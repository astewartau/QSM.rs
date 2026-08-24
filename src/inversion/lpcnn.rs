//! LPCNN dipole inversion (`onnx` feature).
//!
//! LPCNN (Lai 2020) solves the dipole deconvolution by unrolling **proximal
//! gradient descent** (`iter_num = 3`): each iteration takes a k-space
//! data-consistency step with the dipole kernel `D`, then applies a learned 3D-CNN
//! proximal operator `gen`. Only `gen` is a network (exported ONNX, a plain
//! conv/BN/ReLU residual stack); the FFT data-consistency, the unroll, the learned
//! step size `alpha`, and the mean/std normalization live here in Rust.
//!
//! Input is the background-removed **local field in ppm** (single orientation). The
//! authors' Hz↔ppm round-trip (`×tesla·γ` then `÷tesla·γ`) cancels, so the field is
//! consumed directly in ppm. Weights are not bundled; the caller passes `lpcnn.onnx`.

use num_complex::Complex64;

use crate::fft::{fft3d_real, ifft3d_real};
use crate::grid::Grid;
use crate::models::onnx::{OnnxError, OnnxModel, Tensor};

/// Learned data-consistency step size (checkpoint `alpha`, Bmodel).
pub const LPCNN_ALPHA: f64 = 3.718_820_571_899_414;
/// Ground-truth normalization (ppm), baked into the training pipeline.
pub const LPCNN_GT_MEAN: f64 = -0.000_247_246_032_403_825_13;
pub const LPCNN_GT_STD: f64 = 0.028_365_555_003_933_052;
const ITER_NUM: usize = 3;

/// Run LPCNN on a background-removed local field (ppm), column-major `(nx,ny,nz)`.
/// `bdir` is the B0 direction (for the dipole kernel). Returns χ (ppm), masked.
pub fn lpcnn(
    local_field_ppm: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    gen_onnx: &[u8],
) -> Result<Vec<f64>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    assert_eq!(local_field_ppm.len(), n, "field length must match grid");
    assert_eq!(mask.len(), n, "mask length must match grid");

    let model = OnnxModel::load(gen_onnx)?;
    let maskf: Vec<f64> = mask.iter().map(|&m| m as f64).collect();
    let dk = lpcnn_dipole_kernel(grid, bdir);
    let dipole = |v: &[f64]| dipole_apply(v, &dk, grid);

    // x_est = α·D(y)
    let x_est: Vec<f64> = dipole(local_field_ppm).iter().map(|&v| LPCNN_ALPHA * v).collect();

    let mut den = vec![0.0f64; n];
    let mut x_pred = vec![0.0f64; n];
    for i in 0..ITER_NUM {
        // Data-consistency step: pn = α·D(y) at i=0, else den + α·D(y − D(den)).
        let pn: Vec<f64> = if i == 0 {
            x_est.clone()
        } else {
            let dd = dipole(&dipole(&den)); // D²(den)
            (0..n).map(|k| den[k] + x_est[k] - LPCNN_ALPHA * dd[k]).collect()
        };
        // Proximal CNN on the normalized, masked estimate.
        let x_input: Vec<f64> =
            (0..n).map(|k| ((pn[k] - LPCNN_GT_MEAN) / LPCNN_GT_STD) * maskf[k]).collect();
        x_pred = run_gen(&model, &x_input, grid)?;
        den = (0..n)
            .map(|k| (x_pred[k] * LPCNN_GT_STD + LPCNN_GT_MEAN) * maskf[k])
            .collect();
    }
    // Denormalize the final proximal output, masked.
    Ok((0..n)
        .map(|k| (x_pred[k] * LPCNN_GT_STD + LPCNN_GT_MEAN) * maskf[k])
        .collect())
}

/// LPCNN dipole kernel `D = 1/3 − (k·B̂)²/|k|²` with `k` from `fftfreq` (DC at the
/// array corner, `D[0,0,0]=0`) — the convention the model's ortho-FFT expects.
pub(crate) fn lpcnn_dipole_kernel(grid: &Grid, bdir: (f64, f64, f64)) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let (vx, vy, vz) = grid.voxel_size;
    let (bx, by, bz) = bdir;
    let bn = (bx * bx + by * by + bz * bz).sqrt();
    let (bx, by, bz) = (bx / bn, by / bn, bz / bn);
    let freq = |i: usize, ntot: usize, d: f64| -> f64 {
        let ii = if i < ntot.div_ceil(2) { i as f64 } else { i as f64 - ntot as f64 };
        ii / (ntot as f64 * d)
    };
    let mut k = vec![0.0f64; nx * ny * nz];
    for z in 0..nz {
        let kz = freq(z, nz, vz);
        for y in 0..ny {
            let ky = freq(y, ny, vy);
            for x in 0..nx {
                let kx = freq(x, nx, vx);
                let k2 = kx * kx + ky * ky + kz * kz;
                let kb = kx * bx + ky * by + kz * bz;
                k[x + nx * (y + ny * z)] = if k2 > 0.0 { 1.0 / 3.0 - kb * kb / k2 } else { 0.0 };
            }
        }
    }
    k
}

/// `D(v) = real(ifft3(fft3(v)·dk))` with `dk` in fftfreq (unshifted) layout.
fn dipole_apply(v: &[f64], dk: &[f64], grid: &Grid) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let mut spec = fft3d_real(v, nx, ny, nz);
    for (s, &d) in spec.iter_mut().zip(dk) {
        *s *= Complex64::new(d, 0.0);
    }
    ifft3d_real(&spec, nx, ny, nz)
}

/// Run the single-in/single-out proximal CNN on a column-major volume.
fn run_gen(model: &OnnxModel, vol: &[f64], grid: &Grid) -> Result<Vec<f64>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    let mut input = vec![0.0f32; nx * ny * nz];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                input[(x * ny + y) * nz + z] = vol[x + nx * (y + ny * z)] as f32;
            }
        }
    }
    let out = model.run_single(&Tensor::new(vec![1, 1, nx, ny, nz], input))?;
    let mut v = vec![0.0f64; nx * ny * nz];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                v[x + nx * (y + ny * z)] = out.data[(x * ny + y) * nz + z] as f64;
            }
        }
    }
    Ok(v)
}

/// Memory-bounded LPCNN via whole-algorithm overlap-tiling — the entire unrolled net is run on
/// each patch sub-volume so peak memory is bounded (for the 32-bit WASM heap). LPCNN's k-space
/// data-consistency step is global, so tiling is **strongly off-design** and approximate; for
/// real work run whole-volume (e.g. in QSMxT). See [`crate::inversion::tiled::tiled_volume_algorithm`].
pub fn lpcnn_tiled(
    local_field_ppm: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    gen_onnx: &[u8],
    cfg: &super::tiled::TileConfig,
    progress: impl FnMut(usize, usize),
) -> Result<Vec<f64>, OnnxError> {
    super::tiled::tiled_volume_algorithm(
        local_field_ppm, mask, grid, 8, cfg,
        |f, m, g| lpcnn(f, m, g, bdir, gen_onnx),
        progress,
    )
}

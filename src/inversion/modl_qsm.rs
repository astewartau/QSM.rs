//! MoDL-QSM dipole inversion (`onnx` feature).
//!
//! MoDL-QSM (Feng 2021) solves the dipole deconvolution by unrolling a
//! **model-based gradient descent** (`num_iter = 3`): each iteration takes a
//! k-space data-consistency step with the dipole kernel `D` (the `A`/`A^H`
//! operators), then applies a learned 3D-CNN prior. Only the CNN prior is a
//! network (exported ONNX — a conv/BN/ReLU residual stack, 2-channel in/out); the
//! FFT data-consistency, the unroll, the learned step size `alpha`, and the
//! per-channel mean/std normalization live here in Rust.
//!
//! # Field & susceptibility layout
//!
//! Input is the background-removed **local field in ppm** (single orientation).
//! MoDL-QSM's `phi` input is the tissue field normalized to ppm (the repo's example
//! `test_data.mat` fields span ~±0.1–0.2 ppm), so the field is consumed directly.
//!
//! The network is STI-flavored: it works on a **2-channel** susceptibility estimate
//! — channel 0 is the STI tensor component χ33 (comparable to scalar QSM) and
//! channel 1 is the field induced by the χ13/χ23 terms. The `A^H` operator maps a
//! 1-channel field to 2 channels `[ifft(D·fft(φ)), φ]`; the `A` operator maps the
//! 2-channel susceptibility back to a 1-channel field `ifft(D·fft(χ33) + fft(χ13/23))`.
//! We return **channel 0 (χ33)** as the QSM susceptibility, masked (matching the
//! authors' `recon.py`, which keeps `Y[...,0]`).
//!
//! # FFT convention
//!
//! MoDL-QSM's `A`/`A^H` use an **ortho** FFT (`fft/√N`, `ifft·√N`); the `√N` factors
//! cancel through the linear (i)FFT for both operators, so we use the crate's
//! standard normalized `fft3d_real`/`ifft3d_real` pair directly. The dipole kernel
//! `D = 1/3 − (k·B̂)²/|k|²` uses the `fftfreq` convention (DC at the array corner,
//! `D[0,0,0]=0`) — bit-identical to `test_tools.dipole_kernel` (which builds a
//! centered `D` then `fftshift`s it). This is the same kernel LPCNN uses.
//!
//! # Grid requirements
//!
//! `model_test` requires an isotropic 1 mm grid with even dimensions (odd dims are
//! cropped, non-unit voxels are k-space interpolated to 1 mm before inference). The
//! Rust glue therefore assumes 1 mm even-dimension input (the QSM-CI/dev grid);
//! callers needing resampling should do so upstream. Weights are not bundled; the
//! caller passes `modl-qsm.onnx`.

use num_complex::Complex64;

use crate::fft::{fft3d_real, ifft3d_real};
use crate::grid::Grid;
use crate::models::onnx::{OnnxError, OnnxModel, Tensor};

/// Learned data-consistency step size (`Alpha`, checkpoint `logs/last.h5` MyLayer;
/// initializer was 4.0, this is the trained value).
pub const MODL_ALPHA: f64 = 1.101_906_418_800_354;
/// Train-set per-channel mean (`NormFactor.mat` `CosTrnMean`): [χ33, χ13/23 field].
pub const MODL_MEAN: [f64; 2] = [-0.002_256_532_665_342_092_5, -0.000_485_291_442_601_010_2];
/// Train-set per-channel std (`NormFactor.mat` `CosTrnStd`): [χ33, χ13/23 field].
pub const MODL_STD: [f64; 2] = [0.026_074_999_943_375_587, 0.004_082_275_088_876_486];
const NUM_ITER: usize = 3;

/// Run MoDL-QSM on a background-removed local field (ppm), column-major `(nx,ny,nz)`.
/// `bdir` is the B0 direction (for the dipole kernel). Returns χ33 (ppm), masked.
pub fn modl_qsm(
    local_field_ppm: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    prior_onnx: &[u8],
) -> Result<Vec<f64>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    assert_eq!(local_field_ppm.len(), n, "field length must match grid");
    assert_eq!(mask.len(), n, "mask length must match grid");

    let model = OnnxModel::load(prior_onnx)?;
    let maskf: Vec<f64> = mask.iter().map(|&m| m as f64).collect();
    let dk = modl_dipole_kernel(grid, bdir);

    // y_input = alpha · A^H(phi)   (2-channel, interleaved [ch0, ch1] per voxel)
    let y_input = ah_op(local_field_ppm, &dk, grid);
    let y_input: Vec<[f64; 2]> =
        y_input.iter().map(|&[a, b]| [MODL_ALPHA * a, MODL_ALPHA * b]).collect();

    let mut x_output: Vec<[f64; 2]> = vec![[0.0; 2]; n];
    for i in 0..NUM_ITER {
        // Data-consistency: layer_input.
        //   i == 0: layer_input = y_input
        //   i  > 0: layer_input = x_output − alpha·A^H(A(x_output)) + y_input
        let layer_input: Vec<[f64; 2]> = if i == 0 {
            y_input.clone()
        } else {
            let a_x = a_op(&x_output, &dk, grid); // 1-channel field
            let ah_a_x = ah_op(&a_x, &dk, grid); // 2-channel
            (0..n)
                .map(|k| {
                    [
                        x_output[k][0] - MODL_ALPHA * ah_a_x[k][0] + y_input[k][0],
                        x_output[k][1] - MODL_ALPHA * ah_a_x[k][1] + y_input[k][1],
                    ]
                })
                .collect()
        };

        // Normalize per channel, mask, run CNN prior, mask, denormalize per channel.
        let mut norm = vec![0.0f64; 2 * n];
        for k in 0..n {
            norm[2 * k] = ((layer_input[k][0] - MODL_MEAN[0]) / MODL_STD[0]) * maskf[k];
            norm[2 * k + 1] = ((layer_input[k][1] - MODL_MEAN[1]) / MODL_STD[1]) * maskf[k];
        }
        let fx = run_prior(&model, &norm, grid)?; // 2·n interleaved
        for k in 0..n {
            let c0 = fx[2 * k] * maskf[k];
            let c1 = fx[2 * k + 1] * maskf[k];
            x_output[k] = [c0 * MODL_STD[0] + MODL_MEAN[0], c1 * MODL_STD[1] + MODL_MEAN[1]];
        }
    }

    // Keep channel 0 (χ33), masked (recon.py: Y[...,0] · mask).
    Ok((0..n).map(|k| x_output[k][0] * maskf[k]).collect())
}

/// MoDL-QSM dipole kernel `D = 1/3 − (k·B̂)²/|k|²` with `k` from `fftfreq` (DC at the
/// array corner, `D[0,0,0]=0`). Bit-identical to `test_tools.dipole_kernel`.
pub(crate) fn modl_dipole_kernel(grid: &Grid, bdir: (f64, f64, f64)) -> Vec<f64> {
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

/// `A^H` operator: 1-channel field φ → 2-channel `[real(ifft(D·fft(φ))), φ]`.
/// Interleaved output `[ch0, ch1]` per voxel.
fn ah_op(phi: &[f64], dk: &[f64], grid: &Grid) -> Vec<[f64; 2]> {
    let ch0 = dipole_apply(phi, dk, grid);
    ch0.iter().zip(phi).map(|(&a, &p)| [a, p]).collect()
}

/// `A` operator: 2-channel susceptibility → 1-channel field
/// `real(ifft(D·fft(χ33) + fft(χ13/23)))`. Input is interleaved `[ch0, ch1]`.
fn a_op(sus: &[[f64; 2]], dk: &[f64], grid: &Grid) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    let ch0: Vec<f64> = (0..n).map(|k| sus[k][0]).collect();
    let ch1: Vec<f64> = (0..n).map(|k| sus[k][1]).collect();
    let mut spec0 = fft3d_real(&ch0, nx, ny, nz);
    let spec1 = fft3d_real(&ch1, nx, ny, nz);
    for k in 0..n {
        spec0[k] = spec0[k] * Complex64::new(dk[k], 0.0) + spec1[k];
    }
    ifft3d_real(&spec0, nx, ny, nz)
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

/// Run the 2-in / 2-out CNN prior on a column-major volume. `vol` is interleaved
/// `[ch0, ch1]` per voxel (length `2·n`); output is the same interleaved layout.
fn run_prior(model: &OnnxModel, vol: &[f64], grid: &Grid) -> Result<Vec<f64>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    // NCDHW row-major, C=2: [batch, chan, x, y, z].
    let mut input = vec![0.0f32; 2 * n];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let col = x + nx * (y + ny * z);
                let row = (x * ny + y) * nz + z;
                input[row] = vol[2 * col] as f32; // channel 0 plane
                input[n + row] = vol[2 * col + 1] as f32; // channel 1 plane
            }
        }
    }
    let out = model.run_single(&Tensor::new(vec![1, 2, nx, ny, nz], input))?;
    let mut v = vec![0.0f64; 2 * n];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let col = x + nx * (y + ny * z);
                let row = (x * ny + y) * nz + z;
                v[2 * col] = out.data[row] as f64;
                v[2 * col + 1] = out.data[n + row] as f64;
            }
        }
    }
    Ok(v)
}

/// Memory-bounded MoDL-QSM via whole-algorithm overlap-tiling — the full unrolled net runs on
/// each patch sub-volume. MoDL's data-consistency step is a global k-space operation, so tiling
/// is **strongly off-design** and approximate; prefer a whole-volume run (e.g. QSMxT). See
/// [`crate::inversion::tiled::tiled_volume_algorithm`].
pub fn modl_qsm_tiled(
    local_field_ppm: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    prior_onnx: &[u8],
    cfg: &super::tiled::TileConfig,
    progress: impl FnMut(usize, usize),
) -> Result<Vec<f64>, OnnxError> {
    super::tiled::tiled_volume_algorithm(
        local_field_ppm, mask, grid, 1, cfg,
        |f, m, g| modl_qsm(f, m, g, bdir, prior_onnx),
        progress,
    )
}

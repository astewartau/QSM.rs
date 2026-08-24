//! NeXtQSM single-step deep-learning reconstruction (`onnx` feature).
//!
//! NeXtQSM (Cognolato 2023) is a *hybrid* method: a background-removal U-Net
//! followed by a **6-step variational** dipole inversion. Each variational step
//! is a gradient-descent update
//! `x ← x − (λ_k·∇E_D + ∇E_R)·mask`, where `E_D` is an RMSE data-consistency
//! term through the FFT dipole forward and `E_R = mean(|VarNet(x)|)` is a learned
//! regularizer whose gradient is a backprop through the VarNet U-Net.
//!
//! The two U-Net pieces run as ONNX (`nextqsm-bf.onnx` = BFR forward;
//! `nextqsm-vjp.onnx` = the regularizer gradient `∇ₓ mean(|VarNet(x)|)`, hand-coded
//! as a forward graph). The FFT data-consistency gradient and the unroll live here
//! in Rust (`rustfft`), since tract can't do in-graph FFT.
//!
//! Weights are not bundled; the caller passes both ONNX byte buffers.

use num_complex::Complex64;

use crate::fft::{fft3d_real, ifft3d_real};
use crate::grid::Grid;
use crate::models::onnx::{OnnxModel, OnnxError, Tensor};

/// Trained data-consistency weights `λ_k` (one per variational step) shipped with
/// the NeXtQSM checkpoint. Scalars, like a normalization constant.
pub const NEXTQSM_LAMBDAS: [f64; 6] =
    [15.141516, 15.005543, 16.733437, 14.574243, 7.072418, 0.7588475];

/// Run NeXtQSM on a total field map.
///
/// * `total_field` — total field (ppm), column-major `(nx,ny,nz)`.
/// * `mask` — binary brain mask (same layout).
/// * `bdir` — B0 direction; `bf_onnx`/`vjp_onnx` — the two exported graphs.
///
/// The volume is zero-padded (centered) to a multiple of 64 (six pooling levels),
/// reconstructed, and cropped back. Returns susceptibility (ppm), masked.
pub fn nextqsm(
    total_field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    bf_onnx: &[u8],
    vjp_onnx: &[u8],
) -> Result<Vec<f64>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    assert_eq!(total_field.len(), n);
    assert_eq!(mask.len(), n);

    // Centered pad to a multiple of 64.
    let pad = |s: usize| -> (usize, usize) {
        let total = (64 - s % 64) % 64;
        (total / 2, s + total)
    };
    let (bx, px) = pad(nx);
    let (by, py) = pad(ny);
    let (bz, pz) = pad(nz);
    let pgrid = Grid { dims: (px, py, pz), voxel_size: grid.voxel_size };
    let np = px * py * pz;

    let mut field_p = vec![0.0f64; np];
    let mut mask_p = vec![0u8; np];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let s = x + nx * (y + ny * z);
                let d = (x + bx) + px * ((y + by) + py * (z + bz));
                field_p[d] = total_field[s];
                mask_p[d] = mask[s];
            }
        }
    }

    let chi_p = nextqsm_padded(&field_p, &mask_p, &pgrid, bdir, bf_onnx, vjp_onnx, &NEXTQSM_LAMBDAS)?;

    // Crop back.
    let mut chi = vec![0.0f64; n];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                chi[x + nx * (y + ny * z)] = chi_p[(x + bx) + px * ((y + by) + py * (z + bz))];
            }
        }
    }
    Ok(chi)
}

/// NeXtQSM on a grid already sized to a multiple of 64 (no internal padding).
/// Exposed for validation against the reference trajectory.
#[allow(clippy::too_many_arguments)]
pub fn nextqsm_padded(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    bf_onnx: &[u8],
    vjp_onnx: &[u8],
    lambdas: &[f64],
) -> Result<Vec<f64>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    let maskf: Vec<f64> = mask.iter().map(|&m| m as f64).collect();

    // fftshift(dipole kernel) in the crate's column-major layout.
    let kernel = dipole_kernel_shifted(grid, bdir);

    // bf_logits = BFRnet(field·mask)·mask
    let bf = OnnxModel::load(bf_onnx)?;
    let masked_field: Vec<f64> = field.iter().zip(&maskf).map(|(&f, &m)| f * m).collect();
    let bf_out = run_unet(&bf, &masked_field, grid)?;
    let bf_logits: Vec<f64> = bf_out.iter().zip(&maskf).map(|(&v, &m)| v * m).collect();
    let norm_bf = l2(&bf_logits);

    let vjp = OnnxModel::load(vjp_onnx)?;
    let mut x = bf_logits.clone();
    for &lam in lambdas.iter() {
        let dx = dipole_forward(&x, &kernel, grid);
        // u = bf_logits − D(x);  ∇E_D = 100/(‖bf‖·‖u‖) · D(D(x) − bf)
        let u: Vec<f64> = bf_logits.iter().zip(&dx).map(|(&b, &d)| b - d).collect();
        let norm_u = l2(&u);
        let resid: Vec<f64> = dx.iter().zip(&bf_logits).map(|(&d, &b)| d - b).collect();
        let d_resid = dipole_forward(&resid, &kernel, grid);
        let scale = if norm_bf > 0.0 && norm_u > 0.0 { 100.0 / (norm_bf * norm_u) } else { 0.0 };

        // The VJP graph returns the unnormalized gradient ∇ₓ Σ|VarNet(x)|; divide
        // by N to get the mean-gradient ∇ₓ mean(|VarNet(x)|) the regularizer uses.
        let grad_r = run_unet(&vjp, &x, grid)?;
        let inv_n = 1.0 / (n as f64);

        for i in 0..n {
            let de = lam * (scale * d_resid[i]) + grad_r[i] * inv_n;
            x[i] = (x[i] - de) * maskf[i];
        }
    }
    Ok(x)
}

/// Dipole forward `D(y) = real(ifft3(fft3(y) · kernel_shifted))`.
pub(crate) fn dipole_forward(y: &[f64], kernel_shifted: &[f64], grid: &Grid) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let mut spec = fft3d_real(y, nx, ny, nz);
    for (s, &k) in spec.iter_mut().zip(kernel_shifted) {
        *s *= Complex64::new(k, 0.0);
    }
    ifft3d_real(&spec, nx, ny, nz)
}

/// `fftshift(get_dipole_kernel_fourier(...))` in column-major layout.
pub(crate) fn dipole_kernel_shifted(grid: &Grid, bdir: (f64, f64, f64)) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let (vx, vy, vz) = grid.voxel_size;
    let eps = f32::EPSILON as f64;
    let (bx, by, bz) = bdir;
    let mut k = vec![0.0f64; nx * ny * nz];
    // Centered grid, then fftshift => index shift by half in each dim.
    let sh = |i: usize, nsz: usize| (i + nsz / 2) % nsz; // fftshift index
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                // centered coordinate = (idx − n/2); normalized by (n/2) and (2·vox)
                let rx = ((x as f64) - (nx as f64) / 2.0) / ((nx as f64) / 2.0) / (2.0 * vx);
                let ry = ((y as f64) - (ny as f64) / 2.0) / ((ny as f64) / 2.0) / (2.0 * vy);
                let rz = ((z as f64) - (nz as f64) / 2.0) / ((nz as f64) / 2.0) / (2.0 * vz);
                let r2 = rx * rx + ry * ry + rz * rz;
                let dot = rx * bx + ry * by + rz * bz;
                let val = 1.0 / 3.0 - (dot * dot) / (r2 + eps);
                // write to fftshifted position
                let (sx, sy, sz) = (sh(x, nx), sh(y, ny), sh(z, nz));
                k[sx + nx * (sy + ny * sz)] = val;
            }
        }
    }
    k
}

/// Run a single-input/single-output 3D U-Net ONNX on a column-major volume.
fn run_unet(model: &OnnxModel, vol: &[f64], grid: &Grid) -> Result<Vec<f64>, OnnxError> {
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

fn l2(v: &[f64]) -> f64 {
    v.iter().map(|&a| a * a).sum::<f64>().sqrt()
}

/// Memory-bounded NeXtQSM via whole-algorithm overlap-tiling — the full VarNet gradient-descent
/// loop runs on each patch sub-volume (each internally padded to /64). NeXtQSM's forward model is
/// global, so tiling is **strongly off-design** and approximate; prefer a whole-volume run (e.g.
/// QSMxT). See [`crate::inversion::tiled::tiled_volume_algorithm`].
#[allow(clippy::too_many_arguments)]
pub fn nextqsm_tiled(
    total_field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    bf_onnx: &[u8],
    vjp_onnx: &[u8],
    cfg: &super::tiled::TileConfig,
    progress: impl FnMut(usize, usize),
) -> Result<Vec<f64>, OnnxError> {
    super::tiled::tiled_volume_algorithm(
        total_field, mask, grid, 64, cfg,
        |f, m, g| nextqsm(f, m, g, bdir, bf_onnx, vjp_onnx),
        progress,
    )
}

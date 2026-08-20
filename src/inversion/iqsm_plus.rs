//! iQSM+ single-step deep-learning reconstruction (`onnx` feature).
//!
//! iQSM+ (Gao 2024) extends [iQSM](crate::inversion::iqsm) with orientation-adaptive
//! latent feature editing (OA-LFE): the B0 direction (`z_prjs`) is a network input,
//! so oblique/sagittal/coronal acquisitions reconstruct correctly. The exported
//! graph takes six inputs: `phase`, `mask`, `te` (s), `b0` (T), `z_prjs` (B0 dir,
//! `[1,1,3]`), and `border` (the LoT boundary mask, as for iQSM).
//!
//! Pipeline (mirrors the authors' `inference.run_iqsm_plus`): flip the phase sign,
//! erode the mask by a radius-3 sphere, **crop to the brain bounding box + 16-voxel
//! margin**, centre-pad to a multiple of 16, run, ×mask, then paste the result back
//! into the full grid. Multi-echo data is combined with magnitude·TE² weighting.
//!
//! Only axial-ish acquisitions are handled directly here; the authors' extra
//! axis-permutation for strongly oblique fields (`|dir_y| > |dir_z|`) is not applied.
//!
//! Weights are not bundled; the caller passes the exported `iqsm-plus.onnx` bytes.

use crate::grid::Grid;
use crate::inversion::iqsm::sphere_erode;
use crate::models::onnx::{OnnxModel, OnnxError, Tensor};

/// Run iQSM+ on a single echo of wrapped phase.
///
/// * `phase_rad`, `mask` — column-major `(nx,ny,nz)`.
/// * `te` (s), `b0` (T), `b0_dir` — acquisition parameters (`b0_dir` normalized internally).
/// * `phase_sign` (`-1` default), `eroded_rad` (`3` default).
///
/// Returns susceptibility (ppm), masked, in the same layout.
#[allow(clippy::too_many_arguments)]
pub fn iqsm_plus(
    phase_rad: &[f64],
    mask: &[u8],
    grid: &Grid,
    te: f64,
    b0: f64,
    b0_dir: (f64, f64, f64),
    phase_sign: f64,
    eroded_rad: i32,
    model_onnx: &[u8],
) -> Result<Vec<f64>, OnnxError> {
    let model = OnnxModel::load(model_onnx)?;
    iqsm_plus_with(&model, phase_rad, mask, grid, te, b0, b0_dir, phase_sign, eroded_rad)
}

/// Multi-echo iQSM+: reconstruct each echo and combine with magnitude·TE² weights.
#[allow(clippy::too_many_arguments)]
pub fn iqsm_plus_multi_echo(
    phases: &[&[f64]],
    magnitudes: &[&[f64]],
    mask: &[u8],
    grid: &Grid,
    tes: &[f64],
    b0: f64,
    b0_dir: (f64, f64, f64),
    phase_sign: f64,
    eroded_rad: i32,
    model_onnx: &[u8],
) -> Result<Vec<f64>, OnnxError> {
    assert_eq!(phases.len(), tes.len(), "one TE per echo");
    let n = grid.n_total();
    let model = OnnxModel::load(model_onnx)?;
    let mut acc = vec![0.0f64; n];
    let mut wsum = vec![0.0f64; n];
    for (e, &phase) in phases.iter().enumerate() {
        let chi = iqsm_plus_with(&model, phase, mask, grid, tes[e], b0, b0_dir, phase_sign, eroded_rad)?;
        let te2 = tes[e] * tes[e];
        for i in 0..n {
            let w = magnitudes.get(e).map(|m| m[i]).unwrap_or(1.0) * te2;
            acc[i] += w * chi[i];
            wsum[i] += w;
        }
    }
    for i in 0..n {
        acc[i] = if wsum[i] > 0.0 { acc[i] / wsum[i] } else { 0.0 };
    }
    Ok(acc)
}

#[allow(clippy::too_many_arguments)]
fn iqsm_plus_with(
    model: &OnnxModel,
    phase_rad: &[f64],
    mask: &[u8],
    grid: &Grid,
    te: f64,
    b0: f64,
    b0_dir: (f64, f64, f64),
    phase_sign: f64,
    eroded_rad: i32,
) -> Result<Vec<f64>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    assert_eq!(phase_rad.len(), n, "phase length must match grid");
    assert_eq!(mask.len(), n, "mask length must match grid");
    let idx = |x: usize, y: usize, z: usize| x + nx * (y + ny * z);

    let eroded = if eroded_rad > 0 {
        sphere_erode(mask, grid, eroded_rad)
    } else {
        mask.to_vec()
    };

    // Brain bounding box of the eroded mask, expanded by 16 and clamped.
    let (mut x0, mut y0, mut z0) = (nx, ny, nz);
    let (mut x1, mut y1, mut z1) = (0usize, 0usize, 0usize);
    let mut any = false;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if eroded[idx(x, y, z)] != 0 {
                    any = true;
                    x0 = x0.min(x); y0 = y0.min(y); z0 = z0.min(z);
                    x1 = x1.max(x); y1 = y1.max(y); z1 = z1.max(z);
                }
            }
        }
    }
    if !any {
        return Ok(vec![0.0; n]);
    }
    let pad = 16usize;
    let bx0 = x0.saturating_sub(pad);
    let by0 = y0.saturating_sub(pad);
    let bz0 = z0.saturating_sub(pad);
    let bx1 = (x1 + 1 + pad).min(nx);
    let by1 = (y1 + 1 + pad).min(ny);
    let bz1 = (z1 + 1 + pad).min(nz);
    let (cx, cy, cz) = (bx1 - bx0, by1 - by0, bz1 - bz0);

    // Centre-pad the cropped box to a multiple of 16.
    let mpad = |s: usize| -> (usize, usize) {
        let total = (16 - s % 16) % 16;
        (total / 2, s + total)
    };
    let (ox, px) = mpad(cx);
    let (oy, py) = mpad(cy);
    let (oz, pz) = mpad(cz);

    // Fill NCDHW row-major [1,1,px,py,pz] phase/mask/border from the cropped box.
    let mut phase_t = vec![0.0f32; px * py * pz];
    let mut mask_t = vec![0.0f32; px * py * pz];
    let mut border = vec![1.0f32; px * py * pz];
    let rdst = |i: usize, j: usize, k: usize| (k + oz) + pz * ((j + oy) + py * (i + ox));
    for k in 0..cz {
        for j in 0..cy {
            for i in 0..cx {
                let src = idx(bx0 + i, by0 + j, bz0 + k);
                let dst = rdst(i, j, k);
                phase_t[dst] = (phase_sign * phase_rad[src]) as f32;
                mask_t[dst] = eroded[src] as f32;
            }
        }
    }
    for a in 0..px {
        for b in 0..py {
            for c in 0..pz {
                if a == 0 || a == px - 1 || b == 0 || b == py - 1 || c == 0 || c == pz - 1 {
                    border[c + pz * (b + py * a)] = 0.0;
                }
            }
        }
    }

    let norm = (b0_dir.0.powi(2) + b0_dir.1.powi(2) + b0_dir.2.powi(2)).sqrt();
    let zdir = if norm > 0.0 { norm } else { 1.0 };
    let z_prjs = vec![
        (b0_dir.0 / zdir) as f32,
        (b0_dir.1 / zdir) as f32,
        (b0_dir.2 / zdir) as f32,
    ];

    let shape = vec![1, 1, px, py, pz];
    let inputs = [
        Tensor::new(shape.clone(), phase_t),
        Tensor::new(shape.clone(), mask_t),
        Tensor::new(vec![1], vec![te as f32]),
        Tensor::new(vec![1], vec![b0 as f32]),
        Tensor::new(vec![1, 1, 3], z_prjs),
        Tensor::new(shape, border),
    ];
    let out = model.run(&inputs)?;
    let chi_pad = &out[0].data;

    // ×mask, crop pad, paste back into the full grid at the bounding box.
    let mut chi = vec![0.0f64; n];
    for k in 0..cz {
        for j in 0..cy {
            for i in 0..cx {
                let src = idx(bx0 + i, by0 + j, bz0 + k);
                if eroded[src] != 0 {
                    chi[src] = chi_pad[rdst(i, j, k)] as f64;
                }
            }
        }
    }
    Ok(chi)
}

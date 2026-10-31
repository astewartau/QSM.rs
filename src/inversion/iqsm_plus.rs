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
//! into the full grid. Multi-echo data is combined with (magnitude·TE)² weighting.
//!
//! Only axial-ish acquisitions are handled directly here; the authors' extra
//! axis-permutation for strongly oblique fields (`|dir_y| > |dir_z|`) is not applied.
//!
//! Weights are not bundled; the caller passes the exported `iqsm-plus.onnx` bytes.
//!
//! # Reference pipeline
//!
//! As for [iQSM](mod@crate::inversion::iqsm) (normalisation, `bet2 -f 0.5` default mask), with two
//! differences in the authors' `sunhongfu/iQSM_Plus` code:
//!
//! - **Resampling:** `inference.py:443-471` (and `legacy/matlab/iQSM_plus.m:175-210`) interpolates
//!   anisotropic phase to an isotropic grid at the finest voxel size in the complex domain whenever
//!   the rounded grid would change. This crate does not resample.
//! - **Phase masking:** `inference.py:425` and `legacy/matlab/fcns/Save_Input_iQSMplus.m` multiply
//!   the phase by the *unmodified* mask before erosion. This crate does not; with an erosion radius
//!   of at least 1 that cannot change the result, since the 3×3×3 LoT stencil only reaches voxels
//!   inside the unmodified mask wherever its output survives the eroded mask.
//!
//! # Agreement with the original
//!
//! `tests/models_onnx/field.rs` runs it on a 36×32×28 box of in-vivo data (1 mm, 2.89 T, across the
//! brain's edge) against its original's verified QSM-CI entry, iQSM+ a534d94 (`iqsm-plus`)
//! (QSMxT/QSM-CI#316; the fixture is regenerated in Actions by
//! `tests/data/dl_field_qsmci_small.py`): per echo, NRMSE 2.7e-4 to 3.3e-4 %, max |Δ| up to 6.8e-7
//! ppm: float32 rounding.
//!
//! The original runs one echo at a time and combines the echoes afterwards with weights
//! (mag·TE)², in both of the authors' implementations (see [`iqsm_plus_multi_echo`]), and so does
//! this crate. On the box's first and third echoes the driver's combination is (mag·TE)² of its
//! per-echo outputs to 1.6e-3 %, and mag·TE² weighting (iQSM's MATLAB echo fit, which this crate
//! used for iQSM+ until 2026-10-31) differs from it by 3.1 % NRMSE (measured 2026-10-30);
//! [`iqsm_plus_multi_echo`] matches the (mag·TE)² combination of the original's per-echo outputs
//! to 2.5e-4 % (measured 2026-10-31).
//!
//! # Pipeline rules
//!
//! The facts above that a configuration can depart from are published as rules in
//! [`crate::pipeline::rules::RULES`] (each with its strength, basis, reason and source): `iqsm.mask-bet`.

use crate::grid::Grid;
use crate::inversion::iqsm::{combine_echoes, sphere_erode, tiled_echo, IqsmEchoWeighting, Net};
use crate::inversion::tiled::{tile_cost_of, tile_patch_size, TileConfig};
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
    iqsm_plus_with(&|i| model.run(i), phase_rad, mask, grid, te, b0, b0_dir, phase_sign, eroded_rad, true)
}

/// Multi-echo iQSM+: reconstruct each echo and combine them with weights (magnitude·TE)², i.e.
/// [`IqsmEchoWeighting::MagnitudeSquaredTeSquared`]; without `magnitudes`, TE² alone.
///
/// Both of the authors' implementations weight this way (checked at `sunhongfu/iQSM_Plus`
/// 3ba25f5): the MATLAB `legacy/matlab/iQSM_plus.m:273-275`
/// (`weights = (mag.*TE).^2; sum(weights.*chi,4)./sum(weights,4)`) and the Python `run.py:199` and
/// `app.py:415` (`weights = (mag_data * te_bc) ** 2`), and it is the inverse-variance weight. Unlike
/// iQSM and iQFM, whose MATLAB uses mag·TE² (an option there; see [`IqsmEchoWeighting`]), there is
/// no other weighting to offer.
///
/// `tile: None` runs each echo on the brain's bounding box in one pass; `Some` runs it patch by
/// patch, with memory bounded by the patch (an approximation; see [`super::iqsm`]).
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
    tile: Option<&TileConfig>,
) -> Result<Vec<f64>, OnnxError> {
    assert_eq!(phases.len(), tes.len(), "one TE per echo");
    let model = OnnxModel::load(model_onnx)?;
    let whole = |i: &[Tensor]| model.run(i);
    let cost = tile_cost_of("iqsm-plus");
    let tiled = match tile {
        Some(cfg) => {
            let p = tile_patch_size(cfg, cost.divisor);
            let v = [1, 1, p, p, p];
            Some((cfg, model.plan_for(&[&v, &v, &[1], &[1], &[1, 1, 3], &v])?))
        }
        None => None,
    };
    let weighting = IqsmEchoWeighting::MagnitudeSquaredTeSquared;
    combine_echoes(phases, magnitudes, tes, grid.n_total(), weighting, |phase, te| match &tiled {
        None => iqsm_plus_with(&whole, phase, mask, grid, te, b0, b0_dir, phase_sign, eroded_rad, true),
        // A tile is already a multiple of 16 and shaped as the plan is: no crop to the brain.
        Some((cfg, plan)) => tiled_echo(phase, mask, grid, eroded_rad, cfg, &cost, |ph, m, g| {
            iqsm_plus_with(&|i| plan.run(i), ph, m, g, te, b0, b0_dir, phase_sign, 0, false)
        }),
    })
}

/// One pass of the network. With `crop`, on the eroded mask's bounding box plus 16 voxels,
/// centre-padded to a multiple of 16; without, on the whole input, which must already be one.
#[allow(clippy::too_many_arguments)]
fn iqsm_plus_with(
    net: Net,
    phase_rad: &[f64],
    mask: &[u8],
    grid: &Grid,
    te: f64,
    b0: f64,
    b0_dir: (f64, f64, f64),
    phase_sign: f64,
    eroded_rad: i32,
    crop: bool,
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
    let (bx0, by0, bz0, bx1, by1, bz1) = if crop {
        (
            x0.saturating_sub(pad),
            y0.saturating_sub(pad),
            z0.saturating_sub(pad),
            (x1 + 1 + pad).min(nx),
            (y1 + 1 + pad).min(ny),
            (z1 + 1 + pad).min(nz),
        )
    } else {
        (0, 0, 0, nx, ny, nz)
    };
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
    let out = net(&inputs)?;
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

//! iQSM single-step deep-learning reconstruction (`onnx` feature).
//!
//! iQSM (Gao 2022) maps raw wrapped MRI **phase** (radians) directly to
//! susceptibility (ppm) — unwrapping, background removal and dipole inversion in
//! one network (a learnable-Laplacian "LoT" front-end + a U-Net). The exported
//! graph takes four inputs: `phase`, `mask`, `te` (s, scalar), `b0` (T, scalar).
//!
//! This mirrors the authors' `inference.run_iqsm`: flip the phase sign, erode the
//! mask by a radius-3 sphere, centre-pad each dim to a multiple of 16, run the
//! net, multiply by the (padded) mask, and crop back. For multi-echo data each
//! echo is reconstructed and combined with magnitude·TE² weighting
//! ([`iqsm_multi_echo`]).
//!
//! The authors run the whole volume in one pass, and so does [`iqsm`]. Its activations scale with
//! the volume; the multi-echo entry point can instead run each echo patch by patch (a
//! [`TileConfig`]), with memory bounded by the patch: an approximation, since the network's
//! unwrapping, background removal and inversion are not local, as for the other tiled networks
//! (see [`super::tiled`]).
//!
//! Weights are not bundled; the caller passes the exported `iqsm.onnx` bytes.
//!
//! # Reference pipeline
//!
//! From the authors' code (`sunhongfu/iQSM`):
//!
//! - **Normalisation:** the LoT layer scales its output by `3 · 20e-3 / (B0 · TE)`
//!   (`models/unet_blocks.py:21-22`), i.e. to the 3 T, TE = 20 ms reference, so any field strength
//!   and TE are handled through the `te`/`b0` inputs.
//! - **Masking:** with no mask supplied, `run.py:499-506` runs FSL `bet2 -m -f 0.5` on the first
//!   magnitude volume (`bet2_utils.py:47,64`).
//! - **Resolution:** "for data with resolution finer than 0.7 mm isotropic, interpolate to 1 mm
//!   before reconstruction" (`README.md:9`; also `legacy/matlab/demo_single_echo.m:21-23`). The
//!   MATLAB demo interpolates anisotropic data to an isotropic grid at the finest voxel size in the
//!   complex domain (`demo_single_echo.m:52-60`). The Python `inference.py` accepts `voxel_size`
//!   but does not resample. This crate does not resample.
//! - **Phase masking:** the MATLAB `Save_Input.m:25` multiplies the phase by the *eroded* mask; the
//!   Python `inference.py` does not, and neither does this crate. The 3×3×3 LoT stencil output is
//!   multiplied by the eroded mask, so only the outermost shell of the eroded mask can differ.
//!
//! # Pipeline rules
//!
//! The facts above that a configuration can depart from are published as rules in
//! [`crate::pipeline::rules::RULES`] (each with its reason and source): `iqsm.mask-bet`.

use super::tiled::{tile_cost_of, tile_patch_size, tiled_volume_algorithm, TileConfig};
use crate::models::TileCost;
use crate::grid::Grid;
use crate::models::onnx::{OnnxModel, OnnxError, Tensor};

/// Runs the network on its input tensors: a whole-volume model (re-planned per shape) or a plan
/// built once for a tile's shape.
pub(crate) type Net<'a> = &'a (dyn Fn(&[Tensor]) -> Result<Vec<Tensor>, OnnxError> + Sync);

/// Run iQSM on a single echo of wrapped phase.
///
/// * `phase_rad` — wrapped phase (radians), column-major `(nx,ny,nz)`.
/// * `mask` — binary brain mask (same layout).
/// * `te` — echo time in seconds; `b0` — field strength in Tesla.
/// * `phase_sign` — sign convention (`-1` matches the authors' default).
/// * `eroded_rad` — mask erosion radius in voxels (`3` matches the default; `0` disables).
/// * `model_onnx` — bytes of the exported `iqsm.onnx`.
///
/// Returns susceptibility (ppm), masked, in the same layout.
#[allow(clippy::too_many_arguments)]
pub fn iqsm(
    phase_rad: &[f64],
    mask: &[u8],
    grid: &Grid,
    te: f64,
    b0: f64,
    phase_sign: f64,
    eroded_rad: i32,
    model_onnx: &[u8],
) -> Result<Vec<f64>, OnnxError> {
    let model = OnnxModel::load(model_onnx)?;
    iqsm_with(&|i| model.run(i), phase_rad, mask, grid, te, b0, phase_sign, eroded_rad)
}

/// Multi-echo iQSM: reconstruct each echo and combine with magnitude·TE² weights.
/// `phases`/`magnitudes` are per-echo volumes; `tes` the echo times (s). Without
/// `magnitudes` the weights are TE² alone (uniform magnitude).
///
/// The weights are those of the authors' MATLAB echo fit: `sunhongfu/iQSM_Plus`
/// `legacy/matlab/iQSM.m:262-267` fits `TE·χ` against TE with `echofit.m`
/// (`sunhongfu/iQSM` `legacy/matlab/iQSM_fcns/echofit.m`), which is `Σ mag·TE²·χ / Σ mag·TE²`.
/// The authors' current Python drivers (`sunhongfu/iQSM` `run.py:170`, the `--echo_4d` path, and
/// `app.py`) weight by `(mag·TE)²` instead, i.e. magnitude squared; this crate does not.
///
/// `tile: None` runs each echo on the whole volume; `Some` runs it patch by patch (see the
/// module docs), the mask eroded once on the whole volume first.
#[allow(clippy::too_many_arguments)]
pub fn iqsm_multi_echo(
    phases: &[&[f64]],
    magnitudes: &[&[f64]],
    mask: &[u8],
    grid: &Grid,
    tes: &[f64],
    b0: f64,
    phase_sign: f64,
    eroded_rad: i32,
    model_onnx: &[u8],
    tile: Option<&TileConfig>,
) -> Result<Vec<f64>, OnnxError> {
    lot_multi_echo("iqsm", phases, magnitudes, mask, grid, tes, b0, phase_sign, eroded_rad, model_onnx, tile)
}

/// [`iqsm_multi_echo`] for either head of the network: `model_id` (`"iqsm"` or `"iqfm"`) names
/// the registry entry whose tile cost the tiled path budgets with.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lot_multi_echo(
    model_id: &str,
    phases: &[&[f64]],
    magnitudes: &[&[f64]],
    mask: &[u8],
    grid: &Grid,
    tes: &[f64],
    b0: f64,
    phase_sign: f64,
    eroded_rad: i32,
    model_onnx: &[u8],
    tile: Option<&TileConfig>,
) -> Result<Vec<f64>, OnnxError> {
    assert_eq!(phases.len(), tes.len(), "one TE per echo");
    let model = OnnxModel::load(model_onnx)?;
    let whole = |i: &[Tensor]| model.run(i);
    let cost = tile_cost_of(model_id);
    let tiled = match tile {
        Some(cfg) => {
            let p = tile_patch_size(cfg, cost.divisor);
            let v = [1, 1, p, p, p];
            Some((cfg, model.plan_for(&[&v, &v, &[1], &[1], &v])?))
        }
        None => None,
    };
    combine_echoes(phases, magnitudes, tes, grid.n_total(), |phase, te| match &tiled {
        None => iqsm_with(&whole, phase, mask, grid, te, b0, phase_sign, eroded_rad),
        Some((cfg, plan)) => tiled_echo(phase, mask, grid, eroded_rad, cfg, &cost, |ph, m, g| {
            iqsm_with(&|i| plan.run(i), ph, m, g, te, b0, phase_sign, 0)
        }),
    })
}

/// One echo patch by patch: erode the mask on the whole volume, then run `run_patch` (the
/// network with no erosion of its own) on each tile. `cost` is the network's tile cost, whose
/// divisor (16: the LoT-Unet has four pooling levels) pads each patch.
pub(crate) fn tiled_echo(
    phase: &[f64],
    mask: &[u8],
    grid: &Grid,
    eroded_rad: i32,
    cfg: &TileConfig,
    cost: &TileCost,
    run_patch: impl Fn(&[f64], &[u8], &Grid) -> Result<Vec<f64>, OnnxError> + Sync,
) -> Result<Vec<f64>, OnnxError> {
    let eroded = if eroded_rad > 0 { sphere_erode(mask, grid, eroded_rad) } else { mask.to_vec() };
    tiled_volume_algorithm(phase, &eroded, grid, cost, cfg, run_patch, |_, _| {})
}

/// Reconstruct each echo with `echo(phase, te)` and combine with magnitude·TE² weights.
pub(crate) fn combine_echoes(
    phases: &[&[f64]],
    magnitudes: &[&[f64]],
    tes: &[f64],
    n: usize,
    mut echo: impl FnMut(&[f64], f64) -> Result<Vec<f64>, OnnxError>,
) -> Result<Vec<f64>, OnnxError> {
    let mut acc = vec![0.0f64; n];
    let mut wsum = vec![0.0f64; n];
    for (e, &phase) in phases.iter().enumerate() {
        let chi = echo(phase, tes[e])?;
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
fn iqsm_with(
    net: Net,
    phase_rad: &[f64],
    mask: &[u8],
    grid: &Grid,
    te: f64,
    b0: f64,
    phase_sign: f64,
    eroded_rad: i32,
) -> Result<Vec<f64>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    assert_eq!(phase_rad.len(), n, "phase length must match grid");
    assert_eq!(mask.len(), n, "mask length must match grid");

    let eroded = if eroded_rad > 0 {
        sphere_erode(mask, grid, eroded_rad)
    } else {
        mask.to_vec()
    };

    // Centre-pad to a multiple of 16 (four pooling levels).
    let pad = |s: usize| -> (usize, usize) {
        let total = (16 - s % 16) % 16;
        (total / 2, s + total)
    };
    let (bx, px) = pad(nx);
    let (by, py) = pad(ny);
    let (bz, pz) = pad(nz);

    // Repack column-major (nx,ny,nz) → row-major NCDHW [1,1,px,py,pz] f32, centered.
    let mut phase_t = vec![0.0f32; px * py * pz];
    let mut mask_t = vec![0.0f32; px * py * pz];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let src = x + nx * (y + ny * z);
                let dst = (z + bz) + pz * ((y + by) + py * (x + bx));
                phase_t[dst] = (phase_sign * phase_rad[src]) as f32;
                mask_t[dst] = eroded[src] as f32;
            }
        }
    }

    // Border mask: 0 on the padded volume's outer 1-voxel shell, 1 inside. The
    // LoT layer's boundary zeroing is applied inside the graph as `conv * border`
    // (the only tract-friendly encoding of that op).
    let mut border = vec![1.0f32; px * py * pz];
    for a in 0..px {
        for b in 0..py {
            for c in 0..pz {
                if a == 0 || a == px - 1 || b == 0 || b == py - 1 || c == 0 || c == pz - 1 {
                    border[(c) + pz * ((b) + py * a)] = 0.0;
                }
            }
        }
    }

    let shape = vec![1, 1, px, py, pz];
    let inputs = [
        Tensor::new(shape.clone(), phase_t),
        Tensor::new(shape.clone(), mask_t),
        Tensor::new(vec![1], vec![te as f32]),
        Tensor::new(vec![1], vec![b0 as f32]),
        Tensor::new(shape, border),
    ];
    let out = net(&inputs)?;
    let chi_pad = &out[0].data;

    // Multiply by the (padded) mask, crop, unpack → column-major.
    let mut chi = vec![0.0f64; n];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let dst = x + nx * (y + ny * z);
                if eroded[dst] != 0 {
                    let src = (z + bz) + pz * ((y + by) + py * (x + bx));
                    chi[dst] = chi_pad[src] as f64; // mask already applied (eroded[dst]==1 here)
                }
            }
        }
    }
    Ok(chi)
}

/// Binary erosion by a solid sphere of the given radius (voxels), matching
/// `scipy.ndimage.binary_erosion` with `border_value=0` (out-of-bounds = false).
pub(crate) fn sphere_erode(mask: &[u8], grid: &Grid, radius: i32) -> Vec<u8> {
    let (nx, ny, nz) = grid.dims;
    // Precompute sphere offsets (dx²+dy²+dz² ≤ r²).
    let r2 = radius * radius;
    let mut offs: Vec<(i32, i32, i32)> = Vec::new();
    for dz in -radius..=radius {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy + dz * dz <= r2 {
                    offs.push((dx, dy, dz));
                }
            }
        }
    }
    let (nxi, nyi, nzi) = (nx as i32, ny as i32, nz as i32);
    let mut out = vec![0u8; nx * ny * nz];
    for z in 0..nzi {
        for y in 0..nyi {
            for x in 0..nxi {
                let mut keep = true;
                for &(dx, dy, dz) in &offs {
                    let (xx, yy, zz) = (x + dx, y + dy, z + dz);
                    let inside = xx >= 0 && xx < nxi && yy >= 0 && yy < nyi && zz >= 0 && zz < nzi;
                    if !inside
                        || mask[xx as usize + nx * (yy as usize + ny * zz as usize)] == 0
                    {
                        keep = false;
                        break;
                    }
                }
                if keep {
                    out[x as usize + nx * (y as usize + ny * z as usize)] = 1;
                }
            }
        }
    }
    out
}

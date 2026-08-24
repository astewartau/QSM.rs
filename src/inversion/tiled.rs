//! Overlap-tiled inference for fully-convolutional deep-learning inversions (`onnx`).
//!
//! Whole-volume nets (e.g. xQSM, QSMnet) allocate activations proportional to the entire
//! volume, which overflows a 32-bit WASM heap (4 GB ceiling) on clinical-size data. Because
//! these nets are fully convolutional, they can instead be run **patch-by-patch**: each
//! output "core" is produced from a patch that includes a `halo` of surrounding context,
//! then written back — the classic U-Net overlap-tile strategy. Peak memory is bounded by a
//! single patch regardless of volume size.
//!
//! NOTE: tiling a net trained on whole volumes is an **approximation**. Dipole inversion is a
//! global operation, so a patch is blind to distant susceptibility sources; a larger `halo`
//! reduces the resulting low-frequency / boundary error but does not eliminate it.

use crate::grid::Grid;
use crate::models::onnx::{OnnxError, OnnxModel, Tensor};

/// Tiling parameters for [`tiled_field_inversion`].
#[derive(Clone, Copy, Debug)]
pub struct TileConfig {
    /// Output core size per axis — the region each patch contributes to the result.
    pub core: usize,
    /// Context margin (voxels) included on every side of each patch. Larger halos reduce
    /// tile-boundary artifacts at the cost of more work per patch.
    pub halo: usize,
}

impl Default for TileConfig {
    /// 128³ cores with an 8-voxel halo → 144³ patches (~0.4 GB of f32 activations for a typical
    /// net, comfortably within a conservative ~2 GB WASM budget). Empirically the halo barely
    /// affects accuracy here — the tiling error is dominated by *global* low-frequency drift
    /// (a patch can't see distant susceptibility sources), not tile-boundary seams — so this
    /// favours the large core (few patches, ~4× less overlap compute than 64³/32) over a big
    /// halo. On real 3 T data this matched whole-volume xQSM at r≈0.94 in ~1/4 the time.
    fn default() -> Self {
        Self { core: 128, halo: 8 }
    }
}

/// A core-aligned tile: `(x0, y0, z0, cx, cy, cz)` — origin + core extent (clamped at edges).
pub type Tile = (usize, usize, usize, usize, usize, usize);

/// Padded input-patch size per axis for a config and the net's `size_divisor`: `core + 2·halo`
/// rounded up to a multiple of `divisor`. Isotropic, so one value serves all axes. Callers use
/// this to build a reusable [`OnnxModel::plan_for`] plan of shape `[1, 1, p, p, p]`.
pub fn tile_patch_size(cfg: &TileConfig, divisor: usize) -> usize {
    (cfg.core.max(1) + 2 * cfg.halo).div_ceil(divisor.max(1)) * divisor.max(1)
}

/// Shared overlap-tiling driver. Enumerates the core-aligned tiles that actually touch `mask`
/// (all-background tiles are skipped → work is restricted to the mask bounding box for free),
/// runs each through `run_tile`, and scatters the results back into a full-volume buffer
/// (masked). Handles the empty-tile skip, parallel batching, and progress reporting so each
/// model only supplies its own per-patch logic.
///
/// `run_tile(&tile)` must return the tile's **post-processed core block** — row-major
/// `oi,oj,ok`, length `cx·cy·cz` — and be pure + `Sync` (it may run on many threads at once,
/// each holding one patch's activations in the shared wasm heap). `progress(done, total)` is
/// called from the driver thread only (the JS callback isn't `Sync`).
pub fn tiled_scatter(
    grid: &Grid,
    mask: &[u8],
    cfg: &TileConfig,
    run_tile: impl Fn(&Tile) -> Result<Vec<f64>, OnnxError> + Sync,
    mut progress: impl FnMut(usize, usize),
) -> Result<Vec<f64>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    assert_eq!(mask.len(), n, "mask length must match grid");
    let core = cfg.core.max(1);

    // Does the core block at (x0,y0,z0) cover any mask voxel?
    let core_has_mask = |x0: usize, y0: usize, z0: usize, cx: usize, cy: usize, cz: usize| {
        for oj in 0..cy {
            for ok in 0..cz {
                let row = x0 + nx * ((y0 + oj) + ny * (z0 + ok));
                if mask[row..row + cx].iter().any(|&m| m != 0) {
                    return true;
                }
            }
        }
        false
    };

    // Enumerate the core-aligned tiles that actually touch the mask.
    let mut tiles: Vec<Tile> = Vec::new();
    let mut x0 = 0usize;
    while x0 < nx {
        let cx = core.min(nx - x0);
        let mut y0 = 0usize;
        while y0 < ny {
            let cy = core.min(ny - y0);
            let mut z0 = 0usize;
            while z0 < nz {
                let cz = core.min(nz - z0);
                if core_has_mask(x0, y0, z0, cx, cy, cz) {
                    tiles.push((x0, y0, z0, cx, cy, cz));
                }
                z0 += core;
            }
            y0 += core;
        }
        x0 += core;
    }
    let total_tiles = tiles.len();
    progress(0, total_tiles);

    // Scatter a computed core block (row-major oi,oj,ok) into the full volume (masked).
    let write_tile = |chi: &mut [f64], &(x0, y0, z0, cx, cy, cz): &Tile, block: &[f64]| {
        for oi in 0..cx {
            for oj in 0..cy {
                for ok in 0..cz {
                    let dst = (x0 + oi) + nx * ((y0 + oj) + ny * (z0 + ok));
                    if mask[dst] != 0 {
                        chi[dst] = block[(oi * cy + oj) * cz + ok];
                    }
                }
            }
        }
    };

    let mut chi = vec![0.0f64; n];

    // Parallel path: run tiles in batches sized to the rayon pool, then write + report progress
    // from this (driver) thread. Falls back to sequential without the `parallel` feature.
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let batch = rayon::current_num_threads().max(1);
        let mut done = 0usize;
        for chunk in tiles.chunks(batch) {
            let blocks: Vec<Vec<f64>> = chunk.par_iter().map(&run_tile).collect::<Result<_, _>>()?;
            for (tile, block) in chunk.iter().zip(&blocks) {
                write_tile(&mut chi, tile, block);
                done += 1;
                progress(done, total_tiles);
            }
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        for (t, tile) in tiles.iter().enumerate() {
            let block = run_tile(tile)?;
            write_tile(&mut chi, tile, &block);
            progress(t + 1, total_tiles);
        }
    }
    Ok(chi)
}

/// Overlap-tile an **entire volume→volume algorithm** (not just one forward pass). For each
/// tile, a padded `p³` sub-volume of `field`/`mask` is cut out (with `halo` context, zero
/// outside the volume), `run_patch(field_patch, mask_patch, patch_grid)` is run on it, and the
/// central core is written back. Use this for the FFT-unrolled nets (lpcnn/modl-qsm/nextqsm)
/// whose Rust-side physics loop wraps a whole-volume CNN — running the whole algorithm per patch
/// bounds memory. Strongly off-design (the dipole/k-space step then sees only a patch), so results
/// are approximate; callers should warn and steer users to a full-volume run for real work.
#[allow(clippy::too_many_arguments)]
pub fn tiled_volume_algorithm(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    divisor: usize,
    cfg: &TileConfig,
    run_patch: impl Fn(&[f64], &[u8], &Grid) -> Result<Vec<f64>, OnnxError> + Sync,
    progress: impl FnMut(usize, usize),
) -> Result<Vec<f64>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    assert_eq!(field.len(), nx * ny * nz, "field length must match grid");
    let halo = cfg.halo;
    let (nxi, nyi, nzi) = (nx as i64, ny as i64, nz as i64);
    let p = tile_patch_size(cfg, divisor);
    let (vsx, vsy, vsz) = grid.voxel_size;
    let inside = move |x: i64, y: i64, z: i64| x >= 0 && x < nxi && y >= 0 && y < nyi && z >= 0 && z < nzi;

    let run_tile = move |&(x0, y0, z0, cx, cy, cz): &Tile| -> Result<Vec<f64>, OnnxError> {
        // Cut a column-major p³ sub-volume of field + mask (zero/empty outside the volume).
        let mut fpatch = vec![0.0f64; p * p * p];
        let mut mpatch = vec![0u8; p * p * p];
        for k in 0..p {
            let vz = z0 as i64 - halo as i64 + k as i64;
            for j in 0..p {
                let vy = y0 as i64 - halo as i64 + j as i64;
                for i in 0..p {
                    let vx = x0 as i64 - halo as i64 + i as i64;
                    if inside(vx, vy, vz) {
                        let s = vx as usize + nx * (vy as usize + ny * vz as usize);
                        let d = i + p * (j + p * k);
                        fpatch[d] = field[s];
                        mpatch[d] = mask[s];
                    }
                }
            }
        }
        let pgrid = Grid::new(p, p, p, vsx, vsy, vsz);
        let chi = run_patch(&fpatch, &mpatch, &pgrid)?;
        if chi.len() != p * p * p {
            return Err(OnnxError::Run(format!(
                "tiled algorithm returned {} voxels, expected {}", chi.len(), p * p * p
            )));
        }
        // Extract the central core (column-major) at offset `halo`.
        let mut core = vec![0.0f64; cx * cy * cz];
        for oi in 0..cx {
            for oj in 0..cy {
                for ok in 0..cz {
                    let src = (halo + oi) + p * ((halo + oj) + p * (halo + ok));
                    core[(oi * cy + oj) * cz + ok] = chi[src];
                }
            }
        }
        Ok(core)
    };

    tiled_scatter(grid, mask, cfg, run_tile, progress)
}

/// Run a fully-convolutional field→χ ONNX net patch-by-patch with a context halo, bounding
/// peak memory to a single patch.
///
/// Values are fed/read as f32 in the x-outer `NCDHW` layout the nets use; `pre` maps each
/// input field value, `post` maps each raw net output value. Each patch is zero-padded to a
/// multiple of `divisor` (the net's `size_divisor`) and to include the halo. The result is
/// masked, matching the whole-volume wrappers.
#[allow(clippy::too_many_arguments)]
pub fn tiled_field_inversion(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    model: &OnnxModel,
    divisor: usize,
    cfg: &TileConfig,
    pre: impl Fn(f64) -> f32 + Sync,
    post: impl Fn(f32) -> f64 + Sync,
    progress: impl FnMut(usize, usize),
) -> Result<Vec<f64>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    assert_eq!(field.len(), nx * ny * nz, "field length must match grid");
    let halo = cfg.halo;
    let (nxi, nyi, nzi) = (nx as i64, ny as i64, nz as i64);

    // Field sampler: zero outside the volume (edge context / padding).
    let at = |x: i64, y: i64, z: i64| -> f64 {
        if x >= 0 && x < nxi && y >= 0 && y < nyi && z >= 0 && z < nzi {
            field[x as usize + nx * (y as usize + ny * z as usize)]
        } else {
            0.0
        }
    };

    // One fixed patch shape for the whole run → the graph is optimized once and reused.
    let p = tile_patch_size(cfg, divisor);
    let plan = model.plan_for(&[&[1, 1, p, p, p]])?;

    // Per-patch: build the p³ input (NCDHW, x outer, z inner) sampling at (x0-halo+i, …), run the
    // net, and return the central core (offset `halo`) post-processed.
    let run_tile = move |&(x0, y0, z0, cx, cy, cz): &Tile| -> Result<Vec<f64>, OnnxError> {
        let mut inp = vec![0.0f32; p * p * p];
        for i in 0..p {
            let vx = x0 as i64 - halo as i64 + i as i64;
            for j in 0..p {
                let vy = y0 as i64 - halo as i64 + j as i64;
                let base = (i * p + j) * p;
                for k in 0..p {
                    let vz = z0 as i64 - halo as i64 + k as i64;
                    inp[base + k] = pre(at(vx, vy, vz));
                }
            }
        }
        let out = plan.run_single(&Tensor::new(vec![1, 1, p, p, p], inp))?;
        if out.shape != [1, 1, p, p, p] {
            return Err(OnnxError::Run(format!(
                "unexpected patch output shape {:?}, expected [1,1,{p},{p},{p}]",
                out.shape
            )));
        }
        let mut core_block = vec![0.0f64; cx * cy * cz];
        for oi in 0..cx {
            for oj in 0..cy {
                for ok in 0..cz {
                    let src = ((halo + oi) * p + (halo + oj)) * p + (halo + ok);
                    core_block[(oi * cy + oj) * cz + ok] = post(out.data[src]);
                }
            }
        }
        Ok(core_block)
    };

    tiled_scatter(grid, mask, cfg, run_tile, progress)
}

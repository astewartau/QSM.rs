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
//!
//! Tiles run in parallel with the `parallel` feature. One patch bounds memory only if the number
//! of *concurrent* patches is bounded too, which on a 32-bit WASM heap it has to be — see
//! [`tile_concurrency`].

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
    /// 128³ cores with an 8-voxel halo → 144³ patches. Empirically the halo barely affects
    /// accuracy here — the tiling error is dominated by *global* low-frequency drift (a patch
    /// can't see distant susceptibility sources), not tile-boundary seams — so this favours the
    /// large core (few patches, ~4× less overlap compute than 64³/32) over a big halo. On real
    /// 3 T data this matched whole-volume xQSM at r≈0.94 in ~1/4 the time.
    ///
    /// Sized for **native** memory: a 144³ patch costs ~3.1 GB of live f32 activations for xQSM
    /// (see [`ACTIVATION_BYTES_PER_PATCH_VOXEL`]), which leaves a 32-bit WASM host no room for a
    /// second one and not much for anything else. Browser hosts pass a far smaller config —
    /// QSMbly uses `core: 56, halo: 4` (64³ patches, ~270 MB each).
    fn default() -> Self {
        Self { core: 128, halo: 8 }
    }
}

/// Live f32 activation bytes one patch of a tiled net costs, **per patch voxel**.
///
/// Measured by walking the optimized `tract` plan in evaluation order and summing the outlets
/// still live at each step (the real peak, not the sum of all intermediates). At a 64³ patch:
/// xQSM 269.5 MB (1028 B/voxel), QSMnet 147.6 MB, NeXtQSM 134.9 MB, LPCNN 103.9 MB, IR2QSM
/// 93.6 MB. It is linear in patch voxels — xQSM measures the same 1028 B/voxel at 144³ — so one
/// constant covers every patch size. xQSM is both the heaviest and the usual default, so its
/// figure is the one to budget with, rounded up for the caller-side f64 patch buffers.
pub const ACTIVATION_BYTES_PER_PATCH_VOXEL: u64 = 1_100;

/// Activation budget for all concurrent tiles on WASM — see [`tile_concurrency`].
///
/// A threaded WASM host has one shared 32-bit heap with a hard 4 GB ceiling and an allocator
/// that cannot compact, and that heap also holds the model bytes, the full-volume input/output
/// and the host's own data. 1.5 GB leaves room for all of it plus fragmentation.
const WASM_TILE_ACTIVATION_BUDGET: u64 = 1_500_000_000;

/// Largest activation footprint a single patch may have on WASM, running alone.
///
/// [`tile_concurrency`] can bring the patches in flight down to one but not below, so a patch
/// bigger than this cannot run in a browser at all. The rest of the 4 GB heap is not free: a 104³
/// xQSM patch (1.24 GB by [`ACTIVATION_BYTES_PER_PATCH_VOXEL`]) running alone peaked at 1.74 GB,
/// so about 0.5 GB goes to the model, the volumes, the host and fragmentation. 2.5 GB keeps one
/// patch well clear of the 4.24 GB at which a run stalled (#89). It allows patches up to 131³ —
/// `core: 120, halo: 4` (128³) runs; the native 144³ default does not.
pub const WASM_MAX_PATCH_ACTIVATION_BYTES: u64 = 2_500_000_000;

/// Activation bytes one cubic patch of this edge costs. `u64`, because `p³ · bytes` overflows a
/// 32-bit `usize` for large patches.
fn patch_activation_bytes(patch_edge: usize) -> u64 {
    (patch_edge as u64).saturating_pow(3).saturating_mul(ACTIVATION_BYTES_PER_PATCH_VOXEL)
}

/// Largest patch edge that fits [`WASM_MAX_PATCH_ACTIVATION_BYTES`]: 131 voxels.
pub fn max_wasm_patch_edge() -> usize {
    let mut p = 1usize;
    while patch_activation_bytes(p + 1) <= WASM_MAX_PATCH_ACTIVATION_BYTES {
        p += 1;
    }
    p
}

/// Refuse a tile config whose patch is too big for a WASM heap even on its own.
///
/// Running such a patch does not fail cleanly. Near the ceiling the module stops making progress
/// without an error (#89), so it is refused up front with the largest core that would fit.
/// `divisor` is the net's size divisor, which [`tile_patch_size`] rounds the patch up to (pass 1
/// when there is none). The tiled drivers call this on WASM before building a plan; it is callable
/// anywhere, so a native host can check a config before handing it to a browser.
///
/// The footprint is [`ACTIVATION_BYTES_PER_PATCH_VOXEL`], xQSM's figure, for every net, so a
/// lighter net is refused somewhat before it would really run out — the same conservative
/// reckoning [`tile_concurrency`] uses.
pub fn check_wasm_patch_fits(cfg: &TileConfig, divisor: usize) -> Result<(), OnnxError> {
    let p = tile_patch_size(cfg, divisor);
    if patch_activation_bytes(p) <= WASM_MAX_PATCH_ACTIVATION_BYTES {
        return Ok(());
    }
    let d = divisor.max(1);
    let max_edge = max_wasm_patch_edge() / d * d;
    let advice = match max_edge.checked_sub(2 * cfg.halo) {
        Some(core) if core > 0 => format!("at halo {} the largest core that fits is {core}", cfg.halo),
        _ => format!("halo {} alone leaves no room for a core; reduce the halo", cfg.halo),
    };
    Err(OnnxError::Memory(format!(
        "a {p}³ tile patch needs about {:.1} GB of activations, more than a browser's 4 GB WASM \
         heap can hold even one tile at a time (patches up to {max_edge}³ fit); {advice}",
        patch_activation_bytes(p) as f64 / 1e9,
    )))
}

/// How many tiles [`tiled_scatter`] keeps in flight at once.
///
/// Natively this is just the rayon pool: memory is cheap and tile-level parallelism scales far
/// better than `tract`'s intra-op threading. On **WASM** the limit is memory, not cores. Every
/// concurrent tile holds its own set of activations in the single shared heap, so how much of the
/// 4 GB ceiling a tiled run occupies is set by the pool size and the patch size together — and by
/// nothing else, the volume included. Tiling a bigger volume adds tiles, not concurrent ones.
///
/// Measured in Chromium (cross-origin-isolated) before this cap existed, the wasm heap high-water
/// tracked the per-patch arithmetic: at 64³ patches, 1.1–1.4 GB with a 4-thread pool — 4 ×
/// 269.5 MB plus a ~11 MB base and allocator slack — and 2.6–3.4 GB with 14. Raising the core one
/// step is what makes it dangerous: `core: 96` (104³ patches, 1.24 GB each) on a 4-thread pool
/// peaked at **4.24 GB, 98.7% of the ceiling**. It did not abort there. It finished one
/// inference and then stopped making progress, never completing a second within 30 minutes —
/// a stall with no error, which is harder to diagnose than a crash. Nothing in the library
/// noticed (astewartau/QSM.rs#89).
///
/// So on WASM the count is whatever fits [`WASM_TILE_ACTIVATION_BUDGET`], and never more than the
/// pool. That bounds the heap by patch size rather than by core count: 5 tiles at 64³, and one at
/// a time once a single patch costs more than the budget.
pub fn tile_concurrency(cfg: &TileConfig) -> usize {
    #[cfg(not(feature = "parallel"))]
    {
        let _ = cfg;
        1
    }
    #[cfg(feature = "parallel")]
    {
        let pool = rayon::current_num_threads().max(1);
        #[cfg(not(target_family = "wasm"))]
        {
            let _ = cfg;
            pool
        }
        #[cfg(target_family = "wasm")]
        {
            pool.min(tiles_within_wasm_budget(cfg))
        }
    }
}

/// How many tiles of this config fit a browser's share of heap — at least one, however big the
/// patch. [`tile_concurrency`] applies this on WASM; it is callable anywhere so a native host can
/// tell whether a config it is about to hand a browser is feasible there.
///
/// Uses `core + 2·halo` rather than [`tile_patch_size`]'s divisor-rounded value: the two differ by
/// less than one divisor step, far below the precision of a memory budget. Arithmetic is in `u64`
/// because `p³ · bytes` overflows a 32-bit `usize` for large patches.
pub fn tiles_within_wasm_budget(cfg: &TileConfig) -> usize {
    let per_tile = patch_activation_bytes(cfg.core.max(1).saturating_add(2 * cfg.halo));
    (WASM_TILE_ACTIVATION_BUDGET / per_tile.max(1)).max(1) as usize
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
/// `oi,oj,ok`, length `cx·cy·cz` — and be pure + `Sync`: it runs on up to
/// [`tile_concurrency`] threads at once, each holding one patch's activations, which on WASM
/// share a single 4 GB heap. `progress(done, total)` is called from the driver thread only (the
/// JS callback isn't `Sync`).
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
    // Direct callers have no size divisor; the drivers above already checked the rounded patch.
    #[cfg(target_family = "wasm")]
    check_wasm_patch_fits(cfg, 1)?;
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

    // Parallel path: run tiles in batches, then write + report progress from this (driver)
    // thread. Falls back to sequential without the `parallel` feature.
    //
    // The batch size IS the memory bound — see `tile_concurrency`. Batching is how concurrency
    // gets capped: rayon has no per-call concurrency limit, and the obvious alternative (a
    // dedicated pool of that size) cannot be built on wasm32 at all, since
    // `ThreadPoolBuilder::build` needs `std::thread::spawn`. The cost is a barrier per batch —
    // the next wave waits for the slowest tile of this one — which is the price of not holding
    // more patches in the heap than fit.
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let batch = tile_concurrency(cfg);
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
    #[cfg(target_family = "wasm")]
    check_wasm_patch_fits(cfg, divisor)?;
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
    #[cfg(target_family = "wasm")]
    check_wasm_patch_fits(cfg, divisor)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The WASM budget has to translate patch size into a tile count that actually fits a 4 GB
    /// heap. Figures below are `(core + 2·halo)³ · ACTIVATION_BYTES_PER_PATCH_VOXEL` against
    /// `WASM_TILE_ACTIVATION_BUDGET`, i.e. what a browser can hold at once.
    #[test]
    fn wasm_tile_budget_tracks_patch_footprint() {
        // QSMbly's config: 64³ patches, ~288 MB each by the budget's reckoning (269.5 MB
        // measured) → five fit in 1.5 GB.
        assert_eq!(tiles_within_wasm_budget(&TileConfig { core: 56, halo: 4 }), 5);
        // Doubling the patch edge is 8× the memory, so the count collapses fast: 32³ patches are
        // 36 MB each (41 fit), 128³ are 2.3 GB (one, over budget on its own).
        assert_eq!(tiles_within_wasm_budget(&TileConfig { core: 24, halo: 4 }), 41);
        assert_eq!(tiles_within_wasm_budget(&TileConfig { core: 120, halo: 4 }), 1);
        // A patch that alone exceeds the budget still has to be attempted, not divided by zero:
        // 144³ is ~3.1 GB, and the crate's own native default is 192³ on top of that.
        assert_eq!(tiles_within_wasm_budget(&TileConfig { core: 128, halo: 8 }), 1);
        assert_eq!(tiles_within_wasm_budget(&TileConfig { core: 192, halo: 32 }), 1);
        // Degenerate configs must not panic, divide by zero, or overflow into a wrong answer. A
        // patch too small to matter leaves the pool as the only limit; `usize::MAX` saturates to
        // one tile rather than wrapping to a large count.
        assert!(tiles_within_wasm_budget(&TileConfig { core: 0, halo: 0 }) > 1);
        assert_eq!(tiles_within_wasm_budget(&TileConfig { core: usize::MAX, halo: 0 }), 1);

        // The count is monotonically non-increasing in patch size — the property the cap relies
        // on, and the thing a wrong unit or an inverted ratio would break.
        let mut prev = usize::MAX;
        for core in (8..=200).step_by(8) {
            let n = tiles_within_wasm_budget(&TileConfig { core, halo: 4 });
            assert!(n <= prev, "count rose from {prev} to {n} at core {core}");
            assert!(n >= 1);
            prev = n;
        }
    }

    /// A patch that cannot fit a WASM heap even alone is refused with the largest core that would,
    /// and everything a browser actually runs is let through.
    #[test]
    fn wasm_refuses_a_patch_too_big_to_run_alone() {
        assert_eq!(max_wasm_patch_edge(), 131);
        // QSMbly's default, and the largest whole-divisor patch under the limit for xQSM (/8).
        assert!(check_wasm_patch_fits(&TileConfig { core: 56, halo: 4 }, 8).is_ok());
        assert!(check_wasm_patch_fits(&TileConfig { core: 120, halo: 4 }, 8).is_ok());
        // One voxel more rounds up to a 136³ patch, which does not fit.
        assert!(check_wasm_patch_fits(&TileConfig { core: 121, halo: 4 }, 8).is_err());
        // Without a divisor, 131³ fits and 132³ does not.
        assert!(check_wasm_patch_fits(&TileConfig { core: 123, halo: 4 }, 1).is_ok());
        assert!(check_wasm_patch_fits(&TileConfig { core: 124, halo: 4 }, 1).is_err());

        // The crate's native default (144³) is refused, naming the core that would fit at its halo:
        // 128 (131 rounded down to /8) less 2 × 8.
        match check_wasm_patch_fits(&TileConfig::default(), 8) {
            Err(OnnxError::Memory(m)) => {
                assert!(m.contains("144³"), "{m}");
                assert!(m.contains("largest core that fits is 112"), "{m}");
            }
            other => panic!("expected a memory error, got {other:?}"),
        }
        // A halo that leaves no room for any core says so, rather than suggesting core 0.
        match check_wasm_patch_fits(&TileConfig { core: 8, halo: 64 }, 8) {
            Err(OnnxError::Memory(m)) => assert!(m.contains("reduce the halo"), "{m}"),
            other => panic!("expected a memory error, got {other:?}"),
        }
    }

    /// `tile_concurrency` never exceeds the pool, and never reports zero tiles in flight.
    #[test]
    fn tile_concurrency_is_bounded_by_the_pool() {
        for cfg in [
            TileConfig { core: 56, halo: 4 },
            TileConfig { core: 128, halo: 8 },
            TileConfig::default(),
        ] {
            let n = tile_concurrency(&cfg);
            assert!(n >= 1);
            #[cfg(feature = "parallel")]
            assert!(n <= rayon::current_num_threads().max(1));
            #[cfg(not(feature = "parallel"))]
            assert_eq!(n, 1);
        }
    }
}

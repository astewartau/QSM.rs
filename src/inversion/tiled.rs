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
use crate::models::TileCost;

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
    /// Sized for **native** memory: a 144³ patch of xQSM holds about 3.6 GB while it runs (see
    /// [`ModelSpec::tile_bytes_per_voxel`](crate::models::ModelSpec::tile_bytes_per_voxel)), which a 32-bit WASM host cannot fit at all. Browser
    /// hosts pass a far smaller config — QSMbly uses `core: 56, halo: 4` (64³ patches).
    fn default() -> Self {
        Self { core: 128, halo: 8 }
    }
}

/// Heap budget for all concurrent tiles on WASM — see [`tile_concurrency`].
///
/// A threaded WASM host has one shared 32-bit heap with a hard 4 GB ceiling and an allocator
/// that cannot compact, and that heap also holds the model bytes, the full-volume inputs and
/// outputs and the host's own data. 1.5 GB of tiles leaves room for all of it plus fragmentation.
pub const WASM_TILE_ACTIVATION_BUDGET: u64 = 1_500_000_000;

/// Largest footprint a single patch may have on WASM, running alone.
///
/// [`tile_concurrency`] can bring the patches in flight down to one but not below, so a patch
/// bigger than this cannot run in a browser at all. The rest of the 4 GB heap is not free: a 104³
/// xQSM patch running alone peaked at 1.74 GB, so about 0.5 GB goes to the model, the volumes, the
/// host and fragmentation. 2.5 GB keeps one patch well clear of the 4.24 GB at which a run stalled
/// (#89).
pub const WASM_MAX_PATCH_ACTIVATION_BYTES: u64 = 2_500_000_000;

/// The cost a tiled network's wrapper budgets with, from its registry entry
/// ([`ModelSpec::tile_cost`](crate::models::ModelSpec::tile_cost)). Every tiled network has one (a
/// test below checks), so a missing figure is a bug in the registry, not a runtime condition.
pub(crate) fn tile_cost_of(model_id: &str) -> TileCost {
    crate::models::find_model(model_id)
        .and_then(|m| m.tile_cost())
        .unwrap_or_else(|| panic!("{model_id} is tiled but has no tile_bytes_per_voxel in the registry"))
}

/// Heap one tile of this config holds while it runs, for a network of this cost
/// ([`ModelSpec::tile_cost`](crate::models::ModelSpec::tile_cost)): its patch, rounded up to the
/// network's divisor as [`tile_patch_size`] does, times the network's bytes per patch voxel. `u64`,
/// because `p³ · bytes` overflows a 32-bit `usize` for large patches.
pub fn tile_bytes(cfg: &TileConfig, cost: &TileCost) -> u64 {
    let d = cost.divisor.max(1);
    let edge = cfg.core.max(1).saturating_add(cfg.halo.saturating_mul(2)).div_ceil(d).saturating_mul(d);
    patch_bytes(edge, cost.bytes_per_voxel)
}

/// Heap one cubic patch of this edge holds.
fn patch_bytes(patch_edge: usize, bytes_per_voxel: u64) -> u64 {
    (patch_edge as u64).saturating_pow(3).saturating_mul(bytes_per_voxel)
}

/// Largest patch edge that fits [`WASM_MAX_PATCH_ACTIVATION_BYTES`] for a network costing
/// `bytes_per_voxel` per patch voxel.
pub fn max_wasm_patch_edge(bytes_per_voxel: u64) -> usize {
    let mut p = 1usize;
    while patch_bytes(p + 1, bytes_per_voxel) <= WASM_MAX_PATCH_ACTIVATION_BYTES {
        p += 1;
    }
    p
}

/// Refuse a tile config whose patch is too big for a WASM heap even on its own.
///
/// Running such a patch does not fail cleanly. Near the ceiling the module stops making progress
/// without an error (#89), so it is refused up front with the largest core that would fit.
/// `cost` is the network's ([`ModelSpec::tile_cost`](crate::models::ModelSpec::tile_cost)): its
/// size divisor, which [`tile_patch_size`] rounds the patch up to, and its bytes per patch voxel.
/// The tiled drivers call this on WASM before building a plan; it is callable anywhere, so a
/// native host can check a config before handing it to a browser.
pub fn check_wasm_patch_fits(cfg: &TileConfig, cost: &TileCost) -> Result<(), OnnxError> {
    let bytes = tile_bytes(cfg, cost);
    if bytes <= WASM_MAX_PATCH_ACTIVATION_BYTES {
        return Ok(());
    }
    let p = tile_patch_size(cfg, cost.divisor);
    let d = cost.divisor.max(1);
    let max_edge = max_wasm_patch_edge(cost.bytes_per_voxel) / d * d;
    let advice = match max_edge.checked_sub(2 * cfg.halo) {
        Some(core) if core > 0 => format!("at halo {} the largest core that fits is {core}", cfg.halo),
        _ => format!("halo {} alone leaves no room for a core; reduce the halo", cfg.halo),
    };
    Err(OnnxError::Memory(format!(
        "a {p}³ tile patch needs about {:.1} GB, more than a browser's 4 GB WASM heap can hold even \
         one tile at a time (patches up to {max_edge}³ fit for this network); {advice}",
        bytes as f64 / 1e9,
    )))
}

/// How many tiles [`tiled_scatter`] keeps in flight at once.
///
/// Natively this is just the rayon pool: memory is cheap and tile-level parallelism scales far
/// better than `tract`'s intra-op threading. On **WASM** the limit is memory, not cores. Every
/// concurrent tile holds its own activations in the single shared heap, so how much of the 4 GB
/// ceiling a tiled run occupies is set by the pool size, the patch size and the network together —
/// and by nothing else, the volume included. Tiling a bigger volume adds tiles, not concurrent
/// ones.
///
/// Measured in Chromium (cross-origin-isolated) before this cap existed, the wasm heap high-water
/// tracked the per-patch arithmetic: xQSM at 64³ patches, 1.1–1.4 GB with a 4-thread pool and
/// 2.6–3.4 GB with 14. Raising the core one step is what makes it dangerous: `core: 96` (104³
/// patches) on a 4-thread pool peaked at **4.24 GB, 98.7% of the ceiling**. It did not abort
/// there. It finished one inference and then stopped making progress, never completing a second
/// within 30 minutes — a stall with no error, which is harder to diagnose than a crash. Nothing in
/// the library noticed (astewartau/QSM.rs#89).
///
/// So on WASM the count is [`wasm_tiles_in_flight`]: whatever fits
/// [`WASM_TILE_ACTIVATION_BUDGET`] at the network's own cost per tile, and never more than the
/// pool. The cost differs a lot between networks — a 64³ tile of BFRnet holds about three times
/// what one of xQSM does — so one network's figure cannot stand in for the others.
pub fn tile_concurrency(cfg: &TileConfig, cost: &TileCost) -> usize {
    #[cfg(not(feature = "parallel"))]
    {
        let _ = (cfg, cost);
        1
    }
    #[cfg(feature = "parallel")]
    {
        let pool = rayon::current_num_threads().max(1);
        #[cfg(not(target_family = "wasm"))]
        {
            let _ = (cfg, cost);
            pool
        }
        #[cfg(target_family = "wasm")]
        {
            wasm_tiles_in_flight(cfg, cost, pool)
        }
    }
}

/// Tiles a WASM host with `threads` workers keeps in flight: the threads, capped by
/// [`tiles_within_wasm_budget`]. This is what [`tile_concurrency`] returns on WASM; it is callable
/// anywhere, so a host estimating a browser run's memory can use the same count.
pub fn wasm_tiles_in_flight(cfg: &TileConfig, cost: &TileCost, threads: usize) -> usize {
    threads.max(1).min(tiles_within_wasm_budget(cfg, cost))
}

/// How many tiles of this config fit a browser's share of heap — at least one, however big the
/// patch — for a network of this cost ([`ModelSpec::tile_cost`](crate::models::ModelSpec::tile_cost)).
pub fn tiles_within_wasm_budget(cfg: &TileConfig, cost: &TileCost) -> usize {
    (WASM_TILE_ACTIVATION_BUDGET / tile_bytes(cfg, cost).max(1)).max(1) as usize
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
/// share a single 4 GB heap. `cost` is what one tile of the network costs
/// ([`ModelSpec::tile_cost`](crate::models::ModelSpec::tile_cost)), which sets how many fit at
/// once. `progress(done, total)` is called from the driver thread only
/// (the JS callback isn't `Sync`).
pub fn tiled_scatter(
    grid: &Grid,
    mask: &[u8],
    cfg: &TileConfig,
    cost: &TileCost,
    run_tile: impl Fn(&Tile) -> Result<Vec<f64>, OnnxError> + Sync,
    mut progress: impl FnMut(usize, usize),
) -> Result<Vec<f64>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    assert_eq!(mask.len(), n, "mask length must match grid");
    #[cfg(target_family = "wasm")]
    check_wasm_patch_fits(cfg, cost)?;
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
        let batch = tile_concurrency(cfg, cost);
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
        let _ = cost;
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
/// `cost` is the network's tile cost, as for [`tiled_scatter`]; its divisor pads each patch.
#[allow(clippy::too_many_arguments)]
pub fn tiled_volume_algorithm(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    cost: &TileCost,
    cfg: &TileConfig,
    run_patch: impl Fn(&[f64], &[u8], &Grid) -> Result<Vec<f64>, OnnxError> + Sync,
    progress: impl FnMut(usize, usize),
) -> Result<Vec<f64>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    assert_eq!(field.len(), nx * ny * nz, "field length must match grid");
    let halo = cfg.halo;
    let (nxi, nyi, nzi) = (nx as i64, ny as i64, nz as i64);
    #[cfg(target_family = "wasm")]
    check_wasm_patch_fits(cfg, cost)?;
    let p = tile_patch_size(cfg, cost.divisor);
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

    tiled_scatter(grid, mask, cfg, cost, run_tile, progress)
}

/// Run a fully-convolutional field→χ ONNX net patch-by-patch with a context halo, bounding
/// peak memory to a single patch.
///
/// Values are fed/read as f32 in the x-outer `NCDHW` layout the nets use; `pre` maps each
/// input field value, `post` maps each raw net output value. Each patch is zero-padded to a
/// multiple of `cost.divisor` (the net's `size_divisor`) and to include the halo. The result is
/// masked, matching the whole-volume wrappers. `cost` is the network's tile cost, as for
/// [`tiled_scatter`].
#[allow(clippy::too_many_arguments)]
pub fn tiled_field_inversion(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    model: &OnnxModel,
    cost: &TileCost,
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
    check_wasm_patch_fits(cfg, cost)?;
    let p = tile_patch_size(cfg, cost.divisor);
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

    tiled_scatter(grid, mask, cfg, cost, run_tile, progress)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::all_models;

    /// The networks QSM-Core runs tiled, with the divisor each one's wrapper pads its patches to.
    const TILED: [(&str, usize); 11] = [
        ("bfrnet", 8),
        ("xqsm", 8),
        ("qsmnet", 16),
        ("qsmnet-plus", 16),
        ("ir2qsm", 8),
        ("lpcnn", 8),
        ("modl-qsm", 1),
        ("nextqsm", 64),
        ("iqsm", 16),
        ("iqsm-plus", 16),
        ("iqfm", 16),
    ];

    /// QSMbly's tile config: 64³ patches.
    const BROWSER: TileConfig = TileConfig { core: 56, halo: 4 };

    /// A cost for testing the arithmetic on its own: xQSM-sized, no divisor.
    const ARITH: TileCost = TileCost { divisor: 1, bytes_per_voxel: 1_100 };

    /// Exactly the tiled networks carry a tile cost, each with the divisor its wrapper pads to,
    /// so every wrapper's `tile_cost_of` finds one and none budgets with another's figure.
    #[test]
    fn every_tiled_network_has_its_own_tile_cost() {
        for m in all_models() {
            match TILED.iter().find(|(id, _)| *id == m.id) {
                Some(&(id, divisor)) => {
                    let cost = m.tile_cost().unwrap_or_else(|| panic!("{id} is tiled but has no figure"));
                    assert_eq!(cost.divisor, divisor, "{id}");
                    assert_eq!(tile_cost_of(id), cost);
                }
                None => assert!(m.tile_cost().is_none(), "{} is not tiled but has a tile cost", m.id),
            }
        }
        // QSMnet and QSMnet+ share an architecture and one tiled wrapper, which budgets with
        // QSMnet's entry.
        assert_eq!(tile_cost_of("qsmnet"), tile_cost_of("qsmnet-plus"));
    }

    /// For every tiled network, at the browser config and around it, the tiles a WASM pool keeps
    /// in flight times what one of them costs fits the budget: more than one tile is only let
    /// through when they all fit, and a lone tile is either inside the budget or refused by
    /// `check_wasm_patch_fits` if it cannot run even alone.
    #[test]
    fn every_tiled_network_fits_the_budget_at_its_cap() {
        let configs = [
            BROWSER,
            TileConfig { core: 24, halo: 4 },
            TileConfig { core: 40, halo: 8 },
            TileConfig { core: 88, halo: 4 },
            TileConfig { core: 120, halo: 4 },
            TileConfig::default(),
        ];
        for (id, _) in TILED {
            let cost = tile_cost_of(id);
            for cfg in configs {
                let one = tile_bytes(&cfg, &cost);
                for threads in [1, 2, 4, 8, 16] {
                    let n = wasm_tiles_in_flight(&cfg, &cost, threads);
                    assert!((1..=threads).contains(&n), "{id} {cfg:?} on {threads} threads: {n}");
                    if n > 1 {
                        assert!(
                            n as u64 * one <= WASM_TILE_ACTIVATION_BUDGET,
                            "{id} {cfg:?}: {n} tiles of {one} B exceed the budget"
                        );
                    } else if one > WASM_TILE_ACTIVATION_BUDGET {
                        assert_eq!(
                            check_wasm_patch_fits(&cfg, &cost).is_ok(),
                            one <= WASM_MAX_PATCH_ACTIVATION_BYTES,
                            "{id} {cfg:?}"
                        );
                    }
                }
            }
        }
    }

    /// BFRnet was budgeted with xQSM's figure (1 100 B per patch voxel), which let a 4-thread
    /// browser run four of its 64³ tiles at once: about 4 GB of tiles on its own figure. On its
    /// own figure it runs fewer.
    #[test]
    fn bfrnet_runs_fewer_tiles_at_once_than_on_the_shared_figure() {
        let shared = TileCost { divisor: 8, bytes_per_voxel: 1_100 };
        assert_eq!(wasm_tiles_in_flight(&BROWSER, &shared, 4), 4);
        let bfrnet = tile_cost_of("bfrnet");
        let n = wasm_tiles_in_flight(&BROWSER, &bfrnet, 4);
        assert!(n < 4, "BFRnet still runs {n} tiles at once on 4 threads");
        assert_eq!(n, BFRNET_CAP_AT_FOUR);
        // What the four tiles the shared figure allowed would hold by BFRnet's own measure.
        assert!(4 * tile_bytes(&BROWSER, &bfrnet) > WASM_TILE_ACTIVATION_BUDGET);
    }
    const BFRNET_CAP_AT_FOUR: usize = 1;

    /// A tile's bytes follow the patch the network really runs: rounded up to its divisor.
    #[test]
    fn tile_bytes_follow_the_rounded_patch() {
        // 56 + 2·4 = 64 is a multiple of 8 and of 64; 40 + 2·4 = 48 is not a multiple of 64, so
        // NeXtQSM pads it to 64³.
        let c8 = TileCost { divisor: 8, bytes_per_voxel: 1_000 };
        let c64 = TileCost { divisor: 64, bytes_per_voxel: 1_000 };
        assert_eq!(tile_bytes(&BROWSER, &c8), 64 * 64 * 64 * 1_000);
        assert_eq!(tile_bytes(&BROWSER, &c64), 64 * 64 * 64 * 1_000);
        let small = TileConfig { core: 40, halo: 4 };
        assert_eq!(tile_bytes(&small, &c8), 48 * 48 * 48 * 1_000);
        assert_eq!(tile_bytes(&small, &c64), 64 * 64 * 64 * 1_000);
        // Saturates rather than overflowing.
        assert_eq!(tile_bytes(&TileConfig { core: usize::MAX, halo: 8 }, &c8), u64::MAX);
    }

    /// The WASM budget has to translate patch size into a tile count that actually fits a 4 GB
    /// heap. Figures below are `(core + 2·halo)³ · 1 100 B` against `WASM_TILE_ACTIVATION_BUDGET`,
    /// i.e. what a browser can hold at once.
    #[test]
    fn wasm_tile_budget_tracks_patch_footprint() {
        // 64³ patches, ~288 MB each → five fit in 1.5 GB.
        assert_eq!(tiles_within_wasm_budget(&BROWSER, &ARITH), 5);
        // Doubling the patch edge is 8× the memory, so the count collapses fast: 32³ patches are
        // 36 MB each (41 fit), 128³ are 2.3 GB (one, over budget on its own).
        assert_eq!(tiles_within_wasm_budget(&TileConfig { core: 24, halo: 4 }, &ARITH), 41);
        assert_eq!(tiles_within_wasm_budget(&TileConfig { core: 120, halo: 4 }, &ARITH), 1);
        // A patch that alone exceeds the budget still has to be attempted, not divided by zero:
        // 144³ is ~3.3 GB, and the crate's own native default is 192³ on top of that.
        assert_eq!(tiles_within_wasm_budget(&TileConfig { core: 128, halo: 8 }, &ARITH), 1);
        assert_eq!(tiles_within_wasm_budget(&TileConfig { core: 192, halo: 32 }, &ARITH), 1);
        // Degenerate configs must not panic, divide by zero, or overflow into a wrong answer. A
        // patch too small to matter leaves the pool as the only limit; `usize::MAX` saturates to
        // one tile rather than wrapping to a large count.
        assert!(tiles_within_wasm_budget(&TileConfig { core: 0, halo: 0 }, &ARITH) > 1);
        assert_eq!(tiles_within_wasm_budget(&TileConfig { core: usize::MAX, halo: 0 }, &ARITH), 1);

        // The count is monotonically non-increasing in patch size — the property the cap relies
        // on, and the thing a wrong unit or an inverted ratio would break.
        let mut prev = usize::MAX;
        for core in (8..=200).step_by(8) {
            let n = tiles_within_wasm_budget(&TileConfig { core, halo: 4 }, &ARITH);
            assert!(n <= prev, "count rose from {prev} to {n} at core {core}");
            assert!(n >= 1);
            prev = n;
        }
        // And in the network's cost: a heavier network never fits more tiles.
        let heavier = TileCost { bytes_per_voxel: 3 * ARITH.bytes_per_voxel, ..ARITH };
        assert!(tiles_within_wasm_budget(&BROWSER, &heavier) < tiles_within_wasm_budget(&BROWSER, &ARITH));
    }

    /// A patch that cannot fit a WASM heap even alone is refused with the largest core that would,
    /// and everything a browser actually runs is let through.
    #[test]
    fn wasm_refuses_a_patch_too_big_to_run_alone() {
        let by8 = TileCost { divisor: 8, ..ARITH };
        assert_eq!(max_wasm_patch_edge(ARITH.bytes_per_voxel), 131);
        // The browser config, and the largest whole-divisor patch under the limit at /8.
        assert!(check_wasm_patch_fits(&BROWSER, &by8).is_ok());
        assert!(check_wasm_patch_fits(&TileConfig { core: 120, halo: 4 }, &by8).is_ok());
        // One voxel more rounds up to a 136³ patch, which does not fit.
        assert!(check_wasm_patch_fits(&TileConfig { core: 121, halo: 4 }, &by8).is_err());
        // Without a divisor, 131³ fits and 132³ does not.
        assert!(check_wasm_patch_fits(&TileConfig { core: 123, halo: 4 }, &ARITH).is_ok());
        assert!(check_wasm_patch_fits(&TileConfig { core: 124, halo: 4 }, &ARITH).is_err());

        // The crate's native default (144³) is refused, naming the core that would fit at its halo:
        // 128 (131 rounded down to /8) less 2 × 8.
        match check_wasm_patch_fits(&TileConfig::default(), &by8) {
            Err(OnnxError::Memory(m)) => {
                assert!(m.contains("144³"), "{m}");
                assert!(m.contains("largest core that fits is 112"), "{m}");
            }
            other => panic!("expected a memory error, got {other:?}"),
        }
        // A halo that leaves no room for any core says so, rather than suggesting core 0.
        match check_wasm_patch_fits(&TileConfig { core: 8, halo: 64 }, &by8) {
            Err(OnnxError::Memory(m)) => assert!(m.contains("reduce the halo"), "{m}"),
            other => panic!("expected a memory error, got {other:?}"),
        }
        // A heavier network is refused at a smaller patch.
        assert!(max_wasm_patch_edge(3 * ARITH.bytes_per_voxel) < 131);
    }

    /// `tile_concurrency` never exceeds the pool, and never reports zero tiles in flight.
    #[test]
    fn tile_concurrency_is_bounded_by_the_pool() {
        for cfg in [BROWSER, TileConfig { core: 128, halo: 8 }, TileConfig::default()] {
            for (id, _) in TILED {
                let n = tile_concurrency(&cfg, &tile_cost_of(id));
                assert!(n >= 1);
                #[cfg(feature = "parallel")]
                assert!(n <= rayon::current_num_threads().max(1));
                #[cfg(not(feature = "parallel"))]
                assert_eq!(n, 1);
            }
        }
    }
}

//! How much heap a tiled network's tiles take, and how many a browser keeps in flight.
//!
//! The arithmetic the tiled drivers (`inversion::tiled`, `onnx` feature) budget with, kept out of
//! the `onnx` gate so a host that only estimates memory (and does not run networks) computes the
//! same figures. A network's cost per tile comes from the registry
//! ([`ModelSpec::tile_cost`](crate::models::ModelSpec::tile_cost)).

use crate::models::TileCost;

/// Tiling parameters for the tiled drivers (`tiled_field_inversion`, `tiled_volume_algorithm`).
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
    /// Sized for **native** memory: a 144³ patch of xQSM holds about 3.4 GB while it runs (see
    /// [`ModelSpec::tile_bytes_per_voxel`](crate::models::ModelSpec::tile_bytes_per_voxel)), which a 32-bit WASM host cannot fit at all. Browser
    /// hosts pass a far smaller config — QSMbly uses `core: 56, halo: 4` (64³ patches).
    fn default() -> Self {
        Self { core: 128, halo: 8 }
    }
}

/// Heap budget for all concurrent tiles on WASM — see `tile_concurrency` (`onnx` feature).
///
/// A threaded WASM host has one shared 32-bit heap with a hard 4 GB ceiling and an allocator
/// that cannot compact, and that heap also holds the model bytes, the full-volume inputs and
/// outputs and the host's own data. 1.5 GB of tiles leaves room for all of it plus fragmentation.
pub const WASM_TILE_ACTIVATION_BUDGET: u64 = 1_500_000_000;

/// Largest footprint a single patch may have on WASM, running alone.
///
/// `tile_concurrency` (`onnx` feature) can bring the patches in flight down to one but not below, so a patch
/// bigger than this cannot run in a browser at all. The rest of the 4 GB heap is not free: a 104³
/// xQSM patch running alone peaked at 1.74 GB, so about 0.5 GB goes to the model, the volumes, the
/// host and fragmentation. 2.5 GB keeps one patch well clear of the 4.24 GB at which a run stalled
/// (#89).
pub const WASM_MAX_PATCH_ACTIVATION_BYTES: u64 = 2_500_000_000;

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

/// Tiles a WASM host with `threads` workers keeps in flight: the threads, capped by
/// [`tiles_within_wasm_budget`]. This is what `tile_concurrency` (`onnx` feature) returns on WASM; it is callable
/// anywhere, so a host estimating a browser run's memory can use the same count.
pub fn wasm_tiles_in_flight(cfg: &TileConfig, cost: &TileCost, threads: usize) -> usize {
    threads.max(1).min(tiles_within_wasm_budget(cfg, cost))
}

/// How many tiles of this config fit a browser's share of heap — at least one, however big the
/// patch — for a network of this cost ([`ModelSpec::tile_cost`](crate::models::ModelSpec::tile_cost)).
pub fn tiles_within_wasm_budget(cfg: &TileConfig, cost: &TileCost) -> usize {
    (WASM_TILE_ACTIVATION_BUDGET / tile_bytes(cfg, cost).max(1)).max(1) as usize
}

/// Padded input-patch size per axis for a config and the net's `size_divisor`: `core + 2·halo`
/// rounded up to a multiple of `divisor`. Isotropic, so one value serves all axes. Callers use
/// this to build a reusable `OnnxModel::plan_for` plan of shape `[1, 1, p, p, p]`.
pub fn tile_patch_size(cfg: &TileConfig, divisor: usize) -> usize {
    (cfg.core.max(1) + 2 * cfg.halo).div_ceil(divisor.max(1)) * divisor.max(1)
}

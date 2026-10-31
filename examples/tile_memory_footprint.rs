//! Heap one tile of each tiled network holds while it runs, per voxel of its patch.
//!
//! This is where every [`qsm_core::models::ModelSpec::tile_bytes_per_voxel`] comes from, and how
//! to re-derive them after a `tract` upgrade, a change to a tiled wrapper, or for a new tiled net.
//!
//! Each network's own tiled entry point runs single-threaded (built without `parallel`, so exactly
//! one tile is in flight) on an 80³ volume whose mask touches only the first tile. A counting
//! global allocator records the peak heap above what was allocated before the call. That is done
//! at three patch sizes (48³, 64³, 80³; 64³ and 128³ for NeXtQSM, whose patches are multiples of
//! 64), and the least-squares slope against patch voxels is what each extra tile in flight costs
//! per voxel: the tile's activations and everything else the wrapper allocates per tile (input
//! and output patches, and for the unrolled nets their FFT buffers). The intercept (parsed model,
//! optimised plan, full-volume buffers) is paid once per run by the nets that plan once, but once
//! per tile by LPCNN, MoDL-QSM and NeXtQSM, which load their networks inside each patch; for them
//! the tool also prints a 64³ tile's whole peak per patch voxel.
//!
//! Run (weights as fetched by `scripts/fetch_model_weights.py <dir> all`):
//!   cargo run --release --example tile_memory_footprint --features onnx -- <weights dir> [model id…]
//!
//! With `--features "onnx parallel"` and `RAYON_NUM_THREADS=k`, it instead runs k 64³ tiles at
//! once (natively the tiles in flight are the pool) and prints the peak, to set beside
//! `fixed + k · slope · 64³` from the single-threaded run: a check that tiles in flight add up as
//! the budget in `qsm_core::inversion::tiled` assumes.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use qsm_core::grid::Grid;
use qsm_core::inversion::{self, TileConfig};

struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(by: usize) {
    let now = CURRENT.fetch_add(by, Relaxed) + by;
    PEAK.fetch_max(now, Relaxed);
}

// SAFETY: every call forwards to `System` unchanged; the counters are only bookkeeping.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            grew(l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(l);
        if !p.is_null() {
            grew(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l);
        CURRENT.fetch_sub(l.size(), Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = System.realloc(p, l, new);
        if !q.is_null() {
            if new >= l.size() {
                grew(new - l.size());
            } else {
                CURRENT.fetch_sub(l.size() - new, Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Peak heap above the current level while `f` runs.
fn peak_of<T>(f: impl FnOnce() -> T) -> (usize, T) {
    let base = CURRENT.load(Relaxed);
    PEAK.store(base, Relaxed);
    let out = f();
    (PEAK.load(Relaxed) - base, out)
}

const TILED: [&str; 11] = [
    "bfrnet", "xqsm", "qsmnet", "qsmnet-plus", "ir2qsm", "lpcnn", "modl-qsm", "nextqsm", "iqsm",
    "iqsm-plus", "iqfm",
];

/// Run `id`'s tiled entry point once on `field`/`mask` over `grid` with `cfg`.
fn run(id: &str, w: &[Vec<u8>], field: &[f64], mask: &[u8], grid: &Grid, cfg: &TileConfig) -> usize {
    let bdir = (0.0, 0.0, 1.0);
    let none = |_: usize, _: usize| {};
    let (te, b0) = (0.02, 3.0);
    let out = match id {
        "bfrnet" => qsm_core::bgremove::bfrnet::bfrnet_tiled(field, mask, grid, &w[0], cfg, none),
        "xqsm" => inversion::xqsm_tiled(field, mask, grid, &w[0], cfg, none),
        "qsmnet" => inversion::qsmnet_tiled(field, mask, grid, &w[0], &inversion::QsmnetNorm::qsmnet(), cfg, none),
        "qsmnet-plus" => {
            inversion::qsmnet_tiled(field, mask, grid, &w[0], &inversion::QsmnetNorm::qsmnet_plus(), cfg, none)
        }
        "ir2qsm" => inversion::ir2qsm::ir2qsm_tiled(field, mask, grid, &w[0], cfg, none),
        "lpcnn" => inversion::lpcnn::lpcnn_tiled(field, mask, grid, bdir, &w[0], cfg, none),
        "modl-qsm" => inversion::modl_qsm::modl_qsm_tiled(field, mask, grid, bdir, &w[0], cfg, none),
        "nextqsm" => inversion::nextqsm::nextqsm_tiled(field, mask, grid, bdir, &w[0], &w[1], cfg, none),
        // One echo: echoes run one after another, so more of them add no tile in flight.
        "iqsm" => inversion::iqsm_multi_echo(&[field], &[], mask, grid, &[te], b0, 1.0, 0, &w[0], Some(cfg), Default::default()),
        "iqsm-plus" => {
            inversion::iqsm_plus_multi_echo(&[field], &[], mask, grid, &[te], b0, bdir, 1.0, 0, &w[0], Some(cfg))
        }
        "iqfm" => inversion::iqfm_multi_echo(&[field], &[], mask, grid, &[te], b0, 1.0, 0, &w[0], Some(cfg), Default::default()),
        _ => panic!("{id} is not a tiled network"),
    };
    out.unwrap_or_else(|e| panic!("{id}: {e}")).len()
}

/// A smooth field (the values only need to look like data) and a mask that touches only the
/// first `tiles` tiles along x.
fn inputs(dims: (usize, usize, usize), core: usize, tiles: usize) -> (Vec<f64>, Vec<u8>, Grid) {
    let (nx, ny, nz) = dims;
    let n = nx * ny * nz;
    let field = (0..n).map(|l| 0.02 * (l as f64 * 0.11).cos()).collect();
    let mut mask = vec![0u8; n];
    for t in 0..tiles {
        for k in 0..8 {
            for j in 0..8 {
                for i in 0..8 {
                    mask[(t * core + 4 + i) + nx * ((4 + j) + ny * (4 + k))] = 1;
                }
            }
        }
    }
    (field, mask, Grid::new(nx, ny, nz, 1.0, 1.0, 1.0))
}

fn weights(dir: &str, id: &str) -> Vec<Vec<u8>> {
    let spec = qsm_core::models::find_model(id).expect("model in the registry");
    spec.files
        .iter()
        .map(|f| std::fs::read(format!("{dir}/{}", f.name)).unwrap_or_else(|e| panic!("{dir}/{}: {e}", f.name)))
        .collect()
}

/// Tiles in flight: the pool natively with `parallel`, else one.
fn tiles_in_flight() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: tile_memory_footprint <weights dir> [model id…]");
    let ids: Vec<String> = args.collect();
    let ids: Vec<&str> = if ids.is_empty() { TILED.to_vec() } else { ids.iter().map(String::as_str).collect() };
    let halo = 8;
    let k = tiles_in_flight();

    for id in ids {
        let w = weights(&dir, id);
        if k == 1 {
            let divisor = qsm_core::models::find_model(id).unwrap().size_divisor as usize;
            let mut pts = Vec::new();
            let mut at_64 = 0.0;
            for p in patches(divisor) {
                let core = p - 2 * halo;
                let cfg = TileConfig { core, halo };
                assert_eq!(inversion::tile_patch_size(&cfg, divisor), p);
                let (field, mask, grid) = inputs((80, 80, 80), core, 1);
                let (peak, _) = peak_of(|| run(id, &w, &field, &mask, &grid, &cfg));
                println!("{id:<12} {p}³ patch, one tile in flight: peak {:.1} MB", peak as f64 / 1e6);
                let voxels = (p * p * p) as f64;
                if p == 64 {
                    at_64 = peak as f64 / voxels;
                }
                pts.push((voxels, peak as f64));
            }
            let (slope, fixed) = fit(&pts);
            println!(
                "{id:<12} => slope {slope:.0} B per patch voxel, intercept {:.0} MB; a 64³ tile with \
                 the intercept {at_64:.0} B per patch voxel",
                fixed / 1e6
            );
        } else {
            let core = 48;
            let (field, mask, grid) = inputs((core * k, 64, 64), core, k);
            let (peak, _) = peak_of(|| run(id, &w, &field, &mask, &grid, &TileConfig { core, halo }));
            println!("{id:<12} 64³ patch, {k} tiles in flight: peak {:.1} MB", peak as f64 / 1e6);
        }
    }
}

/// Patch edges to measure at: three for the nets that take multiples of 16 or less, the two
/// smallest for NeXtQSM, which takes multiples of 64.
fn patches(divisor: usize) -> Vec<usize> {
    if divisor <= 16 {
        vec![48, 64, 80]
    } else {
        vec![divisor, 2 * divisor]
    }
}

/// Least-squares line through `(x, y)` points: `(slope, intercept)`.
fn fit(pts: &[(f64, f64)]) -> (f64, f64) {
    let n = pts.len() as f64;
    let (mx, my) = pts.iter().fold((0.0, 0.0), |(a, b), &(x, y)| (a + x / n, b + y / n));
    let sxy: f64 = pts.iter().map(|&(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = pts.iter().map(|&(x, _)| (x - mx).powi(2)).sum();
    let slope = sxy / sxx;
    (slope, my - slope * mx)
}

//! Heap one inference of each registry network holds, as a function of its input shape.
//!
//! This is where the figures behind [`qsm_core::models::ModelSpec::memory`] come from, and how to
//! re-derive them after a `tract` upgrade or for a new network. A counting global allocator
//! records, for each network and input shape:
//!
//! - `resident`: what the parsed model ([`OnnxModel::load`]) holds once loaded;
//! - `inference`: the peak above that while one [`OnnxModel::run`] at that shape runs (the plan
//!   tract builds for the shape, the input and output tensors, the activations);
//! - `plan`: what the optimised plan for the shape holds ([`OnnxModel::plan_for`]), and
//!   `plan_run`, the peak above the plan while it runs once (what each further tile in flight on
//!   a shared plan adds).
//!
//! Each line of output is one JSON record. The process's resident-set high-water (`VmHWM`) is
//! printed last, so run one shape per process when that is wanted.
//!
//! Run single-threaded (no `parallel`), as a browser worker runs one inference:
//!   cargo run --release --example inference_memory --features onnx -- net <weights dir> <id> 64x64x64 [...]
//!   cargo run --release --example inference_memory --features onnx -- hdbet <weights dir> <nx>x<ny>x<nz> <vx>x<vy>x<vz> full|window|low
//!
//! The `hdbet` mode runs the whole `bet::hd_bet` step on a synthetic head (an ellipsoid filling
//! the field of view, low noise outside it, so nothing is cropped) and reports its peak, to check
//! `bet::hd_bet_memory_bytes`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use qsm_core::models::onnx::OnnxModel;

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

/// Peak heap above the current level while `f` runs, and what is still held after it.
fn peak_of<T>(f: impl FnOnce() -> T) -> (usize, isize, T) {
    let base = CURRENT.load(Relaxed);
    PEAK.store(base, Relaxed);
    let out = f();
    (PEAK.load(Relaxed) - base, CURRENT.load(Relaxed) as isize - base as isize, out)
}

fn vm_hwm_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

fn parse3<T: std::str::FromStr>(s: &str) -> [T; 3]
where
    T::Err: std::fmt::Debug,
{
    let v: Vec<T> = s.split('x').map(|t| t.parse().unwrap()).collect();
    assert_eq!(v.len(), 3, "expected AxBxC, got {s}");
    let mut it = v.into_iter();
    [it.next().unwrap(), it.next().unwrap(), it.next().unwrap()]
}

#[path = "../tests/support/net_inputs.rs"]
mod net_inputs;
use net_inputs::net_inputs as inputs;

fn net(dir: &str, id: &str, shapes: &[String]) {
    let spec = qsm_core::models::find_model(id).unwrap_or_else(|| panic!("{id} not in the registry"));
    for file in spec.files {
        let bytes = std::fs::read(format!("{dir}/{}", file.name)).unwrap_or_else(|e| panic!("{dir}/{}: {e}", file.name));
        for shape in shapes {
            let s: [usize; 3] = parse3(shape);
            let (load_peak, resident, model) = peak_of(|| OnnxModel::load(&bytes).expect("load"));
            let t = std::time::Instant::now();
            let (inference, _, out) = peak_of(|| {
                let x = inputs(file.name, s);
                model.run(&x).expect("run").len()
            });
            let secs = t.elapsed().as_secs_f64();
            let (plan_peak, plan, p) = peak_of(|| model.plan_for(&inputs(file.name, s).iter().map(|t| t.shape.as_slice()).collect::<Vec<_>>()).expect("plan"));
            let x = inputs(file.name, s);
            let (plan_run, _, _) = peak_of(|| p.run(&x).expect("run").len());
            drop((p, x, model));
            println!(
                "{{\"id\":\"{id}\",\"file\":\"{}\",\"shape\":[{},{},{}],\"voxels\":{},\"resident\":{resident},\"load_peak\":{load_peak},\
                 \"inference\":{inference},\"plan\":{plan},\"plan_peak\":{plan_peak},\"plan_run\":{plan_run},\"outputs\":{out},\"secs\":{secs:.1}}}",
                file.name, s[0], s[1], s[2], s[0] * s[1] * s[2]
            );
        }
    }
    println!("{{\"id\":\"{id}\",\"vm_hwm_kb\":{}}}", vm_hwm_kb());
}

fn hdbet(dir: &str, dims: &str, voxel: &str, mode: &str) {
    use qsm_core::bet::{hd_bet, HdBetInference, HdBetParams};
    let d: [usize; 3] = parse3(dims);
    let v: [f64; 3] = parse3(voxel);
    let grid = qsm_core::Grid::new(d[0], d[1], d[2], v[0], v[1], v[2]);
    let n = d[0] * d[1] * d[2];
    let mag: Vec<f64> = (0..n)
        .map(|l| {
            let (i, j, k) = (l % d[0], (l / d[0]) % d[1], l / (d[0] * d[1]));
            let r = |x: usize, m: usize| ((x as f64 + 0.5) / m as f64 - 0.5) / 0.45;
            let rr = r(i, d[0]).powi(2) + r(j, d[1]).powi(2) + r(k, d[2]).powi(2);
            if rr < 1.0 { 500.0 + 50.0 * (l as f64 * 0.37).sin() } else { 10.0 + (l % 7) as f64 }
        })
        .collect();
    let bytes = std::fs::read(format!("{dir}/hd-bet.onnx")).expect("hd-bet.onnx");
    let params = match mode {
        "full" => HdBetParams::default(),
        "window" => HdBetParams { inference: HdBetInference::SLIDING_WINDOW, ..HdBetParams::default() },
        "low" => HdBetParams::low_memory(),
        _ => panic!("mode is full, window or low"),
    };
    let estimate = qsm_core::bet::hd_bet_memory_bytes(&grid, &params, qsm_core::models::Heap::Native);
    let t = std::time::Instant::now();
    let (peak, _, mask) = peak_of(|| hd_bet(&mag, &grid, &bytes, &params, |_, _| {}).expect("hd_bet"));
    let brain = mask.iter().filter(|&&m| m > 0).count();
    println!(
        "{{\"hdbet\":\"{mode}\",\"dims\":[{},{},{}],\"voxel\":[{},{},{}],\"peak\":{peak},\"estimate\":{estimate},\"brain\":{brain},\"secs\":{:.1},\"vm_hwm_kb\":{}}}",
        d[0], d[1], d[2], v[0], v[1], v[2], t.elapsed().as_secs_f64(), vm_hwm_kb()
    );
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("net") if a.len() >= 5 => net(&a[2], &a[3], &a[4..]),
        Some("hdbet") if a.len() == 6 => hdbet(&a[2], &a[3], &a[4], &a[5]),
        _ => {
            eprintln!("usage: inference_memory net <weights dir> <id> <AxBxC>...\n       inference_memory hdbet <weights dir> <nx>x<ny>x<nz> <vx>x<vy>x<vz> full|window|low");
            std::process::exit(2);
        }
    }
}

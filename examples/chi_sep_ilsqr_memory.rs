//! Heap peak of [`qsm_core::separation::chi_sep_ilsqr`], and its output for comparing versions.
//!
//! This is where the memory figure in `separation::chi_sep_ilsqr`'s *Performance* section comes
//! from. A counting global allocator records the heap peak during one call, on a phantom built
//! as `tests/chi_sep_ilsqr_snu_parity.rs` builds its first case (an ellipsoid with χ+ and χ−
//! blocks at 3 T, 1 mm, the toolbox's default 16-voxel padding), at the grid given:
//!
//!   cargo run --release --example chi_sep_ilsqr_memory --features parallel -- 184x256x128 [out.bin]
//!
//! It prints the NNLS iterations, the run time, the heap peak in bytes and in `f64` per padded
//! voxel, and the process's resident-set high-water (`VmHWM`). With `out.bin`, it writes χ+, χ−
//! and χ_total (`3·n` f64, little-endian), so two builds' outputs can be compared bit for bit.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use num_complex::Complex64;
use qsm_core::fft::{fft3d, ifft3d};
use qsm_core::separation::{chi_sep_ilsqr, ChiSepIlsqrInputs, ChiSepIlsqrParams};
use qsm_core::Grid;

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
        grew(l.size());
        System.alloc(l)
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        grew(l.size());
        System.alloc_zeroed(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        CURRENT.fetch_sub(l.size(), Relaxed);
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        if new > l.size() {
            grew(new - l.size());
        } else {
            CURRENT.fetch_sub(l.size() - new, Relaxed);
        }
        System.realloc(p, l, new)
    }
}

#[global_allocator]
static A: Counting = Counting;

/// The parity test's phantom (its first case: 3 T, 1 mm, B0 along z), at grid `n`.
fn phantom(n: [usize; 3]) -> (Vec<u8>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let vs = [1.0; 3];
    let b0 = 3.0;
    let nt = n[0] * n[1] * n[2];
    let u = |seed: usize, l: usize| ((1103515245.0 * ((l + seed * 7919) as f64) + 12345.0) % 2f64.powi(31)) / 2f64.powi(31);
    let r = [0, 1, 2].map(|a| 0.4 * n[a] as f64 * vs[a]);
    let mut mask = vec![0u8; nt];
    let mut cp = vec![0.0; nt];
    let mut cn = vec![0.0; nt];
    for k in 0..n[2] {
        for j in 0..n[1] {
            for i in 0..n[0] {
                let l = i + n[0] * (j + n[1] * k);
                let p = [i, j, k];
                let x: [f64; 3] = [0, 1, 2].map(|a| ((p[a] + 1) as f64 - (n[a] as f64 + 1.0) / 2.0) * vs[a]);
                let inside = (x[0] / r[0]).powi(2) + (x[1] / r[1]).powi(2) + (x[2] / r[2]).powi(2) <= 1.0;
                let in_csf = inside
                    && (x[0] / (0.25 * r[0])).powi(2) + ((x[1] - 0.2 * r[1]) / (0.15 * r[1])).powi(2) + (x[2] / (0.3 * r[2])).powi(2) <= 1.0;
                mask[l] = inside as u8;
                let blk = |cx: f64, cy: f64, cz: f64, s: f64| {
                    ((x[0] - cx * r[0]).abs() <= s * r[0] && (x[1] - cy * r[1]).abs() <= s * r[1] && (x[2] - cz * r[2]).abs() <= s * r[2]) as u8 as f64
                };
                let keep = (inside as u8 as f64) * (!in_csf as u8 as f64);
                cp[l] = (0.08 * blk(0.4, -0.3, 0.1, 0.18) + 0.05 * blk(-0.35, -0.35, -0.2, 0.15) + 0.02 * ((x[1] < -0.5 * r[1]) as u8 as f64)) * keep;
                cn[l] = (0.04 * blk(-0.3, 0.45, 0.25, 0.2) + 0.03 * blk(0.35, 0.3, -0.3, 0.16) + 0.015 * ((x[0].abs() < 0.3 * r[0]) as u8 as f64)) * keep;
            }
        }
    }
    let cf = 42.58e6 * b0;
    let f = |a: usize, i: usize| {
        let ni = n[a] as i64;
        let fi = if (i as i64) < (ni + 1) / 2 { i as i64 } else { i as i64 - ni };
        fi as f64 / (n[a] as f64 * vs[a])
    };
    let mut buf: Vec<Complex64> = cp.iter().zip(&cn).map(|(a, b)| Complex64::new(a - b, 0.0)).collect();
    fft3d(&mut buf, n[0], n[1], n[2]);
    for k in 0..n[2] {
        for j in 0..n[1] {
            for i in 0..n[0] {
                let (kx, ky, kz) = (f(0, i), f(1, j), f(2, k));
                let k2 = kx * kx + ky * ky + kz * kz;
                let d = if k2 == 0.0 { 0.0 } else { 1.0 / 3.0 - (kz * kz) / k2 };
                buf[i + n[0] * (j + n[1] * k)] *= d;
            }
        }
    }
    ifft3d(&mut buf, n[0], n[1], n[2]);
    let dr = 137.0 / 2.9 * b0;
    let m = |l: usize| mask[l] as f64;
    let field: Vec<f64> = (0..nt).map(|l| (buf[l].re * (cf * 1e-6) + 0.4 * (u(1, l) - 0.5)) * m(l) / cf * 1e6).collect();
    let r2p: Vec<f64> = (0..nt).map(|l| (dr * (cp[l] + cn[l]) + 3.0 + 2.0 * (u(2, l) - 0.5)) * m(l)).collect();
    let qsm: Vec<f64> = (0..nt).map(|l| (cp[l] - cn[l] + 0.004 * (u(3, l) - 0.5)) * m(l)).collect();
    (mask, field, r2p, qsm)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: Vec<usize> = args.get(1).expect("grid, e.g. 184x256x128").split('x').map(|v| v.parse().unwrap()).collect();
    let n = [n[0], n[1], n[2]];
    let (mask, field, r2p, qsm) = phantom(n);
    let grid = Grid::new(n[0], n[1], n[2], 1.0, 1.0, 1.0);
    let params = ChiSepIlsqrParams { cf: 42.58e6 * 3.0, ..ChiSepIlsqrParams::default() };
    let inputs = ChiSepIlsqrInputs { local_field: &field, r2prime: &r2p, qsm: &qsm, mask: &mask };
    let base = CURRENT.load(Relaxed);
    PEAK.store(base, Relaxed);
    let t = std::time::Instant::now();
    let mut iters = 0;
    let (p, q, s) = chi_sep_ilsqr(&inputs, &grid, (0.0, 0.0, 1.0), &params, |i, _| iters = i).unwrap();
    let secs = t.elapsed().as_secs_f64();
    let peak = PEAK.load(Relaxed) - base;
    let pad = params.pad;
    let n_pad = (0..3).map(|a| n[a] + 2 * pad[a]).product::<usize>();
    let n_mask = mask.iter().filter(|&&v| v != 0).count();
    println!(
        "grid {}x{}x{}, padded {} voxels, mask {} ({:.3} of padded); {} NNLS iterations, {:.1} s; heap peak {} B = {:.2} f64 per padded voxel",
        n[0], n[1], n[2], n_pad, n_mask, n_mask as f64 / n_pad as f64, iters, secs, peak, peak as f64 / (8.0 * n_pad as f64)
    );
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        if let Some(l) = status.lines().find(|l| l.starts_with("VmHWM")) {
            println!("{l}");
        }
    }
    if let Some(out) = args.get(2) {
        let bytes: Vec<u8> = p.iter().chain(&q).chain(&s).flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out, bytes).unwrap();
    }
}

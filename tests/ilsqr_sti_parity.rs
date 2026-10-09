//! `inversion::ilsqr` against STI Suite 3.0's `QSM_iLSQR` on a small synthetic case.
//!
//! `tests/data/ilsqr_sti_small.bin` (little-endian, column-major, 20x20x14 grid, voxel
//! 1x1x1.5 mm, B0 direction (0.1, -0.2, sqrt(0.95)), TE 20 ms, 3 T, padsize 8 mm) holds:
//! - the tissue phase (f32, rad; nonzero outside the mask too) and the mask (u8);
//! - inside the mask only: STI's two outputs (f32, ppm) from MATLAB R2023b
//!   `[chi, chi_lsqr] = QSM_iLSQR(phase, mask, 'TE', 20, 'B0', 3, 'H', H, 'padsize', [8 8 8],
//!   'voxelsize', [1 1 1.5])`, then the same two from the MATLAB re-implementation of that
//!   algorithm (`ilsqr_sti_clone.m`, double precision, 5 iterations per LSQR solve; f64).
//!
//! STI runs its LSQR solves in single precision. After a dozen or so iterations LSQR's rounding
//! errors decide the details, so the full-length comparison with STI is loose, and the
//! algorithm itself is pinned by the short double-precision runs.

use qsm_core::inversion::{ilsqr_with_padding, IlsqrParams};
use qsm_core::Grid;

struct Fixture {
    phase: Vec<f64>,
    mask: Vec<u8>,
    inside: Vec<usize>,
    sti: (Vec<f64>, Vec<f64>),
    clone5: (Vec<f64>, Vec<f64>),
}

const DIMS: (usize, usize, usize) = (20, 20, 14);

fn load() -> Fixture {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/ilsqr_sti_small.bin")).expect("fixture");
    let n = DIMS.0 * DIMS.1 * DIMS.2;
    let f32_at = |o: usize| f32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as f64;
    let f64_at = |o: usize| f64::from_le_bytes(raw[o..o + 8].try_into().unwrap());
    let phase: Vec<f64> = (0..n).map(|i| f32_at(4 * i)).collect();
    let mask: Vec<u8> = raw[4 * n..5 * n].to_vec();
    let inside: Vec<usize> = (0..n).filter(|&i| mask[i] != 0).collect();
    let k = inside.len();
    assert_eq!(raw.len(), 5 * n + k * (4 + 4 + 8 + 8));
    let mut o = 5 * n;
    let mut f32s = || { let v: Vec<f64> = (0..k).map(|i| f32_at(o + 4 * i)).collect(); o += 4 * k; v };
    let (s0, s1) = (f32s(), f32s());
    let c0: Vec<f64> = (0..k).map(|i| f64_at(o + 8 * i)).collect();
    let c1: Vec<f64> = (0..k).map(|i| f64_at(o + 8 * (k + i))).collect();
    Fixture { phase, mask, inside, sti: (s0, s1), clone5: (c0, c1) }
}

fn run(fx: &Fixture, max_iter: usize) -> (Vec<f64>, Vec<f64>) {
    let to_ppm = 1.0 / (2.0 * std::f64::consts::PI * 42.575 * 3.0 * 20.0e-3); // STI's scaling
    let grid = Grid::new(DIMS.0, DIMS.1, DIMS.2, 1.0, 1.0, 1.5);
    let params = IlsqrParams { max_iter, ..IlsqrParams::default() };
    let (chi, _, _, x1) = ilsqr_with_padding(&fx.phase, &fx.mask, &grid, (0.1, -0.2, 0.95f64.sqrt()), &params, [8.0; 3], |_, _| {});
    for i in 0..chi.len() { if fx.mask[i] == 0 { assert_eq!(chi[i], 0.0); } }
    let pick = |v: &[f64]| fx.inside.iter().map(|&i| v[i] * to_ppm).collect::<Vec<f64>>();
    (pick(&chi), pick(&x1))
}

fn rms(v: &[f64]) -> f64 { (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt() }
fn maxd(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max) }
fn corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) { ab += (x - ma) * (y - mb); aa += (x - ma).powi(2); bb += (y - mb).powi(2); }
    ab / (aa * bb).sqrt()
}

#[test]
fn ilsqr_matches_the_reverse_engineered_algorithm() {
    let fx = load();
    let (chi, x1) = run(&fx, 5);
    let (d, d1) = (maxd(&chi, &fx.clone5.0), maxd(&x1, &fx.clone5.1));
    println!("5 iterations vs double-precision clone: chi max|d| {:.2e}, initial {:.2e} (rms {:.2e})", d, d1, rms(&fx.clone5.0));
    assert!(d1 < 1e-9 * rms(&fx.clone5.1), "initial LSQR differs by {}", d1);
    assert!(d < 1e-9 * rms(&fx.clone5.0), "chi differs by {}", d);
}

#[test]
fn ilsqr_matches_sti_suite() {
    let fx = load();
    let (chi, x1) = run(&fx, IlsqrParams::default().max_iter);
    let (r, r1) = (corr(&chi, &fx.sti.0), corr(&x1, &fx.sti.1));
    println!("vs STI: chi r {:.6} max|d| {:.2e}; initial r {:.8} max|d| {:.2e} (rms chi {:.2e})",
             r, maxd(&chi, &fx.sti.0), r1, maxd(&x1, &fx.sti.1), rms(&fx.sti.0));
    assert!(r1 > 0.9999, "initial LSQR r = {}", r1);
    assert!(r > 0.99, "chi r = {}", r);
    assert!(maxd(&x1, &fx.sti.1) < 0.05 * rms(&fx.sti.1));
}

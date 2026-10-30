//! `inversion::ilsqr` against STI Suite 3.0's `QSM_iLSQR` on small synthetic cases.
//!
//! Two fixtures:
//!
//! - `tests/data/ilsqr_sti_qsmci_small.bin`, made by `tests/data/ilsqr_sti_qsmci_small.py` with the
//!   verified QSM-CI `ilsqr-sti` container (QSMxT/QSM-CI#304), so it can be regenerated in Actions
//!   (`.github/workflows/regenerate-fixtures.yml`): two phantoms (layout in the script), each with
//!   the local field the container was given and its χ (float32, which is STI's single-precision χ
//!   exactly). QSM.rs runs at its defaults, which are the container's call (100 iterations) with
//!   its `padsize`.
//! - `tests/data/ilsqr_sti_small.bin` (little-endian, column-major, 20x20x14 grid, voxel
//!   1x1x1.5 mm, B0 direction (0.1, -0.2, sqrt(0.95)), TE 20 ms, 3 T, padsize 8 mm), made in
//!   licensed MATLAB R2023b by `tests/data/ilsqr_sti_small.m`, holds:
//!   - the tissue phase (f32, rad; nonzero outside the mask too) and the mask (u8);
//!   - inside the mask only: STI's two outputs (f32, ppm) from
//!     `[chi, chi_lsqr] = QSM_iLSQR(phase, mask, 'TE', 20, 'B0', 3, 'H', H, 'padsize', [8 8 8],
//!     'voxelsize', [1 1 1.5])`, then the same two from the MATLAB re-implementation of that
//!     algorithm (`tests/data/ilsqr_sti_clone.m`, double precision, 5 iterations per LSQR solve;
//!     f64). The container does not return the initial LSQR solution or run the re-implementation,
//!     so this one stays a licensed-MATLAB fixture.
//!
//! STI runs its LSQR solves in single precision. After a dozen or so iterations LSQR's rounding
//! errors decide the details, so the full-length comparison with STI is loose, and the
//! algorithm itself is pinned by the short double-precision runs.

use qsm_core::inversion::{ilsqr, IlsqrParams};
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
    let params = IlsqrParams { max_iter, pad_mm: 8.0, ..IlsqrParams::default() };
    let (chi, _, _, x1) = ilsqr(&fx.phase, &fx.mask, &grid, (0.1, -0.2, 0.95f64.sqrt()), &params, |_, _| {});
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
    // The fixture is QSM_iLSQR without 'niter', i.e. STI's function default of 100.
    let (chi, x1) = run(&fx, 100);
    let (r, r1) = (corr(&chi, &fx.sti.0), corr(&x1, &fx.sti.1));
    println!("vs STI: chi r {:.6} max|d| {:.2e}; initial r {:.8} max|d| {:.2e} (rms chi {:.2e})",
             r, maxd(&chi, &fx.sti.0), r1, maxd(&x1, &fx.sti.1), rms(&fx.sti.0));
    assert!(r1 > 0.9999, "initial LSQR r = {}", r1);
    assert!(r > 0.99, "chi r = {}", r);
    assert!(maxd(&x1, &fx.sti.1) < 0.05 * rms(&fx.sti.1));
}

/// The container fixture: per case, (mask, field in ppm on the grid, STI's χ at the mask voxels).
fn load_qsmci() -> Vec<(Vec<u8>, Vec<f64>, Vec<f64>)> {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/ilsqr_sti_qsmci_small.bin")).expect("fixture");
    let mut o = 0;
    let out = QSMCI_CASES.iter().map(|c| {
        let n = c.dims.0 * c.dims.1 * c.dims.2;
        let m = u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize;
        let mask = raw[o + 4..o + 4 + n].to_vec();
        o += 4 + n;
        let inside: Vec<usize> = (0..n).filter(|&i| mask[i] != 0).collect();
        assert_eq!(inside.len(), m);
        let mut field = vec![0.0; n];
        for (v, &i) in inside.iter().enumerate() {
            field[i] = f64::from_le_bytes(raw[o + 8 * v..o + 8 * v + 8].try_into().unwrap());
        }
        o += 8 * m;
        let chi = (0..m).map(|v| f32::from_le_bytes(raw[o + 4 * v..o + 4 * v + 4].try_into().unwrap()) as f64).collect();
        o += 4 * m;
        (mask, field, chi)
    }).collect();
    assert_eq!(o, raw.len());
    out
}

struct QsmciCase {
    dims: (usize, usize, usize),
    vs: (f64, f64, f64),
    h: (f64, f64, f64),
    pad: f64,
}

/// `CASES` of `tests/data/ilsqr_sti_qsmci_small.py`, in order (B0 and TE only set the phase scale).
const QSMCI_CASES: [QsmciCase; 2] = [
    QsmciCase { dims: (32, 32, 24), vs: (1.0, 1.0, 1.5), h: (0.1, -0.2, 0.974_679_434_480_896_4), pad: 12.0 },
    QsmciCase { dims: (32, 28, 28), vs: (1.0, 1.0, 1.0), h: (0.0, 0.0, 1.0), pad: 8.0 },
];

#[test]
fn ilsqr_defaults_match_the_ilsqr_sti_container() {
    // The container turns ppm into phase with γ = 42.576 MHz/T and STI turns it back with its
    // own 42.575, so STI's χ is the field's χ scaled by 42.576/42.575.
    let gamma_ratio = 42.576 / 42.575;
    for (c, (mask, field, sti)) in QSMCI_CASES.iter().zip(load_qsmci()) {
        let grid = Grid::new(c.dims.0, c.dims.1, c.dims.2, c.vs.0, c.vs.1, c.vs.2);
        let params = IlsqrParams { pad_mm: c.pad, ..IlsqrParams::default() };
        let (chi, _, _, _) = ilsqr(&field, &mask, &grid, c.h, &params, |_, _| {});
        let got: Vec<f64> = (0..chi.len()).filter(|&i| mask[i] != 0).map(|i| chi[i] * gamma_ratio).collect();
        let nrmse = (got.iter().zip(&sti).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / sti.iter().map(|b| b * b).sum::<f64>()).sqrt();
        let (r, d) = (corr(&got, &sti), maxd(&got, &sti));
        println!("{:?} pad {}: vs STI (container) NRMSE {nrmse:.2e}, r {r:.7}, max|Δ| {d:.2e} ppm (rms χ {:.2e})", c.dims, c.pad, rms(&sti));
        // Measured 2026-10-30: NRMSE 3.4 % (r 0.99943) and 1.6 % (r 0.99987). STI runs both LSQR
        // solves in single precision, and after a few dozen iterations that rounding decides the
        // iterates (see the module docs): QSM.rs is closest to STI at STI's own 100 iterations
        // (NRMSE 9.1 % / 7.4 % at 40, 6.4 % / 4.3 % at 200), while rounding QSM.rs's input to
        // float32 moves it by only 5.7e-4 / 1.8e-5. The algorithm itself is pinned to 1e-9 by
        // `ilsqr_matches_the_reverse_engineered_algorithm`.
        assert!(nrmse < 0.05, "{:?}: NRMSE {nrmse}", c.dims);
        assert!(r > 0.999, "{:?}: r {r}", c.dims);
    }
}

//! `separation::chi_sep_medi` against a stored output of its implementation before the speed-up
//! of QSM.rs#296 (commit f16a122), on a small two-source phantom whose grid is padded for the
//! FFTs (30×28×26 → 30×30×27).
//!
//! The speed-up applies the CG operator's D(D(·)) as one transform pair with D², which rounds
//! differently in single precision; everything else it changed gives the same bits. The stored
//! maps are what the old code returned, so this pins the result to the old implementation's.

use qsm_core::separation::{chi_sep_medi, ChiSepParams};
use qsm_core::Grid;

const DIMS: (usize, usize, usize) = (30, 28, 26);
const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/chi_sep_medi_small.bin");

/// A paramagnetic and a diamagnetic sphere in a spherical brain at 7 T, voxels 1×1×1.2 mm,
/// B0 tilted off z; the field is the forward dipole model (by FFT, the same as the solver's)
/// and R2' = Dr·(χ+ + |χ−|), with a deterministic ripple on both so CG has work to do.
struct Phantom {
    field: Vec<f64>,
    r2prime: Vec<f64>,
    magnitude: Vec<f64>,
    mask: Vec<u8>,
    grid: Grid,
    bdir: (f64, f64, f64),
    params: ChiSepParams,
}

fn phantom() -> Phantom {
    let (nx, ny, nz) = DIMS;
    let n = nx * ny * nz;
    let grid = Grid::new(nx, ny, nz, 1.0, 1.0, 1.2);
    let bdir = (0.1, 0.05, (1.0f64 - 0.0125).sqrt());
    let ball = |c: (f64, f64, f64), r: f64| -> Vec<bool> {
        (0..n)
            .map(|l| {
                let (i, j, k) = ((l % nx) as f64, ((l / nx) % ny) as f64, (l / (nx * ny)) as f64);
                (i - c.0).powi(2) + (j - c.1).powi(2) + ((k - c.2) * 1.2).powi(2) <= r * r
            })
            .collect()
    };
    let brain = ball((15.0, 14.0, 13.0), 12.0);
    let para = ball((10.0, 14.0, 13.0), 4.0);
    let dia = ball((20.0, 14.0, 13.0), 4.0);
    let mask: Vec<u8> = brain.iter().map(|&b| b as u8).collect();
    let ripple = |l: usize| 1.0 + 0.2 * ((l as f64) * 0.37).sin();
    let pos: Vec<f64> = (0..n).map(|l| if para[l] { 0.1 * ripple(l) } else { 0.0 }).collect();
    let neg: Vec<f64> = (0..n).map(|l| if dia[l] { -0.05 * ripple(l) } else { 0.0 }).collect();
    let total: Vec<f64> = pos.iter().zip(&neg).map(|(p, q)| p + q).collect();

    let d = qsm_core::kernels::dipole::dipole_kernel(&grid, bdir);
    let f: Vec<_> = qsm_core::fft::fft3d_real(&total, nx, ny, nz).iter().zip(&d).map(|(&c, &k)| c * k).collect();
    let field = qsm_core::fft::ifft3d_real(&f, nx, ny, nz);
    let dr = qsm_core::separation::DR_SHIN_2021;
    let r2prime: Vec<f64> = (0..n).map(|l| dr * (pos[l] - neg[l]) + 0.5 * ((l as f64) * 0.11).cos().abs()).collect();
    let magnitude: Vec<f64> = (0..n).map(|l| mask[l] as f64 * (100.0 - 30.0 * para[l] as u8 as f64 + 20.0 * dia[l] as u8 as f64)).collect();
    let params = ChiSepParams {
        cf: qsm_core::constants::GAMMA_BAR_HZ_PER_T * 7.0,
        max_iter: 5,
        ..ChiSepParams::default()
    };
    Phantom { field, r2prime, magnitude, mask, grid, bdir, params }
}

/// (χ+, χ−) over the mask voxels, in column-major order
fn run() -> (Vec<f64>, Vec<f64>) {
    let Phantom { field, r2prime, magnitude, mask, grid, bdir, params } = phantom();
    let (pos, neg, total) = chi_sep_medi(&field, &r2prime, &magnitude, &mask, &grid, bdir, &params, |_, _| {});
    for l in 0..pos.len() {
        assert_eq!(total[l], pos[l] + neg[l]);
    }
    let inside = |v: &[f64]| -> Vec<f64> { v.iter().zip(&mask).filter(|(_, &m)| m != 0).map(|(&x, _)| x).collect() };
    (inside(&pos), inside(&neg))
}

/// Writes the fixture from the current code (`cargo test --release --test chi_sep_medi_reference
/// -- --ignored dump_fixture`). The committed file was written by the implementation before
/// QSM.rs#296.
#[test]
#[ignore]
fn dump_fixture() {
    let (pos, neg) = run();
    let mut raw = (pos.len() as u32).to_le_bytes().to_vec();
    for v in pos.iter().chain(&neg) {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(FIXTURE, raw).expect("write fixture");
}

/// `tests/data/chi_sep_medi_small.bin` (little-endian): the number of mask voxels `m` (u32),
/// then χ+ and χ− in ppm over the mask voxels in column-major order (f64 each).
fn fixture() -> (Vec<f64>, Vec<f64>) {
    let raw = std::fs::read(FIXTURE).expect("fixture");
    let m = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let vals: Vec<f64> = raw[4..].chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(vals.len(), 2 * m, "fixture length");
    (vals[..m].to_vec(), vals[m..].to_vec())
}

#[test]
fn matches_the_implementation_before_the_speed_up() {
    let (pos, neg) = run();
    let (ref_pos, ref_neg) = fixture();
    assert_eq!(pos.len(), ref_pos.len(), "mask voxel count");
    for (name, got, want) in [("χ+", &pos, &ref_pos), ("χ−", &neg, &ref_neg)] {
        let scale = want.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        let max_diff = got.iter().zip(want.iter()).fold(0.0f64, |a, (&g, &w)| a.max((g - w).abs()));
        println!("{name}: max |Δ| {max_diff:.3e} ppm, {:.3e} of max |{name}| {scale:.4} ppm", max_diff / scale);
        assert!(scale > 1e-3, "{name} is empty");
        assert!(max_diff <= TOL * scale, "{name}: max |Δ| {max_diff:.3e} ppm > {TOL:e} × {scale:.4} ppm");
    }
}

/// Relative to the map's largest magnitude
const TOL: f64 = 1e-5;

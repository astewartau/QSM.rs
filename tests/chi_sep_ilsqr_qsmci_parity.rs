//! `separation::chi_sep_ilsqr` against the SNU χ-separation toolbox v1.1.3's `chi_sep_iLSQR` as
//! the verified QSM-CI `chi-sep-ilsqr` container runs it (QSMxT/QSM-CI#306).
//!
//! `tests/data/chi_sep_ilsqr_qsmci_small.bin` is made by `tests/data/chi_sep_ilsqr_qsmci_small.py`
//! with that container, so it can be regenerated in Actions
//! (`.github/workflows/regenerate-fixtures.yml`); the phantom is rebuilt here exactly as the
//! script builds it, and the fixture holds the container's own QSM (STI Suite's `QSM_iLSQR`),
//! which `chi_sep_iLSQR` requires, and its χ+ and |χ−|. The toolbox function called directly, on
//! other inputs, is checked to double rounding by `chi_sep_ilsqr_snu_parity.rs` (licensed MATLAB).
//! The `chi-sep-medi` container is not used: its call makes the toolbox return its initial guess
//! (see the generator).

use qsm_core::separation::{chi_sep_ilsqr, ChiSepIlsqrInputs, ChiSepIlsqrParams};
use qsm_core::Grid;

const N: [usize; 3] = [24, 22, 18];
const CF: f64 = 42.5774e6 * 3.0;

struct Phantom {
    field: Vec<f64>,
    r2p: Vec<f64>,
    mask: Vec<u8>,
}

/// The generator's `phantom()`.
fn phantom() -> Phantom {
    let nv = N.iter().product();
    let mut p = Phantom { field: vec![0.0; nv], r2p: vec![0.0; nv], mask: vec![0; nv] };
    for k in 0..N[2] {
        for j in 0..N[1] {
            for i in 0..N[0] {
                let l = i + N[0] * (j + N[1] * k);
                let (x, y, z) = (i as f64 - 11.6, j as f64 - 10.4, k as f64 - 8.7);
                if x * x / 110.0 + y * y / 90.0 + z * z / 60.0 >= 1.0 {
                    continue;
                }
                p.mask[l] = 1;
                let d1 = (x - 3.0) * (x - 3.0) + (y + 2.0) * (y + 2.0) + z * z;
                let d2 = (x + 4.0) * (x + 4.0) + (y - 3.0) * (y - 3.0) + (z - 1.0) * (z - 1.0);
                let para = 0.03 + 0.06 / (1.0 + d1 / 8.0);
                let dia = 0.02 + 0.05 / (1.0 + d2 / 10.0);
                p.field[l] = 0.002 * z + 0.02 / (1.0 + d1 / 6.0) - 0.015 / (1.0 + d2 / 8.0) + 0.0001 * x * y;
                p.r2p[l] = 137.0 * (para + dia) + 0.5 * x / 10.0;
            }
        }
    }
    p
}

struct Fixture {
    qsm: Vec<f64>,
    chi: (Vec<f32>, Vec<f32>),
}

fn load() -> Fixture {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/chi_sep_ilsqr_qsmci_small.bin")).expect("fixture");
    let m = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let qsm = (0..m).map(|i| f64::from_le_bytes(raw[4 + 8 * i..12 + 8 * i].try_into().unwrap())).collect();
    let o = 4 + 8 * m;
    let f32s = |o: usize| -> Vec<f32> { (0..m).map(|i| f32::from_le_bytes(raw[o + 4 * i..o + 4 * i + 4].try_into().unwrap())).collect() };
    assert_eq!(o + 8 * m, raw.len(), "fixture length");
    Fixture { qsm, chi: (f32s(o), f32s(o + 4 * m)) }
}

/// (relative L2, max |Δ|) of QSM.rs's χ+ and |χ−| at the mask voxels against the container's
fn compare(p: &Phantom, pos: &[f64], neg: &[f64], want: &(Vec<f32>, Vec<f32>)) -> [(f64, f64); 2] {
    let inside: Vec<usize> = (0..p.mask.len()).filter(|&l| p.mask[l] != 0).collect();
    let stats = |got: Vec<f64>, want: &[f32]| {
        let max = got.iter().zip(want).map(|(a, &b)| (a - b as f64).abs()).fold(0.0, f64::max);
        let rel = (got.iter().zip(want).map(|(a, &b)| (a - b as f64).powi(2)).sum::<f64>() / want.iter().map(|&b| (b as f64).powi(2)).sum::<f64>()).sqrt();
        (rel, max)
    };
    [stats(inside.iter().map(|&l| pos[l]).collect(), &want.0), stats(inside.iter().map(|&l| -neg[l]).collect(), &want.1)]
}

#[test]
fn chi_sep_ilsqr_matches_the_chi_sep_ilsqr_container() {
    let (p, fx) = (phantom(), load());
    // The container's own QSM (STI Suite's QSM_iLSQR), which chi_sep_iLSQR requires
    let mut qsm = vec![0.0; p.mask.len()];
    for (v, l) in (0..p.mask.len()).filter(|&l| p.mask[l] != 0).enumerate() {
        qsm[l] = fx.qsm[v];
    }
    let inputs = ChiSepIlsqrInputs { local_field: &p.field, r2prime: &p.r2p, qsm: &qsm, mask: &p.mask };
    let params = ChiSepIlsqrParams { cf: CF, ..ChiSepIlsqrParams::default() };
    let grid = Grid::new(N[0], N[1], N[2], 1.0, 1.0, 1.0);
    let (pos, neg, _) = chi_sep_ilsqr(&inputs, &grid, (0.0, 0.0, 1.0), &params, |_, _| {}).unwrap();
    let [(rp, dp), (rn, dn)] = compare(&p, &pos, &neg, &fx.chi);
    println!("chi-sep-ilsqr: χ+ rel L2 {rp:.2e}, max|Δ| {dp:.2e} ppm; |χ−| rel L2 {rn:.2e}, max|Δ| {dn:.2e}");
    // Measured 2026-10-30: relative L2 2.9e-8 / 3.0e-8, max |Δ| 3.5e-9 / 7.0e-9 ppm, the
    // container's float32 rounding (χ up to ~0.1 ppm).
    assert!(dp < 1e-8 && dn < 1e-8, "χ+ max|Δ| {dp:e}, |χ−| {dn:e}");
}

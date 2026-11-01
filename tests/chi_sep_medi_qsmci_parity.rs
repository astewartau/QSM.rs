//! `separation::chi_sep_medi`, with its noise map from `utils::fit_ppm_complex_noise`, against the
//! SNU χ-separation toolbox v1.1.3's `chi_sep_MEDI` as the verified QSM-CI `chi-sep-medi`
//! container (v4) runs it (QSMxT/QSM-CI#351, QSM.rs#382).
//!
//! `tests/data/chi_sep_medi_qsmci_small.bin` is made by `tests/data/chi_sep_medi_qsmci_small.py`
//! with that container, so it can be regenerated in Actions
//! (`.github/workflows/regenerate-fixtures.yml`); the phantom is rebuilt here exactly as the
//! script builds it. The fixture holds the container's `N_std` (MEDI `Fit_ppm_complex`'s `dp1`
//! from the echoes' magnitudes), which QSM.rs must reproduce, and its χ+ and |χ−|. The container
//! passes `mask_CSF = []`, which QSM.rs's all-zero CSF mask reproduces. The toolbox function
//! called directly, on other inputs, is checked to double rounding by `chi_sep_medi_snu_parity.rs`
//! (licensed MATLAB).

use qsm_core::separation::{chi_sep_medi, ChiSepMediInputs, ChiSepMediParams};
use qsm_core::utils::fit_ppm_complex_noise;
use qsm_core::Grid;

const N: [usize; 3] = [24, 22, 18];
const TE: [f64; 3] = [0.01, 0.02, 0.03];
const CF: f64 = 42.5774e6 * 3.0;

struct Phantom {
    field: Vec<f64>,
    r2p: Vec<f64>,
    chi: Vec<f64>,
    /// echo after echo
    mag: Vec<f64>,
    mask: Vec<u8>,
}

/// The generator's `phantom()`.
fn phantom() -> Phantom {
    let nv: usize = N.iter().product();
    let mut p = Phantom { field: vec![0.0; nv], r2p: vec![0.0; nv], chi: vec![0.0; nv], mag: vec![0.0; nv * TE.len()], mask: vec![0; nv] };
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
                p.chi[l] = para - dia;
                let m0 = 1.0 + 0.01 * x - 0.005 * y;
                for (e, t) in TE.iter().enumerate() {
                    p.mag[l + nv * e] = m0 / (1.0 + (20.0 + 0.5 * z) * t);
                }
            }
        }
    }
    p
}

struct Fixture {
    n_std: Vec<f64>,
    chi: (Vec<f32>, Vec<f32>),
}

fn load() -> Fixture {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/chi_sep_medi_qsmci_small.bin")).expect("fixture");
    let m = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let n_std = (0..m).map(|i| f64::from_le_bytes(raw[4 + 8 * i..12 + 8 * i].try_into().unwrap())).collect();
    let o = 4 + 8 * m;
    let f32s = |o: usize| -> Vec<f32> { (0..m).map(|i| f32::from_le_bytes(raw[o + 4 * i..o + 4 * i + 4].try_into().unwrap())).collect() };
    assert_eq!(o + 8 * m, raw.len(), "fixture length");
    Fixture { n_std, chi: (f32s(o), f32s(o + 4 * m)) }
}

#[test]
fn chi_sep_medi_matches_the_chi_sep_medi_container() {
    let (p, fx) = (phantom(), load());
    let nv = p.mask.len();
    let inside: Vec<usize> = (0..nv).filter(|&l| p.mask[l] != 0).collect();

    // N_std: Fit_ppm_complex's dp1 from the echoes' magnitudes, zero outside the mask
    let mut n_std = fit_ppm_complex_noise(&p.mag, TE.len());
    for (v, &m) in n_std.iter_mut().zip(&p.mask) {
        if m == 0 {
            *v = 0.0;
        }
    }
    for (v, &l) in inside.iter().enumerate() {
        assert_eq!(n_std[l], fx.n_std[v], "N_std at voxel {l}");
    }

    // mag = sqrt(sum(mag4.^2, 4))
    let rss: Vec<f64> = (0..nv).map(|l| (0..TE.len()).map(|e| p.mag[l + nv * e] * p.mag[l + nv * e]).sum::<f64>().sqrt()).collect();
    let no_csf = vec![0u8; nv];
    let inputs = ChiSepMediInputs {
        magnitude: &rss, local_field: &p.field, r2prime: &p.r2p, n_std: &n_std, qsm: &p.chi, mask: &p.mask,
        mask_csf: &no_csf,
    };
    let params = ChiSepMediParams { cf: CF, ..ChiSepMediParams::default() };
    let grid = Grid::new(N[0], N[1], N[2], 1.0, 1.0, 1.0);
    let (pos, neg, _) = chi_sep_medi(&inputs, &grid, (0.0, 0.0, 1.0), &params, |_, _| {}).unwrap();

    let stats = |got: Vec<f64>, want: &[f32]| {
        let max = got.iter().zip(want).map(|(a, &b)| (a - b as f64).abs()).fold(0.0, f64::max);
        let rel = (got.iter().zip(want).map(|(a, &b)| (a - b as f64).powi(2)).sum::<f64>() / want.iter().map(|&b| (b as f64).powi(2)).sum::<f64>()).sqrt();
        (rel, max)
    };
    let (rp, dp) = stats(inside.iter().map(|&l| pos[l]).collect(), &fx.chi.0);
    let (rn, dn) = stats(inside.iter().map(|&l| -neg[l]).collect(), &fx.chi.1);
    println!("chi-sep-medi: χ+ rel L2 {rp:.2e}, max|Δ| {dp:.2e} ppm; |χ−| rel L2 {rn:.2e}, max|Δ| {dn:.2e}");
    // Measured 2026-11-01: N_std identical; χ+ and |χ−| to the container's float32 rounding.
    assert!(dp < 1e-8 && dn < 1e-8, "χ+ max|Δ| {dp:e}, |χ−| {dn:e}");
}

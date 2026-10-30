//! `separation::wavesep` against WaveSep itself (ZhenghanFang/WaveSep f3a192e).
//!
//! `tests/data/wavesep_small.bin` is made by `tests/data/wavesep_small.py` with the verified QSM-CI
//! `wavesep` image (QSMxT/QSM-CI#302), so it can be regenerated in Actions
//! (`.github/workflows/regenerate-fixtures.yml`): a 24x24x24 synthetic case (χ_total, R2′, mask)
//! and WaveSep's χ+ and |χ−| (float32) at its defaults, as QSM-CI's `wavesep` submission runs it.
//!
//! Both stop after the same 13 iterations and agree to the fixture's float32 rounding. Until the
//! wavelet step used PyWavelets' periodization (astewartau/QSM.rs#327; see `separation::wavesep`)
//! they were ~10 % NRMSE apart here, and 0.11 % (χ+) / 1.08 % (χ−) on QSMxT/QSM-CI#302's larger
//! synthetic case.

use qsm_core::separation::{wavesep, WaveSepParams};
use qsm_core::Grid;

const DIMS: (usize, usize, usize) = (24, 24, 24);

#[test]
fn wavesep_matches_wavesep() {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/wavesep_small.bin")).expect("fixture");
    let n = DIMS.0 * DIMS.1 * DIMS.2;
    let m = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let mask = raw[4..4 + n].to_vec();
    assert_eq!(raw.len(), 4 + n + 16 * m);
    let inside: Vec<usize> = (0..n).filter(|&i| mask[i] != 0).collect();
    assert_eq!(inside.len(), m);
    let at = |a: usize| -> Vec<f64> {
        let o = 4 + n + 4 * a * m;
        (0..m).map(|v| f32::from_le_bytes(raw[o + 4 * v..o + 4 * v + 4].try_into().unwrap()) as f64).collect()
    };
    let grid_vals = |v: Vec<f64>| -> Vec<f64> {
        let mut g = vec![0.0; n];
        for (i, x) in inside.iter().zip(v) {
            g[*i] = x;
        }
        g
    };
    let (chi, r2p) = (grid_vals(at(0)), grid_vals(at(1)));
    let (ref_para, ref_dia) = (at(2), at(3));
    let grid = Grid::new(DIMS.0, DIMS.1, DIMS.2, 1.0, 1.0, 1.0);
    let mut iters = 0;
    let maps = wavesep(&chi, &r2p, &mask, &grid, &WaveSepParams::default(), |i, _| iters = i).expect("wavesep");
    println!("QSM.rs: {iters} iterations");
    let para: Vec<f64> = inside.iter().map(|&i| maps.0[i]).collect();
    let dia: Vec<f64> = inside.iter().map(|&i| -maps.1[i]).collect();
    let nrmse = |a: &[f64], b: &[f64]| (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / b.iter().map(|y| y * y).sum::<f64>()).sqrt();
    let (np, nd) = (nrmse(&para, &ref_para), nrmse(&dia, &ref_dia));
    println!("vs WaveSep: χ+ NRMSE {np:.2e}, |χ−| NRMSE {nd:.2e}");
    // Measured 2026-10-30: χ+ 2.8e-8, |χ−| 2.5e-8 (float32), 13 iterations each; with MATLAB's
    // wavelet alignment it was 10.16 % / 10.04 %.
    assert_eq!(iters, 13, "iterations");
    assert!(np < 1e-6 && nd < 1e-6, "χ+ {np}, χ− {nd}");
}

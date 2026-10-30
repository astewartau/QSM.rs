//! `separation::wavesep` against WaveSep itself (ZhenghanFang/WaveSep f3a192e).
//!
//! `tests/data/wavesep_small.bin` is made by `tests/data/wavesep_small.py` with the verified QSM-CI
//! `wavesep` image (QSMxT/QSM-CI#302), so it can be regenerated in Actions
//! (`.github/workflows/regenerate-fixtures.yml`): a 24x24x24 synthetic case (χ_total, R2′, mask)
//! and WaveSep's χ+ and |χ−| (float32) at its defaults, as QSM-CI's `wavesep` submission runs it.
//!
//! **Known gap** (astewartau/QSM.rs#327): both stop after the same 13 iterations, but differ by
//! ~10 % NRMSE. The wavelet step is the cause: on a smooth 32³ field, QSM.rs's db4 periodized
//! soft-threshold proximal step differs from PyWavelets' (`wavedecn`/`waverecn`,
//! `mode="periodization"`, the same level 2) by 3.4 % relative, so the two bases are not the same.
//! On the larger synthetic case of QSMxT/QSM-CI#302 the gap was 0.11 % (χ+) and 1.08 % (χ−). This
//! test records the measured distance so that a change shows, and is to be tightened once the
//! transform matches PyWavelets'.

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
    println!("vs WaveSep: χ+ NRMSE {:.4} %, |χ−| NRMSE {:.4} %", 100.0 * np, 100.0 * nd);
    // Measured 2026-10-30: χ+ 10.16 %, |χ−| 10.04 %, 13 iterations each (the gap above).
    assert_eq!(iters, 13, "iterations");
    assert!(np < 0.11 && nd < 0.11, "χ+ {np}, χ− {nd}");
}

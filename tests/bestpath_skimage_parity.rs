//! `unwrap::unwrap_bestpath` against scikit-image 0.26.0's 3-D `unwrap_phase` (the original's
//! reference implementation, `unwrap_3d_ljmu.c`).
//!
//! `tests/data/bestpath_skimage_small.bin` is made by `tests/data/bestpath_skimage_small.py` with
//! the verified QSM-CI `py-originals` image (QSMxT/QSM-CI#302), so it can be regenerated in Actions
//! (`.github/workflows/regenerate-fixtures.yml`): two wrapped phantoms, one noiseless and one with
//! noise (σ 0.3 rad), each with the mask, the wrapped phase and scikit-image's unwrapped phase
//! (`rng=0`) at the mask voxels.

use qsm_core::unwrap::{unwrap_bestpath, BestPathParams};
use qsm_core::Grid;

/// Grids of the generator's cases, in order.
const DIMS: [(usize, usize, usize); 2] = [(28, 26, 20), (26, 28, 18)];

/// Per case: (mask, wrapped phase on the grid, scikit-image's unwrapped phase at the mask voxels).
fn load() -> Vec<(Vec<u8>, Vec<f64>, Vec<f64>)> {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/bestpath_skimage_small.bin")).expect("fixture");
    let f64s = |o: usize, k: usize| -> Vec<f64> { (0..k).map(|i| f64::from_le_bytes(raw[o + 8 * i..o + 8 * i + 8].try_into().unwrap())).collect() };
    let mut o = 0;
    let out = DIMS.iter().map(|d| {
        let n = d.0 * d.1 * d.2;
        let m = u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize;
        let mask = raw[o + 4..o + 4 + n].to_vec();
        o += 4 + n;
        let inside: Vec<usize> = (0..n).filter(|&i| mask[i] != 0).collect();
        assert_eq!(inside.len(), m);
        let mut phase = vec![0.0; n];
        for (v, p) in inside.iter().zip(f64s(o, m)) {
            phase[*v] = p;
        }
        let unwrapped = f64s(o + 8 * m, m);
        o += 16 * m;
        (mask, phase, unwrapped)
    }).collect();
    assert_eq!(o, raw.len());
    out
}

#[test]
fn bestpath_matches_scikit_image() {
    for (d, (mask, phase, sk)) in DIMS.iter().zip(load()) {
        let grid = Grid::new(d.0, d.1, d.2, 1.0, 1.0, 1.0);
        let out = unwrap_bestpath(&phase, &mask, &BestPathParams::default(), &grid);
        let got: Vec<f64> = (0..out.len()).filter(|&i| mask[i] != 0).map(|i| out[i]).collect();
        // The mask is one connected component, whose absolute 2π offset is arbitrary (the module
        // docs): compare wrap counts relative to the first voxel's.
        let k0 = ((got[0] - sk[0]) / std::f64::consts::TAU).round();
        let shifted: Vec<f64> = got.iter().map(|a| a - k0 * std::f64::consts::TAU).collect();
        let wraps = shifted.iter().zip(&sk).filter(|(a, b)| ((*a - *b) / std::f64::consts::TAU).round() != 0.0).count();
        let max_d = shifted.iter().zip(&sk).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        println!("{d:?}: global offset {k0}·2π, {wraps} voxels with another relative wrap count, max |Δ| {max_d:.2e} rad");
        // Measured 2026-10-30: the same wrap counts everywhere, and the same values to rounding
        // (max |Δ| 0 and 1.8e-15 rad); in the noisy case QSM.rs's whole volume sits one 2π above
        // scikit-image's.
        assert_eq!(wraps, 0, "{d:?}: wrap counts differ");
        assert!(max_d < 1e-9, "{d:?}: max |Δ| {max_d}");
    }
}

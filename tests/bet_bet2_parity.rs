//! `bet::run_bet` against FSL bet2 (Bostrix/FSL-BET2 d5acd7f), on a small synthetic head.
//!
//! `tests/data/bet_bet2_small.bin` is made by `tests/data/bet_bet2_small.py` with QSM-CI's `bet2`
//! image (QSMxT/QSM-CI#302; built locally from its Dockerfile, as bet2's licence keeps the image
//! unpublished), so it can be regenerated in Actions (`.github/workflows/regenerate-fixtures.yml`).
//! It holds bet2's masks for `-f 0.5` (bet2's default) and `-f 0.2` (QSMART's); the head is
//! rebuilt here exactly as the script builds it.
//!
//! QSM.rs follows bet2 step by step since astewartau/QSM.rs#327 (see `bet::evolution`'s module
//! docs); on a travelling head the masks are identical.

use qsm_core::bet::{run_bet, BetParams, FslLeftRightOrder};
use qsm_core::Grid;

const DIMS: (usize, usize, usize) = (56, 64, 48);

/// The generator's `head()`: 1.5 mm voxels, nested ellipsoids.
fn head() -> Vec<f64> {
    let (nx, ny, nz) = DIMS;
    let mut img = vec![0.0; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let (x, y, z) = (i as f64 - 27.5, j as f64 - 31.5, k as f64 - 23.5);
                let e = |a: f64, b: f64, c: f64| (x / a).powi(2) + (y / b).powi(2) + (z / c).powi(2);
                let v = &mut img[i + nx * (j + ny * k)];
                for (r, val) in [((26.0, 30.0, 22.0), 70.0), ((23.0, 27.0, 19.0), 5.0), ((21.0, 25.0, 17.0), 20.0),
                                 ((19.5, 23.5, 15.5), 80.0), ((16.0, 20.0, 12.5), 110.0)] {
                    if e(r.0, r.1, r.2) < 1.0 {
                        *v = val;
                    }
                }
            }
        }
    }
    img
}

#[test]
fn bet_against_bet2() {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/bet_bet2_small.bin")).expect("fixture");
    let n = DIMS.0 * DIMS.1 * DIMS.2;
    let nb = n.div_ceil(8);
    assert_eq!(raw.len(), 2 * nb);
    let img = head();
    let grid = Grid::new(DIMS.0, DIMS.1, DIMS.2, 1.5, 1.5, 1.5);
    // Measured 2026-10-31: Dice 0.9991 (-f 0.5; 35,875 voxels against 35,870) and 1.0000 (-f 0.2).
    // This piecewise-constant head is sensitive at that level: a 4-ulp change of the initial
    // radius moves the -f 0.5 mask by 2 voxels. (Before the port: 0.9903 and 0.9922.)
    for (c, (f, min_dice)) in [(0.5, 0.998), (0.2, 0.999)].into_iter().enumerate() {
        let bet2: Vec<bool> = (0..n).map(|i| raw[c * nb + i / 8] >> (i % 8) & 1 == 1).collect();
        let params = BetParams { fractional_intensity: f, ..BetParams::default() };
        // nibabel writes diag(1.5, 1.5, 1.5): a positive determinant, so FSL reads the head
        // with x reversed (neurological storage) and bet2 runs on that.
        let ours = run_bet(&img, &grid, &params, FslLeftRightOrder::Neurological, |_, _| {});
        let both = (0..n).filter(|&i| bet2[i] && ours[i] != 0).count();
        let (a, b) = (bet2.iter().filter(|&&v| v).count(), ours.iter().filter(|&&v| v != 0).count());
        let dice = 2.0 * both as f64 / (a + b) as f64;
        println!("-f {f}: Dice {dice:.4}, voxels bet2 {a} / QSM.rs {b}");
        assert!(dice >= min_dice, "-f {f}: Dice {dice}");
    }
}

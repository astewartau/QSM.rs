//! `utils::mppca_denoise` against DIPY 1.12.1's `mppca`, the reference implementation of
//! Veraart et al. 2016.
//!
//! `tests/data/mppca_dipy_small.bin` is made by `tests/data/mppca_dipy_small.py` with the verified
//! QSM-CI `py-originals` image (QSMxT/QSM-CI#302), so it can be regenerated in Actions
//! (`.github/workflows/regenerate-fixtures.yml`): a noisy six-echo 14x14x10 phantom, its mask, and
//! DIPY's output (`patch_radius` 2, `pca_method='eig'`).
//!
//! **Known gap** (astewartau/QSM.rs#327): QSM.rs gives each voxel its own patch's estimate, where
//! DIPY averages the estimates of the overlapping patches, so it is not DIPY's output. This test
//! records the measured distance so that a change shows, and is to be tightened to rounding once
//! the overlapping-patch average reproduces DIPY.

use qsm_core::denoise::mppca_denoise;

const DIMS: (usize, usize, usize) = (14, 14, 10);
const VOLS: usize = 6;

#[test]
fn mppca_against_dipy() {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/mppca_dipy_small.bin")).expect("fixture");
    let n = DIMS.0 * DIMS.1 * DIMS.2;
    assert_eq!(raw.len(), n + 8 * n * VOLS);
    let mask = raw[..n].to_vec();
    let f = |a: usize| -> Vec<f64> {
        let o = n + 4 * a * n * VOLS;
        (0..n * VOLS).map(|i| f32::from_le_bytes(raw[o + 4 * i..o + 4 * i + 4].try_into().unwrap()) as f64).collect()
    };
    let (input, dipy) = (f(0), f(1));
    // Volumes last in the fixture; mppca_denoise takes the volumes of each voxel together.
    let interleave = |v: &[f64]| -> Vec<f64> { (0..n * VOLS).map(|i| v[(i % VOLS) * n + i / VOLS]).collect() };
    let out = mppca_denoise(&interleave(&input), DIMS, VOLS, 2, Some(&mask));
    let (inp, dip) = (interleave(&input), interleave(&dipy));
    let inside: Vec<usize> = (0..n * VOLS).filter(|&i| mask[i / VOLS] != 0).collect();
    let norm = |f: &dyn Fn(usize) -> f64| inside.iter().map(|&i| f(i).powi(2)).sum::<f64>().sqrt();
    let nrmse = norm(&|i| out[i] - dip[i]) / norm(&|i| dip[i]);
    let (dipy_change, our_change) = (norm(&|i| inp[i] - dip[i]) / norm(&|i| dip[i]), norm(&|i| inp[i] - out[i]) / norm(&|i| dip[i]));
    println!("vs DIPY in the mask: NRMSE {:.3} %; change from the input: DIPY {:.3} %, QSM.rs {:.3} %",
             100.0 * nrmse, 100.0 * dipy_change, 100.0 * our_change);
    // Measured 2026-10-30: NRMSE 1.33 %, with DIPY changing the input by 2.28 % and QSM.rs by
    // 1.96 %: the patch-averaging gap above (on a travelling head, 0.97 %).
    assert!(nrmse < 0.02, "NRMSE {nrmse}");
}

//! `utils::gibbs_unring` against DIPY 1.12.1's `gibbs_removal`, the reference implementation of
//! Kellner et al. 2016.
//!
//! `tests/data/gibbs_dipy_small.bin` is made by `tests/data/gibbs_dipy_small.py` with the verified
//! QSM-CI `py-originals` image (QSMxT/QSM-CI#302), so it can be regenerated in Actions
//! (`.github/workflows/regenerate-fixtures.yml`): a 24x24x16 ringing phantom and DIPY's output at
//! its defaults (2-D unringing of each axial slice).
//!
//! **Known gap** (astewartau/QSM.rs#327): QSM.rs unrings in 3-D, generalising DIPY's 2-D weights,
//! so it is not DIPY's output. This test records the measured distance so that a change shows,
//! and is to be tightened to rounding once a 2-D slice-wise mode reproduces DIPY.

use qsm_core::unring::gibbs_unring;

const DIMS: (usize, usize, usize) = (24, 24, 16);

#[test]
fn gibbs_unring_against_dipy() {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/gibbs_dipy_small.bin")).expect("fixture");
    let n = DIMS.0 * DIMS.1 * DIMS.2;
    assert_eq!(raw.len(), 8 * n);
    let f = |a: usize| -> Vec<f64> { (0..n).map(|i| f32::from_le_bytes(raw[4 * (a * n + i)..4 * (a * n + i) + 4].try_into().unwrap()) as f64).collect() };
    let (input, dipy) = (f(0), f(1));
    let out = gibbs_unring(&input, DIMS, 1);
    let norm = |v: &mut dyn Iterator<Item = f64>| v.map(|x| x * x).sum::<f64>().sqrt();
    let dipy_change = norm(&mut input.iter().zip(&dipy).map(|(a, b)| a - b));
    let our_change = norm(&mut input.iter().zip(&out).map(|(a, b)| a - b));
    let nrmse = norm(&mut out.iter().zip(&dipy).map(|(a, b)| a - b)) / norm(&mut dipy.iter().copied());
    println!("vs DIPY: NRMSE {:.3} %; change from the input: DIPY {dipy_change:.1}, QSM.rs {our_change:.1}", 100.0 * nrmse);
    // Measured 2026-10-30: NRMSE 3.02 %, with both changing the input by about as much (DIPY 518,
    // QSM.rs 492, L2): the 3-D vs 2-D gap above (on a travelling head, 2.0 %).
    assert!(nrmse < 0.035, "NRMSE {nrmse}");
}

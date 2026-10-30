//! χ-separation, R2′ and brain extraction against their originals (astewartau/QSM.rs#317):
//!
//! - `tests/data/dl_separation_qsmci_small.bin`, made by `tests/data/dl_separation_qsmci_small.py`
//!   with the `chisepnet` and `susep-net` entries (given the box's local field, χ, R2′ and mask) and
//!   the `r2primenet` entry (given the five echoes' magnitudes and the mask).
//! - `tests/data/dl_hdbet_qsmci_small.bin`, made by `tests/data/dl_hdbet_qsmci_small.py` with the
//!   `hd-bet-qsmci` entry on the 2 mm head of `tests/data/dl_head_small.bin`
//!   (`scripts/dl_fixture_head.py`): HD-BET needs a whole head, which the small box is not.

use qsm_core::Grid;

use super::{assert_parity, data_file, grid, invivo, originals, weights, B0, TES};

/// The originals' outputs, in the generator's order: χ-sepnet χ+ and χ−, SUSEP-Net χ+ and χ− (χ−
/// as a magnitude), and R2PRIMEnet's R2′ (s⁻¹).
fn separation_originals(mask: &[u8]) -> Vec<Vec<f64>> {
    originals("dl_separation_qsmci_small.bin", mask, 5)
}

/// Not to rounding for χ+: 1.7 % NRMSE (χ− 0.045 %), an open gap (astewartau/QSM.rs#330). The
/// original runs SNU's fixed-patch `240904_xsepnet.onnx`, QSM.rs its own export of it. The limits
/// hold it at the size measured; tighten them once it is found.
#[test]
fn chisepnet_matches_its_original() {
    let d = invivo();
    let orig = separation_originals(&d.mask);
    let (pos, neg, _) = qsm_core::separation::chisepnet(
        &d.local_field, &d.chi, &d.r2prime, &d.mask, &grid(), &weights("chi-sepnet"),
        &qsm_core::separation::ChiSepNetNorm::default(), &qsm_core::separation::ChiSepNetParams::default(),
    )
    .unwrap();
    // QSM.rs returns χ− signed (≤ 0); the original writes its magnitude.
    let neg: Vec<f64> = neg.iter().map(|v| -v).collect();
    assert_parity("χ-sepnet χ+", &pos, &d.mask, &orig[0], 1.9, 1.3e-2);
    assert_parity("χ-sepnet χ−", &neg, &d.mask, &orig[1], 6e-2, 3.5e-3);
}

#[test]
fn susep_net_matches_its_original() {
    let d = invivo();
    let orig = separation_originals(&d.mask);
    let (pos, neg, _) = qsm_core::separation::susep_net(
        &d.local_field, &d.chi, &d.r2prime, &d.mask, &grid(), &weights("susep-net"),
        &qsm_core::separation::SusepNetNorm::default(), &qsm_core::separation::SusepNetParams::default(),
        |_, _| {},
    )
    .unwrap();
    let neg: Vec<f64> = neg.iter().map(|v| -v).collect();
    assert_parity("SUSEP-Net χ+", &pos, &d.mask, &orig[2], 2e-4, 2e-6);
    assert_parity("SUSEP-Net χ−", &neg, &d.mask, &orig[3], 2e-4, 2e-6);
}

/// The R2* fit of QSM-CI's `algorithms/r2primenet/recon.py` (`fit_r2star`), its stand-in for
/// ARLO: log-linear least squares over the echoes, weighted by magnitude², clipped at 0.
fn qsmci_r2star(magnitude: &[Vec<f64>]) -> Vec<f64> {
    (0..magnitude[0].len())
        .map(|i| {
            let (mut sw, mut st, mut sl) = (0.0, 0.0, 0.0);
            let m: Vec<f64> = magnitude.iter().map(|e| e[i].max(1e-9)).collect();
            for (e, &v) in m.iter().enumerate() {
                let w = v * v;
                sw += w;
                st += w * TES[e];
                sl += w * v.ln();
            }
            let (tbar, lbar) = (st / sw, sl / sw);
            let (mut cov, mut var) = (0.0, 0.0);
            for (e, &v) in m.iter().enumerate() {
                let w = v * v;
                cov += w * (TES[e] - tbar) * (v.ln() - lbar);
                var += w * (TES[e] - tbar).powi(2);
            }
            let slope = if var > 0.0 { cov / var } else { 0.0 };
            (-slope).max(0.0)
        })
        .collect()
}

/// The network: QSM.rs's `r2primenet` given the R2* the original's wrapper fits. QSM.rs's own entry
/// point from the magnitude, `r2primenet_from_magnitude`, fits R2* with ARLO, as the toolbox's
/// training data did; the wrapper's fit is a stand-in (see `src/utils/r2primenet.rs`).
#[test]
fn r2primenet_matches_its_original() {
    let d = invivo();
    let r2star = qsmci_r2star(&d.magnitude);
    let r2p = qsm_core::utils::r2primenet(
        &r2star, &d.mask, &grid(), B0, &weights("r2primenet"),
        &qsm_core::utils::R2PrimeNetNorm::default(), &qsm_core::utils::R2PrimeNetParams::default(), |_, _| {},
    )
    .unwrap();
    assert_parity("R2PRIMEnet", &r2p, &d.mask, &separation_originals(&d.mask)[4], 2e-4, 5e-4);
}

/// The original runs the HD-BET v2 CLI: nnU-Net's sliding window, no mirroring (`--disable_tta`).
/// The masks are identical (Dice 1); the limit leaves a few edge voxels for other CPUs' rounding.
#[test]
fn hdbet_matches_its_original() {
    const DIMS: [usize; 3] = [64, 80, 48];
    let n = DIMS.iter().product::<usize>();
    let head: Vec<f64> =
        data_file("dl_head_small.bin").as_chunks::<2>().0.iter().map(|&b| u16::from_le_bytes(b) as f64).collect();
    assert_eq!(head.len(), n);
    let bits = data_file("dl_hdbet_qsmci_small.bin");
    let original: Vec<bool> = (0..n).map(|i| bits[i / 8] >> (i % 8) & 1 == 1).collect();
    let params = qsm_core::bet::HdBetParams {
        inference: qsm_core::bet::HdBetInference::SLIDING_WINDOW,
        mirror_tta: false,
    };
    let grid = Grid::new(DIMS[0], DIMS[1], DIMS[2], 2.0, 2.0, 2.0);
    let mask = qsm_core::bet::hd_bet(&head, &grid, &weights("hd-bet"), &params, |_, _| {}).unwrap();
    let (a, b) = (mask.iter().filter(|&&m| m != 0).count(), original.iter().filter(|&&m| m).count());
    let both = mask.iter().zip(&original).filter(|(&m, &o)| m != 0 && o).count();
    let dice = 2.0 * both as f64 / (a + b) as f64;
    println!("HD-BET: Dice {dice:.6} ({a} voxels, the original's {b})");
    assert!(dice >= 0.999, "HD-BET: Dice {dice}");
}


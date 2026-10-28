//! Integration test: χ-separation on the QSM-CI chisep phantom, scored with the
//! QSM-CI metrics (correlation / XSIM / NRMSE per source map) so results are
//! directly comparable to the leaderboard entries:
//!
//! | entry (MATLAB, `-iso` runs)   | χ+ xsim/corr | χ− xsim/corr |
//! |-------------------------------|--------------|--------------|
//! | chi-sep-ilsqr (SNU-LIST)      | 0.590/0.862  | 0.480/0.807  |
//! | chi-sep-medi  (SNU-LIST)      | 0.731/0.904  | 0.580/0.911  |
//!
//! The chi-sep-medi entry's container never iterates (its empty CSF mask makes every step NaN,
//! see `separation::chi_sep_medi`), so its row scores the toolbox's initial guess.
//! | chi-sepnet    (deep learning) | 0.887/0.944  | 0.719/0.884  |
//!
//! Caveat: the phantom was regenerated after those runs (GT corr 0.935 between
//! versions), so posted numbers are a baseline zone, not an exact target.
//! The pipeline mirrors QSM-CI's chi-sep-ilsqr entry: the conventional QSM fed
//! to χ-separation is reconstructed in-house with our iLSQR from the same local
//! field (feeding an external χ_total is known to break the toolbox — see the
//! entry's recon.m). The shared phantom loader and scoring live in [`common`].
//!
//! Run with:
//!   QSMCI_CHISEP=/home/ashley/repos/qsm/qsmci/qsmci/data/sim/chisep \
//!     cargo test --release --features parallel --test chisep_qsmci -- --ignored --nocapture

mod common;

use common::{assert_chisep_separated, chisep_score, load_chisep_phantom, save_center_slices};
use qsm_core::inversion::ilsqr;
use qsm_core::separation::{chi_sep_ilsqr, chi_sep_medi, ChiSepIlsqrInputs, ChiSepIlsqrParams, ChiSepMediInputs, ChiSepMediParams};
use std::time::Instant;

#[test]
#[ignore] // needs the qsmci phantom; run with --release (see module docs)
fn test_chi_sep_ilsqr_qsmci() {
    let Some(ph) = load_chisep_phantom() else { return };
    let (nx, ny, nz) = ph.dims;
    println!("[INFO] phantom {}x{}x{}  cf {:.1} MHz", nx, ny, nz, ph.cf / 1e6);

    // Conventional QSM from the same local field with our iLSQR (ppm in → ppm out).
    let t = Instant::now();
    let (qsm, _, _, _) = ilsqr(
        &ph.local_field_ppm,
        &ph.mask,
        &ph.grid,
        ph.bdir,
        // As the χ-separation toolbox's Chisep_script.m calls QSM_iLSQR (100 iterations, 12 mm).
        &ChiSepIlsqrParams::default().qsm_ilsqr,
        |_, _| {},
    );
    println!("[INFO] iLSQR QSM in {:.1?}", t.elapsed());

    let params = ChiSepIlsqrParams { cf: ph.cf, ..ChiSepIlsqrParams::default() };
    let inputs = ChiSepIlsqrInputs { local_field: &ph.local_field_ppm, r2prime: &ph.r2prime, qsm: &qsm, mask: &ph.mask };
    let t = Instant::now();
    let (chi_pos, chi_neg, _) = chi_sep_ilsqr(&inputs, &ph.grid, ph.bdir, &params, |i, tot| println!("  NNLS iter {}/{}", i, tot))
        .expect("chi_sep_ilsqr");
    let secs = t.elapsed().as_secs_f64();
    println!("[INFO] chi_sep_ilsqr done in {:.1?}", t.elapsed());

    let dia_mag: Vec<f64> = chi_neg.iter().map(|&v| -v).collect();
    println!("[RESULT] chi_sep_ilsqr (target zone: MATLAB chi-sep-ilsqr para 0.59/0.86, dia 0.48/0.81):");
    chisep_score("chi_sep_ilsqr χ+", &chi_pos, &ph.gt_para, &ph.mask, ph.dims, secs);
    chisep_score("chi_sep_ilsqr χ−", &dia_mag, &ph.gt_dia, &ph.mask, ph.dims, secs);

    // Center-slice figures for the CI PR comment (rendered by render_slices.py).
    // The ground truth is saved too, so the consolidated χ-separation montage can show a
    // reference panel alongside the methods, as the BFR and dipole montages already do.
    save_center_slices(&ph.gt_para, &ph.mask, ph.dims, "chisep_para_truth");
    save_center_slices(&ph.gt_dia, &ph.mask, ph.dims, "chisep_dia_truth");
    save_center_slices(&chi_pos, &ph.mask, ph.dims, "chisep_para");
    save_center_slices(&dia_mag, &ph.mask, ph.dims, "chisep_dia");

    // Conservative gates: the CI dataset (chisep-mc, multicompartment) is harder
    // than the local single-compartment phantom, so these are regression floors,
    // not the headline numbers (chisep phantom: para 0.95, dia 0.87).
    assert_chisep_separated("chi_sep_ilsqr", &chi_pos, &dia_mag, &ph.gt_para, &ph.gt_dia, &ph.mask, 0.6, 0.4);
}

/// Regression floors for `chi_sep_medi`'s χ+ / |χ−| correlation with ground truth, kept from the
/// earlier paper-based port; the toolbox port's own scores are in the CI metrics.
const MEDI_PARA_FLOOR: f64 = 0.6;
const MEDI_DIA_FLOOR: f64 = 0.4;

#[test]
#[ignore] // needs the qsmci phantom; run with --release (see module docs)
fn test_chi_sep_medi_qsmci() {
    let Some(ph) = load_chisep_phantom() else { return };
    let n = ph.mask.len();
    let ne = ph.tes.len();
    println!("[INFO] chi_sep_medi on qsmci phantom");
    // The inputs as Chisep_script.m makes them: STI Suite iLSQR QSM of the local field,
    // N_std from Fit_ppm_complex, a CSF mask from extract_CSF on the ARLO R2*.
    let (qsm, _, _, _) = ilsqr(&ph.local_field_ppm, &ph.mask, &ph.grid, ph.bdir, &ChiSepIlsqrParams::default().qsm_ilsqr, |_, _| {});
    let echo_major: Vec<f64> = (0..ne).flat_map(|e| (0..n).map(move |v| (v, e))).map(|(v, e)| ph.mag_voxel_major[v * ne + e]).collect();
    let n_std = qsm_core::utils::fit_ppm_complex_noise(&echo_major, ne);
    let r2s = qsm_core::utils::r2star::arlo(&echo_major, &ph.tes, n).expect("arlo");
    let mask_csf = qsm_core::inversion::extract_csf(&r2s, &ph.mask, &ph.grid, 5.0, true);
    println!("[INFO] CSF mask: {} voxels", mask_csf.iter().filter(|&&m| m != 0).count());
    let inputs = ChiSepMediInputs {
        magnitude: &ph.magnitude_rss,
        local_field: &ph.local_field_ppm,
        r2prime: &ph.r2prime,
        n_std: &n_std,
        qsm: &qsm,
        mask: &ph.mask,
        mask_csf: &mask_csf,
    };
    let params = ChiSepMediParams { cf: ph.cf, ..ChiSepMediParams::default() };
    let t = Instant::now();
    let (chi_pos, chi_neg, _) = chi_sep_medi(&inputs, &ph.grid, ph.bdir, &params, |i, tot| println!("  iter {}/{}", i, tot)).expect("chi_sep_medi");
    let secs = t.elapsed().as_secs_f64();
    println!("[INFO] chi_sep_medi done in {:.1?}", t.elapsed());

    let dia_mag: Vec<f64> = chi_neg.iter().map(|&v| -v).collect();
    println!("[RESULT] chi_sep_medi:");
    chisep_score("chi_sep_medi χ+", &chi_pos, &ph.gt_para, &ph.mask, ph.dims, secs);
    chisep_score("chi_sep_medi χ−", &dia_mag, &ph.gt_dia, &ph.mask, ph.dims, secs);
    save_center_slices(&chi_pos, &ph.mask, ph.dims, "chisep_medi_para");
    save_center_slices(&dia_mag, &ph.mask, ph.dims, "chisep_medi_dia");
    assert_chisep_separated("chi_sep_medi", &chi_pos, &dia_mag, &ph.gt_para, &ph.gt_dia, &ph.mask, MEDI_PARA_FLOOR, MEDI_DIA_FLOOR);
}

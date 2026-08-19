//! Integration test: χ-separation on the QSM-CI chisep phantom, scored with the
//! QSM-CI metrics (correlation / XSIM / NRMSE per source map) so results are
//! directly comparable to the leaderboard entries:
//!
//! | entry (MATLAB, `-iso` runs)   | χ+ xsim/corr | χ− xsim/corr |
//! |-------------------------------|--------------|--------------|
//! | chi-sep-ilsqr (SNU-LIST)      | 0.590/0.862  | 0.480/0.807  |
//! | chi-sep-medi  (SNU-LIST)      | 0.731/0.904  | 0.580/0.911  |
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

use common::{chisep_score, correlation, load_chisep_phantom, save_center_slices};
use qsm_core::inversion::{ilsqr, IlsqrParams};
use qsm_core::separation::{chi_sep_ilsqr, chi_sep_medi, ChiSepIlsqrParams, ChiSepParams};
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
        &IlsqrParams::default(),
        |_, _| {},
    );
    println!("[INFO] iLSQR QSM in {:.1?}", t.elapsed());

    let lambda1: f64 = std::env::var("CHISEP_LAMBDA1")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let params = ChiSepIlsqrParams {
        cf: ph.cf,
        lambda1,
        ..ChiSepIlsqrParams::default()
    };
    println!("[INFO] chi_sep_ilsqr  lambda1 {}", lambda1);
    let t = Instant::now();
    let (chi_pos, chi_neg, _) = chi_sep_ilsqr(
        &ph.local_field_ppm,
        &ph.r2prime,
        &ph.magnitude_rss,
        &qsm,
        &ph.mask,
        &ph.grid,
        ph.bdir,
        &params,
        |i, tot| {
            if i % 5 == 0 || i == tot {
                println!("  iter {}/{}", i, tot);
            }
        },
    );
    let secs = t.elapsed().as_secs_f64();
    println!("[INFO] chi_sep_ilsqr done in {:.1?}", t.elapsed());

    let dia_mag: Vec<f64> = chi_neg.iter().map(|&v| -v).collect();
    println!("[RESULT] chi_sep_ilsqr (target zone: MATLAB chi-sep-ilsqr para 0.59/0.86, dia 0.48/0.81):");
    chisep_score("chi_sep_ilsqr χ+", &chi_pos, &ph.gt_para, &ph.mask, ph.dims, secs);
    chisep_score("chi_sep_ilsqr χ−", &dia_mag, &ph.gt_dia, &ph.mask, ph.dims, secs);

    // Center-slice figures for the CI PR comment (rendered by render_slices.py).
    save_center_slices(&chi_pos, &ph.mask, ph.dims, "chisep_para");
    save_center_slices(&dia_mag, &ph.mask, ph.dims, "chisep_dia");

    // Conservative gates: the CI dataset (chisep-mc, multicompartment) is harder
    // than the local single-compartment phantom, so these are regression floors,
    // not the headline numbers (chisep phantom: para 0.95, dia 0.87).
    let para_corr = correlation(&chi_pos, &ph.gt_para, &ph.mask);
    let dia_corr = correlation(&dia_mag, &ph.gt_dia, &ph.mask);
    assert!(para_corr > 0.6, "chi+ corr {:.3} too low", para_corr);
    assert!(dia_corr > 0.4, "chi- corr {:.3} too low", dia_corr);
}

#[test]
#[ignore] // needs the qsmci phantom; run with --release (see module docs)
fn test_chi_sep_medi_qsmci() {
    let Some(ph) = load_chisep_phantom() else { return };
    println!("[INFO] chi_sep_medi on qsmci phantom (posted MATLAB chi-sep-medi: para 0.73/0.90, dia 0.58/0.91)");

    let params = ChiSepParams {
        cf: ph.cf,
        dr_pos: 137.0,
        dr_neg: 137.0,
        max_iter: 30,
        ..ChiSepParams::default()
    };
    let t = Instant::now();
    let (chi_pos, chi_neg, _) = chi_sep_medi(
        &ph.local_field_ppm,
        &ph.r2prime,
        &ph.magnitude_rss,
        &ph.mask,
        &ph.grid,
        ph.bdir,
        &params,
        |i, tot| {
            if i % 5 == 0 || i == tot {
                println!("  iter {}/{}", i, tot);
            }
        },
    );
    let secs = t.elapsed().as_secs_f64();
    println!("[INFO] chi_sep_medi done in {:.1?}", t.elapsed());

    let dia_mag: Vec<f64> = chi_neg.iter().map(|&v| -v).collect();
    println!("[RESULT] chi_sep_medi:");
    chisep_score("chi_sep_medi χ+", &chi_pos, &ph.gt_para, &ph.mask, ph.dims, secs);
    chisep_score("chi_sep_medi χ−", &dia_mag, &ph.gt_dia, &ph.mask, ph.dims, secs);
}

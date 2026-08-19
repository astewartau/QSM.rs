//! Integration test: relaxometry-based χ-separation on the QSM-CI chisep phantom,
//! scored with the QSM-CI metrics (correlation / XSIM / NRMSE per source map) so
//! results are directly comparable to the leaderboard entries.
//!
//! Four methods that separate χ+ / χ− from a *provided* conventional QSM
//! (χ_total) plus relaxometry / magnitude, without a field-inversion of their own:
//!   - **R2\*-QSM** (Dimov 2022): closed form from χ_total + R2* (fit here from the
//!     multi-echo magnitude). No R2' needed.
//!   - **WaveSep** (Fang 2023): wavelet-L1 proximal-gradient from χ_total + R2'.
//!   - **DECOMPOSE** (Chen 2021): signal-domain 3-compartment per-voxel fit from
//!     χ_total + multi-echo magnitude.
//!   - **HC-ChiSep** (Wharton & Bowtell model): hollow-cylinder fit with signal-
//!     derived fibre orientation from χ_total + R2' + multi-echo magnitude.
//!
//! All consume the phantom's `inputs/chimap.nii.gz` (the provided QSM), matching
//! the QSM-CI chi-separation contract. The shared phantom loader and scoring live
//! in [`common`].
//!
//! Run with:
//!   QSMCI_CHISEP=/home/ashley/repos/qsm/qsmci/qsmci/data/sim/chisep \
//!     cargo test --release --features parallel --test chisep_relaxometry_qsmci -- --ignored --nocapture

mod common;

use common::{chisep_score, correlation, load_chisep_phantom, save_center_slices};
use qsm_core::separation::{
    decompose, hc_chisep, r2star_qsm_from_magnitude, wavesep, DecomposeParams, HcChisepParams,
    R2starQsmParams, WaveSepParams,
};
use std::time::Instant;

#[test]
#[ignore] // needs the qsmci phantom; run with --release (see module docs)
fn test_r2star_qsm_qsmci() {
    let Some(ph) = load_chisep_phantom() else { return };
    let (nx, ny, nz) = ph.dims;
    println!("[INFO] R2*-QSM on qsmci chisep phantom {}x{}x{}  B0 {} T", nx, ny, nz, ph.b0);

    let params = R2starQsmParams { b0: ph.b0, r_const_3t: 274.0 };
    let t = Instant::now();
    let (chi_pos, chi_neg, _chi_total) = r2star_qsm_from_magnitude(
        &ph.mag_voxel_major,
        &ph.tes,
        &ph.chi_total,
        &ph.mask,
        &params,
    );
    let secs = t.elapsed().as_secs_f64();
    println!("[INFO] R2*-QSM done in {:.2?}", t.elapsed());

    // GT diamagnetic map is stored as a positive magnitude; χ− is returned signed.
    let dia_mag: Vec<f64> = chi_neg.iter().map(|&v| -v).collect();
    println!("[RESULT] r2star-qsm (closed-form from χ_total + R2*):");
    chisep_score("R2*-QSM χ+", &chi_pos, &ph.gt_para, &ph.mask, ph.dims, secs);
    chisep_score("R2*-QSM χ−", &dia_mag, &ph.gt_dia, &ph.mask, ph.dims, secs);

    // Figures for the CI PR comment (rendered by render_slices.py).
    save_center_slices(&chi_pos, &ph.mask, ph.dims, "relaxchisep_r2starqsm_para");
    save_center_slices(&dia_mag, &ph.mask, ph.dims, "relaxchisep_r2starqsm_dia");

    // Regression floors — R2*-QSM's single-constant model is a coarse separator,
    // so these are conservative (it carries the irreversible R2 baseline).
    let para_corr = correlation(&chi_pos, &ph.gt_para, &ph.mask);
    assert!(para_corr > 0.3, "R2*-QSM chi+ corr {:.3} too low", para_corr);
}

#[test]
#[ignore] // needs the qsmci phantom; run with --release (see module docs)
fn test_wavesep_qsmci() {
    let Some(ph) = load_chisep_phantom() else { return };
    let (nx, ny, nz) = ph.dims;
    println!("[INFO] WaveSep on qsmci chisep phantom {}x{}x{}  B0 {} T", nx, ny, nz, ph.b0);

    // Dr = 137 Hz/ppm is the qsm-forward phantom's single kernel (Shin 2022).
    let params = WaveSepParams::default();
    let t = Instant::now();
    let (chi_pos, chi_neg, _chi_total) = wavesep(
        &ph.chi_total,
        &ph.r2prime,
        &ph.mask,
        &ph.grid,
        &params,
        |i, tot| {
            if i % 20 == 0 || i == tot {
                println!("  iter {}/{}", i, tot);
            }
        },
    );
    let secs = t.elapsed().as_secs_f64();
    println!("[INFO] WaveSep done in {:.2?}", t.elapsed());

    // GT diamagnetic map is stored as a positive magnitude; χ− is returned signed.
    let dia_mag: Vec<f64> = chi_neg.iter().map(|&v| -v).collect();
    println!("[RESULT] wavesep (wavelet-L1 from χ_total + R2'):");
    chisep_score("WaveSep χ+", &chi_pos, &ph.gt_para, &ph.mask, ph.dims, secs);
    chisep_score("WaveSep χ−", &dia_mag, &ph.gt_dia, &ph.mask, ph.dims, secs);

    // Figures for the CI PR comment (rendered by render_slices.py).
    save_center_slices(&chi_pos, &ph.mask, ph.dims, "relaxchisep_wavesep_para");
    save_center_slices(&dia_mag, &ph.mask, ph.dims, "relaxchisep_wavesep_dia");

    let para_corr = correlation(&chi_pos, &ph.gt_para, &ph.mask);
    let dia_corr = correlation(&dia_mag, &ph.gt_dia, &ph.mask);
    assert!(para_corr > 0.5, "WaveSep chi+ corr {:.3} too low", para_corr);
    assert!(dia_corr > 0.4, "WaveSep chi- corr {:.3} too low", dia_corr);
}

#[test]
#[ignore] // needs the qsmci phantom; run with --release (per-voxel fit is heavy)
fn test_decompose_qsmci() {
    let Some(ph) = load_chisep_phantom() else { return };
    let (nx, ny, nz) = ph.dims;
    println!("[INFO] DECOMPOSE on qsmci chisep phantom {}x{}x{}  B0 {} T", nx, ny, nz, ph.b0);

    // Signal-domain 3-compartment per-voxel fit from the provided QSM + multi-echo
    // magnitude. n_inner can be lowered via env for a quick smoke run.
    let n_inner = std::env::var("DECOMPOSE_NINNER").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let params = DecomposeParams { b0: ph.b0, n_inner, ..DecomposeParams::default() };

    let t = Instant::now();
    let (chi_pos, chi_neg, _chi_total) = decompose(
        &ph.chi_total,
        &ph.mag_voxel_major,
        &ph.tes,
        &ph.mask,
        &params,
        |_, _| {},
    );
    let secs = t.elapsed().as_secs_f64();
    println!("[INFO] DECOMPOSE done in {:.2?}", t.elapsed());

    // GT diamagnetic map is stored as a positive magnitude; χ− is returned signed.
    let dia_mag: Vec<f64> = chi_neg.iter().map(|&v| -v).collect();
    println!("[RESULT] decompose (signal-domain from χ_total + multi-echo magnitude):");
    chisep_score("DECOMPOSE χ+", &chi_pos, &ph.gt_para, &ph.mask, ph.dims, secs);
    chisep_score("DECOMPOSE χ−", &dia_mag, &ph.gt_dia, &ph.mask, ph.dims, secs);

    save_center_slices(&chi_pos, &ph.mask, ph.dims, "relaxchisep_decompose_para");
    save_center_slices(&dia_mag, &ph.mask, ph.dims, "relaxchisep_decompose_dia");

    // Conservative floor: DECOMPOSE is a coarse per-voxel separator on this phantom.
    let para_corr = correlation(&chi_pos, &ph.gt_para, &ph.mask);
    assert!(para_corr > 0.2, "DECOMPOSE chi+ corr {:.3} too low", para_corr);
}

#[test]
#[ignore] // needs the qsmci phantom; run with --release
fn test_hc_chisep_qsmci() {
    let Some(ph) = load_chisep_phantom() else { return };
    let (nx, ny, nz) = ph.dims;
    println!("[INFO] HC-ChiSep on qsmci chisep phantom {}x{}x{}  B0 {} T", nx, ny, nz, ph.b0);

    let params = HcChisepParams {
        b0: ph.b0,
        se_echo_times: ph.se_tes.clone(),
        ..HcChisepParams::default()
    };
    let t = Instant::now();
    let (chi_pos, chi_neg, _chi_total) = hc_chisep(
        &ph.chi_total,
        &ph.r2prime,
        &ph.mag_voxel_major,
        &ph.tes,
        ph.se_mag_voxel_major.as_deref(),
        &ph.mask,
        &ph.grid,
        &params,
        |i, tot| println!("  stage {}/{}", i, tot),
    );
    let secs = t.elapsed().as_secs_f64();
    println!("[INFO] HC-ChiSep done in {:.2?}", t.elapsed());

    // GT diamagnetic map is stored as a positive magnitude; χ− is returned signed.
    let dia_mag: Vec<f64> = chi_neg.iter().map(|&v| -v).collect();
    println!("[RESULT] hc_chisep (hollow-cylinder from χ_total + R2' + magnitude):");
    chisep_score("HC-ChiSep χ+", &chi_pos, &ph.gt_para, &ph.mask, ph.dims, secs);
    chisep_score("HC-ChiSep χ−", &dia_mag, &ph.gt_dia, &ph.mask, ph.dims, secs);

    save_center_slices(&chi_pos, &ph.mask, ph.dims, "relaxchisep_hcchisep_para");
    save_center_slices(&dia_mag, &ph.mask, ph.dims, "relaxchisep_hcchisep_dia");

    let para_corr = correlation(&chi_pos, &ph.gt_para, &ph.mask);
    let dia_corr = correlation(&dia_mag, &ph.gt_dia, &ph.mask);
    assert!(para_corr > 0.4, "HC-ChiSep chi+ corr {:.3} too low", para_corr);
    assert!(dia_corr > 0.4, "HC-ChiSep chi- corr {:.3} too low", dia_corr);
}

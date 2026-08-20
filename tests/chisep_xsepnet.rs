//! Integration test: χ-sepnet (deep learning) on the QSM-CI chisep phantom,
//! scored with the QSM-CI metrics (correlation / XSIM / NRMSE per source map) so
//! results flow into the same summary comment as the classical χ-separation
//! methods. Weights are fetched from the registry at run time (needs the
//! `download` feature, or `$QSM_MODEL_DIR`). Ignored by default.
//!
//! Run with:
//!   QSMCI_CHISEP=/home/ashley/repos/qsm/qsmci/qsmci/data/sim/chisep \
//!     cargo test --release --features "onnx download" --test chisep_xsepnet -- --ignored --nocapture
#![cfg(feature = "onnx")]

mod common;

use common::{chisep_score, load_chisep_phantom, save_center_slices};
use qsm_core::separation::{chisepnet, ChiSepNetNorm};
use std::time::Instant;

#[test]
#[ignore]
fn test_chisep_xsepnet() {
    let Some(ph) = load_chisep_phantom() else {
        println!("Skipping: chisep phantom not found");
        return;
    };
    let (nx, ny, nz) = ph.dims;
    println!("[INFO] phantom {}x{}x{}", nx, ny, nz);

    let spec = qsm_core::models::find_model("chi-sepnet").expect("chi-sepnet in registry");
    let w = qsm_core::models::primary_weight_bytes(spec)
        .expect("chi-sepnet weights (set $QSM_MODEL_DIR or build with 'download')");

    // Feed the provided conventional QSM χ_total (the network's QSM input channel),
    // the local field and R2′ — matching the SNU-LIST recon.py.
    let t = Instant::now();
    let (chi_pos, chi_neg, _tot) = chisepnet(
        &ph.local_field_ppm,
        &ph.chi_total,
        &ph.r2prime,
        &ph.mask,
        &ph.grid,
        &w,
        &ChiSepNetNorm::default(),
    )
    .expect("chisepnet");
    let secs = t.elapsed().as_secs_f64();

    // χ− is returned signed (≤0); the ground truth stores it as a positive magnitude.
    let dia_mag: Vec<f64> = chi_neg.iter().map(|&v| -v).collect();
    chisep_score("chisepnet χ+", &chi_pos, &ph.gt_para, &ph.mask, ph.dims, secs);
    chisep_score("chisepnet χ−", &dia_mag, &ph.gt_dia, &ph.mask, ph.dims, secs);
    save_center_slices(&chi_pos, &ph.mask, ph.dims, "chisep_xsepnet_para");
    save_center_slices(&dia_mag, &ph.mask, ph.dims, "chisep_xsepnet_dia");

    assert!(chi_pos.iter().all(|v| v.is_finite()), "chisepnet produced non-finite χ+");
    assert!(dia_mag.iter().all(|v| v.is_finite()), "chisepnet produced non-finite χ−");
}

//! Integration test: AMP-PE dipole inversion on the qsm-forward BIDS phantom.
//!
//! AMP-PE (Approximate Message Passing with Parameter Estimation; Huang et al.,
//! MRM 2023) is a nonlinear dipole inversion with a sparse-wavelet prior and a
//! Gaussian-mixture noise model. The Rust port here has been verified against the
//! original MATLAB (`github.com/EmoryCN2L/QSM_AMP_PE`) to nrmse ~1e-6 / corr ~1.0
//! on matched inputs (db1/db2, nlevel 2/3, with/without magnitude, up to the full
//! two-stage solve); this test checks end-to-end quality against the ground-truth
//! susceptibility map on real-scale data.
//!
//! AMP-PE returns a near-zero-mean χ within the mask (the constant offset is
//! unresolvable from the local field alone), so the fair metric is the
//! offset-invariant masked correlation.
//!
//! Run with (data path defaults to the local qsm-forward derivatives):
//!   AMPPE_BIDS=/home/ashley/bids/derivatives/qsm-forward/sub-1/anat \
//!     cargo test --release --test amppe_qsmci -- --ignored --nocapture

mod common;

use common::{correlation, load_nifti_file, nrmse};
use qsm_core::inversion::{amp_pe, AmpPeParams};
use qsm_core::Grid;
use std::path::Path;
use std::time::Instant;

fn base_dir() -> String {
    std::env::var("AMPPE_BIDS")
        .unwrap_or_else(|_| "/home/ashley/bids/derivatives/qsm-forward/sub-1/anat".to_string())
}

#[test]
#[ignore] // needs the qsm-forward phantom; run with --release --ignored
fn amp_pe_bids_quality() {
    let base = base_dir();
    if !Path::new(&format!("{}/sub-1_fieldmap-local.nii", base)).exists() {
        println!("Skipping: phantom not found at {}", base);
        return;
    }

    let lf = load_nifti_file(&format!("{}/sub-1_fieldmap-local.nii", base)).unwrap();
    let mask_nii = load_nifti_file(&format!("{}/sub-1_mask.nii", base)).unwrap();
    let gt = load_nifti_file(&format!("{}/sub-1_Chimap.nii", base)).unwrap();

    let dims = lf.dims;
    let (vx, vy, vz) = lf.voxel_size;
    let grid = Grid::new(dims.0, dims.1, dims.2, vx, vy, vz);
    let mask: Vec<u8> = mask_nii.data.iter().map(|&m| if m > 0.5 { 1 } else { 0 }).collect();

    // Iteration count is overridable to keep the test tractable; the default is a
    // reduced count that already recovers the structure (full quality uses 25+25).
    let max_lin: usize = std::env::var("AMPPE_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    let params = AmpPeParams {
        wave_order: 1,
        nlevel: 3,
        max_linearization_ite: max_lin,
        b0: 3.0,
        gyro_ratio: 42.58,
        simulated_te: 8e-3,
        ..Default::default()
    };

    let t = Instant::now();
    let chi = amp_pe(&lf.data, &mask, None, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});
    let elapsed = t.elapsed();

    let corr = correlation(&chi, &gt.data, &mask);
    let nr = nrmse(&chi, &gt.data, &mask);
    println!(
        "AMP-PE ({max_lin}+{max_lin} iters, {:.1}s): masked corr vs GT = {:.4}, nrmse = {:.4}",
        elapsed.as_secs_f64(),
        corr,
        nr
    );

    assert!(chi.iter().all(|v| v.is_finite()), "chi must be finite");
    // Offset-invariant structural agreement with the ground truth. AMP-PE without
    // magnitude weighting reaches ~0.79 masked-corr on this phantom at 8+8 iters;
    // the threshold is a loose smoke-gate (quality, not a benchmark).
    assert!(corr > 0.70, "AMP-PE corr vs GT too low: {corr:.4}");
}

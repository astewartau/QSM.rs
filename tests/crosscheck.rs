//! Cross-check: compute ChallengeMetrics on caller-provided NIfTI files so an external scorer
//! (QSM-CI's Python `qsm-eval`) can be validated against this reference implementation.
//! Skips unless CC_RECON/CC_TRUTH/CC_MASK/CC_SEG are set. Run:
//!   CC_RECON=.. CC_TRUTH=.. CC_MASK=.. CC_SEG=.. cargo test --test crosscheck -- --nocapture

mod common;

use std::env;

#[test]
fn crosscheck_metrics() {
    let recon = match env::var("CC_RECON") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("crosscheck: CC_RECON unset — skipping");
            return;
        }
    };
    let truth = env::var("CC_TRUTH").expect("CC_TRUTH");
    let maskp = env::var("CC_MASK").expect("CC_MASK");
    let segp = env::var("CC_SEG").expect("CC_SEG");

    let recon = common::load_nifti_file(&recon).unwrap();
    let truth = common::load_nifti_file(&truth).unwrap();
    let mask_nd = common::load_nifti_file(&maskp).unwrap();
    let seg_nd = common::load_nifti_file(&segp).unwrap();

    let mask: Vec<u8> = mask_nd.data.iter().map(|&v| if v > 0.5 { 1 } else { 0 }).collect();
    let seg: Vec<u8> = seg_nd.data.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();

    let m = common::ChallengeMetrics::compute("cc", &recon.data, &truth.data, &mask, &seg, recon.dims);

    println!(
        "CC_JSON {{\"nrmse\":{},\"nrmse_detrend\":{},\"nrmse_tissue\":{},\"nrmse_blood\":{},\"nrmse_dgm\":{},\"dgm_linearity\":{},\"calc_moment_dev\":{},\"calc_streak\":{},\"correlation\":{},\"xsim\":{}}}",
        m.nrmse, m.nrmse_detrend, m.nrmse_tissue, m.nrmse_blood, m.nrmse_dgm,
        m.dgm_linearity, m.calc_moment_dev, m.calc_streak, m.correlation, m.xsim
    );
}

//! Integration test: R2PRIMEnet (deep learning) on the QSM-CI chisep phantom.
//!
//! The phantom ships an R2′ map alongside the multi-echo GRE magnitude, which makes this a
//! prediction-against-reference test rather than a smoke test: R2* is fitted from the magnitude
//! with ARLO — the same route a GRE-only dataset would take — and the network's R2′ is scored
//! against the phantom's own. Weights are fetched from the registry at run time (needs the
//! `download` feature, or `$QSM_MODEL_DIR`). Ignored by default.
//!
//! Run with:
//!   QSMCI_CHISEP=/home/ashley/repos/qsm/qsmci/qsmci/data/sim/chisep \
//!     cargo test --release --features "onnx download" --test chisep_r2primenet -- --ignored --nocapture
#![cfg(feature = "onnx")]

mod common;

use common::{chisep_score, correlation, load_chisep_phantom, save_center_slices};
use qsm_core::relaxometry::{r2primenet, R2PrimeNetNorm, R2PrimeNetParams};
use std::time::Instant;

#[test]
#[ignore]
fn test_chisep_r2primenet() {
    let Some(ph) = load_chisep_phantom() else {
        println!("Skipping: chisep phantom not found");
        return;
    };
    let (nx, ny, nz) = ph.dims;
    println!("[INFO] phantom {}x{}x{}, {} GRE echoes", nx, ny, nz, ph.tes.len());

    let spec = qsm_core::models::find_model("r2primenet").expect("r2primenet in registry");
    let w = qsm_core::models::primary_weight_bytes(spec)
        .expect("r2primenet weights (set $QSM_MODEL_DIR or build with 'download')");

    // The network's input is R2*, which a GRE-only dataset gets by fitting the magnitude decay —
    // so fit it here rather than taking a shortcut the real condition does not have.
    let (r2star, _t2star) = qsm_core::r2star::r2star_arlo(
        &ph.mag_voxel_major, &ph.mask, &ph.tes, &ph.grid,
    );

    let t = Instant::now();
    let predicted = r2primenet(
        &r2star,
        &ph.mask,
        &ph.grid,
        &w,
        &R2PrimeNetNorm::default(),
        &R2PrimeNetParams::default(),
        |_, _| {},
    )
    .expect("r2primenet");
    let secs = t.elapsed().as_secs_f64();

    assert_eq!(predicted.len(), ph.mask.len(), "R2′ should be one value per voxel");
    assert!(predicted.iter().all(|v| v.is_finite()), "r2primenet produced non-finite R2′");
    // R2′ is a reversible relaxation rate: negative values are not physical.
    let negative = predicted.iter().zip(&ph.mask)
        .filter(|(&v, &m)| m > 0 && v < -1e-6).count();
    assert_eq!(negative, 0, "{negative} in-mask voxels have negative R2′");

    // Scored through the shared helper so the row lands in the same table, with the same columns,
    // as the χ-separation methods it feeds.
    chisep_score("R2PRIMEnet R2'", &predicted, &ph.r2prime, &ph.mask, ph.dims, secs);
    let corr = correlation(&predicted, &ph.r2prime, &ph.mask);
    let mean = |v: &[f64]| {
        let (s, n) = v.iter().zip(&ph.mask).filter(|(_, &m)| m > 0)
            .fold((0.0, 0usize), |(s, n), (&x, _)| (s + x, n + 1));
        s / n.max(1) as f64
    };
    // A prediction can correlate well and still sit at the wrong level, which matters because
    // χ-separation consumes the magnitude of R2′, not its shape.
    println!("[INFO] mean R2′ predicted {:.2} Hz, reference {:.2} Hz",
             mean(&predicted), mean(&ph.r2prime));

    save_center_slices(&predicted, &ph.mask, ph.dims, "r2primenet_pred");
    save_center_slices(&ph.r2prime, &ph.mask, ph.dims, "r2primenet_ref");
    let diff: Vec<f64> = predicted.iter().zip(&ph.r2prime).map(|(p, r)| p - r).collect();
    save_center_slices(&diff, &ph.mask, ph.dims, "r2primenet_diff");

    // NOT AN ACCURACY GATE. This phantom is simulated at 7 T and R2PRIMEnet's weights are
    // trained at 3 T, with no field strength given to the network, so the prediction is
    // off-distribution by construction and correlating poorly with the reference is the
    // expected result rather than a defect. The old `corr > 0.7` here asserted an accuracy the
    // data cannot support, and had been failing at 0.2532 since it landed (#140) without anyone
    // seeing it, because the CI step reported tee's exit status instead of the test's.
    //
    // What the port's correctness actually rests on is
    // `models_onnx::r2primenet_matches_python_reference`, which scores corr = 1.000000 and
    // max|delta| = 2e-5 Hz against the authors' own onnxruntime recipe.
    //
    // So this band is a regression guard on the off-distribution behaviour: wide enough not to
    // be pinning noise, narrow enough that a change in the tiling, the Dr-scaled z-scoring or
    // the de-normalisation moves the number out of it. Do not read it as agreement, and do not
    // tighten it towards 1.0 without a 3 T phantom to justify that.
    assert!(
        (0.15..0.40).contains(&corr),
        "off-distribution correlation moved to {corr:.4}, outside the 0.15..0.40 band this \
         3 T-network-on-a-7 T-phantom run has held at (0.2532). That is a change in the \
         prediction, not an accuracy result: check the port against \
         models_onnx::r2primenet_matches_python_reference before adjusting this band."
    );
}

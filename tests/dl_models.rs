//! Integration tests for the deep-learning QSM models (`onnx` feature).
//!
//! Each runs a hosted model on the synthetic phantom and reports the same
//! `RESULT:` metrics + centre slices as the classical algorithms, so they flow
//! into the PR summary comment. Weights are fetched from the model registry at
//! run time (needs the `download` feature, or `$QSM_MODEL_DIR`). Ignored by
//! default (network + weights). Metrics are informational — DL models are run on
//! an out-of-distribution synthetic phantom — so tests only assert finiteness.
#![cfg(feature = "onnx")]

mod common;

use std::time::Instant;

use common::{ChallengeMetrics, TestData, TestResult};
use qsm_core::inversion;
use qsm_core::Grid;

fn grid_of(data: &TestData) -> Grid {
    let (nx, ny, nz) = data.dims;
    let (vx, vy, vz) = data.voxel_size;
    Grid::new(nx, ny, nz, vx, vy, vz)
}

/// First weight file of a registry model (downloads + caches with `download`).
fn weights(id: &str) -> Vec<u8> {
    let spec = qsm_core::models::find_model(id).unwrap_or_else(|| panic!("no registry model '{id}'"));
    qsm_core::models::primary_weight_bytes(spec)
        .expect("weights not found (set $QSM_MODEL_DIR or build with the 'download' feature)")
}

/// Report a χ map against the ground-truth susceptibility (Dipole-Inversion table).
fn report_chi(name: &str, slug: &str, chi: &[f64], data: &TestData, elapsed: std::time::Duration) {
    let challenge =
        ChallengeMetrics::compute(name, chi, &data.chi, &data.mask, &data.segmentation, data.dims);
    challenge.print();
    challenge.print_ci_metrics(elapsed);
    common::save_center_slices(chi, &data.mask, data.dims, slug);
    assert!(chi.iter().all(|v| v.is_finite()), "{name} produced non-finite values");
}

#[test]
#[ignore]
fn test_dl_qsmgan() {
    let data = TestData::load().expect("test data");
    let grid = grid_of(&data);
    let w = weights("qsmgan");
    let t = Instant::now();
    let chi = inversion::qsmgan(&data.fieldmap_local, &data.mask, &grid, &w).expect("qsmgan");
    report_chi("QSMGAN", "dl_qsmgan", &chi, &data, t.elapsed());
}

#[test]
#[ignore]
fn test_dl_lpcnn() {
    let data = TestData::load().expect("test data");
    let grid = grid_of(&data);
    let w = weights("lpcnn");
    let t = Instant::now();
    let chi = inversion::lpcnn(&data.fieldmap_local, &data.mask, &grid, data.b0_dir, &w).expect("lpcnn");
    report_chi("LPCNN", "dl_lpcnn", &chi, &data, t.elapsed());
}

#[test]
#[ignore]
fn test_dl_ir2qsm() {
    let data = TestData::load().expect("test data");
    let grid = grid_of(&data);
    let w = weights("ir2qsm");
    let t = Instant::now();
    let chi = inversion::ir2qsm(&data.fieldmap_local, &data.mask, &grid, &w).expect("ir2qsm");
    report_chi("IR2QSM", "dl_ir2qsm", &chi, &data, t.elapsed());
}

#[test]
#[ignore]
fn test_dl_modl_qsm() {
    let data = TestData::load().expect("test data");
    let grid = grid_of(&data);
    let w = weights("modl-qsm");
    let t = Instant::now();
    let chi = inversion::modl_qsm(&data.fieldmap_local, &data.mask, &grid, data.b0_dir, &w).expect("modl_qsm");
    report_chi("MoDL-QSM", "dl_modl_qsm", &chi, &data, t.elapsed());
}

#[test]
#[ignore]
fn test_dl_nextqsm() {
    let data = TestData::load().expect("test data");
    let grid = grid_of(&data);
    let spec = qsm_core::models::find_model("nextqsm").expect("nextqsm in registry");
    let files = qsm_core::models::all_weight_bytes(spec).expect("nextqsm weights");
    // NeXtQSM consumes the TOTAL field (it does its own background removal).
    let t = Instant::now();
    let chi = inversion::nextqsm(&data.fieldmap, &data.mask, &grid, data.b0_dir, &files[0], &files[1])
        .expect("nextqsm");
    report_chi("NeXtQSM", "dl_nextqsm", &chi, &data, t.elapsed());
}

#[test]
#[ignore]
fn test_dl_iqfm() {
    let data = TestData::load().expect("test data");
    let grid = grid_of(&data);
    let w = weights("iqfm");
    // iQFM maps wrapped phase → local field (ppm); compare to the ground-truth local field.
    let phase = &data.phase_echoes[0];
    let te = data.echo_times.first().copied().unwrap_or(0.004);
    let t = Instant::now();
    let lfs = inversion::iqfm(phase, &data.mask, &grid, te, data.field_strength, -1.0, 3, &w).expect("iqfm");
    let res = TestResult::new("iQFM", &lfs, &data.fieldmap_local, &data.mask, data.dims);
    res.print_with_time(t.elapsed());
    res.print_ci_metrics(t.elapsed());
    common::save_center_slices(&lfs, &common::valid_support(&lfs, &data.mask), data.dims, "dl_iqfm");
    assert!(lfs.iter().all(|v| v.is_finite()), "iQFM produced non-finite values");
}

/// R2PRIMEnet on the phantom's multi-echo magnitude: fit R2* with ARLO — the route a GRE-only
/// dataset actually takes — then predict R2′ from it.
///
/// This phantom has no R2′ ground truth, so the check is that the network runs and returns a
/// physically admissible map. `chisep_r2primenet` scores it against a reference R2′.
#[test]
#[ignore]
fn test_dl_r2primenet() {
    let data = TestData::load().expect("test data");
    let grid = grid_of(&data);
    let w = weights("r2primenet");

    let n = data.mag_echoes[0].len();
    let ne = data.mag_echoes.len();
    let mut mag_voxel_major = vec![0.0_f64; n * ne];
    for (e, echo) in data.mag_echoes.iter().enumerate() {
        for (v, &m) in echo.iter().enumerate() {
            mag_voxel_major[v * ne + e] = m;
        }
    }
    let (r2star, _t2star) =
        qsm_core::r2star::r2star_arlo(&mag_voxel_major, &data.mask, &data.echo_times, &grid);

    let t = Instant::now();
    let r2prime = qsm_core::relaxometry::r2primenet(
        &r2star, &data.mask, &grid, &w,
        &qsm_core::relaxometry::R2PrimeNetNorm::default(),
        &qsm_core::relaxometry::R2PrimeNetParams::default(),
        |_, _| {},
    )
    .expect("r2primenet");
    let elapsed = t.elapsed();

    let in_mask = |v: &[f64]| -> (f64, f64) {
        let vals: Vec<f64> = v.iter().zip(&data.mask).filter(|(_, &m)| m > 0).map(|(&x, _)| x).collect();
        let mean = vals.iter().sum::<f64>() / vals.len().max(1) as f64;
        (mean, vals.iter().cloned().fold(f64::MIN, f64::max))
    };
    let (r2s_mean, r2s_max) = in_mask(&r2star);
    let (r2p_mean, r2p_max) = in_mask(&r2prime);
    println!("[INFO] R2* mean {r2s_mean:.2} Hz (max {r2s_max:.2}) → R2′ mean {r2p_mean:.2} Hz (max {r2p_max:.2})");
    println!("R2PRIMEnet     {:>10.2?}", elapsed);
    println!("RESULT:R2PRIMEnet,-,-,-,-,{:.2}", elapsed.as_secs_f64());
    common::save_center_slices(&r2prime, &data.mask, data.dims, "dl_r2primenet");

    assert_eq!(r2prime.len(), n, "R2′ should be one value per voxel");
    assert!(r2prime.iter().all(|v| v.is_finite()), "R2PRIMEnet produced non-finite values");
    // R2′ is a reversible relaxation rate, and it is a *component* of R2*: negative is not
    // physical, and exceeding R2* would mean the irreversible part were negative.
    for (i, (&p, &m)) in r2prime.iter().zip(&data.mask).enumerate() {
        if m > 0 {
            assert!(p >= -1e-6, "negative R2′ {p} at voxel {i}");
        }
    }
    assert!(r2p_mean > 0.0, "R2′ is uniformly zero — the network did not predict anything");
}

/// HD-BET brain extraction on the phantom's root-sum-of-squares magnitude, scored like BET
/// (Dice vs the ground-truth mask) so it lands in the Brain Extraction table next to it.
#[test]
#[ignore]
fn test_dl_hdbet() {
    use qsm_core::bet::{hd_bet, HdBetParams};

    let data = TestData::load().expect("test data");
    let grid = grid_of(&data);
    let w = weights("hd-bet");
    let rss: Vec<f64> = (0..data.mask.len())
        .map(|i| data.mag_echoes.iter().map(|e| e[i] * e[i]).sum::<f64>().sqrt())
        .collect();
    let t = Instant::now();
    let mask = hd_bet(&rss, &grid, &w, &HdBetParams::default(), |_, _| {}).expect("hd_bet");
    let elapsed = t.elapsed();

    let dice = common::dice_coefficient(&mask, &data.mask);
    println!("HD-BET          Dice={:.4}      {:>10.2?}", dice, elapsed);
    println!("RESULT:HD-BET,{:.6},-,-,{:.2}", dice, elapsed.as_secs_f64());

    // Same layout as test_bet: result = predicted mask, mask overlay = ground truth, plus the
    // magnitude HD-BET segmented, so the figure can draw both boundaries on it.
    let predicted: Vec<f64> = mask.iter().map(|&v| v as f64).collect();
    common::save_center_slices(&predicted, &data.mask, data.dims, "dl_hdbet");
    common::save_center_slices(&rss, &data.mask, data.dims, "dl_hdbet_magnitude");

    assert!(dice > 0.9, "HD-BET Dice coefficient too low: {dice}");
}

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
    common::save_center_slices(&lfs, &data.mask, data.dims, "dl_iqfm");
    assert!(lfs.iter().all(|v| v.is_finite()), "iQFM produced non-finite values");
}

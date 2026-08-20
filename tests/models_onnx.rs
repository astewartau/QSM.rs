//! Smoke test for the ONNX inference path (`onnx` feature).
//!
//! Runs a real exported QSM network end-to-end through `models::onnx` to prove
//! the pure-Rust `tract` engine loads and executes it. Ignored by default and
//! points at a local weight file via an env var, since weights are not vendored.
//!
//! ```bash
//! BFRNET_ONNX=~/repos/qsm/qsmci/qsmci/algorithms/bfrnet/BFRnet.onnx \
//!   cargo test --features onnx --test models_onnx -- --ignored --nocapture
//! ```
#![cfg(feature = "onnx")]

use qsm_core::models::onnx::{OnnxModel, Tensor};

#[test]
#[ignore]
fn bfrnet_forward_runs() {
    let path = std::env::var("BFRNET_ONNX")
        .expect("set BFRNET_ONNX to a local BFRnet.onnx to run this test");
    let bytes = std::fs::read(&path).expect("read onnx file");
    println!("loaded {} bytes from {path}", bytes.len());

    let model = OnnxModel::load(&bytes).expect("parse onnx");

    // Fully-convolutional; use a small volume divisible by 8.
    let (d, h, w) = (32usize, 32, 32);
    let field: Vec<f32> = (0..d * h * w).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();

    let out = model
        .run_single(&Tensor::new(vec![1, 1, d, h, w], field))
        .expect("forward pass");

    println!("output shape {:?}", out.shape);
    assert_eq!(out.shape, vec![1, 1, d, h, w], "output should match input shape");
    assert!(out.data.iter().all(|v| v.is_finite()), "output must be finite");
    let mean = out.data.iter().copied().sum::<f32>() / out.data.len() as f32;
    println!("output mean {mean:.6}, first few {:?}", &out.data[..4.min(out.data.len())]);
}

/// Parity: `bgremove::bfrnet` (tract, with the layout repack) must match the
/// authors' ONNX-Runtime reference on a real total-field volume. This validates
/// the column-major↔row-major repack and the crop/mask math, not just that it
/// runs.
///
/// ```bash
/// BFRNET_ONNX=~/repos/qsm/qsmci/qsmci/algorithms/bfrnet/BFRnet.onnx \
///   cargo test --features onnx --test models_onnx bfrnet_matches -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn bfrnet_matches_python_reference() {
    use qsm_core::io::read_nifti_file;
    use std::path::Path;

    let qsmci = "/home/ashley/repos/qsm/qsmci/qsmci";
    let total_p = std::env::var("BFRNET_TOTALFIELD")
        .unwrap_or(format!("{qsmci}/data/sim/dev/groundtruth/totalfield.nii.gz"));
    let mask_p = std::env::var("BFRNET_MASK")
        .unwrap_or(format!("{qsmci}/data/sim/dev/inputs/mask.nii.gz"));
    let ref_p = std::env::var("BFRNET_REF")
        .unwrap_or("/tmp/bfrnet_ref/localfield_ref.nii.gz".to_string());
    let onnx_p = std::env::var("BFRNET_ONNX")
        .unwrap_or(format!("{qsmci}/algorithms/bfrnet/BFRnet.onnx"));

    let total = read_nifti_file(Path::new(&total_p)).expect("total field");
    let mask_nii = read_nifti_file(Path::new(&mask_p)).expect("mask");
    let reference = read_nifti_file(Path::new(&ref_p)).expect("reference localfield");
    let onnx_bytes = std::fs::read(&onnx_p).expect("onnx");

    let grid = qsm_core::Grid {
        dims: total.dims,
        voxel_size: total.voxel_size,
    };
    let mask: Vec<u8> = mask_nii.data.iter().map(|&v| (v > 0.5) as u8).collect();

    let local = qsm_core::bgremove::bfrnet(&total.data, &mask, &grid, &onnx_bytes)
        .expect("bfrnet inference");

    assert_eq!(local.len(), reference.data.len());

    // Correlation + max abs difference over in-mask voxels.
    let (mut sxx, mut syy, mut sxy, mut sx, mut sy, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut max_abs = 0.0f64;
    for i in 0..local.len() {
        if mask[i] == 0 {
            continue;
        }
        let (a, b) = (local[i], reference.data[i]);
        sx += a;
        sy += b;
        sxx += a * a;
        syy += b * b;
        sxy += a * b;
        n += 1.0;
        max_abs = max_abs.max((a - b).abs());
    }
    let cov = sxy - sx * sy / n;
    let corr = cov / ((sxx - sx * sx / n).sqrt() * (syy - sy * sy / n).sqrt());
    println!("BFRnet vs ONNX-Runtime: corr = {corr:.6}, max|Δ| = {max_abs:.6e} ppm, n = {n}");

    assert!(corr > 0.999, "correlation too low: {corr}");
    assert!(max_abs < 5e-3, "max abs diff too high: {max_abs}");
}

/// Parity: `inversion::xqsm` (tract, centered-pad repack) must match the
/// ONNX-Runtime reference on a real local-field volume.
///
/// ```bash
/// cargo test --features onnx --test models_onnx xqsm_matches -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn xqsm_matches_python_reference() {
    use qsm_core::io::read_nifti_file;
    use std::path::Path;

    let qsmci = "/home/ashley/repos/qsm/qsmci/qsmci";
    let field_p = std::env::var("XQSM_LOCALFIELD")
        .unwrap_or(format!("{qsmci}/data/sim/dev/groundtruth/localfield.nii.gz"));
    let mask_p = std::env::var("XQSM_MASK")
        .unwrap_or(format!("{qsmci}/data/sim/dev/inputs/mask.nii.gz"));
    let ref_p = std::env::var("XQSM_REF").unwrap_or("/tmp/xqsm_ref/chi_ref.nii.gz".to_string());
    let onnx_p = std::env::var("XQSM_ONNX").unwrap_or("/tmp/xqsm_export/xqsm.onnx".to_string());

    let field = read_nifti_file(Path::new(&field_p)).expect("local field");
    let mask_nii = read_nifti_file(Path::new(&mask_p)).expect("mask");
    let reference = read_nifti_file(Path::new(&ref_p)).expect("reference chi");
    let onnx_bytes = std::fs::read(&onnx_p).expect("onnx");

    let grid = qsm_core::Grid { dims: field.dims, voxel_size: field.voxel_size };
    let mask: Vec<u8> = mask_nii.data.iter().map(|&v| (v > 0.5) as u8).collect();

    let chi = qsm_core::inversion::xqsm(&field.data, &mask, &grid, &onnx_bytes).expect("xqsm");
    assert_eq!(chi.len(), reference.data.len());

    let (mut sxx, mut syy, mut sxy, mut sx, mut sy, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut max_abs = 0.0f64;
    for i in 0..chi.len() {
        if mask[i] == 0 {
            continue;
        }
        let (a, b) = (chi[i], reference.data[i]);
        sx += a;
        sy += b;
        sxx += a * a;
        syy += b * b;
        sxy += a * b;
        n += 1.0;
        max_abs = max_abs.max((a - b).abs());
    }
    let corr = (sxy - sx * sy / n) / ((sxx - sx * sx / n).sqrt() * (syy - sy * sy / n).sqrt());
    println!("xQSM vs ONNX-Runtime: corr = {corr:.6}, max|Δ| = {max_abs:.6e} ppm, n = {n}");
    assert!(corr > 0.999, "correlation too low: {corr}");
    assert!(max_abs < 5e-3, "max abs diff too high: {max_abs}");
}

/// Parity: `inversion::qsmnet` (tract, NDHWC + norm) vs the ONNX-Runtime
/// reference. Also the first proof that `tract` can run a `tf2onnx`-converted
/// (legacy TensorFlow) graph.
///
/// ```bash
/// cargo test --release --features onnx --test models_onnx qsmnet_matches -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn qsmnet_matches_python_reference() {
    use qsm_core::io::read_nifti_file;
    use std::path::Path;

    let qsmci = "/home/ashley/repos/qsm/qsmci/qsmci";
    let field_p = std::env::var("QSMNET_LOCALFIELD")
        .unwrap_or(format!("{qsmci}/data/sim/dev/groundtruth/localfield.nii.gz"));
    let mask_p = std::env::var("QSMNET_MASK")
        .unwrap_or(format!("{qsmci}/data/sim/dev/inputs/mask.nii.gz"));
    let ref_p = std::env::var("QSMNET_REF").unwrap_or("/tmp/qsmnet_ref/chi_ref.nii.gz".to_string());
    let onnx_p =
        std::env::var("QSMNET_ONNX").unwrap_or("/tmp/qsmnet_export/qsmnet.onnx".to_string());

    let field = read_nifti_file(Path::new(&field_p)).expect("local field");
    let mask_nii = read_nifti_file(Path::new(&mask_p)).expect("mask");
    let reference = read_nifti_file(Path::new(&ref_p)).expect("reference chi");
    let onnx_bytes = std::fs::read(&onnx_p).expect("onnx");

    let grid = qsm_core::Grid { dims: field.dims, voxel_size: field.voxel_size };
    let mask: Vec<u8> = mask_nii.data.iter().map(|&v| (v > 0.5) as u8).collect();

    let chi = qsm_core::inversion::qsmnet(
        &field.data, &mask, &grid, &onnx_bytes, &qsm_core::inversion::QsmnetNorm::default(),
    )
    .expect("qsmnet");
    assert_eq!(chi.len(), reference.data.len());

    let (mut sxx, mut syy, mut sxy, mut sx, mut sy, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut max_abs = 0.0f64;
    for i in 0..chi.len() {
        if mask[i] == 0 {
            continue;
        }
        let (a, b) = (chi[i], reference.data[i]);
        sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b; n += 1.0;
        max_abs = max_abs.max((a - b).abs());
    }
    let corr = (sxy - sx * sy / n) / ((sxx - sx * sx / n).sqrt() * (syy - sy * sy / n).sqrt());
    println!("QSMnet vs ONNX-Runtime: corr = {corr:.6}, max|Δ| = {max_abs:.6e} ppm, n = {n}");
    assert!(corr > 0.999, "correlation too low: {corr}");
    assert!(max_abs < 5e-3, "max abs diff too high: {max_abs}");
}

/// Parity: `inversion::qsmnet` with the QSMnet+ weights + norm vs the
/// ONNX-Runtime reference (same clean-rebuild recipe as QSMnet).
///
/// ```bash
/// cargo test --release --features onnx --test models_onnx qsmnet_plus -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn qsmnet_plus_matches_python_reference() {
    use qsm_core::io::read_nifti_file;
    use std::path::Path;

    let qsmci = "/home/ashley/repos/qsm/qsmci/qsmci";
    let field = read_nifti_file(Path::new(&std::env::var("QSMNETP_LOCALFIELD").unwrap_or(
        format!("{qsmci}/data/sim/dev/groundtruth/localfield.nii.gz"),
    )))
    .expect("field");
    let mask_nii = read_nifti_file(Path::new(
        &std::env::var("QSMNETP_MASK").unwrap_or(format!("{qsmci}/data/sim/dev/inputs/mask.nii.gz")),
    ))
    .expect("mask");
    let reference = read_nifti_file(Path::new(
        &std::env::var("QSMNETP_REF").unwrap_or("/tmp/qsmnetplus_ref/chi_ref.nii.gz".to_string()),
    ))
    .expect("reference");
    let onnx_bytes = std::fs::read(
        std::env::var("QSMNETP_ONNX")
            .unwrap_or("/tmp/qsmnetplus_export/qsmnet_plus_clean.onnx".to_string()),
    )
    .expect("onnx");

    let grid = qsm_core::Grid { dims: field.dims, voxel_size: field.voxel_size };
    let mask: Vec<u8> = mask_nii.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let chi = qsm_core::inversion::qsmnet(
        &field.data, &mask, &grid, &onnx_bytes, &qsm_core::inversion::QsmnetNorm::qsmnet_plus(),
    )
    .expect("qsmnet+");

    let (mut sxx, mut syy, mut sxy, mut sx, mut sy, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut max_abs = 0.0f64;
    for i in 0..chi.len() {
        if mask[i] == 0 {
            continue;
        }
        let (a, b) = (chi[i], reference.data[i]);
        sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b; n += 1.0;
        max_abs = max_abs.max((a - b).abs());
    }
    let corr = (sxy - sx * sy / n) / ((sxx - sx * sx / n).sqrt() * (syy - sy * sy / n).sqrt());
    println!("QSMnet+ vs ONNX-Runtime: corr = {corr:.6}, max|Δ| = {max_abs:.6e} ppm, n = {n}");
    assert!(corr > 0.999, "correlation too low: {corr}");
    assert!(max_abs < 5e-3, "max abs diff too high: {max_abs}");
}

/// Parity: `separation::susep_net` (tract, 3-in/2-out, z-score + de-norm) vs the
/// ONNX-Runtime reference on cropped chi-sep inputs.
///
/// ```bash
/// cargo test --release --features onnx --test models_onnx susep_net_matches -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn susep_net_matches_python_reference() {
    use qsm_core::io::read_nifti_file;
    use std::path::Path;

    let dir = std::env::var("SUSEP_DIR").unwrap_or("/tmp/susep_ref".to_string());
    let rd = |n: &str| read_nifti_file(Path::new(&format!("{dir}/{n}.nii.gz"))).expect(n);
    let field = rd("localfield");
    let qsm = rd("chimap");
    let r2p = rd("r2prime");
    let mask_nii = rd("mask");
    let ref_pos = rd("chi_pos_ref");
    let ref_neg = rd("chi_neg_ref");
    let onnx_bytes =
        std::fs::read(std::env::var("SUSEP_ONNX").unwrap_or("/tmp/susep_export/susep-net.onnx".into()))
            .expect("onnx");

    let grid = qsm_core::Grid { dims: field.dims, voxel_size: field.voxel_size };
    let mask: Vec<u8> = mask_nii.data.iter().map(|&v| (v > 0.5) as u8).collect();

    let (chi_pos, chi_neg, _tot) = qsm_core::separation::susep_net(
        &field.data, &qsm.data, &r2p.data, &mask, &grid, &onnx_bytes,
        &qsm_core::separation::SusepNetNorm::default(),
    )
    .expect("susep-net");

    // Reference χ− is a positive magnitude; our χ− is signed (≤0) — compare magnitude.
    for (name, got, want, sign) in [
        ("chi+", &chi_pos, &ref_pos.data, 1.0f64),
        ("chi-", &chi_neg, &ref_neg.data, -1.0f64),
    ] {
        let (mut sxx, mut syy, mut sxy, mut sx, mut sy, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let mut max_abs = 0.0f64;
        for i in 0..got.len() {
            if mask[i] == 0 {
                continue;
            }
            let (a, b) = (got[i], sign * want[i]);
            sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b; n += 1.0;
            max_abs = max_abs.max((a - b).abs());
        }
        let corr = (sxy - sx * sy / n) / ((sxx - sx * sx / n).sqrt() * (syy - sy * sy / n).sqrt());
        println!("SUSEP-Net {name}: corr = {corr:.6}, max|Δ| = {max_abs:.6e} ppm");
        assert!(corr > 0.999, "{name} correlation too low: {corr}");
        assert!(max_abs < 5e-3, "{name} max abs diff too high: {max_abs}");
    }
}

/// Parity: `inversion::autoqsm` (tract, fixed 64³→32³ + Rust sliding-window
/// tiling/blend) vs the authors' Keras `data_predict` patch-stitched reference.
/// Validates both the clean V-Net re-export and the tiling replica.
///
/// ```bash
/// cargo test --release --features onnx --test models_onnx autoqsm_matches -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn autoqsm_matches_keras_reference() {
    use qsm_core::io::read_nifti_file;
    use std::path::Path;

    let dir = std::env::var("AUTOQSM_DIR").unwrap_or("/tmp/autoqsm_export".to_string());
    let field = read_nifti_file(Path::new(&format!("{dir}/totalfield.nii.gz"))).expect("field");
    let mask_nii = read_nifti_file(Path::new(&format!("{dir}/mask.nii.gz"))).expect("mask");
    let reference = read_nifti_file(Path::new(&format!("{dir}/chi_ref.nii.gz"))).expect("ref");
    let onnx_bytes = std::fs::read(format!("{dir}/autoqsm.onnx")).expect("onnx");

    let grid = qsm_core::Grid { dims: field.dims, voxel_size: field.voxel_size };
    let mask: Vec<u8> = mask_nii.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let chi = qsm_core::inversion::autoqsm(&field.data, &mask, &grid, &onnx_bytes).expect("autoqsm");

    // Reference is whole-head (unmasked); our output is masked — compare in-mask.
    let (mut sxx, mut syy, mut sxy, mut sx, mut sy, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut max_abs = 0.0f64;
    for i in 0..chi.len() {
        if mask[i] == 0 {
            continue;
        }
        let (a, b) = (chi[i], reference.data[i]);
        sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b; n += 1.0;
        max_abs = max_abs.max((a - b).abs());
    }
    let corr = (sxy - sx * sy / n) / ((sxx - sx * sx / n).sqrt() * (syy - sy * sy / n).sqrt());
    println!("AutoQSM vs Keras: corr = {corr:.6}, max|Δ| = {max_abs:.6e} ppm, n = {n}");
    assert!(corr > 0.999, "correlation too low: {corr}");
    assert!(max_abs < 5e-3, "max abs diff too high: {max_abs}");
}

/// Parity: `inversion::iqsm` (tract, 4-input phase→χ incl. scalar te/b0, sphere
/// erosion, LG crop-pad rewrite) vs the authors' original `inference.run_iqsm`
/// (torch) on echo 0 of the dev phase.
///
/// ```bash
/// cargo test --release --features onnx --test models_onnx iqsm_matches -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn iqsm_matches_python_reference() {
    use qsm_core::io::read_nifti_file;
    use std::path::Path;

    let phase = read_nifti_file(Path::new(
        &std::env::var("IQSM_PHASE").unwrap_or("/tmp/iqsm_ref/phase_e0.nii.gz".into()),
    ))
    .expect("phase");
    let mask_nii = read_nifti_file(Path::new(&std::env::var("IQSM_MASK").unwrap_or(
        "/home/ashley/repos/qsm/qsmci/qsmci/data/sim/dev/inputs/mask.nii.gz".into(),
    )))
    .expect("mask");
    let reference = read_nifti_file(Path::new(
        &std::env::var("IQSM_REF").unwrap_or("/tmp/iqsm_ref/iQSM.nii.gz".into()),
    ))
    .expect("ref");
    let onnx_bytes =
        std::fs::read(std::env::var("IQSM_ONNX").unwrap_or("/tmp/iqsm_export/iqsm.onnx".into()))
            .expect("onnx");

    let grid = qsm_core::Grid { dims: phase.dims, voxel_size: phase.voxel_size };
    let mask: Vec<u8> = mask_nii.data.iter().map(|&v| (v > 0.5) as u8).collect();

    // Match the reference: te=0.004 s (echo 0), b0=3 T, phase_sign=-1, erode radius 3.
    let chi = qsm_core::inversion::iqsm(
        &phase.data, &mask, &grid, 0.004, 3.0, -1.0, 3, &onnx_bytes,
    )
    .expect("iqsm");

    // Compare where the reference is non-zero (the eroded mask region).
    let (mut sxx, mut syy, mut sxy, mut sx, mut sy, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut max_abs = 0.0f64;
    for i in 0..chi.len() {
        if reference.data[i] == 0.0 {
            continue;
        }
        let (a, b) = (chi[i], reference.data[i]);
        sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b; n += 1.0;
        max_abs = max_abs.max((a - b).abs());
    }
    let corr = (sxy - sx * sy / n) / ((sxx - sx * sx / n).sqrt() * (syy - sy * sy / n).sqrt());
    println!("iQSM vs Python: corr = {corr:.6}, max|Δ| = {max_abs:.6e} ppm, n = {n}");
    assert!(corr > 0.999, "correlation too low: {corr}");
    assert!(max_abs < 5e-3, "max abs diff too high: {max_abs}");
}

/// Parity: `inversion::iqfm` (tract; the iQSM LoT-Unet's tissue-field head, phase →
/// local field) vs the authors' original `inference.run_iqsm(run_iqfm=True)` on
/// echo 0 of the dev phase. Same code path as iQSM, different weights + output.
///
/// ```bash
/// cargo test --release --features onnx --test models_onnx iqfm_matches -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn iqfm_matches_python_reference() {
    use qsm_core::io::read_nifti_file;
    use std::path::Path;

    let phase = read_nifti_file(Path::new(
        &std::env::var("IQFM_PHASE").unwrap_or("/tmp/iqfm_ref/phase_e0.nii.gz".into()),
    ))
    .expect("phase");
    let mask_nii = read_nifti_file(Path::new(&std::env::var("IQFM_MASK").unwrap_or(
        "/home/ashley/repos/qsm/qsmci/qsmci/data/sim/dev/inputs/mask.nii.gz".into(),
    )))
    .expect("mask");
    let reference = read_nifti_file(Path::new(
        &std::env::var("IQFM_REF").unwrap_or("/tmp/iqfm_ref/iQFM.nii.gz".into()),
    ))
    .expect("ref");
    let onnx_bytes =
        std::fs::read(std::env::var("IQFM_ONNX").unwrap_or("/tmp/iqfm_export/iqfm.onnx".into()))
            .expect("onnx");

    let grid = qsm_core::Grid { dims: phase.dims, voxel_size: phase.voxel_size };
    let mask: Vec<u8> = mask_nii.data.iter().map(|&v| (v > 0.5) as u8).collect();

    // Match the reference: te=0.004 s (echo 0), b0=3 T, phase_sign=-1, erode radius 3.
    let lfs = qsm_core::inversion::iqfm(
        &phase.data, &mask, &grid, 0.004, 3.0, -1.0, 3, &onnx_bytes,
    )
    .expect("iqfm");

    let (mut sxx, mut syy, mut sxy, mut sx, mut sy, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut max_abs = 0.0f64;
    for i in 0..lfs.len() {
        if reference.data[i] == 0.0 {
            continue;
        }
        let (a, b) = (lfs[i], reference.data[i]);
        sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b; n += 1.0;
        max_abs = max_abs.max((a - b).abs());
    }
    let corr = (sxy - sx * sy / n) / ((sxx - sx * sx / n).sqrt() * (syy - sy * sy / n).sqrt());
    println!("iQFM vs Python: corr = {corr:.6}, max|Δ| = {max_abs:.6e} ppm, n = {n}");
    assert!(corr > 0.999, "correlation too low: {corr}");
    assert!(max_abs < 5e-3, "max abs diff too high: {max_abs}");
}

/// Parity: `inversion::iqsm_plus` (tract; OA-LFE, z_prjs input, brain-bbox crop)
/// vs the authors' original `inference.run_iqsm_plus` on echo 0 of the dev phase.
///
/// ```bash
/// cargo test --release --features onnx --test models_onnx iqsm_plus_matches -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn iqsm_plus_matches_python_reference() {
    use qsm_core::io::read_nifti_file;
    use std::path::Path;

    let phase = read_nifti_file(Path::new(
        &std::env::var("IQSMP_PHASE").unwrap_or("/tmp/iqsmplus_ref/phase_e0.nii.gz".into()),
    ))
    .expect("phase");
    let mask_nii = read_nifti_file(Path::new(&std::env::var("IQSMP_MASK").unwrap_or(
        "/home/ashley/repos/qsm/qsmci/qsmci/data/sim/dev/inputs/mask.nii.gz".into(),
    )))
    .expect("mask");
    let reference = read_nifti_file(Path::new(
        &std::env::var("IQSMP_REF").unwrap_or("/tmp/iqsmplus_ref/iQSM_plus.nii.gz".into()),
    ))
    .expect("ref");
    let onnx_bytes = std::fs::read(
        std::env::var("IQSMP_ONNX").unwrap_or("/tmp/iqsmplus_export/iqsm-plus.onnx".into()),
    )
    .expect("onnx");

    let grid = qsm_core::Grid { dims: phase.dims, voxel_size: phase.voxel_size };
    let mask: Vec<u8> = mask_nii.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let chi = qsm_core::inversion::iqsm_plus(
        &phase.data, &mask, &grid, 0.004, 3.0, (0.0, 0.0, 1.0), -1.0, 3, &onnx_bytes,
    )
    .expect("iqsm_plus");

    let (mut sxx, mut syy, mut sxy, mut sx, mut sy, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut max_abs = 0.0f64;
    for i in 0..chi.len() {
        if reference.data[i] == 0.0 {
            continue;
        }
        let (a, b) = (chi[i], reference.data[i]);
        sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b; n += 1.0;
        max_abs = max_abs.max((a - b).abs());
    }
    let corr = (sxy - sx * sy / n) / ((sxx - sx * sx / n).sqrt() * (syy - sy * sy / n).sqrt());
    println!("iQSM+ vs Python: corr = {corr:.6}, max|Δ| = {max_abs:.6e} ppm, n = {n}");
    assert!(corr > 0.999, "correlation too low: {corr}");
    assert!(max_abs < 5e-3, "max abs diff too high: {max_abs}");
}

/// Parity: `inversion::nextqsm_padded` (BFR ONNX + FFT data-consistency gradient
/// + hand-coded VarNet-VJP ONNX, 6-step unroll) vs the `nextqsm` package output.
///
/// ```bash
/// cargo test --release --features onnx --test models_onnx nextqsm_matches -- --ignored --nocapture
/// ```
/// Memory probe: load `PROBE_ONNX`, run one op at `PROBE_SHAPE` (`c,d,h,w`),
/// print peak RSS (VmHWM). Used to check whether tract streams a given op or
/// materializes a giant im2col buffer at full resolution.
#[test]
#[ignore]
fn onnx_op_memory_probe() {
    fn vmhwm_gb() -> f64 {
        let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kb: f64 = rest.split_whitespace().next().unwrap_or("0").parse().unwrap_or(0.0);
                return kb / 1024.0 / 1024.0;
            }
        }
        0.0
    }
    let path = std::env::var("PROBE_ONNX").expect("PROBE_ONNX");
    let shape: Vec<usize> = std::env::var("PROBE_SHAPE")
        .expect("PROBE_SHAPE e.g. 32,192,256,256")
        .split(',').map(|s| s.parse().unwrap()).collect();
    let (c, d, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let bytes = std::fs::read(&path).expect("onnx");
    let model = OnnxModel::load(&bytes).expect("load");
    let n = c * d * h * w;
    let data: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
    println!("probing {path} at [1,{c},{d},{h},{w}] ...");
    match model.run_single(&Tensor::new(vec![1, c, d, h, w], data)) {
        Ok(out) => println!("PROBE OK: out {:?}  peak_RSS={:.2} GB", out.shape, vmhwm_gb()),
        Err(e) => println!("PROBE FAILED: {e}  peak_RSS={:.2} GB", vmhwm_gb()),
    }
}

/// Wall-clock benchmark of the full `inversion::nextqsm` (internal /64 padding,
/// BFR + 6-step unroll) on a real-resolution volume. Reads `BENCH_FIELD` /
/// `BENCH_MASK` niftis and the two onnx from `NEXTQSM_DIR`. Prints load vs
/// inference timing. Not a correctness assertion.
///
/// ```bash
/// BENCH_FIELD=/tmp/bench_field.nii.gz BENCH_MASK=/tmp/bench_mask.nii.gz \
///   cargo test --release --features onnx --test models_onnx nextqsm_bench -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn nextqsm_bench_realsize() {
    use qsm_core::io::read_nifti_file;
    use std::path::Path;
    use std::time::Instant;

    let dir = std::env::var("NEXTQSM_DIR").unwrap_or("/tmp/nextqsm_export".to_string());
    let field = read_nifti_file(Path::new(&std::env::var("BENCH_FIELD").unwrap())).expect("field");
    let mask_nii = read_nifti_file(Path::new(&std::env::var("BENCH_MASK").unwrap())).expect("mask");
    let bf = std::fs::read(format!("{dir}/nextqsm-bf.onnx")).expect("bf onnx");
    let vjp = std::fs::read(format!("{dir}/nextqsm-vjp.onnx")).expect("vjp onnx");
    let grid = qsm_core::Grid { dims: field.dims, voxel_size: (1.0, 1.0, 1.0) };
    let mask: Vec<u8> = mask_nii.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let (dx, dy, dz) = grid.dims;
    let pad = |s: usize| s.div_ceil(64) * 64;
    println!(
        "input {:?} ({} vox) -> padded ({},{},{}) ({} vox)",
        grid.dims, field.data.len(), pad(dx), pad(dy), pad(dz), pad(dx) * pad(dy) * pad(dz)
    );

    let t0 = Instant::now();
    let chi = qsm_core::inversion::nextqsm(&field.data, &mask, &grid, (0.0, 0.0, 1.0), &bf, &vjp)
        .expect("nextqsm");
    let dt = t0.elapsed().as_secs_f64();
    let finite = chi.iter().filter(|v| v.is_finite()).count();
    println!("NeXtQSM (tract) end-to-end on real volume: {dt:.1} s  (finite out: {finite}/{})", chi.len());
}

#[test]
#[ignore]
fn nextqsm_matches_python_reference() {
    use qsm_core::io::read_nifti_file;
    use std::path::Path;

    let dir = std::env::var("NEXTQSM_DIR").unwrap_or("/tmp/nextqsm_export".to_string());
    let rd = |n: &str| read_nifti_file(Path::new(&format!("{dir}/{n}.nii.gz"))).expect(n);
    let source = rd("source_pad");
    let mask_nii = rd("mask_pad");
    let reference = rd("chi_ref");
    let bf = std::fs::read(format!("{dir}/nextqsm-bf.onnx")).expect("bf onnx");
    let vjp = std::fs::read(format!("{dir}/nextqsm-vjp.onnx")).expect("vjp onnx");

    let grid = qsm_core::Grid { dims: source.dims, voxel_size: (1.0, 1.0, 1.0) };
    let mask: Vec<u8> = mask_nii.data.iter().map(|&v| (v > 0.5) as u8).collect();

    let chi = qsm_core::inversion::nextqsm_padded(
        &source.data, &mask, &grid, (0.0, 0.0, 1.0), &bf, &vjp,
        &qsm_core::inversion::NEXTQSM_LAMBDAS,
    )
    .expect("nextqsm");

    let (mut sxx, mut syy, mut sxy, mut sx, mut sy, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut max_abs = 0.0f64;
    for i in 0..chi.len() {
        if mask[i] == 0 {
            continue;
        }
        let (a, b) = (chi[i], reference.data[i]);
        sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b; n += 1.0;
        max_abs = max_abs.max((a - b).abs());
    }
    let corr = (sxy - sx * sy / n) / ((sxx - sx * sx / n).sqrt() * (syy - sy * sy / n).sqrt());
    println!("NeXtQSM vs Python: corr = {corr:.6}, max|Δ| = {max_abs:.6e} ppm, n = {n}");
    // Reference is the genuine `nextqsm` (predict_all.py, TensorFlow) padded output.
    // The 6-step variational unroll is mildly chaotic, so float32 TF-vs-tract
    // differences in the two U-Nets (each matched to ~1e-5 relative in isolation)
    // amplify to ~0.05 ppm at the worst voxels while the map stays essentially
    // identical (corr > 0.9999). A same-engine ONNX-Runtime unroll shows the same
    // ~0.05 spread vs TF, so this is inherent cross-engine drift, not a port error.
    assert!(corr > 0.999, "correlation too low: {corr}");
    assert!(max_abs < 0.1, "max abs diff too high: {max_abs}");
}

/// End-to-end weight fetch: with the `download` feature, download a model from
/// its hosted URL into a fresh cache and verify SHA-256. Exercises the real
/// "download on use" path (QSMxT scenario). Needs network.
///
/// The download → cache → checksum plumbing is **model-agnostic**, so this only
/// fetches one model (the smallest, `xqsm` ~20 MB) — re-downloading every hosted
/// model each run would move ~1 GB for no extra coverage. Each model's hosted
/// URL + hash is verified once at upload time. To check a specific model's live
/// URL, set `QSM_DL_MODEL=<id>`.
///
/// ```bash
/// cargo test --features "onnx download" --test models_onnx osf_download -- --ignored --nocapture
/// ```
#[cfg(feature = "download")]
#[test]
#[ignore]
fn osf_download_and_verify() {
    use qsm_core::models::{download::sha256_hex, find_model, primary_weight_bytes};

    let id = std::env::var("QSM_DL_MODEL").unwrap_or("xqsm".to_string());

    // Fresh cache dir and no bring-your-own override, so this really hits HTTP.
    let tmp = std::env::temp_dir().join(format!("qsm_dl_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::env::set_var("QSM_MODEL_CACHE", &tmp);
    std::env::remove_var("QSM_MODEL_DIR");

    let spec = find_model(&id).unwrap_or_else(|| panic!("unknown model {id}"));
    assert!(spec.is_available(), "{id} should be Available");
    let bytes = primary_weight_bytes(spec).unwrap_or_else(|e| panic!("{id} download: {e}"));
    let f = &spec.files[0];
    assert_eq!(bytes.len() as u64, f.bytes, "{id} size mismatch");
    assert_eq!(sha256_hex(&bytes), f.sha256, "{id} sha mismatch");
    assert!(tmp.join(f.name).is_file(), "{id} should be cached");
    println!("{id}: downloaded {} bytes, sha ok, cached", bytes.len());

    let _ = std::fs::remove_dir_all(&tmp);
}



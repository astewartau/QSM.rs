//! Integration tests for the deep-learning QSM models (`onnx` feature).
//!
//! Each runs a hosted model on the synthetic phantom and reports the same
//! `RESULT:` metrics + centre slices as the classical algorithms, so they flow
//! into the PR summary comment. Weights are fetched from the model registry at
//! run time (needs the `download` feature, or `$QSM_MODEL_DIR`). Ignored by
//! default (network + weights). Metrics are informational — DL models are run on
//! an out-of-distribution synthetic phantom — so tests assert only finiteness and
//! that each output correlates positively with its ground truth above
//! [`MIN_CORR`]: a sign or units convention mix-up shows up as a negative or
//! near-zero correlation, not as a merely poor one.
#![cfg(feature = "onnx")]

mod common;

use std::time::Instant;

use common::{ChallengeMetrics, TestData, TestResult};

/// Correlation floor for every network's output against the phantom's ground truth.
///
/// Out of distribution as the phantom is, the lowest a correctly wired network has scored is
/// ~0.5 (iQFM 0.53, AutoQSM 0.52, the rest 0.6-0.93). A sign convention mix-up scores below
/// zero instead: AutoQSM, fed this crate's field without the negation its training convention
/// needs, scored -0.50 with deep grey matter dark.
const MIN_CORR: f64 = 0.3;

/// Assert the sign/units floor (see [`MIN_CORR`]).
fn assert_corr(name: &str, recon: &[f64], truth: &[f64], mask: &[u8]) {
    let corr = common::correlation(recon, truth, mask);
    assert!(
        corr > MIN_CORR,
        "{name} correlates at {corr:.3} with ground truth (floor {MIN_CORR}): sign or units convention?"
    );
}
use qsm_core::inversion;
use qsm_core::Grid;

fn grid_of(data: &TestData) -> Grid {
    let (nx, ny, nz) = data.dims;
    let (vx, vy, vz) = data.voxel_size;
    Grid::new(nx, ny, nz, vx, vy, vz)
}

/// First weight file of a registry model (downloads + caches with `download`).
fn weights(id: &str) -> qsm_core::models::WeightBytes {
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
    assert_corr(name, chi, &data.chi, &data.mask);
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
    assert_corr("iQFM", &lfs, &data.fieldmap_local, &data.mask);
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
    let r2star: Vec<f64> = qsm_core::r2star::arlo(&data.mag_echoes.concat(), &data.echo_times, n)
        .unwrap()
        .iter()
        .zip(&data.mask)
        .map(|(&r, &m)| if m != 0 { r } else { 0.0 })
        .collect();

    let t = Instant::now();
    // This phantom is 7 T and R2PRIMEnet's weights are 3 T, so the guard would refuse it. What
    // is under test here is the inference path — that the network runs and returns a physically
    // admissible map — not whether the prediction is calibrated for this subject, which at 7 T it
    // is not (see QSM.rs#129). Overridden deliberately, and the numbers printed below should be
    // read as uncalibrated.
    let r2prime = qsm_core::relaxometry::r2primenet(
        &r2star, &data.mask, &grid, data.field_strength, &w,
        &qsm_core::relaxometry::R2PrimeNetNorm::default(),
        &qsm_core::relaxometry::R2PrimeNetParams {
            ignore_field_mismatch: true,
            ..Default::default()
        },
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

// ---------------------------------------------------------------------------------------------
// The rest of the registry. Every model QSM.rs hosts gets a run here, on the same phantom and
// through the same public function the pipeline dispatches to, so a regression in its pre- or
// post-processing shows up in the PR comment rather than only in the reference-parity tests
// (`tests/models_onnx.rs`), which need locally exported Python references and do not run in CI.
// ---------------------------------------------------------------------------------------------

/// iQSM / iQSM+ / iQFM phase-sign and mask-erosion settings, as `pipeline::inversion` uses them.
const IQSM_PHASE_SIGN: f64 = -1.0;
const IQSM_ERODED_RAD: i32 = 3;

/// The phantom's NIfTI affine (row-major 4x4), which MindGrab and SynthSeg orient by.
fn affine_of() -> [f64; 16] {
    common::load_nifti_file("bids/sub-1/anat/sub-1_echo-1_part-mag_MEGRE.nii")
        .expect("phantom magnitude")
        .affine
}

/// Root-sum-of-squares magnitude over echoes: the image the brain-extraction tests segment.
fn rss_magnitude(data: &TestData) -> Vec<f64> {
    (0..data.mask.len())
        .map(|i| data.mag_echoes.iter().map(|e| e[i] * e[i]).sum::<f64>().sqrt())
        .collect()
}

/// Report a local field against the ground-truth local field (Background Field Removal table).
fn report_local_field(name: &str, slug: &str, lfs: &[f64], data: &TestData, elapsed: std::time::Duration) {
    let res = TestResult::new(name, lfs, &data.fieldmap_local, &data.mask, data.dims);
    res.print_with_time(elapsed);
    res.print_ci_metrics(elapsed);
    common::save_center_slices(lfs, &common::valid_support(lfs, &data.mask), data.dims, slug);
    assert!(lfs.iter().all(|v| v.is_finite()), "{name} produced non-finite values");
    assert_corr(name, lfs, &data.fieldmap_local, &data.mask);
}

/// Report a brain mask against the ground-truth mask (Brain Extraction table), in the
/// `RESULT:name,dice,-,-,time` layout `test_bet` and `test_dl_hdbet` use.
fn report_mask(name: &str, slug: &str, mask: &[u8], data: &TestData, elapsed: std::time::Duration) -> f64 {
    assert_eq!(mask.len(), data.mask.len(), "{name}: one mask value per voxel");
    let dice = common::dice_coefficient(mask, &data.mask);
    println!("{name:<15} Dice={dice:.4}      {elapsed:>10.2?}");
    println!("RESULT:{name},{dice:.6},-,-,{:.2}", elapsed.as_secs_f64());
    let predicted: Vec<f64> = mask.iter().map(|&v| v as f64).collect();
    common::save_center_slices(&predicted, &data.mask, data.dims, slug);
    dice
}

#[test]
#[ignore]
fn test_dl_xqsm() {
    let data = TestData::load().expect("test data");
    let grid = grid_of(&data);
    let w = weights("xqsm");
    let t = Instant::now();
    let chi = inversion::xqsm(&data.fieldmap_local, &data.mask, &grid, &w).expect("xqsm");
    report_chi("xQSM", "dl_xqsm", &chi, &data, t.elapsed());
}

#[test]
#[ignore]
fn test_dl_qsmnet() {
    let data = TestData::load().expect("test data");
    let grid = grid_of(&data);
    let w = weights("qsmnet");
    let t = Instant::now();
    let chi = inversion::qsmnet(&data.fieldmap_local, &data.mask, &grid, &w, &inversion::QsmnetNorm::qsmnet())
        .expect("qsmnet");
    report_chi("QSMnet", "dl_qsmnet", &chi, &data, t.elapsed());
}

#[test]
#[ignore]
fn test_dl_qsmnet_plus() {
    let data = TestData::load().expect("test data");
    let grid = grid_of(&data);
    let w = weights("qsmnet-plus");
    let t = Instant::now();
    let chi = inversion::qsmnet(&data.fieldmap_local, &data.mask, &grid, &w, &inversion::QsmnetNorm::qsmnet_plus())
        .expect("qsmnet+");
    report_chi("QSMnet+", "dl_qsmnet_plus", &chi, &data, t.elapsed());
}

/// AutoQSM takes the TOTAL field: it does its own background removal.
#[test]
#[ignore]
fn test_dl_autoqsm() {
    let data = TestData::load().expect("test data");
    let grid = grid_of(&data);
    let w = weights("autoqsm");
    let t = Instant::now();
    let chi = inversion::autoqsm(&data.fieldmap, &data.mask, &grid, &w).expect("autoqsm");
    report_chi("AutoQSM", "dl_autoqsm", &chi, &data, t.elapsed());
}

/// iQSM takes wrapped phase: unwrapping, background removal and inversion in one network.
#[test]
#[ignore]
fn test_dl_iqsm() {
    let data = TestData::load().expect("test data");
    let grid = grid_of(&data);
    let w = weights("iqsm");
    let phases: Vec<&[f64]> = data.phase_echoes.iter().map(|p| p.as_slice()).collect();
    let mags: Vec<&[f64]> = data.mag_echoes.iter().map(|m| m.as_slice()).collect();
    let t = Instant::now();
    let chi = inversion::iqsm_multi_echo(
        &phases, &mags, &data.mask, &grid, &data.echo_times, data.field_strength,
        IQSM_PHASE_SIGN, IQSM_ERODED_RAD, &w, None, Default::default(),
    )
    .expect("iqsm");
    report_chi("iQSM", "dl_iqsm", &chi, &data, t.elapsed());
}

/// iQSM+ is iQSM with the B0 direction as a network input.
#[test]
#[ignore]
fn test_dl_iqsm_plus() {
    let data = TestData::load().expect("test data");
    let grid = grid_of(&data);
    let w = weights("iqsm-plus");
    let phases: Vec<&[f64]> = data.phase_echoes.iter().map(|p| p.as_slice()).collect();
    let mags: Vec<&[f64]> = data.mag_echoes.iter().map(|m| m.as_slice()).collect();
    let t = Instant::now();
    let chi = inversion::iqsm_plus_multi_echo(
        &phases, &mags, &data.mask, &grid, &data.echo_times, data.field_strength, data.b0_dir,
        IQSM_PHASE_SIGN, IQSM_ERODED_RAD, &w, None,
    )
    .expect("iqsm+");
    report_chi("iQSM+", "dl_iqsm_plus", &chi, &data, t.elapsed());
}

/// BFRnet: total field → local field, scored like the classical background-removal methods.
#[test]
#[ignore]
fn test_dl_bfrnet() {
    let data = TestData::load().expect("test data");
    let grid = grid_of(&data);
    let w = weights("bfrnet");
    let t = Instant::now();
    let lfs = qsm_core::bgremove::bfrnet(&data.fieldmap, &data.mask, &grid, &w).expect("bfrnet");
    report_local_field("BFRnet", "dl_bfrnet", &lfs, &data, t.elapsed());
}

/// MindGrab brain extraction on the RSS magnitude, oriented by the phantom's affine.
#[test]
#[ignore]
fn test_dl_mindgrab() {
    use qsm_core::bet::{mindgrab, MindGrabParams};

    let data = TestData::load().expect("test data");
    let grid = grid_of(&data);
    let w = weights("mindgrab");
    let rss = rss_magnitude(&data);
    let affine = affine_of();
    let t = Instant::now();
    let mask = mindgrab(&rss, &grid, &affine, &w, &MindGrabParams::default(), |_, _| {}).expect("mindgrab");
    let dice = report_mask("MindGrab", "dl_mindgrab", &mask, &data, t.elapsed());
    assert!(dice > 0.85, "MindGrab Dice coefficient too low: {dice}");
}

/// RS2-Net is a rodent model and this phantom is a human head, so the phantom is handed to it at
/// a tenth of its voxel size: a 16 x 20 x 20 mm head, the scale of a rat brain, which is what the
/// network's resampling to 0.25 x 0.2 x 0.16 mm expects. The anatomy is still human, so the Dice
/// is informational. What is checked is that preprocessing, tiled inference and the resample back
/// run end to end and return a mask on the input grid that is neither empty nor the whole volume.
#[test]
#[ignore]
fn test_dl_rs2net() {
    use qsm_core::bet::{rs2_net, Rs2NetParams};

    let data = TestData::load().expect("test data");
    let (nx, ny, nz) = data.dims;
    let (vx, vy, vz) = data.voxel_size;
    let rodent_grid = Grid::new(nx, ny, nz, vx * 0.1, vy * 0.1, vz * 0.1);
    let w = weights("rs2-net");
    let t = Instant::now();
    let mask = rs2_net(&data.mag_echoes[0], &rodent_grid, &w, &Rs2NetParams::default(), |_, _| {})
        .expect("rs2_net");
    report_mask("RS2-Net (rodent model; scaled phantom)", "dl_rs2net", &mask, &data, t.elapsed());
    let inside = mask.iter().filter(|&&m| m > 0).count();
    assert!(inside > 0, "RS2-Net returned an empty mask");
    assert!(inside < mask.len(), "RS2-Net masked the whole volume");
}

/// SynthSeg on the RSS magnitude. Scored as the brain mask the labels imply (every non-background
/// label) against the ground-truth mask, plus how many of the version's structures it found;
/// the phantom's own `dseg` uses a different label set, so a per-structure score is not possible.
fn run_synthseg(
    data: &TestData,
    params: &qsm_core::segment::SynthSegParams,
    model: &str,
) -> (qsm_core::segment::SynthSegResult, std::time::Duration) {
    let grid = grid_of(data);
    let w = weights(model);
    let rss = rss_magnitude(data);
    let affine = affine_of();
    let t = Instant::now();
    let seg = qsm_core::segment::synthseg(&rss, &grid, &affine, &w, params, |_, _| {}).expect("synthseg");
    (seg, t.elapsed())
}

fn report_synthseg(
    name: &str,
    slug: &str,
    seg: &qsm_core::segment::SynthSegResult,
    params: &qsm_core::segment::SynthSegParams,
    data: &TestData,
    elapsed: std::time::Duration,
) -> f64 {
    assert_eq!(seg.labels.len(), data.mask.len(), "{name}: one label per voxel");
    let ids = params.version.labels().ids;
    assert!(seg.labels.iter().all(|l| ids.contains(l)), "{name}: label outside the version's table");
    let brain: Vec<u8> = seg.labels.iter().map(|&l| (l != 0) as u8).collect();
    let dice = common::dice_coefficient(&brain, &data.mask);
    let found = ids.iter().filter(|&&id| id != 0 && seg.labels.contains(&id)).count();
    println!("{name:<15} brain Dice={dice:.4}  structures {found}/{}  {elapsed:>10.2?}", ids.len() - 1);
    println!("RESULT:{name},{dice:.6},{found},{},{:.2}", ids.len() - 1, elapsed.as_secs_f64());
    let labels: Vec<f64> = seg.labels.iter().map(|&l| l as f64).collect();
    common::save_center_slices(&labels, &data.mask, data.dims, slug);
    dice
}

/// Mean Dice over the structures either segmentation contains (background excluded).
fn mean_label_dice(a: &[i32], b: &[i32], ids: &[i32]) -> (f64, f64) {
    let mut dices = Vec::new();
    for &id in ids.iter().filter(|&&id| id != 0) {
        let (mut both, mut na, mut nb) = (0usize, 0usize, 0usize);
        for (&x, &y) in a.iter().zip(b) {
            let (ia, ib) = (x == id, y == id);
            both += (ia && ib) as usize;
            na += ia as usize;
            nb += ib as usize;
        }
        if na + nb > 0 {
            dices.push(2.0 * both as f64 / (na + nb) as f64);
        }
    }
    let mean = dices.iter().sum::<f64>() / dices.len().max(1) as f64;
    let min = dices.iter().cloned().fold(1.0, f64::min);
    (mean, min)
}

/// SynthSeg 1.0, whole-volume inference (the CLI's default).
#[test]
#[ignore]
fn test_dl_synthseg() {
    let data = TestData::load().expect("test data");
    let params = qsm_core::segment::SynthSegParams::default();
    let (seg, elapsed) = run_synthseg(&data, &params, "synthseg");
    let dice = report_synthseg("SynthSeg", "dl_synthseg", &seg, &params, &data, elapsed);
    common::save_center_slices(&rss_magnitude(&data), &data.mask, data.dims, "dl_synthseg_magnitude");
    assert!(dice > 0.7, "SynthSeg brain Dice too low: {dice}");
}

/// SynthSeg 1.0 through the memory-bounded sliding window (what the browser runs), against the
/// ground-truth mask and against the whole-volume segmentation it is meant to reproduce.
#[test]
#[ignore]
fn test_dl_synthseg_window() {
    use qsm_core::segment::SynthSegParams;

    let data = TestData::load().expect("test data");
    let windowed = SynthSegParams::low_memory();
    let (seg, elapsed) = run_synthseg(&data, &windowed, "synthseg");
    let dice = report_synthseg("SynthSeg (sliding window)", "dl_synthseg_window", &seg, &windowed, &data, elapsed);

    let (whole, _) = run_synthseg(&data, &SynthSegParams::default(), "synthseg");
    let (mean, min) = mean_label_dice(&seg.labels, &whole.labels, windowed.version.labels().ids);
    println!("[INFO] sliding window vs whole volume: mean structure Dice {mean:.4} (lowest {min:.4})");
    println!("RESULT:SynthSeg window vs whole,{mean:.6},{min:.6}");
    assert!(dice > 0.7, "windowed SynthSeg brain Dice too low: {dice}");
    // On this phantom the two agree at mean 0.82 (one small structure found by only one of them),
    // below the 0.985 measured on an in-vivo scan: the phantom's magnitude is far from SynthSeg's
    // training contrasts, and the windowed posteriors are less certain there. A regression floor.
    assert!(mean > 0.75, "sliding window departs from the whole volume: mean structure Dice {mean:.4}");
}

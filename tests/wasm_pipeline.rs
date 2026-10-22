//! Every `pipeline::*` stage, and every classical background-removal, inversion and
//! separation algorithm, on a tiny phantom.
//!
//! The point is the target, not the numbers: CI runs this file on `wasm32-unknown-unknown` under
//! Node (`wasm-bindgen-test`), where anything that reaches for something the browser does not have
//! (a clock, threads, the file system) panics. It also runs natively as an ordinary test. The
//! checks on the outputs are deliberately loose (right length, finite, not all zero): accuracy is
//! tested elsewhere.
//!
//! Run it under Node with
//! `cargo test --release --target wasm32-unknown-unknown --test wasm_pipeline`, with
//! `wasm-bindgen-test-runner` (from `wasm-bindgen-cli`, the same version as `wasm-bindgen` in
//! `Cargo.lock`) set as the target's runner (`CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER`).
//!
//! The stage cases skip the deep-learning variants (see [`needs_weights`]); each network has a
//! case of its own (`dl_*`, listed in [`CASES`], which [`every_hosted_model_has_a_case`] checks
//! against the registry). Those run on weights handed over in memory, as a browser host runs them:
//! each reads its files from `$QSM_WEIGHTS_DIR` (read at compile time) through Node's `fs`,
//! passes the bytes to `models::provide_weights` and calls the dispatcher a host calls, on the
//! smallest input the network takes. Fetch the weights with
//! `scripts/fetch_model_weights.py <dir> all` (or a list of model ids). Without the variable the
//! cases skip, except in CI.
//!
//! `browser_memory` measures each network's peak WASM memory on a browser-sized scan; it is
//! ignored, for running by hand.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

use num_complex::Complex64;
use qsm_core::bet::BetParams;
use qsm_core::fieldmap::{B0WeightType, FitPpmComplexParams, LinearFitParams, RobustFitParams};
use qsm_core::pipeline::*;
use qsm_core::unwrap::LaplacianSolver;
use qsm_core::utils::morphology::SignalErosionParams;
use std::f64::consts::PI;

const N: usize = 16;
const NV: usize = N * N * N;
const TES: [f64; 4] = [0.004, 0.012, 0.020, 0.028];
const B0_T: f64 = 3.0;
const GAMMA_HZ_PER_T: f64 = 42.576e6;

/// Deep-learning variants need weights (see the module docs), so these cases skip them.
fn needs_weights(model_id: Option<&'static str>) -> bool {
    model_id.is_some()
}

fn meta() -> ScanMetadata {
    ScanMetadata {
        dims: (N, N, N),
        voxel_size: (1.5, 1.5, 1.5),
        echo_times: TES.to_vec(),
        field_strength: B0_T,
        b0_direction: (0.0, 0.0, 1.0),
        slice_geometry: None,
        affine: None,
    }
}

/// A head-like phantom: a ball of tissue with two susceptibility inclusions, a background field
/// from a dipole outside it, wrapped multi-echo phase and decaying magnitude.
struct Phantom {
    mask: Vec<u8>,
    /// χ in ppm
    chi: Vec<f64>,
    /// Local (tissue) field in ppm
    local_ppm: Vec<f64>,
    /// Local plus background field in ppm
    total_ppm: Vec<f64>,
    /// Wrapped phase per echo, in (-π, π]
    phases: Vec<Vec<f64>>,
    magnitudes: Vec<Vec<f64>>,
}

impl Phantom {
    fn new() -> Self {
        let c = (N as f64 - 1.0) / 2.0;
        let r_brain = 0.4 * N as f64;
        let mut mask = vec![0u8; NV];
        let mut chi = vec![0.0; NV];
        let mut background = vec![0.0; NV];
        let mut r2star = vec![0.0; NV];
        for k in 0..N {
            for j in 0..N {
                for i in 0..N {
                    let l = i + j * N + k * N * N;
                    let (x, y, z) = (i as f64 - c, j as f64 - c, k as f64 - c);
                    if x * x + y * y + z * z < r_brain * r_brain {
                        mask[l] = 1;
                        r2star[l] = 20.0;
                    }
                    if (x - 2.0).powi(2) + y * y + z * z < 6.0 {
                        chi[l] = 0.1;
                        r2star[l] = 40.0;
                    }
                    if (x + 2.5).powi(2) + (y - 1.0).powi(2) + (z + 1.0).powi(2) < 4.0 {
                        chi[l] = -0.05;
                    }
                    // field of a dipole below the field of view, plus a linear term
                    let (dx, dy, dz) = (x, y, z + 2.0 * N as f64);
                    let r2 = dx * dx + dy * dy + dz * dz;
                    let cos2 = dz * dz / r2;
                    background[l] = 2e3 * (3.0 * cos2 - 1.0) / r2.powf(1.5) + 0.02 * x;
                }
            }
        }
        let local_ppm = forward(&chi, &mask);
        let total_ppm: Vec<f64> = local_ppm.iter().zip(&background).map(|(a, b)| a + b).collect();
        let rad_per_ppm_s = 2.0 * PI * GAMMA_HZ_PER_T * B0_T * 1e-6;
        let phases = TES
            .iter()
            .map(|&te| {
                total_ppm
                    .iter()
                    .map(|&f| {
                        let p = f * rad_per_ppm_s * te + 0.3;
                        p - 2.0 * PI * ((p + PI) / (2.0 * PI)).floor()
                    })
                    .collect()
            })
            .collect();
        let magnitudes = TES
            .iter()
            .map(|&te| {
                (0..NV)
                    .map(|l| if mask[l] != 0 { 1000.0 * (-r2star[l] * te).exp() } else { 20.0 + (l % 7) as f64 })
                    .collect()
            })
            .collect();
        Self { mask, chi, local_ppm, total_ppm, phases, magnitudes }
    }

    fn phase_refs(&self) -> Vec<&[f64]> {
        self.phases.iter().map(Vec::as_slice).collect()
    }

    fn mag_refs(&self) -> Vec<&[f64]> {
        self.magnitudes.iter().map(Vec::as_slice).collect()
    }

    /// Multi-echo magnitude, voxel-major `(n_voxels, n_echoes)`.
    fn mag_voxel_major(&self) -> Vec<f64> {
        (0..NV).flat_map(|l| self.magnitudes.iter().map(move |m| m[l])).collect()
    }
}

/// Local field (ppm) of `chi` (ppm) inside `mask`: the dipole convolution on a doubled grid.
fn forward(chi: &[f64], mask: &[u8]) -> Vec<f64> {
    let m = 2 * N;
    let grid = qsm_core::Grid::new(m, m, m, 1.5, 1.5, 1.5);
    let d = qsm_core::kernels::dipole::dipole_kernel(&grid, (0.0, 0.0, 1.0));
    let mut big = vec![Complex64::new(0.0, 0.0); m * m * m];
    for k in 0..N {
        for j in 0..N {
            for i in 0..N {
                big[i + j * m + k * m * m].re = chi[i + j * N + k * N * N];
            }
        }
    }
    qsm_core::fft::fft3d(&mut big, m, m, m);
    big.iter_mut().zip(&d).for_each(|(z, &dv)| *z *= dv);
    qsm_core::fft::ifft3d(&mut big, m, m, m);
    let mut out = vec![0.0; NV];
    for k in 0..N {
        for j in 0..N {
            for i in 0..N {
                let l = i + j * N + k * N * N;
                if mask[l] != 0 {
                    out[l] = big[i + j * m + k * m * m].re;
                }
            }
        }
    }
    out
}

fn no_progress() -> impl FnMut(usize, usize) {
    |_, _| {}
}

/// Right length, finite everywhere, and not identically zero.
#[track_caller]
fn check(what: &str, v: &[f64]) {
    assert_eq!(v.len(), NV, "{what}: wrong length");
    assert!(v.iter().all(|x| x.is_finite()), "{what}: non-finite output");
    assert!(v.iter().any(|&x| x != 0.0), "{what}: all zero");
}

#[track_caller]
fn check_mask(what: &str, m: &[u8]) {
    assert_eq!(m.len(), NV, "{what}: wrong length");
    assert!(m.iter().any(|&x| x != 0), "{what}: empty mask");
}

#[test]
fn phase_utils() {
    let p = Phantom::new();
    let mut scaled: Vec<f64> = p.phases[0].iter().map(|v| (v + PI) * 4096.0 / (2.0 * PI)).collect();
    scale_phase_to_pi(&mut scaled);
    check("scale_phase_to_pi", &scaled);
    check("hz_to_ppm", &hz_to_ppm(&p.total_ppm, B0_T));
    check("rads_to_ppm", &rads_to_ppm(&p.total_ppm, B0_T));
    check("rss_combine", &rss_combine(&p.mag_refs()).unwrap());
}

#[test]
fn masking() {
    let p = Phantom::new();
    let m = meta();
    let (phases, mags) = (p.phase_refs(), p.mag_refs());
    let section = |input, generator, refinements| MaskSection { input, generator, refinements };
    let otsu = MaskOp::Threshold { method: MaskThresholdMethod::Otsu, value: None };
    let refinements = vec![
        MaskOp::Dilate { iterations: 1 },
        MaskOp::Close { radius: 1 },
        MaskOp::FillHoles { max_size: 0 },
        MaskOp::GaussianSmooth { sigma_mm: 2.0 },
        MaskOp::SignalErode(SignalErosionParams::default()),
        MaskOp::Erode { iterations: 1 },
    ];
    for op in &refinements {
        assert!(op.dl_model_id().is_none());
    }
    let mut cases = vec![
        MaskingConfig::default().sections,
        vec![section(MaskingInput::MagnitudeFirst, otsu.clone(), refinements)],
        vec![section(
            MaskingInput::Magnitude,
            MaskOp::Threshold { method: MaskThresholdMethod::Fixed, value: Some(100.0) },
            vec![],
        )],
        vec![section(
            MaskingInput::MagnitudeLast,
            MaskOp::Threshold { method: MaskThresholdMethod::Percentile, value: Some(50.0) },
            vec![],
        )],
        vec![section(MaskingInput::Magnitude, MaskOp::Bet(BetParams::default()), vec![])],
    ];
    // two sections, as QSMxT's two-pass masking builds them
    cases.push(vec![
        section(MaskingInput::PhaseQuality, otsu.clone(), vec![]),
        section(MaskingInput::MagnitudeFirst, otsu, vec![]),
    ]);
    cases.push(default_reliable_sections());
    for sections in &cases {
        let mask = run_masking(sections, &phases, &mags, &m).unwrap_or_else(|e| panic!("{sections:?}: {e}"));
        check_mask("run_masking", &mask);
    }
}

#[test]
fn motion_correction() {
    let p = Phantom::new();
    let config = MotionCorrectionConfig { enabled: true, ..Default::default() };
    let r = run_motion_correction(&p.phase_refs(), &p.mag_refs(), &p.mask, &meta(), &config, &mut no_progress())
        .unwrap();
    assert_eq!(r.phases.len(), TES.len());
    check_mask("motion-corrected mask", &r.mask);
}

#[test]
fn field_mapping() {
    let p = Phantom::new();
    let m = meta();
    let (phases, mags) = (p.phase_refs(), p.mag_refs());
    let weightings = [
        B0WeightType::PhaseSNR,
        B0WeightType::PhaseVar,
        B0WeightType::Average,
        B0WeightType::TEs,
        B0WeightType::Mag,
        B0WeightType::AssumedDecay { t2star_s: 0.04 },
    ];
    let mut estimations: Vec<B0EstimationMethod> =
        weightings.iter().map(|&weighting| B0EstimationMethod::WeightedAvg { weighting }).collect();
    estimations.push(B0EstimationMethod::LinearFit(LinearFitParams::default()));
    estimations.push(B0EstimationMethod::RobustFit(RobustFitParams::default()));
    estimations.push(B0EstimationMethod::ComplexFit(FitPpmComplexParams::default()));
    let unwrappers = [
        (UnwrappingAlgorithm::Romeo, LaplacianSolver::Dct),
        (UnwrappingAlgorithm::Laplacian, LaplacianSolver::Dct),
        (UnwrappingAlgorithm::Laplacian, LaplacianSolver::Fft { pad: [4; 3] }),
    ];
    for (unwrapping_algorithm, laplacian_solver) in unwrappers {
        for b0_estimation in &estimations {
            let config = FieldMappingConfig {
                unwrapping_algorithm,
                laplacian_solver,
                b0_estimation: b0_estimation.clone(),
                ..Default::default()
            };
            let r = run_field_mapping(&phases, Some(&mags), &p.mask, &m, &config, &mut no_progress())
                .unwrap_or_else(|e| panic!("{unwrapping_algorithm:?} {b0_estimation:?}: {e}"));
            check("run_field_mapping", &r.b0_field_ppm);
        }
    }
    // bipolar correction, no phase-offset removal, and a single echo without magnitude
    let config = FieldMappingConfig { bipolar_correction: true, phase_offset_removal: false, ..Default::default() };
    let r = run_field_mapping(&phases, Some(&mags), &p.mask, &m, &config, &mut no_progress()).unwrap();
    check("run_field_mapping (bipolar)", &r.b0_field_ppm);
    let m1 = ScanMetadata { echo_times: vec![TES[0]], ..meta() };
    let r = run_field_mapping(&phases[..1], None, &p.mask, &m1, &FieldMappingConfig::default(), &mut no_progress())
        .unwrap();
    check("run_field_mapping (one echo)", &r.b0_field_ppm);
}

#[test]
fn bg_removal() {
    let p = Phantom::new();
    let m = meta();
    // PDF requires the magnitude of every echo (stacked, echo slowest); the others ignore it
    let echoes: Vec<f64> = p.mag_refs().concat();
    for &algorithm in BgRemovalAlgorithm::VARIANTS {
        if needs_weights(algorithm.dl_model_id()) {
            continue;
        }
        for msmv_refine in [false, true] {
            let config = BgRemovalConfig { algorithm, msmv_refine, ..Default::default() };
            let r = run_bg_removal(&p.total_ppm, &p.mask, &m, &config, Some(&echoes), &mut no_progress())
                .unwrap_or_else(|e| panic!("{algorithm:?} (mSMV {msmv_refine}): {e}"));
            check(&format!("{algorithm:?}"), &r.local_field_ppm);
            check_mask(&format!("{algorithm:?} eroded mask"), &r.eroded_mask);
        }
    }
}

#[test]
fn dipole_inversion() {
    let p = Phantom::new();
    let m = meta();
    let mag = rss_combine(&p.mag_refs()).unwrap();
    for &algorithm in InversionAlgorithm::VARIANTS {
        if needs_weights(algorithm.dl_model_id()) {
            continue;
        }
        let config = InversionConfig { algorithm, ..Default::default() };
        let field = match config.input(TES.len()) {
            InversionInput::LocalField => &p.local_ppm,
            InversionInput::TotalField | InversionInput::WrappedPhase => &p.total_ppm,
        };
        let chi = match algorithm {
            InversionAlgorithm::Tgv => run_tgv(
                &p.phase_refs(),
                Some(&p.mag_refs()),
                &p.mask,
                &m,
                &FieldMappingConfig::default(),
                &config.tgv,
                QsmReference::Mean,
                &mut no_progress(),
            ),
            InversionAlgorithm::Qsmart => {
                // QSMART's default SDF widths (sigma 10-14 voxels) are for brains; on this 16^3
                // phantom they round every filter size to 0, which (as in the original) leaves no
                // local field at all. Narrower widths keep this a check of the wasm build.
                let qsmart = qsm_core::utils::QsmartParams {
                    sdf_sigma1_stage1: 2.0, sdf_sigma2_stage1: 2.0, sdf_sigma1_stage2: 2.0, sdf_sigma2_stage2: 1.0,
                    ..qsm_core::utils::QsmartParams::for_field_strength(B0_T)
                };
                let config = InversionConfig { qsmart, ..config };
                run_qsmart(field, &p.mask, None, Some(&mag), &m, &config, QsmReference::Mean, &mut no_progress())
            }
            // MEDI and TFI require the magnitude of every echo (stacked, echo slowest)
            InversionAlgorithm::Medi | InversionAlgorithm::Tfi => {
                let echoes: Vec<f64> = p.mag_refs().concat();
                run_dipole_inversion(field, &p.mask, &m, &config, Some(&echoes), &mut no_progress())
            }
            _ => run_dipole_inversion(field, &p.mask, &m, &config, Some(&mag), &mut no_progress()),
        };
        let chi = chi.unwrap_or_else(|e| panic!("{algorithm:?}: {e}"));
        check(&format!("{algorithm:?}"), &chi);
    }
}

#[test]
fn separation() {
    let p = Phantom::new();
    let m = meta();
    let mag_rss = rss_combine(&p.mag_refs()).unwrap();
    let mag_multi = p.mag_voxel_major();
    let r2prime: Vec<f64> = p.mask.iter().zip(&p.chi).map(|(&k, &x)| if k != 0 { 5.0 + 50.0 * x.abs() } else { 0.0 }).collect();
    let inputs = SeparationInputs {
        local_field_ppm: &p.local_ppm,
        qsm: &p.chi,
        mask: &p.mask,
        r2prime: Some(&r2prime),
        r2star: None,
        magnitude_rss: Some(&mag_rss),
        magnitude_multi: Some(&mag_multi),
        se_magnitude_multi: None,
    };
    for &algorithm in SeparationAlgorithm::VARIANTS {
        if needs_weights(algorithm.dl_model_id()) {
            continue;
        }
        let config = SeparationConfig { algorithm, ..Default::default() };
        let r = run_separation(inputs, &m, &config, &mut no_progress()).unwrap_or_else(|e| panic!("{algorithm:?}: {e}"));
        check(&format!("{algorithm:?} chi_total"), &r.chi_total);
    }
}

#[test]
fn referencing_and_two_pass() {
    let p = Phantom::new();
    for method in [QsmReference::Mean, QsmReference::None] {
        check("apply_reference", &apply_reference(&p.chi, &p.mask, method));
    }
    let main = p.mask.clone();
    let reliable: Vec<u8> = (0..NV).map(|l| (main[l] != 0 && l % 5 != 0) as u8).collect();
    let reliable = restrict_reliable_mask(&reliable, &main).unwrap();
    check_mask("restrict_reliable_mask", &reliable);
    let filled: Vec<f64> = p.chi.iter().map(|x| x + 0.001).collect();
    check("combine_two_pass", &combine_two_pass(&p.chi, &filled, &reliable).unwrap());
}

// ---------------------------------------------------------------------------------------------
// Deep-learning models, on weights provided in memory
// ---------------------------------------------------------------------------------------------

/// Every model in the registry with hosted weights, and the case below that runs it. The
/// coverage check ([`every_hosted_model_has_a_case`]) fails when a model is added to the
/// registry without one, so a new network cannot reach browser hosts untested.
#[cfg(feature = "onnx")]
const CASES: &[(&str, &str)] = &[
    ("hd-bet", "dl_mask_hd_bet"),
    ("rs2-net", "dl_mask_rs2_net"),
    ("mindgrab", "dl_mask_mindgrab"),
    ("synthseg", "dl_seg_synthseg"),
    ("bfrnet", "dl_bfr_bfrnet"),
    ("iqfm", "dl_bfr_iqfm"),
    ("xqsm", "dl_inv_xqsm"),
    ("qsmnet", "dl_inv_qsmnet"),
    ("qsmnet-plus", "dl_inv_qsmnet_plus"),
    ("qsmgan", "dl_inv_qsmgan"),
    ("lpcnn", "dl_inv_lpcnn"),
    ("ir2qsm", "dl_inv_ir2qsm"),
    ("modl-qsm", "dl_inv_modl_qsm"),
    ("autoqsm", "dl_inv_autoqsm"),
    ("nextqsm", "dl_inv_nextqsm"),
    ("iqsm", "dl_inv_iqsm"),
    ("iqsm-plus", "dl_inv_iqsm_plus"),
    ("r2primenet", "dl_sep_r2primenet"),
    ("susep-net", "dl_sep_susep_net"),
    ("chi-sepnet", "dl_sep_chi_sepnet"),
];

#[cfg(feature = "onnx")]
#[test]
fn every_hosted_model_has_a_case() {
    use qsm_core::models::{all_models, WeightStatus};
    let hosted: Vec<&str> =
        all_models().iter().filter(|m| m.status == WeightStatus::Available).map(|m| m.id).collect();
    for id in &hosted {
        assert!(CASES.iter().any(|(m, _)| m == id), "{id} has hosted weights but no wasm case");
    }
    for (id, _) in CASES {
        assert!(hosted.contains(id), "{id} has a case but is not a hosted model");
    }
}

/// The weights directory the cases read from: `$QSM_WEIGHTS_DIR` when this test was compiled,
/// holding each file under its registry name. CI fetches the files and sets it (see the `wasm`
/// job), so there an unset variable is a failure, not a skip.
#[cfg(feature = "onnx")]
fn weights_dir() -> Option<&'static str> {
    let dir = option_env!("QSM_WEIGHTS_DIR");
    if dir.is_none() {
        assert!(option_env!("CI").is_none(), "$QSM_WEIGHTS_DIR must be set in CI");
        println!("skipping: $QSM_WEIGHTS_DIR was not set at compile time");
    }
    dir
}

/// Read a file: through Node's `fs` on wasm32 (into a JS buffer, then copied into the WASM heap
/// as a browser host's fetch is), through `std::fs` natively.
#[cfg(feature = "onnx")]
fn read_file(path: &str) -> Vec<u8> {
    #[cfg(target_arch = "wasm32")]
    {
        #[wasm_bindgen::prelude::wasm_bindgen(module = "fs")]
        extern "C" {
            #[wasm_bindgen(js_name = readFileSync, catch)]
            fn read_file_sync(path: &str) -> Result<js_sys::Uint8Array, wasm_bindgen::JsValue>;
        }
        read_file_sync(path).unwrap_or_else(|e| panic!("read {path}: {e:?}")).to_vec()
    }
    #[cfg(not(target_arch = "wasm32"))]
    std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

/// Weights provided in memory for as long as this lives.
#[cfg(feature = "onnx")]
struct Provided(&'static qsm_core::models::ModelSpec);

#[cfg(feature = "onnx")]
impl Drop for Provided {
    fn drop(&mut self) {
        for f in self.0.files {
            qsm_core::models::forget_weights(f.name);
        }
    }
}

/// Provide every weight file of model `id` from [`weights_dir`], as a browser host does after
/// fetching them; `None` when there is no weights directory.
#[cfg(feature = "onnx")]
fn provide(id: &str) -> Option<Provided> {
    let spec = qsm_core::models::find_model(id).unwrap_or_else(|| panic!("{id} not in the registry"));
    let dir = weights_dir()?;
    for f in spec.files {
        qsm_core::models::provide_weights(f.name, read_file(&format!("{dir}/{}", f.name)))
            .unwrap_or_else(|e| panic!("{id}: {e}"));
    }
    Some(Provided(spec))
}

/// Run model `id` through `run` (the dispatcher a browser host calls) twice: first with no
/// weights provided, where on wasm32 there is nothing else to load them from, so it must fail
/// and say which file to provide; then on the provided weights, returning the output.
///
/// The first half is what makes each case evidence of WASM execution: it shows the run reaches
/// the model registry from the dispatcher and needs these weights, so the second half cannot
/// pass without running the network.
#[cfg(feature = "onnx")]
fn on_provided_weights<T, E: std::fmt::Display>(id: &str, run: impl Fn() -> Result<T, E>) -> Option<T> {
    let spec = qsm_core::models::find_model(id).unwrap_or_else(|| panic!("{id} not in the registry"));
    for f in spec.files {
        qsm_core::models::forget_weights(f.name);
    }
    #[cfg(target_arch = "wasm32")]
    {
        let msg = match run() {
            Ok(_) => panic!("{id} ran without weights"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("provide_weights") && msg.contains(spec.files[0].name),
            "{id} without weights: {msg}"
        );
    }
    let _weights = provide(id)?;
    Some(run().unwrap_or_else(|e| panic!("{id}: {e}")))
}

/// The phantom's scan metadata with only the first echo: the single-echo networks then run
/// once, the smallest run they have.
#[cfg(feature = "onnx")]
fn meta_one_echo() -> ScanMetadata {
    ScanMetadata { echo_times: vec![TES[0]], ..meta() }
}

/// An identity affine for `voxel` (mm), for the networks that place the volume in world space.
#[cfg(feature = "onnx")]
fn affine(voxel: (f64, f64, f64)) -> [f64; 16] {
    let mut a = [0.0; 16];
    (a[0], a[5], a[10], a[15]) = (voxel.0, voxel.1, voxel.2, 1.0);
    a
}

/// A mask section with one deep-learning generator on the first-echo magnitude.
#[cfg(feature = "onnx")]
fn dl_mask(op: MaskOp, phantom: &Phantom, m: &ScanMetadata) -> Result<Vec<u8>, PipelineError> {
    let section = MaskSection { input: MaskingInput::MagnitudeFirst, generator: op, refinements: vec![] };
    run_masking(&[section], &phantom.phase_refs(), &phantom.mag_refs(), m)
}

/// A dipole inversion through `run_dipole_inversion`, on the field its method takes.
#[cfg(feature = "onnx")]
fn dl_inversion(algorithm: InversionAlgorithm, tile: Option<(usize, usize)>) {
    let id = algorithm.dl_model_id().unwrap();
    let p = Phantom::new();
    let m = meta();
    let config = InversionConfig { algorithm, tile, ..Default::default() };
    let field = match config.input(TES.len()) {
        InversionInput::LocalField => &p.local_ppm,
        InversionInput::TotalField => &p.total_ppm,
        InversionInput::WrappedPhase => unreachable!("{algorithm:?} takes wrapped phase"),
    };
    let chi = on_provided_weights(id, || run_dipole_inversion(field, &p.mask, &m, &config, None, &mut no_progress()));
    if let Some(chi) = chi {
        check(id, &chi);
    }
}

/// HD-BET on the phantom: 16³ at 1.5 mm is 24³ at its 1 mm, padded to 32×32×16 multiples, one
/// pass. The cost follows the volume.
#[cfg(feature = "onnx")]
#[test]
fn dl_mask_hd_bet() {
    let p = Phantom::new();
    let m = meta();
    let op = MaskOp::HdBet(qsm_core::bet::HdBetParams::default());
    if let Some(mask) = on_provided_weights("hd-bet", || dl_mask(op.clone(), &p, &m)) {
        assert_eq!(mask.len(), NV);
    }
}

/// RS2-Net on a rodent-sized phantom already at its 0.25×0.2×0.16 mm, so nothing is resampled.
/// The exported graph is traced at a fixed 128×96×128 patch and every volume is padded to at
/// least one, so this costs one full patch, whatever the input: that, not the volume, sets the
/// cost of the smallest run.
#[cfg(feature = "onnx")]
#[test]
fn dl_mask_rs2_net() {
    let p = Phantom::new();
    let m = ScanMetadata { voxel_size: (0.16, 0.25, 0.2), ..meta() };
    let op = MaskOp::Rs2Net(qsm_core::bet::Rs2NetParams::default());
    if let Some(mask) = on_provided_weights("rs2-net", || dl_mask(op.clone(), &p, &m)) {
        assert_eq!(mask.len(), NV);
    }
}

/// MindGrab conforms every input to a 256³ 1 mm volume, so the phantom costs a whole-head pass,
/// and the network's whole-volume instance normalisation keeps it from being tiled. That pass
/// needs about 7 GB, more than a 32-bit heap holds: on wasm32 this aborts in tract's allocator.
/// Ignored until MindGrab has a bounded-memory path; run it with `--include-ignored`.
#[cfg(feature = "onnx")]
#[test]
#[ignore = "MindGrab's 256³ pass does not fit a 32-bit heap"]
fn dl_mask_mindgrab() {
    let p = Phantom::new();
    let m = ScanMetadata { affine: Some(affine((1.5, 1.5, 1.5))), ..meta() };
    let op = MaskOp::MindGrab(qsm_core::bet::MindGrabParams::default());
    if let Some(mask) = on_provided_weights("mindgrab", || dl_mask(op.clone(), &p, &m)) {
        assert_eq!(mask.len(), NV);
    }
}

/// SynthSeg's quickest settings: one pass, no clean-up. What is under test is that the network
/// runs on this target, not how well it segments.
#[cfg(feature = "onnx")]
fn synthseg_quick(inference: qsm_core::segment::SynthSegInference) -> qsm_core::segment::SynthSegParams {
    qsm_core::segment::SynthSegParams {
        inference,
        sigma_smoothing: 0.0,
        ..qsm_core::segment::SynthSegParams::fast()
    }
}

/// SynthSeg as the engine runs it: the weights read back through the registry and handed to
/// `segment::synthseg`.
#[cfg(feature = "onnx")]
fn synthseg_run(
    mag: &[f64],
    grid: &qsm_core::Grid,
    params: &qsm_core::segment::SynthSegParams,
    progress: impl FnMut(usize, usize),
) -> Result<qsm_core::segment::SynthSegResult, String> {
    let weights = qsm_core::models::primary_weight("synthseg")?;
    qsm_core::segment::synthseg(mag, grid, &affine(grid.voxel_size), &weights, params, progress)
        .map_err(|e| e.to_string())
}

/// A 1 mm grid of `dims`, with a bright ellipsoid "head" in it.
#[cfg(feature = "onnx")]
fn synthseg_input(dims: (usize, usize, usize)) -> (Vec<f64>, qsm_core::Grid) {
    let (nx, ny, nz) = dims;
    let mut mag = vec![0.0; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let r = |x: usize, n: usize| (x as f64 + 0.5) / n as f64 - 0.5;
                let d = (r(i, nx) / 0.4).powi(2) + (r(j, ny) / 0.4).powi(2) + (r(k, nz) / 0.4).powi(2);
                mag[i + nx * (j + ny * k)] = if d < 1.0 { 100.0 + 20.0 * d } else { 0.0 };
            }
        }
    }
    (mag, qsm_core::Grid::new(nx, ny, nz, 1.0, 1.0, 1.0))
}

/// SynthSeg on the smallest network input it takes: the 16³ phantom's magnitude, which it pads
/// to 128³ (SynthSeg's minimum, `MIN_PAD`), so this costs one 128³ pass whatever the phantom's
/// size. That fits a 32-bit heap whole.
#[cfg(feature = "onnx")]
#[test]
fn dl_seg_synthseg() {
    use qsm_core::segment::SynthSegInference;
    let p = Phantom::new();
    let grid = qsm_core::Grid::new(N, N, N, 1.0, 1.0, 1.0);
    let params = synthseg_quick(SynthSegInference::FullVolume);
    let out = on_provided_weights("synthseg", || synthseg_run(&p.magnitudes[0], &grid, &params, no_progress()));
    if let Some(out) = out {
        assert_eq!(out.labels.len(), NV);
        assert_eq!(out.volumes.len(), params.version.n_labels());
        assert!(out.volumes.iter().all(|v| v.is_finite()));
    }
}

/// SynthSeg on a volume whose whole-volume pass needs more than a 32-bit heap: 128×128×256 at
/// 1 mm (no resampling, no extra padding) takes about 4.5 GB in one pass, and the wasm build
/// aborts ("memory allocation failed", seen as `RuntimeError: unreachable`). In windows the
/// network sees a patch at a time. A step of a whole patch keeps it to two 128³ windows, as much
/// network as the volume itself.
#[cfg(feature = "onnx")]
#[test]
fn dl_seg_synthseg_beyond_the_heap_in_windows() {
    use qsm_core::segment::SynthSegInference;
    let Some(_weights) = provide("synthseg") else { return };
    let (mag, grid) = synthseg_input((128, 128, 256));
    let params = synthseg_quick(SynthSegInference::SlidingWindow { patch: 128, tile_step: 1.0 });
    let mut passes = 0;
    let out = synthseg_run(&mag, &grid, &params, |done, total| {
        passes = total;
        assert!(done <= total);
    })
    .unwrap_or_else(|e| panic!("synthseg: {e}"));
    assert_eq!(passes, 2, "two windows");
    assert_eq!(out.labels.len(), mag.len());
    assert!(out.labels.iter().any(|&l| l != 0), "found no brain");
}

/// BFRnet through `run_bg_removal`, on the phantom's total field (padded to multiples of 8),
/// whole and in 16³ tiles (`BgRemovalConfig::tile`), the path a browser host takes.
#[cfg(feature = "onnx")]
#[test]
fn dl_bfr_bfrnet() {
    let p = Phantom::new();
    let m = meta();
    for tile in [None, Some((8, 4))] {
        let config = BgRemovalConfig { algorithm: BgRemovalAlgorithm::Bfrnet, tile, ..Default::default() };
        let r = on_provided_weights("bfrnet", || run_bg_removal(&p.total_ppm, &p.mask, &m, &config, None, &mut no_progress()));
        if let Some(r) = r {
            check("bfrnet", &r.local_field_ppm);
        }
    }
}

/// iQFM (wrapped phase → local field) through `run_iqfm`, on one echo, whole and in 16³ tiles.
#[cfg(feature = "onnx")]
#[test]
fn dl_bfr_iqfm() {
    let p = Phantom::new();
    let m = meta_one_echo();
    let (phases, mags) = (p.phase_refs(), p.mag_refs());
    for tile in [None, Some((8, 4))] {
        if let Some(local) = on_provided_weights("iqfm", || run_iqfm(&phases[..1], &mags[..1], &p.mask, &m, tile)) {
            check("iqfm", &local);
        }
    }
}

/// xQSM, whole volume and in tiles: the browser profile runs the field-to-χ networks in tiles
/// (`InversionConfig::tile`), so both paths are a browser host's.
#[cfg(feature = "onnx")]
#[test]
fn dl_inv_xqsm() {
    dl_inversion(InversionAlgorithm::Xqsm, None);
    dl_inversion(InversionAlgorithm::Xqsm, Some((8, 4)));
}

/// QSMnet: 16³ is already a multiple of its 16. What sets the cost here is the 398 MB model.
#[cfg(feature = "onnx")]
#[test]
fn dl_inv_qsmnet() {
    dl_inversion(InversionAlgorithm::Qsmnet, None);
}

/// QSMnet+: the same network as QSMnet with other weights.
#[cfg(feature = "onnx")]
#[test]
fn dl_inv_qsmnet_plus() {
    dl_inversion(InversionAlgorithm::QsmnetPlus, None);
}

/// QSMGAN runs fixed 64³ → 48³ patches, so the phantom costs one 64³ pass.
#[cfg(feature = "onnx")]
#[test]
fn dl_inv_qsmgan() {
    dl_inversion(InversionAlgorithm::Qsmgan, None);
}

#[cfg(feature = "onnx")]
#[test]
fn dl_inv_lpcnn() {
    dl_inversion(InversionAlgorithm::Lpcnn, None);
}

#[cfg(feature = "onnx")]
#[test]
fn dl_inv_ir2qsm() {
    dl_inversion(InversionAlgorithm::Ir2qsm, None);
}

#[cfg(feature = "onnx")]
#[test]
fn dl_inv_modl_qsm() {
    dl_inversion(InversionAlgorithm::ModlQsm, None);
}

/// AutoQSM (total field → χ) runs fixed 64³ → 32³ patches, so the phantom costs one 64³ pass.
#[cfg(feature = "onnx")]
#[test]
fn dl_inv_autoqsm() {
    dl_inversion(InversionAlgorithm::Autoqsm, None);
}

/// NeXtQSM (total field → χ) pads to multiples of 64, so the phantom costs a 64³ BFR pass and
/// six 64³ unrolled steps.
#[cfg(feature = "onnx")]
#[test]
fn dl_inv_nextqsm() {
    dl_inversion(InversionAlgorithm::Nextqsm, None);
}

/// iQSM (wrapped phase → χ) through `run_iqsm`, on one echo, whole and in 16³ tiles.
#[cfg(feature = "onnx")]
#[test]
fn dl_inv_iqsm() {
    let p = Phantom::new();
    let m = meta_one_echo();
    let (phases, mags) = (p.phase_refs(), p.mag_refs());
    for tile in [None, Some((8, 4))] {
        let chi =
            on_provided_weights("iqsm", || run_iqsm(&phases[..1], &mags[..1], &p.mask, &m, QsmReference::None, tile));
        if let Some(chi) = chi {
            check("iqsm", &chi);
        }
    }
}

/// iQSM+ through `run_iqsm_plus`, on one echo, whole and in 16³ tiles.
#[cfg(feature = "onnx")]
#[test]
fn dl_inv_iqsm_plus() {
    let p = Phantom::new();
    let m = meta_one_echo();
    let (phases, mags) = (p.phase_refs(), p.mag_refs());
    for tile in [None, Some((8, 4))] {
        let chi = on_provided_weights("iqsm-plus", || {
            run_iqsm_plus(&phases[..1], &mags[..1], &p.mask, &m, QsmReference::None, tile)
        });
        if let Some(chi) = chi {
            check("iqsm-plus", &chi);
        }
    }
}

/// An R2* map for the phantom (Hz), and the R2′ derived from it, for the separation networks.
#[cfg(feature = "onnx")]
fn r2_maps(p: &Phantom) -> (Vec<f64>, Vec<f64>) {
    let r2star: Vec<f64> =
        p.mask.iter().zip(&p.chi).map(|(&k, &x)| if k != 0 { 20.0 + 100.0 * x.abs() } else { 0.0 }).collect();
    let r2prime = r2star.iter().map(|r| r * 0.25).collect();
    (r2star, r2prime)
}

/// R2PRIMEnet as the engine calls it (`relaxometry::r2primenet`, weights from the registry), on
/// its smallest patch, 16³ (four pooling levels). The volume is padded to at least one patch, so
/// the patch, not the volume, sets the cost: the authors' 192×192×128 makes even a 16³ input a
/// full 192×192×128 pass.
#[cfg(feature = "onnx")]
#[test]
fn dl_sep_r2primenet() {
    use qsm_core::relaxometry::{r2primenet, R2PrimeNetNorm, R2PrimeNetParams};
    let p = Phantom::new();
    let (r2star, _) = r2_maps(&p);
    let params = R2PrimeNetParams { patch: (16, 16, 16), ..Default::default() };
    let r2prime = on_provided_weights("r2primenet", || {
        let weights = qsm_core::models::primary_weight("r2primenet")?;
        r2primenet(&r2star, &p.mask, &meta().grid(), B0_T, &weights, &R2PrimeNetNorm::default(), &params, |_, _| {})
            .map_err(|e| e.to_string())
    });
    if let Some(r2prime) = r2prime {
        check("r2primenet", &r2prime);
    }
}

/// The separation config for `algorithm` with its network's patch set to `patch` (`None`: the
/// authors' setting, SUSEP-Net's whole volume or χ-sepnet's 192×192×128).
#[cfg(feature = "onnx")]
fn separation_config(algorithm: SeparationAlgorithm, patch: Option<(usize, usize, usize)>) -> SeparationConfig {
    let mut config = SeparationConfig { algorithm, ..Default::default() };
    if let Some(p) = patch {
        config.susep_net.patch = Some(p);
        config.chi_sepnet.patch = p;
    }
    config
}

/// A χ-separation network through `run_separation`, on the network patch `patch`.
#[cfg(feature = "onnx")]
fn dl_separation(algorithm: SeparationAlgorithm, patch: Option<(usize, usize, usize)>) {
    let id = algorithm.dl_model_id().unwrap();
    let p = Phantom::new();
    let m = meta();
    let (_, r2prime) = r2_maps(&p);
    let inputs = SeparationInputs {
        local_field_ppm: &p.local_ppm,
        qsm: &p.chi,
        mask: &p.mask,
        r2prime: Some(&r2prime),
        r2star: None,
        magnitude_rss: None,
        magnitude_multi: None,
        se_magnitude_multi: None,
    };
    let config = separation_config(algorithm, patch);
    if let Some(r) = on_provided_weights(id, || run_separation(inputs, &m, &config, &mut no_progress())) {
        check(&format!("{id} chi_total"), &r.chi_total);
    }
}

/// SUSEP-Net: the whole 16³ volume in one pass (it pads to multiples of 8), and in 8³ windows,
/// the sliding window a browser host runs.
#[cfg(feature = "onnx")]
#[test]
fn dl_sep_susep_net() {
    dl_separation(SeparationAlgorithm::SusepNet, None);
    dl_separation(SeparationAlgorithm::SusepNet, Some((8, 8, 8)));
}

/// χ-sepnet pads every volume to at least one patch, so the patch, not the volume, sets the
/// cost: with the authors' 192×192×128 even the 16³ phantom is a full 192×192×128 pass, which
/// does not fit a 32-bit heap. This runs its smallest patch, 16³ (four pooling levels).
#[cfg(feature = "onnx")]
#[test]
fn dl_sep_chi_sepnet() {
    dl_separation(SeparationAlgorithm::ChiSepNet, Some((16, 16, 16)));
}

// ---------------------------------------------------------------------------------------------
// Peak memory on a browser-sized volume (wasm32 only, ignored: run by hand)
// ---------------------------------------------------------------------------------------------

/// Peak WASM memory of each model on a browser-sized scan: 184×256×128 at 1 mm with five echoes,
/// the size of the example scan browser hosts offer. Not run in CI (minutes each, gigabytes of
/// heap); run one at a time, so each starts from a fresh instance:
///
/// ```text
/// QSM_WEIGHTS_DIR=… cargo test --release --target wasm32-unknown-unknown --features "onnx simd" \
///   --test wasm_pipeline -- --include-ignored --nocapture --exact browser_memory::hd_bet_low_memory
/// ```
///
/// A WASM memory only grows, so its size after the run is the run's peak. Each case prints it
/// with the size before the run (the inputs and the provided weights). A run that needs more
/// than the 4 GB a 32-bit heap can address aborts with `RuntimeError: unreachable` after
/// "memory allocation failed". Single-threaded: a threaded host running tiles concurrently
/// holds one tile's activations per worker.
#[cfg(all(feature = "onnx", target_arch = "wasm32"))]
mod browser_memory {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test as test;

    const DIMS: (usize, usize, usize) = (184, 256, 128);
    const ECHOES: [f64; 5] = [0.005, 0.011, 0.017, 0.023, 0.029];
    const BROWSER_TILE: Option<(usize, usize)> = Some((56, 4));

    fn heap_mb() -> f64 {
        core::arch::wasm32::memory_size(0) as f64 * 65536.0 / 1e6
    }

    fn scan_meta(dims: (usize, usize, usize), voxel: (f64, f64, f64)) -> ScanMetadata {
        ScanMetadata {
            dims,
            voxel_size: voxel,
            echo_times: ECHOES.to_vec(),
            field_strength: B0_T,
            b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: None,
            affine: Some(affine(voxel)),
        }
    }

    /// Fields of a head-sized ellipsoid, made voxel by voxel (no FFT, which would itself raise
    /// the heap the measurement reads). The values only need to look like data.
    struct Scan {
        m: ScanMetadata,
    }

    impl Scan {
        fn new() -> Self {
            Self { m: scan_meta(DIMS, (1.0, 1.0, 1.0)) }
        }
        fn n(&self) -> usize {
            let (x, y, z) = self.m.dims;
            x * y * z
        }
        /// Normalised radius of voxel `l` in the head ellipsoid (< 1 inside).
        fn radius(&self, l: usize) -> f64 {
            let (nx, ny, nz) = self.m.dims;
            let (i, j, k) = (l % nx, (l / nx) % ny, l / (nx * ny));
            let r = |x: usize, n: usize| ((x as f64 + 0.5) / n as f64 - 0.5) / 0.42;
            (r(i, nx).powi(2) + r(j, ny).powi(2) + r(k, nz).powi(2)).sqrt()
        }
        fn mask(&self) -> Vec<u8> {
            (0..self.n()).map(|l| (self.radius(l) < 0.85) as u8).collect()
        }
        fn chi(&self) -> Vec<f64> {
            (0..self.n()).map(|l| if self.radius(l) < 0.85 { 0.05 * (l as f64 * 0.37).sin() } else { 0.0 }).collect()
        }
        fn local(&self) -> Vec<f64> {
            (0..self.n()).map(|l| if self.radius(l) < 0.85 { 0.02 * (l as f64 * 0.11).cos() } else { 0.0 }).collect()
        }
        fn total(&self) -> Vec<f64> {
            let nx = self.m.dims.0;
            self.local().iter().enumerate().map(|(l, v)| v + 0.5 * ((l % nx) as f64 / nx as f64 - 0.5)).collect()
        }
        /// Magnitude per echo: decaying tissue in the head, low noise outside it (not zero, as in
        /// a real scan, so nothing is cropped away).
        fn magnitudes(&self) -> Vec<Vec<f64>> {
            ECHOES
                .iter()
                .map(|te| {
                    (0..self.n())
                        .map(|l| if self.radius(l) < 1.0 { 1000.0 * (-30.0 * te).exp() } else { 20.0 + (l % 7) as f64 })
                        .collect()
                })
                .collect()
        }
        fn phases(&self) -> Vec<Vec<f64>> {
            let total = self.total();
            let rad_per_ppm_s = 2.0 * PI * GAMMA_HZ_PER_T * B0_T * 1e-6;
            ECHOES
                .iter()
                .map(|te| {
                    total
                        .iter()
                        .map(|f| {
                            let p = f * rad_per_ppm_s * te;
                            p - 2.0 * PI * ((p + PI) / (2.0 * PI)).floor()
                        })
                        .collect()
                })
                .collect()
        }
        fn r2star(&self) -> Vec<f64> {
            (0..self.n()).map(|l| if self.radius(l) < 0.85 { 25.0 + 5.0 * (l as f64 * 0.21).sin() } else { 0.0 }).collect()
        }
    }

    /// Provide `id`'s weights, run `run`, and print the heap before and after.
    fn measure<T, E: std::fmt::Display>(id: &str, config: &str, run: impl FnOnce() -> Result<T, E>) {
        let Some(_weights) = provide(id) else { return };
        let before = heap_mb();
        wasm_bindgen_test::console_log!("START {id} [{config}] before {before:.0} MB");
        let t = js_sys::Date::now();
        let out = run();
        let secs = (js_sys::Date::now() - t) / 1e3;
        let after = heap_mb();
        if let Err(e) = &out {
            panic!("{id} ({config}): {e}");
        }
        wasm_bindgen_test::console_log!("PEAK {id} [{config}] before {before:.0} MB, peak {after:.0} MB, {secs:.0} s");
    }

    fn mask_op(id: &str, config: &str, op: MaskOp, s: &Scan) {
        let mags = s.magnitudes();
        let refs: Vec<&[f64]> = mags.iter().map(Vec::as_slice).collect();
        let section = MaskSection { input: MaskingInput::MagnitudeFirst, generator: op, refinements: vec![] };
        measure(id, config, || run_masking(&[section], &refs, &refs, &s.m));
    }

    fn inversion(algorithm: InversionAlgorithm, tile: Option<(usize, usize)>) {
        let s = Scan::new();
        let id = algorithm.dl_model_id().unwrap();
        let config = InversionConfig { algorithm, tile, ..Default::default() };
        let field = match config.input(ECHOES.len()) {
            InversionInput::LocalField => s.local(),
            _ => s.total(),
        };
        let mask = s.mask();
        let label = match tile {
            None => "whole volume".to_string(),
            Some((c, h)) => format!("tile ({c}, {h})"),
        };
        measure(id, &label, || run_dipole_inversion(&field, &mask, &s.m, &config, None, &mut |_, _| {}));
    }

    fn tile_label(tile: Option<(usize, usize)>) -> String {
        tile.map_or("whole volume".to_string(), |(c, h)| format!("tile ({c}, {h})"))
    }

    fn bfrnet(tile: Option<(usize, usize)>) {
        let s = Scan::new();
        let (total, mask) = (s.total(), s.mask());
        let config = BgRemovalConfig { algorithm: BgRemovalAlgorithm::Bfrnet, tile, ..Default::default() };
        measure("bfrnet", &tile_label(tile), || run_bg_removal(&total, &mask, &s.m, &config, None, &mut |_, _| {}));
    }

    /// Tiled BFRnet against the 4 GiB heap with the tiles a 4-thread browser keeps in flight.
    ///
    /// Node runs this harness on one thread, so it cannot hold several tiles at once; what it can
    /// show is the measured part and the arithmetic on top of it. It runs the browser config on
    /// the browser-sized scan one tile at a time and reads the heap's high-water, which holds
    /// everything a run needs once (inputs, weights, parsed model, volumes) plus one tile. Each
    /// further tile in flight adds one tile's heap, `inversion::tile_bytes` at BFRnet's registry
    /// figure, which `tile_cost_bfrnet` measures in WASM. So a threaded run's peak is the measured
    /// one plus that for every tile beyond the first, and with the cap a 4-thread pool gets
    /// (`inversion::wasm_tiles_in_flight`), that has to stay under 4 GiB. It also prints what the
    /// four tiles the old shared (xQSM) figure let through would need.
    fn bfrnet_tiled_at_four_threads() {
        use qsm_core::inversion::{tile_bytes, wasm_tiles_in_flight, TileConfig};
        let Some(_weights) = provide("bfrnet") else { return };
        let s = Scan::new();
        let (total, mask) = (s.total(), s.mask());
        let (core, halo) = BROWSER_TILE.unwrap();
        let config = BgRemovalConfig { algorithm: BgRemovalAlgorithm::Bfrnet, tile: BROWSER_TILE, ..Default::default() };
        let base = heap_mb();
        run_bg_removal(&total, &mask, &s.m, &config, None, &mut |_, _| {}).unwrap_or_else(|e| panic!("bfrnet: {e}"));
        let one = heap_mb();
        let cfg = TileConfig { core, halo };
        let cost = qsm_core::models::find_model("bfrnet").unwrap().tile_cost().unwrap();
        let tile_mb = tile_bytes(&cfg, &cost) as f64 / 1e6;
        let cap = wasm_tiles_in_flight(&cfg, &cost, 4);
        let at_cap = one + (cap - 1) as f64 * tile_mb;
        let at_four = one + 3.0 * tile_mb;
        let heap = 4.0 * 1024.0 * 1024.0 * 1024.0 / 1e6;
        wasm_bindgen_test::console_log!(
            "BFRNET before {base:.0} MB, one tile in flight {one:.0} MB, a tile {tile_mb:.0} MB; 4 threads: cap {cap} \
             tiles -> {at_cap:.0} MB; 4 tiles would be {at_four:.0} MB; heap {heap:.0} MB"
        );
        assert!(at_cap < heap, "{cap} BFRnet tiles in flight need {at_cap:.0} MB, over the {heap:.0} MB heap");
    }

    /// The wrapped-phase networks, on all five echoes.
    fn phase_net(id: &str, tile: Option<(usize, usize)>) {
        let s = Scan::new();
        let (ph, mg, mask) = (s.phases(), s.magnitudes(), s.mask());
        let ph: Vec<&[f64]> = ph.iter().map(Vec::as_slice).collect();
        let mg: Vec<&[f64]> = mg.iter().map(Vec::as_slice).collect();
        let label = format!("5 echoes, {}", tile_label(tile));
        measure(id, &label, || match id {
            "iqfm" => run_iqfm(&ph, &mg, &mask, &s.m, tile),
            "iqsm" => run_iqsm(&ph, &mg, &mask, &s.m, QsmReference::None, tile),
            _ => run_iqsm_plus(&ph, &mg, &mask, &s.m, QsmReference::None, tile),
        });
    }

    fn synthseg(config: &str, params: qsm_core::segment::SynthSegParams) {
        let s = Scan::new();
        let mag = s.magnitudes().swap_remove(0);
        measure("synthseg", config, || synthseg_run(&mag, &s.m.grid(), &params, |_, _| {}));
    }

    fn separation(algorithm: SeparationAlgorithm, patch: Option<(usize, usize, usize)>) {
        let s = Scan::new();
        let id = algorithm.dl_model_id().unwrap();
        let (local, chi, mask) = (s.local(), s.chi(), s.mask());
        let r2prime: Vec<f64> = s.r2star().iter().map(|r| 0.25 * r).collect();
        let inputs = SeparationInputs {
            local_field_ppm: &local,
            qsm: &chi,
            mask: &mask,
            r2prime: Some(&r2prime),
            r2star: None,
            magnitude_rss: None,
            magnitude_multi: None,
            se_magnitude_multi: None,
        };
        let config = separation_config(algorithm, patch);
        let label = patch.map_or("authors' setting".to_string(), |(x, y, z)| format!("patch {x}×{y}×{z}"));
        measure(id, &label, || run_separation(inputs, &s.m, &config, &mut |_, _| {}));
    }

    fn r2primenet(config: &str, patch: (usize, usize, usize)) {
        use qsm_core::relaxometry::{R2PrimeNetNorm, R2PrimeNetParams};
        let s = Scan::new();
        let (r2star, mask) = (s.r2star(), s.mask());
        let params = R2PrimeNetParams { patch, ..Default::default() };
        measure("r2primenet", config, || {
            let w = qsm_core::models::primary_weight("r2primenet")?;
            qsm_core::relaxometry::r2primenet(&r2star, &mask, &s.m.grid(), B0_T, &w, &R2PrimeNetNorm::default(), &params, |_, _| {})
                .map_err(|e| e.to_string())
        });
    }

    /// Heap one tile of tiled network `id` costs in WASM, per patch voxel: the figure its registry
    /// entry carries (`ModelSpec::tile_bytes_per_voxel`), measured where it is spent.
    ///
    /// The network's own tiled entry point runs on an 80³ volume whose mask touches only the first
    /// tile, so one tile is in flight, at 48³, 64³ and 80³ patches in turn (64³ and 128³ for
    /// NeXtQSM). The heap after each is
    /// that size's high-water (memory only grows, and a larger run reuses what a smaller one
    /// freed), and the least-squares slope of high-water against patch voxels is the cost of a tile
    /// per voxel, fragmentation included; the intercept (parsed model, plan, volumes) is paid once.
    /// `examples/tile_memory_footprint.rs` measures the same natively, by counting allocations.
    fn tile_cost(id: &str) {
        use qsm_core::inversion::{self, TileConfig};
        let Some(dir) = weights_dir() else { return };
        let spec = qsm_core::models::find_model(id).unwrap();
        let w: Vec<Vec<u8>> = spec.files.iter().map(|f| read_file(&format!("{dir}/{}", f.name))).collect();
        let (nx, ny, nz) = (80, 80, 80);
        let grid = qsm_core::Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
        let field: Vec<f64> = (0..nx * ny * nz).map(|l| 0.02 * (l as f64 * 0.11).cos()).collect();
        let mut mask = vec![0u8; nx * ny * nz];
        for k in 4..12 {
            for j in 4..12 {
                for i in 4..12 {
                    mask[i + nx * (j + ny * k)] = 1;
                }
            }
        }
        let (bdir, te, b0) = ((0.0, 0.0, 1.0), 0.02, 3.0);
        let divisor = spec.size_divisor as usize;
        // Multiples of 16 or less: three sizes. NeXtQSM takes multiples of 64: its two smallest.
        let patches = if divisor <= 16 { vec![48, 64, 80] } else { vec![divisor, 2 * divisor] };
        let before = heap_mb();
        let mut pts = Vec::new();
        let mut at_64 = 0.0;
        for p in patches {
            let cfg = TileConfig { core: p - 16, halo: 8 };
            assert_eq!(inversion::tile_patch_size(&cfg, divisor), p);
            let none = |_: usize, _: usize| {};
            let (f, m, g) = (&field[..], &mask[..], &grid);
            let out = match id {
                "bfrnet" => qsm_core::bgremove::bfrnet::bfrnet_tiled(f, m, g, &w[0], &cfg, none),
                "xqsm" => inversion::xqsm_tiled(f, m, g, &w[0], &cfg, none),
                "qsmnet" => inversion::qsmnet_tiled(f, m, g, &w[0], &inversion::QsmnetNorm::qsmnet(), &cfg, none),
                "qsmnet-plus" => {
                    inversion::qsmnet_tiled(f, m, g, &w[0], &inversion::QsmnetNorm::qsmnet_plus(), &cfg, none)
                }
                "ir2qsm" => inversion::ir2qsm::ir2qsm_tiled(f, m, g, &w[0], &cfg, none),
                "lpcnn" => inversion::lpcnn::lpcnn_tiled(f, m, g, bdir, &w[0], &cfg, none),
                "modl-qsm" => inversion::modl_qsm::modl_qsm_tiled(f, m, g, bdir, &w[0], &cfg, none),
                "nextqsm" => inversion::nextqsm::nextqsm_tiled(f, m, g, bdir, &w[0], &w[1], &cfg, none),
                "iqsm" => inversion::iqsm_multi_echo(&[f], &[], m, g, &[te], b0, 1.0, 0, &w[0], Some(&cfg)),
                "iqsm-plus" => {
                    inversion::iqsm_plus_multi_echo(&[f], &[], m, g, &[te], b0, bdir, 1.0, 0, &w[0], Some(&cfg))
                }
                "iqfm" => inversion::iqfm_multi_echo(&[f], &[], m, g, &[te], b0, 1.0, 0, &w[0], Some(&cfg)),
                _ => panic!("{id} is not a tiled network"),
            };
            out.unwrap_or_else(|e| panic!("{id}: {e}"));
            let mb = heap_mb() - before;
            wasm_bindgen_test::console_log!("TILE {id} {p}³ patch: heap +{mb:.0} MB");
            let voxels = (p * p * p) as f64;
            if p == 64 {
                at_64 = mb * 1e6 / voxels;
            }
            pts.push((voxels, mb * 1e6));
        }
        let n = pts.len() as f64;
        let (mx, my) = pts.iter().fold((0.0, 0.0), |(a, b), &(x, y)| (a + x / n, b + y / n));
        let slope = pts.iter().map(|&(x, y)| (x - mx) * (y - my)).sum::<f64>()
            / pts.iter().map(|&(x, _)| (x - mx).powi(2)).sum::<f64>();
        wasm_bindgen_test::console_log!(
            "TILE {id} => slope {slope:.0} B per patch voxel, intercept {:.0} MB; a 64³ tile with the \
             intercept {at_64:.0} B per patch voxel (registry: {:?})",
            (my - slope * mx) / 1e6,
            spec.tile_bytes_per_voxel
        );
    }

    macro_rules! cases {
        ($($name:ident => $body:expr;)*) => {$(
            #[test]
            #[ignore = "minutes and gigabytes: measures peak memory on a browser-sized scan"]
            fn $name() {
                $body
            }
        )*};
    }

    cases! {
        hd_bet_whole_volume => mask_op("hd-bet", "whole volume", MaskOp::HdBet(qsm_core::bet::HdBetParams::default()), &Scan::new());
        hd_bet_low_memory => mask_op("hd-bet", "low_memory 128×128×64", MaskOp::HdBet(qsm_core::bet::HdBetParams::low_memory()), &Scan::new());
        // A mouse GRE at 0.17×0.20×0.8 mm, the geometry RS2-Net was checked on in vivo.
        rs2_net_mouse => mask_op("rs2-net", "mouse 128×100×20 @ 0.17×0.20×0.8 mm", MaskOp::Rs2Net(Default::default()),
            &Scan { m: scan_meta((128, 100, 20), (0.17, 0.20, 0.8)) });
        mindgrab => mask_op("mindgrab", "default", MaskOp::MindGrab(Default::default()), &Scan::new());
        synthseg_whole_volume => synthseg("whole volume", qsm_core::segment::SynthSegParams::default());
        synthseg_low_memory => synthseg("low_memory 96³", qsm_core::segment::SynthSegParams::low_memory());
        bfrnet_whole_volume => bfrnet(None);
        bfrnet_tiled => bfrnet(BROWSER_TILE);
        bfrnet_tiled_cap_at_four_threads => bfrnet_tiled_at_four_threads();
        iqfm_whole_volume => phase_net("iqfm", None);
        iqfm_tiled => phase_net("iqfm", BROWSER_TILE);
        iqsm_whole_volume => phase_net("iqsm", None);
        iqsm_tiled => phase_net("iqsm", BROWSER_TILE);
        iqsm_plus_whole_volume => phase_net("iqsm-plus", None);
        iqsm_plus_tiled => phase_net("iqsm-plus", BROWSER_TILE);
        xqsm_whole_volume => inversion(InversionAlgorithm::Xqsm, None);
        xqsm_tiled => inversion(InversionAlgorithm::Xqsm, BROWSER_TILE);
        qsmnet_whole_volume => inversion(InversionAlgorithm::Qsmnet, None);
        qsmnet_tiled => inversion(InversionAlgorithm::Qsmnet, BROWSER_TILE);
        qsmnet_plus_whole_volume => inversion(InversionAlgorithm::QsmnetPlus, None);
        qsmnet_plus_tiled => inversion(InversionAlgorithm::QsmnetPlus, BROWSER_TILE);
        ir2qsm_whole_volume => inversion(InversionAlgorithm::Ir2qsm, None);
        ir2qsm_tiled => inversion(InversionAlgorithm::Ir2qsm, BROWSER_TILE);
        lpcnn_whole_volume => inversion(InversionAlgorithm::Lpcnn, None);
        lpcnn_tiled => inversion(InversionAlgorithm::Lpcnn, BROWSER_TILE);
        modl_qsm_whole_volume => inversion(InversionAlgorithm::ModlQsm, None);
        modl_qsm_tiled => inversion(InversionAlgorithm::ModlQsm, BROWSER_TILE);
        nextqsm_whole_volume => inversion(InversionAlgorithm::Nextqsm, None);
        nextqsm_tiled => inversion(InversionAlgorithm::Nextqsm, BROWSER_TILE);
        qsmgan => inversion(InversionAlgorithm::Qsmgan, None);
        autoqsm => inversion(InversionAlgorithm::Autoqsm, None);
        r2primenet_authors_patch => r2primenet("authors' patch 192×192×128", qsm_core::relaxometry::AUTHORS_PATCH);
        r2primenet_wasm_patch => r2primenet("WASM_PATCH 128×128×64", qsm_core::relaxometry::WASM_PATCH);
        susep_net_whole_volume => separation(SeparationAlgorithm::SusepNet, None);
        susep_net_wasm_patch => separation(SeparationAlgorithm::SusepNet, Some(qsm_core::separation::susep_net::WASM_PATCH));
        chi_sepnet_authors_patch => separation(SeparationAlgorithm::ChiSepNet, None);
        chi_sepnet_wasm_patch => separation(SeparationAlgorithm::ChiSepNet, Some(qsm_core::separation::chisepnet::WASM_PATCH));
        tile_cost_bfrnet => tile_cost("bfrnet");
        tile_cost_xqsm => tile_cost("xqsm");
        tile_cost_qsmnet => tile_cost("qsmnet");
        tile_cost_qsmnet_plus => tile_cost("qsmnet-plus");
        tile_cost_ir2qsm => tile_cost("ir2qsm");
        tile_cost_lpcnn => tile_cost("lpcnn");
        tile_cost_modl_qsm => tile_cost("modl-qsm");
        tile_cost_nextqsm => tile_cost("nextqsm");
        tile_cost_iqsm => tile_cost("iqsm");
        tile_cost_iqsm_plus => tile_cost("iqsm-plus");
        tile_cost_iqfm => tile_cost("iqfm");
    }
}

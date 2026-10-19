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
//! Deep-learning variants are skipped (see [`needs_weights`]): they need model weights, which a
//! WASM host hands over in memory. [`xqsm_on_provided_weights`] and the SynthSeg cases
//! ([`synthseg_on_provided_weights`], [`synthseg_beyond_the_heap_in_windows`]) do so: each reads
//! the file named by an environment variable at compile time (`$XQSM_ONNX`, `$SYNTHSEG_ONNX`)
//! through Node's `fs`, passes the bytes to `models::provide_weights` and runs the model. To add
//! another, do the same.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

use num_complex::Complex64;
use qsm_core::bet::BetParams;
use qsm_core::fieldmap::{B0WeightType, LinearFitParams, RobustFitParams};
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
    for &algorithm in BgRemovalAlgorithm::VARIANTS {
        if needs_weights(algorithm.dl_model_id()) {
            continue;
        }
        for msmv_refine in [false, true] {
            let config = BgRemovalConfig { algorithm, msmv_refine, ..Default::default() };
            let r = run_bg_removal(&p.total_ppm, &p.mask, &m, &config, &mut no_progress())
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
                let config = InversionConfig { qsmart: qsm_core::utils::QsmartParams::for_field_strength(B0_T), ..config };
                run_qsmart(field, &p.mask, Some(&mag), &m, &config, QsmReference::Mean, &mut no_progress())
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

/// The bytes of the weights file `path` (an environment variable `var`, read when this test was
/// compiled), or `None` (with a note) when it was unset. CI downloads the file and sets it (see
/// the `wasm` job), so there an unset variable is a failure, not a skip.
#[cfg(feature = "onnx")]
fn weights_bytes(var: &str, path: Option<&'static str>) -> Option<Vec<u8>> {
    let Some(path) = path else {
        assert!(option_env!("CI").is_none(), "${var} must be set in CI");
        println!("skipping: ${var} was not set at compile time");
        return None;
    };
    #[cfg(target_arch = "wasm32")]
    {
        #[wasm_bindgen::prelude::wasm_bindgen(module = "fs")]
        extern "C" {
            #[wasm_bindgen(js_name = readFileSync, catch)]
            fn read_file_sync(path: &str) -> Result<js_sys::Uint8Array, wasm_bindgen::JsValue>;
        }
        let bytes = read_file_sync(path).unwrap_or_else(|e| panic!("read {path}: {e:?}"));
        Some(bytes.to_vec())
    }
    #[cfg(not(target_arch = "wasm32"))]
    Some(std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}")))
}

/// xQSM through `run_dipole_inversion` on weights supplied in memory, as a browser host runs it.
#[cfg(feature = "onnx")]
#[test]
fn xqsm_on_provided_weights() {
    use qsm_core::models;
    const FILE: &str = "xqsm.onnx";
    let Some(bytes) = weights_bytes("XQSM_ONNX", option_env!("XQSM_ONNX")) else { return };
    let p = Phantom::new();
    let m = meta();
    let config = InversionConfig {
        algorithm: InversionAlgorithm::Xqsm,
        ..Default::default()
    };
    assert_eq!(config.input(TES.len()), InversionInput::LocalField);

    // Without supplied bytes there is nothing to load on wasm32 (natively the cache or a download
    // could still find the file), and the error says what to do.
    models::forget_weights(FILE);
    #[cfg(target_arch = "wasm32")]
    {
        let err =
            run_dipole_inversion(&p.local_ppm, &p.mask, &m, &config, None, &mut no_progress())
                .expect_err("xQSM ran without weights");
        let msg = err.to_string();
        assert!(
            msg.contains("provide_weights") && msg.contains(FILE),
            "{msg}"
        );
    }

    models::provide_weights(FILE, bytes).unwrap();
    let chi = run_dipole_inversion(&p.local_ppm, &p.mask, &m, &config, None, &mut no_progress());
    models::forget_weights(FILE);
    check("Xqsm", &chi.unwrap_or_else(|e| panic!("Xqsm: {e}")));
}

/// SynthSeg's weights, as the engine hands them to `segment::synthseg`: provided in memory under
/// the registry's file name and read back through the registry, as a browser host does.
#[cfg(feature = "onnx")]
fn synthseg_weights() -> Option<qsm_core::models::WeightBytes> {
    use qsm_core::models;
    const FILE: &str = "synthseg.onnx";
    let bytes = weights_bytes("SYNTHSEG_ONNX", option_env!("SYNTHSEG_ONNX"))?;
    models::provide_weights(FILE, bytes).unwrap();
    let weights = models::primary_weight("synthseg").unwrap_or_else(|e| panic!("synthseg weights: {e}"));
    models::forget_weights(FILE);
    Some(weights)
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

/// A 1 mm identity-affine grid of `dims`, with a bright ellipsoid "head" in it.
#[cfg(feature = "onnx")]
fn synthseg_input(dims: (usize, usize, usize)) -> (Vec<f64>, qsm_core::Grid, [f64; 16]) {
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
    let grid = qsm_core::Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
    let mut affine = [0.0; 16];
    for a in [0, 5, 10, 15] {
        affine[a] = 1.0;
    }
    (mag, grid, affine)
}

/// SynthSeg on the smallest network input it takes: the 16³ phantom's magnitude, which it pads
/// to 128³ (SynthSeg's minimum, `MIN_PAD`), so this costs one 128³ pass whatever the phantom's
/// size. That fits a 32-bit heap whole.
#[cfg(feature = "onnx")]
#[test]
fn synthseg_on_provided_weights() {
    use qsm_core::segment::{synthseg, SynthSegInference};
    let Some(weights) = synthseg_weights() else { return };
    let p = Phantom::new();
    let grid = qsm_core::Grid::new(N, N, N, 1.0, 1.0, 1.0);
    let (_, _, affine) = synthseg_input((1, 1, 1));
    let params = synthseg_quick(SynthSegInference::FullVolume);
    let out = synthseg(&p.magnitudes[0], &grid, &affine, &weights, &params, no_progress())
        .unwrap_or_else(|e| panic!("synthseg: {e}"));
    assert_eq!(out.labels.len(), NV);
    assert_eq!(out.volumes.len(), params.version.n_labels());
    assert!(out.volumes.iter().all(|v| v.is_finite()));
}

/// SynthSeg on a volume whose whole-volume pass needs more than a 32-bit heap: 128×128×256 at
/// 1 mm (no resampling, no extra padding) takes about 4.5 GB in one pass, and the wasm build
/// aborts ("memory allocation failed", seen as `RuntimeError: unreachable`). In windows the
/// network sees a patch at a time. A step of a whole patch keeps it to two 128³ windows, as much
/// network as the volume itself.
#[cfg(feature = "onnx")]
#[test]
fn synthseg_beyond_the_heap_in_windows() {
    use qsm_core::segment::{synthseg, SynthSegInference};
    let Some(weights) = synthseg_weights() else { return };
    let (mag, grid, affine) = synthseg_input((128, 128, 256));
    let params = synthseg_quick(SynthSegInference::SlidingWindow { patch: 128, tile_step: 1.0 });
    let mut passes = 0;
    let out = synthseg(&mag, &grid, &affine, &weights, &params, |done, total| {
        passes = total;
        assert!(done <= total);
    })
    .unwrap_or_else(|e| panic!("synthseg: {e}"));
    assert_eq!(passes, 2, "two windows");
    assert_eq!(out.labels.len(), mag.len());
    assert!(out.labels.iter().any(|&l| l != 0), "found no brain");
}

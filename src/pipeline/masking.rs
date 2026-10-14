//! Masking stage
//!
//! Generates a binary brain mask from magnitude/phase data using a
//! configurable sequence of operations (threshold, BET, morphological ops).
//! Multiple mask sections can be OR'd together.
//!
//! ## Each section builds its own input
//!
//! The stage takes every echo's magnitude, and each [`MaskSection`] builds what its
//! [`MaskingInput`] asks for, independently of the other sections:
//!
//! | input | thresholded image | magnitude for BET / HD-BET / RS2-Net / signal-gated erosion |
//! |---|---|---|
//! | [`MaskingInput::MagnitudeFirst`] | echo 1 | echo 1 |
//! | [`MaskingInput::Magnitude`] | root-sum-of-squares over echoes | the RSS |
//! | [`MaskingInput::MagnitudeLast`] | last echo | last echo |
//! | [`MaskingInput::PhaseQuality`] | ROMEO voxel quality of echoes 1–2, weighted by the RSS | the RSS |
//!
//! **Phase quality is weighted by the RSS**, not echo 1: it is the highest-SNR magnitude there
//! is, it is the same whatever other sections ask for, and it is what ROMEO itself weights by
//! when unwrapping multi-echo data. With one echo the RSS is that echo.
//!
//! Any homogeneity correction is the caller's, applied to all echoes before they are passed in.

use super::config::*;
use super::phase_utils::rss_combine;
use crate::utils::morphology::{erode_mask, dilate_mask, morphological_close, fill_holes, signal_gated_erosion};

/// Check every volume is the grid's size.
fn check_volumes(name: &str, volumes: &[&[f64]], n_voxels: usize) -> Result<(), PipelineError> {
    for (e, v) in volumes.iter().enumerate() {
        if v.len() != n_voxels {
            return Err(PipelineError::InvalidInput(format!(
                "{name} echo {} has {} voxels, the grid has {n_voxels}", e + 1, v.len()
            )));
        }
    }
    Ok(())
}

/// The magnitude a section with this `input` works from (see the module docs): echo 1, the
/// root-sum-of-squares over echoes, or the last echo. [`MaskingInput::PhaseQuality`] uses the
/// RSS. `None` when there are no magnitudes.
///
/// This is also the magnitude the section's BET, HD-BET, RS2-Net and signal-gated erosion see,
/// so a host that applies a section's ops itself through [`apply_mask_ops`] should pass it as
/// `magnitude`.
///
/// # Errors
/// [`PipelineError::InvalidInput`] if an echo does not match the grid.
pub fn section_magnitude(
    input: MaskingInput,
    magnitudes: &[&[f64]],
    metadata: &ScanMetadata,
) -> Result<Option<Vec<f64>>, PipelineError> {
    check_volumes("magnitude", magnitudes, metadata.grid().n_total())?;
    let (Some(first), Some(last)) = (magnitudes.first(), magnitudes.last()) else {
        return Ok(None);
    };
    Ok(Some(match input {
        MaskingInput::MagnitudeFirst => first.to_vec(),
        MaskingInput::MagnitudeLast => last.to_vec(),
        MaskingInput::Magnitude | MaskingInput::PhaseQuality => rss_combine(magnitudes)?,
    }))
}

/// The image a section with this `input` thresholds, built from `magnitude` (that section's
/// [`section_magnitude`]) and, for phase quality, the phases.
fn section_input(
    input: MaskingInput,
    phases: &[&[f64]],
    magnitude: Option<&[f64]>,
    metadata: &ScanMetadata,
) -> Result<Vec<f64>, PipelineError> {
    match input {
        MaskingInput::MagnitudeFirst | MaskingInput::Magnitude | MaskingInput::MagnitudeLast => {
            magnitude.map(<[f64]>::to_vec).ok_or_else(|| {
                PipelineError::InvalidInput(format!("a {input:?} mask section needs magnitude data"))
            })
        }
        MaskingInput::PhaseQuality => {
            let n_voxels = metadata.grid().n_total();
            check_volumes("phase", phases, n_voxels)?;
            let Some(&phase1) = phases.first() else {
                return Err(PipelineError::InvalidInput(
                    "a PhaseQuality mask section needs phase data".into(),
                ));
            };
            let ones;
            let weight = match magnitude {
                Some(m) => m,
                None => {
                    ones = vec![1.0; n_voxels];
                    &ones
                }
            };
            let all_ones = vec![1u8; n_voxels];
            let grid = metadata.grid();
            Ok(if phases.len() >= 2 && metadata.echo_times.len() >= 2 {
                crate::unwrap::voxel_quality_romeo(
                    phase1, weight,
                    Some(phases[1]),
                    metadata.echo_times[0], metadata.echo_times[1],
                    &all_ones, &grid,
                )
            } else {
                crate::unwrap::voxel_quality_romeo(
                    phase1, weight,
                    None,
                    metadata.echo_times.first().copied().unwrap_or(0.02),
                    0.0, &all_ones, &grid,
                )
            })
        }
    }
}

/// The image a section with this `input` thresholds (see the module docs).
///
/// # Arguments
/// * `input` - Which data source to use
/// * `phases` - Per-echo wrapped phase (for [`MaskingInput::PhaseQuality`])
/// * `magnitudes` - Per-echo magnitude, all echoes (optionally homogeneity-corrected)
/// * `metadata` - Scan metadata
///
/// # Errors
/// [`PipelineError::InvalidInput`] if the input needs data that was not supplied, or a volume
/// does not match the grid.
pub fn resolve_masking_input(
    input: MaskingInput,
    phases: &[&[f64]],
    magnitudes: &[&[f64]],
    metadata: &ScanMetadata,
) -> Result<Vec<f64>, PipelineError> {
    let magnitude = section_magnitude(input, magnitudes, metadata)?;
    section_input(input, phases, magnitude.as_deref(), metadata)
}

/// Build a mask from a single section (generator + refinements). The section builds its own
/// input and magnitude from the echoes (see the module docs).
///
/// # Arguments
/// * `section` - Mask section config (input type, generator, refinements)
/// * `phases` - Per-echo wrapped phase (for [`MaskingInput::PhaseQuality`])
/// * `magnitudes` - Per-echo magnitude, all echoes (optionally homogeneity-corrected)
/// * `metadata` - Scan metadata
pub fn build_mask_section(
    section: &MaskSection,
    phases: &[&[f64]],
    magnitudes: &[&[f64]],
    metadata: &ScanMetadata,
) -> Result<Vec<u8>, PipelineError> {
    let n_voxels = metadata.dims.0 * metadata.dims.1 * metadata.dims.2;
    let magnitude = section_magnitude(section.input, magnitudes, metadata)?;
    let input_data = section_input(section.input, phases, magnitude.as_deref(), metadata)?;
    apply_mask_ops(vec![1u8; n_voxels], &section.all_ops(), &input_data, magnitude.as_deref(), metadata)
}

/// Apply mask operations to an existing mask, in order.
///
/// This is the one implementation of what every mask op *means*; [`build_mask_section`] is this
/// function starting from an all-ones mask, and hosts that drive masking themselves (e.g. an
/// interactive UI applying one refinement at a time) should call it rather than reimplement the
/// operations, so a mask built step-by-step matches the one a `--mask` section would produce.
///
/// **Two different images.** `input_data` is what a generator looks at — thresholding may use a
/// phase-quality map, for instance. `magnitude` is the magnitude image, and is what the ops that
/// need real signal use: [`MaskOp::Bet`], [`MaskOp::HdBet`], [`MaskOp::Rs2Net`], [`MaskOp::MindGrab`] and
/// [`MaskOp::SignalErode`]. Passing
/// the section input as `magnitude` is a bug: signal-gated erosion divides out a receive-coil bias
/// estimate and gates on the in-mask median, which only means anything for a magnitude image.
/// Those ops error when `magnitude` is `None`. To match what [`build_mask_section`] does, pass
/// [`resolve_masking_input`] as `input_data` and [`section_magnitude`] as `magnitude`.
///
/// # Arguments
/// * `mask` - Starting mask (0/1), length `nx*ny*nz`
/// * `ops` - Operations to apply, in order
/// * `input_data` - Image the generators threshold (may be a phase-quality map)
/// * `magnitude` - Magnitude image, for BET / HD-BET / signal-gated erosion
/// * `metadata` - Scan metadata (dims + voxel size)
pub fn apply_mask_ops(
    mask: Vec<u8>,
    ops: &[MaskOp],
    input_data: &[f64],
    magnitude: Option<&[f64]>,
    metadata: &ScanMetadata,
) -> Result<Vec<u8>, PipelineError> {
    let (nx, ny, nz) = metadata.dims;
    let grid = metadata.grid();
    let n_voxels = nx * ny * nz;
    if mask.len() != n_voxels {
        return Err(PipelineError::DimensionMismatch { expected: n_voxels, got: mask.len() });
    }
    let mut mask = mask;

    for op in ops {
        match op {
            MaskOp::Threshold { method, value } => {
                let threshold = match method {
                    MaskThresholdMethod::Otsu => {
                        crate::utils::otsu_threshold(input_data, 256)
                    }
                    MaskThresholdMethod::Fixed => value.unwrap_or(0.5),
                    MaskThresholdMethod::Percentile => {
                        let pct = value.unwrap_or(75.0) / 100.0;
                        let mut sorted: Vec<f64> = input_data.iter()
                            .filter(|v| v.is_finite() && **v > 0.0)
                            .copied().collect();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        if sorted.is_empty() { 0.0 }
                        else {
                            let idx = ((sorted.len() as f64 * pct) as usize).min(sorted.len() - 1);
                            sorted[idx]
                        }
                    }
                };
                mask = input_data.iter()
                    .map(|&v| if v > threshold { 1u8 } else { 0u8 })
                    .collect();
            }
            MaskOp::Bet(params) => {
                let mag_data = magnitude.ok_or_else(|| {
                    PipelineError::InvalidInput("BET requires magnitude data".into())
                })?;
                mask = crate::bet::run_bet(mag_data, &grid, params, |_, _| {});
            }
            MaskOp::Erode { iterations } => {
                mask = erode_mask(&mask, &grid, *iterations);
            }
            MaskOp::Dilate { iterations } => {
                mask = dilate_mask(&mask, &grid, *iterations);
            }
            MaskOp::Close { radius } => {
                mask = morphological_close(&mask, &grid, *radius as i32);
            }
            MaskOp::FillHoles { max_size } => {
                let effective_size = if *max_size == 0 { n_voxels / 20 } else { *max_size };
                mask = fill_holes(&mask, &grid, effective_size);
            }
            MaskOp::GaussianSmooth { sigma_mm } => {
                let sigma = *sigma_mm;
                let mask_f64: Vec<f64> = mask.iter().map(|&m| m as f64).collect();
                let smoothed = crate::utils::gaussian_smooth_3d(
                    &mask_f64,
                    [sigma, sigma, sigma],
                    None, None, 3,
                    &grid,
                );
                mask = smoothed.iter().map(|&v| if v > 0.5 { 1u8 } else { 0u8 }).collect();
            }
            MaskOp::SignalErode(params) => {
                let mag_data = magnitude.ok_or_else(|| {
                    PipelineError::InvalidInput("signal-gated erosion requires magnitude data".into())
                })?;
                mask = signal_gated_erosion(&mask, mag_data, &grid, params);
            }
            MaskOp::HdBet(params) => {
                let mag_data = magnitude.ok_or_else(|| {
                    PipelineError::InvalidInput("HD-BET requires magnitude data".into())
                })?;
                mask = run_hd_bet(mag_data, &grid, params)?;
            }
            MaskOp::Rs2Net(params) => {
                let mag_data = magnitude.ok_or_else(|| {
                    PipelineError::InvalidInput("RS2-Net requires magnitude data".into())
                })?;
                mask = run_rs2_net(mag_data, &grid, params)?;
            }
            MaskOp::MindGrab(params) => {
                let mag_data = magnitude.ok_or_else(|| {
                    PipelineError::InvalidInput("MindGrab requires magnitude data".into())
                })?;
                let affine = metadata.affine.as_ref().ok_or_else(|| {
                    PipelineError::InvalidInput("MindGrab requires the image affine".into())
                })?;
                mask = run_mindgrab(mag_data, &grid, affine, params)?;
            }
        }
    }

    Ok(mask)
}

/// Source the HD-BET weights and run it. Requires the `onnx` feature; weights come from the
/// model registry through [`crate::models`] (supplied bytes first, then local/download on native).
#[cfg(feature = "onnx")]
fn run_hd_bet(
    magnitude: &[f64],
    grid: &crate::Grid,
    params: &crate::bet::HdBetParams,
) -> Result<Vec<u8>, PipelineError> {
    let bytes = crate::models::primary_weight("hd-bet").map_err(PipelineError::InvalidConfig)?;
    crate::bet::hd_bet(magnitude, grid, &bytes, params, |_, _| {})
        .map_err(|e| PipelineError::AlgorithmError(e.to_string()))
}

#[cfg(not(feature = "onnx"))]
fn run_hd_bet(
    _magnitude: &[f64],
    _grid: &crate::Grid,
    _params: &crate::bet::HdBetParams,
) -> Result<Vec<u8>, PipelineError> {
    Err(PipelineError::InvalidConfig(
        "HD-BET requires building qsm-core with the 'onnx' feature".into(),
    ))
}

/// Source the RS2-Net weights and run it. Requires the `onnx` feature; weights come from the
/// model registry through [`crate::models`] (supplied bytes first, then local/download on native).
#[cfg(feature = "onnx")]
fn run_rs2_net(
    magnitude: &[f64],
    grid: &crate::Grid,
    params: &crate::bet::Rs2NetParams,
) -> Result<Vec<u8>, PipelineError> {
    let bytes = crate::models::primary_weight("rs2-net").map_err(PipelineError::InvalidConfig)?;
    crate::bet::rs2_net(magnitude, grid, &bytes, params, |_, _| {})
        .map_err(|e| PipelineError::AlgorithmError(e.to_string()))
}

#[cfg(not(feature = "onnx"))]
fn run_rs2_net(
    _magnitude: &[f64],
    _grid: &crate::Grid,
    _params: &crate::bet::Rs2NetParams,
) -> Result<Vec<u8>, PipelineError> {
    Err(PipelineError::InvalidConfig(
        "RS2-Net requires building qsm-core with the 'onnx' feature".into(),
    ))
}

/// Source the MindGrab weights and run it. Requires the `onnx` feature; weights come from the
/// model registry through [`crate::models`] (supplied bytes first, then local/download on native).
#[cfg(feature = "onnx")]
fn run_mindgrab(
    magnitude: &[f64],
    grid: &crate::Grid,
    affine: &[f64; 16],
    params: &crate::bet::MindGrabParams,
) -> Result<Vec<u8>, PipelineError> {
    let bytes = crate::models::primary_weight("mindgrab").map_err(PipelineError::InvalidConfig)?;
    crate::bet::mindgrab(magnitude, grid, affine, &bytes, params, |_, _| {})
        .map_err(|e| PipelineError::AlgorithmError(e.to_string()))
}

#[cfg(not(feature = "onnx"))]
fn run_mindgrab(
    _magnitude: &[f64],
    _grid: &crate::Grid,
    _affine: &[f64; 16],
    _params: &crate::bet::MindGrabParams,
) -> Result<Vec<u8>, PipelineError> {
    Err(PipelineError::InvalidConfig(
        "MindGrab requires building qsm-core with the 'onnx' feature".into(),
    ))
}

/// Build a mask from multiple sections, OR'd together.
///
/// Each section specifies an input source, a generator (threshold/BET),
/// and optional refinements (erode, dilate, close, fill holes, smooth). Each builds its own
/// input from the echoes, so sections with different inputs do not affect each other.
///
/// # Arguments
/// * `sections` - Mask section configs
/// * `phases` - Per-echo wrapped phase (for [`MaskingInput::PhaseQuality`])
/// * `magnitudes` - Per-echo magnitude, all echoes (optionally homogeneity-corrected)
/// * `metadata` - Scan metadata
pub fn run_masking(
    sections: &[MaskSection],
    phases: &[&[f64]],
    magnitudes: &[&[f64]],
    metadata: &ScanMetadata,
) -> Result<Vec<u8>, PipelineError> {
    let (nx, ny, nz) = metadata.dims;
    let n_voxels = nx * ny * nz;

    if sections.is_empty() {
        return Err(PipelineError::InvalidConfig("no mask sections configured".into()));
    }

    let mut final_mask = vec![0u8; n_voxels];
    for section in sections {
        let section_mask = build_mask_section(section, phases, magnitudes, metadata)?;
        for (f, s) in final_mask.iter_mut().zip(&section_mask) {
            *f |= s;
        }
    }

    Ok(final_mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_metadata() -> ScanMetadata {
        ScanMetadata {
            dims: (8, 8, 8),
            voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005, 0.010],
            field_strength: 3.0,
            b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: None,
            affine: None,
        }
    }

    #[test]
    fn test_run_masking_otsu_threshold() {
        let meta = test_metadata();
        let n = 8 * 8 * 8;
        // Half bright, half dark
        let mut mag = vec![0.1; n];
        for i in n / 2..n {
            mag[i] = 10.0;
        }

        let sections = vec![MaskSection {
            input: MaskingInput::Magnitude,
            generator: MaskOp::Threshold {
                method: MaskThresholdMethod::Otsu,
                value: None,
            },
            refinements: vec![],
        }];

        let result = run_masking(&sections, &[], &[&mag], &meta).unwrap();
        assert_eq!(result.len(), n);

        // Bright half should be masked in, dark half masked out
        let bright_count: usize = result[n / 2..].iter().map(|&m| m as usize).sum();
        let dark_count: usize = result[..n / 2].iter().map(|&m| m as usize).sum();
        assert!(bright_count > dark_count, "Otsu should separate bright from dark");
    }

    #[test]
    fn test_run_masking_with_refinements() {
        let meta = test_metadata();
        let n = 8 * 8 * 8;
        let mag = vec![10.0; n]; // all bright → all masked in

        let sections = vec![MaskSection {
            input: MaskingInput::Magnitude,
            generator: MaskOp::Threshold {
                method: MaskThresholdMethod::Fixed,
                value: Some(0.5),
            },
            refinements: vec![
                MaskOp::Erode { iterations: 1 },
                MaskOp::Dilate { iterations: 1 },
            ],
        }];

        let result = run_masking(&sections, &[], &[&mag], &meta).unwrap();
        assert_eq!(result.len(), n);
        // After erode+dilate, interior should still be masked
        let (nx, ny, nz) = meta.dims;
        let center = nx / 2 + (ny / 2) * nx + (nz / 2) * nx * ny;
        assert_eq!(result[center], 1, "center should survive erode+dilate");
    }

    #[test]
    fn test_run_masking_or_sections() {
        let meta = test_metadata();
        let n = 8 * 8 * 8;
        let mag = vec![10.0; n];

        // Section 1: mask only first half via fixed threshold
        // Section 2: mask only second half
        // OR → should get everything
        let sections = vec![
            MaskSection {
                input: MaskingInput::Magnitude,
                generator: MaskOp::Threshold {
                    method: MaskThresholdMethod::Fixed,
                    value: Some(0.5),
                },
                refinements: vec![],
            },
            MaskSection {
                input: MaskingInput::Magnitude,
                generator: MaskOp::Threshold {
                    method: MaskThresholdMethod::Fixed,
                    value: Some(0.5),
                },
                refinements: vec![],
            },
        ];

        let result = run_masking(&sections, &[], &[&mag], &meta).unwrap();
        let count: usize = result.iter().map(|&m| m as usize).sum();
        assert_eq!(count, n, "OR of identical sections should give full mask");
    }

    #[test]
    fn test_masking_fixed_threshold() {
        let meta = test_metadata();
        let n = 8 * 8 * 8;
        let mag = vec![10.0; n];
        let sections = vec![MaskSection {
            input: MaskingInput::Magnitude,
            generator: MaskOp::Threshold { method: MaskThresholdMethod::Fixed, value: Some(5.0) },
            refinements: vec![],
        }];
        let result = run_masking(&sections, &[], &[&mag], &meta).unwrap();
        let count: usize = result.iter().map(|&m| m as usize).sum();
        assert_eq!(count, n, "all voxels above threshold=5");
    }

    #[test]
    fn test_masking_percentile_threshold() {
        let meta = test_metadata();
        let n = 8 * 8 * 8;
        let mag: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let sections = vec![MaskSection {
            input: MaskingInput::Magnitude,
            generator: MaskOp::Threshold { method: MaskThresholdMethod::Percentile, value: Some(50.0) },
            refinements: vec![],
        }];
        let result = run_masking(&sections, &[], &[&mag], &meta).unwrap();
        let count: usize = result.iter().map(|&m| m as usize).sum();
        assert!(count > 0 && count < n, "percentile should mask ~half");
    }

    #[test]
    fn test_masking_close_and_fill_holes() {
        let meta = test_metadata();
        let n = 8 * 8 * 8;
        let mag = vec![10.0; n];
        let sections = vec![MaskSection {
            input: MaskingInput::Magnitude,
            generator: MaskOp::Threshold { method: MaskThresholdMethod::Fixed, value: Some(0.5) },
            refinements: vec![
                MaskOp::Close { radius: 1 },
                MaskOp::FillHoles { max_size: 0 },
                MaskOp::GaussianSmooth { sigma_mm: 1.0 },
            ],
        }];
        let result = run_masking(&sections, &[], &[&mag], &meta).unwrap();
        assert_eq!(result.len(), n);
    }

    #[test]
    fn test_masking_phase_quality_input() {
        let meta = test_metadata();
        let n = 8 * 8 * 8;
        let phase1 = vec![0.5; n];
        let phase2 = vec![1.0; n];
        let mag = vec![1.0; n];
        let sections = vec![MaskSection {
            input: MaskingInput::PhaseQuality,
            generator: MaskOp::Threshold { method: MaskThresholdMethod::Otsu, value: None },
            refinements: vec![],
        }];
        let result = run_masking(&sections, &[&phase1, &phase2], &[&mag], &meta).unwrap();
        assert_eq!(result.len(), n);
    }

    #[test]
    fn test_masking_signal_erode() {
        let meta = test_metadata();
        let n = 8 * 8 * 8;
        let mag = vec![10.0; n];
        let section = |refinement| vec![MaskSection {
            input: MaskingInput::Magnitude,
            generator: MaskOp::Threshold { method: MaskThresholdMethod::Fixed, value: Some(5.0) },
            refinements: vec![refinement],
        }];
        let params = crate::utils::morphology::SignalErosionParams { min_component: 1, ..Default::default() };
        let result = run_masking(&section(MaskOp::SignalErode(params.clone())), &[], &[&mag], &meta).unwrap();
        // Uniform signal: nothing is gated, so only the one global erosion applies.
        let plain = run_masking(&section(MaskOp::Erode { iterations: 1 }), &[], &[&mag], &meta).unwrap();
        assert_eq!(result, plain);
        // Needs magnitude.
        assert!(run_masking(&section(MaskOp::SignalErode(params)), &[], &[], &meta).is_err());
    }

    #[test]
    fn test_masking_hd_bet_requires_magnitude() {
        let meta = test_metadata();
        let sections = vec![MaskSection {
            input: MaskingInput::Magnitude,
            generator: MaskOp::HdBet(Default::default()),
            refinements: vec![],
        }];
        assert!(run_masking(&sections, &[], &[], &meta).is_err());
        #[cfg(not(feature = "onnx"))]
        assert!(run_masking(&sections, &[], &[&vec![1.0; 512]], &meta).is_err());
    }

    #[test]
    fn test_masking_rs2_net_requires_magnitude() {
        let meta = test_metadata();
        let sections = vec![MaskSection {
            input: MaskingInput::Magnitude,
            generator: MaskOp::Rs2Net(Default::default()),
            refinements: vec![],
        }];
        assert!(run_masking(&sections, &[], &[], &meta).is_err());
        #[cfg(not(feature = "onnx"))]
        assert!(run_masking(&sections, &[], &[&vec![1.0; 512]], &meta).is_err());
    }

    #[test]
    fn test_masking_mindgrab_requires_magnitude_and_affine() {
        let mut meta = test_metadata();
        let sections = vec![MaskSection {
            input: MaskingInput::Magnitude,
            generator: MaskOp::MindGrab(Default::default()),
            refinements: vec![],
        }];
        let mag = vec![1.0; 512];
        assert!(run_masking(&sections, &[], &[], &meta).is_err());
        // no affine
        let err = run_masking(&sections, &[], &[&mag], &meta).unwrap_err();
        assert!(err.to_string().contains("affine"), "{err}");
        meta.affine = Some([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        #[cfg(not(feature = "onnx"))]
        assert!(run_masking(&sections, &[], &[&mag], &meta).is_err());
    }

    /// Applying ops one at a time (what an interactive host does) must equal building the whole
    /// section in one call — same implementation, so a step-by-step mask matches the `--mask`
    /// section a host prints alongside it.
    #[test]
    fn test_apply_mask_ops_matches_build_mask_section() {
        let meta = test_metadata();
        let n = 8 * 8 * 8;
        let e1: Vec<f64> = (0..n).map(|i| 50.0 + ((i * 7) % 13) as f64).collect();
        let e2: Vec<f64> = (0..n).map(|i| ((i * 37) % 19) as f64).collect();
        // what the host would hand apply_mask_ops for a MagnitudeLast section
        let input = section_magnitude(MaskingInput::MagnitudeLast, &[&e1, &e2], &meta).unwrap().unwrap();
        let mag = input.clone();
        assert_eq!(input, e2);
        let section = MaskSection {
            input: MaskingInput::MagnitudeLast,
            generator: MaskOp::Threshold { method: MaskThresholdMethod::Fixed, value: Some(5.0) },
            refinements: vec![
                MaskOp::Dilate { iterations: 1 },
                MaskOp::FillHoles { max_size: 0 },
                MaskOp::Erode { iterations: 1 },
                MaskOp::SignalErode(crate::utils::morphology::SignalErosionParams { min_component: 1, ..Default::default() }),
            ],
        };
        let whole = build_mask_section(&section, &[], &[&e1, &e2], &meta).unwrap();

        let mut step = vec![1u8; n];
        for op in section.all_ops() {
            step = apply_mask_ops(step, std::slice::from_ref(&op), &input, Some(&mag), &meta).unwrap();
        }
        assert_eq!(step, whole);
    }

    #[test]
    fn test_apply_mask_ops_needs_magnitude_and_matching_length() {
        let meta = test_metadata();
        let n = 8 * 8 * 8;
        let input = vec![10.0; n];
        // Signal-gated erosion gates on the magnitude, so without one it is an error rather than
        // silently falling back to the section input.
        let ops = [MaskOp::SignalErode(Default::default())];
        assert!(apply_mask_ops(vec![1u8; n], &ops, &input, None, &meta).is_err());
        // A mask of the wrong size is rejected rather than mis-indexed.
        assert!(apply_mask_ops(vec![1u8; n - 1], &[MaskOp::Erode { iterations: 1 }], &input, Some(&input), &meta).is_err());
    }

    /// Synthetic 3-echo volume on the 8³ test grid: x < 4 is bright on echo 1 and has decayed
    /// by the last echo; x ≥ 4 is dim on echo 1 and bright on the last (#173).
    fn two_region_echoes() -> Vec<Vec<f64>> {
        let n = 8 * 8 * 8;
        let left = |i: usize| i % 8 < 4;
        vec![
            (0..n).map(|i| if left(i) { 10.0 } else { 1.0 }).collect(),
            vec![5.0; n],
            (0..n).map(|i| if left(i) { 1.0 } else { 10.0 }).collect(),
        ]
    }

    fn fixed(input: MaskingInput) -> MaskSection {
        MaskSection {
            input,
            generator: MaskOp::Threshold { method: MaskThresholdMethod::Fixed, value: Some(6.0) },
            refinements: vec![],
        }
    }

    fn count(mask: &[u8]) -> usize {
        mask.iter().map(|&m| m as usize).sum()
    }

    /// The three magnitude inputs are three different images.
    #[test]
    fn magnitude_inputs_are_distinguishable() {
        let meta = test_metadata();
        let echoes = two_region_echoes();
        let mags: Vec<&[f64]> = echoes.iter().map(Vec::as_slice).collect();
        let get = |input| resolve_masking_input(input, &[], &mags, &meta).unwrap();
        assert_eq!(get(MaskingInput::MagnitudeFirst), echoes[0]);
        assert_eq!(get(MaskingInput::MagnitudeLast), echoes[2]);
        let rss = get(MaskingInput::Magnitude);
        assert!((rss[0] - (100.0f64 + 25.0 + 1.0).sqrt()).abs() < 1e-12);
        assert_ne!(rss, echoes[0]);
        assert_ne!(rss, echoes[2]);
    }

    /// Two sections with different inputs each see their own input; swapping one section's
    /// input changes only that section's mask.
    #[test]
    fn each_section_uses_its_own_input() {
        let meta = test_metadata();
        let echoes = two_region_echoes();
        let mags: Vec<&[f64]> = echoes.iter().map(Vec::as_slice).collect();
        let n = 8 * 8 * 8;
        let build = |input| build_mask_section(&fixed(input), &[], &mags, &meta).unwrap();

        let first = build(MaskingInput::MagnitudeFirst);
        let last = build(MaskingInput::MagnitudeLast);
        // echo 1 picks the left half, the last echo the right half
        assert_eq!(count(&first), n / 2);
        assert_eq!(count(&last), n / 2);
        assert!(first.iter().zip(&last).all(|(a, b)| a + b == 1));

        // both sections together: each contributes its own half
        let both = run_masking(&[fixed(MaskingInput::MagnitudeFirst), fixed(MaskingInput::MagnitudeLast)], &[], &mags, &meta).unwrap();
        assert_eq!(count(&both), n);

        // swap the second section to echo 1: only its contribution changes, to the left half
        let swapped = run_masking(&[fixed(MaskingInput::MagnitudeFirst), fixed(MaskingInput::MagnitudeFirst)], &[], &mags, &meta).unwrap();
        assert_eq!(swapped, first);
        // and the first section's own mask is the same whatever the second asks for
        assert_eq!(build(MaskingInput::MagnitudeFirst), first);
    }

    /// Signal-gated ops get the section's own magnitude: a MagnitudeLast section's signal
    /// erosion gates on the last echo, an echo-1 section's on echo 1.
    #[test]
    fn signal_gated_ops_use_the_sections_magnitude() {
        let meta = test_metadata();
        let n = 8 * 8 * 8;
        // echo 1 uniform; the last echo has a dropout along the x = 0 face
        let e1 = vec![10.0; n];
        let e_last: Vec<f64> = (0..n).map(|i| if i % 8 == 0 { 0.5 } else { 10.0 }).collect();
        let mags: [&[f64]; 2] = [&e1, &e_last];
        let params = crate::utils::morphology::SignalErosionParams { min_component: 1, ..Default::default() };
        let section = |input| MaskSection {
            input,
            generator: MaskOp::Threshold { method: MaskThresholdMethod::Fixed, value: Some(0.1) },
            refinements: vec![MaskOp::SignalErode(params.clone())],
        };
        let all = vec![1u8; n];
        let erode_with = |m: &[f64]| apply_mask_ops(all.clone(), &[MaskOp::SignalErode(params.clone())], m, Some(m), &meta).unwrap();
        let on_first = erode_with(&e1);
        let on_last = erode_with(&e_last);
        assert_ne!(on_first, on_last, "test magnitudes do not separate the two cases");

        assert_eq!(build_mask_section(&section(MaskingInput::MagnitudeFirst), &[], &mags, &meta).unwrap(), on_first);
        assert_eq!(build_mask_section(&section(MaskingInput::MagnitudeLast), &[], &mags, &meta).unwrap(), on_last);
    }

    /// Phase quality is weighted by the RSS of all echoes (the documented choice), whatever
    /// the other sections use.
    #[test]
    fn phase_quality_is_weighted_by_the_rss() {
        let meta = test_metadata();
        let n = 8 * 8 * 8;
        let phase1: Vec<f64> = (0..n).map(|i| ((i * 13) % 7) as f64 * 0.4 - 1.2).collect();
        let phase2: Vec<f64> = phase1.iter().map(|p| 2.0 * p).collect();
        let echoes = two_region_echoes();
        let mags: Vec<&[f64]> = echoes.iter().map(Vec::as_slice).collect();
        let rss = rss_combine(&mags).unwrap();
        let phases: [&[f64]; 2] = [&phase1, &phase2];
        let q = resolve_masking_input(MaskingInput::PhaseQuality, &phases, &mags, &meta).unwrap();
        let q_rss = resolve_masking_input(MaskingInput::PhaseQuality, &phases, &[&rss], &meta).unwrap();
        let q_e1 = resolve_masking_input(MaskingInput::PhaseQuality, &phases, &mags[..1], &meta).unwrap();
        assert_eq!(q, q_rss);
        assert_ne!(q, q_e1);
        assert_eq!(section_magnitude(MaskingInput::PhaseQuality, &mags, &meta).unwrap(), Some(rss));
    }

    #[test]
    fn missing_or_mismatched_data_is_an_error() {
        let meta = test_metadata();
        let n = 8 * 8 * 8;
        let mag = vec![1.0; n];
        let short = vec![1.0; n - 1];
        for input in [MaskingInput::MagnitudeFirst, MaskingInput::Magnitude, MaskingInput::MagnitudeLast] {
            assert!(resolve_masking_input(input, &[], &[], &meta).is_err(), "{input:?} without magnitude");
            assert!(resolve_masking_input(input, &[], &[&mag, &short], &meta).is_err(), "{input:?} short echo");
        }
        assert!(resolve_masking_input(MaskingInput::PhaseQuality, &[], &[&mag], &meta).is_err());
        assert!(resolve_masking_input(MaskingInput::PhaseQuality, &[&short], &[&mag], &meta).is_err());
        // phase quality without magnitude is weighted uniformly, as before
        assert!(resolve_masking_input(MaskingInput::PhaseQuality, &[&mag], &[], &meta).is_ok());
    }

    #[test]
    fn test_run_masking_empty_sections() {
        let meta = test_metadata();
        let result = run_masking(&[], &[], &[], &meta);
        assert!(result.is_err());
    }

    /// Every [`BetParams`](crate::bet::BetParams) field set on `MaskOp::Bet` must reach
    /// `run_bet` — not just the fractional intensity and voxel scale the op used to carry.
    #[test]
    fn test_bet_op_passes_every_bet_param_through() {
        use crate::bet::{run_bet, BetParams};
        let n = 20;
        let meta = ScanMetadata { dims: (n, n, n), ..test_metadata() };
        let grid = meta.grid();
        // A bright ellipsoid on a dim background, with a soft edge so the surface parameters matter.
        let c = (n as f64 - 1.0) / 2.0;
        let mag: Vec<f64> = (0..n * n * n).map(|idx| {
            let (i, j, k) = (idx % n, (idx / n) % n, idx / (n * n));
            let r = (((i as f64 - c) / 7.0).powi(2) + ((j as f64 - c) / 6.0).powi(2)
                + ((k as f64 - c) / 5.0).powi(2)).sqrt();
            10.0 + 190.0 / (1.0 + ((r - 1.0) * 2.5).exp())
        }).collect();
        let base = BetParams { iterations: 50, subdivisions: 1, ..BetParams::default() };
        let base_mask = run_bet(&mag, &grid, &base, |_, _| {});

        let variants = [
            BetParams { fractional_intensity: 0.2, ..base.clone() },
            BetParams { voxel_scale: 2.0, ..base.clone() },
            BetParams { smoothness: 3.0, ..base.clone() },
            BetParams { gradient_threshold: 0.5, ..base.clone() },
            BetParams { iterations: 5, ..base.clone() },
            BetParams { subdivisions: 2, ..base.clone() },
        ];
        for params in variants {
            let direct = run_bet(&mag, &grid, &params, |_, _| {});
            assert!(direct != base_mask, "{params:?} should change the mask, or this test proves nothing");
            let op = MaskOp::Bet(params.clone());
            let via_op = apply_mask_ops(vec![1; n * n * n], &[op], &mag, Some(&mag), &meta).unwrap();
            assert_eq!(via_op, direct, "MaskOp::Bet dropped a field of {params:?}");
        }
    }
}

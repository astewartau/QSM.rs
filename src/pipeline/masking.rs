//! Masking stage
//!
//! Generates a binary brain mask from magnitude/phase data using a
//! configurable sequence of operations (threshold, BET, morphological ops).
//! Multiple mask sections can be OR'd together.

use super::config::*;
use super::phase_utils::{erode_mask, dilate_mask};

/// Resolve masking input data based on the MaskingInput type.
///
/// # Arguments
/// * `input` - Which data source to use
/// * `phases` - Per-echo phase arrays
/// * `magnitudes` - Per-echo magnitude arrays (already resolved: RSS for Magnitude,
///   specific echo for First/Last, optionally homogeneity-corrected)
/// * `metadata` - Scan metadata
///
/// # Returns
/// The input data array to threshold/mask from
pub fn resolve_masking_input(
    input: MaskingInput,
    phases: &[&[f64]],
    magnitude: Option<&[f64]>,
    metadata: &ScanMetadata,
) -> Vec<f64> {
    let (nx, ny, nz) = metadata.dims;
    let n_voxels = nx * ny * nz;

    match input {
        MaskingInput::MagnitudeFirst | MaskingInput::Magnitude | MaskingInput::MagnitudeLast => {
            magnitude.map(|m| m.to_vec()).unwrap_or_else(|| vec![0.0; n_voxels])
        }
        MaskingInput::PhaseQuality => {
            if phases.is_empty() {
                return vec![0.0; n_voxels];
            }
            let all_ones = vec![1u8; n_voxels];
            let mag = magnitude.unwrap_or(&[]);
            let mag_data: Vec<f64> = if mag.is_empty() {
                vec![1.0; n_voxels]
            } else {
                mag.to_vec()
            };

            let grid = metadata.grid();
            if phases.len() >= 2 && metadata.echo_times.len() >= 2 {
                crate::unwrap::voxel_quality_romeo(
                    phases[0], &mag_data,
                    Some(phases[1]),
                    metadata.echo_times[0], metadata.echo_times[1],
                    &all_ones, &grid,
                )
            } else {
                crate::unwrap::voxel_quality_romeo(
                    phases[0], &mag_data,
                    None,
                    metadata.echo_times.first().copied().unwrap_or(0.02),
                    0.0, &all_ones, &grid,
                )
            }
        }
    }
}

/// Build a mask from a single section (generator + refinements).
///
/// # Arguments
/// * `section` - Mask section config (input type, generator, refinements)
/// * `input_data` - Pre-resolved input data (from `resolve_masking_input`)
/// * `magnitude` - Magnitude data for BET (optional)
/// * `metadata` - Scan metadata
pub fn build_mask_section(
    section: &MaskSection,
    input_data: &[f64],
    magnitude: Option<&[f64]>,
    metadata: &ScanMetadata,
) -> Result<Vec<u8>, PipelineError> {
    let n_voxels = metadata.dims.0 * metadata.dims.1 * metadata.dims.2;
    apply_mask_ops(vec![1u8; n_voxels], &section.all_ops(), input_data, magnitude, metadata)
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
/// need real signal use: [`MaskOp::Bet`], [`MaskOp::HdBet`] and [`MaskOp::SignalErode`]. Passing
/// the section input as `magnitude` is a bug: signal-gated erosion divides out a receive-coil bias
/// estimate and gates on the in-mask median, which only means anything for a magnitude image.
/// Those ops error when `magnitude` is `None`.
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
    let (vsx, vsy, vsz) = metadata.voxel_size;
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
            MaskOp::Bet { fractional_intensity } => {
                let mag_data = magnitude.ok_or_else(|| {
                    PipelineError::InvalidInput("BET requires magnitude data".into())
                })?;
                let bet_params = crate::bet::BetParams {
                    fractional_intensity: *fractional_intensity,
                    ..crate::bet::BetParams::default()
                };
                let grid = crate::Grid::new(nx, ny, nz, vsx, vsy, vsz);
                mask = crate::bet::run_bet(mag_data, &grid, &bet_params, |_, _| {});
            }
            MaskOp::Erode { iterations } => {
                mask = erode_mask(&mask, &grid, *iterations);
            }
            MaskOp::Dilate { iterations } => {
                mask = dilate_mask(&mask, &grid, *iterations);
            }
            MaskOp::Close { radius } => {
                mask = crate::utils::morphological_close(&mask, &grid, *radius as i32);
            }
            MaskOp::FillHoles { max_size } => {
                let effective_size = if *max_size == 0 { n_voxels / 20 } else { *max_size };
                mask = crate::utils::fill_holes(&mask, &grid, effective_size);
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
                mask = crate::utils::signal_gated_erosion(&mask, mag_data, &grid, params);
            }
            MaskOp::HdBet(params) => {
                let mag_data = magnitude.ok_or_else(|| {
                    PipelineError::InvalidInput("HD-BET requires magnitude data".into())
                })?;
                mask = run_hd_bet(mag_data, &grid, params)?;
            }
        }
    }

    Ok(mask)
}

/// Source the HD-BET weights and run it. Requires the `onnx` feature; weights come from the
/// model registry (local `$QSM_MODEL_DIR`/cache, or the `download` feature).
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

/// Build a mask from multiple sections, OR'd together.
///
/// Each section specifies an input source, a generator (threshold/BET),
/// and optional refinements (erode, dilate, close, fill holes, smooth).
///
/// # Arguments
/// * `sections` - Mask section configs
/// * `phases` - Per-echo phase arrays (for PhaseQuality input)
/// * `magnitude` - Combined magnitude (for Magnitude/BET input)
/// * `metadata` - Scan metadata
pub fn run_masking(
    sections: &[MaskSection],
    phases: &[&[f64]],
    magnitude: Option<&[f64]>,
    metadata: &ScanMetadata,
) -> Result<Vec<u8>, PipelineError> {
    let (nx, ny, nz) = metadata.dims;
    let n_voxels = nx * ny * nz;

    if sections.is_empty() {
        return Err(PipelineError::InvalidConfig("no mask sections configured".into()));
    }

    if sections.len() == 1 {
        let input_data = resolve_masking_input(sections[0].input, phases, magnitude, metadata);
        return build_mask_section(&sections[0], &input_data, magnitude, metadata);
    }

    // Multiple sections: run each, OR together
    let mut final_mask = vec![0u8; n_voxels];
    for section in sections {
        let input_data = resolve_masking_input(section.input, phases, magnitude, metadata);
        let section_mask = build_mask_section(section, &input_data, magnitude, metadata)?;
        for j in 0..n_voxels {
            final_mask[j] |= section_mask[j];
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

        let result = run_masking(&sections, &[], Some(&mag), &meta).unwrap();
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

        let result = run_masking(&sections, &[], Some(&mag), &meta).unwrap();
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

        let result = run_masking(&sections, &[], Some(&mag), &meta).unwrap();
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
        let result = run_masking(&sections, &[], Some(&mag), &meta).unwrap();
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
        let result = run_masking(&sections, &[], Some(&mag), &meta).unwrap();
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
        let result = run_masking(&sections, &[], Some(&mag), &meta).unwrap();
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
        let result = run_masking(&sections, &[&phase1, &phase2], Some(&mag), &meta).unwrap();
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
        let params = crate::utils::SignalErosionParams { min_component: 1, ..Default::default() };
        let result = run_masking(&section(MaskOp::SignalErode(params.clone())), &[], Some(&mag), &meta).unwrap();
        // Uniform signal: nothing is gated, so only the one global erosion applies.
        let plain = run_masking(&section(MaskOp::Erode { iterations: 1 }), &[], Some(&mag), &meta).unwrap();
        assert_eq!(result, plain);
        // Needs magnitude.
        assert!(run_masking(&section(MaskOp::SignalErode(params)), &[], None, &meta).is_err());
    }

    #[test]
    fn test_masking_hd_bet_requires_magnitude() {
        let meta = test_metadata();
        let sections = vec![MaskSection {
            input: MaskingInput::Magnitude,
            generator: MaskOp::HdBet(Default::default()),
            refinements: vec![],
        }];
        assert!(run_masking(&sections, &[], None, &meta).is_err());
        #[cfg(not(feature = "onnx"))]
        assert!(run_masking(&sections, &[], Some(&vec![1.0; 512]), &meta).is_err());
    }

    /// Applying ops one at a time (what an interactive host does) must equal building the whole
    /// section in one call — same implementation, so a step-by-step mask matches the `--mask`
    /// section a host prints alongside it.
    #[test]
    fn test_apply_mask_ops_matches_build_mask_section() {
        let meta = test_metadata();
        let n = 8 * 8 * 8;
        let input: Vec<f64> = (0..n).map(|i| ((i * 37) % 19) as f64).collect();
        let mag: Vec<f64> = (0..n).map(|i| 50.0 + ((i * 7) % 13) as f64).collect();
        let section = MaskSection {
            input: MaskingInput::Magnitude,
            generator: MaskOp::Threshold { method: MaskThresholdMethod::Fixed, value: Some(5.0) },
            refinements: vec![
                MaskOp::Dilate { iterations: 1 },
                MaskOp::FillHoles { max_size: 0 },
                MaskOp::Erode { iterations: 1 },
                MaskOp::SignalErode(crate::utils::SignalErosionParams { min_component: 1, ..Default::default() }),
            ],
        };
        let whole = build_mask_section(&section, &input, Some(&mag), &meta).unwrap();

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

    #[test]
    fn test_run_masking_empty_sections() {
        let meta = test_metadata();
        let result = run_masking(&[], &[], None, &meta);
        assert!(result.is_err());
    }
}

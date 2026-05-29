//! Masking stage
//!
//! Generates a binary brain mask from magnitude/phase data using a
//! configurable sequence of operations (threshold, BET, morphological ops).
//! Multiple mask sections can be OR'd together.

use super::config::*;
use super::phase_utils::{erode_mask, dilate_mask, rss_combine};

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

            if phases.len() >= 2 && metadata.echo_times.len() >= 2 {
                crate::unwrap::voxel_quality_romeo(
                    phases[0], &mag_data,
                    Some(phases[1]),
                    metadata.echo_times[0], metadata.echo_times[1],
                    &all_ones, nx, ny, nz,
                )
            } else {
                crate::unwrap::voxel_quality_romeo(
                    phases[0], &mag_data,
                    None,
                    metadata.echo_times.first().copied().unwrap_or(0.02),
                    0.0, &all_ones, nx, ny, nz,
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
    let (nx, ny, nz) = metadata.dims;
    let (vsx, vsy, vsz) = metadata.voxel_size;
    let n_voxels = nx * ny * nz;
    let mut mask = vec![1u8; n_voxels];

    for op in &section.all_ops() {
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
                let bet_defaults = crate::bet::BetParams::default();
                mask = crate::bet::run_bet(
                    mag_data, nx, ny, nz, vsx, vsy, vsz,
                    *fractional_intensity, bet_defaults.smoothness,
                    bet_defaults.gradient_threshold, bet_defaults.iterations,
                    bet_defaults.subdivisions,
                );
            }
            MaskOp::Erode { iterations } => {
                mask = erode_mask(&mask, nx, ny, nz, *iterations);
            }
            MaskOp::Dilate { iterations } => {
                mask = dilate_mask(&mask, nx, ny, nz, *iterations);
            }
            MaskOp::Close { radius } => {
                mask = crate::utils::morphological_close(&mask, nx, ny, nz, *radius as i32);
            }
            MaskOp::FillHoles { max_size } => {
                let effective_size = if *max_size == 0 { n_voxels / 20 } else { *max_size };
                mask = crate::utils::fill_holes(&mask, nx, ny, nz, effective_size);
            }
            MaskOp::GaussianSmooth { sigma_mm } => {
                let sigma = *sigma_mm;
                let mask_f64: Vec<f64> = mask.iter().map(|&m| m as f64).collect();
                let smoothed = crate::utils::gaussian_smooth_3d(
                    &mask_f64,
                    [sigma, sigma, sigma],
                    None, None, 3,
                    nx, ny, nz,
                );
                mask = smoothed.iter().map(|&v| if v > 0.5 { 1u8 } else { 0u8 }).collect();
            }
        }
    }

    Ok(mask)
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
    fn test_run_masking_empty_sections() {
        let meta = test_metadata();
        let result = run_masking(&[], &[], None, &meta);
        assert!(result.is_err());
    }
}

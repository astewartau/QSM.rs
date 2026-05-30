//! Field mapping stage
//!
//! Multi-echo phase → B0 field map in ppm.
//! Implements the canonical field mapping pipeline:
//! phase offset removal → bipolar correction → unwrapping → B0 estimation.

use super::config::*;
use super::phase_utils::{hz_to_ppm, rads_to_ppm};

/// Run field mapping: convert per-echo phase data to a B0 field map in ppm.
///
/// # Arguments
/// * `phases` - Per-echo wrapped phase arrays (in [-pi, pi])
/// * `magnitudes` - Per-echo magnitude arrays (optional; uniform weights if None)
/// * `mask` - Binary brain mask
/// * `metadata` - Scan metadata (dims, voxel size, echo times in seconds, field strength)
/// * `config` - Field mapping configuration
/// * `progress` - Progress callback (current_step, total_steps)
///
/// # Returns
/// `FieldMappingResult` with B0 field in ppm and optional phase offset
pub fn run_field_mapping(
    phases: &[&[f64]],
    magnitudes: Option<&[&[f64]]>,
    mask: &[u8],
    metadata: &ScanMetadata,
    config: &FieldMappingConfig,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<FieldMappingResult, PipelineError> {
    let (nx, ny, nz) = metadata.dims;
    let (vsx, vsy, vsz) = metadata.voxel_size;
    let n_voxels = nx * ny * nz;
    let n_echoes = phases.len();

    if n_echoes == 0 {
        return Err(PipelineError::InvalidInput("no phase echoes provided".into()));
    }
    if metadata.echo_times.len() != n_echoes {
        return Err(PipelineError::DimensionMismatch {
            expected: n_echoes,
            got: metadata.echo_times.len(),
        });
    }
    for (i, p) in phases.iter().enumerate() {
        if p.len() != n_voxels {
            return Err(PipelineError::DimensionMismatch {
                expected: n_voxels,
                got: p.len(),
            });
        }
        if let Some(ref mags) = magnitudes {
            if i < mags.len() && mags[i].len() != n_voxels {
                return Err(PipelineError::DimensionMismatch {
                    expected: n_voxels,
                    got: mags[i].len(),
                });
            }
        }
    }

    let tes = &metadata.echo_times;
    let is_laplacian = config.unwrapping_algorithm == UnwrappingAlgorithm::Laplacian;
    let do_offset = n_echoes > 1 && config.phase_offset_removal && !is_laplacian;

    // Create uniform magnitude fallback
    let uniform_mag = vec![1.0f64; n_voxels];
    let uniform_mags: Vec<&[f64]> = (0..n_echoes).map(|_| uniform_mag.as_slice()).collect();
    let mag_slices: &[&[f64]] = magnitudes.unwrap_or(&uniform_mags);

    progress(0, 4);

    if n_echoes > 1 && do_offset {
        // ---- Path A: Phase offset removal + unwrap + B0 estimation ----
        field_mapping_with_offset(
            phases, mag_slices, mask, metadata, config, n_voxels, nx, ny, nz, vsx, vsy, vsz, progress,
        )
    } else if n_echoes > 1 {
        // ---- Path B: Direct per-echo unwrapping + linear fit ----
        field_mapping_direct(
            phases, mag_slices, mask, metadata, config, n_voxels, nx, ny, nz, vsx, vsy, vsz, progress,
        )
    } else {
        // ---- Path C: Single echo ----
        field_mapping_single_echo(
            phases[0], mag_slices[0], mask, metadata, config, n_voxels, nx, ny, nz, vsx, vsy, vsz, progress,
        )
    }
}

/// Path A: Multi-echo with phase offset removal
#[allow(clippy::too_many_arguments)]
fn field_mapping_with_offset(
    phases: &[&[f64]],
    mag_slices: &[&[f64]],
    mask: &[u8],
    metadata: &ScanMetadata,
    config: &FieldMappingConfig,
    n_voxels: usize,
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<FieldMappingResult, PipelineError> {
    let tes = &metadata.echo_times;
    let n_echoes = phases.len();

    // Step 1: Phase offset removal
    progress(1, 4);
    let (mut corrected, phase_offset) = crate::utils::phase_offset_removal(
        phases, mag_slices, tes, mask,
        config.phase_offset_sigma, [0, 1],
        crate::unwrap::UnwrapMethod::Romeo,
        [vsx, vsy, vsz],
        nx, ny, nz,
    );

    // Step 2: Bipolar correction (optional, >= 3 echoes)
    if config.bipolar_correction && n_echoes >= 3 {
        crate::utils::bipolar_correction(
            &mut corrected, mag_slices, tes, mask,
            config.phase_offset_sigma, nx, ny, nz,
        );
    }

    // Step 3: Multi-echo unwrapping
    progress(2, 4);
    let unwrapped: Vec<Vec<f64>> = match config.unwrapping_algorithm {
        UnwrappingAlgorithm::Laplacian => {
            corrected.iter()
                .map(|p| crate::unwrap::laplacian_unwrap(p, mask, nx, ny, nz, vsx, vsy, vsz))
                .collect()
        }
        UnwrappingAlgorithm::Romeo => {
            crate::unwrap::unwrap_romeo_multi_echo(
                &corrected, mag_slices, tes, mask,
                &config.romeo_params, nx, ny, nz,
            )
        }
    };

    // Step 4: B0 estimation
    progress(3, 4);
    let b0_hz = match config.b0_estimation {
        B0EstimationMethod::WeightedAvg => {
            crate::utils::calculate_b0_weighted(
                &unwrapped, mag_slices, tes, mask,
                config.b0_weight_type, n_voxels,
            )
        }
        B0EstimationMethod::LinearFit => {
            let uw_refs: Vec<&[f64]> = unwrapped.iter().map(|u| u.as_slice()).collect();
            let fit = crate::utils::multi_echo_linear_fit(
                &uw_refs, mag_slices, tes, mask,
                config.linear_fit_params.estimate_offset,
                config.linear_fit_params.reliability_threshold_percentile,
            );
            crate::utils::field_to_hz(&fit.field)
        }
    };

    progress(4, 4);
    Ok(FieldMappingResult {
        b0_field_ppm: hz_to_ppm(&b0_hz, metadata.field_strength),
        phase_offset: Some(phase_offset),
    })
}

/// Path B: Multi-echo without phase offset removal (per-echo unwrap + linear fit)
#[allow(clippy::too_many_arguments)]
fn field_mapping_direct(
    phases: &[&[f64]],
    mag_slices: &[&[f64]],
    mask: &[u8],
    metadata: &ScanMetadata,
    config: &FieldMappingConfig,
    n_voxels: usize,
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<FieldMappingResult, PipelineError> {
    let tes = &metadata.echo_times;
    let n_echoes = phases.len();

    // Per-echo unwrapping
    progress(1, 4);
    let mut unwrapped: Vec<Vec<f64>> = Vec::with_capacity(n_echoes);
    for e in 0..n_echoes {
        let uw = unwrap_single(
            phases[e], mag_slices.first().copied().unwrap_or(&[]),
            mask, &config, nx, ny, nz, vsx, vsy, vsz,
            if e + 1 < n_echoes { Some(phases[e + 1]) } else { None },
            tes[e],
            if e + 1 < n_echoes { tes[e + 1] } else { 0.0 },
        );
        unwrapped.push(uw);
    }

    // Linear fit (always used in the no-offset path, matching qsmxt.rs reference)
    progress(3, 4);
    let uw_refs: Vec<&[f64]> = unwrapped.iter().map(|u| u.as_slice()).collect();
    let fit = crate::utils::multi_echo_linear_fit(
        &uw_refs, mag_slices, tes, mask,
        config.linear_fit_params.estimate_offset,
        config.linear_fit_params.reliability_threshold_percentile,
    );

    progress(4, 4);
    Ok(FieldMappingResult {
        b0_field_ppm: rads_to_ppm(&fit.field, metadata.field_strength),
        phase_offset: None,
    })
}

/// Path C: Single echo unwrap
#[allow(clippy::too_many_arguments)]
fn field_mapping_single_echo(
    phase: &[f64],
    mag: &[f64],
    mask: &[u8],
    metadata: &ScanMetadata,
    config: &FieldMappingConfig,
    _n_voxels: usize,
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<FieldMappingResult, PipelineError> {
    progress(1, 4);
    let unwrapped = unwrap_single(
        phase, mag, mask, config, nx, ny, nz, vsx, vsy, vsz,
        None, metadata.echo_times[0], 0.0,
    );

    // field = unwrapped / TE → rad/s
    let te = metadata.echo_times[0];
    let field_rads: Vec<f64> = unwrapped.iter().map(|&v| v / te).collect();

    progress(4, 4);
    Ok(FieldMappingResult {
        b0_field_ppm: rads_to_ppm(&field_rads, metadata.field_strength),
        phase_offset: None,
    })
}

/// Unwrap a single echo using the configured algorithm.
#[allow(clippy::too_many_arguments)]
fn unwrap_single(
    phase: &[f64],
    mag: &[f64],
    mask: &[u8],
    config: &FieldMappingConfig,
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    phase2: Option<&[f64]>,
    te1: f64, te2: f64,
) -> Vec<f64> {
    match config.unwrapping_algorithm {
        UnwrappingAlgorithm::Laplacian => {
            crate::unwrap::laplacian_unwrap(phase, mask, nx, ny, nz, vsx, vsy, vsz)
        }
        UnwrappingAlgorithm::Romeo => {
            crate::unwrap::unwrap_romeo(
                phase, mag, phase2, te1, te2,
                mask, &config.romeo_params, nx, ny, nz,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_test_metadata(n_echoes: usize) -> ScanMetadata {
        let tes: Vec<f64> = (0..n_echoes).map(|i| 0.005 + 0.005 * i as f64).collect();
        ScanMetadata {
            dims: (8, 8, 8),
            voxel_size: (1.0, 1.0, 1.0),
            echo_times: tes,
            field_strength: 3.0,
            b0_direction: (0.0, 0.0, 1.0),
        }
    }

    #[test]
    fn test_single_echo_recovers_frequency() {
        // Use ROMEO (not Laplacian, which removes constant component)
        // Uniform phase across all voxels → uniform B0
        // phase(rad) = 2π * f * TE  →  f = phase / (2π * TE)
        let meta = make_test_metadata(1);
        let te = meta.echo_times[0]; // 0.005 s
        let n = 8 * 8 * 8;
        let mask = vec![1u8; n];

        let freq_hz = 50.0;
        let phase_val = 2.0 * PI * freq_hz * te; // ~1.57 rad (no wrapping)
        let phase = vec![phase_val; n];

        let config = FieldMappingConfig {
            unwrapping_algorithm: UnwrappingAlgorithm::Romeo,
            ..Default::default()
        };

        let phase_s: &[f64] = &phase;
        let result = run_field_mapping(
            &[phase_s], None, &mask, &meta, &config, &mut |_, _| {},
        ).unwrap();

        let gamma = 42.576e6;
        let expected_ppm = freq_hz * 1e6 / (gamma * 3.0);

        // ROMEO preserves the constant phase, so B0 should be recovered
        for &v in &result.b0_field_ppm {
            assert!((v - expected_ppm).abs() < 0.01,
                "expected ~{:.4} ppm, got {:.4}", expected_ppm, v);
        }
    }

    #[test]
    fn test_multi_echo_direct_recovers_slope() {
        // 3 echoes with phase = slope * TE (no wrapping, small slope)
        // Use ROMEO so constant field is preserved
        let meta = make_test_metadata(3); // TEs: 0.005, 0.010, 0.015
        let n = 8 * 8 * 8;
        let mask = vec![1u8; n];
        let slope = 100.0; // rad/s → f ≈ 15.9 Hz

        let phases: Vec<Vec<f64>> = meta.echo_times.iter()
            .map(|&te| vec![slope * te; n])
            .collect();
        let phase_refs: Vec<&[f64]> = phases.iter().map(|p| p.as_slice()).collect();
        let mag = vec![1.0; n];
        let mag_refs: Vec<&[f64]> = (0..3).map(|_| mag.as_slice()).collect();

        let config = FieldMappingConfig {
            phase_offset_removal: false,
            unwrapping_algorithm: UnwrappingAlgorithm::Romeo,
            ..Default::default()
        };

        let result = run_field_mapping(
            &phase_refs, Some(&mag_refs), &mask, &meta, &config, &mut |_, _| {},
        ).unwrap();

        // Expected ppm: slope(rad/s) → ppm via rads_to_ppm
        let gamma = 42.576e6;
        let expected_ppm = slope * 1e6 / (2.0 * PI * gamma * 3.0);

        let masked_values: Vec<f64> = result.b0_field_ppm.iter()
            .zip(mask.iter())
            .filter(|(_, &m)| m > 0)
            .map(|(&v, _)| v)
            .collect();

        let mean: f64 = masked_values.iter().sum::<f64>() / masked_values.len() as f64;
        assert!((mean - expected_ppm).abs() < 0.001,
            "expected ~{:.6} ppm, got {:.6}", expected_ppm, mean);
    }

    #[test]
    fn test_multi_echo_with_offset_removal() {
        // Verify the offset-removal path produces output and has phase_offset
        let meta = make_test_metadata(3);
        let n = 8 * 8 * 8;
        let mask = vec![1u8; n];

        let phases: Vec<Vec<f64>> = meta.echo_times.iter()
            .map(|&te| vec![50.0 * te; n]) // small linear phase
            .collect();
        let phase_refs: Vec<&[f64]> = phases.iter().map(|p| p.as_slice()).collect();
        let mag = vec![1.0; n];
        let mag_refs: Vec<&[f64]> = (0..3).map(|_| mag.as_slice()).collect();

        let config = FieldMappingConfig {
            phase_offset_removal: true,
            unwrapping_algorithm: UnwrappingAlgorithm::Romeo,
            b0_estimation: B0EstimationMethod::WeightedAvg,
            ..Default::default()
        };

        let result = run_field_mapping(
            &phase_refs, Some(&mag_refs), &mask, &meta, &config, &mut |_, _| {},
        ).unwrap();

        assert_eq!(result.b0_field_ppm.len(), n);
        assert!(result.phase_offset.is_some(), "offset removal path should return phase offset");
        let offset = result.phase_offset.unwrap();
        assert_eq!(offset.len(), n);

        // All values should be finite
        for &v in &result.b0_field_ppm {
            assert!(v.is_finite(), "B0 field should be finite");
        }
    }

    #[test]
    fn test_validates_echo_time_mismatch() {
        let meta = make_test_metadata(2);
        let n = 8 * 8 * 8;
        let mask = vec![1u8; n];
        let phase = vec![0.0; n];

        let result = run_field_mapping(
            &[&phase[..]], None, &mask, &meta,
            &FieldMappingConfig::default(), &mut |_, _| {},
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_validates_empty_phases() {
        let meta = ScanMetadata {
            dims: (4, 4, 4),
            voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![],
            field_strength: 3.0,
            b0_direction: (0.0, 0.0, 1.0),
        };
        let mask = vec![1u8; 64];
        let result = run_field_mapping(
            &[], None, &mask, &meta,
            &FieldMappingConfig::default(), &mut |_, _| {},
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_b0_estimation_methods_agree_on_linear_data() {
        // For perfectly linear phase data (no offset), WeightedAvg and LinearFit
        // should produce the same B0 estimate. Use ROMEO to preserve constant fields.
        let meta = make_test_metadata(3);
        let n = 8 * 8 * 8;
        let mask = vec![1u8; n];
        let slope = 80.0; // rad/s

        let phases: Vec<Vec<f64>> = meta.echo_times.iter()
            .map(|&te| vec![slope * te; n])
            .collect();
        let phase_refs: Vec<&[f64]> = phases.iter().map(|p| p.as_slice()).collect();
        let mag = vec![1.0; n];
        let mag_refs: Vec<&[f64]> = (0..3).map(|_| mag.as_slice()).collect();

        // Path A: offset removal + weighted avg (uses ROMEO)
        let config_a = FieldMappingConfig {
            phase_offset_removal: true,
            b0_estimation: B0EstimationMethod::WeightedAvg,
            unwrapping_algorithm: UnwrappingAlgorithm::Romeo,
            ..Default::default()
        };
        let result_a = run_field_mapping(
            &phase_refs, Some(&mag_refs), &mask, &meta, &config_a, &mut |_, _| {},
        ).unwrap();

        // Path B: no offset removal → linear fit (uses ROMEO)
        let config_b = FieldMappingConfig {
            phase_offset_removal: false,
            unwrapping_algorithm: UnwrappingAlgorithm::Romeo,
            ..Default::default()
        };
        let result_b = run_field_mapping(
            &phase_refs, Some(&mag_refs), &mask, &meta, &config_b, &mut |_, _| {},
        ).unwrap();

        // Both should recover ~same ppm for perfectly linear data
        let count_a = result_a.b0_field_ppm.iter().filter(|v| v.is_finite() && **v != 0.0).count();
        let count_b = result_b.b0_field_ppm.iter().filter(|v| v.is_finite() && **v != 0.0).count();
        assert!(count_a > 0, "Path A should have non-zero voxels");
        assert!(count_b > 0, "Path B should have non-zero voxels");

        let mean_a: f64 = result_a.b0_field_ppm.iter()
            .filter(|v| v.is_finite() && **v != 0.0).sum::<f64>() / count_a as f64;
        let mean_b: f64 = result_b.b0_field_ppm.iter()
            .filter(|v| v.is_finite() && **v != 0.0).sum::<f64>() / count_b as f64;

        assert!((mean_a - mean_b).abs() < 0.05,
            "WeightedAvg ({:.4}) and LinearFit ({:.4}) should agree on linear data", mean_a, mean_b);
    }
}

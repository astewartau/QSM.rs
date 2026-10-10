//! Field mapping stage
//!
//! Multi-echo phase → B0 field map in ppm.
//! Implements the canonical field mapping pipeline:
//! phase offset removal → bipolar correction → unwrapping → B0 estimation.

use super::config::*;
use super::phase_utils::{hz_to_ppm, rads_to_ppm};
use crate::utils::B0WeightType;

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
        if let Some(mags) = magnitudes {
            if i < mags.len() && mags[i].len() != n_voxels {
                return Err(PipelineError::DimensionMismatch {
                    expected: n_voxels,
                    got: mags[i].len(),
                });
            }
        }
    }

    if let B0WeightType::AssumedDecay { t2star_s } = config.b0_weight_type {
        if !(t2star_s.is_finite() && t2star_s > 0.0) {
            return Err(PipelineError::InvalidConfig(format!(
                "assumed T2* for the assumed-decay B0 weighting must be positive, got {} s", t2star_s
            )));
        }
    }

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
        // ---- Path B: Direct per-echo unwrapping + configured B0 estimation ----
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
    _n_voxels: usize,
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<FieldMappingResult, PipelineError> {
    let tes = &metadata.echo_times;
    let n_echoes = phases.len();

    // Step 1: Phase offset removal
    progress(1, 4);
    let grid = crate::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let (mut corrected, phase_offset) = crate::utils::phase_offset_removal(
        phases, mag_slices, tes, mask,
        config.phase_offset_sigma, [0, 1],
        crate::unwrap::UnwrapMethod::Romeo,
        &grid,
    );

    // Step 2: Bipolar correction (optional, >= 3 echoes)
    if config.bipolar_correction && n_echoes >= 3 {
        crate::utils::bipolar_correction(
            &mut corrected, mag_slices, tes, mask,
            config.phase_offset_sigma, &grid,
        );
    }

    // Step 3: Multi-echo unwrapping
    progress(2, 4);
    let unwrapped: Vec<Vec<f64>> = match config.unwrapping_algorithm {
        UnwrappingAlgorithm::Laplacian => {
            // Neumann, not the ROI-masked variant: this stage wants unwrapping only.
            // Background removal is a later stage, and the masked variant would remove it
            // here first — measurably worse than doing it once, properly.
            corrected.iter()
                .map(|p| crate::unwrap::laplacian_unwrap(p, mask, &grid))
                .collect()
        }
        UnwrappingAlgorithm::Romeo => {
            crate::unwrap::unwrap_romeo_multi_echo(
                &corrected, mag_slices, tes, mask,
                &config.romeo_params, &grid,
            )
        }
    };

    // Step 4: B0 estimation
    progress(3, 4);
    let b0_hz = match config.b0_estimation {
        B0EstimationMethod::WeightedAvg => {
            crate::utils::calculate_b0_weighted(
                &unwrapped, mag_slices, tes, mask,
                config.b0_weight_type, &grid,
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

/// Path B: Multi-echo without phase offset removal (per-echo unwrap + B0 estimation).
///
/// Taken whenever offset removal is off: always with Laplacian unwrapping, and with ROMEO when
/// `phase_offset_removal` is false. Each echo is unwrapped on its own (for Laplacian, with the
/// phase zeroed outside `mask` first), and the echoes are then combined with
/// `config.b0_estimation`, as on the offset-removal path:
///
/// * [`B0EstimationMethod::WeightedAvg`] (the default): each echo is unwrapped only up to a
///   constant of its own, which a weighted mean of `φ/TE` would turn into a field offset (one
///   that varies over the brain for magnitude-dependent weights). So that constant is aligned
///   across echoes first, by the least that the unwrapper leaves undetermined:
///   - Laplacian fixes each echo only up to an arbitrary constant, so the masked mean is
///     subtracted from every echo. Echo `e` is `TEₑ·ω + cₑ` inside the mask; demeaned it is
///     `TEₑ·(ω − ω̄)` for every echo, which loses only a global constant field `ω̄` (removed by
///     background removal and referencing anyway).
///   - ROMEO fixes each echo up to a whole number of turns, so each echo's median wrap count
///     relative to the TE-scaled previous echo is removed
///     ([`correct_multi_echo_wraps`](crate::unwrap::correct_multi_echo_wraps)); this is a no-op
///     when the echoes are already consistent.
///
///   Then `B0 = Σ w (φ/TE) / Σ w` with `config.b0_weight_type`.
/// * [`B0EstimationMethod::LinearFit`]: the magnitude-weighted fit of phase against TE with
///   `config.linear_fit_params`, on the per-echo unwrapped phases as they come out of the
///   unwrapper. This is what this path computed unconditionally before it honoured
///   `b0_estimation` (bit-identical for ROMEO; for Laplacian it now sees the masked phase).
///   With an intercept and two echoes the fit is the echo difference `(φ₂ − φ₁)/(TE₂ − TE₁)`,
///   which discards the absolute phase, so it is only worth choosing when the echoes carry a
///   phase offset that has not been removed.
#[allow(clippy::too_many_arguments)]
fn field_mapping_direct(
    phases: &[&[f64]],
    mag_slices: &[&[f64]],
    mask: &[u8],
    metadata: &ScanMetadata,
    config: &FieldMappingConfig,
    _n_voxels: usize,
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<FieldMappingResult, PipelineError> {
    let tes = &metadata.echo_times;
    let n_echoes = phases.len();

    // Per-echo unwrapping. The Laplacian unwrap is a global Poisson solve, so phase noise
    // outside the brain leaks into it; zero the phase outside the mask first (as STI Suite's
    // `MRPhaseUnwrap(mask .* phase)` is called in the UK Biobank pipeline).
    progress(1, 4);
    let is_laplacian = config.unwrapping_algorithm == UnwrappingAlgorithm::Laplacian;
    let mut unwrapped: Vec<Vec<f64>> = Vec::with_capacity(n_echoes);
    for e in 0..n_echoes {
        let masked;
        let phase: &[f64] = if is_laplacian {
            masked = mask_phase(phases[e], mask);
            &masked
        } else {
            phases[e]
        };
        let uw = unwrap_single(
            phase, mag_slices.first().copied().unwrap_or(&[]),
            mask, config, nx, ny, nz, vsx, vsy, vsz,
            if e + 1 < n_echoes { Some(phases[e + 1]) } else { None },
            tes[e],
            if e + 1 < n_echoes { tes[e + 1] } else { 0.0 },
        );
        unwrapped.push(uw);
    }

    progress(3, 4);
    let b0_field_ppm = match config.b0_estimation {
        B0EstimationMethod::WeightedAvg => {
            align_per_echo_constants(&mut unwrapped, tes, mask, config.unwrapping_algorithm);
            let grid = crate::Grid::new(nx, ny, nz, vsx, vsy, vsz);
            let b0_hz = crate::utils::calculate_b0_weighted(
                &unwrapped, mag_slices, tes, mask,
                config.b0_weight_type, &grid,
            );
            hz_to_ppm(&b0_hz, metadata.field_strength)
        }
        B0EstimationMethod::LinearFit => {
            let uw_refs: Vec<&[f64]> = unwrapped.iter().map(|u| u.as_slice()).collect();
            let fit = crate::utils::multi_echo_linear_fit(
                &uw_refs, mag_slices, tes, mask,
                config.linear_fit_params.estimate_offset,
                config.linear_fit_params.reliability_threshold_percentile,
            );
            rads_to_ppm(&fit.field, metadata.field_strength)
        }
    };

    progress(4, 4);
    Ok(FieldMappingResult {
        b0_field_ppm,
        phase_offset: None,
    })
}

/// `phase` with every voxel outside `mask` set to zero.
fn mask_phase(phase: &[f64], mask: &[u8]) -> Vec<f64> {
    phase.iter().zip(mask).map(|(&p, &m)| if m != 0 { p } else { 0.0 }).collect()
}

/// Remove, from each separately unwrapped echo, the constant its unwrapper leaves undetermined,
/// so that the echoes agree before they are averaged. See [`field_mapping_direct`].
fn align_per_echo_constants(
    unwrapped: &mut [Vec<f64>],
    tes: &[f64],
    mask: &[u8],
    algorithm: UnwrappingAlgorithm,
) {
    match algorithm {
        UnwrappingAlgorithm::Laplacian => {
            for u in unwrapped.iter_mut() {
                remove_masked_mean(u, mask);
            }
        }
        UnwrappingAlgorithm::Romeo => {
            crate::unwrap::correct_multi_echo_wraps(unwrapped, tes, mask);
        }
    }
}

/// Subtract the mean over `mask` from `values` inside `mask` (outside is left untouched).
fn remove_masked_mean(values: &mut [f64], mask: &[u8]) {
    let (sum, n) = values.iter().zip(mask)
        .filter(|(_, &m)| m != 0)
        .fold((0.0, 0usize), |(s, n), (&v, _)| (s + v, n + 1));
    if n == 0 {
        return;
    }
    let mean = sum / n as f64;
    for (v, &m) in values.iter_mut().zip(mask) {
        if m != 0 {
            *v -= mean;
        }
    }
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
    let grid = crate::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    match config.unwrapping_algorithm {
        UnwrappingAlgorithm::Laplacian => {
            crate::unwrap::laplacian_unwrap(phase, mask, &grid)
        }
        UnwrappingAlgorithm::Romeo => {
            crate::unwrap::unwrap_romeo(
                phase, mag, phase2, te1, te2,
                mask, &config.romeo_params, &grid,
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
            slice_geometry: None,
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
            slice_geometry: None,
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

        // Path B: no offset removal, explicit linear fit (uses ROMEO)
        let config_b = FieldMappingConfig {
            phase_offset_removal: false,
            b0_estimation: B0EstimationMethod::LinearFit,
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

    // ---- Path B honours b0_estimation / b0_weight_type ------------------------------------

    const N: usize = 16;

    fn meta_n(tes: &[f64]) -> ScanMetadata {
        ScanMetadata {
            dims: (N, N, N),
            voxel_size: (1.0, 1.0, 1.0),
            echo_times: tes.to_vec(),
            field_strength: 3.0,
            b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: None,
        }
    }

    fn wrap(x: f64) -> f64 { (x + PI).rem_euclid(2.0 * PI) - PI }

    /// Deterministic noise in [-0.5, 0.5).
    fn noise(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    /// Per-echo wrapped phase `wrap(TE·ω + noise)` and a magnitude that varies over the volume
    /// and decays with TE, so magnitude-dependent weights differ from voxel to voxel.
    fn echoes(tes: &[f64], omega: impl Fn(usize, usize, usize) -> f64, noise_amp: f64)
        -> (Vec<Vec<f64>>, Vec<Vec<f64>>)
    {
        let n = N * N * N;
        let mut ph = vec![vec![0.0; n]; tes.len()];
        let mut mg = vec![vec![0.0; n]; tes.len()];
        let mut st = 42u64;
        for k in 0..N { for j in 0..N { for i in 0..N {
            let idx = i + j * N + k * N * N;
            for (e, &te) in tes.iter().enumerate() {
                ph[e][idx] = wrap(te * omega(i, j, k) + noise_amp * noise(&mut st));
                mg[e][idx] = (1.0 + 0.8 * (i as f64 / N as f64)) * (-te / 0.030).exp();
            }
        }}}
        (ph, mg)
    }

    /// Smooth field (rad/s) made of cosine modes, which the Neumann/DCT Laplacian unwrap
    /// reproduces exactly; it wraps several times at the later echoes.
    fn omega_cos(i: usize, j: usize, k: usize) -> f64 {
        let c = |x: usize, m: f64| (PI * m * (x as f64 + 0.5) / N as f64).cos();
        900.0 * c(i, 1.0) + 300.0 * c(j, 1.0) * c(k, 2.0)
    }

    /// Small field (rad/s) that never wraps at the echo times used with it.
    fn omega_small(i: usize, j: usize, _k: usize) -> f64 {
        60.0 + 40.0 * (i as f64 / N as f64) - 30.0 * (j as f64 / N as f64)
    }

    fn refs(v: &[Vec<f64>]) -> Vec<&[f64]> { v.iter().map(|x| x.as_slice()).collect() }

    fn max_abs_diff(a: &[f64], b: &[f64], mask: &[u8]) -> f64 {
        a.iter().zip(b).zip(mask).filter(|(_, &m)| m != 0)
            .map(|((x, y), _)| (x - y).abs()).fold(0.0, f64::max)
    }

    /// Hand-written weighted combination `Σ w (φ/TE) / Σ w` in ppm, with the weights spelled
    /// out here rather than taken from `B0WeightType::weight`.
    fn hand_combination(uw: &[Vec<f64>], mags: &[Vec<f64>], tes: &[f64], mask: &[u8], wt: B0WeightType, b0_t: f64) -> Vec<f64> {
        let gamma = 42.576e6;
        (0..mask.len()).map(|i| {
            if mask[i] == 0 { return 0.0; }
            let (mut num, mut den) = (0.0, 0.0);
            for e in 0..tes.len() {
                let (te, m) = (tes[e], mags[e][i]);
                let w = match wt {
                    B0WeightType::PhaseSNR => m * te,
                    B0WeightType::PhaseVar => m * m * te * te,
                    B0WeightType::Average => 1.0,
                    B0WeightType::TEs => te,
                    B0WeightType::Mag => m,
                    B0WeightType::AssumedDecay { t2star_s } => te * te * (-te / t2star_s).exp(),
                };
                num += w * uw[e][i] / te;
                den += w;
            }
            num / den / (2.0 * PI) * 1e6 / (gamma * b0_t)
        }).collect()
    }

    /// UK Biobank's form of the assumed-decay combination: `Σ Wφ / Σ W·TE`, `W = TE·exp(−TE/T2*)`.
    fn ukb_assumed_decay(uw: &[Vec<f64>], tes: &[f64], mask: &[u8], t2s: f64, b0_t: f64) -> Vec<f64> {
        let gamma = 42.576e6;
        let w: Vec<f64> = tes.iter().map(|&te| te * (-te / t2s).exp()).collect();
        let den: f64 = w.iter().zip(tes).map(|(w, te)| w * te).sum();
        (0..mask.len()).map(|i| {
            if mask[i] == 0 { return 0.0; }
            let num: f64 = (0..tes.len()).map(|e| w[e] * uw[e][i]).sum();
            num / den / (2.0 * PI) * 1e6 / (gamma * b0_t)
        }).collect()
    }

    fn all_weight_types() -> [B0WeightType; 6] {
        [
            B0WeightType::PhaseSNR, B0WeightType::PhaseVar, B0WeightType::Average,
            B0WeightType::TEs, B0WeightType::Mag, B0WeightType::assumed_decay_default(),
        ]
    }

    fn masked_mean(v: &[f64], mask: &[u8]) -> f64 {
        let (s, n) = v.iter().zip(mask).filter(|(_, &m)| m != 0)
            .fold((0.0, 0usize), |(s, n), (&x, _)| (s + x, n + 1));
        s / n as f64
    }

    #[test]
    fn laplacian_direct_honours_weight_type_and_matches_hand_combination() {
        let tes = [0.004, 0.009, 0.014];
        let meta = meta_n(&tes);
        let (ph, mg) = echoes(&tes, omega_cos, 0.4);
        let mask = vec![1u8; N * N * N];
        let grid = crate::Grid::new(N, N, N, 1.0, 1.0, 1.0);

        // Reference: unwrap each echo, remove its masked mean, combine by hand.
        let uw: Vec<Vec<f64>> = ph.iter().map(|p| {
            let mut u = crate::unwrap::laplacian_unwrap(p, &mask, &grid);
            let m = masked_mean(&u, &mask);
            u.iter_mut().for_each(|v| *v -= m);
            u
        }).collect();

        let mut outputs = Vec::new();
        for wt in all_weight_types() {
            let cfg = FieldMappingConfig {
                unwrapping_algorithm: UnwrappingAlgorithm::Laplacian,
                b0_weight_type: wt,
                ..Default::default()
            };
            let got = run_field_mapping(&refs(&ph), Some(&refs(&mg)), &mask, &meta, &cfg, &mut |_, _| {})
                .unwrap().b0_field_ppm;
            let want = hand_combination(&uw, &mg, &tes, &mask, wt, 3.0);
            let d = max_abs_diff(&got, &want, &mask);
            assert!(d < 1e-12, "{wt:?}: differs from the hand combination by {d}");
            outputs.push((wt, got));
        }
        // Every weight type gives a different field (they all used to be identical).
        for a in 0..outputs.len() {
            for b in a + 1..outputs.len() {
                let d = max_abs_diff(&outputs[a].1, &outputs[b].1, &mask);
                assert!(d > 1e-4, "{:?} and {:?} gave the same field (max diff {d})", outputs[a].0, outputs[b].0);
            }
        }
        // Assumed-decay weighting is UK Biobank's Σ Wφ / Σ W·TE.
        let ad = &outputs[5].1;
        let ukb = ukb_assumed_decay(&uw, &tes, &mask, 0.040, 3.0);
        assert!(max_abs_diff(ad, &ukb, &mask) < 1e-12);
    }

    #[test]
    fn laplacian_direct_recovers_wrapped_field() {
        // Noise-free: every weighting recovers the true field up to its masked mean.
        let tes = [0.004, 0.009, 0.014];
        let meta = meta_n(&tes);
        let (ph, mg) = echoes(&tes, omega_cos, 0.0);
        let mask = vec![1u8; N * N * N];
        let gamma = 42.576e6;
        let mut truth: Vec<f64> = (0..N * N * N)
            .map(|idx| omega_cos(idx % N, (idx / N) % N, idx / (N * N)) / (2.0 * PI) * 1e6 / (gamma * 3.0))
            .collect();
        let m = masked_mean(&truth, &mask);
        truth.iter_mut().for_each(|v| *v -= m);
        for wt in all_weight_types() {
            let cfg = FieldMappingConfig {
                unwrapping_algorithm: UnwrappingAlgorithm::Laplacian,
                b0_weight_type: wt,
                ..Default::default()
            };
            let got = run_field_mapping(&refs(&ph), Some(&refs(&mg)), &mask, &meta, &cfg, &mut |_, _| {})
                .unwrap().b0_field_ppm;
            let d = max_abs_diff(&got, &truth, &mask);
            assert!(d < 1e-6, "{wt:?}: max error {d} ppm");
        }
    }

    #[test]
    fn romeo_without_offset_removal_honours_weight_type() {
        // A field that does not wrap, so ROMEO returns each echo's phase unchanged and the
        // expected result is the hand combination of the input phases themselves.
        let tes = [0.004, 0.009, 0.014];
        let meta = meta_n(&tes);
        let (ph, mg) = echoes(&tes, omega_small, 0.3);
        let mask = vec![1u8; N * N * N];

        let mut outputs = Vec::new();
        for wt in all_weight_types() {
            let cfg = FieldMappingConfig {
                unwrapping_algorithm: UnwrappingAlgorithm::Romeo,
                phase_offset_removal: false,
                b0_weight_type: wt,
                ..Default::default()
            };
            let got = run_field_mapping(&refs(&ph), Some(&refs(&mg)), &mask, &meta, &cfg, &mut |_, _| {})
                .unwrap().b0_field_ppm;
            let want = hand_combination(&ph, &mg, &tes, &mask, wt, 3.0);
            let d = max_abs_diff(&got, &want, &mask);
            assert!(d < 1e-12, "{wt:?}: differs from the hand combination by {d}");
            outputs.push((wt, got));
        }
        for a in 0..outputs.len() {
            for b in a + 1..outputs.len() {
                let d = max_abs_diff(&outputs[a].1, &outputs[b].1, &mask);
                assert!(d > 1e-4, "{:?} and {:?} gave the same field (max diff {d})", outputs[a].0, outputs[b].0);
            }
        }
    }

    #[test]
    fn two_echo_weighted_mean_keeps_absolute_phase_linear_fit_is_echo_difference() {
        // The UK Biobank case: two offset-free echoes. The default (weighted mean) uses both
        // echoes' absolute phase; an explicit linear fit with an intercept is the echo difference.
        let tes = [0.00942, 0.0197];
        let meta = meta_n(&tes);
        let (ph, mg) = echoes(&tes, omega_small, 0.3);
        let mask = vec![1u8; N * N * N];
        let gamma = 42.576e6;
        let base = FieldMappingConfig {
            unwrapping_algorithm: UnwrappingAlgorithm::Romeo,
            phase_offset_removal: false,
            ..Default::default()
        };

        let wavg = run_field_mapping(&refs(&ph), Some(&refs(&mg)), &mask, &meta, &base, &mut |_, _| {})
            .unwrap().b0_field_ppm;
        let want = hand_combination(&ph, &mg, &tes, &mask, B0WeightType::PhaseSNR, 3.0);
        assert!(max_abs_diff(&wavg, &want, &mask) < 1e-12);

        let lf_cfg = FieldMappingConfig { b0_estimation: B0EstimationMethod::LinearFit, ..base.clone() };
        let lf = run_field_mapping(&refs(&ph), Some(&refs(&mg)), &mask, &meta, &lf_cfg, &mut |_, _| {})
            .unwrap().b0_field_ppm;
        let diff: Vec<f64> = (0..mask.len())
            .map(|i| (ph[1][i] - ph[0][i]) / (tes[1] - tes[0]) / (2.0 * PI) * 1e6 / (gamma * 3.0))
            .collect();
        assert!(max_abs_diff(&lf, &diff, &mask) < 1e-9, "linear fit is not the echo difference");
        assert!(max_abs_diff(&lf, &wavg, &mask) > 1e-3, "weighted mean should differ from the echo difference");

        // And the echo difference is the noisier of the two: compare residuals to the truth.
        let truth: Vec<f64> = (0..N * N * N)
            .map(|idx| omega_small(idx % N, (idx / N) % N, idx / (N * N)) / (2.0 * PI) * 1e6 / (gamma * 3.0))
            .collect();
        let rms = |v: &[f64]| (v.iter().zip(&truth).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        assert!(rms(&lf) > 1.5 * rms(&wavg), "rms lf {} vs wavg {}", rms(&lf), rms(&wavg));
    }

    /// Sphere mask filling most of the N³ grid.
    fn sphere_mask() -> Vec<u8> {
        let c = (N as f64 - 1.0) / 2.0;
        (0..N * N * N).map(|idx| {
            let (i, j, k) = (idx % N, (idx / N) % N, idx / (N * N));
            let r2 = (i as f64 - c).powi(2) + (j as f64 - c).powi(2) + (k as f64 - c).powi(2);
            (r2 <= (0.45 * N as f64).powi(2)) as u8
        }).collect()
    }

    #[test]
    fn direct_linear_fit_is_the_fit_on_the_unwrapped_echoes() {
        // `b0_estimation = LinearFit` on the direct path is the fit this path always ran:
        // per-echo unwrap (Laplacian: of the masked phase), then multi_echo_linear_fit with
        // an intercept. `b0_weight_type` does not enter it.
        let tes = [0.004, 0.009, 0.014];
        let meta = meta_n(&tes);
        let (ph, mg) = echoes(&tes, omega_cos, 0.4);
        let mask = sphere_mask();
        let grid = crate::Grid::new(N, N, N, 1.0, 1.0, 1.0);
        let cfg = FieldMappingConfig {
            unwrapping_algorithm: UnwrappingAlgorithm::Laplacian,
            b0_estimation: B0EstimationMethod::LinearFit,
            b0_weight_type: B0WeightType::assumed_decay_default(), // must be ignored by the fit
            ..Default::default()
        };
        let got = run_field_mapping(&refs(&ph), Some(&refs(&mg)), &mask, &meta, &cfg, &mut |_, _| {})
            .unwrap().b0_field_ppm;
        let uw: Vec<Vec<f64>> = ph.iter()
            .map(|p| crate::unwrap::laplacian_unwrap(&mask_phase(p, &mask), &mask, &grid))
            .collect();
        let fit = crate::utils::multi_echo_linear_fit(
            &refs(&uw), &refs(&mg), &tes, &mask,
            cfg.linear_fit_params.estimate_offset,
            cfg.linear_fit_params.reliability_threshold_percentile,
        );
        assert_eq!(got, rads_to_ppm(&fit.field, 3.0));
    }

    #[test]
    fn laplacian_direct_ignores_phase_outside_the_mask() {
        // The phase is zeroed outside the mask before the (global) Laplacian solve, so what is
        // there cannot leak into the brain, and the result is the hand combination of the
        // masked, unwrapped, demeaned echoes.
        let tes = [0.004, 0.009, 0.014];
        let meta = meta_n(&tes);
        let (ph, mg) = echoes(&tes, omega_cos, 0.4);
        let mask = sphere_mask();
        assert!(mask.contains(&0) && mask.contains(&1));
        let grid = crate::Grid::new(N, N, N, 1.0, 1.0, 1.0);
        let cfg = FieldMappingConfig {
            unwrapping_algorithm: UnwrappingAlgorithm::Laplacian,
            ..Default::default()
        };
        let run = |p: &[Vec<f64>]| run_field_mapping(&refs(p), Some(&refs(&mg)), &mask, &meta, &cfg, &mut |_, _| {})
            .unwrap().b0_field_ppm;
        let got = run(&ph);

        let mut st = 7u64;
        let scrambled: Vec<Vec<f64>> = ph.iter().map(|p| p.iter().zip(&mask)
            .map(|(&v, &m)| if m != 0 { v } else { wrap(10.0 * noise(&mut st)) }).collect()).collect();
        assert_eq!(got, run(&scrambled));

        let uw: Vec<Vec<f64>> = ph.iter().map(|p| {
            let mut u = crate::unwrap::laplacian_unwrap(&mask_phase(p, &mask), &mask, &grid);
            remove_masked_mean(&mut u, &mask);
            u
        }).collect();
        let want = hand_combination(&uw, &mg, &tes, &mask, B0WeightType::PhaseSNR, 3.0);
        assert!(max_abs_diff(&got, &want, &mask) < 1e-12);
    }

    #[test]
    fn offset_removal_path_accepts_assumed_decay_weighting() {
        // Path A: assumed-decay weighting changes the output, and with T2* → ∞ it is phase-variance
        // weighting at unit magnitude (TE²·exp(−TE/∞) = TE² = mag²·TE²).
        let tes = [0.004, 0.009, 0.014];
        let meta = meta_n(&tes);
        let (ph, _) = echoes(&tes, omega_small, 0.3);
        let mask = vec![1u8; N * N * N];
        let run = |wt| {
            let cfg = FieldMappingConfig { b0_weight_type: wt, ..Default::default() };
            run_field_mapping(&refs(&ph), None, &mask, &meta, &cfg, &mut |_, _| {}).unwrap()
        };
        let snr = run(B0WeightType::PhaseSNR);
        let ad = run(B0WeightType::assumed_decay_default());
        let inf = run(B0WeightType::AssumedDecay { t2star_s: 1e30 });
        let pv = run(B0WeightType::PhaseVar);
        assert!(ad.phase_offset.is_some());
        assert!(max_abs_diff(&snr.b0_field_ppm, &ad.b0_field_ppm, &mask) > 1e-4);
        assert!(max_abs_diff(&inf.b0_field_ppm, &pv.b0_field_ppm, &mask) < 1e-12);
    }

    #[test]
    fn rejects_non_positive_assumed_t2star() {
        let tes = [0.004, 0.009];
        let meta = meta_n(&tes);
        let (ph, _) = echoes(&tes, omega_small, 0.0);
        let mask = vec![1u8; N * N * N];
        for t2 in [0.0, -0.01, f64::NAN] {
            let cfg = FieldMappingConfig {
                unwrapping_algorithm: UnwrappingAlgorithm::Laplacian,
                b0_weight_type: B0WeightType::AssumedDecay { t2star_s: t2 },
                ..Default::default()
            };
            assert!(run_field_mapping(&refs(&ph), None, &mask, &meta, &cfg, &mut |_, _| {}).is_err());
        }
    }
}

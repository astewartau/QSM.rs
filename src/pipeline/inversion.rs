//! Dipole inversion stage
//!
//! Dispatcher for standard dipole inversion algorithms.
//! Also contains TGV and QSMART pipeline runners which combine
//! multiple stages internally.
//!
//! MEDI Hz↔radians conversion is handled internally — the caller
//! passes ppm fields and receives ppm results.

use std::f64::consts::PI;

use super::config::*;

/// Run standard dipole inversion on a local field.
///
/// Handles MEDI's unit conversion (ppm → radians) internally.
///
/// # Arguments
/// * `local_field_ppm` - Local field in ppm (after background removal)
/// * `mask` - Eroded binary mask from background removal
/// * `metadata` - Scan metadata
/// * `config` - Inversion configuration
/// * `magnitude` - Combined magnitude image (needed for MEDI edge weighting)
/// * `progress` - Progress callback (current_iter, max_iter)
///
/// # Returns
/// Susceptibility map in ppm (unreferenced)
pub fn run_dipole_inversion(
    local_field_ppm: &[f64],
    mask: &[u8],
    metadata: &ScanMetadata,
    config: &InversionConfig,
    magnitude: Option<&[f64]>,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Vec<f64>, PipelineError> {
    let grid = metadata.grid();
    let bdir = metadata.b0_direction;
    let n_voxels = grid.n_total();

    if local_field_ppm.len() != n_voxels {
        return Err(PipelineError::DimensionMismatch {
            expected: n_voxels,
            got: local_field_ppm.len(),
        });
    }

    let chi = match config.algorithm {
        InversionAlgorithm::Tkd => {
            crate::inversion::tkd(
                local_field_ppm, mask, &grid, bdir, &config.tkd,
            )
        }
        InversionAlgorithm::Tsvd => {
            crate::inversion::tsvd(
                local_field_ppm, mask, &grid, bdir, &config.tsvd,
            )
        }
        InversionAlgorithm::Tikhonov => {
            crate::inversion::tikhonov(
                local_field_ppm, mask, &grid, bdir, &config.tikhonov,
            )
        }
        InversionAlgorithm::Tv => {
            crate::inversion::tv_admm(
                local_field_ppm, mask, &grid, bdir, &config.tv, progress,
            )
        }
        InversionAlgorithm::Rts => {
            crate::inversion::rts(
                local_field_ppm, mask, &grid, bdir, &config.rts, progress,
            )
        }
        InversionAlgorithm::Nltv => {
            crate::inversion::nltv(
                local_field_ppm, mask, &grid, bdir, &config.nltv, progress,
            )
        }
        InversionAlgorithm::Medi => {
            // MEDI requires field in radians, not ppm
            let gamma_hz = 42.576e6;
            let te1 = metadata.echo_times.first().copied().unwrap_or(0.005);
            let ppm_to_rad = 2.0 * PI * gamma_hz * metadata.field_strength * te1 * 1e-6;

            let local_field_rad: Vec<f64> = local_field_ppm.iter()
                .map(|&v| v * ppm_to_rad)
                .collect();

            let uniform_mag = vec![1.0f64; n_voxels];
            let mag = magnitude.unwrap_or(&uniform_mag);
            let n_std = vec![1.0f64; n_voxels];

            let chi_rad = crate::inversion::medi(
                &local_field_rad, &n_std, mag, mask,
                &grid, bdir, &config.medi, progress,
            );

            let rad_to_ppm = 1.0 / ppm_to_rad;
            chi_rad.iter().map(|&v| v * rad_to_ppm).collect()
        }
        InversionAlgorithm::Ilsqr => {
            let (chi, _, _, _) = crate::inversion::ilsqr(
                local_field_ppm, mask, &grid, bdir, &config.ilsqr, &mut *progress,
            );
            chi
        }
        InversionAlgorithm::Tgv | InversionAlgorithm::Qsmart => {
            return Err(PipelineError::InvalidConfig(
                format!("{:?} should use run_tgv or run_qsmart", config.algorithm),
            ));
        }
    };

    Ok(chi)
}

/// Run TGV single-step QSM reconstruction.
///
/// For multi-echo data, runs field mapping first to get B0, then converts
/// to phase at TE1. For single-echo, uses wrapped phase directly.
/// TGV internally handles unwrapping + background removal + inversion.
///
/// # Returns
/// Susceptibility map in ppm (unreferenced — call `apply_reference` after)
pub fn run_tgv(
    phases: &[&[f64]],
    magnitudes: Option<&[&[f64]]>,
    mask: &[u8],
    metadata: &ScanMetadata,
    field_mapping_config: &FieldMappingConfig,
    tgv_params: &crate::inversion::TgvParams,
    reference: QsmReference,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Vec<f64>, PipelineError> {
    let grid = metadata.grid();
    let (bx, by, bz) = metadata.b0_direction;

    // For multi-echo: compute B0 field map first, then convert to phase
    // For single-echo: use wrapped phase directly
    let phase_data: Vec<f64> = if phases.len() > 1 {
        let field_result = super::field_mapping::run_field_mapping(
            phases, magnitudes, mask, metadata,
            field_mapping_config, &mut |_, _| {},
        )?;
        let gamma_hz = 42.576e6;
        let te1 = metadata.echo_times[0];
        let ppm_to_rad = 2.0 * PI * gamma_hz * metadata.field_strength * te1 * 1e-6;
        field_result.b0_field_ppm.iter().map(|&v| v * ppm_to_rad).collect()
    } else {
        phases[0].to_vec()
    };

    let chi_ppm = crate::inversion::tgv_qsm(
        &phase_data, mask, &grid, tgv_params, (bx, by, bz), &mut *progress,
    );

    Ok(super::referencing::apply_reference(&chi_ppm, mask, reference))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inversion_tkd() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let field = vec![0.01; n];
        let mask = vec![1u8; n];
        let meta = ScanMetadata {
            dims: (nx, ny, nz),
            voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005],
            field_strength: 3.0,
            b0_direction: (0.0, 0.0, 1.0),
        };
        let config = InversionConfig {
            algorithm: InversionAlgorithm::Tkd,
            ..Default::default()
        };

        let result = run_dipole_inversion(
            &field, &mask, &meta, &config, None, &mut |_, _| {},
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), n);
    }

    fn make_inversion_test(alg: InversionAlgorithm) -> Vec<f64> {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let field = vec![0.01; n];
        let mask = vec![1u8; n];
        let meta = ScanMetadata {
            dims: (nx, ny, nz), voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005], field_strength: 3.0, b0_direction: (0.0, 0.0, 1.0),
        };
        let config = InversionConfig { algorithm: alg, ..Default::default() };
        run_dipole_inversion(&field, &mask, &meta, &config, None, &mut |_, _| {}).unwrap()
    }

    #[test]
    fn test_inversion_tsvd() {
        let chi = make_inversion_test(InversionAlgorithm::Tsvd);
        assert_eq!(chi.len(), 8 * 8 * 8);
    }

    #[test]
    fn test_inversion_tikhonov() {
        let chi = make_inversion_test(InversionAlgorithm::Tikhonov);
        assert_eq!(chi.len(), 8 * 8 * 8);
    }

    #[test]
    fn test_inversion_tv() {
        let chi = make_inversion_test(InversionAlgorithm::Tv);
        assert_eq!(chi.len(), 8 * 8 * 8);
    }

    #[test]
    fn test_inversion_rts() {
        let chi = make_inversion_test(InversionAlgorithm::Rts);
        assert_eq!(chi.len(), 8 * 8 * 8);
    }

    #[test]
    fn test_inversion_nltv() {
        let chi = make_inversion_test(InversionAlgorithm::Nltv);
        assert_eq!(chi.len(), 8 * 8 * 8);
    }

    #[test]
    fn test_inversion_ilsqr() {
        let chi = make_inversion_test(InversionAlgorithm::Ilsqr);
        assert_eq!(chi.len(), 8 * 8 * 8);
    }

    #[test]
    fn test_inversion_medi() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let field = vec![0.01; n];
        let mask = vec![1u8; n];
        let mag = vec![1.0; n];
        let meta = ScanMetadata {
            dims: (nx, ny, nz), voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005], field_strength: 3.0, b0_direction: (0.0, 0.0, 1.0),
        };
        let config = InversionConfig { algorithm: InversionAlgorithm::Medi, ..Default::default() };
        let chi = run_dipole_inversion(&field, &mask, &meta, &config, Some(&mag), &mut |_, _| {}).unwrap();
        assert_eq!(chi.len(), n);
    }

    #[test]
    fn test_inversion_rejects_tgv() {
        let n = 64;
        let meta = ScanMetadata {
            dims: (4, 4, 4),
            voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005],
            field_strength: 3.0,
            b0_direction: (0.0, 0.0, 1.0),
        };
        let config = InversionConfig {
            algorithm: InversionAlgorithm::Tgv,
            ..Default::default()
        };

        let result = run_dipole_inversion(
            &vec![0.0; n], &vec![1u8; n], &meta, &config, None, &mut |_, _| {},
        );
        assert!(result.is_err());
    }
}

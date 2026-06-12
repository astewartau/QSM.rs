//! QSMART two-stage reconstruction
//!
//! Two-stage SDF + iLSQR pipeline with vasculature detection.
//! Stage 1: whole-ROI reconstruction. Stage 2: tissue-only reconstruction
//! with vasculature excluded. Final offset adjustment combines both.

use super::config::*;

/// Run QSMART two-stage QSM reconstruction.
///
/// # Arguments
/// * `field_ppm` - Total field in ppm (after field mapping)
/// * `mask` - Binary brain mask
/// * `magnitude` - Combined magnitude (for vasculature detection and, when the inner
///   inversion is MEDI, edge weighting; uniform if None)
/// * `metadata` - Scan metadata
/// * `inversion_config` - Inversion configuration. QSMART-specific settings (SDF,
///   vasculature, iLSQR tolerance) come from `inversion_config.qsmart`; the inner
///   dipole inversion algorithm is selected by `inversion_config.qsmart.inversion`
///   and tuned via the matching per-algorithm field on this config.
/// * `reference` - QSM referencing method
/// * `progress` - Progress callback (current_step, total_steps)
///
/// # Returns
/// Susceptibility map in ppm (referenced)
pub fn run_qsmart(
    field_ppm: &[f64],
    mask: &[u8],
    magnitude: Option<&[f64]>,
    metadata: &ScanMetadata,
    inversion_config: &InversionConfig,
    reference: QsmReference,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Vec<f64>, PipelineError> {
    let (nx, ny, nz) = metadata.dims;
    let bdir = metadata.b0_direction;
    let n_voxels = nx * ny * nz;
    let grid = metadata.grid();
    let qsmart_params = &inversion_config.qsmart;

    // QSMART runs a standard dipole inversion per stage; the two pipeline-level
    // algorithms can't be nested here.
    if matches!(
        qsmart_params.inversion,
        InversionAlgorithm::Qsmart | InversionAlgorithm::Tgv
    ) {
        return Err(PipelineError::InvalidConfig(
            "QSMART inversion cannot be Qsmart or Tgv".into(),
        ));
    }

    // Inner inversion config: same per-algorithm params as the caller, but with the
    // QSMART-selected algorithm and iLSQR params pinned to QSMART's own fields so the
    // default (iLSQR) path is identical to the previous hardcoded call.
    let mut inner_config = inversion_config.clone();
    inner_config.algorithm = qsmart_params.inversion;
    inner_config.ilsqr = crate::inversion::IlsqrParams {
        tol: qsmart_params.ilsqr_tol,
        max_iter: qsmart_params.ilsqr_max_iter,
    };

    // Step 1: Vasculature detection
    progress(1, 6);
    let uniform_mag = vec![1.0f64; n_voxels];
    let mag = magnitude.unwrap_or(&uniform_mag);
    let vasc_params = crate::utils::VasculatureParams {
        sphere_radius: qsmart_params.vasc_sphere_radius,
        frangi_scale_range: qsmart_params.frangi_scale_range,
        frangi_scale_ratio: qsmart_params.frangi_scale_ratio,
        frangi_c: qsmart_params.frangi_c,
    };
    let vasc_mask = crate::utils::generate_vasculature_mask(
        mag, mask, &grid, &vasc_params, |_, _| {},
    );

    let mask_f64: Vec<f64> = mask.iter().map(|&m| m as f64).collect();
    let w1 = crate::utils::compute_weighted_mask_stage1(&mask_f64, &vasc_mask);
    let w2 = crate::utils::compute_weighted_mask_stage2(&mask_f64, &vasc_mask, &vasc_mask);

    // Step 2: SDF stage 1
    progress(2, 6);
    let sdf_params1 = crate::bgremove::SdfParams {
        sigma1: qsmart_params.sdf_sigma1_stage1,
        sigma2: qsmart_params.sdf_sigma2_stage1,
        spatial_radius: qsmart_params.sdf_spatial_radius,
        lower_lim: qsmart_params.sdf_lower_lim,
        curv_constant: qsmart_params.sdf_curv_constant,
        use_curvature: true,
    };
    let lfs1 = crate::bgremove::sdf::sdf(
        field_ppm, &w1, &vasc_mask, &grid, &sdf_params1, |_, _| {},
    );

    // Step 3: dipole inversion stage 1 (iLSQR by default)
    progress(3, 6);
    let chi1 = super::inversion::run_dipole_inversion(
        &lfs1, mask, metadata, &inner_config, magnitude, &mut |_, _| {},
    )?;

    // Step 4: SDF stage 2
    progress(4, 6);
    let sdf_params2 = crate::bgremove::SdfParams {
        sigma1: qsmart_params.sdf_sigma1_stage2,
        sigma2: qsmart_params.sdf_sigma2_stage2,
        spatial_radius: qsmart_params.sdf_spatial_radius,
        lower_lim: qsmart_params.sdf_lower_lim,
        curv_constant: qsmart_params.sdf_curv_constant,
        use_curvature: true,
    };
    let lfs2 = crate::bgremove::sdf::sdf(
        field_ppm, &w2, &vasc_mask, &grid, &sdf_params2, |_, _| {},
    );

    // Step 5: dipole inversion stage 2 (iLSQR by default)
    progress(5, 6);
    let chi2 = super::inversion::run_dipole_inversion(
        &lfs2, mask, metadata, &inner_config, magnitude, &mut |_, _| {},
    )?;

    // Step 6: Combine and reference
    progress(6, 6);
    let chi = crate::utils::adjust_offset(
        &vasc_mask, &lfs1, &chi1, &chi2,
        &grid, bdir, qsmart_params.ppm,
    );

    Ok(super::referencing::apply_reference(&chi, mask, reference))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_qsmart_basic() {
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
            algorithm: InversionAlgorithm::Qsmart,
            qsmart: crate::utils::QsmartParams::for_field_strength(3.0),
            ..Default::default()
        };

        let result = run_qsmart(
            &field, &mask, None, &meta, &config, QsmReference::Mean,
            &mut |_, _| {},
        );
        assert!(result.is_ok());
        let chi = result.unwrap();
        assert_eq!(chi.len(), n);
        for &v in &chi {
            assert!(v.is_finite(), "QSMART output must be finite");
        }
    }

    #[test]
    fn test_run_qsmart_swappable_inversion() {
        // QSMART with a non-default inner inversion (TKD) should run end to end.
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
        let mut qsmart = crate::utils::QsmartParams::for_field_strength(3.0);
        qsmart.inversion = InversionAlgorithm::Tkd;
        let config = InversionConfig {
            algorithm: InversionAlgorithm::Qsmart,
            qsmart,
            ..Default::default()
        };

        let chi = run_qsmart(
            &field, &mask, None, &meta, &config, QsmReference::Mean,
            &mut |_, _| {},
        )
        .unwrap();
        assert_eq!(chi.len(), n);
        for &v in &chi {
            assert!(v.is_finite(), "QSMART output must be finite");
        }
    }

    #[test]
    fn test_run_qsmart_rejects_nested_pipeline_inversion() {
        let (nx, ny, nz) = (4, 4, 4);
        let n = nx * ny * nz;
        let meta = ScanMetadata {
            dims: (nx, ny, nz),
            voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005],
            field_strength: 3.0,
            b0_direction: (0.0, 0.0, 1.0),
        };
        let mut qsmart = crate::utils::QsmartParams::for_field_strength(3.0);
        qsmart.inversion = InversionAlgorithm::Qsmart;
        let config = InversionConfig {
            algorithm: InversionAlgorithm::Qsmart,
            qsmart,
            ..Default::default()
        };

        let result = run_qsmart(
            &vec![0.0; n], &vec![1u8; n], None, &meta, &config,
            QsmReference::Mean, &mut |_, _| {},
        );
        assert!(result.is_err());
    }
}

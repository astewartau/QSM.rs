//! QSMART two-stage reconstruction
//!
//! Two-stage SDF + iLSQR pipeline with vasculature detection, after Yaghmaie et al. (2021)
//! and their MATLAB toolbox (<https://github.com/wtsyeda/QSMART>), which inverts each stage
//! with STI Suite's `QSM_iLSQR`.
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
///   vasculature, iLSQR iteration cap and padding) come from `inversion_config.qsmart`; the
///   inner dipole inversion algorithm is selected by `inversion_config.qsmart.inversion` and
///   tuned via the matching per-algorithm field on this config. With the default iLSQR, the
///   inversion is STI Suite's [`ilsqr`](fn@crate::inversion::ilsqr) with QSMART's arguments
///   (`qsmart.ilsqr_max_iter`, `qsmart.ilsqr_pad_mm`; the artefact tolerance comes from
///   `inversion_config.ilsqr`).
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
    // Refused up front rather than part-way through the two inversions below.
    metadata.require_contiguous_slices()?;
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

    // Inner inversion config: same per-algorithm params as the caller, with the
    // QSMART-selected algorithm.
    let mut inner_config = inversion_config.clone();
    inner_config.algorithm = qsmart_params.inversion;
    // The original QSMART (QSMART_toolbox_v1.0/QSMART.m) calls STI Suite's QSM_iLSQR for both
    // stages as `QSM_iLSQR(lfs, mask, 'H', z_prjs, 'voxelsize', res, 'niter', 50, 'TE', 1000,
    // 'B0', B0)`: no `'padsize'` (STI's default, 6 mm) and both LSQR solves capped at 50
    // iterations. TE/B0 only convert units (the field here is already in ppm).
    let ilsqr_params = crate::inversion::IlsqrParams {
        max_iter: qsmart_params.ilsqr_max_iter,
        ..inversion_config.ilsqr.clone()
    };
    let invert = |lfs: &[f64], m: &[u8]| -> Result<Vec<f64>, PipelineError> {
        if matches!(inner_config.algorithm, InversionAlgorithm::Ilsqr) {
            let (chi, _, _, _) = crate::inversion::ilsqr_with_padding(
                lfs, m, &grid, bdir, &ilsqr_params, [qsmart_params.ilsqr_pad_mm; 3], |_, _| {},
            );
            Ok(chi)
        } else {
            super::inversion::run_dipole_inversion(lfs, m, metadata, &inner_config, magnitude, &mut |_, _| {})
        }
    };

    // Step 1: Vasculature detection (vasc_mask: 1 = tissue, 0 = vessel)
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

    // No reliability map at this layer, so the QSMART weighting mask is just the brain mask.
    let weighted_mask: Vec<f64> = mask.iter().map(|&m| m as f64).collect();
    let ones_vasc = vec![1.0f64; n_voxels];

    // Step 2: SDF stage 1 — background removal over the whole ROI (vessels included).
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
        field_ppm, &weighted_mask, &ones_vasc, &grid, &sdf_params1, |_, _| {},
    );

    // Step 3: dipole inversion stage 1 over the whole ROI.
    progress(3, 6);
    let mask_stage1: Vec<u8> = weighted_mask.iter().map(|&v| if v > 0.1 { 1 } else { 0 }).collect();
    let chi1 = invert(&lfs1, &mask_stage1)?;

    // Step 4: SDF stage 2 — tissue-only, vessel-aware (mask-zeroed field input).
    progress(4, 6);
    let sdf_params2 = crate::bgremove::SdfParams {
        sigma1: qsmart_params.sdf_sigma1_stage2,
        sigma2: qsmart_params.sdf_sigma2_stage2,
        spatial_radius: qsmart_params.sdf_spatial_radius,
        lower_lim: qsmart_params.sdf_lower_lim,
        curv_constant: qsmart_params.sdf_curv_constant,
        use_curvature: true,
    };
    let field_weighted: Vec<f64> = field_ppm.iter()
        .zip(weighted_mask.iter())
        .map(|(&f, &m)| f * m)
        .collect();
    let lfs2 = crate::bgremove::sdf::sdf(
        &field_weighted, &weighted_mask, &vasc_mask, &grid, &sdf_params2, |_, _| {},
    );

    // Step 5: dipole inversion stage 2 over tissue only (in-mask AND not vessel).
    progress(5, 6);
    let mask_stage2: Vec<u8> = weighted_mask.iter()
        .zip(vasc_mask.iter())
        .map(|(&m, &v)| if m > 0.1 && v > 0.5 { 1 } else { 0 })
        .collect();
    let chi2 = invert(&lfs2, &mask_stage2)?;

    // Step 6: offset adjustment (combine stages) and reference.
    // removed_voxels = vessel regions (in mask but excluded from stage 2).
    // Everything is in ppm here, so adjust_offset's lfs rescale is the identity (ppm = 1.0).
    progress(6, 6);
    let removed_voxels: Vec<f64> = weighted_mask.iter()
        .zip(vasc_mask.iter())
        .map(|(&m, &v)| m - v)
        .collect();
    let mut chi = crate::utils::adjust_offset(
        &removed_voxels, &lfs1, &chi1, &chi2, &grid, bdir, 1.0,
    );
    // QSMART susceptibility is only defined inside the brain mask.
    for (c, &m) in chi.iter_mut().zip(mask.iter()) {
        if m == 0 {
            *c = 0.0;
        }
    }

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
            slice_geometry: None,
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
            slice_geometry: None,
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
            slice_geometry: None,
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

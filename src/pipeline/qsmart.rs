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
/// * `magnitude` - Combined magnitude (for vasculature detection; uniform if None)
/// * `metadata` - Scan metadata
/// * `qsmart_params` - QSMART parameters (SDF, iLSQR, vasculature settings)
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
    qsmart_params: &crate::utils::QsmartParams,
    reference: QsmReference,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Vec<f64>, PipelineError> {
    let (nx, ny, nz) = metadata.dims;
    let (vsx, vsy, vsz) = metadata.voxel_size;
    let bdir = metadata.b0_direction;
    let n_voxels = nx * ny * nz;

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
        mag, mask, nx, ny, nz, &vasc_params,
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
        field_ppm, &w1, &vasc_mask, nx, ny, nz, &sdf_params1,
    );

    // Step 3: iLSQR stage 1
    progress(3, 6);
    let (_, _, _, chi1) = crate::inversion::ilsqr(
        &lfs1, mask, nx, ny, nz, vsx, vsy, vsz, bdir,
        qsmart_params.ilsqr_tol, qsmart_params.ilsqr_max_iter,
    );

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
        field_ppm, &w2, &vasc_mask, nx, ny, nz, &sdf_params2,
    );

    // Step 5: iLSQR stage 2
    progress(5, 6);
    let (_, _, _, chi2) = crate::inversion::ilsqr(
        &lfs2, mask, nx, ny, nz, vsx, vsy, vsz, bdir,
        qsmart_params.ilsqr_tol, qsmart_params.ilsqr_max_iter,
    );

    // Step 6: Combine and reference
    progress(6, 6);
    let chi = crate::utils::adjust_offset(
        &vasc_mask, &lfs1, &chi1, &chi2,
        nx, ny, nz, vsx, vsy, vsz, bdir, qsmart_params.ppm,
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
        let params = crate::utils::QsmartParams::for_field_strength(3.0);

        let result = run_qsmart(
            &field, &mask, None, &meta, &params, QsmReference::Mean,
            &mut |_, _| {},
        );
        assert!(result.is_ok());
        let chi = result.unwrap();
        assert_eq!(chi.len(), n);
        for &v in &chi {
            assert!(v.is_finite(), "QSMART output must be finite");
        }
    }
}

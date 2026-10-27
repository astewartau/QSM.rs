//! QSMART two-stage reconstruction
//!
//! Two-stage SDF + iLSQR pipeline with vasculature detection, after Yaghmaie et al. (2021)
//! and their MATLAB toolbox (<https://github.com/wtsyeda/QSMART>), which inverts each stage
//! with STI Suite's `QSM_iLSQR`.
//! Stage 1: whole-ROI reconstruction. Stage 2: tissue-only reconstruction
//! with vasculature excluded. Final offset adjustment combines both.
//!
//! # Reference pipeline
//!
//! What the authors' toolbox (`QSMART_toolbox_v1.0`, wtsyeda/QSMART; "toolbox" below) and paper
//! (Yaghmaie et al. 2021, NeuroImage 231:117701) run before and around the two stages:
//!
//! - **Brain mask: FSL BET**, not HD-BET. Paper §3.1: "a brain mask … is constructed using FSL's
//!   Brain Extraction Tool". Toolbox `brainmask.m` l. 7–33: root-sum-of-squares over coils of
//!   the **first echo** magnitude, ANTs `N4BiasFieldCorrection` (defaults), `bet2 … -f 0.2 -m`;
//!   then voxels whose BET-extracted magnitude is below `mag_threshold` are removed (100 in
//!   `Demo_QSMART.m`; the authors' October 2020 human demo uses the adaptive 2nd percentile),
//!   `bwmorph3(mask, 'majority')`, and `imclose` with `strel('sphere', 2)` (`sph_radius1`,
//!   voxels). No erosion: avoiding it is the point of SDF (paper §1, §5). Rodent masks were
//!   made by non-linear registration to the Waxholm atlases plus manual correction (paper §3.4).
//! - **Coil combination**: POEM (`coil_comb.m`, `'smooth3'`), before anything QSM.rs takes.
//! - **Unwrapping**: Laplacian (`Demo_QSMART.m` l. 32), Hongfu Sun's `lapunwrap.m` per echo on
//!   the unmasked phase (`unwrap_phase.m` l. 5–10):
//!   [`LaplacianSolver::Mirror`](crate::unwrap::LaplacianSolver::Mirror).
//! - **Total field**: `echofit.m`, magnitude-weighted zero-intercept fit of phase against TE,
//!   divided by `γ·B0/1e6` with γ = 2.675e8 rad/s/T and B0 = `qsm_params.field` (7 in the demo);
//!   the reliability map `R_0` from the box-blurred residual (`fit_threshold` 40 in
//!   `Demo_QSMART.m` with `adaptive_threshold` 0; the 99th percentile in the October 2020 demo
//!   and the authors' 7T run): [`B0EstimationMethod::QsmartEchofit`].
//! - **Vessel magnitude**: mean over echoes of the coil-combined magnitude, then N4
//!   (`vasculature_mask.m` l. 8–14): [`qsmart_vessel_magnitude`](crate::utils::qsmart_vessel_magnitude).
//! - **Inversion**: STI Suite **v2.2**'s `QSM_iLSQR` (paper §3.4, "number of iterations = 50"),
//!   called on `mask·R_0` (stage 1) and `mask·vessels·R_0` (stage 2); see the comment in
//!   [`run_qsmart`] for the call.
//! - **Data and field strength**: tuned on 7T Siemens human 3D unipolar multi-echo GRE, 0.6 mm
//!   isotropic, TE 4.80/8.35/11.90/15.45 ms (paper §3.3; README: "parameter settings have been
//!   optimized for multiecho gradient echo data from Siemens 7T"). Also run on 9.4T ex vivo
//!   mouse (125 µm) and 4.7T rat (150 µm) with other settings: Frangi scales 0.01–0.5 in steps
//!   of 0.1, c = 50, σ_C = σ_V = 5 (paper §3.4). No 3T settings are published. The radii of
//!   the morphology (mask closing 2, bottom-hat 8, indent closing 8) are in voxels.
//! - **SDF widths, code vs paper**: the paper gives σ_C = 10, σ_V = 2, σ_I = 10, σ = 10, C_s = 500
//!   (§3.4). The toolbox passes `s1.sdf_sigma1` (10) for both widths in stage 1, so stage 1's σ
//!   is √(10² + 10²) ≈ 14.1, and stage 2 uses σ1 = 8, σ2 = 2 (σ = √68 ≈ 8.2)
//!   (`QSMART_SDF.m` l. 12–13, 20–21; `sdf_curvature.m` l. 8). QSM.rs follows the toolbox.
//!
//! # Pipeline rules
//!
//! The facts above that a configuration can depart from are published as rules in
//! [`crate::pipeline::rules::RULES`] (each with its strength, basis, reason and source): `qsmart.mask-bet`, `qsmart.unwrap-laplacian`, `qsmart.laplacian-solver`, `qsmart.field-echofit`, `qsmart-echofit.unwrap-laplacian`, `qsmart.inner-inversion`, `qsmart.ilsqr-iterations`, `qsmart.ilsqr-pad`.

use super::config::*;
use super::observer::{Observer, VolumeRef};

/// Names of the intermediates [`run_qsmart`] reports through [`Observer::output`], in the order
/// they are produced. Each is reported once per run.
pub mod outputs {
    /// Vessels found by vasculature detection ([`VolumeRef::Mask`](super::VolumeRef::Mask),
    /// 1 = vessel, always inside the brain mask). Stage 2 runs on the brain mask minus these.
    pub const VESSEL_MASK: &str = "vessel_mask";
    /// Stage-1 local field: SDF over the whole brain mask, vessels included (ppm).
    pub const LOCAL_FIELD_STAGE1: &str = "local_field_stage1";
    /// Stage-1 susceptibility: inner inversion of the stage-1 local field over the brain mask
    /// (ppm, unreferenced).
    pub const CHI_STAGE1: &str = "chi_stage1";
    /// Stage-2 local field: vessel-aware SDF on the tissue-only field (ppm).
    pub const LOCAL_FIELD_STAGE2: &str = "local_field_stage2";
    /// Stage-2 susceptibility: inner inversion of the stage-2 local field over tissue only
    /// (ppm, unreferenced, zero in vessels).
    pub const CHI_STAGE2: &str = "chi_stage2";
}

/// Run QSMART two-stage QSM reconstruction.
///
/// # Arguments
/// * `field_ppm` - Total field in ppm (after field mapping)
/// * `mask` - Binary brain mask
/// * `reliability` - QSMART's reliability map `R_0` (1 = reliable), from its echo fit
///   ([`qsmart_echofit`](crate::utils::qsmart_echofit)). Both stages run on `mask` × `R_0`, as
///   `QSMART.m` does; vasculature detection uses `mask` alone. `None` means every voxel is
///   reliable.
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
/// * `observer` - Receives progress (step of 6) and the intermediates named in [`outputs`] as
///   they are produced, as borrowed slices. A plain `FnMut(usize, usize)` progress closure works
///   here too and ignores the intermediates.
///
/// # Returns
/// Susceptibility map in ppm (referenced): the stage-1 and stage-2 χ combined by offset
/// adjustment, zero outside the brain mask.
pub fn run_qsmart(
    field_ppm: &[f64],
    mask: &[u8],
    reliability: Option<&[u8]>,
    magnitude: Option<&[f64]>,
    metadata: &ScanMetadata,
    inversion_config: &InversionConfig,
    reference: QsmReference,
    observer: &mut dyn Observer,
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
        pad_mm: qsmart_params.ilsqr_pad_mm,
        ..inversion_config.ilsqr.clone()
    };
    if matches!(inner_config.algorithm, InversionAlgorithm::Ilsqr) {
        ilsqr_params.validate().map_err(|e| PipelineError::InvalidConfig(format!("QSMART: {e}")))?;
    }
    let invert = |lfs: &[f64], m: &[u8]| -> Result<Vec<f64>, PipelineError> {
        if matches!(inner_config.algorithm, InversionAlgorithm::Ilsqr) {
            let (chi, _, _, _) = crate::inversion::ilsqr(lfs, m, &grid, bdir, &ilsqr_params, |_, _| {});
            Ok(chi)
        } else {
            super::inversion::run_dipole_inversion(lfs, m, metadata, &inner_config, magnitude, &mut |_, _| {})
        }
    };

    // Step 1: Vasculature detection (vasc_mask: 1 = tissue, 0 = vessel)
    observer.progress(1, 6);
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
    )
    .map_err(|e| PipelineError::InvalidConfig(format!("QSMART: {e}")))?;
    {
        let vessel_mask: Vec<u8> = vasc_mask.iter().map(|&v| u8::from(v < 0.5)).collect();
        observer.output(outputs::VESSEL_MASK, VolumeRef::Mask(&vessel_mask));
    }

    // QSMART.m's `mask.*R_0`: the region both stages reconstruct.
    if reliability.is_some_and(|r| r.len() != n_voxels) {
        return Err(PipelineError::InvalidConfig("QSMART: reliability map does not match the grid".into()));
    }
    let weighted_mask: Vec<f64> = (0..n_voxels)
        .map(|i| if mask[i] != 0 && reliability.is_none_or(|r| r[i] != 0) { 1.0 } else { 0.0 })
        .collect();
    let ones_vasc = vec![1.0f64; n_voxels];

    // Step 2: SDF stage 1 — background removal over the whole ROI (vessels included).
    observer.progress(2, 6);
    let sdf_params1 = crate::bgremove::SdfParams {
        sigma1: qsmart_params.sdf_sigma1_stage1,
        sigma2: qsmart_params.sdf_sigma2_stage1,
        lower_lim: qsmart_params.sdf_lower_lim,
        curv_constant: qsmart_params.sdf_curv_constant,
        use_curvature: true,
    };
    let lfs1 = crate::bgremove::sdf::sdf(
        field_ppm, &weighted_mask, &ones_vasc, &grid, &sdf_params1, |_, _| {},
    );
    observer.output(outputs::LOCAL_FIELD_STAGE1, VolumeRef::Real(&lfs1));

    // Step 3: dipole inversion stage 1 over the whole ROI.
    observer.progress(3, 6);
    let mask_stage1: Vec<u8> = weighted_mask.iter().map(|&v| if v > 0.1 { 1 } else { 0 }).collect();
    let chi1 = invert(&lfs1, &mask_stage1)?;
    observer.output(outputs::CHI_STAGE1, VolumeRef::Real(&chi1));

    // Step 4: SDF stage 2 — tissue-only, vessel-aware (mask-zeroed field input).
    observer.progress(4, 6);
    let sdf_params2 = crate::bgremove::SdfParams {
        sigma1: qsmart_params.sdf_sigma1_stage2,
        sigma2: qsmart_params.sdf_sigma2_stage2,
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
    observer.output(outputs::LOCAL_FIELD_STAGE2, VolumeRef::Real(&lfs2));

    // Step 5: dipole inversion stage 2 over tissue only (in-mask AND not vessel).
    observer.progress(5, 6);
    let mask_stage2: Vec<u8> = weighted_mask.iter()
        .zip(vasc_mask.iter())
        .map(|(&m, &v)| if m > 0.1 && v > 0.5 { 1 } else { 0 })
        .collect();
    let chi2 = invert(&lfs2, &mask_stage2)?;
    observer.output(outputs::CHI_STAGE2, VolumeRef::Real(&chi2));

    // Step 6: offset adjustment (combine stages) and reference.
    // removed_voxels = vessel regions (in mask but excluded from stage 2).
    // Everything is in ppm here, so adjust_offset's lfs rescale is the identity (ppm = 1.0).
    observer.progress(6, 6);
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
            affine: None,
        };
        let config = InversionConfig {
            algorithm: InversionAlgorithm::Qsmart,
            qsmart: crate::utils::QsmartParams::for_field_strength(3.0),
            ..Default::default()
        };

        let result = run_qsmart(
            &field, &mask, None, None, &meta, &config, QsmReference::Mean,
            &mut (),
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
            affine: None,
        };
        let mut qsmart = crate::utils::QsmartParams::for_field_strength(3.0);
        qsmart.inversion = InversionAlgorithm::Tkd;
        let config = InversionConfig {
            algorithm: InversionAlgorithm::Qsmart,
            qsmart,
            ..Default::default()
        };

        let chi = run_qsmart(
            &field, &mask, None, None, &meta, &config, QsmReference::Mean,
            &mut (),
        )
        .unwrap();
        assert_eq!(chi.len(), n);
        for &v in &chi {
            assert!(v.is_finite(), "QSMART output must be finite");
        }
    }

    #[test]
    fn test_run_qsmart_reports_intermediates() {
        #[derive(Default)]
        struct Recorder {
            steps: Vec<(usize, usize)>,
            outputs: Vec<(&'static str, Vec<f64>)>,
        }
        impl Observer for Recorder {
            fn progress(&mut self, done: usize, total: usize) {
                self.steps.push((done, total));
            }
            fn output(&mut self, name: &'static str, volume: VolumeRef<'_>) {
                let data = match volume {
                    VolumeRef::Real(v) => v.to_vec(),
                    VolumeRef::Mask(m) => m.iter().map(|&b| b as f64).collect(),
                };
                self.outputs.push((name, data));
            }
        }

        let (nx, ny, nz) = (12, 12, 12);
        let n = nx * ny * nz;
        let field: Vec<f64> = (0..n).map(|i| 0.01 * ((i % 7) as f64 - 3.0)).collect();
        let magnitude: Vec<f64> = (0..n).map(|i| if i % 11 == 0 { 0.2 } else { 1.0 }).collect();
        let mask = vec![1u8; n];
        let meta = ScanMetadata {
            dims: (nx, ny, nz),
            voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005],
            field_strength: 3.0,
            b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: None,
            affine: None,
        };
        let mut qsmart = crate::utils::QsmartParams::for_field_strength(3.0);
        qsmart.inversion = InversionAlgorithm::Tkd;
        let config = InversionConfig { algorithm: InversionAlgorithm::Qsmart, qsmart, ..Default::default() };

        let mut rec = Recorder::default();
        let chi = run_qsmart(
            &field, &mask, None, Some(&magnitude), &meta, &config, QsmReference::None, &mut rec,
        )
        .unwrap();
        // Reporting intermediates does not change the result.
        let plain = run_qsmart(
            &field, &mask, None, Some(&magnitude), &meta, &config, QsmReference::None, &mut (),
        )
        .unwrap();
        assert_eq!(chi, plain);

        assert_eq!(rec.steps, (1..=6).map(|i| (i, 6)).collect::<Vec<_>>());
        let names: Vec<_> = rec.outputs.iter().map(|(name, _)| *name).collect();
        assert_eq!(
            names,
            [
                outputs::VESSEL_MASK,
                outputs::LOCAL_FIELD_STAGE1,
                outputs::CHI_STAGE1,
                outputs::LOCAL_FIELD_STAGE2,
                outputs::CHI_STAGE2,
            ]
        );
        for (name, data) in &rec.outputs {
            assert_eq!(data.len(), n, "{name}");
            assert!(data.iter().all(|v| v.is_finite()), "{name}");
        }
        // Stage 2 inverts tissue only, so its χ is zero in vessels.
        let vessels = &rec.outputs[0].1;
        let chi2 = &rec.outputs[4].1;
        for (&v, &c) in vessels.iter().zip(chi2) {
            assert!(v == 0.0 || v == 1.0);
            if v == 1.0 {
                assert_eq!(c, 0.0);
            }
        }

        // A progress closure is an observer too.
        let mut steps = 0;
        run_qsmart(
            &field, &mask, None, Some(&magnitude), &meta, &config, QsmReference::None,
            &mut |_: usize, _: usize| steps += 1,
        )
        .unwrap();
        assert_eq!(steps, 6);
    }

    #[test]
    fn unreliable_voxels_are_left_out_of_both_stages() {
        // R_0 = 0 takes a voxel out of both stages' region (QSMART.m's `mask.*R_0`), as an
        // unmasked voxel would be, but not out of the vessel search or the brain mask.
        #[derive(Default)]
        struct Chis(Vec<(&'static str, Vec<f64>)>);
        impl Observer for Chis {
            fn progress(&mut self, _: usize, _: usize) {}
            fn output(&mut self, name: &'static str, volume: VolumeRef<'_>) {
                if let VolumeRef::Real(v) = volume { self.0.push((name, v.to_vec())); }
            }
        }
        let (nx, ny, nz) = (12, 12, 12);
        let n = nx * ny * nz;
        let field: Vec<f64> = (0..n).map(|i| 0.01 * ((i % 7) as f64 - 3.0)).collect();
        let mask = vec![1u8; n];
        let reliability: Vec<u8> = (0..n).map(|i| u8::from(i % 5 != 0)).collect();
        let meta = ScanMetadata {
            dims: (nx, ny, nz), voxel_size: (1.0, 1.0, 1.0), echo_times: vec![0.005],
            field_strength: 3.0, b0_direction: (0.0, 0.0, 1.0), slice_geometry: None, affine: None,
        };
        let mut qsmart = crate::utils::QsmartParams::for_field_strength(3.0);
        qsmart.inversion = InversionAlgorithm::Tkd;
        let config = InversionConfig { algorithm: InversionAlgorithm::Qsmart, qsmart, ..Default::default() };
        let mut rec = Chis::default();
        run_qsmart(&field, &mask, Some(&reliability), None, &meta, &config, QsmReference::None, &mut rec).unwrap();
        for (name, v) in &rec.0 {
            for i in (0..n).filter(|&i| reliability[i] == 0) {
                assert_eq!(v[i], 0.0, "{name} at unreliable voxel {i}");
            }
        }
        let short = vec![1u8; n - 1];
        assert!(run_qsmart(&field, &mask, Some(&short), None, &meta, &config, QsmReference::None, &mut ()).is_err());
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
            affine: None,
        };
        let mut qsmart = crate::utils::QsmartParams::for_field_strength(3.0);
        qsmart.inversion = InversionAlgorithm::Qsmart;
        let config = InversionConfig {
            algorithm: InversionAlgorithm::Qsmart,
            qsmart,
            ..Default::default()
        };

        let result = run_qsmart(
            &vec![0.0; n], &vec![1u8; n], None, None, &meta, &config,
            QsmReference::Mean, &mut (),
        );
        assert!(result.is_err());
    }
}

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
        InversionAlgorithm::Tfi => {
            // TFI takes the field in ppm (same convention as NDI and the other inversions —
            // NOT MEDI's radians). It is single-step: `local_field_ppm` is fed as the TOTAL
            // field (caller supplies the pre-background-removal field). Converting to radians
            // here would wrap the large total field in exp(i·field) and destroy the result.
            let uniform_mag = vec![1.0f64; n_voxels];
            let mag = magnitude.unwrap_or(&uniform_mag);
            let n_std = vec![1.0f64; n_voxels];

            crate::inversion::tfi(
                local_field_ppm, &n_std, mag, mask,
                &grid, bdir, &config.tfi, progress,
            )
        }
        InversionAlgorithm::Ilsqr => {
            let (chi, _, _, _) = crate::inversion::ilsqr(
                local_field_ppm, mask, &grid, bdir, &config.ilsqr, &mut *progress,
            );
            chi
        }
        InversionAlgorithm::Ndi => {
            crate::inversion::ndi(
                local_field_ppm, mask, &grid, bdir, &config.ndi, progress,
            )
        }
        InversionAlgorithm::Fansi => {
            let params = crate::inversion::FansiParams { is_tgv: false, ..config.fansi.clone() };
            crate::inversion::fansi(
                local_field_ppm, mask, &grid, bdir, &params, progress,
            )
        }
        InversionAlgorithm::FansiTgv => {
            let params = crate::inversion::FansiParams { is_tgv: true, ..config.fansi.clone() };
            crate::inversion::fansi(
                local_field_ppm, mask, &grid, bdir, &params, progress,
            )
        }
        InversionAlgorithm::L1qsm => {
            crate::inversion::l1qsm(
                local_field_ppm, mask, &grid, bdir, &config.l1qsm, progress,
            )
        }
        InversionAlgorithm::Whqsm => {
            crate::inversion::whqsm(
                local_field_ppm, mask, &grid, bdir, &config.whqsm, progress,
            )
        }
        InversionAlgorithm::Hdqsm => {
            crate::inversion::hdqsm(
                local_field_ppm, mask, &grid, bdir, &config.hdqsm, progress,
            )
        }
        InversionAlgorithm::AmpPe => {
            // AMP-PE takes the local field in ppm (like NDI). `b0` (field strength)
            // is a scan parameter sourced from metadata, not a user knob; magnitude,
            // when present, is the data-fidelity weight + morphology mask.
            let params = crate::inversion::AmpPeParams {
                b0: metadata.field_strength,
                ..config.amp_pe.clone()
            };
            crate::inversion::amp_pe(
                local_field_ppm, mask, magnitude, &grid, bdir, &params, |i, n| progress(i, n),
            )
        }
        InversionAlgorithm::Xqsm => run_xqsm(local_field_ppm, mask, &grid)?,
        InversionAlgorithm::Qsmnet => run_qsmnet(local_field_ppm, mask, &grid, "qsmnet")?,
        InversionAlgorithm::QsmnetPlus => run_qsmnet(local_field_ppm, mask, &grid, "qsmnet-plus")?,
        InversionAlgorithm::Autoqsm => run_autoqsm(local_field_ppm, mask, &grid)?,
        InversionAlgorithm::Tgv | InversionAlgorithm::Qsmart => {
            return Err(PipelineError::InvalidConfig(
                format!("{:?} should use run_tgv or run_qsmart", config.algorithm),
            ));
        }
    };

    Ok(chi)
}

/// Source the xQSM weights and run inference. Requires the `onnx` feature;
/// weights come from the model registry (local `$QSM_MODEL_DIR`/cache, or the
/// `download` feature).
#[cfg(feature = "onnx")]
fn run_xqsm(
    local_field_ppm: &[f64],
    mask: &[u8],
    grid: &crate::Grid,
) -> Result<Vec<f64>, PipelineError> {
    let spec = crate::models::find_model("xqsm")
        .ok_or_else(|| PipelineError::InvalidConfig("xqsm not in model registry".into()))?;
    let bytes = crate::models::primary_weight_bytes(spec)
        .map_err(PipelineError::InvalidConfig)?;
    crate::inversion::xqsm(local_field_ppm, mask, grid, &bytes)
        .map_err(|e| PipelineError::AlgorithmError(e.to_string()))
}

#[cfg(not(feature = "onnx"))]
fn run_xqsm(
    _local_field_ppm: &[f64],
    _mask: &[u8],
    _grid: &crate::Grid,
) -> Result<Vec<f64>, PipelineError> {
    Err(PipelineError::InvalidConfig(
        "xQSM requires building qsm-core with the 'onnx' feature".into(),
    ))
}

/// Run NeXtQSM end-to-end from the **total** field (it does its own background
/// removal), sourcing both weight files from the registry. Unlike the entries in
/// [`InversionAlgorithm`], NeXtQSM spans BFR + dipole inversion, so it is exposed
/// as a standalone reconstruction rather than a dipole-inversion-stage option.
///
/// `total_field_ppm` and `mask` are column-major `(nx,ny,nz)`; `bdir` is the B0
/// direction. Requires the `onnx` feature; weights resolve local-first then via
/// the `download` feature. Returns susceptibility (ppm), masked.
#[cfg(feature = "onnx")]
pub fn run_nextqsm(
    total_field_ppm: &[f64],
    mask: &[u8],
    grid: &crate::Grid,
    bdir: (f64, f64, f64),
) -> Result<Vec<f64>, PipelineError> {
    let spec = crate::models::find_model("nextqsm")
        .ok_or_else(|| PipelineError::InvalidConfig("nextqsm not in model registry".into()))?;
    let files = crate::models::all_weight_bytes(spec).map_err(PipelineError::InvalidConfig)?;
    let [bf, vjp] = &files[..] else {
        return Err(PipelineError::InvalidConfig(
            format!("nextqsm expects 2 weight files (BFR, VJP), got {}", files.len()),
        ));
    };
    crate::inversion::nextqsm(total_field_ppm, mask, grid, bdir, bf, vjp)
        .map_err(|e| PipelineError::AlgorithmError(e.to_string()))
}

/// Stub when built without the `onnx` feature.
#[cfg(not(feature = "onnx"))]
pub fn run_nextqsm(
    _total_field_ppm: &[f64],
    _mask: &[u8],
    _grid: &crate::Grid,
    _bdir: (f64, f64, f64),
) -> Result<Vec<f64>, PipelineError> {
    Err(PipelineError::InvalidConfig(
        "NeXtQSM requires building qsm-core with the 'onnx' feature".into(),
    ))
}

/// Source a QSMnet-family model's weights and run inference (requires the `onnx`
/// feature). `model_id` selects the registry entry (`qsmnet` / `qsmnet-plus`),
/// `norm` supplies that checkpoint's normalization constants.
#[cfg(feature = "onnx")]
fn run_qsmnet(
    local_field_ppm: &[f64],
    mask: &[u8],
    grid: &crate::Grid,
    model_id: &str,
) -> Result<Vec<f64>, PipelineError> {
    use crate::inversion::QsmnetNorm;
    let norm = match model_id {
        "qsmnet-plus" => QsmnetNorm::qsmnet_plus(),
        _ => QsmnetNorm::qsmnet(),
    };
    let spec = crate::models::find_model(model_id).ok_or_else(|| {
        PipelineError::InvalidConfig(format!("{model_id} not in model registry"))
    })?;
    let bytes = crate::models::primary_weight_bytes(spec)
        .map_err(PipelineError::InvalidConfig)?;
    crate::inversion::qsmnet(local_field_ppm, mask, grid, &bytes, &norm)
        .map_err(|e| PipelineError::AlgorithmError(e.to_string()))
}

#[cfg(not(feature = "onnx"))]
fn run_qsmnet(
    _local_field_ppm: &[f64],
    _mask: &[u8],
    _grid: &crate::Grid,
    _model_id: &str,
) -> Result<Vec<f64>, PipelineError> {
    Err(PipelineError::InvalidConfig(
        "QSMnet requires building qsm-core with the 'onnx' feature".into(),
    ))
}

/// Source the AutoQSM weights and run inference (requires the `onnx` feature).
/// `field` is the **total** field (AutoQSM does its own background removal).
#[cfg(feature = "onnx")]
fn run_autoqsm(
    field: &[f64],
    mask: &[u8],
    grid: &crate::Grid,
) -> Result<Vec<f64>, PipelineError> {
    let spec = crate::models::find_model("autoqsm")
        .ok_or_else(|| PipelineError::InvalidConfig("autoqsm not in model registry".into()))?;
    let bytes = crate::models::primary_weight_bytes(spec).map_err(PipelineError::InvalidConfig)?;
    crate::inversion::autoqsm(field, mask, grid, &bytes)
        .map_err(|e| PipelineError::AlgorithmError(e.to_string()))
}

#[cfg(not(feature = "onnx"))]
fn run_autoqsm(
    _field: &[f64],
    _mask: &[u8],
    _grid: &crate::Grid,
) -> Result<Vec<f64>, PipelineError> {
    Err(PipelineError::InvalidConfig(
        "AutoQSM requires building qsm-core with the 'onnx' feature".into(),
    ))
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
    fn test_inversion_amp_pe() {
        let chi = make_inversion_test(InversionAlgorithm::AmpPe);
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

//! Dipole inversion stage
//!
//! Dispatcher for standard dipole inversion algorithms.
//! Also contains TGV and QSMART pipeline runners which combine
//! multiple stages internally.
//!
//! MEDI Hz↔radians conversion is handled internally — the caller
//! passes ppm fields and receives ppm results.


use super::config::*;

/// The echo magnitudes MEDI requires (its edge mask and its `Fit_ppm_complex` noise map): one
/// volume per echo, echo slowest, for evenly spaced echoes (as `Fit_ppm_complex.m` assumes).
pub(crate) fn medi_echo_magnitudes<'a>(
    magnitude: Option<&'a [f64]>,
    echo_times: &[f64],
    n_voxels: usize,
) -> Result<crate::inversion::EchoMagnitudes<'a>, PipelineError> {
    let n_echoes = echo_times.len();
    if n_echoes == 0 {
        return Err(PipelineError::InvalidInput("MEDI needs the echo times".to_string()));
    }
    let spacing: Vec<f64> = echo_times.windows(2).map(|w| w[1] - w[0]).collect();
    if spacing.iter().any(|d| (d - spacing[0]).abs() > 1e-3 * spacing[0].abs()) {
        return Err(PipelineError::InvalidInput(format!(
            "MEDI's noise map (Fit_ppm_complex) needs evenly spaced echoes; got {echo_times:?}"
        )));
    }
    match magnitude {
        Some(m) if m.len() == n_voxels * n_echoes => {
            Ok(crate::inversion::EchoMagnitudes { data: m, n_echoes })
        }
        Some(m) => Err(PipelineError::InvalidInput(format!(
            "MEDI needs the magnitude of every echo ({n_echoes} volumes, {} values); got {} values",
            n_voxels * n_echoes, m.len(),
        ))),
        None => Err(PipelineError::InvalidInput(format!(
            "MEDI needs the magnitude of every echo ({n_echoes} volumes)"
        ))),
    }
}

/// Run a dipole inversion on the field its method takes.
///
/// Which field that is comes from [`InversionConfig::input`]:
///
/// - [`InversionInput::LocalField`] (most methods): the local field after background removal,
///   with the eroded mask background removal returned.
/// - [`InversionInput::TotalField`]: the total field straight from field mapping, with the
///   brain mask and no background removal. Through this function that is TFI, MEDI with SMV,
///   AutoQSM and NeXtQSM. TFI solves over the whole FOV under its preconditioner (1 inside
///   `mask`, `precond` outside) and weights the data by the magnitude inside the mask, so the
///   total field and the brain mask are exactly what it needs; given a local field and an
///   eroded mask it would run, but would be solving a different problem.
/// - [`InversionInput::WrappedPhase`] methods, and TGV and QSMART, have their own entry points
///   and are refused here.
///
/// Handles MEDI's unit conversion (ppm → radians) internally.
///
/// # Arguments
/// * `field_ppm` - The local or total field in ppm, as above
/// * `mask` - The eroded mask from background removal (local field), or the brain mask
///   (total field)
/// * `metadata` - Scan metadata
/// * `config` - Inversion configuration
/// * `magnitude` - Combined magnitude image; MEDI instead requires the magnitude of every echo,
///   stacked with the echo slowest
/// * `progress` - Progress callback (current_iter, max_iter)
///
/// # Returns
/// Susceptibility map in ppm (unreferenced)
pub fn run_dipole_inversion(
    field_ppm: &[f64],
    mask: &[u8],
    metadata: &ScanMetadata,
    config: &InversionConfig,
    magnitude: Option<&[f64]>,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Vec<f64>, PipelineError> {
    // The dipole kernel is an FFT on a uniform grid, so a gapped volume is refused.
    metadata.require_contiguous_slices()?;
    let grid = metadata.grid();
    let bdir = metadata.b0_direction;
    let n_voxels = grid.n_total();

    if field_ppm.len() != n_voxels {
        return Err(PipelineError::DimensionMismatch {
            expected: n_voxels,
            got: field_ppm.len(),
        });
    }

    let chi = match config.algorithm {
        InversionAlgorithm::Tkd => {
            crate::inversion::tkd(
                field_ppm, mask, &grid, bdir, &config.tkd,
            )
        }
        InversionAlgorithm::Tsvd => {
            crate::inversion::tsvd(
                field_ppm, mask, &grid, bdir, &config.tsvd,
            )
        }
        InversionAlgorithm::Tikhonov => {
            crate::inversion::tikhonov(
                field_ppm, mask, &grid, bdir, &config.tikhonov,
            )
        }
        InversionAlgorithm::Tv => {
            crate::inversion::tv_admm(
                field_ppm, mask, &grid, bdir, &config.tv, progress,
            )
        }
        InversionAlgorithm::Rts => {
            crate::inversion::rts(
                field_ppm, mask, &grid, bdir, &config.rts, progress,
            )
        }
        InversionAlgorithm::Nltv => {
            // NLTV takes ppm and converts to radians itself, like NDI; field strength and echo
            // time (the first) come from metadata. The magnitude is the data weight and is
            // required: without it the method can diverge (see the module docs).
            let mag = magnitude.ok_or_else(|| PipelineError::InvalidInput(
                "NLTV needs the magnitude (its data weight)".into(),
            ))?;
            let params = crate::inversion::NltvParams {
                b0: metadata.field_strength,
                te: metadata.echo_times.first().copied().unwrap_or(config.nltv.te),
                ..config.nltv.clone()
            };
            crate::inversion::nltv(
                field_ppm, mask, mag, &grid, bdir, &params, progress,
            )
        }
        InversionAlgorithm::Medi => {
            // MEDI takes ppm like the others and converts to radians itself, over the echo
            // spacing as MEDI's readers set delta_TE: TE(2) - TE(1), or TE for one echo. Field
            // strength and echo spacing are scan parameters from metadata. It requires the
            // magnitude of every echo (edge mask and the Fit_ppm_complex noise map).
            let te = &metadata.echo_times;
            let mag = crate::pipeline::inversion::medi_echo_magnitudes(magnitude, te, n_voxels)?;
            let params = crate::inversion::MediParams {
                b0: metadata.field_strength,
                te: if te.len() == 1 { te[0] } else { te[1] - te[0] },
                ..config.medi.clone()
            };
            crate::inversion::medi(field_ppm, mask, mag, &grid, bdir, &params, progress)
        }
        InversionAlgorithm::Tfi => {
            // TFI takes the field in ppm (same convention as NDI and the other inversions —
            // MEDI's). It is single-step: `field_ppm` is the TOTAL field and `mask`
            // the brain mask (InversionConfig::input). Converting to radians here would wrap
            // the large total field in exp(i·field) and destroy the result.
            let uniform_mag = vec![1.0f64; n_voxels];
            let mag = magnitude.unwrap_or(&uniform_mag);
            let n_std = vec![1.0f64; n_voxels];

            crate::inversion::tfi(
                field_ppm, &n_std, mag, mask,
                &grid, bdir, &config.tfi, progress,
            )
        }
        InversionAlgorithm::Ilsqr => {
            config.ilsqr.validate().map_err(PipelineError::InvalidConfig)?;
            let (chi, _, _, _) = crate::inversion::ilsqr(
                field_ppm, mask, &grid, bdir, &config.ilsqr, &mut *progress,
            );
            chi
        }
        InversionAlgorithm::Star => {
            // STAR-QSM's result depends on the field's scale in radians (see the module docs),
            // so it converts the ppm field to phase at the first echo time and this field
            // strength, as STI Suite is given the phase of one echo.
            let params = crate::inversion::StarParams {
                te: metadata.echo_times.first().copied().unwrap_or(config.star.te),
                b0: metadata.field_strength,
                ..config.star.clone()
            };
            params.validate().map_err(PipelineError::InvalidConfig)?;
            crate::inversion::star_qsm(field_ppm, mask, &grid, bdir, &params, &mut *progress)
        }
        InversionAlgorithm::Ndi => {
            // NDI takes ppm like the others and converts to radians itself. Field strength and
            // echo time (the first, as for MEDI) are scan parameters from metadata, not user
            // knobs; the magnitude is the data weight and is required.
            let magnitude = magnitude.ok_or_else(|| PipelineError::InvalidConfig(
                "NDI needs the magnitude as its data weight".into()))?;
            let params = crate::inversion::NdiParams {
                b0: metadata.field_strength,
                te: metadata.echo_times.first().copied().unwrap_or(config.ndi.te),
                ..config.ndi.clone()
            };
            crate::inversion::ndi(
                field_ppm, mask, magnitude, &grid, bdir, &params, progress,
            )
        }
        InversionAlgorithm::Fansi | InversionAlgorithm::FansiTgv | InversionAlgorithm::L1qsm => {
            // FANSI nlTV/nlTGV and L1-QSM take ppm like the others and convert to radians
            // themselves. Field strength and echo time (the first, as for NDI) are scan
            // parameters from metadata. The magnitude is the data weight and is required:
            // with a flat weight these solvers can diverge.
            let magnitude = magnitude.ok_or_else(|| {
                PipelineError::InvalidInput(format!(
                    "{:?} requires magnitude data (its data weight)", config.algorithm
                ))
            })?;
            let b0 = metadata.field_strength;
            let te = metadata.echo_times.first().copied();
            if config.algorithm == InversionAlgorithm::L1qsm {
                let params = crate::inversion::L1QsmParams {
                    b0,
                    te: te.unwrap_or(config.l1qsm.te),
                    ..config.l1qsm.clone()
                };
                crate::inversion::l1qsm(
                    field_ppm, mask, magnitude, &grid, bdir, &params, progress,
                )
            } else {
                let params = crate::inversion::FansiParams {
                    b0,
                    te: te.unwrap_or(config.fansi.te),
                    is_tgv: config.algorithm == InversionAlgorithm::FansiTgv,
                    ..config.fansi.clone()
                };
                crate::inversion::fansi(
                    field_ppm, mask, magnitude, &grid, bdir, &params, progress,
                )
            }
        }
        InversionAlgorithm::Whqsm => {
            // Like NDI: ppm in and out, radians inside. Field strength and echo time (the
            // first) are scan parameters from metadata; the magnitude, when present, is the
            // data weight.
            let params = crate::inversion::WhQsmParams {
                b0: metadata.field_strength,
                te: metadata.echo_times.first().copied().unwrap_or(config.whqsm.te),
                ..config.whqsm.clone()
            };
            crate::inversion::whqsm(
                field_ppm, mask, magnitude, &grid, bdir, &params, progress,
            )
        }
        InversionAlgorithm::Hdqsm => {
            // The magnitude, when present, is the data weight (HDQSM.m usage example).
            crate::inversion::hdqsm(
                field_ppm, mask, magnitude, &grid, bdir, &config.hdqsm, progress,
            )
        }
        InversionAlgorithm::AmpPe => {
            // AMP-PE takes the local field in ppm (like NDI). `b0` (field strength)
            // is a scan parameter sourced from metadata, not a user knob. The one magnitude
            // this stage has serves as both the data weight and the morphology image; upstream
            // uses the first echo's magnitude for the first and the RSS over echoes for the
            // second, which a caller holding the echoes can pass to `amp_pe` directly.
            let params = crate::inversion::AmpPeParams {
                b0: metadata.field_strength,
                ..config.amp_pe.clone()
            };
            crate::inversion::amp_pe(
                field_ppm, mask, magnitude, None, &grid, bdir, &params, progress,
            )
        }
        InversionAlgorithm::Lsqr => {
            let params = crate::inversion::LsqrQsmParams {
                b0: metadata.field_strength,
                ..config.lsqr.clone()
            };
            crate::inversion::lsqr_qsm(
                field_ppm, mask, magnitude, &grid, bdir, &params, progress,
            )
        }
        InversionAlgorithm::Heidi => {
            // HEIDI is incremental: it keeps the well-conditioned k-space of a
            // seed map and re-derives the rest. The seed is the minimally
            // regularised LSQR solution, left unmasked so the cone projection
            // does not ring off a hard mask edge.
            let lsqr_params = crate::inversion::LsqrQsmParams {
                b0: metadata.field_strength,
                mask_output: false,
                ..config.lsqr.clone()
            };
            let heidi_params = &config.heidi;

            let lsqr_iters = lsqr_params.max_iter;
            let heidi_iters =
                heidi_params.continuation_steps.max(1) * heidi_params.inner_iterations;
            let total = lsqr_iters + heidi_iters;

            let chi_init = crate::inversion::lsqr_qsm(
                field_ppm, mask, magnitude, &grid, bdir, &lsqr_params,
                |i, _| progress(i, total),
            );
            crate::inversion::heidi(
                field_ppm, mask, &chi_init, &grid, bdir, heidi_params,
                |i, _| progress(lsqr_iters + i, total),
            )
        }
        InversionAlgorithm::Xqsm => run_xqsm(field_ppm, mask, &grid, config.tile, progress)?,
        InversionAlgorithm::Qsmnet => run_qsmnet(field_ppm, mask, &grid, "qsmnet", config.tile, progress)?,
        InversionAlgorithm::QsmnetPlus => run_qsmnet(field_ppm, mask, &grid, "qsmnet-plus", config.tile, progress)?,
        InversionAlgorithm::Autoqsm => run_autoqsm(field_ppm, mask, &grid)?,
        InversionAlgorithm::Qsmgan => run_qsmgan(field_ppm, mask, &grid)?,
        InversionAlgorithm::Ir2qsm => run_ir2qsm(field_ppm, mask, &grid, config.tile, progress)?,
        InversionAlgorithm::Lpcnn => run_lpcnn(field_ppm, mask, &grid, bdir, config.tile, progress)?,
        InversionAlgorithm::ModlQsm => run_modl_qsm(field_ppm, mask, &grid, bdir, config.tile, progress)?,
        // NeXtQSM is single-step (own BFR): `field_ppm` must be the total field.
        InversionAlgorithm::Nextqsm => run_nextqsm(field_ppm, mask, metadata, config.tile, progress)?,
        InversionAlgorithm::Tgv | InversionAlgorithm::Qsmart => {
            return Err(PipelineError::InvalidConfig(
                format!("{:?} should use run_tgv or run_qsmart", config.algorithm),
            ));
        }
        // iQSM/iQSM+ reconstruct from wrapped phase (not a local field) — end-to-end.
        InversionAlgorithm::Iqsm | InversionAlgorithm::IqsmPlus => {
            return Err(PipelineError::InvalidConfig(
                format!("{:?} should use run_iqsm / run_iqsm_plus (it takes wrapped phase)", config.algorithm),
            ));
        }
    };

    Ok(chi)
}

/// Source the xQSM weights and run inference. Requires the `onnx` feature;
/// weights resolve through [`crate::models`] (bytes supplied with
/// [`crate::models::provide_weights`], then `$QSM_MODEL_DIR`/cache/download on native).
#[cfg(feature = "onnx")]
fn run_xqsm(
    local_field_ppm: &[f64],
    mask: &[u8],
    grid: &crate::Grid,
    tile: Option<(usize, usize)>,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Vec<f64>, PipelineError> {
    let spec = crate::models::find_model("xqsm")
        .ok_or_else(|| PipelineError::InvalidConfig("xqsm not in model registry".into()))?;
    let bytes = crate::models::primary_weight_bytes(spec)
        .map_err(PipelineError::InvalidConfig)?;
    match tile {
        Some((core, halo)) => crate::inversion::xqsm_tiled(
            local_field_ppm, mask, grid, &bytes, &crate::inversion::TileConfig { core, halo }, progress,
        ),
        None => crate::inversion::xqsm(local_field_ppm, mask, grid, &bytes),
    }
    .map_err(|e| PipelineError::AlgorithmError(e.to_string()))
}

#[cfg(not(feature = "onnx"))]
fn run_xqsm(
    _local_field_ppm: &[f64],
    _mask: &[u8],
    _grid: &crate::Grid,
    _tile: Option<(usize, usize)>,
    _progress: &mut dyn FnMut(usize, usize),
) -> Result<Vec<f64>, PipelineError> {
    Err(PipelineError::InvalidConfig(
        "xQSM requires building qsm-core with the 'onnx' feature".into(),
    ))
}

/// Run NeXtQSM end-to-end from the **total** field (it does its own background
/// removal), sourcing both weight files from the registry.
///
/// NeXtQSM spans BFR + dipole inversion, so unlike the other entries in
/// [`InversionAlgorithm`] its input is the *total* field rather than a local one. It is
/// reachable both ways: as [`InversionAlgorithm::Nextqsm`] through
/// [`run_dipole_inversion`], and directly here for a caller that wants it standalone.
///
/// `total_field_ppm` and `mask` are column-major `(nx,ny,nz)`; the grid and B0 direction come
/// from `metadata`. Requires the `onnx` feature; both weight files resolve through [`crate::models`]
/// (supplied bytes first, as for every network). Returns susceptibility (ppm), masked.
///
/// Refuses a volume whose slices are not contiguous, like every other stage here: NeXtQSM's
/// unrolled data-consistency term is an FFT dipole convolution, so a gap invalidates it as
/// thoroughly as it does a classical inversion. See
/// [`ScanMetadata::require_contiguous_slices`].
#[cfg(feature = "onnx")]
pub fn run_nextqsm(
    total_field_ppm: &[f64],
    mask: &[u8],
    metadata: &ScanMetadata,
    tile: Option<(usize, usize)>,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Vec<f64>, PipelineError> {
    // Checked here rather than only in `run_dipole_inversion`, because this is also a public
    // entry point in its own right. Reached through the dispatcher it runs twice, which costs
    // one float comparison.
    metadata.require_contiguous_slices()?;
    let grid = &metadata.grid();
    let bdir = metadata.b0_direction;
    let spec = crate::models::find_model("nextqsm")
        .ok_or_else(|| PipelineError::InvalidConfig("nextqsm not in model registry".into()))?;
    let files = crate::models::all_weight_bytes(spec).map_err(PipelineError::InvalidConfig)?;
    let [bf, vjp] = &files[..] else {
        return Err(PipelineError::InvalidConfig(
            format!("nextqsm expects 2 weight files (BFR, VJP), got {}", files.len()),
        ));
    };
    match tile {
        Some((core, halo)) => crate::inversion::nextqsm_tiled(
            total_field_ppm, mask, grid, bdir, bf, vjp, &crate::inversion::TileConfig { core, halo }, progress,
        ),
        None => crate::inversion::nextqsm(total_field_ppm, mask, grid, bdir, bf, vjp),
    }
    .map_err(|e| PipelineError::AlgorithmError(e.to_string()))
}

/// Stub when built without the `onnx` feature.
#[cfg(not(feature = "onnx"))]
pub fn run_nextqsm(
    _total_field_ppm: &[f64],
    _mask: &[u8],
    _metadata: &ScanMetadata,
    _tile: Option<(usize, usize)>,
    _progress: &mut dyn FnMut(usize, usize),
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
    tile: Option<(usize, usize)>,
    progress: &mut dyn FnMut(usize, usize),
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
    match tile {
        Some((core, halo)) => crate::inversion::qsmnet_tiled(
            local_field_ppm, mask, grid, &bytes, &norm, &crate::inversion::TileConfig { core, halo }, progress,
        ),
        None => crate::inversion::qsmnet(local_field_ppm, mask, grid, &bytes, &norm),
    }
    .map_err(|e| PipelineError::AlgorithmError(e.to_string()))
}

#[cfg(not(feature = "onnx"))]
fn run_qsmnet(
    _local_field_ppm: &[f64],
    _mask: &[u8],
    _grid: &crate::Grid,
    _model_id: &str,
    _tile: Option<(usize, usize)>,
    _progress: &mut dyn FnMut(usize, usize),
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

/// Source the QSMGAN generator weights and run inference (requires the `onnx` feature).
#[cfg(feature = "onnx")]
fn run_qsmgan(local_field_ppm: &[f64], mask: &[u8], grid: &crate::Grid) -> Result<Vec<f64>, PipelineError> {
    let bytes = crate::models::primary_weight("qsmgan").map_err(PipelineError::InvalidConfig)?;
    crate::inversion::qsmgan(local_field_ppm, mask, grid, &bytes)
        .map_err(|e| PipelineError::AlgorithmError(e.to_string()))
}

/// Source the IR2QSM weights and run inference (requires the `onnx` feature).
#[cfg(feature = "onnx")]
fn run_ir2qsm(
    local_field_ppm: &[f64],
    mask: &[u8],
    grid: &crate::Grid,
    tile: Option<(usize, usize)>,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Vec<f64>, PipelineError> {
    let bytes = crate::models::primary_weight("ir2qsm").map_err(PipelineError::InvalidConfig)?;
    match tile {
        Some((core, halo)) => crate::inversion::ir2qsm_tiled(
            local_field_ppm, mask, grid, &bytes, &crate::inversion::TileConfig { core, halo }, progress,
        ),
        None => crate::inversion::ir2qsm(local_field_ppm, mask, grid, &bytes),
    }
    .map_err(|e| PipelineError::AlgorithmError(e.to_string()))
}

/// Source the LPCNN proximal-CNN weights and run inference (requires the `onnx` feature).
/// LPCNN's k-space data-consistency step uses the B0 direction.
#[cfg(feature = "onnx")]
fn run_lpcnn(
    local_field_ppm: &[f64],
    mask: &[u8],
    grid: &crate::Grid,
    bdir: (f64, f64, f64),
    tile: Option<(usize, usize)>,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Vec<f64>, PipelineError> {
    let bytes = crate::models::primary_weight("lpcnn").map_err(PipelineError::InvalidConfig)?;
    match tile {
        Some((core, halo)) => crate::inversion::lpcnn_tiled(
            local_field_ppm, mask, grid, bdir, &bytes, &crate::inversion::TileConfig { core, halo }, progress,
        ),
        None => crate::inversion::lpcnn(local_field_ppm, mask, grid, bdir, &bytes),
    }
    .map_err(|e| PipelineError::AlgorithmError(e.to_string()))
}

/// Source the MoDL-QSM prior-CNN weights and run inference (requires the `onnx` feature).
/// Output is the STI χ33 component.
#[cfg(feature = "onnx")]
fn run_modl_qsm(
    local_field_ppm: &[f64],
    mask: &[u8],
    grid: &crate::Grid,
    bdir: (f64, f64, f64),
    tile: Option<(usize, usize)>,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Vec<f64>, PipelineError> {
    let bytes = crate::models::primary_weight("modl-qsm").map_err(PipelineError::InvalidConfig)?;
    match tile {
        Some((core, halo)) => crate::inversion::modl_qsm_tiled(
            local_field_ppm, mask, grid, bdir, &bytes, &crate::inversion::TileConfig { core, halo }, progress,
        ),
        None => crate::inversion::modl_qsm(local_field_ppm, mask, grid, bdir, &bytes),
    }
    .map_err(|e| PipelineError::AlgorithmError(e.to_string()))
}

#[cfg(not(feature = "onnx"))]
fn run_qsmgan(_f: &[f64], _m: &[u8], _g: &crate::Grid) -> Result<Vec<f64>, PipelineError> {
    Err(PipelineError::InvalidConfig("QSMGAN requires building qsm-core with the 'onnx' feature".into()))
}
#[cfg(not(feature = "onnx"))]
fn run_ir2qsm(
    _f: &[f64],
    _m: &[u8],
    _g: &crate::Grid,
    _tile: Option<(usize, usize)>,
    _progress: &mut dyn FnMut(usize, usize),
) -> Result<Vec<f64>, PipelineError> {
    Err(PipelineError::InvalidConfig("IR2QSM requires building qsm-core with the 'onnx' feature".into()))
}
#[cfg(not(feature = "onnx"))]
fn run_lpcnn(
    _f: &[f64],
    _m: &[u8],
    _g: &crate::Grid,
    _b: (f64, f64, f64),
    _tile: Option<(usize, usize)>,
    _progress: &mut dyn FnMut(usize, usize),
) -> Result<Vec<f64>, PipelineError> {
    Err(PipelineError::InvalidConfig("LPCNN requires building qsm-core with the 'onnx' feature".into()))
}
#[cfg(not(feature = "onnx"))]
fn run_modl_qsm(_f: &[f64], _m: &[u8], _g: &crate::Grid, _b: (f64, f64, f64), _tile: Option<(usize, usize)>, _progress: &mut dyn FnMut(usize, usize)) -> Result<Vec<f64>, PipelineError> {
    Err(PipelineError::InvalidConfig("MoDL-QSM requires building qsm-core with the 'onnx' feature".into()))
}

/// The authors' fixed inference conventions for the iQSM/iQFM LoT-Unet family:
/// phase sign `-1` and a 3-voxel mask-erosion radius. These are training-time
/// constants, not user knobs, so the pipeline runners bake them in.
#[cfg(feature = "onnx")]
const IQSM_PHASE_SIGN: f64 = -1.0;
#[cfg(feature = "onnx")]
const IQSM_ERODED_RAD: i32 = 3;

/// Run iQSM single-step reconstruction from wrapped **phase** (joint unwrap +
/// background removal + dipole inversion). Multi-echo inputs are reconstructed
/// per echo and magnitude·TE²-combined. Requires the `onnx` feature + `iqsm` weights.
///
/// `phases`/`magnitudes` are per-echo wrapped-phase / magnitude volumes (column-major);
/// `metadata` supplies the grid, echo times, and B0. Returns susceptibility (ppm),
/// referenced per `reference`.
#[cfg(feature = "onnx")]
pub fn run_iqsm(
    phases: &[&[f64]],
    magnitudes: &[&[f64]],
    mask: &[u8],
    metadata: &ScanMetadata,
    reference: QsmReference,
    tile: Option<(usize, usize)>,
) -> Result<Vec<f64>, PipelineError> {
    metadata.require_contiguous_slices()?;
    let grid = metadata.grid();
    let bytes = crate::models::primary_weight("iqsm").map_err(PipelineError::InvalidConfig)?;
    let tile = tile.map(|(core, halo)| crate::inversion::TileConfig { core, halo });
    let chi = crate::inversion::iqsm_multi_echo(
        phases, magnitudes, mask, &grid, &metadata.echo_times,
        metadata.field_strength, IQSM_PHASE_SIGN, IQSM_ERODED_RAD, &bytes, tile.as_ref(),
    ).map_err(|e| PipelineError::AlgorithmError(e.to_string()))?;
    Ok(super::referencing::apply_reference(&chi, mask, reference))
}

#[cfg(not(feature = "onnx"))]
pub fn run_iqsm(
    _phases: &[&[f64]], _magnitudes: &[&[f64]], _mask: &[u8],
    _metadata: &ScanMetadata, _reference: QsmReference, _tile: Option<(usize, usize)>,
) -> Result<Vec<f64>, PipelineError> {
    Err(PipelineError::InvalidConfig("iQSM requires building qsm-core with the 'onnx' feature".into()))
}

/// Run iQSM+ single-step reconstruction from wrapped **phase** (orientation-adaptive
/// variant; the B0 direction is a genuine network input). See [`run_iqsm`].
#[cfg(feature = "onnx")]
pub fn run_iqsm_plus(
    phases: &[&[f64]],
    magnitudes: &[&[f64]],
    mask: &[u8],
    metadata: &ScanMetadata,
    reference: QsmReference,
    tile: Option<(usize, usize)>,
) -> Result<Vec<f64>, PipelineError> {
    metadata.require_contiguous_slices()?;
    let grid = metadata.grid();
    let bytes = crate::models::primary_weight("iqsm-plus").map_err(PipelineError::InvalidConfig)?;
    let tile = tile.map(|(core, halo)| crate::inversion::TileConfig { core, halo });
    let chi = crate::inversion::iqsm_plus_multi_echo(
        phases, magnitudes, mask, &grid, &metadata.echo_times,
        metadata.field_strength, metadata.b0_direction, IQSM_PHASE_SIGN, IQSM_ERODED_RAD, &bytes,
        tile.as_ref(),
    ).map_err(|e| PipelineError::AlgorithmError(e.to_string()))?;
    Ok(super::referencing::apply_reference(&chi, mask, reference))
}

#[cfg(not(feature = "onnx"))]
pub fn run_iqsm_plus(
    _phases: &[&[f64]], _magnitudes: &[&[f64]], _mask: &[u8],
    _metadata: &ScanMetadata, _reference: QsmReference, _tile: Option<(usize, usize)>,
) -> Result<Vec<f64>, PipelineError> {
    Err(PipelineError::InvalidConfig("iQSM+ requires building qsm-core with the 'onnx' feature".into()))
}

/// Run iQFM: joint unwrapping + background removal from wrapped **phase** in one
/// network (the tissue-field head of the iQSM LoT-Unet). Returns the local
/// (background-removed) field in ppm — feed it to any dipole inversion. This is a
/// deep-learning replacement for the unwrap + BFR stages, not an inversion.
/// Requires the `onnx` feature + `iqfm` weights.
#[cfg(feature = "onnx")]
pub fn run_iqfm(
    phases: &[&[f64]],
    magnitudes: &[&[f64]],
    mask: &[u8],
    metadata: &ScanMetadata,
    tile: Option<(usize, usize)>,
) -> Result<Vec<f64>, PipelineError> {
    metadata.require_contiguous_slices()?;
    let grid = metadata.grid();
    let bytes = crate::models::primary_weight("iqfm").map_err(PipelineError::InvalidConfig)?;
    let tile = tile.map(|(core, halo)| crate::inversion::TileConfig { core, halo });
    crate::inversion::iqfm_multi_echo(
        phases, magnitudes, mask, &grid, &metadata.echo_times,
        metadata.field_strength, IQSM_PHASE_SIGN, IQSM_ERODED_RAD, &bytes, tile.as_ref(),
    ).map_err(|e| PipelineError::AlgorithmError(e.to_string()))
}

#[cfg(not(feature = "onnx"))]
pub fn run_iqfm(
    _phases: &[&[f64]], _magnitudes: &[&[f64]], _mask: &[u8], _metadata: &ScanMetadata,
    _tile: Option<(usize, usize)>,
) -> Result<Vec<f64>, PipelineError> {
    Err(PipelineError::InvalidConfig("iQFM requires building qsm-core with the 'onnx' feature".into()))
}

/// Run TGV single-step QSM reconstruction.
///
/// Multi-echo data goes through field mapping first and TGV runs on the field in ppm
/// ([`TgvInput::FieldPpm`](crate::inversion::TgvInput::FieldPpm)). Single-echo data goes in
/// as the wrapped phase with its echo time and the scan's field strength
/// ([`TgvInput::WrappedPhase`](crate::inversion::TgvInput::WrappedPhase)), and TGV handles
/// unwrapping as well. Either way TGV also does background removal and inversion.
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
    use crate::inversion::TgvInput;

    metadata.require_contiguous_slices()?;
    let grid = metadata.grid();
    let (bx, by, bz) = metadata.b0_direction;

    let chi_ppm = if phases.len() > 1 {
        let field_result = super::field_mapping::run_field_mapping(
            phases, magnitudes, mask, metadata,
            field_mapping_config, &mut |_, _| {},
        )?;
        crate::inversion::tgv_qsm(
            TgvInput::FieldPpm(&field_result.b0_field_ppm),
            mask, &grid, tgv_params, (bx, by, bz), &mut *progress,
        )?
    } else {
        let te_s = *metadata.echo_times.first().ok_or_else(|| {
            PipelineError::InvalidInput("TGV needs the echo time of the phase".into())
        })?;
        let b0_t = metadata.field_strength;
        if !(te_s.is_finite() && te_s > 0.0 && b0_t.is_finite() && b0_t > 0.0) {
            return Err(PipelineError::InvalidInput(format!(
                "single-echo TGV needs a positive echo time and field strength, got TE = {te_s} s, B0 = {b0_t} T"
            )));
        }
        let input = TgvInput::WrappedPhase {
            phase: phases[0],
            te_s,
            b0_t,
        };
        crate::inversion::tgv_qsm(input, mask, &grid, tgv_params, (bx, by, bz), &mut *progress)?
    };

    Ok(super::referencing::apply_reference(&chi_ppm, mask, reference))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Calls the stage, rather than inspecting its source as the coverage test in `config.rs`
    /// does, so the refusal is proven to happen for a real inversion.
    /// NeXtQSM is reachable both as an `InversionAlgorithm` and as a standalone entry point, so
    /// both ways in have to refuse a gap. This covers the dispatcher route, which works in any
    /// build; the standalone route needs the `onnx` feature, because without it `run_nextqsm`
    /// is the stub and reporting the missing feature is more use than reporting the gap. What
    /// guarantees the standalone route in a default build is the source-level coverage test in
    /// `config.rs`, which reads the `onnx` body and requires the call to be there.
    ///
    /// Note what this does *not* pin: the NeXtQSM route is now defended twice, once in the
    /// dispatcher and once inside `run_nextqsm`, so deleting either one on its own leaves this
    /// test passing. Checked by perturbation — removing the dispatcher's call was caught only
    /// by the coverage test and by the default-algorithm test below, which has no second line
    /// of defence behind it. Both call sites are pinned by the coverage test, not by this one.
    #[test]
    fn nextqsm_refuses_a_gapped_acquisition_through_the_dispatcher() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let config = InversionConfig {
            algorithm: InversionAlgorithm::Nextqsm,
            ..InversionConfig::default()
        };
        let err = run_dipole_inversion(
            &vec![0.01; n], &vec![1u8; n], &gapped_meta(nx, ny, nz), &config, None, &mut |_, _| {},
        )
        .expect_err("a gapped volume must be refused");
        let PipelineError::InvalidInput(msg) = &err else { panic!("{err:?}") };
        assert!(msg.contains("1.0000 mm gap"), "{msg}");
    }

    /// The standalone entry point. Needs `onnx` for the real function to exist, but still does
    /// no I/O: the check runs before the weight registry is touched, which is the whole point
    /// of refusing before any work.
    #[cfg(feature = "onnx")]
    #[test]
    fn nextqsm_refuses_a_gapped_acquisition_standalone() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let err = run_nextqsm(
            &vec![0.01; n], &vec![1u8; n], &gapped_meta(nx, ny, nz), None, &mut |_, _| {},
        )
        .expect_err("a gapped volume must be refused");
        let PipelineError::InvalidInput(msg) = &err else { panic!("{err:?}") };
        assert!(msg.contains("1.0000 mm gap"), "{msg}");
    }

    /// 3 mm pitch with 2 mm slices: a 1 mm gap.
    fn gapped_meta(nx: usize, ny: usize, nz: usize) -> ScanMetadata {
        ScanMetadata {
            dims: (nx, ny, nz),
            voxel_size: (1.0, 1.0, 3.0),
            echo_times: vec![0.005],
            field_strength: 3.0,
            b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: Some(crate::grid::SliceGeometry { thickness: 2.0, axis: 2 }),
            affine: None,
        }
    }

    #[test]
    fn a_gapped_acquisition_is_refused_before_any_work() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let field = vec![0.01; n];
        let mask = vec![1u8; n];
        let mut meta = ScanMetadata {
            dims: (nx, ny, nz),
            voxel_size: (1.0, 1.0, 3.0),
            echo_times: vec![0.005],
            field_strength: 3.0,
            b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: None,
            affine: None,
        };
        let config = InversionConfig::default();

        assert!(run_dipole_inversion(&field, &mask, &meta, &config, None, &mut |_, _| {}).is_ok());
        meta.slice_geometry = Some(crate::grid::SliceGeometry { thickness: 3.0, axis: 2 });
        assert!(run_dipole_inversion(&field, &mask, &meta, &config, None, &mut |_, _| {}).is_ok());

        meta.slice_geometry = Some(crate::grid::SliceGeometry { thickness: 2.0, axis: 2 });
        let err = run_dipole_inversion(&field, &mask, &meta, &config, None, &mut |_, _| {})
            .expect_err("a gapped volume must be refused");
        let PipelineError::InvalidInput(msg) = &err else { panic!("{err:?}") };
        assert!(msg.contains("1.0000 mm gap"), "{msg}");
    }

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
            slice_geometry: None,
            affine: None,
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
        make_inversion_test_with(InversionConfig { algorithm: alg, ..Default::default() })
    }

    fn make_inversion_test_with(config: InversionConfig) -> Vec<f64> {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let field = vec![0.01; n];
        let mask = vec![1u8; n];
        let meta = ScanMetadata {
            dims: (nx, ny, nz), voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005], field_strength: 3.0, b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: None,
            affine: None,
        };
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
        let n = 8 * 8 * 8;
        let field = vec![0.01; n];
        let mask = vec![1u8; n];
        let mag = vec![1.0; n];
        let meta = ScanMetadata {
            dims: (8, 8, 8), voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005], field_strength: 3.0, b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: None,
            affine: None,
        };
        let config = InversionConfig { algorithm: InversionAlgorithm::Nltv, ..Default::default() };
        let chi = run_dipole_inversion(&field, &mask, &meta, &config, Some(&mag), &mut |_, _| {}).unwrap();
        assert_eq!(chi.len(), n);
        // the magnitude is required
        let r = run_dipole_inversion(&field, &mask, &meta, &config, None, &mut |_, _| {});
        assert!(matches!(r, Err(PipelineError::InvalidInput(_))));
    }

    #[test]
    fn test_inversion_ilsqr() {
        let ilsqr = crate::inversion::IlsqrParams { max_iter: 3, ..Default::default() };
        let chi = make_inversion_test_with(InversionConfig { algorithm: InversionAlgorithm::Ilsqr, ilsqr, ..Default::default() });
        assert_eq!(chi.len(), 8 * 8 * 8);
    }

    #[test]
    fn test_inversion_ilsqr_rejects_bad_padding() {
        let meta = ScanMetadata {
            dims: (8, 8, 8), voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005], field_strength: 3.0, b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: None,
            affine: None,
        };
        for pad_mm in [-1.0, f64::NAN, f64::INFINITY] {
            let ilsqr = crate::inversion::IlsqrParams { pad_mm, ..Default::default() };
            let config = InversionConfig { algorithm: InversionAlgorithm::Ilsqr, ilsqr, ..Default::default() };
            let r = run_dipole_inversion(&[0.01; 512], &[1u8; 512], &meta, &config, None, &mut |_, _| {});
            assert!(matches!(r, Err(PipelineError::InvalidConfig(_))), "pad {pad_mm} accepted");
        }
    }

    #[test]
    fn test_inversion_star() {
        let star = crate::inversion::StarParams { pad: 4, ..Default::default() };
        let chi = make_inversion_test_with(InversionConfig { algorithm: InversionAlgorithm::Star, star, ..Default::default() });
        assert_eq!(chi.len(), 8 * 8 * 8);
        assert!(chi.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_inversion_star_rejects_a_bad_field_strength() {
        let meta = ScanMetadata {
            dims: (8, 8, 8), voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005], field_strength: 0.0, b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: None,
            affine: None,
        };
        let config = InversionConfig { algorithm: InversionAlgorithm::Star, ..Default::default() };
        let r = run_dipole_inversion(&[0.01; 512], &[1u8; 512], &meta, &config, None, &mut |_, _| {});
        assert!(matches!(r, Err(PipelineError::InvalidConfig(_))));
    }

    #[test]
    fn test_fansi_family_requires_magnitude() {
        let meta = ScanMetadata {
            dims: (8, 8, 8), voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005], field_strength: 3.0, b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: None,
            affine: None,
        };
        for algorithm in [InversionAlgorithm::Fansi, InversionAlgorithm::FansiTgv, InversionAlgorithm::L1qsm] {
            let mut config = InversionConfig { algorithm, ..Default::default() };
            config.fansi.max_iter = 2;
            config.l1qsm.max_iter = 2;
            let r = run_dipole_inversion(&[0.01; 512], &[1u8; 512], &meta, &config, None, &mut |_, _| {});
            assert!(matches!(r, Err(PipelineError::InvalidInput(_))), "{algorithm:?} ran without magnitude");
            let mag = [1.0; 512];
            let r = run_dipole_inversion(&[0.01; 512], &[1u8; 512], &meta, &config, Some(&mag), &mut |_, _| {});
            assert!(r.is_ok(), "{algorithm:?} failed with magnitude");
        }
    }

    #[test]
    fn test_inversion_amp_pe() {
        let chi = make_inversion_test(InversionAlgorithm::AmpPe);
        assert_eq!(chi.len(), 8 * 8 * 8);
    }

    #[test]
    fn test_inversion_lsqr() {
        let chi = make_inversion_test(InversionAlgorithm::Lsqr);
        assert_eq!(chi.len(), 8 * 8 * 8);
        assert!(chi.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_inversion_heidi() {
        let chi = make_inversion_test(InversionAlgorithm::Heidi);
        assert_eq!(chi.len(), 8 * 8 * 8);
        assert!(chi.iter().all(|v| v.is_finite()));
    }

    /// HEIDI runs LSQR first, so its progress must advance monotonically across
    /// both stages and end at the combined total rather than restarting.
    #[test]
    fn test_heidi_progress_is_monotonic() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let meta = ScanMetadata {
            dims: (nx, ny, nz), voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005], field_strength: 3.0, b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: None,
            affine: None,
        };
        let config = InversionConfig {
            algorithm: InversionAlgorithm::Heidi,
            heidi: crate::inversion::HeidiParams {
                continuation_steps: 2,
                inner_iterations: 4,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut seen: Vec<(usize, usize)> = Vec::new();
        run_dipole_inversion(
            &vec![0.01; n], &vec![1u8; n], &meta, &config, None,
            &mut |i, t| seen.push((i, t)),
        )
        .unwrap();

        assert!(!seen.is_empty(), "no progress was reported");
        assert!(
            seen.windows(2).all(|w| w[0].0 <= w[1].0),
            "progress went backwards: {seen:?}"
        );
        let total = seen[0].1;
        assert!(seen.iter().all(|&(_, t)| t == total), "total changed mid-run");
        assert!(seen.last().unwrap().0 <= total);
    }

    /// MEDI gets the ppm field, the field strength and the echo spacing (MEDI's delta_TE:
    /// TE(2) - TE(1), or TE for one echo), and needs the magnitude.
    #[test]
    fn test_inversion_medi() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let field: Vec<f64> = (0..n).map(|i| 0.01 * ((i * 7) % 5) as f64).collect();
        let mask = vec![1u8; n];
        let mag: Vec<f64> = (0..n).map(|i| 1.0 + ((i * 3) % 7) as f64).collect();
        let medi = crate::inversion::MediParams { max_iter: 2, cg_max_iter: 5, smv: false, ..Default::default() };
        let config = InversionConfig { algorithm: InversionAlgorithm::Medi, medi: medi.clone(), ..Default::default() };
        use crate::inversion::EchoMagnitudes;
        let echoes: Vec<f64> = (0..3 * n).map(|i| 1.0 + ((i * 5) % 9) as f64).collect();
        let cases: [(Vec<f64>, f64, &[f64], usize); 2] = [
            (vec![0.004, 0.010, 0.016], 0.006, &echoes, 3),
            (vec![0.02], 0.02, &mag, 1),
        ];
        for (tes, delta_te, given, n_echoes) in cases {
            let meta = ScanMetadata {
                dims: (nx, ny, nz), voxel_size: (1.0, 1.0, 1.0),
                echo_times: tes, field_strength: 7.0, b0_direction: (0.0, 0.0, 1.0),
                slice_geometry: None,
                affine: None,
            };
            let via_stage = run_dipole_inversion(&field, &mask, &meta, &config, Some(given), &mut |_, _| {}).unwrap();
            let params = crate::inversion::MediParams { b0: 7.0, te: delta_te, ..medi.clone() };
            let direct = crate::inversion::medi(
                &field, &mask, EchoMagnitudes { data: given, n_echoes }, &meta.grid(), meta.b0_direction, &params, |_, _| {},
            );
            assert_eq!(via_stage, direct);
            assert!(run_dipole_inversion(&field, &mask, &meta, &config, None, &mut |_, _| {}).is_err());
            assert!(run_dipole_inversion(&field, &mask, &meta, &config, Some(&given[..n - 1]), &mut |_, _| {}).is_err());
        }
        // a combined magnitude of multi-echo data, or uneven echoes, are refused
        let meta = |tes: Vec<f64>| ScanMetadata {
            dims: (nx, ny, nz), voxel_size: (1.0, 1.0, 1.0),
            echo_times: tes, field_strength: 7.0, b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: None,
            affine: None,
        };
        assert!(run_dipole_inversion(&field, &mask, &meta(vec![0.004, 0.010, 0.016]), &config, Some(&mag), &mut |_, _| {}).is_err());
        assert!(run_dipole_inversion(&field, &mask, &meta(vec![0.004, 0.010, 0.020]), &config, Some(&echoes), &mut |_, _| {}).is_err());
    }

    /// TFI is a total-field method (`InversionConfig::input`): the dispatcher must hand the
    /// field to `tfi` in ppm, untouched (no ppm→rad scaling as for MEDI), with the brain mask.
    #[test]
    fn tfi_gets_the_total_field_unchanged() {
        let (nx, ny, nz) = (10, 10, 10);
        let n = nx * ny * nz;
        // a background ramp plus a local bump: a total field
        let field: Vec<f64> = (0..n)
            .map(|i| {
                let (x, y, z) = ((i % nx) as f64, ((i / nx) % ny) as f64, (i / (nx * ny)) as f64);
                0.5 * (z - 4.5) / 4.5 + 0.02 * (-((x - 4.5).powi(2) + (y - 4.5).powi(2) + (z - 4.5).powi(2)) / 4.0).exp()
            })
            .collect();
        let mask: Vec<u8> = (0..n)
            .map(|i| {
                let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
                u8::from((2..8).contains(&x) && (2..8).contains(&y) && (2..8).contains(&z))
            })
            .collect();
        let meta = ScanMetadata {
            dims: (nx, ny, nz), voxel_size: (1.0, 1.0, 1.0),
            echo_times: vec![0.005, 0.010], field_strength: 3.0, b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: None,
            affine: None,
        };
        let config = InversionConfig { algorithm: InversionAlgorithm::Tfi, ..Default::default() };
        assert_eq!(config.input(2), InversionInput::TotalField);
        let tfi = crate::inversion::TfiParams { max_iter: 2, cg_max_iter: 5, ..config.tfi.clone() };
        let config = InversionConfig { tfi: tfi.clone(), ..config };
        let via_stage = run_dipole_inversion(&field, &mask, &meta, &config, None, &mut |_, _| {}).unwrap();
        let direct = crate::inversion::tfi(
            &field, &vec![1.0; n], &vec![1.0; n], &mask, &meta.grid(), meta.b0_direction, &tfi, |_, _| {},
        );
        assert_eq!(via_stage, direct);
        assert!(via_stage.iter().all(|v| v.is_finite()));
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
            slice_geometry: None,
            affine: None,
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

//! Field mapping stage
//!
//! Multi-echo phase → B0 field map in ppm.
//! Implements the canonical field mapping pipeline:
//! phase offset removal → bipolar correction → unwrapping → B0 estimation,
//! or MEDI's: complex fit of the wrapped echoes → unwrapping.
//!
//! # Reference pipeline
//!
//! The default ([`FieldMappingConfig::default`]: MCPC-3D-S offset removal, ROMEO template
//! unwrapping, `:phase_snr` weighted average) is the ROMEO authors' B0 pipeline: the `romeo` CLI
//! with `--compute-B0` turns MCPC-3D-S on for multi-echo data (ROMEO.jl 08776d4,
//! `ext/RomeoApp/caller.jl:80-91`), unwraps temporally from echo 1 (`argparse.jl:101-105`) and
//! combines with `phase_snr` (`argparse.jl:67-70`; MriResearchTools 771eadf
//! `src/romeofunctions.jl:13-24`), whose docstring requires the phase offsets to have been
//! removed first (`romeofunctions.jl:3-7`). Where QSM.rs differs:
//!
//! - **Smoothing sigma.** `phase_offset_sigma_mm` defaults to the ROMEO command line's 7 mm per
//!   axis, divided by the voxel size (`argparse.jl:79-91`, `caller.jl:137-141`); MriResearchTools'
//!   `mcpc3ds` function itself defaults to `[10, 10, 5]` voxels (`src/mcpc3ds.jl:17,39`). The
//!   command line says to reduce it on a small field of view and to use 0 along an axis with a
//!   single slice; that is left to the caller.
//! - **Masks.** MCPC-3D-S makes its own `robustmask` of `sqrt(|HIP|)`, as the reference
//!   `mcpc3ds` does (`mcpc3ds.jl:44-46`; [`crate::utils::phase_offset_removal`]). ROMEO and the
//!   bipolar fit use the brain mask passed in, as the CLI does when given one (`-k`); without
//!   one the CLI unwraps inside `robustmask` of the template magnitude (`caller.jl:201-204`).
//! - **ROMEO settings.** The default is ROMEO.jl's API defaults
//!   ([`RomeoParams::default`](crate::unwrap::RomeoParams::default): `:romeo` weights,
//!   `temporal_uncertain_unwrapping` 0.5). The CLI turns `-w romeo` into `romeo3` when a
//!   magnitude is given and does no temporal re-unwrapping;
//!   [`RomeoParams::romeo_cli`](crate::unwrap::RomeoParams::romeo_cli) sets those, and with it
//!   the result matches the CLI up to the CLI's single-precision arithmetic (see
//!   [`crate::unwrap::romeo`], *Agreement with the reference*).
//! - **Offset removal off, or Laplacian unwrapping.** Phase-offset removal and bipolar correction
//!   run only with ROMEO (`do_offset` below); with Laplacian they are skipped whatever the
//!   config says. Without them each echo is unwrapped on its own (the template and `individual`
//!   settings of `romeo_params` are not used), with echo 1's magnitude and the next echo as
//!   `phase2` (none for the last echo), and the per-echo constants aligned before combining (see
//!   `field_mapping_direct`). ROMEO.jl's individual mode uses each echo's own magnitude and the
//!   previous echo as `phase2` (echo 2 for echo 1) (`src/unwrapping.jl:151-160`).
//! - **Bipolar correction** is opt-in in the reference too (`argparse.jl:71-76`, "requires ≥ 3
//!   echoes"); with fewer than 3 echoes it is skipped here silently.
//!
//! Field strength enters only in the conversion to ppm; nothing assumes a resolution or B0
//! direction.
//!
//! # Pipeline rules
//!
//! The facts above that a configuration can depart from are published as rules in
//! [`crate::pipeline::rules::RULES`] (each with its strength, basis, reason and source): `field-mapping.phase-offsets`, `field-mapping.bipolar-needs-three-echoes`, `field-mapping.bipolar-not-with-laplacian`, `field-mapping.phase-offset-sigma`.

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
///
/// # UK Biobank's field map
///
/// On offset-free echoes (e.g. MCPC-3D-S coil-combined), Laplacian unwrapping with
/// `laplacian_solver: LaplacianSolver::Fft { pad: [64; 3] }` and
/// `b0_estimation: B0EstimationMethod::WeightedAvg { weighting: B0WeightType::AssumedDecay { t2star_s: 0.040 } }`
/// reproduces the UK Biobank QSM pipeline's field map: STI Suite's `MRPhaseUnwrap(mask .* φ)`
/// per echo, then `Σ Wφ / Σ W·TE` with `W = TE·exp(−TE/T2*)`. It matches up to a global
/// constant (each echo's masked mean is removed before combining, see the direct path below),
/// to ≈1e-14 rad in the combined phase on in-vivo two-echo data.
///
/// ```
/// use qsm_core::pipeline::{B0EstimationMethod, FieldMappingConfig, UnwrappingAlgorithm};
/// use qsm_core::unwrap::LaplacianSolver;
/// use qsm_core::utils::B0WeightType;
/// let ukb = FieldMappingConfig {
///     unwrapping_algorithm: UnwrappingAlgorithm::Laplacian,
///     laplacian_solver: LaplacianSolver::Fft { pad: [64; 3] },
///     b0_estimation: B0EstimationMethod::WeightedAvg {
///         weighting: B0WeightType::assumed_decay_default(),
///     },
///     ..Default::default()
/// };
/// # let _ = ukb;
/// ```
///
/// # MEDI's field map
///
/// [`B0EstimationMethod::ComplexFit`] runs MEDI's `Fit_ppm_complex` on the wrapped complex echoes,
/// unwraps the fitted phase per echo spacing spatially with `unwrapping_algorithm`, and divides
/// by the echo spacing. It needs `magnitudes` and evenly spaced echoes, and also returns the
/// field's noise map in [`FieldMappingResult::field_noise_ppm`] (MEDI's `N_std`):
///
/// ```
/// use qsm_core::fieldmap::FitPpmComplexParams;
/// use qsm_core::pipeline::{B0EstimationMethod, FieldMappingConfig};
/// let medi = FieldMappingConfig {
///     b0_estimation: B0EstimationMethod::ComplexFit(FitPpmComplexParams::default()),
///     ..Default::default()
/// };
/// # let _ = medi;
/// ```
///
/// # Motion-robust echo combination
///
/// [`B0EstimationMethod::RobustFit`] combines the echoes with
/// [`multi_echo_robust_fit`](crate::utils::multi_echo_robust_fit) on whichever path runs, so it
/// composes with ROMEO and Laplacian unwrapping and with phase-offset removal on or off. It
/// needs [`RobustFitParams::min_echoes`](crate::utils::RobustFitParams::min_echoes) echoes or
/// more (fewer is [`PipelineError::InvalidConfig`]) and returns its per-echo report in
/// [`FieldMappingResult::robust_fit`]:
///
/// ```
/// use qsm_core::pipeline::{B0EstimationMethod, FieldMappingConfig};
/// use qsm_core::utils::RobustFitParams;
/// let robust = FieldMappingConfig {
///     b0_estimation: B0EstimationMethod::RobustFit(RobustFitParams::default()),
///     ..Default::default()
/// };
/// # let _ = robust;
/// ```
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

    match &config.b0_estimation {
        B0EstimationMethod::WeightedAvg { weighting: B0WeightType::AssumedDecay { t2star_s } } => {
            if !(t2star_s.is_finite() && *t2star_s > 0.0) {
                return Err(PipelineError::InvalidConfig(format!(
                    "assumed T2* for the assumed-decay B0 weighting must be positive, got {} s", t2star_s
                )));
            }
        }
        B0EstimationMethod::RobustFit(params) => {
            params.validate().map_err(PipelineError::InvalidConfig)?;
            // The library falls back to the plain fit, silently, below this. Asking for the robust
            // fit and getting a plain one with an empty report is worse than being told.
            if n_echoes < params.min_echoes() {
                return Err(PipelineError::InvalidConfig(format!(
                    "the robust B0 fit needs at least {} echoes ({}), got {}; use the weighted \
                     average or the linear fit for this series",
                    params.min_echoes(),
                    if params.estimate_offset {
                        "4 with estimate_offset, which costs a degree of freedom"
                    } else {
                        "3 without estimate_offset"
                    },
                    n_echoes,
                )));
            }
        }
        B0EstimationMethod::QsmartEchofit { fit_threshold } => {
            fit_threshold.validate().map_err(PipelineError::InvalidConfig)?;
        }
        B0EstimationMethod::WeightedAvg { .. } | B0EstimationMethod::LinearFit(_) => {}
        // Not a combination of unwrapped echoes: the fit runs on the wrapped echoes and is
        // unwrapped after, so it has a path of its own.
        B0EstimationMethod::ComplexFit(params) => {
            return field_mapping_complex_fit(phases, magnitudes, mask, metadata, config, params, progress);
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
        phases, mag_slices, tes,
        config.phase_offset_sigma_voxels((vsx, vsy, vsz)), [0, 1],
        crate::unwrap::UnwrapMethod::Romeo,
        &grid,
    );

    // Step 2: Bipolar correction (optional, >= 3 echoes)
    if config.bipolar_correction && n_echoes >= 3 {
        crate::utils::bipolar_correction(
            &mut corrected, mag_slices, tes, mask,
            config.phase_offset_sigma_voxels((vsx, vsy, vsz)), &grid,
        );
    }

    // Step 3: Multi-echo unwrapping
    progress(2, 4);
    let unwrapped: Vec<Vec<f64>> = match config.unwrapping_algorithm {
        UnwrappingAlgorithm::Laplacian => {
            // Neumann, not the ROI-masked variant: this stage wants unwrapping only.
            // Background removal is a later stage, and the masked variant would remove it
            // here first — measurably worse than doing it once, properly. The phase is zeroed
            // outside the mask first (MCPC-3D-S corrects every voxel, and the Laplacian unwrap
            // takes the whole array), as on the direct path.
            corrected.iter()
                .map(|p| crate::unwrap::laplacian_unwrap(&mask_phase(p, mask), mask, &grid, config.laplacian_solver))
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
    let mut robust_fit = None;
    if let B0EstimationMethod::QsmartEchofit { fit_threshold } = config.b0_estimation {
        progress(4, 4);
        return Ok(FieldMappingResult { phase_offset: Some(phase_offset), ..qsmart_fit(&unwrapped, mag_slices, metadata, fit_threshold) });
    }
    let b0_hz = match &config.b0_estimation {
        B0EstimationMethod::WeightedAvg { weighting } => {
            crate::utils::calculate_b0_weighted(
                &unwrapped, mag_slices, tes, mask,
                *weighting, &grid,
            )
        }
        B0EstimationMethod::LinearFit(params) => {
            let uw_refs: Vec<&[f64]> = unwrapped.iter().map(|u| u.as_slice()).collect();
            let fit = crate::utils::multi_echo_linear_fit(
                &uw_refs, mag_slices, tes, mask,
                params.estimate_offset,
            );
            crate::utils::field_to_hz(&fit.field)
        }
        B0EstimationMethod::RobustFit(params) => {
            let (field_rads, report) = robust_fit_echoes(&unwrapped, mag_slices, tes, mask, params);
            robust_fit = Some(report);
            crate::utils::field_to_hz(&field_rads)
        }
        B0EstimationMethod::QsmartEchofit { .. } => unreachable!("returned above"),
        B0EstimationMethod::ComplexFit(_) => unreachable!("run_field_mapping takes the complex fit's own path"),
    };

    progress(4, 4);
    Ok(FieldMappingResult {
        b0_field_ppm: hz_to_ppm(&b0_hz, metadata.field_strength),
        phase_offset: Some(phase_offset),
        robust_fit,
        reliability: None,
        qsmart_fit_threshold: None,
        field_noise_ppm: None,
    })
}

/// Path B: Multi-echo without phase offset removal (per-echo unwrap + B0 estimation).
///
/// Taken whenever offset removal is off: always with Laplacian unwrapping, and with ROMEO when
/// `phase_offset_removal` is false. Each echo is unwrapped on its own (for Laplacian, with the
/// phase zeroed outside `mask` first unless `config.laplacian_unmasked`, and `config.laplacian_solver` as the Poisson solver), and the echoes are then combined with
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
///   Then `B0 = Σ w (φ/TE) / Σ w` with the variant's `weighting`.
///
///   This alignment is QSM.rs's own (kept, Ashley 2026-10-28): each echo's Laplacian unwrap is
///   determined only up to a constant of its own, so the constants are removed before the
///   echoes are combined, which loses only a global constant field. No reference combines
///   Laplacian-unwrapped echoes this way. The MATLAB pipelines QSM.rs reproduces with other
///   settings: UK Biobank (STI Suite `MRPhaseUnwrap` on the masked phase, then the T2*-weighted
///   average; [`LaplacianSolver::Fft`](crate::unwrap::LaplacianSolver::Fft) `{ pad: [64; 3] }` with
///   [`B0WeightType::AssumedDecay`], matched to ≈1e-14 rad up to a global constant, see
///   [`run_field_mapping`]), QSMART (Hongfu Sun's `lapunwrap` per echo on the raw phase, then
///   `echofit`; [`LaplacianSolver::Mirror`](crate::unwrap::LaplacianSolver::Mirror) with
///   `laplacian_unmasked` and [`B0EstimationMethod::QsmartEchofit`], which aligns nothing) and
///   STI Suite's GUI (`MRPhaseUnwrap` on the raw phase, echoes summed: `laplacian_unmasked` with
///   [`B0WeightType::TEs`]). QSM.jl's README unwraps the raw phase and inverts each echo's field
///   separately, so it has no echo combination to compare with.
///   Measured on the QSM-CI in-vivo travelling-head scan (Prisma 2.89 T, 1 mm, 5 echoes)
///   against QSM.jl's `unwrap_laplacian(…; solver = :dct)` with each echo scaled by its TE and
///   combined by QSM.jl's `multi_echo_average` (`W` = magnitude), in the QSM-CI `julia-qsm`
///   image (`sha256:b3c048ac…`): with the phase masked and each echo's masked mean removed on the
///   QSM.jl side too, the field maps agree to NRMSE 1.4e-9 % (float32 output rounding); with
///   QSM.jl's raw phase and no mean removal, NRMSE 46 % against this default and 28 % against
///   `laplacian_unmasked: true`, the rest being the per-echo constants this alignment removes.
/// * [`B0EstimationMethod::LinearFit`]: the magnitude-weighted fit of phase against TE with
///   the variant's [`LinearFitParams`](crate::utils::LinearFitParams). With ROMEO it runs on the
///   per-echo unwrapped phases as they come out of the unwrapper, which is what this path
///   computed unconditionally before it honoured `b0_estimation` (bit-identical). With Laplacian
///   the per-echo constants are aligned first, as for the weighted mean: a constant `cₑ` that
///   differs between echoes is not linear in TE, so the fit turns it into a slope error, and
///   because the fit is magnitude-weighted that error varies over the brain rather than being a
///   constant that background removal would take out.
///   With an intercept and two echoes the fit is the echo difference `(φ₂ − φ₁)/(TE₂ − TE₁)`,
///   which discards the absolute phase, so it is only worth choosing when the echoes carry a
///   phase offset that has not been removed.
/// * [`B0EstimationMethod::RobustFit`]: the per-echo constants are aligned exactly as for the
///   weighted mean (an unaligned constant is a departure from linearity in TE that the robust
///   loss would blame on the echo), then the robust fit runs on the aligned echoes. Keep
///   `estimate_offset` on unless the echoes are known to be offset-free (e.g. MCPC-3D-S
///   combined): the offset removal that would otherwise take the intercept out does not run here.
/// * [`B0EstimationMethod::QsmartEchofit`]: QSMART's `echofit.m` on the echoes as unwrapped. With
///   Laplacian nothing is aligned, as in QSMART (whose `unwrap_phase` → `echofit` removes no
///   per-echo constant), and with `laplacian_unmasked` every echo is kept over the whole volume,
///   because `echofit.m` fits every voxel and takes its `R_0` threshold from all of them. With
///   ROMEO, whole turns are aligned as for the weighted mean: a fit through the origin turns a
///   turn between echoes into a slope error.
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
    // `MRPhaseUnwrap(mask .* phase)` is called in the UK Biobank pipeline), unless
    // `laplacian_unmasked` asks for the raw phase (STI Suite's GUI, QSMART).
    progress(1, 4);
    let is_laplacian = config.unwrapping_algorithm == UnwrappingAlgorithm::Laplacian;
    let region = LaplacianRegion::new(config, mask);
    let mut unwrapped: Vec<Vec<f64>> = Vec::with_capacity(n_echoes);
    for e in 0..n_echoes {
        let masked;
        let phase: &[f64] = if region.zero_outside {
            masked = mask_phase(phases[e], mask);
            &masked
        } else {
            phases[e]
        };
        let uw = unwrap_single(
            phase, mag_slices.first().copied().unwrap_or(&[]),
            region.keep(mask), config, nx, ny, nz, vsx, vsy, vsz,
            if e + 1 < n_echoes { Some(phases[e + 1]) } else { None },
            tes[e],
            if e + 1 < n_echoes { tes[e + 1] } else { 0.0 },
        );
        unwrapped.push(uw);
    }

    progress(3, 4);
    if let B0EstimationMethod::QsmartEchofit { fit_threshold } = config.b0_estimation {
        if !is_laplacian {
            align_per_echo_constants(&mut unwrapped, tes, mask, config.unwrapping_algorithm);
        }
        progress(4, 4);
        return Ok(qsmart_fit(&unwrapped, mag_slices, metadata, fit_threshold));
    }
    let mut robust_fit = None;
    let b0_field_ppm = match &config.b0_estimation {
        B0EstimationMethod::WeightedAvg { weighting } => {
            align_per_echo_constants(&mut unwrapped, tes, mask, config.unwrapping_algorithm);
            let grid = crate::Grid::new(nx, ny, nz, vsx, vsy, vsz);
            let b0_hz = crate::utils::calculate_b0_weighted(
                &unwrapped, mag_slices, tes, mask,
                *weighting, &grid,
            );
            hz_to_ppm(&b0_hz, metadata.field_strength)
        }
        B0EstimationMethod::LinearFit(params) => {
            // Laplacian only: ROMEO's whole-turn alignment would change what this path has
            // always computed for it.
            if is_laplacian {
                align_per_echo_constants(&mut unwrapped, tes, mask, config.unwrapping_algorithm);
            }
            let uw_refs: Vec<&[f64]> = unwrapped.iter().map(|u| u.as_slice()).collect();
            let fit = crate::utils::multi_echo_linear_fit(
                &uw_refs, mag_slices, tes, mask,
                params.estimate_offset,
            );
            rads_to_ppm(&fit.field, metadata.field_strength)
        }
        B0EstimationMethod::RobustFit(params) => {
            // Aligned first, as for the weighted mean. A per-echo constant left by the unwrapper is
            // exactly what the robust loss is built to see as a bad echo — a departure from a
            // straight line in TE at every voxel — so unaligned echoes would be down-weighted for
            // the unwrapper's convention, not for anything in the data. Both alignments keep a
            // series that is linear in TE linear in TE (Laplacian: the demeaned echo
            // `φ₀ − φ̄₀ + TEₑ(ω − ω̄)` still has one intercept and one slope; ROMEO: whole turns
            // only, and a no-op on consistent echoes), so the fit itself loses nothing but a
            // global constant field.
            align_per_echo_constants(&mut unwrapped, tes, mask, config.unwrapping_algorithm);
            let (field_rads, report) = robust_fit_echoes(&unwrapped, mag_slices, tes, mask, params);
            robust_fit = Some(report);
            rads_to_ppm(&field_rads, metadata.field_strength)
        }
        B0EstimationMethod::QsmartEchofit { .. } => unreachable!("returned above"),
        B0EstimationMethod::ComplexFit(_) => unreachable!("run_field_mapping takes the complex fit's own path"),
    };

    progress(4, 4);
    Ok(FieldMappingResult {
        b0_field_ppm,
        phase_offset: None,
        robust_fit,
        field_noise_ppm: None,
        reliability: None,
        qsmart_fit_threshold: None,
    })
}

/// [`B0EstimationMethod::QsmartEchofit`] on unwrapped echoes: QSMART's `echofit.m`
/// ([`qsmart_echofit`](crate::utils::qsmart_echofit)) with QSMART's γ, giving the total field
/// over every voxel and the reliability map `R_0`.
fn qsmart_fit(
    unwrapped: &[Vec<f64>],
    mag_slices: &[&[f64]],
    metadata: &ScanMetadata,
    fit_threshold: crate::utils::QsmartFitThreshold,
) -> FieldMappingResult {
    let uw: Vec<&[f64]> = unwrapped.iter().map(|u| u.as_slice()).collect();
    let ppm = crate::utils::QSMART_GYRO * metadata.field_strength / 1e6;
    let fit = crate::utils::qsmart_echofit(&uw, mag_slices, &metadata.echo_times, &metadata.grid(), ppm, fit_threshold);
    FieldMappingResult {
        b0_field_ppm: fit.field_ppm,
        phase_offset: None,
        robust_fit: None,
        reliability: Some(fit.reliability),
        qsmart_fit_threshold: Some(fit.fit_threshold),
        field_noise_ppm: None,
    }
}

/// Where Laplacian unwrapping works ([`FieldMappingConfig::laplacian_unmasked`]): on the phase
/// zeroed outside the mask, keeping the result inside it (the default); or on the raw phase.
/// The raw phase's result is kept over the whole volume only for QSMART's echo fit, which fits
/// every voxel (`echofit.m`); the other combinations use only the voxels inside the mask.
struct LaplacianRegion {
    zero_outside: bool,
    everywhere: Option<Vec<u8>>,
}

impl LaplacianRegion {
    fn new(config: &FieldMappingConfig, mask: &[u8]) -> Self {
        let unmasked = config.unwrapping_algorithm == UnwrappingAlgorithm::Laplacian && config.laplacian_unmasked;
        let echofit = matches!(config.b0_estimation, B0EstimationMethod::QsmartEchofit { .. });
        LaplacianRegion {
            zero_outside: config.unwrapping_algorithm == UnwrappingAlgorithm::Laplacian && !unmasked,
            everywhere: (unmasked && echofit).then(|| vec![1; mask.len()]),
        }
    }

    /// The mask the unwrapped result is kept in.
    fn keep<'a>(&'a self, mask: &'a [u8]) -> &'a [u8] {
        self.everywhere.as_deref().unwrap_or(mask)
    }
}

/// [`multi_echo_robust_fit`](crate::utils::multi_echo_robust_fit) on unwrapped echoes: the field
/// in rad/s and the report. Echo count and params are checked by [`run_field_mapping`].
fn robust_fit_echoes(
    unwrapped: &[Vec<f64>],
    mag_slices: &[&[f64]],
    tes: &[f64],
    mask: &[u8],
    params: &crate::utils::RobustFitParams,
) -> (Vec<f64>, RobustFitReport) {
    let r = crate::utils::multi_echo_robust_fit(unwrapped, mag_slices, tes, mask, params);
    (r.fit.field, RobustFitReport { quality: r.quality, robust_weights: r.robust_weights })
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
    let unwrapped = unwrap_one_echo(phase, mag, mask, metadata.echo_times[0], config, nx, ny, nz, vsx, vsy, vsz);
    if let B0EstimationMethod::QsmartEchofit { fit_threshold } = config.b0_estimation {
        progress(4, 4);
        return Ok(qsmart_fit(&[unwrapped], &[mag], metadata, fit_threshold));
    }
    progress(4, 4);
    Ok(single_echo_result(&unwrapped, mask, metadata.echo_times[0], metadata.field_strength, config))
}

/// One echo's phase, unwrapped. Laplacian: as on the multi-echo path, the phase is zeroed outside
/// the mask first (a global Poisson solve otherwise carries the noise there into the brain; not
/// with `laplacian_unmasked`).
#[allow(clippy::too_many_arguments)]
fn unwrap_one_echo(
    phase: &[f64],
    mag: &[f64],
    mask: &[u8],
    te: f64,
    config: &FieldMappingConfig,
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    let region = LaplacianRegion::new(config, mask);
    let masked;
    let phase: &[f64] = if region.zero_outside {
        masked = mask_phase(phase, mask);
        &masked
    } else {
        phase
    };
    unwrap_single(phase, mag, region.keep(mask), config, nx, ny, nz, vsx, vsy, vsz, None, te, 0.0)
}

/// One echo's field from its unwrapped phase: the masked mean a Laplacian unwrap leaves
/// undetermined removed (so one echo gives the same field as each echo of a series does), then
/// divided by the echo time.
fn single_echo_result(
    unwrapped: &[f64],
    mask: &[u8],
    te: f64,
    field_strength: f64,
    config: &FieldMappingConfig,
) -> FieldMappingResult {
    let mut unwrapped = unwrapped.to_vec();
    if config.unwrapping_algorithm == UnwrappingAlgorithm::Laplacian {
        remove_masked_mean(&mut unwrapped, mask);
    }
    // field = unwrapped / TE → rad/s
    let field_rads: Vec<f64> = unwrapped.iter().map(|&v| v / te).collect();
    FieldMappingResult {
        b0_field_ppm: rads_to_ppm(&field_rads, field_strength),
        phase_offset: None,
        robust_fit: None,
        reliability: None,
        qsmart_fit_threshold: None,
        field_noise_ppm: None,
    }
}

/// The field of **each echo on its own**, for a per-echo reconstruction
/// ([`QsmPipelineConfig::qsm_per_echo`](super::config::QsmPipelineConfig::qsm_per_echo); see
/// [`super::per_echo`]): echo `e`'s phase is unwrapped with
/// [`FieldMappingConfig::unwrapping_algorithm`] (ROMEO with that echo's magnitude) and divided by
/// its own echo time, exactly as a single-echo series is ([`run_field_mapping`] with one echo).
/// The echo combination ([`FieldMappingConfig::b0_estimation`]), phase-offset removal and bipolar
/// correction combine or compare echoes, so they do not apply and are not run. This is the
/// DECOMPOSE authors' per-echo field: `MRPhaseUnwrap` of each echo's raw phase, then that echo's
/// background removal and STAR-QSM at its TE (Chen et al. 2021, NeuroImage 242:118477, §3.6).
///
/// Returns one [`FieldMappingResult`] per echo, in echo order.
pub fn run_field_mapping_per_echo(
    phases: &[&[f64]],
    magnitudes: Option<&[&[f64]]>,
    mask: &[u8],
    metadata: &ScanMetadata,
    config: &FieldMappingConfig,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Vec<FieldMappingResult>, PipelineError> {
    let (nx, ny, nz) = metadata.dims;
    let (vsx, vsy, vsz) = metadata.voxel_size;
    let n_voxels = nx * ny * nz;
    let n_echoes = phases.len();
    if n_echoes == 0 {
        return Err(PipelineError::InvalidInput("no phase echoes provided".into()));
    }
    if metadata.echo_times.len() != n_echoes {
        return Err(PipelineError::DimensionMismatch { expected: n_echoes, got: metadata.echo_times.len() });
    }
    if mask.len() != n_voxels {
        return Err(PipelineError::DimensionMismatch { expected: n_voxels, got: mask.len() });
    }
    let uniform = vec![1.0f64; n_voxels];
    let mut out = Vec::with_capacity(n_echoes);
    for (e, p) in phases.iter().enumerate() {
        let mag: &[f64] = magnitudes.and_then(|m| m.get(e).copied()).unwrap_or(&uniform);
        if p.len() != n_voxels || mag.len() != n_voxels {
            return Err(PipelineError::DimensionMismatch { expected: n_voxels, got: p.len().min(mag.len()) });
        }
        progress(e, n_echoes);
        let te = metadata.echo_times[e];
        let unwrapped = unwrap_one_echo(p, mag, mask, te, config, nx, ny, nz, vsx, vsy, vsz);
        out.push(single_echo_result(&unwrapped, mask, te, metadata.field_strength, config));
    }
    progress(n_echoes, n_echoes);
    Ok(out)
}

/// [`B0EstimationMethod::ComplexFit`]: MEDI's field map. `Fit_ppm_complex` on the wrapped
/// complex echoes gives the phase per echo spacing, wrapped; it is unwrapped spatially, as MEDI's
/// `iFreq = unwrapPhase(iMag, iFreq_raw, matrix_size)` does, and divided by the echo spacing.
/// The noise map (`dp1`, MEDI's `N_std`) is converted the same way, to ppm.
///
/// MEDI's unwrapper is a region-growing method that is not ported; the configured
/// `unwrapping_algorithm` is used instead. ROMEO gets MEDI's `iMag` (root-sum-of-squares of the
/// echo magnitudes) as its magnitude. Laplacian unwrapping is done as on the other paths: phase
/// zeroed outside the mask first, masked mean removed after (it leaves a constant undetermined).
fn field_mapping_complex_fit(
    phases: &[&[f64]],
    magnitudes: Option<&[&[f64]]>,
    mask: &[u8],
    metadata: &ScanMetadata,
    config: &FieldMappingConfig,
    params: &crate::utils::FitPpmComplexParams,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<FieldMappingResult, PipelineError> {
    let mags = magnitudes.ok_or_else(|| PipelineError::InvalidInput(
        "the complex fit (Fit_ppm_complex) needs the magnitude of every echo".into(),
    ))?;
    if mags.len() != phases.len() {
        return Err(PipelineError::DimensionMismatch { expected: phases.len(), got: mags.len() });
    }
    if config.bipolar_correction {
        return Err(PipelineError::InvalidConfig(
            "bipolar correction is not available with the complex fit (MEDI's \
             Fit_ppm_complex_bipolar is not ported)".into(),
        ));
    }
    progress(1, 4);
    let fit = crate::utils::fit_ppm_complex(mags, phases, &metadata.echo_times, params)?;

    progress(2, 4);
    let (nx, ny, nz) = metadata.dims;
    let (vsx, vsy, vsz) = metadata.voxel_size;
    let grid = crate::Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let unwrapped = match config.unwrapping_algorithm {
        UnwrappingAlgorithm::Laplacian => {
            let mut u = crate::unwrap::laplacian_unwrap(
                &mask_phase(&fit.field, mask), mask, &grid, config.laplacian_solver,
            );
            remove_masked_mean(&mut u, mask);
            u
        }
        UnwrappingAlgorithm::Romeo => {
            // iMag = sqrt(sum(abs(iField).^2, 4))
            let n = fit.field.len();
            let imag: Vec<f64> = (0..n)
                .map(|i| mags.iter().map(|m| m[i] * m[i]).sum::<f64>().sqrt())
                .collect();
            crate::unwrap::unwrap_romeo(
                &fit.field, &imag, None, fit.echo_spacing, 0.0,
                mask, &config.romeo_params, &grid,
            )
        }
    };

    progress(3, 4);
    let per_second = |v: &[f64]| -> Vec<f64> { v.iter().map(|&x| x / fit.echo_spacing).collect() };
    let b0_field_ppm = rads_to_ppm(&per_second(&unwrapped), metadata.field_strength);
    let field_noise_ppm = rads_to_ppm(&per_second(&fit.noise), metadata.field_strength);
    progress(4, 4);
    Ok(FieldMappingResult {
        b0_field_ppm,
        phase_offset: None,
        robust_fit: None,
        reliability: None,
        qsmart_fit_threshold: None,
        field_noise_ppm: Some(field_noise_ppm),
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
            crate::unwrap::laplacian_unwrap(phase, mask, &grid, config.laplacian_solver)
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
    use crate::utils::{LinearFitParams, RobustFitParams};
    use crate::unwrap::LaplacianSolver;
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
            affine: None,
        }
    }

    #[test]
    fn qsmart_echofit_is_unmasked_laplacian_then_echofit() {
        // QSMART's route, from the regular settings: every echo's raw phase unwrapped over the
        // whole volume with the mirror solver (the mask plays no part), then qsmart_echofit,
        // which also returns R_0. Bit for bit what QSMART's own route computed.
        let meta = make_test_metadata(3);
        let n = 8 * 8 * 8;
        let phases: Vec<Vec<f64>> = meta.echo_times.iter()
            .map(|te| (0..n).map(|i| { let t = 400.0 * te * (1.0 + (i as f64 * 0.37).sin()) + 0.2 * (i as f64).cos(); t.sin().atan2(t.cos()) }).collect())
            .collect();
        let mags: Vec<Vec<f64>> = (0..3).map(|e| (0..n).map(|i| 1.0 + ((i + e) % 5) as f64).collect()).collect();
        let p: Vec<&[f64]> = phases.iter().map(|v| v.as_slice()).collect();
        let m: Vec<&[f64]> = mags.iter().map(|v| v.as_slice()).collect();
        let mut mask = vec![0u8; n];
        mask[100..300].fill(1);
        let grid = meta.grid();
        let uw: Vec<Vec<f64>> = p.iter().map(|ph| crate::unwrap::laplacian_unwrap(ph, &vec![1; n], &grid, LaplacianSolver::Mirror)).collect();
        let uw: Vec<&[f64]> = uw.iter().map(|v| v.as_slice()).collect();
        let ppm = crate::utils::QSMART_GYRO * 3.0 / 1e6;
        // A threshold at the median residual, so R_0 has both values.
        let mut blur = crate::utils::qsmart_echofit(&uw, &m, &meta.echo_times, &grid, ppm, crate::utils::QsmartFitThreshold::Absolute(1.0)).fit_residual_blur;
        blur.sort_by(f64::total_cmp);
        let thr = blur[n / 2];
        let want = crate::utils::qsmart_echofit(&uw, &m, &meta.echo_times, &grid, ppm, crate::utils::QsmartFitThreshold::Absolute(thr));
        let config = FieldMappingConfig {
            unwrapping_algorithm: UnwrappingAlgorithm::Laplacian,
            laplacian_solver: LaplacianSolver::Mirror,
            laplacian_unmasked: true,
            b0_estimation: B0EstimationMethod::QsmartEchofit { fit_threshold: crate::utils::QsmartFitThreshold::Absolute(thr) },
            ..Default::default()
        };
        let got = run_field_mapping(&p, Some(&m), &mask, &meta, &config, &mut |_, _| {}).unwrap();
        assert_eq!(got.b0_field_ppm, want.field_ppm);
        assert_eq!(got.reliability, Some(want.reliability));
        assert!(got.reliability.as_ref().unwrap().contains(&0) && got.reliability.as_ref().unwrap().contains(&1));
        assert_eq!(got.qsmart_fit_threshold, Some(thr));
        // An adaptive threshold is reported as the residual the percentile resolved to.
        let pct = crate::utils::QsmartFitThreshold::Percentile(99.0);
        let want_pct = crate::utils::qsmart_echofit(&uw, &m, &meta.echo_times, &grid, ppm, pct);
        let config_pct = FieldMappingConfig { b0_estimation: B0EstimationMethod::QsmartEchofit { fit_threshold: pct }, ..config.clone() };
        let got_pct = run_field_mapping(&p, Some(&m), &mask, &meta, &config_pct, &mut |_, _| {}).unwrap();
        assert_eq!(got_pct.qsmart_fit_threshold, Some(want_pct.fit_threshold));
        assert!(want_pct.fit_threshold > 0.0 && want_pct.fit_threshold != thr);

        let bad = FieldMappingConfig { b0_estimation: B0EstimationMethod::QsmartEchofit { fit_threshold: crate::utils::QsmartFitThreshold::Percentile(0.0) }, ..config.clone() };
        assert!(run_field_mapping(&p, Some(&m), &mask, &meta, &bad, &mut |_, _| {}).is_err());

        // The phase masked first (the default) is a different unwrap, kept inside the mask.
        let masked = FieldMappingConfig { laplacian_unmasked: false, ..config.clone() };
        let got_masked = run_field_mapping(&p, Some(&m), &mask, &meta, &masked, &mut |_, _| {}).unwrap();
        assert_ne!(got_masked.b0_field_ppm, want.field_ppm);
        assert!(got_masked.b0_field_ppm.iter().zip(&mask).all(|(&f, &k)| k != 0 || f == 0.0));

        // ROMEO, with or without phase-offset removal or bipolar correction, feeds the same fit:
        // a departure from QSMART, not a refusal.
        for other in [
            FieldMappingConfig { unwrapping_algorithm: UnwrappingAlgorithm::Romeo, ..config.clone() },
            FieldMappingConfig { unwrapping_algorithm: UnwrappingAlgorithm::Romeo, phase_offset_removal: false, ..config.clone() },
            FieldMappingConfig { unwrapping_algorithm: UnwrappingAlgorithm::Romeo, bipolar_correction: true, ..config.clone() },
        ] {
            let got = run_field_mapping(&p, Some(&m), &mask, &meta, &other, &mut |_, _| {}).unwrap();
            assert!(got.reliability.is_some() && got.qsmart_fit_threshold.is_some(), "{other:?}");
            assert!(got.b0_field_ppm.iter().all(|f| f.is_finite()));
        }
        // Every other method leaves R_0 out.
        let linear = FieldMappingConfig { b0_estimation: B0EstimationMethod::LinearFit(LinearFitParams::default()), ..config };
        let got_linear = run_field_mapping(&p, Some(&m), &mask, &meta, &linear, &mut |_, _| {}).unwrap();
        assert!(got_linear.reliability.is_none() && got_linear.qsmart_fit_threshold.is_none());
    }

    #[test]
    fn qsmart_echofit_on_one_echo_is_unmasked_laplacian_then_echofit() {
        let meta = make_test_metadata(1);
        let n = 8 * 8 * 8;
        let phase: Vec<f64> = (0..n).map(|i| { let t = 2.0 * (1.0 + (i as f64 * 0.37).sin()); t.sin().atan2(t.cos()) }).collect();
        let mag: Vec<f64> = (0..n).map(|i| 1.0 + (i % 5) as f64).collect();
        let mut mask = vec![0u8; n];
        mask[100..300].fill(1);
        let grid = meta.grid();
        let uw = crate::unwrap::laplacian_unwrap(&phase, &vec![1; n], &grid, LaplacianSolver::Mirror);
        let fit_threshold = crate::utils::QsmartFitThreshold::default();
        let want = crate::utils::qsmart_echofit(&[&uw], &[&mag], &meta.echo_times, &grid, crate::utils::QSMART_GYRO * 3.0 / 1e6, fit_threshold);
        let config = FieldMappingConfig {
            unwrapping_algorithm: UnwrappingAlgorithm::Laplacian,
            laplacian_solver: LaplacianSolver::Mirror,
            laplacian_unmasked: true,
            b0_estimation: B0EstimationMethod::QsmartEchofit { fit_threshold },
            ..Default::default()
        };
        let got = run_field_mapping(&[&phase], Some(&[&mag]), &mask, &meta, &config, &mut |_, _| {}).unwrap();
        assert_eq!(got.b0_field_ppm, want.field_ppm);
        assert_eq!(got.reliability, Some(want.reliability));
    }

    /// Unmasked Laplacian unwrapping solves on the raw phase; the other combinations still keep
    /// the field inside the mask.
    #[test]
    fn laplacian_unmasked_unwraps_the_raw_phase() {
        let meta = make_test_metadata(3);
        let n = 8 * 8 * 8;
        let phases: Vec<Vec<f64>> = meta.echo_times.iter()
            .map(|te| (0..n).map(|i| { let t = 400.0 * te * (1.0 + (i as f64 * 0.37).sin()); t.sin().atan2(t.cos()) }).collect())
            .collect();
        let p: Vec<&[f64]> = phases.iter().map(|v| v.as_slice()).collect();
        let mut mask = vec![0u8; n];
        mask[100..300].fill(1);
        let grid = meta.grid();
        let base = FieldMappingConfig { unwrapping_algorithm: UnwrappingAlgorithm::Laplacian, ..Default::default() };
        let unmasked = FieldMappingConfig { laplacian_unmasked: true, ..base.clone() };
        let got = run_field_mapping(&p, None, &mask, &meta, &unmasked, &mut |_, _| {}).unwrap().b0_field_ppm;
        // As the multi-echo path computes it, from echoes unwrapped on the raw phase.
        let mut uw: Vec<Vec<f64>> = p.iter().map(|ph| crate::unwrap::laplacian_unwrap(ph, &mask, &grid, LaplacianSolver::Dct)).collect();
        align_per_echo_constants(&mut uw, &meta.echo_times, &mask, UnwrappingAlgorithm::Laplacian);
        let ones = vec![1.0; n];
        let mags: Vec<&[f64]> = (0..3).map(|_| ones.as_slice()).collect();
        let want = hz_to_ppm(&crate::utils::calculate_b0_weighted(&uw, &mags, &meta.echo_times, &mask, B0WeightType::PhaseSNR, &grid), 3.0);
        assert_eq!(got, want);
        let masked = run_field_mapping(&p, None, &mask, &meta, &base, &mut |_, _| {}).unwrap().b0_field_ppm;
        assert_ne!(got, masked);
        assert!(got.iter().zip(&mask).all(|(&f, &k)| k != 0 || f == 0.0));
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

        let gamma = crate::constants::GAMMA_BAR_HZ_PER_T;
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
        let gamma = crate::constants::GAMMA_BAR_HZ_PER_T;
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
            b0_estimation: B0EstimationMethod::default(),
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
    fn complex_fit_gives_the_field_and_its_noise_in_ppm() {
        // 4 echoes 5 ms apart, ω = 50·(1 + x) rad/s and an offset of 1 rad: the echo phases wrap
        // (up to 9 rad), the phase per echo spacing (up to 2 rad) does not.
        let meta = make_test_metadata(4);
        let n = 8 * 8 * 8;
        let mask = vec![1u8; n];
        let omega: Vec<f64> = (0..n).map(|i| 50.0 * (1.0 + (i % 8) as f64)).collect();
        let wrap = |x: f64| (x + PI).rem_euclid(2.0 * PI) - PI;
        let phases: Vec<Vec<f64>> = meta.echo_times.iter()
            .map(|&te| omega.iter().map(|&w| wrap(1.0 + w * te)).collect())
            .collect();
        let phase_refs: Vec<&[f64]> = phases.iter().map(|p| p.as_slice()).collect();
        let mag = vec![1.0; n];
        let mag_refs: Vec<&[f64]> = (0..4).map(|_| mag.as_slice()).collect();
        let fit = B0EstimationMethod::ComplexFit(crate::utils::FitPpmComplexParams::default());
        for unwrapping_algorithm in [UnwrappingAlgorithm::Romeo, UnwrappingAlgorithm::Laplacian] {
            let config = FieldMappingConfig { unwrapping_algorithm, b0_estimation: fit.clone(), ..Default::default() };
            let r = run_field_mapping(&phase_refs, Some(&mag_refs), &mask, &meta, &config, &mut |_, _| {}).unwrap();
            let want = rads_to_ppm(&omega, 3.0);
            // ROMEO keeps the absolute field; Laplacian only up to the masked mean
            let shift = if unwrapping_algorithm == UnwrappingAlgorithm::Laplacian {
                (want.iter().sum::<f64>() - r.b0_field_ppm.iter().sum::<f64>()) / n as f64
            } else {
                0.0
            };
            for (got, w) in r.b0_field_ppm.iter().zip(&want) {
                assert!((got + shift - w).abs() < 1e-6, "{unwrapping_algorithm:?}: {got} vs {w}");
            }
            // dp1 for four unit echoes: sqrt(4 / (4·14 − 36)) rad per spacing
            let noise = r.field_noise_ppm.expect("complex fit returns its noise map");
            let want_noise = rads_to_ppm(&[(0.2f64).sqrt() / 0.005], 3.0)[0];
            assert!(noise.iter().all(|v| (v - want_noise).abs() < 1e-12));
        }
    }

    #[test]
    fn complex_fit_needs_magnitudes_and_refuses_bipolar_correction() {
        let meta = make_test_metadata(3);
        let n = 8 * 8 * 8;
        let mask = vec![1u8; n];
        let p = vec![0.1; n];
        let phases: Vec<&[f64]> = vec![&p, &p, &p];
        let fit = B0EstimationMethod::ComplexFit(crate::utils::FitPpmComplexParams::default());
        let config = FieldMappingConfig { b0_estimation: fit.clone(), ..Default::default() };
        assert!(matches!(run_field_mapping(&phases, None, &mask, &meta, &config, &mut |_, _| {}),
            Err(PipelineError::InvalidInput(_))));
        let config = FieldMappingConfig { b0_estimation: fit, bipolar_correction: true, ..Default::default() };
        assert!(matches!(run_field_mapping(&phases, Some(&phases), &mask, &meta, &config, &mut |_, _| {}),
            Err(PipelineError::InvalidConfig(_))));
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
            affine: None,
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
            b0_estimation: B0EstimationMethod::default(),
            unwrapping_algorithm: UnwrappingAlgorithm::Romeo,
            ..Default::default()
        };
        let result_a = run_field_mapping(
            &phase_refs, Some(&mag_refs), &mask, &meta, &config_a, &mut |_, _| {},
        ).unwrap();

        // Path B: no offset removal, explicit linear fit (uses ROMEO)
        let config_b = FieldMappingConfig {
            phase_offset_removal: false,
            b0_estimation: B0EstimationMethod::LinearFit(LinearFitParams::default()),
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

    // ---- Path B honours b0_estimation and its weighting ---------------------------------------

    const N: usize = 16;

    fn meta_n(tes: &[f64]) -> ScanMetadata {
        ScanMetadata {
            dims: (N, N, N),
            voxel_size: (1.0, 1.0, 1.0),
            echo_times: tes.to_vec(),
            field_strength: 3.0,
            b0_direction: (0.0, 0.0, 1.0),
            slice_geometry: None,
            affine: None,
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
        let gamma = crate::constants::GAMMA_BAR_HZ_PER_T;
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
        let gamma = crate::constants::GAMMA_BAR_HZ_PER_T;
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
            let mut u = crate::unwrap::laplacian_unwrap(p, &mask, &grid, LaplacianSolver::Dct);
            let m = masked_mean(&u, &mask);
            u.iter_mut().for_each(|v| *v -= m);
            u
        }).collect();

        let mut outputs = Vec::new();
        for wt in all_weight_types() {
            let cfg = FieldMappingConfig {
                unwrapping_algorithm: UnwrappingAlgorithm::Laplacian,
                b0_estimation: B0EstimationMethod::WeightedAvg { weighting: wt },
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

    /// Laplacian + linear fit aligns the per-echo constants (as the weighted mean does), and a
    /// single Laplacian echo is masked and demeaned like each echo of a series.
    #[test]
    fn laplacian_linear_fit_and_single_echo_align_like_the_weighted_mean() {
        let tes = [0.004, 0.009, 0.014];
        let meta = meta_n(&tes);
        let (mut ph, mg) = echoes(&tes, omega_cos, 0.4);
        // A sphere mask with noise outside it, and the FFT solver, whose unwraps are not
        // zero-mean over the mask: the cases where alignment and masking matter.
        let c = N as f64 / 2.0;
        let mask: Vec<u8> = (0..N * N * N).map(|idx| {
            let d = |x: usize| x as f64 + 0.5 - c;
            (d(idx % N).powi(2) + d((idx / N) % N).powi(2) + d(idx / (N * N)).powi(2) < (c - 2.0).powi(2)) as u8
        }).collect();
        let mut st = 5u64;
        for p in ph.iter_mut() {
            for (v, &m) in p.iter_mut().zip(&mask) {
                if m == 0 { *v = noise(&mut st) * PI; }
            }
        }
        let grid = crate::Grid::new(N, N, N, 1.0, 1.0, 1.0);
        let solver = LaplacianSolver::Fft { pad: [8, 8, 6] };
        let uw: Vec<Vec<f64>> = ph.iter().map(|p| {
            let masked = mask_phase(p, &mask);
            let mut u = crate::unwrap::laplacian_unwrap(&masked, &mask, &grid, solver);
            remove_masked_mean(&mut u, &mask);
            u
        }).collect();

        for estimate_offset in [true, false] {
            let p = LinearFitParams { estimate_offset };
            let cfg = FieldMappingConfig {
                unwrapping_algorithm: UnwrappingAlgorithm::Laplacian,
                b0_estimation: B0EstimationMethod::LinearFit(p.clone()),
                laplacian_solver: solver,
                ..Default::default()
            };
            let got = run_field_mapping(&refs(&ph), Some(&refs(&mg)), &mask, &meta, &cfg, &mut |_, _| {})
                .unwrap().b0_field_ppm;
            let fit = crate::utils::multi_echo_linear_fit(
                &refs(&uw), &refs(&mg), &tes, &mask, p.estimate_offset);
            let want = rads_to_ppm(&fit.field, 3.0);
            let d = max_abs_diff(&got, &want, &mask);
            assert!(d < 1e-12, "estimate_offset {estimate_offset}: differs from the aligned fit by {d}");
        }

        let one = meta_n(&tes[..1]);
        let cfg = FieldMappingConfig {
            unwrapping_algorithm: UnwrappingAlgorithm::Laplacian,
            laplacian_solver: solver,
            ..Default::default()
        };
        let got = run_field_mapping(&[&ph[0]], None, &mask, &one, &cfg, &mut |_, _| {}).unwrap().b0_field_ppm;
        let want: Vec<f64> = uw[0].iter().map(|&u| u / tes[0]).collect();
        let d = max_abs_diff(&got, &rads_to_ppm(&want, 3.0), &mask);
        assert!(d < 1e-12, "single echo differs from the masked, demeaned unwrap by {d}");
    }

    #[test]
    fn laplacian_direct_recovers_wrapped_field() {
        // Noise-free: every weighting recovers the true field up to its masked mean.
        let tes = [0.004, 0.009, 0.014];
        let meta = meta_n(&tes);
        let (ph, mg) = echoes(&tes, omega_cos, 0.0);
        let mask = vec![1u8; N * N * N];
        let gamma = crate::constants::GAMMA_BAR_HZ_PER_T;
        let mut truth: Vec<f64> = (0..N * N * N)
            .map(|idx| omega_cos(idx % N, (idx / N) % N, idx / (N * N)) / (2.0 * PI) * 1e6 / (gamma * 3.0))
            .collect();
        let m = masked_mean(&truth, &mask);
        truth.iter_mut().for_each(|v| *v -= m);
        for wt in all_weight_types() {
            let cfg = FieldMappingConfig {
                unwrapping_algorithm: UnwrappingAlgorithm::Laplacian,
                b0_estimation: B0EstimationMethod::WeightedAvg { weighting: wt },
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
                b0_estimation: B0EstimationMethod::WeightedAvg { weighting: wt },
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
        let gamma = crate::constants::GAMMA_BAR_HZ_PER_T;
        let base = FieldMappingConfig {
            unwrapping_algorithm: UnwrappingAlgorithm::Romeo,
            phase_offset_removal: false,
            ..Default::default()
        };

        let wavg = run_field_mapping(&refs(&ph), Some(&refs(&mg)), &mask, &meta, &base, &mut |_, _| {})
            .unwrap().b0_field_ppm;
        let want = hand_combination(&ph, &mg, &tes, &mask, B0WeightType::PhaseSNR, 3.0);
        assert!(max_abs_diff(&wavg, &want, &mask) < 1e-12);

        let lf_cfg = FieldMappingConfig { b0_estimation: B0EstimationMethod::LinearFit(LinearFitParams::default()), ..base.clone() };
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
        // `b0_estimation = LinearFit` on the Laplacian direct path: per-echo unwrap of the
        // masked phase, each echo's masked mean removed (the per-echo alignment), then
        // multi_echo_linear_fit with an intercept (the variant carries no weighting).
        let tes = [0.004, 0.009, 0.014];
        let meta = meta_n(&tes);
        let (ph, mg) = echoes(&tes, omega_cos, 0.4);
        let mask = sphere_mask();
        let grid = crate::Grid::new(N, N, N, 1.0, 1.0, 1.0);
        let cfg = FieldMappingConfig {
            unwrapping_algorithm: UnwrappingAlgorithm::Laplacian,
            b0_estimation: B0EstimationMethod::LinearFit(LinearFitParams::default()),
            ..Default::default()
        };
        let got = run_field_mapping(&refs(&ph), Some(&refs(&mg)), &mask, &meta, &cfg, &mut |_, _| {})
            .unwrap().b0_field_ppm;
        let uw: Vec<Vec<f64>> = ph.iter()
            .map(|p| {
                let mut u = crate::unwrap::laplacian_unwrap(&mask_phase(p, &mask), &mask, &grid, LaplacianSolver::Dct);
                remove_masked_mean(&mut u, &mask);
                u
            })
            .collect();
        let fit = crate::utils::multi_echo_linear_fit(
            &refs(&uw), &refs(&mg), &tes, &mask,
            LinearFitParams::default().estimate_offset,
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
            let mut u = crate::unwrap::laplacian_unwrap(&mask_phase(p, &mask), &mask, &grid, LaplacianSolver::Dct);
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
            let cfg = FieldMappingConfig { b0_estimation: B0EstimationMethod::WeightedAvg { weighting: wt }, ..Default::default() };
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
                b0_estimation: B0EstimationMethod::WeightedAvg { weighting: B0WeightType::AssumedDecay { t2star_s: t2 } },
                ..Default::default()
            };
            assert!(run_field_mapping(&refs(&ph), None, &mask, &meta, &cfg, &mut |_, _| {}).is_err());
        }
    }

    // ---- Laplacian solver choice on the direct path --------------------------------------

    /// Masked phase, unwrapped with `solver`, as the direct path does before aligning echoes.
    fn masked_unwrap(ph: &[Vec<f64>], mask: &[u8], solver: LaplacianSolver) -> Vec<Vec<f64>> {
        let grid = crate::Grid::new(N, N, N, 1.0, 1.0, 1.0);
        ph.iter().map(|p| crate::unwrap::laplacian_unwrap(&mask_phase(p, mask), mask, &grid, solver)).collect()
    }

    #[test]
    fn laplacian_direct_uses_the_configured_solver() {
        let tes = [0.004, 0.009, 0.014];
        let meta = meta_n(&tes);
        let (ph, mg) = echoes(&tes, omega_cos, 0.4);
        let mask = sphere_mask();
        let fft = LaplacianSolver::Fft { pad: [4; 3] };
        let run = |solver| {
            let cfg = FieldMappingConfig {
                unwrapping_algorithm: UnwrappingAlgorithm::Laplacian,
                laplacian_solver: solver,
                ..Default::default()
            };
            run_field_mapping(&refs(&ph), Some(&refs(&mg)), &mask, &meta, &cfg, &mut |_, _| {})
                .unwrap().b0_field_ppm
        };
        assert_eq!(FieldMappingConfig::default().laplacian_solver, LaplacianSolver::Dct);
        let (dct, got) = (run(LaplacianSolver::Dct), run(fft));
        let mut uw = masked_unwrap(&ph, &mask, fft);
        uw.iter_mut().for_each(|u| remove_masked_mean(u, &mask));
        let want = hand_combination(&uw, &mg, &tes, &mask, B0WeightType::PhaseSNR, 3.0);
        assert!(max_abs_diff(&got, &want, &mask) < 1e-12);
        assert!(max_abs_diff(&got, &dct, &mask) > 1e-6, "solver setting had no effect");
    }

    #[test]
    fn assumed_decay_with_fft_is_ukb_up_to_a_constant() {
        // UK Biobank: MRPhaseUnwrap(mask .* φ) per echo (Fft), then Σ Wφ / Σ W·TE with
        // W = TE·exp(−TE/40 ms). The direct path removes each echo's masked mean first, which
        // with voxel-independent weights shifts the field by one constant and nothing else.
        let tes = [0.00942, 0.0197];
        let meta = meta_n(&tes);
        let (ph, mg) = echoes(&tes, omega_cos, 0.4);
        let mask = sphere_mask();
        let fft = LaplacianSolver::Fft { pad: [8; 3] };
        let cfg = FieldMappingConfig {
            unwrapping_algorithm: UnwrappingAlgorithm::Laplacian,
            laplacian_solver: fft,
            b0_estimation: B0EstimationMethod::WeightedAvg { weighting: B0WeightType::assumed_decay_default() },
            ..Default::default()
        };
        let got = run_field_mapping(&refs(&ph), Some(&refs(&mg)), &mask, &meta, &cfg, &mut |_, _| {})
            .unwrap().b0_field_ppm;
        let ukb = ukb_assumed_decay(&masked_unwrap(&ph, &mask, fft), &tes, &mask, 0.040, 3.0);
        let d: Vec<f64> = got.iter().zip(&ukb).map(|(a, b)| a - b).collect();
        let c = masked_mean(&d, &mask);
        let dev = d.iter().zip(&mask).filter(|(_, &m)| m != 0).map(|(v, _)| (v - c).abs()).fold(0.0, f64::max);
        assert!(dev < 1e-12, "not a constant offset: {dev}");
    }

    #[test]
    fn assumed_decay_effective_te_matches_ukb() {
        // UK Biobank's effective TE Σ W·TE / Σ W for TE 9.42 / 19.7 ms and T2* = 40 ms is
        // 15.772350 ms (from the MATLAB run). The phase weight is W = w / TE.
        let tes = [9.42e-3, 19.7e-3];
        let wt = B0WeightType::assumed_decay_default();
        let w: Vec<f64> = tes.iter().map(|&te| wt.weight(te, 1.0) / te).collect();
        assert!((w[0] - 9.42e-3 * (-9.42f64 / 40.0).exp()).abs() < 1e-15);
        let te_eff = (w[0] * tes[0] + w[1] * tes[1]) / (w[0] + w[1]);
        assert!((te_eff - 15.772350e-3).abs() < 1e-9, "{te_eff}");
    }

    // ---- B0EstimationMethod::RobustFit ----------------------------------------------------------

    const TES4: [f64; 4] = [0.004, 0.009, 0.014, 0.019];
    /// The echo the corruption tests damage. Not the first two, which the offset removal reads,
    /// and not the last, which a linear fit follows (see `multi_echo_robust_fit`'s docs).
    const BAD: usize = 2;

    fn robust_cfg(unwrapping_algorithm: UnwrappingAlgorithm, phase_offset_removal: bool) -> FieldMappingConfig {
        FieldMappingConfig {
            unwrapping_algorithm,
            phase_offset_removal,
            b0_estimation: B0EstimationMethod::RobustFit(RobustFitParams::default()),
            ..Default::default()
        }
    }

    /// A four-echo series `wrap(φ₀ + TE·ω + noise/mag)` over a sphere, with phase noise scaled
    /// as `1/magnitude` (phase SNR is magnitude SNR, the model the robust scale assumes). With
    /// `corrupt`, echo [`BAD`] loses 70% of its magnitude and gains a smooth phase error of up to
    /// ~0.8 rad over a slab of slices — what a shot taken mid-movement does to one echo.
    ///
    /// Returns `(phases, mags, mask, true field in ppm, slab mask)`.
    #[allow(clippy::type_complexity)]
    fn robust_series(corrupt: bool) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<u8>, Vec<f64>, Vec<u8>) {
        let n = N * N * N;
        let mask = sphere_mask();
        let mut ph = vec![vec![0.0; n]; TES4.len()];
        let mut mg = vec![vec![0.0; n]; TES4.len()];
        let mut truth = vec![0.0; n];
        let mut slab = vec![0u8; n];
        let mut st = 7u64;
        let ppm = 1e6 / (2.0 * PI * crate::constants::GAMMA_BAR_HZ_PER_T * 3.0);
        for k in 0..N { for j in 0..N { for i in 0..N {
            let idx = i + j * N + k * N * N;
            let w = omega_small(i, j, k);
            let phi0 = 0.2 * (i as f64 * 0.3).cos() * (k as f64 * 0.2).sin();
            truth[idx] = w * ppm;
            let in_slab = (5..11).contains(&k);
            slab[idx] = (in_slab && mask[idx] != 0) as u8;
            for (e, &te) in TES4.iter().enumerate() {
                let mut m = (1.0 + 0.8 * (i as f64 / N as f64)) * (-te / 0.030).exp();
                let mut p = phi0 + te * w + 0.02 * noise(&mut st) / m;
                if corrupt && e == BAD && in_slab {
                    m *= 0.3;
                    p += 0.8 * ((2.5 * i as f64 / N as f64).sin() + (1.7 * j as f64 / N as f64).cos()) * 0.5;
                }
                ph[e][idx] = wrap(p);
                mg[e][idx] = m;
            }
        }}}
        (ph, mg, mask, truth, slab)
    }

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }

    fn mean_abs_err_demeaned(got: &[f64], want: &[f64], mask: &[u8]) -> f64 {
        // The direct path loses a global constant field (removed by referencing anyway).
        let d: Vec<f64> = got.iter().zip(want).map(|(a, b)| a - b).collect();
        let c = masked_mean(&d, mask);
        let (s, n) = d.iter().zip(mask).filter(|(_, &m)| m != 0)
            .fold((0.0, 0usize), |(s, n), (&x, _)| (s + (x - c).abs(), n + 1));
        s / n as f64
    }

    #[test]
    fn default_estimation_is_the_phase_snr_weighted_average_and_reports_nothing() {
        let cfg = FieldMappingConfig::default();
        assert_eq!(cfg.b0_estimation, B0EstimationMethod::WeightedAvg { weighting: B0WeightType::PhaseSNR });
        let (ph, mg, mask, _, _) = robust_series(false);
        let meta = meta_n(&TES4);
        let r = run_field_mapping(&refs(&ph), Some(&refs(&mg)), &mask, &meta, &cfg, &mut |_, _| {}).unwrap();
        assert!(r.robust_fit.is_none());
        let lf = FieldMappingConfig {
            b0_estimation: B0EstimationMethod::LinearFit(LinearFitParams::default()),
            ..Default::default()
        };
        let r = run_field_mapping(&refs(&ph), Some(&refs(&mg)), &mask, &meta, &lf, &mut |_, _| {}).unwrap();
        assert!(r.robust_fit.is_none());
    }

    /// Selecting the robust fit in the config runs `multi_echo_robust_fit` on exactly the echoes
    /// each path would hand any other estimator — bit for bit, and with its report returned.
    #[test]
    fn robust_fit_is_the_library_fit_on_every_path() {
        let (ph, mg, mask, _, _) = robust_series(true);
        let meta = meta_n(&TES4);
        let grid = crate::Grid::new(N, N, N, 1.0, 1.0, 1.0);
        let params = RobustFitParams::default();
        let run = |cfg: &FieldMappingConfig| {
            run_field_mapping(&refs(&ph), Some(&refs(&mg)), &mask, &meta, cfg, &mut |_, _| {}).unwrap()
        };

        // Path A: ROMEO with phase-offset removal (the default unwrapping).
        let cfg = robust_cfg(UnwrappingAlgorithm::Romeo, true);
        let got = run(&cfg);
        let (corrected, _) = crate::utils::phase_offset_removal(
            &ph, &mg, &TES4, cfg.phase_offset_sigma_voxels((1.0, 1.0, 1.0)), [0, 1],
            crate::unwrap::UnwrapMethod::Romeo, &grid,
        );
        let uw = crate::unwrap::unwrap_romeo_multi_echo(&corrected, &refs(&mg), &TES4, &mask, &cfg.romeo_params, &grid);
        let want = crate::utils::multi_echo_robust_fit(&uw, &mg, &TES4, &mask, &params);
        assert_eq!(got.b0_field_ppm, hz_to_ppm(&crate::utils::field_to_hz(&want.fit.field), 3.0));
        let rep = got.robust_fit.expect("path A must return the robust-fit report");
        assert_eq!(rep.robust_weights, want.robust_weights);
        assert_eq!(rep.quality.flagged, want.quality.flagged);
        assert!(got.phase_offset.is_some());

        // Path B: Laplacian (always direct): masked unwrap, demean, fit.
        let got = run(&robust_cfg(UnwrappingAlgorithm::Laplacian, true));
        let mut uw = masked_unwrap(&ph, &mask, LaplacianSolver::Dct);
        uw.iter_mut().for_each(|u| remove_masked_mean(u, &mask));
        let want = crate::utils::multi_echo_robust_fit(&uw, &mg, &TES4, &mask, &params);
        assert_eq!(got.b0_field_ppm, rads_to_ppm(&want.fit.field, 3.0));
        assert_eq!(got.robust_fit.expect("Laplacian path report").robust_weights, want.robust_weights);

        // Path B: ROMEO without phase-offset removal: per-echo unwrap, wrap alignment, fit.
        let cfg = robust_cfg(UnwrappingAlgorithm::Romeo, false);
        let got = run(&cfg);
        let mut uw: Vec<Vec<f64>> = (0..TES4.len()).map(|e| unwrap_single(
            &ph[e], &mg[0], &mask, &cfg, N, N, N, 1.0, 1.0, 1.0,
            ph.get(e + 1).map(|p| p.as_slice()), TES4[e], TES4.get(e + 1).copied().unwrap_or(0.0),
        )).collect();
        crate::unwrap::correct_multi_echo_wraps(&mut uw, &TES4, &mask);
        let want = crate::utils::multi_echo_robust_fit(&uw, &mg, &TES4, &mask, &params);
        assert_eq!(got.b0_field_ppm, rads_to_ppm(&want.fit.field, 3.0));
        assert!(got.robust_fit.is_some() && got.phase_offset.is_none());
    }

    /// The point of the option, end to end through `run_field_mapping`: one echo corrupted over a
    /// slab is named, is down-weighted where it is corrupted (with ROMEO, only there), and the field map
    /// is closer to the truth than either of the existing estimators manages — on every path.
    /// On the clean series nothing is flagged.
    #[test]
    fn robust_fit_downweights_a_corrupted_echo_end_to_end() {
        let meta = meta_n(&TES4);
        let (ph, mg, mask, truth, slab) = robust_series(true);
        let (cph, cmg, _, _, _) = robust_series(false);
        let outside: Vec<u8> = mask.iter().zip(&slab).map(|(&m, &s)| (m != 0 && s == 0) as u8).collect();
        let in_set = |w: &[f64], set: &[u8]| -> Vec<f64> {
            w.iter().zip(set).filter(|(_, &m)| m != 0).map(|(&x, _)| x).collect()
        };

        for (alg, offset) in [
            (UnwrappingAlgorithm::Romeo, true),
            (UnwrappingAlgorithm::Romeo, false),
            (UnwrappingAlgorithm::Laplacian, true),
        ] {
            let with = |b0_estimation: B0EstimationMethod, ph: &[Vec<f64>], mg: &[Vec<f64>]| {
                let cfg = FieldMappingConfig {
                    unwrapping_algorithm: alg,
                    phase_offset_removal: offset,
                    b0_estimation,
                    ..Default::default()
                };
                run_field_mapping(&refs(ph), Some(&refs(mg)), &mask, &meta, &cfg, &mut |_, _| {}).unwrap()
            };
            let robust = with(B0EstimationMethod::RobustFit(RobustFitParams::default()), &ph, &mg);
            let linear = with(B0EstimationMethod::LinearFit(LinearFitParams::default()), &ph, &mg);
            let wavg = with(B0EstimationMethod::default(), &ph, &mg);
            let clean = with(B0EstimationMethod::RobustFit(RobustFitParams::default()), &cph, &cmg);

            let rep = robust.robust_fit.as_ref().unwrap();
            let tag = format!("{alg:?}, offset removal {offset}");
            assert_eq!(rep.quality.flagged, vec![BAD], "{tag}: must name exactly the corrupted echo ({:?})", rep.quality.outlier_score);
            assert!(clean.robust_fit.as_ref().unwrap().quality.flagged.is_empty(),
                "{tag}: clean series flagged: {:?}", clean.robust_fit.as_ref().unwrap().quality.outlier_score);

            let bad_in = median(in_set(&rep.robust_weights[BAD], &slab));
            let bad_out = median(in_set(&rep.robust_weights[BAD], &outside));
            let good_in = median(in_set(&rep.robust_weights[0], &slab));
            assert!(bad_in < 0.5, "{tag}: corrupted echo kept median weight {bad_in} in the slab");
            println!("{tag}: median robust weight, bad echo in slab {bad_in}, outside {bad_out}; good echo in slab {good_in}");
            // Laplacian unwrapping is a global Poisson solve, so a corrupted slab leaves that echo
            // wrong (smeared, then shifted by its demeaning) everywhere, and the fit is right to
            // drop it everywhere: it does, and gets the lowest error of the three paths. ROMEO
            // keeps the damage local, so there the weight must stay up outside the slab.
            let local = alg == UnwrappingAlgorithm::Romeo;
            assert!((bad_out > 0.9 || !local) && good_in > 0.9,
                "{tag}: weights moved where nothing is wrong: bad echo outside {bad_out}, good echo inside {good_in}");

            let e_rob = mean_abs_err_demeaned(&robust.b0_field_ppm, &truth, &mask);
            let e_lin = mean_abs_err_demeaned(&linear.b0_field_ppm, &truth, &mask);
            let e_avg = mean_abs_err_demeaned(&wavg.b0_field_ppm, &truth, &mask);
            let e_cln = mean_abs_err_demeaned(&clean.b0_field_ppm, &truth, &mask);
            println!("{tag}: field error (ppm) robust {e_rob:.5}, linear {e_lin:.5}, weighted avg {e_avg:.5}, robust on clean {e_cln:.5}");
            assert!(e_rob < e_lin && e_rob < e_avg, "{tag}: robust {e_rob} vs linear {e_lin} / weighted avg {e_avg}");
        }
    }

    #[test]
    fn robust_fit_refuses_a_series_too_short_to_judge() {
        let (ph, mg, mask, _, _) = robust_series(false);
        let run = |n: usize, estimate_offset: bool| {
            let cfg = FieldMappingConfig {
                b0_estimation: B0EstimationMethod::RobustFit(RobustFitParams { estimate_offset, ..Default::default() }),
                ..Default::default()
            };
            run_field_mapping(&refs(&ph[..n]), Some(&refs(&mg[..n])), &mask, &meta_n(&TES4[..n]), &cfg, &mut |_, _| {})
        };
        for (n, offset) in [(1, true), (2, true), (3, true), (1, false), (2, false)] {
            match run(n, offset) {
                Err(PipelineError::InvalidConfig(msg)) => assert!(msg.contains("at least"), "{msg}"),
                other => panic!("{n} echoes, estimate_offset {offset}: expected InvalidConfig, got {:?}", other.map(|r| r.robust_fit.is_some())),
            }
        }
        assert!(run(3, false).unwrap().robust_fit.is_some());
        assert!(run(4, true).unwrap().robust_fit.is_some());
    }

    #[test]
    fn robust_fit_rejects_out_of_range_parameters() {
        let (ph, mg, mask, _, _) = robust_series(false);
        let d = RobustFitParams::default();
        assert_eq!(d.min_echoes(), 4);
        assert!(d.validate().is_ok());
        let bad = [
            RobustFitParams { tuning: Some(0.0), ..d.clone() },
            RobustFitParams { tuning: Some(-1.0), ..d.clone() },
            RobustFitParams { tuning: Some(f64::NAN), ..d.clone() },
            RobustFitParams { tuning: Some(f64::INFINITY), ..d.clone() },
            RobustFitParams { downweight_level: 0.0, ..d.clone() },
            RobustFitParams { downweight_level: 1.5, ..d.clone() },
            RobustFitParams { downweight_level: f64::NAN, ..d.clone() },
            RobustFitParams { flag_ratio: 0.5, ..d.clone() },
            RobustFitParams { flag_ratio: f64::NAN, ..d.clone() },
        ];
        for p in bad {
            let cfg = FieldMappingConfig { b0_estimation: B0EstimationMethod::RobustFit(p.clone()), ..Default::default() };
            let r = run_field_mapping(&refs(&ph), Some(&refs(&mg)), &mask, &meta_n(&TES4), &cfg, &mut |_, _| {});
            assert!(matches!(r, Err(PipelineError::InvalidConfig(_))), "{p:?} was accepted");
        }
        // The edges of every range are valid; iterations = 0 is the report-only mode.
        let edges = [
            RobustFitParams { tuning: Some(1e-3), ..d.clone() },
            RobustFitParams { downweight_level: 1.0, ..d.clone() },
            RobustFitParams { flag_ratio: 1.0, ..d.clone() },
            RobustFitParams { flag_ratio: f64::INFINITY, ..d.clone() },
            RobustFitParams { iterations: 0, ..d.clone() },
        ];
        for p in edges {
            assert!(p.validate().is_ok(), "{p:?} was refused");
        }
    }

    /// `iterations: 0` leaves every robust weight at 1, so the field map is the plain linear fit
    /// (same estimate_offset) bit for bit, while the report is still produced and still names the
    /// corrupted echo.
    #[test]
    fn robust_fit_with_no_iterations_is_the_linear_fit_plus_a_report() {
        let (ph, mg, mask, _, _) = robust_series(true);
        let meta = meta_n(&TES4);
        let run = |b0_estimation| {
            let cfg = FieldMappingConfig { b0_estimation, ..Default::default() };
            run_field_mapping(&refs(&ph), Some(&refs(&mg)), &mask, &meta, &cfg, &mut |_, _| {}).unwrap()
        };
        let r = run(B0EstimationMethod::RobustFit(RobustFitParams { iterations: 0, ..Default::default() }));
        let l = run(B0EstimationMethod::LinearFit(LinearFitParams::default()));
        assert_eq!(r.b0_field_ppm, l.b0_field_ppm);
        let rep = r.robust_fit.unwrap();
        assert!(rep.robust_weights.iter().all(|w| w.iter().all(|&x| x == 1.0)));
        assert_eq!(rep.quality.flagged, vec![BAD]);
    }
}

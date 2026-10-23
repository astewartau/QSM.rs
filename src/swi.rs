//! Susceptibility Weighted Imaging (CLEAR-SWI) and Susceptibility Map-Weighted
//! Imaging (SMWI)
//!
//! [`calculate_swi`] implements CLEAR-SWI (Eckstein et al. 2021), as CLEARSWI.jl does with its
//! default options. All echoes contribute:
//!
//! 1. **Magnitude.** The echoes are combined ([`MagCombine`]; root-sum-of-squares by default),
//!    divided by a receive-sensitivity estimate from the first echo ([`Sensitivity`]; the
//!    CLEAR bias-field estimate of [`get_sensitivity`]) and scaled with a softplus curve
//!    ([`softplus_scaling`]) whose offset is half the 80th intensity percentile.
//! 2. **Phase.** Each echo is unwrapped ([`SwiUnwrap`]; Laplacian by default), high-pass
//!    filtered in-plane ([`highpass_filter`]) and the echoes are combined into a frequency map
//!    by weighted averaging of `φ/TE` ([`B0WeightType`]; `mag·TE`, the phase-SNR weighting, by
//!    default). With [`SwiUnwrap::Romeo`] the echoes are unwrapped together and the combined
//!    frequency map is filtered instead.
//! 3. **Phase mask.** The filtered phase is mapped to [0, 1] ([`create_phase_mask`]; a tanh
//!    sigmoid by default), and SWI = magnitude × phase mask.
//!
//! A single echo is the special case without combination: the filtered phase itself drives the
//! phase mask, and the magnitude is that echo's magnitude.
//!
//! [`create_mip`] gives the minimum-intensity projection (7 slices by default).
//!
//! SMWI ([`calculate_smwi`]) builds the same kind of [0, 1] mask from a
//! susceptibility map instead of from filtered phase, which avoids the
//! blooming and residual-wrap artifacts a phase mask inherits from the
//! non-local dipole field.
//!
//! References:
//! Eckstein, K., Bachrata, B., Hangel, G., Widhalm, G., Enzinger, C., Barth, M., Trattnig, S.,
//! Robinson, S.D. (2021). "Improved susceptibility weighted imaging at ultra-high field using
//! bipolar multi-echo acquisition and optimized image processing: CLEAR-SWI." NeuroImage 237,
//! 118175. https://doi.org/10.1016/j.neuroimage.2021.118175
//!
//! Gho, S.-M., et al. (2014). "Susceptibility map-weighted imaging (SMWI)
//! for neuroimaging." Magnetic Resonance in Medicine, 72:337-346.
//! https://doi.org/10.1002/mrm.24920
//!
//! Nam, Y., et al. (2017). "Imaging of nigrosome 1 in substantia nigra at 3T
//! using multiecho susceptibility map-weighted imaging (SMWI)." Journal of
//! Magnetic Resonance Imaging, 46:528-536. https://doi.org/10.1002/jmri.25553
//!
//! Reference implementations: https://github.com/korbinian90/CLEARSWI.jl
//! (SWI; `calculateSWI` with default `Options`), https://github.com/kschan0214/sepia
//! (SMWI, `misc/swi_smwi/smwi`)
//!
//! # Differences from CLEARSWI.jl
//!
//! - Laplacian unwrapping uses [`laplacian_unwrap`] with the DCT solver, whose discrete
//!   Laplacian accounts for the voxel size and the matrix size of each axis. MriResearchTools'
//!   `laplacianunwrap` uses eigenvalues `p² + q² + t²` (`p = 1..N`) in index units, which
//!   weights the axes by matrix size rather than by voxel spacing. The difference is a smooth
//!   field that the high-pass filter largely removes.
//! - Quantiles for the softplus offset and the rescaling phase masks use the same block
//!   sampling as MriResearchTools' `estimatequantile`; the brain-agnostic mask comes from
//!   [`robust_mask`].
//! - The rescaling phase masks ([`PhaseScaling::Positive`] and friends) clamp the rescaled
//!   value to [0, 1] before raising it to `strength`, so a value beyond the 1st/99th-percentile
//!   extrapolation cannot exceed 1 (or go negative under a non-integer power).

use crate::error::InputError;
use crate::unwrap::{laplacian_unwrap, unwrap_romeo_multi_echo, LaplacianSolver, RomeoParams};
use crate::utils::{calculate_b0_weighted, gaussian_smooth_3d, get_sensitivity, robust_mask, B0WeightType};
use crate::Grid;

/// CLEAR-SWI parameters. The defaults are CLEARSWI.jl's `Options()` defaults.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct SwiParams {
    /// How the echoes' magnitudes are combined (default: [`MagCombine::Rss`], CLEARSWI.jl's `:SNR`)
    pub mag_combine: MagCombine,
    /// Receive-sensitivity (bias-field) correction of the combined magnitude
    /// (default: [`Sensitivity::Estimate`] with CLEARSWI.jl's default sigma)
    pub sensitivity: Sensitivity,
    /// Scale the corrected magnitude with [`softplus_scaling`], offset at half the 80th
    /// percentile (default: true)
    pub softplus: bool,
    /// Phase unwrapping (default: [`SwiUnwrap::Laplacian`])
    pub unwrap: SwiUnwrap,
    /// Echo weighting when combining the filtered phases into a frequency map
    /// (default: [`B0WeightType::PhaseSNR`], `mag·TE`)
    pub echo_weighting: B0WeightType,
    /// High-pass filter sigma in voxels [x, y, z] (default: [4, 4, 0])
    pub hp_sigma: [f64; 3],
    /// Phase mask scaling (default: [`PhaseScaling::Tanh`])
    pub scaling: PhaseScaling,
    /// Phase mask strength; for the rescaling masks, the power (default: 4)
    pub strength: f64,
    /// Minimum-intensity projection window in slices (default: 7)
    pub mip_window: usize,
}

impl Default for SwiParams {
    fn default() -> Self {
        Self {
            mag_combine: MagCombine::Rss,
            sensitivity: Sensitivity::Estimate { sigma_mm: None },
            softplus: true,
            unwrap: SwiUnwrap::Laplacian,
            echo_weighting: B0WeightType::PhaseSNR,
            hp_sigma: [4.0, 4.0, 0.0],
            scaling: PhaseScaling::Tanh,
            strength: 4.0,
            mip_window: 7,
        }
    }
}

/// Magnitude echo combination for SWI
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MagCombine {
    /// Root sum of squares, `sqrt(Σ mag²)` (CLEARSWI.jl `:SNR`)
    Rss,
    /// Sum over echoes (CLEARSWI.jl `:average`, which sums; the scale cancels in the softplus)
    Average,
    /// The last echo only
    Last,
    /// One echo, 0-indexed
    Echo(usize),
}

/// Receive-sensitivity correction of the combined SWI magnitude
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sensitivity {
    /// No correction
    Off,
    /// Divide by the CLEAR bias-field estimate of the first echo ([`get_sensitivity`]).
    /// `sigma_mm: None` uses CLEARSWI.jl's default: 7 mm, or the median field of view if
    /// that is smaller.
    Estimate {
        /// Bias-field smoothing sigma in mm
        sigma_mm: Option<f64>,
    },
}

/// Phase unwrapping for SWI
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwiUnwrap {
    /// Each echo unwrapped with [`laplacian_unwrap`] (DCT solver) and high-pass filtered,
    /// then the echoes combined (CLEARSWI.jl `:laplacian`)
    Laplacian,
    /// As [`Self::Laplacian`], slice by slice in 2D (CLEARSWI.jl `:laplacianslice`); for 2D
    /// multi-slice acquisitions
    LaplacianSlice,
    /// All echoes unwrapped together with ROMEO (template echo 1), combined, then the combined
    /// map high-pass filtered in-plane (CLEARSWI.jl `:romeo`)
    Romeo,
}

/// Phase mask scaling type
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhaseScaling {
    /// Sigmoid weighting: `(1 + tanh(1 - x/m)) / 2`
    /// where `m = median(positive phase in mask) * 10 / strength`
    Tanh,
    /// Negate phase first, then apply Tanh
    NegativeTanh,
    /// Traditional SWI: positive phase suppressed, negative → 1
    Positive,
    /// Traditional SWI: negative phase suppressed, positive → 1
    Negative,
    /// Both positive and negative phase suppressed
    Triangular,
}

/// CLEAR-SWI and the images it was made from.
#[derive(Debug, Clone)]
pub struct SwiResult {
    /// SWI: `magnitude × phase_mask`
    pub swi: Vec<f64>,
    /// The SWI magnitude: combined, sensitivity-corrected and softplus-scaled
    pub magnitude: Vec<f64>,
    /// The phase mask, in [0, 1]
    pub phase_mask: Vec<f64>,
    /// The high-pass filtered (combined) phase the mask was made from: a frequency map in Hz
    /// when the echo times are in seconds and there are several echoes, radians for one echo
    pub filtered_phase: Vec<f64>,
    /// The mask the phase was filtered and scaled within
    pub mask: Vec<u8>,
}

/// High-pass filter by subtracting a masked Gaussian-smoothed version
///
/// The smoothing is MriResearchTools' masked `gaussiansmooth3d` (4 box passes, NaN
/// extrapolation into the masked-out region). An axis with sigma 0 is not smoothed.
///
/// # Arguments
/// * `data` - Input data (e.g. unwrapped phase)
/// * `mask` - Binary mask (1 = inside, 0 = outside)
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `sigma` - Gaussian sigma for each dimension in voxels (e.g. [4, 4, 0])
///
/// # Returns
/// High-pass filtered data, zero outside `mask`
pub fn highpass_filter(
    data: &[f64],
    mask: &[u8],
    grid: &Grid,
    sigma: [f64; 3],
) -> Vec<f64> {
    let nbox = 4; // masked smoothing uses nbox=4 in MriResearchTools
    let smoothed = gaussian_smooth_3d(data, sigma, Some(mask), None, nbox, grid);
    let n_total = grid.n_total();
    let mut result = vec![0.0; n_total];
    for i in 0..n_total {
        if mask[i] != 0 {
            result[i] = data[i] - smoothed[i];
        }
    }
    result
}

/// Create the SWI phase mask from filtered phase
///
/// As CLEARSWI.jl's `createphasemask!`: the phase is set to zero outside `mask`, then the
/// scaling is applied to every voxel, so outside the mask the weight is the scaling's value at
/// zero phase (≈ 0.88 for [`PhaseScaling::Tanh`], 1 for [`PhaseScaling::Positive`]) rather
/// than zero. The SWI therefore keeps the whole head visible, as CLEAR-SWI does.
///
/// # Arguments
/// * `phase` - High-pass filtered phase (any unit; every scaling is scale-invariant)
/// * `mask` - Binary mask
/// * `scaling` - Phase scaling type
/// * `strength` - Scaling strength (higher = stronger phase contrast)
///
/// # Returns
/// Phase mask with values in [0, 1]
pub fn create_phase_mask(
    phase: &[f64],
    mask: &[u8],
    scaling: PhaseScaling,
    strength: f64,
) -> Vec<f64> {
    let mut result: Vec<f64> = phase
        .iter()
        .zip(mask)
        .map(|(&p, &m)| if m != 0 { p } else { 0.0 })
        .collect();
    // `pos` is taken before any negation, as in CLEARSWI.jl.
    let pos: Vec<bool> = result.iter().map(|&v| v > 0.0).collect();

    let scaling = if scaling == PhaseScaling::NegativeTanh {
        for v in result.iter_mut() {
            *v = -*v;
        }
        PhaseScaling::Tanh
    } else {
        scaling
    };

    match scaling {
        PhaseScaling::Tanh => {
            let mut positives: Vec<f64> = result
                .iter()
                .zip(mask)
                .filter(|&(&v, &m)| m != 0 && v > 0.0)
                .map(|(&v, _)| v)
                .collect();
            let m = match median(&mut positives) {
                Some(med) if med > 0.0 => med * 10.0 / strength,
                _ => 1.0,
            };
            for v in result.iter_mut() {
                *v = (1.0 + (1.0 - *v / m).tanh()) / 2.0;
            }
        }
        PhaseScaling::Positive => {
            rescale_selected(&mut result, &pos, true, 1.0, 0.0, strength);
            fill_selected(&mut result, &pos, false, 1.0);
        }
        PhaseScaling::Negative => {
            rescale_selected(&mut result, &pos, false, 0.0, 1.0, strength);
            fill_selected(&mut result, &pos, true, 1.0);
        }
        PhaseScaling::Triangular => {
            rescale_selected(&mut result, &pos, true, 1.0, 0.0, strength);
            rescale_selected(&mut result, &pos, false, 0.0, 1.0, strength);
        }
        PhaseScaling::NegativeTanh => unreachable!(),
    }

    for v in &mut result {
        if *v < 0.0 || v.is_nan() {
            *v = 0.0;
        }
    }
    result
}

/// Combine the magnitude echoes for SWI.
///
/// `mags` holds one volume per echo.
///
/// # Errors
/// If there are no echoes, or [`MagCombine::Echo`] names an echo that does not exist.
pub fn combine_magnitude<M: AsRef<[f64]>>(
    mags: &[M],
    combine: MagCombine,
) -> Result<Vec<f64>, InputError> {
    let n_echoes = mags.len();
    if n_echoes == 0 {
        return Err(InputError::new("SWI needs at least one magnitude echo"));
    }
    if n_echoes == 1 {
        return Ok(mags[0].as_ref().to_vec());
    }
    let n = mags[0].as_ref().len();
    Ok(match combine {
        MagCombine::Rss => (0..n)
            .map(|i| mags.iter().map(|m| m.as_ref()[i].powi(2)).sum::<f64>().sqrt())
            .collect(),
        MagCombine::Average => (0..n).map(|i| mags.iter().map(|m| m.as_ref()[i]).sum()).collect(),
        MagCombine::Last => mags[n_echoes - 1].as_ref().to_vec(),
        MagCombine::Echo(e) => mags
            .get(e)
            .ok_or_else(|| {
                InputError::new(format!("SWI magnitude echo {} requested, but there are {n_echoes}", e + 1))
            })?
            .as_ref()
            .to_vec(),
    })
}

/// The SWI magnitude: combined echoes, sensitivity-corrected, softplus-scaled.
///
/// As CLEARSWI.jl's `getswimag`. The sensitivity is estimated from the first echo.
pub fn swi_magnitude<M: AsRef<[f64]>>(
    mags: &[M],
    grid: &Grid,
    params: &SwiParams,
) -> Result<Vec<f64>, InputError> {
    let mut combined = combine_magnitude(mags, params.mag_combine)?;
    if let Sensitivity::Estimate { sigma_mm } = params.sensitivity {
        let sigma_mm = sigma_mm.unwrap_or_else(|| default_sensitivity_sigma_mm(grid));
        let sensitivity = get_sensitivity(mags[0].as_ref(), grid, sigma_mm, 15)?;
        for (c, &s) in combined.iter_mut().zip(&sensitivity) {
            if s.is_finite() && s > 0.0 {
                *c /= s;
            }
        }
    }
    if params.softplus {
        let q = estimate_quantile(&combined, 0.8);
        combined = softplus_scaling(&combined, q / 2.0, 2.0);
    }
    Ok(combined)
}

/// CLEARSWI.jl's default sensitivity sigma: 7 mm, or the median field of view if smaller.
fn default_sensitivity_sigma_mm(grid: &Grid) -> f64 {
    let (nx, ny, nz) = grid.dims;
    let (vx, vy, vz) = grid.voxel_size;
    let mut fov = [nx as f64 * vx, ny as f64 * vy, nz as f64 * vz];
    fov.sort_by(f64::total_cmp);
    fov[1].min(7.0)
}

/// The high-pass filtered, echo-combined phase that drives the SWI phase mask.
///
/// As CLEARSWI.jl's `getcombinedphase` for its three unwrapping options. With several echoes
/// the result is a frequency map (`φ / (2π·TE)`, Hz for TEs in seconds); with one echo it is
/// the filtered phase in radians. Zero outside `mask`.
pub fn swi_phase<P: AsRef<[f64]> + Sync, M: AsRef<[f64]> + Sync>(
    phases: &[P],
    mags: &[M],
    tes: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &SwiParams,
) -> Vec<f64> {
    let n_total = grid.n_total();
    let everywhere = vec![1u8; n_total];
    let combine = |filtered: &[Vec<f64>]| -> Vec<f64> {
        if filtered.len() == 1 {
            filtered[0].clone()
        } else {
            calculate_b0_weighted(filtered, mags, tes, mask, params.echo_weighting, grid)
        }
    };

    match params.unwrap {
        SwiUnwrap::Laplacian => {
            let filtered: Vec<Vec<f64>> = phases
                .iter()
                .map(|p| {
                    let unwrapped = laplacian_unwrap(p.as_ref(), &everywhere, grid, LaplacianSolver::Dct);
                    highpass_filter(&unwrapped, mask, grid, params.hp_sigma)
                })
                .collect();
            combine(&filtered)
        }
        SwiUnwrap::LaplacianSlice => {
            let (nx, ny, nz) = grid.dims;
            let (vx, vy, vz) = grid.voxel_size;
            let slice_grid = Grid::new(nx, ny, 1, vx, vy, vz);
            let nxy = nx * ny;
            let ones = vec![1u8; nxy];
            let sigma = [params.hp_sigma[0], params.hp_sigma[1], 0.0];
            let filtered: Vec<Vec<f64>> = phases
                .iter()
                .map(|p| {
                    let mut out = vec![0.0; n_total];
                    for k in 0..nz {
                        let range = k * nxy..(k + 1) * nxy;
                        let unwrapped =
                            laplacian_unwrap(&p.as_ref()[range.clone()], &ones, &slice_grid, LaplacianSolver::Dct);
                        let hp = highpass_filter(&unwrapped, &mask[range.clone()], &slice_grid, sigma);
                        out[range].copy_from_slice(&hp);
                    }
                    out
                })
                .collect();
            combine(&filtered)
        }
        SwiUnwrap::Romeo => {
            // CLEARSWI.jl calls `romeo(phase; mag, TEs)` without a mask.
            let unwrapped = unwrap_romeo_multi_echo(phases, mags, tes, &everywhere, &RomeoParams::default(), grid);
            let combined = combine(&unwrapped);
            // CLEARSWI.jl filters the combined map in-plane only (`dims=1:2`).
            highpass_filter(&combined, mask, grid, [params.hp_sigma[0], params.hp_sigma[1], 0.0])
        }
    }
}

/// CLEAR-SWI from multi-echo (or single-echo) phase and magnitude.
///
/// As CLEARSWI.jl's `calculateSWI(Data(mag, phase, header, TEs), options)`. See the
/// [module docs](self) for the steps.
///
/// # Arguments
/// * `phases` - Wrapped phase, one volume per echo, in radians
/// * `mags` - Magnitude, one volume per echo
/// * `tes` - Echo times, one per echo (seconds; only their ratios and the weighting use them)
/// * `mask` - Mask the phase is filtered and scaled within. `None` uses CLEAR-SWI's own:
///   [`robust_mask`] of the first echo's magnitude, which covers the head
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `params` - CLEAR-SWI parameters
///
/// # Errors
/// If there are no echoes, the phase, magnitude and echo-time counts differ, an echo time is
/// not positive and finite (several echoes only), or a volume does not match `grid`.
pub fn calculate_swi<P: AsRef<[f64]> + Sync, M: AsRef<[f64]> + Sync>(
    phases: &[P],
    mags: &[M],
    tes: &[f64],
    mask: Option<&[u8]>,
    grid: &Grid,
    params: &SwiParams,
) -> Result<SwiResult, InputError> {
    let n_total = grid.n_total();
    let n_echoes = phases.len();
    if n_echoes == 0 {
        return Err(InputError::new("SWI needs at least one echo"));
    }
    if mags.len() != n_echoes || tes.len() != n_echoes {
        return Err(InputError::new(format!(
            "SWI got {} phase echoes, {} magnitude echoes and {} echo times; they must match",
            n_echoes, mags.len(), tes.len(),
        )));
    }
    if n_echoes > 1 && !tes.iter().all(|&t| t.is_finite() && t > 0.0) {
        return Err(InputError::new(format!("SWI echo times must be positive, got {tes:?}")));
    }
    for (e, (p, m)) in phases.iter().zip(mags).enumerate() {
        crate::error::check_len(&format!("SWI phase echo {}", e + 1), p.as_ref().len(), n_total)?;
        crate::error::check_len(&format!("SWI magnitude echo {}", e + 1), m.as_ref().len(), n_total)?;
    }
    if let Some(m) = mask {
        crate::error::check_len("SWI mask", m.len(), n_total)?;
    }

    let mask: Vec<u8> = match mask {
        Some(m) => m.iter().map(|&v| u8::from(v != 0)).collect(),
        None => robust_mask(mags[0].as_ref(), grid),
    };
    let magnitude = swi_magnitude(mags, grid, params)?;
    let filtered_phase = swi_phase(phases, mags, tes, &mask, grid, params);
    let phase_mask = create_phase_mask(&filtered_phase, &mask, params.scaling, params.strength);
    let swi = magnitude.iter().zip(&phase_mask).map(|(&m, &p)| m * p).collect();
    Ok(SwiResult { swi, magnitude, phase_mask, filtered_phase, mask })
}


/// A minimum-intensity projection, and the geometry it actually lives on.
///
/// A projection is not the volume it came from: it is shorter along the slice axis, and each of
/// its slices stands for a slab rather than a plane. Both facts travel with the data here so a
/// caller cannot write it out under the source volume's geometry — which produces a file whose
/// header promises more voxels than it holds, and whose slices are misplaced in world space.
#[derive(Debug, Clone)]
pub struct Mip {
    /// Projection values, Fortran order, `grid.n_total()` of them.
    pub data: Vec<f64>,
    /// Dimensions and voxel sizes of the projection.
    pub grid: Grid,
    /// Row-major 4×4 voxel→world affine, with the origin at the centre of the first slab.
    pub affine: [f64; 16],
}

/// Geometry of a sliding projection over `window` slices, without computing one.
///
/// Returns the projection's grid and affine. Use it to size or place a projection ahead of
/// time — to validate a window before doing the work, or to allocate an output file.
///
/// The slice axis loses `window - 1` slices, since a sliding window of `window` slices has
/// `nz - window + 1` positions. Each output slice represents the **centre** of its slab, so the
/// origin moves `(window - 1) / 2` slices along the slice direction. That shift follows the
/// affine's third column rather than world z, which keeps it correct for an oblique
/// acquisition; an even window lands the origin on a half-slice offset, as the convention
/// implies.
///
/// # Errors
/// If `window` is zero, or deeper than the volume.
pub fn mip_geometry(
    grid: &Grid,
    affine: &[f64; 16],
    window: usize,
) -> Result<(Grid, [f64; 16]), String> {
    let (nx, ny, nz) = grid.dims;
    if window == 0 {
        return Err("MIP window must be at least 1 slice".to_string());
    }
    if window > nz {
        return Err(format!(
            "MIP window of {} slices is deeper than the {}-slice volume",
            window, nz,
        ));
    }

    let (vsx, vsy, vsz) = grid.voxel_size;
    let mip_grid = Grid::new(nx, ny, nz - window + 1, vsx, vsy, vsz);

    let mut mip_affine = *affine;
    let slabs = (window - 1) as f64 / 2.0;
    for row in 0..3 {
        // Row-major 4×4: column 2 is the slice direction, column 3 the origin.
        mip_affine[row * 4 + 3] = affine[row * 4 + 3] + affine[row * 4 + 2] * slabs;
    }

    Ok((mip_grid, mip_affine))
}

/// Minimum intensity projection along the z-axis.
///
/// For each (x, y) position, takes the minimum value over a sliding window of `window` slices
/// along z. The result is `window - 1` slices shorter than `data` and sits half a slab further
/// along the slice direction, so it comes back as a [`Mip`] carrying its own grid and affine —
/// see [`mip_geometry`] for the convention.
///
/// # Arguments
/// * `data` - 3D volume (Fortran order)
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `affine` - Row-major 4×4 voxel→world affine of `data`. Pass an identity affine if the
///   caller has no world geometry; the projection's affine is then identity-with-origin-shift.
/// * `window` - Number of slices in the projection window
///
/// # Errors
/// If `window` is zero or deeper than the volume, or if `data` does not match `grid`.
pub fn create_mip(
    data: &[f64],
    grid: &Grid,
    affine: &[f64; 16],
    window: usize,
) -> Result<Mip, String> {
    let (nx, ny, nz) = grid.dims;
    if data.len() != grid.n_total() {
        return Err(format!(
            "MIP input is {} values but its {}x{}x{} grid has {} voxels",
            data.len(), nx, ny, nz, grid.n_total(),
        ));
    }
    let (mip_grid, mip_affine) = mip_geometry(grid, affine, window)?;

    let nz_out = mip_grid.nz();
    let nxy = nx * ny;
    let mut mip = vec![0.0; nxy * nz_out];

    for k_out in 0..nz_out {
        for j in 0..ny {
            for i in 0..nx {
                let idx_xy = i + j * nx;
                let mut min_val = data[idx_xy + k_out * nxy];
                for kw in 1..window {
                    let val = data[idx_xy + (k_out + kw) * nxy];
                    if val < min_val {
                        min_val = val;
                    }
                }
                mip[idx_xy + k_out * nxy] = min_val;
            }
        }
    }

    debug_assert_eq!(mip.len(), mip_grid.n_total());
    Ok(Mip { data: mip, grid: mip_grid, affine: mip_affine })
}

/// Softplus magnitude scaling for enhanced contrast
///
/// Applies a shifted softplus function: `softplus(x) - softplus(0)` where
/// `softplus(x) = (log(1 + exp(-|f*(x-offset)|)) + max(0, f*(x-offset))) / f`
/// and `f = factor / offset`.
///
/// # Arguments
/// * `magnitude` - Input magnitude data
/// * `offset` - Softplus offset (controls transition point)
/// * `factor` - Steepness factor (default 2.0)
///
/// # Returns
/// Scaled magnitude
pub fn softplus_scaling(
    magnitude: &[f64],
    offset: f64,
    factor: f64,
) -> Vec<f64> {
    if offset.abs() < 1e-20 {
        return magnitude.to_vec();
    }

    let f = factor / offset;

    // softplus(0) for baseline subtraction
    let arg0 = f * (0.0 - offset);
    let sp0 = ((1.0 + (-arg0.abs()).exp()).ln() + arg0.max(0.0)) / f;

    magnitude.iter().map(|&val| {
        let arg = f * (val - offset);
        let sp = ((1.0 + (-arg.abs()).exp()).ln() + arg.max(0.0)) / f;
        sp - sp0
    }).collect()
}

// ---- Susceptibility Map-Weighted Imaging (SMWI) ----

/// SMWI algorithm parameters
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct SmwiParams {
    /// Susceptibility threshold in ppm: the |χ| at which the weighting mask
    /// reaches zero. Must be positive; non-positive values fall back to 1 ppm.
    pub threshold_ppm: f64,
    /// Power the mask is raised to (contrast strength)
    pub power: f64,
    /// mIP window size in slices
    pub mip_window: usize,
}

impl Default for SmwiParams {
    fn default() -> Self {
        Self {
            threshold_ppm: 1.0,
            power: 4.0,
            mip_window: 4,
        }
    }
}

/// Which susceptibility sources the weighting mask suppresses
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmwiContrast {
    /// Suppress paramagnetic sources (χ > 0): veins, iron, microbleeds
    Paramagnetic,
    /// Suppress diamagnetic sources (χ < 0): calcification, myelin
    Diamagnetic,
}

/// Build the SMWI weighting mask from a susceptibility map
///
/// ```text
/// paramagnetic: w(χ) = clamp(1 - χ/t, 0, 1)
/// diamagnetic:  w(χ) = clamp(1 + χ/t, 0, 1)
/// ```
///
/// Non-finite susceptibilities map to 0 so they cannot propagate into the
/// weighted image or a later mIP.
///
/// # Arguments
/// * `chi_ppm` - Susceptibility map in ppm
/// * `threshold_ppm` - Susceptibility threshold in ppm (non-positive → 1 ppm)
/// * `contrast` - Which sources to suppress
///
/// # Returns
/// Weighting mask with values in [0, 1]
pub fn susceptibility_mask(
    chi_ppm: &[f64],
    threshold_ppm: f64,
    contrast: SmwiContrast,
) -> Vec<f64> {
    let t = if threshold_ppm > 0.0 { threshold_ppm } else { 1.0 };
    let sign = match contrast {
        SmwiContrast::Paramagnetic => -1.0,
        SmwiContrast::Diamagnetic => 1.0,
    };

    chi_ppm
        .iter()
        .map(|&chi| {
            if chi.is_finite() {
                (1.0 + sign * chi / t).clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .collect()
}

/// Calculate SMWI from magnitude and a susceptibility map
///
/// Pipeline: susceptibility map → weighting mask → magnitude × mask^power.
/// Unlike SWI, the weighting comes from χ rather than high-pass filtered
/// phase, so it is free of the blooming and residual-wrap artifacts that the
/// non-local dipole field introduces into a phase mask.
///
/// `magnitude` may hold several echoes stacked along the slowest axis
/// (length `n_echoes × n_total`, as in `[x, y, z, echo]` Fortran order); the
/// single χ-derived mask is then broadcast across echoes — the multi-echo
/// SMWI of Nam et al. Use [`average_echoes`] to combine the result.
///
/// # Arguments
/// * `magnitude` - Magnitude image(s), `n_total` or `n_echoes × n_total` values
/// * `chi_ppm` - Susceptibility map in ppm (`n_total` values)
/// * `mask` - Optional binary brain mask (1 = inside); outside is zeroed
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `params` - SMWI algorithm parameters
///
/// # Returns
/// `(paramagnetic, diamagnetic)` weighted images, each shaped like `magnitude`
///
/// # Panics
/// If `chi_ppm` is not `n_total` long, or `magnitude` is not a non-zero
/// multiple of `n_total`.
pub fn calculate_smwi(
    magnitude: &[f64],
    chi_ppm: &[f64],
    mask: Option<&[u8]>,
    grid: &Grid,
    params: &SmwiParams,
) -> (Vec<f64>, Vec<f64>) {
    let n_total = grid.n_total();
    assert_eq!(
        chi_ppm.len(),
        n_total,
        "SMWI: susceptibility map has {} voxels, grid has {}",
        chi_ppm.len(),
        n_total
    );
    assert!(
        !magnitude.is_empty() && magnitude.len().is_multiple_of(n_total),
        "SMWI: magnitude length {} is not a non-zero multiple of the {} grid voxels",
        magnitude.len(),
        n_total
    );

    let p_mask = susceptibility_mask(chi_ppm, params.threshold_ppm, SmwiContrast::Paramagnetic);
    let d_mask = susceptibility_mask(chi_ppm, params.threshold_ppm, SmwiContrast::Diamagnetic);

    // Weight once per voxel, then broadcast over echoes
    let p_weight: Vec<f64> = p_mask.iter().map(|&w| w.powf(params.power)).collect();
    let d_weight: Vec<f64> = d_mask.iter().map(|&w| w.powf(params.power)).collect();

    let mut p_smwi = vec![0.0; magnitude.len()];
    let mut d_smwi = vec![0.0; magnitude.len()];

    let inside = |v: usize| mask.is_none_or(|m| m[v] != 0);

    for (i, &m) in magnitude.iter().enumerate() {
        let v = i % n_total;
        if !inside(v) {
            continue;
        }
        p_smwi[i] = m * p_weight[v];
        d_smwi[i] = m * d_weight[v];
    }

    (p_smwi, d_smwi)
}

/// Average a multi-echo volume over its echoes
///
/// Input is `n_echoes × n_total` values stacked along the slowest axis;
/// output is a single `n_total` volume. A single-echo input is returned
/// unchanged.
///
/// # Panics
/// If `data` is not a non-zero multiple of `grid.n_total()`.
pub fn average_echoes(data: &[f64], grid: &Grid) -> Vec<f64> {
    let n_total = grid.n_total();
    assert!(
        !data.is_empty() && data.len().is_multiple_of(n_total),
        "average_echoes: length {} is not a non-zero multiple of the {} grid voxels",
        data.len(),
        n_total
    );

    let n_echoes = data.len() / n_total;
    let mut out = vec![0.0; n_total];
    for (i, &v) in data.iter().enumerate() {
        out[i % n_total] += v;
    }
    for v in &mut out {
        *v /= n_echoes as f64;
    }
    out
}

// ---- Helpers ----

/// Median of `v` (reordered in place), or `None` if empty.
fn median(v: &mut [f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_unstable_by(f64::total_cmp);
    let mid = v.len() / 2;
    Some(if v.len().is_multiple_of(2) { (v[mid - 1] + v[mid]) / 2.0 } else { v[mid] })
}

/// MriResearchTools' `sample`: up to 1e5 finite values taken as `√n` evenly spaced blocks of
/// `√n` consecutive values, so a quantile of a large volume is cheap and deterministic.
fn sample(data: &[f64]) -> Vec<f64> {
    let n = data.len().min(100_000);
    if n == 0 {
        return Vec::new();
    }
    let len = (n as f64).sqrt().ceil() as usize;
    let last_start = (data.len() - len) as f64;
    let mut out = Vec::with_capacity(len * len);
    for b in 0..len {
        // Julia's `round(Int, range(0, N - len; length = len))`: ties to even.
        let start = if len == 1 { 0.0 } else { last_start * b as f64 / (len - 1) as f64 };
        let start = start.round_ties_even() as usize;
        out.extend(data[start..start + len].iter().copied().filter(|v| v.is_finite()));
    }
    if out.is_empty() {
        out = data.iter().copied().filter(|v| v.is_finite()).collect();
    }
    out
}

/// MriResearchTools' `estimatequantile`: quantiles `ps` of a block sample of `data`, with
/// Julia's default (type 7, linear) interpolation. NaN if `data` holds no finite value.
fn estimate_quantiles(data: &[f64], ps: &[f64]) -> Vec<f64> {
    let mut s = sample(data);
    if s.is_empty() {
        return vec![f64::NAN; ps.len()];
    }
    s.sort_unstable_by(f64::total_cmp);
    ps.iter()
        .map(|&p| {
            let h = (s.len() - 1) as f64 * p;
            let lo = h.floor() as usize;
            let hi = (lo + 1).min(s.len() - 1);
            s[lo] + (h - lo as f64) * (s[hi] - s[lo])
        })
        .collect()
}

fn estimate_quantile(data: &[f64], p: f64) -> f64 {
    estimate_quantiles(data, &[p])[0]
}

/// CLEARSWI.jl's `robustrescale(x[sel], new_min, new_max) .^ strength` on the voxels whose
/// `pos` equals `positive`: the 1st and 99th percentiles, extrapolated to 0 % and 100 %, map
/// to `new_min` and `new_max`. Clamped to [0, 1] before the power.
fn rescale_selected(
    data: &mut [f64],
    pos: &[bool],
    positive: bool,
    new_min: f64,
    new_max: f64,
    strength: f64,
) {
    let selected: Vec<f64> = data.iter().zip(pos).filter(|&(_, &p)| p == positive).map(|(&v, _)| v).collect();
    if selected.is_empty() {
        return;
    }
    let q = estimate_quantiles(&selected, &[0.01, 0.99]);
    let old_range = (q[1] - q[0]) / 0.98;
    let old_min = q[0] - 0.01 * old_range;
    for (v, &p) in data.iter_mut().zip(pos) {
        if p != positive {
            continue;
        }
        let r = if old_range.abs() > 1e-20 && old_range.is_finite() {
            (*v - old_min) * ((new_max - new_min) / old_range) + new_min
        } else {
            (new_min + new_max) / 2.0
        };
        *v = r.clamp(0.0, 1.0).powf(strength);
    }
}

/// Set the voxels whose `pos` equals `positive` to `value`.
fn fill_selected(data: &mut [f64], pos: &[bool], positive: bool, value: f64) {
    for (v, &p) in data.iter_mut().zip(pos) {
        if p == positive {
            *v = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const F0: f64 = 0.880_797_077_977_882_3; // (1 + tanh(1)) / 2

    /// A ball of radius `r` voxels at the centre of an `n`³ grid.
    fn ball(n: usize, r: f64) -> Vec<bool> {
        let c = (n as f64 - 1.0) / 2.0;
        (0..n * n * n)
            .map(|i| {
                let (x, y, z) = ((i % n) as f64, ((i / n) % n) as f64, (i / (n * n)) as f64);
                (x - c).powi(2) + (y - c).powi(2) + (z - c).powi(2) <= r * r
            })
            .collect()
    }

    /// Echo phases `ω·TE` (wrapped) of a frequency map `ω` in rad/s.
    fn echoes(omega: &[f64], tes: &[f64]) -> Vec<Vec<f64>> {
        let wrap = |p: f64| (p + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI;
        tes.iter().map(|&te| omega.iter().map(|&w| wrap(w * te)).collect()).collect()
    }

    /// A smooth background frequency plus a small paramagnetic bump, in rad/s.
    fn frequency(n: usize, background: f64, bump: f64) -> Vec<f64> {
        let c = (n as f64 - 1.0) / 2.0;
        (0..n * n * n)
            .map(|i| {
                let (x, y, z) = ((i % n) as f64, ((i / n) % n) as f64, (i / (n * n)) as f64);
                let d2 = (x - c - 3.0).powi(2) + (y - c).powi(2) + (z - c).powi(2);
                background * (x - c) / n as f64 + bump * (-d2 / 4.0).exp()
            })
            .collect()
    }

    #[test]
    fn test_calculate_swi_zero_phase() {
        let n = 8;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let phase = vec![vec![0.0; n * n * n]];
        let mag = vec![vec![1.0; n * n * n]];
        let mask = vec![1u8; n * n * n];
        let r = calculate_swi(&phase, &mag, &[0.01], Some(&mask), &grid, &SwiParams::default()).unwrap();
        // Zero phase: no positive phase, so m = 1 and every voxel gets tanh's value at zero.
        for (&p, &s) in r.phase_mask.iter().zip(&r.swi) {
            assert!((p - F0).abs() < 1e-12, "{p}");
            assert!(s.is_finite() && s >= 0.0);
        }
    }

    #[test]
    fn test_phase_mask_outside_mask_is_the_zero_phase_weight() {
        // CLEARSWI.jl zeroes the phase outside the mask, then scales every voxel: the head
        // outside the mask keeps its magnitude (× f(0)) rather than going black.
        let phase = vec![0.3, -0.2, 0.5, 0.7];
        let mask = vec![1u8, 1, 0, 0];
        let tanh = create_phase_mask(&phase, &mask, PhaseScaling::Tanh, 4.0);
        assert!((tanh[2] - F0).abs() < 1e-12 && (tanh[3] - F0).abs() < 1e-12, "{tanh:?}");
        let positive = create_phase_mask(&phase, &mask, PhaseScaling::Positive, 4.0);
        assert_eq!(&positive[2..], &[1.0, 1.0]);
        // Negative phase is untouched by the positive mask.
        assert_eq!(positive[1], 1.0);
    }

    #[test]
    fn test_tanh_mask_formula() {
        // m = median(positive phase in mask) · 10 / strength
        let phase = vec![0.1, 0.2, 0.3, -0.4];
        let mask = vec![1u8; 4];
        let pm = create_phase_mask(&phase, &mask, PhaseScaling::Tanh, 4.0);
        let m = 0.2 * 10.0 / 4.0;
        for (i, &p) in phase.iter().enumerate() {
            let want = (1.0 + (1.0 - p / m).tanh()) / 2.0;
            assert!((pm[i] - want).abs() < 1e-12, "{i}: {} vs {want}", pm[i]);
        }
        // NegativeTanh is Tanh of the negated phase.
        let neg: Vec<f64> = phase.iter().map(|p| -p).collect();
        assert_eq!(
            create_phase_mask(&phase, &mask, PhaseScaling::NegativeTanh, 4.0),
            create_phase_mask(&neg, &mask, PhaseScaling::Tanh, 4.0),
        );
    }

    #[test]
    fn test_estimate_quantile_matches_mriresearchtools() {
        // Reference values from MriResearchTools 2.2 `estimatequantile` (Julia 1.9).
        let x: Vec<f64> = (1..=10).map(|i| (i * i) as f64).collect();
        assert_eq!(estimate_quantile(&x, 0.8), 64.0);
        let y: Vec<f64> = (1..=1000).map(|i| (0.37 * i as f64).sin() * i as f64).collect();
        let q = estimate_quantiles(&y, &[0.01, 0.99]);
        assert!((q[0] - -893.956_867_457_096_1).abs() < 1e-9, "{q:?}");
        assert!((q[1] - 898.654_032_022_898_7).abs() < 1e-9, "{q:?}");
        let z: Vec<f64> = (1..=300_000).map(|i| (0.011 * i as f64).cos() + 0.001 * i as f64).collect();
        assert!((estimate_quantile(&z, 0.8) - 240.122_157_209_175_4).abs() < 1e-9);
    }

    #[test]
    fn test_combine_magnitude() {
        let mags = vec![vec![3.0, 1.0], vec![4.0, 2.0]];
        assert_eq!(combine_magnitude(&mags, MagCombine::Rss).unwrap(), vec![5.0, 5f64.sqrt()]);
        assert_eq!(combine_magnitude(&mags, MagCombine::Average).unwrap(), vec![7.0, 3.0]);
        assert_eq!(combine_magnitude(&mags, MagCombine::Last).unwrap(), vec![4.0, 2.0]);
        assert_eq!(combine_magnitude(&mags, MagCombine::Echo(0)).unwrap(), vec![3.0, 1.0]);
        assert!(combine_magnitude(&mags, MagCombine::Echo(2)).is_err());
        // One echo is that echo, whatever the combination.
        assert_eq!(combine_magnitude(&mags[..1], MagCombine::Rss).unwrap(), vec![3.0, 1.0]);
    }

    #[test]
    fn test_calculate_swi_rejects_mismatched_echoes() {
        let grid = Grid::new(2, 2, 2, 1.0, 1.0, 1.0);
        let v = vec![vec![0.0; 8]; 2];
        let err = calculate_swi(&v, &v[..1], &[0.01, 0.02], None, &grid, &SwiParams::default()).unwrap_err();
        assert!(err.0.contains("must match"), "{err}");
        let err = calculate_swi(&v, &v, &[0.01, 0.0], None, &grid, &SwiParams::default()).unwrap_err();
        assert!(err.0.contains("positive"), "{err}");
        let short = vec![vec![0.0; 7]];
        assert!(calculate_swi(&short, &short, &[0.01], None, &grid, &SwiParams::default()).is_err());
        let none: Vec<Vec<f64>> = Vec::new();
        assert!(calculate_swi(&none, &none, &[], None, &grid, &SwiParams::default()).is_err());
    }

    #[test]
    fn test_multi_echo_phase_matches_single_echo_without_wraps() {
        // Noise-free phase linear in TE, small enough not to wrap: every echo carries the same
        // frequency map, so the echo-combined filtered phase is the single-echo filtered phase
        // over 2π·TE, and the (scale-invariant) phase mask is the same.
        let n = 24;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let omega = frequency(n, 20.0, 60.0);
        let tes = [0.004, 0.008, 0.012];
        let phases = echoes(&omega, &tes);
        let mags = vec![vec![100.0; n * n * n], vec![80.0; n * n * n], vec![60.0; n * n * n]];
        let mask: Vec<u8> = ball(n, 10.0).iter().map(|&b| u8::from(b)).collect();
        let params = SwiParams::default();

        let multi = calculate_swi(&phases, &mags, &tes, Some(&mask), &grid, &params).unwrap();
        let single = calculate_swi(&phases[2..], &mags[2..], &tes[2..], Some(&mask), &grid, &params).unwrap();
        for i in 0..n * n * n {
            if mask[i] != 0 {
                let want = single.filtered_phase[i] / (2.0 * std::f64::consts::PI * tes[2]);
                assert!((multi.filtered_phase[i] - want).abs() < 1e-6, "{i}: {} vs {want}", multi.filtered_phase[i]);
            }
            assert!((multi.phase_mask[i] - single.phase_mask[i]).abs() < 1e-9, "{i}");
        }
        // The bump is paramagnetic: the phase mask darkens it.
        let centre = (n / 2 + 3) + (n / 2) * n + (n / 2) * n * n;
        assert!(multi.phase_mask[centre] < 0.5, "{}", multi.phase_mask[centre]);
    }

    #[test]
    fn test_romeo_and_laplacian_agree_on_wrapped_phase() {
        // A background gradient that wraps the later echoes several times: both unwrappers
        // must recover the bump.
        let n = 24;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let omega = frequency(n, 1200.0, 300.0);
        let tes = [0.005, 0.010, 0.015];
        let phases = echoes(&omega, &tes);
        let peak = omega.iter().fold(0.0f64, |a, w| a.max(w.abs())) * tes[2];
        assert!(peak > 2.0 * std::f64::consts::PI, "test phase should wrap, peak {peak}");
        let mags = vec![vec![100.0; n * n * n]; 3];
        let mask: Vec<u8> = ball(n, 10.0).iter().map(|&b| u8::from(b)).collect();
        let centre = (n / 2 + 3) + (n / 2) * n + (n / 2) * n * n;
        for unwrap in [SwiUnwrap::Laplacian, SwiUnwrap::LaplacianSlice, SwiUnwrap::Romeo] {
            let params = SwiParams { unwrap, ..SwiParams::default() };
            let r = calculate_swi(&phases, &mags, &tes, Some(&mask), &grid, &params).unwrap();
            assert!(r.phase_mask[centre] < 0.5, "{unwrap:?}: bump weight {}", r.phase_mask[centre]);
            assert!(r.swi.iter().all(|v| v.is_finite()), "{unwrap:?}");
        }
    }

    #[test]
    fn test_default_mask_is_robust_mask_of_first_echo() {
        let n = 16;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let inside = ball(n, 6.0);
        let mag: Vec<f64> = inside.iter().map(|&b| if b { 100.0 } else { 1.0 }).collect();
        let phase = vec![0.0; n * n * n];
        let r = calculate_swi(&[phase], std::slice::from_ref(&mag), &[0.02], None, &grid, &SwiParams::default()).unwrap();
        assert_eq!(r.mask, robust_mask(&mag, &grid));
    }

    #[test]
    fn test_phase_mask_range() {
        let n = 10;
        let nn = n * n * n;
        let phase: Vec<f64> = (0..nn).map(|i| (i as f64 * 0.01) - 5.0).collect();
        let mask = vec![1u8; nn];

        for scaling in &[
            PhaseScaling::Tanh,
            PhaseScaling::NegativeTanh,
            PhaseScaling::Positive,
            PhaseScaling::Negative,
            PhaseScaling::Triangular,
        ] {
            let pm = create_phase_mask(&phase, &mask, *scaling, 4.0);
            for (i, &v) in pm.iter().enumerate() {
                assert!(v >= 0.0, "{:?}: value at {} = {} < 0", scaling, i, v);
                assert!(v <= 1.0 + 1e-10, "{:?}: value at {} = {} > 1", scaling, i, v);
            }
        }
    }

    #[test]
    fn test_highpass_filter_constant() {
        // Constant input should give zero output (constant is its own smooth)
        let n = 16;
        let nn = n * n * n;
        let data = vec![5.0; nn];
        let mask = vec![1u8; nn];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let result = highpass_filter(&data, &mask, &grid, [2.0, 2.0, 0.0]);

        for &v in &result {
            assert!(v.abs() < 1.0, "High-pass of constant should be near zero, got {}", v);
        }
    }

    const IDENTITY: [f64; 16] = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];

    #[test]
    fn test_mip_basic() {
        // 3x3x5 volume, mIP with window=3 → 3x3x3 output
        let (nx, ny, nz) = (3, 3, 5);
        let grid = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
        let mut data = vec![10.0; nx * ny * nz];
        // Place a low value at slice 2
        let idx = 1 + nx + 2 * nx * ny; // (x, y, z) = (1, 1, 2)
        data[idx] = 1.0;

        let mip = create_mip(&data, &grid, &IDENTITY, 3).unwrap();
        assert_eq!(mip.data.len(), nx * ny * 3);

        // The minimum at (1,1) should appear in every window that includes z=2: the windows
        // starting at z = 0, 1 and 2 cover slices 0-2, 1-3 and 2-4 respectively.
        // (A closure rather than three `1 + 1 * nx + k * nx * ny` expressions, which kept the
        // three lines visually parallel at the cost of a `0 *` that clippy denies.)
        let mip_idx = |window: usize| 1 + nx + window * nx * ny;
        for window in 0..3 {
            assert_eq!(
                mip.data[mip_idx(window)], 1.0,
                "window starting at z={window} covers z=2 and must see the minimum"
            );
        }
    }

    /// The projection must describe itself: a caller that trusts the source grid writes a file
    /// whose header promises more voxels than it holds (QSMxT#211).
    #[test]
    fn test_mip_reports_its_own_dimensions() {
        let grid = Grid::new(32, 32, 32, 1.0, 1.0, 1.0);
        let mip = create_mip(&vec![1.0; grid.n_total()], &grid, &IDENTITY, 7).unwrap();
        assert_eq!(mip.grid.dims, (32, 32, 26));
        assert_eq!(mip.data.len(), mip.grid.n_total());
        // Voxel sizes are unchanged by a projection.
        assert_eq!(mip.grid.voxel_size, grid.voxel_size);
    }

    #[test]
    fn test_mip_origin_moves_to_the_slab_centre() {
        let grid = Grid::new(4, 4, 16, 1.0, 1.0, 2.0);
        let mut affine = IDENTITY;
        affine[10] = 2.0; // 2 mm slices
        affine[11] = 10.0;
        let (mip_grid, mip_affine) = mip_geometry(&grid, &affine, 7).unwrap();
        assert_eq!(mip_grid.dims, (4, 4, 10));
        // Three slices of 2 mm past the first slab's first slice.
        assert_eq!(mip_affine[11], 16.0);
        // Nothing but the origin moves.
        assert_eq!(mip_affine[..11], affine[..11]);
    }

    #[test]
    fn test_mip_origin_follows_an_oblique_slice_direction() {
        // Slice direction tilted 45° in the y/z plane, 2 mm slices.
        let s = 2.0 / 2f64.sqrt();
        let affine = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, s, 0.0,
            0.0, 0.0, s, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let grid = Grid::new(8, 8, 20, 1.0, 1.0, 2.0);
        let (mip_grid, mip_affine) = mip_geometry(&grid, &affine, 7).unwrap();
        assert_eq!(mip_grid.dims, (8, 8, 14));
        // Both y and z move: three slice steps along the tilted direction, not along world z.
        assert!((mip_affine[7] - 3.0 * s).abs() < 1e-12, "{:?}", mip_affine);
        assert!((mip_affine[11] - 3.0 * s).abs() < 1e-12, "{:?}", mip_affine);
        assert_eq!(mip_affine[3], 0.0);
    }

    #[test]
    fn test_mip_even_window_lands_on_a_half_slice() {
        let grid = Grid::new(4, 4, 16, 1.0, 1.0, 1.0);
        let (_, mip_affine) = mip_geometry(&grid, &IDENTITY, 4).unwrap();
        assert_eq!(mip_affine[11], 1.5);
    }

    #[test]
    fn test_mip_window_of_one_is_the_volume_itself() {
        let grid = Grid::new(3, 3, 5, 1.0, 1.0, 1.0);
        let data: Vec<f64> = (0..grid.n_total()).map(|i| i as f64).collect();
        let mip = create_mip(&data, &grid, &IDENTITY, 1).unwrap();
        assert_eq!(mip.grid.dims, grid.dims);
        assert_eq!(mip.data, data);
        assert_eq!(mip.affine, IDENTITY);
    }

    #[test]
    fn test_mip_window_of_full_depth_leaves_one_slice() {
        let grid = Grid::new(3, 3, 5, 1.0, 1.0, 1.0);
        let mip = create_mip(&vec![1.0; grid.n_total()], &grid, &IDENTITY, 5).unwrap();
        assert_eq!(mip.grid.dims, (3, 3, 1));
        assert_eq!(mip.affine[11], 2.0);
    }

    #[test]
    fn test_mip_window_too_large() {
        let grid = Grid::new(3, 3, 3, 1.0, 1.0, 1.0);
        let err = create_mip(&[1.0; 27], &grid, &IDENTITY, 10).unwrap_err();
        assert!(err.contains("deeper than the 3-slice volume"), "{}", err);
    }

    #[test]
    fn test_mip_zero_window() {
        let grid = Grid::new(3, 3, 3, 1.0, 1.0, 1.0);
        let err = create_mip(&[1.0; 27], &grid, &IDENTITY, 0).unwrap_err();
        assert!(err.contains("at least 1 slice"), "{}", err);
    }

    #[test]
    fn test_mip_rejects_data_that_does_not_match_the_grid() {
        let grid = Grid::new(3, 3, 3, 1.0, 1.0, 1.0);
        let err = create_mip(&[1.0; 26], &grid, &IDENTITY, 3).unwrap_err();
        assert!(err.contains("26 values"), "{}", err);
        assert!(err.contains("27 voxels"), "{}", err);
    }

    #[test]
    fn test_softplus_scaling() {
        let mag = vec![0.0, 0.5, 1.0, 2.0];
        let result = softplus_scaling(&mag, 1.0, 2.0);

        // softplus(0, offset=1, factor=2) should be 0 (baseline subtracted)
        assert!(result[0].abs() < 1e-10, "softplus(0) should be ~0, got {}", result[0]);
        // Values should increase monotonically
        for i in 1..result.len() {
            assert!(result[i] >= result[i - 1], "softplus should be monotonically increasing");
        }
    }

    // ---- SMWI ----

    #[test]
    fn test_susceptibility_mask_paramagnetic() {
        // t = 0.5 ppm: 1 below zero, linear ramp to 0 at the threshold
        let chi = vec![-1.0, -0.001, 0.0, 0.125, 0.25, 0.5, 0.9];
        let w = susceptibility_mask(&chi, 0.5, SmwiContrast::Paramagnetic);
        let expected = [1.0, 1.0, 1.0, 0.75, 0.5, 0.0, 0.0];
        for (i, (&got, &want)) in w.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-12, "index {}: got {}, want {}", i, got, want);
        }
    }

    #[test]
    fn test_susceptibility_mask_diamagnetic() {
        // Mirror image of the paramagnetic mask about χ = 0
        let chi = vec![-0.9, -0.5, -0.25, 0.0, 0.25, 1.0];
        let w = susceptibility_mask(&chi, 0.5, SmwiContrast::Diamagnetic);
        let expected = [0.0, 0.0, 0.5, 1.0, 1.0, 1.0];
        for (i, (&got, &want)) in w.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-12, "index {}: got {}, want {}", i, got, want);
        }
    }

    #[test]
    fn test_susceptibility_mask_edge_cases() {
        // Non-finite χ is suppressed rather than propagated
        let chi = vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        for contrast in &[SmwiContrast::Paramagnetic, SmwiContrast::Diamagnetic] {
            let w = susceptibility_mask(&chi, 1.0, *contrast);
            assert!(w.iter().all(|&v| v == 0.0), "{:?}: non-finite χ should give 0", contrast);
        }

        // Non-positive threshold falls back to 1 ppm
        let chi = vec![0.5];
        let fallback = susceptibility_mask(&chi, -1.0, SmwiContrast::Paramagnetic);
        let one_ppm = susceptibility_mask(&chi, 1.0, SmwiContrast::Paramagnetic);
        assert_eq!(fallback, one_ppm);
    }

    #[test]
    fn test_smwi_reference_formula() {
        // magn .* max(0, 1 - χ/t).^m, hand-computed at t = 0.2 ppm, m = 4
        let grid = Grid::new(2, 1, 1, 1.0, 1.0, 1.0);
        let chi = vec![0.05, -0.05];
        let magnitude = vec![100.0, 100.0];
        let params = SmwiParams { threshold_ppm: 0.2, power: 4.0, mip_window: 4 };

        let (para, dia) = calculate_smwi(&magnitude, &chi, None, &grid, &params);

        // χ = +0.05: paramagnetic weight (1 - 0.25)^4 = 0.31640625, diamagnetic 1
        assert!((para[0] - 31.640625).abs() < 1e-9, "got {}", para[0]);
        assert!((dia[0] - 100.0).abs() < 1e-9, "got {}", dia[0]);
        // χ = -0.05: mirror image
        assert!((para[1] - 100.0).abs() < 1e-9, "got {}", para[1]);
        assert!((dia[1] - 31.640625).abs() < 1e-9, "got {}", dia[1]);
    }

    #[test]
    fn test_smwi_multi_echo_broadcast() {
        // One χ mask applied to every echo of a 4D magnitude
        let grid = Grid::new(2, 2, 2, 1.0, 1.0, 1.0);
        let n = grid.n_total();
        let chi: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let magnitude: Vec<f64> = (0..3 * n).map(|i| (i + 1) as f64).collect();
        let params = SmwiParams::default();

        let (para, _dia) = calculate_smwi(&magnitude, &chi, None, &grid, &params);
        assert_eq!(para.len(), magnitude.len());

        let weight = susceptibility_mask(&chi, params.threshold_ppm, SmwiContrast::Paramagnetic);
        for i in 0..magnitude.len() {
            let want = magnitude[i] * weight[i % n].powf(params.power);
            assert!((para[i] - want).abs() < 1e-12, "index {}: got {}, want {}", i, para[i], want);
        }

        // Single-echo call on the first echo must agree with the broadcast one
        let (para1, _) = calculate_smwi(&magnitude[..n], &chi, None, &grid, &params);
        assert_eq!(&para[..n], &para1[..]);
    }

    #[test]
    fn test_smwi_brain_mask() {
        let grid = Grid::new(2, 2, 2, 1.0, 1.0, 1.0);
        let n = grid.n_total();
        let chi = vec![0.0; n];
        let magnitude = vec![5.0; 2 * n];
        let mut mask = vec![1u8; n];
        mask[0] = 0;

        let (para, dia) = calculate_smwi(&magnitude, &chi, Some(&mask), &grid, &SmwiParams::default());

        // Masked-out voxel is zero in every echo, χ = 0 leaves the rest untouched
        assert_eq!(para[0], 0.0);
        assert_eq!(para[n], 0.0);
        assert_eq!(dia[0], 0.0);
        assert!((para[1] - 5.0).abs() < 1e-12);
        assert!((para[n + 1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_average_echoes() {
        let grid = Grid::new(2, 1, 1, 1.0, 1.0, 1.0);
        let data = vec![1.0, 2.0, 3.0, 4.0, 8.0, 12.0];
        let avg = average_echoes(&data, &grid);
        assert_eq!(avg, vec![4.0, 6.0]);

        // Single echo passes through
        let single = vec![7.0, 9.0];
        assert_eq!(average_echoes(&single, &grid), single);
    }

    #[test]
    #[should_panic(expected = "not a non-zero multiple")]
    fn test_smwi_rejects_ragged_magnitude() {
        let grid = Grid::new(2, 2, 2, 1.0, 1.0, 1.0);
        let chi = vec![0.0; grid.n_total()];
        let magnitude = vec![1.0; grid.n_total() + 1];
        let _ = calculate_smwi(&magnitude, &chi, None, &grid, &SmwiParams::default());
    }
}

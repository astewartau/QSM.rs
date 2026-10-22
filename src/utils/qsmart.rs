//! QSMART (Quantitative Susceptibility Mapping Artifact Reduction Technique)
//!
//! This module provides the complete QSMART pipeline including:
//! - Two-stage QSM reconstruction (whole ROI + tissue only)
//! - Offset adjustment for combining the two stages
//! - Integration of SDF background removal with iLSQR inversion
//!
//! Reference:
//! Yaghmaie, N., Syeda, W., et al. (2021).
//! "QSMART: Quantitative Susceptibility Mapping Artifact Reduction Technique."
//! NeuroImage, 231:117701. https://doi.org/10.1016/j.neuroimage.2020.117701
//!
//! Reference implementation: https://github.com/wtsyeda/QSMART

use crate::fft::{fft3d, ifft3d};
use crate::kernels::dipole::dipole_kernel;
use crate::pipeline::config::InversionAlgorithm;
use num_complex::Complex64;

/// Adjust offset between two-stage QSMART susceptibility maps
///
/// Combines chi_1 (whole ROI) and chi_2 (tissue only) with offset adjustment
/// to ensure consistency with the original field data.
///
/// # Arguments
/// * `removed_voxels` - Mask of removed voxels (mask * R_0 - vasc_only), indicates where stage 1 but not stage 2 was applied
/// * `lfs_sdf` - Local field shift from stage 1 SDF (in ppm, will be scaled back)
/// * `chi_1` - Susceptibility from stage 1 (whole ROI)
/// * `chi_2` - Susceptibility from stage 2 (tissue only)
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `b0_dir` - B0 field direction (unit vector)
/// * `ppm` - PPM conversion factor (to scale lfs_sdf back)
///
/// # Returns
/// Combined and offset-adjusted susceptibility map
pub fn adjust_offset(
    removed_voxels: &[f64],
    lfs_sdf: &[f64],
    chi_1: &[f64],
    chi_2: &[f64],
    grid: &crate::Grid,
    b0_dir: (f64, f64, f64),
    ppm: f64,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    // Scale lfs_sdf back (it was multiplied by ppm in the QSMART pipeline)
    let lfs_scaled: Vec<f64> = lfs_sdf.iter().map(|&v| v / ppm).collect();

    // Clean removed_voxels (clamp negatives to 0)
    let removed_clean: Vec<f64> = removed_voxels.iter()
        .map(|&v| if v < 0.0 { 0.0 } else { v })
        .collect();

    // Zero out chi_1 where removed_voxels is 0
    let chi_1_masked: Vec<f64> = chi_1.iter()
        .zip(removed_clean.iter())
        .map(|(&c, &r)| if r > 0.0 { c } else { 0.0 })
        .collect();

    // Combined chi = chi_1_masked + chi_2
    let combined_chi: Vec<f64> = chi_1_masked.iter()
        .zip(chi_2.iter())
        .map(|(&c1, &c2)| c1 + c2)
        .collect();

    // Get dipole kernel
    let d_kernel = dipole_kernel(grid, b0_dir);

    // Compute offset using Fourier-space relationship
    // x1 = ifft(D * fft(removed_voxels))
    // x2 = lfs_sdf - ifft(D * fft(combined_chi))
    // offset = real(x1' * x2) / real(x1' * x1)

    // Convert to complex for FFT
    let mut removed_complex: Vec<Complex64> = removed_clean.iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect();
    let mut combined_complex: Vec<Complex64> = combined_chi.iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect();

    // FFT of removed_voxels and combined_chi
    fft3d(&mut removed_complex, nx, ny, nz);
    fft3d(&mut combined_complex, nx, ny, nz);

    // Multiply by dipole kernel
    let mut d_removed: Vec<Complex64> = removed_complex.iter()
        .zip(d_kernel.iter())
        .map(|(&c, &d)| c * d)
        .collect();

    let mut d_combined: Vec<Complex64> = combined_complex.iter()
        .zip(d_kernel.iter())
        .map(|(&c, &d)| c * d)
        .collect();

    // Inverse FFT
    ifft3d(&mut d_removed, nx, ny, nz);
    ifft3d(&mut d_combined, nx, ny, nz);

    // x1 = real part of ifft(D * fft(removed))
    let x1: Vec<f64> = d_removed.iter().map(|c| c.re).collect();

    // x2 = lfs_sdf - real(ifft(D * fft(combined_chi)))
    let x2: Vec<f64> = lfs_scaled.iter()
        .zip(d_combined.iter())
        .map(|(&lfs, c)| lfs - c.re)
        .collect();

    // Compute offset: o = (x1' * x2) / (x1' * x1)
    let x1_dot_x2: f64 = x1.iter().zip(x2.iter()).map(|(&a, &b)| a * b).sum();
    let x1_dot_x1: f64 = x1.iter().map(|&a| a * a).sum();

    let offset = if x1_dot_x1.abs() > 1e-10 {
        x1_dot_x2 / x1_dot_x1
    } else {
        0.0
    };

    // Adjusted combined chi = combined_chi + offset * removed_voxels
    let adjusted: Vec<f64> = combined_chi.iter()
        .zip(removed_clean.iter())
        .map(|(&c, &r)| c + offset * r)
        .collect();

    adjusted
}

/// QSMART's proton gyromagnetic ratio in rad/s/T (`Demo_QSMART.m`'s `gyro`): the reference
/// implementation's value, not `crate::constants`. Keep it.
pub const QSMART_GYRO: f64 = 2.675e8;

/// QSMART's fixed echo-fit residual threshold for `R_0` (`Demo_QSMART.m`'s `fit_threshold`,
/// used when its `adaptive_threshold` is off).
pub const QSMART_FIT_THRESHOLD: f64 = 40.0;

/// QSMART's adaptive echo-fit threshold: the percentile of the blurred residual
/// (`fit_thresh_percentile`) that its authors use.
pub const QSMART_FIT_PERCENTILE: f64 = 99.0;

/// How [`qsmart_echofit`] sets the residual threshold for `R_0` (`echofit.m`'s
/// `adaptive_threshold`).
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QsmartFitThreshold {
    /// Adaptive: the given percentile (0–100] of the nonzero blurred residuals over the whole
    /// volume, as MATLAB's `prctile(fit_residual_blur(fit_residual_blur~=0), p)`. QSMART's authors
    /// run with [`QSMART_FIT_PERCENTILE`] (99); the default.
    Percentile(f64),
    /// A fixed residual threshold ([`QSMART_FIT_THRESHOLD`] = 40 in `Demo_QSMART.m`).
    Absolute(f64),
}

impl Default for QsmartFitThreshold {
    fn default() -> Self {
        QsmartFitThreshold::Percentile(QSMART_FIT_PERCENTILE)
    }
}

impl QsmartFitThreshold {
    /// Checks the value: a percentile in (0, 100], or a positive, finite threshold.
    pub fn validate(&self) -> Result<(), String> {
        match *self {
            QsmartFitThreshold::Percentile(p) if !(p > 0.0 && p <= 100.0) => {
                Err(format!("QSMART's echo-fit percentile must be in (0, 100], got {p}"))
            }
            QsmartFitThreshold::Absolute(t) if !(t.is_finite() && t > 0.0) => {
                Err(format!("QSMART's echo-fit threshold must be positive, got {t}"))
            }
            _ => Ok(()),
        }
    }
}

/// MATLAB's `prctile(x, p)` (its default method): the sorted values sit at percentiles
/// `100 * (i - 0.5) / n`, linearly interpolated between and clamped beyond the first and last.
/// NaNs are left out. `None` when nothing is left.
fn prctile_matlab(values: impl Iterator<Item = f64>, p: f64) -> Option<f64> {
    let mut x: Vec<f64> = values.filter(|v| !v.is_nan()).collect();
    if x.is_empty() {
        return None;
    }
    x.sort_by(f64::total_cmp);
    let n = x.len() as f64;
    // Position on the 1-based index scale: p = 100 (i - 0.5) / n  =>  i = n p / 100 + 0.5
    let i = n * p / 100.0 + 0.5;
    if i <= 1.0 {
        return Some(x[0]);
    }
    if i >= n {
        return Some(x[x.len() - 1]);
    }
    let lo = i.floor();
    let f = i - lo;
    let k = lo as usize - 1;
    Some(x[k] + f * (x[k + 1] - x[k]))
}

/// QSMART's echo fit ([`qsmart_echofit`]): the total field and the reliability map `R_0`.
pub struct QsmartEchofit {
    /// Total field in ppm: the magnitude-weighted, zero-intercept fit of phase to echo time
    pub field_ppm: Vec<f64>,
    /// The fit's magnitude-weighted residual, normalised by the magnitude sum and times the
    /// number of echoes (0 where undefined)
    pub fit_residual: Vec<f64>,
    /// `fit_residual` box-filtered over about 1 mm (`smooth3(res, 'box', round(1./vox)*2+1)`)
    pub fit_residual_blur: Vec<f64>,
    /// The residual threshold used (the percentile's value when adaptive)
    pub fit_threshold: f64,
    /// `R_0`: 1 where `fit_residual_blur < fit_threshold`, else 0
    pub reliability: Vec<u8>,
}

/// QSMART's `echofit.m` (zero intercept): fits the unwrapped phase of each echo to its echo time,
/// weighted by magnitude, and marks voxels whose blurred fit residual reaches the threshold as
/// unreliable (`R_0` = 0). The threshold is `fit_threshold`: by default the 99th percentile of the
/// blurred residual ([`QsmartFitThreshold::Percentile`], as the authors run it), or a fixed value.
///
/// `phases` are unwrapped phases in radians, `magnitudes` the matching magnitudes, one volume per
/// echo; `echo_times` in seconds; `ppm` converts rad/s to ppm (QSMART: `gyro * B0 / 1e6` with
/// `gyro = 2.675e8`).
pub fn qsmart_echofit(
    phases: &[&[f64]],
    magnitudes: &[&[f64]],
    echo_times: &[f64],
    grid: &crate::Grid,
    ppm: f64,
    fit_threshold: QsmartFitThreshold,
) -> QsmartEchofit {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    let ne = echo_times.len();
    assert!(ne > 0 && phases.len() == ne && magnitudes.len() == ne, "qsmart_echofit: one phase and magnitude per echo time");
    assert!(phases.iter().chain(magnitudes).all(|v| v.len() == n), "qsmart_echofit: volumes must match the grid");
    let mut field_ppm = vec![0.0; n];
    let mut fit_residual = vec![0.0; n];
    for v in 0..n {
        // tfs = sum(mag.*ph.*TE) ./ (sum(mag.*TE.^2) + eps)
        let (mut num, mut den, mut wsum) = (0.0, 0.0, 0.0);
        for e in 0..ne {
            let (m, te) = (magnitudes[e][v], echo_times[e]);
            num += m * phases[e][v] * te;
            den += m * te * te;
            wsum += m;
        }
        let tfs = num / (den + f64::EPSILON);
        let mut r = 0.0;
        for e in 0..ne {
            let d = phases[e][v] - tfs * echo_times[e];
            r += d * magnitudes[e][v] * d;
        }
        let res = r / wsum * ne as f64;
        fit_residual[v] = if res.is_finite() { res } else { 0.0 };
        field_ppm[v] = tfs / ppm;
    }
    let (vx, vy, vz) = grid.voxel_size;
    let half = |vs: f64| (1.0 / vs).round() as usize;
    let fit_residual_blur = box_mean_replicate(&fit_residual, grid.dims, [half(vx), half(vy), half(vz)]);
    let fit_threshold = match fit_threshold {
        QsmartFitThreshold::Absolute(t) => t,
        // With no nonzero residual there is nothing to exclude.
        QsmartFitThreshold::Percentile(p) => {
            prctile_matlab(fit_residual_blur.iter().copied().filter(|&b| b != 0.0), p).unwrap_or(f64::INFINITY)
        }
    };
    let reliability = fit_residual_blur.iter().map(|&b| u8::from(b < fit_threshold || b.is_nan())).collect();
    QsmartEchofit { field_ppm, fit_residual, fit_residual_blur, fit_threshold, reliability }
}

/// MATLAB's `smooth3(data, 'box', 2*half+1)`: the mean over a box, replicate-padded.
fn box_mean_replicate(data: &[f64], dims: (usize, usize, usize), half: [usize; 3]) -> Vec<f64> {
    let (nx, ny, nz) = dims;
    let mut out = data.to_vec();
    for (axis, &h) in half.iter().enumerate() {
        if h == 0 {
            continue;
        }
        let (len, stride) = [(nx, 1), (ny, nx), (nz, nx * ny)][axis];
        let src = out.clone();
        let w = (2 * h + 1) as f64;
        for i in 0..data.len() {
            let c = (i / stride) % len;
            let base = i - c * stride;
            let mut acc = 0.0;
            for k in 0..=2 * h {
                let p = (c + k).saturating_sub(h).min(len - 1);
                acc += src[base + p * stride];
            }
            out[i] = acc / w;
        }
    }
    out
}

/// The magnitude QSMART detects vessels on (`vasculature_mask.m`): the mean over echoes, then
/// N4 bias correction with ANTs' defaults and no mask
/// ([`n4_bias_correction`](crate::utils::n4_bias_correction)), as QSMART runs
/// `N4BiasFieldCorrection -i AvgEcho.nii -o AvgEcho_n4.nii`. Pass the result to
/// [`run_qsmart`](crate::pipeline::run_qsmart) as its magnitude.
///
/// QSMART's vessel threshold depends on the magnitude's intensity scale, so the mean (not the sum
/// or the root sum of squares) and N4 (not another bias correction) matter.
///
/// # Errors
/// If there are no echoes, or an echo does not match the grid.
pub fn qsmart_vessel_magnitude(
    magnitudes: &[&[f64]],
    grid: &crate::Grid,
) -> Result<Vec<f64>, crate::error::InputError> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    if magnitudes.is_empty() {
        return Err(crate::error::InputError::new("QSMART's vessel magnitude needs at least one magnitude echo"));
    }
    if let Some(e) = magnitudes.iter().position(|m| m.len() != n) {
        return Err(crate::error::InputError::new(format!(
            "QSMART's vessel magnitude: echo {} has {} voxels but the grid has {n}",
            e + 1,
            magnitudes[e].len()
        )));
    }
    let ne = magnitudes.len() as f64;
    let mean: Vec<f64> = (0..n).map(|i| magnitudes.iter().map(|m| m[i]).sum::<f64>() / ne).collect();
    Ok(crate::utils::n4_bias_correction(&mean, None, grid, &crate::utils::N4Params::default(), |_, _| {}).corrected)
}

/// Complete QSMART pipeline parameters
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct QsmartParams {
    /// PPM conversion factor: (gyro * field) / 1e6
    pub ppm: f64,
    /// SDF parameters for stage 1
    pub sdf_sigma1_stage1: f64,
    pub sdf_sigma2_stage1: f64,
    /// SDF parameters for stage 2
    pub sdf_sigma1_stage2: f64,
    pub sdf_sigma2_stage2: f64,
    /// SDF lower limit for proximity clamping
    pub sdf_lower_lim: f64,
    /// SDF curvature constant
    pub sdf_curv_constant: f64,
    /// Vasculature sphere radius for bottom-hat
    pub vasc_sphere_radius: i32,
    /// Frangi scale range for vessel detection
    pub frangi_scale_range: [f64; 2],
    /// Frangi scale ratio
    pub frangi_scale_ratio: f64,
    /// Frangi C parameter
    pub frangi_c: f64,
    /// iLSQR iteration cap for both LSQR solves: STI's `'niter'`, which QSMART passes as 50
    pub ilsqr_max_iter: usize,
    /// iLSQR zero padding in mm: STI's `'padsize'`, which QSMART leaves at STI's default
    /// ([`crate::inversion::STI_DEFAULT_PAD_MM`])
    pub ilsqr_pad_mm: f64,
    /// Dipole inversion algorithm used for both QSMART stages
    pub inversion: InversionAlgorithm,
    /// B0 field direction
    pub b0_dir: (f64, f64, f64),
}

impl Default for QsmartParams {
    fn default() -> Self {
        // Default values for 7T human brain
        Self {
            ppm: 2.675e8 * 7.0 / 1e6, // gyro * field / 1e6
            sdf_sigma1_stage1: 10.0,
            sdf_sigma2_stage1: 10.0,
            sdf_sigma1_stage2: 8.0,
            sdf_sigma2_stage2: 2.0,
            sdf_lower_lim: 0.6,
            sdf_curv_constant: 500.0,
            vasc_sphere_radius: 8,
            // QSMART Demo defaults: FrangiScaleRange=[0.5,6], FrangiScaleRatio=0.5
            frangi_scale_range: [0.5, 6.0],
            frangi_scale_ratio: 0.5,
            frangi_c: 500.0,
            ilsqr_max_iter: 50,
            ilsqr_pad_mm: crate::inversion::STI_DEFAULT_PAD_MM,
            inversion: InversionAlgorithm::Ilsqr,
            b0_dir: (0.0, 0.0, 1.0),
        }
    }
}

impl QsmartParams {
    /// Create parameters for specific field strength
    pub fn for_field_strength(field_tesla: f64) -> Self {
        Self {
            ppm: QSMART_GYRO * field_tesla / 1e6,
            ..Default::default()
        }
    }
}

/// Result of QSMART pipeline
pub struct QsmartResult {
    /// Final combined and offset-adjusted susceptibility map
    pub chi_qsmart: Vec<f64>,
    /// Stage 1 susceptibility (whole ROI)
    pub chi_stage1: Vec<f64>,
    /// Stage 2 susceptibility (tissue only)
    pub chi_stage2: Vec<f64>,
    /// Local field from stage 1
    pub lfs_stage1: Vec<f64>,
    /// Local field from stage 2
    pub lfs_stage2: Vec<f64>,
    /// Vasculature mask (1 = tissue, 0 = vessel)
    pub vasc_mask: Vec<f64>,
}

/// Compute removed voxels mask for offset adjustment
///
/// removed_voxels = (mask * R_0) - vasc_only
/// This represents voxels processed in stage 1 but not in stage 2
pub fn compute_removed_voxels(
    mask: &[f64],
    r_0: &[f64],
    vasc_only: &[f64],
) -> Vec<f64> {
    mask.iter()
        .zip(r_0.iter())
        .zip(vasc_only.iter())
        .map(|((&m, &r), &v)| m * r - v)
        .collect()
}

/// Compute weighted mask for iLSQR
///
/// For stage 1: mask * R_0
/// For stage 2: mask * vasc_only * R_0
pub fn compute_weighted_mask_stage1(mask: &[f64], r_0: &[f64]) -> Vec<f64> {
    mask.iter().zip(r_0.iter()).map(|(&m, &r)| m * r).collect()
}

pub fn compute_weighted_mask_stage2(mask: &[f64], r_0: &[f64], vasc_only: &[f64]) -> Vec<f64> {
    mask.iter()
        .zip(r_0.iter())
        .zip(vasc_only.iter())
        .map(|((&m, &r), &v)| m * v * r)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vessel_magnitude_is_n4_of_the_echo_mean() {
        let grid = crate::Grid::new(12, 10, 8, 1.0, 1.0, 1.0);
        let n = 12 * 10 * 8;
        let echoes: Vec<Vec<f64>> = (0..3)
            .map(|e| (0..n).map(|i| 100.0 + 40.0 * (i as f64 * 0.11).sin() + 10.0 * e as f64 + (i % 12) as f64).collect())
            .collect();
        let refs: Vec<&[f64]> = echoes.iter().map(|v| v.as_slice()).collect();
        let mean: Vec<f64> = (0..n).map(|i| (echoes[0][i] + echoes[1][i] + echoes[2][i]) / 3.0).collect();
        let want = crate::utils::n4_bias_correction(&mean, None, &grid, &crate::utils::N4Params::default(), |_, _| {}).corrected;
        assert_eq!(qsmart_vessel_magnitude(&refs, &grid).unwrap(), want);
        assert!(qsmart_vessel_magnitude(&[], &grid).is_err());
        assert!(qsmart_vessel_magnitude(&[&echoes[0][..n - 1]], &grid).is_err());
    }

    #[test]
    fn prctile_matches_matlab() {
        // MATLAB R2023b prctile([3 1 4 1 5 9 2 6 5 3 5], p).
        let v = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0];
        for (p, want) in [(99.0, 9.0), (50.0, 4.0), (0.0, 1.0), (100.0, 9.0), (3.0, 1.0), (97.0, 9.0)] {
            assert_eq!(prctile_matlab(v.iter().copied(), p), Some(want), "p = {p}");
        }
        // Interpolated: positions 100 (i - 0.5) / 4 = 12.5, 37.5, 62.5, 87.5.
        assert_eq!(prctile_matlab([10.0, 20.0, 30.0, 40.0].into_iter(), 25.0), Some(15.0));
        assert_eq!(prctile_matlab([f64::NAN, 2.0].into_iter(), 50.0), Some(2.0));
        assert_eq!(prctile_matlab(std::iter::empty(), 50.0), None);
    }

    #[test]
    fn adaptive_threshold_is_the_percentile_of_the_nonzero_residuals() {
        let grid = crate::Grid::new(6, 5, 4, 1.0, 1.0, 1.0);
        let te = [0.004, 0.009, 0.014];
        let n = 120;
        let mags: Vec<Vec<f64>> = (0..3).map(|e| (0..n).map(|i| if i % 13 == 0 { 0.0 } else { 1.0 + ((i + e) % 7) as f64 }).collect()).collect();
        let phases: Vec<Vec<f64>> = (0..3).map(|e| (0..n).map(|i| 30.0 * te[e] + 0.4 * (i as f64 * 1.3 + e as f64).cos()).collect()).collect();
        let p: Vec<&[f64]> = phases.iter().map(|v| v.as_slice()).collect();
        let m: Vec<&[f64]> = mags.iter().map(|v| v.as_slice()).collect();
        let fit = qsmart_echofit(&p, &m, &te, &grid, 1.0, QsmartFitThreshold::Percentile(80.0));
        let want = prctile_matlab(fit.fit_residual_blur.iter().copied().filter(|&b| b != 0.0), 80.0).unwrap();
        assert_eq!(fit.fit_threshold, want);
        let excluded = fit.reliability.iter().filter(|&&r| r == 0).count();
        assert!(excluded > 0 && excluded < n / 2, "{excluded}");
        assert_eq!(QsmartFitThreshold::default(), QsmartFitThreshold::Percentile(99.0));
        assert!(QsmartFitThreshold::Percentile(0.0).validate().is_err());
        assert!(QsmartFitThreshold::Percentile(101.0).validate().is_err());
        assert!(QsmartFitThreshold::Absolute(0.0).validate().is_err());
        assert!(QsmartFitThreshold::Absolute(40.0).validate().is_ok());
    }

    #[test]
    fn echofit_matches_matlab() {
        // QSMART_toolbox_v1.0/echofit.m (MATLAB R2023b) on the same synthetic data: anisotropic
        // voxels (box 3x5x1) and zero-magnitude voxels (residual 0/0 -> 0).
        let grid = crate::Grid::new(6, 5, 4, 1.0, 0.5, 2.0);
        let te = [0.004, 0.009, 0.014];
        let n = 120;
        let mags: Vec<Vec<f64>> = (0..3)
            .map(|e| (0..n).map(|i| if i % 13 == 0 { 0.0 } else { 1.0 + ((i * 7 + e * 3) % 11) as f64 / 5.0 }).collect())
            .collect();
        let phases: Vec<Vec<f64>> = (0..3)
            .map(|e| (0..n).map(|i| 30.0 * te[e] * (1.0 + (i as f64 * 0.7).sin()) + 0.4 * (i as f64 * 1.3 + e as f64 * 2.1).cos()).collect())
            .collect();
        let p: Vec<&[f64]> = phases.iter().map(|v| v.as_slice()).collect();
        let m: Vec<&[f64]> = mags.iter().map(|v| v.as_slice()).collect();
        let fit = qsmart_echofit(&p, &m, &te, &grid, 2.675e8 * 3.0 / 1e6, QsmartFitThreshold::Absolute(0.19491409729604658));
        let close = |a: f64, b: f64| assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0), "{a} vs {b}");
        close(fit.field_ppm.iter().sum(), 4.228895044177502);
        close(fit.fit_residual.iter().sum(), 23.334665367421614);
        close(fit.fit_residual_blur.iter().sum(), 23.280564657566376);
        close(fit.fit_residual_blur.iter().map(|v| v * v).sum(), 4.534541884237994);
        close(fit.field_ppm[7], 0.009255448272120092);
        close(fit.fit_residual[7], 0.19925363423077985);
        close(fit.fit_residual_blur[7], 0.1834123131209351);
        close(fit.fit_residual_blur[0], 0.15544118128559214);
        let r0: String = fit.reliability.iter().map(|&r| char::from(b'0' + r)).collect();
        assert_eq!(r0, "110000110000111100111100111100110011110011110100111100111000001011000011001110001110000010111000111000001010001110001110");
    }
    use crate::Grid;

    #[test]
    fn test_offset_adjustment_basic() {
        // Basic test: should run without panic
        let n = 8;
        let n_total = n * n * n;
        let g = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let removed = vec![1.0f64; n_total];
        let lfs = vec![0.0f64; n_total];
        let chi_1 = vec![0.1f64; n_total];
        let chi_2 = vec![0.1f64; n_total];

        let result = adjust_offset(
            &removed, &lfs, &chi_1, &chi_2,
            &g, (0.0, 0.0, 1.0), 1.0
        );

        assert_eq!(result.len(), n_total);
    }

    #[test]
    fn test_params_for_field_strength() {
        let params_7t = QsmartParams::for_field_strength(7.0);
        let params_3t = QsmartParams::for_field_strength(3.0);

        // 7T should have higher PPM than 3T
        assert!(params_7t.ppm > params_3t.ppm);
    }
}

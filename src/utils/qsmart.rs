//! QSMART (Quantitative Susceptibility Mapping Artifact Reduction Technique)
//!
//! This module provides the complete QSMART pipeline including:
//! - Two-stage QSM reconstruction (whole ROI + tissue only)
//! - Offset adjustment for combining the two stages
//! - Integration of SDF background removal with iLSQR inversion
//!
//! Reference: Syeda et al., QSMART: Quantitative Susceptibility Mapping Artifact Reduction Technique

use crate::fft::{fft3d, ifft3d};
use crate::kernels::dipole::dipole_kernel;
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
/// * `nx`, `ny`, `nz` - Volume dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
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
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    b0_dir: (f64, f64, f64),
    ppm: f64,
) -> Vec<f64> {
    let n_total = nx * ny * nz;

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
    let d_kernel = dipole_kernel(nx, ny, nz, vsx, vsy, vsz, b0_dir);

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

/// Complete QSMART pipeline parameters
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
    /// SDF spatial radius for morphological closing
    pub sdf_spatial_radius: i32,
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
    /// iLSQR tolerance
    pub ilsqr_tol: f64,
    /// iLSQR max iterations
    pub ilsqr_max_iter: usize,
    /// B0 field direction
    pub b0_dir: (f64, f64, f64),
}

impl Default for QsmartParams {
    fn default() -> Self {
        // Default values for 7T human brain
        Self {
            ppm: 2.675e8 * 7.0 / 1e6, // gyro * field / 1e6
            sdf_sigma1_stage1: 10.0,
            sdf_sigma2_stage1: 0.0,
            sdf_sigma1_stage2: 8.0,
            sdf_sigma2_stage2: 2.0,
            sdf_spatial_radius: 8,
            sdf_lower_lim: 0.6,
            sdf_curv_constant: 500.0,
            vasc_sphere_radius: 8,
            // QSMART reference defaults: FrangiScaleRange=[1,10], FrangiScaleRatio=2
            frangi_scale_range: [1.0, 10.0],
            frangi_scale_ratio: 2.0,
            frangi_c: 500.0,
            ilsqr_tol: 0.01,
            ilsqr_max_iter: 50,
            b0_dir: (0.0, 0.0, 1.0),
        }
    }
}

impl QsmartParams {
    /// Create parameters for specific field strength
    pub fn for_field_strength(field_tesla: f64) -> Self {
        let gyro = 2.675e8; // Proton gyromagnetic ratio
        Self {
            ppm: gyro * field_tesla / 1e6,
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
    fn test_offset_adjustment_basic() {
        // Basic test: should run without panic
        let n = 8;
        let n_total = n * n * n;

        let removed = vec![1.0f64; n_total];
        let lfs = vec![0.0f64; n_total];
        let chi_1 = vec![0.1f64; n_total];
        let chi_2 = vec![0.1f64; n_total];

        let result = adjust_offset(
            &removed, &lfs, &chi_1, &chi_2,
            n, n, n, 1.0, 1.0, 1.0,
            (0.0, 0.0, 1.0), 1.0
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

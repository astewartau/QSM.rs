//! SHARP background field removal
//!
//! Sophisticated Harmonic Artifact Reduction for Phase data.
//! Uses the spherical mean value property of harmonic functions to
//! separate local from background fields.
//!
//! Reference:
//! Schweser, F., Deistung, A., Lehr, B.W., Reichenbach, J.R. (2011).
//! "Quantitative imaging of intrinsic magnetic tissue properties using MRI signal phase."
//! NeuroImage, 54(4):2789-2807. https://doi.org/10.1016/j.neuroimage.2010.10.070
//!
//! Reference implementation: https://github.com/kamesy/QSM.jl

use num_complex::Complex64;
use crate::Grid;
use crate::fft::{fft3d, ifft3d, fft_real_kernel};
use crate::kernels::smv::{smv_kernel, erode_mask_smv};

/// SHARP algorithm parameters
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct SharpParams {
    /// Deconvolution threshold
    pub threshold: f64,
    /// SMV kernel radius in mm (default: 6.0)
    pub radius: f64,
}

impl Default for SharpParams {
    fn default() -> Self {
        Self {
            threshold: 0.05,
            radius: 6.0,
        }
    }
}

/// SHARP background field removal
///
/// Uses spherical mean value (SMV) filtering to remove background field.
/// The local field is obtained by deconvolving the SMV-filtered field.
///
/// # Arguments
/// * `field` - Unwrapped total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `grid` - Volume dimensions and voxel sizes
/// * `params` - SHARP parameters (threshold, radius in mm)
///
/// # Returns
/// (local_field, eroded_mask)
pub fn sharp(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &SharpParams,
) -> (Vec<f64>, Vec<u8>) {
    sharp_core(field, mask, grid, params.threshold, params.radius)
}

/// SHARP with an explicit absolute kernel radius (mm).
///
/// Internal entry point shared with V-SHARP; the public [`sharp`] wrapper
/// derives the radius from [`SharpParams`].
pub(crate) fn sharp_core(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    threshold: f64,
    radius: f64,
) -> (Vec<f64>, Vec<u8>) {
    let (nx, ny, nz) = grid.dims;
    let n_total = nx * ny * nz;

    // Generate SMV kernel and FFT it
    let s_kernel = smv_kernel(grid, radius);
    let s_fft = fft_real_kernel(&s_kernel, nx, ny, nz);

    // Erode mask via SMV convolution
    let eroded_mask = erode_mask_smv(mask, &s_fft, grid, 1.0 - 1e-7_f64.sqrt());

    // Apply SHARP:
    // 1. Multiply field by (1-S) in k-space (high-pass filter)
    // 2. Apply eroded mask
    // 3. Divide by (1-S) with threshold (deconvolution)
    // 4. Apply eroded mask

    // FFT of field
    let mut field_complex: Vec<Complex64> = field.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut field_complex, nx, ny, nz);

    // High-pass filter: multiply by (1-S)
    for i in 0..n_total {
        field_complex[i] *= 1.0 - s_fft[i];
    }

    // IFFT
    ifft3d(&mut field_complex, nx, ny, nz);

    // Apply eroded mask
    for i in 0..n_total {
        if eroded_mask[i] == 0 {
            field_complex[i] = Complex64::new(0.0, 0.0);
        }
    }

    // FFT again for deconvolution
    fft3d(&mut field_complex, nx, ny, nz);

    // Deconvolution: divide by (1-S) with threshold
    for i in 0..n_total {
        let one_minus_s = 1.0 - s_fft[i];
        if one_minus_s.abs() < threshold {
            field_complex[i] = Complex64::new(0.0, 0.0);
        } else {
            field_complex[i] /= one_minus_s;
        }
    }

    // Final IFFT
    ifft3d(&mut field_complex, nx, ny, nz);

    // Apply eroded mask and extract real part
    let local_field: Vec<f64> = field_complex.iter()
        .enumerate()
        .map(|(i, c)| if eroded_mask[i] == 1 { c.re } else { 0.0 })
        .collect();

    (local_field, eroded_mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharp_zero_field() {
        // Zero field should give zero local field
        let n = 16;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        // Use small radius for small test array
        let (local, _) = sharp(&field, &mask, &grid, &SharpParams { threshold: 0.05, radius: 2.0 });

        for &val in local.iter() {
            assert!(val.abs() < 1e-8, "Zero field should give zero local field, got {}", val);
        }
    }

    #[test]
    fn test_sharp_finite() {
        // Result should be finite (no NaN or Inf)
        let n = 16;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.01).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let (local, eroded) = sharp(&field, &mask, &grid, &SharpParams { threshold: 0.05, radius: 2.0 });

        for (i, &val) in local.iter().enumerate() {
            assert!(val.is_finite(), "Local field should be finite at index {}", i);
        }

        // Eroded mask should have at least some voxels
        let eroded_count: usize = eroded.iter().map(|&m| m as usize).sum();
        assert!(eroded_count > 0, "Eroded mask should have some voxels");
    }
}

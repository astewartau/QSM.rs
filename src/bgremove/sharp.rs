//! SHARP background field removal
//!
//! Sophisticated Harmonic Artifact Reduction for Phase data.
//! Uses the spherical mean value property of harmonic functions to
//! separate local from background fields.
//!
//! Reference:
//! Schweser F, et al. Quantitative imaging of intrinsic magnetic tissue
//! properties using MRI signal phase. Neuroimage. 2011;54(4):2789-807.

use num_complex::Complex64;
use crate::fft::{fft3d, ifft3d};
use crate::kernels::smv::smv_kernel;

/// SHARP background field removal
///
/// Uses spherical mean value (SMV) filtering to remove background field.
/// The local field is obtained by deconvolving the SMV-filtered field.
///
/// # Arguments
/// * `field` - Unwrapped total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `radius` - SMV kernel radius in mm
/// * `threshold` - High-pass filter threshold (typically 0.05)
///
/// # Returns
/// (local_field, eroded_mask)
pub fn sharp(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64,
    threshold: f64,
) -> (Vec<f64>, Vec<u8>) {
    let n_total = nx * ny * nz;

    // Generate SMV kernel and FFT it
    let s_kernel = smv_kernel(nx, ny, nz, vsx, vsy, vsz, radius);

    // FFT of SMV kernel
    let mut s_complex: Vec<Complex64> = s_kernel.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut s_complex, nx, ny, nz);

    // S is the real part of FFT(smv_kernel)
    // 1-S is the high-pass kernel
    let s_fft: Vec<f64> = s_complex.iter().map(|c| c.re).collect();

    // Erode mask: convolve mask with SMV kernel
    // Voxels where convolution result < 1 are near boundary
    let mask_f64: Vec<f64> = mask.iter().map(|&m| m as f64).collect();
    let mut mask_complex: Vec<Complex64> = mask_f64.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();

    fft3d(&mut mask_complex, nx, ny, nz);

    // Convolve mask with SMV kernel
    for i in 0..n_total {
        mask_complex[i] *= s_complex[i].re;  // S is real
    }

    ifft3d(&mut mask_complex, nx, ny, nz);

    // Eroded mask: values close to 1 are fully inside
    let delta = 1.0 - 1e-7_f64.sqrt();  // ~1 - eps
    let eroded_mask: Vec<u8> = mask_complex.iter()
        .map(|c| if c.re > delta { 1 } else { 0 })
        .collect();

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

/// SHARP with default parameters
pub fn sharp_default(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> (Vec<f64>, Vec<u8>) {
    // Default radius: 18 * minimum voxel size
    let radius = 18.0 * vsx.min(vsy).min(vsz);
    sharp(field, mask, nx, ny, nz, vsx, vsy, vsz, radius, 0.05)
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

        // Use small radius for small test array
        let (local, _) = sharp(&field, &mask, n, n, n, 1.0, 1.0, 1.0, 2.0, 0.05);

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

        let (local, eroded) = sharp(&field, &mask, n, n, n, 1.0, 1.0, 1.0, 2.0, 0.05);

        for (i, &val) in local.iter().enumerate() {
            assert!(val.is_finite(), "Local field should be finite at index {}", i);
        }

        // Eroded mask should have at least some voxels
        let eroded_count: usize = eroded.iter().map(|&m| m as usize).sum();
        assert!(eroded_count > 0, "Eroded mask should have some voxels");
    }
}

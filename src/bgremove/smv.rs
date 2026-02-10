//! Simple Spherical Mean Value (SMV) background field removal
//!
//! Basic SMV filtering: subtracts the spherical mean of the field.
//! Simpler than SHARP (no deconvolution step).
//!
//! local_field = field - SMV(field)
//!
//! Reference:
//! Schweser F, et al. Quantitative imaging of intrinsic magnetic tissue
//! properties using MRI signal phase. Neuroimage. 2011;54(4):2789-807.

use num_complex::Complex64;
use crate::fft::{fft3d, ifft3d};
use crate::kernels::smv::smv_kernel;

/// Simple SMV background field removal
///
/// Computes: local_field = field - SMV(field)
///
/// # Arguments
/// * `field` - Unwrapped total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `radius` - SMV kernel radius in mm
///
/// # Returns
/// (local_field, eroded_mask)
pub fn smv(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64,
) -> (Vec<f64>, Vec<u8>) {
    let n_total = nx * ny * nz;

    // Generate SMV kernel and FFT it
    let s_kernel = smv_kernel(nx, ny, nz, vsx, vsy, vsz, radius);

    // FFT of SMV kernel
    let mut s_complex: Vec<Complex64> = s_kernel.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut s_complex, nx, ny, nz);
    let s_fft = s_complex;

    // Erode mask: convolve mask with SMV kernel
    let mask_f64: Vec<f64> = mask.iter().map(|&m| m as f64).collect();
    let mut mask_complex: Vec<Complex64> = mask_f64.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();

    fft3d(&mut mask_complex, nx, ny, nz);

    // Convolve mask with SMV kernel
    for i in 0..n_total {
        mask_complex[i] *= s_fft[i].re;
    }

    ifft3d(&mut mask_complex, nx, ny, nz);

    // Eroded mask: values close to 1 are fully inside
    let delta = 1.0 - 1e-10;
    let eroded_mask: Vec<u8> = mask_complex.iter()
        .map(|c| if c.re > delta { 1 } else { 0 })
        .collect();

    // Compute SMV(field) = background field estimate
    let mut field_complex: Vec<Complex64> = field.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut field_complex, nx, ny, nz);

    // Multiply by SMV kernel in k-space
    for i in 0..n_total {
        field_complex[i] *= s_fft[i].re;
    }

    ifft3d(&mut field_complex, nx, ny, nz);

    // Local field = field - SMV(field), within eroded mask
    let local_field: Vec<f64> = field.iter()
        .zip(field_complex.iter())
        .enumerate()
        .map(|(i, (&f, smv_f))| {
            if eroded_mask[i] == 1 {
                f - smv_f.re
            } else {
                0.0
            }
        })
        .collect();

    (local_field, eroded_mask)
}

/// Simple SMV with default parameters
pub fn smv_default(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> (Vec<f64>, Vec<u8>) {
    // Default radius: 5mm (typical for brain imaging)
    let radius = 5.0;
    smv(field, mask, nx, ny, nz, vsx, vsy, vsz, radius)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smv_zero_field() {
        let n = 16;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        let (local, _) = smv(&field, &mask, n, n, n, 1.0, 1.0, 1.0, 2.0);

        for &val in local.iter() {
            assert!(val.abs() < 1e-8, "Zero field should give zero local field, got {}", val);
        }
    }

    #[test]
    fn test_smv_finite() {
        let n = 16;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.01).collect();
        let mask = vec![1u8; n * n * n];

        let (local, eroded) = smv(&field, &mask, n, n, n, 1.0, 1.0, 1.0, 2.0);

        for (i, &val) in local.iter().enumerate() {
            assert!(val.is_finite(), "Local field should be finite at index {}", i);
        }

        let eroded_count: usize = eroded.iter().map(|&m| m as usize).sum();
        assert!(eroded_count > 0, "Eroded mask should have some voxels");
    }
}

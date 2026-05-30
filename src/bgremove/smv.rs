//! Simple Spherical Mean Value (SMV) background field removal
//!
//! Basic SMV filtering: subtracts the spherical mean of the field.
//! Simpler than SHARP (no deconvolution step).
//!
//! local_field = field - SMV(field)
//!
//! Reference:
//! Schweser, F., Deistung, A., Lehr, B.W., Reichenbach, J.R. (2011).
//! "Quantitative imaging of intrinsic magnetic tissue properties using MRI signal phase."
//! NeuroImage, 54(4):2789-2807. https://doi.org/10.1016/j.neuroimage.2010.10.070
//!
//! Reference implementation: https://github.com/kamesy/QSM.jl

use crate::fft::fft_real_kernel;
use crate::kernels::smv::{smv_kernel, erode_mask_smv};

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
    // Generate SMV kernel and FFT it
    let s_kernel = smv_kernel(nx, ny, nz, vsx, vsy, vsz, radius);
    let s_fft = fft_real_kernel(&s_kernel, nx, ny, nz);

    // Erode mask via SMV convolution
    let eroded_mask = erode_mask_smv(mask, &s_fft, nx, ny, nz, 1.0 - 1e-10);

    // Compute SMV(field) = background field estimate
    let smv_field = crate::fft::apply_real_kernel(field, &s_fft, nx, ny, nz);

    // Local field = field - SMV(field), within eroded mask
    let local_field: Vec<f64> = field.iter()
        .zip(smv_field.iter())
        .enumerate()
        .map(|(i, (&f, &smv_f))| {
            if eroded_mask[i] == 1 { f - smv_f } else { 0.0 }
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

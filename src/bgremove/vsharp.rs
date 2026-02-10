//! V-SHARP background field removal
//!
//! Variable kernel SHARP uses multiple SMV kernel radii
//! to preserve more brain tissue at edges while still
//! removing background fields.
//!
//! Reference:
//! Wu B, et al. Whole brain susceptibility mapping using compressed sensing.
//! Magnetic Resonance in Medicine. 2012;67(1):137-47.

use num_complex::Complex64;
use crate::fft::{fft3d, ifft3d};
use crate::kernels::smv::smv_kernel;

/// V-SHARP background field removal
///
/// Uses multiple SMV kernel radii, starting from largest and decreasing.
/// At each voxel, uses the smallest radius that doesn't touch the boundary.
///
/// # Arguments
/// * `field` - Unwrapped total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `radii` - SMV kernel radii in mm (should be sorted large to small)
/// * `threshold` - High-pass filter threshold (typically 0.05)
///
/// # Returns
/// (local_field, eroded_mask)
pub fn vsharp(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radii: &[f64],
    threshold: f64,
) -> (Vec<f64>, Vec<u8>) {
    if radii.is_empty() {
        return (vec![0.0; nx * ny * nz], mask.to_vec());
    }

    // If only one radius, use regular SHARP
    if radii.len() == 1 {
        return crate::bgremove::sharp::sharp(
            field, mask, nx, ny, nz, vsx, vsy, vsz, radii[0], threshold
        );
    }

    let n_total = nx * ny * nz;

    // Sort radii from largest to smallest
    let mut sorted_radii = radii.to_vec();
    sorted_radii.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // FFT of field
    let mut field_complex: Vec<Complex64> = field.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut field_complex, nx, ny, nz);
    let field_fft = field_complex.clone();

    // Track which voxels have been processed and final mask
    let mut processed = vec![false; n_total];
    let mut local_field = vec![0.0; n_total];
    let mut final_mask = vec![0u8; n_total];

    // Process each radius (large to small)
    let delta = 1.0 - 1e-7_f64.sqrt();

    // Store the first (largest) kernel's inverse for deconvolution
    let mut inverse_kernel: Option<Vec<f64>> = None;

    for &radius in &sorted_radii {
        // Generate SMV kernel
        let s_kernel = smv_kernel(nx, ny, nz, vsx, vsy, vsz, radius);

        // FFT of SMV kernel
        let mut s_complex: Vec<Complex64> = s_kernel.iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect();
        fft3d(&mut s_complex, nx, ny, nz);
        let s_fft: Vec<f64> = s_complex.iter().map(|c| c.re).collect();

        // Store inverse of first (largest) kernel
        if inverse_kernel.is_none() {
            inverse_kernel = Some(s_fft.iter().map(|&s| {
                let one_minus_s = 1.0 - s;
                if one_minus_s.abs() < threshold {
                    0.0
                } else {
                    1.0 / one_minus_s
                }
            }).collect());
        }

        // Erode mask for this radius
        let mask_f64: Vec<f64> = mask.iter().map(|&m| m as f64).collect();
        let mut mask_complex: Vec<Complex64> = mask_f64.iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect();

        fft3d(&mut mask_complex, nx, ny, nz);

        // Convolve mask with SMV kernel
        for i in 0..n_total {
            mask_complex[i] *= s_fft[i];
        }

        ifft3d(&mut mask_complex, nx, ny, nz);

        // Current eroded mask
        let current_mask: Vec<bool> = mask_complex.iter()
            .map(|c| c.re > delta)
            .collect();

        // Apply high-pass filter: multiply by (1-S)
        let mut filtered = field_fft.clone();
        for i in 0..n_total {
            filtered[i] *= 1.0 - s_fft[i];
        }

        ifft3d(&mut filtered, nx, ny, nz);

        // For voxels that are in current mask but not yet processed,
        // store the filtered value
        for i in 0..n_total {
            if current_mask[i] && !processed[i] {
                local_field[i] = filtered[i].re;
                processed[i] = true;
                final_mask[i] = 1;
            }
        }
    }

    // Deconvolution with largest kernel's inverse
    if let Some(inv_kernel) = inverse_kernel {
        let mut local_complex: Vec<Complex64> = local_field.iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect();

        fft3d(&mut local_complex, nx, ny, nz);

        for i in 0..n_total {
            local_complex[i] *= inv_kernel[i];
        }

        ifft3d(&mut local_complex, nx, ny, nz);

        // Apply final mask
        for i in 0..n_total {
            local_field[i] = if final_mask[i] == 1 { local_complex[i].re } else { 0.0 };
        }
    }

    (local_field, final_mask)
}

/// V-SHARP with progress callback
///
/// Same as `vsharp` but calls `progress_callback(radius_index, total_radii)` for each radius.
pub fn vsharp_with_progress<F>(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radii: &[f64],
    threshold: f64,
    mut progress_callback: F,
) -> (Vec<f64>, Vec<u8>)
where
    F: FnMut(usize, usize),
{
    if radii.is_empty() {
        return (vec![0.0; nx * ny * nz], mask.to_vec());
    }

    // If only one radius, use regular SHARP
    if radii.len() == 1 {
        progress_callback(1, 1);
        return crate::bgremove::sharp::sharp(
            field, mask, nx, ny, nz, vsx, vsy, vsz, radii[0], threshold
        );
    }

    let n_total = nx * ny * nz;
    let n_radii = radii.len();

    // Sort radii from largest to smallest
    let mut sorted_radii = radii.to_vec();
    sorted_radii.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // FFT of field
    let mut field_complex: Vec<Complex64> = field.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut field_complex, nx, ny, nz);
    let field_fft = field_complex.clone();

    // Track which voxels have been processed and final mask
    let mut processed = vec![false; n_total];
    let mut local_field = vec![0.0; n_total];
    let mut final_mask = vec![0u8; n_total];

    let delta = 1.0 - 1e-7_f64.sqrt();
    let mut inverse_kernel: Option<Vec<f64>> = None;

    for (idx, &radius) in sorted_radii.iter().enumerate() {
        // Report progress
        progress_callback(idx + 1, n_radii);

        // Generate SMV kernel
        let s_kernel = smv_kernel(nx, ny, nz, vsx, vsy, vsz, radius);

        // FFT of SMV kernel
        let mut s_complex: Vec<Complex64> = s_kernel.iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect();
        fft3d(&mut s_complex, nx, ny, nz);
        let s_fft: Vec<f64> = s_complex.iter().map(|c| c.re).collect();

        // Store inverse of first (largest) kernel
        if inverse_kernel.is_none() {
            inverse_kernel = Some(s_fft.iter().map(|&s| {
                let one_minus_s = 1.0 - s;
                if one_minus_s.abs() < threshold {
                    0.0
                } else {
                    1.0 / one_minus_s
                }
            }).collect());
        }

        // Erode mask for this radius
        let mask_f64: Vec<f64> = mask.iter().map(|&m| m as f64).collect();
        let mut mask_complex: Vec<Complex64> = mask_f64.iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect();

        fft3d(&mut mask_complex, nx, ny, nz);

        for i in 0..n_total {
            mask_complex[i] *= s_fft[i];
        }

        ifft3d(&mut mask_complex, nx, ny, nz);

        let current_mask: Vec<bool> = mask_complex.iter()
            .map(|c| c.re > delta)
            .collect();

        // Apply high-pass filter
        let mut filtered = field_fft.clone();
        for i in 0..n_total {
            filtered[i] *= 1.0 - s_fft[i];
        }

        ifft3d(&mut filtered, nx, ny, nz);

        for i in 0..n_total {
            if current_mask[i] && !processed[i] {
                local_field[i] = filtered[i].re;
                processed[i] = true;
                final_mask[i] = 1;
            }
        }
    }

    // Deconvolution
    if let Some(inv_kernel) = inverse_kernel {
        let mut local_complex: Vec<Complex64> = local_field.iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect();

        fft3d(&mut local_complex, nx, ny, nz);

        for i in 0..n_total {
            local_complex[i] *= inv_kernel[i];
        }

        ifft3d(&mut local_complex, nx, ny, nz);

        for i in 0..n_total {
            local_field[i] = if final_mask[i] == 1 { local_complex[i].re } else { 0.0 };
        }
    }

    (local_field, final_mask)
}

/// V-SHARP with default parameters
pub fn vsharp_default(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> (Vec<f64>, Vec<u8>) {
    let min_vox = vsx.min(vsy).min(vsz);
    let max_vox = vsx.max(vsy).max(vsz);

    // Default radii: 18*min_vox down to 2*max_vox in steps of 2*max_vox
    let mut radii = Vec::new();
    let mut r = 18.0 * min_vox;
    while r >= 2.0 * max_vox {
        radii.push(r);
        r -= 2.0 * max_vox;
    }

    if radii.is_empty() {
        radii.push(18.0 * min_vox);
    }

    vsharp(field, mask, nx, ny, nz, vsx, vsy, vsz, &radii, 0.05)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vsharp_zero_field() {
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        let radii = vec![4.0, 3.0, 2.0];
        let (local, _) = vsharp(&field, &mask, n, n, n, 1.0, 1.0, 1.0, &radii, 0.05);

        for &val in local.iter() {
            assert!(val.abs() < 1e-10);
        }
    }

    #[test]
    fn test_vsharp_preserves_more_than_sharp() {
        let n = 16;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        // V-SHARP with multiple radii
        let radii = vec![5.0, 4.0, 3.0, 2.0];
        let (_, vsharp_mask) = vsharp(&field, &mask, n, n, n, 1.0, 1.0, 1.0, &radii, 0.05);

        // SHARP with single large radius
        let (_, sharp_mask) = crate::bgremove::sharp::sharp(
            &field, &mask, n, n, n, 1.0, 1.0, 1.0, 5.0, 0.05
        );

        let vsharp_count: usize = vsharp_mask.iter().map(|&m| m as usize).sum();
        let sharp_count: usize = sharp_mask.iter().map(|&m| m as usize).sum();

        // V-SHARP should preserve at least as many voxels as SHARP
        assert!(vsharp_count >= sharp_count,
            "V-SHARP {} should preserve at least as many as SHARP {}",
            vsharp_count, sharp_count);
    }
}

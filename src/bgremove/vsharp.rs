//! V-SHARP background field removal
//!
//! Variable kernel SHARP uses multiple SMV kernel radii
//! to preserve more brain tissue at edges while still
//! removing background fields.
//!
//! Each voxel takes the SMV high-pass `φ − S_r∗φ` of the largest radius `r`
//! whose ball fits inside the mask. By default that high-passed field is then
//! deconvolved by the largest kernel, `1/(1 − Ŝ)` with TSVD at `threshold`
//! (Wu et al. 2012; QSM.jl). With [`VsharpParams::deconvolve`] off the
//! high-passed field is returned as is, which is what STI Suite's `V_SHARP`
//! does. The deconvolution restores low spatial frequencies; on real data those
//! are dominated by residual background and field-map error, so turning it off
//! gives a more reproducible local field at the cost of attenuating genuinely
//! low-frequency tissue contrast.
//!
//! Reference:
//! Wu, B., Li, W., Guidon, A., Liu, C. (2012).
//! "Whole brain susceptibility mapping using compressed sensing."
//! Magnetic Resonance in Medicine, 67(1):137-147. https://doi.org/10.1002/mrm.23000
//!
//! Reference implementation: https://github.com/kamesy/QSM.jl

use num_complex::Complex64;
use crate::Grid;
use crate::fft::{fft3d, ifft3d};
use crate::kernels::smv::{smv_kernel, erode_mask_smv};

/// V-SHARP algorithm parameters
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct VsharpParams {
    /// Deconvolution (TSVD) threshold: k-space frequencies where |1 - S| < threshold are dropped
    /// rather than divided through. Higher = more regularisation, more robust to a noisy input field
    /// (default: 0.2). Only used when [`deconvolve`](Self::deconvolve) is true.
    pub threshold: f64,
    /// Maximum (starting) SMV kernel radius in mm (default: 12.0)
    pub max_radius: f64,
    /// Minimum SMV kernel radius in mm — also the step between successive radii (default: 1.0)
    pub min_radius: f64,
    /// Deconvolve the SMV high-passed field by the largest kernel, `1/(1 − Ŝ)` with TSVD at
    /// [`threshold`](Self::threshold) (default: true, Wu et al. 2012 / QSM.jl). When false,
    /// return the high-passed field `φ − S_r∗φ` itself, each voxel with the largest radius
    /// whose ball fits in the mask (as STI Suite's `V_SHARP` does). The deconvolution restores
    /// low spatial frequencies that `1 − Ŝ` attenuated; on real data those are dominated by
    /// residual background and field-map error, so `false` is the more reproducible choice
    /// there, while `true` recovers more low-frequency tissue contrast on a clean field.
    pub deconvolve: bool,
}

impl Default for VsharpParams {
    fn default() -> Self {
        Self {
            // 0.2 (was 0.05): the deconvolution amplifies structured error in a noisy estimated field,
            // and real pipelines always feed V-SHARP an estimated (not ground-truth) field. A stronger
            // regularisation is robust to that; 0.05 under-regularises and only wins on a clean field
            // (which BFR is never run on in practice). See the PR for the benchmark.
            threshold: 0.2,
            max_radius: 12.0,
            min_radius: 1.0,
            deconvolve: true,
        }
    }
}

/// V-SHARP background field removal
///
/// Uses multiple SMV kernel radii, starting from largest and decreasing.
/// At each voxel, uses the largest radius whose ball fits inside the mask,
/// then (if `params.deconvolve`) deconvolves by the largest kernel.
///
/// # Arguments
/// * `field` - Unwrapped total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `grid` - Volume dimensions and voxel sizes
/// * `params` - V-SHARP parameters (min/max radius in mm, deconvolution on/off and its threshold)
/// * `progress` - Progress callback (radius_index, total_radii)
///
/// # Returns
/// (local_field, eroded_mask)
pub fn vsharp(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &VsharpParams,
    progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<u8>) {
    let radii = vsharp_radii(params);
    let deconvolve = params.deconvolve.then_some(params.threshold);
    vsharp_with_radii(field, mask, grid, &radii, deconvolve, progress)
}

/// Build the descending list of SMV kernel radii (mm) from the params.
///
/// Starts at `max_radius` (mm) and steps down by `min_radius` (mm), which is
/// also the smallest kernel used.
fn vsharp_radii(params: &VsharpParams) -> Vec<f64> {
    let mut radii = Vec::new();
    let mut r = params.max_radius;
    let step = params.min_radius;
    while r >= step {
        radii.push(r);
        r -= step;
    }
    if radii.is_empty() {
        radii.push(params.max_radius);
    }
    radii
}

/// V-SHARP with an explicit list of SMV kernel radii (mm).
///
/// Internal entry point; the public [`vsharp`] wrapper derives the radii
/// from [`VsharpParams`]. `deconvolve` is the TSVD threshold of the final
/// deconvolution, or `None` to return the SMV high-passed field undeconvolved.
pub(crate) fn vsharp_with_radii(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    radii: &[f64],
    deconvolve: Option<f64>,
    mut progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<u8>) {
    let (nx, ny, nz) = grid.dims;

    if radii.is_empty() {
        return (vec![0.0; nx * ny * nz], mask.to_vec());
    }

    // If only one radius and deconvolving, this is regular SHARP
    if let (1, Some(threshold)) = (radii.len(), deconvolve) {
        progress(1, 1);
        return crate::bgremove::sharp::sharp_core(
            field, mask, grid, threshold, radii[0]
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
        progress(idx + 1, n_radii);

        // Generate SMV kernel
        let s_kernel = smv_kernel(grid, radius);

        // FFT of SMV kernel
        let mut s_complex: Vec<Complex64> = s_kernel.iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect();
        fft3d(&mut s_complex, nx, ny, nz);
        let s_fft: Vec<f64> = s_complex.iter().map(|c| c.re).collect();

        // Store inverse of first (largest) kernel
        if let (0, Some(threshold)) = (idx, deconvolve) {
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
        let eroded = erode_mask_smv(mask, &s_fft, grid, delta);
        let current_mask: Vec<bool> = eroded.iter().map(|&m| m == 1).collect();

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

    // Deconvolution (only built when `deconvolve` is set)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vsharp_zero_field() {
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let radii = vec![4.0, 3.0, 2.0];
        let (local, _) = vsharp_with_radii(&field, &mask, &grid, &radii, Some(0.05), |_, _| {});

        for &val in local.iter() {
            assert!(val.abs() < 1e-10);
        }
    }

    #[test]
    fn test_vsharp_preserves_more_than_sharp() {
        let n = 16;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        // V-SHARP with multiple radii
        let radii = vec![5.0, 4.0, 3.0, 2.0];
        let (_, vsharp_mask) = vsharp_with_radii(&field, &mask, &grid, &radii, Some(0.05), |_, _| {});

        // SHARP with single large radius
        let (_, sharp_mask) = crate::bgremove::sharp::sharp_core(
            &field, &mask, &grid, 0.05, 5.0
        );

        let vsharp_count: usize = vsharp_mask.iter().map(|&m| m as usize).sum();
        let sharp_count: usize = sharp_mask.iter().map(|&m| m as usize).sum();

        // V-SHARP should preserve at least as many voxels as SHARP
        assert!(vsharp_count >= sharp_count,
            "V-SHARP {} should preserve at least as many as SHARP {}",
            vsharp_count, sharp_count);
    }

    #[test]
    fn test_vsharp_nonuniform_voxels() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 0.5, 1.0, 2.0);

        // Anisotropic voxel sizes
        let radii = vec![4.0, 3.0, 2.0];
        let (local, final_mask) = vsharp_with_radii(
            &field, &mask, &grid, &radii, Some(0.05), |_, _| {}
        );

        // All values should be finite
        for (i, &val) in local.iter().enumerate() {
            assert!(val.is_finite(), "V-SHARP nonuniform voxels: finite at index {}", i);
        }

        // Final mask should have some voxels
        let mask_count: usize = final_mask.iter().map(|&m| m as usize).sum();
        assert!(mask_count > 0, "V-SHARP nonuniform: final mask should have some voxels");
    }

    #[test]
    fn test_vsharp_single_radius() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        // Single radius should delegate to SHARP
        let radii = vec![3.0];
        let (local, final_mask) = vsharp_with_radii(
            &field, &mask, &grid, &radii, Some(0.05), |_, _| {}
        );

        // All values should be finite
        for (i, &val) in local.iter().enumerate() {
            assert!(val.is_finite(), "V-SHARP single radius: finite at index {}", i);
        }

        // Result should match SHARP with same radius
        let (sharp_local, sharp_mask) = crate::bgremove::sharp::sharp_core(
            &field, &mask, &grid, 0.05, 3.0
        );

        for i in 0..n*n*n {
            assert!(
                (local[i] - sharp_local[i]).abs() < 1e-10,
                "Single-radius V-SHARP should match SHARP at index {}", i
            );
        }

        assert_eq!(final_mask, sharp_mask, "Single-radius V-SHARP mask should match SHARP mask");
    }

    #[test]
    fn test_vsharp_empty_radii() {
        // Empty radii should return zeros and the original mask
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let (local, returned_mask) = vsharp_with_radii(
            &field, &mask, &grid, &[], Some(0.05), |_, _| {}
        );

        for &val in &local {
            assert_eq!(val, 0.0, "Empty radii should return zero local field");
        }
        assert_eq!(returned_mask, mask, "Empty radii should return original mask");
    }

    #[test]
    fn test_vsharp_larger_volume() {
        // 16x16x16 volume with spherical mask
        let n = 16;
        let mut field = vec![0.0; n * n * n];
        // Linear background field in z
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    field[x + y * n + z * n * n] = (z as f64) * 0.1;
                }
            }
        }

        // Spherical mask
        let mut mask = vec![0u8; n * n * n];
        let center = n / 2;
        let radius = n / 3;
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let dx = (x as i32) - (center as i32);
                    let dy = (y as i32) - (center as i32);
                    let dz = (z as i32) - (center as i32);
                    if dx * dx + dy * dy + dz * dz <= (radius * radius) as i32 {
                        mask[x + y * n + z * n * n] = 1;
                    }
                }
            }
        }

        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let radii = vec![6.0, 4.0, 3.0, 2.0];
        let (local, final_mask) = vsharp_with_radii(
            &field, &mask, &grid, &radii, Some(0.05), |_, _| {}
        );

        assert_eq!(local.len(), n * n * n);
        for &val in &local {
            assert!(val.is_finite(), "V-SHARP larger volume values should be finite");
        }

        let mask_count: usize = final_mask.iter().map(|&m| m as usize).sum();
        assert!(mask_count > 0, "V-SHARP larger volume should have voxels in final mask");

        // Voxels outside the final mask should be zero
        for i in 0..n * n * n {
            if final_mask[i] == 0 {
                assert_eq!(local[i], 0.0, "Outside final mask should be zero");
            }
        }
    }

    #[test]
    fn test_vsharp_with_progress() {
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let radii = vec![4.0, 3.0, 2.0];
        let mut progress_calls = Vec::new();
        let (local, _) = vsharp_with_radii(
            &field, &mask, &grid, &radii, Some(0.05),
            |idx, total| { progress_calls.push((idx, total)); }
        );

        assert_eq!(local.len(), n * n * n);
        assert!(!progress_calls.is_empty(), "Progress should be called");
        for &val in &local {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_vsharp_with_progress_single_radius() {
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let radii = vec![3.0];
        let mut progress_calls = Vec::new();
        let (local, _) = vsharp_with_radii(
            &field, &mask, &grid, &radii, Some(0.05),
            |idx, total| { progress_calls.push((idx, total)); }
        );

        assert_eq!(local.len(), n * n * n);
        assert!(!progress_calls.is_empty(), "Progress should be called for single radius");
        for &val in &local {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_vsharp_with_progress_empty_radii() {
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let mut progress_calls = Vec::new();
        let (local, returned_mask) = vsharp_with_radii(
            &field, &mask, &grid, &[], Some(0.05),
            |idx, total| { progress_calls.push((idx, total)); }
        );

        for &val in &local {
            assert_eq!(val, 0.0);
        }
        assert_eq!(returned_mask, mask);
    }

    #[test]
    fn test_vsharp_unsorted_radii() {
        // Radii given in arbitrary order - should be sorted internally
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let radii_sorted = vec![4.0, 3.0, 2.0];
        let radii_unsorted = vec![2.0, 4.0, 3.0];

        let (local_sorted, mask_sorted) = vsharp_with_radii(
            &field, &mask, &grid, &radii_sorted, Some(0.05), |_, _| {}
        );
        let (local_unsorted, mask_unsorted) = vsharp_with_radii(
            &field, &mask, &grid, &radii_unsorted, Some(0.05), |_, _| {}
        );

        // Results should be the same regardless of input order
        assert_eq!(mask_sorted, mask_unsorted, "Sorted and unsorted radii should give same mask");
        for i in 0..n * n * n {
            assert!(
                (local_sorted[i] - local_unsorted[i]).abs() < 1e-10,
                "Results should match at index {}",
                i
            );
        }
    }

    /// Spherical mask of radius `rad` voxels centred in an `n`³ volume, and a smooth
    /// non-harmonic field over it.
    fn sphere_case(n: usize, rad: f64) -> (Vec<f64>, Vec<u8>) {
        let c = n as f64 / 2.0;
        let mut field = vec![0.0; n * n * n];
        let mut mask = vec![0u8; n * n * n];
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let (dx, dy, dz) = (x as f64 - c, y as f64 - c, z as f64 - c);
                    let i = x + y * n + z * n * n;
                    field[i] = 0.01 * dz + 0.002 * (dx * dx - dy * dy) + (0.3 * dx).sin() * 0.05;
                    mask[i] = (dx * dx + dy * dy + dz * dz <= rad * rad) as u8;
                }
            }
        }
        (field, mask)
    }

    #[test]
    fn test_vsharp_default_deconvolves() {
        assert!(VsharpParams::default().deconvolve);
    }

    #[test]
    fn test_vsharp_no_deconvolve_is_smv_highpass() {
        // Without deconvolution each voxel is φ − S_r∗φ for the largest radius r whose
        // ball fits the mask; check against that computed directly, radius by radius.
        let n = 20;
        let (field, mask) = sphere_case(n, 8.0);
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let radii = [4.0, 3.0, 2.0];
        let (local, out_mask) = vsharp_with_radii(&field, &mask, &grid, &radii, None, |_, _| {});

        let delta = 1.0 - 1e-7_f64.sqrt();
        let mut expected = vec![0.0; n * n * n];
        let mut expected_mask = vec![0u8; n * n * n];
        for &r in &radii {
            let mut s: Vec<Complex64> = smv_kernel(&grid, r).iter().map(|&x| Complex64::new(x, 0.0)).collect();
            fft3d(&mut s, n, n, n);
            let s_fft: Vec<f64> = s.iter().map(|c| c.re).collect();
            let fits = erode_mask_smv(&mask, &s_fft, &grid, delta);
            let mut sf: Vec<Complex64> = field.iter().map(|&x| Complex64::new(x, 0.0)).collect();
            fft3d(&mut sf, n, n, n);
            for i in 0..sf.len() { sf[i] *= s_fft[i]; }
            ifft3d(&mut sf, n, n, n);
            for i in 0..n * n * n {
                if fits[i] == 1 && expected_mask[i] == 0 {
                    expected[i] = field[i] - sf[i].re;
                    expected_mask[i] = 1;
                }
            }
        }
        assert_eq!(out_mask, expected_mask);
        assert!(out_mask.contains(&1));
        for i in 0..n * n * n {
            assert!((local[i] - expected[i]).abs() < 1e-12, "voxel {i}: {} vs {}", local[i], expected[i]);
        }
    }

    #[test]
    fn test_vsharp_deconvolve_flag() {
        let n = 20;
        let (field, mask) = sphere_case(n, 8.0);
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let base = VsharpParams { max_radius: 4.0, min_radius: 1.0, ..Default::default() };
        let (dec, dec_mask) = vsharp(&field, &mask, &grid, &base, |_, _| {});
        let off = VsharpParams { deconvolve: false, ..base.clone() };
        let (hp, hp_mask) = vsharp(&field, &mask, &grid, &off, |_, _| {});
        // Same voxels either way; the deconvolution changes the values.
        assert_eq!(dec_mask, hp_mask);
        assert!(dec.iter().zip(&hp).any(|(a, b)| (a - b).abs() > 1e-6));
        // The threshold only applies to the deconvolution.
        let (hp2, _) = vsharp(&field, &mask, &grid, &VsharpParams { threshold: 0.9, ..off }, |_, _| {});
        assert_eq!(hp, hp2);
        for i in 0..n * n * n {
            if hp_mask[i] == 0 { assert_eq!(hp[i], 0.0); }
        }
    }

    #[test]
    fn test_vsharp_no_deconvolve_single_radius() {
        // One radius without deconvolution is the plain SMV high-pass, not SHARP.
        let n = 20;
        let (field, mask) = sphere_case(n, 8.0);
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let (hp, hp_mask) = vsharp_with_radii(&field, &mask, &grid, &[3.0], None, |_, _| {});
        let (sh, sh_mask) = crate::bgremove::sharp::sharp_core(&field, &mask, &grid, 0.05, 3.0);
        assert_eq!(hp_mask, sh_mask);
        assert!(hp.iter().zip(&sh).any(|(a, b)| (a - b).abs() > 1e-6));
    }
}

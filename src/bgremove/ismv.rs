//! Iterative Spherical Mean Value (iSMV) background field removal
//!
//! Iterative approach that avoids mask erosion by iteratively
//! correcting boundary values.
//!
//! Reference:
//! Wen, Y., Zhou, D., Liu, T., Spincemaille, P., Wang, Y. (2014).
//! "An iterative spherical mean value method for background field removal in MRI."
//! Magnetic Resonance in Medicine, 72(4):1065-1071. https://doi.org/10.1002/mrm.24998
//!
//! Reference implementation: https://github.com/kamesy/QSM.jl

use num_complex::Complex64;
use crate::Grid;
use crate::fft::{fft3d, ifft3d, fft_real_kernel};
use crate::kernels::smv::{smv_kernel, erode_mask_smv};
use crate::utils::vec_norm;

/// iSMV algorithm parameters
#[derive(Clone, Debug)]
pub struct IsmvParams {
    /// Convergence tolerance
    pub tol: f64,
    /// Maximum iterations
    pub max_iter: usize,
    /// Kernel radius factor (multiplied by max voxel size; default: 2.0)
    pub radius_factor: f64,
}

impl Default for IsmvParams {
    fn default() -> Self {
        Self { tol: 1e-3, max_iter: 500, radius_factor: 2.0 }
    }
}

/// iSMV background field removal
///
/// # Arguments
/// * `field` - Total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = brain, 0 = background
/// * `grid` - Volume dimensions and voxel sizes
/// * `radius` - SMV kernel radius in mm
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
/// * `progress` - Progress callback (iteration, max_iter)
///
/// # Returns
/// Tuple of (local field, eroded mask)
pub fn ismv(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    radius: f64,
    tol: f64,
    max_iter: usize,
    mut progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<u8>) {
    let (nx, ny, nz) = grid.dims;
    let n_total = nx * ny * nz;

    // Generate SMV kernel
    let smv = smv_kernel(grid, radius);
    let smv_fft_real = fft_real_kernel(&smv, nx, ny, nz);

    // Complex FFT needed for iteration loop
    let mut smv_complex: Vec<Complex64> = smv.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut smv_complex, nx, ny, nz);
    let smv_fft = smv_complex;

    // Convert mask to f64
    let m0: Vec<f64> = mask.iter()
        .map(|&m| if m != 0 { 1.0 } else { 0.0 })
        .collect();

    // Erode mask using SMV
    let eroded_mask = erode_mask_f64(mask, &smv_fft_real, grid);

    // Boundary mask: original mask minus eroded mask
    let boundary: Vec<f64> = m0.iter()
        .zip(eroded_mask.iter())
        .map(|(&m, &e)| m - e)
        .collect();

    // Initialize: f = field
    let mut f: Vec<f64> = field.to_vec();

    // f0 = eroded_mask * field (for residual calculation)
    let mut f0: Vec<f64> = field.iter()
        .zip(eroded_mask.iter())
        .map(|(&fi, &m)| fi * m)
        .collect();

    // Boundary correction: bc = boundary * field
    let bc: Vec<f64> = field.iter()
        .zip(boundary.iter())
        .map(|(&fi, &b)| fi * b)
        .collect();

    // Initial residual norm
    let mut nr = vec_norm(&f0);
    let eps = tol * nr;

    // iSMV iterations
    for iter in 0..max_iter {
        // Report progress
        progress(iter + 1, max_iter);

        if nr <= eps {
            progress(iter + 1, iter + 1);
            break;
        }

        // f = SMV(f)
        let mut f_complex: Vec<Complex64> = f.iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect();

        fft3d(&mut f_complex, nx, ny, nz);

        for i in 0..n_total {
            f_complex[i] *= smv_fft[i];
        }

        ifft3d(&mut f_complex, nx, ny, nz);

        // f = eroded_mask * f + bc
        for i in 0..n_total {
            f[i] = eroded_mask[i] * f_complex[i].re + bc[i];
        }

        // Compute residual: ||f0 - f||
        let mut residual_sq = 0.0;
        for i in 0..n_total {
            let diff = f0[i] - f[i];
            residual_sq += diff * diff;
            f0[i] = f[i];
        }
        nr = residual_sq.sqrt();
    }

    // Compute local field: m * (field - f)
    let mut local_field = vec![0.0; n_total];
    for i in 0..n_total {
        if mask[i] != 0 {
            local_field[i] = field[i] - f[i];
        }
    }

    // Convert eroded mask to u8
    let eroded_mask_u8: Vec<u8> = eroded_mask.iter()
        .map(|&m| if m > 0.5 { 1 } else { 0 })
        .collect();

    (local_field, eroded_mask_u8)
}

/// Erode mask using SMV convolution (returns f64 for arithmetic compatibility)
fn erode_mask_f64(mask: &[u8], smv_kernel_fft: &[f64], grid: &Grid) -> Vec<f64> {
    let eroded = erode_mask_smv(mask, smv_kernel_fft, grid, 1.0 - 1e-10);
    eroded.iter().map(|&m| m as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ismv_zero_field() {
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let (local, eroded) = ismv(
            &field, &mask, &grid,
            2.0, 1e-3, 10, |_, _| {}
        );

        for &val in local.iter() {
            assert!(val.abs() < 1e-10, "Zero field should give zero local field");
        }

        // Some voxels should be in eroded mask
        let eroded_count: usize = eroded.iter().map(|&m| m as usize).sum();
        assert!(eroded_count > 0, "Eroded mask should have some voxels");
    }

    #[test]
    fn test_ismv_finite() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let (local, _eroded) = ismv(
            &field, &mask, &grid,
            2.0, 1e-3, 20, |_, _| {}
        );

        for (i, &val) in local.iter().enumerate() {
            assert!(val.is_finite(), "Local field should be finite at index {}", i);
        }
    }

    #[test]
    fn test_ismv_preserves_interior() {
        // iSMV should preserve some of the mask interior
        let n = 16;
        let field = vec![0.1; n * n * n];

        // Create a spherical mask
        let mut mask = vec![0u8; n * n * n];
        let center = n / 2;
        let radius = n / 3;

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let di = (i as i32) - (center as i32);
                    let dj = (j as i32) - (center as i32);
                    let dk = (k as i32) - (center as i32);
                    if di*di + dj*dj + dk*dk <= (radius * radius) as i32 {
                        mask[i * n * n + j * n + k] = 1;
                    }
                }
            }
        }

        let mask_count: usize = mask.iter().map(|&m| m as usize).sum();
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        // Use small radius for less erosion
        let (_, eroded) = ismv(
            &field, &mask, &grid,
            1.5, 1e-3, 50, |_, _| {}
        );

        let eroded_count: usize = eroded.iter().map(|&m| m as usize).sum();

        // Eroded mask should have fewer voxels than original
        assert!(eroded_count <= mask_count, "Eroded mask should be smaller than original");
        // Should preserve at least some interior voxels
        assert!(eroded_count > 0, "Eroded mask should have some voxels");
    }

    #[test]
    fn test_ismv_convergence() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        // Run with more iterations and tight tolerance
        let (local_many, _) = ismv(
            &field, &mask, &grid,
            2.0, 1e-6, 100, |_, _| {}
        );

        // Run with fewer iterations
        let (local_few, _) = ismv(
            &field, &mask, &grid,
            2.0, 1e-6, 5, |_, _| {}
        );

        // Both should be finite
        for (i, &val) in local_many.iter().enumerate() {
            assert!(val.is_finite(), "iSMV many iters: finite at index {}", i);
        }
        for (i, &val) in local_few.iter().enumerate() {
            assert!(val.is_finite(), "iSMV few iters: finite at index {}", i);
        }

        // More iterations should give different (hopefully more converged) result
        // or the same if already converged
        let diff_norm: f64 = local_many.iter()
            .zip(local_few.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        // The difference should be finite (no NaN/Inf)
        assert!(diff_norm.is_finite(), "Difference between runs should be finite");
    }

    #[test]
    fn test_ismv_different_radius() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        // Small radius
        let (local_small, eroded_small) = ismv(
            &field, &mask, &grid,
            1.5, 1e-3, 20, |_, _| {}
        );

        // Larger radius
        let (local_large, eroded_large) = ismv(
            &field, &mask, &grid,
            3.0, 1e-3, 20, |_, _| {}
        );

        // Both should produce finite results
        for (i, &val) in local_small.iter().enumerate() {
            assert!(val.is_finite(), "iSMV small radius: finite at index {}", i);
        }
        for (i, &val) in local_large.iter().enumerate() {
            assert!(val.is_finite(), "iSMV large radius: finite at index {}", i);
        }

        // Larger radius should erode more
        let small_count: usize = eroded_small.iter().map(|&m| m as usize).sum();
        let large_count: usize = eroded_large.iter().map(|&m| m as usize).sum();
        assert!(
            large_count <= small_count,
            "Larger radius should erode more: large={}, small={}",
            large_count, small_count
        );
    }

    #[test]
    fn test_ismv_larger_volume() {
        // Test with 16x16x16 volume and spherical mask
        let n = 16;

        // Create a field with a linear background
        let mut field = vec![0.0; n * n * n];
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
        let (local, eroded) = ismv(
            &field, &mask, &grid,
            2.0, 1e-3, 50, |_, _| {}
        );

        assert_eq!(local.len(), n * n * n);
        for &val in &local {
            assert!(val.is_finite());
        }

        // Eroded mask should be non-empty
        let eroded_count: usize = eroded.iter().map(|&m| m as usize).sum();
        assert!(eroded_count > 0, "Eroded mask should have some voxels");

        // Outside original mask should be zero
        for i in 0..n * n * n {
            if mask[i] == 0 {
                assert_eq!(local[i], 0.0, "Outside mask should be zero");
            }
        }
    }

    #[test]
    fn test_ismv_with_progress() {
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let mut progress_calls = Vec::new();
        let (local, _) = ismv(
            &field, &mask, &grid,
            2.0, 1e-3, 20,
            |iter, max| { progress_calls.push((iter, max)); }
        );

        assert_eq!(local.len(), n * n * n);
        assert!(!progress_calls.is_empty(), "Progress callback should be called");
        for &val in &local {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_ismv_anisotropic_voxels() {
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 0.5, 1.0, 2.0);

        // Anisotropic voxel sizes
        let (local, eroded) = ismv(
            &field, &mask, &grid,
            3.0, 1e-3, 20, |_, _| {}
        );

        for &val in &local {
            assert!(val.is_finite());
        }
        // Should still produce some eroded mask
        let count: usize = eroded.iter().map(|&m| m as usize).sum();
        assert!(count <= n * n * n);
    }

    #[test]
    fn test_ismv_tight_convergence() {
        // Test tight convergence to ensure the convergence check branch is hit
        let n = 8;
        let field = vec![0.5; n * n * n]; // Constant field
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let (local, _) = ismv(
            &field, &mask, &grid,
            2.0, 1e-12, 200, |_, _| {} // Very tight tolerance, many iterations allowed
        );

        for &val in &local {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_ismv_with_background_mask() {
        // Test where some voxels are outside the mask
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();

        // Mask with zero border
        let mut mask = vec![0u8; n * n * n];
        for z in 1..(n - 1) {
            for y in 1..(n - 1) {
                for x in 1..(n - 1) {
                    mask[x + y * n + z * n * n] = 1;
                }
            }
        }

        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let (local, eroded) = ismv(
            &field, &mask, &grid,
            1.5, 1e-3, 30, |_, _| {}
        );

        for &val in &local {
            assert!(val.is_finite());
        }

        // Outside mask should be zero
        for i in 0..n * n * n {
            if mask[i] == 0 {
                assert_eq!(local[i], 0.0, "Outside mask should be zero at index {}", i);
            }
        }

        // Eroded mask should be subset of original mask
        for i in 0..n * n * n {
            if eroded[i] != 0 {
                assert_eq!(mask[i], 1, "Eroded voxel must be inside original mask");
            }
        }
    }
}

//! Spherical Mean Value (SMV) kernel
//!
//! Creates a binary sphere kernel used for background field removal methods
//! like SHARP and V-SHARP.

use crate::Grid;

/// Generate SMV kernel in image space
///
/// Creates a binary sphere of given radius, normalized so sum = 1.
/// Kernel is centered at index (0, 0, 0) for FFT compatibility.
///
/// # Arguments
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `radius` - Sphere radius in mm
///
/// # Returns
/// Flattened SMV kernel array of size nx*ny*nz in C order, normalized
pub fn smv_kernel(
    grid: &Grid,
    radius: f64,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = grid.voxel_size;
    let n_total = nx * ny * nz;
    let mut s = vec![0.0; n_total];
    let r_squared = radius * radius;

    // Create binary sphere centered at (0,0,0) with wraparound
    let mut count = 0.0;

    // Fortran order: index = i + j*nx + k*nx*ny
    for k in 0..nz {
        let zk = if k <= nz / 2 { k as f64 } else { (k as i64 - nz as i64) as f64 };
        let dz = zk * vsz;

        for j in 0..ny {
            let yj = if j <= ny / 2 { j as f64 } else { (j as i64 - ny as i64) as f64 };
            let dy = yj * vsy;

            for i in 0..nx {
                // Distance from center in x (with wraparound)
                let xi = if i <= nx / 2 { i as f64 } else { (i as i64 - nx as i64) as f64 };
                let dx = xi * vsx;

                let dist_sq = dx * dx + dy * dy + dz * dz;

                let idx = i + j * nx + k * nx * ny;

                if dist_sq <= r_squared {
                    s[idx] = 1.0;
                    count += 1.0;
                }
            }
        }
    }

    // Normalize so sum = 1
    if count > 0.0 {
        let norm = 1.0 / count;
        for val in s.iter_mut() {
            *val *= norm;
        }
    }

    s
}

/// Erode a binary mask using FFT-based SMV convolution.
///
/// Convolves the mask with an SMV kernel in Fourier space and thresholds.
/// Voxels where SMV(mask) > delta are kept; others are removed.
///
/// # Arguments
/// * `mask` - Binary input mask
/// * `smv_kernel_fft` - Pre-computed FFT of SMV kernel (real parts via `fft_real_kernel`)
/// * `nx`, `ny`, `nz` - Volume dimensions
/// * `delta` - Threshold (e.g. `1.0 - 1e-7_f64.sqrt()`)
pub fn erode_mask_smv(
    mask: &[u8],
    smv_kernel_fft: &[f64],
    grid: &Grid,
    delta: f64,
) -> Vec<u8> {
    use num_complex::Complex64;
    use crate::fft::{fft3d, ifft3d};

    let (nx, ny, nz) = grid.dims;
    let n_total = nx * ny * nz;
    let mut mask_complex: Vec<Complex64> = mask.iter()
        .map(|&m| Complex64::new(m as f64, 0.0))
        .collect();

    fft3d(&mut mask_complex, nx, ny, nz);

    for i in 0..n_total {
        mask_complex[i] *= smv_kernel_fft[i];
    }

    ifft3d(&mut mask_complex, nx, ny, nz);

    mask_complex.iter()
        .map(|c| if c.re > delta { 1 } else { 0 })
        .collect()
}

// ============================================================================
// F32 (Single Precision) SMV Kernel Functions
// ============================================================================

/// Generate SMV kernel in image space (f32 version for WASM performance)
///
/// Creates a binary sphere of given radius, normalized so sum = 1.
/// Kernel is centered at index (0, 0, 0) for FFT compatibility.
pub fn smv_kernel_f32(
    grid: &Grid,
    radius: f32,
) -> Vec<f32> {
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = (grid.vsx() as f32, grid.vsy() as f32, grid.vsz() as f32);
    let n_total = nx * ny * nz;
    let mut s = vec![0.0f32; n_total];
    let r_squared = radius * radius;

    // Create binary sphere centered at (0,0,0) with wraparound
    let mut count = 0.0f32;

    // Fortran order: index = i + j*nx + k*nx*ny
    for k in 0..nz {
        let zk = if k <= nz / 2 { k as f32 } else { (k as i64 - nz as i64) as f32 };
        let dz = zk * vsz;

        for j in 0..ny {
            let yj = if j <= ny / 2 { j as f32 } else { (j as i64 - ny as i64) as f32 };
            let dy = yj * vsy;

            for i in 0..nx {
                // Distance from center in x (with wraparound)
                let xi = if i <= nx / 2 { i as f32 } else { (i as i64 - nx as i64) as f32 };
                let dx = xi * vsx;

                let dist_sq = dx * dx + dy * dy + dz * dz;

                let idx = i + j * nx + k * nx * ny;

                if dist_sq <= r_squared {
                    s[idx] = 1.0;
                    count += 1.0;
                }
            }
        }
    }

    // Normalize so sum = 1
    if count > 0.0 {
        let norm = 1.0 / count;
        for val in s.iter_mut() {
            *val *= norm;
        }
    }

    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Grid;

    fn grid(n: usize) -> Grid {
        Grid::new(n, n, n, 1.0, 1.0, 1.0)
    }

    #[test]
    fn test_smv_kernel_normalization() {
        let s = smv_kernel(&grid(16), 3.0);
        let sum: f64 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "SMV kernel should sum to 1, got {}", sum);
    }

    #[test]
    fn test_smv_kernel_center() {
        let s = smv_kernel(&grid(16), 2.0);
        assert!(s[0] > 0.0, "Center voxel should be in sphere");
    }

    #[test]
    fn test_smv_kernel_radius() {
        let s = smv_kernel(&grid(8), 0.5);
        let count: usize = s.iter().filter(|&&v| v > 0.0).count();
        assert_eq!(count, 1, "Radius 0.5 should only include center");
    }
}

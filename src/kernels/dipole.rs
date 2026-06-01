//! Dipole kernel for QSM
//!
//! The dipole kernel describes the relationship between magnetic susceptibility
//! and the induced magnetic field in MRI. In k-space:
//!
//! D(k) = 1/3 - (k·B)² / |k|²
//!
//! where B is the B0 field direction (typically [0, 0, 1]).

use crate::fft::fftfreq;
use crate::Grid;

/// Generate dipole kernel in k-space
///
/// Creates the dipole kernel D(k) = 1/3 - (kz)²/|k|² (for B = [0,0,1])
/// centered at index (0, 0, 0) (not shifted).
///
/// # Arguments
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `bdir` - B0 field direction as (bx, by, bz), default (0, 0, 1)
///
/// # Returns
/// Flattened dipole kernel array of size nx*ny*nz in C order
pub fn dipole_kernel(
    grid: &Grid,
    bdir: (f64, f64, f64),
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = grid.voxel_size;
    let n_total = nx * ny * nz;
    let mut d = vec![0.0; n_total];

    // Generate frequency grids
    let kx = fftfreq(nx, vsx);
    let ky = fftfreq(ny, vsy);
    let kz = fftfreq(nz, vsz);

    // Normalize B direction
    let (bx, by, bz) = bdir;
    let bnorm = (bx * bx + by * by + bz * bz).sqrt();
    let bx = bx / bnorm;
    let by = by / bnorm;
    let bz = bz / bnorm;

    let one_third = 1.0 / 3.0;

    // Compute dipole kernel: D(k) = 1/3 - (k·B)² / |k|²
    // Fortran order: index = i + j*nx + k*nx*ny
    for k in 0..nz {
        let kz_val = kz[k];
        for j in 0..ny {
            let ky_val = ky[j];
            for i in 0..nx {
                let kx_val = kx[i];

                let k_dot_b = kx_val * bx + ky_val * by + kz_val * bz;
                let k_squared = kx_val * kx_val + ky_val * ky_val + kz_val * kz_val;

                let idx = i + j * nx + k * nx * ny;

                if k_squared > 1e-20 {
                    d[idx] = one_third - (k_dot_b * k_dot_b) / k_squared;
                } else {
                    // DC term (k = 0)
                    d[idx] = 0.0;
                }
            }
        }
    }

    d
}

// ============================================================================
// F32 (Single Precision) Dipole Kernel Functions
// ============================================================================

use crate::fft::fftfreq_f32;

/// Generate dipole kernel in k-space (f32 version for WASM performance)
///
/// Creates the dipole kernel D(k) = 1/3 - (kz)²/|k|² (for B = [0,0,1])
/// centered at index (0, 0, 0) (not shifted).
pub fn dipole_kernel_f32(
    grid: &Grid,
    bdir: (f32, f32, f32),
) -> Vec<f32> {
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = (grid.vsx() as f32, grid.vsy() as f32, grid.vsz() as f32);
    let n_total = nx * ny * nz;
    let mut d = vec![0.0f32; n_total];

    // Generate frequency grids
    let kx = fftfreq_f32(nx, vsx);
    let ky = fftfreq_f32(ny, vsy);
    let kz = fftfreq_f32(nz, vsz);

    // Normalize B direction
    let (bx, by, bz) = bdir;
    let bnorm = (bx * bx + by * by + bz * bz).sqrt();
    let bx = bx / bnorm;
    let by = by / bnorm;
    let bz = bz / bnorm;

    let one_third = 1.0f32 / 3.0;

    // Compute dipole kernel: D(k) = 1/3 - (k·B)² / |k|²
    // Fortran order: index = i + j*nx + k*nx*ny
    for k in 0..nz {
        let kz_val = kz[k];
        for j in 0..ny {
            let ky_val = ky[j];
            for i in 0..nx {
                let kx_val = kx[i];

                let k_dot_b = kx_val * bx + ky_val * by + kz_val * bz;
                let k_squared = kx_val * kx_val + ky_val * ky_val + kz_val * kz_val;

                let idx = i + j * nx + k * nx * ny;

                if k_squared > 1e-10 {
                    d[idx] = one_third - (k_dot_b * k_dot_b) / k_squared;
                } else {
                    // DC term (k = 0)
                    d[idx] = 0.0;
                }
            }
        }
    }

    d
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::Grid;

    fn grid(n: usize) -> Grid {
        Grid::new(n, n, n, 1.0, 1.0, 1.0)
    }

    #[test]
    fn test_dipole_kernel_dc() {
        let d = dipole_kernel(&grid(4), (0.0, 0.0, 1.0));
        assert!(d[0].abs() < 1e-10, "DC component should be 0, got {}", d[0]);
    }

    #[test]
    fn test_dipole_kernel_range() {
        let d = dipole_kernel(&grid(8), (0.0, 0.0, 1.0));

        for (i, &val) in d.iter().enumerate() {
            if i > 0 {
                assert!(val >= -2.0/3.0 - 1e-10 && val <= 1.0/3.0 + 1e-10,
                    "Dipole value {} out of range at index {}", val, i);
            }
        }
    }

    #[test]
    fn test_dipole_kernel_symmetry() {
        let n = 8;
        let d = dipole_kernel(&grid(n), (0.0, 0.0, 1.0));

        // D(kx, ky, kz) should equal D(-kx, -ky, -kz)
        for i in 1..n/2 {
            let i_neg = n - i;
            for j in 1..n/2 {
                let j_neg = n - j;
                for k in 1..n/2 {
                    let k_neg = n - k;

                    let idx1 = i * n * n + j * n + k;
                    let idx2 = i_neg * n * n + j_neg * n + k_neg;

                    let diff = (d[idx1] - d[idx2]).abs();
                    assert!(diff < 1e-10,
                        "Symmetry broken at ({},{},{}) vs ({},{},{}): {} vs {}",
                        i, j, k, i_neg, j_neg, k_neg, d[idx1], d[idx2]);
                }
            }
        }
    }
}

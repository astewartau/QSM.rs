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

/// Largest |D(k)| the dipole kernel takes: `D = 1/3 − cos²θ` spans `[−2/3, 1/3]`, so |D| ≤ 2/3.
///
/// A k-space truncation threshold on |D| (TKD, TSVD, RTS's well-conditioned set, HEIDI's
/// cone) at or above this value truncates every frequency, and the result is all zero or all
/// regularisation. Hosts bounding such a threshold should take the limit from here.
pub const MAX_ABS_DIPOLE: f64 = 2.0 / 3.0;

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

/// The MEDI toolbox's `dipole_kernel.m` k-space kernel exactly as MATLAB builds it: the grid
/// `-N/2:N/2-1` per axis divided by `N·voxel`, `D = 1/3 − (k·B0)²/|k|²` (0 where `k = 0`), then
/// `fftshift`. `bdir` is used as given (MATLAB does not normalise it).
///
/// On an axis with an even number of voxels this is [`dipole_kernel`] (to rounding). On an odd
/// axis `-N/2:N/2-1` runs over half-integers, so every sample sits half a frequency step off the
/// DFT grid and index 0 holds `k = −1/2` rather than DC; the kernel is then not
/// Hermitian-symmetric and has no zero at DC. [`dipole_kernel`] samples the DFT grid. Ports of
/// MEDI functions that run on odd grids use this one where they must reproduce `dipole_kernel.m`.
pub fn dipole_kernel_medi(grid: &Grid, bdir: (f64, f64, f64)) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = grid.voxel_size;
    // fftshift(-N/2:N/2-1)(i) = -N/2 + mod(i + floor(N/2), N), divided by N*voxel as MATLAB does
    let axis = |n: usize, vs: f64| -> Vec<f64> {
        (0..n).map(|i| (-(n as f64) / 2.0 + ((i + n / 2) % n) as f64) / (n as f64 * vs)).collect()
    };
    let (kx, ky, kz) = (axis(nx, vsx), axis(ny, vsy), axis(nz, vsz));
    let (bx, by, bz) = bdir;
    let mut d = vec![0.0; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let (x, y, z) = (kx[i], ky[j], kz[k]);
                let kb = x * bx + y * by + z * bz;
                let k2 = x * x + y * y + z * z;
                d[i + nx * (j + ny * k)] = if k2 == 0.0 { 0.0 } else { 1.0 / 3.0 - kb * kb / k2 };
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
    fn medi_kernel_is_the_dft_kernel_on_even_grids_and_off_grid_on_odd() {
        let b = (0.1, 0.2, 0.97f64);
        let n = (b.0 * b.0 + b.1 * b.1 + b.2 * b.2).sqrt();
        let b = (b.0 / n, b.1 / n, b.2 / n);
        let even = Grid::new(8, 6, 4, 1.0, 1.1, 1.2);
        for (x, y) in dipole_kernel(&even, b).iter().zip(dipole_kernel_medi(&even, b)) {
            assert!((x - y).abs() < 1e-15, "{x} {y}");
        }
        // odd x: index 0 holds kx = -1/2 step, so D(0) = 1/3 - (bx/2)^2/(1/2)^2 = 1/3 - bx^2
        let odd = Grid::new(7, 6, 4, 1.0, 1.0, 1.0);
        let d = dipole_kernel_medi(&odd, b);
        assert!((d[0] - (1.0 / 3.0 - b.0 * b.0)).abs() < 1e-15, "{}", d[0]);
        assert_eq!(dipole_kernel(&odd, b)[0], 0.0);
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
                let (lo, hi) = (-2.0 / 3.0 - 1e-10, 1.0 / 3.0 + 1e-10);
                assert!((lo..=hi).contains(&val),
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

    #[test]
    fn max_abs_dipole_bounds_the_kernel() {
        let grid = Grid::new(9, 8, 7, 1.0, 1.2, 0.8);
        for bdir in [(0.0, 0.0, 1.0), (0.3, 0.2, 0.93)] {
            let d = dipole_kernel(&grid, bdir);
            let max = d.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert!(max <= MAX_ABS_DIPOLE + 1e-12, "{max}");
        }
    }
}

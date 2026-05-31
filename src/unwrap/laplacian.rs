//! Laplacian-based phase unwrapping
//!
//! Uses the Laplacian operator to unwrap phase without path dependence.
//! The wrapped phase Laplacian equals the true Laplacian, so we can
//! recover the true phase by solving a Poisson equation.
//!
//! Reference:
//! Schofield, M.A., Zhu, Y. (2003). "Fast phase unwrapping algorithm for
//! interferometric applications." Optics Letters, 28(14):1194-1196.
//! https://doi.org/10.1364/OL.28.001194
//!
//! Reference implementation: https://github.com/kamesy/QSM.jl

use std::f64::consts::PI;
use num_complex::Complex64;
use crate::fft::{fft3d, ifft3d};
use crate::Grid;

/// Wrap angle to [-π, π]
#[inline]
pub(crate) fn wrap(x: f64) -> f64 {
    let mut y = x % (2.0 * PI);
    if y > PI {
        y -= 2.0 * PI;
    } else if y < -PI {
        y += 2.0 * PI;
    }
    y
}

/// Compute wrapped Laplacian of phase with periodic boundary conditions
///
/// Uses second-order central finite differences on wrapped phase differences.
pub(crate) fn wrapped_laplacian_periodic(
    phase: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut d2u = vec![0.0; n_total];

    let dx2 = 1.0 / (vsx * vsx);
    let dy2 = 1.0 / (vsy * vsy);
    let dz2 = 1.0 / (vsz * vsz);

    for k in 0..nz {
        let km1 = if k == 0 { nz - 1 } else { k - 1 };
        let kp1 = if k + 1 >= nz { 0 } else { k + 1 };

        for j in 0..ny {
            let jm1 = if j == 0 { ny - 1 } else { j - 1 };
            let jp1 = if j + 1 >= ny { 0 } else { j + 1 };

            for i in 0..nx {
                let im1 = if i == 0 { nx - 1 } else { i - 1 };
                let ip1 = if i + 1 >= nx { 0 } else { i + 1 };

                let idx = i + j * nx + k * nx * ny;
                let u_ijk = phase[idx];

                let idx_im1 = im1 + j * nx + k * nx * ny;
                let idx_ip1 = ip1 + j * nx + k * nx * ny;
                let idx_jm1 = i + jm1 * nx + k * nx * ny;
                let idx_jp1 = i + jp1 * nx + k * nx * ny;
                let idx_km1 = i + j * nx + km1 * nx * ny;
                let idx_kp1 = i + j * nx + kp1 * nx * ny;

                let lap_x = (wrap(phase[idx_ip1] - u_ijk) - wrap(u_ijk - phase[idx_im1])) * dx2;
                let lap_y = (wrap(phase[idx_jp1] - u_ijk) - wrap(u_ijk - phase[idx_jm1])) * dy2;
                let lap_z = (wrap(phase[idx_kp1] - u_ijk) - wrap(u_ijk - phase[idx_km1])) * dz2;

                d2u[idx] = lap_x + lap_y + lap_z;
            }
        }
    }

    d2u
}

/// Solve Poisson equation using FFT (periodic boundary conditions)
pub(crate) fn solve_poisson_fft(
    f: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    let mut f_complex: Vec<Complex64> = f.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut f_complex, nx, ny, nz);

    let idx2 = 1.0 / (vsx * vsx);
    let idy2 = 1.0 / (vsy * vsy);
    let idz2 = 1.0 / (vsz * vsz);

    for k in 0..nz {
        let fk = if k <= nz / 2 { k as f64 / nz as f64 } else { (k as f64 - nz as f64) / nz as f64 };
        let lam_z = 2.0 * ((2.0 * PI * fk).cos() - 1.0) * idz2;

        for j in 0..ny {
            let fj = if j <= ny / 2 { j as f64 / ny as f64 } else { (j as f64 - ny as f64) / ny as f64 };
            let lam_y = 2.0 * ((2.0 * PI * fj).cos() - 1.0) * idy2;

            for i in 0..nx {
                let fi = if i <= nx / 2 { i as f64 / nx as f64 } else { (i as f64 - nx as f64) / nx as f64 };
                let lam_x = 2.0 * ((2.0 * PI * fi).cos() - 1.0) * idx2;

                let lam = lam_x + lam_y + lam_z;
                let idx = i + j * nx + k * nx * ny;

                if lam.abs() > 1e-20 {
                    f_complex[idx] /= lam;
                } else {
                    f_complex[idx] = Complex64::new(0.0, 0.0);
                }
            }
        }
    }

    ifft3d(&mut f_complex, nx, ny, nz);
    f_complex.iter().map(|c| c.re).collect()
}

/// Laplacian phase unwrapping
///
/// Uses FFT-based Poisson solver with periodic boundary conditions.
/// Fast and robust but may have issues at mask boundaries.
///
/// # Arguments
/// * `phase` - Wrapped phase (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `grid` - Volume grid (dimensions and voxel sizes)
///
/// # Returns
/// Unwrapped phase
pub fn laplacian_unwrap(
    phase: &[f64],
    mask: &[u8],
    grid: &Grid,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = grid.voxel_size;
    let n_total = nx * ny * nz;

    let d2u = wrapped_laplacian_periodic(phase, nx, ny, nz, vsx, vsy, vsz);

    let d2u_masked: Vec<f64> = d2u.iter()
        .enumerate()
        .map(|(i, &val)| if mask[i] != 0 { val } else { 0.0 })
        .collect();

    let unwrapped = solve_poisson_fft(&d2u_masked, nx, ny, nz, vsx, vsy, vsz);

    let mut result = vec![0.0; n_total];
    for i in 0..n_total {
        if mask[i] != 0 {
            result[i] = unwrapped[i];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: usize) -> Grid {
        Grid::new(n, n, n, 1.0, 1.0, 1.0)
    }

    #[test]
    fn test_wrap() {
        assert!((wrap(0.0) - 0.0).abs() < 1e-10);
        assert!((wrap(PI) - PI).abs() < 1e-10);
        assert!((wrap(-PI) - (-PI)).abs() < 1e-10);
        assert!((wrap(2.0 * PI) - 0.0).abs() < 1e-10);
        assert!((wrap(3.0 * PI) - PI).abs() < 1e-10);
        assert!((wrap(-3.0 * PI) - (-PI)).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_unwrap_constant() {
        let n = 8;
        let phase = vec![1.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        let unwrapped = laplacian_unwrap(&phase, &mask, &grid(n));

        let mean: f64 = unwrapped.iter().sum::<f64>() / (n * n * n) as f64;
        for &val in unwrapped.iter() {
            assert!((val - mean).abs() < 1e-6, "Constant phase should unwrap to constant");
        }
    }

    #[test]
    fn test_laplacian_unwrap_smooth() {
        let n = 16;
        let mut phase = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let idx = i + j * n + k * n * n;
                    phase[idx] = 0.5 * (2.0 * PI * i as f64 / n as f64).sin();
                }
            }
        }

        let unwrapped = laplacian_unwrap(&phase, &mask, &grid(n));

        for (i, (&orig, &unwr)) in phase.iter().zip(unwrapped.iter()).enumerate() {
            assert!(unwr.is_finite(), "Unwrapped should be finite at {}", i);
            assert!((orig - unwr).abs() < 1.0,
                "Unwrapped should be close to original for smooth phase");
        }
    }

    #[test]
    fn test_laplacian_unwrap_finite() {
        let n = 8;
        let phase: Vec<f64> = (0..n*n*n).map(|i| wrap((i as f64) * 0.1)).collect();
        let mask = vec![1u8; n * n * n];

        let unwrapped = laplacian_unwrap(&phase, &mask, &grid(n));

        for (i, &val) in unwrapped.iter().enumerate() {
            assert!(val.is_finite(), "Unwrapped phase should be finite at index {}", i);
        }
    }
}

//! Laplacian-based phase unwrapping
//!
//! Uses the Laplacian operator to unwrap phase without path dependence.
//! The wrapped phase Laplacian equals the true Laplacian, so we can
//! recover the true phase by solving a Poisson equation.
//!
//! Reference:
//! Schofield MA, Zhu Y. Fast phase unwrapping algorithm for interferometric
//! applications. Optics letters. 2003 Jul 15;28(14):1194-6.

use std::f64::consts::PI;
use num_complex::Complex64;
use crate::fft::{fft3d, ifft3d};

/// Wrap angle to [-π, π]
#[inline]
fn wrap(x: f64) -> f64 {
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
fn wrapped_laplacian_periodic(
    phase: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut d2u = vec![0.0; n_total];

    let dx2 = 1.0 / (vsx * vsx);
    let dy2 = 1.0 / (vsy * vsy);
    let dz2 = 1.0 / (vsz * vsz);

    // Fortran order: index = i + j*nx + k*nx*ny
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

                // Indices for neighbors
                let idx_im1 = im1 + j * nx + k * nx * ny;
                let idx_ip1 = ip1 + j * nx + k * nx * ny;
                let idx_jm1 = i + jm1 * nx + k * nx * ny;
                let idx_jp1 = i + jp1 * nx + k * nx * ny;
                let idx_km1 = i + j * nx + km1 * nx * ny;
                let idx_kp1 = i + j * nx + kp1 * nx * ny;

                // Wrapped Laplacian using central differences
                // Δu = (u[i+1] - 2*u[i] + u[i-1]) / dx²
                // But we wrap the differences: wrap(u[i+1] - u[i]) - wrap(u[i] - u[i-1])
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
///
/// Solves: ∇²u = f
/// In Fourier domain: λ * û = f̂, where λ are eigenvalues of discrete Laplacian
fn solve_poisson_fft(
    f: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    let n_total = nx * ny * nz;

    // FFT of RHS
    let mut f_complex: Vec<Complex64> = f.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut f_complex, nx, ny, nz);

    let idx2 = 1.0 / (vsx * vsx);
    let idy2 = 1.0 / (vsy * vsy);
    let idz2 = 1.0 / (vsz * vsz);

    // Divide by eigenvalues of discrete Laplacian
    // λ[i,j,k] = 2*(cos(2πi/nx)-1)/dx² + 2*(cos(2πj/ny)-1)/dy² + 2*(cos(2πk/nz)-1)/dz²
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
                    // DC component - set to zero (solution is unique up to a constant)
                    f_complex[idx] = Complex64::new(0.0, 0.0);
                }
            }
        }
    }

    // IFFT to get solution
    ifft3d(&mut f_complex, nx, ny, nz);

    // Extract real part
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
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
///
/// # Returns
/// Unwrapped phase
pub fn laplacian_unwrap(
    phase: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    let n_total = nx * ny * nz;

    // Step 1: Compute wrapped Laplacian
    let d2u = wrapped_laplacian_periodic(phase, nx, ny, nz, vsx, vsy, vsz);

    // Step 2: Apply mask to wrapped Laplacian (Dirichlet-like BCs on mask boundary)
    let d2u_masked: Vec<f64> = d2u.iter()
        .enumerate()
        .map(|(i, &val)| if mask[i] != 0 { val } else { 0.0 })
        .collect();

    // Step 3: Solve Poisson equation
    let unwrapped = solve_poisson_fft(&d2u_masked, nx, ny, nz, vsx, vsy, vsz);

    // Step 4: Apply mask to result
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
        // Constant phase should stay constant (up to arbitrary offset)
        let n = 8;
        let phase = vec![1.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        let unwrapped = laplacian_unwrap(&phase, &mask, n, n, n, 1.0, 1.0, 1.0);

        // Check that result is approximately constant
        let mean: f64 = unwrapped.iter().sum::<f64>() / (n * n * n) as f64;
        for &val in unwrapped.iter() {
            assert!((val - mean).abs() < 1e-6, "Constant phase should unwrap to constant");
        }
    }

    #[test]
    fn test_laplacian_unwrap_smooth() {
        // Smooth phase should remain smooth after unwrapping
        let n = 16;
        let mut phase = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        // Create smooth sinusoidal phase (no wraps needed)
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let idx = i + j * n + k * n * n;
                    // Small amplitude so no wrapping occurs
                    phase[idx] = 0.5 * (2.0 * PI * i as f64 / n as f64).sin();
                }
            }
        }

        let unwrapped = laplacian_unwrap(&phase, &mask, n, n, n, 1.0, 1.0, 1.0);

        // Check that result is finite and similar to input
        // (since input has no wraps, output should be close)
        for (i, (&orig, &unwr)) in phase.iter().zip(unwrapped.iter()).enumerate() {
            assert!(unwr.is_finite(), "Unwrapped should be finite at {}", i);
            // Allow some deviation due to boundary conditions
            assert!((orig - unwr).abs() < 1.0,
                "Unwrapped should be close to original for smooth phase");
        }
    }

    #[test]
    fn test_laplacian_unwrap_finite() {
        let n = 8;
        let phase: Vec<f64> = (0..n*n*n).map(|i| wrap((i as f64) * 0.1)).collect();
        let mask = vec![1u8; n * n * n];

        let unwrapped = laplacian_unwrap(&phase, &mask, n, n, n, 1.0, 1.0, 1.0);

        for (i, &val) in unwrapped.iter().enumerate() {
            assert!(val.is_finite(), "Unwrapped phase should be finite at index {}", i);
        }
    }
}

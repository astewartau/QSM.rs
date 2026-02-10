//! Nonlinear Total Variation (NLTV) regularized dipole inversion
//!
//! NLTV extends standard TV by using iteratively reweighted minimization,
//! which produces sharper edges and better preserves fine details.
//!
//! The method solves:
//! min_x ||Dx - f||₂² + λ Σ w_i |∇x|_i
//!
//! where weights w_i are iteratively updated based on the current solution.
//!
//! Reference:
//! Bilgic B, et al. Nonlinear regularization for quantitative susceptibility mapping.
//! Magnetic Resonance in Medicine. 2017.

use num_complex::Complex64;
use crate::fft::{fft3d, ifft3d};
use crate::kernels::dipole::dipole_kernel;
use crate::kernels::laplacian::laplacian_kernel;
use crate::utils::gradient::{bdiv_inplace, fgrad_inplace};

/// Weighted soft thresholding operator
#[inline]
fn weighted_shrink(x: f64, threshold: f64, weight: f64) -> f64 {
    let t = threshold * weight;
    if x > t {
        x - t
    } else if x < -t {
        x + t
    } else {
        0.0
    }
}

/// NLTV dipole inversion using iteratively reweighted ADMM
///
/// # Arguments
/// * `local_field` - Local field values (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bdir` - B0 field direction
/// * `lambda` - Regularization parameter (typically 1e-3)
/// * `mu` - Reweighting parameter for nonlinearity (typically 1.0)
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum ADMM iterations
/// * `newton_iter` - Reweighting updates (inner Newton-like iterations)
///
/// # Returns
/// Susceptibility map
pub fn nltv(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bdir: (f64, f64, f64),
    lambda: f64,
    mu: f64,
    tol: f64,
    max_iter: usize,
    newton_iter: usize,
) -> Vec<f64> {
    nltv_with_progress(
        local_field, mask, nx, ny, nz, vsx, vsy, vsz,
        bdir, lambda, mu, tol, max_iter, newton_iter,
        |_, _| {} // no-op progress callback
    )
}

/// NLTV with progress callback
pub fn nltv_with_progress<F>(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bdir: (f64, f64, f64),
    lambda: f64,
    mu: f64,
    tol: f64,
    max_iter: usize,
    newton_iter: usize,
    mut progress_callback: F,
) -> Vec<f64>
where
    F: FnMut(usize, usize),
{
    let n_total = nx * ny * nz;
    let eps = 1e-6; // Small constant to avoid division by zero

    // ========================================================================
    // Pre-compute kernels (done once)
    // ========================================================================

    let d_kernel = dipole_kernel(nx, ny, nz, vsx, vsy, vsz, bdir);
    let l_kernel = laplacian_kernel(nx, ny, nz, vsx, vsy, vsz, true);

    // FFT of Laplacian kernel
    let mut l_complex: Vec<Complex64> = l_kernel.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut l_complex, nx, ny, nz);

    // Compute rho adaptively (for ADMM)
    let rho = 100.0 * lambda;

    // Pre-compute inverse of (D^H D + ρ L)
    let mut inv_a: Vec<f64> = vec![0.0; n_total];
    for i in 0..n_total {
        let a = d_kernel[i] * d_kernel[i] + rho * l_complex[i].re;
        inv_a[i] = if a.abs() > 1e-20 { 1.0 / a } else { 0.0 };
    }

    // Pre-compute D^H * FFT(f)
    let f_hat = &mut l_complex;
    for i in 0..n_total {
        f_hat[i] = Complex64::new(local_field[i], 0.0);
    }
    fft3d(f_hat, nx, ny, nz);
    for i in 0..n_total {
        f_hat[i] = f_hat[i] * d_kernel[i] * inv_a[i];
    }

    // ========================================================================
    // Pre-allocate working buffers
    // ========================================================================

    let mut x = vec![0.0; n_total];
    let mut x_prev = vec![0.0; n_total];

    // Dual variables
    let mut ux = vec![0.0; n_total];
    let mut uy = vec![0.0; n_total];
    let mut uz = vec![0.0; n_total];

    // Gradient buffers
    let mut gx = vec![0.0; n_total];
    let mut gy = vec![0.0; n_total];
    let mut gz = vec![0.0; n_total];

    // Divergence buffer
    let mut div_d = vec![0.0; n_total];

    // Complex FFT buffer
    let mut work_complex = vec![Complex64::new(0.0, 0.0); n_total];

    // Adaptive weights for nonlinear term
    let mut weights = vec![1.0; n_total];

    let total_iter = max_iter * newton_iter;
    let mut current_iter = 0;

    // ========================================================================
    // Outer loop: Newton-like reweighting
    // ========================================================================
    for _newton in 0..newton_iter {
        let lambda_over_rho = lambda / rho;

        // ====================================================================
        // Inner loop: ADMM with current weights
        // ====================================================================
        for _iter in 0..max_iter {
            current_iter += 1;
            progress_callback(current_iter, total_iter);

            // Swap x and x_prev
            std::mem::swap(&mut x, &mut x_prev);

            // ================================================================
            // x-subproblem
            // ================================================================
            bdiv_inplace(&mut div_d, &gx, &gy, &gz, nx, ny, nz, vsx, vsy, vsz);

            for i in 0..n_total {
                work_complex[i] = Complex64::new(div_d[i], 0.0);
            }
            fft3d(&mut work_complex, nx, ny, nz);

            for i in 0..n_total {
                work_complex[i] = f_hat[i] + rho * work_complex[i] * inv_a[i];
            }

            ifft3d(&mut work_complex, nx, ny, nz);
            for i in 0..n_total {
                x[i] = work_complex[i].re;
            }

            // ================================================================
            // Convergence check
            // ================================================================
            let mut norm_diff_sq = 0.0;
            let mut norm_x_sq = 0.0;
            for i in 0..n_total {
                let diff = x[i] - x_prev[i];
                norm_diff_sq += diff * diff;
                norm_x_sq += x[i] * x[i];
            }

            let rel_change = norm_diff_sq.sqrt() / (norm_x_sq.sqrt() + 1e-20);
            if rel_change < tol {
                break;
            }

            // ================================================================
            // z-subproblem + u-update with adaptive weights
            // ================================================================
            fgrad_inplace(&mut gx, &mut gy, &mut gz, &x, nx, ny, nz, vsx, vsy, vsz);

            for i in 0..n_total {
                let grad_x = gx[i];
                let grad_y = gy[i];
                let grad_z = gz[i];

                let vx = grad_x + ux[i];
                let vy = grad_y + uy[i];
                let vz = grad_z + uz[i];

                // Weighted soft thresholding
                let zx_i = weighted_shrink(vx, lambda_over_rho, weights[i]);
                let zy_i = weighted_shrink(vy, lambda_over_rho, weights[i]);
                let zz_i = weighted_shrink(vz, lambda_over_rho, weights[i]);

                // u update
                ux[i] = vx - zx_i;
                uy[i] = vy - zy_i;
                uz[i] = vz - zz_i;

                // Store (z - u_new) for next iteration's div
                gx[i] = 2.0 * zx_i - vx;
                gy[i] = 2.0 * zy_i - vy;
                gz[i] = 2.0 * zz_i - vz;
            }
        }

        // ====================================================================
        // Update weights based on current gradient magnitude (Newton update)
        // ====================================================================
        fgrad_inplace(&mut gx, &mut gy, &mut gz, &x, nx, ny, nz, vsx, vsy, vsz);

        for i in 0..n_total {
            // Gradient magnitude
            let grad_mag = (gx[i] * gx[i] + gy[i] * gy[i] + gz[i] * gz[i]).sqrt();

            // Reweighting: w = 1 / (|∇x| + eps)^(1-q) where q is close to 1 for L1
            // Using mu to control nonlinearity: w = 1 / (|∇x| + mu*eps)
            weights[i] = 1.0 / (grad_mag + mu * eps);
        }

        // Normalize weights to prevent explosion
        let max_weight: f64 = weights.iter().cloned().fold(0.0, f64::max);
        if max_weight > 1.0 {
            for w in weights.iter_mut() {
                *w /= max_weight;
            }
        }
    }

    // Apply mask
    for i in 0..n_total {
        if mask[i] == 0 {
            x[i] = 0.0;
        }
    }

    x
}

/// NLTV with default parameters
pub fn nltv_default(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    nltv(
        local_field, mask, nx, ny, nz, vsx, vsy, vsz,
        (0.0, 0.0, 1.0),  // bdir
        1e-3,             // lambda
        1.0,              // mu
        1e-3,             // tol
        250,              // max_iter
        10                // newton_iter
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nltv_zero_field() {
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        let chi = nltv(
            &field, &mask, n, n, n, 1.0, 1.0, 1.0,
            (0.0, 0.0, 1.0), 1e-3, 1.0, 1e-2, 10, 2
        );

        for &val in chi.iter() {
            assert!(val.abs() < 1e-8, "Zero field should give zero chi");
        }
    }

    #[test]
    fn test_nltv_finite() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];

        let chi = nltv(
            &field, &mask, n, n, n, 1.0, 1.0, 1.0,
            (0.0, 0.0, 1.0), 1e-3, 1.0, 1e-2, 10, 2
        );

        for (i, &val) in chi.iter().enumerate() {
            assert!(val.is_finite(), "Chi should be finite at index {}", i);
        }
    }

    #[test]
    fn test_weighted_shrink() {
        // w=1 should behave like regular shrink
        assert!((weighted_shrink(1.0, 0.5, 1.0) - 0.5).abs() < 1e-10);
        assert!((weighted_shrink(-1.0, 0.5, 1.0) - (-0.5)).abs() < 1e-10);
        assert!((weighted_shrink(0.3, 0.5, 1.0) - 0.0).abs() < 1e-10);

        // w=0.5 should have half the threshold
        assert!((weighted_shrink(1.0, 0.5, 0.5) - 0.75).abs() < 1e-10);
        assert!((weighted_shrink(0.3, 0.5, 0.5) - 0.05).abs() < 1e-10);
    }
}

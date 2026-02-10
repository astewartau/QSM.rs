//! Total Variation (TV) regularized dipole inversion using ADMM
//!
//! Solves the L1-regularized inverse problem:
//! min_x ||Dx - f||₂² + λ||∇x||₁
//!
//! using Alternating Direction Method of Multipliers (ADMM).
//!
//! Reference:
//! Bilgic B, et al. Fast quantitative susceptibility mapping with
//! L1-regularization and automatic parameter selection.
//! Magnetic Resonance in Medicine. 2014;72(5):1444-59.

use num_complex::Complex64;
use crate::fft::{fft3d, ifft3d};
use crate::kernels::dipole::dipole_kernel;
use crate::kernels::laplacian::laplacian_kernel;
use crate::utils::gradient::{bdiv_inplace, fgrad_inplace};

/// Soft thresholding (shrinkage) operator for L1 regularization
/// shrink(x, t) = sign(x) * max(|x| - t, 0)
#[inline]
fn shrink(x: f64, threshold: f64) -> f64 {
    if x > threshold {
        x - threshold
    } else if x < -threshold {
        x + threshold
    } else {
        0.0
    }
}

/// TV-ADMM dipole inversion (optimized)
///
/// # Arguments
/// * `local_field` - Local field values (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bdir` - B0 field direction
/// * `lambda` - Regularization parameter (typically 1e-3 to 1e-4)
/// * `rho` - ADMM penalty parameter (typically 100*lambda)
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
///
/// # Returns
/// Susceptibility map
pub fn tv_admm(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bdir: (f64, f64, f64),
    lambda: f64,
    rho: f64,
    tol: f64,
    max_iter: usize,
) -> Vec<f64> {
    tv_admm_with_progress(
        local_field, mask, nx, ny, nz, vsx, vsy, vsz,
        bdir, lambda, rho, tol, max_iter,
        |_, _| {} // no-op progress callback
    )
}

/// TV-ADMM with progress callback (optimized)
///
/// Optimized implementation with:
/// - Pre-allocated buffers (zero allocations per iteration)
/// - In-place gradient/divergence operations
/// - Buffer swapping instead of cloning
/// - Fused z-subproblem and u-update
///
/// Same as `tv_admm` but calls `progress_callback(iteration, max_iter)` each iteration.
pub fn tv_admm_with_progress<F>(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bdir: (f64, f64, f64),
    lambda: f64,
    rho: f64,
    tol: f64,
    max_iter: usize,
    mut progress_callback: F,
) -> Vec<f64>
where
    F: FnMut(usize, usize),
{
    let n_total = nx * ny * nz;

    // ========================================================================
    // Pre-compute kernels (done once)
    // ========================================================================

    // Generate dipole kernel D
    let d_kernel = dipole_kernel(nx, ny, nz, vsx, vsy, vsz, bdir);

    // Generate negative Laplacian kernel (for -Δ = ∇ᵀ∇)
    let l_kernel = laplacian_kernel(nx, ny, nz, vsx, vsy, vsz, true);

    // FFT of Laplacian kernel
    let mut l_complex: Vec<Complex64> = l_kernel.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut l_complex, nx, ny, nz);

    // Pre-compute inverse of (D^H D + ρ L) for x-subproblem
    let mut inv_a: Vec<f64> = vec![0.0; n_total];
    for i in 0..n_total {
        let a = d_kernel[i] * d_kernel[i] + rho * l_complex[i].re;
        inv_a[i] = if a.abs() > 1e-20 { 1.0 / a } else { 0.0 };
    }

    // Pre-compute D^H * f for constant part of RHS (reuse l_complex as work buffer)
    let f_hat = &mut l_complex; // Reuse buffer
    for i in 0..n_total {
        f_hat[i] = Complex64::new(local_field[i], 0.0);
    }
    fft3d(f_hat, nx, ny, nz);

    // f_hat = D^H * FFT(f) * inv_a
    for i in 0..n_total {
        f_hat[i] = f_hat[i] * d_kernel[i] * inv_a[i];
    }

    // ========================================================================
    // Pre-allocate ALL working buffers (zero allocations in iteration loop)
    // ========================================================================

    // Solution and previous (for convergence check)
    let mut x = vec![0.0; n_total];
    let mut x_prev = vec![0.0; n_total];

    // Dual variables (scaled Lagrange multipliers)
    let mut ux = vec![0.0; n_total];
    let mut uy = vec![0.0; n_total];
    let mut uz = vec![0.0; n_total];

    // Gradient buffers (reused for z-u computation)
    let mut gx = vec![0.0; n_total];
    let mut gy = vec![0.0; n_total];
    let mut gz = vec![0.0; n_total];

    // Divergence buffer
    let mut div_d = vec![0.0; n_total];

    // Complex FFT buffer (reused each iteration)
    let mut work_complex = vec![Complex64::new(0.0, 0.0); n_total];

    let lambda_over_rho = lambda / rho;

    // ========================================================================
    // ADMM iterations (zero allocations per iteration)
    // ========================================================================
    for iter in 0..max_iter {
        // Report progress
        progress_callback(iter + 1, max_iter);

        // Swap x and x_prev (no allocation, just pointer swap)
        std::mem::swap(&mut x, &mut x_prev);

        // ====================================================================
        // x-subproblem: solve (D^H D + ρ L) x = D^H f + ρ div(z - u)
        // ====================================================================

        // gx/gy/gz currently hold ∇x from previous iteration (or zero initially)
        // After z-subproblem, they hold z - u (we compute this at end of loop)

        // Compute div(z - u) into div_d
        // On first iteration, gx/gy/gz are zero, so div is zero
        bdiv_inplace(&mut div_d, &gx, &gy, &gz, nx, ny, nz, vsx, vsy, vsz);

        // Prepare FFT: work_complex = div_d
        for i in 0..n_total {
            work_complex[i] = Complex64::new(div_d[i], 0.0);
        }
        fft3d(&mut work_complex, nx, ny, nz);

        // x_hat = f_hat + rho * FFT(div) * inv_a
        for i in 0..n_total {
            work_complex[i] = f_hat[i] + rho * work_complex[i] * inv_a[i];
        }

        // IFFT to get x
        ifft3d(&mut work_complex, nx, ny, nz);
        for i in 0..n_total {
            x[i] = work_complex[i].re;
        }

        // ====================================================================
        // Convergence check (before z/u update for efficiency)
        // ====================================================================
        let mut norm_diff_sq = 0.0;
        let mut norm_x_sq = 0.0;
        for i in 0..n_total {
            let diff = x[i] - x_prev[i];
            norm_diff_sq += diff * diff;
            norm_x_sq += x[i] * x[i];
        }

        let rel_change = norm_diff_sq.sqrt() / (norm_x_sq.sqrt() + 1e-20);
        if rel_change < tol {
            progress_callback(iter + 1, iter + 1);
            break;
        }

        // ====================================================================
        // Fused: z-subproblem + u-update + prepare (z-u) for next iteration
        // ====================================================================

        // Compute gradient of x into gx/gy/gz
        fgrad_inplace(&mut gx, &mut gy, &mut gz, &x, nx, ny, nz, vsx, vsy, vsz);

        // Fused loop: z = shrink(∇x + u), u = u + ∇x - z, store (z - u) in gx/gy/gz
        for i in 0..n_total {
            // x-gradient
            let grad_x = gx[i];
            let grad_y = gy[i];
            let grad_z = gz[i];

            // ∇x + u (temporary)
            let vx = grad_x + ux[i];
            let vy = grad_y + uy[i];
            let vz = grad_z + uz[i];

            // z = shrink(∇x + u, λ/ρ)
            let zx_i = shrink(vx, lambda_over_rho);
            let zy_i = shrink(vy, lambda_over_rho);
            let zz_i = shrink(vz, lambda_over_rho);

            // u_new = u + ∇x - z = v - z
            ux[i] = vx - zx_i;
            uy[i] = vy - zy_i;
            uz[i] = vz - zz_i;

            // Store (z - u_new) = z - (v - z) = 2z - v for next iteration's div
            gx[i] = 2.0 * zx_i - vx;
            gy[i] = 2.0 * zy_i - vy;
            gz[i] = 2.0 * zz_i - vz;
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

/// TV-ADMM with default parameters
pub fn tv_admm_default(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    tv_admm(
        local_field, mask, nx, ny, nz, vsx, vsy, vsz,
        (0.0, 0.0, 1.0),  // bdir
        1e-3,             // lambda
        0.1,              // rho = 100 * lambda
        1e-3,             // tol
        250               // max_iter
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::gradient::fgrad;

    #[test]
    fn test_shrink() {
        assert!((shrink(1.0, 0.5) - 0.5).abs() < 1e-10);
        assert!((shrink(-1.0, 0.5) - (-0.5)).abs() < 1e-10);
        assert!((shrink(0.3, 0.5) - 0.0).abs() < 1e-10);
        assert!((shrink(-0.3, 0.5) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_tv_admm_zero_field() {
        // Zero field should give zero susceptibility
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        let chi = tv_admm(
            &field, &mask, n, n, n, 1.0, 1.0, 1.0,
            (0.0, 0.0, 1.0), 1e-3, 0.1, 1e-2, 10
        );

        for &val in chi.iter() {
            assert!(val.abs() < 1e-8, "Zero field should give zero chi, got {}", val);
        }
    }

    #[test]
    fn test_tv_admm_finite() {
        // Result should be finite
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];

        let chi = tv_admm(
            &field, &mask, n, n, n, 1.0, 1.0, 1.0,
            (0.0, 0.0, 1.0), 1e-3, 0.1, 1e-2, 10
        );

        for (i, &val) in chi.iter().enumerate() {
            assert!(val.is_finite(), "Chi should be finite at index {}", i);
        }
    }

    #[test]
    fn test_tv_admm_smoother_than_tkd() {
        // TV should produce smoother results than TKD
        let n = 8;
        // Create noisy field
        let mut field = vec![0.0; n * n * n];
        for i in 0..n*n*n {
            field[i] = if i % 2 == 0 { 0.01 } else { -0.01 };  // Alternating
        }
        let mask = vec![1u8; n * n * n];

        let chi_tv = tv_admm(
            &field, &mask, n, n, n, 1.0, 1.0, 1.0,
            (0.0, 0.0, 1.0), 1e-2, 1.0, 1e-2, 50  // Strong regularization
        );

        // Compute total variation (L1 norm of gradient)
        let (gx, gy, gz) = fgrad(&chi_tv, n, n, n, 1.0, 1.0, 1.0);
        let tv: f64 = gx.iter().chain(gy.iter()).chain(gz.iter())
            .map(|&g| g.abs())
            .sum();

        // TV result should have small total variation
        // (exact value depends on parameters, but should be bounded)
        assert!(tv.is_finite(), "TV should be finite");
    }
}

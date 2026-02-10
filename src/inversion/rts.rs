//! Rapid Two-Step (RTS) dipole inversion
//!
//! Two-step approach that combines:
//! 1. LSMR for well-conditioned k-space regions
//! 2. TV regularization for ill-conditioned regions
//!
//! Reference:
//! Kames C, Wiggermann V, Rauscher A. Rapid two-step dipole inversion for
//! susceptibility mapping with sparsity priors.
//! Neuroimage. 2018 Feb 15;167:276-83.

use num_complex::Complex64;
use crate::fft::{fft3d, ifft3d};
use crate::kernels::dipole::dipole_kernel;
use crate::kernels::laplacian::laplacian_kernel;
use crate::utils::gradient::{fgrad_inplace, bdiv_inplace};

/// Soft thresholding (shrinkage) operator
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

/// RTS dipole inversion (optimized)
///
/// # Arguments
/// * `local_field` - Local field values (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bdir` - B0 field direction
/// * `delta` - Threshold for ill-conditioned region (typically 0.15)
/// * `mu` - Regularization parameter for well-conditioned region (typically 1e5)
/// * `rho` - ADMM penalty parameter (typically 10)
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum ADMM iterations
/// * `lsmr_iter` - LSMR iterations for step 1 (typically 4)
///
/// # Returns
/// Susceptibility map
pub fn rts(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bdir: (f64, f64, f64),
    delta: f64,
    mu: f64,
    rho: f64,
    tol: f64,
    max_iter: usize,
    lsmr_iter: usize,
) -> Vec<f64> {
    rts_with_progress(
        local_field, mask, nx, ny, nz, vsx, vsy, vsz,
        bdir, delta, mu, rho, tol, max_iter, lsmr_iter,
        |_, _| {} // no-op progress callback
    )
}

/// RTS with progress callback (optimized)
///
/// Optimized implementation with:
/// - Pre-allocated buffers (zero allocations per iteration)
/// - In-place gradient/divergence operations
/// - Buffer swapping instead of cloning
/// - Fused z-subproblem and u-update
///
/// Same as `rts` but calls `progress_callback(iteration, max_iter)` each iteration.
pub fn rts_with_progress<F>(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bdir: (f64, f64, f64),
    delta: f64,
    mu: f64,
    rho: f64,
    tol: f64,
    max_iter: usize,
    lsmr_iter: usize,
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

    // Generate negative Laplacian kernel
    let l_kernel = laplacian_kernel(nx, ny, nz, vsx, vsy, vsz, true);

    // FFT of Laplacian kernel (reuse buffer for other purposes later)
    let mut work_complex: Vec<Complex64> = l_kernel.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut work_complex, nx, ny, nz);

    // Compute well-conditioned mask M and inverse operator iA
    let mut m_mask: Vec<f64> = vec![0.0; n_total];
    let mut inv_a: Vec<f64> = vec![0.0; n_total];

    for i in 0..n_total {
        let l_fft_i = work_complex[i].re;
        if d_kernel[i].abs() > delta {
            m_mask[i] = mu;
        }
        let a = m_mask[i] + rho * l_fft_i;
        if a.abs() > 1e-20 {
            inv_a[i] = rho / a;
        }
    }

    // ========================================================================
    // Step 1: Well-conditioned k-space (simplified LSMR)
    // ========================================================================

    // FFT of field (reuse work_complex)
    for i in 0..n_total {
        work_complex[i] = Complex64::new(local_field[i], 0.0);
    }
    fft3d(&mut work_complex, nx, ny, nz);

    // Store field_fft for LSMR iterations
    let field_fft: Vec<Complex64> = work_complex.clone();

    // Initial estimate: chi = D * f / (D^2 + epsilon) for well-conditioned
    // Stored in work_complex
    for i in 0..n_total {
        let d = d_kernel[i];
        if d.abs() > delta {
            work_complex[i] = field_fft[i] * d / (d * d + 1e-6);
        } else {
            work_complex[i] = Complex64::new(0.0, 0.0);
        }
    }

    // Simple iterative refinement for well-conditioned region
    // Use a temporary buffer for residual
    let mut residual = vec![Complex64::new(0.0, 0.0); n_total];
    for _ in 0..lsmr_iter {
        // residual = f - D * chi
        for i in 0..n_total {
            residual[i] = field_fft[i] - work_complex[i] * d_kernel[i];
        }

        // update chi for well-conditioned region
        for i in 0..n_total {
            let d = d_kernel[i];
            if d.abs() > delta {
                work_complex[i] += residual[i] * d / (d * d + 1e-6);
            }
        }
    }

    // Transform to spatial domain
    ifft3d(&mut work_complex, nx, ny, nz);

    // Initialize x and apply mask
    let mut x = vec![0.0; n_total];
    for i in 0..n_total {
        x[i] = if mask[i] != 0 { work_complex[i].re } else { 0.0 };
    }

    // ========================================================================
    // Pre-compute constant part of RHS for ADMM
    // ========================================================================

    // F_hat = inv_a * M * FFT(x) / rho
    for i in 0..n_total {
        work_complex[i] = Complex64::new(x[i], 0.0);
    }
    fft3d(&mut work_complex, nx, ny, nz);

    let mut f_hat: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n_total];
    for i in 0..n_total {
        if m_mask[i].abs() > 1e-20 && inv_a[i].abs() > 1e-20 {
            f_hat[i] = work_complex[i] * (m_mask[i] / rho) * inv_a[i];
        }
    }

    // ========================================================================
    // Pre-allocate ALL working buffers for ADMM (zero allocations in loop)
    // ========================================================================

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
    let mut div_v = vec![0.0; n_total];

    let inv_rho = 1.0 / rho;

    // ========================================================================
    // Step 2: ADMM iterations (zero allocations per iteration)
    // ========================================================================

    for iter in 0..max_iter {
        progress_callback(iter + 1, max_iter);

        // Swap x and x_prev (no allocation)
        std::mem::swap(&mut x, &mut x_prev);

        // ====================================================================
        // x-subproblem: (M + ρL)x = F + ρ∇ᵀ(z - u)
        // ====================================================================

        // gx/gy/gz hold (z - u) from previous iteration (or zero initially)
        bdiv_inplace(&mut div_v, &gx, &gy, &gz, nx, ny, nz, vsx, vsy, vsz);

        // Prepare FFT
        for i in 0..n_total {
            work_complex[i] = Complex64::new(div_v[i], 0.0);
        }
        fft3d(&mut work_complex, nx, ny, nz);

        // x_hat = f_hat + inv_a * div_hat
        for i in 0..n_total {
            work_complex[i] = f_hat[i] + work_complex[i] * inv_a[i];
        }

        // IFFT to get x
        ifft3d(&mut work_complex, nx, ny, nz);
        for i in 0..n_total {
            x[i] = work_complex[i].re;
        }

        // ====================================================================
        // Convergence check
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

        fgrad_inplace(&mut gx, &mut gy, &mut gz, &x, nx, ny, nz, vsx, vsy, vsz);

        for i in 0..n_total {
            let grad_x = gx[i];
            let grad_y = gy[i];
            let grad_z = gz[i];

            // ∇x + u
            let vx = grad_x + ux[i];
            let vy = grad_y + uy[i];
            let vz = grad_z + uz[i];

            // z = shrink(∇x + u, 1/ρ)
            let zx_i = shrink(vx, inv_rho);
            let zy_i = shrink(vy, inv_rho);
            let zz_i = shrink(vz, inv_rho);

            // u_new = u + ∇x - z = v - z
            ux[i] = vx - zx_i;
            uy[i] = vy - zy_i;
            uz[i] = vz - zz_i;

            // Store (z - u_new) = 2z - v for next iteration
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

/// RTS with default parameters
pub fn rts_default(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    rts(
        local_field, mask, nx, ny, nz, vsx, vsy, vsz,
        (0.0, 0.0, 1.0),  // bdir
        0.15,              // delta
        1e5,               // mu
        10.0,              // rho
        1e-2,              // tol
        20,                // max_iter
        4                  // lsmr_iter
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rts_zero_field() {
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        let chi = rts(
            &field, &mask, n, n, n, 1.0, 1.0, 1.0,
            (0.0, 0.0, 1.0), 0.15, 1e5, 10.0, 1e-2, 5, 2
        );

        for &val in chi.iter() {
            assert!(val.abs() < 1e-6, "Zero field should give near-zero chi");
        }
    }

    #[test]
    fn test_rts_finite() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];

        let chi = rts(
            &field, &mask, n, n, n, 1.0, 1.0, 1.0,
            (0.0, 0.0, 1.0), 0.15, 1e5, 10.0, 1e-2, 5, 2
        );

        for (i, &val) in chi.iter().enumerate() {
            assert!(val.is_finite(), "Chi should be finite at index {}", i);
        }
    }

    #[test]
    fn test_rts_mask() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mut mask = vec![1u8; n * n * n];
        // Zero out some mask values
        mask[0] = 0;
        mask[10] = 0;

        let chi = rts(
            &field, &mask, n, n, n, 1.0, 1.0, 1.0,
            (0.0, 0.0, 1.0), 0.15, 1e5, 10.0, 1e-2, 5, 2
        );

        assert_eq!(chi[0], 0.0, "Masked voxel should be zero");
        assert_eq!(chi[10], 0.0, "Masked voxel should be zero");
    }
}

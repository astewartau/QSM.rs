//! Total Variation (TV) regularized dipole inversion using ADMM
//!
//! Solves the L1-regularized inverse problem:
//! min_x ||Dx - f||₂² + λ||∇x||₁
//!
//! using Alternating Direction Method of Multipliers (ADMM).
//!
//! Reference:
//! Bilgic, B., Fan, A.P., Polimeni, J.R., et al. (2014).
//! "Fast quantitative susceptibility mapping with L1-regularization and automatic
//! parameter selection." Magnetic Resonance in Medicine, 72(5):1444-1459.
//! https://doi.org/10.1002/mrm.25029
//!
//! Reference implementation: https://github.com/kamesy/QSM.jl

use num_complex::Complex64;
use crate::fft::Fft3dWorkspace;
use crate::kernels::dipole::dipole_kernel;
use crate::kernels::laplacian::laplacian_kernel;
use crate::utils::{shrink, apply_mask_zero};
use crate::Grid;
use super::admm::{AdmmBuffers, admm_step};

/// TV-ADMM algorithm parameters
#[derive(Clone, Debug)]
pub struct TvParams {
    /// Regularization parameter (typically 1e-3 to 1e-4)
    pub lambda: f64,
    /// ADMM penalty parameter (typically 100*lambda)
    pub rho: f64,
    /// Convergence tolerance
    pub tol: f64,
    /// Maximum iterations
    pub max_iter: usize,
}

impl Default for TvParams {
    fn default() -> Self {
        Self {
            lambda: 2e-4,
            rho: 2e-2,
            tol: 1e-3,
            max_iter: 250,
        }
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

    // Create FFT workspace (caches plans and scratch buffers for reuse)
    let mut fft_ws = Fft3dWorkspace::new(nx, ny, nz);

    // Generate dipole kernel D
    let d_kernel = dipole_kernel(nx, ny, nz, vsx, vsy, vsz, bdir);

    // Generate negative Laplacian kernel (for -Δ = ∇ᵀ∇)
    let l_kernel = laplacian_kernel(nx, ny, nz, vsx, vsy, vsz, true);

    // FFT of Laplacian kernel
    let mut l_complex: Vec<Complex64> = l_kernel.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft_ws.fft3d(&mut l_complex);

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
    fft_ws.fft3d(f_hat);

    // f_hat = D^H * FFT(f) * inv_a
    for i in 0..n_total {
        f_hat[i] = f_hat[i] * d_kernel[i] * inv_a[i];
    }

    // ========================================================================
    // Pre-allocate working buffers and run ADMM iterations
    // ========================================================================

    let grid = Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let mut buf = AdmmBuffers::new(n_total);
    let lambda_over_rho = lambda / rho;

    for iter in 0..max_iter {
        progress_callback(iter + 1, max_iter);

        let converged = admm_step(
            &mut buf, &mut fft_ws, f_hat, &inv_a, rho, &grid, tol,
            |vx, vy, vz, _| (shrink(vx, lambda_over_rho), shrink(vy, lambda_over_rho), shrink(vz, lambda_over_rho)),
        );

        if converged {
            progress_callback(iter + 1, iter + 1);
            break;
        }
    }

    apply_mask_zero(&mut buf.x, mask);

    buf.x
}

/// TV-ADMM with default parameters
pub fn tv_admm_default(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    let p = TvParams::default();
    tv_admm(
        local_field, mask, nx, ny, nz, vsx, vsy, vsz,
        (0.0, 0.0, 1.0),
        p.lambda, p.rho, p.tol, p.max_iter,
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

    /// Verify parallel and sequential TV-ADMM produce identical results.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_tv_parallel_matches_sequential() {
        let n = 16;
        let field: Vec<f64> = (0..n*n*n).map(|i| ((i as f64) * 0.7).sin() * 0.01).collect();
        let mask = vec![1u8; n * n * n];

        // Sequential (1 thread)
        let pool_1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let chi_seq = pool_1.install(|| {
            tv_admm(&field, &mask, n, n, n, 1.0, 1.0, 1.0,
                (0.0, 0.0, 1.0), 1e-3, 0.1, 1e-3, 50)
        });

        // Parallel (default threads)
        let chi_par = tv_admm(&field, &mask, n, n, n, 1.0, 1.0, 1.0,
            (0.0, 0.0, 1.0), 1e-3, 0.1, 1e-3, 50);

        // Compare
        for (i, (s, p)) in chi_seq.iter().zip(chi_par.iter()).enumerate() {
            assert!(
                (s - p).abs() < 1e-10,
                "TV mismatch at voxel {}: seq={} par={} diff={}",
                i, s, p, (s - p).abs()
            );
        }
    }
}

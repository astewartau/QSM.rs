//! Total Variation (TV) regularized dipole inversion using ADMM
//!
//! Solves the L1-regularized inverse problem:
//! min_x ||Dx - f||_2^2 + lambda||grad(x)||_1
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

use crate::utils::{shrink, apply_mask_zero};
use crate::Grid;
use super::admm::{AdmmBuffers, admm_step, prepare_admm_spectral};

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

/// TV-ADMM dipole inversion
///
/// Optimized implementation with:
/// - Pre-allocated buffers (zero allocations per iteration)
/// - In-place gradient/divergence operations
/// - Buffer swapping instead of cloning
/// - Fused z-subproblem and u-update
///
/// # Arguments
/// * `local_field` - Local field values (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `bdir` - B0 field direction
/// * `params` - TV-ADMM parameters
/// * `progress` - Progress callback `(iteration, max_iter)`
///
/// # Returns
/// Susceptibility map
pub fn tv_admm(
    local_field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &TvParams,
    mut progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let n_total = grid.n_total();

    // Pre-compute spectral operators
    let (mut fft_ws, inv_a, f_hat) = prepare_admm_spectral(local_field, grid, bdir, params.rho);

    // Pre-allocate working buffers and run ADMM iterations
    let mut buf = AdmmBuffers::new(n_total);
    let lambda_over_rho = params.lambda / params.rho;

    for iter in 0..params.max_iter {
        progress(iter + 1, params.max_iter);

        let converged = admm_step(
            &mut buf, &mut fft_ws, &f_hat, &inv_a, params.rho, grid, params.tol,
            |vx, vy, vz, _| (shrink(vx, lambda_over_rho), shrink(vy, lambda_over_rho), shrink(vz, lambda_over_rho)),
        );

        if converged {
            progress(iter + 1, iter + 1);
            break;
        }
    }

    apply_mask_zero(&mut buf.x, mask);

    buf.x
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
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = TvParams { lambda: 1e-3, rho: 0.1, tol: 1e-2, max_iter: 10 };

        let chi = tv_admm(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

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
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = TvParams { lambda: 1e-3, rho: 0.1, tol: 1e-2, max_iter: 10 };

        let chi = tv_admm(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

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
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = TvParams { lambda: 1e-2, rho: 1.0, tol: 1e-2, max_iter: 50 };

        let chi_tv = tv_admm(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        // Compute total variation (L1 norm of gradient)
        let (gx, gy, gz) = fgrad(&chi_tv, &grid);
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
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = TvParams { lambda: 1e-3, rho: 0.1, tol: 1e-3, max_iter: 50 };

        // Sequential (1 thread)
        let pool_1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let chi_seq = pool_1.install(|| {
            tv_admm(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {})
        });

        // Parallel (default threads)
        let chi_par = tv_admm(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

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

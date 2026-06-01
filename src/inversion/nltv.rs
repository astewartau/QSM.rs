//! Nonlinear Total Variation (NLTV) regularized dipole inversion
//!
//! NLTV extends standard TV by using iteratively reweighted minimization,
//! which produces sharper edges and better preserves fine details.
//!
//! The method solves:
//! min_x ||Dx - f||_2^2 + lambda * sum w_i |grad(x)|_i
//!
//! where weights w_i are iteratively updated based on the current solution.
//!
//! Reference:
//! Kames, C., Wiggermann, V., Rauscher, A. (2018).
//! "Rapid two-step dipole inversion for susceptibility mapping with sparsity priors."
//! NeuroImage, 167:276-283. https://doi.org/10.1016/j.neuroimage.2017.11.018
//!
//! Reference implementation: https://github.com/kamesy/QSM.jl

use crate::utils::gradient::fgrad_inplace;
use crate::utils::{weighted_shrink, apply_mask_zero};
use crate::Grid;
use super::admm::{AdmmBuffers, admm_step, prepare_admm_spectral};

/// NLTV algorithm parameters
#[derive(Clone, Debug)]
pub struct NltvParams {
    /// Regularization parameter
    pub lambda: f64,
    /// Penalty parameter
    pub mu: f64,
    /// Convergence tolerance
    pub tol: f64,
    /// Maximum ADMM iterations
    pub max_iter: usize,
    /// Newton iterations for weight update
    pub newton_iter: usize,
}

impl Default for NltvParams {
    fn default() -> Self {
        Self {
            lambda: 1e-3,
            mu: 1.0,
            tol: 1e-3,
            max_iter: 250,
            newton_iter: 10,
        }
    }
}

/// NLTV dipole inversion using iteratively reweighted ADMM
///
/// # Arguments
/// * `local_field` - Local field values (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `bdir` - B0 field direction
/// * `params` - NLTV parameters
/// * `progress` - Progress callback `(iteration, total_iterations)`
///
/// # Returns
/// Susceptibility map
pub fn nltv(
    local_field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &NltvParams,
    mut progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let n_total = grid.n_total();
    let eps = 1e-6; // Small constant to avoid division by zero

    // Compute rho adaptively (for ADMM)
    let rho = 100.0 * params.lambda;

    // Pre-compute spectral operators
    let (mut fft_ws, inv_a, f_hat) = prepare_admm_spectral(local_field, grid, bdir, rho);

    // Pre-allocate working buffers and run ADMM iterations
    let mut buf = AdmmBuffers::new(n_total);
    let mut weights = vec![1.0; n_total];

    let total_iter = params.max_iter * params.newton_iter;
    let mut current_iter = 0;

    // Outer loop: Newton-like reweighting
    for _newton in 0..params.newton_iter {
        let lambda_over_rho = params.lambda / rho;

        // Inner loop: ADMM with current weights
        for _iter in 0..params.max_iter {
            current_iter += 1;
            progress(current_iter, total_iter);

            let converged = admm_step(
                &mut buf, &mut fft_ws, &f_hat, &inv_a, rho, grid, params.tol,
                |vx, vy, vz, i| (
                    weighted_shrink(vx, lambda_over_rho, weights[i]),
                    weighted_shrink(vy, lambda_over_rho, weights[i]),
                    weighted_shrink(vz, lambda_over_rho, weights[i]),
                ),
            );

            if converged {
                break;
            }
        }

        // Update weights based on current gradient magnitude
        fgrad_inplace(&mut buf.gx, &mut buf.gy, &mut buf.gz, &buf.x, grid);

        for i in 0..n_total {
            let grad_mag = (buf.gx[i] * buf.gx[i] + buf.gy[i] * buf.gy[i] + buf.gz[i] * buf.gz[i]).sqrt();
            weights[i] = 1.0 / (grad_mag + params.mu * eps);
        }

        // Normalize weights to prevent explosion
        let max_weight: f64 = weights.iter().cloned().fold(0.0, f64::max);
        if max_weight > 1.0 {
            for w in weights.iter_mut() {
                *w /= max_weight;
            }
        }
    }

    apply_mask_zero(&mut buf.x, mask);

    buf.x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::weighted_shrink;

    #[test]
    fn test_nltv_zero_field() {
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = NltvParams { lambda: 1e-3, mu: 1.0, tol: 1e-2, max_iter: 10, newton_iter: 2 };

        let chi = nltv(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        for &val in chi.iter() {
            assert!(val.abs() < 1e-8, "Zero field should give zero chi");
        }
    }

    #[test]
    fn test_nltv_finite() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = NltvParams { lambda: 1e-3, mu: 1.0, tol: 1e-2, max_iter: 10, newton_iter: 2 };

        let chi = nltv(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

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

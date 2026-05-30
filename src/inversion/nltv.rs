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
//! Kames, C., Wiggermann, V., Rauscher, A. (2018).
//! "Rapid two-step dipole inversion for susceptibility mapping with sparsity priors."
//! NeuroImage, 167:276-283. https://doi.org/10.1016/j.neuroimage.2017.11.018
//!
//! Reference implementation: https://github.com/kamesy/QSM.jl

use num_complex::Complex64;
use crate::fft::Fft3dWorkspace;
use crate::kernels::dipole::dipole_kernel;
use crate::kernels::laplacian::laplacian_kernel;
use crate::utils::gradient::fgrad_inplace;
use crate::utils::{weighted_shrink, apply_mask_zero};
use crate::Grid;
use super::admm::{AdmmBuffers, admm_step};

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

    // Create FFT workspace (caches plans and scratch buffers for reuse)
    let mut fft_ws = Fft3dWorkspace::new(nx, ny, nz);

    let d_kernel = dipole_kernel(nx, ny, nz, vsx, vsy, vsz, bdir);
    let l_kernel = laplacian_kernel(nx, ny, nz, vsx, vsy, vsz, true);

    // FFT of Laplacian kernel
    let mut l_complex: Vec<Complex64> = l_kernel.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft_ws.fft3d(&mut l_complex);

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
    fft_ws.fft3d(f_hat);
    for i in 0..n_total {
        f_hat[i] = f_hat[i] * d_kernel[i] * inv_a[i];
    }

    // ========================================================================
    // Pre-allocate working buffers and run ADMM iterations
    // ========================================================================

    let grid = Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let mut buf = AdmmBuffers::new(n_total);
    let mut weights = vec![1.0; n_total];

    let total_iter = max_iter * newton_iter;
    let mut current_iter = 0;

    // Outer loop: Newton-like reweighting
    for _newton in 0..newton_iter {
        let lambda_over_rho = lambda / rho;

        // Inner loop: ADMM with current weights
        for _iter in 0..max_iter {
            current_iter += 1;
            progress_callback(current_iter, total_iter);

            let converged = admm_step(
                &mut buf, &mut fft_ws, f_hat, &inv_a, rho, &grid, tol,
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
        fgrad_inplace(&mut buf.gx, &mut buf.gy, &mut buf.gz, &buf.x, nx, ny, nz, vsx, vsy, vsz);

        for i in 0..n_total {
            let grad_mag = (buf.gx[i] * buf.gx[i] + buf.gy[i] * buf.gy[i] + buf.gz[i] * buf.gz[i]).sqrt();
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

    apply_mask_zero(&mut buf.x, mask);

    buf.x
}

/// NLTV with default parameters (matches QSM.jl nltv.jl defaults)
pub fn nltv_default(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    nltv(
        local_field, mask, nx, ny, nz, vsx, vsy, vsz,
        (0.0, 0.0, 1.0),  // bdir
        1e-3,             // lambda (QSM.jl default)
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

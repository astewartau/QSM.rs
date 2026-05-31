//! Rapid Two-Step (RTS) dipole inversion
//!
//! Two-step approach that combines:
//! 1. LSMR for well-conditioned k-space regions
//! 2. TV regularization for ill-conditioned regions
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
use crate::utils::{shrink, apply_mask_zero};
use crate::Grid;
use super::admm::{AdmmBuffers, admm_step};

/// RTS algorithm parameters
#[derive(Clone, Debug)]
pub struct RtsParams {
    /// Threshold for ill-conditioned region (typically 0.15)
    pub delta: f64,
    /// Regularization parameter for well-conditioned region (typically 1e5)
    pub mu: f64,
    /// ADMM penalty parameter (typically 10)
    pub rho: f64,
    /// Convergence tolerance
    pub tol: f64,
    /// Maximum ADMM iterations
    pub max_iter: usize,
    /// LSMR iterations for step 1 (typically 4)
    pub lsmr_iter: usize,
}

impl Default for RtsParams {
    fn default() -> Self {
        Self {
            delta: 0.15,
            mu: 1e5,
            rho: 10.0,
            tol: 1e-2,
            max_iter: 20,
            lsmr_iter: 4,
        }
    }
}

/// RTS dipole inversion
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
/// * `params` - RTS parameters
/// * `progress` - Progress callback `(iteration, max_iter)`
///
/// # Returns
/// Susceptibility map
pub fn rts(
    local_field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &RtsParams,
    mut progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let n_total = grid.n_total();

    // ========================================================================
    // Pre-compute kernels (done once)
    // ========================================================================

    // Create FFT workspace (caches plans and scratch buffers for reuse)
    let mut fft_ws = Fft3dWorkspace::new(nx, ny, nz);

    // Generate dipole kernel D
    let d_kernel = dipole_kernel(grid, bdir);

    // Generate negative Laplacian kernel
    let l_kernel = laplacian_kernel(grid, true);

    // FFT of Laplacian kernel (reuse buffer for other purposes later)
    let mut work_complex: Vec<Complex64> = l_kernel.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft_ws.fft3d(&mut work_complex);

    // Compute well-conditioned mask M and inverse operator iA
    let mut m_mask: Vec<f64> = vec![0.0; n_total];
    let mut inv_a: Vec<f64> = vec![0.0; n_total];

    for i in 0..n_total {
        let l_fft_i = work_complex[i].re;
        if d_kernel[i].abs() > params.delta {
            m_mask[i] = params.mu;
        }
        let a = m_mask[i] + params.rho * l_fft_i;
        if a.abs() > 1e-20 {
            inv_a[i] = params.rho / a;
        }
    }

    // ========================================================================
    // Step 1: Well-conditioned k-space (simplified LSMR)
    // ========================================================================

    // FFT of field (reuse work_complex)
    for i in 0..n_total {
        work_complex[i] = Complex64::new(local_field[i], 0.0);
    }
    fft_ws.fft3d(&mut work_complex);

    // Store field_fft for LSMR iterations
    let field_fft: Vec<Complex64> = work_complex.clone();

    // Initial estimate: chi = D * f / (D^2 + epsilon) for well-conditioned
    // Stored in work_complex
    for i in 0..n_total {
        let d = d_kernel[i];
        if d.abs() > params.delta {
            work_complex[i] = field_fft[i] * d / (d * d + 1e-6);
        } else {
            work_complex[i] = Complex64::new(0.0, 0.0);
        }
    }

    // Simple iterative refinement for well-conditioned region
    // Use a temporary buffer for residual
    let mut residual = vec![Complex64::new(0.0, 0.0); n_total];
    for _ in 0..params.lsmr_iter {
        // residual = f - D * chi
        for i in 0..n_total {
            residual[i] = field_fft[i] - work_complex[i] * d_kernel[i];
        }

        // update chi for well-conditioned region
        for i in 0..n_total {
            let d = d_kernel[i];
            if d.abs() > params.delta {
                work_complex[i] += residual[i] * d / (d * d + 1e-6);
            }
        }
    }

    // Transform to spatial domain
    fft_ws.ifft3d(&mut work_complex);

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
    fft_ws.fft3d(&mut work_complex);

    let mut f_hat: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n_total];
    for i in 0..n_total {
        if m_mask[i].abs() > 1e-20 && inv_a[i].abs() > 1e-20 {
            f_hat[i] = work_complex[i] * (m_mask[i] / params.rho) * inv_a[i];
        }
    }

    // ========================================================================
    // Pre-allocate working buffers and run ADMM iterations
    // ========================================================================

    let mut buf = AdmmBuffers::new(n_total);
    // Copy LSMR result into buf.x
    buf.x.copy_from_slice(&x);

    let inv_rho = 1.0 / params.rho;

    for iter in 0..params.max_iter {
        progress(iter + 1, params.max_iter);

        // RTS uses rho=1.0 in admm_step because inv_a already incorporates rho
        let converged = admm_step(
            &mut buf, &mut fft_ws, &f_hat, &inv_a, 1.0, grid, params.tol,
            |vx, vy, vz, _| (shrink(vx, inv_rho), shrink(vy, inv_rho), shrink(vz, inv_rho)),
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

    #[test]
    fn test_rts_zero_field() {
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = RtsParams { delta: 0.15, mu: 1e5, rho: 10.0, tol: 1e-2, max_iter: 5, lsmr_iter: 2 };

        let chi = rts(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        for &val in chi.iter() {
            assert!(val.abs() < 1e-6, "Zero field should give near-zero chi");
        }
    }

    #[test]
    fn test_rts_finite() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = RtsParams { delta: 0.15, mu: 1e5, rho: 10.0, tol: 1e-2, max_iter: 5, lsmr_iter: 2 };

        let chi = rts(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

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
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = RtsParams { delta: 0.15, mu: 1e5, rho: 10.0, tol: 1e-2, max_iter: 5, lsmr_iter: 2 };

        let chi = rts(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        assert_eq!(chi[0], 0.0, "Masked voxel should be zero");
        assert_eq!(chi[10], 0.0, "Masked voxel should be zero");
    }

    /// Verify parallel and sequential RTS produce identical results.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_rts_parallel_matches_sequential() {
        let n = 16;
        let field: Vec<f64> = (0..n*n*n).map(|i| ((i as f64) * 0.7).sin() * 0.01).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = RtsParams { delta: 0.15, mu: 1e5, rho: 10.0, tol: 1e-4, max_iter: 20, lsmr_iter: 4 };

        // Sequential (1 thread)
        let pool_1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let chi_seq = pool_1.install(|| {
            rts(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {})
        });

        // Parallel (default threads)
        let chi_par = rts(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        // Compare
        for (i, (s, p)) in chi_seq.iter().zip(chi_par.iter()).enumerate() {
            assert!(
                (s - p).abs() < 1e-10,
                "RTS mismatch at voxel {}: seq={} par={} diff={}",
                i, s, p, (s - p).abs()
            );
        }
    }
}

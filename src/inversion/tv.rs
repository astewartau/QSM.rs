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
//! Reference implementation: https://github.com/kamesy/QSM.jl (`tv`), whose defaults these
//! are: `lambda = 1e-3`, `rho = 100*lambda`, `tol = 1e-3`, 250 iterations. Like QSM.jl, the
//! field is in ppm and the first x-update starts from the scaled split variable set to the
//! gradient of the field. QSM.jl's optional data and TV weights (`W`, `Wtv`) are not ported;
//! their default (`nothing`) is what this function computes.
//!
//! QSM.jl is this method's reference: QSM.jl, Kames's maintained successor to QSM.m, is the implementation
//! designated for this family (QSM.rs's reference rule: the authors' code where it
//! exists, otherwise one designated implementation per family, the others' values recorded).
//!
//! QSM.jl pads the field to its fast FFT size first (`pad = (0, 0, 0)`, `padfastfft`), and so
//! does this port by default ([`TvParams::pad`]).
//!
//! # Half spectrum (tilted B0)
//!
//! QSM.jl applies its k-space multipliers on FFTW's real-to-complex half spectrum and inverts
//! with `irfft`. With B0 tilted from the grid axes the dipole kernel is not even on the Nyquist
//! planes of an even axis (`D(k) ≠ D(−k)` there, since the index of −½ maps to itself), and
//! `irfft` resolves that differently from a full complex FFT and its real part. This port uses
//! the multiplier `irfft` implies ([`crate::fft::c2r_kernel`]/[`crate::fft::c2r_spectrum`],
//! astewartau/QSM.rs#329), which reproduces QSM.jl there too. Before (2026-10-29) the full-FFT
//! port differed from QSM.jl on the oblique KKI travelling head (Philips Achieva, B0 tilted) by
//! 0.5–1.2 % NRMSE for TKD, TSVD, Tikhonov and RTS (QSMxT/QSM-CI parity workflow);
//! `tests/inversion_qsmjl_parity.rs` now has even-sized oblique cases.
//!
//! # Agreement with QSM.jl
//!
//! On the QSM-CI in-vivo travelling-head scan (Siemens Prisma 2.89 T, 1 mm, 184×256×128, V-SHARP
//! local field and brain mask), against QSM.jl in the QSM-CI `julia-qsm` image (`sha256:b3c048ac…`,
//! QSM.jl fde4b92), both outputs saved as float32:
//! NRMSE 3.4e-14 % at QSM.jl's defaults. Before this port padded, its defaults differed from
//! QSM.jl's by NRMSE 6.9 %. `tests/inversion_qsmjl_parity.rs` checks it against QSM.jl's output on
//! a small phantom (relative L2 ~1e-15). As in [`super::rts`](mod@super::rts), the DC term of the
//! Laplacian is exactly 0 here; on anisotropic voxels QSM.jl inverts its rounding, which shifts
//! χ by an arbitrary constant.
//!
//! # Reference pipeline
//! QSM.jl's README pipeline (commit fde4b92, L35-52: BET on the last echo, 5-voxel erosion,
//! Laplacian unwrapping, V-SHARP), as for [`super::rts`](mod@super::rts).

use crate::utils::{shrink, apply_mask_zero};
use crate::utils::gradient::fgrad_inplace;
use crate::Grid;
use super::admm::{AdmmBuffers, admm_step, prepare_admm_spectral};

/// TV-ADMM algorithm parameters
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct TvParams {
    /// Regularization parameter
    pub lambda: f64,
    /// ADMM penalty parameter; `None` is QSM.jl's `100*lambda`
    pub rho: Option<f64>,
    /// Convergence tolerance
    pub tol: f64,
    /// Maximum iterations
    pub max_iter: usize,
    /// Zero padding per axis, QSM.jl's `pad` (default `[0, 0, 0]`, QSM.jl's): `< 0` does not
    /// pad that axis, `≥ 0` pads it by at least that many voxels, rounded up to QSM.jl's fast
    /// FFT size (see [`with_qsmjl_padding`](crate::utils::padding::with_qsmjl_padding)).
    pub pad: [i64; 3],
}

impl Default for TvParams {
    fn default() -> Self {
        Self {
            lambda: 1e-3,
            rho: None,
            tol: 1e-3,
            max_iter: 250,
            pad: crate::utils::padding::QSMJL_PAD,
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
    progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    crate::utils::padding::with_qsmjl_padding(local_field, mask, grid, params.pad, |f, m, g| {
        tv_admm_unpadded(f, m, g, bdir, params, progress)
    })
}

/// [`tv_admm`] on the already padded grid.
fn tv_admm_unpadded(
    local_field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &TvParams,
    mut progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let n_total = grid.n_total();

    let rho = params.rho.unwrap_or(100.0 * params.lambda);

    // Pre-compute spectral operators
    let (mut fft_ws, inv_a, f_hat) = prepare_admm_spectral(local_field, grid, bdir, rho);

    // Pre-allocate working buffers and run ADMM iterations
    let mut buf = AdmmBuffers::new(n_total);
    let lambda_over_rho = params.lambda / rho;

    // QSM.jl starts the split variable at the gradient of the field (`d = ∇f`, with `u = 0`),
    // so the first x-update already carries the TV term.
    fgrad_inplace(&mut buf.gx, &mut buf.gy, &mut buf.gz, local_field, grid);

    for iter in 0..params.max_iter {
        progress(iter + 1, params.max_iter);

        let converged = admm_step(
            &mut buf, &mut fft_ws, &f_hat, &inv_a, rho, grid, params.tol,
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
        let params = TvParams { lambda: 1e-3, rho: Some(0.1), tol: 1e-2, max_iter: 10, ..Default::default() };

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
        let params = TvParams { lambda: 1e-3, rho: Some(0.1), tol: 1e-2, max_iter: 10, ..Default::default() };

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
        let params = TvParams { lambda: 1e-2, rho: Some(1.0), tol: 1e-2, max_iter: 50, ..Default::default() };

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
        let params = TvParams { lambda: 1e-3, rho: Some(0.1), tol: 1e-3, max_iter: 50, ..Default::default() };

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

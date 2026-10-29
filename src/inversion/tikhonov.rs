//! Tikhonov regularization for QSM
//!
//! Tikhonov regularization adds an L2 penalty term to stabilize the inversion:
//!
//! chi = argmin_x ||Dx - f||_2^2 + lambda||Gamma*x||_2^2
//!
//! This has a closed-form solution in k-space:
//! chi_hat = D* * f_hat / (|D|^2 + lambda|Gamma|^2)
//!
//! Reference:
//! Bilgic, B., et al. (2014). "Fast image reconstruction with L2-regularization."
//! Journal of Magnetic Resonance Imaging, 40(1):181-191. https://doi.org/10.1002/jmri.24365
//!
//! Reference implementation: https://github.com/kamesy/QSM.jl
//!
//! # Reference pipeline
//! QSM.jl (commit fde4b92) `tikh` (`src/inversion/direct.jl` L100-155): lambda 1e-2 and
//! `reg = :identity` in the code (L149-150); its docstring's signature line says `:gradient`
//! (L109) but its keyword list says `:identity` (L131). Same README pipeline as
//! [`super::rts`](mod@super::rts).
//!
//! QSM.jl is this method's reference: its authors released no code, and QSM.jl, Kames's maintained successor to
//! QSM.m, is the implementation designated for this family (QSM.rs's reference rule: the authors' code where it
//! exists, otherwise one designated implementation per family, the others' values recorded).
//!
//! QSM.jl pads the field to its fast FFT size first (`pad = (0, 0, 0)`, `padfastfft`,
//! `src/inversion/direct.jl:182-184`), and so does this port by default ([`TikhonovParams::pad`]).
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
//! identical (NRMSE 0) at QSM.jl's defaults and with padding off on both. Before this port
//! padded, its defaults differed from QSM.jl's by NRMSE 1.9 %. `tests/inversion_qsmjl_parity.rs`
//! checks the identity and gradient regularisers against QSM.jl's output on small phantoms.

use num_complex::Complex64;
use crate::fft::{fft3d, ifft3d};
use crate::kernels::dipole::dipole_kernel;
use crate::kernels::laplacian::laplacian_kernel;
use crate::utils::apply_mask_zero;
use crate::Grid;

/// Regularization type for Tikhonov
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug)]
pub enum Regularization {
    /// Identity: lambda||x||_2^2
    Identity,
    /// Gradient: lambda||grad(x)||_2^2 (uses negative Laplacian)
    Gradient,
    /// Laplacian: lambda||lap(x)||_2^2
    Laplacian,
}

/// Tikhonov algorithm parameters
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct TikhonovParams {
    /// Regularization weight
    pub lambda: f64,
    /// Regularization type
    pub reg: Regularization,
    /// Zero padding per axis, QSM.jl's `pad` (default `[0, 0, 0]`, QSM.jl's): `< 0` does not
    /// pad that axis, `≥ 0` pads it by at least that many voxels, rounded up to QSM.jl's fast
    /// FFT size (see [`with_qsmjl_padding`](crate::utils::padding::with_qsmjl_padding)).
    pub pad: [i64; 3],
}

impl Default for TikhonovParams {
    fn default() -> Self {
        Self { lambda: 0.01, reg: Regularization::Identity, pad: crate::utils::padding::QSMJL_PAD }
    }
}

/// Tikhonov regularization for dipole inversion
///
/// # Arguments
/// * `local_field` - Local field values
/// * `mask` - Binary mask (1 = inside ROI, 0 = outside)
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `bdir` - B0 field direction
/// * `params` - Tikhonov parameters (lambda, regularization type)
///
/// # Returns
/// Susceptibility map
pub fn tikhonov(
    local_field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &TikhonovParams,
) -> Vec<f64> {
    crate::utils::padding::with_qsmjl_padding(local_field, mask, grid, params.pad, |f, m, g| {
        tikhonov_unpadded(f, m, g, bdir, params)
    })
}

/// [`tikhonov`] on the already padded grid.
fn tikhonov_unpadded(
    local_field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &TikhonovParams,
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let n_total = grid.n_total();
    let lambda = params.lambda;

    // Generate dipole kernel
    let d = dipole_kernel(grid, bdir);

    // Generate regularization kernel and FFT it
    let gamma: Vec<f64> = match params.reg {
        Regularization::Identity => {
            vec![1.0; n_total]
        }
        Regularization::Gradient => {
            // Negative Laplacian for gradient regularization
            let l = laplacian_kernel(grid, true);
            // FFT to get frequency response
            let mut l_complex: Vec<Complex64> = l.iter()
                .map(|&x| Complex64::new(x, 0.0))
                .collect();
            fft3d(&mut l_complex, nx, ny, nz);
            // Take real part (Laplacian FFT is real)
            l_complex.iter().map(|c| c.re).collect()
        }
        Regularization::Laplacian => {
            // Laplacian squared
            let l = laplacian_kernel(grid, false);
            let mut l_complex: Vec<Complex64> = l.iter()
                .map(|&x| Complex64::new(x, 0.0))
                .collect();
            fft3d(&mut l_complex, nx, ny, nz);
            // |Gamma|^2
            l_complex.iter().map(|c| c.re * c.re).collect()
        }
    };

    // Compute Tikhonov inverse: D / (D^2 + lambda*Gamma)
    let inv_d: Vec<f64> = d.iter().zip(gamma.iter()).map(|(&dval, &gval)| {
        let denom = dval * dval + lambda * gval;
        if denom.abs() > 1e-20 {
            dval / denom
        } else {
            0.0
        }
    }).collect();

    // QSM.jl applies this multiplier on its real-to-complex half spectrum; the kernel `irfft`
    // implies, so the full FFT reproduces it with tilted B0 (astewartau/QSM.rs#329).
    let inv_d = crate::fft::c2r_kernel(&inv_d, nx, ny, nz);

    // Convert local field to complex
    let mut field_complex: Vec<Complex64> = local_field.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();

    // FFT of local field
    fft3d(&mut field_complex, nx, ny, nz);

    // Multiply by Tikhonov inverse
    for i in 0..n_total {
        field_complex[i] *= inv_d[i];
    }

    // IFFT to get susceptibility
    ifft3d(&mut field_complex, nx, ny, nz);

    // Extract real part and apply mask
    let mut chi: Vec<f64> = field_complex.iter()
        .map(|c| c.re)
        .collect();

    apply_mask_zero(&mut chi, mask);

    chi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tikhonov_zero_field() {
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let chi = tikhonov(&field, &mask, &grid, (0.0, 0.0, 1.0), &TikhonovParams::default());

        for val in chi.iter() {
            assert!(val.abs() < 1e-10, "Zero field should give zero chi");
        }
    }

    #[test]
    fn test_tikhonov_finite() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.01).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let chi = tikhonov(&field, &mask, &grid, (0.0, 0.0, 1.0), &TikhonovParams::default());

        for (i, val) in chi.iter().enumerate() {
            assert!(val.is_finite(), "Chi should be finite at index {}", i);
        }
    }

    #[test]
    fn test_tikhonov_regularization_types() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| ((i as f64) * 0.1).sin()).collect();
        let mask = vec![1u8; n * n * n];
        let bdir = (0.0, 0.0, 1.0);
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let chi_id = tikhonov(&field, &mask, &grid, bdir,
                             &TikhonovParams { lambda: 0.01, reg: Regularization::Identity, ..Default::default() });
        let chi_grad = tikhonov(&field, &mask, &grid, bdir,
                               &TikhonovParams { lambda: 0.01, reg: Regularization::Gradient, ..Default::default() });
        let chi_lap = tikhonov(&field, &mask, &grid, bdir,
                              &TikhonovParams { lambda: 0.01, reg: Regularization::Laplacian, ..Default::default() });

        // All should be different
        let diff_ig: f64 = chi_id.iter().zip(chi_grad.iter())
            .map(|(a, b)| (a - b).abs()).sum();
        let diff_gl: f64 = chi_grad.iter().zip(chi_lap.iter())
            .map(|(a, b)| (a - b).abs()).sum();

        assert!(diff_ig > 1e-10, "Identity and Gradient should differ");
        assert!(diff_gl > 1e-10, "Gradient and Laplacian should differ");
    }
}

//! Tikhonov regularization for QSM
//!
//! Tikhonov regularization adds an L2 penalty term to stabilize the inversion:
//!
//! χ = argmin_x ||Dx - f||₂² + λ||Γx||₂²
//!
//! This has a closed-form solution in k-space:
//! χ̂ = D* · f̂ / (|D|² + λ|Γ|²)
//!
//! Reference:
//! Bilgic B, et al. Fast image reconstruction with L2-regularization.
//! JMRI 2014;40(1):181-91.

use num_complex::Complex64;
use crate::fft::{fft3d, ifft3d};
use crate::kernels::dipole::dipole_kernel;
use crate::kernels::laplacian::laplacian_kernel;

/// Regularization type for Tikhonov
#[derive(Clone, Copy, Debug)]
pub enum Regularization {
    /// Identity: λ||x||₂²
    Identity,
    /// Gradient: λ||∇x||₂² (uses negative Laplacian)
    Gradient,
    /// Laplacian: λ||∆x||₂²
    Laplacian,
}

/// Tikhonov regularization for dipole inversion
///
/// # Arguments
/// * `local_field` - Local field values
/// * `mask` - Binary mask (1 = inside ROI, 0 = outside)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bdir` - B0 field direction
/// * `lambda` - Regularization parameter (typically 1e-2 to 1e-4)
/// * `reg` - Type of regularization
///
/// # Returns
/// Susceptibility map
pub fn tikhonov(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bdir: (f64, f64, f64),
    lambda: f64,
    reg: Regularization,
) -> Vec<f64> {
    let n_total = nx * ny * nz;

    // Generate dipole kernel
    let d = dipole_kernel(nx, ny, nz, vsx, vsy, vsz, bdir);

    // Generate regularization kernel and FFT it
    let gamma: Vec<f64> = match reg {
        Regularization::Identity => {
            vec![1.0; n_total]
        }
        Regularization::Gradient => {
            // Negative Laplacian for gradient regularization
            let l = laplacian_kernel(nx, ny, nz, vsx, vsy, vsz, true);
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
            let l = laplacian_kernel(nx, ny, nz, vsx, vsy, vsz, false);
            let mut l_complex: Vec<Complex64> = l.iter()
                .map(|&x| Complex64::new(x, 0.0))
                .collect();
            fft3d(&mut l_complex, nx, ny, nz);
            // |Γ|²
            l_complex.iter().map(|c| c.re * c.re).collect()
        }
    };

    // Compute Tikhonov inverse: D / (D² + λΓ)
    let inv_d: Vec<f64> = d.iter().zip(gamma.iter()).map(|(&dval, &gval)| {
        let denom = dval * dval + lambda * gval;
        if denom.abs() > 1e-20 {
            dval / denom
        } else {
            0.0
        }
    }).collect();

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

    // Apply mask
    for i in 0..n_total {
        if mask[i] == 0 {
            chi[i] = 0.0;
        }
    }

    chi
}

/// Tikhonov with default parameters (gradient regularization, λ=0.01)
pub fn tikhonov_default(
    local_field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    tikhonov(
        local_field, mask, nx, ny, nz, vsx, vsy, vsz,
        (0.0, 0.0, 1.0), 0.01, Regularization::Gradient
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tikhonov_zero_field() {
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        let chi = tikhonov_default(&field, &mask, n, n, n, 1.0, 1.0, 1.0);

        for val in chi.iter() {
            assert!(val.abs() < 1e-10, "Zero field should give zero chi");
        }
    }

    #[test]
    fn test_tikhonov_finite() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.01).collect();
        let mask = vec![1u8; n * n * n];

        let chi = tikhonov_default(&field, &mask, n, n, n, 1.0, 1.0, 1.0);

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

        let chi_id = tikhonov(&field, &mask, n, n, n, 1.0, 1.0, 1.0,
                             bdir, 0.01, Regularization::Identity);
        let chi_grad = tikhonov(&field, &mask, n, n, n, 1.0, 1.0, 1.0,
                               bdir, 0.01, Regularization::Gradient);
        let chi_lap = tikhonov(&field, &mask, n, n, n, 1.0, 1.0, 1.0,
                              bdir, 0.01, Regularization::Laplacian);

        // All should be different
        let diff_ig: f64 = chi_id.iter().zip(chi_grad.iter())
            .map(|(a, b)| (a - b).abs()).sum();
        let diff_gl: f64 = chi_grad.iter().zip(chi_lap.iter())
            .map(|(a, b)| (a - b).abs()).sum();

        assert!(diff_ig > 1e-10, "Identity and Gradient should differ");
        assert!(diff_gl > 1e-10, "Gradient and Laplacian should differ");
    }
}

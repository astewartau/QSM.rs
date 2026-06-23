//! RESHARP background field removal
//!
//! Regularized SHARP — uses Tikhonov regularization instead of TSVD
//! truncation for more robust deconvolution of the SMV-filtered field.
//!
//! Solves: argmin ||M·ifft(C·fft(x)) - M·ifft(C·fft(y))||² + λ||x||²
//!
//! Where C = (1 - S) is the high-pass (delta minus SMV) kernel in k-space,
//! M is the eroded mask, and λ is the Tikhonov regularization parameter.
//!
//! Reference:
//! Sun, H. and Wilman, A.H. (2013).
//! "Background field removal using spherical mean value filtering and Tikhonov regularization."
//! Magn Reson Med, 71(3):1151-1157. https://doi.org/10.1002/mrm.24765

use num_complex::Complex64;
use crate::Grid;
use crate::fft::{fft3d, ifft3d, fft_real_kernel};
use crate::kernels::smv::{smv_kernel, erode_mask_smv};
use crate::solvers::cg_solve_with_progress;

/// RESHARP algorithm parameters
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct ResharpParams {
    /// SMV kernel radius in mm
    pub radius: f64,
    /// Tikhonov regularization parameter
    pub tik_reg: f64,
    /// CG convergence tolerance
    pub tol: f64,
    /// Maximum CG iterations
    pub max_iter: usize,
}

impl Default for ResharpParams {
    fn default() -> Self {
        Self {
            radius: 6.0,
            tik_reg: 1e-4,
            tol: 1e-6,
            max_iter: 30,
        }
    }
}

/// RESHARP background field removal
///
/// # Arguments
/// * `field` - Unwrapped total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `grid` - Volume dimensions and voxel sizes
/// * `params` - RESHARP algorithm parameters
/// * `progress` - Progress callback (current_iteration, max_iterations)
///
/// # Returns
/// (local_field, eroded_mask)
pub fn resharp(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &ResharpParams,
    mut progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<u8>) {
    let (nx, ny, nz) = grid.dims;
    let n_total = nx * ny * nz;

    // Generate SMV kernel and FFT
    let s_kernel = smv_kernel(grid, params.radius);
    let s_fft = fft_real_kernel(&s_kernel, nx, ny, nz);

    // DKER = 1 - S (high-pass / delta-kernel) in k-space
    let dker: Vec<f64> = s_fft.iter().map(|&s| 1.0 - s).collect();

    // Erode mask via SMV convolution
    let eroded_mask = erode_mask_smv(mask, &s_fft, grid, 1.0 - 1e-7_f64.sqrt());
    let eroded_mask_f64: Vec<f64> = eroded_mask.iter()
        .map(|&m| m as f64)
        .collect();

    // Compute RHS: b = H'(H(field))
    // H(x) = M * ifft(DKER * fft(x))
    // H'(y) = ifft(DKER * fft(M * y))   [DKER is real so conj(DKER) = DKER]
    //
    // b = ifft(DKER * fft(M * ifft(DKER * fft(field))))
    let b = apply_ht_h(&dker, &eroded_mask_f64, field, nx, ny, nz);

    // Solve (H'H + λI)x = b via CG
    let tik_reg = params.tik_reg;
    let x0 = vec![0.0; n_total];
    let x = cg_solve_with_progress(
        |x_vec| {
            let mut result = apply_ht_h(&dker, &eroded_mask_f64, x_vec, nx, ny, nz);
            // Add Tikhonov term: + λx
            for i in 0..n_total {
                result[i] += tik_reg * x_vec[i];
            }
            result
        },
        &b,
        &x0,
        params.tol,
        params.max_iter,
        &mut progress,
    );

    // Apply eroded mask to result
    let local_field: Vec<f64> = x.iter()
        .enumerate()
        .map(|(i, &v)| if eroded_mask[i] == 1 { v } else { 0.0 })
        .collect();

    (local_field, eroded_mask)
}

/// Apply H'H to a vector: ifft(DKER * fft(M * ifft(DKER * fft(x))))
///
/// H(x)  = M * ifft(DKER * fft(x))
/// H'(y) = ifft(DKER * fft(M * y))
fn apply_ht_h(
    dker: &[f64],
    mask: &[f64],
    x: &[f64],
    nx: usize, ny: usize, nz: usize,
) -> Vec<f64> {
    let n_total = nx * ny * nz;

    // Step 1: H(x) = M * ifft(DKER * fft(x))
    let mut tmp: Vec<Complex64> = x.iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect();
    fft3d(&mut tmp, nx, ny, nz);

    for i in 0..n_total {
        tmp[i] *= dker[i];
    }
    ifft3d(&mut tmp, nx, ny, nz);

    // Apply mask
    for i in 0..n_total {
        tmp[i] *= mask[i];
    }

    // Step 2: H'(Hx) = ifft(DKER * fft(M * Hx))
    fft3d(&mut tmp, nx, ny, nz);

    for i in 0..n_total {
        tmp[i] *= dker[i];
    }
    ifft3d(&mut tmp, nx, ny, nz);

    tmp.iter().map(|c| c.re).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resharp_zero_field() {
        let n = 16;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = ResharpParams { radius: 2.0, tik_reg: 1e-4, tol: 1e-6, max_iter: 50 };

        let (local, _) = resharp(&field, &mask, &grid, &params, |_, _| {});

        for &val in local.iter() {
            assert!(val.abs() < 1e-8, "Zero field should give zero local field, got {}", val);
        }
    }

    #[test]
    fn test_resharp_finite() {
        let n = 16;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.01).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = ResharpParams { radius: 2.0, tik_reg: 1e-4, tol: 1e-6, max_iter: 50 };

        let (local, eroded) = resharp(&field, &mask, &grid, &params, |_, _| {});

        for (i, &val) in local.iter().enumerate() {
            assert!(val.is_finite(), "Local field should be finite at index {}", i);
        }

        let eroded_count: usize = eroded.iter().map(|&m| m as usize).sum();
        assert!(eroded_count > 0, "Eroded mask should have some voxels");
    }
}

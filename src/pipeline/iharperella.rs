//! iHARPERELLA — integrated phase unwrapping and background field removal
//!
//! Performs Laplacian-based phase unwrapping and SMV background removal
//! simultaneously in a single iterative loop. Works directly on wrapped phase.
//!
//! Each iteration:
//! 1. Computes residual wrapped phase
//! 2. Laplacian-unwraps the residual (FFT Poisson solver)
//! 3. SMV-filters to estimate background field
//! 4. Accumulates the local (tissue) field contribution
//!
//! Reference:
//! Li, W., Avram, A.V., Wu, B., Xiao, X., Liu, C. (2014).
//! "Integrated Laplacian-based phase unwrapping and background phase removal
//! for quantitative susceptibility mapping."
//! NMR in Biomedicine, 27(2):219-227. https://doi.org/10.1002/nbm.3056

use num_complex::Complex64;
use crate::fft::{fft3d, ifft3d};
use crate::kernels::smv::smv_kernel;
use crate::unwrap::laplacian::{wrap, wrapped_laplacian_periodic, solve_poisson_fft};

/// iHARPERELLA algorithm parameters
#[derive(Clone, Debug)]
pub struct IharperellaParams {
    /// SMV kernel radius in mm
    pub radius: f64,
    /// Number of iterations
    pub niter: usize,
}

impl Default for IharperellaParams {
    fn default() -> Self {
        Self {
            radius: 5.0,
            niter: 40,
        }
    }
}

/// iHARPERELLA — integrated phase unwrapping and background removal
///
/// # Arguments
/// * `phase` - Wrapped phase in radians (nx * ny * nz)
/// * `mask` - Binary brain mask (nx * ny * nz), 1 = inside
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `radius` - SMV kernel radius in mm
/// * `niter` - Number of iterations (typically 40)
///
/// # Returns
/// (tissue_phase, eroded_mask) — unwrapped local field and eroded mask
pub fn iharperella(
    phase: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64,
    niter: usize,
) -> (Vec<f64>, Vec<u8>) {
    iharperella_with_progress(phase, mask, nx, ny, nz, vsx, vsy, vsz,
                              radius, niter, |_, _| {})
}

/// iHARPERELLA with default parameters
pub fn iharperella_default(
    phase: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> (Vec<f64>, Vec<u8>) {
    let p = IharperellaParams::default();
    iharperella(phase, mask, nx, ny, nz, vsx, vsy, vsz, p.radius, p.niter)
}

/// iHARPERELLA with progress callback
///
/// Callback receives (current_iteration, total_iterations).
pub fn iharperella_with_progress<F>(
    phase: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64,
    niter: usize,
    mut callback: F,
) -> (Vec<f64>, Vec<u8>)
where
    F: FnMut(usize, usize),
{
    let n_total = nx * ny * nz;

    // Pre-compute SMV kernel FFT for background estimation
    let s_kernel = smv_kernel(nx, ny, nz, vsx, vsy, vsz, radius);
    let mut s_complex: Vec<Complex64> = s_kernel.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut s_complex, nx, ny, nz);
    let s_fft: Vec<f64> = s_complex.iter().map(|c| c.re).collect();

    // Erode mask: convolve mask with SMV kernel, keep where result ≈ 1
    let mask_f64: Vec<f64> = mask.iter().map(|&m| m as f64).collect();
    let mut mask_complex: Vec<Complex64> = mask_f64.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut mask_complex, nx, ny, nz);
    for i in 0..n_total {
        mask_complex[i] *= s_fft[i];
    }
    ifft3d(&mut mask_complex, nx, ny, nz);

    let delta = 1.0 - 1e-7_f64.sqrt();
    let eroded_mask: Vec<u8> = mask_complex.iter()
        .map(|c| if c.re > delta { 1 } else { 0 })
        .collect();

    // Iterative unwrapping + background removal
    let mut tissue_phase = vec![0.0; n_total];

    for iter in 0..niter {
        callback(iter + 1, niter);

        // Step 1: Compute residual wrapped phase
        let residual: Vec<f64> = phase.iter()
            .zip(tissue_phase.iter())
            .map(|(&p, &t)| wrap(p - t))
            .collect();

        // Step 2: Laplacian-unwrap the residual
        let d2u = wrapped_laplacian_periodic(&residual, nx, ny, nz, vsx, vsy, vsz);

        // Mask the Laplacian
        let d2u_masked: Vec<f64> = d2u.iter()
            .enumerate()
            .map(|(i, &val)| if mask[i] != 0 { val } else { 0.0 })
            .collect();

        // Solve Poisson equation
        let unwrapped = solve_poisson_fft(&d2u_masked, nx, ny, nz, vsx, vsy, vsz);

        // Step 3: SMV-filter the unwrapped residual to estimate background
        let mut field_complex: Vec<Complex64> = unwrapped.iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect();
        fft3d(&mut field_complex, nx, ny, nz);

        for i in 0..n_total {
            field_complex[i] *= s_fft[i];
        }
        ifft3d(&mut field_complex, nx, ny, nz);

        // Step 4: Accumulate tissue contribution = unwrapped - SMV(unwrapped)
        // Subtract mean of the update within eroded mask to prevent DC bias
        // accumulation (the Poisson solver enforces zero mean over the full grid,
        // not within the mask, causing a small DC offset each iteration)
        let mut update_sum = 0.0;
        let mut update_count = 0usize;
        for i in 0..n_total {
            if eroded_mask[i] == 1 {
                update_sum += unwrapped[i] - field_complex[i].re;
                update_count += 1;
            }
        }
        let update_mean = if update_count > 0 { update_sum / update_count as f64 } else { 0.0 };

        for i in 0..n_total {
            if eroded_mask[i] == 1 {
                tissue_phase[i] += (unwrapped[i] - field_complex[i].re) - update_mean;
            }
        }
    }

    // Final masking
    for i in 0..n_total {
        if eroded_mask[i] == 0 {
            tissue_phase[i] = 0.0;
        }
    }

    (tissue_phase, eroded_mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iharperella_zero_phase() {
        let n = 16;
        let phase = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        let (tissue, _) = iharperella(&phase, &mask, n, n, n, 1.0, 1.0, 1.0, 2.0, 5);

        for &val in tissue.iter() {
            assert!(val.abs() < 1e-8, "Zero phase should give zero tissue phase, got {}", val);
        }
    }

    #[test]
    fn test_iharperella_finite() {
        let n = 16;
        let phase: Vec<f64> = (0..n*n*n).map(|i| wrap((i as f64) * 0.1)).collect();
        let mask = vec![1u8; n * n * n];

        let (tissue, eroded) = iharperella(&phase, &mask, n, n, n, 1.0, 1.0, 1.0, 2.0, 5);

        for (i, &val) in tissue.iter().enumerate() {
            assert!(val.is_finite(), "Tissue phase should be finite at index {}", i);
        }

        let eroded_count: usize = eroded.iter().map(|&m| m as usize).sum();
        assert!(eroded_count > 0, "Eroded mask should have some voxels");
    }
}

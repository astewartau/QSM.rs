//! Spatially Dependent Filtering (SDF) for QSMART
//!
//! SDF is the background field removal method used in QSMART. It uses variable-radius
//! Gaussian filtering where the kernel size depends on the proximity to the brain boundary.
//! This allows for aggressive filtering in the brain interior while preserving details
//! near the surface.
//!
//! The algorithm includes optional curvature-based weighting to further reduce artifacts
//! at highly curved brain regions.
//!
//! Reference: Syeda et al., QSMART: Quantitative Susceptibility Mapping Artifact Reduction Technique

use crate::utils::curvature::calculate_curvature_proximity;

/// Parameters for SDF background field removal
#[derive(Clone, Debug)]
pub struct SdfParams {
    /// Sigma parameter for initial proximity map (default 10 for stage 1, 8 for stage 2)
    pub sigma1: f64,
    /// Sigma parameter for vasculature proximity (default 0 for stage 1, 2 for stage 2)
    pub sigma2: f64,
    /// Spatial radius for morphological closing of indents (default 8 voxels)
    pub spatial_radius: i32,
    /// Lower limit for clamping proximity values (default 0.6)
    pub lower_lim: f64,
    /// Scaling constant for curvature (default 500)
    pub curv_constant: f64,
    /// Whether to use curvature-based edge refinement
    pub use_curvature: bool,
}

impl Default for SdfParams {
    fn default() -> Self {
        Self {
            sigma1: 10.0,
            // MATLAB passes sigma1 for both sigma args in stage 1.
            // sigma2 affects the combined sigma even though vasculature
            // proximity is skipped when vasc_only is all-ones.
            sigma2: 10.0,
            spatial_radius: 8,
            lower_lim: 0.6,
            curv_constant: 500.0,
            use_curvature: true,
        }
    }
}

impl SdfParams {
    /// Create parameters for QSMART Stage 1
    /// MATLAB passes sigma1 for both sigma args in stage 1.
    pub fn stage1() -> Self {
        Self {
            sigma1: 10.0,
            sigma2: 10.0,
            spatial_radius: 8,
            lower_lim: 0.6,
            curv_constant: 500.0,
            use_curvature: true,
        }
    }

    /// Create parameters for QSMART Stage 2
    pub fn stage2() -> Self {
        Self {
            sigma1: 8.0,
            sigma2: 2.0,
            spatial_radius: 8,
            lower_lim: 0.6,
            curv_constant: 500.0,
            use_curvature: true,
        }
    }
}

/// SDF background field removal
///
/// Removes background field from total field shift using spatially dependent filtering.
///
/// # Arguments
/// * `tfs` - Total field shift (unwrapped phase / ppm)
/// * `mask` - Binary brain mask (weighted by reliability if R_0 is incorporated)
/// * `vasc_only` - Vasculature-only mask (1 = tissue, 0 = vessel). Pass all-ones for stage 1.
/// * `nx`, `ny`, `nz` - Volume dimensions
/// * `params` - SDF parameters
///
/// # Returns
/// Local field shift (background removed)
pub fn sdf(
    tfs: &[f64],
    mask: &[f64],
    vasc_only: &[f64],
    nx: usize, ny: usize, nz: usize,
    params: &SdfParams,
) -> Vec<f64> {
    sdf_with_progress(tfs, mask, vasc_only, nx, ny, nz, params, |_, _| {})
}

/// SDF with progress callback
pub fn sdf_with_progress<F>(
    tfs: &[f64],
    mask: &[f64],
    vasc_only: &[f64],
    nx: usize, ny: usize, nz: usize,
    params: &SdfParams,
    progress_callback: F,
) -> Vec<f64>
where
    F: Fn(usize, usize),
{
    let n_total = nx * ny * nz;

    // Convert mask to binary for morphological operations
    let mask_binary: Vec<u8> = mask.iter().map(|&v| if v > 0.0 { 1 } else { 0 }).collect();

    // Combined sigma for n calculation
    let sigma = (params.sigma1 * params.sigma1 + params.sigma2 * params.sigma2).sqrt();
    let n = if sigma > 0.0 { -sigma.ln() / 0.5_f64.ln() } else { 0.0 };

    // Calculate initial proximity map (prox1)
    // Gaussian smoothing of mask with anisotropic kernel [sigma1, 2*sigma1, 2*sigma1]
    let prox1 = if params.sigma1 > 0.0 {
        gaussian_smooth_3d_masked_f64(mask, mask, nx, ny, nz, &[params.sigma1, 2.0 * params.sigma1, 2.0 * params.sigma1])
    } else {
        mask.to_vec()
    };

    // Calculate curvature-based proximity if enabled
    let prox = if params.use_curvature {
        let (prox_curv, _curv_i) = calculate_curvature_proximity(
            &mask_binary,
            &prox1,
            params.lower_lim,
            params.curv_constant,
            params.sigma1,
            nx, ny, nz,
        );
        prox_curv
    } else {
        // Even without curvature, clamp proximity to lower_lim to prevent
        // filter sizes from getting too small at the edges
        // (matching calculate_curvature.m line 45: prox(prox < lowerLim & prox ~= 0) = lowerLim)
        prox1.iter()
            .zip(mask.iter())
            .map(|(&p, &m)| {
                if m > 0.0 && p > 0.0 && p < params.lower_lim {
                    params.lower_lim
                } else {
                    p
                }
            })
            .collect()
    };

    // Calculate vasculature proximity (prox2) for stage 2
    let prox_final = if params.sigma2 > 0.0 {
        let prox2 = gaussian_smooth_3d_masked_f64(vasc_only, mask, nx, ny, nz, &[params.sigma2, params.sigma2, params.sigma2]);
        // Multiply prox * prox2
        prox.iter().zip(prox2.iter()).map(|(&p, &p2)| p * p2).collect()
    } else {
        prox
    };

    // Calculate alpha = sigma * round(prox^n, 2)
    // Alpha determines the local smoothing kernel size
    let mut alpha: Vec<f64> = prox_final.iter()
        .zip(mask.iter())
        .map(|(&p, &m)| {
            if m > 0.0 {
                let val = sigma * (p.powf(n) * 100.0).round() / 100.0;
                val
            } else {
                0.0
            }
        })
        .collect();

    // Set alpha=1 for vessel regions within mask
    // (vasc_only=0 means vessel, matching sdf_curvature.m line 27)
    for i in 0..n_total {
        if mask[i] > 0.0 && vasc_only[i] == 0.0 {
            alpha[i] = 1.0;
        }
    }

    // Get unique alpha values and sort
    let mut unique_alphas: Vec<f64> = alpha.iter()
        .filter(|&&a| a > 0.0)
        .copied()
        .collect();
    unique_alphas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_alphas.dedup();

    let total_alphas = unique_alphas.len();

    // Create index map: for each voxel, which alpha index does it belong to?
    let mut alpha_index: Vec<usize> = vec![0; n_total];
    for i in 0..n_total {
        if alpha[i] > 0.0 {
            // Find index in unique_alphas
            let idx = unique_alphas.iter().position(|&a| (a - alpha[i]).abs() < 1e-10).unwrap_or(0);
            alpha_index[i] = idx + 1; // 1-indexed to distinguish from background
        }
    }

    // Apply spatially dependent filtering
    // For each unique alpha, smooth and assign to corresponding voxels
    let mut background = vec![0.0f64; n_total];

    // Pre-compute filter size
    let filter_size = 2 * (2.0 * sigma).ceil() as usize + 1;

    for (alpha_idx, &current_alpha) in unique_alphas.iter().enumerate() {
        progress_callback(alpha_idx, total_alphas);

        // Compute smoothed field for this alpha
        let smoothed: Vec<f64> = if current_alpha > 0.0 {
            // Smooth tfs * mask with Gaussian kernel of size current_alpha
            let weighted_tfs: Vec<f64> = tfs.iter()
                .zip(mask.iter())
                .map(|(&t, &m)| t * m)
                .collect();

            let num = gaussian_smooth_3d_with_filter_size(&weighted_tfs, nx, ny, nz, current_alpha, filter_size);
            let denom = gaussian_smooth_3d_with_filter_size(&mask.to_vec(), nx, ny, nz, current_alpha, filter_size);

            // Divide: num / denom
            num.iter()
                .zip(denom.iter())
                .map(|(&n, &d)| if d.abs() > 1e-10 { n / d } else { 0.0 })
                .collect()
        } else {
            // alpha=0: just use tfs*mask
            tfs.iter().zip(mask.iter()).map(|(&t, &m)| t * m).collect()
        };

        // Assign to voxels with this alpha
        for i in 0..n_total {
            if alpha_index[i] == alpha_idx + 1 {
                background[i] = smoothed[i];
            }
        }
    }

    progress_callback(total_alphas, total_alphas);

    // Compute local field: (tfs - background) * mask
    let local_field: Vec<f64> = tfs.iter()
        .zip(background.iter())
        .zip(mask.iter())
        .map(|((&t, &b), &m)| (t - b) * m)
        .collect();

    local_field
}

/// SDF with curvature-based weighting (full QSMART pipeline)
///
/// This is the main entry point matching QSMART's sdf_curvature function.
pub fn sdf_curvature(
    tfs: &[f64],
    mask: &[f64],
    vasc_only: &[f64],
    nx: usize, ny: usize, nz: usize,
    params: &SdfParams,
) -> Vec<f64> {
    // Ensure curvature is enabled
    let params_with_curv = SdfParams {
        use_curvature: true,
        ..params.clone()
    };

    sdf(tfs, mask, vasc_only, nx, ny, nz, &params_with_curv)
}

/// 3D Gaussian smoothing with specified filter size
fn gaussian_smooth_3d_with_filter_size(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    sigma: f64,
    filter_size: usize,
) -> Vec<f64> {
    if sigma <= 0.0 {
        return data.to_vec();
    }

    // Create 1D Gaussian kernel
    let kernel_radius = (filter_size - 1) / 2;
    let mut kernel = vec![0.0f64; filter_size];

    let mut sum = 0.0;
    for i in 0..filter_size {
        let x = i as f64 - kernel_radius as f64;
        kernel[i] = (-x * x / (2.0 * sigma * sigma)).exp();
        sum += kernel[i];
    }

    // Normalize
    for k in kernel.iter_mut() {
        *k /= sum;
    }

    // Apply separable convolution
    let smoothed_x = convolve_1d_direction(data, nx, ny, nz, &kernel, 'x');
    let smoothed_xy = convolve_1d_direction(&smoothed_x, nx, ny, nz, &kernel, 'y');
    let smoothed_xyz = convolve_1d_direction(&smoothed_xy, nx, ny, nz, &kernel, 'z');

    smoothed_xyz
}

/// Gaussian smoothing with anisotropic sigma and mask
fn gaussian_smooth_3d_masked_f64(
    data: &[f64],
    mask: &[f64],
    nx: usize, ny: usize, nz: usize,
    sigmas: &[f64; 3],
) -> Vec<f64> {
    // Apply separable 1D convolutions
    let smoothed_x = convolve_1d_direction_sigma(data, nx, ny, nz, sigmas[0], 'x');
    let smoothed_xy = convolve_1d_direction_sigma(&smoothed_x, nx, ny, nz, sigmas[1], 'y');
    let smoothed_xyz = convolve_1d_direction_sigma(&smoothed_xy, nx, ny, nz, sigmas[2], 'z');

    // Apply mask
    smoothed_xyz.iter()
        .zip(mask.iter())
        .map(|(&v, &m)| if m > 0.0 { v } else { 0.0 })
        .collect()
}

/// 1D convolution along specified axis with replicate padding
/// Matches MATLAB's imgaussfilt3 default behavior
fn convolve_1d_direction(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    kernel: &[f64],
    direction: char,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut result = vec![0.0f64; n_total];
    let kernel_radius = (kernel.len() - 1) / 2;

    let idx = |i: usize, j: usize, k: usize| i + j * nx + k * nx * ny;

    // Helper to clamp index for replicate padding
    let clamp_x = |x: isize| -> usize { x.max(0).min(nx as isize - 1) as usize };
    let clamp_y = |y: isize| -> usize { y.max(0).min(ny as isize - 1) as usize };
    let clamp_z = |z: isize| -> usize { z.max(0).min(nz as isize - 1) as usize };

    match direction {
        'x' => {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let mut sum = 0.0;

                        for ki in 0..kernel.len() {
                            let offset = ki as isize - kernel_radius as isize;
                            let ni = clamp_x(i as isize + offset);
                            sum += data[idx(ni, j, k)] * kernel[ki];
                        }

                        result[idx(i, j, k)] = sum;
                    }
                }
            }
        }
        'y' => {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let mut sum = 0.0;

                        for ki in 0..kernel.len() {
                            let offset = ki as isize - kernel_radius as isize;
                            let nj = clamp_y(j as isize + offset);
                            sum += data[idx(i, nj, k)] * kernel[ki];
                        }

                        result[idx(i, j, k)] = sum;
                    }
                }
            }
        }
        'z' => {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let mut sum = 0.0;

                        for ki in 0..kernel.len() {
                            let offset = ki as isize - kernel_radius as isize;
                            let nk = clamp_z(k as isize + offset);
                            sum += data[idx(i, j, nk)] * kernel[ki];
                        }

                        result[idx(i, j, k)] = sum;
                    }
                }
            }
        }
        _ => panic!("Invalid convolution direction"),
    }

    result
}

/// 1D convolution with specified sigma
fn convolve_1d_direction_sigma(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    sigma: f64,
    direction: char,
) -> Vec<f64> {
    if sigma <= 0.0 {
        return data.to_vec();
    }

    // Create 1D Gaussian kernel
    // Match MATLAB's imgaussfilt3 default: filterSize = 2*ceil(2*sigma)+1
    let kernel_radius = (2.0 * sigma).ceil() as usize;
    let kernel_size = 2 * kernel_radius + 1;
    let mut kernel = vec![0.0f64; kernel_size];

    let mut sum = 0.0;
    for i in 0..kernel_size {
        let x = i as f64 - kernel_radius as f64;
        kernel[i] = (-x * x / (2.0 * sigma * sigma)).exp();
        sum += kernel[i];
    }

    // Normalize
    for k in kernel.iter_mut() {
        *k /= sum;
    }

    convolve_1d_direction(data, nx, ny, nz, &kernel, direction)
}

/// Simple SDF without curvature (faster, for testing)
pub fn sdf_simple(
    tfs: &[f64],
    mask: &[f64],
    nx: usize, ny: usize, nz: usize,
    sigma1: f64,
) -> Vec<f64> {
    let vasc_only = vec![1.0f64; mask.len()];
    let params = SdfParams {
        sigma1,
        sigma2: 0.0,
        use_curvature: false,
        ..Default::default()
    };

    sdf(tfs, mask, &vasc_only, nx, ny, nz, &params)
}

/// Default SDF parameters for stage 1
pub fn sdf_default_stage1(
    tfs: &[f64],
    mask: &[f64],
    nx: usize, ny: usize, nz: usize,
) -> Vec<f64> {
    let vasc_only = vec![1.0f64; mask.len()];
    sdf(tfs, mask, &vasc_only, nx, ny, nz, &SdfParams::stage1())
}

/// Default SDF parameters for stage 2
pub fn sdf_default_stage2(
    tfs: &[f64],
    mask: &[f64],
    vasc_only: &[f64],
    nx: usize, ny: usize, nz: usize,
) -> Vec<f64> {
    sdf(tfs, mask, vasc_only, nx, ny, nz, &SdfParams::stage2())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdf_simple() {
        // Simple test: constant field should give zero local field
        let n = 10;
        let n_total = n * n * n;

        let tfs = vec![1.0f64; n_total];
        let mask = vec![1.0f64; n_total];

        let lfs = sdf_simple(&tfs, &mask, n, n, n, 2.0);

        // Local field should be near zero for constant total field
        let max_lfs = lfs.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        assert!(max_lfs < 0.1, "Max LFS was {}", max_lfs);
    }

    #[test]
    fn test_gaussian_smooth_constant() {
        // Smoothing constant field should give same constant
        let data = vec![5.0f64; 27];
        let smoothed = gaussian_smooth_3d_with_filter_size(&data, 3, 3, 3, 1.0, 5);

        for &v in &smoothed {
            assert!((v - 5.0).abs() < 0.1);
        }
    }
}

//! Vasculature Mask Generation for QSMART
//!
//! This module generates a binary mask identifying blood vessels in the brain.
//! The vasculature mask is used in QSMART's two-stage processing to separate
//! tissue from vessel regions, reducing streaking artifacts near veins.
//!
//! The algorithm:
//! 1. Apply morphological bottom-hat filtering to enhance dark tubular structures
//! 2. Apply Frangi vesselness filter to detect vessels
//! 3. Use Otsu's thresholding to create binary mask
//! 4. Return complementary mask (1 = tissue, 0 = vessel)

use crate::utils::frangi::{FrangiParams, frangi_filter_3d};
use crate::utils::threshold::otsu_threshold;

/// Parameters for vasculature mask generation
#[derive(Clone, Debug)]
pub struct VasculatureParams {
    /// Radius for bottom-hat morphological filter (default 8 voxels)
    pub sphere_radius: i32,
    /// Frangi filter scale range [min, max] (default [0.5, 6.0])
    pub frangi_scale_range: [f64; 2],
    /// Frangi filter scale ratio (default 0.5)
    pub frangi_scale_ratio: f64,
    /// Frangi C parameter (default 500)
    pub frangi_c: f64,
}

impl Default for VasculatureParams {
    fn default() -> Self {
        Self {
            sphere_radius: 8,
            // QSMART Demo defaults: FrangiScaleRange=[0.5,6], FrangiScaleRatio=0.5
            frangi_scale_range: [0.5, 6.0],
            frangi_scale_ratio: 0.5,
            frangi_c: 500.0,
        }
    }
}

/// Generate vasculature mask from magnitude image
///
/// # Arguments
/// * `magnitude` - Average magnitude image (ideally bias-corrected)
/// * `mask` - Binary brain mask
/// * `nx`, `ny`, `nz` - Volume dimensions
/// * `params` - Vasculature detection parameters
///
/// # Returns
/// Complementary vasculature mask (1 = tissue, 0 = vessel)
pub fn generate_vasculature_mask(
    magnitude: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    params: &VasculatureParams,
) -> Vec<f64> {
    generate_vasculature_mask_with_progress(magnitude, mask, nx, ny, nz, params, |_, _| {})
}

/// Generate vasculature mask with progress callback
pub fn generate_vasculature_mask_with_progress<F>(
    magnitude: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    params: &VasculatureParams,
    progress_callback: F,
) -> Vec<f64>
where
    F: Fn(usize, usize),
{
    let n_total = nx * ny * nz;

    // Step 1: Apply morphological bottom-hat filter to enhance dark structures
    // Bottom-hat = closing(image) - image
    progress_callback(0, 4);

    let bottom_hat = morphological_bottom_hat(magnitude, nx, ny, nz, params.sphere_radius);

    // Step 2: Apply mask to bottom-hat result
    progress_callback(1, 4);

    let masked_bottom_hat: Vec<f64> = bottom_hat.iter()
        .zip(mask.iter())
        .map(|(&v, &m)| if m != 0 { v } else { 0.0 })
        .collect();

    // Step 3: Apply Frangi vesselness filter
    progress_callback(2, 4);

    let frangi_params = FrangiParams {
        scale_range: params.frangi_scale_range,
        scale_ratio: params.frangi_scale_ratio,
        alpha: 0.5,
        beta: 0.5,
        c: params.frangi_c,
        black_white: false, // Detect bright structures (after bottom-hat)
    };

    let frangi_result = frangi_filter_3d(&masked_bottom_hat, nx, ny, nz, &frangi_params);
    let enhanced = frangi_result.vesselness;

    // Step 4: Otsu's thresholding
    progress_callback(3, 4);

    let threshold = otsu_threshold(&enhanced, 256);

    // Create binary mask
    let mut vasc_mask = vec![0.0f64; n_total];
    for i in 0..n_total {
        if enhanced[i] > threshold {
            vasc_mask[i] = 1.0;
        }
    }

    // Apply brain mask
    for i in 0..n_total {
        if mask[i] == 0 {
            vasc_mask[i] = 0.0;
        }
    }

    // Return complementary mask (1 = tissue, 0 = vessel)
    let vasc_only: Vec<f64> = vasc_mask.iter()
        .map(|&v| if v > 0.0 { 0.0 } else { 1.0 })
        .collect();

    progress_callback(4, 4);

    vasc_only
}

/// Spherical structuring element with pre-computed linear offsets for fast access
struct SphereKernel {
    /// Linear offsets for interior voxels (no bounds checking needed)
    linear_offsets: Vec<isize>,
    /// Coordinate offsets for boundary voxels (need bounds checking)
    coord_offsets: Vec<(i32, i32, i32)>,
    /// Radius of the sphere
    radius: usize,
}

impl SphereKernel {
    /// Create a spherical kernel with pre-computed offsets
    fn new(radius: i32, nx: usize, ny: usize) -> Self {
        let mut linear_offsets = Vec::new();
        let mut coord_offsets = Vec::new();
        let r2 = (radius * radius) as f64;
        let stride_y = nx as isize;
        let stride_z = (nx * ny) as isize;

        for dk in -radius..=radius {
            for dj in -radius..=radius {
                for di in -radius..=radius {
                    let dist2 = (di * di + dj * dj + dk * dk) as f64;
                    if dist2 <= r2 {
                        // Pre-compute linear offset: di + dj*nx + dk*nx*ny
                        let linear = di as isize + dj as isize * stride_y + dk as isize * stride_z;
                        linear_offsets.push(linear);
                        coord_offsets.push((di, dj, dk));
                    }
                }
            }
        }

        Self {
            linear_offsets,
            coord_offsets,
            radius: radius as usize,
        }
    }
}

/// Morphological bottom-hat filter (closing - original)
///
/// Enhances dark features smaller than the structuring element.
/// Uses true spherical structuring element for accuracy.
/// Optimized with pre-computed linear offsets and interior/boundary separation.
fn morphological_bottom_hat(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    radius: i32,
) -> Vec<f64> {
    // Pre-compute spherical kernel with linear offsets
    let kernel = SphereKernel::new(radius, nx, ny);

    // Closing = dilation followed by erosion with spherical SE
    let dilated = dilate_grayscale_optimized(data, nx, ny, nz, &kernel);
    let closed = erode_grayscale_optimized(&dilated, nx, ny, nz, &kernel);

    // Bottom-hat = closed - original
    closed.iter()
        .zip(data.iter())
        .map(|(&c, &d)| (c - d).max(0.0))
        .collect()
}

/// Optimized grayscale dilation with spherical structuring element
/// Uses pre-computed linear offsets and separates interior from boundary processing
fn dilate_grayscale_optimized(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    kernel: &SphereKernel,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut result = vec![0.0f64; n_total];
    let r = kernel.radius;
    let stride_z = nx * ny;

    // Process interior voxels (no bounds checking needed) - this is the fast path
    if nx > 2 * r && ny > 2 * r && nz > 2 * r {
        for k in r..(nz - r) {
            let k_offset = k * stride_z;
            for j in r..(ny - r) {
                let jk_offset = j * nx + k_offset;
                for i in r..(nx - r) {
                    let idx = i + jk_offset;
                    let mut max_val = f64::NEG_INFINITY;

                    // Fast path: use pre-computed linear offsets, no bounds checking
                    for &offset in &kernel.linear_offsets {
                        let neighbor_idx = (idx as isize + offset) as usize;
                        // Use unsafe for additional speed in inner loop
                        let val = unsafe { *data.get_unchecked(neighbor_idx) };
                        if val > max_val {
                            max_val = val;
                        }
                    }

                    unsafe { *result.get_unchecked_mut(idx) = max_val };
                }
            }
        }
    }

    // Process boundary voxels (need bounds checking) - slower but less frequent
    let nx_i32 = nx as i32;
    let ny_i32 = ny as i32;
    let nz_i32 = nz as i32;

    for k in 0..nz {
        let k_offset = k * stride_z;
        let k_is_boundary = k < r || k >= nz - r;

        for j in 0..ny {
            let jk_offset = j * nx + k_offset;
            let j_is_boundary = j < r || j >= ny - r;

            for i in 0..nx {
                let i_is_boundary = i < r || i >= nx - r;

                // Skip interior voxels (already processed)
                if !k_is_boundary && !j_is_boundary && !i_is_boundary {
                    continue;
                }

                let idx = i + jk_offset;
                let mut max_val = f64::NEG_INFINITY;

                // Slow path: need to check bounds for each neighbor
                for &(di, dj, dk) in &kernel.coord_offsets {
                    let ni = i as i32 + di;
                    let nj = j as i32 + dj;
                    let nk = k as i32 + dk;

                    if ni >= 0 && ni < nx_i32 &&
                       nj >= 0 && nj < ny_i32 &&
                       nk >= 0 && nk < nz_i32 {
                        let neighbor_idx = ni as usize + nj as usize * nx + nk as usize * stride_z;
                        max_val = max_val.max(data[neighbor_idx]);
                    }
                }

                result[idx] = if max_val.is_finite() { max_val } else { data[idx] };
            }
        }
    }

    result
}

/// Optimized grayscale erosion with spherical structuring element
fn erode_grayscale_optimized(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    kernel: &SphereKernel,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut result = vec![0.0f64; n_total];
    let r = kernel.radius;
    let stride_z = nx * ny;

    // Process interior voxels (no bounds checking needed) - this is the fast path
    if nx > 2 * r && ny > 2 * r && nz > 2 * r {
        for k in r..(nz - r) {
            let k_offset = k * stride_z;
            for j in r..(ny - r) {
                let jk_offset = j * nx + k_offset;
                for i in r..(nx - r) {
                    let idx = i + jk_offset;
                    let mut min_val = f64::INFINITY;

                    // Fast path: use pre-computed linear offsets, no bounds checking
                    for &offset in &kernel.linear_offsets {
                        let neighbor_idx = (idx as isize + offset) as usize;
                        let val = unsafe { *data.get_unchecked(neighbor_idx) };
                        if val < min_val {
                            min_val = val;
                        }
                    }

                    unsafe { *result.get_unchecked_mut(idx) = min_val };
                }
            }
        }
    }

    // Process boundary voxels (need bounds checking)
    let nx_i32 = nx as i32;
    let ny_i32 = ny as i32;
    let nz_i32 = nz as i32;

    for k in 0..nz {
        let k_offset = k * stride_z;
        let k_is_boundary = k < r || k >= nz - r;

        for j in 0..ny {
            let jk_offset = j * nx + k_offset;
            let j_is_boundary = j < r || j >= ny - r;

            for i in 0..nx {
                let i_is_boundary = i < r || i >= nx - r;

                // Skip interior voxels (already processed)
                if !k_is_boundary && !j_is_boundary && !i_is_boundary {
                    continue;
                }

                let idx = i + jk_offset;
                let mut min_val = f64::INFINITY;

                for &(di, dj, dk) in &kernel.coord_offsets {
                    let ni = i as i32 + di;
                    let nj = j as i32 + dj;
                    let nk = k as i32 + dk;

                    if ni >= 0 && ni < nx_i32 &&
                       nj >= 0 && nj < ny_i32 &&
                       nk >= 0 && nk < nz_i32 {
                        let neighbor_idx = ni as usize + nj as usize * nx + nk as usize * stride_z;
                        min_val = min_val.min(data[neighbor_idx]);
                    }
                }

                result[idx] = if min_val.is_finite() { min_val } else { data[idx] };
            }
        }
    }

    result
}


/// Simple wrapper with default parameters
pub fn generate_vasculature_mask_default(
    magnitude: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
) -> Vec<f64> {
    generate_vasculature_mask(magnitude, mask, nx, ny, nz, &VasculatureParams::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bottom_hat_basic() {
        // Constant image should give zero bottom-hat
        let data = vec![1.0f64; 27];
        let result = morphological_bottom_hat(&data, 3, 3, 3, 1);

        for &v in &result {
            assert!(v.abs() < 0.01, "Bottom-hat of constant should be ~0");
        }
    }

    #[test]
    fn test_vasculature_mask_basic() {
        // Basic test: should run without panic
        let magnitude = vec![1.0f64; 1000];
        let mask = vec![1u8; 1000];

        let params = VasculatureParams {
            sphere_radius: 2,
            frangi_scale_range: [1.0, 2.0],
            frangi_scale_ratio: 1.0,
            frangi_c: 100.0,
        };

        let result = generate_vasculature_mask(&magnitude, &mask, 10, 10, 10, &params);
        assert_eq!(result.len(), 1000);
    }
}

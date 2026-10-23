//! Vasculature Mask Generation for QSMART
//!
//! This module generates a binary mask identifying blood vessels in the brain.
//! The vasculature mask is used in QSMART's two-stage processing to separate
//! tissue from vessel regions, reducing streaking artifacts near veins.
//!
//! The algorithm:
//! 1. Apply morphological bottom-hat filtering to enhance dark tubular structures
//!    (MATLAB's `imbothat` with `strel('sphere', 8)`, edge handling included)
//! 2. Apply Frangi vesselness filter to detect vessels (single precision, as QSMART's
//!    `FrangiFilter3D` runs on its single-precision magnitude; see `frangi`)
//! 3. Threshold: vesselness above Otsu's threshold (MATLAB's `graythresh`) = vessel
//! 4. Return complementary mask (1 = tissue, 0 = vessel)
//!
//! ## Reference pipeline
//!
//! QSMART's `vasculature_mask.m` runs this on the mean over echoes of the coil-combined
//! magnitude after ANTs N4 (l. 8–14; [`crate::utils::qsmart_vessel_magnitude`]), restricted to
//! the BET brain mask (`FrangiFilter3D(test.*mask, …)`, l. 27), with `FrangiAlpha` = `FrangiBeta`
//! = 0.5, `BlackWhite` false and the demo's scales 0.5:0.5:6 and c = 500 (`Demo_QSMART.m`
//! l. 41–43; paper §3.4, 7T human, 0.6 mm). The paper's rodent runs used scales 0.01–0.5 in
//! steps of 0.1 and c = 50. The bottom-hat radius (8) and Frangi scales are in voxels, and
//! Otsu's threshold depends on the magnitude's intensity distribution, so the vessel mask
//! depends on that magnitude being the one QSMART uses.

use crate::error::InputError;
use crate::Grid;
use crate::utils::frangi::{FrangiParams, frangi_filter_3d};
use crate::utils::threshold::graythresh;

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
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `params` - Vasculature detection parameters
/// * `progress` - Progress callback `(current_step, total_steps)`
///
/// # Returns
/// Complementary vasculature mask (1 = tissue, 0 = vessel)
///
/// # Errors
/// If the Frangi parameters are invalid (see [`FrangiParams::validate`]).
pub fn generate_vasculature_mask(
    magnitude: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &VasculatureParams,
    progress_callback: impl Fn(usize, usize),
) -> Result<Vec<f64>, InputError> {
    let (nx, ny, nz) = grid.dims;
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

    let frangi_result = frangi_filter_3d(&masked_bottom_hat, grid, &frangi_params, |_, _| {})?;
    let enhanced = frangi_result.vesselness;

    // Step 4: Otsu threshold, as vasculature_mask.m does: `T = graythresh(enhanced)` over the
    // whole volume, vessels where `enhanced > T`, nothing outside the brain mask.
    progress_callback(3, 4);

    let threshold = graythresh(&enhanced);

    // Return complementary mask (1 = tissue, 0 = vessel)
    let vasc_only: Vec<f64> = enhanced.iter()
        .zip(mask.iter())
        .map(|(&v, &m)| if m != 0 && v > threshold { 0.0 } else { 1.0 })
        .collect();

    progress_callback(4, 4);

    Ok(vasc_only)
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

/// Morphological bottom-hat filter (closing - original), as MATLAB's
/// `imbothat(I, strel('sphere', radius))`.
///
/// Enhances dark features smaller than the structuring element. Like MATLAB's `imclose`,
/// the closing pads the volume with -Inf by the radius before dilating, so near the
/// volume's edge the erosion also takes the dilated padding into account (without the
/// padding, voxels within `radius` of the edge close to higher values; on QSMART's 7T
/// exemplar that changed 5,128 brain voxels). Matches `imbothat` exactly on QSMART's
/// exemplar and three 3T cases.
fn morphological_bottom_hat(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    radius: i32,
) -> Vec<f64> {
    let r = radius.max(0) as usize;
    let (px, py, pz) = (nx + 2 * r, ny + 2 * r, nz + 2 * r);
    let mut padded = vec![f64::NEG_INFINITY; px * py * pz];
    for k in 0..nz {
        for j in 0..ny {
            let src = j * nx + k * nx * ny;
            let dst = r + (j + r) * px + (k + r) * px * py;
            padded[dst..dst + nx].copy_from_slice(&data[src..src + nx]);
        }
    }
    let kernel = SphereKernel::new(radius, px, py);
    let dilated = dilate_grayscale_optimized(&padded, px, py, pz, &kernel);
    let closed = erode_grayscale_optimized(&dilated, px, py, pz, &kernel);
    let mut out = vec![0.0; data.len()];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = i + j * nx + k * nx * ny;
                let c = closed[i + r + (j + r) * px + (k + r) * px * py];
                out[idx] = (c - data[idx]).max(0.0);
            }
        }
    }
    out
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


#[cfg(test)]
mod tests {
    use super::*;

    fn grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 1.0, 1.0, 1.0)
    }

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
    fn bottom_hat_edge_matches_imbothat() {
        // MATLAB R2023b: rng(3); x = randi([0 9], 4, 4, 4); imbothat(x, strel('sphere', 1)).
        // In a volume this small every voxel is near the edge; closing without the padding
        // (imerode(imdilate(x, se), se)) differs at 33 of the 64 voxels.
        let x = [
            5., 7., 2., 5., 8., 8., 1., 2., 0., 4., 0., 4., 6., 2., 6., 5., 0., 5., 2., 4., 2., 6.,
            4., 1., 5., 7., 3., 2., 3., 9., 9., 6., 9., 8., 3., 0., 6., 5., 3., 2., 4., 4., 2., 2.,
            4., 8., 5., 2., 2., 4., 2., 2., 5., 0., 4., 3., 7., 7., 6., 6., 3., 6., 3., 5.,
        ];
        let mut expected = vec![0.0; 64];
        expected[26] = 3.0;
        expected[41] = 2.0;
        expected[42] = 3.0;
        assert_eq!(morphological_bottom_hat(&x, 4, 4, 4, 1), expected);
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

        let result = generate_vasculature_mask(&magnitude, &mask, &grid(10, 10, 10), &params, |_, _| {}).unwrap();
        assert_eq!(result.len(), 1000);
    }
}

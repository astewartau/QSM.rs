//! Mask utilities
//!
//! Provides functions for creating, eroding, dilating, and applying masks on 3D volumes.

use crate::Grid;

/// Create a binary sphere mask on a 3D volume
///
/// Generates a mask where voxels within the specified radius of the center
/// are set to 1, and all others are 0. Uses Fortran (column-major) ordering
/// to match NIfTI convention: index = x + y*nx + z*nx*ny.
///
/// # Arguments
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `center_x`, `center_y`, `center_z` - Sphere center in voxel coordinates
/// * `radius` - Sphere radius in voxels
///
/// # Returns
/// Flattened binary mask of length nx*ny*nz
pub fn create_sphere_mask(
    grid: &Grid,
    center_x: f64, center_y: f64, center_z: f64,
    radius: f64,
) -> Vec<u8> {
    let (nx, ny, nz) = grid.dims;
    let mut mask = vec![0u8; nx * ny * nz];
    let r2 = radius * radius;

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let dx = i as f64 - center_x;
                let dy = j as f64 - center_y;
                let dz = k as f64 - center_z;
                if dx * dx + dy * dy + dz * dz <= r2 {
                    mask[i + j * nx + k * nx * ny] = 1;
                }
            }
        }
    }

    mask
}

/// Zero out elements where mask is 0.
#[inline]
pub fn apply_mask_zero(data: &mut [f64], mask: &[u8]) {
    for i in 0..data.len() {
        if mask[i] == 0 {
            data[i] = 0.0;
        }
    }
}

/// Erode a binary mask by removing boundary voxels (6-connectivity).
///
/// Each iteration removes voxels that have any 6-connected neighbor equal to 0
/// or that sit on the volume boundary.
pub fn erode_mask(mask: &[u8], grid: &Grid, iterations: usize) -> Vec<u8> {
    let (nx, ny, nz) = grid.dims;
    let mut current = mask.to_vec();
    for _ in 0..iterations {
        let mut eroded = current.clone();
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = x + y * nx + z * nx * ny;
                    if current[idx] == 0 {
                        continue;
                    }
                    if x == 0
                        || x == nx - 1
                        || y == 0
                        || y == ny - 1
                        || z == 0
                        || z == nz - 1
                        || current[idx - 1] == 0
                        || current[idx + 1] == 0
                        || current[idx - nx] == 0
                        || current[idx + nx] == 0
                        || current[idx - nx * ny] == 0
                        || current[idx + nx * ny] == 0
                    {
                        eroded[idx] = 0;
                    }
                }
            }
        }
        current = eroded;
    }
    current
}

/// Dilate a binary mask by expanding into neighboring voxels (6-connectivity).
///
/// Each iteration adds voxels that have any 6-connected neighbor equal to 1.
pub fn dilate_mask(mask: &[u8], grid: &Grid, iterations: usize) -> Vec<u8> {
    let (nx, ny, nz) = grid.dims;
    let mut current = mask.to_vec();
    for _ in 0..iterations {
        let mut dilated = current.clone();
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = x + y * nx + z * nx * ny;
                    if current[idx] == 1 {
                        continue;
                    }
                    let has_neighbor = (x > 0 && current[idx - 1] == 1)
                        || (x < nx - 1 && current[idx + 1] == 1)
                        || (y > 0 && current[idx - nx] == 1)
                        || (y < ny - 1 && current[idx + nx] == 1)
                        || (z > 0 && current[idx - nx * ny] == 1)
                        || (z < nz - 1 && current[idx + nx * ny] == 1);
                    if has_neighbor {
                        dilated[idx] = 1;
                    }
                }
            }
        }
        current = dilated;
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 1.0, 1.0, 1.0)
    }

    #[test]
    fn test_sphere_mask_basic() {
        let mask = create_sphere_mask(&grid(10, 10, 10), 5.0, 5.0, 5.0, 3.0);
        assert_eq!(mask.len(), 1000);

        // Center voxel should be inside
        assert_eq!(mask[5 + 5 * 10 + 5 * 100], 1);

        // Corner should be outside
        assert_eq!(mask[0], 0);

        // Count should be reasonable for a sphere of radius 3
        let count: usize = mask.iter().map(|&m| m as usize).sum();
        assert!(count > 50 && count < 200, "Sphere voxel count {} seems wrong", count);
    }

    #[test]
    fn test_sphere_mask_non_cubic() {
        let mask = create_sphere_mask(&grid(20, 10, 5), 10.0, 5.0, 2.5, 2.0);
        assert_eq!(mask.len(), 1000);

        // Center should be inside
        assert_eq!(mask[10 + 5 * 20 + 2 * 20 * 10], 1);
    }

    #[test]
    fn test_sphere_mask_zero_radius() {
        let mask = create_sphere_mask(&grid(5, 5, 5), 2.0, 2.0, 2.0, 0.0);
        // Only the exact center voxel (distance 0 <= 0)
        let count: usize = mask.iter().map(|&m| m as usize).sum();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_apply_mask_zero() {
        let mask = vec![1, 0, 1, 0, 1];
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        apply_mask_zero(&mut data, &mask);
        assert_eq!(data, vec![1.0, 0.0, 3.0, 0.0, 5.0]);
    }

    #[test]
    fn test_erode_mask_cube() {
        // 3x3x3 all-ones mask: erosion should remove everything (all on boundary)
        let mask = vec![1u8; 27];
        let result = erode_mask(&mask, &grid(3, 3, 3), 1);
        assert_eq!(result.iter().filter(|&&v| v == 1).count(), 1); // only center
    }

    #[test]
    fn test_dilate_mask_single_voxel() {
        // Single voxel in center of 5x5x5: dilation should add 6 neighbors
        let mut mask = vec![0u8; 125];
        mask[2 + 2 * 5 + 2 * 25] = 1; // center
        let result = dilate_mask(&mask, &grid(5, 5, 5), 1);
        assert_eq!(result.iter().filter(|&&v| v == 1).count(), 7); // center + 6 neighbors
    }
}

//! Mask generation utilities
//!
//! Provides functions for creating geometric masks (e.g. spheres) on 3D volumes.

/// Create a binary sphere mask on a 3D volume
///
/// Generates a mask where voxels within the specified radius of the center
/// are set to 1, and all others are 0. Uses Fortran (column-major) ordering
/// to match NIfTI convention: index = x + y*nx + z*nx*ny.
///
/// # Arguments
/// * `nx`, `ny`, `nz` - Volume dimensions
/// * `center_x`, `center_y`, `center_z` - Sphere center in voxel coordinates
/// * `radius` - Sphere radius in voxels
///
/// # Returns
/// Flattened binary mask of length nx*ny*nz
pub fn create_sphere_mask(
    nx: usize, ny: usize, nz: usize,
    center_x: f64, center_y: f64, center_z: f64,
    radius: f64,
) -> Vec<u8> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_mask_basic() {
        let mask = create_sphere_mask(10, 10, 10, 5.0, 5.0, 5.0, 3.0);
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
        let mask = create_sphere_mask(20, 10, 5, 10.0, 5.0, 2.5, 2.0);
        assert_eq!(mask.len(), 1000);

        // Center should be inside
        assert_eq!(mask[10 + 5 * 20 + 2 * 20 * 10], 1);
    }

    #[test]
    fn test_sphere_mask_zero_radius() {
        let mask = create_sphere_mask(5, 5, 5, 2.0, 2.0, 2.0, 0.0);
        // Only the exact center voxel (distance 0 <= 0)
        let count: usize = mask.iter().map(|&m| m as usize).sum();
        assert_eq!(count, 1);
    }
}

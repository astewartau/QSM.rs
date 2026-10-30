//! Binary mask morphology on 3D volumes (Fortran order, `x + y*nx + z*nx*ny`).
//!
//! - [`erode_mask`] / [`dilate_mask`] — iterated 6-connected erosion and dilation
//! - [`morphological_close`] — closing with a spherical structuring element
//! - [`fill_holes`] — fill enclosed background components up to a size limit
//! - [`signal_gated_erosion`] — erode only low-signal boundary voxels ([`signal_erosion`])

use crate::Grid;
use crate::utils::bias_correction::{check_volume, idx3d};

pub mod signal_erosion;
pub use signal_erosion::{signal_gated_erosion, SignalErosionParams};

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

/// Erode a binary mask with a spherical structuring element of `radius` voxels
/// (MATLAB `imerode(mask, strel('sphere', radius))`). Voxels whose ball crosses the
/// volume edge are removed.
pub(crate) fn erode_ball(mask: &[u8], nx: usize, ny: usize, nz: usize, radius: i32) -> Vec<u8> {
    let n_total = nx * ny * nz;
    let mut eroded = vec![0u8; n_total];

    let idx = |i: usize, j: usize, k: usize| i + j * nx + k * nx * ny;

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if mask[idx(i, j, k)] == 0 {
                    continue;
                }

                let mut all_inside = true;

                'outer: for dz in -radius..=radius {
                    for dy in -radius..=radius {
                        for dx in -radius..=radius {
                            let dist2 = dx * dx + dy * dy + dz * dz;
                            if dist2 > radius * radius {
                                continue;
                            }

                            let ni = i as i32 + dx;
                            let nj = j as i32 + dy;
                            let nk = k as i32 + dz;

                            if ni < 0 || ni >= nx as i32 ||
                               nj < 0 || nj >= ny as i32 ||
                               nk < 0 || nk >= nz as i32 {
                                all_inside = false;
                                break 'outer;
                            }

                            if mask[idx(ni as usize, nj as usize, nk as usize)] == 0 {
                                all_inside = false;
                                break 'outer;
                            }
                        }
                    }
                }

                if all_inside {
                    eroded[idx(i, j, k)] = 1;
                }
            }
        }
    }

    eroded
}

/// Dilate a binary mask with a spherical structuring element of `radius` voxels
/// (MATLAB `imdilate(mask, strel('sphere', radius))`).
pub(crate) fn dilate_ball(mask: &[u8], nx: usize, ny: usize, nz: usize, radius: i32) -> Vec<u8> {
    let n_total = nx * ny * nz;
    let mut dilated = vec![0u8; n_total];

    let idx = |i: usize, j: usize, k: usize| i + j * nx + k * nx * ny;

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if mask[idx(i, j, k)] != 0 {
                    // Set all neighbors within radius
                    for dz in -radius..=radius {
                        for dy in -radius..=radius {
                            for dx in -radius..=radius {
                                let dist2 = dx * dx + dy * dy + dz * dz;
                                if dist2 > radius * radius {
                                    continue;
                                }

                                let ni = i as i32 + dx;
                                let nj = j as i32 + dy;
                                let nk = k as i32 + dz;

                                if ni >= 0 && ni < nx as i32 &&
                                   nj >= 0 && nj < ny as i32 &&
                                   nk >= 0 && nk < nz as i32 {
                                    dilated[idx(ni as usize, nj as usize, nk as usize)] = 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    dilated
}

/// Morphological closing (dilation followed by erosion) with a spherical structuring element
/// of `radius` voxels.
pub fn morphological_close(mask: &[u8], grid: &Grid, radius: i32) -> Vec<u8> {
    let (nx, ny, nz) = grid.dims;
    let dilated = dilate_ball(mask, nx, ny, nz, radius);
    erode_ball(&dilated, nx, ny, nz, radius)
}

/// Find connected component using flood fill (6-connectivity in 3D)
fn flood_fill_component(
    mask: &[u8],
    visited: &mut [bool],
    start: usize,
    nx: usize, ny: usize, nz: usize,
) -> Vec<usize> {
    let mut component = Vec::new();
    let mut stack = vec![start];

    while let Some(idx) = stack.pop() {
        if visited[idx] || mask[idx] != 0 {
            continue;
        }

        visited[idx] = true;
        component.push(idx);

        // Get 3D coordinates
        let k = idx / (nx * ny);
        let rem = idx % (nx * ny);
        let j = rem / nx;
        let i = rem % nx;

        // 6-connectivity neighbors
        if i > 0 {
            let n = idx3d(i - 1, j, k, nx, ny);
            if !visited[n] && mask[n] == 0 { stack.push(n); }
        }
        if i + 1 < nx {
            let n = idx3d(i + 1, j, k, nx, ny);
            if !visited[n] && mask[n] == 0 { stack.push(n); }
        }
        if j > 0 {
            let n = idx3d(i, j - 1, k, nx, ny);
            if !visited[n] && mask[n] == 0 { stack.push(n); }
        }
        if j + 1 < ny {
            let n = idx3d(i, j + 1, k, nx, ny);
            if !visited[n] && mask[n] == 0 { stack.push(n); }
        }
        if k > 0 {
            let n = idx3d(i, j, k - 1, nx, ny);
            if !visited[n] && mask[n] == 0 { stack.push(n); }
        }
        if k + 1 < nz {
            let n = idx3d(i, j, k + 1, nx, ny);
            if !visited[n] && mask[n] == 0 { stack.push(n); }
        }
    }

    component
}

/// Fill holes in a binary mask
///
/// Matches MriResearchTools.jl's fill_holes function (771eadf; to the voxel against it in the
/// verified QSM-CI `julia-qsm` image, `tests/mritools_parity.rs`).
/// Fills connected components of zeros (holes) up to max_hole_size.
/// Uses 6-connectivity for 3D.
pub fn fill_holes(mask: &[u8], grid: &Grid, max_hole_size: usize) -> Vec<u8> {
    let (nx, ny, nz) = grid.dims;
    let n_total = check_volume("fill_holes: mask", mask.len(), grid);
    let mut result = mask.to_vec();
    let mut visited = vec![false; n_total];

    // Find all connected components of zeros (potential holes)
    for idx in 0..n_total {
        if mask[idx] == 0 && !visited[idx] {
            let component = flood_fill_component(mask, &mut visited, idx, nx, ny, nz);

            // Check if this component touches the boundary
            let mut touches_boundary = false;
            for &cidx in &component {
                let k = cidx / (nx * ny);
                let rem = cidx % (nx * ny);
                let j = rem / nx;
                let i = rem % nx;

                if i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1 {
                    touches_boundary = true;
                    break;
                }
            }

            // Fill if it's a hole (doesn't touch boundary) and small enough
            if !touches_boundary && component.len() <= max_hole_size {
                for cidx in component {
                    result[cidx] = 1;
                }
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

    /// Create a solid sphere mask centered in an n x n x n volume.
    fn make_sphere_mask(n: usize, radius: f64) -> Vec<u8> {
        let center = n as f64 / 2.0;
        let mut mask = vec![0u8; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let dx = i as f64 - center;
                    let dy = j as f64 - center;
                    let dz = k as f64 - center;
                    if (dx * dx + dy * dy + dz * dz).sqrt() < radius {
                        mask[i + j * n + k * n * n] = 1;
                    }
                }
            }
        }
        mask
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

    #[test]
    fn test_erode_ball() {
        // 5x5x5 solid cube
        let mask = vec![1u8; 125];
        let eroded = erode_ball(&mask, 5, 5, 5, 1);

        // Center 3x3x3 should remain
        let count: usize = eroded.iter().map(|&v| v as usize).sum();
        assert!(count > 0);
        assert!(count < 125);
    }

    #[test]
    fn test_dilate_ball() {
        // Single center voxel in 5x5x5
        let mut mask = vec![0u8; 125];
        mask[62] = 1; // Center

        let dilated = dilate_ball(&mask, 5, 5, 5, 1);

        // Should expand to 6-connectivity
        let count: usize = dilated.iter().map(|&v| v as usize).sum();
        assert!(count >= 7); // At least 7 voxels (center + 6 neighbors)
    }

    // =====================================================================
    // Tests for erode_ball
    // =====================================================================

    #[test]
    fn test_erode_ball_sphere() {
        let n = 10;
        let mask = make_sphere_mask(n, 4.0);
        let eroded = erode_ball(&mask, n, n, n, 1);

        let orig_count: usize = mask.iter().map(|&v| v as usize).sum();
        let eroded_count: usize = eroded.iter().map(|&v| v as usize).sum();
        assert!(
            eroded_count < orig_count,
            "Eroded sphere should be smaller: {} < {}",
            eroded_count,
            orig_count
        );
        assert!(eroded_count > 0, "Eroded sphere should not be empty");

        // Center should still be in eroded mask
        let center = n / 2 + (n / 2) * n + (n / 2) * n * n;
        assert_eq!(eroded[center], 1, "Center should survive erosion");
    }

    #[test]
    fn test_erode_ball_single_voxel() {
        // A single voxel should be eroded away
        let mut mask = vec![0u8; 125];
        mask[62] = 1; // center of 5x5x5
        let eroded = erode_ball(&mask, 5, 5, 5, 1);
        let count: usize = eroded.iter().map(|&v| v as usize).sum();
        assert_eq!(count, 0, "Single voxel should be fully eroded");
    }

    // =====================================================================
    // Tests for dilate_ball
    // =====================================================================

    #[test]
    fn test_dilate_ball_sphere() {
        let n = 10;
        let mask = make_sphere_mask(n, 3.0);
        let dilated = dilate_ball(&mask, n, n, n, 1);

        let orig_count: usize = mask.iter().map(|&v| v as usize).sum();
        let dilated_count: usize = dilated.iter().map(|&v| v as usize).sum();
        assert!(
            dilated_count > orig_count,
            "Dilated sphere should be larger: {} > {}",
            dilated_count,
            orig_count
        );
    }

    #[test]
    fn test_dilate_ball_radius_2() {
        let mut mask = vec![0u8; 125];
        mask[62] = 1; // center of 5x5x5
        let dilated = dilate_ball(&mask, 5, 5, 5, 2);
        let count: usize = dilated.iter().map(|&v| v as usize).sum();
        // Should be more than radius=1 dilation
        assert!(count > 7, "Radius-2 dilation should produce more than 7 voxels, got {}", count);
    }

    // =====================================================================
    // Tests for morphological_close
    // =====================================================================

    #[test]
    fn test_morphological_close_fills_small_gaps() {
        let n = 10;
        let mut mask = make_sphere_mask(n, 4.0);
        // Remove a surface voxel to create a small gap
        let eroded = erode_ball(&mask, n, n, n, 1);
        let surface: Vec<usize> = (0..mask.len()).filter(|&i| mask[i] != 0 && eroded[i] == 0).collect();
        if !surface.is_empty() {
            mask[surface[0]] = 0;
        }

        let closed = morphological_close(&mask, &grid(n, n, n), 1);
        let orig_count: usize = mask.iter().map(|&v| v as usize).sum();
        let closed_count: usize = closed.iter().map(|&v| v as usize).sum();
        // Closing should recover the gap or at least not shrink significantly
        assert!(
            closed_count >= orig_count,
            "Closing should not reduce mask size: {} vs {}",
            closed_count,
            orig_count
        );
    }

    #[test]
    fn test_morphological_close_empty() {
        let mask = vec![0u8; 27];
        let closed = morphological_close(&mask, &grid(3, 3, 3), 1);
        let count: usize = closed.iter().map(|&v| v as usize).sum();
        assert_eq!(count, 0, "Closing empty mask should stay empty");
    }

    // =====================================================================
    // Tests for fill_holes
    // =====================================================================

    #[test]
    fn test_fill_holes_basic() {
        // 3x3x3 cube with a hole in the center
        let mut mask = vec![1u8; 27];
        mask[13] = 0; // center voxel

        let filled = fill_holes(&mask, &grid(3, 3, 3), 5);
        assert_eq!(filled[13], 1, "Center hole should be filled");
    }

    // =====================================================================
    // Tests for fill_holes (more complex cases)
    // =====================================================================

    #[test]
    fn test_fill_holes_boundary_hole_not_filled() {
        // A hole touching the boundary should NOT be filled
        let mut mask = vec![1u8; 125]; // 5x5x5
        mask[0] = 0; // corner voxel - touches boundary
        let filled = fill_holes(&mask, &grid(5, 5, 5), 100);
        assert_eq!(filled[0], 0, "Boundary hole should not be filled");
    }

    #[test]
    fn test_fill_holes_large_hole_not_filled() {
        // A hole larger than max_hole_size should NOT be filled
        let n = 7;
        let n_total = n * n * n;
        let mut mask = vec![1u8; n_total];
        // Create a large internal hole (3x3x3 = 27 voxels)
        for k in 2..5 {
            for j in 2..5 {
                for i in 2..5 {
                    mask[i + j * n + k * n * n] = 0;
                }
            }
        }
        let filled = fill_holes(&mask, &grid(n, n, n), 5); // max_hole_size=5, hole is 27
        // Hole should NOT be filled because it's too large
        let center = 3 + 3 * n + 3 * n * n;
        assert_eq!(filled[center], 0, "Large hole should not be filled");
    }

    #[test]
    #[should_panic(expected = "fill_holes: mask has 216 voxels but the grid is 8x8x8 (512 voxels)")]
    fn fill_holes_refuses_mask_not_matching_grid() {
        fill_holes(&vec![1u8; 6 * 6 * 6], &grid(8, 8, 8), 10);
    }

    // =====================================================================
    // Tests for flood_fill_component
    // =====================================================================

    #[test]
    fn test_flood_fill_component_single() {
        let mask = vec![1u8; 27]; // 3x3x3 all filled => no zeros
        let mut visited = vec![false; 27];
        // Start at a filled voxel -> should find nothing (mask[idx] != 0)
        let component = flood_fill_component(&mask, &mut visited, 0, 3, 3, 3);
        assert!(component.is_empty(), "All filled mask should have no zero component");
    }

    #[test]
    fn test_flood_fill_component_connected_zeros() {
        let n = 5;
        let n_total = n * n * n;
        let mut mask = vec![1u8; n_total];
        // Create a connected line of zeros along x at j=2, k=2
        for i in 1..4 {
            mask[i + 2 * n + 2 * n * n] = 0;
        }
        let mut visited = vec![false; n_total];
        let start = 1 + 2 * n + 2 * n * n;
        let component = flood_fill_component(&mask, &mut visited, start, n, n, n);
        assert_eq!(component.len(), 3, "Should find 3 connected zero voxels");
    }
}

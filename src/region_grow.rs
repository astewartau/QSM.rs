use crate::priority_queue::BucketQueue;
use std::f64::consts::PI;

/// Neighbor offsets: (dimension_for_weight, di, dj, dk)
/// dimension 0 = x edges, 1 = y edges, 2 = z edges
const NEIGHBOR_OFFSETS: [(usize, i32, i32, i32); 6] = [
    (0, 1, 0, 0),
    (0, -1, 0, 0),
    (1, 0, 1, 0),
    (1, 0, -1, 0),
    (2, 0, 0, 1),
    (2, 0, 0, -1),
];

const TWO_PI: f64 = 2.0 * PI;

/// Convert 3D index to flat index (Fortran order / column-major, matches NIfTI)
/// For array shape (nx, nx, ny), index [i,j,k] maps to: i + j*nx + k*nx*ny
#[inline(always)]
fn idx3d(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
    i + j * nx + k * nx * ny
}

/// Convert 4D index (for weights array) to flat index (Fortran order)
/// weights layout: [dim][i][j][k] where dim is 0,1,2 for x,y,z
/// For array shape (3, nx, nx, ny), index [dim,i,j,k] maps to: i + j*nx + k*nx*ny + dim*nx*ny*nz
#[inline(always)]
fn idx4d(dim: usize, i: usize, j: usize, k: usize, nx: usize, ny: usize, nz: usize) -> usize {
    i + j * nx + k * nx * ny + dim * nx * ny * nz
}

/// Queue item: (target_i, target_j, target_k, ref_i, ref_j, ref_k)
/// Stores both the target voxel to unwrap AND the reference voxel to use
type QueueItem = (usize, usize, usize, usize, usize, usize);

/// Region growing phase unwrapping (matches Python implementation exactly)
///
/// # Arguments
/// * `phase` - Mutable slice of phase values (nx * ny * nz), will be modified in-place
/// * `weights` - Weight values (3 * nx * ny * nz), layout [dim][x][y][z]
/// * `mask` - Boolean mask (nx * ny * nz), 1 = process, 0 = skip
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `seed_i`, `seed_j`, `seed_k` - Seed point coordinates
///
/// # Returns
/// Number of voxels processed
pub fn grow_region_unwrap(
    phase: &mut [f64],
    weights: &[u8],
    mask: &mut [u8],  // Used as visited array (modified in-place)
    nx: usize,
    ny: usize,
    nz: usize,
    seed_i: usize,
    seed_j: usize,
    seed_k: usize,
) -> usize {
    let mut pq: BucketQueue<QueueItem> = BucketQueue::new(256);
    let mut processed = 0usize;

    // Mark seed as visited
    let seed_idx = idx3d(seed_i, seed_j, seed_k, nx, ny);

    // If mask[seed] is 0, seed is not in ROI - find alternative
    if mask[seed_idx] == 0 {
        return 0;
    }

    // Use mask as visited array: 2 = visited, 1 = in ROI but not visited, 0 = not in ROI
    mask[seed_idx] = 2;
    processed += 1;

    // Add initial edges from seed (matching Python's queue item format)
    for &(dim, di, dj, dk) in &NEIGHBOR_OFFSETS {
        let ni = seed_i as i32 + di;
        let nj = seed_j as i32 + dj;
        let nk = seed_k as i32 + dk;

        if ni >= 0 && ni < nx as i32 && nj >= 0 && nj < ny as i32 && nk >= 0 && nk < nz as i32 {
            let ni = ni as usize;
            let nj = nj as usize;
            let nk = nk as usize;
            let n_idx = idx3d(ni, nj, nk, nx, ny);

            // Check if neighbor is in mask and not visited
            if mask[n_idx] == 1 {
                // Get weight at edge (min coordinates)
                let ei = seed_i.min(ni);
                let ej = seed_j.min(nj);
                let ek = seed_k.min(nk);
                let weight = weights[idx4d(dim, ei, ej, ek, nx, ny, nz)] as usize;

                if weight > 0 {
                    // Store (target, reference) coordinates like Python
                    pq.push(weight, (ni, nj, nk, seed_i, seed_j, seed_k));
                }
            }
        }
    }

    // Main region growing loop
    while let Some((ni, nj, nk, oi, oj, ok)) = pq.pop() {
        let n_idx = idx3d(ni, nj, nk, nx, ny);

        // Skip if already visited
        if mask[n_idx] != 1 {
            continue;
        }

        // Unwrap using stored reference (exact Python match)
        let new_val = phase[n_idx];
        let old_val = phase[idx3d(oi, oj, ok, nx, ny)];

        // Unwrap: new_val - 2π * round((new_val - old_val) / 2π)
        let diff = new_val - old_val;
        let n_wraps = (diff / TWO_PI).round();
        phase[n_idx] = new_val - TWO_PI * n_wraps;

        // Mark as visited
        mask[n_idx] = 2;
        processed += 1;

        // Add new edges to unvisited neighbors
        for &(dim, di, dj, dk) in &NEIGHBOR_OFFSETS {
            let nni = ni as i32 + di;
            let nnj = nj as i32 + dj;
            let nnk = nk as i32 + dk;

            if nni >= 0 && nni < nx as i32 && nnj >= 0 && nnj < ny as i32 && nnk >= 0 && nnk < nz as i32 {
                let nni = nni as usize;
                let nnj = nnj as usize;
                let nnk = nnk as usize;
                let nn_idx = idx3d(nni, nnj, nnk, nx, ny);

                // Only add if in mask and not visited
                if mask[nn_idx] == 1 {
                    let ei = ni.min(nni);
                    let ej = nj.min(nnj);
                    let ek = nk.min(nnk);
                    let weight = weights[idx4d(dim, ei, ej, ek, nx, ny, nz)] as usize;

                    if weight > 0 {
                        // Store current voxel as reference for neighbor
                        pq.push(weight, (nni, nnj, nnk, ni, nj, nk));
                    }
                }
            }
        }
    }

    processed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_unwrap() {
        // 3x3x3 test case
        let nx = 3;
        let ny = 3;
        let nz = 3;

        // Create wrapped phase with a 2π jump
        let mut phase = vec![0.0f64; nx * ny * nz];
        phase[idx3d(0, 0, 0, nx, ny)] = 0.0;
        phase[idx3d(1, 0, 0, nx, ny)] = 0.1;
        phase[idx3d(2, 0, 0, nx, ny)] = 0.2 - TWO_PI; // Wrapped value

        // All weights = 255 (high quality)
        let weights = vec![255u8; 3 * nx * ny * nz];

        // All voxels in mask (1 = in ROI, not visited)
        let mut mask = vec![1u8; nx * ny * nz];

        // Unwrap from center
        let processed = grow_region_unwrap(&mut phase, &weights, &mut mask, nx, ny, nz, 1, 1, 1);

        assert!(processed > 0);

        // Check that the wrapped value was unwrapped
        let unwrapped_val = phase[idx3d(2, 0, 0, nx, ny)];
        assert!((unwrapped_val - 0.2).abs() < 0.5, "Expected ~0.2, got {}", unwrapped_val);
    }
}

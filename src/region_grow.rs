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

/// Region growing phase unwrapping starting from pre-visited boundary.
///
/// Similar to [`grow_region_unwrap`] but starts from all boundary edges between
/// visited (mask=2) and unvisited (mask=1) voxels simultaneously. This is used
/// for the temporal uncertain unwrapping fallback in ROMEO multi-echo mode,
/// where certain voxels act as seeds for re-unwrapping uncertain neighbors.
///
/// # Arguments
/// * `phase` - Mutable slice of phase values, modified in-place
/// * `weights` - Weight values (3 * nx * ny * nz), layout [dim][x][y][z]
/// * `mask` - Visited array: 0 = not in ROI, 1 = uncertain (to unwrap), 2 = certain (seed)
/// * `nx`, `ny`, `nz` - Array dimensions
///
/// # Returns
/// Number of voxels processed
pub fn grow_region_unwrap_from_visited(
    phase: &mut [f64],
    weights: &[u8],
    mask: &mut [u8],
    nx: usize,
    ny: usize,
    nz: usize,
) -> usize {
    let mut pq: BucketQueue<QueueItem> = BucketQueue::new(256);
    let mut processed = 0usize;

    // Find all boundary edges: visited (2) → unvisited (1)
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = idx3d(i, j, k, nx, ny);
                if mask[idx] != 2 {
                    continue;
                }

                for &(dim, di, dj, dk) in &NEIGHBOR_OFFSETS {
                    let ni = i as i32 + di;
                    let nj = j as i32 + dj;
                    let nk = k as i32 + dk;

                    if ni >= 0 && ni < nx as i32 && nj >= 0 && nj < ny as i32 && nk >= 0 && nk < nz as i32 {
                        let ni = ni as usize;
                        let nj = nj as usize;
                        let nk = nk as usize;
                        let n_idx = idx3d(ni, nj, nk, nx, ny);

                        if mask[n_idx] == 1 {
                            let ei = i.min(ni);
                            let ej = j.min(nj);
                            let ek = k.min(nk);
                            let weight = weights[idx4d(dim, ei, ej, ek, nx, ny, nz)] as usize;

                            if weight > 0 {
                                pq.push(weight, (ni, nj, nk, i, j, k));
                            }
                        }
                    }
                }
            }
        }
    }

    // Same growth loop as grow_region_unwrap
    while let Some((ni, nj, nk, oi, oj, ok)) = pq.pop() {
        let n_idx = idx3d(ni, nj, nk, nx, ny);

        if mask[n_idx] != 1 {
            continue;
        }

        let new_val = phase[n_idx];
        let old_val = phase[idx3d(oi, oj, ok, nx, ny)];
        let diff = new_val - old_val;
        let n_wraps = (diff / TWO_PI).round();
        phase[n_idx] = new_val - TWO_PI * n_wraps;

        mask[n_idx] = 2;
        processed += 1;

        for &(dim, di, dj, dk) in &NEIGHBOR_OFFSETS {
            let nni = ni as i32 + di;
            let nnj = nj as i32 + dj;
            let nnk = nk as i32 + dk;

            if nni >= 0 && nni < nx as i32 && nnj >= 0 && nnj < ny as i32 && nnk >= 0 && nnk < nz as i32 {
                let nni = nni as usize;
                let nnj = nnj as usize;
                let nnk = nnk as usize;
                let nn_idx = idx3d(nni, nnj, nnk, nx, ny);

                if mask[nn_idx] == 1 {
                    let ei = ni.min(nni);
                    let ej = nj.min(nnj);
                    let ek = nk.min(nnk);
                    let weight = weights[idx4d(dim, ei, ej, ek, nx, ny, nz)] as usize;

                    if weight > 0 {
                        pq.push(weight, (nni, nnj, nnk, ni, nj, nk));
                    }
                }
            }
        }
    }

    processed
}

/// Full-featured region growing phase unwrapping matching ROMEO.jl.
///
/// Supports wrap_addition (linear extrapolation), multi-seed auto-selection,
/// region tracking, and multi-echo seed correction.
///
/// # Arguments
/// * `phase` - Mutable phase values, modified in-place
/// * `weights` - Edge weights (3 * nx * ny * nz), 0 = no edge, 1-255 = valid
/// * `mask` - Read-only binary mask (>0 = in ROI)
/// * `visited` - Output region IDs (0 = unvisited, 1..255 = region)
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `wrap_addition` - Extra phase tolerance for linear unwrapping [0, π]
/// * `max_seeds` - Maximum number of seed regions (1-255)
/// * `seed_phase2` - Optional second echo phase for multi-echo seed correction
/// * `seed_tes` - Optional (te1, te2) for multi-echo seed correction
///
/// # Returns
/// Number of regions created
pub fn grow_region_unwrap_full(
    phase: &mut [f64],
    weights: &[u8],
    mask: &[u8],
    visited: &mut [u8],
    nx: usize,
    ny: usize,
    nz: usize,
    wrap_addition: f64,
    max_seeds: u8,
    seed_phase2: Option<&[f64]>,
    seed_tes: Option<(f64, f64)>,
) -> u8 {
    let n_total = nx * ny * nz;
    let max_seeds = max_seeds.max(1);
    let mut pq: BucketQueue<QueueItem> = BucketQueue::new(256);
    let mut num_seeds: u8 = 0;
    let mut seed_thresh: usize = 0;

    // Build sorted seed candidates: voxels ordered by sum of outgoing edge weights (descending)
    let mut seed_candidates: Vec<(u32, usize, usize, usize)> = Vec::new();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = idx3d(i, j, k, nx, ny);
                if mask[idx] == 0 { continue; }
                let mut sum = 0u32;
                // Sum weights in 3 positive directions from this voxel
                for dim in 0..3 {
                    let (ni, nj, nk) = match dim {
                        0 => (i + 1, j, k),
                        1 => (i, j + 1, k),
                        _ => (i, j, k + 1),
                    };
                    if ni < nx && nj < ny && nk < nz {
                        let w = weights[idx4d(dim, i, j, k, nx, ny, nz)];
                        sum += w as u32;
                    }
                }
                seed_candidates.push((sum, i, j, k));
            }
        }
    }
    seed_candidates.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    let mut seed_cursor = 0;

    // Add first seed
    if !add_seed_full(
        phase, weights, mask, visited, &mut pq,
        &seed_candidates, &mut seed_cursor, &mut num_seeds,
        &mut seed_thresh,
        seed_phase2, seed_tes,
        nx, ny, nz,
    ) {
        return 0;
    }

    // MST loop
    loop {
        // Check if we need a new seed before popping
        if num_seeds < max_seeds {
            if let Some(best_prio) = pq.peek_max_priority() {
                if best_prio < seed_thresh {
                    add_seed_full(
                        phase, weights, mask, visited, &mut pq,
                        &seed_candidates, &mut seed_cursor, &mut num_seeds,
                        &mut seed_thresh,
                        seed_phase2, seed_tes,
                        nx, ny, nz,
                    );
                }
            }
        }

        let (ni, nj, nk, oi, oj, ok) = match pq.pop() {
            Some(item) => item,
            None => break,
        };

        let n_idx = idx3d(ni, nj, nk, nx, ny);
        if visited[n_idx] != 0 || mask[n_idx] == 0 {
            continue;
        }

        // Unwrap with wrap_addition
        let o_idx = idx3d(oi, oj, ok, nx, ny);
        unwrap_edge_wa(phase, visited, n_idx, o_idx, ni, nj, nk, oi, oj, ok, wrap_addition, nx, ny, nz);

        // Assign same region as reference
        visited[n_idx] = visited[o_idx];

        // Add neighbors to queue
        for &(dim, di, dj, dk) in &NEIGHBOR_OFFSETS {
            let nni = ni as i32 + di;
            let nnj = nj as i32 + dj;
            let nnk = nk as i32 + dk;

            if nni >= 0 && nni < nx as i32 && nnj >= 0 && nnj < ny as i32 && nnk >= 0 && nnk < nz as i32 {
                let nni = nni as usize;
                let nnj = nnj as usize;
                let nnk = nnk as usize;
                let nn_idx = idx3d(nni, nnj, nnk, nx, ny);

                if mask[nn_idx] > 0 && visited[nn_idx] == 0 {
                    let ei = ni.min(nni);
                    let ej = nj.min(nnj);
                    let ek = nk.min(nnk);
                    let weight = weights[idx4d(dim, ei, ej, ek, nx, ny, nz)] as usize;

                    if weight > 0 {
                        pq.push(weight, (nni, nnj, nnk, ni, nj, nk));
                    }
                }
            }
        }
    }

    num_seeds
}

/// Add a new seed from the candidate list.
///
/// Returns true if a seed was added, false if no more candidates.
fn add_seed_full(
    phase: &mut [f64],
    weights: &[u8],
    mask: &[u8],
    visited: &mut [u8],
    pq: &mut BucketQueue<QueueItem>,
    candidates: &[(u32, usize, usize, usize)],
    cursor: &mut usize,
    num_seeds: &mut u8,
    seed_thresh: &mut usize,
    seed_phase2: Option<&[f64]>,
    seed_tes: Option<(f64, f64)>,
    nx: usize, ny: usize, nz: usize,
) -> bool {
    // Find next unvisited candidate
    while *cursor < candidates.len() {
        let (quality, si, sj, sk) = candidates[*cursor];
        *cursor += 1;
        let idx = idx3d(si, sj, sk, nx, ny);
        if mask[idx] > 0 && visited[idx] == 0 {
            *num_seeds += 1;
            visited[idx] = *num_seeds;

            // Seed correction: multi-echo aware if phase2/TEs available
            seed_correction(phase, idx, seed_phase2, seed_tes);

            // Add seed's neighbors to queue
            for &(dim, di, dj, dk) in &NEIGHBOR_OFFSETS {
                let ni = si as i32 + di;
                let nj = sj as i32 + dj;
                let nk = sk as i32 + dk;
                if ni >= 0 && ni < nx as i32 && nj >= 0 && nj < ny as i32 && nk >= 0 && nk < nz as i32 {
                    let ni = ni as usize;
                    let nj = nj as usize;
                    let nk = nk as usize;
                    let n_idx = idx3d(ni, nj, nk, nx, ny);
                    if mask[n_idx] > 0 && visited[n_idx] == 0 {
                        let ei = si.min(ni);
                        let ej = sj.min(nj);
                        let ek = sk.min(nk);
                        let weight = weights[idx4d(dim, ei, ej, ek, nx, ny, nz)] as usize;
                        if weight > 0 {
                            pq.push(weight, (ni, nj, nk, si, sj, sk));
                        }
                    }
                }
            }

            // Compute new seed threshold: avg quality / 2 (in our convention, higher = better)
            *seed_thresh = (quality as usize) / 6; // quality is sum of 3 weights, /6 = avg/2
            return true;
        }
    }
    false
}

/// Unwrap edge with wrap_addition (linear extrapolation).
///
/// Matches ROMEO.jl `unwrapedge!`: looks at the voxel behind the reference
/// to predict the phase trend, clamped by wrap_addition.
fn unwrap_edge_wa(
    phase: &mut [f64],
    visited: &[u8],
    new_idx: usize, old_idx: usize,
    ni: usize, nj: usize, nk: usize,
    oi: usize, oj: usize, ok: usize,
    wrap_addition: f64,
    nx: usize, ny: usize, nz: usize,
) {
    let new_val = phase[new_idx];
    let old_val = phase[old_idx];

    let mut d = 0.0;
    if wrap_addition > 0.0 {
        // Voxel "behind" old: mirror of new through old
        let bhi = 2 * oi as i32 - ni as i32;
        let bhj = 2 * oj as i32 - nj as i32;
        let bhk = 2 * ok as i32 - nk as i32;

        if bhi >= 0 && bhi < nx as i32 && bhj >= 0 && bhj < ny as i32 && bhk >= 0 && bhk < nz as i32 {
            let bh_idx = idx3d(bhi as usize, bhj as usize, bhk as usize, nx, ny);
            if visited[bh_idx] != 0 {
                let v = old_val - phase[bh_idx];
                d = v.clamp(-wrap_addition, wrap_addition);
            }
        }
    }

    phase[new_idx] = new_val - TWO_PI * ((new_val - (old_val + d)) / TWO_PI).round();
}

/// Seed correction matching ROMEO.jl `seedcorrection!`.
///
/// With multi-echo data: finds the 2π offset for the seed that best aligns
/// phase/TE1 with phase2/TE2 (tests ±2 wraps for phase, ±1 for phase2).
/// Without multi-echo data: wraps seed phase to [-π, π].
fn seed_correction(
    phase: &mut [f64],
    idx: usize,
    phase2: Option<&[f64]>,
    tes: Option<(f64, f64)>,
) {
    if let (Some(p2), Some((te1, te2))) = (phase2, tes) {
        if te1.abs() > 1e-10 && te2.abs() > 1e-10 {
            let mut best_diff = f64::INFINITY;
            let mut best_offset = 0i32;
            for off1 in -2i32..=2 {
                for off2 in -1i32..=1 {
                    let diff = ((phase[idx] + TWO_PI * off1 as f64) / te1
                              - (p2[idx] + TWO_PI * off2 as f64) / te2).abs()
                        + (off1.unsigned_abs() + off2.unsigned_abs()) as f64 / 100.0;
                    if diff < best_diff {
                        best_diff = diff;
                        best_offset = off1;
                    }
                }
            }
            phase[idx] += TWO_PI * best_offset as f64;
            return;
        }
    }
    // Fallback: wrap to [-π, π]
    phase[idx] = wrap_to_pi(phase[idx]);
}

/// Wrap angle to [-π, π].
#[inline]
fn wrap_to_pi(x: f64) -> f64 {
    let mut a = x % TWO_PI;
    if a > PI { a -= TWO_PI; }
    else if a < -PI { a += TWO_PI; }
    a
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

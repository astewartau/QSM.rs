//! Laplacian Boundary Value (LBV) background field removal
//!
//! LBV removes background fields by solving the Laplace equation inside the mask
//! with Dirichlet boundary conditions from the total field at the mask boundary.
//!
//! The method exploits that background fields satisfy ∇²b = 0 inside the ROI.
//!
//! Reference:
//! Zhou D, Liu T, Spincemaille P, Wang Y. Background field removal by solving the
//! Laplacian boundary value problem. NMR in Biomedicine. 2014;27(3):312-319.

/// LBV background field removal
///
/// Solves ∇²b = 0 inside mask with b = f on boundary to find background field,
/// then computes local field as l = f - b.
///
/// # Arguments
/// * `field` - Total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = brain, 0 = background
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `tol` - Convergence tolerance for iterative solver
/// * `max_iter` - Maximum iterations
///
/// # Returns
/// Tuple of (local_field, eroded_mask)
pub fn lbv(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, Vec<u8>) {
    let n_total = nx * ny * nz;

    // Compute inverse squared voxel sizes for Laplacian
    let dx2_inv = 1.0 / (vsx * vsx);
    let dy2_inv = 1.0 / (vsy * vsy);
    let dz2_inv = 1.0 / (vsz * vsz);
    let diag = -2.0 * (dx2_inv + dy2_inv + dz2_inv);

    // Find interior and boundary voxels
    // Interior: mask=1 and all 6 neighbors have mask=1
    // Boundary: mask=1 and at least one neighbor has mask=0
    let mut interior = vec![false; n_total];
    let mut boundary = vec![false; n_total];

    for z in 1..(nz - 1) {
        for y in 1..(ny - 1) {
            for x in 1..(nx - 1) {
                let idx = x + y * nx + z * nx * ny;
                if mask[idx] == 0 {
                    continue;
                }

                // Check 6-connected neighbors
                let neighbors = [
                    idx.wrapping_sub(1),      // x-1
                    idx + 1,                  // x+1
                    idx.wrapping_sub(nx),     // y-1
                    idx + nx,                 // y+1
                    idx.wrapping_sub(nx * ny), // z-1
                    idx + nx * ny,            // z+1
                ];

                let all_inside = neighbors.iter().all(|&n| n < n_total && mask[n] != 0);

                if all_inside {
                    interior[idx] = true;
                } else {
                    boundary[idx] = true;
                }
            }
        }
    }

    // Edge voxels are boundary by definition
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if z == 0 || z == nz - 1 || y == 0 || y == ny - 1 || x == 0 || x == nx - 1 {
                    let idx = x + y * nx + z * nx * ny;
                    if mask[idx] != 0 {
                        boundary[idx] = true;
                        interior[idx] = false;
                    }
                }
            }
        }
    }

    // Initialize background field with total field
    // Background field = total field on boundary, solve for interior
    let mut bg_field = field.to_vec();

    // Solve ∇²b = 0 on interior voxels using Gauss-Seidel with over-relaxation
    // The boundary values are fixed (Dirichlet BC)
    let omega = 1.5; // Over-relaxation parameter

    for _iter in 0..max_iter {
        let mut max_change = 0.0f64;

        for z in 1..(nz - 1) {
            for y in 1..(ny - 1) {
                for x in 1..(nx - 1) {
                    let idx = x + y * nx + z * nx * ny;

                    if !interior[idx] {
                        continue;
                    }

                    // Compute Laplacian stencil weighted sum
                    let sum = dx2_inv * (bg_field[idx - 1] + bg_field[idx + 1])
                            + dy2_inv * (bg_field[idx - nx] + bg_field[idx + nx])
                            + dz2_inv * (bg_field[idx - nx * ny] + bg_field[idx + nx * ny]);

                    // Gauss-Seidel update: solve diag * b_new = -sum
                    let new_val = -sum / diag;

                    // SOR update
                    let old_val = bg_field[idx];
                    let updated = old_val + omega * (new_val - old_val);

                    max_change = max_change.max((updated - old_val).abs());
                    bg_field[idx] = updated;
                }
            }
        }

        // Check convergence
        if max_change < tol {
            break;
        }
    }

    // Compute local field = total field - background field
    let mut local_field = vec![0.0; n_total];
    let mut eroded_mask = vec![0u8; n_total];

    for i in 0..n_total {
        if interior[i] {
            local_field[i] = field[i] - bg_field[i];
            eroded_mask[i] = 1;
        }
    }

    (local_field, eroded_mask)
}

/// LBV with progress callback
pub fn lbv_with_progress<F>(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    tol: f64,
    max_iter: usize,
    mut progress_callback: F,
) -> (Vec<f64>, Vec<u8>)
where
    F: FnMut(usize, usize),
{
    let n_total = nx * ny * nz;

    let dx2_inv = 1.0 / (vsx * vsx);
    let dy2_inv = 1.0 / (vsy * vsy);
    let dz2_inv = 1.0 / (vsz * vsz);
    let diag = -2.0 * (dx2_inv + dy2_inv + dz2_inv);

    let mut interior = vec![false; n_total];
    let mut boundary = vec![false; n_total];

    for z in 1..(nz - 1) {
        for y in 1..(ny - 1) {
            for x in 1..(nx - 1) {
                let idx = x + y * nx + z * nx * ny;
                if mask[idx] == 0 {
                    continue;
                }

                let neighbors = [
                    idx.wrapping_sub(1),
                    idx + 1,
                    idx.wrapping_sub(nx),
                    idx + nx,
                    idx.wrapping_sub(nx * ny),
                    idx + nx * ny,
                ];

                let all_inside = neighbors.iter().all(|&n| n < n_total && mask[n] != 0);

                if all_inside {
                    interior[idx] = true;
                } else {
                    boundary[idx] = true;
                }
            }
        }
    }

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if z == 0 || z == nz - 1 || y == 0 || y == ny - 1 || x == 0 || x == nx - 1 {
                    let idx = x + y * nx + z * nx * ny;
                    if mask[idx] != 0 {
                        boundary[idx] = true;
                        interior[idx] = false;
                    }
                }
            }
        }
    }

    let mut bg_field = field.to_vec();
    let omega = 1.5;

    for iter in 0..max_iter {
        if iter % 10 == 0 {
            progress_callback(iter, max_iter);
        }

        let mut max_change = 0.0f64;

        for z in 1..(nz - 1) {
            for y in 1..(ny - 1) {
                for x in 1..(nx - 1) {
                    let idx = x + y * nx + z * nx * ny;

                    if !interior[idx] {
                        continue;
                    }

                    let sum = dx2_inv * (bg_field[idx - 1] + bg_field[idx + 1])
                            + dy2_inv * (bg_field[idx - nx] + bg_field[idx + nx])
                            + dz2_inv * (bg_field[idx - nx * ny] + bg_field[idx + nx * ny]);

                    let new_val = -sum / diag;
                    let old_val = bg_field[idx];
                    let updated = old_val + omega * (new_val - old_val);

                    max_change = max_change.max((updated - old_val).abs());
                    bg_field[idx] = updated;
                }
            }
        }

        if max_change < tol {
            progress_callback(iter + 1, iter + 1);
            break;
        }
    }

    progress_callback(max_iter, max_iter);

    let mut local_field = vec![0.0; n_total];
    let mut eroded_mask = vec![0u8; n_total];

    for i in 0..n_total {
        if interior[i] {
            local_field[i] = field[i] - bg_field[i];
            eroded_mask[i] = 1;
        }
    }

    (local_field, eroded_mask)
}

/// LBV with default parameters
pub fn lbv_default(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> (Vec<f64>, Vec<u8>) {
    lbv(field, mask, nx, ny, nz, vsx, vsy, vsz, 0.001, 500)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lbv_zero_field() {
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];

        let (local, _) = lbv(&field, &mask, n, n, n, 1.0, 1.0, 1.0, 1e-6, 100);

        for &val in local.iter() {
            assert!(val.abs() < 1e-10, "Zero field should give zero local field");
        }
    }

    #[test]
    fn test_lbv_finite() {
        let n = 16;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();

        // Create spherical mask
        let mut mask = vec![0u8; n * n * n];
        let center = n / 2;
        let radius = n / 3;

        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let dx = (x as i32) - (center as i32);
                    let dy = (y as i32) - (center as i32);
                    let dz = (z as i32) - (center as i32);
                    if dx*dx + dy*dy + dz*dz <= (radius * radius) as i32 {
                        mask[x + y * n + z * n * n] = 1;
                    }
                }
            }
        }

        let (local, eroded_mask) = lbv(&field, &mask, n, n, n, 1.0, 1.0, 1.0, 1e-5, 100);

        for (i, &val) in local.iter().enumerate() {
            assert!(val.is_finite(), "Local field should be finite at index {}", i);
        }

        // Eroded mask should be smaller than original
        let eroded_count: usize = eroded_mask.iter().map(|&x| x as usize).sum();
        let mask_count: usize = mask.iter().map(|&x| x as usize).sum();
        assert!(eroded_count <= mask_count, "Eroded mask should be <= original mask");
        assert!(eroded_count > 0, "Eroded mask should not be empty for reasonable-sized input");
    }

    #[test]
    fn test_lbv_harmonic_removal() {
        // Create a harmonic background field (satisfies ∇²b = 0)
        // and verify LBV removes it
        let n = 16;
        let mut field = vec![0.0; n * n * n];

        // Add linear field (which is harmonic)
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let idx = x + y * n + z * n * n;
                    field[idx] = (z as f64) * 0.1; // Linear in z
                }
            }
        }

        // Create spherical mask
        let mut mask = vec![0u8; n * n * n];
        let center = n / 2;
        let radius = n / 3;

        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let dx = (x as i32) - (center as i32);
                    let dy = (y as i32) - (center as i32);
                    let dz = (z as i32) - (center as i32);
                    if dx*dx + dy*dy + dz*dz <= (radius * radius) as i32 {
                        mask[x + y * n + z * n * n] = 1;
                    }
                }
            }
        }

        let (local, eroded_mask) = lbv(&field, &mask, n, n, n, 1.0, 1.0, 1.0, 1e-6, 500);

        // Local field should be close to zero for interior voxels
        // since the input is purely harmonic
        let mut max_local: f64 = 0.0;
        for i in 0..n*n*n {
            if eroded_mask[i] != 0 {
                max_local = max_local.max(local[i].abs());
            }
        }

        // Allow some tolerance due to discrete Laplacian
        assert!(max_local < 0.5, "Harmonic field should be mostly removed, got max {}", max_local);
    }
}

//! Exact Euclidean distance transform (EDT) for binary masks.
//!
//! Implements the separable, O(n) algorithm of Felzenszwalb & Huttenlocher,
//! "Distance Transforms of Sampled Functions" (2012), generalized to
//! anisotropic voxel spacing. For each masked (foreground) voxel it returns the
//! Euclidean distance **in millimetres** to the nearest background (`mask == 0`)
//! voxel; background voxels get distance 0.
//!
//! This is primarily used to reason about how morphological / SMV-kernel erosion
//! will shrink a mask before running a pipeline: a voxel survives erosion by a
//! spherical kernel of radius `r` iff its distance-to-boundary is `>= r`.

use crate::Grid;

/// 1-D squared distance transform along one axis with squared spacing `a`.
///
/// Computes `d[q] = min_p ( a * (q - p)^2 + f[p] )` (lower envelope of
/// parabolas). `f` holds the running squared distances from previous passes;
/// source voxels start at 0 and non-source at a large finite sentinel.
fn dt_1d(f: &[f64], a: f64, d: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        let mut s;
        loop {
            let vk = v[k];
            // Intersection of the parabola rooted at q with the one at v[k],
            // derived from a(s-q)^2 + f[q] = a(s-vk)^2 + f[vk].
            s = (((f[q] - f[vk]) / a) + (q * q) as f64 - (vk * vk) as f64)
                / (2.0 * (q as f64 - vk as f64));
            if s <= z[k] {
                // The z[0] = -inf sentinel guarantees this stops at k == 0.
                k -= 1;
            } else {
                break;
            }
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }

    k = 0;
    for q in 0..n {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let vk = v[k];
        let dq = q as f64 - vk as f64;
        d[q] = a * dq * dq + f[vk];
    }
}

/// Compute the exact Euclidean distance transform of a binary mask.
///
/// Returns, for every voxel, the distance in millimetres to the nearest
/// `mask == 0` voxel (0 for background voxels themselves). Only in-volume
/// background counts; if the mask fills the whole volume the returned distances
/// are the (large) sentinel-derived values rather than a boundary distance.
pub fn distance_transform_edt(mask: &[u8], grid: &Grid) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = grid.voxel_size;
    let n = nx * ny * nz;
    assert_eq!(mask.len(), n, "mask length must equal grid voxel count");

    // Finite sentinel larger than any achievable squared distance, so that
    // source-less 1-D lines never produce INF-INF = NaN in the arithmetic.
    let diag_sq = (nx as f64 * vsx).powi(2)
        + (ny as f64 * vsy).powi(2)
        + (nz as f64 * vsz).powi(2);
    let big = diag_sq * 4.0 + 1.0;

    let mut f: Vec<f64> = mask.iter().map(|&m| if m == 0 { 0.0 } else { big }).collect();

    let max_dim = nx.max(ny).max(nz);
    let mut line = vec![0.0f64; max_dim];
    let mut d = vec![0.0f64; max_dim];
    let mut v = vec![0usize; max_dim];
    let mut z = vec![0.0f64; max_dim + 1];

    // Pass 1: along x (contiguous in Fortran order).
    for zi in 0..nz {
        for yi in 0..ny {
            let base = yi * nx + zi * nx * ny;
            for x in 0..nx {
                line[x] = f[base + x];
            }
            dt_1d(&line[..nx], vsx * vsx, &mut d[..nx], &mut v[..nx], &mut z[..nx + 1]);
            for x in 0..nx {
                f[base + x] = d[x];
            }
        }
    }

    // Pass 2: along y.
    for zi in 0..nz {
        for xi in 0..nx {
            for y in 0..ny {
                line[y] = f[xi + y * nx + zi * nx * ny];
            }
            dt_1d(&line[..ny], vsy * vsy, &mut d[..ny], &mut v[..ny], &mut z[..ny + 1]);
            for y in 0..ny {
                f[xi + y * nx + zi * nx * ny] = d[y];
            }
        }
    }

    // Pass 3: along z.
    for yi in 0..ny {
        for xi in 0..nx {
            for zc in 0..nz {
                line[zc] = f[xi + yi * nx + zc * nx * ny];
            }
            dt_1d(&line[..nz], vsz * vsz, &mut d[..nz], &mut v[..nz], &mut z[..nz + 1]);
            for zc in 0..nz {
                f[xi + yi * nx + zc * nx * ny] = d[zc];
            }
        }
    }

    for val in f.iter_mut() {
        *val = val.max(0.0).sqrt();
    }
    f
}

/// Maximum distance-to-boundary over all foreground voxels, in millimetres.
///
/// This is the "radius of the deepest interior point" — the largest SMV/erosion
/// kernel radius for which *any* voxel survives.
pub fn mask_max_thickness(mask: &[u8], grid: &Grid) -> f64 {
    let edt = distance_transform_edt(mask, grid);
    edt.iter()
        .zip(mask.iter())
        .filter(|(_, &m)| m != 0)
        .map(|(&d, _)| d)
        .fold(0.0, f64::max)
}

/// Number of foreground voxels that survive erosion by a spherical kernel of
/// radius `radius_mm` (i.e. whose distance-to-boundary is `>= radius_mm`).
pub fn erosion_survivors(mask: &[u8], grid: &Grid, radius_mm: f64) -> usize {
    let edt = distance_transform_edt(mask, grid);
    edt.iter()
        .zip(mask.iter())
        .filter(|(&d, &m)| m != 0 && d >= radius_mm)
        .count()
}

/// Suggest the largest erosion/SMV radius (mm) that retains at least
/// `keep_fraction` of the foreground voxels.
///
/// Returns 0.0 for an empty mask. `keep_fraction` is clamped to (0, 1].
/// Computed from the distribution of foreground distance-to-boundary values:
/// the radius equal to the `(1 - keep_fraction)` quantile keeps `keep_fraction`
/// of voxels.
pub fn suggest_radius_for_fraction(mask: &[u8], grid: &Grid, keep_fraction: f64) -> f64 {
    let keep = keep_fraction.clamp(1e-6, 1.0);
    let edt = distance_transform_edt(mask, grid);
    let mut depths: Vec<f64> = edt
        .iter()
        .zip(mask.iter())
        .filter(|(_, &m)| m != 0)
        .map(|(&d, _)| d)
        .collect();
    if depths.is_empty() {
        return 0.0;
    }
    depths.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Keep the deepest `keep` fraction: threshold at the (1 - keep) quantile.
    let idx = (((1.0 - keep) * depths.len() as f64).floor() as usize).min(depths.len() - 1);
    depths[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1d_line_unit_spacing() {
        // 7x1x1, background only at x=0 → distances 0,1,2,...,6
        let grid = Grid::new(7, 1, 1, 1.0, 1.0, 1.0);
        let mut mask = vec![1u8; 7];
        mask[0] = 0;
        let edt = distance_transform_edt(&mask, &grid);
        for x in 0..7 {
            assert!((edt[x] - x as f64).abs() < 1e-9, "x={} got {}", x, edt[x]);
        }
    }

    #[test]
    fn test_1d_anisotropic_spacing() {
        // 3 voxels along x with 2mm spacing, background at x=0 → 0,2,4
        let grid = Grid::new(3, 1, 1, 2.0, 1.0, 1.0);
        let mut mask = vec![1u8; 3];
        mask[0] = 0;
        let edt = distance_transform_edt(&mask, &grid);
        assert!((edt[0] - 0.0).abs() < 1e-9);
        assert!((edt[1] - 2.0).abs() < 1e-9);
        assert!((edt[2] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_3d_corner_background() {
        // 3x3x3, background only at (0,0,0). Distance at (2,2,2) = sqrt(12).
        let (nx, ny, nz) = (3, 3, 3);
        let grid = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
        let mut mask = vec![1u8; nx * ny * nz];
        mask[0] = 0; // (0,0,0)
        let edt = distance_transform_edt(&mask, &grid);
        let corner = 2 + 2 * nx + 2 * nx * ny;
        assert!((edt[corner] - 12.0_f64.sqrt()).abs() < 1e-9, "got {}", edt[corner]);
        // Face-adjacent voxel (1,0,0) is distance 1.
        assert!((edt[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_background_is_zero() {
        let grid = Grid::new(4, 4, 4, 1.0, 1.0, 1.0);
        let mask = vec![0u8; 64];
        let edt = distance_transform_edt(&mask, &grid);
        assert!(edt.iter().all(|&d| d == 0.0));
    }

    #[test]
    fn test_erosion_survivors_monotonic() {
        // Solid 11^3 cube of foreground surrounded by background.
        let n = 15;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let mut mask = vec![0u8; n * n * n];
        for z in 2..13 {
            for y in 2..13 {
                for x in 2..13 {
                    mask[x + y * n + z * n * n] = 1;
                }
            }
        }
        let s0 = erosion_survivors(&mask, &grid, 0.0);
        let s2 = erosion_survivors(&mask, &grid, 2.0);
        let s4 = erosion_survivors(&mask, &grid, 4.0);
        let s7 = erosion_survivors(&mask, &grid, 7.0);
        assert_eq!(s0, 11 * 11 * 11); // radius 0 keeps everything
        assert!(s2 < s0 && s4 < s2, "survivors must shrink with radius: {s0},{s2},{s4}");
        assert_eq!(s7, 0, "radius exceeding max thickness erases the mask");
        // Deepest interior point of an 11-wide cube (indices 2..12) is 6 voxels
        // from the nearest background (at index 1 / 13).
        let t = mask_max_thickness(&mask, &grid);
        assert!((t - 6.0).abs() < 1e-9, "max thickness {}", t);
    }

    #[test]
    fn test_suggest_radius_retains_fraction() {
        let n = 15;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let mut mask = vec![0u8; n * n * n];
        for z in 2..13 {
            for y in 2..13 {
                for x in 2..13 {
                    mask[x + y * n + z * n * n] = 1;
                }
            }
        }
        let count: usize = mask.iter().map(|&m| m as usize).sum();
        let r = suggest_radius_for_fraction(&mask, &grid, 0.5);
        let survivors = erosion_survivors(&mask, &grid, r);
        assert!(
            survivors as f64 >= 0.5 * count as f64,
            "radius {r} kept {survivors}/{count}, expected >= 50%"
        );
    }
}

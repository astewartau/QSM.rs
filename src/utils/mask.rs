//! Mask utilities
//!
//! Provides functions for creating and applying masks on 3D volumes, and for finding where a
//! scanner wrote no phase ([`scanner_masked_phase`]). Morphology (erosion,
//! dilation, closing, hole filling) is in [`crate::utils::morphology`].

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

/// Smallest share of the volume a single exact phase value must cover to count as a scanner's
/// fill value in [`scanner_masked_phase`]: 5 %.
///
/// Measured phase is quantised (12 bits on most scanners, 4,096 levels), so no single value
/// covers more than a fraction of a percent of a real image, smooth background included; the
/// fill values found so far cover 56–67 % of the field of view (Philips, Q0748 audit, 2026-11).
pub const SCANNER_FILL_MIN_FRACTION: f64 = 0.05;

/// Where a scanner wrote a fill value instead of a measured phase: [`scanner_masked_phase`].
#[derive(Clone, Debug, PartialEq)]
pub struct ScannerMaskedPhase {
    /// 1 where the phase holds data, 0 where it is the scanner's fill value.
    pub has_data: Vec<u8>,
    /// The fill value, if one was found.
    pub fill_value: Option<f64>,
    /// Number of voxels without data.
    pub n_masked: usize,
}

/// Find the voxels where the scanner wrote no phase: a constant fill value, the same in every
/// echo, covering a large part of the volume.
///
/// Some scanners mask the phase they export. Philips writes the lowest phase value (−π once
/// scaled) outside its own signal mask, in every echo, with magnitude 0 or noise; on the Philips
/// series of the Q0748 audit (2026-11) that was 56–67 % of the field of view. A phase that is flat
/// in space and constant over echoes looks like the most reliable phase there is: ROMEO's phase
/// quality (and ROMEO.jl's `voxelquality`) rates it far above the brain, so a phase-quality mask
/// takes the background in (27–50 % of the default mask's voxels on those series), and MCPC-3D-S
/// and the unwrapper treat it as signal.
///
/// A voxel has no data when its phase is **bit-identical in every echo** and equal to the
/// volume's most frequent exact value, provided that value covers at least
/// [`SCANNER_FILL_MIN_FRACTION`] of the volume. With one echo only the second condition applies.
/// Both conditions are needed: with two echoes of a 12-bit phase, about 1 in 4,096 real voxels
/// repeats its value by chance, and a mode alone could be a genuine (if implausible) phase.
/// Measured phase never meets both, so this is not a threshold on the data, only the detection
/// of a value the scanner wrote; when no value qualifies, every voxel has data.
///
/// `phases` holds each echo's phase, in any units (the comparison is exact).
pub fn scanner_masked_phase(phases: &[&[f64]]) -> ScannerMaskedPhase {
    let n = phases.first().map_or(0, |p| p.len());
    let all_data = || ScannerMaskedPhase { has_data: vec![1; n], fill_value: None, n_masked: 0 };
    if n == 0 || phases.iter().any(|p| p.len() != n) {
        return all_data();
    }
    let same_in_every_echo = |i: usize| {
        let v = phases[0][i].to_bits();
        phases[1..].iter().all(|p| p[i].to_bits() == v)
    };
    // The most frequent exact value among the voxels that repeat it in every echo.
    let mut counts: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    for i in 0..n {
        if phases[0][i].is_finite() && same_in_every_echo(i) {
            *counts.entry(phases[0][i].to_bits()).or_insert(0) += 1;
        }
    }
    // Ties go to the smaller value, so the result does not depend on the hash order.
    let Some((bits, count)) = counts.into_iter().max_by(|a, b| {
        a.1.cmp(&b.1).then_with(|| f64::from_bits(b.0).total_cmp(&f64::from_bits(a.0)))
    }) else {
        return all_data();
    };
    if (count as f64) < SCANNER_FILL_MIN_FRACTION * n as f64 {
        return all_data();
    }
    let has_data: Vec<u8> = (0..n)
        .map(|i| u8::from(!(phases[0][i].to_bits() == bits && same_in_every_echo(i))))
        .collect();
    ScannerMaskedPhase { has_data, fill_value: Some(f64::from_bits(bits)), n_masked: count }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 1.0, 1.0, 1.0)
    }

    fn noisy(n: usize, seed: u64) -> Vec<f64> {
        // a 12-bit quantised "measured" phase
        (0..n).map(|i| {
            let h = (i as u64).wrapping_mul(2654435761).wrapping_add(seed.wrapping_mul(40503)) % 4096;
            -std::f64::consts::PI + h as f64 * (2.0 * std::f64::consts::PI / 4096.0)
        }).collect()
    }

    #[test]
    fn scanner_fill_value_is_found_in_every_echo() {
        let n = 4000;
        let fill = -std::f64::consts::PI;
        let mut echoes: Vec<Vec<f64>> = (0..3).map(|e| noisy(n, e + 1)).collect();
        for e in echoes.iter_mut() {
            for v in e[..1500].iter_mut() { *v = fill; }
        }
        // a voxel that holds the fill value in two echoes only is measured phase
        echoes[2][1499] = 0.25;
        let refs: Vec<&[f64]> = echoes.iter().map(|e| e.as_slice()).collect();
        let r = scanner_masked_phase(&refs);
        assert_eq!(r.fill_value, Some(fill));
        assert_eq!(r.n_masked, 1499);
        assert!(r.has_data[..1499].iter().all(|&h| h == 0));
        assert!(r.has_data[1499..].iter().all(|&h| h == 1));
    }

    #[test]
    fn single_echo_uses_the_dominant_value_and_noise_has_none() {
        let n = 4000;
        let mut one = noisy(n, 7);
        let r = scanner_masked_phase(&[&one]);
        assert_eq!((r.fill_value, r.n_masked), (None, 0));
        assert!(r.has_data.iter().all(|&h| h == 1));
        // below the minimum share: not a fill value
        for v in one[..150].iter_mut() { *v = 0.0; }
        assert_eq!(scanner_masked_phase(&[&one]).n_masked, 0);
        for v in one[..400].iter_mut() { *v = 0.0; }
        let r = scanner_masked_phase(&[&one]);
        assert_eq!((r.fill_value, r.n_masked), (Some(0.0), 400 + one[400..].iter().filter(|&&v| v == 0.0).count()));
        // two echoes of independent noise repeat values by chance, but no repeated value dominates
        let (a, b) = (noisy(n, 3), noisy(n, 4));
        assert_eq!(scanner_masked_phase(&[&a, &b]).n_masked, 0);
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
}

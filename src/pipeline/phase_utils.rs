//! Phase and field conversion utilities
//!
//! Canonical implementations shared by all pipeline consumers.

use std::f64::consts::PI;

/// Scale phase data to [-pi, pi] range in-place.
///
/// If data is already approximately in [-pi, pi] (within 10% tolerance),
/// leaves it unchanged. Otherwise linearly maps from [min, max] to [-pi, pi].
/// NaN/Inf values are replaced with 0.
pub fn scale_phase_to_pi(data: &mut [f64]) {
    if data.is_empty() {
        return;
    }

    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for &v in data.iter() {
        if v.is_finite() {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }
    }

    // Replace non-finite values with 0
    for v in data.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }

    let range = max_val - min_val;
    if range < 1e-10 {
        return;
    }

    // Check if already approximately in [-pi, pi]
    let tol = 0.1 * PI;
    if (min_val + PI).abs() < tol && (max_val - PI).abs() < tol {
        return;
    }

    // Linearly rescale to [-pi, pi]
    let scale = 2.0 * PI / range;
    for v in data.iter_mut() {
        *v = (*v - min_val) * scale - PI;
    }
}

/// Convert field from Hz to ppm.
pub fn hz_to_ppm(field_hz: &[f64], field_strength: f64) -> Vec<f64> {
    let gamma_hz = 42.576e6; // Hz/T (proton gyromagnetic ratio)
    let scale = 1e6 / (gamma_hz * field_strength);
    field_hz.iter().map(|&v| v * scale).collect()
}

/// Convert field from rad/s to ppm.
pub fn rads_to_ppm(field_rads: &[f64], field_strength: f64) -> Vec<f64> {
    let gamma_hz = 42.576e6;
    let scale = 1e6 / (2.0 * PI * gamma_hz * field_strength);
    field_rads.iter().map(|&v| v * scale).collect()
}

/// Root-sum-of-squares combination of multiple magnitude images.
pub fn rss_combine(magnitudes: &[&[f64]]) -> Vec<f64> {
    if magnitudes.is_empty() {
        return Vec::new();
    }
    let n = magnitudes[0].len();
    let mut combined = vec![0.0f64; n];
    for mag in magnitudes {
        for (i, &v) in mag.iter().enumerate() {
            combined[i] += v * v;
        }
    }
    for v in &mut combined {
        *v = v.sqrt();
    }
    combined
}

// Re-export mask operations from canonical location
pub use crate::utils::mask::{erode_mask, dilate_mask};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_phase_already_in_range() {
        let mut data = vec![-PI, 0.0, PI];
        let original = data.clone();
        scale_phase_to_pi(&mut data);
        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_scale_phase_rescales_0_to_4096() {
        let mut data = vec![0.0, 2048.0, 4096.0];
        scale_phase_to_pi(&mut data);
        assert!((data[0] - (-PI)).abs() < 1e-10);
        assert!((data[2] - PI).abs() < 1e-10);
        assert!(data[1].abs() < 1e-10);
    }

    #[test]
    fn test_scale_phase_nan_replaced() {
        let mut data = vec![0.0, f64::NAN, 4096.0];
        scale_phase_to_pi(&mut data);
        assert!(data[1].is_finite());
    }

    #[test]
    fn test_scale_phase_empty() {
        let mut data: Vec<f64> = vec![];
        scale_phase_to_pi(&mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn test_hz_to_ppm_3t() {
        let gamma_hz = 42.576e6;
        let field = vec![gamma_hz * 3.0];
        let ppm = hz_to_ppm(&field, 3.0);
        assert!((ppm[0] - 1e6).abs() < 1.0);
    }

    #[test]
    fn test_rads_to_ppm_3t() {
        let gamma_hz = 42.576e6;
        let rads = vec![2.0 * PI * gamma_hz * 3.0];
        let ppm = rads_to_ppm(&rads, 3.0);
        assert!((ppm[0] - 1e6).abs() < 1.0);
    }

    #[test]
    fn test_rss_combine() {
        let a = vec![3.0, 0.0];
        let b = vec![4.0, 1.0];
        let result = rss_combine(&[&a, &b]);
        assert!((result[0] - 5.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rss_combine_empty() {
        let result = rss_combine(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_erode_mask_cube() {
        let mask = vec![1u8; 27]; // 3x3x3
        let grid = crate::Grid::new(3, 3, 3, 1.0, 1.0, 1.0);
        let result = erode_mask(&mask, &grid, 1);
        let center = 1 + 3 + 9;
        assert_eq!(result[center], 1);
        let total: u32 = result.iter().map(|&v| v as u32).sum();
        assert_eq!(total, 1);
    }

    #[test]
    fn test_dilate_mask_single() {
        let mut mask = vec![0u8; 27];
        mask[1 + 3 + 9] = 1; // center
        let grid = crate::Grid::new(3, 3, 3, 1.0, 1.0, 1.0);
        let result = dilate_mask(&mask, &grid, 1);
        let total: u32 = result.iter().map(|&v| v as u32).sum();
        assert_eq!(total, 7); // center + 6 neighbors
    }
}

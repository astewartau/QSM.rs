//! Automatic thresholding algorithms
//!
//! Provides Otsu's method for computing optimal thresholds on bimodal histograms.

/// Otsu's method for automatic threshold selection
///
/// Finds the threshold that maximizes inter-class variance.
/// Matches MATLAB's graythresh: operates on all values including zeros,
/// normalizes to [0,1] range, and returns threshold at bin edge.
///
/// # Arguments
/// * `data` - Input data (e.g. flattened 3D image)
/// * `num_bins` - Number of histogram bins (typically 256)
///
/// # Returns
/// The optimal threshold value
pub fn otsu_threshold(data: &[f64], num_bins: usize) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let min_val = data.iter().fold(f64::MAX, |a, &b| a.min(b));
    let max_val = data.iter().fold(f64::MIN, |a, &b| a.max(b));

    if (max_val - min_val).abs() < 1e-10 {
        return min_val;
    }

    // Build histogram over full range including zeros
    let bin_width = (max_val - min_val) / num_bins as f64;
    let mut histogram = vec![0usize; num_bins];

    for &v in data {
        let bin = ((v - min_val) / bin_width).floor() as usize;
        let bin = bin.min(num_bins - 1);
        histogram[bin] += 1;
    }

    let total_pixels = data.len() as f64;

    // Compute cumulative sums
    let mut sum_total = 0.0;
    for (i, &count) in histogram.iter().enumerate() {
        sum_total += i as f64 * count as f64;
    }

    let mut sum_background = 0.0;
    let mut weight_background = 0.0;
    let mut max_variance = 0.0;
    let mut optimal_threshold_bin = 0;

    for (t, &count) in histogram.iter().enumerate() {
        weight_background += count as f64;
        if weight_background == 0.0 {
            continue;
        }

        let weight_foreground = total_pixels - weight_background;
        if weight_foreground == 0.0 {
            break;
        }

        sum_background += t as f64 * count as f64;

        let mean_background = sum_background / weight_background;
        let mean_foreground = (sum_total - sum_background) / weight_foreground;

        // Inter-class variance
        let variance = weight_background * weight_foreground
            * (mean_background - mean_foreground).powi(2);

        if variance > max_variance {
            max_variance = variance;
            optimal_threshold_bin = t;
        }
    }

    // Convert bin to threshold value (bin edge, matching MATLAB's graythresh)
    min_val + optimal_threshold_bin as f64 * bin_width
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_otsu_threshold_bimodal() {
        // Bimodal distribution with spread around two clusters
        let mut data = Vec::new();
        for i in 0..100 {
            data.push(0.1 + 0.2 * (i as f64 / 100.0)); // 0.1 to 0.3
        }
        for i in 0..100 {
            data.push(0.7 + 0.2 * (i as f64 / 100.0)); // 0.7 to 0.9
        }

        let threshold = otsu_threshold(&data, 256);
        assert!(threshold > 0.2 && threshold < 0.8,
            "Threshold {} should be between the two clusters", threshold);
    }

    #[test]
    fn test_otsu_threshold_empty() {
        assert_eq!(otsu_threshold(&[], 256), 0.0);
    }

    #[test]
    fn test_otsu_threshold_constant() {
        let data = vec![5.0; 100];
        assert_eq!(otsu_threshold(&data, 256), 5.0);
    }
}

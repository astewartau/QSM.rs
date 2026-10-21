//! Automatic thresholding algorithms
//!
//! Provides Otsu's method for computing optimal thresholds on bimodal histograms.

/// Otsu's method for automatic threshold selection
///
/// Finds the threshold that maximizes inter-class variance over a histogram spanning the data's
/// own range (all values, zeros included), and returns the threshold at a bin edge. For MATLAB's
/// `graythresh` on double data, which does not rescale, see [`graythresh`].
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

    // Convert bin to threshold value (bin edge)
    min_val + optimal_threshold_bin as f64 * bin_width
}

/// MATLAB's `graythresh` on double data: Otsu's threshold, in [0, 1].
///
/// As `graythresh` does, the data are converted with `im2uint8` (clipped to [0, 1], scaled by
/// 255 and rounded; NaN becomes 0), not rescaled to their own range, and Otsu's method runs on
/// the 256-bin histogram of every value (`otsuthresh`: ties take the mean of the maximising bins;
/// a non-finite maximum gives 0).
pub fn graythresh(data: &[f64]) -> f64 {
    let mut counts = [0f64; 256];
    for &v in data {
        let q = if v.is_nan() { 0.0 } else { (v.clamp(0.0, 1.0) * 255.0).round() };
        counts[q as usize] += 1.0;
    }
    let total: f64 = counts.iter().sum();
    if total == 0.0 {
        return 0.0;
    }
    let (mut omega, mut mu) = ([0f64; 256], [0f64; 256]);
    let (mut o, mut m) = (0.0, 0.0);
    for i in 0..256 {
        let p = counts[i] / total;
        o += p;
        m += p * (i + 1) as f64;
        omega[i] = o;
        mu[i] = m;
    }
    let mu_t = mu[255];
    let sigma: Vec<f64> = (0..256)
        .map(|i| (mu_t * omega[i] - mu[i]).powi(2) / (omega[i] * (1.0 - omega[i])))
        .collect();
    let max = sigma.iter().copied().filter(|v| !v.is_nan()).fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return 0.0;
    }
    let idx: Vec<f64> = (0..256).filter(|&i| sigma[i] == max).map(|i| (i + 1) as f64).collect();
    let idx = idx.iter().sum::<f64>() / idx.len() as f64;
    (idx - 1.0) / 255.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graythresh_matches_matlab() {
        // MATLAB R2023b graythresh on the same vectors.
        let a = [0.0, 0.0, 0.0, 0.1, 0.2, 0.8, 0.9, 0.95, 1.2, -0.1, f64::NAN, 0.5, 0.51];
        assert!((graythresh(&a) - 0.349019607843137).abs() < 1e-14);
        let b: Vec<f64> = std::iter::repeat_n(0.0, 1000)
            .chain((1..=50).map(|i| 0.02 * i as f64))
            .chain((1..=30).map(|i| 0.6 + 0.003 * i as f64))
            .collect();
        assert!((graythresh(&b) - 0.329411764705882).abs() < 1e-14);
        assert_eq!(graythresh(&[0.3; 10]), 0.0);
        // A tie across bins 1..254 takes their mean.
        assert!((graythresh(&[0.004, 0.005, 0.006, 0.996]) - 0.5).abs() < 1e-14);
    }

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

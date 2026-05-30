//! QSM referencing
//!
//! Adjusts the susceptibility map by subtracting the mean value
//! within the mask (mean referencing) or zeroing outside the mask.

use super::config::QsmReference;

/// Apply QSM referencing to a susceptibility map.
///
/// - `Mean`: subtract the mean of masked voxels, zero outside mask
/// - `None`: zero outside mask only
pub fn apply_reference(chi: &[f64], mask: &[u8], method: QsmReference) -> Vec<f64> {
    let n = chi.len();
    let mut result = vec![0.0; n];

    match method {
        QsmReference::Mean => {
            let mut sum = 0.0;
            let mut count = 0usize;
            for i in 0..n {
                if mask[i] > 0 {
                    sum += chi[i];
                    count += 1;
                }
            }
            let mean = if count > 0 { sum / count as f64 } else { 0.0 };
            for i in 0..n {
                if mask[i] > 0 {
                    result[i] = chi[i] - mean;
                }
            }
        }
        QsmReference::None => {
            for i in 0..n {
                if mask[i] > 0 {
                    result[i] = chi[i];
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_mean() {
        let chi = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![1, 1, 1, 0];
        let result = apply_reference(&chi, &mask, QsmReference::Mean);
        // Mean of masked = (1+2+3)/3 = 2.0
        assert!((result[0] - (-1.0)).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
        assert!((result[2] - 1.0).abs() < 1e-10);
        assert_eq!(result[3], 0.0); // masked out
    }

    #[test]
    fn test_reference_none() {
        let chi = vec![1.0, 2.0, 3.0];
        let mask = vec![1, 0, 1];
        let result = apply_reference(&chi, &mask, QsmReference::None);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 3.0);
    }

    #[test]
    fn test_reference_mean_empty_mask() {
        let chi = vec![1.0, 2.0];
        let mask = vec![0, 0];
        let result = apply_reference(&chi, &mask, QsmReference::Mean);
        assert_eq!(result, vec![0.0, 0.0]);
    }
}

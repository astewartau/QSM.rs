//! Phase and field conversion utilities
//!
//! Canonical implementations shared by all pipeline consumers.

use std::f64::consts::PI;

use crate::constants::GAMMA_BAR_HZ_PER_T;

/// Bring one phase volume into radians in `[-π, π]`, in place: [`scale_phases_to_pi`] for a
/// single echo. Use [`scale_phases_to_pi`] for multi-echo phase, so every echo gets the same
/// conversion.
pub fn scale_phase_to_pi(data: &mut [f64]) {
    scale_phases_to_pi(&mut [data]);
}

/// Bring multi-echo phase into radians in `[-π, π]`, in place, with ONE conversion shared by
/// every echo. Non-finite values become 0.
///
/// Phase reaches QSM.rs in radians or in a scanner's or converter's own units. The rule is
/// chosen from the finite minimum and maximum over all echoes together:
///
/// 1. **Already radians** (every value within `[-π, π]`, give or take [`PHASE_RADIANS_TOL`]):
///    left exactly as it is. This includes phase that does not reach ±π, such as phase that does
///    not wrap (short echoes, a good shim, a small field of view, synthetic phantoms) or phase
///    that was unwrapped or processed before. An earlier rule stretched any volume whose minimum
///    and maximum were not near ±π onto `[-π, π]`, one echo at a time; on such phase that
///    changed every value and scaled each echo differently, destroying the field (a non-wrapping
///    phantom's complex-fit field came out at 1e-8 ppm with correlation 0.06 to the truth).
/// 2. **Radians in `[0, 2π]`** (minimum ≥ −tolerance, maximum ≤ 2π + tolerance): values above π
///    are wrapped by −2π. The same angles, so nothing a field map sees changes; a linear map to
///    `[-π, π]` would instead add π to every echo, which is wrong for a single-echo field φ/TE.
/// 3. **Anything else** (integer DICOM phase such as 0..4095 or −4096..4095, degrees): mapped
///    linearly from `[min, max]` over ALL echoes onto `[-π, π]`. Integer phase of every echo
///    spans its full range, so the shared range is the scanner's; one scale and offset for every
///    echo keeps the phase differences between echoes, which per-echo ranges would not.
///
/// A volume with no range (constant, or nothing finite) is left as it is (after the non-finite
/// values are zeroed).
pub fn scale_phases_to_pi<T: AsMut<[f64]>>(echoes: &mut [T]) {
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for echo in echoes.iter_mut() {
        for v in echo.as_mut().iter_mut() {
            if v.is_finite() {
                min_val = min_val.min(*v);
                max_val = max_val.max(*v);
            } else {
                *v = 0.0;
            }
        }
    }
    // No range: constant, or nothing finite (then -inf - inf = -inf); NaN cannot arise but is
    // treated the same.
    let range = max_val - min_val;
    if range.is_nan() || range < 1e-10 {
        return;
    }
    let tol = PHASE_RADIANS_TOL;
    if min_val >= -PI - tol && max_val <= PI + tol {
        return;
    }
    if min_val >= -tol && max_val <= 2.0 * PI + tol {
        for echo in echoes.iter_mut() {
            for v in echo.as_mut().iter_mut() {
                if *v > PI {
                    *v -= 2.0 * PI;
                }
            }
        }
        return;
    }
    let scale = 2.0 * PI / range;
    for echo in echoes.iter_mut() {
        for v in echo.as_mut().iter_mut() {
            *v = (*v - min_val) * scale - PI;
        }
    }
}

/// How far outside `[-π, π]` (or `[0, 2π]`) phase may reach and still be taken as radians by
/// [`scale_phases_to_pi`]: float32 storage and converters' rescale slopes put ±π a little off.
/// 10 %, as the earlier rule's "near ±π" test; far below any other unit's range (degrees reach
/// 180, integer phase 4095).
pub const PHASE_RADIANS_TOL: f64 = 0.1 * PI;

/// Convert field from Hz to ppm.
pub fn hz_to_ppm(field_hz: &[f64], field_strength: f64) -> Vec<f64> {
    let gamma_hz = GAMMA_BAR_HZ_PER_T;
    let scale = 1e6 / (gamma_hz * field_strength);
    field_hz.iter().map(|&v| v * scale).collect()
}

/// Convert field from rad/s to ppm.
pub fn rads_to_ppm(field_rads: &[f64], field_strength: f64) -> Vec<f64> {
    let gamma_hz = GAMMA_BAR_HZ_PER_T;
    let scale = 1e6 / (2.0 * PI * gamma_hz * field_strength);
    field_rads.iter().map(|&v| v * scale).collect()
}

/// Convert field from ppm to Hz: the inverse of [`hz_to_ppm`].
pub fn ppm_to_hz(field_ppm: &[f64], field_strength: f64) -> Vec<f64> {
    let scale = GAMMA_BAR_HZ_PER_T * field_strength * 1e-6;
    field_ppm.iter().map(|&v| v * scale).collect()
}

/// Convert field from ppm to rad/s: the inverse of [`rads_to_ppm`].
pub fn ppm_to_rads(field_ppm: &[f64], field_strength: f64) -> Vec<f64> {
    let scale = 2.0 * PI * GAMMA_BAR_HZ_PER_T * field_strength * 1e-6;
    field_ppm.iter().map(|&v| v * scale).collect()
}

/// Root-sum-of-squares combination of multiple magnitude images.
///
/// Every echo must hold the same number of voxels. The combined volume inherits
/// that size and is then indexed by the run's grid downstream (bias correction,
/// masking), so an echo of a different matrix size would otherwise either be
/// silently truncated or make the result the wrong size for every later stage.
///
/// # Errors
/// If the echoes do not all have the same length. No echoes gives an empty volume.
pub fn rss_combine(magnitudes: &[&[f64]]) -> Result<Vec<f64>, crate::error::InputError> {
    if magnitudes.is_empty() {
        return Ok(Vec::new());
    }
    let n = magnitudes[0].len();
    for (e, mag) in magnitudes.iter().enumerate() {
        if mag.len() != n {
            return Err(crate::error::InputError(format!(
                "rss_combine: echo {e} has {} voxels but echo 0 has {n} \
                 (all echoes must come from the same acquisition)",
                mag.len()
            )));
        }
    }
    let mut combined = vec![0.0f64; n];
    for mag in magnitudes {
        for (i, &v) in mag.iter().enumerate() {
            combined[i] += v * v;
        }
    }
    for v in &mut combined {
        *v = v.sqrt();
    }
    Ok(combined)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase that does not wrap (it never reaches ±π) is radians already: left untouched, every
    /// echo. The old rule stretched each echo onto [-π, π] separately.
    #[test]
    fn scale_phases_leaves_non_wrapping_radians_alone() {
        let mut echoes = vec![vec![0.23, 0.30, 0.42], vec![0.16, 0.30, 0.55], vec![0.03, 0.30, 0.80]];
        let original = echoes.clone();
        scale_phases_to_pi(&mut echoes);
        assert_eq!(echoes, original);
        // float32-rounded ±π is still radians
        let mut edge = vec![vec![-(PI as f32) as f64, 0.0, (PI as f32) as f64]];
        let original = edge.clone();
        scale_phases_to_pi(&mut edge);
        assert_eq!(edge, original);
    }

    /// Integer phase maps to [-π, π] with one scale and offset for every echo: the same count is
    /// the same angle in every echo, whatever range each echo happens to cover.
    #[test]
    fn scale_phases_maps_integer_phase_identically_across_echoes() {
        for (lo, hi) in [(0.0, 4095.0), (-4096.0, 4095.0)] {
            let mid = ((lo + hi) / 2.0f64).round();
            // echo 2 does not reach either end
            let mut echoes = vec![vec![lo, mid, hi], vec![mid - 100.0, mid, mid + 100.0]];
            scale_phases_to_pi(&mut echoes);
            let map = |v: f64| (v - lo) * 2.0 * PI / (hi - lo) - PI;
            assert!((echoes[0][0] + PI).abs() < 1e-12 && (echoes[0][2] - PI).abs() < 1e-12);
            assert_eq!(echoes[0][1], echoes[1][1], "the same count, the same angle");
            for (got, v) in echoes[1].iter().zip([mid - 100.0, mid, mid + 100.0]) {
                assert!((got - map(v)).abs() < 1e-12, "{lo}..{hi}: {got} vs {}", map(v));
            }
        }
    }

    /// Radians in [0, 2π) are wrapped, not shifted: the same angles, so no echo gains π.
    #[test]
    fn scale_phases_wraps_zero_to_two_pi() {
        let mut echoes = vec![vec![0.0, 1.0, 4.0, 2.0 * PI - 1e-3], vec![0.5, 3.0, 5.0, 6.0]];
        let original = echoes.clone();
        scale_phases_to_pi(&mut echoes);
        for (e, o) in echoes.iter().flatten().zip(original.iter().flatten()) {
            assert!((-PI..=PI).contains(e), "{e}");
            assert!(((e - o).rem_euclid(2.0 * PI)).min((o - e).rem_euclid(2.0 * PI)) < 1e-12, "{e} vs {o}");
        }
        assert_eq!(echoes[0][1], 1.0);
    }

    /// Degrees are not radians: mapped linearly, shared across echoes.
    #[test]
    fn scale_phases_maps_degrees() {
        let mut echoes = vec![vec![-180.0, 0.0, 180.0], vec![-90.0, 90.0, 0.0]];
        scale_phases_to_pi(&mut echoes);
        assert!((echoes[1][0] + PI / 2.0).abs() < 1e-12 && (echoes[1][1] - PI / 2.0).abs() < 1e-12);
        assert!(echoes[1][2].abs() < 1e-12);
    }

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

    /// A constant volume has no range to map: it is left as it is.
    #[test]
    fn test_scale_phase_constant_is_left_alone() {
        let mut data = vec![5.0, 5.0, 5.0];
        scale_phase_to_pi(&mut data);
        assert!(data.iter().all(|v| (v - 5.0).abs() < 1e-10), "{data:?}");
    }

    /// Nothing finite: every value becomes 0, and there is then no range to map.
    #[test]
    fn test_scale_phase_all_nan_is_zero() {
        let mut data = vec![f64::NAN, f64::INFINITY, f64::NAN];
        scale_phase_to_pi(&mut data);
        assert_eq!(data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_hz_to_ppm_3t() {
        let gamma_hz = GAMMA_BAR_HZ_PER_T;
        let field = vec![gamma_hz * 3.0];
        let ppm = hz_to_ppm(&field, 3.0);
        assert!((ppm[0] - 1e6).abs() < 1.0);
    }

    #[test]
    fn test_rads_to_ppm_3t() {
        let gamma_hz = GAMMA_BAR_HZ_PER_T;
        let rads = vec![2.0 * PI * gamma_hz * 3.0];
        let ppm = rads_to_ppm(&rads, 3.0);
        assert!((ppm[0] - 1e6).abs() < 1.0);
    }

    #[test]
    fn ppm_hz_rads_round_trip() {
        let ppm = vec![-0.25, 0.0, 1.5];
        for (a, b) in hz_to_ppm(&ppm_to_hz(&ppm, 7.0), 7.0).iter().zip(&ppm) {
            assert!((a - b).abs() < 1e-12);
        }
        for (a, b) in rads_to_ppm(&ppm_to_rads(&ppm, 3.0), 3.0).iter().zip(&ppm) {
            assert!((a - b).abs() < 1e-12);
        }
        // 1 ppm at 3 T is 127.728 Hz.
        assert!((ppm_to_hz(&[1.0], 3.0)[0] - 127.728).abs() < 1e-9);
    }

    #[test]
    fn test_rss_combine() {
        let a = vec![3.0, 0.0];
        let b = vec![4.0, 1.0];
        let result = rss_combine(&[&a, &b]).unwrap();
        assert!((result[0] - 5.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rss_combine_empty() {
        let result = rss_combine(&[]).unwrap();
        assert!(result.is_empty());
    }

    // Issue #70: the combined volume inherits echo 0's size and is then indexed
    // by the run's grid, so a mismatched echo either panicked on an index or made
    // every later stage read past the end of the result. Now it is an error.
    #[test]
    fn rss_combine_refuses_a_longer_echo() {
        let e0 = vec![1.0; 512];
        let e1 = vec![1.0; 576];
        let err = rss_combine(&[&e0, &e1]).unwrap_err();
        assert!(err.to_string().contains("rss_combine: echo 1 has 576 voxels but echo 0 has 512"), "{err}");
    }

    #[test]
    fn rss_combine_refuses_a_shorter_echo() {
        // The short case never panicked - it silently summed only the part of
        // the stack that echo 1 covered.
        let e0 = vec![1.0; 512];
        let e1 = vec![1.0; 216];
        let err = rss_combine(&[&e0, &e1]).unwrap_err();
        assert!(err.to_string().contains("rss_combine: echo 1 has 216 voxels but echo 0 has 512"), "{err}");
    }
}

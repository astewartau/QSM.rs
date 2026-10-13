//! Real/imaginary input: conversion to magnitude and phase, with slice-chopping correction.
//!
//! Some scanners (GE in particular) export the complex image as separate real and imaginary
//! volumes instead of magnitude and phase. Converting them is `|z|` and `arg z`, except for one
//! artefact: when the reconstruction centres k-space without the matching image-domain shift,
//! every other slice comes out multiplied by −1 — a π phase jump between neighbouring slices
//! ("chopping"). Magnitude does not see it; phase does, and nothing downstream (unwrapping, field
//! fitting, background removal) can tell a π jump per slice from a field.
//!
//! [`complex_to_mag_phase`] detects that per axis and undoes it with a `(−1)^k` factor along the
//! affected axis before taking the phase. The factor is the same for every echo: the chop comes
//! from the reconstruction, not the signal, so all echoes share it. Flipping the same slices in
//! every echo leaves at most a common π offset, which phase-offset removal absorbs; flipping
//! different slices per echo would put a slice-alternating offset into the field.
//!
//! The sign of the phase is a vendor convention too: GE's real/imaginary pair gives phase of the
//! opposite sign to Siemens and Philips phase images (on the same subject, high-pass field maps
//! anti-correlate). Nothing in the data says which convention it follows, so the caller states
//! it ([`ComplexToMagPhaseParams::negate_phase`]); conjugating `z` brings it to the convention
//! the rest of the crate assumes.

use std::f64::consts::PI;

use crate::utils::threshold::otsu_threshold;
use crate::Grid;

/// Coherence below which an axis's chop decision is uncertain; see
/// [`SliceChopping::ambiguous_axes`].
///
/// Smooth phase gives a neighbour coherence near +1, chopped phase near −1. Values close to zero
/// come from data with too little signal, or phase varying too fast between voxels, for the
/// sign to be trusted either way.
pub const CHOP_AMBIGUOUS_COHERENCE: f64 = 0.5;

/// How [`complex_to_mag_phase`] treats alternate-slice π chopping.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ChopCorrection {
    /// Detect chopping along each axis from the data and correct the axes where it is found.
    #[default]
    Auto,
    /// Correct exactly these axes (`[x, y, z]`) whatever the data says. `[false; 3]` converts
    /// without any correction.
    Axes([bool; 3]),
}

/// Options for [`complex_to_mag_phase`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ComplexToMagPhaseParams {
    /// How alternate-slice π chopping is handled.
    pub chop: ChopCorrection,
    /// Negate the phase (take `arg z̄`). Set it for data whose phase runs opposite to the
    /// Siemens/Philips convention the rest of the crate assumes, such as GE real/imaginary
    /// exports; it cannot be detected from the data.
    pub negate_phase: bool,
}

/// What [`complex_to_mag_phase`] found and did about slice chopping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SliceChopping {
    /// Magnitude-weighted coherence of neighbouring voxels along each axis (`[x, y, z]`), in
    /// `[-1, 1]`, pooled over all echoes and measured on the input as given: about +1 for smooth
    /// phase, about −1 for an axis chopped by π. See [`slice_chopping_coherence`].
    pub coherence: [f64; 3],
    /// Axes the `(−1)^k` correction was applied along.
    pub corrected: [bool; 3],
    /// Whether `corrected` came from detection ([`ChopCorrection::Auto`]) or was imposed.
    pub detected: bool,
}

impl SliceChopping {
    /// Axes whose coherence is too close to zero for the chop decision to be trusted
    /// (`|coherence| < CHOP_AMBIGUOUS_COHERENCE`).
    pub fn ambiguous_axes(&self) -> Vec<usize> {
        (0..3).filter(|&a| self.coherence[a].abs() < CHOP_AMBIGUOUS_COHERENCE).collect()
    }
}

/// Multi-echo magnitude and phase converted from real/imaginary input.
#[derive(Clone, Debug)]
pub struct MagPhaseEchoes {
    /// Per-echo magnitude `|z|`.
    pub magnitudes: Vec<Vec<f64>>,
    /// Per-echo phase `arg z` in `[-π, π]`, after the chop correction (and negated when
    /// `phase_negated`).
    pub phases: Vec<Vec<f64>>,
    pub chopping: SliceChopping,
    /// Whether the phase was negated ([`ComplexToMagPhaseParams::negate_phase`]).
    pub phase_negated: bool,
}

/// Magnitude-weighted neighbour coherence along each axis, pooled over echoes.
///
/// For each axis this is `Re Σ z_v z̄_{v+1} / Σ |z_v| |z_{v+1}|` over neighbouring pairs that both
/// lie in the signal mask, summed over echoes: the magnitude-weighted mean of `cos Δφ` between
/// neighbours. Weighting by magnitude lets the high-signal echoes and voxels dominate, which is
/// where the neighbour phase difference is small and the sign of the result unambiguous.
///
/// The mask is an Otsu threshold on the first echo's magnitude, so background noise (whose
/// neighbour phase differences are uniform and average to zero, diluting the result) does not
/// count. If nothing clears it, every voxel counts.
///
/// # Panics
/// If there are no echoes, `real` and `imag` hold different numbers of echoes, or a volume does
/// not have `grid.n_total()` voxels.
pub fn slice_chopping_coherence(real: &[&[f64]], imag: &[&[f64]], grid: &Grid) -> [f64; 3] {
    check_echoes(real, imag, grid);
    let (nx, ny, nz) = grid.dims;
    let n = grid.n_total();

    let mag0: Vec<f64> = (0..n).map(|v| finite(real[0][v]).hypot(finite(imag[0][v]))).collect();
    let threshold = otsu_threshold(&mag0, 256);
    let mut mask: Vec<bool> = mag0.iter().map(|&m| m > threshold).collect();
    if !mask.iter().any(|&m| m) {
        mask.fill(true);
    }

    let strides = [1, nx, nx * ny];
    let mut num = [0.0f64; 3];
    let mut den = [0.0f64; 3];
    for (re, im) in real.iter().zip(imag) {
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let v = i + j * nx + k * nx * ny;
                    if !mask[v] {
                        continue;
                    }
                    let (a_re, a_im) = (finite(re[v]), finite(im[v]));
                    for (axis, &inside) in [i + 1 < nx, j + 1 < ny, k + 1 < nz].iter().enumerate() {
                        let w = v + strides[axis];
                        if !inside || !mask[w] {
                            continue;
                        }
                        let (b_re, b_im) = (finite(re[w]), finite(im[w]));
                        // Re(a · conj(b)) and |a| |b|
                        num[axis] += a_re * b_re + a_im * b_im;
                        den[axis] += a_re.hypot(a_im) * b_re.hypot(b_im);
                    }
                }
            }
        }
    }
    [0, 1, 2].map(|a| if den[a] > 0.0 { num[a] / den[a] } else { 0.0 })
}

/// Convert multi-echo real/imaginary volumes to magnitude and phase, correcting alternate-slice
/// π chopping.
///
/// With [`ChopCorrection::Auto`] an axis is corrected when its
/// [`slice_chopping_coherence`] is negative, i.e. neighbours along it are, on balance, closer
/// to anti-phase than in-phase. One decision covers all echoes. Check
/// [`SliceChopping::ambiguous_axes`] on the result for axes where that decision was close.
///
/// The correction negates `z` on odd-indexed planes along each corrected axis before taking
/// `arg z`; magnitude is unchanged by it. With `negate_phase` the phase is `arg z̄` instead,
/// which leaves the chop detection unchanged. Non-finite inputs are treated as zero.
///
/// # Panics
/// As [`slice_chopping_coherence`].
pub fn complex_to_mag_phase(
    real: &[&[f64]],
    imag: &[&[f64]],
    grid: &Grid,
    params: &ComplexToMagPhaseParams,
) -> MagPhaseEchoes {
    let coherence = slice_chopping_coherence(real, imag, grid);
    let (corrected, detected) = match params.chop {
        ChopCorrection::Auto => (coherence.map(|c| c < 0.0), true),
        ChopCorrection::Axes(axes) => (axes, false),
    };

    let (nx, ny, nz) = grid.dims;
    let n = grid.n_total();
    let mut magnitudes = Vec::with_capacity(real.len());
    let mut phases = Vec::with_capacity(real.len());
    for (re, im) in real.iter().zip(imag) {
        let mut mag = vec![0.0f64; n];
        let mut phase = vec![0.0f64; n];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let v = i + j * nx + k * nx * ny;
                    let (r, q) = (finite(re[v]), finite(im[v]));
                    let q = if params.negate_phase { -q } else { q };
                    let odd = (corrected[0] && i % 2 == 1) as u8
                        + (corrected[1] && j % 2 == 1) as u8
                        + (corrected[2] && k % 2 == 1) as u8;
                    let (r, q) = if odd % 2 == 1 { (-r, -q) } else { (r, q) };
                    mag[v] = r.hypot(q);
                    phase[v] = q.atan2(r);
                }
            }
        }
        magnitudes.push(mag);
        phases.push(phase);
    }
    debug_assert!(phases.iter().flatten().all(|p| p.abs() <= PI));

    MagPhaseEchoes {
        magnitudes,
        phases,
        chopping: SliceChopping { coherence, corrected, detected },
        phase_negated: params.negate_phase,
    }
}

#[inline]
fn finite(x: f64) -> f64 {
    if x.is_finite() { x } else { 0.0 }
}

fn check_echoes(real: &[&[f64]], imag: &[&[f64]], grid: &Grid) {
    assert!(!real.is_empty(), "complex_to_mag_phase: no echoes");
    assert_eq!(
        real.len(), imag.len(),
        "complex_to_mag_phase: {} real echoes but {} imaginary",
        real.len(), imag.len()
    );
    let n = grid.n_total();
    for (e, (re, im)) in real.iter().zip(imag).enumerate() {
        assert!(
            re.len() == n && im.len() == n,
            "complex_to_mag_phase: echo {e} has {} real and {} imaginary voxels, the grid has {n}",
            re.len(), im.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth multi-echo complex image on a 16³ grid: an ellipsoid of signal in noise-free
    /// background, phase a smooth field times TE plus a constant offset.
    fn smooth_echoes(grid: &Grid, n_echoes: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let (nx, ny, nz) = grid.dims;
        let (mut re, mut im, mut ph) = (Vec::new(), Vec::new(), Vec::new());
        for e in 0..n_echoes {
            let te = (e + 1) as f64;
            let (mut r, mut q, mut p) = (vec![0.0; grid.n_total()], vec![0.0; grid.n_total()], vec![0.0; grid.n_total()]);
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let (x, y, z) = (i as f64 / nx as f64 - 0.5, j as f64 / ny as f64 - 0.5, k as f64 / nz as f64 - 0.5);
                        let v = i + j * nx + k * nx * ny;
                        let inside = x * x / 0.16 + y * y / 0.12 + z * z / 0.14 < 1.0;
                        let m = if inside { 100.0 * (-0.1 * te).exp() } else { 0.5 };
                        let phase = wrap(0.3 + te * (1.5 * x + 0.8 * y * y - 1.2 * z));
                        r[v] = m * phase.cos();
                        q[v] = m * phase.sin();
                        p[v] = phase;
                    }
                }
            }
            re.push(r);
            im.push(q);
            ph.push(p);
        }
        (re, im, ph)
    }

    fn wrap(p: f64) -> f64 {
        (p + PI).rem_euclid(2.0 * PI) - PI
    }

    /// Negate odd-indexed planes along `axis`, as a chopped reconstruction does.
    fn chop(data: &mut [Vec<f64>], grid: &Grid, axis: usize) {
        let (nx, ny, nz) = grid.dims;
        for vol in data.iter_mut() {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        if [i, j, k][axis] % 2 == 1 {
                            vol[i + j * nx + k * nx * ny] *= -1.0;
                        }
                    }
                }
            }
        }
    }

    fn auto() -> ComplexToMagPhaseParams {
        ComplexToMagPhaseParams::default()
    }

    fn axes(a: [bool; 3]) -> ComplexToMagPhaseParams {
        ComplexToMagPhaseParams { chop: ChopCorrection::Axes(a), ..Default::default() }
    }

    fn refs(v: &[Vec<f64>]) -> Vec<&[f64]> {
        v.iter().map(|x| x.as_slice()).collect()
    }

    /// Largest wrapped difference between two phase volumes.
    fn max_phase_err(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| wrap(x - y).abs()).fold(0.0, f64::max)
    }

    #[test]
    fn unchopped_input_is_left_alone() {
        let grid = Grid::new(16, 16, 16, 1.0, 1.0, 1.0);
        let (re, im, ph) = smooth_echoes(&grid, 3);
        let out = complex_to_mag_phase(&refs(&re), &refs(&im), &grid, &auto());
        assert_eq!(out.chopping.corrected, [false; 3]);
        assert!(out.chopping.detected);
        assert!(out.chopping.coherence.iter().all(|&c| c > 0.8), "{:?}", out.chopping.coherence);
        assert!(out.chopping.ambiguous_axes().is_empty());
        for e in 0..3 {
            assert!(max_phase_err(&out.phases[e], &ph[e]) < 1e-12);
            for v in 0..grid.n_total() {
                assert!((out.magnitudes[e][v] - re[e][v].hypot(im[e][v])).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn chopping_is_detected_and_undone_on_each_axis() {
        let grid = Grid::new(16, 14, 12, 1.0, 1.0, 2.0);
        for axis in 0..3 {
            let (mut re, mut im, ph) = smooth_echoes(&grid, 3);
            chop(&mut re, &grid, axis);
            chop(&mut im, &grid, axis);
            let out = complex_to_mag_phase(&refs(&re), &refs(&im), &grid, &auto());
            let mut expected = [false; 3];
            expected[axis] = true;
            assert_eq!(out.chopping.corrected, expected, "axis {axis}: {:?}", out.chopping.coherence);
            assert!(out.chopping.coherence[axis] < -0.8, "axis {axis}: {:?}", out.chopping.coherence);
            // Even planes are untouched, so the phase is recovered exactly, with no global offset.
            for e in 0..3 {
                assert!(max_phase_err(&out.phases[e], &ph[e]) < 1e-12, "axis {axis} echo {e}");
            }
        }
    }

    #[test]
    fn chopping_on_two_axes_is_undone() {
        let grid = Grid::new(16, 16, 16, 1.0, 1.0, 1.0);
        let (mut re, mut im, ph) = smooth_echoes(&grid, 2);
        for axis in [0, 2] {
            chop(&mut re, &grid, axis);
            chop(&mut im, &grid, axis);
        }
        let out = complex_to_mag_phase(&refs(&re), &refs(&im), &grid, &auto());
        assert_eq!(out.chopping.corrected, [true, false, true]);
        assert!(max_phase_err(&out.phases[1], &ph[1]) < 1e-12);
    }

    // One decision for all echoes: a late echo whose own evidence is weak still gets the same
    // planes flipped as the first, so no echo-dependent π offset appears between slices.
    #[test]
    fn one_decision_covers_every_echo() {
        let grid = Grid::new(16, 16, 16, 1.0, 1.0, 1.0);
        let (mut re, mut im, ph) = smooth_echoes(&grid, 2);
        chop(&mut re, &grid, 2);
        chop(&mut im, &grid, 2);
        // Scramble the second echo's phase so it carries no chop evidence of its own.
        let mut seed = 12345u64;
        for v in 0..grid.n_total() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let p = (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 * PI;
            let m = re[1][v].hypot(im[1][v]) * 0.01;
            re[1][v] = m * p.cos();
            im[1][v] = m * p.sin();
        }
        let out = complex_to_mag_phase(&refs(&re), &refs(&im), &grid, &auto());
        assert_eq!(out.chopping.corrected, [false, false, true]);
        assert!(max_phase_err(&out.phases[0], &ph[0]) < 1e-12);
    }

    #[test]
    fn override_is_obeyed_against_the_data() {
        let grid = Grid::new(16, 16, 16, 1.0, 1.0, 1.0);
        let (mut re, mut im, ph) = smooth_echoes(&grid, 2);
        chop(&mut re, &grid, 1);
        chop(&mut im, &grid, 1);

        let off = complex_to_mag_phase(&refs(&re), &refs(&im), &grid, &axes([false; 3]));
        assert_eq!(off.chopping.corrected, [false; 3]);
        assert!(!off.chopping.detected);
        // The coherence is still reported, so a caller can see the override disagreed.
        assert!(off.chopping.coherence[1] < -0.8);
        assert!(max_phase_err(&off.phases[0], &ph[0]) > 3.0);

        let forced = complex_to_mag_phase(&refs(&re), &refs(&im), &grid, &axes([false, true, false]));
        assert_eq!(forced.chopping.corrected, [false, true, false]);
        assert!(max_phase_err(&forced.phases[0], &ph[0]) < 1e-12);
    }

    #[test]
    fn negate_phase_conjugates_and_keeps_the_chop_decision() {
        let grid = Grid::new(16, 16, 16, 1.0, 1.0, 1.0);
        let (mut re, mut im, ph) = smooth_echoes(&grid, 2);
        chop(&mut re, &grid, 2);
        chop(&mut im, &grid, 2);
        let plain = complex_to_mag_phase(&refs(&re), &refs(&im), &grid, &auto());
        let params = ComplexToMagPhaseParams { negate_phase: true, ..Default::default() };
        let negated = complex_to_mag_phase(&refs(&re), &refs(&im), &grid, &params);
        assert!(!plain.phase_negated && negated.phase_negated);
        assert_eq!(negated.chopping, plain.chopping);
        for e in 0..2 {
            let flipped: Vec<f64> = ph[e].iter().map(|p| -p).collect();
            assert!(max_phase_err(&negated.phases[e], &flipped) < 1e-12);
            assert_eq!(negated.magnitudes[e], plain.magnitudes[e]);
        }
    }

    #[test]
    fn noise_only_input_is_ambiguous() {
        let grid = Grid::new(16, 16, 16, 1.0, 1.0, 1.0);
        let mut seed = 7u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let re: Vec<f64> = (0..grid.n_total()).map(|_| next()).collect();
        let im: Vec<f64> = (0..grid.n_total()).map(|_| next()).collect();
        let out = complex_to_mag_phase(&[&re], &[&im], &grid, &auto());
        assert_eq!(out.chopping.ambiguous_axes(), vec![0, 1, 2]);
    }

    #[test]
    fn non_finite_input_becomes_zero() {
        let grid = Grid::new(4, 4, 4, 1.0, 1.0, 1.0);
        let mut re = vec![1.0; 64];
        let im = vec![0.0; 64];
        re[5] = f64::NAN;
        let out = complex_to_mag_phase(&[&re], &[&im], &grid, &axes([false; 3]));
        assert_eq!(out.magnitudes[0][5], 0.0);
        assert_eq!(out.phases[0][5], 0.0);
    }

    #[test]
    #[should_panic(expected = "complex_to_mag_phase: echo 0 has 64 real and 63 imaginary voxels")]
    fn mismatched_volumes_are_refused() {
        let grid = Grid::new(4, 4, 4, 1.0, 1.0, 1.0);
        complex_to_mag_phase(&[&[0.0; 64]], &[&[0.0; 63]], &grid, &auto());
    }
}

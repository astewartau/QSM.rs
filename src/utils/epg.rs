//! EPG-based T2/R2 mapping from multi-echo spin-echo (MESE) magnitude data.
//!
//! A multi-echo spin-echo (CPMG) train uses repeated refocusing pulses. Real
//! refocusing pulses are not perfect 180° flips (B1 inhomogeneity, slice
//! profile), so they generate stimulated and indirect echoes that add signal at
//! later echoes. A naive mono-exponential fit (e.g. [`r2star_arlo`]) reads that
//! extra signal as slower decay and therefore *overestimates* T2 (underestimates
//! R2). The Extended Phase Graph (EPG) formalism models the full echo train,
//! including the stimulated-echo pathways, as a function of `(T2, T1, B1)` and so
//! removes that bias.
//!
//! This module provides:
//! - [`epg_cpmg_echoes`]: the forward model — simulated CPMG echo amplitudes.
//! - [`r2_epg`]: per-voxel R2 mapping by EPG dictionary matching.
//! - [`r2prime`]: R2' = R2* − R2 (the input chi-separation needs).
//!
//! When the refocusing flip angle is exactly 180° (`b1 = 1.0`), the EPG echo
//! train reduces exactly to the mono-exponential `S(TE_n) = exp(-TE_n / T2)`,
//! so on perfectly-refocused data EPG and ARLO agree (see the unit tests).
//!
//! References:
//! - Weigel, M. (2015). "Extended phase graphs: dephasing, RF pulses, and echoes
//!   — pure and simple." JMRI 41(2):266-295.
//! - Ben-Eliezer, N., et al. (2015). "Rapid and accurate T2 mapping from
//!   multi-spin-echo data using Bloch-simulation-based reconstruction." MRM
//!   73(2):809-817.

use num_complex::Complex64;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

const PI: f64 = std::f64::consts::PI;

/// EPG configuration state: three complex vectors indexed by dephasing order
/// `k = 0..=k_max`. `fp[k]` is the F+ (positive dephasing) coherence at order
/// `k`, `fm[k]` the F- (negative dephasing) coherence, and `z[k]` the
/// longitudinal coherence.
struct EpgState {
    fp: Vec<Complex64>,
    fm: Vec<Complex64>,
    z: Vec<Complex64>,
}

impl EpgState {
    fn new(k_max: usize) -> Self {
        let zero = Complex64::new(0.0, 0.0);
        EpgState {
            fp: vec![zero; k_max + 1],
            fm: vec![zero; k_max + 1],
            z: vec![zero; k_max + 1],
        }
    }

    /// Relaxation + longitudinal regrowth (toward M0 = 1) over one interval.
    /// `e1 = exp(-dt/T1)`, `e2 = exp(-dt/T2)`.
    #[inline]
    fn relax(&mut self, e1: f64, e2: f64) {
        for v in self.fp.iter_mut() {
            *v *= e2;
        }
        for v in self.fm.iter_mut() {
            *v *= e2;
        }
        for v in self.z.iter_mut() {
            *v *= e1;
        }
        self.z[0] += 1.0 - e1;
    }

    /// Unit positive dephasing gradient: shift coherence orders. F+ moves to
    /// higher orders, F- to lower orders, and the observable F+(0) is refilled
    /// from conj(F-(0)) (Weigel 2015; Hargreaves EPG `epg_grad`).
    #[inline]
    fn grad(&mut self) {
        let k_max = self.fp.len() - 1;
        // F+ up: fp[k] = fp[k-1]
        for k in (1..=k_max).rev() {
            self.fp[k] = self.fp[k - 1];
        }
        // F- down: fm[k] = fm[k+1]
        for k in 0..k_max {
            self.fm[k] = self.fm[k + 1];
        }
        self.fm[k_max] = Complex64::new(0.0, 0.0);
        // Refill the observable from the conjugate-symmetric partner.
        self.fp[0] = self.fm[0].conj();
    }

    /// Instantaneous RF pulse of flip angle `alpha` (rad) and phase `phi` (rad),
    /// applied to every coherence order (Weigel 2015, Hargreaves `epg_rf`).
    #[inline]
    fn rf(&mut self, alpha: f64, phi: f64) {
        let c = (alpha / 2.0).cos();
        let s = (alpha / 2.0).sin();
        let c2 = c * c;
        let s2 = s * s;
        let sa = alpha.sin();
        let ca = alpha.cos();
        let i = Complex64::i();
        let eip = Complex64::from_polar(1.0, phi); // e^{i phi}
        let ei2p = Complex64::from_polar(1.0, 2.0 * phi); // e^{2i phi}

        let m00 = Complex64::new(c2, 0.0);
        let m01 = ei2p * s2;
        let m02 = -i * eip * sa;
        let m10 = ei2p.conj() * s2;
        let m11 = Complex64::new(c2, 0.0);
        let m12 = i * eip.conj() * sa;
        let m20 = -i * 0.5 * eip.conj() * sa;
        let m21 = i * 0.5 * eip * sa;
        let m22 = Complex64::new(ca, 0.0);

        for k in 0..self.fp.len() {
            let a = self.fp[k];
            let b = self.fm[k];
            let d = self.z[k];
            self.fp[k] = m00 * a + m01 * b + m02 * d;
            self.fm[k] = m10 * a + m11 * b + m12 * d;
            self.z[k] = m20 * a + m21 * b + m22 * d;
        }
    }
}

/// Simulate the CPMG multi-echo spin-echo magnitude train.
///
/// Models a 90° excitation followed by `n_echoes` refocusing pulses of nominal
/// 180° scaled by `b1` (so the actual refocusing flip is `b1 * 180°`), spaced by
/// echo spacing `esp` (seconds). Echoes form at `TE_n = n * esp`.
///
/// # Arguments
/// * `t2` - Transverse relaxation time (seconds)
/// * `t1` - Longitudinal relaxation time (seconds)
/// * `b1` - Refocusing efficiency: fraction of the nominal 180° flip (1.0 = perfect)
/// * `esp` - Echo spacing (seconds)
/// * `n_echoes` - Number of echoes to simulate
///
/// # Returns
/// `n_echoes` echo magnitudes, normalized so equilibrium magnetization M0 = 1.
///
/// With `b1 = 1.0` this returns exactly `exp(-n * esp / t2)` (perfect refocusing).
pub fn epg_cpmg_echoes(t2: f64, t1: f64, b1: f64, esp: f64, n_echoes: usize) -> Vec<f64> {
    // Coherence can reach up to one order per gradient (two per echo interval).
    let k_max = 2 * n_echoes + 2;
    let mut st = EpgState::new(k_max);

    // Equilibrium then 90° excitation about y (phi = pi/2). CPMG requires the
    // excitation and refocusing pulses to be 90° out of phase.
    st.z[0] = Complex64::new(1.0, 0.0);
    st.rf(PI / 2.0, PI / 2.0);

    let e1 = (-esp / 2.0 / t1).exp();
    let e2 = (-esp / 2.0 / t2).exp();
    let alpha = b1 * PI; // refocusing flip angle (rad)

    let mut echoes = Vec::with_capacity(n_echoes);
    for _ in 0..n_echoes {
        // First half-interval: relax + dephase.
        st.relax(e1, e2);
        st.grad();
        // Refocusing pulse about x (phi = 0).
        st.rf(alpha, 0.0);
        // Second half-interval: relax + rephase.
        st.relax(e1, e2);
        st.grad();
        // Echo forms at the observable coherence order 0.
        echoes.push(st.fp[0].norm());
    }
    echoes
}

/// Parameters for [`r2_epg`] dictionary matching.
pub struct R2EpgParams {
    /// Assumed T1 in seconds (weak influence on the T2 estimate).
    pub t1: f64,
    /// Candidate T2 values (seconds) — the dictionary's T2 axis.
    pub t2_grid: Vec<f64>,
    /// Candidate B1 (refocusing efficiency) values — the dictionary's B1 axis.
    pub b1_grid: Vec<f64>,
}

impl Default for R2EpgParams {
    fn default() -> Self {
        // T2 log-spaced 5 ms .. 500 ms (R2 ~ 2 .. 200 Hz).
        let n_t2 = 150;
        let (t2_lo, t2_hi) = (0.005_f64, 0.5_f64);
        let ln_lo = t2_lo.ln();
        let ln_hi = t2_hi.ln();
        let t2_grid: Vec<f64> = (0..n_t2)
            .map(|i| (ln_lo + (ln_hi - ln_lo) * i as f64 / (n_t2 - 1) as f64).exp())
            .collect();

        // B1 (refocusing efficiency) 0.6 .. 1.0 in steps of 0.02.
        let n_b1 = 21;
        let (b1_lo, b1_hi) = (0.6_f64, 1.0_f64);
        let b1_grid: Vec<f64> = (0..n_b1)
            .map(|i| b1_lo + (b1_hi - b1_lo) * i as f64 / (n_b1 - 1) as f64)
            .collect();

        R2EpgParams {
            t1: 1.0,
            t2_grid,
            b1_grid,
        }
    }
}

/// One dictionary atom: a normalized echo train and its `(T2, B1)` labels.
struct DictAtom {
    signal: Vec<f64>, // L2-normalized echo amplitudes
    t2: f64,
    b1: f64,
}

/// Build the EPG dictionary for a given echo spacing and echo count.
fn build_dictionary(params: &R2EpgParams, esp: f64, n_echoes: usize) -> Vec<DictAtom> {
    let mut dict = Vec::with_capacity(params.t2_grid.len() * params.b1_grid.len());
    for &t2 in &params.t2_grid {
        for &b1 in &params.b1_grid {
            let mut sig = epg_cpmg_echoes(t2, params.t1, b1, esp, n_echoes);
            let norm: f64 = sig.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if norm > 1e-30 {
                for v in sig.iter_mut() {
                    *v /= norm;
                }
                dict.push(DictAtom { signal: sig, t2, b1 });
            }
        }
    }
    dict
}

/// R2 mapping from multi-echo spin-echo magnitude via EPG dictionary matching.
///
/// For each masked voxel the (normalized) measured echo train is matched against
/// a dictionary of EPG-simulated trains over a `(T2, B1)` grid; the R2 = 1/T2 of
/// the best match is returned. Normalization removes the proton-density / receive
/// gain, so only the decay *shape* is fit. This corrects the stimulated-echo bias
/// that a mono-exponential fit ([`r2star_arlo`]) suffers on imperfectly-refocused
/// (B1 < 1) data.
///
/// # Arguments
/// * `magnitude` - MESE magnitude, flattened `[v0_e0, v0_e1, ..., v1_e0, ...]`
///   (row-major `(n_voxels, n_echoes)`, same layout as [`r2star_arlo`])
/// * `mask` - Binary brain mask `[nx*ny*nz]` (1 = process, 0 = skip)
/// * `echo_times` - Spin-echo times in seconds `[n_echoes]` (equi-spaced, ≥3)
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `params` - Dictionary grids and assumed T1
///
/// # Returns
/// `(r2_map, b1_map)` - R2 in Hz and the fitted B1 (refocusing efficiency), both
/// `[nx*ny*nz]`.
///
/// # Panics
/// Panics if `echo_times.len() < 3`, echo times are not equi-spaced, or the
/// `magnitude`/`mask` lengths are inconsistent with `grid`.
pub fn r2_epg(
    magnitude: &[f64],
    mask: &[u8],
    echo_times: &[f64],
    grid: &crate::Grid,
    params: &R2EpgParams,
) -> (Vec<f64>, Vec<f64>) {
    let n_echoes = echo_times.len();
    assert!(n_echoes >= 3, "EPG R2 fitting requires at least 3 echoes");
    let n_voxels = grid.n_total();
    assert_eq!(
        magnitude.len(),
        n_voxels * n_echoes,
        "magnitude length must be n_voxels * n_echoes"
    );
    assert_eq!(mask.len(), n_voxels, "mask length must be n_voxels");

    // Sort echoes by time and require (approximately) uniform spacing.
    let sort_indices: Vec<usize> = {
        let mut idx: Vec<usize> = (0..n_echoes).collect();
        idx.sort_by(|&a, &b| echo_times[a].partial_cmp(&echo_times[b]).unwrap());
        idx
    };
    let mut te_sorted: Vec<f64> = echo_times.to_vec();
    te_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let esp = te_sorted[1] - te_sorted[0];
    let diffs: Vec<f64> = te_sorted.windows(2).map(|w| w[1] - w[0]).collect();
    let max_dev = diffs.iter().map(|&d| (d - esp).abs()).fold(0.0_f64, f64::max);
    assert!(
        max_dev <= 1e-4,
        "EPG R2 fitting requires equi-spaced echo times"
    );
    // EPG forms echoes at TE_n = n*esp; require the acquisition to start at esp.
    // (A constant offset would need a different first-interval model.)

    let dict = build_dictionary(params, esp, n_echoes);
    assert!(!dict.is_empty(), "EPG dictionary is empty");

    let mut r2_map = vec![0.0_f64; n_voxels];
    let mut b1_map = vec![0.0_f64; n_voxels];

    // Match each voxel against the dictionary (parallel over voxels).
    let mut out: Vec<(f64, f64)> = vec![(0.0, 0.0); n_voxels];
    maybe_par_chunks_mut!(out.as_mut_slice(), 1)
        .enumerate()
        .for_each(|(v, slot)| {
            if mask[v] == 0 {
                return;
            }
            // Extract and normalize the measured echo train.
            let mut sig: Vec<f64> = sort_indices
                .iter()
                .map(|&ei| magnitude[v * n_echoes + ei])
                .collect();
            let norm: f64 = sig.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm < 1e-20 {
                return;
            }
            for x in sig.iter_mut() {
                *x /= norm;
            }

            // Best match = maximum normalized dot product (cosine similarity).
            let mut best_dot = f64::NEG_INFINITY;
            let mut best = (0.0_f64, 0.0_f64);
            for atom in &dict {
                let dot: f64 = sig
                    .iter()
                    .zip(atom.signal.iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                if dot > best_dot {
                    best_dot = dot;
                    best = (1.0 / atom.t2, atom.b1);
                }
            }
            slot[0] = best;
        });

    for v in 0..n_voxels {
        r2_map[v] = out[v].0;
        b1_map[v] = out[v].1;
    }

    (r2_map, b1_map)
}

/// Compute R2' = R2* − R2, clamped at zero, within the mask.
///
/// R2* (from a gradient-echo acquisition) captures reversible + irreversible
/// dephasing; R2 (from a spin-echo acquisition) captures only the irreversible
/// part. Their difference R2' is the reversible dephasing that chi-separation
/// uses to constrain the paramagnetic/diamagnetic split. Noise can make R2* < R2
/// in some voxels, so negative values are clamped to zero.
///
/// # Arguments
/// * `r2star` - R2* map in Hz `[nx*ny*nz]`
/// * `r2` - R2 map in Hz `[nx*ny*nz]` (co-registered to `r2star`)
/// * `mask` - Binary brain mask `[nx*ny*nz]`
///
/// # Returns
/// R2' map in Hz `[nx*ny*nz]`.
pub fn r2prime(r2star: &[f64], r2: &[f64], mask: &[u8]) -> Vec<f64> {
    assert_eq!(r2star.len(), r2.len(), "r2star and r2 must be the same length");
    assert_eq!(r2star.len(), mask.len(), "mask must match map length");
    (0..r2star.len())
        .map(|i| {
            if mask[i] != 0 {
                (r2star[i] - r2[i]).max(0.0)
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Grid;

    /// With perfect 180° refocusing (b1 = 1), the EPG train must reduce exactly
    /// to the mono-exponential S(TE_n) = exp(-TE_n / T2).
    #[test]
    fn test_epg_perfect_refocus_is_monoexponential() {
        let t2 = 0.060; // 60 ms
        let t1 = 1.2;
        let esp = 0.010; // 10 ms
        let n = 8;
        let echoes = epg_cpmg_echoes(t2, t1, 1.0, esp, n);
        for k in 0..n {
            let te = (k + 1) as f64 * esp;
            let expected = (-te / t2).exp();
            let err = (echoes[k] - expected).abs() / expected;
            assert!(
                err < 1e-6,
                "echo {} EPG {} vs mono-exp {} (rel err {:.2e})",
                k,
                echoes[k],
                expected,
                err
            );
        }
    }

    /// With imperfect refocusing (b1 < 1), stimulated-echo pathways make later
    /// echoes DECAY MORE SLOWLY than a mono-exponential — i.e. the apparent T2 is
    /// biased upward. Verify the later echoes sit above the mono-exp curve.
    #[test]
    fn test_epg_imperfect_refocus_slower_decay() {
        let t2 = 0.060;
        let t1 = 1.2;
        let esp = 0.010;
        let n = 8;
        let echoes = epg_cpmg_echoes(t2, t1, 0.7, esp, n);
        // Normalize both to first echo for a fair shape comparison.
        let first = echoes[0];
        for k in 2..n {
            let te = (k + 1) as f64 * esp;
            let te1 = esp;
            let mono = (-(te - te1) / t2).exp(); // relative to echo 1
            let epg_rel = echoes[k] / first;
            assert!(
                epg_rel > mono,
                "echo {}: EPG rel {} should exceed mono-exp rel {}",
                k,
                epg_rel,
                mono
            );
        }
    }

    /// The headline result: on imperfectly-refocused data, EPG recovers the true
    /// T2 while a mono-exponential (log-linear) fit is biased. We simulate a
    /// b1 = 0.75 train, fit R2 with both, and check EPG is far closer to truth.
    #[test]
    fn test_epg_beats_monoexp_on_imperfect_refocus() {
        let t2_true = 0.070; // 70 ms  -> R2 = 14.29 Hz
        let r2_true = 1.0 / t2_true;
        let t1 = 1.2;
        let b1_true = 0.75;
        let esp = 0.010;
        let n = 10;
        let te: Vec<f64> = (1..=n).map(|k| k as f64 * esp).collect();

        // Simulate one voxel's MESE train (scaled by an arbitrary M0).
        let m0 = 850.0;
        let train = epg_cpmg_echoes(t2_true, t1, b1_true, esp, n);
        let mag: Vec<f64> = train.iter().map(|&v| v * m0).collect();

        let grid = Grid::new(1, 1, 1, 1.0, 1.0, 1.0);
        let mask = vec![1u8];

        // EPG fit (dictionary includes b1 = 0.75 and t2 near 70 ms).
        let params = R2EpgParams::default();
        let (r2_epg_map, b1_map) = r2_epg(&mag, &mask, &te, &grid, &params);
        let r2_epg_val = r2_epg_map[0];
        let epg_err = (r2_epg_val - r2_true).abs() / r2_true;

        // Mono-exponential (ARLO) fit for comparison.
        let (r2_arlo_map, _) = crate::utils::r2star::r2star_arlo(&mag, &mask, &te, &grid);
        let arlo_err = (r2_arlo_map[0] - r2_true).abs() / r2_true;

        // EPG should be within ~5% (grid-limited) and clearly better than ARLO.
        assert!(
            epg_err < 0.06,
            "EPG R2 {:.3} Hz vs true {:.3} Hz (err {:.1}%)",
            r2_epg_val,
            r2_true,
            epg_err * 100.0
        );
        assert!(
            epg_err < arlo_err,
            "EPG err {:.1}% should beat ARLO err {:.1}% (EPG {:.2} Hz, ARLO {:.2} Hz, true {:.2} Hz)",
            epg_err * 100.0,
            arlo_err * 100.0,
            r2_epg_val,
            r2_arlo_map[0],
            r2_true
        );
        // The fitted B1 should land near the truth.
        assert!(
            (b1_map[0] - b1_true).abs() <= 0.06,
            "fitted B1 {} vs true {}",
            b1_map[0],
            b1_true
        );
    }

    /// On perfectly-refocused data, EPG and ARLO should agree (both correct).
    #[test]
    fn test_epg_matches_arlo_on_perfect_refocus() {
        let t2_true = 0.050;
        let r2_true = 1.0 / t2_true;
        let esp = 0.012;
        let n = 6;
        let te: Vec<f64> = (1..=n).map(|k| k as f64 * esp).collect();
        let train = epg_cpmg_echoes(t2_true, 1.0, 1.0, esp, n);
        let mag: Vec<f64> = train.iter().map(|&v| v * 500.0).collect();

        let grid = Grid::new(1, 1, 1, 1.0, 1.0, 1.0);
        let mask = vec![1u8];
        let (r2_epg_map, _) = r2_epg(&mag, &mask, &te, &grid, &R2EpgParams::default());
        let err = (r2_epg_map[0] - r2_true).abs() / r2_true;
        assert!(err < 0.05, "EPG R2 {} vs true {}", r2_epg_map[0], r2_true);
    }

    #[test]
    fn test_r2prime_subtract_and_clamp() {
        let r2star = vec![50.0, 30.0, 10.0, 0.0];
        let r2 = vec![20.0, 35.0, 10.0, 5.0];
        let mask = vec![1u8, 1, 0, 1];
        let rp = r2prime(&r2star, &r2, &mask);
        assert!((rp[0] - 30.0).abs() < 1e-12); // 50 - 20
        assert!(rp[1].abs() < 1e-12); // clamp negative (30 - 35)
        assert!(rp[2].abs() < 1e-12); // masked out
        assert!(rp[3].abs() < 1e-12); // clamp negative (0 - 5)
    }
}

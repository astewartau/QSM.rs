//! DECOMPOSE-QSM: signal-domain paramagnetic/diamagnetic source separation.
//!
//! DECOMPOSE (Chen et al., NeuroImage 2021) separates paramagnetic (χ+, iron) and
//! diamagnetic (χ−, myelin·calcium) susceptibility by fitting a three-compartment
//! complex multi-echo gradient-echo signal per voxel. Each voxel's signal is
//!
//! ```text
//!   S(t) = C+ · exp(−( a·χ+  + R2*₀ + i·(2/3)·χ+·γ·B0)·t)
//!        + C− · exp(−(−a·χ−  + R2*₀ + i·(2/3)·χ−·γ·B0)·t)
//!        + C₀ · exp(−R2*₀·t)
//! ```
//!
//! with `γ = 42.58·2π`, `a = 2π·γ·B0 / (9√3)` the static-dephasing broadening
//! coefficient, χ+ ≥ 0, χ− ≤ 0 (ppm-scale). C+, C−, C₀ are the paramagnetic,
//! diamagnetic and neutral compartment amplitudes and R2*₀ the baseline decay.
//!
//! There are two entry points. [`decompose_complex`] fits this model to the
//! measured complex multi-echo signal (magnitude and phase), with a free
//! per-voxel phase offset and frequency; see its documentation. [`decompose`],
//! described in the rest of this header, follows the QSM-CI reference instead.
//!
//! Like the QSM-CI reference, [`decompose`] synthesizes the *phase* from the provided
//! conventional QSM (χ_total) rather than re-deriving it from raw multi-echo phase —
//! this isolates DECOMPOSE's separation step. The observed complex data per echo
//! is `y = |mag_norm| · exp(−i·(2/3)·χ_total·γ·B0·TE)` (magnitude normalised by its
//! global maximum).
//!
//! The per-voxel fit is a 3-stage alternating bounded least-squares, repeated
//! `n_inner` times: (1) amplitudes `[C+,C−,C₀]` against the linear signal, then
//! (2) `R2*₀` and (3) `[χ+,χ−]` against `log(signal)` (complex log). The residual
//! is the complex difference packed as `[Re; −Im]`, matching the reference.
//!
//! Each source is reconstructed from the fitted parameters via a per-compartment
//! phase accumulation, `−Σ angle(model) / ((2/3)·γ·B0·ΣTE)`: the paramagnetic
//! sub-model (`pscModel`) gives χ+ and the diamagnetic sub-model (`dscModel`)
//! gives |χ−|. χ+ is returned ≥ 0, χ− ≤ 0.
//!
//! **Phase unwrapping.** Every phase the fit and the reconstruction see arrives
//! through `atan2`, on `(−π, π]`. A compartment accumulates `(2/3)·χ·γ·B0·TE`
//! radians, so above `|χ_total| = π / ((2/3)·γ·B0·TE_max)` the late echoes fold
//! back onto that branch — 0.09 ppm at 7 T with a 28 ms last echo, which is
//! ordinary globus pallidus. Left folded, the alternation converges to a
//! degenerate fixed point (χ− pinned at `chi_bound`, R2*₀ and the amplitudes at
//! zero) and returns χ+ = 0 exactly, so the most paramagnetic structure in the
//! volume reads as a hole. Both the stage-2/3 residuals and `recon_phase`
//! therefore walk the echoes in order and keep each phase on the branch nearest
//! its predecessor ([`unwrap_near`]); the synthesized data phase is known in
//! closed form and is used unwrapped rather than round-tripped through `atan2`.
//! The original method, published at 3 T, does not meet this limit in practice —
//! the threshold there is ~0.2 ppm.
//!
//! Unwrapping is only unambiguous while the phase step *between* echoes stays
//! under π, so the χ search is additionally capped at
//! `π / ((2/3)·γ·B0·ΔTE_max)` — 0.31 ppm at 7 T with 8 ms spacing, inside the
//! 0.5 ppm default [`DecomposeParams::chi_bound`]. Past that the model matches a
//! given χ on more than one branch, the stage-3 landscape grows spurious minima,
//! and scattered voxels fall into them; capping the search is what removes the
//! speckle of dropouts that unwrapping alone left in the recovered pallidum.
//!
//! Note that `−Σ angle / den` remains a *compressive* estimator of χ: its gain
//! rises with χ itself (≈0.03 at white-matter χ+, ≈0.87 in the pallidum on the
//! QSM-CI phantom), so it flattens contrast between weak sources. That is
//! inherent to the reference's reconstruction, not to this port — the fitted
//! compartment amplitudes set the gain, and they are only weakly identifiable
//! from data that carries no sub-voxel compartment signature.
//!
//! **Output-mapping note.** The QSM-CI reference *comment* claims a para/dia
//! output swap (χ+ = |DSC|); on the qsm-forward phantom that swap anti-correlates
//! with the ground truth, while the physically-consistent mapping used here
//! (χ+ = |PSC|, from the paramagnetic sub-model) correlates strongly. We therefore
//! do not replicate the reference's swap.
//!
//! Reference:
//! Chen, J., et al. (2021). "Decompose quantitative susceptibility mapping (QSM)
//! to sub-voxel diamagnetic and paramagnetic components based on gradient-echo MRI
//! data." NeuroImage 242:118735. https://doi.org/10.1016/j.neuroimage.2021.118735
//! Reference implementation (QSM-CI port of Tim Ho's open MATLAB DECOMPOSE-QSM).

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use std::f64::consts::PI;

/// Gyromagnetic constant used by the DECOMPOSE reference: `42.58 · 2π`.
const GAMMA: f64 = 42.58 * 2.0 * PI;

/// Parameters for [`decompose`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct DecomposeParams {
    /// Main field strength in Tesla.
    pub b0: f64,
    /// Number of alternating 3-stage fit passes per voxel (reference default: 10).
    pub n_inner: usize,
    /// Upper bound on |χ| in the fit, ppm (reference default: 0.5).
    pub chi_bound: f64,
    /// Max Levenberg–Marquardt iterations per fit stage (reference lsqcurvefit
    /// defaults to a trust-region solve; 30 is ample for these 1–3 parameter fits).
    pub max_lm_iter: usize,
}

impl Default for DecomposeParams {
    fn default() -> Self {
        Self {
            b0: 7.0,
            n_inner: 10,
            chi_bound: 0.5,
            max_lm_iter: 30,
        }
    }
}

/// DECOMPOSE source separation from a provided QSM and multi-echo magnitude.
///
/// # Arguments
/// * `chi_total` — Conventional QSM χ_total in **ppm** (`n_voxels`), used to
///   synthesize the per-echo phase.
/// * `magnitude` — Multi-echo magnitude, flattened as `(n_voxels, n_echoes)` in
///   row-major order (echo fastest per voxel) — the same layout as
///   [`crate::separation::r2star_qsm_from_magnitude`]. Normalised internally by
///   its global maximum.
/// * `echo_times` — Echo times in **seconds** (`n_echoes`).
/// * `mask` — Binary brain mask (`n_voxels`, 1 = inside).
/// * `params` — See [`DecomposeParams`].
/// * `progress` — Progress callback `(voxels_done, voxels_total)`, called
///   periodically over the fitted (masked) voxels.
///
/// # Returns
/// `(chi_pos, chi_neg, chi_total)` in ppm, restricted to `mask` — matching the
/// [`chi_sep_ilsqr`](super::chi_sep_ilsqr)/[`chi_sep_medi`](super::chi_sep_medi)
/// convention: `chi_pos` ≥ 0 (paramagnetic), `chi_neg` ≤ 0 (diamagnetic, signed),
/// and `chi_total = chi_pos + chi_neg`.
pub fn decompose(
    chi_total: &[f64],
    magnitude: &[f64],
    echo_times: &[f64],
    mask: &[u8],
    params: &DecomposeParams,
    mut progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = chi_total.len();
    let ne = echo_times.len();
    assert_eq!(
        magnitude.len(),
        n * ne,
        "magnitude must be n_voxels * n_echoes"
    );
    assert_eq!(mask.len(), n, "mask length must match chi_total");
    assert!(ne >= 3, "DECOMPOSE needs at least 3 echoes");

    let b0 = params.b0;
    let te = echo_times;

    // Global-max magnitude normalisation (matches the reference).
    let gmax = magnitude.iter().cloned().fold(0.0_f64, f64::max).max(1e-30);

    // Reconstruction denominator: (2/3)·γ·B0·ΣTE.
    let den = (2.0 / 3.0) * GAMMA * b0 * te.iter().sum::<f64>();

    // Masked voxel indices (only these are fit).
    let voxels: Vec<usize> = (0..n).filter(|&i| mask[i] != 0).collect();
    let total = voxels.len();

    // Per-voxel fit → (index, chi_pos, chi_neg). Embarrassingly parallel.
    //
    // The fit allocates nothing: every buffer lives in a `Scratch` that rayon
    // hands out once per worker thread and we reuse across voxels. This matters
    // far beyond the usual allocator overhead — the released binaries are static
    // musl builds, whose allocator serialises under concurrent small
    // allocations, and the churn this loop used to produce (~10⁴ allocations per
    // voxel) made the loop get *slower* as threads were added.
    let results: Vec<(usize, f64, f64)> = maybe_par_map_init!(
        voxels,
        || Scratch::new(ne),
        |ws: &mut Scratch, &i: &usize| {
            // Build the complex data y[e] = mag_norm · exp(-i·(2/3)·χ_total·γ·B0·TE).
            let chi = chi_total[i];
            let mut all_zero = true;
            for e in 0..ne {
                let m = magnitude[i * ne + e] / gmax;
                if m > 0.0 {
                    all_zero = false;
                }
                let ph = -(2.0 / 3.0) * chi * GAMMA * b0 * te[e];
                ws.y[e] = (m * ph.cos(), m * ph.sin());
                // log(y) for stages 2 & 3. `ph` is the phase we just synthesized,
                // so keep it as it is rather than recovering a ±π-wrapped copy of
                // it through `atan2` — see the note above `unwrap_near`.
                ws.logy[e] = (0.5 * (m * m).max(1e-300).ln(), ph);
            }
            if all_zero {
                return (i, 0.0, 0.0);
            }

            let (cp, cm, c0, r0, chip, chim) =
                fit_voxel(te, &ws.y, &ws.logy, b0, params, &mut ws.lm);

            // Reconstruct χ+ (from dscModel) and χ− (from pscModel), both as the
            // per-compartment phase accumulation −Σangle/den.
            let dsc = recon_phase(te, b0, den, |t| {
                dsc_model(t, cp, cm, c0, chip, chim, r0, b0)
            });
            let psc = recon_phase(te, b0, den, |t| {
                psc_model(t, cp, cm, c0, chip, chim, r0, b0)
            });
            // pscModel recovers the paramagnetic term (∝ χ+), dscModel the
            // diamagnetic term (∝ |χ−|). We use this physically-correct mapping
            // (χ+ = |PSC|, χ− = −|DSC|), which correlates positively with the
            // qsm-forward phantom's ground truth. Note: the QSM-CI reference
            // *comment* claims the opposite ("χ+ = |DSC|"); that swap anti-
            // correlates with this GT, so we do not replicate it. See module docs.
            (i, psc.abs(), -dsc.abs())
        }
    )
    .collect();

    let mut chi_pos = vec![0.0_f64; n];
    let mut chi_neg = vec![0.0_f64; n];
    let mut chi_out = vec![0.0_f64; n];
    for (i, pos, neg) in results {
        chi_pos[i] = pos;
        chi_neg[i] = neg;
        chi_out[i] = pos + neg;
    }
    progress(total, total);
    (chi_pos, chi_neg, chi_out)
}

/// Parameters for [`decompose_complex`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct DecomposeComplexParams {
    /// Main field strength in Tesla.
    pub b0: f64,
    /// Upper bound on the pool susceptibilities |χc±| in the fit, ppm.
    pub chi_bound: f64,
    /// Upper bound on the baseline decay rate R2*₀ in the fit, Hz.
    pub r2_bound: f64,
    /// Max Levenberg–Marquardt iterations per start.
    pub max_lm_iter: usize,
}

impl Default for DecomposeComplexParams {
    fn default() -> Self {
        Self {
            b0: 7.0,
            chi_bound: 0.5,
            r2_bound: 100.0,
            max_lm_iter: 100,
        }
    }
}

/// `(χc+, χc−)` starting points of the multi-start search in [`decompose_complex`].
/// The joint fit has distinct local minima for weak/strong pool susceptibility;
/// these four cover both orderings of |χc+| vs |χc−| at two scales.
const COMPLEX_STARTS: [(f64, f64); 4] = [(0.05, -0.05), (0.2, -0.2), (0.1, -0.02), (0.02, -0.1)];

/// DECOMPOSE source separation fitted to the measured multi-echo complex signal.
///
/// [`decompose`] fits DECOMPOSE's three-pool model to a signal whose phase is
/// rebuilt as linear in TE from a conventional QSM. The pools' own phase is not
/// linear in TE, so that rebuilt signal is not one the model can produce, and
/// the fit compresses both sources. This entry point instead fits the model to
/// the measured complex data,
///
/// ```text
///   S(t) = [ C+ · exp(−( a·χc+ + R2*₀ + i·(2/3)·χc+·γ·B0)·t)
///          + C− · exp(−(−a·χc− + R2*₀ + i·(2/3)·χc−·γ·B0)·t)
///          + C₀ · exp(−R2*₀·t) ] · exp(i·(φ₀ + Δω·t))
/// ```
///
/// with a free per-voxel phase offset φ₀ and frequency Δω, which absorb the
/// transmit/receive phase and every field not produced inside the voxel
/// (background and neighbouring sources). No QSM is needed. All eight
/// parameters are fitted jointly by bounded Levenberg–Marquardt with an
/// analytic Jacobian, from each of four `(χc+, χc−)` starting points, both
/// directly and after a first pass with the pool χ held at the start. φ₀ and
/// Δω are searched within one period of their seeds (a line through the
/// echo-unwrapped phase). The lowest-cost fit wins, except that a fit whose
/// neutral pool has vanished ([`MIN_NEUTRAL_FRACTION`]) yields to any that has
/// one. The sources returned are the voxel averages
/// `χ+ = C+/(C++C−+C₀)·χc+` and `χ− = C−/(C++C−+C₀)·χc−`.
///
/// The fitted frequency Δω also carries the pools' mean phase slope, so χ_total
/// is not identified from Δω; it is the sum of the two voxel-average sources.
///
/// # Arguments
/// * `magnitude` — Multi-echo magnitude, `(n_voxels, n_echoes)` row-major (echo
///   fastest per voxel), as in [`decompose`]. Each voxel is normalised by its
///   largest echo, so the scale is arbitrary.
/// * `phase` — Multi-echo phase in radians, same layout. Wrapped or unwrapped,
///   background field included or not. Sign convention: phase = +γ·ΔB·TE, so a
///   paramagnetic source advances the phase — the convention the rest of this
///   crate uses for field maps. The phase is unwrapped along the echoes per voxel
///   to seed φ₀ and Δω; the fit itself compares complex signals, so wraps do
///   not enter it.
/// * `echo_times` — Echo times in **seconds** (`n_echoes ≥ 3`).
/// * `mask` — Binary mask (`n_voxels`, 1 = inside).
/// * `params` — See [`DecomposeComplexParams`].
/// * `progress` — Progress callback `(voxels_done, voxels_total)`.
///
/// # Returns
/// `(chi_pos, chi_neg, chi_total)` in ppm, zero outside `mask`; `chi_pos ≥ 0`,
/// `chi_neg ≤ 0`, `chi_total = chi_pos + chi_neg`.
pub fn decompose_complex(
    magnitude: &[f64],
    phase: &[f64],
    echo_times: &[f64],
    mask: &[u8],
    params: &DecomposeComplexParams,
    mut progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = mask.len();
    let ne = echo_times.len();
    assert_eq!(magnitude.len(), n * ne, "magnitude must be n_voxels * n_echoes");
    assert_eq!(phase.len(), n * ne, "phase must be n_voxels * n_echoes");
    assert!(ne >= 3, "DECOMPOSE needs at least 3 echoes");

    let voxels: Vec<usize> = (0..n).filter(|&i| mask[i] != 0).collect();
    let total = voxels.len();
    let te = echo_times;

    let results: Vec<(usize, f64, f64)> = maybe_par_map_init!(
        voxels,
        || ComplexScratch::new(ne),
        |ws: &mut ComplexScratch, &i: &usize| {
            let (pos, neg) = fit_voxel_complex(
                &magnitude[i * ne..(i + 1) * ne],
                &phase[i * ne..(i + 1) * ne],
                te,
                params,
                ws,
            );
            (i, pos, neg)
        }
    )
    .collect();

    let mut chi_pos = vec![0.0_f64; n];
    let mut chi_neg = vec![0.0_f64; n];
    let mut chi_out = vec![0.0_f64; n];
    for (i, pos, neg) in results {
        chi_pos[i] = pos;
        chi_neg[i] = neg;
        chi_out[i] = pos + neg;
    }
    progress(total, total);
    (chi_pos, chi_neg, chi_out)
}

/// Reusable scratch for one voxel of [`decompose_complex`]; every buffer is
/// overwritten before it is read.
struct ComplexScratch {
    /// Normalised complex data per echo, in DECOMPOSE's phase convention.
    y: Vec<(f64, f64)>,
    /// Levenberg–Marquardt buffers.
    lm: LmScratch,
}

impl ComplexScratch {
    fn new(ne: usize) -> Self {
        Self { y: vec![(0.0, 0.0); ne], lm: LmScratch::new(2 * ne) }
    }
}

/// The complex DECOMPOSE model of [`decompose_complex`] over one voxel's echoes,
/// parameterised as `x = [C+, C−, C₀, R2*₀, χc+, χc−, φ₀, Δω]`.
struct ComplexModel<'a> {
    te: &'a [f64],
    /// Static-dephasing broadening coefficient `2π·γ·B0 / (9√3)`.
    a: f64,
    /// Pool frequency per ppm, `(2/3)·γ·B0`.
    w: f64,
}

impl ComplexModel<'_> {
    /// Each pool's term at `t` without the shared `exp(i·(φ₀ + Δω·t))`, as
    /// `[E+, E−, E₀]`, and that shared factor.
    #[inline]
    fn terms(&self, x: &[f64], t: f64) -> ([(f64, f64); 3], (f64, f64)) {
        let (r0, xp, xm) = (x[3], x[4], x[5]);
        let ep = (-(self.a * xp + r0) * t).exp();
        let em = (-(-self.a * xm + r0) * t).exp();
        let (sp, cp) = (-self.w * xp * t).sin_cos();
        let (sm, cm) = (-self.w * xm * t).sin_cos();
        let (sn, cs) = (x[6] + x[7] * t).sin_cos();
        ([(ep * cp, ep * sp), (em * cm, em * sm), ((-r0 * t).exp(), 0.0)], (cs, sn))
    }

    /// Complex signal at `t`.
    #[inline]
    fn signal(&self, x: &[f64], t: f64) -> (f64, f64) {
        let ([tp, tm, t0], rot) = self.terms(x, t);
        let s = (
            x[0] * tp.0 + x[1] * tm.0 + x[2] * t0.0,
            x[0] * tp.1 + x[1] * tm.1 + x[2] * t0.1,
        );
        cmul(s, rot)
    }

    /// Residual `[Re(S − y); Im(S − y)]`.
    fn resid(&self, x: &[f64], y: &[(f64, f64)], out: &mut [f64]) {
        let ne = self.te.len();
        for (e, &t) in self.te.iter().enumerate() {
            let mdl = self.signal(x, t);
            out[e] = mdl.0 - y[e].0;
            out[ne + e] = mdl.1 - y[e].1;
        }
    }

    /// Analytic Jacobian of [`Self::resid`], `2·n_echoes × 8`, column-major.
    fn jac(&self, x: &[f64], jac: &mut [f64]) {
        let ne = self.te.len();
        let m = 2 * ne;
        let (a, w) = (self.a, self.w);
        for (e, &t) in self.te.iter().enumerate() {
            let ([tp, tm, t0], rot) = self.terms(x, t);
            let ep = cmul(tp, rot);
            let em = cmul(tm, rot);
            let e0 = cmul(t0, rot);
            let mdl = (
                x[0] * ep.0 + x[1] * em.0 + x[2] * e0.0,
                x[0] * ep.1 + x[1] * em.1 + x[2] * e0.1,
            );
            // ∂/∂χc+ = C+·E+·(−(a + i·w)·t);  ∂/∂χc− = C−·E−·((a − i·w)·t).
            let dxp = cmul((x[0] * ep.0, x[0] * ep.1), (-a * t, -w * t));
            let dxm = cmul((x[1] * em.0, x[1] * em.1), (a * t, -w * t));
            let cols = [
                ep,
                em,
                e0,
                (-t * mdl.0, -t * mdl.1),
                dxp,
                dxm,
                (-mdl.1, mdl.0),
                (-t * mdl.1, t * mdl.0),
            ];
            for (j, c) in cols.iter().enumerate() {
                jac[j * m + e] = c.0;
                jac[j * m + ne + e] = c.1;
            }
        }
    }
}

/// Complex product of `(re, im)` pairs.
#[inline]
fn cmul(p: (f64, f64), q: (f64, f64)) -> (f64, f64) {
    (p.0 * q.0 - p.1 * q.1, p.0 * q.1 + p.1 * q.0)
}

/// Largest |χ| the echo spacing represents without the per-echo phase step of a
/// pool exceeding π, capped at `chi_bound`. Past it the model matches a given χ
/// on more than one branch (see the module notes).
fn chi_search_bound(te: &[f64], b0: f64, chi_bound: f64) -> f64 {
    let dte = te.windows(2).map(|w| w[1] - w[0]).fold(te[0], f64::max);
    chi_bound.min(PI / ((2.0 / 3.0) * GAMMA * b0 * dte))
}

/// Joint 8-parameter fit of one voxel's complex signal. Returns the voxel-average
/// `(χ+, χ−)`; `(0, 0)` for a voxel without signal.
fn fit_voxel_complex(
    mag: &[f64],
    phase: &[f64],
    te: &[f64],
    params: &DecomposeComplexParams,
    ws: &mut ComplexScratch,
) -> (f64, f64) {
    let ne = te.len();
    let smax = mag.iter().cloned().fold(0.0_f64, f64::max);
    if smax.is_nan() || smax <= 0.0 || phase.iter().any(|p| !p.is_finite()) {
        return (0.0, 0.0);
    }
    // Data in DECOMPOSE's convention, where a paramagnetic pool's phase is
    // −(2/3)·χ·γ·B0·t: the conjugate of the input's +γ·ΔB·TE convention. The
    // phase, unwrapped along the echoes, is fitted with a line to seed φ₀ and Δω.
    // Nothing is seeded from the magnitude decay: beating between the pools
    // modulates it, and a log-linear R2* or S₀ from it lands the fit in a wrong
    // basin more often than the fixed seeds below do.
    let (mut st, mut stt, mut sp, mut stp) = (0.0, 0.0, 0.0, 0.0);
    let mut prev = 0.0;
    for e in 0..ne {
        let m = mag[e] / smax;
        let ph = -phase[e];
        ws.y[e] = (m * ph.cos(), m * ph.sin());
        let ang = if e == 0 { ph } else { unwrap_near(ph, prev) };
        prev = ang;
        let t = te[e];
        st += t;
        stt += t * t;
        sp += ang;
        stp += t * ang;
    }
    let nf = ne as f64;
    let dw0 = (nf * stp - st * sp) / (nf * stt - st * st);
    let phi0 = (sp - dw0 * st) / nf;

    let b0 = params.b0;
    let a = (2.0 * PI * GAMMA * b0) / (9.0 * 3.0_f64.sqrt());
    let w = (2.0 / 3.0) * GAMMA * b0;
    let ub = chi_search_bound(te, b0, params.chi_bound);
    // The data are normalised to their largest echo, so the pool amplitudes sum
    // to ~exp(R2*·TE₁) ≥ 1; 3 leaves room for any plausible decay before TE₁.
    let amp_ub = 3.0;
    let r2_ub = params.r2_bound;
    // φ₀ and Δω are only defined modulo 2π and (for evenly spaced echoes)
    // modulo 2π/ΔTE, so a window of one period around each seed loses no
    // solution, and keeps a long first step from carrying the fit into a
    // distant, aliased one.
    let dte = te.windows(2).map(|w| w[1] - w[0]).fold(0.0, f64::max);
    let dw_half = PI / dte;
    let lb = [0.0, 0.0, 0.0, 0.0, 0.0, -ub, phi0 - PI, 0.0];
    let ubv = [amp_ub, amp_ub, amp_ub, r2_ub, ub, 0.0, phi0 + PI, 0.0];

    let model = ComplexModel { te, a, w };
    let y = &ws.y;
    let resid = |x: &[f64], out: &mut [f64]| model.resid(x, y, out);
    let jac = |x: &[f64], s: &mut LmScratch| model.jac(x, &mut s.jac);
    let maxit = params.max_lm_iter;

    let mut best: Option<([f64; LM_MAX_N], f64)> = None;
    for &(xp0, xm0) in &COMPLEX_STARTS {
        let xp0 = xp0.min(ub);
        let xm0 = xm0.max(-ub);
        // The pools' mean phase slope is part of the model; take it out of the
        // measured slope so the seed's total frequency matches the data.
        let dw_seed = dw0 + w * 0.2 * (xp0 + xm0);
        let x0 = [0.2, 0.2, 0.6, 10.0_f64.min(r2_ub), xp0, xm0, phi0, dw_seed];
        let (mut lo, mut hi) = (lb, ubv);
        (lo[7], hi[7]) = (dw_seed - dw_half, dw_seed + dw_half);
        // Each start is fitted twice: jointly from the seed, and jointly after
        // first settling the amplitudes, R2*₀, φ₀ and Δω with the pool χ held at
        // the start. Neither path dominates: the direct one is the more reliable
        // with few echoes at 3 T, the staged one where the pools dephase within
        // the first echoes (7 T), which leaves the linear phase seed far off.
        let direct = lm_core(&resid, jac, &x0, &lo, &hi, maxit, true, &mut ws.lm);
        let (mut lo_held, mut hi_held) = (lo, hi);
        (lo_held[4], hi_held[4], lo_held[5], hi_held[5]) = (xp0, xp0, xm0, xm0);
        let pre = lm_core(&resid, jac, &x0, &lo_held, &hi_held, maxit, true, &mut ws.lm);
        let staged = lm_core(&resid, jac, &pre.0[..8], &lo, &hi, maxit, true, &mut ws.lm);
        for fit in [direct, staged] {
            if best.is_none_or(|b| better_complex_fit(&fit, &b)) {
                best = Some(fit);
            }
        }
    }
    let x = best.map_or([0.0; LM_MAX_N], |b| b.0);
    let sum = x[0] + x[1] + x[2];
    if sum.is_nan() || sum <= 0.0 {
        return (0.0, 0.0);
    }
    ((x[0] / sum * x[4]).max(0.0), (x[1] / sum * x[5]).min(0.0))
}

/// Reusable scratch for one voxel's fit. Every buffer is fully overwritten
/// before it is read, so a `Scratch` shared across voxels (or across a rayon
/// worker's whole share of them) cannot carry state between them.
struct Scratch {
    /// Complex data per echo.
    y: Vec<(f64, f64)>,
    /// `log(y)` per echo.
    logy: Vec<(f64, f64)>,
    /// Levenberg–Marquardt buffers.
    lm: LmScratch,
}

impl Scratch {
    fn new(ne: usize) -> Self {
        Self {
            y: vec![(0.0, 0.0); ne],
            logy: vec![(0.0, 0.0); ne],
            lm: LmScratch::new(2 * ne),
        }
    }
}

/// Smallest neutral-pool fraction `C₀/(C++C−+C₀)` a [`decompose_complex`] fit is
/// preferred with. Below it the fit is the pool-relabelling degeneracy rather
/// than a DECOMPOSE voxel: one source pool stands in for the tissue water, its
/// broadening `a·|χc|` playing R2*₀ and Δω shifting to absorb its frequency.
/// Under noise that relabelling often fits a few hundredths better than the
/// physical solution and then wins on cost alone, returning a voxel whose whole
/// signal is a source; on DECOMPOSE-model voxels at 3 T and 7 T it was most of
/// the gross outliers.
const MIN_NEUTRAL_FRACTION: f64 = 0.02;

/// Whether candidate fit `a` beats `b` (`(x, cost)` pairs): a fit with a
/// neutral pool of at least [`MIN_NEUTRAL_FRACTION`] beats one without, and
/// otherwise the lower cost wins.
fn better_complex_fit(a: &([f64; LM_MAX_N], f64), b: &([f64; LM_MAX_N], f64)) -> bool {
    let physical = |x: &[f64; LM_MAX_N]| {
        let sum = x[0] + x[1] + x[2];
        sum > 0.0 && x[2] >= MIN_NEUTRAL_FRACTION * sum
    };
    match (physical(&a.0), physical(&b.0)) {
        (true, false) => true,
        (false, true) => false,
        _ => a.1 < b.1,
    }
}

/// The three-stage alternating fit for a single voxel. Returns
/// `(C+, C−, C₀, R2*₀, χ+, χ−)`.
#[allow(clippy::too_many_arguments)]
fn fit_voxel(
    te: &[f64],
    y: &[(f64, f64)],
    logy: &[(f64, f64)],
    b0: f64,
    params: &DecomposeParams,
    lm: &mut LmScratch,
) -> (f64, f64, f64, f64, f64, f64) {
    let ne = te.len();
    // Initial values (reference).
    let mut c = [0.3_f64, 0.3, 0.4]; // [C+, C−, C₀]
    let mut r0 = 25.0_f64;
    let mut chi = [0.05_f64, -0.05]; // [χ+, χ−]

    // Cap the χ search at the largest |χ| the echo spacing can represent without
    // aliasing. Unwrapping puts each echo's phase on the branch nearest the
    // previous one, which is only unambiguous while the step between echoes stays
    // under π; past that the model can match a given χ on more than one branch,
    // the stage-3 landscape grows spurious minima, and scattered voxels fall into
    // them. At 7 T with 8 ms spacing the limit is 0.31 ppm, well inside the 0.5
    // ppm default — which is what left the recovered pallidum speckled with
    // dropouts even after the phases were unwrapped.
    let ub = chi_search_bound(te, b0, params.chi_bound);
    let inf = f64::INFINITY;
    let maxit = params.max_lm_iter;

    for _ in 0..params.n_inner {
        // Stage 1: amplitudes C against the linear signal y.
        {
            let resid = |x: &[f64], out: &mut [f64]| {
                for e in 0..ne {
                    let m = signal_model(te[e], x[0], x[1], x[2], chi[0], chi[1], r0, b0);
                    out[e] = m.0 - y[e].0;
                    out[ne + e] = -(m.1 - y[e].1);
                }
            };
            let x = lm_bounded(resid, &c, &[0.0, 0.0, 0.0], &[inf, inf, inf], maxit, lm);
            c = [x[0], x[1], x[2]];
        }
        // Stage 2: R2*₀ against log(y).
        {
            let resid = |x: &[f64], out: &mut [f64]| {
                let mut prev = 0.0;
                for e in 0..ne {
                    let m = clog(signal_model(
                        te[e], c[0], c[1], c[2], chi[0], chi[1], x[0], b0,
                    ));
                    let ang = if e == 0 { m.1 } else { unwrap_near(m.1, prev) };
                    prev = ang;
                    out[e] = m.0 - logy[e].0;
                    out[ne + e] = -(ang - logy[e].1);
                }
            };
            let x = lm_bounded(resid, &[r0], &[0.0], &[inf], maxit, lm);
            r0 = x[0];
        }
        // Stage 3: [χ+, χ−] against log(y).
        {
            let resid = |x: &[f64], out: &mut [f64]| {
                let mut prev = 0.0;
                for e in 0..ne {
                    let m = clog(signal_model(te[e], c[0], c[1], c[2], x[0], x[1], r0, b0));
                    let ang = if e == 0 { m.1 } else { unwrap_near(m.1, prev) };
                    prev = ang;
                    out[e] = m.0 - logy[e].0;
                    out[ne + e] = -(ang - logy[e].1);
                }
            };
            let x = lm_bounded(resid, &chi, &[0.0, -ub], &[ub, 0.0], maxit, lm);
            chi = [x[0], x[1]];
        }
    }
    (c[0], c[1], c[2], r0, chi[0], chi[1])
}

/// Complex DECOMPOSE signal model at echo time `t` (seconds). Returns `(re, im)`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn signal_model(
    t: f64,
    cp: f64,
    cm: f64,
    c0: f64,
    chip: f64,
    chim: f64,
    r0: f64,
    b0: f64,
) -> (f64, f64) {
    let a = (2.0 * PI * GAMMA * b0) / (9.0 * 3.0_f64.sqrt());
    // term1: paramagnetic
    let d1 = a * chip + r0;
    let w1 = (2.0 / 3.0) * chip * GAMMA * b0;
    let t1 = cexp_decay(cp, d1, w1, t);
    // term2: diamagnetic (χ− ≤ 0, so −a·χ− ≥ 0 adds decay)
    let d2 = -a * chim + r0;
    let w2 = (2.0 / 3.0) * chim * GAMMA * b0;
    let t2 = cexp_decay(cm, d2, w2, t);
    // term3: neutral
    let t3 = (c0 * (-r0 * t).exp(), 0.0);
    (t1.0 + t2.0 + t3.0, t1.1 + t2.1 + t3.1)
}

/// Paramagnetic-only reconstruction sub-model (`pscModel.m`).
#[allow(clippy::too_many_arguments)]
#[inline]
fn psc_model(
    t: f64,
    cp: f64,
    cm: f64,
    c0: f64,
    chip: f64,
    _chim: f64,
    r0: f64,
    b0: f64,
) -> (f64, f64) {
    let a = (2.0 * PI * GAMMA * b0) / (9.0 * 3.0_f64.sqrt());
    let d1 = a * chip + r0;
    let w1 = (2.0 / 3.0) * chip * GAMMA * b0;
    let t1 = cexp_decay(cp, d1, w1, t);
    let t3 = ((c0 + cm) * (-r0 * t).exp(), 0.0);
    (t1.0 + t3.0, t1.1 + t3.1)
}

/// Diamagnetic-only reconstruction sub-model (`dscModel.m`). Note the flipped
/// imaginary sign `(2/3)·(−χ−)·γ·B0` relative to [`signal_model`].
#[allow(clippy::too_many_arguments)]
#[inline]
fn dsc_model(
    t: f64,
    cp: f64,
    cm: f64,
    c0: f64,
    _chip: f64,
    chim: f64,
    r0: f64,
    b0: f64,
) -> (f64, f64) {
    let a = (2.0 * PI * GAMMA * b0) / (9.0 * 3.0_f64.sqrt());
    let d2 = -a * chim + r0;
    let w2 = (2.0 / 3.0) * (-chim) * GAMMA * b0; // flipped sign
    let t2 = cexp_decay(cm, d2, w2, t);
    let t3 = ((c0 + cp) * (-r0 * t).exp(), 0.0);
    (t2.0 + t3.0, t2.1 + t3.1)
}

/// `amp · exp(−(decay + i·freq)·t)` as `(re, im)`.
#[inline]
fn cexp_decay(amp: f64, decay: f64, freq: f64, t: f64) -> (f64, f64) {
    let mag = amp * (-decay * t).exp();
    let ang = -freq * t;
    (mag * ang.cos(), mag * ang.sin())
}

/// Complex natural log: `log(z) = log|z| + i·angle(z)`.
#[inline]
fn clog(z: (f64, f64)) -> (f64, f64) {
    let mag2 = z.0 * z.0 + z.1 * z.1;
    (0.5 * mag2.max(1e-300).ln(), z.1.atan2(z.0))
}

/// `2π`, the period of the phase branch cut.
const TAU: f64 = 2.0 * PI;

/// Put `ang` on the 2π-branch nearest `prev`.
///
/// Every phase here reaches us through `atan2`, which folds onto `(−π, π]`. That
/// fold is not harmless: a compartment accumulates `(2/3)·χ·γ·B0·TE` radians, so
/// at `|χ_total| > π / ((2/3)·γ·B0·TE_max)` — 0.09 ppm at 7 T with a 28 ms last
/// echo, i.e. inside the globus pallidus — the late echoes fold back and the
/// alternating fit chases the fold into a degenerate fixed point (χ− pinned at
/// its bound, R2*₀ and the amplitudes driven to zero, χ+ stuck at 0). Walking the
/// echoes in order and keeping each phase on the branch nearest its predecessor
/// restores the monotone ramp the model assumes.
///
/// The first echo is left on its principal branch: it anchors the series, and at
/// any sane `TE₁` its phase is well inside `(−π, π]` for `|χ| ≤ chi_bound`.
#[inline]
fn unwrap_near(ang: f64, prev: f64) -> f64 {
    ang + ((prev - ang) / TAU).round() * TAU
}

/// Reconstruct a source value: `−Σ angle(model(TE)) / den`, with the phase
/// unwrapped along the echo axis so a strong source is not folded back to zero
/// (see [`unwrap_near`]).
fn recon_phase(te: &[f64], _b0: f64, den: f64, model: impl Fn(f64) -> (f64, f64)) -> f64 {
    let mut s = 0.0;
    let mut prev = 0.0;
    for (e, &t) in te.iter().enumerate() {
        let z = model(t);
        let raw = z.1.atan2(z.0);
        let ang = if e == 0 { raw } else { unwrap_near(raw, prev) };
        prev = ang;
        s += ang;
    }
    -s / den
}

// ---------------------------------------------------------------------------
// Bounded Levenberg–Marquardt for small (n ≤ 3) least-squares problems.
// ---------------------------------------------------------------------------

/// Largest parameter count any DECOMPOSE fit solves for: the 8-parameter joint
/// fit of [`decompose_complex`]. The alternating stages of [`decompose`] use 1–3.
/// Keeps the LM state on the stack.
const LM_MAX_N: usize = 8;

/// Scratch buffers for [`lm_bounded`], sized once for a residual of length `m`
/// and reused across every stage, LM iteration and voxel. Contents are always
/// written before they are read.
struct LmScratch {
    /// Residual at the current `x`.
    r: Vec<f64>,
    /// Residual at the finite-difference probe point.
    rp: Vec<f64>,
    /// Residual at the trial point.
    rn: Vec<f64>,
    /// Jacobian, `m × LM_MAX_N`, column-major.
    jac: Vec<f64>,
}

impl LmScratch {
    fn new(m: usize) -> Self {
        Self {
            r: vec![0.0; m],
            rp: vec![0.0; m],
            rn: vec![0.0; m],
            jac: vec![0.0; m * LM_MAX_N],
        }
    }
}

/// Minimise ‖resid(x)‖² over the box `[lb, ub]` by Levenberg–Marquardt with a
/// forward finite-difference Jacobian and per-step projection onto the box.
/// Sized for the 1–3 parameter DECOMPOSE stages.
///
/// `resid` writes the residual for a parameter vector into the caller's buffer
/// rather than returning a fresh one, and all other working storage is either a
/// stack array (`n ≤ LM_MAX_N`) or a buffer in `s`, so a solve allocates
/// nothing. Only the first `x0.len()` entries of the returned array are
/// meaningful.
fn lm_bounded<F>(
    resid: F,
    x0: &[f64],
    lb: &[f64],
    ub: &[f64],
    max_iter: usize,
    s: &mut LmScratch,
) -> [f64; LM_MAX_N]
where
    F: Fn(&[f64], &mut [f64]),
{
    let fd_jac = |x: &[f64], s: &mut LmScratch| {
        let n = x.len();
        let m = s.r.len();
        for j in 0..n {
            let h = 1e-6 * x[j].abs().max(1e-3);
            let mut xp = [0.0_f64; LM_MAX_N];
            xp[..n].copy_from_slice(x);
            xp[j] = (x[j] + h).clamp(lb[j], ub[j]);
            let hj = xp[j] - x[j];
            let step = if hj.abs() < 1e-30 { h } else { hj };
            if hj.abs() < 1e-30 {
                xp[j] = x[j] + h; // allow out-of-box probe when pinned at a bound
            }
            resid(&xp[..n], &mut s.rp);
            for k in 0..m {
                s.jac[j * m + k] = (s.rp[k] - s.r[k]) / step;
            }
        }
    };
    lm_core(&resid, fd_jac, x0, lb, ub, max_iter, false, s).0
}

/// Levenberg–Marquardt over the box `[lb, ub]` with a caller-supplied
/// Jacobian. `jac(x, s)` fills `s.jac` (`m × n`, column-major) at `x`; it may
/// read `s.r`, which holds the residual at `x`, and use `s.rp` as scratch.
/// Returns the solution and its cost `‖r‖²`. Allocates nothing.
///
/// With `active_set`, a parameter sitting on a bound that the gradient pushes
/// outward is held there for the step instead of being stepped and clamped.
/// Clamping alone distorts the step of every other parameter, the damped
/// retries then shrink it until the decrease falls under `ftol`, and the solve
/// stops short of the minimum — at noise levels where a pool amplitude, R2*₀
/// or a pool χ belongs on its bound, that left a sizeable fraction of
/// [`decompose_complex`] voxels several-fold above the optimum cost. The
/// alternating fit of [`decompose`] keeps the plain clamped step its results
/// were validated with.
fn lm_core<F, J>(
    resid: &F,
    jac: J,
    x0: &[f64],
    lb: &[f64],
    ub: &[f64],
    max_iter: usize,
    active_set: bool,
    s: &mut LmScratch,
) -> ([f64; LM_MAX_N], f64)
where
    F: Fn(&[f64], &mut [f64]),
    J: Fn(&[f64], &mut LmScratch),
{
    let n = x0.len();
    debug_assert!(n <= LM_MAX_N);
    let m = s.r.len();
    let mut x = [0.0_f64; LM_MAX_N];
    for d in 0..n {
        x[d] = x0[d].clamp(lb[d], ub[d]);
    }
    resid(&x[..n], &mut s.r);
    let mut cost = dot(&s.r, &s.r);
    let mut lambda = 1e-3;
    let ftol = 1e-10;

    for _ in 0..max_iter {
        jac(&x[..n], s);
        // Normal equations A = JᵀJ (n×n), g = Jᵀr (n).
        let mut a = [0.0_f64; LM_MAX_N * LM_MAX_N];
        let mut g = [0.0_f64; LM_MAX_N];
        for jc in 0..n {
            for jr in 0..=jc {
                let acc = dot(&s.jac[jr * m..(jr + 1) * m], &s.jac[jc * m..(jc + 1) * m]);
                a[jr * n + jc] = acc;
                a[jc * n + jr] = acc;
            }
            g[jc] = dot(&s.jac[jc * m..(jc + 1) * m], &s.r);
        }

        // Parameters held on their bound for this step (see `active_set`).
        let mut held = [false; LM_MAX_N];
        if active_set {
            for d in 0..n {
                // The step is x − dx with dx ≈ A⁻¹g, so g > 0 pushes x down.
                held[d] = (x[d] <= lb[d] && g[d] > 0.0) || (x[d] >= ub[d] && g[d] < 0.0);
            }
        }

        // Inner loop: inflate lambda until a step decreases the cost.
        let mut improved = false;
        for _ in 0..30 {
            let mut al = a;
            let mut gl = g;
            for d in 0..n {
                al[d * n + d] += lambda * a[d * n + d].max(1e-12);
                if held[d] {
                    for k in 0..n {
                        al[d * n + k] = 0.0;
                        al[k * n + d] = 0.0;
                    }
                    al[d * n + d] = 1.0;
                    gl[d] = 0.0;
                }
            }
            let g = &gl;
            let Some(dx) = solve_small(&al, g, n) else {
                lambda *= 2.5;
                continue;
            };
            let mut xn = [0.0_f64; LM_MAX_N];
            for d in 0..n {
                xn[d] = (x[d] - dx[d]).clamp(lb[d], ub[d]);
            }
            resid(&xn[..n], &mut s.rn);
            let cn = dot(&s.rn, &s.rn);
            if cn < cost {
                let rel = (cost - cn) / cost.max(1e-300);
                x = xn;
                s.r.copy_from_slice(&s.rn);
                cost = cn;
                lambda = (lambda * 0.4).max(1e-12);
                improved = true;
                if rel < ftol {
                    return (x, cost);
                }
                break;
            } else {
                lambda *= 2.5;
                if lambda > 1e12 {
                    break;
                }
            }
        }
        if !improved {
            break;
        }
    }
    (x, cost)
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Solve `A x = b` for small `n` (≤ `LM_MAX_N`) by Gaussian elimination with
/// partial pivoting. `a` is row-major `n×n`. Returns `None` if singular. Only
/// the first `n` entries of the inputs and the result are used.
fn solve_small(
    a: &[f64; LM_MAX_N * LM_MAX_N],
    b: &[f64; LM_MAX_N],
    n: usize,
) -> Option<[f64; LM_MAX_N]> {
    let mut m = *a;
    let mut y = *b;
    for col in 0..n {
        // Partial pivot.
        let mut piv = col;
        let mut best = m[col * n + col].abs();
        for r in (col + 1)..n {
            let v = m[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-300 {
            return None;
        }
        if piv != col {
            for c in 0..n {
                m.swap(col * n + c, piv * n + c);
            }
            y.swap(col, piv);
        }
        let d = m[col * n + col];
        for r in (col + 1)..n {
            let f = m[r * n + col] / d;
            if f != 0.0 {
                for c in col..n {
                    m[r * n + c] -= f * m[col * n + c];
                }
                y[r] -= f * y[col];
            }
        }
    }
    let mut x = [0.0_f64; LM_MAX_N];
    for col in (0..n).rev() {
        let mut acc = y[col];
        for c in (col + 1)..n {
            acc -= m[col * n + c] * x[c];
        }
        x[col] = acc / m[col * n + col];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lm_recovers_linear_fit() {
        // Fit y = m·t + c via LM on residual [m·t+c − data].
        let t = [0.0, 1.0, 2.0, 3.0];
        let data = [1.0, 3.0, 5.0, 7.0]; // m=2, c=1
        let resid = |x: &[f64], out: &mut [f64]| {
            for (k, (&ti, d)) in t.iter().zip(data).enumerate() {
                out[k] = x[0] * ti + x[1] - d;
            }
        };
        let mut lm = LmScratch::new(t.len());
        let x = lm_bounded(
            resid,
            &[0.0, 0.0],
            &[-10.0, -10.0],
            &[10.0, 10.0],
            50,
            &mut lm,
        );
        assert!((x[0] - 2.0).abs() < 1e-6, "slope {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-6, "intercept {}", x[1]);
    }

    #[test]
    fn solve_small_3x3() {
        // A x = b with a known solution.
        let a: [f64; 9] = [2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 0.0];
        let x_true = [1.0, 2.0, 3.0];
        let b: [f64; 3] = [
            a[0] * x_true[0] + a[1] * x_true[1] + a[2] * x_true[2],
            a[3] * x_true[0] + a[4] * x_true[1] + a[5] * x_true[2],
            a[6] * x_true[0] + a[7] * x_true[1] + a[8] * x_true[2],
        ];
        // `solve_small` takes LM_MAX_N-sized buffers, using the leading n×n block.
        let mut ap = [0.0; LM_MAX_N * LM_MAX_N];
        ap[..9].copy_from_slice(&a);
        let mut bp = [0.0; LM_MAX_N];
        bp[..3].copy_from_slice(&b);
        let x = solve_small(&ap, &bp, 3).unwrap();
        for i in 0..3 {
            assert!((x[i] - x_true[i]).abs() < 1e-9, "x{i} = {}", x[i]);
        }
    }

    /// Sign/plumbing check on a synthetic voxel. Verifies the output sign
    /// convention and the physically-correct mapping: χ+ = |PSC| (paramagnetic
    /// sub-model), χ− = −|DSC| (diamagnetic sub-model). A voxel with a large
    /// paramagnetic fit amplitude/χ therefore yields a dominant χ+.
    #[test]
    fn decompose_signs_on_synthetic_voxel() {
        let te = [0.004, 0.012, 0.020, 0.028];
        let b0 = 7.0;
        let params = DecomposeParams {
            b0,
            n_inner: 12,
            chi_bound: 0.5,
            max_lm_iter: 40,
        };

        // Forward: paramagnetic-dominant fit parameters.
        let (cp, cm, c0, chip, chim, r0) = (0.5, 0.1, 0.4, 0.10, -0.02, 20.0);
        let sig: Vec<(f64, f64)> = te
            .iter()
            .map(|&t| signal_model(t, cp, cm, c0, chip, chim, r0, b0))
            .collect();
        let chi_total_val = chip + chim;
        let n = 1;
        let ne = te.len();
        let mut mag = vec![0.0_f64; n * ne];
        for e in 0..ne {
            mag[e] = (sig[e].0 * sig[e].0 + sig[e].1 * sig[e].1).sqrt();
        }
        let chi_total = vec![chi_total_val; n];
        let mask = vec![1u8; n];
        let (pos, neg, tot) = decompose(&chi_total, &mag, &te, &mask, &params, |_, _| {});
        // Sign convention + invariant.
        assert!(pos[0] >= 0.0, "χ+ must be ≥ 0");
        assert!(neg[0] <= 0.0, "χ− must be ≤ 0");
        assert!(
            (tot[0] - (pos[0] + neg[0])).abs() < 1e-12,
            "χ_total = χ+ + χ−"
        );
        // Physically-correct mapping: paramagnetic fit → χ+ (|PSC|) dominates.
        assert!(
            pos[0] > neg[0].abs(),
            "expected χ+ dominant: χ+={} χ−={}",
            pos[0],
            neg[0]
        );
    }

    /// Regression: χ+ must survive past the phase-wrap threshold.
    ///
    /// A compartment accumulates `(2/3)·χ·γ·B0·TE` radians, so with `atan2`'s
    /// `(−π, π]` fold and no unwrapping the alternating fit collapses to a
    /// degenerate fixed point above `|χ_total| = π / ((2/3)·γ·B0·TE_max)` and
    /// returns χ+ = 0 exactly. At the phantom's 7 T and 28 ms last echo that
    /// threshold is 0.090 ppm — inside the globus pallidus, which consequently
    /// reconstructed as a hole. Sweep across it and require a response that is
    /// non-zero and increasing, not a dropout.
    #[test]
    fn chi_pos_survives_the_phase_wrap_threshold() {
        let te = [0.004, 0.012, 0.020, 0.028]; // 7 T phantom sampling
        let b0 = 7.0;
        let params = DecomposeParams {
            b0,
            n_inner: 10,
            chi_bound: 0.5,
            max_lm_iter: 30,
        };
        let threshold = PI / ((2.0 / 3.0) * GAMMA * b0 * te[te.len() - 1]);
        assert!(
            (0.06..0.12).contains(&threshold),
            "test assumes a threshold inside the swept range, got {threshold}"
        );

        // One voxel forward-simulated from the model, for a range of true χ+ that
        // straddles the threshold.
        let (cp, cm, c0, chim, r0) = (0.35, 0.25, 0.40, -0.03, 25.0);
        let mut prev = 0.0_f64;
        for k in 4..=13 {
            let chip = 0.02 * k as f64;
            let mag: Vec<f64> = te
                .iter()
                .map(|&t| {
                    let z = signal_model(t, cp, cm, c0, chip, chim, r0, b0);
                    (z.0 * z.0 + z.1 * z.1).sqrt()
                })
                .collect();
            let (pos, _neg, _tot) =
                decompose(&[chip + chim], &mag, &te, &[1u8], &params, |_, _| {});
            assert!(
                pos[0] > 0.0,
                "χ+ collapsed to zero at χ+={chip} (χ_total={}, threshold {threshold:.4})",
                chip + chim
            );
            assert!(
                pos[0] > prev,
                "χ+ must increase with the true source: {} -> {} at χ+={chip}",
                prev,
                pos[0]
            );
            prev = pos[0];
        }
    }

    /// Regression: no scattered dropouts across the deep-grey χ+ range.
    ///
    /// Unwrapping alone was not enough. The χ search ran to `chi_bound` (0.5 ppm
    /// by default), past the 0.31 ppm that 8 ms echo spacing can represent
    /// without the per-echo phase step exceeding π. In that aliased region the
    /// stage-3 landscape grows spurious minima, and voxels at particular
    /// (χ, R2*) combinations fell into them — leaving a recovered pallidum that
    /// was bright but speckled with black. Feed single-compartment data built the
    /// way the phantom builds it (magnitude decaying at R2* = R2 + Dr·|χ|, phase
    /// from χ_total) and require every voxel across the band to recover.
    #[test]
    fn chi_pos_has_no_dropouts_across_deep_grey() {
        let te = [0.004, 0.012, 0.020, 0.028];
        let b0 = 7.0;
        let params = DecomposeParams {
            b0,
            n_inner: 10,
            chi_bound: 0.5,
            max_lm_iter: 30,
        };
        let (dr_pos, dr_neg, r2) = (320.0_f64, 397.0_f64, 30.0_f64);
        let chim = -0.015_f64;
        for k in 3..=30 {
            let chip = 0.01 * k as f64;
            let r2s = r2 + dr_pos * chip + dr_neg * chim.abs();
            let mag: Vec<f64> = te.iter().map(|&t| (-r2s * t).exp()).collect();
            let (pos, _neg, _tot) =
                decompose(&[chip + chim], &mag, &te, &[1u8], &params, |_, _| {});
            assert!(
                pos[0] > 0.3 * chip,
                "dropout at χ+={chip:.2}: recovered {:.5} (gain {:.3})",
                pos[0],
                pos[0] / chip
            );
        }
    }

    /// Multi-voxel run exercising the masked-out, all-zero, and progress paths.
    #[test]
    fn decompose_multivoxel_mask_and_zero_paths() {
        let te = [0.004, 0.012, 0.020, 0.028];
        let b0 = 7.0;
        let ne = te.len();
        let params = DecomposeParams { b0, n_inner: 4, chi_bound: 0.5, max_lm_iter: 20 };

        // A real signal for the fitted voxels.
        let (cp, cm, c0, chip, chim, r0) = (0.4, 0.2, 0.4, 0.06, -0.05, 22.0);
        let sig: Vec<(f64, f64)> = te
            .iter()
            .map(|&t| signal_model(t, cp, cm, c0, chip, chim, r0, b0))
            .collect();
        let smag: Vec<f64> = sig.iter().map(|z| (z.0 * z.0 + z.1 * z.1).sqrt()).collect();

        // 4 voxels: [0]=masked-out, [1]=all-zero magnitude, [2],[3]=real signal.
        let n = 4;
        let mut mag = vec![0.0_f64; n * ne];
        for e in 0..ne {
            mag[2 * ne + e] = smag[e];
            mag[3 * ne + e] = smag[e];
        }
        let chi_total = vec![chip + chim; n];
        let mask = vec![0u8, 1, 1, 1];

        let mut last = (0usize, 0usize);
        let (pos, neg, tot) =
            decompose(&chi_total, &mag, &te, &mask, &params, |i, t| last = (i, t));

        // Masked-out and all-zero voxels stay at zero.
        assert_eq!((pos[0], neg[0], tot[0]), (0.0, 0.0, 0.0), "masked voxel");
        assert_eq!((pos[1], neg[1], tot[1]), (0.0, 0.0, 0.0), "all-zero voxel");
        // Fitted voxels obey the sign convention + invariant.
        for i in [2usize, 3] {
            assert!(pos[i] >= 0.0 && neg[i] <= 0.0, "signs at voxel {i}");
            assert!((tot[i] - (pos[i] + neg[i])).abs() < 1e-12, "invariant at voxel {i}");
        }
        // Progress reported completion over the 3 masked voxels.
        assert_eq!(last, (3, 3), "final progress");
    }

    // ── decompose_complex ──

    /// qsm-forward's DECOMPOSE pool split (χc = 0.2 ppm, pool fraction ≤ 0.45):
    /// returns `(C+, C−, C₀, χc+, χc−)` for voxel sources χ+ ≥ 0, χ− ≤ 0.
    fn pools(chi_pos: f64, chi_neg: f64) -> (f64, f64, f64, f64, f64) {
        let (chi_c, f_max) = (0.2, 0.45);
        let cp = (chi_pos / chi_c).min(f_max);
        let cm = (chi_neg.abs() / chi_c).min(f_max);
        let xp = if cp > 0.0 { chi_pos / cp } else { 0.0 };
        let xm = if cm > 0.0 { chi_neg / cm } else { 0.0 };
        (cp, cm, 1.0 - cp - cm, xp, xm)
    }

    /// One voxel's (magnitude, wrapped phase) in the input convention (phase =
    /// +γ·ΔB·TE), forward-simulated from the complex model at `x`.
    fn simulate(x: &[f64; 8], te: &[f64], b0: f64, scale: f64) -> (Vec<f64>, Vec<f64>) {
        let a = (2.0 * PI * GAMMA * b0) / (9.0 * 3.0_f64.sqrt());
        let w = (2.0 / 3.0) * GAMMA * b0;
        let model = ComplexModel { te, a, w };
        let (mut mag, mut ph) = (Vec::new(), Vec::new());
        for &t in te {
            let z = model.signal(x, t);
            mag.push(scale * (z.0 * z.0 + z.1 * z.1).sqrt());
            ph.push(-z.1.atan2(z.0)); // conjugate into the input convention
        }
        (mag, ph)
    }

    const TE_3T: [f64; 6] = [0.003, 0.007, 0.011, 0.015, 0.019, 0.023];

    #[test]
    fn complex_jacobian_matches_finite_differences() {
        let b0 = 3.0;
        let a = (2.0 * PI * GAMMA * b0) / (9.0 * 3.0_f64.sqrt());
        let w = (2.0 / 3.0) * GAMMA * b0;
        let model = ComplexModel { te: &TE_3T, a, w };
        let x = [0.3, 0.2, 0.5, 17.0, 0.15, -0.08, 0.4, 150.0];
        let y = vec![(0.1, -0.2); TE_3T.len()];
        let m = 2 * TE_3T.len();
        let mut jac = vec![0.0; m * 8];
        model.jac(&x, &mut jac);
        let (mut r0, mut r1) = (vec![0.0; m], vec![0.0; m]);
        model.resid(&x, &y, &mut r0);
        for j in 0..8 {
            let h = 1e-7 * x[j].abs().max(1e-3);
            let mut xp = x;
            xp[j] += h;
            model.resid(&xp, &y, &mut r1);
            for k in 0..m {
                let fd = (r1[k] - r0[k]) / h;
                let an = jac[j * m + k];
                assert!(
                    (fd - an).abs() <= 1e-4 * an.abs().max(1.0),
                    "∂r[{k}]/∂x[{j}]: analytic {an}, finite difference {fd}"
                );
            }
        }
    }

    /// Planted sources come back from the measured signal, through an arbitrary
    /// phase offset, a background frequency that wraps the phase between echoes,
    /// and an arbitrary magnitude scale.
    #[test]
    fn decompose_complex_recovers_planted_sources() {
        let b0 = 3.0;
        let r2 = 15.0;
        let cases = [(0.02, -0.01), (0.15, -0.03), (0.05, -0.09), (0.19, -0.08), (0.1, -0.05)];
        let (phi0, dw) = (2.6, 2.0 * PI * 70.0);
        let ne = TE_3T.len();
        let n = cases.len();
        let (mut mag, mut ph) = (Vec::new(), Vec::new());
        for &(chp, chm) in &cases {
            let (cp, cm, c0, xp, xm) = pools(chp, chm);
            let (m, p) = simulate(&[cp, cm, c0, r2, xp, xm, phi0, dw], &TE_3T, b0, 830.0);
            mag.extend(m);
            ph.extend(p);
        }
        let params = DecomposeComplexParams { b0, ..Default::default() };
        let (pos, neg, tot) =
            decompose_complex(&mag, &ph, &TE_3T, &vec![1u8; n], &params, |_, _| {});
        assert_eq!(pos.len(), n);
        assert_eq!(mag.len(), n * ne);
        for (k, &(chp, chm)) in cases.iter().enumerate() {
            assert!(
                (pos[k] - chp).abs() < 2e-3 && (neg[k] - chm).abs() < 2e-3,
                "voxel {k}: planted (χ+, χ−) = ({chp}, {chm}), recovered ({}, {})",
                pos[k],
                neg[k]
            );
            assert!((tot[k] - (pos[k] + neg[k])).abs() < 1e-12);
        }
    }

    /// Masked-out, zero-signal and non-finite-phase voxels stay zero; progress
    /// reports the masked count.
    #[test]
    fn decompose_complex_mask_and_degenerate_paths() {
        let b0 = 3.0;
        let ne = TE_3T.len();
        let (cp, cm, c0, xp, xm) = pools(0.1, -0.05);
        let (m, p) = simulate(&[cp, cm, c0, 15.0, xp, xm, 0.0, 0.0], &TE_3T, b0, 1.0);
        // [0] masked out, [1] zero magnitude, [2] NaN phase, [3] real signal.
        let mut mag = vec![0.0; 4 * ne];
        let mut ph = vec![0.0; 4 * ne];
        for v in [0, 2, 3] {
            mag[v * ne..(v + 1) * ne].copy_from_slice(&m);
            ph[v * ne..(v + 1) * ne].copy_from_slice(&p);
        }
        ph[2 * ne + 1] = f64::NAN;
        let params = DecomposeComplexParams { b0, ..Default::default() };
        let mut last = (0, 0);
        let (pos, neg, _) =
            decompose_complex(&mag, &ph, &TE_3T, &[0, 1, 1, 1], &params, |i, t| last = (i, t));
        for v in 0..3 {
            assert_eq!((pos[v], neg[v]), (0.0, 0.0), "voxel {v}");
        }
        assert!(pos[3] > 0.0 && neg[3] < 0.0);
        assert_eq!(last, (3, 3));
    }
}

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
//! **Input: one QSM per echo.** The data the model is fitted to is built from the
//! multi-echo magnitude `M(t)` and a QSM reconstructed *separately at each echo*,
//! χ(t) (the paper's Eq. 4):
//!
//! ```text
//!   y(t) = M(t) · exp(−i·(2/3)·χ(t)·γ·B0·t)
//! ```
//!
//! The paper is explicit that the echo-dependent QSM is required: the
//! sub-voxel compartments make the apparent susceptibility drift with TE, and
//! that drift is the signature DECOMPOSE separates. A single echo-averaged χ
//! gives every echo the same apparent χ, a phase linear in TE, and so no
//! compartment information beyond what the magnitude carries. `chi_echoes` is
//! therefore one χ map per echo, in the same echo order as `magnitude`. The
//! reference implementation computes it with STI Suite per echo (Laplacian
//! unwrapping → V-SHARP → STAR-QSM with that echo's TE). `M(t)` is normalised by
//! its global maximum, as in the reference.
//!
//! The per-voxel fit is a 3-stage alternating bounded least-squares, repeated
//! `n_inner` times: (1) amplitudes `[C+,C−,C₀]` against the linear signal, then
//! (2) `R2*₀` and (3) `[χ+,χ−]` against `log(signal)` (complex log). The residual
//! is the complex difference packed as `[Re; −Im]`, matching the reference.
//!
//! Each source is reconstructed from the fitted parameters via a per-compartment
//! phase accumulation, `−Σ angle(model) / ((2/3)·γ·B0·ΣTE)`: the paramagnetic
//! sub-model (`pscModel`, the paper's PCS, Eq. 6) gives χ+ and the diamagnetic
//! sub-model (`dscModel`, the DCS, Eq. 7) gives |χ−|. χ+ is returned ≥ 0, χ− ≤ 0.
//!
//! **Paper vs. the reference implementation.** This port follows Tim Ho's open
//! MATLAB re-implementation, which differs from the paper's description in ways
//! this port inherits (they are recorded here, not changed):
//!
//! - *Amplitude constraints.* The paper constrains `C+ + C− + C₀ = 1` with
//!   `0 < C < 1`. The reference bounds each amplitude only below (`C ≥ 0`,
//!   unbounded above) and imposes no sum-to-one; so does this port.
//! - *Solvers.* The paper solves sub-problems 5.1 (amplitudes) and 5.2 (R2*₀) with
//!   `lsqlin` and 5.3 (χ+, χ−) with trust-region `lsqnonlin`. The reference uses
//!   `lsqcurvefit` for all three; this port uses a bounded Levenberg–Marquardt
//!   for all three.
//! - *Normalisation and initial values.* The paper's main text states neither
//!   how `M(t)` is normalised nor the initial values. The reference normalises
//!   `M(t)` by its global maximum and starts from `C = [0.3, 0.3, 0.4]`,
//!   `R2*₀ = 25 Hz`, `χ = [0.05, −0.05]` ppm with `|χ| ≤ 0.5` ppm; this port does
//!   the same.
//! - *Output files.* The reference demo writes DSC to the file `XPSC` and PSC to
//!   `XDSC`. Per the paper the paramagnetic component is the PCS (Eq. 6), so this
//!   port returns χ+ = |PSC| and χ− = −|DSC|.
//!
//! The port's own additions, for stability rather than as changes to the model,
//! are the phase unwrapping along echoes and the χ search cap, below.
//!
//! **Phase unwrapping.** Every phase the fit and the reconstruction see arrives
//! through `atan2`, on `(−π, π]`. A compartment accumulates `(2/3)·χ·γ·B0·TE`
//! radians, so above `|χ| = π / ((2/3)·γ·B0·TE_max)` the late echoes fold
//! back onto that branch — 0.09 ppm at 7 T with a 28 ms last echo, which is
//! ordinary globus pallidus. Left folded, the alternation converges to a
//! degenerate fixed point (χ− pinned at `chi_bound`, R2*₀ and the amplitudes at
//! zero) and returns χ+ = 0 exactly, so the most paramagnetic structure in the
//! volume reads as a hole. Both the stage-2/3 residuals and `recon_phase`
//! therefore walk the echoes in order and keep each phase on the branch nearest
//! its predecessor ([`unwrap_near`]); the data phase `−(2/3)·χ(t)·γ·B0·t` is known
//! in closed form from the per-echo QSM and is used as it is rather than
//! round-tripped through `atan2`.
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
//! Note that `−Σ angle / den` is a *compressive* estimator of χ: its gain rises
//! with χ itself, so it flattens contrast between weak sources. That is inherent
//! to the reference's reconstruction, not to this port — the fitted compartment
//! amplitudes set the gain. Fed an echo-independent χ instead of χ(t), the
//! amplitudes are only weakly identifiable (the data then carries no sub-voxel
//! compartment signature in its phase) and the gain is small: ≈0.03 at
//! white-matter χ+ on the QSM-CI phantom.
//!
//! Reference:
//! Chen, J., Gong, N.-J., Chaim, K.T., Otaduy, M.C.G., Liu, C. (2021). "Decompose
//! quantitative susceptibility mapping (QSM) to sub-voxel diamagnetic and
//! paramagnetic components based on gradient-echo MRI data." NeuroImage
//! 242:118477. https://doi.org/10.1016/j.neuroimage.2021.118477
//! Reference implementation: Tim Ho's open MATLAB DECOMPOSE-QSM,
//! <https://github.com/timwahoo/DECOMPOSE-QSM> (`DECOMPOSE_analysis.m`).

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

/// DECOMPOSE source separation from a per-echo QSM and the multi-echo magnitude.
///
/// # Arguments
/// * `chi_echoes` — QSM of each echo, χ(t), in **ppm**, flattened as
///   `(n_voxels, n_echoes)` in row-major order (echo fastest per voxel), the same
///   layout and echo order as `magnitude`. This is the TE-dependent QSM the
///   method requires (see the module docs); it builds the fitted data as
///   `y(t) = M(t)·exp(−i·(2/3)·χ(t)·γ·B0·t)`. Passing one map repeated for every
///   echo runs the algorithm, but leaves it without the echo dependence it
///   separates on.
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
    chi_echoes: &[f64],
    magnitude: &[f64],
    echo_times: &[f64],
    mask: &[u8],
    params: &DecomposeParams,
    mut progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = mask.len();
    let ne = echo_times.len();
    assert_eq!(
        magnitude.len(),
        n * ne,
        "magnitude must be n_voxels * n_echoes"
    );
    assert_eq!(
        chi_echoes.len(),
        n * ne,
        "chi_echoes must be n_voxels * n_echoes (one QSM per echo)"
    );
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
            // The fitted data, Eq. 4: y[e] = M(TE_e) · exp(−i·(2/3)·χ(TE_e)·γ·B0·TE_e),
            // from this echo's own QSM.
            let mut all_zero = true;
            for e in 0..ne {
                let m = magnitude[i * ne + e] / gmax;
                if m > 0.0 {
                    all_zero = false;
                }
                let ph = -(2.0 / 3.0) * chi_echoes[i * ne + e] * GAMMA * b0 * te[e];
                ws.y[e] = (m * ph.cos(), m * ph.sin());
                // log(y) for stages 2 & 3. `ph` is known in closed form, so keep
                // it as it is rather than recovering a ±π-wrapped copy of it
                // through `atan2` — see the note above `unwrap_near`.
                ws.logy[e] = (0.5 * (m * m).max(1e-300).ln(), ph);
            }
            if all_zero {
                return (i, 0.0, 0.0);
            }

            let (cp, cm, c0, r0, chip, chim) =
                fit_voxel(te, &ws.y, &ws.logy, b0, params, &mut ws.lm);

            // Reconstruct χ+ (from pscModel) and χ− (from dscModel), both as the
            // per-compartment phase accumulation −Σangle/den.
            let dsc = recon_phase(te, b0, den, |t| {
                dsc_model(t, cp, cm, c0, chip, chim, r0, b0)
            });
            let psc = recon_phase(te, b0, den, |t| {
                psc_model(t, cp, cm, c0, chip, chim, r0, b0)
            });
            // The paper's PCS (Eq. 6, paramagnetic sub-model) is χ+ and its DCS
            // (Eq. 7, diamagnetic sub-model) is χ−. The reference demo's output
            // files swap the two names; that is not followed. See module docs.
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
    let dte = te.windows(2).map(|w| w[1] - w[0]).fold(te[0], f64::max);
    let ub = params.chi_bound.min(PI / ((2.0 / 3.0) * GAMMA * b0 * dte));
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

/// Largest parameter count any DECOMPOSE stage fits (stage 1: `[C+, C−, C₀]`).
/// Keeps the LM state on the stack.
const LM_MAX_N: usize = 3;

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
        // Forward-difference Jacobian J (m×n), column-major in `s.jac`.
        for j in 0..n {
            let h = 1e-6 * x[j].abs().max(1e-3);
            let mut xp = x;
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
        // Normal equations A = JᵀJ (n×n), g = Jᵀr (n).
        let mut a = [0.0_f64; LM_MAX_N * LM_MAX_N];
        let mut g = [0.0_f64; LM_MAX_N];
        for jc in 0..n {
            for jr in 0..n {
                let mut acc = 0.0;
                for k in 0..m {
                    acc += s.jac[jr * m + k] * s.jac[jc * m + k];
                }
                a[jr * n + jc] = acc;
            }
            let mut acc = 0.0;
            for k in 0..m {
                acc += s.jac[jc * m + k] * s.r[k];
            }
            g[jc] = acc;
        }

        // Inner loop: inflate lambda until a step decreases the cost.
        let mut improved = false;
        for _ in 0..30 {
            let mut al = a;
            for d in 0..n {
                al[d * n + d] += lambda * a[d * n + d].max(1e-12);
            }
            let Some(dx) = solve_small(&al, &g, n) else {
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
                    return x;
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
    x
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
        let x = solve_small(&a, &b, 3).unwrap();
        for i in 0..3 {
            assert!((x[i] - x_true[i]).abs() < 1e-9, "x{i} = {}", x[i]);
        }
    }

    /// One voxel planted from DECOMPOSE's own three-compartment forward model
    /// ([`signal_model`]): its multi-echo magnitude and its per-echo QSM χ(t), the
    /// apparent susceptibility a QSM reconstructed at echo `t` sees,
    /// `χ(t) = −angle(S(t)) / ((2/3)·γ·B0·t)` (phase unwrapped along echoes).
    #[allow(clippy::too_many_arguments)]
    fn planted_voxel(
        te: &[f64],
        cp: f64,
        cm: f64,
        c0: f64,
        chip: f64,
        chim: f64,
        r0: f64,
        b0: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut mag = Vec::with_capacity(te.len());
        let mut chi = Vec::with_capacity(te.len());
        let mut prev = 0.0;
        for (e, &t) in te.iter().enumerate() {
            let z = signal_model(t, cp, cm, c0, chip, chim, r0, b0);
            mag.push((z.0 * z.0 + z.1 * z.1).sqrt());
            let raw = z.1.atan2(z.0);
            let ang = if e == 0 { raw } else { unwrap_near(raw, prev) };
            prev = ang;
            chi.push(-ang / ((2.0 / 3.0) * GAMMA * b0 * t));
        }
        (mag, chi)
    }

    /// The same map for every echo — what an echo-averaged QSM provides.
    fn repeated(chi: &[f64], ne: usize) -> Vec<f64> {
        chi.iter().flat_map(|&c| std::iter::repeat(c).take(ne)).collect()
    }

    /// Eq. 4 wiring: given a voxel's per-echo QSM χ(t) and magnitude, the data
    /// `decompose` fits is `M(t)·exp(−i·(2/3)·χ(t)·γ·B0·t)`. For a voxel planted
    /// from the three-compartment model that is the planted complex signal itself
    /// (scaled by the global-max normalisation), so the output must equal a fit of
    /// that signal reconstructed with χ+ = |PSC|, χ− = −|DSC|.
    #[test]
    fn per_echo_qsm_builds_the_eq4_data() {
        let b0 = 3.0;
        let te: Vec<f64> = (0..6).map(|k| 0.003 + 0.004 * k as f64).collect();
        let params = DecomposeParams { b0, ..DecomposeParams::default() };
        let (cp, cm, c0, chip, chim, r0) = (0.3, 0.3, 0.4, 0.12, -0.06, 20.0);
        let (mag, chi) = planted_voxel(&te, cp, cm, c0, chip, chim, r0, b0);

        // χ(t) is genuinely echo-dependent for a mixed voxel.
        let spread = chi.iter().cloned().fold(f64::MIN, f64::max)
            - chi.iter().cloned().fold(f64::MAX, f64::min);
        assert!(spread > 1e-3, "planted χ(t) should drift with TE, spread {spread}");

        let (pos, neg, tot) = decompose(&chi, &mag, &te, &[1u8], &params, |_, _| {});

        // The planted signal, normalised as decompose normalises the magnitude.
        let gmax = mag.iter().cloned().fold(0.0, f64::max);
        let mut y = Vec::new();
        let mut logy = Vec::new();
        let mut prev = 0.0;
        for (e, &t) in te.iter().enumerate() {
            let z = signal_model(t, cp, cm, c0, chip, chim, r0, b0);
            let z = (z.0 / gmax, z.1 / gmax);
            let l = clog(z);
            let ang = if e == 0 { l.1 } else { unwrap_near(l.1, prev) };
            prev = ang;
            y.push(z);
            logy.push((l.0, ang));
        }
        let mut lm = LmScratch::new(2 * te.len());
        let (fcp, fcm, fc0, fr0, fchip, fchim) = fit_voxel(&te, &y, &logy, b0, &params, &mut lm);
        let den = (2.0 / 3.0) * GAMMA * b0 * te.iter().sum::<f64>();
        let want_pos =
            recon_phase(&te, b0, den, |t| psc_model(t, fcp, fcm, fc0, fchip, fchim, fr0, b0))
                .abs();
        let want_neg =
            -recon_phase(&te, b0, den, |t| dsc_model(t, fcp, fcm, fc0, fchip, fchim, fr0, b0))
                .abs();
        // χ(t) round-trips the phase through floating point, and the alternating
        // fit amplifies last-bit differences, so agreement is to a tolerance —
        // one the echo-independent input misses by far.
        let tol = 5e-4;
        assert!((pos[0] - want_pos).abs() < tol, "χ+ {} vs {want_pos}", pos[0]);
        assert!((neg[0] - want_neg).abs() < tol, "χ− {} vs {want_neg}", neg[0]);
        assert!((tot[0] - (pos[0] + neg[0])).abs() < 1e-12, "χ_total = χ+ + χ−");

        let flat = repeated(&[chip + chim], te.len());
        let (fpos, fneg, _) = decompose(&flat, &mag, &te, &[1u8], &params, |_, _| {});
        assert!(
            (fpos[0] - want_pos).abs() + (fneg[0] - want_neg).abs() > 10.0 * tol,
            "an echo-independent χ should not reproduce the per-echo fit: \
             χ+ {} χ− {} vs {want_pos} {want_neg}",
            fpos[0],
            fneg[0]
        );
    }

    /// Echo `e` of the input is the QSM of echo `e`: reversing the echo order of
    /// `chi_echoes` against the magnitude must change the result.
    #[test]
    fn per_echo_qsm_is_read_in_echo_order() {
        let b0 = 3.0;
        let te: Vec<f64> = (0..6).map(|k| 0.003 + 0.004 * k as f64).collect();
        let params = DecomposeParams { b0, n_inner: 4, ..DecomposeParams::default() };
        let (mag, chi) = planted_voxel(&te, 0.3, 0.3, 0.4, 0.12, -0.06, 20.0, b0);
        let rev: Vec<f64> = chi.iter().rev().cloned().collect();
        let (p1, n1, _) = decompose(&chi, &mag, &te, &[1u8], &params, |_, _| {});
        let (p2, n2, _) = decompose(&rev, &mag, &te, &[1u8], &params, |_, _| {});
        assert!(
            (p1[0] - p2[0]).abs() + (n1[0] - n2[0]).abs() > 1e-4,
            "echo order of chi_echoes was ignored"
        );
    }

    /// Sign/plumbing check on a synthetic voxel. Verifies the output sign
    /// convention and the mapping χ+ = |PSC| (paramagnetic sub-model),
    /// χ− = −|DSC| (diamagnetic sub-model). A voxel with a large paramagnetic
    /// amplitude/χ therefore yields a dominant χ+.
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
        let (mag, chi) = planted_voxel(&te, 0.5, 0.1, 0.4, 0.10, -0.02, 20.0, b0);
        let (pos, neg, tot) = decompose(&chi, &mag, &te, &[1u8], &params, |_, _| {});
        assert!(pos[0] >= 0.0, "χ+ must be ≥ 0");
        assert!(neg[0] <= 0.0, "χ− must be ≤ 0");
        assert!((tot[0] - (pos[0] + neg[0])).abs() < 1e-12, "χ_total = χ+ + χ−");
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
    /// degenerate fixed point above `|χ| = π / ((2/3)·γ·B0·TE_max)` and returns
    /// χ+ = 0 exactly. At 7 T and a 28 ms last echo that threshold is 0.090 ppm —
    /// inside the globus pallidus, which consequently reconstructed as a hole.
    /// Sweep across it and require a response that is non-zero and increasing,
    /// not a dropout.
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

        // Magnitude forward-simulated from the model and one χ for every echo
        // (the echo-independent input that exposed the fold), for a range of true
        // χ+ that straddles the threshold.
        let (cp, cm, c0, chim, r0) = (0.35, 0.25, 0.40, -0.03, 25.0);
        let mut prev = 0.0_f64;
        for k in 4..=13 {
            let chip = 0.02 * k as f64;
            let (mag, _) = planted_voxel(&te, cp, cm, c0, chip, chim, r0, b0);
            let chi = repeated(&[chip + chim], te.len());
            let (pos, _neg, _tot) = decompose(&chi, &mag, &te, &[1u8], &params, |_, _| {});
            assert!(
                pos[0] > 0.0,
                "χ+ collapsed to zero at χ+={chip} (threshold {threshold:.4})"
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
    /// The χ search used to run to `chi_bound` (0.5 ppm by default), past the
    /// 0.31 ppm that 8 ms echo spacing can represent without the per-echo phase
    /// step exceeding π. In that aliased region the stage-3 landscape grows
    /// spurious minima, and voxels at particular (χ, R2*) combinations fell into
    /// them. This feeds the echo-independent data that exposed it (magnitude
    /// decaying at R2* = R2 + Dr·|χ|, one χ for every echo) and requires every
    /// voxel across the band to recover.
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
            let chi = repeated(&[chip + chim], te.len());
            let (pos, _neg, _tot) = decompose(&chi, &mag, &te, &[1u8], &params, |_, _| {});
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
        let (smag, schi) = planted_voxel(&te, 0.4, 0.2, 0.4, 0.06, -0.05, 22.0, b0);

        // 4 voxels: [0]=masked-out, [1]=all-zero magnitude, [2],[3]=real signal.
        let n = 4;
        let mut mag = vec![0.0_f64; n * ne];
        let mut chi = vec![0.0_f64; n * ne];
        for v in [2usize, 3] {
            mag[v * ne..(v + 1) * ne].copy_from_slice(&smag);
            chi[v * ne..(v + 1) * ne].copy_from_slice(&schi);
        }
        let mask = vec![0u8, 1, 1, 1];

        let mut last = (0usize, 0usize);
        let (pos, neg, tot) = decompose(&chi, &mag, &te, &mask, &params, |i, t| last = (i, t));

        assert_eq!((pos[0], neg[0], tot[0]), (0.0, 0.0, 0.0), "masked voxel");
        assert_eq!((pos[1], neg[1], tot[1]), (0.0, 0.0, 0.0), "all-zero voxel");
        for i in [2usize, 3] {
            assert!(pos[i] >= 0.0 && neg[i] <= 0.0, "signs at voxel {i}");
            assert!((tot[i] - (pos[i] + neg[i])).abs() < 1e-12, "invariant at voxel {i}");
        }
        assert_eq!(last, (3, 3), "final progress");
    }
}

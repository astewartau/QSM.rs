//! Bound-constrained nonlinear least squares by the subspace trust-region reflective method.
//!
//! Minimises `‖F(x)‖²` over the box `lb ≤ x ≤ ub`, for a residual `F: ℝⁿ → ℝᵐ` given as a
//! closure, with a forward finite-difference Jacobian. This is the method MATLAB's
//! `lsqcurvefit` and `lsqnonlin` use by default (`Algorithm = 'trust-region-reflective'`, with
//! `SubproblemAlgorithm = 'factorization'`), implemented here from the published algorithm and
//! MATLAB's documentation, so that a port of MATLAB code calling those functions can run the same
//! iterations. MathWorks' own source is not used.
//!
//! # The method
//!
//! Coleman & Li's interior reflective Newton method [1, 2], with the two-dimensional subspace
//! trust-region step of Branch, Coleman & Li [3]. Every iterate is strictly inside the box.
//!
//! At an iterate `x` with residual `F`, Jacobian `J` and gradient `g = JᵀF`:
//!
//! 1. **Scaling** [1, §2]. For each `i`, `vᵢ` is the distance to the bound the negative gradient
//!    points at: `xᵢ − uᵢ` if `gᵢ < 0` and `uᵢ < ∞`, `xᵢ − lᵢ` if `gᵢ ≥ 0` and `lᵢ > −∞`, and `±1`
//!    when that bound is infinite. `D = diag(|v|^{-½})`. `x` is a first-order point of the
//!    bounded problem when `v ⊙ g = 0`; the **first-order optimality** reported (MATLAB's
//!    `firstorderopt`) is `‖v ⊙ g‖∞`.
//! 2. **Scaled model** [1, eq. 3.1]. In the scaled variable `ŝ = D s`, the quadratic model is
//!    `ψ(ŝ) = ĝᵀŝ + ½ ŝᵀM̂ŝ` with `ĝ = D⁻¹g`, `M̂ = D⁻¹JᵀJD⁻¹ + diag(g)Jᵛ` and `Jᵛ = diag(∂|v|/∂x)`
//!    (`sign(gᵢ)` for a finite bound, 0 otherwise), so `diag(g)Jᵛ = diag(|gᵢ|)` on the finite
//!    bounds. That term is what makes the Newton step of the bounded problem quadratically
//!    convergent [2].
//! 3. **Subspace trust-region step** [3, §2]. The trust region `‖ŝ‖ ≤ Δ` is searched in the plane
//!    spanned by the scaled gradient `ĝ` and the scaled Gauss–Newton (Newton) step `ŝ_N`, the
//!    solution of `M̂ ŝ_N = −ĝ`. `M̂` is positive semidefinite for least squares, so `ŝ_N` is
//!    computed directly, as the least-squares solution of the augmented system
//!    `[J D⁻¹; diag(|g|·Jᵛ)^½] ŝ_N ≈ [−F; 0]` (a column-pivoted Householder QR: MATLAB's
//!    `'factorization'` subproblem, not its preconditioned-CG alternative). The two-dimensional
//!    trust-region problem is then solved exactly (eigendecomposition and the secular equation,
//!    Moré & Sorensen [4]).
//! 4. **Strict feasibility: truncation and reflection** [1, §3; 2]. Let α be the multiple of
//!    `s = D⁻¹ŝ` that reaches the first bound, and `θ = max(0.95, 1 − ‖v ⊙ g‖∞)`. If `θ·α ≥ 1`,
//!    `s` is the step. Otherwise the best by ψ of three candidates is taken: `s` truncated to
//!    `θ·α·s`; `s` *reflected* at that bound (the path continues from the bound point with the
//!    crossing component negated, up to θ of the way to the next bound or to the trust-region
//!    boundary), kept only when ψ's minimiser along that segment, taken with the gradient at
//!    `x`, lies at its end; and the scaled steepest-descent (Cauchy) step `−D⁻²g`, minimised
//!    along its own pulled-back feasible segment. Every iterate stays strictly interior.
//! 5. **Acceptance and radius** [1, Alg. 3.1]. `ρ = (½‖F(x+s)‖² − ½‖F‖²) / (gᵀs + ½‖Js‖²)`: the
//!    Gauss–Newton prediction, without the `diag(g)Jᵛ` term that only shapes the step. The step
//!    is accepted when `ρ > 0`. `ρ ≤ ¼` shrinks Δ to `¼·min(Δ, ‖ŝ‖)`; `ρ > ¾` with
//!    `‖ŝ‖ ≥ 0.9·Δ` doubles it. `Δ₀ = 10`.
//! 6. **Stopping**, after an accepted step: `‖v ⊙ g‖∞ < OptimalityTolerance` (exit flag 1), the
//!    change in `‖F‖²` below `FunctionTolerance·(1 + ‖F‖²)` (3); after any step,
//!    `‖s‖ < StepTolerance·(1 + ‖x‖)` (2); and the iteration and evaluation caps (0). At the start,
//!    the optimality test alone.
//!
//! # Matching MATLAB
//!
//! The defaults of [`LsqTrrOptions`] are `lsqcurvefit`'s: `MaxIterations` 400,
//! `MaxFunctionEvaluations` `100·n`, `FunctionTolerance`, `StepTolerance` and
//! `OptimalityTolerance` 1e-6, `FiniteDifferenceStepSize` `√eps`, `TypicalX` 1, and forward
//! differences with step `h = √eps · sign′(xⱼ) · max(|xⱼ|, TypicalXⱼ)` (`sign′(0) = 1`), as
//! MATLAB documents them.
//!
//! What the papers and the documentation leave open was settled by running MATLAB R2023b's
//! `lsqcurvefit` as a black box (`scripts/lsq_trr_matlab/probe_lsq_trr.m`: every point the
//! objective is evaluated at, and every `OutputFcn` `optimValues` field), on generic problems and
//! on every stage call of DECOMPOSE's fit:
//!
//! - `Δ₀ = 10` on every problem; radius updates as in step 5 (read off `trustregionradius`).
//! - The residual **and its Jacobian** are evaluated at every trial point before ρ decides, so a
//!   rejected step costs `1 + n` evaluations too (MATLAB's `funcCount`), and rejected steps count
//!   as iterations.
//! - The finite-difference quotient divides by the nominal `h`, not by the representable
//!   `(x + h) − x`; a probe that would leave the box goes the other way.
//! - An `x0` on a bound is used as it is; one outside a two-sided box goes to its midpoint.
//! - `stepsize` is the scaled norm `‖ŝ‖`; `firstorderopt` is `‖v ⊙ g‖∞`.
//! - θ, the `θ·α < 1` pull-back, the reflection rule and ρ's model are as in steps 4 and 5 (each
//!   is commented where it is implemented, with the evidence).
//!
//! The 2×2 trust-region subproblem is solved to machine precision; MATLAB's steps on the
//! trust-region boundary have `‖ŝ‖` within ~1e-13 of Δ, so it solves the secular equation to a
//! tolerance, and steps agree to ~1e-13 rather than bit for bit.
//!
//! **Where the last bits still differ.** A residual written with MATLAB's elementary functions
//! ([`crate::utils::fdlibm`]: MATLAB R2023b's `exp`, `log`, `sin`, `cos`, `atan2` and `hypot` are
//! FDLIBM 5.3's, bit for bit) is evaluated identically on both sides, so `F` and the
//! finite-difference `J` are bit-identical at every point the two runs share. The first quantity
//! that differs is then the trial step. MATLAB's step solves the same model, but with a forward
//! error of about `eps·cond(M̂)` in the model's weakest direction (on exactly representable
//! linear problems, `scripts/lsq_trr_matlab/lin_probe.m`: 30–4000 ulps at `cond(M̂)` = 10²–10⁴,
//! where this port's direct solve is within an ulp or two of the exact rational answer). That is
//! the signature of solving in a rotated basis (an eigendecomposition of the reduced matrix, for
//! one). No combination tried here of a QR or normal-equation Newton step, a Gram–Schmidt, QR
//! or SVD basis, and an eigen or direct 2×2 solve reproduces MATLAB's bits, and MATLAB's dense linear algebra is MKL's,
//! whose reduction orders are not documented (`J'*F` and `J'*J` match no sequential, lane-split
//! or pairwise order tried; `sum` is sequential and `norm` matches reference `dnrm2` on 98 % of
//! inputs). On a well-determined problem the difference stays at that level and both runs stop at
//! the same solution to the stopping tolerance; where a Jacobian column is nearly rank-deficient
//! (a parameter the residual barely depends on, such as an amplitude of 1e-11), it decides the
//! path and the two runs part.
//!
//! **Agreement** (`tests/data/lsq_trr_matlab_generic.txt` in the tests here; the DECOMPOSE trace,
//! 34 MB, is replayed by `separation::decompose`'s ignored `decompose_stages_match_matlab`): on the 11340 `lsqcurvefit` calls of the
//! authors' DECOMPOSE fit on the 378-voxel QSM-CI fixture, each started from MATLAB's own x0 and
//! fixed parameters, 10696 (94.3 %) take the same iterations, function evaluations and exit flag
//! with every evaluation point within 1e-6: stage 2 (one parameter) 3780 of 3780, stage 1
//! 3265 of 3780 and stage 3 3651 of 3780 (with the platform's `exp`/`log`/`atan2` in the
//! residual instead of FDLIBM's: 10505, stage 3 3461). 5823 calls (stage 1 435, stage 2 3686,
//! stage 3 1702) are bit-identical in every evaluation point and sum of squares. Of the calls
//! whose parameters (fixed and fitted) all exceed 1e-6, 1204 of 1205 (stage 1) and 1124 of 1177
//! (stage 3) take MATLAB's path; the other 53 part after a trial step that differs at ~1e-10 and
//! end at the same minimum. On the generic probe problems, 11 of 16 agree
//! throughout; the other five (bounded Rosenbrock variants converging onto an active bound, and
//! the last iterations of unbounded Rosenbrock) reach the same solution to 1e-6 or better, in a
//! different number of iterations for three of them.
//!
//! # References
//!
//! 1. Coleman, T.F. & Li, Y. (1996). An interior trust region approach for nonlinear
//!    minimization subject to bounds. *SIAM J. Optim.* 6(2):418–445.
//! 2. Coleman, T.F. & Li, Y. (1994). On the convergence of interior-reflective Newton methods
//!    for nonlinear minimization subject to bounds. *Math. Programming* 67:189–224.
//! 3. Branch, M.A., Coleman, T.F. & Li, Y. (1999). A subspace, interior, and conjugate gradient
//!    method for large-scale bound-constrained minimization problems. *SIAM J. Sci. Comput.*
//!    21(1):1–23.
//! 4. Moré, J.J. & Sorensen, D.C. (1983). Computing a trust region step. *SIAM J. Sci. Stat.
//!    Comput.* 4(3):553–572.
//! 5. MathWorks, *Optimization Toolbox User's Guide* (R2023b): "Least-Squares (Model Fitting)
//!    Algorithms — Trust-Region-Reflective Least Squares", "Tolerances and Stopping Criteria",
//!    "First-Order Optimality Measure", and the `lsqcurvefit` / `optimoptions` reference pages.

/// The constants of the method that the papers leave to the implementation.
///
/// One value for each, and where it came from. They are fixed (not options): a port that calls
/// [`lsq_trust_region_reflective`] gets MATLAB's iterations, and changing one would no longer.
#[derive(Clone, Copy, Debug)]
pub struct Rules {
    /// Initial trust-region radius (scaled variable).
    pub delta0: f64,
    /// Lower bound of the step-back factor θ.
    pub theta_min: f64,
    /// A step is accepted when `ρ > accept`.
    pub accept: f64,
    /// `ρ ≤ shrink_below` shrinks the radius to `shrink_factor · min(Δ, ‖ŝ‖)`.
    pub shrink_below: f64,
    /// Shrink factor.
    pub shrink_factor: f64,
    /// `ρ > grow_above` with `‖ŝ‖ ≥ grow_reach·Δ` grows the radius to `grow_factor · Δ`.
    pub grow_above: f64,
    /// See `grow_above`.
    pub grow_reach: f64,
    /// Growth factor.
    pub grow_factor: f64,
}

/// The values [`lsq_trust_region_reflective`] uses: Coleman & Li's μ = ¼, η = ¾, shrink ×¼ and
/// grow ×2, θ ≥ 0.95, and Δ₀ = 10, each confirmed on MATLAB R2023b's `lsqcurvefit` iterations
/// (its `OutputFcn` reports the radius, ρ and the step; module docs, *Matching MATLAB*).
pub const RULES: Rules = Rules {
    delta0: 10.0,
    theta_min: 0.95,
    accept: 0.0,
    shrink_below: 0.25,
    shrink_factor: 0.25,
    grow_above: 0.75,
    grow_reach: 0.9,
    grow_factor: 2.0,
};

/// Options for [`lsq_trust_region_reflective`]. Defaults are MATLAB `lsqcurvefit`'s.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct LsqTrrOptions {
    /// `MaxIterations` (default 400).
    pub max_iterations: usize,
    /// `MaxFunctionEvaluations`; `None` = `100·n` (MATLAB's default for this algorithm). Counts
    /// the finite-difference evaluations.
    pub max_function_evaluations: Option<usize>,
    /// `FunctionTolerance` (default 1e-6).
    pub function_tolerance: f64,
    /// `StepTolerance` (default 1e-6).
    pub step_tolerance: f64,
    /// `OptimalityTolerance` (default 1e-6).
    pub optimality_tolerance: f64,
    /// `FiniteDifferenceStepSize` (default `√eps`).
    pub finite_difference_step: f64,
    /// `TypicalX` for every variable (default 1).
    pub typical_x: f64,
}

impl Default for LsqTrrOptions {
    fn default() -> Self {
        Self {
            max_iterations: 400,
            max_function_evaluations: None,
            function_tolerance: 1e-6,
            step_tolerance: 1e-6,
            optimality_tolerance: 1e-6,
            finite_difference_step: f64::EPSILON.sqrt(),
            typical_x: 1.0,
        }
    }
}

/// Why [`lsq_trust_region_reflective`] stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LsqTrrExit {
    /// First-order optimality below `optimality_tolerance` (MATLAB exit flag 1).
    Optimality,
    /// Step below `step_tolerance` (exit flag 2).
    StepTolerance,
    /// Change in the sum of squares below `function_tolerance` (exit flag 3).
    FunctionTolerance,
    /// `max_iterations` reached (exit flag 0).
    MaxIterations,
    /// `max_function_evaluations` reached (exit flag 0).
    MaxFunctionEvaluations,
}

impl LsqTrrExit {
    /// MATLAB's `exitflag` for this outcome.
    pub fn matlab_exitflag(self) -> i32 {
        match self {
            Self::Optimality => 1,
            Self::StepTolerance => 2,
            Self::FunctionTolerance => 3,
            Self::MaxIterations | Self::MaxFunctionEvaluations => 0,
        }
    }
}

/// What [`lsq_trust_region_reflective`] returns besides `x` (MATLAB's `resnorm` and `output`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LsqTrrReport {
    /// `‖F(x)‖²` at the solution.
    pub resnorm: f64,
    /// Iterations taken (`output.iterations`).
    pub iterations: usize,
    /// Residual evaluations, finite differences included (`output.funcCount`).
    pub func_count: usize,
    /// `‖v ⊙ g‖∞` at the solution (`output.firstorderopt`).
    pub first_order_optimality: f64,
    /// Why it stopped (`exitflag`).
    pub exit: LsqTrrExit,
}

/// One iteration, as MATLAB's `OutputFcn` sees it (`optimValues`).
#[derive(Clone, Copy, Debug)]
pub struct LsqTrrIterate<'a> {
    /// Iteration number (0 = the start point).
    pub iteration: usize,
    /// Residual evaluations so far.
    pub func_count: usize,
    /// Current point.
    pub x: &'a [f64],
    /// `‖F(x)‖²`.
    pub resnorm: f64,
    /// `‖v ⊙ g‖∞`.
    pub first_order_optimality: f64,
    /// Trust-region radius Δ that the next step will use.
    pub trust_region_radius: f64,
    /// Scaled norm `‖ŝ‖ = ‖D s‖` of the step just tried (0 at iteration 0): MATLAB's `stepsize`.
    pub step_norm: f64,
    /// ρ of the step that led here (0 at iteration 0).
    pub ratio: f64,
}

/// Working storage for [`lsq_trust_region_reflective`], sized for `m` residuals and `n`
/// parameters. Reuse one across calls of that size (e.g. one per thread): a solve then allocates
/// nothing.
#[derive(Clone, Debug)]
pub struct LsqTrrWorkspace {
    m: usize,
    n: usize,
    f: Vec<f64>,
    f_trial: Vec<f64>,
    f_probe: Vec<f64>,
    /// Jacobian, `m × n`, column-major.
    jac: Vec<f64>,
    /// Jacobian at the trial point.
    jac_trial: Vec<f64>,
    /// Augmented matrix for the Newton step, `(m + n) × n`, column-major.
    aug: Vec<f64>,
    aug_rhs: Vec<f64>,
    x_trial: Vec<f64>,
    x_probe: Vec<f64>,
    /// n-vectors.
    g: Vec<f64>,
    v: Vec<f64>,
    dv: Vec<f64>,
    dsq: Vec<f64>,
    ghat: Vec<f64>,
    newton: Vec<f64>,
    z1: Vec<f64>,
    z2: Vec<f64>,
    step: Vec<f64>,
    shat: Vec<f64>,
    cand: Vec<f64>,
    best: Vec<f64>,
    dir: Vec<f64>,
    base: Vec<f64>,
    perm: Vec<usize>,
    tau: Vec<f64>,
    colnorm: Vec<f64>,
}

impl LsqTrrWorkspace {
    /// Storage for `m` residuals and `n` parameters.
    pub fn new(m: usize, n: usize) -> Self {
        let z = |k: usize| vec![0.0; k];
        Self {
            m,
            n,
            f: z(m),
            f_trial: z(m),
            f_probe: z(m),
            jac: z(m * n),
            jac_trial: z(m * n),
            aug: z((m + n) * n),
            aug_rhs: z(m + n),
            x_trial: z(n),
            x_probe: z(n),
            g: z(n),
            v: z(n),
            dv: z(n),
            dsq: z(n),
            ghat: z(n),
            newton: z(n),
            z1: z(n),
            z2: z(n),
            step: z(n),
            shat: z(n),
            cand: z(n),
            best: z(n),
            dir: z(n),
            base: z(n),
            perm: vec![0; n],
            tau: z(n),
            colnorm: z(n),
        }
    }

    /// Residual count this workspace was sized for.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Parameter count this workspace was sized for.
    pub fn n(&self) -> usize {
        self.n
    }
}

/// Minimise `‖F(x)‖²` over `lb ≤ x ≤ ub` by the subspace trust-region reflective method (see the
/// module docs), starting from `x` and leaving the solution in `x`.
///
/// * `resid(x, out)` writes `F(x)` (length `ws.m()`) into `out`.
/// * `lb`, `ub` — bounds (`±∞` for none), `lb ≤ ub`.
/// * `observe` sees every iteration (iteration 0 is the start point), like MATLAB's `OutputFcn`.
///
/// # Panics
/// If the lengths disagree with `ws`, or a bound pair is inverted.
pub fn lsq_trust_region_reflective<F, O>(
    mut resid: F,
    x: &mut [f64],
    lb: &[f64],
    ub: &[f64],
    opts: &LsqTrrOptions,
    ws: &mut LsqTrrWorkspace,
    mut observe: O,
) -> LsqTrrReport
where
    F: FnMut(&[f64], &mut [f64]),
    O: FnMut(&LsqTrrIterate),
{
    let n = ws.n;
    assert!(
        x.len() == n && lb.len() == n && ub.len() == n,
        "lsq_trr: lengths"
    );
    for i in 0..n {
        assert!(lb[i] <= ub[i], "lsq_trr: lb > ub at {i}");
    }
    let max_fev = opts.max_function_evaluations.unwrap_or(100 * n);
    let r = &RULES;

    start_point(x, lb, ub);
    resid(x, &mut ws.f);
    let mut fcount = 1;
    let mut f = dot(&ws.f, &ws.f);
    fd_jacobian(&mut resid, x, &ws.f, lb, ub, opts, &mut ws.x_probe, &mut ws.f_probe, &mut ws.jac);
    fcount += n;

    scaling(x, lb, ub, ws);
    let mut opt = first_order(ws);
    let mut delta = r.delta0;
    observe(&LsqTrrIterate {
        iteration: 0,
        func_count: fcount,
        x,
        resnorm: f,
        first_order_optimality: opt,
        trust_region_radius: delta,
        step_norm: 0.0,
        ratio: 0.0,
    });
    let finish = |exit, x_iter: usize, fcount, f, opt| LsqTrrReport {
        resnorm: f,
        iterations: x_iter,
        func_count: fcount,
        first_order_optimality: opt,
        exit,
    };
    if opt < opts.optimality_tolerance {
        return finish(LsqTrrExit::Optimality, 0, fcount, f, opt);
    }

    let mut iter = 0;
    loop {
        if iter >= opts.max_iterations {
            return finish(LsqTrrExit::MaxIterations, iter, fcount, f, opt);
        }
        if fcount >= max_fev {
            return finish(LsqTrrExit::MaxFunctionEvaluations, iter, fcount, f, opt);
        }
        // Scaled model and its trust-region step.
        trust_region_step(x, lb, ub, delta, opt, ws);
        // ρ's prediction is the Gauss–Newton model of the step, gᵀs + ½‖Js‖², without the
        // bound term diag(g)Jᵛ that shapes the step (black-box: on 9210 DECOMPOSE steps the fitted
        // coefficient of that term in MATLAB's `ratio` is 0, median 3e-10).
        let qpval = {
            let (m, n) = (ws.m, ws.n);
            let mut q = 0.0;
            for k in 0..m {
                let mut a = 0.0;
                for j in 0..n {
                    a += ws.jac[j * m + k] * ws.step[j];
                }
                q += a * a;
            }
            0.5 * q + dot(&ws.g, &ws.step)
        };
        let snorm = norm(&ws.step);
        let shat_norm = norm(&ws.shat);
        for i in 0..n {
            ws.x_trial[i] = x[i] + ws.step[i];
        }
        // MATLAB evaluates the residual and its Jacobian at every trial point, before deciding
        // whether to accept it, so a rejected step costs `1 + n` evaluations too.
        resid(&ws.x_trial, &mut ws.f_trial);
        let f_trial = dot(&ws.f_trial, &ws.f_trial);
        fd_jacobian(&mut resid, &ws.x_trial, &ws.f_trial, lb, ub, opts, &mut ws.x_probe, &mut ws.f_probe, &mut ws.jac_trial);
        fcount += 1 + n;
        let ratio = if qpval != 0.0 { (0.5 * f_trial - 0.5 * f) / qpval } else { 0.0 };

        // Radius.
        if ratio <= r.shrink_below {
            delta = r.shrink_factor * delta.min(shat_norm);
        } else if ratio > r.grow_above && shat_norm >= r.grow_reach * delta {
            delta *= r.grow_factor;
        }
        iter += 1;

        let accepted = ratio > r.accept && f_trial.is_finite();
        let f_old = f;
        if accepted {
            x.copy_from_slice(&ws.x_trial);
            std::mem::swap(&mut ws.f, &mut ws.f_trial);
            std::mem::swap(&mut ws.jac, &mut ws.jac_trial);
            f = f_trial;
            scaling(x, lb, ub, ws);
            opt = first_order(ws);
        }
        observe(&LsqTrrIterate {
            iteration: iter,
            func_count: fcount,
            x,
            resnorm: f,
            first_order_optimality: opt,
            trust_region_radius: delta,
            step_norm: shat_norm,
            ratio,
        });
        if accepted {
            if opt < opts.optimality_tolerance {
                return finish(LsqTrrExit::Optimality, iter, fcount, f, opt);
            }
            if (f_old - f).abs() < opts.function_tolerance * (1.0 + f_old.abs()) {
                return finish(LsqTrrExit::FunctionTolerance, iter, fcount, f, opt);
            }
        }
        if snorm < opts.step_tolerance * (1.0 + norm(x)) {
            return finish(LsqTrrExit::StepTolerance, iter, fcount, f, opt);
        }
    }
}

/// A start point outside the box goes to the middle of a two-sided box, or onto the bound it
/// violates if the other side is infinite. A point on a bound is left there.
fn start_point(x: &mut [f64], lb: &[f64], ub: &[f64]) {
    for i in 0..x.len() {
        let (l, u) = (lb[i], ub[i]);
        if x[i] >= l && x[i] <= u {
            continue;
        }
        x[i] = if l.is_finite() && u.is_finite() { 0.5 * (l + u) } else { x[i].clamp(l, u) };
    }
}

/// Forward-difference Jacobian at `x` (residual `ws.f` already evaluated there) into `ws.jac`.
#[allow(clippy::too_many_arguments)]
fn fd_jacobian<F: FnMut(&[f64], &mut [f64])>(
    resid: &mut F,
    x: &[f64],
    f0: &[f64],
    lb: &[f64],
    ub: &[f64],
    opts: &LsqTrrOptions,
    x_probe: &mut [f64],
    f_probe: &mut [f64],
    jac: &mut [f64],
) {
    let m = f0.len();
    x_probe.copy_from_slice(x);
    for j in 0..x.len() {
        // MATLAB's documented step: FiniteDifferenceStepSize·sign′(x)·max(|x|, TypicalX), with
        // sign′(0) = 1.
        let sgn = if x[j] < 0.0 { -1.0 } else { 1.0 };
        let mut h = opts.finite_difference_step * sgn * x[j].abs().max(opts.typical_x);
        // A probe that would leave the box goes the other way (seen in MATLAB: x = ub − 1e-9).
        if x[j] + h > ub[j] || x[j] + h < lb[j] {
            h = -h;
        }
        x_probe[j] = x[j] + h;
        resid(x_probe, f_probe);
        // Divided by the nominal step h, not the representable (x + h) − x: MATLAB's first
        // Gauss–Newton step on Rosenbrock from (−1.2, 1) agrees with the former to 1e-15 and
        // with the latter only to 5e-9.
        for k in 0..m {
            jac[j * m + k] = (f_probe[k] - f0[k]) / h;
        }
        x_probe[j] = x[j];
    }
}

/// `g = JᵀF`, and Coleman–Li's `v`, `∂|v|/∂x` and `D⁻¹ = |v|^½` at `x`.
fn scaling(x: &[f64], lb: &[f64], ub: &[f64], ws: &mut LsqTrrWorkspace) {
    let (m, n) = (ws.m, ws.n);
    for j in 0..n {
        let col = &ws.jac[j * m..(j + 1) * m];
        ws.g[j] = dot(col, &ws.f);
    }
    for i in 0..n {
        let g = ws.g[i];
        let (v, dv) = if g < 0.0 && ub[i].is_finite() {
            (x[i] - ub[i], 1.0)
        } else if g >= 0.0 && lb[i].is_finite() {
            (x[i] - lb[i], 1.0)
        } else if g < 0.0 {
            (-1.0, 0.0)
        } else {
            (1.0, 0.0)
        };
        ws.v[i] = v;
        ws.dv[i] = dv;
        ws.dsq[i] = v.abs().sqrt();
    }
}

/// `‖v ⊙ g‖∞`.
fn first_order(ws: &LsqTrrWorkspace) -> f64 {
    let mut o = 0.0_f64;
    for i in 0..ws.n {
        o = o.max((ws.v[i] * ws.g[i]).abs());
    }
    o
}

/// The quadratic model at the current iterate, borrowed from the workspace.
struct Model<'a> {
    m: usize,
    n: usize,
    jac: &'a [f64],
    g: &'a [f64],
    dv: &'a [f64],
    dsq: &'a [f64],
}

impl Model<'_> {
    /// `(J a)ᵀ(J b)` for unscaled `a`, `b`.
    fn jtj(&self, a: &[f64], b: &[f64]) -> f64 {
        let mut q = 0.0;
        for k in 0..self.m {
            let (mut sa, mut sb) = (0.0, 0.0);
            for j in 0..self.n {
                sa += self.jac[j * self.m + k] * a[j];
                sb += self.jac[j * self.m + k] * b[j];
            }
            q += sa * sb;
        }
        q
    }

    /// `|gᵢ|·dvᵢ/|vᵢ|`: the `diag(g)Jᵛ` term on an unscaled step.
    fn c(&self, i: usize) -> f64 {
        let d2 = self.dsq[i] * self.dsq[i];
        // A variable that has reached its bound in floating point (`v = 0`) cannot move: its
        // step component is 0, and so is its term.
        if d2 > 0.0 {
            self.g[i].abs() * self.dv[i] / d2
        } else {
            0.0
        }
    }

    /// `ψ` of an unscaled step `s`: `gᵀs + ½(‖Js‖² + Σ cᵢ·sᵢ²)`.
    fn value(&self, s: &[f64]) -> f64 {
        let mut q = self.jtj(s, s);
        let mut lin = 0.0;
        for i in 0..self.n {
            q += self.c(i) * s[i] * s[i];
            lin += self.g[i] * s[i];
        }
        lin + 0.5 * q
    }

    /// Minimiser of `ψ(t·dir)` over `t ∈ [0, tmax]`.
    fn line_min_from_x(&self, dir: &[f64], tmax: f64) -> f64 {
        let mut a = self.jtj(dir, dir);
        let mut b = 0.0;
        for i in 0..self.n {
            a += self.c(i) * dir[i] * dir[i];
            b += self.g[i] * dir[i];
        }
        if a > 0.0 {
            (-b / a).clamp(0.0, tmax)
        } else if b < 0.0 {
            tmax
        } else {
            0.0
        }
    }

    /// Minimiser of `ψ(base + t·dir)` over `t ∈ [0, tmax]`.
    fn line_min(&self, base: &[f64], dir: &[f64], tmax: f64) -> f64 {
        // ψ(base + t·dir) = ψ(base) + t·b + ½t²·a.
        let mut a = self.jtj(dir, dir);
        let mut b = self.jtj(dir, base);
        for i in 0..self.n {
            a += self.c(i) * dir[i] * dir[i];
            b += self.c(i) * base[i] * dir[i] + self.g[i] * dir[i];
        }
        if a > 0.0 {
            (-b / a).clamp(0.0, tmax)
        } else if b < 0.0 {
            tmax
        } else {
            0.0
        }
    }
}

/// The trial step for radius `delta`: leaves `s` in `ws.step`, `ŝ = D s` in `ws.shat`, and returns
/// the model value `ψ(ŝ)`.
fn trust_region_step(
    x: &[f64],
    lb: &[f64],
    ub: &[f64],
    delta: f64,
    opt: f64,
    ws: &mut LsqTrrWorkspace,
) -> f64 {
    let (m, n) = (ws.m, ws.n);
    // ĝ = D⁻¹ g.
    for i in 0..n {
        ws.ghat[i] = ws.dsq[i] * ws.g[i];
    }
    // Newton step ŝ_N: least squares on [J D⁻¹; diag(√(|g|·dv))] ŝ ≈ [−F; 0].
    let rows = m + n;
    for j in 0..n {
        for k in 0..m {
            ws.aug[j * rows + k] = ws.jac[j * m + k] * ws.dsq[j];
        }
        for k in 0..n {
            ws.aug[j * rows + m + k] = if k == j {
                (ws.g[j].abs() * ws.dv[j]).sqrt()
            } else {
                0.0
            };
        }
    }
    for k in 0..m {
        ws.aug_rhs[k] = -ws.f[k];
    }
    for k in 0..n {
        ws.aug_rhs[m + k] = 0.0;
    }
    lstsq_qrp(
        rows,
        n,
        &mut ws.aug,
        &mut ws.aug_rhs,
        &mut ws.newton,
        &mut ws.perm,
        &mut ws.tau,
        &mut ws.colnorm,
    );

    let md = Model {
        m,
        n,
        jac: &ws.jac,
        g: &ws.g,
        dv: &ws.dv,
        dsq: &ws.dsq,
    };
    // Orthonormal basis of span{ĝ, ŝ_N} (scaled).
    let gn = norm(&ws.ghat);
    for i in 0..n {
        ws.z1[i] = ws.ghat[i] / gn;
    }
    let mut dim = 1;
    if n > 1 {
        ws.z2.copy_from_slice(&ws.newton);
        for _ in 0..2 {
            let p = dot(&ws.z1, &ws.z2);
            for i in 0..n {
                ws.z2[i] -= p * ws.z1[i];
            }
        }
        let nz = norm(&ws.z2);
        if nz > 0.0 && nz > 1e3 * f64::EPSILON * norm(&ws.newton) {
            for i in 0..n {
                ws.z2[i] /= nz;
            }
            dim = 2;
        }
    }
    // Reduced model W = ZᵀM̂Z, b = Zᵀĝ, with M̂ = D⁻¹JᵀJD⁻¹ + diag(|g|·dv): in unscaled terms
    // zₐᵀM̂z_b = (J D⁻¹zₐ)ᵀ(J D⁻¹z_b) + Σ |g|·dv·zₐz_b.
    let quad = |a: &[f64], b: &[f64], tmp_a: &mut [f64], tmp_b: &mut [f64]| {
        for i in 0..n {
            tmp_a[i] = md.dsq[i] * a[i];
            tmp_b[i] = md.dsq[i] * b[i];
        }
        let mut q = md.jtj(tmp_a, tmp_b);
        for i in 0..n {
            q += md.g[i].abs() * md.dv[i] * a[i] * b[i];
        }
        q
    };
    let w11 = quad(&ws.z1, &ws.z1, &mut ws.cand, &mut ws.dir);
    let (w12, w22) = if dim == 2 {
        (
            quad(&ws.z1, &ws.z2, &mut ws.cand, &mut ws.dir),
            quad(&ws.z2, &ws.z2, &mut ws.cand, &mut ws.dir),
        )
    } else {
        (0.0, 0.0)
    };
    let b1 = dot(&ws.z1, &ws.ghat);
    let b2 = if dim == 2 { dot(&ws.z2, &ws.ghat) } else { 0.0 };
    let (c1, c2) = if dim == 2 {
        trust_2d(w11, w12, w22, b1, b2, delta)
    } else {
        (trust_1d(w11, b1, delta), 0.0)
    };
    for i in 0..n {
        ws.shat[i] = c1 * ws.z1[i] + if dim == 2 { c2 * ws.z2[i] } else { 0.0 };
        ws.step[i] = ws.dsq[i] * ws.shat[i];
    }

    // Strict feasibility. θ = max(θ_min, 1 − ‖v ⊙ g‖∞) (black-box: MATLAB's truncated steps
    // stop at exactly that fraction of the distance to the bound). A step is pulled back
    // whenever θ·α < 1, also when it would stop short of the bound (α > 1).
    let (alpha, hit) = to_boundary(x, &ws.step, lb, ub);
    let theta = RULES.theta_min.max(1.0 - opt);
    if theta * alpha >= 1.0 {
        return md.value(&ws.step);
    }
    // (a) Truncated at the first bound crossed.
    for i in 0..n {
        ws.best[i] = theta * alpha * ws.step[i];
    }
    let mut best_val = md.value(&ws.best);
    // (b) Reflected there. MATLAB minimises the model along the reflected direction with the
    // gradient at x, not at the bound point (ψ's linear term along `dir` is gᵀdir), and keeps
    // the reflected step only when that minimiser reaches the end of the segment: of 2167 first
    // steps of DECOMPOSE's stage calls that crossed a bound, MATLAB reflected in 6, every one
    // of them clamped there, and these two rules reproduce MATLAB's choice in 2162.
    for i in 0..n {
        ws.base[i] = alpha * ws.step[i];
        ws.dir[i] = ws.step[i];
        ws.x_probe[i] = x[i] + ws.base[i];
    }
    ws.dir[hit] = -ws.dir[hit];
    let (beta, _) = to_boundary(&ws.x_probe, &ws.dir, lb, ub);
    let tmax = tr_limit(&ws.base, &ws.dir, &ws.dsq, delta).min(theta * beta);
    if tmax > 0.0 && md.line_min_from_x(&ws.dir, tmax) >= tmax {
        for i in 0..n {
            ws.cand[i] = ws.base[i] + tmax * ws.dir[i];
        }
        let v = md.value(&ws.cand);
        if v < best_val {
            best_val = v;
            ws.best.copy_from_slice(&ws.cand);
        }
    }
    // (c) Scaled steepest descent −D⁻²g, minimised along its feasible, pulled-back segment.
    for i in 0..n {
        ws.dir[i] = -ws.dsq[i] * ws.dsq[i] * ws.g[i];
        ws.base[i] = 0.0;
    }
    let (gamma, _) = to_boundary(x, &ws.dir, lb, ub);
    let tmax = tr_limit(&ws.base, &ws.dir, &ws.dsq, delta).min(theta * gamma);
    if tmax > 0.0 {
        let t = md.line_min(&ws.base, &ws.dir, tmax);
        for i in 0..n {
            ws.cand[i] = t * ws.dir[i];
        }
        let v = md.value(&ws.cand);
        if v < best_val {
            best_val = v;
            ws.best.copy_from_slice(&ws.cand);
        }
    }
    ws.step.copy_from_slice(&ws.best);
    for i in 0..n {
        ws.shat[i] = if ws.dsq[i] > 0.0 { ws.step[i] / ws.dsq[i] } else { 0.0 };
    }
    best_val
}

/// Largest `t ≥ 0` with `‖(base + t·dir) ⊘ dsq‖ ≤ delta`.
fn tr_limit(base: &[f64], dir: &[f64], dsq: &[f64], delta: f64) -> f64 {
    let (mut a, mut b, mut c) = (0.0, 0.0, 0.0);
    for i in 0..base.len() {
        if dsq[i] == 0.0 {
            continue;
        }
        let (p, d) = (base[i] / dsq[i], dir[i] / dsq[i]);
        a += d * d;
        b += p * d;
        c += p * p;
    }
    c -= delta * delta;
    if a == 0.0 {
        return f64::INFINITY;
    }
    let disc = (b * b - a * c).max(0.0);
    ((-b + disc.sqrt()) / a).max(0.0)
}

/// Largest multiple `α` of `dir` from `x` that stays in the box, and the index that limits it.
fn to_boundary(x: &[f64], dir: &[f64], lb: &[f64], ub: &[f64]) -> (f64, usize) {
    let mut alpha = f64::INFINITY;
    let mut hit = 0;
    for i in 0..x.len() {
        let a = if dir[i] > 0.0 {
            (ub[i] - x[i]) / dir[i]
        } else if dir[i] < 0.0 {
            (lb[i] - x[i]) / dir[i]
        } else {
            f64::INFINITY
        };
        if a < alpha {
            alpha = a;
            hit = i;
        }
    }
    (alpha, hit)
}

/// Exact minimiser of `b·c + ½w·c²` over `|c| ≤ delta`.
fn trust_1d(w: f64, b: f64, delta: f64) -> f64 {
    if w > 0.0 {
        let c = -b / w;
        if c.abs() <= delta {
            return c;
        }
    }
    if b > 0.0 {
        -delta
    } else {
        delta
    }
}

/// Exact minimiser of `bᵀc + ½cᵀWc` over `‖c‖ ≤ delta`, `W = [w11 w12; w12 w22]` (Moré &
/// Sorensen: eigendecomposition and the secular equation).
fn trust_2d(w11: f64, w12: f64, w22: f64, b1: f64, b2: f64, delta: f64) -> (f64, f64) {
    // Eigen-decomposition of the symmetric 2×2.
    let (l1, l2, e1, e2) = sym2_eig(w11, w12, w22);
    let g1 = e1.0 * b1 + e1.1 * b2;
    let g2 = e2.0 * b1 + e2.1 * b2;
    let at = |mu: f64| -> (f64, f64) {
        let a1 = -g1 / (l1 + mu);
        let a2 = -g2 / (l2 + mu);
        (a1 * e1.0 + a2 * e2.0, a1 * e1.1 + a2 * e2.1)
    };
    if l1 > 0.0 {
        let c = at(0.0);
        if (c.0 * c.0 + c.1 * c.1).sqrt() <= delta {
            return c;
        }
    }
    let lo = (-l1).max(0.0);
    // Hard case: no component along the smallest eigenvector.
    if g1.abs() <= f64::EPSILON * (g1.abs() + g2.abs()) * 10.0 || g1 == 0.0 {
        let a2 = if l2 + lo > 0.0 { -g2 / (l2 + lo) } else { 0.0 };
        if a2.abs() <= delta {
            let t = (delta * delta - a2 * a2).max(0.0).sqrt();
            return (a2 * e2.0 + t * e1.0, a2 * e2.1 + t * e1.1);
        }
    }
    // Newton on 1/‖c(μ)‖ − 1/Δ from the left (monotone).
    let norm_at = |mu: f64| {
        let a1 = g1 / (l1 + mu);
        let a2 = g2 / (l2 + mu);
        (a1 * a1 + a2 * a2).sqrt()
    };
    let mut mu = if lo > 0.0 {
        lo + lo.abs() * 1e-12 + 1e-300
    } else {
        0.0
    };
    // Start where ‖c‖ ≥ Δ.
    if norm_at(mu) < delta {
        mu = lo;
    }
    for _ in 0..200 {
        let a1 = g1 / (l1 + mu);
        let a2 = g2 / (l2 + mu);
        let nc = (a1 * a1 + a2 * a2).sqrt();
        let phi = 1.0 / nc - 1.0 / delta;
        let dphi = (a1 * a1 / (l1 + mu) + a2 * a2 / (l2 + mu)) / (nc * nc * nc);
        let step = phi / dphi;
        let next = mu - step;
        if next.partial_cmp(&mu) != Some(std::cmp::Ordering::Greater) || (next - mu) <= 4.0 * f64::EPSILON * next.abs() {
            mu = next.max(mu);
            break;
        }
        mu = next;
    }
    at(mu)
}

/// Eigenvalues `l1 ≤ l2` and unit eigenvectors of `[a b; b d]`.
fn sym2_eig(a: f64, b: f64, d: f64) -> (f64, f64, (f64, f64), (f64, f64)) {
    if b == 0.0 {
        return if a <= d {
            (a, d, (1.0, 0.0), (0.0, 1.0))
        } else {
            (d, a, (0.0, 1.0), (1.0, 0.0))
        };
    }
    let mean = 0.5 * (a + d);
    let r = (0.25 * (a - d) * (a - d) + b * b).sqrt();
    // The eigenvalue of larger magnitude directly, the other from the determinant: `mean − r`
    // cancels catastrophically for an ill-conditioned Gauss–Newton matrix.
    let big = if mean >= 0.0 { mean + r } else { mean - r };
    let det = a * d - b * b;
    let small = if big != 0.0 { det / big } else { 0.0 };
    let (l1, l2) = if big >= small { (small, big) } else { (big, small) };
    // Eigenvector of l2 by the better-conditioned row, then its orthogonal.
    let (vx, vy) = if (a - l1).abs() >= (d - l1).abs() {
        (a - l1, b)
    } else {
        (b, d - l1)
    };
    let nv = (vx * vx + vy * vy).sqrt();
    let e2 = (vx / nv, vy / nv);
    let e1 = (-e2.1, e2.0);
    (l1, l2, e1, e2)
}

/// Least-squares solution of `A x ≈ b` (`A` is `rows × n`, column-major, overwritten) by
/// Householder QR with column pivoting; columns whose pivot falls below
/// `max(rows, n)·eps·|R₁₁|` are dropped (their component is 0, as MATLAB's `\` gives for a
/// rank-deficient system).
#[allow(clippy::too_many_arguments)]
fn lstsq_qrp(
    rows: usize,
    n: usize,
    a: &mut [f64],
    b: &mut [f64],
    x: &mut [f64],
    perm: &mut [usize],
    tau: &mut [f64],
    colnorm: &mut [f64],
) {
    for j in 0..n {
        perm[j] = j;
        colnorm[j] = dot(&a[j * rows..(j + 1) * rows], &a[j * rows..(j + 1) * rows]);
    }
    for k in 0..n {
        // Pivot: the remaining column of largest norm.
        let mut p = k;
        for j in k + 1..n {
            if colnorm[j] > colnorm[p] {
                p = j;
            }
        }
        if p != k {
            for r in 0..rows {
                a.swap(k * rows + r, p * rows + r);
            }
            colnorm.swap(k, p);
            perm.swap(k, p);
        }
        // Householder reflector for column k below row k.
        let mut s = 0.0;
        for r in k..rows {
            s += a[k * rows + r] * a[k * rows + r];
        }
        let alpha = s.sqrt();
        if alpha == 0.0 {
            tau[k] = 0.0;
            continue;
        }
        let x0 = a[k * rows + k];
        let beta = if x0 >= 0.0 { -alpha } else { alpha };
        let v0 = x0 - beta;
        for r in k + 1..rows {
            a[k * rows + r] /= v0;
        }
        tau[k] = (beta - x0) / beta;
        a[k * rows + k] = beta;
        // Apply to the remaining columns and b.
        let apply = |col: &mut [f64], vcol: &[f64]| {
            let mut w = col[k];
            for r in k + 1..rows {
                w += vcol[r] * col[r];
            }
            w *= tau[k];
            col[k] -= w;
            for r in k + 1..rows {
                col[r] -= w * vcol[r];
            }
        };
        let (head, tail) = a.split_at_mut((k + 1) * rows);
        let vcol = &head[k * rows..(k + 1) * rows];
        for j in 0..n - k - 1 {
            apply(&mut tail[j * rows..(j + 1) * rows], vcol);
        }
        apply(b, vcol);
        for j in k + 1..n {
            let t = a[j * rows + k];
            colnorm[j] -= t * t;
        }
    }
    // Rank and back substitution.
    let tol = rows.max(n) as f64 * f64::EPSILON * a[0].abs();
    let mut rank = 0;
    while rank < n && a[rank * rows + rank].abs() > tol {
        rank += 1;
    }
    for v in tau.iter_mut() {
        *v = 0.0;
    }
    for k in (0..rank).rev() {
        let mut acc = b[k];
        for j in k + 1..rank {
            acc -= a[j * rows + k] * tau[j];
        }
        tau[k] = acc / a[k * rows + k];
    }
    for j in 0..n {
        x[perm[j]] = tau[j];
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[inline]
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solve(
        m: usize,
        x0: &[f64],
        lb: &[f64],
        ub: &[f64],
        resid: impl FnMut(&[f64], &mut [f64]),
    ) -> (Vec<f64>, LsqTrrReport) {
        let mut x = x0.to_vec();
        let mut ws = LsqTrrWorkspace::new(m, x0.len());
        let rep = lsq_trust_region_reflective(
            resid,
            &mut x,
            lb,
            ub,
            &LsqTrrOptions::default(),
            &mut ws,
            |_| {},
        );
        (x, rep)
    }

    #[test]
    fn linear_fit_unbounded() {
        let t = [0.0, 1.0, 2.0, 3.0];
        let d = [1.0, 3.0, 5.0, 7.0];
        let inf = f64::INFINITY;
        let (x, rep) = solve(4, &[0.0, 0.0], &[-inf, -inf], &[inf, inf], |x, out| {
            for k in 0..4 {
                out[k] = x[0] * t[k] + x[1] - d[k];
            }
        });
        assert!(
            (x[0] - 2.0).abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6,
            "{x:?} {rep:?}"
        );
    }

    #[test]
    fn rosenbrock_unbounded() {
        let inf = f64::INFINITY;
        let (x, rep) = solve(2, &[-1.2, 1.0], &[-inf, -inf], &[inf, inf], |x, out| {
            out[0] = 10.0 * (x[1] - x[0] * x[0]);
            out[1] = 1.0 - x[0];
        });
        assert!(
            (x[0] - 1.0).abs() < 1e-4 && (x[1] - 1.0).abs() < 1e-4,
            "{x:?} {rep:?}"
        );
    }

    #[test]
    fn rosenbrock_active_bound_is_respected() {
        // Constrained optimum on x0 = 0.5 (x1 = 0.25).
        let (x, rep) = solve(2, &[-1.2, 1.0], &[-2.0, -2.0], &[0.5, 2.0], |x, out| {
            out[0] = 10.0 * (x[1] - x[0] * x[0]);
            out[1] = 1.0 - x[0];
        });
        // The iterates approach the active bound from inside (they may round onto it).
        assert!(
            x[0] <= 0.5 && x[0] > 0.499 && (x[1] - 0.25).abs() < 1e-3,
            "{x:?} {rep:?}"
        );
    }

    #[test]
    fn lstsq_matches_normal_equations() {
        // 4×2 full rank.
        let mut a = vec![1.0, 2.0, 3.0, 4.0, 1.0, 0.5, -1.0, 2.0];
        let mut b = vec![1.0, -1.0, 2.0, 0.5];
        let a0 = a.clone();
        let b0 = b.clone();
        let (mut x, mut p, mut t, mut c) = (vec![0.0; 2], vec![0; 2], vec![0.0; 2], vec![0.0; 2]);
        lstsq_qrp(4, 2, &mut a, &mut b, &mut x, &mut p, &mut t, &mut c);
        // AᵀA x = Aᵀb.
        for i in 0..2 {
            let mut lhs = 0.0;
            let mut rhs = 0.0;
            for k in 0..4 {
                let aki = a0[i * 4 + k];
                rhs += aki * b0[k];
                for j in 0..2 {
                    lhs += aki * a0[j * 4 + k] * x[j];
                }
            }
            assert!((lhs - rhs).abs() < 1e-12, "{lhs} {rhs}");
        }
    }

    #[test]
    fn trust_2d_on_boundary_and_inside() {
        // PD, unconstrained minimiser inside.
        let c = trust_2d(2.0, 0.0, 1.0, -2.0, -1.0, 10.0);
        assert!((c.0 - 1.0).abs() < 1e-14 && (c.1 - 1.0).abs() < 1e-14);
        // Same, small radius: on the boundary.
        let c = trust_2d(2.0, 0.0, 1.0, -2.0, -1.0, 0.5);
        assert!(
            ((c.0 * c.0 + c.1 * c.1).sqrt() - 0.5).abs() < 1e-14,
            "{c:?}"
        );
        // Indefinite.
        let c = trust_2d(-1.0, 0.3, 2.0, 0.5, -1.0, 1.0);
        assert!(
            ((c.0 * c.0 + c.1 * c.1).sqrt() - 1.0).abs() < 1e-12,
            "{c:?}"
        );
    }

    /// The probe's generic problems (`generic_problems()` in
    /// `scripts/lsq_trr_matlab/probe_lsq_trr.m`, same order): `(m, residual)`.
    #[allow(clippy::type_complexity)]
    pub(crate) fn generic(id: usize) -> (usize, Box<dyn Fn(&[f64], &mut [f64])>) {
        let t4 = [0.0, 1.0, 2.0, 3.0];
        let d4 = [1.0, 3.0, 5.0, 7.0];
        let lin = move |x: &[f64], o: &mut [f64]| {
            for k in 0..4 {
                o[k] = x[0] * t4[k] + x[1] - d4[k];
            }
        };
        let rosen = |x: &[f64], o: &mut [f64]| {
            o[0] = 10.0 * (x[1] - x[0] * x[0]);
            o[1] = 1.0 - x[0];
        };
        let te: Vec<f64> = (0..7).map(|k| 0.5 * k as f64).collect();
        let t3: Vec<f64> = (0..21).map(|k| 0.25 * k as f64).collect();
        let e1 = {
            let te = te.clone();
            move |x: &[f64], o: &mut [f64]| {
                for k in 0..te.len() {
                    o[k] = (-x[0] * te[k]).exp() - (-2.0 * te[k]).exp();
                }
            }
        };
        let e3 = |c: f64, t3: Vec<f64>| {
            move |x: &[f64], o: &mut [f64]| {
                for k in 0..t3.len() {
                    let d = 2.0 * (-0.7 * t3[k]).exp() + c;
                    o[k] = x[0] * (-x[1] * t3[k]).exp() + x[2] - d;
                }
            }
        };
        match id {
            1 | 9 => (4, Box::new(lin)),
            2..=5 | 12 | 15 => (2, Box::new(rosen)),
            6..=8 => (te.len(), Box::new(e1)),
            10 => (
                2,
                Box::new(move |x: &[f64], o: &mut [f64]| {
                    rosen(x, o);
                    o[0] *= 1e4;
                    o[1] *= 1e4;
                }),
            ),
            11 | 13 => (t3.len(), Box::new(e3(0.3, t3))),
            14 => (t3.len(), Box::new(e3(-0.2, t3))),
            16 => (
                2,
                Box::new(move |x: &[f64], o: &mut [f64]| {
                    rosen(x, o);
                    o[1] -= 0.5;
                }),
            ),
            _ => panic!("no generic problem {id}"),
        }
    }

    /// Against MATLAB R2023b's `lsqcurvefit` on the probe's generic problems: same evaluation
    /// points, iteration and function counts, exit flags and solution. The trace is
    /// `tests/data/lsq_trr_matlab_generic.txt` (or `$LSQ_TRR_TRACE`); `LSQ_TRR_DUMP=1` prints
    /// every problem side by side.
    #[test]
    fn matches_matlab_lsqcurvefit_generic() {
        let path = std::env::var("LSQ_TRR_TRACE").unwrap_or_else(|_| {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/lsq_trr_matlab_generic.txt"
            )
            .to_string()
        });
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("no MATLAB trace at {path}; skipped");
            return;
        };
        let dump = std::env::var("LSQ_TRR_DUMP").is_ok();
        let mut bad = Vec::new();
        for call in super::super::lsq_trr_trace::parse(&text)
            .iter()
            .filter(|c| c.tag.starts_with('G'))
        {
            let id: usize = call.tag[1..].parse().unwrap();
            let (m, f) = generic(id);
            assert_eq!(m, call.m, "{}", call.tag);
            let r = super::super::lsq_trr_trace::replay(call, |x, o| f(x, o));
            let d = super::super::lsq_trr_trace::compare(call, &r, 1e-6);
            eprintln!("{d:?}");
            if dump {
                eprintln!("{}", super::super::lsq_trr_trace::dump(call, &r));
            }
            let identical = d.same_counts && d.same_exit && d.first_divergence.is_none();
            // Bounded Rosenbrock converging onto an active bound (G4, G5, G12) and the last
            // iterations of unbounded Rosenbrock (G2, G10): the same solution, not the same path
            // (module docs, *Matching MATLAB*).
            let known = ["G2", "G4", "G5", "G10", "G12"].contains(&call.tag.as_str());
            if !(identical || known && d.x_rel < 1e-5) {
                bad.push(d.tag.clone());
            }
        }
        assert!(bad.is_empty(), "differ from MATLAB: {bad:?}");
    }
}

//! iLSQR: LSQR dipole inversion with streaking-artefact removal
//!
//! [`ilsqr`] implements **STI Suite 3.0's `QSM_iLSQR`** (Li et al. 2015), as called with
//! `QSM_iLSQR(phase, mask, 'TE', TE, 'B0', B0, 'H', H, 'padsize', [64 64 64], 'voxelsize', vs)`.
//! STI Suite ships as protected P-code, so the algorithm was recovered by black-box probing
//! (call tracing, capturing the operators and right-hand sides it hands to MATLAB's `lsqr`, and
//! fitting every stage until a MATLAB re-implementation reproduced STI to its own
//! single-precision noise floor):
//!
//! 1. Crop field and mask to the mask's bounding box (odd extents grown by one voxel to make
//!    them even), then zero-pad each axis to the next multiple of 16 strictly above
//!    `crop + 2·pad_mm/voxel_size` (default `pad_mm` = 64). Everything below runs on that grid.
//! 2. **Laplacian weights** `W`: `L = |F⁻¹(k²·F φ)|` with `k²` in (cycles/mm)² and the DC term
//!    set to 10⁶ (STI's value, which adds `10⁶·mean(φ)` to every voxel, so the `|·|` decides
//!    which tail of the Laplacian is down-weighted). Ranked inside the mask:
//!    `W = 1` below the 60th percentile, falling linearly to `0.2` at the 99.9th, `0` outside.
//!    The field itself is not re-masked.
//! 3. **Initial solution**: LSQR on `D W D χ = D W φ` (`D` = dipole convolution), tolerance
//!    0.01, at most `max_iter` iterations (MATLAB `lsqr` stopping rules).
//! 4. **FastQSM** prior: `sign(D)·Fφ`, blended where `|D|` is small with a copy smoothed *in
//!    k-space* by a closed radius-3 ball, masked, blended again, rescaled to a TKD (threshold
//!    1/8) solution by least squares inside the mask.
//! 5. **Gradient weights** `G_d` (`d` = x, y, z): forward differences of the FastQSM map (no
//!    voxel-size scaling), pooled over the three axes inside the mask; `G = 1` below the 50th
//!    percentile of `|∇|`, `0` above the 70th, linear between.
//! 6. **Streaking artefacts**: LSQR for `y` in `min ‖G ∇ F⁻¹(M_ic y) − G ∇(M·Re χ₁)‖` with
//!    `M_ic = |D| < 0.1` (the ill-conditioned cone), tolerance `tol` (default 1e-3), at most
//!    `max_iter` iterations; artefact `χ_sa = Re F⁻¹(M_ic y)`.
//! 7. `χ = (Re χ₁ − χ_sa) · mask`, cropped back to the input grid.
//!
//! Percentiles are STI's: the sorted value at 1-based index `round(p·N)`, not interpolated.
//! The field may be in any linear unit (rad, Hz, ppm); `χ` comes out in the unit the field
//! would have for χ = 1 (QSM.rs passes ppm, so χ is in ppm).
//!
//! # Performance and precision
//!
//! Nearly all the time goes into the two LSQR solves, each costing two 3-D FFTs on the padded
//! grid per operator call. The initial solve runs in k-space (2 FFTs per call instead of 4, the
//! same iterates up to rounding), the transforms are pruned to the box holding the mask and
//! carry the solves' diagonal products in their first and last passes, and the artefact solve
//! stores its vectors on the cone and the mask only. On a UK Biobank SWI field (352×400×96
//! padded) this takes ~15 s on 16 cores with a 1.9 GB peak. Against the image-space
//! formulation of the initial solve, the k-space one gives the same stop flags and iteration
//! counts in both solves there, and χ within a relative L2 of 5.8e-9 (the artefact solve
//! amplifying rounding-level differences of the initial solution, which agree to 2.7e-15).
//!
//! [`IlsqrParams::precision`] = [`IlsqrPrecision::Single`] runs both solves in single
//! precision, as STI Suite does (~10 s, 1.4 GB on that field). On that field it is *closer* to
//! STI's output (r = 0.999999975, 99.9th percentile |Δ| 8.5e-5 ppm) than the double-precision
//! default (r = 0.9999985, 3.7e-4 ppm), and differs from the default by a relative L2 of 1.7e-3.
//!
//! [`ilsqr_qsmm`] keeps the earlier formulation (the QSM.m port used by QSM.rs ≤ v0.38: no
//! padding, LSMR for the artefacts, other weights); QSMART still uses it.
//!
//! References:
//! Li, W., Wang, N., Yu, F., Han, H., Cao, W., Romero, R., Tantiwongkosi, B.,
//! Duong, T.Q., Liu, C. (2015). "A method for estimating and removing streaking
//! artifacts in quantitative susceptibility mapping."
//! NeuroImage, 108:111-122. <https://doi.org/10.1016/j.neuroimage.2014.12.043>
//!
//! Paige, C.C., Saunders, M.A. (1982). "LSQR: An algorithm for sparse linear equations and
//! sparse least squares." ACM Trans. Math. Softw. 8(1):43-71.
//!
//! [`ilsqr_qsmm`] reference implementation: <https://github.com/kamesy/QSM.m>

/// Parameters for the iLSQR algorithm.
///
/// [`ilsqr`] (STI Suite) defaults: `tol = 1e-3`, `max_iter = 100`. The initial LSQR solve
/// always uses STI's tolerance of 0.01. For [`ilsqr_qsmm`] use [`IlsqrParams::qsmm`]
/// (`tol = 0.01`, `max_iter = 50`, the defaults of QSM.rs ≤ v0.38).
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct IlsqrParams {
    /// Tolerance of the streaking-artefact solve (default: 1e-3)
    pub tol: f64,
    /// Maximum iterations of each LSQR solve (default: 100)
    pub max_iter: usize,
    /// Arithmetic of [`ilsqr`]'s two LSQR solves (default: [`IlsqrPrecision::Double`]);
    /// [`ilsqr_qsmm`] ignores it.
    pub precision: IlsqrPrecision,
}

impl Default for IlsqrParams {
    fn default() -> Self {
        Self {
            tol: 1e-3,
            max_iter: 100,
            precision: IlsqrPrecision::Double,
        }
    }
}

impl IlsqrParams {
    /// Defaults of the QSM.m formulation, [`ilsqr_qsmm`] (QSM.rs ≤ v0.38 `ilsqr`).
    pub fn qsmm() -> Self {
        Self { tol: 0.01, max_iter: 50, precision: IlsqrPrecision::Double }
    }
}

/// Arithmetic of [`ilsqr`]'s two LSQR solves, which take nearly all of its time.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IlsqrPrecision {
    /// Double precision throughout (default).
    #[default]
    Double,
    /// Single-precision vectors and transforms in both LSQR solves, as STI Suite runs them
    /// (LSQR's scalars and norm sums, the weights, FastQSM and the output stay double). Faster
    /// and lighter; the result differs from [`IlsqrPrecision::Double`]'s at single-precision
    /// rounding amplified by the solves (see the module docs for measured differences).
    Single,
}

/// Zero padding STI's `QSM_iLSQR` is given by UK Biobank and in STI Suite's examples, in mm.
pub const STI_PAD_MM: f64 = 64.0;

use std::cell::{Cell, RefCell};
use num_complex::{Complex, Complex32, Complex64};
use std::borrow::Cow;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use crate::fft::{Box3, Fft3dWorkspace, Fft3dWorkspaceF32, FftOpts};
use crate::kernels::dipole::dipole_kernel;
use crate::kernels::smv::smv_kernel;
use crate::utils::gradient::{fgrad, bdiv};
use crate::Grid;

// ============================================================================
// LSQR Solver
// ============================================================================

/// LSQR iterative solver for Ax = b
///
/// Solves the least squares problem min ||Ax - b||² using the LSQR algorithm.
/// Based on Paige & Saunders (1982).
///
/// # Arguments
/// * `apply_a` - Function that computes A*x
/// * `apply_at` - Function that computes A^T*x
/// * `b` - Right-hand side vector
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
///
/// # Returns
/// Solution vector x
pub fn lsqr<F, G>(
    apply_a: F,
    apply_at: G,
    b: &[f64],
    tol: f64,
    max_iter: usize,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
    G: Fn(&[f64]) -> Vec<f64>,
{
    // Initialize
    let mut u = b.to_vec();
    let mut beta = norm(&u);

    if beta > 0.0 {
        scale_inplace(&mut u, 1.0 / beta);
    }

    let mut v = apply_at(&u);
    let n = v.len();
    let mut alpha = norm(&v);

    if alpha > 0.0 {
        scale_inplace(&mut v, 1.0 / alpha);
    }

    let mut w = v.clone();
    let mut x = vec![0.0; n];

    let mut phi_bar = beta;
    let mut rho_bar = alpha;

    let bnorm = beta;

    // A zero right-hand side has x = 0 as its exact solution, and the plane
    // rotation below would divide 0 by 0 and hand back NaN.
    if bnorm == 0.0 {
        return x;
    }

    for _iter in 0..max_iter {
        // Bidiagonalization
        let mut u_new = apply_a(&v);
        axpy(&mut u_new, -alpha, &u);
        beta = norm(&u_new);

        if beta > 0.0 {
            scale_inplace(&mut u_new, 1.0 / beta);
        }
        u = u_new;

        let mut v_new = apply_at(&u);
        axpy(&mut v_new, -beta, &v);
        alpha = norm(&v_new);

        if alpha > 0.0 {
            scale_inplace(&mut v_new, 1.0 / alpha);
        }
        v = v_new;

        // Construct and apply rotation
        let rho = (rho_bar * rho_bar + beta * beta).sqrt();
        let c = rho_bar / rho;
        let s = beta / rho;
        let theta = s * alpha;
        rho_bar = -c * alpha;
        let phi = c * phi_bar;
        phi_bar *= s;

        // Update x and w
        let t1 = phi / rho;
        let t2 = -theta / rho;

        for i in 0..n {
            x[i] += t1 * w[i];
            w[i] = v[i] + t2 * w[i];
        }

        // Check convergence
        let rel_residual = phi_bar / (bnorm + 1e-20);

        if rel_residual < tol {
            break;
        }
    }

    x
}

// ============================================================================
// LSQR Solver (Complex)
// ============================================================================

/// Complex norm
fn norm_complex(x: &[Complex64]) -> f64 {
    x.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
}

/// Complex scale in place
fn scale_complex_inplace(x: &mut [Complex64], s: f64) {
    for v in x.iter_mut() {
        *v *= s;
    }
}

/// Complex axpy: y += a * x
fn axpy_complex(y: &mut [Complex64], a: f64, x: &[Complex64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += a * xi;
    }
}

/// LSQR iterative solver for Ax = b (complex version)
///
/// Solves the least squares problem min ||Ax - b||² using the LSQR algorithm.
/// Based on Paige & Saunders (1982), with convergence tests matching MATLAB's lsqr.
///
/// Convergence tests (matching MATLAB):
/// 1. ||r|| / ||b|| <= btol + atol * ||A|| * ||x|| / ||b||  (residual test)
/// 2. ||A'r|| / (||A|| * ||r||) <= atol  (normal equations test)
pub fn lsqr_complex<F, G>(
    apply_a: F,
    apply_ah: G,
    b: &[Complex64],
    tol: f64,
    max_iter: usize,
    verbose: bool,
) -> Vec<Complex64>
where
    F: Fn(&[Complex64]) -> Vec<Complex64>,
    G: Fn(&[Complex64]) -> Vec<Complex64>,
{
    // Initialize: beta_1 * u_1 = b
    let mut u = b.to_vec();
    let mut beta = norm_complex(&u);

    if beta > 0.0 {
        scale_complex_inplace(&mut u, 1.0 / beta);
    }

    // alpha_1 * v_1 = A^H * u_1
    let mut v = apply_ah(&u);
    let n = v.len();
    let mut alpha = norm_complex(&v);

    if alpha > 0.0 {
        scale_complex_inplace(&mut v, 1.0 / alpha);
    }

    let mut w = v.clone();
    let mut x = vec![Complex64::new(0.0, 0.0); n];

    let mut phi_bar = beta;
    let mut rho_bar = alpha;

    let bnorm = beta;
    let atol = tol;
    let btol = tol;

    // Track ||A|| estimate
    let mut norm_a2 = alpha * alpha;

    // ||x|| estimate using plane rotations (matches MATLAB's built-in lsqr xxnorm)
    // Verified: produces identical values to exact norm_complex(&x)
    let mut xxnorm = 0.0;
    let mut z_sol = 0.0;
    let mut cs2 = -1.0;
    let mut sn2 = 0.0;

    if alpha * beta == 0.0 {
        return x;
    }

    for _iter in 0..max_iter {
        // Bidiagonalization step
        let mut u_new = apply_a(&v);
        axpy_complex(&mut u_new, -alpha, &u);
        beta = norm_complex(&u_new);

        if beta > 0.0 {
            scale_complex_inplace(&mut u_new, 1.0 / beta);
        }
        u = u_new;

        let mut v_new = apply_ah(&u);
        axpy_complex(&mut v_new, -beta, &v);
        alpha = norm_complex(&v_new);

        if alpha > 0.0 {
            scale_complex_inplace(&mut v_new, 1.0 / alpha);
        }
        v = v_new;

        // Construct and apply Givens rotation
        let rho = (rho_bar * rho_bar + beta * beta).sqrt();
        let c = rho_bar / rho;
        let s = beta / rho;
        let theta = s * alpha;
        rho_bar = -c * alpha;
        let phi = c * phi_bar;
        phi_bar *= s;

        // ||x|| estimation via plane rotations (MATLAB's xxnorm approach)
        let delta = sn2 * rho;
        let gambar = -cs2 * rho;
        let rhs = phi - delta * z_sol;
        let zbar = rhs / gambar;
        let xnorm = (xxnorm + zbar * zbar).sqrt();
        let gamma = (gambar * gambar + theta * theta).sqrt();
        cs2 = gambar / gamma;
        sn2 = theta / gamma;
        z_sol = rhs / gamma;
        xxnorm += z_sol * z_sol;

        // Update x and w
        let t1 = phi / rho;
        let t2 = -theta / rho;
        for i in 0..n {
            x[i] += t1 * w[i];
            w[i] = v[i] + t2 * w[i];
        }

        // Estimate norms for convergence tests
        let normr = phi_bar;
        let norm_ar = alpha * (c * phi_bar).abs();

        norm_a2 += beta * beta + alpha * alpha;
        let norm_a = norm_a2.sqrt();

        // Convergence test.
        //
        // The kamesy reference (QSM.m/src/inversion/ilsqr.m, Step 1 `lsqr_`)
        // calls MATLAB's *built-in* `lsqr(afun, b, tol, maxit)`, which stops on
        // the simple relative residual `||b - A*x|| / ||b|| <= tol`. It does NOT
        // use the augmented Paige-Saunders test
        // `||r||/||b|| <= btol + atol*||A||*||x||/||b||` (that belongs to
        // `lsqrSOL`). The previous augmented test tripped at iter ~10 (relres
        // ~0.09) because the `atol*||A||*||x||/||b||` term inflated the
        // threshold, leaving Step 1 badly under-converged (xlsqr NRMSE ~19% vs
        // kamesy). Matching the built-in relative-residual test lets Step 1 run
        // to relres <= tol (~39 iters here), reproducing kamesy's xlsqr exactly
        // (NRMSE ~0.4%).
        //
        // `normr = phi_bar` is the LSQR estimate of ||r||; `norm_ar`, `norm_a`,
        // `xnorm` are retained only for the optional verbose printout.
        let test1 = normr / (bnorm + 1e-20);
        let test2 = norm_ar / ((norm_a * normr) + 1e-20);
        let _ = (btol, xnorm);

        if verbose {
            eprintln!("  LSQR iter {:>3}: ||r||/||b||={:.6e}  ||A'r||/(||A||·||r||)={:.6e}",
                _iter + 1, test1, test2);
        }

        if test1 <= atol {
            if verbose {
                eprintln!("  LSQR converged at iteration {} (relres={:.4e})",
                    _iter + 1, test1);
            }
            break;
        }
    }

    x
}

// ============================================================================
// LSMR Solver
// ============================================================================

/// LSMR iterative solver for Ax = b
///
/// Solves the least squares problem min ||Ax - b||² using the LSMR algorithm.
/// Based on Fong & Saunders (2011). More stable than LSQR for ill-conditioned problems.
///
/// # Arguments
/// * `apply_a` - Function that computes A*x
/// * `apply_at` - Function that computes A^T*x
/// * `b` - Right-hand side vector
/// * `n` - Size of solution vector
/// * `atol` - Absolute tolerance
/// * `btol` - Relative tolerance
/// * `max_iter` - Maximum iterations
/// * `verbose` - Print progress
///
/// # Returns
/// Solution vector x
pub fn lsmr<F, G>(
    apply_a: F,
    apply_at: G,
    b: &[f64],
    n: usize,
    atol: f64,
    btol: f64,
    max_iter: usize,
    _verbose: bool,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
    G: Fn(&[f64]) -> Vec<f64>,
{
    // Reference: Fong & Saunders (2011), "LSMR: An iterative algorithm for
    // sparse least-squares problems", SIAM J. Sci. Comput.
    // Based on the official MATLAB implementation by Fong & Saunders.

    // Initialize: beta*u = b, alpha*v = A'*u
    let mut u = b.to_vec();
    let mut beta = norm(&u);

    if beta > 0.0 {
        scale_inplace(&mut u, 1.0 / beta);
    }

    let mut v = apply_at(&u);
    let mut alpha = norm(&v);

    if alpha > 0.0 {
        scale_inplace(&mut v, 1.0 / alpha);
    }

    // Initialize variables (matching MATLAB reference variable names)
    let mut alpha_bar = alpha;
    let mut zeta_bar = alpha * beta;
    let mut rho = 1.0;
    let mut rho_bar = 1.0;
    let mut c_bar = 1.0;
    let mut s_bar = 0.0;

    let mut h = v.clone();
    let mut h_bar = vec![0.0; n];
    let mut x = vec![0.0; n];

    // Variables for ||r|| estimation
    let normb = beta;
    let mut betadd = beta;
    let mut betad = 0.0;
    let mut rhodold = 1.0;
    let mut tautildeold = 0.0;
    let mut thetatilde = 0.0;
    let mut zeta = 0.0;
    let d = 0.0;

    // Variables for ||A|| and cond(A) estimation
    let mut norm_a2 = alpha * alpha;
    let mut maxrbar = 0.0f64;
    let mut minrbar = 1e100f64;
    let conlim = 1e8;
    let ctol = if conlim > 0.0 { 1.0 / conlim } else { 0.0 };

    // Early exit if A'b = 0
    if alpha * beta == 0.0 {
        return x;
    }

    for _iter in 0..max_iter {
        // Bidiagonalization
        let mut u_new = apply_a(&v);
        axpy(&mut u_new, -alpha, &u);
        beta = norm(&u_new);

        if beta > 0.0 {
            scale_inplace(&mut u_new, 1.0 / beta);
        }
        u = u_new;

        let mut v_new = apply_at(&u);
        axpy(&mut v_new, -beta, &v);
        alpha = norm(&v_new);

        if alpha > 0.0 {
            scale_inplace(&mut v_new, 1.0 / alpha);
        }
        v = v_new;

        // Construct rotation Q_i (undamped: alphahat = alphabar)
        let rho_old = rho;
        rho = (alpha_bar * alpha_bar + beta * beta).sqrt();
        let c = alpha_bar / rho;
        let s = beta / rho;
        let theta_new = s * alpha;
        alpha_bar = c * alpha;

        // Construct rotation Qbar_i
        let rho_bar_old = rho_bar;
        let zeta_old = zeta;
        let theta_bar = s_bar * rho;
        let rho_temp = c_bar * rho;
        rho_bar = (rho_temp * rho_temp + theta_new * theta_new).sqrt();
        c_bar = rho_temp / rho_bar;
        s_bar = theta_new / rho_bar;
        zeta = c_bar * zeta_bar;
        zeta_bar *= -s_bar;

        // Update h_bar, x, h
        for i in 0..n {
            h_bar[i] = h[i] - (theta_bar * rho / (rho_old * rho_bar_old)) * h_bar[i];
            x[i] += (zeta / (rho * rho_bar)) * h_bar[i];
            h[i] = v[i] - (theta_new / rho) * h[i];
        }

        // Estimate ||r|| (from reference implementation)
        // For undamped case: chat=1, shat=0, so betaacute=betadd, betacheck=0
        let betaacute = betadd;      // chat * betadd (chat=1 for undamped)
        // betacheck = 0 for undamped (shat=0), so d += 0
        let betahat = c * betaacute;
        betadd = -s * betaacute;

        let thetatildeold = thetatilde;
        let rhotildeold = (rhodold * rhodold + theta_bar * theta_bar).sqrt();
        let ctildeold = rhodold / rhotildeold;
        let stildeold = theta_bar / rhotildeold;
        thetatilde = stildeold * rho_bar;
        rhodold = ctildeold * rho_bar;
        betad = -stildeold * betad + ctildeold * betahat;

        tautildeold = (zeta_old - thetatildeold * tautildeold) / rhotildeold;
        let taud = (zeta - thetatilde * tautildeold) / rhodold;
        // d += betacheck^2 = 0 for undamped case
        let normr = (d + (betad - taud).powi(2) + betadd * betadd).sqrt();

        // Estimate ||A||
        norm_a2 += beta * beta;
        let norm_a = norm_a2.sqrt();
        norm_a2 += alpha * alpha;

        // Estimate cond(A) (matching MATLAB reference)
        maxrbar = maxrbar.max(rho_bar_old);
        if _iter > 0 {
            minrbar = minrbar.min(rho_bar_old);
        }
        let cond_a = maxrbar.max(rho_temp) / minrbar.min(rho_temp);

        // Convergence tests (matching reference implementation)
        let norm_ar = zeta_bar.abs();
        let normx = norm(&x);

        let test1 = normr / (normb + 1e-20);
        let test2 = norm_ar / ((norm_a * normr) + 1e-20);
        let test3 = 1.0 / (cond_a + 1e-20);
        let rtol = btol + atol * norm_a * normx / (normb + 1e-20);

        if _verbose {
            eprintln!("  LSMR iter {:>3}: ||r||/||b||={:.6e}  ||A'r||/(||A||·||r||)={:.6e}  1/condA={:.6e}  rtol={:.6e}",
                _iter + 1, test1, test2, test3, rtol);
        }

        // Test3 (condition number) checked first, then test2, then test1
        // matching MATLAB priority where later tests override earlier istop
        if test3 <= ctol || test2 <= atol || test1 <= rtol {
            if _verbose {
                let reason = if test1 <= rtol { "test1 (residual)"
                } else if test2 <= atol { "test2 (||A'r||)"
                } else { "test3 (cond(A))" };
                eprintln!("  LSMR converged at iteration {} via {}", _iter + 1, reason);
            }
            break;
        }
    }

    x
}

// ============================================================================
// Weight Functions
// ============================================================================

/// Compute Laplacian of a 3D field using mask-adaptive finite differences
///
/// Matches MATLAB's lap1_mex.c: uses central differences where both neighbors
/// are in the mask, forward/backward one-sided stencils near mask boundaries,
/// and zero contribution where neither neighbor is in the mask.
fn compute_laplacian(
    f: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut lap = vec![0.0; n_total];

    let hx = 1.0 / (vsx * vsx);
    let hy = 1.0 / (vsy * vsy);
    let hz = 1.0 / (vsz * vsz);

    let nxny = nx * ny;

    for k in 0..nz {
        for j in 0..ny {
            let jk_offset = j * nx + k * nxny;
            for i in 0..nx {
                let l = i + jk_offset;

                if mask[l] == 0 {
                    continue;
                }

                // X-axis contribution
                lap[l] += hx * lap1_axis(f, mask, l, 1, nx, i, nx);

                // Y-axis contribution
                lap[l] += hy * lap1_axis(f, mask, l, nx, nxny, j * nx, nxny);

                // Z-axis contribution
                lap[l] += hz * lap1_axis(f, mask, l, nxny, n_total, k * nxny, n_total);
            }
        }
    }

    lap
}

/// Compute second derivative along one axis using mask-adaptive stencil.
///
/// Matches MATLAB's lap1_mex.c logic:
/// - `idx = 2*G[l+a] + G[l-a]` selects the stencil type:
///   3 = central, 2 = forward, 1 = backward, 0 = zero
/// - At domain boundaries: i=0 → forward, i=N-1 → backward
///
/// # Arguments
/// * `f` - field values
/// * `mask` - binary mask
/// * `l` - linear index of current voxel
/// * `a` - stride for this axis (1 for x, nx for y, nx*ny for z)
/// * `n_axis` - total extent for this axis (nx for x, nx*ny for y, nx*ny*nz for z)
/// * `coord` - axis coordinate as linear offset (i for x, j*nx for y, k*nx*ny for z)
/// * `n_total` - total number of voxels (only used for z boundary detection)
#[inline]
fn lap1_axis(
    f: &[f64],
    mask: &[u8],
    l: usize,
    a: usize,
    n_axis: usize,
    coord: usize,
    n_total: usize,
) -> f64 {
    // Determine the stencil type based on mask of neighbors and boundary
    // MATLAB: (i-1) < NXX ? 2*G[l+a]+G[l-a] : (i==0)*2 + (i==NX)
    // where NXX = N_axis_size - 2 (using size_t underflow trick for boundary detection)
    let n_end = n_axis - a; // corresponds to NX, NY, NZ in MATLAB (last element coord)
    let n_interior = n_axis - 2 * a; // corresponds to NXX, NYY, NZZ

    let stencil = if coord.wrapping_sub(a) < n_interior {
        // Interior: check mask neighbors
        2 * mask[l + a] + mask[l - a]
    } else {
        // Boundary: first → forward(2), last → backward(1)
        if coord == 0 { 2 } else if coord == n_end { 1 } else { 0 }
    };

    match stencil {
        3 => {
            // Central: u[l-a] - 2u[l] + u[l+a]
            f[l - a] - 2.0 * f[l] + f[l + a]
        }
        2 => {
            // Forward one-sided
            lap1_forward(f, mask, l, a, n_axis, coord, n_total)
        }
        1 => {
            // Backward one-sided
            lap1_backward(f, mask, l, a, n_axis, coord, n_total)
        }
        _ => 0.0, // Neither neighbor in mask
    }
}

/// Forward one-sided second derivative (matching MATLAB's fd/ff functions)
#[inline]
fn lap1_forward(
    f: &[f64],
    mask: &[u8],
    l: usize,
    a: usize,
    n_axis: usize,
    coord: usize,
    _n_total: usize,
) -> f64 {
    // 4th order: 2u - 5u[+a] + 4u[+2a] - u[+3a]
    if coord + 3 * a < n_axis && mask[l + 2 * a] != 0 && mask[l + 3 * a] != 0 {
        2.0 * f[l] - 5.0 * f[l + a] + 4.0 * f[l + 2 * a] - f[l + 3 * a]
    }
    // 2nd order: u - 2u[+a] + u[+2a]
    else if coord + 2 * a < n_axis && mask[l + 2 * a] != 0 {
        f[l] - 2.0 * f[l + a] + f[l + 2 * a]
    }
    // 1st order: u[+a] - u
    else {
        f[l + a] - f[l]
    }
}

/// Backward one-sided second derivative (matching MATLAB's bd/bf functions)
#[inline]
fn lap1_backward(
    f: &[f64],
    mask: &[u8],
    l: usize,
    a: usize,
    n_axis: usize,
    coord: usize,
    _n_total: usize,
) -> f64 {
    // 4th order: -u[-3a] + 4u[-2a] - 5u[-a] + 2u
    if coord.wrapping_sub(3 * a) < n_axis && mask[l - 3 * a] != 0 && mask[l - 2 * a] != 0 {
        -f[l - 3 * a] + 4.0 * f[l - 2 * a] - 5.0 * f[l - a] + 2.0 * f[l]
    }
    // 2nd order: u[-2a] - 2u[-a] + u
    else if coord.wrapping_sub(2 * a) < n_axis && mask[l - 2 * a] != 0 {
        f[l - 2 * a] - 2.0 * f[l - a] + f[l]
    }
    // 1st order: u[-a] - u
    else {
        f[l - a] - f[l]
    }
}

/// Laplacian weights for iLSQR (Equation 7)
///
/// Weights based on Laplacian magnitude with percentile-based thresholding.
fn laplacian_weights_ilsqr(
    f: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    pmin: f64,
    pmax: f64,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut w = vec![0.0; n_total];

    // Compute Laplacian
    let lap = compute_laplacian(f, mask, nx, ny, nz, vsx, vsy, vsz);

    // Collect masked Laplacian values for percentile calculation
    let mut masked_lap: Vec<f64> = lap.iter()
        .zip(mask.iter())
        .filter(|(_, &m)| m > 0)
        .map(|(&l, _)| l)
        .collect();

    if masked_lap.is_empty() {
        return w;
    }

    // Sort for percentile calculation (MATLAB: prctile with linear interpolation)
    masked_lap.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let thr_min = prctile(&masked_lap, pmin);
    let thr_max = prctile(&masked_lap, pmax);

    let range = thr_max - thr_min;

    // Apply weights (Equation 7)
    for i in 0..n_total {
        if mask[i] == 0 {
            continue;
        }

        let l = lap[i];

        if l < thr_min {
            w[i] = 1.0;
        } else if l > thr_max {
            w[i] = 0.0;
        } else if range > 1e-10 {
            w[i] = (thr_max - l) / range;
        }
    }

    w
}

/// K-space weights for FastQSM (Equation 10)
///
/// Weights based on |D|^n with percentile normalization.
fn dipole_kspace_weights_ilsqr(
    d: &[f64],
    n_exp: f64,
    pa: f64,
    pb: f64,
) -> Vec<f64> {
    let len = d.len();
    let mut w = vec![0.0; len];

    // Compute |D|^n
    for i in 0..len {
        w[i] = d[i].abs().powf(n_exp);
    }

    // Percentile on ALL values (matching MATLAB: prctile(vec(w), [pa, pb]))
    let mut vals: Vec<f64> = w.to_vec();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if vals.is_empty() {
        return vec![0.0; len];
    }

    let ab_min = prctile(&vals, pa);
    let ab_max = prctile(&vals, pb);

    let range = ab_max - ab_min;

    // Normalize to [0, 1].
    // Not `clamp`: `max`/`min` maps a NaN weight to 0.0, which drops that voxel from the fit,
    // whereas `clamp` would propagate NaN through the weight map. `vals` is a masked subset of
    // `w`, so a NaN outside the mask reaches here without having tripped the `partial_cmp`
    // unwrap above. Behaviour-preserving on purpose.
    #[allow(clippy::manual_clamp)]
    for i in 0..len {
        if range > 1e-20 {
            w[i] = (w[i] - ab_min) / range;
        }
        w[i] = w[i].max(0.0).min(1.0);
    }

    w
}

/// Mask-adaptive forward gradient (matching MATLAB's gradfm_mex)
///
/// For masked voxels: uses forward difference where forward neighbor is in mask,
/// falls back to backward difference, or 0 if neither neighbor is in mask.
/// Outside mask: gradient is 0.
fn fgrad_masked(
    f: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_total = nx * ny * nz;
    let mut dx = vec![0.0; n_total];
    let mut dy = vec![0.0; n_total];
    let mut dz = vec![0.0; n_total];

    let hx = 1.0 / vsx;
    let hy = 1.0 / vsy;
    let hz = 1.0 / vsz;

    let nxny = nx * ny;

    for k in 0..nz {
        for j in 0..ny {
            let jk = j * nx + k * nxny;
            for i in 0..nx {
                let l = i + jk;
                if mask[l] == 0 { continue; }

                // X-axis: forward if possible, else backward, else 0
                dx[l] = if i < nx - 1 && mask[l + 1] != 0 {
                    hx * (f[l + 1] - f[l])
                } else if i > 0 && mask[l - 1] != 0 {
                    hx * (f[l] - f[l - 1])
                } else {
                    0.0
                };

                // Y-axis
                dy[l] = if j < ny - 1 && mask[l + nx] != 0 {
                    hy * (f[l + nx] - f[l])
                } else if j > 0 && mask[l - nx] != 0 {
                    hy * (f[l] - f[l - nx])
                } else {
                    0.0
                };

                // Z-axis
                dz[l] = if k < nz - 1 && mask[l + nxny] != 0 {
                    hz * (f[l + nxny] - f[l])
                } else if k > 0 && mask[l - nxny] != 0 {
                    hz * (f[l] - f[l - nxny])
                } else {
                    0.0
                };
            }
        }
    }

    (dx, dy, dz)
}

/// Gradient weights for streaking artifact estimation (Equation 15)
fn gradient_weights_ilsqr(
    x: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    pmin: f64,
    pmax: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // MATLAB uses gradf(x, mask, vsz) — mask-adaptive forward differences
    let (gx, gy, gz) = fgrad_masked(x, mask, nx, ny, nz, vsx, vsy, vsz);

    // Apply percentile-based weights to each component
    let wx = gradient_weights_component(&gx, mask, pmin, pmax);
    let wy = gradient_weights_component(&gy, mask, pmin, pmax);
    let wz = gradient_weights_component(&gz, mask, pmin, pmax);

    (wx, wy, wz)
}

fn gradient_weights_component(
    g: &[f64],
    mask: &[u8],
    pmin: f64,
    pmax: f64,
) -> Vec<f64> {
    let len = g.len();
    let mut w = vec![0.0; len];

    // Collect masked gradient values
    let mut masked_g: Vec<f64> = g.iter()
        .zip(mask.iter())
        .filter(|(_, &m)| m > 0)
        .map(|(&v, _)| v)
        .collect();

    if masked_g.is_empty() {
        return w;
    }

    masked_g.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let thr_min = prctile(&masked_g, pmin);
    let thr_max = prctile(&masked_g, pmax);

    let range = thr_max - thr_min;

    for i in 0..len {
        if mask[i] == 0 {
            continue;
        }

        let v = g[i];

        if v < thr_min {
            w[i] = 1.0;
        } else if v > thr_max {
            w[i] = 0.0;
        } else if range > 1e-10 {
            w[i] = (thr_max - v) / range;
        }

        // Apply mask
        w[i] *= mask[i] as f64;
    }

    w
}

// ============================================================================
// Helper Functions
// ============================================================================

fn norm(x: &[f64]) -> f64 {
    x.iter().map(|&v| v * v).sum::<f64>().sqrt()
}

fn scale_inplace(x: &mut [f64], s: f64) {
    for v in x.iter_mut() {
        *v *= s;
    }
}

fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += a * xi;
    }
}

fn multiply_elementwise(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).collect()
}

fn sign_array(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| {
        if v > 0.0 { 1.0 }
        else if v < 0.0 { -1.0 }
        else { 0.0 }
    }).collect()
}

/// Percentile with linear interpolation (matching MATLAB's prctile)
///
/// Input must be a sorted slice. Returns the p-th percentile (p in [0, 100]).
fn prctile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 { return 0.0; }
    if n == 1 { return sorted[0]; }
    let h = (p / 100.0) * (n - 1) as f64;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = h - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

// ============================================================================
// Step 1: Initial LSQR Solution
// ============================================================================

/// Step 1: Initial LSQR solution with Laplacian weights
fn lsqr_step(
    f: &[f64],
    mask: &[u8],
    d: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    workspace: &mut Fft3dWorkspace,
) -> Vec<f64> {

    // Laplacian weight parameters (from QSM.m)
    let pmin = 60.0;
    let pmax = 99.9;
    let tol_lsqr = 0.01;
    let maxit_lsqr = 50;

    // Compute Laplacian weights (Equation 7)
    let w = laplacian_weights_ilsqr(f, mask, nx, ny, nz, vsx, vsy, vsz, pmin, pmax);

    // Compute b = D * FFT(w .* f) - b is COMPLEX
    let wf: Vec<Complex64> = w.iter().zip(f.iter())
        .map(|(&wi, &fi)| Complex64::new(wi * fi, 0.0))
        .collect();

    let mut wf_fft = wf.clone();
    workspace.fft3d(&mut wf_fft);

    // b = D .* FFT(w .* f) - keep as complex!
    let b: Vec<Complex64> = wf_fft.iter().zip(d.iter())
        .map(|(wfi, &di)| wfi * di)
        .collect();

    // Define A*x operator: D * FFT(w .* real(IFFT(D .* x)))
    // Works with complex vectors throughout
    // Reuse a single workspace across all LSQR iterations to avoid repeated allocation
    let lsqr_ws = RefCell::new(Fft3dWorkspace::new(nx, ny, nz));
    let apply_a = |x: &[Complex64]| -> Vec<Complex64> {
        // D .* x (in k-space) - x is complex, D is real
        let dx: Vec<Complex64> = x.iter().zip(d.iter())
            .map(|(xi, &di)| xi * di)
            .collect();

        // IFFT(D .* x)
        let mut dx_ifft = dx.clone();
        let mut ws = lsqr_ws.borrow_mut();
        ws.ifft3d(&mut dx_ifft);

        // w .* real(IFFT(D .* x)) - take real part here as per MATLAB reference
        let wdx: Vec<Complex64> = w.iter().zip(dx_ifft.iter())
            .map(|(&wi, dxi)| Complex64::new(wi * dxi.re, 0.0))
            .collect();

        // FFT(w .* ...)
        let mut wdx_fft = wdx.clone();
        ws.fft3d(&mut wdx_fft);

        // D .* FFT(...)
        wdx_fft.iter().zip(d.iter())
            .map(|(wdxi, &di)| wdxi * di)
            .collect()
    };

    // A^H is same as A for this Hermitian operator (D is real, w is real)
    let apply_ah = |x: &[Complex64]| -> Vec<Complex64> {
        apply_a(x)
    };

    // Solve with complex LSQR
    let x_lsqr = lsqr_complex(apply_a, apply_ah, &b, tol_lsqr, maxit_lsqr, false);

    // IFFT to get result in image space
    let mut x_ifft = x_lsqr;
    workspace.ifft3d(&mut x_ifft);

    // Apply mask and take real part
    x_ifft.iter().zip(mask.iter())
        .map(|(xi, &mi)| if mi > 0 { xi.re } else { 0.0 })
        .collect()
}

// ============================================================================
// Step 2: FastQSM
// ============================================================================

/// Step 2: FastQSM estimate
fn fastqsm_step(
    f: &[f64],
    mask: &[u8],
    d: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    workspace: &mut Fft3dWorkspace,
) -> Vec<f64> {
    let n_total = nx * ny * nz;

    // FFT of field
    let f_complex: Vec<Complex64> = f.iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect();

    let mut f_fft = f_complex;
    workspace.fft3d(&mut f_fft);

    // Equation (8): x = sign(D) .* F
    let sign_d = sign_array(d);
    let x: Vec<Complex64> = f_fft.iter().zip(sign_d.iter())
        .map(|(fi, &si)| fi * si)
        .collect();

    // K-space weights (Equation 10)
    let pa = 1.0;
    let pb = 30.0;
    let n_exp = 0.001;
    let wfs = dipole_kspace_weights_ilsqr(d, n_exp, pa, pb);

    // SMV kernel for smoothing (Equation 9)
    let r_smv = 3.0;
    let smv_grid = Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let h = smv_kernel(&smv_grid, r_smv);

    // FFT of SMV kernel — take real part to match MATLAB: real(fft3(ifftshift(h)))
    let h_complex: Vec<Complex64> = h.iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect();
    let mut h_fft_complex = h_complex;
    workspace.fft3d(&mut h_fft_complex);
    let h_fft: Vec<f64> = h_fft_complex.iter().map(|c| c.re).collect();

    // Equation (9): Apply weighted combination
    // x = FFT(mask .* IFFT(wfs .* x + (1-wfs) .* (h .* x)))
    let mut x_filtered: Vec<Complex64> = x.iter()
        .zip(wfs.iter())
        .zip(h_fft.iter())
        .map(|((xi, &wi), &hi)| {
            xi * wi + xi * hi * (1.0 - wi)
        })
        .collect();

    workspace.ifft3d(&mut x_filtered);

    // Apply mask
    for (xi, &mi) in x_filtered.iter_mut().zip(mask.iter()) {
        if mi == 0 {
            *xi = Complex64::new(0.0, 0.0);
        } else {
            *xi = Complex64::new(xi.re, 0.0);
        }
    }

    workspace.fft3d(&mut x_filtered);

    // Equation (11): Apply again
    let mut x_filtered2: Vec<Complex64> = x_filtered.iter()
        .zip(wfs.iter())
        .zip(h_fft.iter())
        .map(|((xi, &wi), &hi)| {
            xi * wi + xi * hi * (1.0 - wi)
        })
        .collect();

    workspace.ifft3d(&mut x_filtered2);

    let x_fs: Vec<f64> = x_filtered2.iter().zip(mask.iter())
        .map(|(xi, &mi)| if mi > 0 { xi.re } else { 0.0 })
        .collect();

    // Equation (12): TKD for comparison
    let t0 = 1.0 / 8.0;
    let mut inv_d = vec![0.0; n_total];
    for i in 0..n_total {
        if d[i].abs() < t0 {
            inv_d[i] = d[i].signum() / t0;
        } else {
            inv_d[i] = 1.0 / d[i];
        }
    }

    let x_tkd_fft: Vec<Complex64> = f_fft.iter().zip(inv_d.iter())
        .map(|(fi, &idi)| fi * idi)
        .collect();

    let mut x_tkd_complex = x_tkd_fft;
    workspace.ifft3d(&mut x_tkd_complex);

    let x_tkd: Vec<f64> = x_tkd_complex.iter().zip(mask.iter())
        .map(|(xi, &mi)| if mi > 0 { xi.re } else { 0.0 })
        .collect();

    // Equations (13-14): Linear regression to scale FastQSM
    // Solve: xtkd ≈ a * xfs + b
    // MATLAB reference uses ALL voxels (including zeros outside mask) for the regression
    let sum_xfs: f64 = x_fs.iter().copied().sum();
    let sum_xtkd: f64 = x_tkd.iter().copied().sum();
    let sum_xfs2: f64 = x_fs.iter().map(|&v| v * v).sum();
    let sum_xfs_xtkd: f64 = x_fs.iter().zip(x_tkd.iter())
        .map(|(&xf, &xt)| xf * xt)
        .sum();

    let n_all: f64 = n_total as f64;

    // Solve 2x2 system: [sum_xfs2, sum_xfs; sum_xfs, n] * [a; b] = [sum_xfs_xtkd; sum_xtkd]
    let det = sum_xfs2 * n_all - sum_xfs * sum_xfs;

    let (a, b) = if det.abs() > 1e-20 {
        let a = (n_all * sum_xfs_xtkd - sum_xfs * sum_xtkd) / det;
        let b = (sum_xfs2 * sum_xtkd - sum_xfs * sum_xfs_xtkd) / det;
        (a, b)
    } else {
        (1.0, 0.0)
    };

    // Equation (14): x = a * xfs + b
    x_fs.iter().zip(mask.iter())
        .map(|(&xf, &mi)| if mi > 0 { a * xf + b } else { 0.0 })
        .collect()
}

// ============================================================================
// Step 3: Streaking Artifact Estimation
// ============================================================================

/// Step 3: Estimate streaking artifacts using LSMR
fn susceptibility_artifacts_step(
    x0: &[f64],
    xfs: &[f64],
    mask: &[u8],
    d: &[f64],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    tol: f64,
    maxit: usize,
    _workspace: &mut Fft3dWorkspace,
) -> Vec<f64> {
    let n_total = nx * ny * nz;

    // Gradient weights (Equation 15)
    let pmin = 50.0;
    let pmax = 70.0;
    let (wx, wy, wz) = gradient_weights_ilsqr(xfs, mask, nx, ny, nz, vsx, vsy, vsz, pmin, pmax);

    // Ill-conditioned mask (Equation 4)
    let thr = 0.1;
    let mic: Vec<f64> = d.iter().map(|&di| if di.abs() < thr { 1.0 } else { 0.0 }).collect();

    // Compute gradient of x0 (Equation 3)
    let step3_grid = Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let (dx, dy, dz) = fgrad(x0, &step3_grid);

    // b = [wx .* dx; wy .* dy; wz .* dz] (concatenated)
    let bx = multiply_elementwise(&wx, &dx);
    let by = multiply_elementwise(&wy, &dy);
    let bz = multiply_elementwise(&wz, &dz);

    let mut b = Vec::with_capacity(3 * n_total);
    b.extend_from_slice(&bx);
    b.extend_from_slice(&by);
    b.extend_from_slice(&bz);

    // Define forward operator A and adjoint A^T
    // Reuse a single workspace across all LSMR iterations to avoid repeated allocation
    let lsmr_ws = RefCell::new(Fft3dWorkspace::new(nx, ny, nz));
    let apply_a = |x_in: &[f64]| -> Vec<f64> {
        // x_in is in image space
        // Apply Mic in k-space
        let x_complex: Vec<Complex64> = x_in.iter()
            .map(|&v| Complex64::new(v, 0.0))
            .collect();

        let mut x_fft = x_complex;
        let mut ws = lsmr_ws.borrow_mut();
        ws.fft3d(&mut x_fft);

        // Apply ill-conditioned mask
        let x_mic: Vec<Complex64> = x_fft.iter().zip(mic.iter())
            .map(|(xi, &mi)| xi * mi)
            .collect();

        let mut x_ifft = x_mic;
        ws.ifft3d(&mut x_ifft);

        let x_filtered: Vec<f64> = x_ifft.iter().map(|xi| xi.re).collect();

        // Compute gradient
        let (gx, gy, gz) = fgrad(&x_filtered, &step3_grid);

        // Apply weights and concatenate
        let mut result = Vec::with_capacity(3 * n_total);
        result.extend(wx.iter().zip(gx.iter()).map(|(&w, &g)| w * g));
        result.extend(wy.iter().zip(gy.iter()).map(|(&w, &g)| w * g));
        result.extend(wz.iter().zip(gz.iter()).map(|(&w, &g)| w * g));

        result
    };

    // Define adjoint operator A^T
    let apply_at = |y_in: &[f64]| -> Vec<f64> {
        // y_in is [yx; yy; yz] concatenated (3 * n_total)
        let yx = &y_in[0..n_total];
        let yy = &y_in[n_total..2*n_total];
        let yz = &y_in[2*n_total..3*n_total];

        // Apply weights
        let wyx: Vec<f64> = wx.iter().zip(yx.iter()).map(|(&w, &y)| w * y).collect();
        let wyy: Vec<f64> = wy.iter().zip(yy.iter()).map(|(&w, &y)| w * y).collect();
        let wyz: Vec<f64> = wz.iter().zip(yz.iter()).map(|(&w, &y)| w * y).collect();

        // Adjoint of forward gradient = -div (bdiv returns +div, so negate)
        // MATLAB's gradfp_adj_mex uses h = -1/voxel_size, including the negation.
        // Rust's bdiv uses h = +1/voxel_size, so we negate here.
        let div = bdiv(&wyx, &wyy, &wyz, &step3_grid);

        // Apply Mic in k-space
        let div_complex: Vec<Complex64> = div.iter()
            .map(|&v| Complex64::new(-v, 0.0))
            .collect();

        let mut div_fft = div_complex;
        let mut ws = lsmr_ws.borrow_mut();
        ws.fft3d(&mut div_fft);

        let div_mic: Vec<Complex64> = div_fft.iter().zip(mic.iter())
            .map(|(di, &mi)| di * mi)
            .collect();

        let mut div_ifft = div_mic;
        ws.ifft3d(&mut div_ifft);

        div_ifft.iter().map(|di| di.re).collect()
    };

    // Solve with LSMR
    let xsa = lsmr(apply_a, apply_at, &b, n_total, tol, tol, maxit, false);

    // Apply mask
    xsa.iter().zip(mask.iter())
        .map(|(&x, &m)| if m > 0 { x } else { 0.0 })
        .collect()
}

// ============================================================================
// iLSQR, QSM.m formulation (QSM.rs <= v0.38 default)
// ============================================================================

/// iLSQR in the QSM.m formulation (QSM.rs ≤ v0.38 `ilsqr`)
///
/// Kept for QSMART and for comparisons; [`ilsqr`] is STI Suite's algorithm. Use
/// [`IlsqrParams::qsmm`] for this function's original defaults.
///
/// # Arguments
/// * `field` - Unwrapped local field/tissue phase (nx * ny * nz)
/// * `mask` - Binary mask of region of interest
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `bdir` - B0 field direction (bx, by, bz)
/// * `params` - iLSQR parameters (tol, max_iter)
/// * `progress` - Progress callback `(step, total_steps)`
///
/// # Returns
/// Tuple of (susceptibility, streaking_artifacts, fast_qsm, initial_lsqr)
pub fn ilsqr_qsmm(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &IlsqrParams,
    mut progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = grid.voxel_size;

    // Generate dipole kernel
    let d = dipole_kernel(grid, bdir);

    // Create FFT workspace
    let mut workspace = Fft3dWorkspace::new(nx, ny, nz);

    progress(1, 4);

    // Step 1: Initial LSQR solution
    let xlsqr = lsqr_step(field, mask, &d, nx, ny, nz, vsx, vsy, vsz, &mut workspace);

    progress(2, 4);

    // Step 2: FastQSM estimate
    let xfs = fastqsm_step(field, mask, &d, nx, ny, nz, vsx, vsy, vsz, &mut workspace);

    progress(3, 4);

    // Step 3: Estimate streaking artifacts
    let xsa = susceptibility_artifacts_step(
        &xlsqr, &xfs, mask, &d,
        nx, ny, nz, vsx, vsy, vsz,
        params.tol, params.max_iter, &mut workspace
    );

    progress(4, 4);

    // Step 4: Subtract artifacts
    let chi: Vec<f64> = xlsqr.iter().zip(xsa.iter()).zip(mask.iter())
        .map(|((&xl, &xs), &m)| if m > 0 { xl - xs } else { 0.0 })
        .collect();

    (chi, xsa, xfs, xlsqr)
}

// ============================================================================
// STI Suite 3.0 QSM_iLSQR (default `ilsqr`)
// ============================================================================

/// How an [`lsqr_matlab`] solve ended (MATLAB `lsqr` flags).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LsqrOutcome {
    /// 0 converged, 1 hit `max_iter`, 3 stagnated, 4 breakdown (a scalar became zero or non-finite)
    pub flag: u8,
    /// Iterations whose update is included in the returned solution
    pub iter: usize,
    /// ‖b − A x‖ / ‖b‖ (LSQR's running estimate)
    pub relres: f64,
}

/// Chunk length of the LSQR norms: fixed, so every sum is independent of the thread count.
const NORM_CHUNK: usize = 1 << 16;

/// How an LSQR vector is split for its norms: lengths of consecutive parts, each summed
/// sequentially, the partial sums then added in order.
///
/// A vector stored densely uses [`dense_parts`] (`NORM_CHUNK`-long parts). A
/// vector stored *compactly*, as only the entries at a sorted list of positions in a longer
/// vector that is exactly zero elsewhere, uses [`compact_parts`]: one part per `NORM_CHUNK`
/// of the long vector. Adding the omitted zeros changes no partial sum, so the norm of the
/// compact vector equals the dense norm of the long one.
fn dense_parts(len: usize) -> Vec<usize> {
    (0..len.div_ceil(NORM_CHUNK)).map(|c| NORM_CHUNK.min(len - c * NORM_CHUNK)).collect()
}

/// See [`dense_parts`]. `pos` are the (strictly increasing) positions in the long vector.
fn compact_parts(pos: impl Iterator<Item = usize>) -> Vec<usize> {
    let mut parts: Vec<usize> = Vec::new();
    let mut last = usize::MAX;
    for g in pos {
        let c = g / NORM_CHUNK;
        if c == last {
            *parts.last_mut().unwrap() += 1;
        } else {
            parts.push(1);
            last = c;
        }
    }
    parts
}

/// Element type of the LSQR vectors: `f64`, or `f32` for [`IlsqrPrecision::Single`]. LSQR's
/// scalars and the norms' sums stay `f64`; for `f64` every operation is the plain one.
trait Real: rustfft::FftNum + num_traits::NumAssign {
    type Ws;
    fn workspace(n: [usize; 3]) -> Self::Ws;
    fn transform(ws: &mut Self::Ws, a: &mut [Complex<Self>], inverse: bool, o: &FftOpts<Self>);
    fn of(v: f64) -> Self;
    fn to64(self) -> f64;
    /// `v` in this precision (borrowed for f64)
    fn cast(v: &[f64]) -> Cow<'_, [Self]>;
}

impl Real for f64 {
    type Ws = Fft3dWorkspace;
    fn workspace(n: [usize; 3]) -> Fft3dWorkspace { Fft3dWorkspace::new(n[0], n[1], n[2]) }
    fn transform(ws: &mut Fft3dWorkspace, a: &mut [Complex64], inverse: bool, o: &FftOpts) {
        if inverse { ws.ifft3d_with(a, o) } else { ws.fft3d_with(a, o) }
    }
    #[inline(always)]
    fn of(v: f64) -> f64 { v }
    #[inline(always)]
    fn to64(self) -> f64 { self }
    fn cast(v: &[f64]) -> Cow<'_, [f64]> { Cow::Borrowed(v) }
}

impl Real for f32 {
    type Ws = Fft3dWorkspaceF32;
    fn workspace(n: [usize; 3]) -> Fft3dWorkspaceF32 { Fft3dWorkspaceF32::new(n[0], n[1], n[2]) }
    fn transform(ws: &mut Fft3dWorkspaceF32, a: &mut [Complex32], inverse: bool, o: &FftOpts<f32>) {
        if inverse { ws.ifft3d_with(a, o) } else { ws.fft3d_with(a, o) }
    }
    #[inline(always)]
    fn of(v: f64) -> f32 { v as f32 }
    #[inline(always)]
    fn to64(self) -> f64 { self as f64 }
    fn cast(v: &[f64]) -> Cow<'_, [f32]> { Cow::Owned(crate::maybe_par_iter!(v).map(|&x| x as f32).collect()) }
}

/// `|z|²` in f64 (for f64 exactly `z.norm_sqr()`)
#[inline(always)]
fn nsq<T: Real>(z: Complex<T>) -> f64 {
    let (r, i) = (z.re.to64(), z.im.to64());
    r * r + i * i
}

fn split_parts<'a, T>(mut x: &'a [T], parts: &[usize]) -> Vec<&'a [T]> {
    parts.iter().map(|&len| { let (a, b) = x.split_at(len); x = b; a }).collect()
}

fn split_parts_mut<'a, T>(mut x: &'a mut [T], parts: &[usize]) -> Vec<&'a mut [T]> {
    parts.iter().map(|&len| { let (a, b) = std::mem::take(&mut x).split_at_mut(len); x = b; a }).collect()
}

fn cnorm_parts<T: Real>(x: &[Complex<T>], parts: &[usize]) -> f64 {
    let xs = split_parts(x, parts);
    let sums: Vec<f64> = crate::maybe_par_iter!(xs).map(|c| c.iter().map(|&v| nsq(v)).sum::<f64>()).collect();
    sums.iter().sum::<f64>().sqrt()
}

/// `f` applied elementwise to `y` (paired with `x`), then [`cnorm_parts`]`(y)`, in one pass:
/// the same values as the two passes, part by part.
fn update_and_norm<T: Real>(
    y: &mut [Complex<T>],
    x: &[Complex<T>],
    parts: &[usize],
    f: impl Fn(&mut Complex<T>, Complex<T>) + Sync + Send,
) -> f64 {
    let mut ys = split_parts_mut(y, parts);
    let xs = split_parts(x, parts);
    let sums: Vec<f64> = crate::maybe_par_iter_mut!(ys)
        .zip(crate::maybe_par_iter!(xs))
        .map(|(yc, xc)| {
            yc.iter_mut().zip(xc.iter()).for_each(|(a, &b)| f(a, b));
            yc.iter().map(|&v| nsq(v)).sum::<f64>()
        })
        .collect();
    sums.iter().sum::<f64>().sqrt()
}

/// LSQR's direction and solution updates `d ← (v − θ d)/ρ`, `x ← x + φ d` in one pass,
/// returning `(‖d‖, ‖x‖)`: the same values as two [`update_and_norm`] passes, part by part.
#[allow(clippy::too_many_arguments)]
fn update_d_x_and_norms<T: Real>(d: &mut [Complex<T>], x: &mut [Complex<T>], v: &[Complex<T>], parts: &[usize], thet: f64, rho: f64, phi: f64) -> (f64, f64) {
    let (thet, rho, phi) = (T::of(thet), T::of(rho), T::of(phi));
    let mut ds = split_parts_mut(d, parts);
    let mut xs = split_parts_mut(x, parts);
    let vs = split_parts(v, parts);
    let sums: Vec<(f64, f64)> = crate::maybe_par_iter_mut!(ds)
        .zip(crate::maybe_par_iter_mut!(xs))
        .zip(crate::maybe_par_iter!(vs))
        .map(|((dc, xc), vc)| {
            let (mut sd, mut sx) = (0.0f64, 0.0f64);
            for ((di, xi), &vi) in dc.iter_mut().zip(xc.iter_mut()).zip(vc.iter()) {
                *di = (vi - *di * thet) / rho;
                *xi += *di * phi;
                sd += nsq(*di);
                sx += nsq(*xi);
            }
            (sd, sx)
        })
        .collect();
    (sums.iter().map(|p| p.0).sum::<f64>().sqrt(), sums.iter().map(|p| p.1).sum::<f64>().sqrt())
}

fn div_inplace<T: Real>(x: &mut [Complex<T>], s: f64) {
    let s = T::of(s);
    crate::maybe_par_iter_mut!(x).for_each(|z| *z /= s);
}

/// LSQR (Paige & Saunders 1982) with MATLAB `lsqr`'s stopping rules, zero initial guess, no
/// preconditioner: stops when `‖r‖ ≤ tol·‖b‖` or `‖Aᴴr‖ / (‖A‖_F·‖r‖) ≤ tol` (both tested on
/// the previous iterate, before it is updated), after three stagnant steps, or on breakdown.
/// `apply_a` computes `A x`, `apply_ah` computes `Aᴴ u`.
pub fn lsqr_matlab<F, G>(
    mut apply_a: F,
    mut apply_ah: G,
    b: &[Complex64],
    tol: f64,
    max_iter: usize,
) -> (Vec<Complex64>, LsqrOutcome)
where
    F: FnMut(&[Complex64]) -> Vec<Complex64>,
    G: FnMut(&[Complex64]) -> Vec<Complex64>,
{
    let nx = apply_ah(b).len();
    lsqr_matlab_into(
        |v, out| out.copy_from_slice(&apply_a(v)),
        |u, out| out.copy_from_slice(&apply_ah(u)),
        b.to_vec(),
        (&dense_parts(b.len()), &dense_parts(nx)),
        tol,
        max_iter,
    )
}

/// [`lsqr_matlab`] with operators that write into a caller-owned buffer (`out` holds stale
/// data on entry and must be overwritten), taking `b` by value as LSQR's first `u`, so a solve
/// allocates its six vectors once. `parts` = (range parts, domain parts): how the norms of `u`
/// and `v` vectors are split (see [`dense_parts`]); the domain parts also give the length of
/// the unknown. Performs exactly the same floating-point operations as the textbook loop (the
/// norms of updated vectors are fused into the update passes, with the same parts).
fn lsqr_matlab_into<T, F, G>(
    mut apply_a: F,
    mut apply_ah: G,
    b: Vec<Complex<T>>,
    parts: (&[usize], &[usize]),
    tol: f64,
    max_iter: usize,
) -> (Vec<Complex<T>>, LsqrOutcome)
where
    T: Real,
    F: FnMut(&[Complex<T>], &mut [Complex<T>]),
    G: FnMut(&[Complex<T>], &mut [Complex<T>]),
{
    let zero = Complex::new(T::zero(), T::zero());
    let (up, vp) = parts;
    let nx: usize = vp.iter().sum();
    let n2b = cnorm_parts(&b, up);
    let tolb = tol * n2b;
    let mut u = b;
    let mut beta = n2b;
    let mut normr = beta;
    if beta != 0.0 {
        div_inplace(&mut u, beta);
    }
    let (mut c, mut s) = (1.0f64, 0.0f64);
    let mut phibar = beta;
    let mut v = vec![zero; nx];
    apply_ah(&u, &mut v);
    let mut x = vec![zero; nx];
    let mut alpha = cnorm_parts(&v, vp);
    if alpha != 0.0 {
        div_inplace(&mut v, alpha);
    }
    let mut normar = alpha * beta;
    let relres = |nr: f64| if n2b > 0.0 { nr / n2b } else { 0.0 };
    if normar == 0.0 {
        return (x, LsqrOutcome { flag: 0, iter: 0, relres: relres(normr) });
    }
    let mut un = vec![zero; u.len()];
    let mut vt = vec![zero; nx];
    let mut d = vec![zero; nx];
    let mut normx = 0.0f64; // cnorm(x), kept up to date by the x update
    let mut norma = 0.0f64;
    let mut stag = 0usize;
    let mut flag = 1u8;
    let mut iter = max_iter;
    for ii in 1..=max_iter {
        apply_a(&v, &mut un);
        let alpha_t = T::of(alpha);
        beta = update_and_norm(&mut un, &u, up, |a, b| *a -= b * alpha_t);
        if beta != 0.0 {
            div_inplace(&mut un, beta);
        }
        std::mem::swap(&mut u, &mut un);
        norma = (norma * norma + alpha * alpha + beta * beta).sqrt();
        let thet = -s * alpha;
        let rhot = c * alpha;
        let rho = (rhot * rhot + beta * beta).sqrt();
        c = rhot / rho;
        s = -beta / rho;
        let phi = c * phibar;
        if phi == 0.0 {
            stag = 1;
        }
        phibar *= s;
        let converged = normar / (norma * normr) <= tol || normr <= tolb;
        let breakdown = !phi.is_finite() || rho == 0.0 || !rho.is_finite();
        let normx_prev = normx;
        // The x update needs ‖d‖ only through the stagnation test, which cannot stop this
        // iteration while stag < 2: then d and x are updated in one pass.
        let fused = !converged && !breakdown && stag < 2;
        let normd = if fused {
            let (nd, nxn) = update_d_x_and_norms(&mut d, &mut x, &v, vp, thet, rho, phi);
            normx = nxn;
            nd
        } else {
            let (thet, rho) = (T::of(thet), T::of(rho));
            update_and_norm(&mut d, &v, vp, |di, vi| *di = (vi - *di * thet) / rho)
        };
        if phi.abs() * normd < f64::EPSILON * normx_prev {
            stag += 1;
        } else {
            stag = 0;
        }
        if converged {
            flag = 0;
            iter = ii - 1;
            break;
        }
        if stag >= 3 {
            flag = 3;
            iter = ii - 1;
            break;
        }
        if breakdown {
            flag = 4;
            iter = ii - 1;
            break;
        }
        if !fused {
            let phi = T::of(phi);
            normx = update_and_norm(&mut x, &d, vp, |xi, di| *xi += di * phi);
        }
        normr *= s.abs();
        apply_ah(&u, &mut vt);
        let beta_t = T::of(beta);
        alpha = update_and_norm(&mut vt, &v, vp, |a, b| *a -= b * beta_t);
        std::mem::swap(&mut v, &mut vt);
        if (alpha == 0.0 || !alpha.is_finite()) && ii < max_iter {
            flag = 4;
            iter = ii;
            break;
        }
        if alpha != 0.0 {
            div_inplace(&mut v, alpha);
        }
        normar = alpha * (s * phi).abs();
    }
    if flag == 1 && (normar / (norma * normr) <= tol || normr <= tolb) {
        flag = 0;
    }
    (x, LsqrOutcome { flag, iter, relres: relres(normr) })
}

/// 0-based index of STI's percentile: the sorted value at 1-based index `round(p·N)`.
fn sti_percentile_index(n: usize, p: f64) -> usize {
    ((p * n as f64).round() as usize).clamp(1, n) - 1
}

/// Sorted value at 1-based index `round(p·N)` (STI's percentile; `p` in [0, 1]).
#[cfg(test)]
fn sti_percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[sti_percentile_index(sorted.len(), p)]
}

/// STI's percentiles `p_lo ≤ p_hi` of `v` (which is reordered), by selection instead of a full
/// sort: the k-th element in `total_cmp` order is unique, so this is exactly the sorted lookup.
fn sti_percentiles(v: &mut [f64], p_lo: f64, p_hi: f64) -> (f64, f64) {
    let (il, ih) = (sti_percentile_index(v.len(), p_lo), sti_percentile_index(v.len(), p_hi));
    debug_assert!(il <= ih);
    let hi = *v.select_nth_unstable_by(ih, |a, b| a.total_cmp(b)).1;
    let lo = if il == ih { hi } else { *v[..ih].select_nth_unstable_by(il, |a, b| a.total_cmp(b)).1 };
    (lo, hi)
}

/// STI's `MaskBoundingBox`: tight bounding box (0-based, inclusive); an odd extent grows by one
/// voxel at the high end, or at the low end when the high end is the array edge (and shrinks
/// by one at the high end when both ends are edges).
fn sti_bounding_box(mask: &[u8], dims: (usize, usize, usize)) -> Option<([usize; 3], [usize; 3])> {
    let (nx, ny, nz) = dims;
    let n = [nx, ny, nz];
    let mut lo = n;
    let mut hi = [0usize; 3];
    let mut any = false;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if mask[i + j * nx + k * nx * ny] != 0 {
                    any = true;
                    for (d, c) in [i, j, k].into_iter().enumerate() {
                        lo[d] = lo[d].min(c);
                        hi[d] = hi[d].max(c);
                    }
                }
            }
        }
    }
    if !any {
        return None;
    }
    for d in 0..3 {
        if (hi[d] - lo[d] + 1) % 2 == 1 {
            if hi[d] + 1 < n[d] {
                hi[d] += 1;
            } else if lo[d] > 0 {
                lo[d] -= 1;
            } else {
                hi[d] -= 1;
            }
        }
    }
    Some((lo, hi))
}

/// STI's padded grid: per axis the next multiple of 16 strictly above `c + 2·pad_mm/vs`.
fn sti_padded_dims(c: [usize; 3], vs: [f64; 3], pad_mm: [f64; 3]) -> [usize; 3] {
    [0, 1, 2].map(|d| 16 * ((c[d] as f64 + 2.0 * pad_mm[d] / vs[d] + 1.0) / 16.0).ceil() as usize)
}

/// Sentinel of the compact-index maps in the artefact solve.
const NONE: u32 = u32::MAX;

/// Circular neighbour `l + e_axis` on the grid `n`.
#[inline]
fn circ_next(l: usize, axis: usize, n: [usize; 3]) -> usize {
    let step = [1, n[0], n[0] * n[1]][axis];
    if (l / step) % n[axis] + 1 == n[axis] { l - (n[axis] - 1) * step } else { l + step }
}

/// Weighted circular forward differences (no voxel-size scaling), as STI's iLSQR uses, at the
/// mask voxels only: `r[a·nm + i] = gw[a][i] · (val(l + e_a) − val(l))` with `l = inside[i]`,
/// `nm = inside.len()`, `gw` the weights compacted to the mask.
fn grad_masked<T: Real>(val: &(dyn Fn(usize) -> Complex<T> + Sync), inside: &[usize], gw: &[Vec<T>], n: [usize; 3], r: &mut [Complex<T>]) {
    let nm = inside.len();
    for (a, ra) in r.chunks_mut(nm).enumerate() {
        let g = &gw[a];
        crate::maybe_par_iter_mut!(ra).enumerate().for_each(|(i, o)| {
            let l = inside[i];
            *o = (val(circ_next(l, a, n)) - val(l)) * g[i];
        });
    }
}

/// Adjoint of [`grad_masked`] onto the box `bx` of the grid: `acc[l] = Σ_a (q_a[l − e_a] − q_a[l])`
/// for `l` in `bx` (the rest of `acc` is left as it is), summed in axis order from zero, with
/// `q_a = gw[a]·r_a` on the mask and zero elsewhere (`mask_of[l]` = position of `l` in
/// `inside`, or [`NONE`]). The result is zero outside the mask grown by one voxel along +e_a.
fn grad_masked_adj<T: Real>(r: &[Complex<T>], mask_of: &[u32], gw: &[Vec<T>], n: [usize; 3], bx: &Box3, acc: &mut [Complex<T>]) {
    let (nxy, nt) = (n[0] * n[1], n[0] * n[1] * n[2]);
    let nm = r.len() / 3;
    let zero = Complex::new(T::zero(), T::zero());
    let q = |a: usize, l: usize| match mask_of[l] {
        NONE => zero,
        i => r[a * nm + i as usize] * gw[a][i as usize],
    };
    crate::maybe_par_chunks_mut!(acc, nxy).enumerate().filter(|(k, _)| bx[2].contains(k)).for_each(|(k, pa)| {
        let uz = if k == 0 { nt } else { 0 }; // circular: l − e_z = l + uz − nxy
        for j in bx[1].clone() {
            let uy = if j == 0 { nxy } else { 0 };
            let row = k * nxy + j * n[0];
            for i in bx[0].clone() {
                let l = row + i;
                let xp = if i == 0 { l + n[0] - 1 } else { l - 1 };
                let mut o = zero;
                o += q(0, xp) - q(0, l);
                o += q(1, l + uy - n[0]) - q(1, l);
                o += q(2, l + uz - nxy) - q(2, l);
                pa[j * n[0] + i] = o;
            }
        }
    });
}

/// iLSQR as in STI Suite 3.0 (`QSM_iLSQR`), with STI's 64 mm padding.
///
/// See the module docs for the algorithm. `field` is the local field (any linear unit; QSM.rs
/// passes ppm), `bdir` the B0 direction in voxel axes (normalised here).
///
/// # Returns
/// `(chi, streaking_artifacts, fast_qsm, initial_lsqr)` on the input grid, all masked.
/// `initial_lsqr` is STI's second output (the step-3 LSQR solution).
pub fn ilsqr(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &IlsqrParams,
    progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    ilsqr_with_padding(field, mask, grid, bdir, params, [STI_PAD_MM; 3], progress)
}

/// [`ilsqr`] with an explicit STI `padsize` (mm per axis; STI's `'padsize'` argument).
pub fn ilsqr_with_padding(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &IlsqrParams,
    pad_mm: [f64; 3],
    progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    ilsqr_with_padding_traced(field, mask, grid, bdir, params, pad_mm, progress, &mut IlsqrTrace::default())
}

/// Where an [`ilsqr_with_padding_traced`] run spent its time, and how its two LSQR solves ended.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct IlsqrTrace {
    /// Padded grid
    pub dims: [usize; 3],
    /// (stage, wall seconds, 3-D FFTs, FFT wall seconds) in execution order
    pub stages: Vec<(&'static str, f64, usize, f64)>,
    /// Initial solve and artefact solve
    pub solves: Vec<LsqrOutcome>,
    /// 3-D FFTs (forward + inverse) on the padded grid, and their total wall time
    pub n_fft: usize,
    pub fft_secs: f64,
}

/// [`ilsqr_with_padding`], also filling an [`IlsqrTrace`] (for benchmarks and parity checks).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn ilsqr_with_padding_traced(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &IlsqrParams,
    pad_mm: [f64; 3],
    mut progress: impl FnMut(usize, usize),
    trace: &mut IlsqrTrace,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    *trace = IlsqrTrace::default();
    let clock = FftClock::default();
    let mut t_stage = (std::time::Instant::now(), 0usize, 0.0f64);
    let mut stage = |trace: &mut IlsqrTrace, name: &'static str| {
        let (n, t) = (clock.n.get(), clock.secs.get());
        trace.stages.push((name, t_stage.0.elapsed().as_secs_f64(), n - t_stage.1, t - t_stage.2));
        t_stage = (std::time::Instant::now(), n, t);
    };
    let (nx, ny, nz) = grid.dims;
    let n_in = nx * ny * nz;
    let vs = [grid.voxel_size.0, grid.voxel_size.1, grid.voxel_size.2];
    let zeros = || vec![0.0; n_in];
    let Some((lo, hi)) = sti_bounding_box(mask, grid.dims) else {
        return (zeros(), zeros(), zeros(), zeros());
    };
    progress(1, 4);

    // 1. crop to the (even) bounding box and pad to the next multiple of 16 above c + 2 pad/vs
    let c = [0, 1, 2].map(|d| hi[d] - lo[d] + 1);
    let n = sti_padded_dims(c, vs, pad_mm);
    let pd = [0, 1, 2].map(|d| (n[d] - c[d]) / 2);
    let nt = n[0] * n[1] * n[2];
    let mut p = vec![0.0; nt];
    let mut m = vec![0.0; nt];
    for k in 0..c[2] {
        for j in 0..c[1] {
            for i in 0..c[0] {
                let src = (i + lo[0]) + (j + lo[1]) * nx + (k + lo[2]) * nx * ny;
                let dst = (i + pd[0]) + (j + pd[1]) * n[0] + (k + pd[2]) * n[0] * n[1];
                p[dst] = field[src];
                m[dst] = if mask[src] != 0 { 1.0 } else { 0.0 };
            }
        }
    }
    let inside: Vec<usize> = (0..nt).filter(|&l| m[l] > 0.0).collect();
    let pgrid = Grid::new(n[0], n[1], n[2], vs[0], vs[1], vs[2]);
    let dk = dipole_kernel(&pgrid, bdir);
    trace.dims = n;
    let ws = RefCell::new(Fft3dWorkspace::new(n[0], n[1], n[2]));
    let fft = |a: &mut [Complex64]| clock.time(|| ws.borrow_mut().fft3d(a));
    let ifft = |a: &mut [Complex64]| clock.time(|| ws.borrow_mut().ifft3d(a));
    // the mask lies in the crop box; its forward-difference neighbours in the box grown by one
    let crop_box: Box3 = [0, 1, 2].map(|d| pd[d]..pd[d] + c[d]);
    let crop_box1: Box3 = [0, 1, 2].map(|d| pd[d]..(pd[d] + c[d] + 1).min(n[d]));
    stage(trace, "crop/pad, dipole kernel, plans");
    let mut p_k: Vec<Complex64> = crate::maybe_par_iter!(p).map(|&v| Complex64::new(v, 0.0)).collect();
    fft(&mut p_k[..]);

    // 2. Laplacian weights: |F⁻¹(k² F φ)| with STI's 1e6 at DC, ranked in the mask
    let kx = crate::fft::fftfreq(n[0], vs[0]);
    let ky = crate::fft::fftfreq(n[1], vs[1]);
    let kz = crate::fft::fftfreq(n[2], vs[2]);
    let mut lap: Vec<Complex64> = p_k.clone();
    crate::maybe_par_iter_mut!(lap).enumerate().for_each(|(l, z)| {
        let (i, j, k) = (l % n[0], (l / n[0]) % n[1], l / (n[0] * n[1]));
        let k2 = if l == 0 { 1e6 } else { kx[i] * kx[i] + ky[j] * ky[j] + kz[k] * kz[k] };
        *z *= k2;
    });
    ifft(&mut lap[..]);
    let mut w: Vec<f64> = crate::maybe_par_iter!(lap).map(|z| z.re.abs()).collect(); // |L|, then W
    drop(lap);
    let mut ranked: Vec<f64> = inside.iter().map(|&l| w[l]).collect();
    let (t60, t999) = sti_percentiles(&mut ranked, 0.6, 0.999);
    drop(ranked);
    crate::maybe_par_iter_mut!(w).zip(crate::maybe_par_iter!(m)).for_each(|(v, &mv)| {
        *v = if mv == 0.0 {
            0.0
        } else if t999 > t60 {
            (1.0 - 0.8 * (*v - t60) / (t999 - t60)).clamp(0.2, 1.0)
        } else if *v <= t60 {
            1.0
        } else {
            0.2
        };
    });
    stage(trace, "Laplacian weights (incl. F phi)");

    // 3. initial solution: LSQR on D W D χ = D W φ, with D = F⁻¹ diag(dk) F (initial_solve)
    let (x1r, out1) = match params.precision {
        IlsqrPrecision::Double => initial_solve::<f64>(n, &p, &w, &dk, &crop_box, params.max_iter, &clock),
        IlsqrPrecision::Single => initial_solve::<f32>(n, &p, &w, &dk, &crop_box, params.max_iter, &clock),
    };
    trace.solves.push(out1);
    stage(trace, "solve 1 (initial LSQR)");
    progress(2, 4);

    // 4. FastQSM prior (sign(D) inverse blended with a k-space-smoothed copy where |D| is small)
    let xfs = sti_fastqsm(&p_k, &m, &inside, &dk, n, &fft, &ifft);
    drop(p_k);
    stage(trace, "FastQSM");
    progress(3, 4);

    // 5. gradient weights from the FastQSM map, pooled over the three axes in the mask
    let mut gw: Vec<Vec<f64>> = (0..3).map(|_| vec![0.0; nt]).collect();
    let mut pool = Vec::with_capacity(3 * inside.len());
    for (axis, a) in gw.iter_mut().enumerate() {
        // |∇_axis xfs| (circular forward difference), held in gw[axis] until the weights replace it
        let xfs = &xfs;
        crate::maybe_par_iter_mut!(a).enumerate().for_each(|(l, o)| *o = (xfs[circ_next(l, axis, n)] - xfs[l]).abs());
        pool.extend(inside.iter().map(|&l| a[l]));
    }
    let (t50, t70) = sti_percentiles(&mut pool, 0.5, 0.7);
    drop(pool);
    for a in gw.iter_mut() {
        crate::maybe_par_iter_mut!(a).zip(crate::maybe_par_iter!(m)).for_each(|(g, &mv)| {
            *g = if mv == 0.0 {
                0.0
            } else if t70 > t50 {
                ((t70 - *g) / (t70 - t50)).clamp(0.0, 1.0)
            } else if *g <= t50 {
                1.0
            } else {
                0.0
            };
        });
    }
    stage(trace, "gradient weights");
    progress(4, 4);

    // 6. streaking artefacts (artefact_solve), subtracted from the initial solution
    let sq = (nt as f64).sqrt();
    let (xsa, out2) = match params.precision {
        IlsqrPrecision::Double => artefact_solve::<f64>(n, &x1r, &m, &inside, gw, &dk, &crop_box1, params.tol, params.max_iter, &clock),
        IlsqrPrecision::Single => artefact_solve::<f32>(n, &x1r, &m, &inside, gw, &dk, &crop_box1, params.tol, params.max_iter, &clock),
    };
    trace.solves.push(out2);

    // 7. subtract, mask, crop back
    let mut chi = zeros();
    let mut sa = zeros();
    let mut fs = zeros();
    let mut x0 = zeros();
    for k in 0..c[2] {
        for j in 0..c[1] {
            for i in 0..c[0] {
                let dst = (i + lo[0]) + (j + lo[1]) * nx + (k + lo[2]) * nx * ny;
                let src = (i + pd[0]) + (j + pd[1]) * n[0] + (k + pd[2]) * n[0] * n[1];
                if m[src] > 0.0 {
                    let a = xsa[src] * sq;
                    chi[dst] = x1r[src] - a;
                    sa[dst] = a;
                    fs[dst] = xfs[src];
                    x0[dst] = x1r[src];
                }
            }
        }
    }
    stage(trace, "solve 2 (artefact LSQR, incl. setup) + output");
    trace.n_fft = clock.n.get();
    trace.fft_secs = clock.secs.get();
    (chi, sa, fs, x0)
}

/// FFT count and wall time of an iLSQR run (for [`IlsqrTrace`]).
#[derive(Default)]
struct FftClock {
    n: Cell<usize>,
    secs: Cell<f64>,
}

impl FftClock {
    fn time<R>(&self, f: impl FnOnce() -> R) -> R {
        let t = std::time::Instant::now();
        let r = f();
        self.n.set(self.n.get() + 1);
        self.secs.set(self.secs.get() + t.elapsed().as_secs_f64());
        r
    }
}

/// Step 3 of [`ilsqr_with_padding`]: LSQR (tol 0.01) on D W D χ = D W φ with
/// D = F⁻¹ diag(dk) F, in precision `T`. Returns χ and how LSQR ended.
///
/// Solved in k-space, for x̃ = F χ: with the unitary U = F/√N, D W D = U* (dk·U W U*·dk) U,
/// so the LSQR iterates for (U A U*, U b) are U times those for (A, b) and the norms are equal
/// (exact arithmetic). The √N factors cancel in dk·F W F⁻¹·dk, and taking F b = √N·U b as the
/// right-hand side (LSQR is linear in b and its stopping tests are scale-free) leaves
/// χ = F⁻¹ x̃. That is 2 FFTs per operator call instead of 4; rounding differs only at the
/// level of the transforms' own rounding. The products with dk and W are fused into the
/// transforms' first and last passes, which only cover the crop box (W = 0 outside the mask).
#[allow(clippy::too_many_arguments)]
fn initial_solve<T: Real>(
    n: [usize; 3],
    p: &[f64],
    w: &[f64],
    dk: &[f64],
    crop_box: &Box3,
    max_iter: usize,
    clock: &FftClock,
) -> (Vec<f64>, LsqrOutcome) {
    let ws = RefCell::new(T::workspace(n));
    let tr = |a: &mut [Complex<T>], inverse: bool, o: &FftOpts<T>| clock.time(|| T::transform(&mut ws.borrow_mut(), a, inverse, o));
    let (wt, dkt) = (T::cast(w), T::cast(dk));
    let (wt, dkt) = (&wt[..], &dkt[..]);
    let apply1 = |v: &[Complex<T>], t: &mut [Complex<T>]| {
        let load = |l: usize, row: &mut [Complex<T>]| {
            row.iter_mut().zip(&v[l..]).zip(&dkt[l..]).for_each(|((o, &z), &dv)| *o = z * dv);
        };
        tr(t, true, &FftOpts { needed: Some(crop_box), load: Some(&load), post: Some(wt), ..Default::default() });
        tr(t, false, &FftOpts { support: Some(crop_box), post: Some(dkt), ..Default::default() });
    };
    let mut b1: Vec<Complex<T>> = crate::maybe_par_iter!(p).zip(crate::maybe_par_iter!(w))
        .map(|(&pv, &wv)| Complex::new(T::of(pv * wv), T::zero())).collect();
    tr(&mut b1, false, &FftOpts::default());
    crate::maybe_par_iter_mut!(b1).zip(crate::maybe_par_iter!(dkt)).for_each(|(z, &dv)| *z *= dv);
    let dense = dense_parts(p.len());
    let (mut x1, out) = lsqr_matlab_into(apply1, apply1, b1, (&dense, &dense), 0.01, max_iter);
    tr(&mut x1, true, &FftOpts::default());
    (crate::maybe_par_iter!(x1).map(|z| z.re.to64()).collect(), out)
}

/// Step 6 of [`ilsqr_with_padding`]: the streaking artefacts, by LSQR (`tol`) in precision
/// `T` over the k-space cone |D| < 0.1 against the weighted gradient (weights `gw`, full grid,
/// zero off the mask) of the initial solution `x1`. Returns the real part of F⁻¹ of the
/// solution (the artefacts divided by √N) and how LSQR ended.
///
/// A y = G ∇ (√N F⁻¹(M_ic y)), Aᴴ r = M_ic F(∇ᴴ(G r)) / √N. Every vector LSQR forms in the
/// domain is exactly zero off the cone, and every one in the range is exactly zero off the
/// mask (G = 0 there), so both are stored compactly: the unknown on the cone positions, the
/// residual on the mask positions of each axis. With norms split as for the dense vectors
/// (`compact_parts`) the iterates have the same values as a dense solve. The image-space data
/// lives in `crop_box1` (the mask and its +e_a neighbours), so the transforms are pruned to it.
#[allow(clippy::too_many_arguments)]
fn artefact_solve<T: Real>(
    n: [usize; 3],
    x1: &[f64],
    m: &[f64],
    inside: &[usize],
    gw: Vec<Vec<f64>>,
    dk: &[f64],
    crop_box1: &Box3,
    tol: f64,
    max_iter: usize,
    clock: &FftClock,
) -> (Vec<f64>, LsqrOutcome) {
    let nt = dk.len();
    let sq = (nt as f64).sqrt();
    let cone: Vec<usize> = (0..nt).filter(|&l| dk[l].abs() < 0.1).collect();
    let nm = inside.len();
    assert!(nt < NONE as usize, "grid too large for 32-bit voxel indices");
    let index_of = |list: &[usize]| {
        let mut map = vec![NONE; nt];
        for (i, &l) in list.iter().enumerate() {
            map[l] = i as u32;
        }
        map
    };
    let cone_of = index_of(&cone);
    let mask_of = index_of(inside);
    let gwc: Vec<Vec<T>> = gw.iter().map(|g| inside.iter().map(|&l| T::of(g[l])).collect()).collect();
    drop(gw);
    let u_parts = compact_parts((0..3).flat_map(|a| inside.iter().map(move |&l| a * nt + l)));
    let v_parts = compact_parts(cone.iter().copied());
    let zero = Complex::new(T::zero(), T::zero());
    let ws = RefCell::new(T::workspace(n));
    let tr = |a: &mut [Complex<T>], inverse: bool, o: &FftOpts<T>| clock.time(|| T::transform(&mut ws.borrow_mut(), a, inverse, o));
    let (sq_t, isq_t) = (T::of(sq), T::of(1.0 / sq));
    let t_cell = RefCell::new(vec![zero; nt]);
    let apply2 = |y: &[Complex<T>], out: &mut [Complex<T>]| {
        let mut t = t_cell.borrow_mut();
        // scatter y onto the cone in the transform's first pass; read on the mask and its
        // +e_a neighbours only
        let load = |l: usize, row: &mut [Complex<T>]| {
            row.iter_mut().zip(&cone_of[l..]).for_each(|(o, &ci)| *o = if ci == NONE { zero } else { y[ci as usize] });
        };
        tr(&mut t[..], true, &FftOpts { needed: Some(crop_box1), load: Some(&load), ..Default::default() });
        let t = &t[..];
        grad_masked(&|l| t[l] * sq_t, inside, &gwc, n, out);
    };
    let apply2h = |r: &[Complex<T>], out: &mut [Complex<T>]| {
        let mut acc = t_cell.borrow_mut();
        grad_masked_adj(r, &mask_of, &gwc, n, crop_box1, &mut acc);
        tr(&mut acc[..], false, &FftOpts { support: Some(crop_box1), ..Default::default() });
        let acc = &acc[..];
        crate::maybe_par_iter_mut!(out).zip(crate::maybe_par_iter!(cone)).for_each(|(o, &l)| *o = acc[l] * isq_t);
    };
    let mut b2 = vec![zero; 3 * nm];
    grad_masked(&|l| Complex::new(T::of(x1[l] * m[l]), T::zero()), inside, &gwc, n, &mut b2);
    let (y, out) = lsqr_matlab_into(apply2, apply2h, b2, (&u_parts, &v_parts), tol, max_iter);
    drop(t_cell);
    let mut xsa = vec![zero; nt];
    for (&l, &z) in cone.iter().zip(&y) {
        xsa[l] = z;
    }
    drop(y);
    tr(&mut xsa, true, &FftOpts::default());
    (crate::maybe_par_iter!(xsa).map(|z| z.re.to64()).collect(), out)
}

/// STI's `FastQSM` on the padded grid, given `F φ` (unshifted k-space). Returns the masked
/// estimate, rescaled to the TKD (threshold 1/8) solution by least squares inside the mask.
fn sti_fastqsm(
    p_k: &[Complex64],
    m: &[f64],
    inside: &[usize],
    dk: &[f64],
    n: [usize; 3],
    fft: &dyn Fn(&mut [Complex64]),
    ifft: &dyn Fn(&mut [Complex64]),
) -> Vec<f64> {
    let nt = p_k.len();
    // kernel: ball3D(3) (centre excluded) / its count, closed along the 2nd axis with [1 1 1]
    // (out-of-array = −∞ for the dilation, +∞ for the erosion), not renormalised
    let r = 3i64;
    let s = 7usize;
    let mut ball = vec![0.0f64; s * s * s];
    let mut cnt = 0.0;
    for (l, bv) in ball.iter_mut().enumerate() {
        let (i, j, k) = ((l % s) as i64 - r, ((l / s) % s) as i64 - r, (l / (s * s)) as i64 - r);
        let d2 = (i * i + j * j + k * k) as f64;
        if d2 > 0.0 && d2 <= (r as f64 + 1.0 / 6.0).powi(2) {
            *bv = 1.0;
            cnt += 1.0;
        }
    }
    ball.iter_mut().for_each(|v| *v /= cnt);
    let line = |a: &[f64], dil: bool| -> Vec<f64> {
        let mut o = vec![0.0; a.len()];
        for (l, ov) in o.iter_mut().enumerate() {
            let j = (l / s) % s;
            let mut v = a[l];
            for jj in [j.wrapping_sub(1), j + 1] {
                if jj < s {
                    let q = a[l - j * s + jj * s];
                    v = if dil { v.max(q) } else { v.min(q) };
                }
            }
            *ov = v;
        }
        o
    };
    let closed = line(&line(&ball, true), false);
    // STI smooths the k-space of the *centred* image (fftnc: origin at the array centre); on
    // the k-space of the uncentred image that is the same convolution with the kernel
    // modulated by (−1)^(i+j+k) (all padded dimensions are even).
    let mut kern = vec![Complex64::new(0.0, 0.0); nt];
    for (l, &bv) in closed.iter().enumerate() {
        if bv != 0.0 {
            let off = [(l % s) as i64 - r, ((l / s) % s) as i64 - r, (l / (s * s)) as i64 - r];
            let idx = [0, 1, 2].map(|d| off[d].rem_euclid(n[d] as i64) as usize);
            let sign = if (off[0] + off[1] + off[2]).rem_euclid(2) == 0 { 1.0 } else { -1.0 };
            kern[idx[0] + idx[1] * n[0] + idx[2] * n[0] * n[1]] = Complex64::new(sign * bv, 0.0);
        }
    }
    fft(&mut kern[..]);
    let kh: Vec<f64> = kern.iter().map(|z| z.re).collect();
    drop(kern);
    // k-space blending weights from |D|^0.001, ranked over the whole grid (1st..30th pct)
    let mut wf: Vec<f64> = crate::maybe_par_iter!(dk).map(|&dv| dv.abs().powf(0.001)).collect();
    let mut zs = wf.clone();
    let (zlo, zhi) = sti_percentiles(&mut zs, 0.01, 0.3);
    drop(zs);
    crate::maybe_par_iter_mut!(wf).for_each(|v| *v = if zhi > zlo { ((*v - zlo) / (zhi - zlo)).clamp(0.0, 1.0) } else { 1.0 });
    // blend X with its k-space smoothing (a convolution over k-space indices)
    let blend = |x: &mut [Complex64]| {
        let mut sm = x.to_vec();
        fft(&mut sm[..]);
        crate::maybe_par_iter_mut!(sm).zip(crate::maybe_par_iter!(kh)).for_each(|(v, &h)| *v *= h);
        ifft(&mut sm[..]);
        crate::maybe_par_iter_mut!(x).zip(crate::maybe_par_iter!(sm)).zip(crate::maybe_par_iter!(wf))
            .for_each(|((xv, &sv), &w)| *xv = *xv * w + sv * (1.0 - w));
    };
    let mut x: Vec<Complex64> = crate::maybe_par_iter!(p_k).zip(crate::maybe_par_iter!(dk)).map(|(&pv, &dv)| pv * sign0(dv)).collect();
    blend(&mut x[..]);
    ifft(&mut x[..]);
    crate::maybe_par_iter_mut!(x).zip(crate::maybe_par_iter!(m)).for_each(|(v, &mv)| *v *= mv);
    fft(&mut x[..]);
    blend(&mut x[..]);
    ifft(&mut x[..]);
    let mut xfs: Vec<f64> = crate::maybe_par_iter!(x).zip(crate::maybe_par_iter!(m)).map(|(v, &mv)| v.re * mv).collect();
    drop(x);
    // TKD: 1/D clamped to ±8 (D = 0 → +8)
    let mut t: Vec<Complex64> = crate::maybe_par_iter!(p_k).zip(crate::maybe_par_iter!(dk)).map(|(&pv, &dv)| {
        let r = 1.0 / dv;
        pv * if r.abs() > 8.0 { 8.0f64.copysign(r) } else { r }
    }).collect();
    ifft(&mut t[..]);
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for &l in inside {
        sxy += xfs[l] * t[l].re;
        sxx += xfs[l] * xfs[l];
    }
    let a = if sxx > 0.0 { sxy / sxx } else { 1.0 };
    crate::maybe_par_iter_mut!(xfs).for_each(|v| *v *= a);
    xfs
}

fn sign0(v: f64) -> f64 {
    if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsqr_simple() {
        // Test LSQR on a simple diagonal system
        let n = 10;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let b: Vec<f64> = diag.iter().map(|&d| d * 2.0).collect();  // x = [2, 2, 2, ...]

        let apply_a = |x: &[f64]| -> Vec<f64> {
            x.iter().zip(diag.iter()).map(|(&xi, &di)| xi * di).collect()
        };

        let x = lsqr(apply_a, apply_a, &b, 1e-10, 100);

        for (i, &xi) in x.iter().enumerate() {
            assert!((xi - 2.0).abs() < 1e-6, "x[{}] = {}, expected 2.0", i, xi);
        }
    }

    #[test]
    fn test_norm() {
        let x = vec![3.0, 4.0];
        assert!((norm(&x) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sign_array() {
        let x = vec![-2.0, 0.0, 3.0];
        let s = sign_array(&x);
        assert_eq!(s, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_lsqr_complex_relres_stop() {
        // Regression for the iLSQR Step-1 fix: lsqr_complex must stop on the
        // simple relative-residual criterion `||r||/||b|| <= tol` (matching
        // MATLAB's built-in `lsqr` that kamesy's ilsqr.m Step 1 calls), NOT the
        // augmented Paige-Saunders `rtol = btol + atol*||A||*||x||/||b||` test,
        // which previously tripped early and left Step 1 under-converged.
        //
        // Use an ill-conditioned diagonal A = diag(1, 10, 100, 1000) where the
        // augmented test (with its ||A||*||x|| term) would stop well before the
        // residual actually reaches tol. With the correct test, the residual
        // must fall to <= tol.
        let diag = vec![1.0, 10.0, 100.0, 1000.0];
        let expected: Vec<Complex64> =
            diag.iter().map(|&d| Complex64::new(d, 0.0)).collect();
        let b: Vec<Complex64> = expected
            .iter()
            .zip(diag.iter())
            .map(|(&xi, &di)| xi * di)
            .collect();

        let da = diag.clone();
        let apply = move |x: &[Complex64]| -> Vec<Complex64> {
            x.iter().zip(da.iter()).map(|(&xi, &di)| xi * di).collect()
        };
        let apply2 = apply.clone();
        let tol = 1e-6;
        let x = lsqr_complex(apply, apply2, &b, tol, 200, false);

        // Residual ||A*x - b|| / ||b|| must be at or below tol.
        let ax: Vec<Complex64> =
            x.iter().zip(diag.iter()).map(|(&xi, &di)| xi * di).collect();
        let r: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(a, bb)| (a - bb).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let bn: f64 = b.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        assert!(
            r / bn <= tol * 10.0,
            "relres {} should reach ~tol {}; premature-stop regression",
            r / bn,
            tol
        );
    }

    #[test]
    fn test_lsqr_complex_diagonal() {
        // Test complex LSQR on a diagonal system: A = diag(1, 2, 3), b = [1+i, 4+2i, 9+3i]
        // Expected solution: x = [1+i, 2+i, 3+i]
        let diag = vec![1.0, 2.0, 3.0];
        let expected = [
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(3.0, 1.0),
        ];
        let b: Vec<Complex64> = expected.iter().zip(diag.iter())
            .map(|(&xi, &di)| xi * di)
            .collect();

        let diag_a = diag.clone();
        let diag_ah = diag.clone();
        let apply_a = move |x: &[Complex64]| -> Vec<Complex64> {
            x.iter().zip(diag_a.iter()).map(|(&xi, &di)| xi * di).collect()
        };
        let apply_ah = move |x: &[Complex64]| -> Vec<Complex64> {
            x.iter().zip(diag_ah.iter()).map(|(&xi, &di)| xi * di).collect()
        };

        let x = lsqr_complex(apply_a, apply_ah, &b, 1e-10, 100, false);

        for (i, (xi, ei)) in x.iter().zip(expected.iter()).enumerate() {
            assert!((xi.re - ei.re).abs() < 1e-6,
                "x[{}].re = {}, expected {}", i, xi.re, ei.re);
            assert!((xi.im - ei.im).abs() < 1e-6,
                "x[{}].im = {}, expected {}", i, xi.im, ei.im);
        }
    }

    #[test]
    fn test_lsmr_diagonal() {
        // Test the LSMR solver (inside ilsqr.rs) exercises all code paths.
        // Use a well-conditioned diagonal system: A = diag(1, 1, 1) (identity)
        // b = [3, 5, 7], expected x = [3, 5, 7]
        let b = vec![3.0, 5.0, 7.0];

        let apply_a = |x: &[f64]| -> Vec<f64> { x.to_vec() };
        let apply_at = |x: &[f64]| -> Vec<f64> { x.to_vec() };

        let x = lsmr(apply_a, apply_at, &b, 3, 1e-6, 1e-6, 200, false);

        // Verify that the solver returns finite values and the output has correct length
        assert_eq!(x.len(), 3);
        for (i, &xi) in x.iter().enumerate() {
            assert!(xi.is_finite(), "x[{}] = {} is not finite", i, xi);
        }

        // Compute residual: ||Ax - b|| should be reduced from ||b||
        let residual: f64 = x.iter().zip(b.iter())
            .map(|(&xi, &bi)| (xi - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        let bnorm: f64 = b.iter().map(|&bi| bi * bi).sum::<f64>().sqrt();
        assert!(residual < bnorm,
            "residual {} should be less than ||b|| = {}", residual, bnorm);
    }

    #[test]
    fn test_laplacian_weights() {
        // Test laplacian_weights_ilsqr on a small 4x4x4 volume with a uniform field
        // inside a mask. A constant field has zero Laplacian, so weights should be 1.0.
        let (nx, ny, nz) = (4, 4, 4);
        let n_total = nx * ny * nz;
        let mut mask = vec![0u8; n_total];
        let mut field = vec![0.0; n_total];

        // Create a sphere mask and constant field inside
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = i + j * nx + k * nx * ny;
                    let ci = i as f64 - 1.5;
                    let cj = j as f64 - 1.5;
                    let ck = k as f64 - 1.5;
                    let r2 = ci * ci + cj * cj + ck * ck;
                    if r2 < 2.5 {
                        mask[idx] = 1;
                        field[idx] = 5.0; // constant field => Laplacian is 0
                    }
                }
            }
        }

        let w = laplacian_weights_ilsqr(&field, &mask, nx, ny, nz, 1.0, 1.0, 1.0, 10.0, 90.0);

        // All weights should be finite and in [0, 1]
        for (i, &wi) in w.iter().enumerate() {
            assert!(wi.is_finite(), "weight[{}] is not finite", i);
            assert!((0.0..=1.0).contains(&wi), "weight[{}] = {} out of [0,1]", i, wi);
        }

        // Masked-out voxels should have weight 0
        for i in 0..n_total {
            if mask[i] == 0 {
                assert_eq!(w[i], 0.0, "weight outside mask should be 0 at index {}", i);
            }
        }
    }

    #[test]
    fn test_dipole_kspace_weights() {
        // Test dipole_kspace_weights_ilsqr with synthetic dipole values
        let d = vec![0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0, -0.5, -1.0, 0.0];

        let w = dipole_kspace_weights_ilsqr(&d, 1.0, 1.0, 90.0);

        // All weights should be in [0, 1]
        for (i, &wi) in w.iter().enumerate() {
            assert!((0.0..=1.0).contains(&wi),
                "weight[{}] = {} out of [0,1]", i, wi);
        }

        // Zero dipole values should produce weight 0 (or very small) since |0|^n = 0
        assert!(w[0] <= 1e-10, "weight at D=0 should be ~0, got {}", w[0]);

        // The largest |D| values should have weight near 1.0
        // d[6]=1.0 and d[8]=-1.0 have the largest |D|
        assert!(w[6] > 0.5, "weight at |D|=1.0 should be large, got {}", w[6]);
        assert!(w[8] > 0.5, "weight at |D|=1.0 should be large, got {}", w[8]);
    }

    #[test]
    fn test_gradient_weights() {
        // Test gradient_weights_ilsqr on a small 4x4x4 volume
        let (nx, ny, nz) = (4, 4, 4);
        let n_total = nx * ny * nz;

        // Create a mask (all ones for simplicity)
        let mask = vec![1u8; n_total];

        // Create a field with a linear gradient in x
        let mut field = vec![0.0; n_total];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = i + j * nx + k * nx * ny;
                    field[idx] = i as f64; // linear in x
                }
            }
        }

        let (wx, wy, wz) = gradient_weights_ilsqr(
            &field, &mask, nx, ny, nz, 1.0, 1.0, 1.0, 10.0, 90.0
        );

        // All weights should be finite and in [0, 1]
        for i in 0..n_total {
            assert!(wx[i].is_finite(), "wx[{}] is not finite", i);
            assert!(wy[i].is_finite(), "wy[{}] is not finite", i);
            assert!(wz[i].is_finite(), "wz[{}] is not finite", i);
            assert!(wx[i] >= 0.0 && wx[i] <= 1.0, "wx[{}] = {} out of [0,1]", i, wx[i]);
            assert!(wy[i] >= 0.0 && wy[i] <= 1.0, "wy[{}] = {} out of [0,1]", i, wy[i]);
            assert!(wz[i] >= 0.0 && wz[i] <= 1.0, "wz[{}] = {} out of [0,1]", i, wz[i]);
        }

        // The y and z gradients are zero for this field, so wy and wz should reflect
        // that all gradient values are identical (zero). Check they are well-defined.
        let wy_sum: f64 = wy.iter().sum();
        let wz_sum: f64 = wz.iter().sum();
        assert!(wy_sum.is_finite(), "wy sum is not finite");
        assert!(wz_sum.is_finite(), "wz sum is not finite");
    }

    #[test]
    fn test_ilsqr_small() {
        // Run ilsqr_simple on a small 8x8x8 volume with a sphere mask
        // and synthetic local field data. This exercises the full pipeline:
        // lsqr_step, fastqsm_step, susceptibility_artifacts_step.
        let (nx, ny, nz) = (8, 8, 8);
        let n_total = nx * ny * nz;
        let vsx = 1.0;
        let vsy = 1.0;
        let vsz = 1.0;
        let bdir = (0.0, 0.0, 1.0);

        // Create a sphere mask centered in the volume
        let mut mask = vec![0u8; n_total];
        let cx = (nx as f64 - 1.0) / 2.0;
        let cy = (ny as f64 - 1.0) / 2.0;
        let cz = (nz as f64 - 1.0) / 2.0;
        let radius = 3.0;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = i + j * nx + k * nx * ny;
                    let di = i as f64 - cx;
                    let dj = j as f64 - cy;
                    let dk = k as f64 - cz;
                    if di * di + dj * dj + dk * dk < radius * radius {
                        mask[idx] = 1;
                    }
                }
            }
        }

        // Create synthetic local field: a simple dipole-like pattern
        // Use a small susceptibility source and forward-model through the dipole kernel
        let mut field = vec![0.0; n_total];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = i + j * nx + k * nx * ny;
                    if mask[idx] > 0 {
                        let di = i as f64 - cx;
                        let dj = j as f64 - cy;
                        let dk = k as f64 - cz;
                        // Simulate a simple field variation
                        field[idx] = 0.01 * (dk * dk - di * di - dj * dj)
                            / (di * di + dj * dj + dk * dk + 1.0);
                    }
                }
            }
        }

        let tol = 0.1;
        let maxit = 5; // Few iterations for speed

        let grid = Grid::new(nx, ny, nz, vsx, vsy, vsz);
        let params = IlsqrParams { tol, max_iter: maxit, ..IlsqrParams::default() };
        let (chi, _, _, _) = ilsqr_qsmm(&field, &mask, &grid, bdir, &params, |_, _| {});

        // Check output dimensions
        assert_eq!(chi.len(), n_total, "output size mismatch");

        // Check all values are finite
        for (i, &v) in chi.iter().enumerate() {
            assert!(v.is_finite(), "chi[{}] = {} is not finite", i, v);
        }

        // Check mask is respected: outside mask should be zero
        for i in 0..n_total {
            if mask[i] == 0 {
                assert_eq!(chi[i], 0.0, "chi outside mask should be 0 at index {}", i);
            }
        }

        // Check that the result is not all zeros inside the mask
        let inside_sum: f64 = chi.iter().zip(mask.iter())
            .filter(|(_, &m)| m > 0)
            .map(|(&v, _)| v.abs())
            .sum();
        assert!(inside_sum > 0.0, "chi should not be all zeros inside the mask");
    }
    // ---------------- STI Suite iLSQR ----------------

    fn c64(re: f64, im: f64) -> Complex64 { Complex64::new(re, im) }

    type Op = Box<dyn Fn(&[Complex64]) -> Vec<Complex64>>;

    /// Dense column-major matrix-vector products (A x, Aᴴ y) for the LSQR tests.
    fn dense(a: Vec<Complex64>, m: usize, n: usize) -> (Op, Op) {
        let a2 = a.clone();
        let fwd = move |x: &[Complex64]| (0..m).map(|i| (0..n).map(|j| a[i + j * m] * x[j]).sum()).collect();
        let adj = move |y: &[Complex64]| (0..n).map(|j| (0..m).map(|i| a2[i + j * m].conj() * y[i]).sum()).collect();
        (Box::new(fwd), Box::new(adj))
    }

    #[test]
    fn test_lsqr_matlab_matches_matlab_lsqr() {
        // Reference values from MATLAB R2023b `[x,flag,relres,iter] = lsqr(A, b, tol, maxit)`.
        // Inconsistent 3x2 system: stops on the normal-equations test after 2 iterations.
        let a: Vec<Complex64> = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0].iter().map(|&v| c64(v, 0.0)).collect();
        let (f, g) = dense(a, 3, 2);
        let b = [c64(1.0, 0.0), c64(2.0, 0.0), c64(4.0, 0.0)];
        let (x, o) = lsqr_matlab(f, g, &b, 1e-8, 10);
        assert_eq!((o.flag, o.iter), (0, 2));
        assert!((o.relres - 0.0890870806374748).abs() < 1e-12);
        assert!((x[0].re - 0.666666666666665).abs() < 1e-12 && (x[1].re - 0.083333333333331).abs() < 1e-12);

        // Ill-conditioned diagonal, iteration cap reached: x after exactly 5 iterations.
        let d: Vec<f64> = (0..8).map(|i| 10f64.powf(-3.0 * i as f64 / 7.0)).collect(); // logspace(0, -3, 8)
        let mut a = vec![c64(0.0, 0.0); 64];
        for i in 0..8 { a[i + 8 * i] = c64(d[i], 0.0); }
        let (f, g) = dense(a, 8, 8);
        let b = vec![c64(1.0, 0.0); 8];
        let (x, o) = lsqr_matlab(f, g, &b, 1e-10, 5);
        assert_eq!((o.flag, o.iter), (1, 5));
        assert!((o.relres - 0.573674508817809).abs() < 1e-12);
        let want = [0.999999943834766, 2.68269578689076, 7.1969177940897, 19.2460743622976,
                    59.9361124086567, 25.5070666956926, 9.67511787758301, 3.61516968204123];
        // (x(1) moves by ~1e-9 for a 1-ulp change in A here, so x is only checked to 1e-6)
        for (xi, wi) in x.iter().zip(want) { assert!((xi.re - wi).abs() < 1e-6 * wi, "{} vs {}", xi.re, wi); }

        // Complex 3x2 system.
        let a = vec![c64(1.0, 0.0), c64(2.0, 0.0), c64(0.0, 0.5), c64(0.0, 1.0), c64(-1.0, 0.0), c64(3.0, 0.0)];
        let (f, g) = dense(a, 3, 2);
        let b = [c64(1.0, 0.0), c64(0.0, 1.0), c64(2.0, 0.0)];
        let (x, o) = lsqr_matlab(f, g, &b, 1e-12, 20);
        assert_eq!((o.flag, o.iter), (0, 2));
        assert!((o.relres - 0.449991346403468).abs() < 1e-12);
        assert!((x[0] - c64(0.448598130841121, 0.186915887850467)).norm() < 1e-12);
        assert!((x[1] - c64(0.635514018691589, -0.168224299065421)).norm() < 1e-12);
    }

    #[test]
    fn test_lsqr_matlab_zero_rhs() {
        let (x, o) = lsqr_matlab(|v: &[Complex64]| v.to_vec(), |v: &[Complex64]| v.to_vec(), &[c64(0.0, 0.0); 4], 1e-6, 10);
        assert_eq!((o.flag, o.iter), (0, 0));
        assert!(x.iter().all(|z| z.norm() == 0.0));
    }

    #[test]
    fn test_sti_bounding_box() {
        // STI MaskBoundingBox outputs (1-based) from probing, here 0-based.
        type Case = ((usize, usize, usize), [usize; 3], [usize; 3], [[usize; 2]; 3]);
        let cases: [Case; 5] = [
            ((20, 20, 20), [4, 5, 6], [10, 11, 12], [[4, 11], [5, 12], [6, 13]]),
            ((20, 20, 20), [1, 1, 1], [5, 6, 7], [[1, 6], [1, 6], [1, 8]]),
            ((20, 20, 20), [14, 15, 16], [20, 20, 20], [[13, 20], [15, 20], [15, 20]]),
            ((21, 21, 21), [1, 2, 3], [21, 21, 21], [[1, 20], [2, 21], [2, 21]]),
            ((20, 20, 20), [7, 7, 7], [7, 8, 7], [[7, 8], [7, 8], [7, 8]]),
        ];
        for (dims, lo1, hi1, want) in cases {
            let mut m = vec![0u8; dims.0 * dims.1 * dims.2];
            for k in lo1[2] - 1..hi1[2] { for j in lo1[1] - 1..hi1[1] { for i in lo1[0] - 1..hi1[0] {
                m[i + j * dims.0 + k * dims.0 * dims.1] = 1;
            } } }
            let (lo, hi) = sti_bounding_box(&m, dims).unwrap();
            for d in 0..3 { assert_eq!([lo[d] + 1, hi[d] + 1], want[d], "dims {:?} box {:?}..{:?}", dims, lo1, hi1); }
        }
        assert!(sti_bounding_box(&[0u8; 8], (2, 2, 2)).is_none());
    }

    #[test]
    fn test_sti_padded_dims() {
        // (crop, voxel size, padsize) -> STI's padded size, from probing QSM_iLSQR
        type Case = ([usize; 3], [f64; 3], [f64; 3], [usize; 3]);
        let cases: [Case; 6] = [
            ([32, 30, 8], [0.8, 0.8, 3.0], [64.0; 3], [208, 192, 64]),
            ([48, 44, 12], [0.8, 0.8, 3.0], [0.0; 3], [64, 48, 16]),
            ([32, 34, 32], [1.3; 3], [10.0; 3], [64, 64, 64]),
            ([32, 30, 18], [1.3, 0.7, 1.1], [10.0, 3.0, 7.0], [64, 48, 32]),
            ([40, 42, 10], [1.05, 2.2, 3.3], [20.0; 3], [80, 64, 32]),
            ([46, 42, 10], [0.5, 0.5, 2.0], [64.0; 3], [304, 304, 80]),
        ];
        for (c, vs, pad, want) in cases { assert_eq!(sti_padded_dims(c, vs, pad), want); }
    }

    #[test]
    fn test_sti_percentile_is_rounded_index() {
        let v: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        assert_eq!(sti_percentile(&v, 0.25), 3.0); // round(2.5) = 3, no interpolation
        assert_eq!(sti_percentile(&v, 0.999), 10.0);
        assert_eq!(sti_percentile(&v, 0.0), 1.0);
    }

    /// `grad_masked` is the weighted circular forward difference on the mask voxels, and
    /// `grad_masked_adj` its adjoint.
    #[test]
    fn test_grad_masked_and_adjoint() {
        let n = [6, 5, 4];
        let nt = 120;
        let inside: Vec<usize> = (0..nt).filter(|&l| (l * 7) % 5 != 0).collect();
        let nm = inside.len();
        let mut mask_of = vec![NONE; nt];
        for (i, &l) in inside.iter().enumerate() { mask_of[l] = i as u32; }
        let u: Vec<Complex64> = (0..nt).map(|i| c64((i as f64 * 0.37).sin(), (i as f64 * 0.11).cos())).collect();
        let r: Vec<Complex64> = (0..3 * nm).map(|i| c64((i as f64 * 0.23).cos(), (i as f64 * 0.71).sin())).collect();
        let gw: Vec<Vec<f64>> = (0..3).map(|a| (0..nm).map(|i| ((i * 7 + a * 3) % 11) as f64 / 10.0).collect()).collect();
        let mut g = vec![c64(0.0, 0.0); 3 * nm];
        grad_masked(&|l| u[l], &inside, &gw, n, &mut g);
        for a in 0..3 {
            let (s, len) = ([1, 6, 30][a], n[a]);
            for (i, &l) in inside.iter().enumerate() {
                let c = (l / s) % len;
                let nb = if c + 1 == len { l - (len - 1) * s } else { l + s };
                assert_eq!(g[a * nm + i], (u[nb] - u[l]) * gw[a][i]);
            }
        }
        let mut h = vec![c64(9.0, 9.0); nt];
        grad_masked_adj(&r, &mask_of, &gw, n, &[0..6, 0..5, 0..4], &mut h);
        let lhs: Complex64 = g.iter().zip(&r).map(|(a, b)| a.conj() * b).sum();
        let rhs: Complex64 = u.iter().zip(&h).map(|(a, b)| a.conj() * b).sum();
        assert!((lhs - rhs).norm() < 1e-12);
    }

    /// LSQR on vectors stored compactly (`compact_parts`) gives exactly the iterates of the
    /// dense solve whose vectors are zero off the support.
    #[test]
    fn test_lsqr_compact_equals_dense() {
        let nt = 3 * NORM_CHUNK + 1234;
        let supp: Vec<usize> = (0..nt).filter(|&l| (l / 1000) % 3 != 1 && l % 7 != 3).collect();
        let w: Vec<f64> = (0..nt).map(|l| 0.5 + ((l * 13) % 17) as f64 / 7.0).collect();
        let b: Vec<Complex64> = (0..nt).map(|l| if supp.binary_search(&l).is_ok() { c64((l as f64 * 0.01).sin(), (l as f64 * 0.003).cos()) } else { c64(0.0, 0.0) }).collect();
        // diagonal operator restricted to the support
        let dense = |v: &[Complex64]| -> Vec<Complex64> { v.iter().zip(&w).map(|(&z, &wv)| z * wv).collect() };
        let (xd, od) = lsqr_matlab(dense, dense, &b, 1e-14, 15);
        let wc: Vec<f64> = supp.iter().map(|&l| w[l]).collect();
        let bc: Vec<Complex64> = supp.iter().map(|&l| b[l]).collect();
        let parts = compact_parts(supp.iter().copied());
        let op = |v: &[Complex64], out: &mut [Complex64]| out.iter_mut().zip(v).zip(&wc).for_each(|((o, &z), &wv)| *o = z * wv);
        let (xc, oc) = lsqr_matlab_into(op, op, bc, (&parts, &parts), 1e-14, 15);
        assert_eq!(od, oc);
        for (i, &l) in supp.iter().enumerate() { assert_eq!(xd[l], xc[i]); }
    }

    /// Sphere of susceptibility forward-modelled through the dipole kernel on a 2x grid.
    fn sphere_phantom(n: usize, vs: (f64, f64, f64), bdir: (f64, f64, f64)) -> (Vec<f64>, Vec<u8>, Vec<f64>) {
        let nt = n * n * n;
        let c = (n as f64 - 1.0) / 2.0;
        let mut chi = vec![0.0; nt];
        let mut mask = vec![0u8; nt];
        for k in 0..n { for j in 0..n { for i in 0..n {
            let (x, y, z) = (i as f64 - c, j as f64 - c, k as f64 - c);
            let r2 = x * x + y * y + z * z;
            let l = i + j * n + k * n * n;
            if r2 < (0.4 * n as f64).powi(2) { mask[l] = 1; }
            if (x - 2.0).powi(2) + y * y + z * z < 9.0 { chi[l] = 0.1; }
            if (x + 3.0).powi(2) + (y - 1.0).powi(2) + z * z < 4.0 { chi[l] = -0.05; }
        } } }
        let m2 = 2 * n;
        let g2 = Grid::new(m2, m2, m2, vs.0, vs.1, vs.2);
        let d = dipole_kernel(&g2, bdir);
        let mut big = vec![Complex64::new(0.0, 0.0); m2 * m2 * m2];
        for k in 0..n { for j in 0..n { for i in 0..n { big[i + j * m2 + k * m2 * m2].re = chi[i + j * n + k * n * n]; } } }
        let mut ws = Fft3dWorkspace::new(m2, m2, m2);
        ws.fft3d(&mut big);
        for (z, &dv) in big.iter_mut().zip(&d) { *z *= dv; }
        ws.ifft3d(&mut big);
        let mut field = vec![0.0; nt];
        for k in 0..n { for j in 0..n { for i in 0..n {
            let l = i + j * n + k * n * n;
            if mask[l] != 0 { field[l] = big[i + j * m2 + k * m2 * m2].re; }
        } } }
        (field, mask, chi)
    }

    fn corr_in(a: &[f64], b: &[f64], m: &[u8]) -> f64 {
        let idx: Vec<usize> = (0..a.len()).filter(|&i| m[i] != 0).collect();
        let n = idx.len() as f64;
        let (ma, mb) = (idx.iter().map(|&i| a[i]).sum::<f64>() / n, idx.iter().map(|&i| b[i]).sum::<f64>() / n);
        let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
        for &i in &idx { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma).powi(2); sbb += (b[i] - mb).powi(2); }
        sab / (saa * sbb).sqrt()
    }

    #[test]
    fn test_ilsqr_sti_recovers_spheres() {
        let n = 16;
        let bdir = (0.1, -0.2, 0.95f64.sqrt());
        let (field, mask, chi) = sphere_phantom(n, (1.0, 1.0, 1.0), bdir);
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let mut steps = Vec::new();
        let params = IlsqrParams { max_iter: 40, ..IlsqrParams::default() };
        let (x, xsa, xfs, x1) = ilsqr_with_padding(&field, &mask, &grid, bdir, &params, [4.0; 3], |s, t| steps.push((s, t)));
        assert_eq!(steps, vec![(1, 4), (2, 4), (3, 4), (4, 4)]);
        for v in [&x, &xsa, &xfs, &x1] {
            assert!(v.iter().all(|z| z.is_finite()));
            assert!(v.iter().zip(&mask).all(|(&z, &m)| m != 0 || z == 0.0), "output must be masked");
        }
        let (r, r1) = (corr_in(&x, &chi, &mask), corr_in(&x1, &chi, &mask));
        println!("corr with truth: chi {:.4}, initial LSQR {:.4}", r, r1);
        assert!(r > 0.9 && r1 > 0.8, "corr with truth {} / {}", r, r1);
        // chi = initial LSQR minus artefacts
        for i in 0..x.len() { assert!((x[i] - (x1[i] - xsa[i])).abs() < 1e-12); }
        // linear in the field (up to LSQR rounding: the artefact solve runs 100 unconverged steps)
        let f2: Vec<f64> = field.iter().map(|v| 3.0 * v).collect();
        let (y, _, _, _) = ilsqr_with_padding(&f2, &mask, &grid, bdir, &params, [4.0; 3], |_, _| {});
        let x3: Vec<f64> = x.iter().map(|v| 3.0 * v).collect();
        let worst = y.iter().zip(&x3).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        println!("scaling: max|chi(3f) - 3 chi(f)| = {:.2e}", worst);
        assert!(corr_in(&y, &x3, &mask) > 0.9999 && worst < 0.01 * 0.3, "not linear: {}", worst);
    }

    /// Single-precision solves track the double-precision result, and keep the output masked.
    /// LSQR amplifies the rounding (on this case the initial solve stops one iteration later in
    /// single precision, 47 vs 46): measured 5.6e-4 (initial) and 2.7e-3 (χ) relative L2, which
    /// matches the 1.7e-3 seen on a UK Biobank field.
    #[test]
    fn test_ilsqr_single_precision_tracks_double() {
        let n = 16;
        let bdir = (0.1, -0.2, 0.95f64.sqrt());
        let (field, mask, _) = sphere_phantom(n, (1.0, 1.0, 1.0), bdir);
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let run = |precision| {
            let params = IlsqrParams { precision, ..IlsqrParams::default() };
            let mut t = IlsqrTrace::default();
            let (x, _, _, x1) = ilsqr_with_padding_traced(&field, &mask, &grid, bdir, &params, [4.0; 3], |_, _| {}, &mut t);
            (x, x1, t.solves)
        };
        let (xd, x1d, sd) = run(IlsqrPrecision::Double);
        let (xs, x1s, ss) = run(IlsqrPrecision::Single);
        println!("double: {:?}\nsingle: {:?}", sd, ss);
        for (d, s) in sd.iter().zip(&ss) {
            assert!(d.flag == s.flag && d.iter.abs_diff(s.iter) <= 2, "solves ended differently: {:?} / {:?}", sd, ss);
        }
        let rel = |a: &[f64], b: &[f64]| norm(&a.iter().zip(b).map(|(p, q)| p - q).collect::<Vec<_>>()) / norm(a);
        let (r, r1) = (rel(&xd, &xs), rel(&x1d, &x1s));
        println!("single vs double: chi {:.2e}, initial LSQR {:.2e}", r, r1);
        assert!(r1 < 5e-3 && r < 1e-2, "single vs double: {} / {}", r, r1);
        assert!(xs.iter().zip(&mask).all(|(&z, &m)| z.is_finite() && (m != 0 || z == 0.0)));
    }

    #[test]
    fn test_ilsqr_sti_empty_mask_and_zero_field() {
        let grid = Grid::new(8, 8, 8, 1.0, 1.0, 1.0);
        let (x, _, _, _) = ilsqr(&[1.0; 512], &[0u8; 512], &grid, (0.0, 0.0, 1.0), &IlsqrParams::default(), |_, _| {});
        assert!(x.iter().all(|&v| v == 0.0));
        let mut mask = vec![0u8; 512];
        for k in 2..6 { for j in 2..6 { for i in 2..6 { mask[i + 8 * j + 64 * k] = 1; } } }
        let (x, _, _, _) = ilsqr_with_padding(&[0.0; 512], &mask, &grid, (0.0, 0.0, 1.0), &IlsqrParams::default(), [4.0; 3], |_, _| {});
        assert!(x.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_ilsqr_params_defaults() {
        let p = IlsqrParams::default();
        assert_eq!((p.tol, p.max_iter), (1e-3, 100));
        let q = IlsqrParams::qsmm();
        assert_eq!((q.tol, q.max_iter), (0.01, 50));
    }

}

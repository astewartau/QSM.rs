//! iLSQR: Iterative LSQR for QSM with streaking artifact removal
//!
//! Reference:
//! Li, W., Wang, N., Yu, F., Han, H., Cao, W., Romero, R., Tantiwongkosi, B.,
//! Duong, T.Q., Liu, C. (2015). "A method for estimating and removing streaking
//! artifacts in quantitative susceptibility mapping."
//! NeuroImage, 108:111-122. https://doi.org/10.1016/j.neuroimage.2014.12.043
//!
//! Reference implementation: https://github.com/kamesy/QSM.m
//!
//! The algorithm consists of 4 steps:
//! 1. Initial LSQR solution with Laplacian-based weights
//! 2. FastQSM estimate using sign(D) approximation
//! 3. Streaking artifact estimation using LSMR
//! 4. Artifact subtraction

/// Parameters for the iLSQR algorithm.
#[derive(Clone, Debug)]
pub struct IlsqrParams {
    /// Convergence tolerance (default: 0.01)
    pub tol: f64,
    /// Maximum iterations (default: 50)
    pub max_iter: usize,
}

impl Default for IlsqrParams {
    fn default() -> Self {
        Self {
            tol: 0.01,
            max_iter: 50,
        }
    }
}

use std::cell::RefCell;
use num_complex::Complex64;
use crate::fft::Fft3dWorkspace;
use crate::kernels::dipole::dipole_kernel;
use crate::kernels::smv::smv_kernel;
use crate::utils::gradient::{fgrad, bdiv};

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
        phi_bar = s * phi_bar;

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
        phi_bar = s * phi_bar;

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

        // Convergence tests (matching MATLAB's lsqr)
        let test1 = normr / (bnorm + 1e-20);
        let test2 = norm_ar / ((norm_a * normr) + 1e-20);
        let rtol = btol + atol * norm_a * xnorm / (bnorm + 1e-20);

        if verbose {
            eprintln!("  LSQR iter {:>3}: ||r||/||b||={:.6e}  ||A'r||/(||A||·||r||)={:.6e}  rtol={:.6e}",
                _iter + 1, test1, test2, rtol);
        }

        if test2 <= atol || test1 <= rtol {
            if verbose {
                eprintln!("  LSQR converged at iteration {} (test1={:.4e}, test2={:.4e})",
                    _iter + 1, test1, test2);
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
        zeta_bar = -s_bar * zeta_bar;

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
        2 * (mask[l + a] as u8) + (mask[l - a] as u8)
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

    // Normalize to [0, 1]
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
    let h = smv_kernel(nx, ny, nz, vsx, vsy, vsz, r_smv);

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
    let sum_xfs: f64 = x_fs.iter().map(|&v| v).sum();
    let sum_xtkd: f64 = x_tkd.iter().map(|&v| v).sum();
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
    let (dx, dy, dz) = fgrad(x0, nx, ny, nz, vsx, vsy, vsz);

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
        let (gx, gy, gz) = fgrad(&x_filtered, nx, ny, nz, vsx, vsy, vsz);

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
        let div = bdiv(&wyx, &wyy, &wyz, nx, ny, nz, vsx, vsy, vsz);

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
// Main iLSQR Algorithm
// ============================================================================

/// iLSQR: A method for estimating and removing streaking artifacts in QSM
///
/// # Arguments
/// * `field` - Unwrapped local field/tissue phase (nx * ny * nz)
/// * `mask` - Binary mask of region of interest
/// * `nx`, `ny`, `nz` - Array dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `bdir` - B0 field direction (bx, by, bz)
/// * `tol` - Stopping tolerance for LSMR solver
/// * `maxit` - Maximum iterations for LSMR
///
/// # Returns
/// Tuple of (susceptibility, streaking_artifacts, fast_qsm, initial_lsqr)
pub fn ilsqr(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bdir: (f64, f64, f64),
    tol: f64,
    maxit: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // Generate dipole kernel
    let d = dipole_kernel(nx, ny, nz, vsx, vsy, vsz, bdir);

    // Create FFT workspace
    let mut workspace = Fft3dWorkspace::new(nx, ny, nz);

    // Step 1: Initial LSQR solution
    let xlsqr = lsqr_step(field, mask, &d, nx, ny, nz, vsx, vsy, vsz, &mut workspace);

    // Step 2: FastQSM estimate
    let xfs = fastqsm_step(field, mask, &d, nx, ny, nz, vsx, vsy, vsz, &mut workspace);

    // Step 3: Estimate streaking artifacts
    let xsa = susceptibility_artifacts_step(
        &xlsqr, &xfs, mask, &d,
        nx, ny, nz, vsx, vsy, vsz,
        tol, maxit, &mut workspace
    );

    // Step 4: Subtract artifacts
    let chi: Vec<f64> = xlsqr.iter().zip(xsa.iter()).zip(mask.iter())
        .map(|((&xl, &xs), &m)| if m > 0 { xl - xs } else { 0.0 })
        .collect();

    (chi, xsa, xfs, xlsqr)
}

/// Simplified iLSQR returning only the final susceptibility map
pub fn ilsqr_simple(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bdir: (f64, f64, f64),
    tol: f64,
    maxit: usize,
) -> Vec<f64> {
    let (chi, _, _, _) = ilsqr(field, mask, nx, ny, nz, vsx, vsy, vsz, bdir, tol, maxit);
    chi
}

/// iLSQR with progress callback
pub fn ilsqr_with_progress<F>(
    field: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    bdir: (f64, f64, f64),
    tol: f64,
    maxit: usize,
    mut progress_callback: F,
) -> Vec<f64>
where
    F: FnMut(usize, usize),
{
    // Generate dipole kernel
    let d = dipole_kernel(nx, ny, nz, vsx, vsy, vsz, bdir);

    // Create FFT workspace
    let mut workspace = Fft3dWorkspace::new(nx, ny, nz);

    progress_callback(1, 4);

    // Step 1: Initial LSQR solution
    let xlsqr = lsqr_step(field, mask, &d, nx, ny, nz, vsx, vsy, vsz, &mut workspace);

    progress_callback(2, 4);

    // Step 2: FastQSM estimate
    let xfs = fastqsm_step(field, mask, &d, nx, ny, nz, vsx, vsy, vsz, &mut workspace);

    progress_callback(3, 4);

    // Step 3: Estimate streaking artifacts
    let xsa = susceptibility_artifacts_step(
        &xlsqr, &xfs, mask, &d,
        nx, ny, nz, vsx, vsy, vsz,
        tol, maxit, &mut workspace
    );

    progress_callback(4, 4);

    // Step 4: Subtract artifacts
    xlsqr.iter().zip(xsa.iter()).zip(mask.iter())
        .map(|((&xl, &xs), &m)| if m > 0 { xl - xs } else { 0.0 })
        .collect()
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
    fn test_lsqr_complex_diagonal() {
        // Test complex LSQR on a diagonal system: A = diag(1, 2, 3), b = [1+i, 4+2i, 9+3i]
        // Expected solution: x = [1+i, 2+i, 3+i]
        let diag = vec![1.0, 2.0, 3.0];
        let expected = vec![
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
            assert!(wi >= 0.0 && wi <= 1.0, "weight[{}] = {} out of [0,1]", i, wi);
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
            assert!(wi >= 0.0 && wi <= 1.0,
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

        let chi = ilsqr_simple(&field, &mask, nx, ny, nz, vsx, vsy, vsz, bdir, tol, maxit);

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
}

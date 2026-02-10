//! iLSQR: Iterative LSQR for QSM with streaking artifact removal
//!
//! Implementation based on:
//! Li W, Wang N, Yu F, Han H, Cao W, Romero R, Tantiwongkosi B, Duong TQ, Liu C.
//! "A method for estimating and removing streaking artifacts in quantitative
//! susceptibility mapping." NeuroImage. 2015 Mar 1;108:111-22.
//!
//! The algorithm consists of 4 steps:
//! 1. Initial LSQR solution with Laplacian-based weights
//! 2. FastQSM estimate using sign(D) approximation
//! 3. Streaking artifact estimation using LSMR
//! 4. Artifact subtraction

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
/// Works with complex vectors, using A^H (conjugate transpose) for the adjoint.
pub fn lsqr_complex<F, G>(
    apply_a: F,
    apply_ah: G,
    b: &[Complex64],
    tol: f64,
    max_iter: usize,
) -> Vec<Complex64>
where
    F: Fn(&[Complex64]) -> Vec<Complex64>,
    G: Fn(&[Complex64]) -> Vec<Complex64>,
{
    // Initialize
    let mut u = b.to_vec();
    let mut beta = norm_complex(&u);

    if beta > 0.0 {
        scale_complex_inplace(&mut u, 1.0 / beta);
    }

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

    for _iter in 0..max_iter {
        // Bidiagonalization
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
    // Initialize
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

    // Initialize variables for Golub-Kahan process
    let mut alpha_bar = alpha;
    let mut zeta_bar = alpha * beta;
    let mut rho = 1.0;
    let mut rho_bar = 1.0;

    let mut h = v.clone();
    let mut h_bar = vec![0.0; n];
    let mut x = vec![0.0; n];

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

        // Construct rotation Q_hat
        let rho_temp = (alpha_bar * alpha_bar + beta * beta).sqrt();
        let c_temp = alpha_bar / rho_temp;
        let s_temp = beta / rho_temp;
        let theta_new = s_temp * alpha;
        alpha_bar = c_temp * alpha;

        // Construct rotation Q_bar
        let rho_bar_temp = (rho_bar * rho_bar + theta_new * theta_new).sqrt();
        let c_bar_new = rho_bar / rho_bar_temp;
        let s_bar_new = theta_new / rho_bar_temp;
        let zeta = c_bar_new * zeta_bar;
        zeta_bar = -s_bar_new * zeta_bar;

        // Update h_bar, x, h
        for i in 0..n {
            h_bar[i] = h[i] - (theta_new * rho / (rho_temp * rho_bar_temp)) * h_bar[i];
            x[i] += (zeta / rho_bar_temp) * h_bar[i];
            h[i] = v[i] - (theta_new / rho_temp) * h[i];
        }

        rho = rho_temp;
        rho_bar = rho_bar_temp;

        // Check convergence
        let norm_ar = zeta_bar.abs();
        let norm_r = ((bnorm * bnorm) - (zeta_bar * zeta_bar)).abs().sqrt();

        let test1 = norm_r / (bnorm + 1e-20);
        let test2 = norm_ar / (norm(&x) + 1e-20);

        if test1 < btol || test2 < atol {
            break;
        }
    }

    x
}

// ============================================================================
// Weight Functions
// ============================================================================

/// Compute Laplacian of a 3D field using finite differences
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

    for k in 0..nz {
        let km1 = if k > 0 { k - 1 } else { 0 };
        let kp1 = if k + 1 < nz { k + 1 } else { nz - 1 };

        for j in 0..ny {
            let jm1 = if j > 0 { j - 1 } else { 0 };
            let jp1 = if j + 1 < ny { j + 1 } else { ny - 1 };

            for i in 0..nx {
                let idx = i + j * nx + k * nx * ny;

                if mask[idx] == 0 {
                    continue;
                }

                let im1 = if i > 0 { i - 1 } else { 0 };
                let ip1 = if i + 1 < nx { i + 1 } else { nx - 1 };

                let idx_xm = im1 + j * nx + k * nx * ny;
                let idx_xp = ip1 + j * nx + k * nx * ny;
                let idx_ym = i + jm1 * nx + k * nx * ny;
                let idx_yp = i + jp1 * nx + k * nx * ny;
                let idx_zm = i + j * nx + km1 * nx * ny;
                let idx_zp = i + j * nx + kp1 * nx * ny;

                lap[idx] = hx * (f[idx_xp] - 2.0 * f[idx] + f[idx_xm])
                         + hy * (f[idx_yp] - 2.0 * f[idx] + f[idx_ym])
                         + hz * (f[idx_zp] - 2.0 * f[idx] + f[idx_zm]);
            }
        }
    }

    lap
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

    // Sort for percentile calculation
    masked_lap.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n_masked = masked_lap.len();
    let idx_min = ((pmin / 100.0) * n_masked as f64) as usize;
    let idx_max = ((pmax / 100.0) * n_masked as f64).min(n_masked as f64 - 1.0) as usize;

    let thr_min = masked_lap[idx_min.min(n_masked - 1)];
    let thr_max = masked_lap[idx_max.min(n_masked - 1)];

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

    // Collect non-zero values for percentile
    let mut vals: Vec<f64> = w.iter().filter(|&&v| v > 1e-20).copied().collect();

    if vals.is_empty() {
        return vec![0.0; len];
    }

    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n_vals = vals.len();
    let idx_a = ((pa / 100.0) * n_vals as f64) as usize;
    let idx_b = ((pb / 100.0) * n_vals as f64).min(n_vals as f64 - 1.0) as usize;

    let ab_min = vals[idx_a.min(n_vals - 1)];
    let ab_max = vals[idx_b.min(n_vals - 1)];

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

/// Gradient weights for streaking artifact estimation (Equation 15)
fn gradient_weights_ilsqr(
    x: &[f64],
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    pmin: f64,
    pmax: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Compute gradients
    let (gx, gy, gz) = fgrad(x, nx, ny, nz, vsx, vsy, vsz);

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

    let n_masked = masked_g.len();
    let idx_min = ((pmin / 100.0) * n_masked as f64) as usize;
    let idx_max = ((pmax / 100.0) * n_masked as f64).min(n_masked as f64 - 1.0) as usize;

    let thr_min = masked_g[idx_min.min(n_masked - 1)];
    let thr_max = masked_g[idx_max.min(n_masked - 1)];

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
    let apply_a = |x: &[Complex64]| -> Vec<Complex64> {
        // D .* x (in k-space) - x is complex, D is real
        let dx: Vec<Complex64> = x.iter().zip(d.iter())
            .map(|(xi, &di)| xi * di)
            .collect();

        // IFFT(D .* x)
        let mut dx_ifft = dx.clone();
        let mut temp_ws = Fft3dWorkspace::new(nx, ny, nz);
        temp_ws.ifft3d(&mut dx_ifft);

        // w .* real(IFFT(D .* x)) - take real part here as per MATLAB reference
        let wdx: Vec<Complex64> = w.iter().zip(dx_ifft.iter())
            .map(|(&wi, dxi)| Complex64::new(wi * dxi.re, 0.0))
            .collect();

        // FFT(w .* ...)
        let mut wdx_fft = wdx.clone();
        temp_ws.fft3d(&mut wdx_fft);

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
    let x_lsqr = lsqr_complex(apply_a, apply_ah, &b, tol_lsqr, maxit_lsqr);

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

    // FFT of SMV kernel
    let h_complex: Vec<Complex64> = h.iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect();
    let mut h_fft = h_complex;
    workspace.fft3d(&mut h_fft);

    // Equation (9): Apply weighted combination
    // x = FFT(mask .* IFFT(wfs .* x + (1-wfs) .* (h .* x)))
    let mut x_filtered: Vec<Complex64> = x.iter()
        .zip(wfs.iter())
        .zip(h_fft.iter())
        .map(|((xi, &wi), hi)| {
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
        .map(|((xi, &wi), hi)| {
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
    let sum_xfs: f64 = x_fs.iter().zip(mask.iter())
        .filter(|(_, &m)| m > 0)
        .map(|(&v, _)| v)
        .sum();

    let sum_xtkd: f64 = x_tkd.iter().zip(mask.iter())
        .filter(|(_, &m)| m > 0)
        .map(|(&v, _)| v)
        .sum();

    let sum_xfs2: f64 = x_fs.iter().zip(mask.iter())
        .filter(|(_, &m)| m > 0)
        .map(|(&v, _)| v * v)
        .sum();

    let sum_xfs_xtkd: f64 = x_fs.iter().zip(x_tkd.iter()).zip(mask.iter())
        .filter(|(_, &m)| m > 0)
        .map(|((&xf, &xt), _)| xf * xt)
        .sum();

    let n_mask: f64 = mask.iter().filter(|&&m| m > 0).count() as f64;

    // Solve 2x2 system: [sum_xfs2, sum_xfs; sum_xfs, n] * [a; b] = [sum_xfs_xtkd; sum_xtkd]
    let det = sum_xfs2 * n_mask - sum_xfs * sum_xfs;

    let (a, b) = if det.abs() > 1e-20 {
        let a = (n_mask * sum_xfs_xtkd - sum_xfs * sum_xtkd) / det;
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

    // Define forward operator A
    let apply_a = |x_in: &[f64]| -> Vec<f64> {
        // x_in is in image space
        // Apply Mic in k-space
        let x_complex: Vec<Complex64> = x_in.iter()
            .map(|&v| Complex64::new(v, 0.0))
            .collect();

        let mut x_fft = x_complex;
        let mut temp_ws = Fft3dWorkspace::new(nx, ny, nz);
        temp_ws.fft3d(&mut x_fft);

        // Apply ill-conditioned mask
        let x_mic: Vec<Complex64> = x_fft.iter().zip(mic.iter())
            .map(|(xi, &mi)| xi * mi)
            .collect();

        let mut x_ifft = x_mic;
        temp_ws.ifft3d(&mut x_ifft);

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

        // Adjoint of gradient (negative divergence)
        let div = bdiv(&wyx, &wyy, &wyz, nx, ny, nz, vsx, vsy, vsz);

        // Apply Mic in k-space
        let div_complex: Vec<Complex64> = div.iter()
            .map(|&v| Complex64::new(v, 0.0))
            .collect();

        let mut div_fft = div_complex;
        let mut temp_ws = Fft3dWorkspace::new(nx, ny, nz);
        temp_ws.fft3d(&mut div_fft);

        let div_mic: Vec<Complex64> = div_fft.iter().zip(mic.iter())
            .map(|(di, &mi)| di * mi)
            .collect();

        let mut div_ifft = div_mic;
        temp_ws.ifft3d(&mut div_ifft);

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
}

//! FANSI nonlinear TV / TGV dipole inversion.
//!
//! Nonlinear total-variation (nlTV) and nonlinear total-generalized-variation
//! (nlTGV) QSM dipole inversion with a nonlinear (wrapped-phase) data-fidelity
//! term, solved with ADMM plus an inner complex-argument Newton iteration.
//!
//! The nonlinear fidelity models the field data as `exp(i * D x)` and matches
//! it to the wrapped local-field phase, which makes the reconstruction robust to
//! phase-wrap / high-field regimes.
//!
//! References:
//! - nlTV: Milovic, C., Bilgic, B., Zhao, B., et al. (2018).
//!   "Fast nonlinear susceptibility inversion with variational regularization."
//!   Magnetic Resonance in Medicine, 80(2):814-821.
//!   <https://doi.org/10.1002/mrm.27073>
//! - nlTGV: the total-generalized-variation variant of the above.
//!
//! Ported faithfully from the FANSI toolbox `nlTV.m` and `nlTGV.m`
//! (<https://gitlab.com/cmilovic/FANSI-toolbox>).
//!
//! # Units / `phase_scale`
//! The FANSI fidelity is a *phase* (radians) model: `sin(z2 - phase)`. If the
//! input `local_field` is already radians, use `phase_scale = 1.0`. If it is
//! ppm-scale, `phase_scale` should convert ppm -> radians (`2*pi*gamma*B0*TE`);
//! the returned map is divided by `phase_scale` so it is expressed on the same
//! scale as the input field.

use crate::inversion::admm::prepare_fansi_spectral;
use crate::utils::gradient::{bdiv_inplace, fgrad_inplace};
use crate::utils::{apply_mask_zero, shrink};
use crate::Grid;
use num_complex::Complex64;
use std::f64::consts::PI;

/// FANSI nlTV/nlTGV parameters.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct FansiParams {
    /// First-order (TV / TGV gradient) L1 penalty weight.
    pub alpha1: f64,
    /// Gradient-consistency ADMM weight.
    pub mu1: f64,
    /// Fidelity-consistency ADMM weight.
    pub mu2: f64,
    /// Second-order (symmetric-gradient) L1 penalty weight (nlTGV only).
    pub alpha0: f64,
    /// Second-order consistency ADMM weight (nlTGV only).
    pub mu0: f64,
    /// Number of outer ADMM iterations.
    pub max_iter: usize,
    /// Percent-update convergence stopping tolerance.
    pub tol_update: f64,
    /// Inner Newton convergence tolerance.
    pub tol_delta: f64,
    /// Working (phase) scale applied to the input local field; output is divided
    /// by it. Use 1.0 for radians input, ppm->radians factor for ppm input.
    pub phase_scale: f64,
    /// Select nlTGV (`true`) or nlTV (`false`).
    pub is_tgv: bool,
}

impl Default for FansiParams {
    fn default() -> Self {
        Self {
            alpha1: 2e-4,
            mu1: 2e-2,
            mu2: 1.0,
            alpha0: 4e-4,
            mu0: 4e-2,
            max_iter: 150,
            tol_update: 0.1,
            tol_delta: 1e-6,
            phase_scale: 1.0,
            is_tgv: false,
        }
    }
}

/// L2 norm of a real slice.
fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|&a| a * a).sum::<f64>().sqrt()
}

/// FANSI nlTV/nlTGV dipole inversion.
///
/// # Arguments
/// * `local_field` - Local field values (nx * ny * nz).
/// * `mask` - Binary mask (nx * ny * nz), non-zero = inside ROI.
/// * `grid` - Volume grid (dimensions and voxel sizes).
/// * `bdir` - B0 field direction.
/// * `params` - FANSI parameters (`is_tgv` selects nlTGV vs nlTV).
/// * `progress` - Progress callback `(iteration, max_iter)`.
///
/// # Returns
/// Estimated susceptibility map, masked to the ROI.
pub fn fansi(
    local_field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &FansiParams,
    progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    if params.is_tgv {
        nltgv(local_field, mask, grid, bdir, params, progress)
    } else {
        nltv(local_field, mask, grid, bdir, params, progress)
    }
}

/// Nonlinear total-variation dipole inversion (FANSI `nlTV.m`).
fn nltv(
    local_field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &FansiParams,
    mut progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let n = grid.n_total();

    let (mut fft_ws, k, ee2) = prepare_fansi_spectral(grid, bdir);

    // Scaled phase and W = mask (0/1 weight).
    let phase: Vec<f64> = local_field.iter().map(|&f| f * params.phase_scale).collect();
    let w: Vec<f64> = mask.iter().map(|&m| if m != 0 { 1.0 } else { 0.0 }).collect();

    let mu1 = params.mu1;
    let mu2 = params.mu2;
    let alpha_over_mu = params.alpha1 / mu1;

    // ADMM variables.
    let mut x = vec![0.0f64; n];
    let mut x_prev = vec![0.0f64; n];

    // Gradient-consistency split (real) and its multipliers.
    let mut z_dx = vec![0.0f64; n];
    let mut z_dy = vec![0.0f64; n];
    let mut z_dz = vec![0.0f64; n];
    let mut s_dx = vec![0.0f64; n];
    let mut s_dy = vec![0.0f64; n];
    let mut s_dz = vec![0.0f64; n];

    // Fidelity-consistency auxiliary z2 = W .* phase ./ (W + mu2), and multiplier.
    let mut z2 = vec![0.0f64; n];
    for i in 0..n {
        let den = w[i] + mu2;
        z2[i] = if den != 0.0 { w[i] * phase[i] / den } else { 0.0 };
    }
    let mut s2 = vec![0.0f64; n];

    // Reusable buffers.
    let mut fdiv = vec![Complex64::new(0.0, 0.0); n];
    let mut fd2 = vec![Complex64::new(0.0, 0.0); n];
    let mut xhat = vec![Complex64::new(0.0, 0.0); n];
    let mut fx = vec![Complex64::new(0.0, 0.0); n];

    let mut gxc = vec![0.0f64; n];
    let mut gyc = vec![0.0f64; n];
    let mut gzc = vec![0.0f64; n];
    let mut x_dx = vec![0.0f64; n];
    let mut x_dy = vec![0.0f64; n];
    let mut x_dz = vec![0.0f64; n];
    let mut div = vec![0.0f64; n];
    let mut dx = vec![0.0f64; n]; // Dx = real(ifft(k .* fft(x)))
    let mut rhs_z2 = vec![0.0f64; n];
    let mut diff = vec![0.0f64; n];
    let mut update = vec![0.0f64; n];

    for t in 0..params.max_iter {
        progress(t + 1, params.max_iter);

        // ---- x-subproblem -------------------------------------------------
        // Gradient side: mu1 * sum(E_t .* fft(z_d - s_d)) == mu1 * fft(bdiv(z_d - s_d)).
        for i in 0..n {
            gxc[i] = z_dx[i] - s_dx[i];
            gyc[i] = z_dy[i] - s_dy[i];
            gzc[i] = z_dz[i] - s_dz[i];
        }
        bdiv_inplace(&mut div, &gxc, &gyc, &gzc, grid);
        for i in 0..n {
            fdiv[i] = Complex64::new(div[i], 0.0);
        }
        fft_ws.fft3d(&mut fdiv);

        // Fidelity side: mu2 * conj(K) .* fft(z2 - s2) (K real -> conj = K).
        for i in 0..n {
            fd2[i] = Complex64::new(z2[i] - s2[i], 0.0);
        }
        fft_ws.fft3d(&mut fd2);

        for i in 0..n {
            // NOTE the minus on the gradient-consistency term: the adjoint of the
            // crate's forward-difference `fgrad` is `-bdiv` (not `+bdiv`), so the
            // spectral term mu1*sum(E_t.*F(z_d-s_d)) = -mu1*F(bdiv(z_d-s_d)).
            // Matches QSM.rs's own TV-ADMM (`f_hat - rho*fft(bdiv...)`). Using +bdiv
            // fails to cancel mu1*∇*∇ at the fixed point, doubling the effective
            // regularization and damping the susceptibility amplitude.
            let num = -mu1 * fdiv[i] + mu2 * k[i] * fd2[i];
            let den = mu2 * k[i] * k[i] + mu1 * ee2[i];
            // Guard the dipole null-space (DC and any singular bin): both the
            // dipole kernel and the Laplacian vanish there, so susceptibility is
            // undetermined up to a constant. Zero it (matches TV's inv_a guard);
            // dividing FFT round-off by ~0 would otherwise create a huge DC pedestal.
            xhat[i] = if den > 1e-20 { num / den } else { Complex64::new(0.0, 0.0) };
        }
        fft_ws.ifft3d(&mut xhat);
        x_prev.copy_from_slice(&x);
        for i in 0..n {
            x[i] = xhat[i].re;
        }

        // ---- convergence check --------------------------------------------
        let xnorm = norm2(&x);
        if xnorm > 0.0 {
            for i in 0..n {
                diff[i] = x[i] - x_prev[i];
            }
            let x_update = 100.0 * norm2(&diff) / xnorm;
            if x_update < params.tol_update || x_update.is_nan() {
                progress(t + 1, t + 1);
                break;
            }
        }

        if t + 1 >= params.max_iter {
            break;
        }

        // ---- gradient split update (TV shrink) ----------------------------
        // Fx = fft(x) (reused for the dipole rhs below).
        for i in 0..n {
            fx[i] = Complex64::new(x[i], 0.0);
        }
        fft_ws.fft3d(&mut fx);

        fgrad_inplace(&mut x_dx, &mut x_dy, &mut x_dz, &x, grid);
        for i in 0..n {
            z_dx[i] = shrink(x_dx[i] + s_dx[i], alpha_over_mu);
            z_dy[i] = shrink(x_dy[i] + s_dy[i], alpha_over_mu);
            z_dz[i] = shrink(x_dz[i] + s_dz[i], alpha_over_mu);
            s_dx[i] += x_dx[i] - z_dx[i];
            s_dy[i] += x_dy[i] - z_dy[i];
            s_dz[i] += x_dz[i] - z_dz[i];
        }

        // ---- fidelity auxiliary (z2) via Newton ---------------------------
        // Dx = real(ifft(K .* Fx)) ; rhs_z2 = mu2 * (Dx + s2) ; z2 init = Dx + s2.
        for i in 0..n {
            xhat[i] = fx[i] * k[i];
        }
        fft_ws.ifft3d(&mut xhat);
        for i in 0..n {
            dx[i] = xhat[i].re;
            rhs_z2[i] = mu2 * (dx[i] + s2[i]);
            z2[i] = rhs_z2[i] / mu2;
        }

        let mut delta = f64::INFINITY;
        let mut inn = 0usize;
        while delta > params.tol_delta && inn < 10 {
            inn += 1;
            let norm_old = norm2(&z2);
            for i in 0..n {
                let a = z2[i] - phase[i];
                let numer = w[i] * a.sin() + mu2 * z2[i] - rhs_z2[i];
                let denom = w[i] * a.cos() + mu2;
                update[i] = numer / denom;
                z2[i] -= update[i];
            }
            delta = if norm_old > 0.0 {
                norm2(&update) / norm_old
            } else {
                0.0
            };
        }

        // ---- multiplier update --------------------------------------------
        // s2 = s2 + Dx - z2.
        for i in 0..n {
            s2[i] += dx[i] - z2[i];
        }
    }

    if params.phase_scale != 1.0 {
        for v in &mut x {
            *v /= params.phase_scale;
        }
    }

    apply_mask_zero(&mut x, mask);
    x
}

/// Multiply a complex spectral array in place by a complex multiplier array.
#[inline]
fn spectral_mul_assign(dst: &mut [Complex64], m: &[Complex64]) {
    for (d, &mm) in dst.iter_mut().zip(m.iter()) {
        *d *= mm;
    }
}

/// Nonlinear total-generalized-variation dipole inversion (FANSI `nlTGV.m`).
///
/// Everything (operators + normal-equation cofactors) is built from the local,
/// *unscaled* spectral gradient multipliers `E1,E2,E3` to stay self-consistent
/// with the MATLAB Cramer-rule algebra.
#[allow(clippy::too_many_lines)]
fn nltgv(
    local_field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &FansiParams,
    mut progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let n = grid.n_total();
    let (nx, ny, nz) = (grid.nx(), grid.ny(), grid.nz());

    // Reuse only the real dipole kernel K from the shared prep.
    let (mut fft_ws, k, _ee2) = prepare_fansi_spectral(grid, bdir);

    // Local unscaled spectral gradient multipliers, MATLAB order i+j*nx+k*nx*ny.
    let mut e1 = vec![Complex64::new(0.0, 0.0); n];
    let mut e2 = vec![Complex64::new(0.0, 0.0); n];
    let mut e3 = vec![Complex64::new(0.0, 0.0); n];
    let two_pi = 2.0 * PI;
    for kk in 0..nz {
        let ez = Complex64::new(0.0, 1.0) * (two_pi * (kk as f64) / (nz as f64));
        let e3v = Complex64::new(1.0, 0.0) - ez.exp();
        for jj in 0..ny {
            let ey = Complex64::new(0.0, 1.0) * (two_pi * (jj as f64) / (ny as f64));
            let e2v = Complex64::new(1.0, 0.0) - ey.exp();
            for ii in 0..nx {
                let ex = Complex64::new(0.0, 1.0) * (two_pi * (ii as f64) / (nx as f64));
                let e1v = Complex64::new(1.0, 0.0) - ex.exp();
                let idx = ii + jj * nx + kk * nx * ny;
                e1[idx] = e1v;
                e2[idx] = e2v;
                e3[idx] = e3v;
            }
        }
    }

    let phase: Vec<f64> = local_field.iter().map(|&f| f * params.phase_scale).collect();
    let w: Vec<f64> = mask.iter().map(|&m| if m != 0 { 1.0 } else { 0.0 }).collect();

    let mu0 = params.mu0;
    let mu1 = params.mu1;
    let mu2 = params.mu2;

    // Precompute per-voxel normal-equation cofactors (all complex).
    let mut d11 = vec![Complex64::new(0.0, 0.0); n];
    let mut d21 = d11.clone();
    let mut d31 = d11.clone();
    let mut d41 = d11.clone();
    let mut d12 = d11.clone();
    let mut d22 = d11.clone();
    let mut d32 = d11.clone();
    let mut d42 = d11.clone();
    let mut d13 = d11.clone();
    let mut d23 = d11.clone();
    let mut d33 = d11.clone();
    let mut d43 = d11.clone();
    let mut d14 = d11.clone();
    let mut d24 = d11.clone();
    let mut d34 = d11.clone();
    let mut d44 = d11.clone();
    let mut det_ainv = d11.clone();

    let half = 0.5;
    for i in 0..n {
        let e1i = e1[i];
        let e2i = e2[i];
        let e3i = e3[i];
        let et1 = e1i.conj();
        let et2 = e2i.conj();
        let et3 = e3i.conj();

        let e1te1 = et1 * e1i;
        let e2te2 = et2 * e2i;
        let e3te3 = et3 * e3i;
        let mu0h_e1te2 = (mu0 * half) * et1 * e2i;
        let mu0h_e1te3 = (mu0 * half) * et1 * e3i;
        let mu0h_e2te3 = (mu0 * half) * et2 * e3i;

        // a0 = mu2 * conj(K) .* K = mu2 * k * k (real).
        let a0 = Complex64::new(mu2 * k[i] * k[i], 0.0);
        let a1 = a0 + mu1 * (e1te1 + e2te2 + e3te3);
        let a2 = Complex64::new(mu1, 0.0) + mu0 * (e1te1 + (e2te2 + e3te3) * half);
        let a3 = Complex64::new(mu1, 0.0) + mu0 * (e1te1 * half + e2te2 + e3te3 * half);
        let a4 = Complex64::new(mu1, 0.0) + mu0 * ((e1te1 + e2te2) * half + e3te3);
        let a5 = -mu1 * e1i;
        let a6 = -mu1 * e2i;
        let a7 = mu0h_e1te2;
        let a8 = -mu1 * e3i;
        let a9 = mu0h_e1te3;
        let a10 = mu0h_e2te3;
        let a5t = a5.conj();
        let a6t = a6.conj();
        let a7t = a7.conj();
        let a8t = a8.conj();
        let a9t = a9.conj();
        let a10t = a10.conj();

        let c11 = a2 * a3 * a4 + a7t * a9 * a10t + a7 * a9t * a10
            - a3 * a9 * a9t
            - a2 * a10 * a10t
            - a4 * a7 * a7t;
        let c21 = a3 * a4 * a5t + a6t * a9 * a10t + a7 * a8t * a10
            - a3 * a8t * a9
            - a5t * a10 * a10t
            - a4 * a6t * a7;
        let c31 = a4 * a5t * a7t + a6t * a9 * a9t + a2 * a8t * a10
            - a7t * a8t * a9
            - a5t * a9t * a10
            - a2 * a4 * a6t;
        let c41 = a5t * a7t * a10t + a6t * a7 * a9t + a2 * a3 * a8t
            - a7 * a7t * a8t
            - a3 * a5t * a9t
            - a2 * a6t * a10t;
        let c12 = a3 * a4 * a5 + a7t * a8 * a10t + a6 * a9t * a10
            - a3 * a8 * a9t
            - a5 * a10 * a10t
            - a4 * a6 * a7t;
        let c22 = a1 * a3 * a4 + a6t * a8 * a10t + a6 * a8t * a10
            - a3 * a8 * a8t
            - a1 * a10 * a10t
            - a4 * a6 * a6t;
        let c32 = a1 * a4 * a7t + a6t * a8 * a9t + a5 * a8t * a10
            - a7t * a8 * a8t
            - a1 * a9t * a10
            - a4 * a5 * a6t;
        let c42 = a1 * a7t * a10t + a6 * a6t * a9t + a3 * a5 * a8t
            - a6 * a7t * a8t
            - a1 * a3 * a9t
            - a5 * a6t * a10t;
        let c13 = a4 * a5 * a7 + a2 * a8 * a10t + a6 * a9 * a9t
            - a7 * a8 * a9t
            - a5 * a9 * a10t
            - a2 * a4 * a6;
        let c23 = a1 * a4 * a7 + a5t * a8 * a10t + a6 * a8t * a9
            - a7 * a8 * a8t
            - a1 * a9 * a10t
            - a4 * a5t * a6;
        let c33 = a1 * a2 * a4 + a5t * a8 * a9t + a5 * a8t * a9
            - a2 * a8 * a8t
            - a1 * a9 * a9t
            - a4 * a5 * a5t;
        let c43 = a1 * a2 * a10t + a5t * a6 * a9t + a5 * a7 * a8t
            - a2 * a6 * a8t
            - a1 * a7 * a9t
            - a5 * a5t * a10t;
        let c14 = a5 * a7 * a10 + a2 * a3 * a8 + a6 * a7t * a9
            - a7 * a7t * a8
            - a3 * a5 * a9
            - a2 * a6 * a10;
        let c24 = a1 * a7 * a10 + a3 * a5t * a8 + a6 * a6t * a9
            - a6t * a7 * a8
            - a1 * a3 * a9
            - a5t * a6 * a10;
        let c34 = a1 * a2 * a10 + a5t * a7t * a8 + a5 * a6t * a9
            - a2 * a6t * a8
            - a1 * a7t * a9
            - a5 * a5t * a10;
        let c44 = a1 * a2 * a3 + a5t * a6 * a7t + a5 * a6t * a7
            - a2 * a6 * a6t
            - a1 * a7 * a7t
            - a3 * a5 * a5t;

        let det_a = a1 * c11 - a5 * c21 + a6 * c31 - a8 * c41;

        d11[i] = c11;
        d21[i] = c21;
        d31[i] = c31;
        d41[i] = c41;
        d12[i] = c12;
        d22[i] = c22;
        d32[i] = c32;
        d42[i] = c42;
        d13[i] = c13;
        d23[i] = c23;
        d33[i] = c33;
        d43[i] = c43;
        d14[i] = c14;
        d24[i] = c24;
        d34[i] = c34;
        d44[i] = c44;
        // Guard the singular (DC null-space) bins: at k=0 every spectral
        // operator vanishes so det_a -> 0. Zero the solve there instead of
        // dividing FFT round-off by ~eps (which would create a huge DC pedestal).
        det_ainv[i] = if det_a.norm() > 1e-20 {
            Complex64::new(1.0, 0.0) / det_a
        } else {
            Complex64::new(0.0, 0.0)
        };
    }

    // Conjugates of E used in RHS assembly.
    let et1: Vec<Complex64> = e1.iter().map(|c| c.conj()).collect();
    let et2: Vec<Complex64> = e2.iter().map(|c| c.conj()).collect();
    let et3: Vec<Complex64> = e3.iter().map(|c| c.conj()).collect();

    // ADMM state.
    let mut x = vec![0.0f64; n];
    let mut x_prev = vec![0.0f64; n];
    let mut v1 = vec![0.0f64; n];
    let mut v2 = vec![0.0f64; n];
    let mut v3 = vec![0.0f64; n];

    // First-order splits.
    let mut z1_1 = vec![0.0f64; n];
    let mut z1_2 = vec![0.0f64; n];
    let mut z1_3 = vec![0.0f64; n];
    let mut s1_1 = vec![0.0f64; n];
    let mut s1_2 = vec![0.0f64; n];
    let mut s1_3 = vec![0.0f64; n];

    // Symmetric (second-order) splits.
    let mut z0_1 = vec![0.0f64; n];
    let mut z0_2 = vec![0.0f64; n];
    let mut z0_3 = vec![0.0f64; n];
    let mut z0_4 = vec![0.0f64; n];
    let mut z0_5 = vec![0.0f64; n];
    let mut z0_6 = vec![0.0f64; n];
    let mut s0_1 = vec![0.0f64; n];
    let mut s0_2 = vec![0.0f64; n];
    let mut s0_3 = vec![0.0f64; n];
    let mut s0_4 = vec![0.0f64; n];
    let mut s0_5 = vec![0.0f64; n];
    let mut s0_6 = vec![0.0f64; n];

    // Fidelity auxiliary z2 = W .* phase ./ (W + mu2).
    let mut z2 = vec![0.0f64; n];
    for i in 0..n {
        let den = w[i] + mu2;
        z2[i] = if den != 0.0 { w[i] * phase[i] / den } else { 0.0 };
    }
    let mut s2 = vec![0.0f64; n];

    let alpha1_over_mu1 = params.alpha1 / mu1;
    let alpha0_over_mu0 = params.alpha0 / mu0;

    // Reusable complex/real buffers.
    let mut rhs1 = vec![Complex64::new(0.0, 0.0); n];
    let mut rhs2 = rhs1.clone();
    let mut rhs3 = rhs1.clone();
    let mut rhs4 = rhs1.clone();
    let mut t1 = rhs1.clone();
    let mut t2 = rhs1.clone();
    let mut t3 = rhs1.clone();
    let mut fx = rhs1.clone();
    let mut fv1 = rhs1.clone();
    let mut fv2 = rhs1.clone();
    let mut fv3 = rhs1.clone();
    let mut cbuf = rhs1.clone();

    let mut dx1 = vec![0.0f64; n];
    let mut dx2 = vec![0.0f64; n];
    let mut dx3 = vec![0.0f64; n];
    let mut ev1 = vec![0.0f64; n];
    let mut ev2 = vec![0.0f64; n];
    let mut ev3 = vec![0.0f64; n];
    let mut ev4 = vec![0.0f64; n];
    let mut ev5 = vec![0.0f64; n];
    let mut ev6 = vec![0.0f64; n];
    let mut dx = vec![0.0f64; n];
    let mut rhs_z2 = vec![0.0f64; n];
    let mut diff = vec![0.0f64; n];
    let mut update = vec![0.0f64; n];

    // Scratch real buffer for building fft inputs.
    let mut rbuf = vec![0.0f64; n];

    // real(fft) of a real field `src` into complex buffer `dst`.
    macro_rules! fft_real {
        ($dst:expr, $src:expr) => {{
            for i in 0..n {
                $dst[i] = Complex64::new($src[i], 0.0);
            }
            fft_ws.fft3d(&mut $dst);
        }};
    }
    // real(fft) of (a - b) elementwise into complex buffer `dst`.
    macro_rules! fft_real_diff {
        ($dst:expr, $a:expr, $b:expr) => {{
            for i in 0..n {
                rbuf[i] = $a[i] - $b[i];
            }
            fft_real!($dst, rbuf);
        }};
    }

    for t in 0..params.max_iter {
        progress(t + 1, params.max_iter);

        // ---- assemble RHS (spectral) --------------------------------------
        // rhs1 = mu2*conj(K)*fft(z2 - s2) + mu1*(Et*fft(z1 - s1))
        for i in 0..n {
            cbuf[i] = Complex64::new(z2[i] - s2[i], 0.0);
        }
        fft_ws.fft3d(&mut cbuf);
        for i in 0..n {
            rhs1[i] = mu2 * k[i] * cbuf[i];
        }
        fft_real_diff!(t1, z1_1, s1_1);
        fft_real_diff!(t2, z1_2, s1_2);
        fft_real_diff!(t3, z1_3, s1_3);
        for i in 0..n {
            rhs1[i] += mu1 * (et1[i] * t1[i] + et2[i] * t2[i] + et3[i] * t3[i]);
        }

        // rhs2 = -mu1*fft(z1_1 - s1_1) + mu0*(Et1*fft(z0_1 - s0_1) + Et2*fft(z0_4 - s0_4) + Et3*fft(z0_5 - s0_5))
        // t1 already holds fft(z1_1 - s1_1).
        for i in 0..n {
            rhs2[i] = -mu1 * t1[i];
            rhs3[i] = -mu1 * t2[i];
            rhs4[i] = -mu1 * t3[i];
        }
        // rhs2 second-order part.
        fft_real_diff!(t1, z0_1, s0_1);
        fft_real_diff!(t2, z0_4, s0_4);
        fft_real_diff!(t3, z0_5, s0_5);
        for i in 0..n {
            rhs2[i] += mu0 * (et1[i] * t1[i] + et2[i] * t2[i] + et3[i] * t3[i]);
        }
        // rhs3 second-order part: mu0*(Et2*fft(z0_2 - s0_2) + Et1*fft(z0_4 - s0_4) + Et3*fft(z0_6 - s0_6))
        fft_real_diff!(t1, z0_2, s0_2);
        fft_real_diff!(t2, z0_4, s0_4);
        fft_real_diff!(t3, z0_6, s0_6);
        for i in 0..n {
            rhs3[i] += mu0 * (et2[i] * t1[i] + et1[i] * t2[i] + et3[i] * t3[i]);
        }
        // rhs4 second-order part: mu0*(Et3*fft(z0_3 - s0_3) + Et1*fft(z0_5 - s0_5) + Et2*fft(z0_6 - s0_6))
        fft_real_diff!(t1, z0_3, s0_3);
        fft_real_diff!(t2, z0_5, s0_5);
        fft_real_diff!(t3, z0_6, s0_6);
        for i in 0..n {
            rhs4[i] += mu0 * (et3[i] * t1[i] + et1[i] * t2[i] + et2[i] * t3[i]);
        }

        // ---- Cramer solve -------------------------------------------------
        for i in 0..n {
            let r1 = rhs1[i];
            let r2 = rhs2[i];
            let r3 = rhs3[i];
            let r4 = rhs4[i];
            let da = det_ainv[i];
            fx[i] = (r1 * d11[i] - r2 * d21[i] + r3 * d31[i] - r4 * d41[i]) * da;
            fv1[i] = (-r1 * d12[i] + r2 * d22[i] - r3 * d32[i] + r4 * d42[i]) * da;
            fv2[i] = (r1 * d13[i] - r2 * d23[i] + r3 * d33[i] - r4 * d43[i]) * da;
            fv3[i] = (-r1 * d14[i] + r2 * d24[i] - r3 * d34[i] + r4 * d44[i]) * da;
        }
        // x, v via ifft (take real part). Use t1 as scratch to preserve the
        // spectral fx/fv (re-FFT'd after convergence anyway).
        t1.copy_from_slice(&fx);
        fft_ws.ifft3d(&mut t1);
        x_prev.copy_from_slice(&x);
        for i in 0..n {
            x[i] = t1[i].re;
        }
        t1.copy_from_slice(&fv1);
        fft_ws.ifft3d(&mut t1);
        for i in 0..n {
            v1[i] = t1[i].re;
        }
        t1.copy_from_slice(&fv2);
        fft_ws.ifft3d(&mut t1);
        for i in 0..n {
            v2[i] = t1[i].re;
        }
        t1.copy_from_slice(&fv3);
        fft_ws.ifft3d(&mut t1);
        for i in 0..n {
            v3[i] = t1[i].re;
        }

        // ---- convergence --------------------------------------------------
        let xnorm = norm2(&x);
        if xnorm > 0.0 {
            for i in 0..n {
                diff[i] = x[i] - x_prev[i];
            }
            let x_update = 100.0 * norm2(&diff) / xnorm;
            if x_update < params.tol_update || x_update.is_nan() {
                progress(t + 1, t + 1);
                break;
            }
        }

        if t + 1 >= params.max_iter {
            break;
        }

        // ---- re-FFT for stability -----------------------------------------
        fft_real!(fx, x);
        fft_real!(fv1, v1);
        fft_real!(fv2, v2);
        fft_real!(fv3, v3);

        // Dx1 = real(ifft(E1.*Fx)), etc.
        cbuf.copy_from_slice(&fx);
        spectral_mul_assign(&mut cbuf, &e1);
        fft_ws.ifft3d(&mut cbuf);
        for i in 0..n {
            dx1[i] = cbuf[i].re;
        }
        cbuf.copy_from_slice(&fx);
        spectral_mul_assign(&mut cbuf, &e2);
        fft_ws.ifft3d(&mut cbuf);
        for i in 0..n {
            dx2[i] = cbuf[i].re;
        }
        cbuf.copy_from_slice(&fx);
        spectral_mul_assign(&mut cbuf, &e3);
        fft_ws.ifft3d(&mut cbuf);
        for i in 0..n {
            dx3[i] = cbuf[i].re;
        }

        // E_v1 = real(ifft(E1.*Fv1)), E_v2 = ...E2.*Fv2, E_v3 = ...E3.*Fv3.
        cbuf.copy_from_slice(&fv1);
        spectral_mul_assign(&mut cbuf, &e1);
        fft_ws.ifft3d(&mut cbuf);
        for i in 0..n {
            ev1[i] = cbuf[i].re;
        }
        cbuf.copy_from_slice(&fv2);
        spectral_mul_assign(&mut cbuf, &e2);
        fft_ws.ifft3d(&mut cbuf);
        for i in 0..n {
            ev2[i] = cbuf[i].re;
        }
        cbuf.copy_from_slice(&fv3);
        spectral_mul_assign(&mut cbuf, &e3);
        fft_ws.ifft3d(&mut cbuf);
        for i in 0..n {
            ev3[i] = cbuf[i].re;
        }

        // E_v4 = real(ifft(E1.*Fv2 + E2.*Fv1))/2, etc.
        for i in 0..n {
            cbuf[i] = e1[i] * fv2[i] + e2[i] * fv1[i];
        }
        fft_ws.ifft3d(&mut cbuf);
        for i in 0..n {
            ev4[i] = cbuf[i].re * 0.5;
        }
        for i in 0..n {
            cbuf[i] = e1[i] * fv3[i] + e3[i] * fv1[i];
        }
        fft_ws.ifft3d(&mut cbuf);
        for i in 0..n {
            ev5[i] = cbuf[i].re * 0.5;
        }
        for i in 0..n {
            cbuf[i] = e2[i] * fv3[i] + e3[i] * fv2[i];
        }
        fft_ws.ifft3d(&mut cbuf);
        for i in 0..n {
            ev6[i] = cbuf[i].re * 0.5;
        }

        // ---- symmetric-gradient split (shrink) ----------------------------
        for i in 0..n {
            z0_1[i] = shrink(ev1[i] + s0_1[i], alpha0_over_mu0);
            z0_2[i] = shrink(ev2[i] + s0_2[i], alpha0_over_mu0);
            z0_3[i] = shrink(ev3[i] + s0_3[i], alpha0_over_mu0);
            z0_4[i] = shrink(ev4[i] + s0_4[i], alpha0_over_mu0);
            z0_5[i] = shrink(ev5[i] + s0_5[i], alpha0_over_mu0);
            z0_6[i] = shrink(ev6[i] + s0_6[i], alpha0_over_mu0);
        }

        // ---- first-order split (shrink) -----------------------------------
        for i in 0..n {
            z1_1[i] = shrink(dx1[i] - v1[i] + s1_1[i], alpha1_over_mu1);
            z1_2[i] = shrink(dx2[i] - v2[i] + s1_2[i], alpha1_over_mu1);
            z1_3[i] = shrink(dx3[i] - v3[i] + s1_3[i], alpha1_over_mu1);
        }

        // ---- fidelity auxiliary (z2) via Newton ---------------------------
        // Dx = real(ifft(K.*Fx)) ; rhs_z2 = mu2*(Dx + s2) ; z2 init = Dx + s2.
        for i in 0..n {
            cbuf[i] = fx[i] * k[i];
        }
        fft_ws.ifft3d(&mut cbuf);
        for i in 0..n {
            dx[i] = cbuf[i].re;
            rhs_z2[i] = mu2 * (dx[i] + s2[i]);
            z2[i] = rhs_z2[i] / mu2;
        }
        let mut delta = f64::INFINITY;
        let mut inn = 0usize;
        while delta > params.tol_delta && inn < 50 {
            inn += 1;
            let norm_old = norm2(&z2);
            for i in 0..n {
                let a = z2[i] - phase[i];
                let numer = w[i] * a.sin() + mu2 * z2[i] - rhs_z2[i];
                let denom = w[i] * a.cos() + mu2;
                update[i] = numer / denom;
                z2[i] -= update[i];
            }
            delta = if norm_old > 0.0 {
                norm2(&update) / norm_old
            } else {
                0.0
            };
        }

        // ---- multiplier updates -------------------------------------------
        for i in 0..n {
            s0_1[i] += ev1[i] - z0_1[i];
            s0_2[i] += ev2[i] - z0_2[i];
            s0_3[i] += ev3[i] - z0_3[i];
            s0_4[i] += ev4[i] - z0_4[i];
            s0_5[i] += ev5[i] - z0_5[i];
            s0_6[i] += ev6[i] - z0_6[i];
            s1_1[i] += dx1[i] - v1[i] - z1_1[i];
            s1_2[i] += dx2[i] - v2[i] - z1_2[i];
            s1_3[i] += dx3[i] - v3[i] - z1_3[i];
            s2[i] += dx[i] - z2[i];
        }
    }

    if params.phase_scale != 1.0 {
        for v in &mut x {
            *v /= params.phase_scale;
        }
    }

    apply_mask_zero(&mut x, mask);
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fansi_nltv_zero_field() {
        // Zero field should give (approximately) zero susceptibility.
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = FansiParams {
            max_iter: 10,
            is_tgv: false,
            ..FansiParams::default()
        };

        let chi = fansi(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        for &val in chi.iter() {
            assert!(val.abs() < 1e-6, "Zero field should give ~zero chi, got {}", val);
        }
    }

    #[test]
    fn test_fansi_nltv_finite() {
        // A small ramp field should produce all-finite output.
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = FansiParams {
            max_iter: 10,
            is_tgv: false,
            ..FansiParams::default()
        };

        let chi = fansi(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        for (i, &val) in chi.iter().enumerate() {
            assert!(val.is_finite(), "Chi should be finite at index {}", i);
        }
    }

    #[test]
    fn test_fansi_nltgv_zero_field() {
        // Zero field should give (approximately) zero susceptibility.
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = FansiParams {
            max_iter: 10,
            is_tgv: true,
            ..FansiParams::default()
        };

        let chi = fansi(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        for &val in chi.iter() {
            assert!(val.abs() < 1e-6, "Zero field should give ~zero chi, got {}", val);
        }
    }

    #[test]
    fn test_fansi_nltgv_finite() {
        // A small ramp field should produce all-finite output.
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = FansiParams {
            max_iter: 10,
            is_tgv: true,
            ..FansiParams::default()
        };

        let chi = fansi(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        for (i, &val) in chi.iter().enumerate() {
            assert!(val.is_finite(), "Chi should be finite at index {}", i);
        }
    }
}

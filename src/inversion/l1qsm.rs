//! L1-QSM: nonlinear L1 data-fidelity dipole inversion with TV regularization
//!
//! Solves the nonlinear QSM inverse problem with an L1 data-fidelity term
//! (robust to phase outliers / noise) and a total-variation prior:
//!
//! min_x  || W .* ( exp(i * D x) - exp(i * phi) ) ||_1  +  alpha1 * ||grad(x)||_1
//!
//! where `D` is the dipole operator, `phi` is the (scaled) local field phase,
//! and `W = lambda * mask`. The nonlinear fidelity is linearized around the
//! current auxiliary phase and solved with an inner complex-argument Newton
//! iteration, all embedded in an ADMM splitting.
//!
//! Reference:
//! Milovic, C., Tejos, C., Acosta-Cabronero, J., et al. (2022).
//! "The 2016 QSM Challenge: L1-QSM and PI-QSM — comparison of nonlinear
//! L1 and phase-integral data-fidelity QSM reconstructions."
//! Magnetic Resonance in Medicine (MRM), 2022.
//!
//! Ported from the FANSI toolbox `nlL1TV.m`
//! (https://gitlab.com/cmilovic/FANSI-toolbox).

use crate::inversion::admm::prepare_fansi_spectral;
use crate::utils::gradient::{bdiv_inplace, fgrad_inplace};
use crate::utils::{apply_mask_zero, shrink};
use crate::Grid;
use num_complex::Complex64;

/// L1-QSM algorithm parameters.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct L1QsmParams {
    /// Gradient (TV) L1 penalty weight.
    pub alpha1: f64,
    /// Gradient-consistency ADMM weight (typically 100 * alpha1).
    pub mu1: f64,
    /// Fidelity-consistency ADMM weight.
    pub mu2: f64,
    /// L1 proximal ADMM weight.
    pub mu3: f64,
    /// L1 fidelity strength; effective weight is `lambda * mask`.
    pub lambda: f64,
    /// Number of outer ADMM iterations.
    pub max_iter: usize,
    /// Percent-update convergence stopping tolerance.
    pub tol_update: f64,
    /// Inner Newton convergence tolerance.
    pub tol_delta: f64,
    /// ppm -> working (phase) scale applied to the input local field.
    pub phase_scale: f64,
}

impl Default for L1QsmParams {
    fn default() -> Self {
        Self {
            alpha1: 2e-4,
            mu1: 2e-2,
            mu2: 1.0,
            mu3: 1.0,
            lambda: 1.0,
            max_iter: 50,
            tol_update: 1.0,
            tol_delta: 1e-6,
            phase_scale: 1.0,
        }
    }
}

/// L2 norm of a real slice.
fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|&a| a * a).sum::<f64>().sqrt()
}

/// L1-QSM nonlinear dipole inversion.
///
/// # Arguments
/// * `local_field` - Local field values, ppm-scale (nx * ny * nz).
/// * `mask` - Binary mask (nx * ny * nz), non-zero = inside ROI.
/// * `grid` - Volume grid (dimensions and voxel sizes).
/// * `bdir` - B0 field direction.
/// * `params` - L1-QSM parameters.
/// * `progress` - Progress callback `(iteration, max_iter)`.
///
/// # Returns
/// Estimated susceptibility map (ppm-scale), masked to the ROI.
pub fn l1qsm(
    local_field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &L1QsmParams,
    mut progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let n = grid.n_total();

    let (mut fft_ws, k, ee2) = prepare_fansi_spectral(grid, bdir);

    // Scaled phase and L1 fidelity weight W = lambda * mask.
    let phase: Vec<f64> = local_field.iter().map(|&f| f * params.phase_scale).collect();
    let w: Vec<f64> = mask
        .iter()
        .map(|&m| if m != 0 { params.lambda } else { 0.0 })
        .collect();

    // IS = exp(1i * phase)
    let is: Vec<Complex64> = phase
        .iter()
        .map(|&p| Complex64::new(p.cos(), p.sin()))
        .collect();

    let mu1 = params.mu1;
    let mu2 = params.mu2;
    let mu3 = params.mu3;
    let alpha_over_mu = params.alpha1 / mu1;

    // ADMM variables.
    let mut x = vec![0.0f64; n];
    let mut x_prev = vec![0.0f64; n];

    // Gradient-consistency split (real).
    let mut z_dx = vec![0.0f64; n];
    let mut z_dy = vec![0.0f64; n];
    let mut z_dz = vec![0.0f64; n];
    let mut s_dx = vec![0.0f64; n];
    let mut s_dy = vec![0.0f64; n];
    let mut s_dz = vec![0.0f64; n];

    // Fidelity-consistency split (real phase auxiliary) and its multiplier.
    // isPrecond=true: z2 = W .* phase / max(W)
    let w_max = w.iter().cloned().fold(0.0f64, f64::max);
    let mut z2 = vec![0.0f64; n];
    if w_max > 0.0 {
        for i in 0..n {
            z2[i] = w[i] * phase[i] / w_max;
        }
    }
    let mut s2 = vec![0.0f64; n];

    // L1 proximal split (complex) and its multiplier.
    let mut z3 = vec![Complex64::new(0.0, 0.0); n];
    let mut s3 = vec![Complex64::new(0.0, 0.0); n];

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

    for t in 0..params.max_iter {
        progress(t + 1, params.max_iter);

        // ---- x-subproblem (gradient consistency) --------------------------
        // gradient side: mu1 * sum(E_t .* fft(z_d - s_d)) == mu1 * fft(bdiv(z_d - s_d))
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

        // fidelity side: mu2 * conj(K) .* fft(z2 - s2); K real so conj(K) = K,
        // applied after the FFT below.
        for i in 0..n {
            fd2[i] = Complex64::new(z2[i] - s2[i], 0.0);
        }
        fft_ws.fft3d(&mut fd2);

        // Combine in k-space.
        for i in 0..n {
            let num = mu1 * fdiv[i] + mu2 * k[i] * fd2[i];
            // Guard the dipole null-space (DC/singular bins): both dipole kernel
            // and Laplacian vanish there. Zero it instead of dividing FFT
            // round-off by ~0, which would otherwise create a huge DC pedestal.
            let den = mu2 * k[i] * k[i] + mu1 * ee2[i];
            xhat[i] = if den > 1e-20 { num / den } else { Complex64::new(0.0, 0.0) };
        }
        fft_ws.ifft3d(&mut xhat);
        x_prev.copy_from_slice(&x);
        for i in 0..n {
            x[i] = xhat[i].re;
        }

        // ---- convergence check (percent update) ---------------------------
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
        // Fx = fft(x)  (reused for the dipole rhs below)
        for i in 0..n {
            fx[i] = Complex64::new(x[i], 0.0);
        }
        fft_ws.fft3d(&mut fx);

        // x_d{x,y,z} = fgrad(x)
        fgrad_inplace(&mut x_dx, &mut x_dy, &mut x_dz, &x, grid);
        for i in 0..n {
            z_dx[i] = shrink(x_dx[i] + s_dx[i], alpha_over_mu);
            z_dy[i] = shrink(x_dy[i] + s_dy[i], alpha_over_mu);
            z_dz[i] = shrink(x_dz[i] + s_dz[i], alpha_over_mu);
            s_dx[i] += x_dx[i] - z_dx[i];
            s_dy[i] += x_dy[i] - z_dy[i];
            s_dz[i] += x_dz[i] - z_dz[i];
        }

        // ---- L1 fidelity proximal (z3) ------------------------------------
        // Y3 = exp(1i*z2) - IS + s3 ;  z3 = soft_threshold_complex(Y3, W/mu3)
        for i in 0..n {
            let ez2 = Complex64::new(z2[i].cos(), z2[i].sin());
            let y3 = ez2 - is[i] + s3[i];
            let mag = y3.norm();
            let thr = w[i] / (mu3 + f64::EPSILON);
            let shr = (mag - thr).max(0.0);
            z3[i] = if mag > 0.0 {
                y3 * (shr / mag)
            } else {
                Complex64::new(0.0, 0.0)
            };
        }

        // ---- fidelity auxiliary (z2) via complex-arg Newton ---------------
        // rhs_z2 = mu2 * (Dx + s2), with Dx = real(ifft(K .* Fx))
        for i in 0..n {
            xhat[i] = fx[i] * k[i];
        }
        fft_ws.ifft3d(&mut xhat);
        for i in 0..n {
            dx[i] = xhat[i].re;
            rhs_z2[i] = mu2 * (dx[i] + s2[i]);
            // init z2 = rhs_z2 / mu2 (== Dx + s2)
            z2[i] = rhs_z2[i] / mu2;
        }

        // Newton per-voxel: minimize mu3*|exp(iz2)-(IS+z3-s3)| + mu2/2*(z2-...)^2
        // Uses complex-argument trigonometric linearization (see module notes).
        // yphase = angle(IS+z3-s3), ym = abs(IS+z3-s3), b = ln(ym).
        let mut yphase = vec![0.0f64; n];
        let mut cosh_b = vec![0.0f64; n];
        let mut sinh_b = vec![0.0f64; n];
        for i in 0..n {
            let yc = is[i] + z3[i] - s3[i];
            yphase[i] = yc.arg();
            let m = yc.norm();
            if m > 0.0 {
                cosh_b[i] = 0.5 * (m + 1.0 / m); // cosh(ln m)
                sinh_b[i] = 0.5 * (m - 1.0 / m); // sinh(ln m)
            } else {
                cosh_b[i] = 1.0;
                sinh_b[i] = 0.0;
            }
        }

        let mut delta = f64::INFINITY;
        let mut inn = 0usize;
        let mut update = vec![0.0f64; n];
        while delta > params.tol_delta && inn < 4 {
            inn += 1;
            let norm_old = norm2(&z2);

            for i in 0..n {
                let a = z2[i] - yphase[i];
                let (sa, ca) = a.sin_cos();
                // sin(a - i b) = sin(a)cosh(b) - i cos(a)sinh(b)
                let sin_arg = Complex64::new(sa * cosh_b[i], -ca * sinh_b[i]);
                // cos(a - i b) = cos(a)cosh(b) + i sin(a)sinh(b)
                let cos_arg = Complex64::new(ca * cosh_b[i], sa * sinh_b[i]);

                let temp = mu3 * cos_arg + Complex64::new(mu2 + f64::EPSILON, 0.0);
                let numer = mu3 * sin_arg + Complex64::new(mu2 * z2[i] - rhs_z2[i], 0.0);

                // max(abs(temp), 0.05) .* sign(temp): floor magnitude at 0.05.
                let tm = temp.norm();
                let denom = if tm < 0.05 {
                    if tm > 0.0 {
                        temp * (0.05 / tm)
                    } else {
                        Complex64::new(0.05, 0.0)
                    }
                } else {
                    temp
                };

                update[i] = (numer / denom).re;
                z2[i] -= update[i];
            }

            let delta_new = if norm_old > 0.0 {
                norm2(&update) / norm_old
            } else {
                0.0
            };
            if delta_new > delta {
                break;
            }
            delta = delta_new;
        }

        // ---- multiplier updates -------------------------------------------
        // s2 = s2 + Dx - z2
        for i in 0..n {
            s2[i] += dx[i] - z2[i];
        }
        // s3 = exp(1i*z2) - IS + s3 - z3
        for i in 0..n {
            let ez2 = Complex64::new(z2[i].cos(), z2[i].sin());
            s3[i] = ez2 - is[i] + s3[i] - z3[i];
        }
    }

    // Undo working-scale so the output is ppm-scale.
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
    fn test_l1qsm_zero_field() {
        // Zero field should give (approximately) zero susceptibility.
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = L1QsmParams {
            max_iter: 10,
            ..L1QsmParams::default()
        };

        let chi = l1qsm(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        for &val in chi.iter() {
            assert!(val.abs() < 1e-6, "Zero field should give ~zero chi, got {}", val);
        }
    }

    #[test]
    fn test_l1qsm_finite() {
        // A small ramp field should produce all-finite output.
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = L1QsmParams {
            max_iter: 10,
            ..L1QsmParams::default()
        };

        let chi = l1qsm(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        for (i, &val) in chi.iter().enumerate() {
            assert!(val.is_finite(), "Chi should be finite at index {}", i);
        }
    }
}

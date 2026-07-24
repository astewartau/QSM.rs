//! Weak-Harmonic QSM (WH-QSM) dipole inversion
//!
//! Jointly estimates the susceptibility map `x` AND a residual harmonic
//! background field `phi_h`, so that any harmonic (Laplacian-null) field
//! remaining after background-field removal is absorbed into `phi_h` instead
//! of corrupting the susceptibility estimate. This makes the reconstruction
//! robust to imperfect background-field removal.
//!
//! The optimization solves a nonlinear total-variation problem via ADMM,
//! with an additional weak-harmonic regularization term coupling `x` and the
//! harmonic field. The data-fidelity term is nonlinear (sine of the phase),
//! and is solved with an inner Newton iteration.
//!
//! Reference:
//! Milovic, C., Bilgic, B., Zhao, B., Acosta-Cabronero, J., Tejos, C. (2019).
//! "Weak-harmonic regularization for quantitative susceptibility mapping."
//! Magnetic Resonance in Medicine, 81(2):1399-1411.
//! <https://doi.org/10.1002/mrm.27483>
//!
//! Ported from FANSI's `WH_nlTV.m`.
//!
//! # Units / `phase_scale`
//! The nonlinear data term operates on a wrapped phase. The input
//! `local_field` is multiplied by `params.phase_scale` to form the internal
//! `phase`, and the final susceptibility map is divided by `phase_scale`. For
//! field data already in ppm-consistent units (as the rest of QSM.rs assumes),
//! use `phase_scale = 1.0`. When the input is a raw radians phase and a
//! ppm-scaled output is desired, set `phase_scale` to the radians→ppm factor.

use crate::inversion::admm::prepare_fansi_spectral;
use crate::utils::gradient::{bdiv_inplace, fgrad_inplace};
use crate::utils::{apply_mask_zero, shrink};
use crate::Grid;
use num_complex::Complex64;

/// WH-QSM parameters.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct WhQsmParams {
    /// TV regularization weight.
    pub alpha1: f64,
    /// ADMM penalty for the TV splitting (mu).
    pub mu1: f64,
    /// ADMM penalty for the data-fidelity splitting.
    pub mu2: f64,
    /// Weak-harmonic ROI penalty (constrains the harmonic field inside the mask).
    pub beta: f64,
    /// ADMM penalty for the harmonic-field splitting.
    pub muh: f64,
    /// Maximum outer iterations.
    pub max_iter: usize,
    /// Percent-update stopping tolerance on `x`.
    pub tol_update: f64,
    /// Inner Newton stopping tolerance.
    pub tol_delta: f64,
    /// Phase scaling factor (see module docs).
    pub phase_scale: f64,
}

impl Default for WhQsmParams {
    fn default() -> Self {
        Self {
            alpha1: 2e-4,
            mu1: 2e-2,
            mu2: 1.0,
            beta: 150.0,
            muh: 3.0,
            max_iter: 300,
            tol_update: 0.1,
            tol_delta: 1e-6,
            phase_scale: 1.0,
        }
    }
}

/// L2 norm of a real vector.
#[inline]
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|&a| a * a).sum::<f64>().sqrt()
}

/// Weak-Harmonic QSM dipole inversion.
///
/// Jointly estimates the susceptibility map and a residual harmonic
/// background field, returning the susceptibility map (`out.x`).
///
/// # Arguments
/// * `local_field` - Local field values (nx * ny * nz).
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI.
/// * `grid` - Volume grid (dimensions and voxel sizes).
/// * `bdir` - B0 field direction.
/// * `params` - WH-QSM parameters.
/// * `progress` - Progress callback `(iteration, max_iter)`.
///
/// # Returns
/// Susceptibility map.
pub fn whqsm(
    local_field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &WhQsmParams,
    mut progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let n = grid.n_total();

    let alpha = params.alpha1;
    let mu = params.mu1;
    let mu2 = params.mu2;
    let beta = params.beta;
    let muh = params.muh;
    let alpha_over_mu = alpha / mu;

    // Spectral operators: k = real dipole kernel D(k) (DC=0), ee2 = real Laplacian |E|^2.
    let (mut fft_ws, k, ee2) = prepare_fansi_spectral(grid, bdir);

    // Weight W = mask (weight = mask, W = weight^2 = mask since 0/1).
    // mask_f: floating mask.
    let mask_f: Vec<f64> = mask.iter().map(|&m| m as f64).collect();
    let w: Vec<f64> = mask_f.clone();

    // phase = local_field * phase_scale
    let phase: Vec<f64> = local_field.iter().map(|&f| f * params.phase_scale).collect();

    // Denominator for x-subproblem: 1e-30 + mu2*k^2 + mu*ee2
    let x_denom: Vec<f64> = (0..n)
        .map(|i| 1e-30 + mu2 * k[i] * k[i] + mu * ee2[i])
        .collect();
    // Denominator for phi_h subproblem: eps + mu2 + muh*ee2^2
    let phi_denom: Vec<f64> = (0..n)
        .map(|i| 1e-30 + mu2 + muh * ee2[i] * ee2[i])
        .collect();

    // State variables.
    let mut x = vec![0.0f64; n];
    let mut z_dx = vec![0.0f64; n];
    let mut z_dy = vec![0.0f64; n];
    let mut z_dz = vec![0.0f64; n];
    let mut s_dx = vec![0.0f64; n];
    let mut s_dy = vec![0.0f64; n];
    let mut s_dz = vec![0.0f64; n];

    let mut phi_h = vec![0.0f64; n];
    let mut z_h = vec![0.0f64; n];
    let mut s_h = vec![0.0f64; n];

    // z2 = phase .* W ; s2 = 0
    let mut z2: Vec<f64> = (0..n).map(|i| phase[i] * w[i]).collect();
    let mut s2 = vec![0.0f64; n];

    // Scratch buffers.
    let mut cbuf = vec![Complex64::new(0.0, 0.0); n]; // general complex scratch
    let mut cbuf2 = vec![Complex64::new(0.0, 0.0); n]; // second complex scratch
    let mut fx = vec![Complex64::new(0.0, 0.0); n]; // Fx = fft(x)
    let mut real_scratch = vec![0.0f64; n];
    let mut gx = vec![0.0f64; n];
    let mut gy = vec![0.0f64; n];
    let mut gz = vec![0.0f64; n];
    let mut dg = vec![0.0f64; n]; // divergence buffer
    let mut dx_real = vec![0.0f64; n]; // Dx = real(ifft(k .* Fx))
    let mut ee2_phi = vec![0.0f64; n]; // real(ifft(EE2 .* Fphi_h))

    for t in 0..params.max_iter {
        progress(t + 1, params.max_iter);

        // --- x-subproblem (masked) ---
        // numerator k-space = mu * fft(bdiv(z_d - s_d)) + K .* fft(z2 - s2 - phi_h)
        // x = mask .* real(ifft( numerator ./ x_denom ))

        // bdiv(z_d - s_d) in real space, then fft.
        for i in 0..n {
            gx[i] = z_dx[i] - s_dx[i];
            gy[i] = z_dy[i] - s_dy[i];
            gz[i] = z_dz[i] - s_dz[i];
        }
        bdiv_inplace(&mut dg, &gx, &gy, &gz, grid);
        for i in 0..n {
            cbuf[i] = Complex64::new(dg[i], 0.0);
        }
        fft_ws.fft3d(&mut cbuf); // cbuf = fft(bdiv(z_d - s_d))

        // Dt_kspace source in real space = z2 - s2 - phi_h ; fft it.
        for i in 0..n {
            fx[i] = Complex64::new(z2[i] - s2[i] - phi_h[i], 0.0);
        }
        fft_ws.fft3d(&mut fx); // fx = fft(z2 - s2 - phi_h)

        // numerator ./ x_denom  (k real -> K.* = k[i] * fx[i]; conj(K)=K since real)
        for i in 0..n {
            // Minus on the gradient term: adjoint of crate `fgrad` is `-bdiv` (matches
            // QSM.rs TV-ADMM). `+bdiv` doubles the effective regularization. See fansi.rs.
            let num = -mu * cbuf[i] + k[i] * fx[i];
            // Guard the dipole null-space (DC/singular bins): x_denom -> ~0 there
            // (both dipole kernel and Laplacian vanish). Zero it instead of
            // dividing FFT round-off by ~1e-30, which blows up and (via the
            // harmonic-field coupling) diverges the whole solve.
            cbuf[i] = if x_denom[i] > 1e-20 { num / x_denom[i] } else { Complex64::new(0.0, 0.0) };
        }
        fft_ws.ifft3d(&mut cbuf);

        // Compute x_update = 100 * norm(x - x_prev) / norm(x) after updating.
        // Keep old x in real_scratch for the diff.
        real_scratch.copy_from_slice(&x);
        for i in 0..n {
            x[i] = mask_f[i] * cbuf[i].re;
        }
        // x_update.
        let mut diff_norm = 0.0;
        for i in 0..n {
            let d = x[i] - real_scratch[i];
            diff_norm += d * d;
        }
        let diff_norm = diff_norm.sqrt();
        let x_norm = norm(&x);
        let x_update = if x_norm > 0.0 {
            100.0 * diff_norm / x_norm
        } else {
            // If ||x|| == 0, treat as no meaningful update (avoid div by zero).
            0.0
        };
        if x_update < params.tol_update {
            progress(t + 1, t + 1);
            break;
        }

        if t + 1 < params.max_iter {
            // Fx = fft(x)
            for i in 0..n {
                fx[i] = Complex64::new(x[i], 0.0);
            }
            fft_ws.fft3d(&mut fx);

            // Forward gradient of x (== [real(ifft(E1.*Fx)), ...]).
            fgrad_inplace(&mut gx, &mut gy, &mut gz, &x, grid);

            // z_d = shrink(grad + s_d, alpha_over_mu) ; s_d += grad - z_d.
            for i in 0..n {
                let ax = gx[i] + s_dx[i];
                let ay = gy[i] + s_dy[i];
                let az = gz[i] + s_dz[i];
                z_dx[i] = shrink(ax, alpha_over_mu);
                z_dy[i] = shrink(ay, alpha_over_mu);
                z_dz[i] = shrink(az, alpha_over_mu);
                s_dx[i] += gx[i] - z_dx[i];
                s_dy[i] += gy[i] - z_dy[i];
                s_dz[i] += gz[i] - z_dz[i];
            }

            // Dx = real(ifft(K .* Fx))
            for i in 0..n {
                cbuf[i] = k[i] * fx[i];
            }
            fft_ws.ifft3d(&mut cbuf);
            for i in 0..n {
                dx_real[i] = cbuf[i].re;
            }

            // rhs_z2 = mu2 * (Dx + s2 + phi_h) ; z2 = rhs_z2 / mu2 = Dx + s2 + phi_h.
            // Store rhs_z2 in real_scratch.
            for i in 0..n {
                real_scratch[i] = mu2 * (dx_real[i] + s2[i] + phi_h[i]);
                z2[i] = real_scratch[i] / mu2;
            }

            // Newton iteration on z2 (nonlinear data term).
            let mut delta = f64::INFINITY;
            let mut inn = 0;
            while delta > params.tol_delta && inn < 50 {
                inn += 1;
                let norm_old = norm(&z2);
                // update = (W.*sin(z2-phase) + mu2*z2 - rhs_z2) ./ (W.*cos(z2-phase) + mu2)
                let mut upd_norm2 = 0.0;
                for i in 0..n {
                    let dphi = z2[i] - phase[i];
                    let numr = w[i] * dphi.sin() + mu2 * z2[i] - real_scratch[i];
                    let denr = w[i] * dphi.cos() + mu2;
                    let u = numr / denr;
                    z2[i] -= u;
                    upd_norm2 += u * u;
                }
                let upd_norm = upd_norm2.sqrt();
                delta = if norm_old > 0.0 {
                    upd_norm / norm_old
                } else {
                    // guard: if z2 became zero, stop.
                    0.0
                };
            }

            // --- Harmonic field update (phi_h) ---
            // Fphi_h = ( muh*EE2.*fft(z_h - s_h) + mu2*fft(z2 - s2) - mu2*K.*Fx )
            //          ./ (eps + mu2 + muh*EE2.^2)
            // fft(z_h - s_h)
            for i in 0..n {
                cbuf[i] = Complex64::new(z_h[i] - s_h[i], 0.0);
            }
            fft_ws.fft3d(&mut cbuf); // cbuf = fft(z_h - s_h)

            // fft(z2 - s2)  -> reuse a fresh complex buffer via dx_real? Need another buffer.
            // Use `phi_h`-sized complex scratch: allocate once via a persistent buffer.
            // We reuse `cbuf` after combining; but we need fft(z2 - s2) simultaneously.
            // Compute fft(z2 - s2) into a second complex buffer.
            // (fx currently holds Fx; we still need Fx for the -mu2*K.*Fx term.)
            for i in 0..n {
                cbuf2[i] = Complex64::new(z2[i] - s2[i], 0.0);
            }
            fft_ws.fft3d(&mut cbuf2); // cbuf2 = fft(z2 - s2)

            // Fphi_h (store into cbuf).
            for i in 0..n {
                let numer =
                    muh * ee2[i] * cbuf[i] + mu2 * cbuf2[i] - mu2 * (k[i] * fx[i]);
                cbuf[i] = numer / phi_denom[i];
            }
            // cbuf now = Fphi_h.

            // EE2_phi = real(ifft(EE2 .* Fphi_h)) : first compute with EE2 multiply.
            for i in 0..n {
                cbuf2[i] = ee2[i] * cbuf[i];
            }
            fft_ws.ifft3d(&mut cbuf2);
            for i in 0..n {
                ee2_phi[i] = cbuf2[i].re;
            }

            // phi_h = real(ifft(Fphi_h))
            fft_ws.ifft3d(&mut cbuf);
            for i in 0..n {
                phi_h[i] = cbuf[i].re;
            }

            // z_h = muh*(EE2_phi + s_h) ./ (muh + beta*mask)   (mask 0/1)
            for i in 0..n {
                z_h[i] = muh * (ee2_phi[i] + s_h[i]) / (muh + beta * mask_f[i]);
            }

            // --- dual updates ---
            // s2 = s2 + real(ifft(K.*Fx)) - z2 + phi_h  (= s2 + Dx - z2 + phi_h)
            for i in 0..n {
                s2[i] = s2[i] + dx_real[i] - z2[i] + phi_h[i];
            }
            // s_h = s_h + EE2_phi - z_h
            for i in 0..n {
                s_h[i] = s_h[i] + ee2_phi[i] - z_h[i];
            }
        }
    }

    // Divide out phase_scale and apply mask.
    if params.phase_scale != 1.0 {
        let inv = 1.0 / params.phase_scale;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
    apply_mask_zero(&mut x, mask);

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whqsm_zero_field() {
        // Zero field should give (near) zero susceptibility.
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = WhQsmParams {
            max_iter: 15,
            ..Default::default()
        };

        let chi = whqsm(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        for &val in chi.iter() {
            assert!(
                val.abs() < 1e-6,
                "Zero field should give ~zero chi, got {}",
                val
            );
        }
    }

    #[test]
    fn test_whqsm_finite() {
        // Small ramp field -> all outputs finite.
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = WhQsmParams {
            max_iter: 15,
            ..Default::default()
        };

        let chi = whqsm(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        for (i, &val) in chi.iter().enumerate() {
            assert!(val.is_finite(), "Chi should be finite at index {}", i);
        }
    }
}

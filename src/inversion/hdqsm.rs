//! HD-QSM: Hybrid data-fidelity two-stage linear dipole inversion.
//!
//! A linear dipole-inversion method that runs in two stages. Stage 1 solves an
//! L1 data-fidelity problem (robust to phase/model errors) and derives a
//! spatially-varying discrepancy weighting. Stage 2 solves an L2 data-fidelity
//! problem reweighted by that discrepancy map. Both stages use ADMM with an
//! L1 total-variation regularizer.
//!
//! Because the whole method is linear and scale-consistent, feed the local field
//! in ppm directly and the output susceptibility is in ppm as well (no phase
//! scaling required).
//!
//! Reference:
//! Lambert, M., Tejos, C., Langkammer, C., et al. (2022).
//! "Hybrid data fidelity term approach for quantitative susceptibility mapping."
//! Magnetic Resonance in Medicine, 87(6):3059-3072.
//! https://doi.org/10.1002/mrm.29218
//!
//! Reference implementation: HDQSM.m, https://github.com/mglambert/HD-QSM (MIT).

use num_complex::Complex64;
use crate::inversion::admm::prepare_fansi_spectral;
use crate::utils::gradient::{bdiv_inplace, fgrad_inplace};
use crate::utils::{apply_mask_zero, shrink};
use crate::Grid;

/// HD-QSM algorithm parameters.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct HdQsmParams {
    /// L2-stage TV weight (default 1e-4).
    pub alpha_l2: f64,
    /// L2-stage gradient-consistency ADMM weight (default 1e-2 = 100*alpha_l2).
    ///
    /// The TV soft-threshold is `alpha_l2 / mu1_l2`; the FANSI HDQSM.m example's
    /// `mu1_l2 = 10*alpha_l2` locks that ratio at 0.1, which over-smooths on the
    /// ppm-scaled QSM-CI phantom (the README warns the example is calibrated to a
    /// different, normalized brain). We use `mu1_l2 = 100*alpha_l2` (ratio 0.01),
    /// matching QSM.rs's proven TV-ADMM balance — corr jumps from ~0.48 to ~0.84.
    pub mu1_l2: f64,
    /// Fidelity consistency weight (default 1.0).
    pub mu2: f64,
    /// Stage-1 (L1) iterations (default 20).
    pub max_iter_l1: usize,
    /// Stage-2 (L2) iterations (default 80).
    pub max_iter_l2: usize,
    /// Stage-2 percent-update stopping tolerance (default 1.0).
    pub tol_update: f64,
    // Stage-1 alpha_l1 / mu1_l1 default to sqrt(alpha_l2) / sqrt(mu1_l2) internally.
}

impl Default for HdQsmParams {
    fn default() -> Self {
        Self {
            alpha_l2: 1e-4,
            mu1_l2: 1e-2,
            mu2: 1.0,
            max_iter_l1: 20,
            max_iter_l2: 80,
            tol_update: 1.0,
        }
    }
}

/// L2 norm of a slice.
#[inline]
fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Percent update `100 * ||x - x_prev|| / ||x||`.
#[inline]
fn percent_update(x: &[f64], x_prev: &[f64]) -> f64 {
    let diff: f64 = x.iter().zip(x_prev).map(|(&a, &b)| (a - b) * (a - b)).sum::<f64>().sqrt();
    let nx = norm2(x);
    if nx > 0.0 { 100.0 * diff / nx } else { 0.0 }
}

/// Forward dipole convolution: `Dx = real(ifft(kernel .* fft(x)))`.
///
/// `cbuf` is scratch of length n_total; result written to `out`.
fn apply_dipole(
    fft_ws: &mut crate::fft::Fft3dWorkspace,
    k: &[f64],
    x: &[f64],
    cbuf: &mut [Complex64],
    out: &mut [f64],
) {
    for (c, &xv) in cbuf.iter_mut().zip(x.iter()) {
        *c = Complex64::new(xv, 0.0);
    }
    fft_ws.fft3d(cbuf);
    for (c, &kv) in cbuf.iter_mut().zip(k.iter()) {
        *c *= kv;
    }
    fft_ws.ifft3d(cbuf);
    for (o, c) in out.iter_mut().zip(cbuf.iter()) {
        *o = c.re;
    }
}

/// HD-QSM dipole inversion.
///
/// # Arguments
/// * `local_field` - Local field values in ppm (nx * ny * nz).
/// * `mask` - Binary ROI mask (nx * ny * nz), 1 = inside.
/// * `grid` - Volume grid (dimensions and voxel sizes).
/// * `bdir` - B0 field direction.
/// * `params` - HD-QSM parameters.
/// * `progress` - Progress callback `(global_iter, max_iter_l1 + max_iter_l2)`.
///
/// # Returns
/// Susceptibility map in ppm.
pub fn hdqsm(
    local_field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &HdQsmParams,
    mut progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let n = grid.n_total();
    let (mut fft_ws, k, ee2) = prepare_fansi_spectral(grid, bdir);
    // K2 = abs(kernel).^2 = k^2 (kernel is real). mu*EE2 is added per stage.
    let denom_base: Vec<f64> = k.iter()
        .map(|&kk| 1e-30 + params.mu2 * kk * kk)
        .collect();

    // weight = mask (as f64 in [0,1]).
    let weight_mask: Vec<f64> = mask.iter().map(|&m| if m != 0 { 1.0 } else { 0.0 }).collect();

    let total_iter = params.max_iter_l1 + params.max_iter_l2;
    let mut global_iter = 0usize;

    // Working buffers.
    let mut x = vec![0.0f64; n];
    let mut x_prev = vec![0.0f64; n];

    let mut z_dx = vec![0.0f64; n];
    let mut z_dy = vec![0.0f64; n];
    let mut z_dz = vec![0.0f64; n];
    let mut s_dx = vec![0.0f64; n];
    let mut s_dy = vec![0.0f64; n];
    let mut s_dz = vec![0.0f64; n];
    let mut x_dx = vec![0.0f64; n];
    let mut x_dy = vec![0.0f64; n];
    let mut x_dz = vec![0.0f64; n];

    let mut z2 = vec![0.0f64; n];
    let mut s2 = vec![0.0f64; n];
    let mut dx = vec![0.0f64; n]; // Dx = real(ifft(K.*Fx))

    let mut div = vec![0.0f64; n];
    let mut gx = vec![0.0f64; n];
    let mut gy = vec![0.0f64; n];
    let mut gz = vec![0.0f64; n];

    let mut cbuf = vec![Complex64::new(0.0, 0.0); n];
    let mut fdiv = vec![Complex64::new(0.0, 0.0); n];
    let mut fdt = vec![Complex64::new(0.0, 0.0); n];

    // ---- Helper closures cannot capture &mut fft_ws + buffers simultaneously,
    //      so the x-subproblem is inlined below. ----

    // =========================================================================
    // STAGE 1: L1 data fidelity.
    // =========================================================================
    {
        let mu = params.mu1_l2.sqrt();
        let alpha = params.alpha_l2.sqrt();
        let mu2 = params.mu2;
        let ll = alpha / mu;

        // Denominator for stage 1: eps + mu2*K2 + mu*EE2.
        let denom: Vec<f64> = (0..n).map(|i| denom_base[i] + mu * ee2[i]).collect();

        // Wy = input.
        let wy = local_field;

        for t in 0..params.max_iter_l1 {
            x_prev.copy_from_slice(&x);

            // Dt_kspace_arg = z2 - s2 + Wy.
            for i in 0..n {
                gx[i] = z2[i] - s2[i] + wy[i]; // reuse gx as dt_arg scratch
            }
            // fft(dt_arg) -> fdt
            for (c, &v) in fdt.iter_mut().zip(gx.iter()) {
                *c = Complex64::new(v, 0.0);
            }
            fft_ws.fft3d(&mut fdt);

            // fft(bdiv(z_d - s_d)) -> fdiv
            for i in 0..n {
                gx[i] = z_dx[i] - s_dx[i];
                gy[i] = z_dy[i] - s_dy[i];
                gz[i] = z_dz[i] - s_dz[i];
            }
            bdiv_inplace(&mut div, &gx, &gy, &gz, grid);
            for (c, &v) in fdiv.iter_mut().zip(div.iter()) {
                *c = Complex64::new(v, 0.0);
            }
            fft_ws.fft3d(&mut fdiv);

            // num = mu*fdiv + mu2*conj(K)*fdt  (K real -> conj(K)=K); divide by denom.
            // Guard the dipole null-space (DC/singular bins): denom -> ~0 there
            // (dipole kernel and Laplacian both vanish). Zero it instead of
            // dividing FFT round-off by ~0, which would create a huge DC pedestal.
            for i in 0..n {
                // Minus on the gradient term: adjoint of crate `fgrad` is `-bdiv` (matches
                // QSM.rs TV-ADMM). `+bdiv` doubles the effective regularization. See fansi.rs.
                let num = -fdiv[i] * mu + fdt[i] * (mu2 * k[i]);
                cbuf[i] = if denom[i] > 1e-20 { num / denom[i] } else { Complex64::new(0.0, 0.0) };
            }
            fft_ws.ifft3d(&mut cbuf);
            for i in 0..n {
                x[i] = cbuf[i].re;
            }

            global_iter += 1;
            progress(global_iter, total_iter);

            if t < params.max_iter_l1 - 1 {
                // fgrad(x) -> x_d
                fgrad_inplace(&mut x_dx, &mut x_dy, &mut x_dz, &x, grid);
                // z_d = shrink(x_d + s_d, ll); s_d += x_d - z_d.
                for i in 0..n {
                    let ax = x_dx[i] + s_dx[i];
                    let ay = x_dy[i] + s_dy[i];
                    let az = x_dz[i] + s_dz[i];
                    z_dx[i] = shrink(ax, ll);
                    z_dy[i] = shrink(ay, ll);
                    z_dz[i] = shrink(az, ll);
                    s_dx[i] += x_dx[i] - z_dx[i];
                    s_dy[i] += x_dy[i] - z_dy[i];
                    s_dz[i] += x_dz[i] - z_dz[i];
                }

                // Dx = real(ifft(K.*Fx))
                apply_dipole(&mut fft_ws, &k, &x, &mut cbuf, &mut dx);
                // z2_inner = Dx + s2 - Wy; z2 = shrink(z2_inner, weight/mu2); s2 = z2_inner - z2.
                for i in 0..n {
                    let z2_inner = dx[i] + s2[i] - wy[i];
                    z2[i] = shrink(z2_inner, weight_mask[i] / mu2);
                    s2[i] = z2_inner - z2[i];
                }
            }
        }
    }

    // Discrepancy factor: dphi = ifft(fft(input) - fft(x).*K); dphi = abs(dphi).*mask; dphi /= max.
    // = |input - Dx| .* mask, normalized.  (fft(input) - K.*fft(x) -> ifft = input - Dx)
    let mut dphi = vec![0.0f64; n];
    {
        // Dx = real(ifft(K.*Fx)) with current x.
        apply_dipole(&mut fft_ws, &k, &x, &mut cbuf, &mut dx);
        let mut maxv = 0.0f64;
        for i in 0..n {
            let v = (local_field[i] - dx[i]).abs() * (if mask[i] != 0 { 1.0 } else { 0.0 });
            dphi[i] = v;
            if v > maxv {
                maxv = v;
            }
        }
        if maxv > 0.0 {
            for v in dphi.iter_mut() {
                *v /= maxv;
            }
        }
    }

    // =========================================================================
    // STAGE 2: L2 data fidelity, reweighted by discrepancy.
    // =========================================================================
    {
        let mu = params.mu1_l2;
        let alpha = params.alpha_l2;
        let mu2 = params.mu2;
        let ll = alpha / mu;

        // Denominator for stage 2: eps + mu2*K2 + mu*EE2.
        let denom: Vec<f64> = (0..n).map(|i| denom_base[i] + mu * ee2[i]).collect();

        // weight = weight.*weight.*(1-dphi).*(1-dphi).
        let weight: Vec<f64> = (0..n).map(|i| {
            let w = weight_mask[i];
            let d = 1.0 - dphi[i];
            w * w * d * d
        }).collect();
        // Wy = weight.*input./(weight+mu2).
        let wy: Vec<f64> = (0..n).map(|i| weight[i] * local_field[i] / (weight[i] + mu2)).collect();

        // Reset dual/aux for stage 2 (x carried over).
        for i in 0..n {
            z_dx[i] = 0.0; z_dy[i] = 0.0; z_dz[i] = 0.0;
            s_dx[i] = 0.0; s_dy[i] = 0.0; s_dz[i] = 0.0;
            s2[i] = 0.0;
        }

        for _t in 0..params.max_iter_l2 {
            x_prev.copy_from_slice(&x);

            // z/s updates BEFORE the x-update (unlike stage 1).
            fgrad_inplace(&mut x_dx, &mut x_dy, &mut x_dz, &x, grid);
            for i in 0..n {
                let ax = x_dx[i] + s_dx[i];
                let ay = x_dy[i] + s_dy[i];
                let az = x_dz[i] + s_dz[i];
                z_dx[i] = shrink(ax, ll);
                z_dy[i] = shrink(ay, ll);
                z_dz[i] = shrink(az, ll);
                s_dx[i] += x_dx[i] - z_dx[i];
                s_dy[i] += x_dy[i] - z_dy[i];
                s_dz[i] += x_dz[i] - z_dz[i];
            }

            // Dx = real(ifft(K.*Fx))
            apply_dipole(&mut fft_ws, &k, &x, &mut cbuf, &mut dx);
            // z2 = Wy + mu2*(Dx + s2)./(weight + mu2); s2 = s2 + Dx - z2.
            for i in 0..n {
                z2[i] = wy[i] + mu2 * (dx[i] + s2[i]) / (weight[i] + mu2);
                s2[i] = s2[i] + dx[i] - z2[i];
            }

            // x update. Dt_kspace_arg = z2 - s2.
            for i in 0..n {
                gx[i] = z2[i] - s2[i];
            }
            for (c, &v) in fdt.iter_mut().zip(gx.iter()) {
                *c = Complex64::new(v, 0.0);
            }
            fft_ws.fft3d(&mut fdt);

            for i in 0..n {
                gx[i] = z_dx[i] - s_dx[i];
                gy[i] = z_dy[i] - s_dy[i];
                gz[i] = z_dz[i] - s_dz[i];
            }
            bdiv_inplace(&mut div, &gx, &gy, &gz, grid);
            for (c, &v) in fdiv.iter_mut().zip(div.iter()) {
                *c = Complex64::new(v, 0.0);
            }
            fft_ws.fft3d(&mut fdiv);

            for i in 0..n {
                // Minus on the gradient term: adjoint of crate `fgrad` is `-bdiv` (matches
                // QSM.rs TV-ADMM). `+bdiv` doubles the effective regularization. See fansi.rs.
                let num = -fdiv[i] * mu + fdt[i] * (mu2 * k[i]);
                // Guard the dipole null-space (DC/singular bins) — see stage 1.
                cbuf[i] = if denom[i] > 1e-20 { num / denom[i] } else { Complex64::new(0.0, 0.0) };
            }
            fft_ws.ifft3d(&mut cbuf);
            for i in 0..n {
                x[i] = cbuf[i].re;
            }

            global_iter += 1;
            progress(global_iter, total_iter);

            let upd = percent_update(&x, &x_prev);
            if upd < params.tol_update {
                break;
            }
        }
    }

    // Ensure progress reaches the total even if stage 2 stopped early.
    if global_iter < total_iter {
        progress(total_iter, total_iter);
    }

    apply_mask_zero(&mut x, mask);
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdqsm_zero_field() {
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = HdQsmParams {
            max_iter_l1: 5,
            max_iter_l2: 10,
            ..HdQsmParams::default()
        };

        let chi = hdqsm(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        for &val in chi.iter() {
            assert!(val.abs() < 1e-6, "Zero field should give zero chi, got {}", val);
        }
    }

    #[test]
    fn test_hdqsm_finite() {
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = HdQsmParams {
            max_iter_l1: 5,
            max_iter_l2: 10,
            ..HdQsmParams::default()
        };

        let chi = hdqsm(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        for (i, &val) in chi.iter().enumerate() {
            assert!(val.is_finite(), "Chi should be finite at index {}", i);
        }
    }
}

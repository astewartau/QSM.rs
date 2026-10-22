//! Nonlinear Dipole Inversion (NDI) for QSM
//!
//! Solves the dipole inversion problem with a nonlinear data-consistency term,
//! modelling the wrapped phase directly via `sin(Dx - phase)` and minimizing it
//! with a simple L2-regularized gradient descent:
//!
//! min_x || W .* (exp(i*Dx) - exp(i*phase)) ||^2 + alpha ||x||^2
//!
//! which yields the update
//!
//! x <- x - tau * D^H(W^2 .* sin(Dx - phase)) - tau*alpha*x
//!
//! `W` is the data weight: the magnitude, normalised as in FANSI's in vivo
//! example (`min(2*mask*mag/max(mag), 1)`). The magnitude is required: with a
//! uniform weight the update can diverge where the phase nears ±π.
//!
//! Because the dipole kernel `D` is real, `conj(D) = D` and the adjoint is just
//! another forward dipole application.
//!
//! Reference:
//! Polak, D., Chatnuntawech, I., Yoon, J., Iyer, S.S., Milovic, C., Lee, J.,
//! Bachert, P., Adalsteinsson, E., Setsompop, K., Bilgic, B. (2020).
//! "Nonlinear dipole inversion (NDI) enables robust quantitative susceptibility
//! mapping (QSM)." NMR in Biomedicine, 33(12):e4271.
//! https://doi.org/10.1002/nbm.4271
//!
//! Ported from FANSI's `ndi.m`, and `ndi_auto.m` for the automatic stopping
//! criterion (Milovic et al., ISMRM 2021, p. 3982)
//! (https://gitlab.com/cmilovic/FANSI-toolbox).
//!
//! # Units
//! Like the other inversions, NDI takes the local field in ppm and returns ppm.
//! FANSI's solver works on the phase in radians, so the field is converted with
//! `2*pi*gamma*b0*te` first and the result converted back at the end; left in
//! ppm, `sin(x) ~ x` and the data term would not be nonlinear. `b0` and `te`
//! are scan parameters that the pipeline fills from metadata.

use crate::fft::Fft3dWorkspace;
use crate::utils::apply_mask_zero;
use crate::Grid;
use super::admm::prepare_fansi_spectral;
use num_complex::Complex64;

/// NDI algorithm parameters
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct NdiParams {
    /// Gradient-descent step size
    pub tau: f64,
    /// L2 regularization weight
    pub alpha: f64,
    /// Number of iterations (the maximum, with `auto_stop`)
    pub max_iter: usize,
    /// Stop once the high-frequency energy of the estimate along the dipole cone exceeds that
    /// just off it (FANSI `ndi_auto.m`)
    pub auto_stop: bool,
    /// Field strength in Tesla (scan parameter, for the radian conversion)
    pub b0: f64,
    /// Echo time in seconds (scan parameter, for the radian conversion)
    pub te: f64,
}

impl Default for NdiParams {
    /// FANSI `ndi.m` defaults.
    fn default() -> Self {
        Self {
            tau: 2.0,
            alpha: 1e-5,
            max_iter: 100,
            auto_stop: false,
            b0: 3.0,
            te: 0.02,
        }
    }
}

impl NdiParams {
    /// FANSI `ndi_auto.m` defaults: automatic stopping, at most 500 iterations.
    pub fn auto() -> Self {
        Self {
            tau: 1.0,
            alpha: 1e-6,
            max_iter: 500,
            auto_stop: true,
            ..Self::default()
        }
    }
}

/// The two k-space bands `ndi_auto.m` compares (FANSI `create_freqmasks_auto.m`): a radial
/// shell of high frequencies, split by |D| into a band close to the cone (0.15 < |D| <= 0.2)
/// and one further off it (0.225 < |D| <= 0.275). `k` is the kernel in FFT order.
fn auto_stop_masks(grid: &Grid, k: &[f64]) -> (Vec<bool>, Vec<bool>) {
    let (nx, ny, nz) = grid.dims;
    let res = [grid.vsx(), grid.vsy(), grid.vsz()];
    let rmax = res.iter().cloned().fold(f64::MIN, f64::max);
    let (rin, rout) = (rmax * 0.65 / 2.0, rmax * 0.95 / 2.0);
    // MATLAB builds the shell on a centred grid ((1:N) - (1 + floor(N/2))) * res/N and then
    // fftshifts it; index i in FFT order is centred index (i + floor(N/2)) mod N.
    let coord = |i: usize, n: usize, r: f64| {
        let c = (i + n / 2) % n;
        (c as f64 - (n / 2) as f64) * r / n as f64
    };
    let mut m1 = vec![false; k.len()];
    let mut m2 = vec![false; k.len()];
    for z in 0..nz {
        let kz = coord(z, nz, res[2]);
        for y in 0..ny {
            let ky = coord(y, ny, res[1]);
            for x in 0..nx {
                let kx = coord(x, nx, res[0]);
                let k2 = kx * kx + ky * ky + kz * kz;
                if k2 > rout * rout || k2 < rin * rin {
                    continue;
                }
                let i = x + nx * (y + ny * z);
                let a = k[i].abs();
                m1[i] = a > 0.15 && a <= 0.2;
                m2[i] = a > 0.225 && a <= 0.275;
            }
        }
    }
    (m1, m2)
}

/// Apply the forward dipole operator `D * x` in-place buffers.
///
/// Equivalent to MATLAB `susc2field(kernel, x) = real(ifftn(D .* fftn(x)))`.
/// Since the dipole kernel `k` is real, `conj(k) = k`, so the same routine
/// computes the adjoint `D^H`.
fn apply_dipole(
    fft_ws: &mut Fft3dWorkspace,
    k: &[f64],
    x: &[f64],
    buf: &mut [Complex64],
    out: &mut [f64],
) {
    for i in 0..x.len() {
        buf[i] = Complex64::new(x[i], 0.0);
    }
    fft_ws.fft3d(buf);
    for i in 0..x.len() {
        buf[i] *= k[i];
    }
    fft_ws.ifft3d(buf);
    for i in 0..x.len() {
        out[i] = buf[i].re;
    }
}

/// Nonlinear Dipole Inversion (NDI)
///
/// # Arguments
/// * `local_field` - Local field in ppm (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), non-zero = inside ROI
/// * `magnitude` - Magnitude image, the data weight (see module docs)
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `bdir` - B0 field direction
/// * `params` - NDI parameters
/// * `progress` - Progress callback `(iteration, max_iter)`
///
/// # Returns
/// Estimated susceptibility map (ppm-scale), masked to the ROI.
pub fn ndi(
    local_field: &[f64],
    mask: &[u8],
    magnitude: &[f64],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &NdiParams,
    mut progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let n = grid.n_total();

    let (mut fft_ws, k_kernel, _ee2) = prepare_fansi_spectral(grid, bdir);

    // ppm -> radians, as FANSI expects.
    // FANSI's scripts use `gyro = 2*pi*42.58`; QSM.rs uses GAMMA_BAR_MHZ_PER_T, which is closer
    // to the CODATA shielded-proton value (see `crate::constants`).
    let ppm_to_rad = 2.0 * std::f64::consts::PI * crate::constants::GAMMA_BAR_MHZ_PER_T * params.b0 * params.te;
    let phase: Vec<f64> = local_field.iter().map(|&f| f * ppm_to_rad).collect();

    // W = min(2*mask*mag/max(mag), 1) (FANSI script_invivo.m), squared as in ndi.m.
    let mag_max = magnitude.iter().cloned().fold(0.0f64, f64::max);
    let w: Vec<f64> = (0..n)
        .map(|i| {
            if mask[i] == 0 || mag_max <= 0.0 {
                return 0.0;
            }
            let v = (2.0 * magnitude[i] / mag_max).min(1.0);
            v * v
        })
        .collect();
    let stop_masks = params.auto_stop.then(|| auto_stop_masks(grid, &k_kernel));

    let mut x = vec![0.0f64; n];

    // Reusable buffers.
    let mut buf = vec![Complex64::new(0.0, 0.0); n];
    let mut phix = vec![0.0f64; n];
    let mut resid = vec![0.0f64; n];
    let mut grad = vec![0.0f64; n];

    for t in 0..params.max_iter {
        progress(t + 1, params.max_iter);

        // phix = D * x
        apply_dipole(&mut fft_ws, &k_kernel, &x, &mut buf, &mut phix);

        // resid = W .* sin(phix - phase)
        for i in 0..n {
            resid[i] = w[i] * (phix[i] - phase[i]).sin();
        }

        // grad = D^H * resid  (conj(kernel) = kernel, real)
        apply_dipole(&mut fft_ws, &k_kernel, &resid, &mut buf, &mut grad);

        // x <- x - tau*grad - tau*alpha*x
        for i in 0..n {
            x[i] -= params.tau * grad[i] + params.tau * params.alpha * x[i];
        }

        // ndi_auto.m: stop once the mean |FFT(x)| in the band near the cone exceeds the band
        // off it (MATLAB's `t > 3` is 1-based, i.e. from the fourth iteration on).
        if let Some((m1, m2)) = &stop_masks {
            for i in 0..n {
                buf[i] = Complex64::new(x[i], 0.0);
            }
            fft_ws.fft3d(&mut buf);
            let (mut s1, mut n1, mut s2, mut n2) = (0.0, 0usize, 0.0, 0usize);
            for i in 0..n {
                if m1[i] {
                    s1 += buf[i].norm();
                    n1 += 1;
                } else if m2[i] {
                    s2 += buf[i].norm();
                    n2 += 1;
                }
            }
            if t > 2 && n1 > 0 && n2 > 0 && s1 / n1 as f64 > s2 / n2 as f64 {
                break;
            }
        }
    }

    // Back to ppm.
    for v in &mut x {
        *v /= ppm_to_rad;
    }

    apply_mask_zero(&mut x, mask);
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndi_zero_field() {
        // Zero field should give zero susceptibility.
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = NdiParams { max_iter: 20, ..Default::default() };

        let chi = ndi(&field, &mask, &vec![1.0; n * n * n], &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        for &val in chi.iter() {
            assert!(val.abs() < 1e-6, "Zero field should give zero chi, got {}", val);
        }
    }

    #[test]
    fn test_ndi_finite() {
        // Result should be finite for a small ramp field.
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = NdiParams { max_iter: 20, ..Default::default() };

        let chi = ndi(&field, &mask, &vec![1.0; n * n * n], &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        for (i, &val) in chi.iter().enumerate() {
            assert!(val.is_finite(), "Chi should be finite at index {}", i);
        }
    }
}

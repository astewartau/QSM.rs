//! Nonlinear Dipole Inversion (NDI) for QSM
//!
//! Solves the dipole inversion problem with a nonlinear data-consistency term,
//! modelling the wrapped phase directly via `sin(Dx - phase)` and minimizing it
//! with a simple L2-regularized gradient descent:
//!
//! min_x || W .* (exp(i*Dx) - exp(i*phase)) ||^2 + alpha ||x||^2
//!
//! which yields the update (weight W = mask here):
//!
//! x <- x - tau * D^H(W .* sin(Dx - phase)) - tau*alpha*x
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
//! Ported from FANSI's `ndi.m`
//! (https://gitlab.com/cmilovic/FANSI-toolbox).
//!
//! # Note on `phase_scale`
//! FANSI's NDI operates on the phase in radians. In this crate the input local
//! field is provided at ppm-scale. With the default `phase_scale = 1.0` the
//! solver runs at the native ppm scale; for small ppm values `sin(x) ~ x`, so
//! the behaviour is numerically well-conditioned and scale-consistent with the
//! rest of the crate. A user may set `phase_scale` to convert the input to
//! radians (e.g. via the Hz->rad / ppm->rad factor for their acquisition) to
//! recover the true nonlinear behaviour; the output is divided back by the same
//! factor so the returned susceptibility stays at ppm-scale.

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
    /// Number of iterations
    pub max_iter: usize,
    /// ppm -> working-scale multiplier (see module docs)
    pub phase_scale: f64,
}

impl Default for NdiParams {
    fn default() -> Self {
        Self {
            tau: 2.0,
            alpha: 1e-5,
            max_iter: 200,
            phase_scale: 1.0,
        }
    }
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
        buf[i] = buf[i] * k[i];
    }
    fft_ws.ifft3d(buf);
    for i in 0..x.len() {
        out[i] = buf[i].re;
    }
}

/// Nonlinear Dipole Inversion (NDI)
///
/// # Arguments
/// * `local_field` - Local field values, ppm-scale (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), non-zero = inside ROI
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
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &NdiParams,
    mut progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let n = grid.n_total();

    let (mut fft_ws, k_kernel, _ee2) = prepare_fansi_spectral(grid, bdir);

    // Scaled phase and weight (W = mask.*mask = mask for binary mask).
    let phase: Vec<f64> = local_field.iter().map(|&f| f * params.phase_scale).collect();
    let w: Vec<f64> = mask
        .iter()
        .map(|&m| if m != 0 { 1.0 } else { 0.0 })
        .collect();

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
    }

    // Undo the working-scale so the output stays at ppm-scale.
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
    fn test_ndi_zero_field() {
        // Zero field should give zero susceptibility.
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = NdiParams { max_iter: 20, ..Default::default() };

        let chi = ndi(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

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

        let chi = ndi(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        for (i, &val) in chi.iter().enumerate() {
            assert!(val.is_finite(), "Chi should be finite at index {}", i);
        }
    }
}

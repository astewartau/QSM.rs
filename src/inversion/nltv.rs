//! Nonlinear total variation (NLTV) dipole inversion
//!
//! TV-regularised dipole inversion with a nonlinear data term on the wrapped phase:
//!
//! min_x || W (exp(i D x) - exp(i f)) ||_2^2 + lambda || grad x ||_1
//!
//! solved with ADMM. The data term gets its own auxiliary variable `z = D x`, whose
//! subproblem `W^2 sin(z - f) + mu (z - (D x + u)) = 0` is solved per voxel with Newton's
//! method.
//!
//! Reference:
//! Milovic, C., Bilgic, B., Zhao, B., Acosta-Cabronero, J., Tejos, C. (2018).
//! "Fast nonlinear susceptibility inversion with variational regularization."
//! Magnetic Resonance in Medicine, 80(2):814-821. https://doi.org/10.1002/mrm.27073
//!
//! Ported line for line from QSM.jl's `nltv` (`src/inversion/nltv.jl`,
//! https://github.com/kamesy/QSM.jl), including its defaults, its zero padding to a fast FFT
//! size, and its starting point (`z - u = grad f` for the TV term).
//!
//! Two places where this port has to choose, because QSM.jl's code is not well defined:
//! - QSM.jl's `W` is optional, but its unweighted path (`W = nothing`) never refreshes the FFT
//!   of `x` before forming `D x` for the Newton step, and with `W = 1` and `mu = 1` the Newton
//!   step's derivative `W^2/mu*cos(z - f) + 1` reaches zero: QSM.jl itself diverges on fields of
//!   more than a few radians (the qsm-forward 7T phantom). This port therefore requires the
//!   magnitude.
//! - QSM.jl takes the data weight `W` from the caller and gives no rule for making it. Here
//!   `W = min(2*mask*mag/max(mag), 1)`, the normalisation of FANSI's in vivo example by the same
//!   authors as the paper (and of [`super::ndi`]); QSM.jl squares it.
//! - QSM.jl allocates the data-term multiplier without initialising it; here it starts at zero.
//!
//! QSM.jl uses real FFTs, this port complex FFTs and the real part. When B0 is tilted from the
//! grid axes the dipole kernel is not symmetric on the Nyquist planes, and the two treat those
//! bins differently. That is the only difference between the two: about 1e-4 ppm RMS on a 3T
//! scan tilted by 6 degrees, and none for axial B0.
//!
//! # Units
//! Like the other inversions, NLTV takes the local field in ppm and returns ppm. The nonlinear
//! data term needs the phase in radians, so the field is converted with `2*pi*gamma*b0*te`
//! first and the result converted back at the end. `b0` and `echo_times` are scan parameters
//! that the pipeline fills from metadata, and [`NltvParams::te`] picks the echo time the
//! conversion uses (by default the reference's choice; see there).
//!
//! # Reference pipeline
//! QSM.jl's README example (commit fde4b92, L35-52), the pipeline its author ships for all its
//! inversions: brain mask by FSL BET on the last echo's magnitude (`-f 0.5`), eroded by 5
//! voxels; Laplacian unwrapping of each echo (which also removes harmonic background), each echo
//! scaled to ppm by its own TE; V-SHARP with QSM.jl's defaults (radii `18*min(vsz)` down to
//! `2*max(vsz)` mm in steps of `2*max(vsz)`, threshold 0.05; `src/bgremove/sharp.jl` L136-137).
//! QSM.jl's inversions take 3D or 4D (per-echo) fields; the pipeline here inverts one combined
//! field.

use crate::utils::gradient::{bdiv_inplace, fgrad_inplace};
use crate::utils::padding::{pad3d, unpad3d};
use crate::utils::apply_mask_zero;
use crate::{EchoTime, Grid};
use super::admm::prepare_fansi_spectral;
use num_complex::Complex64;

/// NLTV algorithm parameters (QSM.jl `nltv` defaults)
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct NltvParams {
    /// TV regularisation parameter
    pub lambda: f64,
    /// ADMM penalty for the TV term; `None` uses QSM.jl's default, `100 * lambda`
    pub rho: Option<f64>,
    /// ADMM penalty for the data term
    pub mu: f64,
    /// Stopping tolerance on ||x - x_prev|| / ||x||
    pub tol: f64,
    /// Maximum ADMM iterations
    pub max_iter: usize,
    /// Stopping tolerance for the Newton iterations of the data subproblem
    pub tol_newton: f64,
    /// Maximum Newton iterations per ADMM iteration
    pub max_iter_newton: usize,
    /// Field strength in Tesla (scan parameter, for the radian conversion)
    pub b0: f64,
    /// Echo times in seconds, in acquisition order (scan parameter, for the radian
    /// conversion).
    pub echo_times: Vec<f64>,
    /// Which echo time the radian conversion uses. Default [`EchoTime::First`]: QSM.jl's `nltv`
    /// (`src/inversion/nltv.jl`, fde4b92) takes the field or phase of each echo separately and
    /// gives no rule for a field combined over echoes, so the first echo time, the default where
    /// the reference does not say.
    pub te: EchoTime,
}

impl Default for NltvParams {
    fn default() -> Self {
        Self {
            lambda: 1e-3,
            rho: None,
            mu: 1.0,
            tol: 1e-3,
            max_iter: 250,
            tol_newton: 1e-6,
            max_iter_newton: 10,
            b0: 3.0,
            echo_times: vec![0.02],
            te: EchoTime::First,
        }
    }
}

/// Next size >= n whose only prime factors are 2, 3, 5, 7 (Julia `nextprod((2,3,5,7), n)`).
fn next_fast_fft(n: usize) -> usize {
    let mut s = n.max(1);
    loop {
        let mut m = s;
        for p in [2, 3, 5, 7] {
            while m.is_multiple_of(p) {
                m /= p;
            }
        }
        if m == 1 {
            return s;
        }
        s += 1;
    }
}

/// QSM.jl `fastfftsize(sz, (0, 0, 0); rfft=true)`: the fast FFT size, with an even first
/// dimension where one is close.
fn fast_fft_dims(dims: (usize, usize, usize)) -> (usize, usize, usize) {
    let s0 = next_fast_fft(dims.0);
    let mut s = s0;
    for _ in 0..3 {
        if s.is_multiple_of(2) {
            break;
        }
        s = next_fast_fft(s + 1);
    }
    let sx = if s.is_multiple_of(2) { s } else { s0 + 1 };
    (sx, next_fast_fft(dims.1), next_fast_fft(dims.2))
}

#[inline]
fn shrink(v: f64, t: f64) -> f64 {
    if v.abs() <= t { 0.0 } else { (v.abs() - t).copysign(v) }
}

/// NLTV dipole inversion (QSM.jl `nltv`)
///
/// # Arguments
/// * `local_field` - Local field in ppm (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz); the result is zeroed outside it
/// * `magnitude` - Magnitude image (required), made into the data weight `W` (see module docs;
///   zero in the padding)
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `bdir` - B0 field direction
/// * `params` - NLTV parameters
/// * `progress` - Progress callback `(iteration, max_iter)`
///
/// # Returns
/// Susceptibility map in ppm, masked to the ROI.
pub fn nltv(
    local_field: &[f64],
    mask: &[u8],
    magnitude: &[f64],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &NltvParams,
    mut progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let rho = params.rho.unwrap_or(100.0 * params.lambda);
    let mu = params.mu;
    let lambda_over_rho = params.lambda / rho;
    let tol_newton2 = params.tol_newton * params.tol_newton;

    // ppm -> radians, as the nonlinear data term needs.
    let ppm_to_rad = 2.0 * std::f64::consts::PI * crate::constants::GAMMA_BAR_MHZ_PER_T * params.b0 * params.te.seconds(&params.echo_times, "NLTV");

    // Zero padding to a fast FFT size (QSM.jl `pad = (0, 0, 0)`). Every step is
    // shift-equivariant, so placing the data in the corner is the same as centring it.
    let dims = grid.dims;
    let pdims = fast_fft_dims(dims);
    let pgrid = Grid::new(pdims.0, pdims.1, pdims.2, grid.vsx(), grid.vsy(), grid.vsz());
    let n = pgrid.n_total();

    let phase: Vec<f64> = local_field.iter().map(|&f| f * ppm_to_rad).collect();
    let fp = pad3d(&phase, dims, pdims);
    // W^2 / mu
    let mag_max = magnitude.iter().cloned().fold(0.0f64, f64::max);
    let w: Vec<f64> = (0..magnitude.len())
        .map(|i| {
            if mask[i] == 0 || mag_max <= 0.0 { 0.0 } else { (2.0 * magnitude[i] / mag_max).min(1.0) }
        })
        .collect();
    let w2: Vec<f64> = pad3d(&w, dims, pdims).into_iter().map(|v| v * v / mu).collect();

    let (mut fft_ws, d, l) = prepare_fansi_spectral(&pgrid, bdir);

    // iA = (mu D^H D - rho Laplacian)^-1
    let ia: Vec<f64> = (0..n)
        .map(|i| {
            let a = mu * d[i] * d[i] + rho * l[i];
            if a == 0.0 { 0.0 } else { 1.0 / a }
        })
        .collect();

    let mut f_hat = vec![Complex64::new(0.0, 0.0); n];
    let mut x_hat = vec![Complex64::new(0.0, 0.0); n];
    let mut work = vec![Complex64::new(0.0, 0.0); n];

    // previous iterate; only the first convergence check sees this starting value
    let mut x = vec![0.0; n];
    let mut x0 = vec![0.0; n];
    let (mut ux, mut uy, mut uz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let (mut dx, mut dy, mut dz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut ue = vec![0.0; n];
    let mut ze = vec![0.0; n];
    let mut div = vec![0.0; n];

    // d = z - u = grad f to start
    fgrad_inplace(&mut dx, &mut dy, &mut dz, &fp, &pgrid);

    for it in 1..=params.max_iter {
        progress(it, params.max_iter);
        std::mem::swap(&mut x, &mut x0);

        // x-subproblem: x = iA (rho grad^T (z - u) + mu D^H (z_e - u_e)), grad^T = -bdiv
        bdiv_inplace(&mut div, &dx, &dy, &dz, &pgrid);
        for i in 0..n {
            x_hat[i] = Complex64::new(-div[i], 0.0);
        }
        fft_ws.fft3d(&mut x_hat);
        for i in 0..n {
            x_hat[i] = rho * ia[i] * x_hat[i] + f_hat[i];
            work[i] = x_hat[i];
        }
        fft_ws.ifft3d(&mut work);
        for i in 0..n {
            x[i] = work[i].re;
        }

        // convergence check
        let (mut ndx, mut nx) = (0.0, 0.0);
        for i in 0..n {
            let (a, b) = (x[i], x0[i]);
            ndx += (a - b) * (a - b);
            nx += a * a;
        }
        if ndx.sqrt() < params.tol * nx.sqrt() || it == params.max_iter {
            break;
        }

        // TV term: z = shrink(grad x + u, lambda/rho), u = u + grad x - z, d = z - u
        fgrad_inplace(&mut dx, &mut dy, &mut dz, &x, &pgrid);
        for (u, g) in [(&mut ux, &mut dx), (&mut uy, &mut dy), (&mut uz, &mut dz)] {
            for i in 0..n {
                let v = u[i] + g[i];
                let z = shrink(v, lambda_over_rho);
                u[i] = v - z;
                g[i] = z - v + z;
            }
        }

        // Data term: v = D x + u_e; Newton on W^2/mu sin(z - f) + z - v = 0, from z = v
        for i in 0..n {
            work[i] = x_hat[i] * d[i];
        }
        fft_ws.ifft3d(&mut work);
        for i in 0..n {
            let v = work[i].re + ue[i];
            ze[i] = v;
            ue[i] = v;
        }
        for _ in 0..params.max_iter_newton {
            let (mut nd, mut nz) = (0.0, 0.0);
            for i in 0..n {
                let (v, w, z) = (ue[i], w2[i], ze[i]);
                let (s, c) = (z - fp[i]).sin_cos();
                let delta = (w * s + z - v) / (w * c + 1.0);
                ze[i] = z - delta;
                nd += delta * delta;
                nz += z * z;
            }
            if nd < nz * tol_newton2 {
                break;
            }
        }
        // u_e = v - z; data part of the next right-hand side: mu D^H (z - u_e) iA
        for i in 0..n {
            let (v, z) = (ue[i], ze[i]);
            ue[i] = v - z;
            f_hat[i] = Complex64::new(z - v + z, 0.0);
        }
        fft_ws.fft3d(&mut f_hat);
        for i in 0..n {
            f_hat[i] *= mu * d[i] * ia[i];
        }
    }

    let mut chi = unpad3d(&x, pdims, dims);
    for v in chi.iter_mut() {
        *v /= ppm_to_rad;
    }
    apply_mask_zero(&mut chi, mask);
    chi
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_params() -> NltvParams {
        NltvParams { max_iter: 20, ..NltvParams::default() }
    }

    #[test]
    fn test_nltv_zero_field() {
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let mag = vec![1.0; field.len()];
        let chi = nltv(&field, &mask, &mag, &grid, (0.0, 0.0, 1.0), &small_params(), |_, _| {});
        assert!(chi.iter().all(|v| v.abs() < 1e-12), "zero field should give zero chi");
    }

    #[test]
    fn test_nltv_magnitude_weight() {
        // a dimmer region gets a smaller weight and changes the result
        let (nx, ny, nz) = (10, 9, 8);
        let nt = nx * ny * nz;
        let field: Vec<f64> = (0..nt).map(|i| ((i * 37 % 101) as f64 - 50.0) * 1e-3).collect();
        let mask = vec![1u8; nt];
        let grid = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
        let p = small_params();
        let mag = vec![5.0; nt];
        let a = nltv(&field, &mask, &mag, &grid, (0.1, 0.0, 1.0), &p, |_, _| {});
        let mut mag2 = mag.clone();
        mag2[..nt / 2].iter_mut().for_each(|v| *v = 1.0);
        let c = nltv(&field, &mask, &mag2, &grid, (0.1, 0.0, 1.0), &p, |_, _| {});
        assert!(a.iter().all(|v| v.is_finite()));
        assert!(a.iter().zip(&c).any(|(x, y)| (x - y).abs() > 1e-9));
    }

    #[test]
    fn test_nltv_masks_output() {
        let n = 11; // padded to 12
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 1e-4).collect();
        let mut mask = vec![1u8; n * n * n];
        mask[0] = 0;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let mag = vec![1.0; field.len()];
        let chi = nltv(&field, &mask, &mag, &grid, (0.0, 0.0, 1.0), &small_params(), |_, _| {});
        assert_eq!(chi.len(), n * n * n);
        assert_eq!(chi[0], 0.0);
        assert!(chi.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_fast_fft_dims() {
        // values from QSM.jl fastfftsize(sz, (0, 0, 0); rfft=true)
        assert_eq!(fast_fft_dims((11, 11, 13)), (12, 12, 14));
        assert_eq!(fast_fft_dims((176, 240, 150)), (180, 240, 150));
        assert_eq!(fast_fft_dims((255, 97, 49)), (256, 98, 49));
        assert_eq!(fast_fft_dims((15, 15, 15)), (16, 15, 15));
    }
}

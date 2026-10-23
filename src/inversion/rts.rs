//! Rapid Two-Step (RTS) dipole inversion
//!
//! Two steps:
//! 1. **Well-conditioned k-space:** a few LSMR iterations on `D χ̂ = F f` (all of k-space,
//!    from zero), then the result is masked.
//! 2. **Ill-conditioned k-space:** TV-regularised ADMM that holds the well-conditioned
//!    coefficients (`|D| > delta`) of step 1 with weight `mu`, starting from the gradient of the
//!    step-1 map, and raising the penalty `rho` by `gamma` whenever the primal residual falls by
//!    less than a factor `tau`.
//!
//! Reference:
//! Kames, C., Wiggermann, V., Rauscher, A. (2018).
//! "Rapid two-step dipole inversion for susceptibility mapping with sparsity priors."
//! NeuroImage, 167:276-283. https://doi.org/10.1016/j.neuroimage.2017.11.018
//!
//! Reference implementation: QSM.jl's `rts` (https://github.com/kamesy/QSM.jl, by the
//! method's first author), ported line for line with its defaults. The field is in ppm, as in
//! QSM.jl. QSM.jl works on real-to-complex FFTs; here the full complex FFT is used, and the
//! LSMR inner products are taken over the half spectrum QSM.jl stores (`kx` index
//! `0..=nx/2`) so its scalars are the same.
//!
//! # Reference pipeline
//! QSM.jl's README example (commit fde4b92, L35-52) ends in `rts`: FSL BET on the last echo's
//! magnitude (`-f 0.5`), eroded by 5 voxels; Laplacian unwrapping of each echo, each scaled to
//! ppm by its own TE; V-SHARP with QSM.jl's defaults (radii `18*min(vsz)` down to `2*max(vsz)`
//! mm in steps of `2*max(vsz)`, threshold 0.05; `src/bgremove/sharp.jl` L136-137); then `rts` on
//! the (per-echo) local field. QSM.jl's `pad = (0, 0, 0)` pads to a fast FFT size only; this
//! port does not pad.
//!
//! # Pipeline rules
//!
//! The facts above that a configuration can depart from are published as rules in
//! [`crate::pipeline::rules::RULES`] (each with its reason and source): `rts.bg-vsharp`.

use num_complex::Complex64;
use crate::fft::Fft3dWorkspace;
use crate::kernels::dipole::dipole_kernel;
use crate::kernels::laplacian::laplacian_kernel;
use crate::utils::apply_mask_zero;
use crate::utils::gradient::{bdiv_inplace, fgrad_inplace};
use crate::Grid;

/// RTS algorithm parameters (QSM.jl `rts` keywords and defaults)
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct RtsParams {
    /// Threshold on |D| for the ill-conditioned k-space region (`delta`)
    pub delta: f64,
    /// Weight of the well-conditioned k-space data in step 2 (`mu`)
    pub mu: f64,
    /// Initial ADMM penalty parameter (`rho`)
    pub rho: f64,
    /// Stopping tolerance on the relative change of χ (`tol`)
    pub tol: f64,
    /// Maximum ADMM iterations (`maxit`)
    pub max_iter: usize,
    /// LSMR iterations in step 1 (`lstol`)
    pub lsmr_iter: usize,
    /// Raise `rho` when the primal residual falls by less than this factor (`tau`)
    pub tau: f64,
    /// Factor `rho` is raised by (`gamma`)
    pub gamma: f64,
}

impl Default for RtsParams {
    fn default() -> Self {
        Self {
            delta: 0.15,
            mu: 1e5,
            rho: 10.0,
            tol: 1e-2,
            max_iter: 20,
            lsmr_iter: 4,
            tau: 0.7,
            gamma: 5.0,
        }
    }
}

/// Norm of a Hermitian-symmetric spectrum as QSM.jl sees it: over the `kx = 0..=nx/2` half
/// that its real-to-complex FFT stores.
fn half_norm(v: &[Complex64], nx: usize) -> f64 {
    let half = nx / 2;
    v.iter()
        .enumerate()
        .filter(|(i, _)| i % nx <= half)
        .map(|(_, c)| c.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

/// LSMR for the diagonal operator `diag(d)` on a spectrum, from x = 0, with `lambda = 0`,
/// `atol = btol = 0` and `conlim = 1e8`, as QSM.jl's `rts` calls its copy of
/// IterativeSolvers.jl's `lsmr!` (v0.9.2).
fn lsmr_diag(d: &[f64], b: &[Complex64], nx: usize, maxit: usize) -> Vec<Complex64> {
    let n = b.len();
    let zero = Complex64::new(0.0, 0.0);
    let mut x = vec![zero; n];

    let mut u = b.to_vec();
    let mut beta = half_norm(&u, nx);
    if beta > 0.0 {
        u.iter_mut().for_each(|c| *c /= beta);
    }
    let mut v: Vec<Complex64> = (0..n).map(|i| u[i] * d[i]).collect();
    let mut alpha = half_norm(&v, nx);
    if alpha * beta == 0.0 {
        return x;
    }
    if alpha > 0.0 {
        v.iter_mut().for_each(|c| *c /= alpha);
    }

    let mut zetabar = alpha * beta;
    let mut alphabar = alpha;
    let mut rho = 1.0f64;
    let mut rhobar = 1.0f64;
    let mut cbar = 1.0f64;
    let mut sbar = 0.0f64;

    let mut hbar = vec![zero; n];
    let mut h = v.clone();

    let mut betadd = beta;
    let mut betad = 0.0f64;
    let mut rhodold = 1.0f64;
    let mut tautildeold = 0.0f64;
    let mut thetatilde = 0.0f64;
    let mut zeta = 0.0f64;
    let mut dsum = 0.0f64;

    let mut norm_a2 = alpha * alpha;
    let mut maxrbar = 0.0f64;
    let mut minrbar = f64::MAX;
    let normb = beta;
    let ctol = 1e-8; // 1/conlim

    let mut i = 0;
    while i < maxit {
        i += 1;

        // u = A v - alpha u
        for k in 0..n {
            u[k] = v[k] * d[k] - u[k] * alpha;
        }
        beta = half_norm(&u, nx);
        if beta > 0.0 {
            u.iter_mut().for_each(|c| *c /= beta);
            for k in 0..n {
                v[k] = u[k] * d[k] - v[k] * beta;
            }
            alpha = half_norm(&v, nx);
            if alpha > 0.0 {
                v.iter_mut().for_each(|c| *c /= alpha);
            }
        }

        // lambda = 0: chat = 1, shat = 0
        let alphahat = alphabar.hypot(0.0);
        let chat = alphabar / alphahat;
        let shat = 0.0 / alphahat;

        let rhoold = rho;
        rho = alphahat.hypot(beta);
        let c = alphahat / rho;
        let s = beta / rho;
        let thetanew = s * alpha;
        alphabar = c * alpha;

        let rhobarold = rhobar;
        let zetaold = zeta;
        let thetabar = sbar * rho;
        let rhotmp = cbar * rho;
        rhobar = rhotmp.hypot(thetanew);
        cbar = rhotmp / rhobar;
        sbar = thetanew / rhobar;
        zeta = cbar * zetabar;
        zetabar *= -sbar;

        let delta = -thetabar * rho / (rhoold * rhobarold);
        let sigma = zeta / (rho * rhobar);
        let nu = -thetanew / rho;
        let mut normx2 = 0.0;
        for k in 0..n {
            hbar[k] = hbar[k] * delta + h[k];
            x[k] += hbar[k] * sigma;
            h[k] = h[k] * nu + v[k];
            if k % nx <= nx / 2 {
                normx2 += x[k].norm_sqr();
            }
        }
        let normx = normx2.sqrt();

        let betaacute = chat * betadd;
        let betacheck = -shat * betadd;
        let betahat = c * betaacute;
        betadd = -s * betaacute;

        let thetatildeold = thetatilde;
        let rhotildeold = rhodold.hypot(thetabar);
        let ctildeold = rhodold / rhotildeold;
        let stildeold = thetabar / rhotildeold;
        thetatilde = stildeold * rhobar;
        rhodold = ctildeold * rhobar;
        betad = -stildeold * betad + ctildeold * betahat;

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold;
        let taud = (zeta - thetatilde * tautildeold) / rhodold;

        dsum += betacheck * betacheck;
        let normr = (dsum + (betad - taud).powi(2) + betadd * betadd).sqrt();

        norm_a2 += beta * beta;
        let norm_a = norm_a2.sqrt();
        norm_a2 += alpha * alpha;

        maxrbar = maxrbar.max(rhobarold);
        if i > 1 {
            minrbar = minrbar.min(rhobarold);
        }
        let cond_a = maxrbar.max(rhotmp) / minrbar.min(rhotmp);

        let norm_ar = zetabar.abs();
        let test1 = normr / normb;
        let test2 = norm_ar / (norm_a * normr);
        let test3 = 1.0 / cond_a;
        let t1 = test1 / (1.0 + norm_a * normx / normb);
        let eps_r = 0.0; // btol + atol*... with atol = btol = 0

        if test1 <= eps_r
            || test2 <= 0.0
            || test3 <= ctol
            || 1.0 + t1 <= 1.0
            || 1.0 + test2 <= 1.0
            || 1.0 + test3 <= 1.0
            || i >= maxit
        {
            break;
        }
    }
    x
}

/// RTS dipole inversion (QSM.jl `rts`)
///
/// # Arguments
/// * `local_field` - Local field in ppm (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `bdir` - B0 field direction
/// * `params` - RTS parameters
/// * `progress` - Progress callback `(iteration, max_iter)`
///
/// # Returns
/// Susceptibility map in ppm, masked
pub fn rts(
    local_field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &RtsParams,
    mut progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let n = grid.n_total();
    let zero = Complex64::new(0.0, 0.0);

    let mut fft_ws = Fft3dWorkspace::new(nx, ny, nz);
    let d = dipole_kernel(grid, bdir);

    // Negative Laplacian kernel in k-space (L)
    let mut work: Vec<Complex64> = laplacian_kernel(grid, true)
        .iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect();
    fft_ws.fft3d(&mut work);
    let l: Vec<f64> = work.iter().map(|c| c.re).collect();

    // M = mu on the well-conditioned frequencies
    let m: Vec<f64> = d.iter().map(|&dv| if dv.abs() > params.delta { params.mu } else { 0.0 }).collect();

    // ---- Step 1: LSMR on D x = F f, then mask ----
    for i in 0..n {
        work[i] = Complex64::new(local_field[i], 0.0);
    }
    fft_ws.fft3d(&mut work);
    let mut xhat = lsmr_diag(&d, &work, nx, params.lsmr_iter);
    fft_ws.ifft3d(&mut xhat);
    let mut x: Vec<f64> = (0..n).map(|i| if mask[i] != 0 { xhat[i].re } else { 0.0 }).collect();

    // ---- Step 2: TV for the ill-conditioned frequencies ----
    let mut rho = params.rho;
    for i in 0..n {
        work[i] = Complex64::new(x[i], 0.0);
    }
    fft_ws.fft3d(&mut work);
    let mut f_hat = vec![zero; n];
    let mut i_a = vec![0.0f64; n];
    for i in 0..n {
        let a = rho * l[i] + m[i];
        if a != 0.0 {
            let ia = 1.0 / a;
            f_hat[i] = work[i] * (ia * m[i]);
            i_a[i] = rho * ia;
        }
    }

    let mut px = vec![0.0f64; n];
    let mut py = vec![0.0f64; n];
    let mut pz = vec![0.0f64; n];
    // v = y - p, starting from v = ∇x
    let mut vx = vec![0.0f64; n];
    let mut vy = vec![0.0f64; n];
    let mut vz = vec![0.0f64; n];
    fgrad_inplace(&mut vx, &mut vy, &mut vz, &x, grid);
    let mut dx = vec![0.0f64; n];
    let mut dy = vec![0.0f64; n];
    let mut dz = vec![0.0f64; n];
    let mut div = vec![0.0f64; n];
    let mut x0 = vec![0.0f64; n];
    let mut nr = f64::MAX;

    for it in 1..=params.max_iter {
        progress(it, params.max_iter);
        std::mem::swap(&mut x, &mut x0);

        // x-subproblem: x = F⁻¹(iA · F(∇ᵀ v) + F̂), with ∇ᵀ = -div
        bdiv_inplace(&mut div, &vx, &vy, &vz, grid);
        for i in 0..n {
            work[i] = Complex64::new(-div[i], 0.0);
        }
        fft_ws.fft3d(&mut work);
        for i in 0..n {
            work[i] = work[i] * i_a[i] + f_hat[i];
        }
        fft_ws.ifft3d(&mut work);
        for i in 0..n {
            x[i] = work[i].re;
        }

        // convergence check
        let (mut ndx2, mut nx2) = (0.0, 0.0);
        for i in 0..n {
            let (a, b) = (x[i], x0[i]);
            ndx2 += (a - b) * (a - b);
            nx2 += a * a;
        }
        if ndx2.sqrt() < params.tol * nx2.sqrt() || it == params.max_iter {
            progress(it, it);
            break;
        }

        // y-subproblem and multiplier update
        fgrad_inplace(&mut dx, &mut dy, &mut dz, &x, grid);
        let thr = 1.0 / rho;
        let update = |p: &mut [f64], v: &mut [f64], dd: &mut [f64]| {
            for i in 0..n {
                let di = dd[i];
                let pi = p[i] + di;
                let y = if pi.abs() <= thr { 0.0 } else { (pi.abs() - thr).copysign(pi) };
                p[i] = pi - y;
                v[i] = y - pi + y;
                dd[i] = di - y;
            }
        };
        update(&mut px, &mut vx, &mut dx);
        update(&mut py, &mut vy, &mut dy);
        update(&mut pz, &mut vz, &mut dz);

        // penalty update on the primal residual Σ|∇x - y|
        let nr0 = nr;
        nr = (0..n).map(|i| (dx[i] * dx[i] + dy[i] * dy[i] + dz[i] * dz[i]).sqrt()).sum();
        if nr > params.tau * nr0 {
            let rho0 = rho;
            rho *= params.gamma;
            for i in 0..n {
                let iaprev = i_a[i];
                if iaprev != 0.0 {
                    let ia = 1.0 / (rho * l[i] + m[i]);
                    f_hat[i] *= ia * rho0 / iaprev;
                    i_a[i] = rho * ia;
                }
            }
            let ig = 1.0 / params.gamma;
            px.iter_mut().for_each(|v| *v *= ig);
            py.iter_mut().for_each(|v| *v *= ig);
            pz.iter_mut().for_each(|v| *v *= ig);
        }
    }

    apply_mask_zero(&mut x, mask);
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rts_zero_field() {
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = RtsParams { max_iter: 5, lsmr_iter: 2, ..Default::default() };

        let chi = rts(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        for &val in chi.iter() {
            assert!(val.abs() < 1e-6, "Zero field should give near-zero chi");
        }
    }

    #[test]
    fn test_rts_finite() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = RtsParams { max_iter: 5, lsmr_iter: 2, ..Default::default() };

        let chi = rts(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        for (i, &val) in chi.iter().enumerate() {
            assert!(val.is_finite(), "Chi should be finite at index {}", i);
        }
    }

    #[test]
    fn test_rts_mask() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mut mask = vec![1u8; n * n * n];
        // Zero out some mask values
        mask[0] = 0;
        mask[10] = 0;
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = RtsParams { max_iter: 5, lsmr_iter: 2, ..Default::default() };

        let chi = rts(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        assert_eq!(chi[0], 0.0, "Masked voxel should be zero");
        assert_eq!(chi[10], 0.0, "Masked voxel should be zero");
    }

    /// Verify parallel and sequential RTS produce identical results.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_rts_parallel_matches_sequential() {
        let n = 16;
        let field: Vec<f64> = (0..n*n*n).map(|i| ((i as f64) * 0.7).sin() * 0.01).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let params = RtsParams { tol: 1e-4, ..Default::default() };

        // Sequential (1 thread)
        let pool_1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let chi_seq = pool_1.install(|| {
            rts(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {})
        });

        // Parallel (default threads)
        let chi_par = rts(&field, &mask, &grid, (0.0, 0.0, 1.0), &params, |_, _| {});

        // Compare
        for (i, (s, p)) in chi_seq.iter().zip(chi_par.iter()).enumerate() {
            assert!(
                (s - p).abs() < 1e-10,
                "RTS mismatch at voxel {}: seq={} par={} diff={}",
                i, s, p, (s - p).abs()
            );
        }
    }
}

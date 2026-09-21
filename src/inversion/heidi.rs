//! HEIDI — Homogeneity Enabled Incremental Dipole Inversion
//!
//! HEIDI splits k-space at the dipole cone and treats the two halves differently:
//!
//! * **Well-conditioned coefficients** (`|D(k)| > threshold`) are taken straight
//!   from an already-computed χ map — normally the minimally regularised LSQR
//!   solution of [`super::lsqr_qsm`] — and held as *exact* constraints.
//! * **Ill-conditioned coefficients** inside the cone, where direct division
//!   amplifies noise without bound, are recovered by minimising a weighted total
//!   variation subject to those constraints.
//!
//! ```text
//! minimise  Σ_v ‖ (M_x ∂_x χ, M_y ∂_y χ, M_z ∂_z χ) ‖   subject to   (F χ)|cone = (F χ_init)|cone
//! ```
//!
//! The "homogeneity" of the name is the weighting `M`. Each direction gets its own
//! 0/1 mask, built by thresholding the gradient of the *field* map: where the field
//! is locally flat the susceptibility is expected to be flat too, and TV is applied;
//! across an anatomical edge the mask opens and the gradient goes unpenalised. A
//! Laplacian test additionally opens the mask wherever the field curves sharply.
//! Voxels that are an edge in every direction keep a small residual weight
//! ([`HeidiParams::gradient_mask_floor`]) rather than being fully unconstrained,
//! which suppresses noise in otherwise-free regions.
//!
//! The constrained minimisation is solved with NESTA (Nesterov's smoothing with
//! continuation): the TV term is smoothed by a parameter μ that is annealed down
//! over several outer steps, each of which runs a fixed number of accelerated
//! gradient iterations. Because the data constraint is an orthogonal projection
//! (`A·Aᵀ = I` for a restricted FFT), the projection step is exact and costs one
//! FFT round trip.
//!
//! Units: the field is in ppm and χ comes back in ppm. The gradient and Laplacian
//! thresholds below are therefore in ppm/mm and ppm/voxel², independent of echo
//! time and field strength — upstream expresses the same numbers in units of
//! phase/(TE·B₀), which is the same quantity.
//!
//! Reference:
//! Schweser, F., Sommer, K., Deistung, A., Reichenbach, J.R. (2012).
//! "Quantitative susceptibility mapping for investigating subtle susceptibility
//! variations in the human brain." NeuroImage, 62(3):2083-2100.
//! https://doi.org/10.1016/j.neuroimage.2012.05.067
//!
//! NESTA:
//! Becker, S., Bobin, J., Candès, E.J. (2011). "NESTA: A fast and accurate
//! first-order method for sparse recovery." SIAM Journal on Imaging Sciences,
//! 4(1):1-39. https://doi.org/10.1137/090756855

use num_complex::Complex64;

use crate::fft::Fft3dWorkspace;
use crate::kernels::dipole::dipole_kernel;
use crate::kernels::laplacian::laplacian_kernel;
use crate::utils::anisotropic_diffusion::{
    gradient_anisotropic_diffusion, AnisotropicDiffusionParams,
};
use crate::utils::apply_mask_zero;
use crate::utils::padding::{next_fast_fft_size, pad3d, unpad3d};
use crate::Grid;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Gyromagnetic ratio of the proton, in rad/s/T.
const GAMMA: f64 = 2.67522128e8;

/// Upstream expresses its mask thresholds per unit of `phase / (TE[ms] · B₀[T])`.
/// One ppm of field is `γ · 1e-3 · 1e-6` of that, so dividing an upstream threshold
/// by this constant converts it to ppm.
const UPSTREAM_PPM_SCALE: f64 = GAMMA * 1e-9;

/// Upstream `DEFAULT_GRADIENTMASK_THRESHOLD`, converted to ppm/mm.
pub const DEFAULT_GRADIENT_THRESHOLD_PPM: f64 = 0.00105 / UPSTREAM_PPM_SCALE;

/// Upstream `DEFAULT_GRADIENTMASK_LAPLACIANTRHESHOLD`, converted to ppm/voxel².
pub const DEFAULT_LAPLACIAN_THRESHOLD_PPM: f64 = 0.0159 / UPSTREAM_PPM_SCALE;

/// Parameters for [`heidi`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct HeidiParams {
    /// `|D(k)|` above which a coefficient counts as well-conditioned and is taken
    /// from `chi_init` unchanged.
    pub cone_threshold: f64,
    /// Field-gradient threshold in ppm/mm. Below it, a direction is considered
    /// homogeneous and its TV weight is switched on.
    pub gradient_threshold: f64,
    /// Open the homogeneity masks where the field's Laplacian is large.
    pub apply_laplacian_correction: bool,
    /// Field-Laplacian threshold in ppm per voxel² (the stencil is in voxel units,
    /// matching upstream).
    pub laplacian_threshold: f64,
    /// Anisotropic-diffusion smoothing applied to the field before the gradients
    /// are taken. `None` skips it.
    pub denoise: Option<AnisotropicDiffusionParams>,
    /// TV weight kept at voxels that read as an edge in *every* direction.
    ///
    /// Upstream's shipped code passes the masks through a `> 0.5` resize test that
    /// silently rounds this floor away, but its own changelog calls binarising the
    /// masks a bug ("must contain values of 0.1"), so the documented intent is the
    /// default here. Set it to `0.0` to reproduce the shipped behaviour.
    pub gradient_mask_floor: f64,
    /// NESTA continuation steps on μ.
    pub continuation_steps: usize,
    /// Accelerated-gradient iterations per continuation step.
    pub inner_iterations: usize,
    /// Final (smallest) smoothing parameter μ.
    pub mu_min: f64,
    /// Relative objective change that ends an inner loop early.
    ///
    /// This is upstream's own `DEFAULT_TV_TOLVAR`. Its shipped driver overrides it
    /// to machine epsilon, which disables early stopping and always burns the full
    /// `continuation_steps × inner_iterations` budget — thousands of FFT round
    /// trips, and hours on a whole brain. At `1e-3` the continuation levels settle
    /// on their own, typically in a few hundred iterations total.
    pub tol: f64,
}

impl Default for HeidiParams {
    fn default() -> Self {
        Self {
            // Upstream's `computesusceptibility_heidi` defaults to 0.14; the
            // shipped driver `inversion_heidi.m` overrides it to 0.1.
            cone_threshold: 0.1,
            gradient_threshold: DEFAULT_GRADIENT_THRESHOLD_PPM,
            apply_laplacian_correction: true,
            laplacian_threshold: DEFAULT_LAPLACIAN_THRESHOLD_PPM,
            denoise: Some(AnisotropicDiffusionParams {
                iterations: 5,
                time_step: 0.05,
                conductance: 1.0,
            }),
            gradient_mask_floor: 0.1,
            continuation_steps: 8,
            inner_iterations: 500,
            mu_min: f64::EPSILON,
            tol: 1e-3,
        }
    }
}

/// The three per-direction homogeneity weights, as used by the TV term.
pub struct HomogeneityMasks {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
}

/// HEIDI dipole inversion.
///
/// # Arguments
/// * `local_field` - Local field in ppm (`nx * ny * nz`, Fortran order)
/// * `mask` - Binary brain mask
/// * `chi_init` - Starting χ map in ppm, supplying the well-conditioned k-space
///   coefficients. [`super::lsqr_qsm`] with `mask_output: false` is the intended
///   source; any inversion will work, but the cone content is then that method's.
/// * `grid` - Volume grid
/// * `bdir` - B0 direction
/// * `params` - Algorithm parameters
/// * `progress` - Progress callback `(iteration, total_iterations)`
///
/// # Returns
/// Susceptibility map in ppm, masked.
pub fn heidi(
    local_field: &[f64],
    mask: &[u8],
    chi_init: &[f64],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &HeidiParams,
    progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    // Thousands of FFT round trips make the transform size the dominant cost, so
    // grow the grid to one that factors into small primes first. Zero padding is
    // transparent here: the dipole cone and the TV masks are both rebuilt on the
    // padded grid, and the mask keeps the extra voxels out of the result.
    let dims = grid.dims;
    let fast = (
        next_fast_fft_size(dims.0),
        next_fast_fft_size(dims.1),
        next_fast_fft_size(dims.2),
    );
    if fast == dims {
        return heidi_core(local_field, mask, chi_init, grid, bdir, params, progress);
    }
    let (vsx, vsy, vsz) = grid.voxel_size;
    let pgrid = Grid::new(fast.0, fast.1, fast.2, vsx, vsy, vsz);
    let chi = heidi_core(
        &pad3d(local_field, dims, fast),
        &pad3d(mask, dims, fast),
        &pad3d(chi_init, dims, fast),
        &pgrid,
        bdir,
        params,
        progress,
    );
    unpad3d(&chi, fast, dims)
}

fn heidi_core(
    local_field: &[f64],
    mask: &[u8],
    chi_init: &[f64],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &HeidiParams,
    progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let n_total = grid.n_total();
    assert_eq!(local_field.len(), n_total, "field length must match grid");
    assert_eq!(mask.len(), n_total, "mask length must match grid");
    assert_eq!(chi_init.len(), n_total, "chi_init length must match grid");

    let masks = homogeneity_masks(local_field, mask, grid, params);
    let cone = cone_mask(grid, bdir, params.cone_threshold);

    let mut chi = nesta_cone_fill(chi_init, &cone, &masks, grid, params, progress);
    apply_mask_zero(&mut chi, mask);
    chi
}

/// `|D(k)| > threshold`: the k-space region the dipole model conditions well.
fn cone_mask(grid: &Grid, bdir: (f64, f64, f64), threshold: f64) -> Vec<bool> {
    dipole_kernel(grid, bdir)
        .into_iter()
        .map(|d| d.abs() > threshold)
        .collect()
}

/// Build the three per-direction homogeneity weights from the field map.
///
/// Public so callers can inspect or reuse the weighting; [`heidi`] calls it itself.
pub fn homogeneity_masks(
    local_field: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &HeidiParams,
) -> HomogeneityMasks {
    let (nx, ny, nz) = grid.dims;
    let n_total = grid.n_total();

    // Upstream masks the field before denoising, so the diffusion sees a clean
    // zero outside the brain rather than background junk.
    let mut field = local_field.to_vec();
    apply_mask_zero(&mut field, mask);

    let smoothed = match &params.denoise {
        Some(d) => gradient_anisotropic_diffusion(&field, grid, d),
        None => field,
    };

    // A direction is "homogeneous" where the field gradient is small.
    let mut mx = threshold_gradient(&smoothed, grid, 0, params.gradient_threshold);
    let mut my = threshold_gradient(&smoothed, grid, 1, params.gradient_threshold);
    let mut mz = threshold_gradient(&smoothed, grid, 2, params.gradient_threshold);

    if params.apply_laplacian_correction {
        // Sharp field curvature means an unresolved source: drop the TV weight
        // there in every direction. The stencil is in voxel units, as upstream's.
        let unit_grid = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
        let lap_kernel = laplacian_kernel(&unit_grid, false);

        let mut ws = Fft3dWorkspace::new(nx, ny, nz);
        let mut kernel_ft: Vec<Complex64> =
            lap_kernel.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        ws.fft3d(&mut kernel_ft);

        let mut buf: Vec<Complex64> =
            smoothed.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        ws.fft3d(&mut buf);
        for (b, k) in buf.iter_mut().zip(&kernel_ft) {
            *b *= *k;
        }
        ws.ifft3d(&mut buf);

        for i in 0..n_total {
            if buf[i].re.abs() > params.laplacian_threshold {
                mx[i] = 0.0;
                my[i] = 0.0;
                mz[i] = 0.0;
            }
        }
    }

    // Restrict to the brain, then give all-edge voxels a residual weight so they
    // are not left completely unconstrained.
    for i in 0..n_total {
        if mask[i] == 0 {
            mx[i] = 0.0;
            my[i] = 0.0;
            mz[i] = 0.0;
        } else if mx[i] == 0.0 && my[i] == 0.0 && mz[i] == 0.0 {
            mx[i] = params.gradient_mask_floor;
            my[i] = params.gradient_mask_floor;
            mz[i] = params.gradient_mask_floor;
        }
    }

    HomogeneityMasks { x: mx, y: my, z: mz }
}

/// 1.0 where `|∂_axis field| < threshold`, else 0.0.
fn threshold_gradient(field: &[f64], grid: &Grid, axis: usize, threshold: f64) -> Vec<f64> {
    let g = discrete_gradient(field, grid, axis, false);
    g.into_iter()
        .map(|v| if v.abs() < threshold { 1.0 } else { 0.0 })
        .collect()
}

/// Backward difference along `axis`, scaled by voxel size, zero on the leading face.
///
/// `transpose` gives the operator upstream pairs with it: a forward difference,
/// zero on the *trailing* face. The two agree with a true adjoint pair except on
/// the two boundary faces, which carry no TV weight (the brain mask is zero there),
/// so the discrepancy never reaches the solution.
fn discrete_gradient_into(x: &[f64], out: &mut [f64], grid: &Grid, axis: usize, transpose: bool) {
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = grid.voxel_size;
    let (n_axis, voxel_size) = match axis {
        0 => (nx, vsx),
        1 => (ny, vsy),
        _ => (nz, vsz),
    };

    out.fill(0.0);
    if n_axis < 2 {
        return;
    }
    let inv = 1.0 / voxel_size;
    let slab = nx * ny;

    // Split the volume into independent pieces: one row per chunk along x, one
    // z-slab along y and z. Each chunk writes only its own output but may read
    // its neighbour's input, which is fine — `x` is shared immutably.
    let chunk = if axis == 0 { nx } else { slab };

    crate::maybe_par_chunks_mut!(out, chunk)
        .enumerate()
        .for_each(|(c, o)| {
            let base = c * chunk;
            match axis {
                0 => {
                    if transpose {
                        for p in 0..nx - 1 {
                            o[p] = (x[base + p] - x[base + p + 1]) * inv;
                        }
                    } else {
                        for p in 1..nx {
                            o[p] = (x[base + p] - x[base + p - 1]) * inv;
                        }
                    }
                }
                1 => {
                    if transpose {
                        for j in 0..ny - 1 {
                            for i in 0..nx {
                                let q = i + j * nx;
                                o[q] = (x[base + q] - x[base + q + nx]) * inv;
                            }
                        }
                    } else {
                        for j in 1..ny {
                            for i in 0..nx {
                                let q = i + j * nx;
                                o[q] = (x[base + q] - x[base + q - nx]) * inv;
                            }
                        }
                    }
                }
                _ => {
                    // `c` is the slice index along z.
                    if transpose {
                        if c + 1 < nz {
                            for p in 0..slab {
                                o[p] = (x[base + p] - x[base + p + slab]) * inv;
                            }
                        }
                    } else if c >= 1 {
                        for p in 0..slab {
                            o[p] = (x[base + p] - x[base + p - slab]) * inv;
                        }
                    }
                }
            }
        });
}

/// Allocating wrapper around [`discrete_gradient_into`].
fn discrete_gradient(x: &[f64], grid: &Grid, axis: usize, transpose: bool) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    discrete_gradient_into(x, &mut out, grid, axis, transpose);
    out
}

/// Scratch buffers for the NESTA loop, so the inner iterations allocate nothing.
struct TvWorkspace {
    /// Per-direction weighted gradients, reused to carry `M·u` into the adjoint pass.
    d1: Vec<f64>,
    d2: Vec<f64>,
    d3: Vec<f64>,
    /// One `(semi_norm, ‖u‖²)` partial per z-slab, summed after the parallel pass.
    partials: Vec<(f64, f64)>,
}

impl TvWorkspace {
    fn new(grid: &Grid) -> Self {
        let n = grid.n_total();
        Self {
            d1: vec![0.0; n],
            d2: vec![0.0; n],
            d3: vec![0.0; n],
            partials: vec![(0.0, 0.0); grid.nz()],
        }
    }
}

/// Smoothed weighted-TV objective and its gradient.
///
/// Returns `f_μ(x)` and writes `∇f_μ(x)` into `df`. Two passes over the volume,
/// each parallel over z-slabs: the first builds the masked gradients and the
/// smoothing weight `w`, the second applies the adjoint differences.
fn tv_objective(
    x: &[f64],
    mu: f64,
    masks: &HomogeneityMasks,
    grid: &Grid,
    ws: &mut TvWorkspace,
    df: &mut [f64],
) -> f64 {
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = grid.voxel_size;
    let slab = nx * ny;
    let (ix, iy, iz) = (1.0 / vsx, 1.0 / vsy, 1.0 / vsz);

    // ── Pass 1: masked gradients, Nesterov smoothing, objective partials ──
    //
    // `w = max(μ, ‖∇x‖)` replaces the TV norm with a Huber-like function that is
    // quadratic below μ and linear above it. The buffers come out holding
    // `M·u = M²·Dx / w`, which is exactly what the adjoint pass needs.
    crate::maybe_par_chunks_mut!(ws.d1, slab)
        .zip(crate::maybe_par_chunks_mut!(ws.d2, slab))
        .zip(crate::maybe_par_chunks_mut!(ws.d3, slab))
        .zip(crate::maybe_par_chunks_mut!(ws.partials, 1))
        .enumerate()
        .for_each(|(k, (((d1s, d2s), d3s), acc))| {
            let base = k * slab;
            let (mut semi, mut nsq) = (0.0, 0.0);

            for j in 0..ny {
                for i in 0..nx {
                    let q = i + j * nx;
                    let idx = base + q;

                    // Backward differences, zero on the leading face of each axis.
                    let gx = if i >= 1 { (x[idx] - x[idx - 1]) * ix } else { 0.0 };
                    let gy = if j >= 1 { (x[idx] - x[idx - nx]) * iy } else { 0.0 };
                    let gz = if k >= 1 { (x[idx] - x[idx - slab]) * iz } else { 0.0 };

                    let (a, b, c) = (
                        masks.x[idx] * gx,
                        masks.y[idx] * gy,
                        masks.z[idx] * gz,
                    );
                    let mag_sq = a * a + b * b + c * c;
                    let w = mag_sq.sqrt().max(mu);
                    semi += mag_sq / w;
                    nsq += mag_sq / (w * w);

                    let inv_w = 1.0 / w;
                    d1s[q] = masks.x[idx] * a * inv_w;
                    d2s[q] = masks.y[idx] * b * inv_w;
                    d3s[q] = masks.z[idx] * c * inv_w;
                }
            }

            acc[0] = (semi, nsq);
        });

    let semi_norm: f64 = ws.partials.iter().map(|p| p.0).sum();
    let norm_u_sq: f64 = ws.partials.iter().map(|p| p.1).sum();

    // ── Pass 2: ∇f = Dᵀ(M u), summed over the three directions ──
    let (d1, d2, d3) = (&ws.d1, &ws.d2, &ws.d3);
    crate::maybe_par_chunks_mut!(df, slab)
        .enumerate()
        .for_each(|(k, out)| {
            let base = k * slab;
            for j in 0..ny {
                for i in 0..nx {
                    let q = i + j * nx;
                    let idx = base + q;

                    // Forward differences, zero on the trailing face of each axis.
                    let mut v = 0.0;
                    if i + 1 < nx {
                        v += (d1[idx] - d1[idx + 1]) * ix;
                    }
                    if j + 1 < ny {
                        v += (d2[idx] - d2[idx + nx]) * iy;
                    }
                    if k + 1 < nz {
                        v += (d3[idx] - d3[idx + slab]) * iz;
                    }
                    out[q] = v;
                }
            }
        });

    semi_norm - 0.5 * mu * norm_u_sq
}

/// NESTA with continuation: fill the dipole cone under a weighted-TV prior while
/// holding the well-conditioned coefficients of `chi_init` fixed.
fn nesta_cone_fill(
    chi_init: &[f64],
    cone: &[bool],
    masks: &HomogeneityMasks,
    grid: &Grid,
    params: &HeidiParams,
    mut progress: impl FnMut(usize, usize),
) -> Vec<f64> {
    let (nx, ny, nz) = grid.dims;
    let n = grid.n_total();

    let mut fft = Fft3dWorkspace::new(nx, ny, nz);
    let mut cbuf: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n];

    // P(x) = F⁻¹( cone · F x ): projection onto the well-conditioned subspace.
    // A = restrict∘F and Aᵀ = F⁻¹∘zero-fill satisfy A·Aᵀ = I, so P is idempotent
    // and the NESTA data step below is an exact projection.
    let project = |x: &[f64], out: &mut [f64], fft: &mut Fft3dWorkspace, cbuf: &mut Vec<Complex64>| {
        for (c, &v) in cbuf.iter_mut().zip(x) {
            *c = Complex64::new(v, 0.0);
        }
        fft.fft3d(cbuf);
        for (c, &keep) in cbuf.iter_mut().zip(cone) {
            if !keep {
                *c = Complex64::new(0.0, 0.0);
            }
        }
        fft.ifft3d(cbuf);
        for (o, c) in out.iter_mut().zip(cbuf.iter()) {
            *o = c.re;
        }
    };

    // The data term: the cone content of chi_init, which every iterate must match.
    let mut p_b = vec![0.0; n];
    project(chi_init, &mut p_b, &mut fft, &mut cbuf);

    // TV gradients use relative voxel sizes, as upstream normalises them.
    let (vsx, vsy, vsz) = grid.voxel_size;
    let vmax = vsx.max(vsy).max(vsz);
    let tv_grid = Grid::new(nx, ny, nz, vsx / vmax, vsy / vmax, vsz / vmax);

    // Lipschitz constant of the smoothed TV gradient: ‖D‖² / μ, with ‖D‖² summed
    // over the three difference operators (2/v per direction).
    let lip_geometry = 4.0 / (vsx / vmax).powi(2)
        + 4.0 / (vsy / vmax).powi(2)
        + 4.0 / (vsz / vmax).powi(2);

    let mu0 = 0.9 * p_b.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    if mu0 <= 0.0 {
        // Nothing in the cone — no data to honour, and TV alone gives zero.
        return vec![0.0; n];
    }

    let steps = params.continuation_steps.max(1);
    let gamma = (params.mu_min / mu0).powf(1.0 / steps as f64);

    let mut ws = TvWorkspace::new(&tv_grid);

    let mut xplug = p_b.clone();
    let mut df = vec![0.0; n];
    let mut wk = vec![0.0; n];
    let mut cp = vec![0.0; n];
    let mut proj = vec![0.0; n];
    let mut yk = vec![0.0; n];
    let mut zk = vec![0.0; n];

    let total_iters = steps * params.inner_iterations;
    let mut done = 0usize;
    let mut mu = mu0;

    for _ in 0..steps {
        mu *= gamma;
        let lmu = lip_geometry / mu;
        let inv_lmu = 1.0 / lmu;

        let mut xk = xplug.clone();
        wk.fill(0.0);
        // `fmean` is a short history of objective values; the stopping test
        // compares the current objective against their mean.
        let mut fmean: Vec<f64> = vec![f64::MIN_POSITIVE / 10.0];

        for k in 0..params.inner_iterations {
            let fx = tv_objective(&xk, mu, masks, &tv_grid, &mut ws, &mut df);

            // Upstream aborts the whole run on divergence. Rewinding to this
            // step's starting point instead keeps the previous continuation
            // level's answer, which is a usable map rather than an error.
            if !fx.is_finite() {
                xk.copy_from_slice(&xplug);
                break;
            }

            // yk: gradient step from xk, projected back onto the constraint set.
            for i in 0..n {
                cp[i] = xk[i] - inv_lmu * df[i];
            }
            project(&cp, &mut proj, &mut fft, &mut cbuf);
            for i in 0..n {
                yk[i] = p_b[i] + cp[i] - proj[i];
            }

            let fbar = fmean.iter().sum::<f64>() / fmean.len() as f64;
            let qp = (fx - fbar).abs() / fbar;

            done += 1;
            progress(done, total_iters);

            // Upstream breaks before the momentum update, keeping the current
            // iterate rather than the freshly projected `yk`.
            if qp <= params.tol {
                break;
            }

            fmean.insert(0, fx);
            fmean.truncate(10);

            // zk: step from the anchor point along the accumulated weighted
            // gradient — the Nesterov momentum term.
            let apk = 0.5 * (k as f64 + 1.0);
            let tauk = 2.0 / (k as f64 + 3.0);
            for i in 0..n {
                wk[i] += apk * df[i];
                cp[i] = xplug[i] - inv_lmu * wk[i];
            }
            project(&cp, &mut proj, &mut fft, &mut cbuf);
            for i in 0..n {
                zk[i] = p_b[i] + cp[i] - proj[i];
                xk[i] = tauk * zk[i] + (1.0 - tauk) * yk[i];
            }
        }

        xplug = xk;
    }

    xplug
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_grid(n: usize) -> Grid {
        Grid::new(n, n, n, 1.0, 1.0, 1.0)
    }

    #[test]
    fn discrete_gradient_matches_upstream_edges() {
        let g = Grid::new(4, 1, 1, 1.0, 1.0, 1.0);
        let x = vec![1.0, 3.0, 6.0, 10.0];

        let fwd = discrete_gradient(&x, &g, 0, false);
        assert_eq!(fwd, vec![0.0, 2.0, 3.0, 4.0]);

        let tr = discrete_gradient(&x, &g, 0, true);
        assert_eq!(tr, vec![-2.0, -3.0, -4.0, 0.0]);
    }

    #[test]
    fn discrete_gradient_scales_by_voxel_size() {
        let g = Grid::new(3, 1, 1, 2.0, 1.0, 1.0);
        let x = vec![0.0, 4.0, 10.0];
        let fwd = discrete_gradient(&x, &g, 0, false);
        assert_eq!(fwd, vec![0.0, 2.0, 3.0]);
    }

    #[test]
    fn thresholds_convert_from_upstream_units() {
        // A 0.00105 threshold in phase/(TE[ms]·B0[T]) is ~3.93e-3 ppm/mm.
        assert!((DEFAULT_GRADIENT_THRESHOLD_PPM - 3.9249e-3).abs() < 1e-6);
        assert!((DEFAULT_LAPLACIAN_THRESHOLD_PPM - 0.059434).abs() < 1e-5);
    }

    #[test]
    fn homogeneity_masks_open_at_edges() {
        let n = 12;
        let g = test_grid(n);
        let n_total = g.n_total();

        // Flat left half, steep ramp on the right.
        let mut field = vec![0.0; n_total];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    field[i + j * n + k * n * n] =
                        if i < n / 2 { 0.0 } else { 0.05 * (i - n / 2) as f64 };
                }
            }
        }
        let mask = vec![1u8; n_total];
        let params = HeidiParams {
            denoise: None,
            apply_laplacian_correction: false,
            ..Default::default()
        };
        let m = homogeneity_masks(&field, &mask, &g, &params);

        // Deep in the flat region the x mask is on.
        let flat = 2 + (n / 2) * n + (n / 2) * n * n;
        assert_eq!(m.x[flat], 1.0);
        // On the ramp the gradient is 0.05 ppm/mm, far above the 3.9e-3 threshold.
        let ramp = (n / 2 + 2) + (n / 2) * n + (n / 2) * n * n;
        assert_eq!(m.x[ramp], 0.0);
        // y and z are flat everywhere, so the all-edge floor never triggers.
        assert_eq!(m.y[ramp], 1.0);
    }

    #[test]
    fn all_edge_voxels_get_the_floor() {
        let n = 8;
        let g = test_grid(n);
        let n_total = g.n_total();

        // A field that ramps steeply in every direction.
        let mut field = vec![0.0; n_total];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    field[i + j * n + k * n * n] = 0.1 * (i + j + k) as f64;
                }
            }
        }
        let mask = vec![1u8; n_total];
        let params = HeidiParams {
            denoise: None,
            apply_laplacian_correction: false,
            gradient_mask_floor: 0.1,
            ..Default::default()
        };
        let m = homogeneity_masks(&field, &mask, &g, &params);

        let inner = 4 + 4 * n + 4 * n * n;
        assert_eq!(m.x[inner], 0.1);
        assert_eq!(m.y[inner], 0.1);
        assert_eq!(m.z[inner], 0.1);
    }

    #[test]
    fn masks_are_zero_outside_the_brain() {
        let n = 8;
        let g = test_grid(n);
        let n_total = g.n_total();
        let field = vec![0.0; n_total];
        let mut mask = vec![0u8; n_total];
        mask[4 + 4 * n + 4 * n * n] = 1;

        let params = HeidiParams { denoise: None, ..Default::default() };
        let m = homogeneity_masks(&field, &mask, &g, &params);
        for (i, &inside) in mask.iter().enumerate() {
            if inside == 0 {
                assert_eq!(m.x[i], 0.0);
                assert_eq!(m.y[i], 0.0);
                assert_eq!(m.z[i], 0.0);
            }
        }
    }

    /// The defining property: coefficients outside the cone are free, but the
    /// well-conditioned ones must come back unchanged.
    #[test]
    fn cone_coefficients_are_preserved() {
        let n = 16;
        let g = test_grid(n);
        let n_total = g.n_total();
        let bdir = (0.0, 0.0, 1.0);

        let mut chi_init = vec![0.0; n_total];
        let c = n as f64 / 2.0;
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let r = ((i as f64 - c).powi(2)
                        + (j as f64 - c).powi(2)
                        + (k as f64 - c).powi(2))
                    .sqrt();
                    chi_init[i + j * n + k * n * n] = if r < 4.0 { 0.1 } else { 0.0 };
                }
            }
        }

        let field = vec![0.0; n_total];
        let mask = vec![1u8; n_total];
        let params = HeidiParams {
            denoise: None,
            continuation_steps: 3,
            inner_iterations: 20,
            ..Default::default()
        };
        let chi = heidi(&field, &mask, &chi_init, &g, bdir, &params, |_, _| {});

        // Compare the two maps' k-space inside the cone.
        let cone = cone_mask(&g, bdir, params.cone_threshold);
        let mut ws = Fft3dWorkspace::new(n, n, n);
        let mut a: Vec<Complex64> = chi_init.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        let mut b: Vec<Complex64> = chi.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        ws.fft3d(&mut a);
        ws.fft3d(&mut b);

        let mut worst: f64 = 0.0;
        let mut scale: f64 = 0.0;
        for i in 0..n_total {
            if cone[i] {
                worst = worst.max((a[i] - b[i]).norm());
                scale = scale.max(a[i].norm());
            }
        }
        assert!(
            worst < 1e-6 * scale.max(1.0),
            "cone coefficients drifted by {worst} (scale {scale})"
        );
    }

    /// The fused two-pass TV kernel must agree with the straightforward
    /// composition of `discrete_gradient` it replaced.
    #[test]
    fn fused_tv_matches_the_naive_composition() {
        let (nx, ny, nz) = (9usize, 7usize, 8usize);
        let g = Grid::new(nx, ny, nz, 1.0, 1.3, 0.8);
        let n = g.n_total();

        let x: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.37).sin()).collect();
        let masks = HomogeneityMasks {
            x: (0..n).map(|i| if i % 5 == 0 { 0.1 } else { 1.0 }).collect(),
            y: (0..n).map(|i| if i % 3 == 0 { 0.0 } else { 1.0 }).collect(),
            z: (0..n).map(|i| if i % 7 == 0 { 0.1 } else { 1.0 }).collect(),
        };
        let mu = 0.05;

        let mut ws = TvWorkspace::new(&g);
        let mut df = vec![0.0; n];
        let fx = tv_objective(&x, mu, &masks, &g, &mut ws, &mut df);

        // Reference: build each direction separately, then apply the adjoints.
        let mut d1 = discrete_gradient(&x, &g, 0, false);
        let mut d2 = discrete_gradient(&x, &g, 1, false);
        let mut d3 = discrete_gradient(&x, &g, 2, false);
        for i in 0..n {
            d1[i] *= masks.x[i];
            d2[i] *= masks.y[i];
            d3[i] *= masks.z[i];
        }
        let mut semi = 0.0;
        let mut nsq = 0.0;
        for i in 0..n {
            let mag_sq = d1[i] * d1[i] + d2[i] * d2[i] + d3[i] * d3[i];
            let w = mag_sq.sqrt().max(mu);
            semi += mag_sq / w;
            nsq += mag_sq / (w * w);
            d1[i] = masks.x[i] * d1[i] / w;
            d2[i] = masks.y[i] * d2[i] / w;
            d3[i] = masks.z[i] * d3[i] / w;
        }
        let ref_fx = semi - 0.5 * mu * nsq;

        let gx = discrete_gradient(&d1, &g, 0, true);
        let gy = discrete_gradient(&d2, &g, 1, true);
        let gz = discrete_gradient(&d3, &g, 2, true);

        assert!((fx - ref_fx).abs() < 1e-12, "objective {fx} vs {ref_fx}");
        for i in 0..n {
            let expect = gx[i] + gy[i] + gz[i];
            assert!(
                (df[i] - expect).abs() < 1e-12,
                "gradient mismatch at {i}: {} vs {expect}",
                df[i]
            );
        }
    }

    /// A prime-factored grid takes the padding path; the result must still come
    /// back at the caller's dimensions and match an already-fast grid's behaviour.
    #[test]
    fn padding_path_preserves_dimensions() {
        let (nx, ny, nz) = (7usize, 11usize, 13usize);
        assert_ne!(
            (
                crate::utils::padding::next_fast_fft_size(nx),
                crate::utils::padding::next_fast_fft_size(ny),
                crate::utils::padding::next_fast_fft_size(nz),
            ),
            (nx, ny, nz),
            "test grid must actually need padding"
        );

        let g = Grid::new(nx, ny, nz, 1.0, 1.0, 1.0);
        let n_total = g.n_total();
        let chi_init: Vec<f64> = (0..n_total).map(|i| ((i as f64) * 0.31).sin() * 0.05).collect();
        let params = HeidiParams {
            denoise: None,
            continuation_steps: 2,
            inner_iterations: 10,
            ..Default::default()
        };
        let chi = heidi(
            &vec![0.0; n_total],
            &vec![1u8; n_total],
            &chi_init,
            &g,
            (0.0, 0.0, 1.0),
            &params,
            |_, _| {},
        );
        assert_eq!(chi.len(), n_total);
        assert!(chi.iter().all(|v| v.is_finite()));
        assert!(chi.iter().any(|v| v.abs() > 0.0), "padded run produced an empty map");
    }

    #[test]
    fn zero_init_gives_zero_output() {
        let n = 8;
        let g = test_grid(n);
        let n_total = g.n_total();
        let params = HeidiParams {
            denoise: None,
            continuation_steps: 2,
            inner_iterations: 5,
            ..Default::default()
        };
        let chi = heidi(
            &vec![0.0; n_total],
            &vec![1u8; n_total],
            &vec![0.0; n_total],
            &g,
            (0.0, 0.0, 1.0),
            &params,
            |_, _| {},
        );
        assert!(chi.iter().all(|v| v.abs() < 1e-12));
    }

    /// The cone is where the TV prior actually does work: HEIDI should reduce
    /// total variation relative to just low-passing the initial map.
    #[test]
    fn tv_is_reduced_against_the_bare_cone_projection() {
        let n = 16;
        let g = test_grid(n);
        let n_total = g.n_total();
        let bdir = (0.0, 0.0, 1.0);

        let mut chi_init = vec![0.0; n_total];
        let c = n as f64 / 2.0;
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let r = ((i as f64 - c).powi(2)
                        + (j as f64 - c).powi(2)
                        + (k as f64 - c).powi(2))
                    .sqrt();
                    chi_init[i + j * n + k * n * n] = if r < 5.0 { 0.1 } else { 0.0 };
                }
            }
        }

        let field = vec![0.0; n_total];
        let mask = vec![1u8; n_total];
        let params = HeidiParams {
            denoise: None,
            continuation_steps: 4,
            inner_iterations: 60,
            ..Default::default()
        };
        let chi = heidi(&field, &mask, &chi_init, &g, bdir, &params, |_, _| {});

        // Bare projection = what you get with no prior at all.
        let cone = cone_mask(&g, bdir, params.cone_threshold);
        let mut ws = Fft3dWorkspace::new(n, n, n);
        let mut buf: Vec<Complex64> =
            chi_init.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        ws.fft3d(&mut buf);
        for (b, &keep) in buf.iter_mut().zip(&cone) {
            if !keep {
                *b = Complex64::new(0.0, 0.0);
            }
        }
        ws.ifft3d(&mut buf);
        let projected: Vec<f64> = buf.iter().map(|c| c.re).collect();

        let tv = |x: &[f64]| -> f64 {
            let dx = discrete_gradient(x, &g, 0, false);
            let dy = discrete_gradient(x, &g, 1, false);
            let dz = discrete_gradient(x, &g, 2, false);
            (0..n_total)
                .map(|i| (dx[i] * dx[i] + dy[i] * dy[i] + dz[i] * dz[i]).sqrt())
                .sum()
        };

        let tv_heidi = tv(&chi);
        let tv_proj = tv(&projected);
        assert!(
            tv_heidi < tv_proj,
            "HEIDI TV {tv_heidi} is not below the bare projection's {tv_proj}"
        );
    }
}

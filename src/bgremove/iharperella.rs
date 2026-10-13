//! HARPERELLA and iHARPERELLA — integrated phase unwrapping and background field removal
//!
//! Both algorithms simultaneously unwrap phase and remove background field by
//! estimating the phase Laplacian outside the brain. The background phase is
//! harmonic inside the brain (nabla^2 phi_bg = 0), so the wrapped Laplacian inside
//! the brain contains only tissue sources.
//!
//! ## Input
//!
//! What they start from is explicit ([`HarperellaInput`]), as for TGV-QSM:
//!
//! - [`HarperellaInput::WrappedPhase`]: one wrapped echo in rad, the published method. The
//!   Laplacian is the wrapped one (each neighbour difference wrapped to `(-π, π]`), which is
//!   what unwraps the phase. The result is divided by
//!   [`rad_per_ppm`](crate::inversion::rad_per_ppm)`(te_s, b0_t)`.
//! - [`HarperellaInput::FieldPpm`]: an unwrapped total field in ppm, e.g. a multi-echo field
//!   map. The Laplacian is the ordinary one: wrapping a ppm field's neighbour differences, as
//!   if they were radians, takes `sin`-like liberties where the field is steep (sinuses,
//!   strong sources), and only there.
//!
//! Either way the output is the local field in **ppm**.
//!
//! Both solvers are linear in the data and stop on a *relative* residual (`tol` × the norm of
//! the right-hand side), and there is no regularisation weight. So the same `max_iter`/`tol`
//! mean the same thing for either input, whatever the field's units: scaling the data by `c`
//! scales the solution by `c` and leaves the iteration count unchanged (exactly in exact
//! arithmetic; a CG stopped well short of convergence amplifies rounding, so two inputs that
//! differ only by rounding agree to the solver's accuracy, not to the last bit).
//!
//! **HARPERELLA** estimates the exterior Laplacian by making the SMV of the total
//! Laplacian uniform across the FOV (Eq [3] in the paper):
//!   min ||S(nabla^2 phi_E) + S(nabla^2 phi_brain) - delta||_2
//!
//! **iHARPERELLA** instead directly minimizes the weighted resulting phase
//! (Eq [3] in the ISMRM abstract):
//!   min ||W * inv_lap(nabla^2 phi_brain + nabla^2 phi_out)||_2
//! providing more robust low-frequency background suppression.
//!
//! References:
//! - HARPERELLA: Li, W., et al. (2014). "Integrated Laplacian-based phase
//!   unwrapping and background phase removal for quantitative susceptibility
//!   mapping." NMR in Biomedicine, 27(2):219-227. doi:10.1002/nbm.3056
//! - iHARPERELLA: Li, W., Wu, B., Liu, C. (2015). "iHARPERELLA: an improved
//!   method for integrated 3D phase unwrapping and background phase removal."
//!   Proc. ISMRM 23, p.3313.

use num_complex::Complex64;
use crate::Grid;
use crate::error::{check_len, InputError};
use crate::fft::{fft3d, ifft3d};
use crate::kernels::smv::smv_kernel;
use crate::unwrap::laplacian::{laplacian_periodic, wrapped_laplacian_periodic};

/// What HARPERELLA / iHARPERELLA start from. See the module docs.
#[derive(Clone, Copy, Debug)]
pub enum HarperellaInput<'a> {
    /// Unwrapped total field in ppm (e.g. a multi-echo field map). Ordinary discrete Laplacian.
    FieldPpm(&'a [f64]),
    /// One wrapped echo in rad (the published method). Wrapped Laplacian; the result is
    /// converted to ppm with [`rad_per_ppm`](crate::inversion::rad_per_ppm)`(te_s, b0_t)`.
    WrappedPhase {
        /// Wrapped phase in rad.
        phase: &'a [f64],
        /// Echo time of `phase` in seconds.
        te_s: f64,
        /// Field strength in Tesla.
        b0_t: f64,
    },
}

impl HarperellaInput<'_> {
    /// Check the input against the grid and mask; return its Laplacian in the input's own
    /// units, and for phase the rad-per-ppm factor the result is divided by.
    fn laplacian(&self, mask: &[u8], grid: &Grid) -> Result<(Vec<f64>, Option<f64>), InputError> {
        let (nx, ny, nz) = grid.dims;
        let (vsx, vsy, vsz) = grid.voxel_size;
        let n = grid.n_total();
        check_len("mask", mask.len(), n)?;
        match *self {
            HarperellaInput::FieldPpm(field) => {
                check_len("field", field.len(), n)?;
                Ok((laplacian_periodic(field, nx, ny, nz, vsx, vsy, vsz), None))
            }
            HarperellaInput::WrappedPhase { phase, te_s, b0_t } => {
                check_len("phase", phase.len(), n)?;
                if !(te_s.is_finite() && te_s > 0.0 && b0_t.is_finite() && b0_t > 0.0) {
                    return Err(InputError(format!(
                        "wrapped-phase input needs a positive echo time and field strength, \
                         got TE = {te_s} s, B0 = {b0_t} T"
                    )));
                }
                Ok((
                    wrapped_laplacian_periodic(phase, nx, ny, nz, vsx, vsy, vsz),
                    Some(crate::inversion::rad_per_ppm(te_s, b0_t)),
                ))
            }
        }
    }
}

/// HARPERELLA parameters.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct HarperellaParams {
    /// SMV kernel radius in mm (paper uses 10mm for in vivo)
    pub radius: f64,
    /// Maximum CG iterations for exterior Laplacian estimation
    pub max_iter: usize,
    /// CG convergence tolerance, relative to the right-hand side
    pub tol: f64,
}

impl Default for HarperellaParams {
    fn default() -> Self {
        Self {
            radius: 10.0,
            max_iter: 40,
            tol: 1e-6,
        }
    }
}

/// iHARPERELLA parameters. It has no SMV kernel, so no radius.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct IharperellaParams {
    /// Maximum CG iterations for exterior Laplacian estimation
    pub max_iter: usize,
    /// CG convergence tolerance, relative to the right-hand side
    pub tol: f64,
}

impl Default for IharperellaParams {
    fn default() -> Self {
        Self {
            max_iter: 40,
            tol: 1e-6,
        }
    }
}

// =============================================================================
// HARPERELLA (Li et al., NMR Biomed 2014)
// Exterior estimation via SMV uniformity in Laplacian domain
// =============================================================================

/// HARPERELLA — integrated phase unwrapping and background removal
///
/// Estimates exterior Laplacian by enforcing uniform SMV across the FOV.
///
/// # Arguments
/// * `input` - Wrapped phase (rad) or unwrapped total field (ppm); see [`HarperellaInput`]
/// * `mask` - Binary brain mask (nx * ny * nz), 1 = inside
/// * `grid` - Volume dimensions and voxel sizes
/// * `params` - Algorithm parameters
/// * `progress` - Progress callback (iteration, max_iter)
///
/// # Returns
/// `(local_field_ppm, mask)`: the background-free local field in ppm, and the brain mask.
///
/// # Errors
/// A buffer whose length does not match `grid`, or a `WrappedPhase` echo time or field
/// strength that is not positive.
pub fn harperella(
    input: HarperellaInput,
    mask: &[u8],
    grid: &Grid,
    params: &HarperellaParams,
    mut progress: impl FnMut(usize, usize),
) -> Result<(Vec<f64>, Vec<u8>), InputError> {
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = grid.voxel_size;
    let n_total = nx * ny * nz;
    let (lap, rad_per_ppm) = input.laplacian(mask, grid)?;
    let (lap_brain, interior_mask, exterior_mask) =
        prepare_laplacian(lap, mask, nx, ny, nz, vsx, vsy, vsz);

    // SMV kernel
    let s_fft = compute_smv_fft(nx, ny, nz, vsx, vsy, vsz, params.radius);

    // Compute delta = mean of S(nabla^2 phi) over trustable interior I (Eq [4])
    let smv_lap = apply_smv(&lap_brain, &s_fft, nx, ny, nz);
    let mut delta_sum = 0.0;
    let mut delta_count = 0usize;
    for i in 0..n_total {
        if interior_mask[i] == 1 {
            delta_sum += smv_lap[i];
            delta_count += 1;
        }
    }
    let delta_val = if delta_count > 0 { delta_sum / delta_count as f64 } else { 0.0 };

    // Solve: min ||S(nabla^2 phi_E) + S(nabla^2 phi_brain) - delta||_2  (Eq [3])
    // Normal equations: ext * S(S(ext * x)) = ext * S(delta - S(lap_brain))
    let rhs: Vec<f64> = (0..n_total).map(|i| delta_val - smv_lap[i]).collect();
    let smv_rhs = apply_smv(&rhs, &s_fft, nx, ny, nz);
    let atb: Vec<f64> = (0..n_total).map(|i| exterior_mask[i] * smv_rhs[i]).collect();

    let lap_ext = cg_solve_masked(
        &atb, &exterior_mask, params.max_iter, params.tol, &mut progress,
        |x| {
            let masked: Vec<f64> = (0..n_total).map(|i| exterior_mask[i] * x[i]).collect();
            let sx = apply_smv(&masked, &s_fft, nx, ny, nz);
            let ssx = apply_smv(&sx, &s_fft, nx, ny, nz);
            (0..n_total).map(|i| exterior_mask[i] * ssx[i]).collect()
        },
    );

    // Combine and inverse Laplacian
    let mut lap_fov: Vec<f64> = lap_brain;
    for i in 0..n_total { lap_fov[i] += exterior_mask[i] * lap_ext[i]; }
    let tissue_phase = solve_poisson(&lap_fov, nx, ny, nz, vsx, vsy, vsz);
    Ok((to_local_field(&tissue_phase, mask, rad_per_ppm), mask.to_vec()))
}

// =============================================================================
// iHARPERELLA (Li et al., ISMRM 2015 #3313)
// Exterior estimation via direct phase minimization
// =============================================================================

/// iHARPERELLA — improved integrated phase unwrapping and background removal
///
/// More robust low-frequency background suppression than HARPERELLA.
/// Estimates exterior Laplacian by directly minimizing the weighted resulting
/// phase rather than enforcing SMV uniformity.
///
/// # Arguments
/// * `input` - Wrapped phase (rad) or unwrapped total field (ppm); see [`HarperellaInput`]
/// * `mask` - Binary brain mask (nx * ny * nz), 1 = inside
/// * `grid` - Volume dimensions and voxel sizes
/// * `params` - Algorithm parameters
/// * `progress` - Progress callback (iteration, max_iter)
///
/// # Returns
/// `(local_field_ppm, mask)`
///
/// # Errors
/// As [`harperella`].
pub fn iharperella(
    input: HarperellaInput,
    mask: &[u8],
    grid: &Grid,
    params: &IharperellaParams,
    progress: impl FnMut(usize, usize),
) -> Result<(Vec<f64>, Vec<u8>), InputError> {
    let w_brain: Vec<f64> = mask.iter().map(|&m| m as f64).collect();
    iharperella_with_weights(input, mask, grid, params, &w_brain, progress)
}

/// iHARPERELLA with custom W_Brain weighting
///
/// Allows specifying an arbitrary weighting function for the phase minimization.
/// The weighting should be non-negative and defined over the full volume (nx*ny*nz).
///
/// # Arguments
/// * `input` - Wrapped phase (rad) or unwrapped total field (ppm); see [`HarperellaInput`]
/// * `mask` - Binary brain mask (nx * ny * nz), 1 = inside
/// * `grid` - Volume dimensions and voxel sizes
/// * `params` - Algorithm parameters
/// * `w_brain` - Weighting function for phase minimization
/// * `progress` - Progress callback (iteration, max_iter)
///
/// # Returns
/// `(local_field_ppm, mask)`
///
/// # Errors
/// As [`harperella`], and a `w_brain` whose length does not match `grid`.
pub fn iharperella_with_weights(
    input: HarperellaInput,
    mask: &[u8],
    grid: &Grid,
    params: &IharperellaParams,
    w_brain: &[f64],
    mut progress: impl FnMut(usize, usize),
) -> Result<(Vec<f64>, Vec<u8>), InputError> {
    let (nx, ny, nz) = grid.dims;
    let (vsx, vsy, vsz) = grid.voxel_size;
    let n_total = nx * ny * nz;
    let (max_iter, tol) = (params.max_iter, params.tol);
    let (lap, rad_per_ppm) = input.laplacian(mask, grid)?;
    check_len("w_brain", w_brain.len(), n_total)?;
    let (lap_brain, _interior_mask, exterior_mask) =
        prepare_laplacian(lap, mask, nx, ny, nz, vsx, vsy, vsz);

    // Pre-compute Laplacian eigenvalues for inverse Laplacian
    let lap_eig = compute_laplacian_eigenvalues(nx, ny, nz, vsx, vsy, vsz);

    // Solve: min ||W * inv_lap(nabla^2 phi_brain + nabla^2 phi_out * (1-M))||_2  (Eq [3])
    //
    // A(x) = W * inv_lap(x * (1-M))
    // A'(y) = (1-M) * inv_lap(W * y)    [inv_lap is self-adjoint]
    // b = -W * inv_lap(nabla^2 phi_brain)
    //
    // Normal equations: A'A(x) = A'(-b)
    // (1-M) * inv_lap(W^2 * inv_lap((1-M) * x)) = (1-M) * inv_lap(W^2 * inv_lap(nabla^2 phi_brain))

    // Compute -A'c = -(1-M) * inv_lap(W^2 * inv_lap(nabla^2 phi_brain))
    // The objective is min ||Ax + c||_2 where c = W * inv_lap(nabla^2 phi_brain * M)
    // Normal equations: A'Ax = -A'c
    let phase_from_brain = apply_inv_lap(&lap_brain, &lap_eig, nx, ny, nz);
    let w2_phase: Vec<f64> = (0..n_total).map(|i| w_brain[i] * w_brain[i] * phase_from_brain[i]).collect();
    let atb_raw = apply_inv_lap(&w2_phase, &lap_eig, nx, ny, nz);
    let atb: Vec<f64> = (0..n_total).map(|i| -exterior_mask[i] * atb_raw[i]).collect();

    let lap_ext = cg_solve_masked(
        &atb, &exterior_mask, max_iter, tol, &mut progress,
        |x| {
            // A'A(x) = (1-M) * inv_lap(W^2 * inv_lap((1-M) * x))
            let masked: Vec<f64> = (0..n_total).map(|i| exterior_mask[i] * x[i]).collect();
            let inv1 = apply_inv_lap(&masked, &lap_eig, nx, ny, nz);
            let w2_inv1: Vec<f64> = (0..n_total).map(|i| w_brain[i] * w_brain[i] * inv1[i]).collect();
            let inv2 = apply_inv_lap(&w2_inv1, &lap_eig, nx, ny, nz);
            (0..n_total).map(|i| exterior_mask[i] * inv2[i]).collect()
        },
    );

    // Combine and inverse Laplacian
    let mut lap_fov = lap_brain;
    for i in 0..n_total { lap_fov[i] += exterior_mask[i] * lap_ext[i]; }
    let tissue_phase = solve_poisson(&lap_fov, nx, ny, nz, vsx, vsy, vsz);
    Ok((to_local_field(&tissue_phase, mask, rad_per_ppm), mask.to_vec()))
}

// =============================================================================
// Shared infrastructure
// =============================================================================

/// Mask the result to the brain and, for phase input, convert rad to ppm.
fn to_local_field(tissue: &[f64], mask: &[u8], rad_per_ppm: Option<f64>) -> Vec<f64> {
    tissue.iter().zip(mask)
        .map(|(&v, &m)| match (m != 0, rad_per_ppm) {
            (false, _) => 0.0,
            (true, None) => v,
            (true, Some(c)) => v / c,
        })
        .collect()
}

/// Erode mask, prepare interior/exterior masks, keep the Laplacian (Eq [1], computed by
/// [`HarperellaInput::laplacian`]) only at trustable interior voxels.
#[allow(clippy::too_many_arguments)]
fn prepare_laplacian(
    lap: Vec<f64>, mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> (Vec<f64>, Vec<u8>, Vec<f64>) {
    let n_total = nx * ny * nz;

    // Erode mask by ~3 voxels
    let erode_radius = 3.0 * vsx.min(vsy).min(vsz);
    let erode_fft = compute_smv_fft(nx, ny, nz, vsx, vsy, vsz, erode_radius);

    let mask_f64: Vec<f64> = mask.iter().map(|&m| m as f64).collect();
    let mut mask_conv: Vec<Complex64> = mask_f64.iter()
        .map(|&x| Complex64::new(x, 0.0)).collect();
    fft3d(&mut mask_conv, nx, ny, nz);
    for i in 0..n_total { mask_conv[i] *= erode_fft[i]; }
    ifft3d(&mut mask_conv, nx, ny, nz);

    let delta_thresh = 1.0 - 1e-7_f64.sqrt();
    let interior_mask: Vec<u8> = mask_conv.iter()
        .map(|c| if c.re > delta_thresh { 1 } else { 0 }).collect();
    let exterior_mask: Vec<f64> = mask.iter()
        .map(|&m| if m == 0 { 1.0 } else { 0.0 }).collect();

    // Keep Laplacian only at trustable interior voxels
    let mut lap_brain = vec![0.0; n_total];
    for i in 0..n_total {
        if interior_mask[i] == 1 { lap_brain[i] = lap[i]; }
    }

    (lap_brain, interior_mask, exterior_mask)
}

/// Compute FFT of SMV kernel (returns real part, since kernel is symmetric)
fn compute_smv_fft(
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64, radius: f64,
) -> Vec<f64> {
    let grid = Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let s_kernel = smv_kernel(&grid, radius);
    let mut c: Vec<Complex64> = s_kernel.iter()
        .map(|&x| Complex64::new(x, 0.0)).collect();
    fft3d(&mut c, nx, ny, nz);
    c.iter().map(|v| v.re).collect()
}

/// Apply SMV filter in k-space: S(x) = ifft(s_fft * fft(x))
fn apply_smv(x: &[f64], s_fft: &[f64], nx: usize, ny: usize, nz: usize) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut c: Vec<Complex64> = x.iter()
        .map(|&v| Complex64::new(v, 0.0)).collect();
    fft3d(&mut c, nx, ny, nz);
    for i in 0..n_total { c[i] *= s_fft[i]; }
    ifft3d(&mut c, nx, ny, nz);
    c.iter().map(|v| v.re).collect()
}

/// Compute discrete Laplacian eigenvalues for inverse Laplacian
fn compute_laplacian_eigenvalues(
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    use std::f64::consts::PI;
    let n_total = nx * ny * nz;
    let idx2 = 1.0 / (vsx * vsx);
    let idy2 = 1.0 / (vsy * vsy);
    let idz2 = 1.0 / (vsz * vsz);
    let mut eig = vec![0.0; n_total];

    for k in 0..nz {
        let fk = if k <= nz / 2 { k as f64 / nz as f64 } else { (k as f64 - nz as f64) / nz as f64 };
        let lz = 2.0 * ((2.0 * PI * fk).cos() - 1.0) * idz2;
        for j in 0..ny {
            let fj = if j <= ny / 2 { j as f64 / ny as f64 } else { (j as f64 - ny as f64) / ny as f64 };
            let ly = 2.0 * ((2.0 * PI * fj).cos() - 1.0) * idy2;
            for i in 0..nx {
                let fi = if i <= nx / 2 { i as f64 / nx as f64 } else { (i as f64 - nx as f64) / nx as f64 };
                let lx = 2.0 * ((2.0 * PI * fi).cos() - 1.0) * idx2;
                eig[i + j * nx + k * nx * ny] = lx + ly + lz;
            }
        }
    }
    eig
}

/// Apply inverse Laplacian using pre-computed eigenvalues
fn apply_inv_lap(f: &[f64], eig: &[f64], nx: usize, ny: usize, nz: usize) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut c: Vec<Complex64> = f.iter()
        .map(|&x| Complex64::new(x, 0.0)).collect();
    fft3d(&mut c, nx, ny, nz);
    for i in 0..n_total {
        if eig[i].abs() > 1e-20 {
            c[i] /= eig[i];
        } else {
            c[i] = Complex64::new(0.0, 0.0);
        }
    }
    ifft3d(&mut c, nx, ny, nz);
    c.iter().map(|v| v.re).collect()
}

/// Solve Poisson equation via FFT: nabla^2 u = f -> u
fn solve_poisson(
    f: &[f64], nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
) -> Vec<f64> {
    let eig = compute_laplacian_eigenvalues(nx, ny, nz, vsx, vsy, vsz);
    apply_inv_lap(f, &eig, nx, ny, nz)
}

/// Generic CG solver for masked normal equations
fn cg_solve_masked<F, Op>(
    b: &[f64],
    _mask: &[f64],
    max_iter: usize,
    tol: f64,
    callback: &mut F,
    apply_ata: Op,
) -> Vec<f64>
where
    F: FnMut(usize, usize),
    Op: Fn(&[f64]) -> Vec<f64>,
{
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rsold: f64 = r.iter().map(|&v| v * v).sum();
    let b_norm: f64 = b.iter().map(|&v| v * v).sum::<f64>().sqrt();

    if b_norm < 1e-20 { return x; }

    for iter in 0..max_iter {
        callback(iter + 1, max_iter);

        let ap = apply_ata(&p);
        let pap: f64 = p.iter().zip(ap.iter()).map(|(&pi, &api)| pi * api).sum();

        if pap.abs() < 1e-20 { break; }

        let alpha = rsold / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let rsnew: f64 = r.iter().map(|&v| v * v).sum();
        if rsnew.sqrt() < tol * b_norm { break; }

        let beta = rsnew / rsold;
        for i in 0..n { p[i] = r[i] + beta * p[i]; }
        rsold = rsnew;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unwrap::laplacian::wrap;

    const N: usize = 16;

    fn grid() -> Grid {
        Grid::new(N, N, N, 1.0, 1.0, 1.0)
    }

    /// A sphere of radius 6 voxels in the middle of the FOV.
    fn sphere_mask() -> Vec<u8> {
        let c = (N as f64 - 1.0) / 2.0;
        (0..N * N * N)
            .map(|idx| {
                let (i, j, k) = (idx % N, (idx / N) % N, idx / (N * N));
                let r2 = (i as f64 - c).powi(2) + (j as f64 - c).powi(2) + (k as f64 - c).powi(2);
                u8::from(r2 <= 36.0)
            })
            .collect()
    }

    /// A smooth field in ppm: a Gaussian blob (1.5 ppm peak, slope at most 0.32 ppm per voxel)
    /// inside the mask on a smooth periodic background (0.3 ppm, slope at most 0.12).
    fn field_ppm(scale: f64) -> Vec<f64> {
        let c = (N as f64 - 1.0) / 2.0;
        (0..N * N * N)
            .map(|idx| {
                let (x, y, z) = ((idx % N) as f64 - c, ((idx / N) % N) as f64 - c, (idx / (N * N)) as f64 - c);
                let blob = 1.5 * (-(x * x + y * y + z * z) / 16.0).exp();
                let bg = 0.3 * (2.0 * std::f64::consts::PI * x / N as f64).sin();
                scale * (blob + bg)
            })
            .collect()
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    fn max_abs(a: &[f64]) -> f64 {
        a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
    }

    fn run_h(input: HarperellaInput, mask: &[u8]) -> Vec<f64> {
        let p = HarperellaParams { radius: 3.0, max_iter: 30, tol: 1e-8 };
        harperella(input, mask, &grid(), &p, |_, _| {}).unwrap().0
    }

    fn run_ih(input: HarperellaInput, mask: &[u8]) -> Vec<f64> {
        let p = IharperellaParams { max_iter: 30, tol: 1e-8 };
        iharperella(input, mask, &grid(), &p, |_, _| {}).unwrap().0
    }

    #[test]
    fn zero_field_gives_zero() {
        let zero = vec![0.0; N * N * N];
        let mask = sphere_mask();
        for out in [run_h(HarperellaInput::FieldPpm(&zero), &mask), run_ih(HarperellaInput::FieldPpm(&zero), &mask)] {
            assert!(max_abs(&out) < 1e-12);
        }
    }

    #[test]
    fn wrapped_phase_is_finite() {
        let phase: Vec<f64> = (0..N * N * N).map(|i| wrap((i as f64) * 0.1)).collect();
        let mask = vec![1u8; N * N * N];
        let input = HarperellaInput::WrappedPhase { phase: &phase, te_s: 0.02, b0_t: 3.0 };
        assert!(run_h(input, &mask).iter().all(|v| v.is_finite()));
        assert!(run_ih(input, &mask).iter().all(|v| v.is_finite()));
    }

    /// Wrapped phase of a field whose neighbour differences stay below π is the same data as
    /// the field itself: the Laplacian both methods start from (`HarperellaInput::laplacian`)
    /// is the same to rounding once divided by rad/ppm, whatever the echo time; and the local
    /// field iHARPERELLA solves for from it is the same to the solver's accuracy. Run to
    /// convergence: a CG stopped early amplifies the rounding-level difference between the two
    /// right-hand sides (to ~1e-2 after 30 iterations here). HARPERELLA shares the Laplacian
    /// and needs ~400 iterations to converge on this volume, too slow for a unit test.
    #[test]
    fn wrapped_phase_matches_field_ppm_and_is_independent_of_te() {
        let field = field_ppm(1.0);
        let mask = sphere_mask();
        let b0 = 3.0;
        let (lap_field, none) = HarperellaInput::FieldPpm(&field).laplacian(&mask, &grid()).unwrap();
        assert!(none.is_none());
        let p = IharperellaParams { max_iter: 150, tol: 1e-12 };
        let from_field = iharperella(HarperellaInput::FieldPpm(&field), &mask, &grid(), &p, |_, _| {}).unwrap().0;
        assert!(max_abs(&from_field) > 1e-3, "nothing came through");
        // c = 5.6 and 6.4 rad/ppm: the field (peak 1.5 ppm, slope under 0.44 ppm per voxel)
        // wraps, and neighbour differences stay under π.
        for te in [0.007, 0.008] {
            let c = crate::inversion::rad_per_ppm(te, b0);
            assert!(max_abs(&field) * c > 2.0 * std::f64::consts::PI, "test field does not wrap");
            let phase: Vec<f64> = field.iter().map(|f| wrap(f * c)).collect();
            let input = HarperellaInput::WrappedPhase { phase: &phase, te_s: te, b0_t: b0 };
            let (lap_phase, rad_per_ppm) = input.laplacian(&mask, &grid()).unwrap();
            assert_eq!(rad_per_ppm, Some(c));
            let lap_phase: Vec<f64> = lap_phase.iter().map(|v| v / c).collect();
            let d = max_abs_diff(&lap_phase, &lap_field) / max_abs(&lap_field);
            assert!(d < 1e-12, "TE={te}: Laplacians differ by {d:e}");
            let from_phase = iharperella(input, &mask, &grid(), &p, |_, _| {}).unwrap().0;
            let d = max_abs_diff(&from_phase, &from_field) / max_abs(&from_field);
            assert!(d < 1e-5, "TE={te}: phase vs field differ by {d:e}");
        }
    }

    /// No regularisation and a relative stopping rule: scaling the field scales the result,
    /// so `tol` and `max_iter` mean the same for a field in ppm as for phase in rad. (A
    /// power-of-two factor scales every floating-point operation exactly, so the comparison is
    /// to rounding; any other factor agrees to the CG tolerance.)
    #[test]
    fn field_input_is_scale_equivariant() {
        let mask = sphere_mask();
        let (f1, f2) = (field_ppm(1.0), field_ppm(16.0));
        for (run, name) in [(run_h as fn(HarperellaInput, &[u8]) -> Vec<f64>, "HARPERELLA"), (run_ih, "iHARPERELLA")] {
            let a: Vec<f64> = run(HarperellaInput::FieldPpm(&f1), &mask).iter().map(|v| v * 16.0).collect();
            let b = run(HarperellaInput::FieldPpm(&f2), &mask);
            let d = max_abs_diff(&a, &b) / max_abs(&b);
            assert!(d < 1e-12, "{name}: not scale-equivariant ({d:e})");
        }
    }

    /// A field steep enough that neighbours differ by more than π (a ppm field read as rad
    /// would be wrapped there) goes through the plain Laplacian, so it is not distorted:
    /// the result is the scaled result of the gentle field.
    #[test]
    fn steep_field_is_not_wrapped() {
        let mask = sphere_mask();
        let steep = field_ppm(32.0);
        let lap_plain = laplacian_periodic(&steep, N, N, N, 1.0, 1.0, 1.0);
        let lap_wrapped = wrapped_laplacian_periodic(&steep, N, N, N, 1.0, 1.0, 1.0);
        assert!(max_abs_diff(&lap_plain, &lap_wrapped) > 1.0, "test field is not steep enough");
        let gentle: Vec<f64> = run_ih(HarperellaInput::FieldPpm(&field_ppm(1.0)), &mask)
            .iter().map(|v| v * 32.0).collect();
        let out = run_ih(HarperellaInput::FieldPpm(&steep), &mask);
        assert!(max_abs_diff(&out, &gentle) / max_abs(&gentle) < 1e-12);
    }

    #[test]
    fn bad_input_is_an_error_not_a_panic() {
        let n = N * N * N;
        let (field, mask) = (vec![0.0; n], vec![1u8; n]);
        let (g, hp, ip) = (grid(), HarperellaParams::default(), IharperellaParams::default());
        let short = vec![0.0; n - 1];
        assert!(harperella(HarperellaInput::FieldPpm(&short), &mask, &g, &hp, |_, _| {}).is_err());
        assert!(iharperella(HarperellaInput::FieldPpm(&field), &mask[1..], &g, &ip, |_, _| {}).is_err());
        for (te_s, b0_t) in [(0.0, 3.0), (0.02, 0.0), (f64::NAN, 3.0), (-0.01, 3.0)] {
            let input = HarperellaInput::WrappedPhase { phase: &field, te_s, b0_t };
            let err = harperella(input, &mask, &g, &hp, |_, _| {}).unwrap_err();
            assert!(err.to_string().contains("positive echo time"), "{err}");
            assert!(iharperella(input, &mask, &g, &ip, |_, _| {}).is_err());
        }
        assert!(iharperella_with_weights(HarperellaInput::FieldPpm(&field), &mask, &g, &ip, &short, |_, _| {})
            .is_err());
    }
}

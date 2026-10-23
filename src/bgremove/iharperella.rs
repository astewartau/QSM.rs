//! HARPERELLA and iHARPERELLA: integrated phase unwrapping and background field removal
//!
//! Both methods remove the background field through the Laplacian. The background field is
//! harmonic inside the brain, so the Laplacian there holds only tissue sources. They estimate
//! the Laplacian outside the brain, add it to the brain's, and invert the Laplacian over the
//! whole field of view.
//!
//! iHARPERELLA is a port of STI Suite 3.0's `iHARPERELLA` (its protected P-code was recovered by
//! black-box probing: logging the calls it makes to its helper functions and to MATLAB's
//! `lsqr`, and probing the `lsqr` operator it builds). HARPERELLA is not in STI Suite 3.0; it
//! follows the paper, and uses STI's building blocks for what the paper leaves open (see
//! [`harperella`]).
//!
//! ## Building blocks (STI Suite)
//!
//! - **Laplacian**: spectral, `L(x) = F⁻¹[k² F(x)]` with `k` in cycles/mm (STI's `fftnc` grid),
//!   which is `−∇²x / 4π²`. Its inverse divides by `k²` and sets the DC term to zero. The scale
//!   cancels between the two, so the result is in the input's units.
//! - **Boundary rim**: brain voxels with more than one voxel outside the brain among the 146
//!   neighbours of STI's `ball3D(3)` (the offsets with `1 ≤ |d|² ≤ 10` voxels²), i.e.
//!   `SMVFiltering(mask, 3) < 0.99`. Within the rim, the 20 % of Laplacian values largest in
//!   magnitude are set to zero (the paper's "large Laplacian values near the boundary of the
//!   brain ... were considered inaccurate and set to zero"). Outside the brain the Laplacian is
//!   zero.
//! - **Solver**: MATLAB's `lsqr` (no preconditioner, zero initial guess) at tolerance `tol`
//!   for at most `max_iter` iterations; STI passes 0.001 and `niter` (default 40).
//! - **Padding**: `pad` zero voxels on each side (STI's `padsize`, default 0).
//!
//! Past about ten iterations LSQR on these operators amplifies rounding: two implementations
//! whose residuals agree to 1e-14 for the first ten iterations differ by ~1e-3 by the twelfth.
//! So results agree with STI Suite to rounding for a few iterations and to a few percent at 40.
//!
//! ## Input
//!
//! - [`HarperellaInput::FieldPpm`]: an unwrapped total field in ppm, e.g. a multi-echo field
//!   map. This is what STI's `iHARPERELLA` does with its input: it takes the spectral
//!   Laplacian of it as it is (it does not unwrap).
//! - [`HarperellaInput::WrappedPhase`]: one wrapped echo in rad, as in the HARPERELLA paper:
//!   the Laplacian is `cos φ L(sin φ) − sin φ L(cos φ)` (the paper's Eq. 1) with the spectral
//!   `L`, and the result is divided by
//!   [`rad_per_ppm`](crate::inversion::rad_per_ppm)`(te_s, b0_t)`.
//!
//! Either way the output is the local field in **ppm**, inside the mask.
//!
//! ## Reference pipeline
//!
//! - **Input**: STI's help text and `STI_Templates.m` call it on the "raw phase", but probed
//!   in MATLAB R2023b, STI 3.0's `iHARPERELLA` does not unwrap: on a synthetic head whose
//!   phase wraps in 59 % of the brain, its output from the wrapped phase correlates −0.03 with
//!   the true local phase, from the unwrapped phase 0.98. Hence [`HarperellaInput::FieldPpm`].
//! - **Defaults**: `niter` 40; `padsize` 0 (probed: bitwise equal to `[0 0 0]`, while
//!   `STI_Templates.m` lists `[12 12 12]`). [`IharperellaParams::default`] matches the function.
//! - **Chain**: STI Suite 2.01's manual has the GUI combine the Laplacians of all echoes
//!   ("Combine Laplacian") and then run iHARPERELLA, as the alternative to "Unwrapping" +
//!   V-SHARP, before iLSQR; the mask is the GUI's magnitude threshold of the 3rd or 4th echo.
//!   STI Suite 3.0's GUI has no iHARPERELLA button (only the function).
//!
//! References:
//! - HARPERELLA: Li, W., Avram, A. V., Wu, B., Xiao, X., Liu, C. (2014). "Integrated
//!   Laplacian-based phase unwrapping and background phase removal for quantitative
//!   susceptibility mapping." NMR in Biomedicine, 27(2):219-227. doi:10.1002/nbm.3056
//! - iHARPERELLA: Li, W., Wu, B., Liu, C. (2015). "iHARPERELLA: an improved method for
//!   integrated 3D phase unwrapping and background phase removal." Proc. ISMRM 23, p.3313.
//!   Code: STI Suite 3.0, `Core_Functions_P/iHARPERELLA.p`.

use num_complex::Complex64;
use crate::Grid;
use crate::error::{check_len, InputError};
use crate::fft::{fft3d, ifft3d};
use crate::inversion::ilsqr::lsqr_matlab;
use crate::kernels::smv::smv_kernel;
use crate::utils::padding::{pad3d, unpad3d};

/// What HARPERELLA / iHARPERELLA start from. See the module docs.
#[derive(Clone, Copy, Debug)]
pub enum HarperellaInput<'a> {
    /// Unwrapped total field in ppm (e.g. a multi-echo field map). Spectral Laplacian.
    FieldPpm(&'a [f64]),
    /// One wrapped echo in rad. Laplacian by the paper's Eq. 1; the result is converted to
    /// ppm with [`rad_per_ppm`](crate::inversion::rad_per_ppm)`(te_s, b0_t)`.
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
    /// Check the input against the grid and mask; return the data and, for phase, the
    /// rad-per-ppm factor the result is divided by.
    fn data(&self, mask: &[u8], grid: &Grid) -> Result<(&[f64], Option<f64>), InputError> {
        let n = grid.n_total();
        check_len("mask", mask.len(), n)?;
        match *self {
            HarperellaInput::FieldPpm(field) => {
                check_len("field", field.len(), n)?;
                Ok((field, None))
            }
            HarperellaInput::WrappedPhase { phase, te_s, b0_t } => {
                check_len("phase", phase.len(), n)?;
                if !(te_s.is_finite() && te_s > 0.0 && b0_t.is_finite() && b0_t > 0.0) {
                    return Err(InputError(format!(
                        "wrapped-phase input needs a positive echo time and field strength, \
                         got TE = {te_s} s, B0 = {b0_t} T"
                    )));
                }
                Ok((phase, Some(crate::inversion::rad_per_ppm(te_s, b0_t))))
            }
        }
    }
}

/// HARPERELLA parameters.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct HarperellaParams {
    /// SMV kernel radius in mm (the paper uses 10 mm in vivo)
    pub radius: f64,
    /// Maximum LSQR iterations for the exterior Laplacian (STI's `niter`, default 40)
    pub max_iter: usize,
    /// LSQR tolerance (MATLAB `lsqr`'s `tol`; 0.001 as in STI's iHARPERELLA)
    pub tol: f64,
    /// Zero padding in voxels on each side (STI's `padsize`, default 0)
    pub pad: [usize; 3],
}

impl Default for HarperellaParams {
    fn default() -> Self {
        Self { radius: 10.0, max_iter: 40, tol: 1e-3, pad: [0, 0, 0] }
    }
}

/// iHARPERELLA parameters (STI Suite's `iHARPERELLA` options). It has no SMV kernel.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct IharperellaParams {
    /// Maximum LSQR iterations (STI's `niter`, default 40)
    pub max_iter: usize,
    /// LSQR tolerance (STI passes 0.001 to MATLAB's `lsqr`)
    pub tol: f64,
    /// Zero padding in voxels on each side (STI's `padsize`, default 0)
    pub pad: [usize; 3],
}

impl Default for IharperellaParams {
    fn default() -> Self {
        Self { max_iter: 40, tol: 1e-3, pad: [0, 0, 0] }
    }
}

/// The padded working volume: data, mask, Laplacian (rim cleaned), and the padded grid.
struct Work {
    dims: (usize, usize, usize),
    vs: (f64, f64, f64),
    mask: Vec<u8>,
    k2: Vec<f64>,
    lap: Vec<f64>,
    rim: Vec<u8>,
}

impl Work {
    fn new(data: &[f64], mask: &[u8], grid: &Grid, pad: [usize; 3], wrapped: bool) -> Self {
        let (nx, ny, nz) = grid.dims;
        let dims = (nx + 2 * pad[0], ny + 2 * pad[1], nz + 2 * pad[2]);
        let vs = grid.voxel_size;
        let pd = pad_centred(data, grid.dims, dims, pad);
        let pm = pad_centred(mask, grid.dims, dims, pad);
        let k2 = k2_kernel(dims, vs);
        let mut lap = if wrapped {
            let (sin, cos): (Vec<f64>, Vec<f64>) = pd.iter().map(|p| p.sin_cos()).unzip();
            let ls = apply_kernel(&sin, &k2, dims);
            let lc = apply_kernel(&cos, &k2, dims);
            (0..pd.len()).map(|i| cos[i] * ls[i] - sin[i] * lc[i]).collect()
        } else {
            apply_kernel(&pd, &k2, dims)
        };
        for (l, &m) in lap.iter_mut().zip(&pm) {
            if m == 0 { *l = 0.0; }
        }
        let rim = sti_rim(&pm, dims);
        drop_large_rim_values(&mut lap, &rim);
        Self { dims, vs, mask: pm, k2, lap, rim }
    }

    fn n(&self) -> usize { self.mask.len() }

    /// Invert the Laplacian, crop the padding, keep the brain, convert to ppm.
    fn finish(&self, lap_fov: &[f64], mask: &[u8], grid: &Grid, pad: [usize; 3], rad_per_ppm: Option<f64>) -> Vec<f64> {
        let tissue = apply_inv_kernel(lap_fov, &self.k2, self.dims);
        let tissue = unpad_centred(&tissue, self.dims, grid.dims, pad);
        tissue.iter().zip(mask)
            .map(|(&v, &m)| match (m != 0, rad_per_ppm) {
                (false, _) => 0.0,
                (true, None) => v,
                (true, Some(c)) => v / c,
            })
            .collect()
    }
}

// =============================================================================
// HARPERELLA (Li et al., NMR Biomed 2014)
// =============================================================================

/// HARPERELLA: the exterior Laplacian makes the SMV of the total Laplacian uniform.
///
/// Solves the paper's Eq. 3, `min ‖S(∇²φ_E) + S(∇²φ_{I+O}) − δ‖₂` over the field of view, for
/// the Laplacian `∇²φ_E` outside the brain, where `S` is the spherical mean of `radius` mm and
/// `δ` (Eq. 4) is the mean of `S(∇²φ)` over the trustable interior I. The paper splits the
/// tissue into I and the boundary region O, "the set of tissue voxels next to the boundaries
/// that are within a distance of the radius of the spherical mean value filter", so I is the
/// mask eroded by `radius`. The local field is the inverse Laplacian of the brain's Laplacian
/// plus the exterior one (Eqs. 5-6).
///
/// STI Suite 3.0 does not include HARPERELLA, so where the paper is silent this uses what STI's
/// iHARPERELLA does (module docs): the spectral Laplacian and its inverse (the paper's Eq. 6
/// is the same inverse), the boundary rim and its 20 % of large values set to zero (the paper:
/// "large Laplacian values near the boundary of the brain (<3 voxels from the boundary)"),
/// MATLAB `lsqr` at tolerance 0.001 (the paper: "the preconditioned LSQR method", preconditioner
/// unstated), and no padding by default (the paper pads, size unstated).
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
    let (data, rad_per_ppm) = input.data(mask, grid)?;
    let wrapped = rad_per_ppm.is_some();
    let w = Work::new(data, mask, grid, params.pad, wrapped);
    let (nx, ny, nz) = w.dims;
    let (vsx, vsy, vsz) = w.vs;
    let n = w.n();

    let s_fft = compute_smv_fft(nx, ny, nz, vsx, vsy, vsz, params.radius);
    let smv_lap = apply_smv(&w.lap, &s_fft, nx, ny, nz);

    // delta (Eq. 4): mean of S(Laplacian) over the interior I, the mask eroded by the radius.
    let interior = erode(&w.mask, nx, ny, nz, vsx, vsy, vsz, params.radius);
    let (sum, count) = smv_lap.iter().zip(&interior)
        .filter(|(_, &i)| i == 1)
        .fold((0.0, 0usize), |(s, c), (&v, _)| (s + v, c + 1));
    let delta = if count > 0 { sum / count as f64 } else { 0.0 };

    // Eq. 3 as least squares: A x = S(E x), b = delta - S(Laplacian); Aᵀ y = E S(y).
    let ext: Vec<f64> = w.mask.iter().map(|&m| if m == 0 { 1.0 } else { 0.0 }).collect();
    let b: Vec<Complex64> = smv_lap.iter().map(|&v| Complex64::new(delta - v, 0.0)).collect();
    let mut it = 0usize;
    let max_iter = params.max_iter;
    let (x, _) = lsqr_matlab(
        |v| {
            it += 1;
            progress(it.min(max_iter), max_iter);
            let ev: Vec<f64> = (0..n).map(|i| ext[i] * v[i].re).collect();
            to_complex(apply_smv(&ev, &s_fft, nx, ny, nz))
        },
        |u| {
            let re: Vec<f64> = u.iter().map(|c| c.re).collect();
            let su = apply_smv(&re, &s_fft, nx, ny, nz);
            (0..n).map(|i| Complex64::new(ext[i] * su[i], 0.0)).collect()
        },
        &b,
        params.tol,
        max_iter,
    );

    let lap_fov: Vec<f64> = (0..n).map(|i| w.lap[i] + ext[i] * x[i].re).collect();
    Ok((w.finish(&lap_fov, mask, grid, params.pad, rad_per_ppm), mask.to_vec()))
}

// =============================================================================
// iHARPERELLA (Li et al., ISMRM 2015 #3313; STI Suite 3.0 iHARPERELLA)
// =============================================================================

/// iHARPERELLA: the exterior Laplacian minimises the weighted resulting field in the brain.
///
/// STI Suite 3.0's `iHARPERELLA`: with `L` the brain's Laplacian (module docs), `E` the
/// exterior and `W` = 1 in the brain, 2 in the boundary rim and 0 outside, it solves
/// `min_x ‖W L⁻¹(E x) − W L⁻¹(L)‖₂` with MATLAB `lsqr` and returns `L⁻¹(L − x)` in the mask:
/// the field from the brain's Laplacian minus the field the exterior sources `x` explain.
///
/// # Arguments
/// * `input` - Unwrapped total field (ppm) or wrapped phase (rad); see [`HarperellaInput`]
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
    mut progress: impl FnMut(usize, usize),
) -> Result<(Vec<f64>, Vec<u8>), InputError> {
    let (data, rad_per_ppm) = input.data(mask, grid)?;
    let w = Work::new(data, mask, grid, params.pad, rad_per_ppm.is_some());
    let dims = w.dims;
    let n = w.n();
    let ext: Vec<f64> = w.mask.iter().map(|&m| if m == 0 { 1.0 } else { 0.0 }).collect();
    let wt: Vec<f64> = (0..n).map(|i| w.mask[i] as f64 + w.rim[i] as f64).collect();

    let inv_lap = apply_inv_kernel(&w.lap, &w.k2, dims);
    let b: Vec<Complex64> = (0..n).map(|i| Complex64::new(wt[i] * inv_lap[i], 0.0)).collect();
    let mut it = 0usize;
    let max_iter = params.max_iter;
    let (x, _) = lsqr_matlab(
        |v| {
            it += 1;
            progress(it.min(max_iter), max_iter);
            let ev: Vec<f64> = (0..n).map(|i| ext[i] * v[i].re).collect();
            let f = apply_inv_kernel(&ev, &w.k2, dims);
            (0..n).map(|i| Complex64::new(wt[i] * f[i], 0.0)).collect()
        },
        |u| {
            let wu: Vec<f64> = (0..n).map(|i| wt[i] * u[i].re).collect();
            let f = apply_inv_kernel(&wu, &w.k2, dims);
            (0..n).map(|i| Complex64::new(ext[i] * f[i], 0.0)).collect()
        },
        &b,
        params.tol,
        max_iter,
    );

    let lap_fov: Vec<f64> = (0..n).map(|i| w.lap[i] - x[i].re).collect();
    Ok((w.finish(&lap_fov, mask, grid, params.pad, rad_per_ppm), mask.to_vec()))
}

// =============================================================================
// Shared infrastructure
// =============================================================================

fn to_complex(v: Vec<f64>) -> Vec<Complex64> {
    v.into_iter().map(|x| Complex64::new(x, 0.0)).collect()
}

/// Pad with `pad` zero voxels on each side (STI's `padarray(x, padsize)`).
fn pad_centred<T: Copy + Default>(
    x: &[T], dims: (usize, usize, usize), new: (usize, usize, usize), pad: [usize; 3],
) -> Vec<T> {
    if pad == [0, 0, 0] { return x.to_vec(); }
    let corner = pad3d(x, dims, new);
    shift(&corner, new, [pad[0] as isize, pad[1] as isize, pad[2] as isize])
}

/// Inverse of [`pad_centred`].
fn unpad_centred(x: &[f64], padded: (usize, usize, usize), dims: (usize, usize, usize), pad: [usize; 3]) -> Vec<f64> {
    if pad == [0, 0, 0] { return x.to_vec(); }
    let back = shift(x, padded, [-(pad[0] as isize), -(pad[1] as isize), -(pad[2] as isize)]);
    unpad3d(&back, padded, dims)
}

/// Circular shift by `s` voxels.
fn shift<T: Copy + Default>(x: &[T], dims: (usize, usize, usize), s: [isize; 3]) -> Vec<T> {
    let (nx, ny, nz) = dims;
    let mut out = vec![T::default(); x.len()];
    let w = |i: usize, s: isize, n: usize| ((i as isize + s).rem_euclid(n as isize)) as usize;
    for k in 0..nz {
        let kk = w(k, s[2], nz);
        for j in 0..ny {
            let jj = w(j, s[1], ny);
            for i in 0..nx {
                out[w(i, s[0], nx) + jj * nx + kk * nx * ny] = x[i + j * nx + k * nx * ny];
            }
        }
    }
    out
}

/// STI's spectral Laplacian kernel `k² = kx² + ky² + kz²`, `k` in cycles/mm, in FFT order.
fn k2_kernel(dims: (usize, usize, usize), vs: (f64, f64, f64)) -> Vec<f64> {
    let (nx, ny, nz) = dims;
    let f = |i: usize, n: usize, v: f64| {
        let k = if i <= n / 2 { i as f64 } else { i as f64 - n as f64 };
        let k = k / (n as f64 * v);
        k * k
    };
    let mut out = vec![0.0; nx * ny * nz];
    for k in 0..nz {
        let kz = f(k, nz, vs.2);
        for j in 0..ny {
            let ky = f(j, ny, vs.1);
            for i in 0..nx {
                out[i + j * nx + k * nx * ny] = f(i, nx, vs.0) + ky + kz;
            }
        }
    }
    out
}

/// `F⁻¹[k F(x)]`, real part.
fn apply_kernel(x: &[f64], k: &[f64], dims: (usize, usize, usize)) -> Vec<f64> {
    let (nx, ny, nz) = dims;
    let mut c: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    fft3d(&mut c, nx, ny, nz);
    for (ci, &ki) in c.iter_mut().zip(k) { *ci *= ki; }
    ifft3d(&mut c, nx, ny, nz);
    c.iter().map(|v| v.re).collect()
}

/// `F⁻¹[F(x) / k]` with the `k = 0` term set to zero, real part.
fn apply_inv_kernel(x: &[f64], k: &[f64], dims: (usize, usize, usize)) -> Vec<f64> {
    let (nx, ny, nz) = dims;
    let mut c: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    fft3d(&mut c, nx, ny, nz);
    for (ci, &ki) in c.iter_mut().zip(k) {
        *ci = if ki > 0.0 { *ci / ki } else { Complex64::new(0.0, 0.0) };
    }
    ifft3d(&mut c, nx, ny, nz);
    c.iter().map(|v| v.re).collect()
}

/// STI's boundary rim: mask voxels where `SMVFiltering(mask, 3)`, the mean of the mask over
/// the 146 neighbours of `ball3D(3)` (offsets with `1 ≤ |d|² ≤ 10` voxels², periodic), is
/// below 0.99, i.e. more than one of those neighbours is outside the mask.
fn sti_rim(mask: &[u8], dims: (usize, usize, usize)) -> Vec<u8> {
    let (nx, ny, nz) = dims;
    let mut offs = Vec::new();
    for dz in -3i64..=3 {
        for dy in -3i64..=3 {
            for dx in -3i64..=3 {
                let r2 = dx * dx + dy * dy + dz * dz;
                if (1..=10).contains(&r2) { offs.push((dx, dy, dz)); }
            }
        }
    }
    debug_assert_eq!(offs.len(), 146);
    let total = offs.len() as f64;
    let w = |i: usize, d: i64, n: usize| (i as i64 + d).rem_euclid(n as i64) as usize;
    let mut rim = vec![0u8; mask.len()];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = i + j * nx + k * nx * ny;
                if mask[idx] == 0 { continue; }
                let inside = offs.iter()
                    .filter(|&&(dx, dy, dz)| mask[w(i, dx, nx) + w(j, dy, ny) * nx + w(k, dz, nz) * nx * ny] != 0)
                    .count();
                rim[idx] = u8::from((inside as f64) / total < 0.99);
            }
        }
    }
    rim
}

/// Set the 20 % of rim Laplacian values largest in magnitude to zero: keep `|L| ≤` the sorted
/// rim magnitude at 1-based index `round(0.8 N)`.
fn drop_large_rim_values(lap: &mut [f64], rim: &[u8]) {
    let mut mags: Vec<f64> = lap.iter().zip(rim).filter(|(_, &r)| r != 0).map(|(l, _)| l.abs()).collect();
    if mags.is_empty() { return; }
    let k = ((0.8 * mags.len() as f64).round() as usize).clamp(1, mags.len()) - 1;
    let thr = *mags.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1;
    for (l, &r) in lap.iter_mut().zip(rim) {
        if r != 0 && l.abs() > thr { *l = 0.0; }
    }
}

/// Erode `mask` by a sphere of `radius` mm: keep the voxels whose whole sphere lies in the mask.
///
/// The radius is widened by one part in 10^6 so that a voxel exactly `radius` away counts as
/// inside the sphere whatever the rounding of the voxel size: a header voxel size of
/// 0.99999994 mm must not drop the voxels at exactly `radius` from the sphere.
#[allow(clippy::too_many_arguments)]
fn erode(
    mask: &[u8],
    nx: usize, ny: usize, nz: usize,
    vsx: f64, vsy: f64, vsz: f64,
    radius: f64,
) -> Vec<u8> {
    let erode_fft = compute_smv_fft(nx, ny, nz, vsx, vsy, vsz, radius * (1.0 + 1e-6));
    let mut mask_conv: Vec<Complex64> = mask.iter()
        .map(|&m| Complex64::new(m as f64, 0.0)).collect();
    fft3d(&mut mask_conv, nx, ny, nz);
    for (c, &e) in mask_conv.iter_mut().zip(&erode_fft) { *c *= e; }
    ifft3d(&mut mask_conv, nx, ny, nz);
    let thresh = 1.0 - 1e-7_f64.sqrt();
    mask_conv.iter().map(|c| u8::from(c.re > thresh)).collect()
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
    apply_kernel(x, s_fft, (nx, ny, nz))
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

    /// A smooth field in ppm: a Gaussian blob (1.5 ppm peak) inside the mask on a smooth
    /// periodic background (0.3 ppm).
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
        let p = HarperellaParams { radius: 3.0, max_iter: 30, ..Default::default() };
        harperella(input, mask, &grid(), &p, |_, _| {}).unwrap().0
    }

    fn run_ih(input: HarperellaInput, mask: &[u8]) -> Vec<f64> {
        let p = IharperellaParams { max_iter: 30, ..Default::default() };
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

    /// The spectral Laplacian and its inverse undo each other up to the mean.
    #[test]
    fn spectral_laplacian_inverts() {
        let dims = (N, N, 12);
        let k2 = k2_kernel(dims, (1.0, 0.9, 1.5));
        let x: Vec<f64> = (0..N * N * 12).map(|i| ((i * 37 % 101) as f64).sin()).collect();
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        let back = apply_inv_kernel(&apply_kernel(&x, &k2, dims), &k2, dims);
        let centred: Vec<f64> = x.iter().map(|v| v - mean).collect();
        assert!(max_abs_diff(&back, &centred) < 1e-10);
        // k² is -∇²/4π²: on a plane wave of 2 cycles over 16 mm along x, k² = (2/16)².
        let wave: Vec<f64> = (0..N * N * 12).map(|i| (2.0 * std::f64::consts::PI * 2.0 * (i % N) as f64 / N as f64).cos()).collect();
        let lw = apply_kernel(&wave, &k2, dims);
        let expect: Vec<f64> = wave.iter().map(|v| v * (2.0f64 / 16.0).powi(2)).collect();
        assert!(max_abs_diff(&lw, &expect) < 1e-12);
    }

    /// STI's rim: a voxel with one ball neighbour outside the mask is not rim, with two it is.
    #[test]
    fn rim_needs_two_outside_neighbours() {
        let dims = (N, N, N);
        let mut mask = vec![1u8; N * N * N];
        let at = |i: usize, j: usize, k: usize| i + j * N + k * N * N;
        mask[at(3, 8, 8)] = 0;
        let rim = sti_rim(&mask, dims);
        assert_eq!(rim.iter().filter(|&&r| r != 0).count(), 0, "one hole: 145/146 > 0.99");
        mask[at(4, 8, 8)] = 0;
        let rim = sti_rim(&mask, dims);
        // voxels with both holes in their ball: offsets to both within |d|² <= 10
        assert!(rim[at(6, 8, 8)] == 1 && rim[at(1, 8, 8)] == 1 && rim[at(8, 8, 8)] == 0);
    }

    /// The rim keeps round(0.8 N) values and zeroes the rest, the largest in magnitude.
    #[test]
    fn rim_drops_the_largest_fifth() {
        for n in [1usize, 4, 5, 9, 13] {
            let mut lap: Vec<f64> = (0..n + 3).map(|i| if i % 2 == 0 { i as f64 } else { -(i as f64) }).collect();
            let mut rim = vec![1u8; n];
            rim.extend([0u8; 3]);
            let outside_before: Vec<f64> = lap[n..].to_vec();
            drop_large_rim_values(&mut lap, &rim);
            let kept = lap[..n].iter().filter(|&&v| v != 0.0).count() + usize::from(n > 0); // value 0 at i = 0
            assert_eq!(kept, ((0.8 * n as f64).round() as usize).max(1), "n = {n}");
            assert_eq!(&lap[n..], &outside_before[..], "outside the rim is untouched");
        }
    }

    #[test]
    fn padding_round_trips() {
        let dims = (5, 4, 3);
        let x: Vec<f64> = (0..60).map(|i| i as f64).collect();
        let pad = [2, 1, 3];
        let new = (9, 6, 9);
        let p = pad_centred(&x, dims, new, pad);
        assert_eq!(p[2 + new.0 + 3 * new.0 * new.1], x[0]);
        assert_eq!(unpad_centred(&p, new, dims, pad), x);
    }

    /// Wrapped phase goes through the paper's Eq. 1, which agrees with the field's Laplacian
    /// where the field is smooth: the two local fields agree closely.
    #[test]
    fn wrapped_phase_agrees_with_field_ppm() {
        let field = field_ppm(1.0);
        let mask = sphere_mask();
        let te = 0.007;
        let b0 = 3.0;
        let c = crate::inversion::rad_per_ppm(te, b0);
        let phase: Vec<f64> = field.iter().map(|f| wrap(f * c)).collect();
        assert!(max_abs(&field) * c > 2.0 * std::f64::consts::PI, "test field does not wrap");
        let input = HarperellaInput::WrappedPhase { phase: &phase, te_s: te, b0_t: b0 };
        for (from_phase, from_field, name) in [
            (run_ih(input, &mask), run_ih(HarperellaInput::FieldPpm(&field), &mask), "iHARPERELLA"),
            (run_h(input, &mask), run_h(HarperellaInput::FieldPpm(&field), &mask), "HARPERELLA"),
        ] {
            let d = max_abs_diff(&from_phase, &from_field) / max_abs(&from_field);
            assert!(d < 0.1, "{name}: phase vs field differ by {d:e}");
        }
    }

    /// LSQR's stopping rule is relative, so scaling the field scales the result. (A
    /// power-of-two factor scales every floating-point operation exactly.)
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

    /// Padding enlarges the working volume and changes the result, but the output stays on
    /// the input grid, zero outside the mask.
    #[test]
    fn padding_is_applied() {
        let mask = sphere_mask();
        let field = field_ppm(1.0);
        let p0 = IharperellaParams { max_iter: 10, ..Default::default() };
        let p4 = IharperellaParams { pad: [4, 4, 4], ..p0.clone() };
        let a = iharperella(HarperellaInput::FieldPpm(&field), &mask, &grid(), &p0, |_, _| {}).unwrap().0;
        let b = iharperella(HarperellaInput::FieldPpm(&field), &mask, &grid(), &p4, |_, _| {}).unwrap().0;
        assert_eq!(b.len(), N * N * N);
        assert!(max_abs_diff(&a, &b) > 1e-6);
        assert!(b.iter().zip(&mask).all(|(&v, &m)| m != 0 || v == 0.0));
    }

    /// The erosions do not depend on how the voxel size was rounded: a header voxel size of
    /// 0.99999994 mm erodes like 1 mm.
    #[test]
    fn erosion_ignores_voxel_size_rounding() {
        let mask = sphere_mask();
        for radius in [2.0, 3.0] {
            let exact = erode(&mask, N, N, N, 1.0, 1.0, 1.0, radius);
            let rounded = erode(&mask, N, N, N, 1.0, 1.0, 0.999_999_94, radius * 0.999_999_94);
            assert!(exact.contains(&1), "radius {radius}: nothing left");
            assert_eq!(exact, rounded, "radius {radius}");
        }
    }

    /// HARPERELLA's trustable interior I, where delta is averaged, is the mask eroded by the
    /// SMV radius. A radius wider than the brain leaves I empty; delta is then 0 and the result
    /// is still finite.
    #[test]
    fn empty_interior_gives_zero_delta() {
        let mask = sphere_mask();
        assert!(erode(&mask, N, N, N, 1.0, 1.0, 1.0, 6.5).iter().all(|&v| v == 0));
        assert!(erode(&mask, N, N, N, 1.0, 1.0, 1.0, 3.0).contains(&1));
        let field = field_ppm(1.0);
        let p = HarperellaParams { radius: 6.5, max_iter: 30, ..Default::default() };
        let out = harperella(HarperellaInput::FieldPpm(&field), &mask, &grid(), &p, |_, _| {}).unwrap().0;
        assert!(out.iter().all(|v| v.is_finite()) && max_abs(&out) > 1e-3);
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
    }
}

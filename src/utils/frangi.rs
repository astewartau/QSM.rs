//! Frangi vesselness filter for 3D tubular structure detection
//!
//! Uses the eigenvalues of the scale-normalised Hessian to score tubular (vessel-like)
//! structures, as Frangi et al. (1998) describe.
//!
//! This is a port of Dirk-Jan Kroon's `FrangiFilter3D` (MATLAB File Exchange 24409:
//! `FrangiFilter3D.m`, `Hessian3D.m`, `imgaussian.m` and the `eig3volume.c` mex), the
//! filter QSMART uses for its vessel mask. It follows that code's **single-precision**
//! path, because that is the path QSMART takes: its magnitude comes from ANTs N4's
//! float32 output via `load_nii`, and `FrangiFilter3D` keeps single input single. The
//! precision decides the result: at small scales the vesselness is ~1e-6 and
//! `1 - exp(-S^2/2c^2)` rounds to exactly 0 in single for most voxels, so on QSMART's
//! 7T exemplar (scales 0.01:0.01:0.05) single precision gives 137,710 nonzero voxels
//! (bit-identical to the exemplar) where double gives 523,844.
//!
//! Each step does what MATLAB R2023b does for single input (checked against MATLAB):
//! - `imgaussian`: kernel `exp(-x^2/(2 sigma^2))` over `x = -ceil(3 sigma)..ceil(3 sigma)`,
//!   normalised in double and then cast to single; three separable `imfilter` passes
//!   (x, y, then z; z only when there are at least 4 slices) with replicate padding.
//!   `imfilter` on single input accumulates in single, each product rounded before it
//!   is added, and stores each pass as single. The order of the sum matters at that
//!   precision and is MATLAB's: along x the taps go first to last; along y and z the
//!   taps whose source is inside the volume go last to first, then the replicated
//!   taps beyond the edge, last to first.
//! - `gradient3`: central differences in single, one-sided at the edges; the Hessian is
//!   `Dxx = d/dx(Dx)`, `Dxy = d/dy(Dx)`, `Dxz = d/dz(Dx)`, `Dyy = d/dy(Dy)`,
//!   `Dyz = d/dz(Dy)`, `Dzz = d/dz(Dz)`, each first derivative stored as single.
//! - `sigma^2 * D`: a double scalar times a single array is evaluated in double and
//!   rounded to single (likewise the divisions by `2 alpha^2`, `2 beta^2` and `2 c^2`).
//! - `eig3volume`: each single Hessian is widened to double, decomposed with JAMA's
//!   Householder tridiagonalisation and QL iteration, sorted by magnitude, and the
//!   eigenvalues are rounded back to single.
//! - `Ra`, `Rb`, `S` and the exponentials are evaluated in single. MATLAB's single `exp`
//!   is not correctly rounded (it differs from the correctly rounded result by 1 ulp for
//!   ~6 % of arguments with |x| > 3.6e-6); here `exp` is correctly rounded, so some
//!   vesselness values differ from MATLAB's in the last bits. Which voxels are zero does
//!   not change: for |x| below 3.6e-6, where `exp` rounds to 1, the two agree. With
//!   MATLAB's `exp` swapped for a correctly rounded one, `FrangiFilter3D` and this port
//!   agree bit for bit (QSMART's 7T exemplar and three 3T cases).
//! - Scales follow MATLAB's colon operator (`a:r:b`, first half `a + k r`, second half
//!   counted back from the end), and the output is the maximum over scales.
//!
//! ## Reference pipeline
//!
//! QSMART calls `FrangiFilter3D` with `FrangiAlpha` 0.5, `FrangiBeta` 0.5, `BlackWhite` false
//! and the demo's `FrangiScaleRange` [0.5 6], `FrangiScaleRatio` 0.5, `FrangiC` 500
//! (`vasculature_mask.m` l. 25–27, `Demo_QSMART.m` l. 41–43), which the paper calls Kroon's
//! defaults for vessels (7T human, 0.6 mm; Yaghmaie et al. 2021, §3.4). Its rodent runs used
//! scales 0.01–0.5 in steps of 0.1 and c = 50; the authors' 7T exemplar run used scales
//! 0.01:0.01:0.05 (the exemplar above). Scales are in voxels.
//!
//! References:
//! Frangi, A.F., Niessen, W.J., Vincken, K.L., Viergever, M.A. (1998).
//! "Multiscale vessel enhancement filtering." MICCAI'98, LNCS vol 1496, 130-137.
//! https://doi.org/10.1007/BFb0056195
//!
//! Reference implementation: https://www.mathworks.com/matlabcentral/fileexchange/24409-hessian-based-frangi-vesselness-filter

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::error::InputError;

/// Parameters for Frangi vesselness filter
#[derive(Clone, Debug)]
pub struct FrangiParams {
    /// Range of sigma values [min, max] for multi-scale analysis
    pub scale_range: [f64; 2],
    /// Step size between sigma values (default 0.5)
    pub scale_ratio: f64,
    /// Alpha parameter: sensitivity to plate-like structures (Ra term), default 0.5
    pub alpha: f64,
    /// Beta parameter: sensitivity to blob-like structures (Rb term), default 0.5
    pub beta: f64,
    /// C parameter: sensitivity to noise/background (S term), default 500
    /// Threshold between eigenvalues of noise and vessel structure
    pub c: f64,
    /// Detect black vessels (true) or white vessels (false)
    pub black_white: bool,
}

impl FrangiParams {
    /// Checks `scale_ratio`, the step between scales: it must be finite and > 0. MATLAB's
    /// `a:0:b` is empty and the reference then fails; a tiny or zero step used to leave the
    /// scale loop running indefinitely.
    pub fn validate(&self) -> Result<(), InputError> {
        if !(self.scale_ratio.is_finite() && self.scale_ratio > 0.0) {
            return Err(InputError::param("Frangi scale_ratio", self.scale_ratio, "finite and > 0"));
        }
        Ok(())
    }
}

impl Default for FrangiParams {
    fn default() -> Self {
        Self {
            // QSMART Demo defaults: FrangiScaleRange=[0.5,6], FrangiScaleRatio=0.5
            scale_range: [0.5, 6.0],
            scale_ratio: 0.5,
            alpha: 0.5,
            beta: 0.5,
            c: 500.0,
            black_white: false, // White vessels (bright structures)
        }
    }
}

/// Result of Frangi filter including vesselness and scale information
pub struct FrangiResult {
    /// Vesselness response (0 to 1); single-precision values widened to f64
    pub vesselness: Vec<f64>,
    /// Scale at which maximum vesselness was found
    pub scale: Vec<f64>,
}

/// Apply 3D Frangi vesselness filter
///
/// The input is rounded to single precision first, as `FrangiFilter3D` does with
/// non-double input, and the whole filter then runs in single precision (see the module
/// docs).
///
/// # Arguments
/// * `data` - Input 3D volume (nx * ny * nz)
/// * `grid` - Volume grid (dimensions and voxel sizes)
/// * `params` - Frangi filter parameters
/// * `progress` - Progress callback `(current_scale, total_scales)`
///
/// # Returns
/// FrangiResult with vesselness map and optimal scale map
///
/// # Errors
/// If [`FrangiParams::validate`] rejects `params`.
pub fn frangi_filter_3d(
    data: &[f64],
    grid: &crate::Grid,
    params: &FrangiParams,
    progress: impl FnMut(usize, usize),
) -> Result<FrangiResult, InputError> {
    params.validate()?;
    let (nx, ny, nz) = grid.dims;
    let input: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    Ok(frangi_filter_3d_inner(&input, nx, ny, nz, params, progress))
}

/// MATLAB's colon operator `a:d:b` for `d > 0`: the first half of the points is
/// `a + k d`, the second half `b - (n - k) d`, so both ends are hit exactly.
fn matlab_colon(a: f64, d: f64, b: f64) -> Vec<f64> {
    if d.is_nan() || d <= 0.0 || b < a || !a.is_finite() || !b.is_finite() {
        return Vec::new();
    }
    let mut n = ((b - a) / d).floor();
    let tol = 2.0 * f64::EPSILON * a.abs().max(b.abs());
    if a + (n + 1.0) * d - b <= tol {
        n += 1.0;
    }
    let end = if (a + n * d - b).abs() <= tol { b } else { a + n * d };
    let n = n as usize;
    (0..=n)
        .map(|k| if k <= n / 2 { a + k as f64 * d } else { end - (n - k) as f64 * d })
        .collect()
}

/// One `imfilter(I, H, 'same', 'replicate')` pass along one axis, single precision.
fn imfilter_axis(data: &[f32], dims: (usize, usize, usize), kernel: &[f32], axis: usize) -> Vec<f32> {
    let (nx, ny, nz) = dims;
    let nxy = nx * ny;
    let r = (kernel.len() - 1) / 2;
    let (len, stride) = match axis {
        0 => (nx, 1),
        1 => (ny, nx),
        _ => (nz, nxy),
    };
    let mut out = vec![0f32; data.len()];
    if out.is_empty() {
        return out;
    }
    crate::maybe_par_chunks_mut!(out, nxy).enumerate().for_each(|(k, slice)| {
        for j in 0..ny {
            for i in 0..nx {
                let pos = [i, j, k][axis];
                let base = i + j * nx + k * nxy - pos * stride;
                let tap = |t: usize| kernel[t] * data[base + (pos + t).saturating_sub(r).min(len - 1) * stride];
                let mut acc = 0f32;
                if axis == 0 {
                    for t in 0..kernel.len() {
                        acc += tap(t);
                    }
                } else {
                    // Taps whose source lies inside the volume, last to first, then the
                    // replicated ones beyond the edge, last to first.
                    let inside = |t: usize| pos + t >= r && pos + t - r < len;
                    for t in (0..kernel.len()).rev().filter(|&t| inside(t)) {
                        acc += tap(t);
                    }
                    for t in (0..kernel.len()).rev().filter(|&t| !inside(t)) {
                        acc += tap(t);
                    }
                }
                slice[i + j * nx] = acc;
            }
        }
    });
    out
}

/// `imgaussian(I, sigma)` for single `I`.
fn imgaussian(data: &[f32], dims: (usize, usize, usize), sigma: f64) -> Vec<f32> {
    let half = (sigma * 6.0 / 2.0).ceil() as i64;
    let h: Vec<f64> = (-half..=half)
        .map(|x| {
            let x = x as f64;
            (-(x * x / (2.0 * (sigma * sigma)))).exp()
        })
        .collect();
    let sum: f64 = h.iter().sum();
    let kernel: Vec<f32> = h.iter().map(|&v| (v / sum) as f32).collect();
    let fx = imfilter_axis(data, dims, &kernel, 0);
    let fxy = imfilter_axis(&fx, dims, &kernel, 1);
    if dims.2 < 4 {
        // imgaussian treats fewer than 4 slices as a stack of 2D images.
        return fxy;
    }
    imfilter_axis(&fxy, dims, &kernel, 2)
}

/// `gradient3` at one voxel: `pos` is the coordinate along the axis of length `len`.
#[inline(always)]
fn gradient3_at(f: &[f32], idx: usize, pos: usize, len: usize, stride: usize) -> f32 {
    if len < 2 {
        0.0
    } else if pos == 0 {
        f[idx + stride] - f[idx]
    } else if pos == len - 1 {
        f[idx] - f[idx - stride]
    } else {
        (f[idx + stride] - f[idx - stride]) / 2.0
    }
}

/// `gradient3(F, axis)` over the whole volume.
fn gradient3(f: &[f32], dims: (usize, usize, usize), axis: usize) -> Vec<f32> {
    let (nx, ny, _) = dims;
    let nxy = nx * ny;
    let (len, stride) = match axis {
        0 => (dims.0, 1),
        1 => (dims.1, nx),
        _ => (dims.2, nxy),
    };
    let mut out = vec![0f32; f.len()];
    if out.is_empty() {
        return out;
    }
    crate::maybe_par_chunks_mut!(out, nxy).enumerate().for_each(|(k, slice)| {
        for j in 0..ny {
            for i in 0..nx {
                let idx = i + j * nx + k * nxy;
                slice[i + j * nx] = gradient3_at(f, idx, [i, j, k][axis], len, stride);
            }
        }
    });
    out
}

/// JAMA's symmetric Householder reduction to tridiagonal form (`tred2` in eig3volume.c).
/// The eigenvectors are not needed, but `V` carries the reduction.
pub(crate) fn tred2(v: &mut [[f64; 3]; 3], d: &mut [f64; 3], e: &mut [f64; 3]) {
    const N: usize = 3;
    *d = v[N - 1];
    for i in (1..N).rev() {
        let mut scale = 0.0;
        let mut h = 0.0;
        for dk in d.iter().take(i) {
            scale += dk.abs();
        }
        if scale == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = v[i - 1][j];
                v[i][j] = 0.0;
                v[j][i] = 0.0;
            }
        } else {
            for dk in d.iter_mut().take(i) {
                *dk /= scale;
                h += *dk * *dk;
            }
            let mut f = d[i - 1];
            let mut g = h.sqrt();
            if f > 0.0 {
                g = -g;
            }
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            for ej in e.iter_mut().take(i) {
                *ej = 0.0;
            }
            for j in 0..i {
                f = d[j];
                v[j][i] = f;
                g = e[j] + v[j][j] * f;
                for k in (j + 1)..i {
                    g += v[k][j] * d[k];
                    e[k] += v[k][j] * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for j in 0..i {
                e[j] /= h;
                f += e[j] * d[j];
            }
            let hh = f / (h + h);
            for j in 0..i {
                e[j] -= hh * d[j];
            }
            for j in 0..i {
                f = d[j];
                g = e[j];
                for k in j..i {
                    v[k][j] -= f * e[k] + g * d[k];
                }
                d[j] = v[i - 1][j];
                v[i][j] = 0.0;
            }
        }
        d[i] = h;
    }
    for i in 0..N - 1 {
        v[N - 1][i] = v[i][i];
        v[i][i] = 1.0;
        let h = d[i + 1];
        if h != 0.0 {
            for k in 0..=i {
                d[k] = v[k][i + 1] / h;
            }
            for j in 0..=i {
                let mut g = 0.0;
                for k in 0..=i {
                    g += v[k][i + 1] * v[k][j];
                }
                for k in 0..=i {
                    v[k][j] -= g * d[k];
                }
            }
        }
        for row in v.iter_mut().take(i + 1) {
            row[i + 1] = 0.0;
        }
    }
    for j in 0..N {
        d[j] = v[N - 1][j];
        v[N - 1][j] = 0.0;
    }
    v[N - 1][N - 1] = 1.0;
    e[0] = 0.0;
}

/// JAMA's symmetric tridiagonal QL algorithm (`tql2` in eig3volume.c), eigenvalues only:
/// the eigenvector accumulation does not feed back into them. Ends with the eigenvalues
/// in ascending order.
pub(crate) fn tql2(d: &mut [f64; 3], e: &mut [f64; 3]) {
    const N: usize = 3;
    for i in 1..N {
        e[i - 1] = e[i];
    }
    e[N - 1] = 0.0;
    let mut f = 0.0;
    let mut tst1: f64 = 0.0;
    let eps = 2f64.powi(-52);
    for l in 0..N {
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < N {
            if e[m].abs() <= eps * tst1 {
                break;
            }
            m += 1;
        }
        // e[N-1] is 0 and the input is finite, so this stops at m < N.
        if m > l {
            loop {
                let mut g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = (p * p + 1.0).sqrt();
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for di in d.iter_mut().skip(l + 2) {
                    *di -= h;
                }
                f += h;
                p = d[m];
                let mut c = 1.0;
                let mut c2 = c;
                let mut c3 = c;
                let el1 = e[l + 1];
                let mut s = 0.0;
                let mut s2 = 0.0;
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = (p * p + e[i] * e[i]).sqrt();
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if e[l].abs() <= eps * tst1 {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
    for i in 0..N - 1 {
        let mut k = i;
        let mut p = d[i];
        for (j, &dj) in d.iter().enumerate().skip(i + 1) {
            if dj < p {
                k = j;
                p = dj;
            }
        }
        if k != i {
            d[k] = d[i];
            d[i] = p;
        }
    }
}

/// `eig3volume` for one voxel of single input: eigenvalues of the symmetric matrix
/// `[[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]` in double, sorted as eig3volume.c sorts
/// them (by magnitude, ascending), rounded to single.
fn eig3volume_single(xx: f32, xy: f32, xz: f32, yy: f32, yz: f32, zz: f32) -> (f32, f32, f32) {
    if ![xx, xy, xz, yy, yz, zz].iter().all(|v| v.is_finite()) {
        // The C code does not handle these (NaN can index past its arrays); the
        // vesselness would be non-finite, which FrangiFilter3D zeroes.
        return (f32::NAN, f32::NAN, f32::NAN);
    }
    let mut v = [
        [xx as f64, xy as f64, xz as f64],
        [xy as f64, yy as f64, yz as f64],
        [xz as f64, yz as f64, zz as f64],
    ];
    let mut d = [0f64; 3];
    let mut e = [0f64; 3];
    tred2(&mut v, &mut d, &mut e);
    tql2(&mut d, &mut e);
    let mut da = [d[0].abs(), d[1].abs(), d[2].abs()];
    if da[0] >= da[1] && da[0] > da[2] {
        d.swap(0, 2);
        da.swap(0, 2);
    } else if da[1] >= da[0] && da[1] > da[2] {
        d.swap(1, 2);
        da.swap(1, 2);
    }
    if da[0] > da[1] {
        d.swap(0, 1);
    }
    (d[0] as f32, d[1] as f32, d[2] as f32)
}

/// Correctly rounded single-precision `exp` (see the module docs on MATLAB's).
#[inline(always)]
fn exp_single(x: f32) -> f32 {
    (x as f64).exp() as f32
}

/// `FrangiFilter3D`'s vesselness for one voxel's sorted single eigenvalues, in single.
/// `a`, `b`, `c` are `2 alpha^2`, `2 beta^2` and `2 c^2` (doubles, as in MATLAB).
#[inline(always)]
fn vesselness_single(l1: f32, l2: f32, l3: f32, a: f64, b: f64, c: f64, black_white: bool) -> f32 {
    let (a1, a2, a3) = (l1.abs(), l2.abs(), l3.abs());
    let ra = a2 / a3;
    let rb = a1 / (a2 * a3).sqrt();
    let s = (a1 * a1 + a2 * a2 + a3 * a3).sqrt();
    // single ./ double scalar: evaluated in double, rounded to single.
    let exp_ra = 1.0 - exp_single(-(((ra * ra) as f64 / a) as f32));
    let exp_rb = exp_single(-(((rb * rb) as f64 / b) as f32));
    let exp_s = 1.0 - exp_single(((-(s * s)) as f64 / c) as f32);
    let v = exp_ra * exp_rb * exp_s;
    let wrong_sign = if black_white { l2 < 0.0 || l3 < 0.0 } else { l2 > 0.0 || l3 > 0.0 };
    if wrong_sign || !v.is_finite() {
        0.0
    } else {
        v
    }
}

/// Inner implementation of the Frangi filter (single-precision input, raw dimensions).
///
/// Per scale it keeps the smoothed volume's three first derivatives (single) and computes
/// the six second derivatives, the eigenvalues and the vesselness voxel by voxel, rather
/// than storing six Hessian volumes as `Hessian3D` does; the arithmetic is the same.
fn frangi_filter_3d_inner<F>(
    input: &[f32],
    nx: usize, ny: usize, nz: usize,
    params: &FrangiParams,
    mut progress_callback: F,
) -> FrangiResult
where
    F: FnMut(usize, usize),
{
    let n_total = nx * ny * nz;
    let dims = (nx, ny, nz);
    let nxy = nx * ny;

    let mut sigmas = matlab_colon(params.scale_range[0], params.scale_ratio, params.scale_range[1]);
    if sigmas.is_empty() {
        sigmas.push(params.scale_range[0]);
    }
    let total_scales = sigmas.len();

    let mut best = vec![0f32; n_total];
    let mut best_scale = vec![sigmas[0]; n_total];
    let mut voxel = vec![0f32; n_total];

    let a = 2.0 * (params.alpha * params.alpha);
    let b = 2.0 * (params.beta * params.beta);
    let c = 2.0 * (params.c * params.c);

    for (scale_idx, &sigma) in sigmas.iter().enumerate() {
        progress_callback(scale_idx, total_scales);
        if n_total == 0 {
            continue;
        }

        let (dx, dy, dz) = {
            let smoothed = if sigma > 0.0 { imgaussian(input, dims, sigma) } else { input.to_vec() };
            (gradient3(&smoothed, dims, 0), gradient3(&smoothed, dims, 1), gradient3(&smoothed, dims, 2))
        };
        // FrangiFilter3D scales by sigma^2 only for sigma > 0.
        let scale = if sigma > 0.0 { Some(sigma * sigma) } else { None };
        let mul = |d: f32| match scale {
            Some(s2) => (s2 * d as f64) as f32,
            None => d,
        };

        crate::maybe_par_chunks_mut!(voxel, nxy).enumerate().for_each(|(k, slice)| {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = i + j * nx + k * nxy;
                    let dxx = mul(gradient3_at(&dx, idx, i, nx, 1));
                    let dxy = mul(gradient3_at(&dx, idx, j, ny, nx));
                    let dxz = mul(gradient3_at(&dx, idx, k, nz, nxy));
                    let dyy = mul(gradient3_at(&dy, idx, j, ny, nx));
                    let dyz = mul(gradient3_at(&dy, idx, k, nz, nxy));
                    let dzz = mul(gradient3_at(&dz, idx, k, nz, nxy));
                    // An all-zero Hessian gives Ra = 0/0, which FrangiFilter3D zeroes.
                    slice[i + j * nx] = if dxx == 0.0 && dxy == 0.0 && dxz == 0.0
                        && dyy == 0.0 && dyz == 0.0 && dzz == 0.0
                    {
                        0.0
                    } else {
                        let (l1, l2, l3) = eig3volume_single(dxx, dxy, dxz, dyy, dyz, dzz);
                        vesselness_single(l1, l2, l3, a, b, c, params.black_white)
                    };
                }
            }
        });

        if scale_idx == 0 {
            best.copy_from_slice(&voxel);
        } else {
            for ((bv, bs), &v) in best.iter_mut().zip(best_scale.iter_mut()).zip(voxel.iter()) {
                if v > *bv {
                    *bv = v;
                    *bs = sigma;
                }
            }
        }
    }

    progress_callback(total_scales, total_scales);

    FrangiResult {
        vesselness: best.iter().map(|&v| v as f64).collect(),
        scale: best_scale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Grid;

    fn grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 1.0, 1.0, 1.0)
    }

    fn sorted(t: (f32, f32, f32)) -> [f32; 3] {
        let mut v = [t.0, t.1, t.2];
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    }

    /// MATLAB R2023b `FrangiFilter3D` on single input: a 16x18x20 oblique tube with
    /// integer intensities. Layout (little-endian, column-major): the u16 input, then f32
    /// volumes for scales 0.01:0.01:0.05 with C = 500 (single-precision underflow decides
    /// which voxels are nonzero; double gives ~8x as many), first as MATLAB returns it and
    /// then with MATLAB's single `exp` swapped for a correctly rounded one, and last for
    /// scales 0.5:0.5:2 with C = 200 (the Gaussian path), also with that `exp`.
    #[test]
    fn matches_matlab_single_precision() {
        let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/frangi_matlab_single_small.bin"))
            .expect("fixture");
        let (nx, ny, nz) = (16, 18, 20);
        let n = nx * ny * nz;
        assert_eq!(raw.len(), 2 * n + 12 * n);
        let input: Vec<f64> = raw[..2 * n].as_chunks::<2>().0.iter().map(|&b| u16::from_le_bytes(b) as f64).collect();
        let volume = |i: usize| -> Vec<f32> {
            raw[2 * n + 4 * n * i..2 * n + 4 * n * (i + 1)]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect()
        };
        let run = |params: FrangiParams| -> Vec<f32> {
            frangi_filter_3d(&input, &grid(nx, ny, nz), &params, |_, _| {}).unwrap().vesselness.iter().map(|&v| v as f32).collect()
        };
        let small = run(FrangiParams { scale_range: [0.01, 0.05], scale_ratio: 0.01, c: 500.0, ..Default::default() });
        let gauss = run(FrangiParams { scale_range: [0.5, 2.0], scale_ratio: 0.5, c: 200.0, ..Default::default() });

        // Bit-identical to FrangiFilter3D apart from MATLAB's own single `exp`.
        assert_eq!(small, volume(1));
        assert_eq!(gauss, volume(2));
        // Against MATLAB as is: the same nonzero voxels, values a last-bit `exp` apart.
        let matlab = volume(0);
        let nonzero = matlab.iter().filter(|&&v| v != 0.0).count();
        assert!(nonzero > 0 && nonzero < n / 10, "{nonzero}");
        for (g, e) in small.iter().zip(matlab.iter()) {
            assert_eq!(*g != 0.0, *e != 0.0, "got {g}, MATLAB {e}");
            assert!((g - e).abs() <= 1e-2 * e, "got {g}, MATLAB {e}");
        }
    }

    #[test]
    fn colon_matches_matlab() {
        // MATLAB: 0.01:0.01:0.05 hits 0.03 as 0.01 + 2*0.01 and 0.04 as 0.05 - 0.01.
        assert_eq!(matlab_colon(0.01, 0.01, 0.05), vec![0.01, 0.02, 0.01 + 2.0 * 0.01, 0.05 - 0.01, 0.05]);
        assert_eq!(matlab_colon(0.5, 0.5, 6.0).len(), 12);
        assert_eq!(matlab_colon(0.1, 0.1, 0.3).len(), 3);
        assert!(matlab_colon(5.0, 1.0, 1.0).is_empty());
    }

    #[test]
    fn eig3volume_eigenvalues() {
        assert_eq!(sorted(eig3volume_single(1.0, 0.0, 0.0, 2.0, 0.0, 3.0)), [1.0, 2.0, 3.0]);
        // [[2, 1, 0], [1, 2, 0], [0, 0, 1]] has eigenvalues 1, 1, 3.
        let v = sorted(eig3volume_single(2.0, 1.0, 0.0, 2.0, 0.0, 1.0));
        for (got, want) in v.iter().zip([1.0, 1.0, 3.0]) {
            assert!((got - want).abs() < 1e-6);
        }
        // Sorted by magnitude, ascending, signs kept.
        let (l1, l2, l3) = eig3volume_single(-5.0, 0.0, 0.0, 0.1, 0.0, -2.0);
        assert_eq!((l1, l2, l3), (0.1, -2.0, -5.0));
        assert!(eig3volume_single(f32::NAN, 0.0, 0.0, 1.0, 0.0, 1.0).0.is_nan());
    }

    #[test]
    fn gradient3_edges_and_interior() {
        let (nx, ny, nz) = (4, 3, 5);
        let data: Vec<f32> = (0..nx * ny * nz).map(|i| ((i / (nx * ny)) as f32).powi(2)).collect();
        let g = gradient3(&data, (nx, ny, nz), 2);
        let at = |k: usize| g[k * nx * ny];
        assert_eq!([at(0), at(1), at(2), at(3), at(4)], [1.0, 2.0, 4.0, 6.0, 7.0]);
        assert!(gradient3(&data, (nx, ny, nz), 0).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn imgaussian_preserves_constant() {
        let n = 8;
        let data = vec![5.0f32; n * n * n];
        for &v in &imgaussian(&data, (n, n, n), 1.0) {
            assert!((v - 5.0).abs() < 1e-5);
        }
        // Tiny sigma: the kernel is [0, 1, 0] in single, so the volume is unchanged.
        let ramp: Vec<f32> = (0..n * n * n).map(|i| i as f32 * 0.37).collect();
        assert_eq!(imgaussian(&ramp, (n, n, n), 0.03), ramp);
    }

    #[test]
    fn vesselness_sign_and_degenerate_cases() {
        let (a, b, c) = (0.5, 0.5, 2.0 * 500.0 * 500.0);
        assert!(vesselness_single(0.01, -100.0, -100.0, a, b, c, false) > 0.0);
        assert_eq!(vesselness_single(0.01, 100.0, 100.0, a, b, c, false), 0.0);
        assert!(vesselness_single(0.01, 100.0, 100.0, a, b, c, true) > 0.0);
        assert_eq!(vesselness_single(0.01, -100.0, -100.0, a, b, c, true), 0.0);
        assert_eq!(vesselness_single(0.0, 0.0, 0.0, a, b, c, false), 0.0);
        // S^2 / 2c^2 below single precision's resolution of 1: exp rounds to 1.
        assert_eq!(vesselness_single(1e-6, -1e-2, -1e-2, a, b, c, false), 0.0);
    }

    #[test]
    fn frangi_detects_tube() {
        let n = 12;
        let mut data = vec![0.0; n * n * n];
        let (cy, cz) = (n as f64 / 2.0, n as f64 / 2.0);
        for x in 1..(n - 1) {
            for y in 0..n {
                for z in 0..n {
                    let r2 = (y as f64 - cy).powi(2) + (z as f64 - cz).powi(2);
                    data[x + y * n + z * n * n] = (-r2 / (2.0 * 1.5 * 1.5)).exp();
                }
            }
        }
        let params = FrangiParams { scale_range: [1.0, 2.0], scale_ratio: 1.0, c: 10.0, ..Default::default() };
        let result = frangi_filter_3d(&data, &grid(n, n, n), &params, |_, _| {}).unwrap();
        assert!(result.vesselness.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
        assert!(result.vesselness.iter().cloned().fold(0.0, f64::max) > 0.0);
        assert!(result.scale.iter().all(|s| *s == 1.0 || *s == 2.0));
    }

    #[test]
    fn frangi_zero_input_progress_and_empty_scales() {
        let n = 8;
        let data = vec![0.0; n * n * n];
        let mut calls = Vec::new();
        let params = FrangiParams { scale_range: [1.0, 2.0], scale_ratio: 1.0, ..Default::default() };
        let result = frangi_filter_3d(&data, &grid(n, n, n), &params, |c, t| calls.push((c, t))).unwrap();
        assert!(result.vesselness.iter().all(|&v| v == 0.0));
        assert_eq!(calls.last(), Some(&(2, 2)));
        // min > max: falls back to the first scale.
        let params = FrangiParams { scale_range: [5.0, 1.0], scale_ratio: 1.0, ..Default::default() };
        let result = frangi_filter_3d(&data, &grid(n, n, n), &params, |_, _| {}).unwrap();
        assert_eq!(result.vesselness.len(), n * n * n);
    }

    /// #199: a `scale_ratio` that is not > 0 is an error, not an endless (or silently
    /// single-scale) scale loop.
    #[test]
    fn scale_ratio_that_is_not_positive_is_an_error() {
        let n = 4;
        for bad in [0.0, -0.5, f64::NAN] {
            let params = FrangiParams { scale_ratio: bad, ..Default::default() };
            let err = frangi_filter_3d(&vec![0.0; n * n * n], &grid(n, n, n), &params, |_, _| {})
                .err()
                .expect("a scale_ratio <= 0 should be refused");
            assert!(err.0.contains("Frangi scale_ratio") && err.0.contains("> 0"), "{err}");
        }
    }
}

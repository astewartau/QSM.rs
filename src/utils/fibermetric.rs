//! MATLAB's `fibermetric` (Image Processing Toolbox R2023b) for double-precision volumes
//!
//! `fibermetric(V, thickness, 'ObjectPolarity', polarity)` is a multiscale Frangi vesselness:
//! for each thickness `t` (in voxels) it smooths `V` with `imgaussfilt3(V, t/6, 'FilterSize',
//! 2*ceil(t/2)+1)`, takes the Hessian of the smoothed volume scaled by `sigma^2`, scores each
//! voxel from the Hessian's eigenvalues, and keeps the maximum over thicknesses. The mSMV
//! authors build their vessel mask with it ([`msmv`](mod@crate::inversion::msmv)), which is why it
//! is ported here. This is not Kroon's `FrangiFilter3D` ([`frangi`](super::frangi)): the
//! Hessian, the scale parameter, the default `c` and the precision all differ.
//!
//! What each step does, and how it was established:
//! - `fibermetric.m` itself is M-code: `sigma = thickness/6`, the default
//!   `StructureSensitivity` is `diff(getrangefromclass(V))/100`, i.e. **0.01 for double input**
//!   (since R2018b; before that it was half the maximum Hessian norm), double input stays
//!   double, and the output is `B = max(B, out)` over thicknesses, starting from zeros (so a
//!   NaN response is dropped by `max`).
//! - `imgaussfilt3` is M-code too: the kernel `exp(-x^2/(2 sigma^2))` over
//!   `x = -(h-1)/2..(h-1)/2`, entries below `eps*max` set to 0, normalised to sum 1, applied
//!   separably (y, then x, then z) with replicate padding. For large volumes and kernels
//!   (>= 2.5e6 voxels and >= 9^3 taps) MATLAB filters in the frequency domain instead, with
//!   the same replicate padding; that is the same linear filter and differs only by rounding.
//! - The Hessian and the vesselness are in a compiled builtin
//!   (`images.internal.builtins.fibermetric(Ig, c, isBright, sigma)`). Its behaviour was read
//!   off by calling the builtin directly in MATLAB R2023b on random and tubular volumes
//!   (four sigmas, both polarities, two values of `c`) and matching it to 1e-15 with:
//!   - first derivatives by central differences with replicate padding,
//!     `G[i] = (f[i+1] - f[i-1])/2`, so `G[0] = (f[1]-f[0])/2` at the edges;
//!   - mixed second derivatives `H_ab` as the same central difference of `G_a` along `b`;
//!   - pure second derivatives `H_aa` as the central difference of `G_a` along `a`, except at
//!     the first and last voxel along `a`, where the builtin uses `(f[2] + f[1] - 2 f[0])/4`
//!     (and its mirror image at the far end), i.e. `(G[0] + G[1])/2`;
//!   - every entry multiplied by `sigma^2`, eigenvalues sorted by magnitude
//!     `|l1| <= |l2| <= |l3|`, and Frangi's measure with `alpha = beta = 0.5`:
//!     `(1 - exp(-Ra^2/(2 alpha^2))) exp(-Rb^2/(2 beta^2)) (1 - exp(-S^2/(2 c^2)))`,
//!     `Ra = |l2|/|l3|`, `Rb = |l1|/sqrt(|l2 l3|)`, `S = sqrt(l1^2 + l2^2 + l3^2)`;
//!   - set to 0 where `l2 > 0` or `l3 > 0` for bright structures (`l2 < 0` or `l3 < 0` for
//!     dark ones).
//!
//! Eigenvalues are computed with JAMA's tridiagonalisation and QL iteration (shared with
//! [`frangi`](super::frangi)); MATLAB's eigen solver is not known, so values agree to rounding,
//! not bit for bit. Every axis needs at least three voxels (`fibermetric` refuses volumes with
//! a singleton dimension).
//!
//! Reference:
//! Frangi, A.F., Niessen, W.J., Vincken, K.L., Viergever, M.A. (1998).
//! "Multiscale vessel enhancement filtering." MICCAI'98, LNCS vol 1496, 130-137.
//! https://doi.org/10.1007/BFb0056195

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// `fibermetric`'s `ObjectPolarity`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiberPolarity {
    /// Structures brighter than the background (`'bright'`, MATLAB's default)
    Bright,
    /// Structures darker than the background (`'dark'`)
    Dark,
}

/// The default `StructureSensitivity` for double input: `diff(getrangefromclass(V))/100`.
pub const FIBERMETRIC_DEFAULT_C: f64 = 0.01;

/// MATLAB's `fibermetric(V, thickness, 'ObjectPolarity', polarity, 'StructureSensitivity', c)`
/// on a double volume (see the module docs).
///
/// # Arguments
/// * `data` - The volume (`nx*ny*nz`, x fastest), finite
/// * `dims` - `(nx, ny, nz)`, each at least 3
/// * `thickness` - Thicknesses in voxels (positive integers), `sigma = thickness/6`
/// * `c` - `StructureSensitivity` ([`FIBERMETRIC_DEFAULT_C`] is MATLAB's default for double)
/// * `polarity` - `ObjectPolarity`
/// * `roi` - If given, the response is computed only where `roi` is true and is 0 elsewhere
///   (the smoothing still uses the whole volume, so values inside `roi` are unchanged)
///
/// # Returns
/// The maximum response over thicknesses, 0 where every response is 0 or undefined.
pub fn fibermetric(
    data: &[f64],
    dims: (usize, usize, usize),
    thickness: &[usize],
    c: f64,
    polarity: FiberPolarity,
    roi: Option<&[bool]>,
) -> Vec<f64> {
    let (nx, ny, nz) = dims;
    let n = nx * ny * nz;
    assert_eq!(data.len(), n, "fibermetric: data length does not match dims");
    assert!(nx >= 3 && ny >= 3 && nz >= 3, "fibermetric: every axis needs at least 3 voxels");
    if let Some(r) = roi {
        assert_eq!(r.len(), n, "fibermetric: roi length does not match dims");
    }
    let mut out = vec![0.0f64; n];
    let two_c2 = 2.0 * c * c;
    for &t in thickness {
        assert!(t > 0, "fibermetric: thickness must be positive");
        let sigma = t as f64 / 6.0;
        let hsize = 2 * (3.0 * sigma).ceil() as usize + 1;
        let ig = imgaussfilt3(data, dims, sigma, hsize);
        // First derivatives, central with replicate padding.
        let g: [Vec<f64>; 3] = [0, 1, 2].map(|a| central(&ig, dims, a));
        let s2 = sigma * sigma;
        let score = |p: usize| -> f64 {
            let (i, j, k) = (p % nx, (p / nx) % ny, p / (nx * ny));
            let idx = [i, j, k];
            let h = |a: usize, b: usize| -> f64 {
                if a == b {
                    second_pure(&ig, &g[a], dims, a, idx, p)
                } else {
                    central_at(&g[a], dims, b, idx, p)
                }
            };
            let (xx, yy, zz) = (h(0, 0) * s2, h(1, 1) * s2, h(2, 2) * s2);
            let (xy, xz, yz) = (h(0, 1) * s2, h(0, 2) * s2, h(1, 2) * s2);
            let (l1, l2, l3) = eig_sorted_abs(xx, xy, xz, yy, yz, zz);
            let wrong = match polarity {
                FiberPolarity::Bright => l2 > 0.0 || l3 > 0.0,
                FiberPolarity::Dark => l2 < 0.0 || l3 < 0.0,
            };
            if wrong {
                return 0.0;
            }
            let (a1, a2, a3) = (l1.abs(), l2.abs(), l3.abs());
            let ra = a2 / a3;
            let rb = a1 / (a2 * a3).sqrt();
            let s = (l1 * l1 + l2 * l2 + l3 * l3).sqrt();
            // alpha = beta = 0.5, so 2 alpha^2 = 2 beta^2 = 0.5
            (1.0 - (-(ra * ra) / 0.5).exp()) * (-(rb * rb) / 0.5).exp() * (1.0 - (-(s * s) / two_c2).exp())
        };
        let update = |(p, o): (usize, &mut f64)| {
            if roi.is_some_and(|r| !r[p]) {
                return;
            }
            let v = score(p);
            // B = max(B, out): MATLAB's max drops NaN
            if v > *o {
                *o = v;
            }
        };
        #[cfg(feature = "parallel")]
        out.par_iter_mut().enumerate().with_min_len(1 << 12).for_each(update);
        #[cfg(not(feature = "parallel"))]
        out.iter_mut().enumerate().for_each(update);
    }
    out
}

/// `images.internal.createGaussianKernel(sigma, hsize)`, 1-D
fn gaussian_kernel(sigma: f64, hsize: usize) -> Vec<f64> {
    let r = (hsize as f64 - 1.0) / 2.0;
    let mut h: Vec<f64> = (0..hsize)
        .map(|q| {
            let x = q as f64 - r;
            (-(x * x / (sigma * sigma)) / 2.0).exp()
        })
        .collect();
    let max = h.iter().cloned().fold(0.0, f64::max);
    for v in h.iter_mut() {
        if *v < f64::EPSILON * max {
            *v = 0.0;
        }
    }
    let sum: f64 = h.iter().sum();
    if sum != 0.0 {
        for v in h.iter_mut() {
            *v /= sum;
        }
    }
    h
}

/// `imgaussfilt3(V, sigma, 'FilterSize', hsize)`: separable, replicate padding, y then x then z
fn imgaussfilt3(data: &[f64], dims: (usize, usize, usize), sigma: f64, hsize: usize) -> Vec<f64> {
    let h = gaussian_kernel(sigma, hsize);
    let a = filter_axis(data, dims, &h, 1);
    let b = filter_axis(&a, dims, &h, 0);
    filter_axis(&b, dims, &h, 2)
}

/// One `imfilter(A, h, 'replicate', 'conv', 'same')` pass along `axis` (h is symmetric)
fn filter_axis(data: &[f64], dims: (usize, usize, usize), h: &[f64], axis: usize) -> Vec<f64> {
    let (nx, ny, nz) = dims;
    let len = [nx, ny, nz][axis];
    let stride = [1, nx, nx * ny][axis];
    let r = (h.len() - 1) / 2;
    let mut out = vec![0.0f64; data.len()];
    let row = |p: usize, o: &mut f64| {
        let pos = (p / stride) % len;
        let base = p - pos * stride;
        let mut acc = 0.0;
        for (q, &w) in h.iter().enumerate() {
            let src = (pos as isize + q as isize - r as isize).clamp(0, len as isize - 1) as usize;
            acc += w * data[base + src * stride];
        }
        *o = acc;
    };
    #[cfg(feature = "parallel")]
    out.par_iter_mut().enumerate().with_min_len(1 << 12).for_each(|(p, o)| row(p, o));
    #[cfg(not(feature = "parallel"))]
    out.iter_mut().enumerate().for_each(|(p, o)| row(p, o));
    out
}

fn axis_geom(dims: (usize, usize, usize), axis: usize) -> (usize, usize) {
    let (nx, ny, nz) = dims;
    ([nx, ny, nz][axis], [1, nx, nx * ny][axis])
}

/// Central difference of `f` along `axis` at flat index `p` (position `idx`), replicate padding
#[inline]
fn central_at(f: &[f64], dims: (usize, usize, usize), axis: usize, idx: [usize; 3], p: usize) -> f64 {
    let (len, stride) = axis_geom(dims, axis);
    let i = idx[axis];
    let lo = if i == 0 { p } else { p - stride };
    let hi = if i + 1 == len { p } else { p + stride };
    (f[hi] - f[lo]) / 2.0
}

/// `H_aa` at `p`: the central difference of `G_a`, except `(f[2] + f[1] - 2 f[0])/4` at the
/// first voxel along `a` and its mirror image at the last (what the builtin computes)
#[inline]
fn second_pure(f: &[f64], ga: &[f64], dims: (usize, usize, usize), axis: usize, idx: [usize; 3], p: usize) -> f64 {
    let (len, stride) = axis_geom(dims, axis);
    let i = idx[axis];
    if i == 0 {
        (f[p + 2 * stride] + f[p + stride] - 2.0 * f[p]) / 4.0
    } else if i + 1 == len {
        (f[p - 2 * stride] + f[p - stride] - 2.0 * f[p]) / 4.0
    } else {
        (ga[p + stride] - ga[p - stride]) / 2.0
    }
}

/// The whole central-difference first derivative along `axis`
fn central(f: &[f64], dims: (usize, usize, usize), axis: usize) -> Vec<f64> {
    let (nx, ny, _) = dims;
    let mut out = vec![0.0f64; f.len()];
    let fill = |(p, o): (usize, &mut f64)| {
        let idx = [p % nx, (p / nx) % ny, p / (nx * ny)];
        *o = central_at(f, dims, axis, idx, p);
    };
    #[cfg(feature = "parallel")]
    out.par_iter_mut().enumerate().with_min_len(1 << 12).for_each(fill);
    #[cfg(not(feature = "parallel"))]
    out.iter_mut().enumerate().for_each(fill);
    out
}

/// Eigenvalues of the symmetric 3x3 matrix, sorted by magnitude (ascending)
fn eig_sorted_abs(xx: f64, xy: f64, xz: f64, yy: f64, yz: f64, zz: f64) -> (f64, f64, f64) {
    if ![xx, xy, xz, yy, yz, zz].iter().all(|v| v.is_finite()) {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mut v = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]];
    let mut d = [0f64; 3];
    let mut e = [0f64; 3];
    super::frangi::tred2(&mut v, &mut d, &mut e);
    super::frangi::tql2(&mut d, &mut e);
    d.sort_by(|a, b| a.abs().total_cmp(&b.abs()));
    (d[0], d[1], d[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_kernel_matches_matlab_shape() {
        // createGaussianKernel(1/6, 3): exp(-18) relative weight at +-1, normalised
        let h = gaussian_kernel(1.0 / 6.0, 3);
        let e = (-18.0f64).exp();
        assert!((h[1] - 1.0 / (1.0 + 2.0 * e)).abs() < 1e-15);
        assert!((h[0] - h[2]).abs() == 0.0);
    }

    #[test]
    fn bright_tube_scores_and_dark_does_not() {
        let (nx, ny, nz) = (15, 15, 12);
        let mut v = vec![0.0; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let r2 = (i as f64 - 7.0).powi(2) + (j as f64 - 7.0).powi(2);
                    v[i + nx * (j + ny * k)] = 50.0 * (-r2 / 3.0).exp();
                }
            }
        }
        let b = fibermetric(&v, (nx, ny, nz), &[2, 3, 4], FIBERMETRIC_DEFAULT_C, FiberPolarity::Bright, None);
        let d = fibermetric(&v, (nx, ny, nz), &[2, 3, 4], FIBERMETRIC_DEFAULT_C, FiberPolarity::Dark, None);
        let centre = 7 + nx * (7 + ny * 6);
        assert!(b[centre] > 0.5, "tube centre {}", b[centre]);
        assert_eq!(d[centre], 0.0);
    }
}

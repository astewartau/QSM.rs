//! iLSQR: LSQR dipole inversion with streaking-artefact removal
//!
//! [`ilsqr`] implements **STI Suite 3.0's `QSM_iLSQR`** (Li et al. 2015), as called with
//! `QSM_iLSQR(phase, mask, 'TE', TE, 'B0', B0, 'H', H, 'padsize', [64 64 64], 'voxelsize', vs)`.
//! STI Suite ships as protected P-code, so the algorithm was recovered by black-box probing
//! (call tracing, capturing the operators and right-hand sides it hands to MATLAB's `lsqr`, and
//! fitting every stage until a MATLAB re-implementation reproduced STI to its own
//! single-precision noise floor):
//!
//! 1. Crop field and mask to the mask's bounding box (odd extents grown by one voxel to make
//!    them even), then zero-pad each axis to the next multiple of 16 strictly above
//!    `crop + 2·pad_mm/voxel_size` (default `pad_mm` = 64). Everything below runs on that grid.
//! 2. **Laplacian weights** `W`: `L = |F⁻¹(k²·F φ)|` with `k²` in (cycles/mm)² and the DC term
//!    set to 10⁶ (STI's value, which adds `10⁶·mean(φ)` to every voxel, so the `|·|` decides
//!    which tail of the Laplacian is down-weighted). Ranked inside the mask:
//!    `W = 1` below the 60th percentile, falling linearly to `0.2` at the 99.9th, `0` outside.
//!    The field itself is not re-masked.
//! 3. **Initial solution**: LSQR on `D W D χ = D W φ` (`D` = dipole convolution), tolerance
//!    0.01, at most `max_iter` iterations (MATLAB `lsqr` stopping rules).
//! 4. **FastQSM** prior: `sign(D)·Fφ`, blended where `|D|` is small with a copy smoothed *in
//!    k-space* by a closed radius-3 ball, masked, blended again, rescaled to a TKD (threshold
//!    1/8) solution by least squares inside the mask.
//! 5. **Gradient weights** `G_d` (`d` = x, y, z): forward differences of the FastQSM map (no
//!    voxel-size scaling), pooled over the three axes inside the mask; `G = 1` below the 50th
//!    percentile of `|∇|`, `0` above the 70th, linear between.
//! 6. **Streaking artefacts**: LSQR for `y` in `min ‖G ∇ F⁻¹(M_ic y) − G ∇(M·Re χ₁)‖` with
//!    `M_ic = |D| < 0.1` (the ill-conditioned cone), tolerance `tol` (default 1e-3), at most
//!    `max_iter` iterations; artefact `χ_sa = Re F⁻¹(M_ic y)`.
//! 7. `χ = (Re χ₁ − χ_sa) · mask`, cropped back to the input grid.
//!
//! Percentiles are STI's: the sorted value at 1-based index `round(p·N)`, not interpolated.
//! The field may be in any linear unit (rad, Hz, ppm); `χ` comes out in the unit the field
//! would have for χ = 1 (QSM.rs passes ppm, so χ is in ppm).
//!
//! # Performance and precision
//!
//! Nearly all the time goes into the two LSQR solves, each costing two 3-D FFTs on the padded
//! grid per operator call. The initial solve runs in k-space (2 FFTs per call instead of 4, the
//! same iterates up to rounding), the transforms are pruned to the box holding the mask and
//! carry the solves' diagonal products in their first and last passes, and the artefact solve
//! stores its vectors on the cone and the mask only. On a UK Biobank SWI field (352×400×96
//! padded) this takes ~15 s on 16 cores with a 1.9 GB peak. Against the image-space
//! formulation of the initial solve, the k-space one gives the same stop flags and iteration
//! counts in both solves there, and χ within a relative L2 of 5.8e-9 (the artefact solve
//! amplifying rounding-level differences of the initial solution, which agree to 2.7e-15).
//!
//! Everything runs in double precision. STI Suite runs its LSQR solves in single precision;
//! the remaining difference to STI's output (r = 0.9999985, 99.9th percentile |Δ| 3.7e-4 ppm
//! on that field) is that rounding, amplified by the solves.
//!
//! QSM.rs ≤ v0.38 shipped a different iLSQR (a port of QSM.m's formulation: no padding, LSMR
//! for the artefacts, other weights). It was removed: STI Suite's algorithm is the reference
//! iLSQR, and it is also what the original QSMART calls (see [`crate::pipeline::run_qsmart`]).
//!
//! References:
//! Li, W., Wang, N., Yu, F., Han, H., Cao, W., Romero, R., Tantiwongkosi, B.,
//! Duong, T.Q., Liu, C. (2015). "A method for estimating and removing streaking
//! artifacts in quantitative susceptibility mapping."
//! NeuroImage, 108:111-122. <https://doi.org/10.1016/j.neuroimage.2014.12.043>
//!
//! Paige, C.C., Saunders, M.A. (1982). "LSQR: An algorithm for sparse linear equations and
//! sparse least squares." ACM Trans. Math. Softw. 8(1):43-71.

/// Parameters for the iLSQR algorithm.
///
/// STI Suite defaults: `tol = 1e-3`, `max_iter = 100`. The initial LSQR solve always uses
/// STI's tolerance of 0.01. `max_iter` is STI's `'niter'` argument (it caps both solves).
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct IlsqrParams {
    /// Tolerance of the streaking-artefact solve (default: 1e-3)
    pub tol: f64,
    /// Maximum iterations of each LSQR solve (default: 100)
    pub max_iter: usize,
}

impl Default for IlsqrParams {
    fn default() -> Self {
        Self {
            tol: 1e-3,
            max_iter: 100,
        }
    }
}

/// Zero padding STI's `QSM_iLSQR` is given by UK Biobank and in STI Suite's examples, in mm.
pub const STI_PAD_MM: f64 = 64.0;

/// Zero padding STI's `QSM_iLSQR` uses when called without `'padsize'` (as QSMART calls it),
/// in mm. Probed in STI Suite 3.0: output bitwise equal to an explicit `padsize` of 6 mm
/// (and different at 5.5 and 6.5) on five crop / voxel-size combinations.
pub const STI_DEFAULT_PAD_MM: f64 = 6.0;

use std::cell::{Cell, RefCell};
use num_complex::Complex64;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use crate::fft::{Box3, Fft3dWorkspace, FftOpts};
use crate::kernels::dipole::dipole_kernel;
use crate::Grid;

// ============================================================================
// LSQR Solver
// ============================================================================

/// LSQR iterative solver for Ax = b
///
/// Solves the least squares problem min ||Ax - b||² using the LSQR algorithm.
/// Based on Paige & Saunders (1982).
///
/// # Arguments
/// * `apply_a` - Function that computes A*x
/// * `apply_at` - Function that computes A^T*x
/// * `b` - Right-hand side vector
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
///
/// # Returns
/// Solution vector x
pub fn lsqr<F, G>(
    apply_a: F,
    apply_at: G,
    b: &[f64],
    tol: f64,
    max_iter: usize,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
    G: Fn(&[f64]) -> Vec<f64>,
{
    // Initialize
    let mut u = b.to_vec();
    let mut beta = norm(&u);

    if beta > 0.0 {
        scale_inplace(&mut u, 1.0 / beta);
    }

    let mut v = apply_at(&u);
    let n = v.len();
    let mut alpha = norm(&v);

    if alpha > 0.0 {
        scale_inplace(&mut v, 1.0 / alpha);
    }

    let mut w = v.clone();
    let mut x = vec![0.0; n];

    let mut phi_bar = beta;
    let mut rho_bar = alpha;

    let bnorm = beta;

    // A zero right-hand side has x = 0 as its exact solution, and the plane
    // rotation below would divide 0 by 0 and hand back NaN.
    if bnorm == 0.0 {
        return x;
    }

    for _iter in 0..max_iter {
        // Bidiagonalization
        let mut u_new = apply_a(&v);
        axpy(&mut u_new, -alpha, &u);
        beta = norm(&u_new);

        if beta > 0.0 {
            scale_inplace(&mut u_new, 1.0 / beta);
        }
        u = u_new;

        let mut v_new = apply_at(&u);
        axpy(&mut v_new, -beta, &v);
        alpha = norm(&v_new);

        if alpha > 0.0 {
            scale_inplace(&mut v_new, 1.0 / alpha);
        }
        v = v_new;

        // Construct and apply rotation
        let rho = (rho_bar * rho_bar + beta * beta).sqrt();
        let c = rho_bar / rho;
        let s = beta / rho;
        let theta = s * alpha;
        rho_bar = -c * alpha;
        let phi = c * phi_bar;
        phi_bar *= s;

        // Update x and w
        let t1 = phi / rho;
        let t2 = -theta / rho;

        for i in 0..n {
            x[i] += t1 * w[i];
            w[i] = v[i] + t2 * w[i];
        }

        // Check convergence
        let rel_residual = phi_bar / (bnorm + 1e-20);

        if rel_residual < tol {
            break;
        }
    }

    x
}

fn norm(x: &[f64]) -> f64 {
    x.iter().map(|&v| v * v).sum::<f64>().sqrt()
}

fn scale_inplace(x: &mut [f64], s: f64) {
    for v in x.iter_mut() {
        *v *= s;
    }
}

fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += a * xi;
    }
}

// ============================================================================
// STI Suite 3.0 QSM_iLSQR (default `ilsqr`)
// ============================================================================

/// How an [`lsqr_matlab`] solve ended (MATLAB `lsqr` flags).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LsqrOutcome {
    /// 0 converged, 1 hit `max_iter`, 3 stagnated, 4 breakdown (a scalar became zero or non-finite)
    pub flag: u8,
    /// Iterations whose update is included in the returned solution
    pub iter: usize,
    /// ‖b − A x‖ / ‖b‖ (LSQR's running estimate)
    pub relres: f64,
}

/// Chunk length of the LSQR norms: fixed, so every sum is independent of the thread count.
const NORM_CHUNK: usize = 1 << 16;

/// How an LSQR vector is split for its norms: lengths of consecutive parts, each summed
/// sequentially, the partial sums then added in order.
///
/// A vector stored densely uses [`dense_parts`] (`NORM_CHUNK`-long parts). A
/// vector stored *compactly*, as only the entries at a sorted list of positions in a longer
/// vector that is exactly zero elsewhere, uses [`compact_parts`]: one part per `NORM_CHUNK`
/// of the long vector. Adding the omitted zeros changes no partial sum, so the norm of the
/// compact vector equals the dense norm of the long one.
fn dense_parts(len: usize) -> Vec<usize> {
    (0..len.div_ceil(NORM_CHUNK)).map(|c| NORM_CHUNK.min(len - c * NORM_CHUNK)).collect()
}

/// See [`dense_parts`]. `pos` are the (strictly increasing) positions in the long vector.
fn compact_parts(pos: impl Iterator<Item = usize>) -> Vec<usize> {
    let mut parts: Vec<usize> = Vec::new();
    let mut last = usize::MAX;
    for g in pos {
        let c = g / NORM_CHUNK;
        if c == last {
            *parts.last_mut().unwrap() += 1;
        } else {
            parts.push(1);
            last = c;
        }
    }
    parts
}

/// One of [`Fft3dWorkspace`]'s pruned / fused transforms (forward, or inverse if `inverse`).
fn transform(ws: &mut Fft3dWorkspace, a: &mut [Complex64], inverse: bool, o: &FftOpts) {
    if inverse { ws.ifft3d_with(a, o) } else { ws.fft3d_with(a, o) }
}

/// `|z|²` (exactly `z.norm_sqr()`)
#[inline(always)]
fn nsq(z: Complex64) -> f64 {
    z.re * z.re + z.im * z.im
}

fn split_parts<'a, T>(mut x: &'a [T], parts: &[usize]) -> Vec<&'a [T]> {
    parts.iter().map(|&len| { let (a, b) = x.split_at(len); x = b; a }).collect()
}

fn split_parts_mut<'a, T>(mut x: &'a mut [T], parts: &[usize]) -> Vec<&'a mut [T]> {
    parts.iter().map(|&len| { let (a, b) = std::mem::take(&mut x).split_at_mut(len); x = b; a }).collect()
}

fn cnorm_parts(x: &[Complex64], parts: &[usize]) -> f64 {
    let xs = split_parts(x, parts);
    let sums: Vec<f64> = crate::maybe_par_iter!(xs).map(|c| c.iter().map(|&v| nsq(v)).sum::<f64>()).collect();
    sums.iter().sum::<f64>().sqrt()
}

/// `f` applied elementwise to `y` (paired with `x`), then [`cnorm_parts`]`(y)`, in one pass:
/// the same values as the two passes, part by part.
fn update_and_norm(
    y: &mut [Complex64],
    x: &[Complex64],
    parts: &[usize],
    f: impl Fn(&mut Complex64, Complex64) + Sync + Send,
) -> f64 {
    let mut ys = split_parts_mut(y, parts);
    let xs = split_parts(x, parts);
    let sums: Vec<f64> = crate::maybe_par_iter_mut!(ys)
        .zip(crate::maybe_par_iter!(xs))
        .map(|(yc, xc)| {
            yc.iter_mut().zip(xc.iter()).for_each(|(a, &b)| f(a, b));
            yc.iter().map(|&v| nsq(v)).sum::<f64>()
        })
        .collect();
    sums.iter().sum::<f64>().sqrt()
}

/// LSQR's direction and solution updates `d ← (v − θ d)/ρ`, `x ← x + φ d` in one pass,
/// returning `(‖d‖, ‖x‖)`: the same values as two [`update_and_norm`] passes, part by part.
#[allow(clippy::too_many_arguments)]
fn update_d_x_and_norms(d: &mut [Complex64], x: &mut [Complex64], v: &[Complex64], parts: &[usize], thet: f64, rho: f64, phi: f64) -> (f64, f64) {
    let mut ds = split_parts_mut(d, parts);
    let mut xs = split_parts_mut(x, parts);
    let vs = split_parts(v, parts);
    let sums: Vec<(f64, f64)> = crate::maybe_par_iter_mut!(ds)
        .zip(crate::maybe_par_iter_mut!(xs))
        .zip(crate::maybe_par_iter!(vs))
        .map(|((dc, xc), vc)| {
            let (mut sd, mut sx) = (0.0f64, 0.0f64);
            for ((di, xi), &vi) in dc.iter_mut().zip(xc.iter_mut()).zip(vc.iter()) {
                *di = (vi - *di * thet) / rho;
                *xi += *di * phi;
                sd += nsq(*di);
                sx += nsq(*xi);
            }
            (sd, sx)
        })
        .collect();
    (sums.iter().map(|p| p.0).sum::<f64>().sqrt(), sums.iter().map(|p| p.1).sum::<f64>().sqrt())
}

fn div_inplace(x: &mut [Complex64], s: f64) {
    crate::maybe_par_iter_mut!(x).for_each(|z| *z /= s);
}

/// LSQR (Paige & Saunders 1982) with MATLAB `lsqr`'s stopping rules, zero initial guess, no
/// preconditioner: stops when `‖r‖ ≤ tol·‖b‖` or `‖Aᴴr‖ / (‖A‖_F·‖r‖) ≤ tol` (both tested on
/// the previous iterate, before it is updated), after three stagnant steps, or on breakdown.
/// `apply_a` computes `A x`, `apply_ah` computes `Aᴴ u`.
pub fn lsqr_matlab<F, G>(
    mut apply_a: F,
    mut apply_ah: G,
    b: &[Complex64],
    tol: f64,
    max_iter: usize,
) -> (Vec<Complex64>, LsqrOutcome)
where
    F: FnMut(&[Complex64]) -> Vec<Complex64>,
    G: FnMut(&[Complex64]) -> Vec<Complex64>,
{
    let nx = apply_ah(b).len();
    lsqr_matlab_into(
        |v, out| out.copy_from_slice(&apply_a(v)),
        |u, out| out.copy_from_slice(&apply_ah(u)),
        b.to_vec(),
        (&dense_parts(b.len()), &dense_parts(nx)),
        tol,
        max_iter,
    )
}

/// [`lsqr_matlab`] with operators that write into a caller-owned buffer (`out` holds stale
/// data on entry and must be overwritten), taking `b` by value as LSQR's first `u`, so a solve
/// allocates its six vectors once. `parts` = (range parts, domain parts): how the norms of `u`
/// and `v` vectors are split (see [`dense_parts`]); the domain parts also give the length of
/// the unknown. Performs exactly the same floating-point operations as the textbook loop (the
/// norms of updated vectors are fused into the update passes, with the same parts).
fn lsqr_matlab_into<F, G>(
    mut apply_a: F,
    mut apply_ah: G,
    b: Vec<Complex64>,
    parts: (&[usize], &[usize]),
    tol: f64,
    max_iter: usize,
) -> (Vec<Complex64>, LsqrOutcome)
where
    F: FnMut(&[Complex64], &mut [Complex64]),
    G: FnMut(&[Complex64], &mut [Complex64]),
{
    let zero = Complex64::new(0.0, 0.0);
    let (up, vp) = parts;
    let nx: usize = vp.iter().sum();
    let n2b = cnorm_parts(&b, up);
    let tolb = tol * n2b;
    let mut u = b;
    let mut beta = n2b;
    let mut normr = beta;
    if beta != 0.0 {
        div_inplace(&mut u, beta);
    }
    let (mut c, mut s) = (1.0f64, 0.0f64);
    let mut phibar = beta;
    let mut v = vec![zero; nx];
    apply_ah(&u, &mut v);
    let mut x = vec![zero; nx];
    let mut alpha = cnorm_parts(&v, vp);
    if alpha != 0.0 {
        div_inplace(&mut v, alpha);
    }
    let mut normar = alpha * beta;
    let relres = |nr: f64| if n2b > 0.0 { nr / n2b } else { 0.0 };
    if normar == 0.0 {
        return (x, LsqrOutcome { flag: 0, iter: 0, relres: relres(normr) });
    }
    let mut un = vec![zero; u.len()];
    let mut vt = vec![zero; nx];
    let mut d = vec![zero; nx];
    let mut normx = 0.0f64; // cnorm(x), kept up to date by the x update
    let mut norma = 0.0f64;
    let mut stag = 0usize;
    let mut flag = 1u8;
    let mut iter = max_iter;
    for ii in 1..=max_iter {
        apply_a(&v, &mut un);
        beta = update_and_norm(&mut un, &u, up, |a, b| *a -= b * alpha);
        if beta != 0.0 {
            div_inplace(&mut un, beta);
        }
        std::mem::swap(&mut u, &mut un);
        norma = (norma * norma + alpha * alpha + beta * beta).sqrt();
        let thet = -s * alpha;
        let rhot = c * alpha;
        let rho = (rhot * rhot + beta * beta).sqrt();
        c = rhot / rho;
        s = -beta / rho;
        let phi = c * phibar;
        if phi == 0.0 {
            stag = 1;
        }
        phibar *= s;
        let converged = normar / (norma * normr) <= tol || normr <= tolb;
        let breakdown = !phi.is_finite() || rho == 0.0 || !rho.is_finite();
        let normx_prev = normx;
        // The x update needs ‖d‖ only through the stagnation test, which cannot stop this
        // iteration while stag < 2: then d and x are updated in one pass.
        let fused = !converged && !breakdown && stag < 2;
        let normd = if fused {
            let (nd, nxn) = update_d_x_and_norms(&mut d, &mut x, &v, vp, thet, rho, phi);
            normx = nxn;
            nd
        } else {
            update_and_norm(&mut d, &v, vp, |di, vi| *di = (vi - *di * thet) / rho)
        };
        if phi.abs() * normd < f64::EPSILON * normx_prev {
            stag += 1;
        } else {
            stag = 0;
        }
        if converged {
            flag = 0;
            iter = ii - 1;
            break;
        }
        if stag >= 3 {
            flag = 3;
            iter = ii - 1;
            break;
        }
        if breakdown {
            flag = 4;
            iter = ii - 1;
            break;
        }
        if !fused {
            normx = update_and_norm(&mut x, &d, vp, |xi, di| *xi += di * phi);
        }
        normr *= s.abs();
        apply_ah(&u, &mut vt);
        alpha = update_and_norm(&mut vt, &v, vp, |a, b| *a -= b * beta);
        std::mem::swap(&mut v, &mut vt);
        if (alpha == 0.0 || !alpha.is_finite()) && ii < max_iter {
            flag = 4;
            iter = ii;
            break;
        }
        if alpha != 0.0 {
            div_inplace(&mut v, alpha);
        }
        normar = alpha * (s * phi).abs();
    }
    if flag == 1 && (normar / (norma * normr) <= tol || normr <= tolb) {
        flag = 0;
    }
    (x, LsqrOutcome { flag, iter, relres: relres(normr) })
}

/// 0-based index of STI's percentile: the sorted value at 1-based index `round(p·N)`.
fn sti_percentile_index(n: usize, p: f64) -> usize {
    ((p * n as f64).round() as usize).clamp(1, n) - 1
}

/// Sorted value at 1-based index `round(p·N)` (STI's percentile; `p` in [0, 1]).
#[cfg(test)]
fn sti_percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[sti_percentile_index(sorted.len(), p)]
}

/// STI's percentiles `p_lo ≤ p_hi` of `v` (which is reordered), by selection instead of a full
/// sort: the k-th element in `total_cmp` order is unique, so this is exactly the sorted lookup.
fn sti_percentiles(v: &mut [f64], p_lo: f64, p_hi: f64) -> (f64, f64) {
    let (il, ih) = (sti_percentile_index(v.len(), p_lo), sti_percentile_index(v.len(), p_hi));
    debug_assert!(il <= ih);
    let hi = *v.select_nth_unstable_by(ih, |a, b| a.total_cmp(b)).1;
    let lo = if il == ih { hi } else { *v[..ih].select_nth_unstable_by(il, |a, b| a.total_cmp(b)).1 };
    (lo, hi)
}

/// STI's `MaskBoundingBox`: tight bounding box (0-based, inclusive); an odd extent grows by one
/// voxel at the high end, or at the low end when the high end is the array edge (and shrinks
/// by one at the high end when both ends are edges).
fn sti_bounding_box(mask: &[u8], dims: (usize, usize, usize)) -> Option<([usize; 3], [usize; 3])> {
    let (nx, ny, nz) = dims;
    let n = [nx, ny, nz];
    let mut lo = n;
    let mut hi = [0usize; 3];
    let mut any = false;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if mask[i + j * nx + k * nx * ny] != 0 {
                    any = true;
                    for (d, c) in [i, j, k].into_iter().enumerate() {
                        lo[d] = lo[d].min(c);
                        hi[d] = hi[d].max(c);
                    }
                }
            }
        }
    }
    if !any {
        return None;
    }
    for d in 0..3 {
        if (hi[d] - lo[d] + 1) % 2 == 1 {
            if hi[d] + 1 < n[d] {
                hi[d] += 1;
            } else if lo[d] > 0 {
                lo[d] -= 1;
            } else {
                hi[d] -= 1;
            }
        }
    }
    Some((lo, hi))
}

/// STI's padded grid: per axis the next multiple of 16 strictly above `c + 2·pad_mm/vs`.
fn sti_padded_dims(c: [usize; 3], vs: [f64; 3], pad_mm: [f64; 3]) -> [usize; 3] {
    [0, 1, 2].map(|d| 16 * ((c[d] as f64 + 2.0 * pad_mm[d] / vs[d] + 1.0) / 16.0).ceil() as usize)
}

/// Sentinel of the compact-index maps in the artefact solve.
const NONE: u32 = u32::MAX;

/// Circular neighbour `l + e_axis` on the grid `n`.
#[inline]
fn circ_next(l: usize, axis: usize, n: [usize; 3]) -> usize {
    let step = [1, n[0], n[0] * n[1]][axis];
    if (l / step) % n[axis] + 1 == n[axis] { l - (n[axis] - 1) * step } else { l + step }
}

/// Weighted circular forward differences (no voxel-size scaling), as STI's iLSQR uses, at the
/// mask voxels only: `r[a·nm + i] = gw[a][i] · (val(l + e_a) − val(l))` with `l = inside[i]`,
/// `nm = inside.len()`, `gw` the weights compacted to the mask.
fn grad_masked(val: &(dyn Fn(usize) -> Complex64 + Sync), inside: &[usize], gw: &[Vec<f64>], n: [usize; 3], r: &mut [Complex64]) {
    let nm = inside.len();
    for (a, ra) in r.chunks_mut(nm).enumerate() {
        let g = &gw[a];
        crate::maybe_par_iter_mut!(ra).enumerate().for_each(|(i, o)| {
            let l = inside[i];
            *o = (val(circ_next(l, a, n)) - val(l)) * g[i];
        });
    }
}

/// Adjoint of [`grad_masked`] onto the box `bx` of the grid: `acc[l] = Σ_a (q_a[l − e_a] − q_a[l])`
/// for `l` in `bx` (the rest of `acc` is left as it is), summed in axis order from zero, with
/// `q_a = gw[a]·r_a` on the mask and zero elsewhere (`mask_of[l]` = position of `l` in
/// `inside`, or [`NONE`]). The result is zero outside the mask grown by one voxel along +e_a.
fn grad_masked_adj(r: &[Complex64], mask_of: &[u32], gw: &[Vec<f64>], n: [usize; 3], bx: &Box3, acc: &mut [Complex64]) {
    let (nxy, nt) = (n[0] * n[1], n[0] * n[1] * n[2]);
    let nm = r.len() / 3;
    let zero = Complex64::new(0.0, 0.0);
    let q = |a: usize, l: usize| match mask_of[l] {
        NONE => zero,
        i => r[a * nm + i as usize] * gw[a][i as usize],
    };
    crate::maybe_par_chunks_mut!(acc, nxy).enumerate().filter(|(k, _)| bx[2].contains(k)).for_each(|(k, pa)| {
        let uz = if k == 0 { nt } else { 0 }; // circular: l − e_z = l + uz − nxy
        for j in bx[1].clone() {
            let uy = if j == 0 { nxy } else { 0 };
            let row = k * nxy + j * n[0];
            for i in bx[0].clone() {
                let l = row + i;
                let xp = if i == 0 { l + n[0] - 1 } else { l - 1 };
                let mut o = zero;
                o += q(0, xp) - q(0, l);
                o += q(1, l + uy - n[0]) - q(1, l);
                o += q(2, l + uz - nxy) - q(2, l);
                pa[j * n[0] + i] = o;
            }
        }
    });
}

/// iLSQR as in STI Suite 3.0 (`QSM_iLSQR`), with STI's 64 mm padding.
///
/// See the module docs for the algorithm. `field` is the local field (any linear unit; QSM.rs
/// passes ppm), `bdir` the B0 direction in voxel axes (normalised here).
///
/// # Returns
/// `(chi, streaking_artifacts, fast_qsm, initial_lsqr)` on the input grid, all masked.
/// `initial_lsqr` is STI's second output (the step-3 LSQR solution).
pub fn ilsqr(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &IlsqrParams,
    progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    ilsqr_with_padding(field, mask, grid, bdir, params, [STI_PAD_MM; 3], progress)
}

/// [`ilsqr`] with an explicit STI `padsize` (mm per axis; STI's `'padsize'` argument).
pub fn ilsqr_with_padding(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &IlsqrParams,
    pad_mm: [f64; 3],
    progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    ilsqr_with_padding_traced(field, mask, grid, bdir, params, pad_mm, progress, &mut IlsqrTrace::default())
}

/// Where an [`ilsqr_with_padding_traced`] run spent its time, and how its two LSQR solves ended.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct IlsqrTrace {
    /// Padded grid
    pub dims: [usize; 3],
    /// (stage, wall seconds, 3-D FFTs, FFT wall seconds) in execution order
    pub stages: Vec<(&'static str, f64, usize, f64)>,
    /// Initial solve and artefact solve
    pub solves: Vec<LsqrOutcome>,
    /// 3-D FFTs (forward + inverse) on the padded grid, and their total wall time
    pub n_fft: usize,
    pub fft_secs: f64,
}

/// [`ilsqr_with_padding`], also filling an [`IlsqrTrace`] (for benchmarks and parity checks).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn ilsqr_with_padding_traced(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &IlsqrParams,
    pad_mm: [f64; 3],
    mut progress: impl FnMut(usize, usize),
    trace: &mut IlsqrTrace,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    *trace = IlsqrTrace::default();
    let clock = FftClock::default();
    let mut t_stage = (std::time::Instant::now(), 0usize, 0.0f64);
    let mut stage = |trace: &mut IlsqrTrace, name: &'static str| {
        let (n, t) = (clock.n.get(), clock.secs.get());
        trace.stages.push((name, t_stage.0.elapsed().as_secs_f64(), n - t_stage.1, t - t_stage.2));
        t_stage = (std::time::Instant::now(), n, t);
    };
    let (nx, ny, nz) = grid.dims;
    let n_in = nx * ny * nz;
    let vs = [grid.voxel_size.0, grid.voxel_size.1, grid.voxel_size.2];
    let zeros = || vec![0.0; n_in];
    let Some((lo, hi)) = sti_bounding_box(mask, grid.dims) else {
        return (zeros(), zeros(), zeros(), zeros());
    };
    progress(1, 4);

    // 1. crop to the (even) bounding box and pad to the next multiple of 16 above c + 2 pad/vs
    let c = [0, 1, 2].map(|d| hi[d] - lo[d] + 1);
    let n = sti_padded_dims(c, vs, pad_mm);
    let pd = [0, 1, 2].map(|d| (n[d] - c[d]) / 2);
    let nt = n[0] * n[1] * n[2];
    let mut p = vec![0.0; nt];
    let mut m = vec![0.0; nt];
    for k in 0..c[2] {
        for j in 0..c[1] {
            for i in 0..c[0] {
                let src = (i + lo[0]) + (j + lo[1]) * nx + (k + lo[2]) * nx * ny;
                let dst = (i + pd[0]) + (j + pd[1]) * n[0] + (k + pd[2]) * n[0] * n[1];
                p[dst] = field[src];
                m[dst] = if mask[src] != 0 { 1.0 } else { 0.0 };
            }
        }
    }
    let inside: Vec<usize> = (0..nt).filter(|&l| m[l] > 0.0).collect();
    let pgrid = Grid::new(n[0], n[1], n[2], vs[0], vs[1], vs[2]);
    let dk = dipole_kernel(&pgrid, bdir);
    trace.dims = n;
    let ws = RefCell::new(Fft3dWorkspace::new(n[0], n[1], n[2]));
    let fft = |a: &mut [Complex64]| clock.time(|| ws.borrow_mut().fft3d(a));
    let ifft = |a: &mut [Complex64]| clock.time(|| ws.borrow_mut().ifft3d(a));
    // the mask lies in the crop box; its forward-difference neighbours in the box grown by one
    let crop_box: Box3 = [0, 1, 2].map(|d| pd[d]..pd[d] + c[d]);
    let crop_box1: Box3 = [0, 1, 2].map(|d| pd[d]..(pd[d] + c[d] + 1).min(n[d]));
    stage(trace, "crop/pad, dipole kernel, plans");
    let mut p_k: Vec<Complex64> = crate::maybe_par_iter!(p).map(|&v| Complex64::new(v, 0.0)).collect();
    fft(&mut p_k[..]);

    // 2. Laplacian weights: |F⁻¹(k² F φ)| with STI's 1e6 at DC, ranked in the mask
    let kx = crate::fft::fftfreq(n[0], vs[0]);
    let ky = crate::fft::fftfreq(n[1], vs[1]);
    let kz = crate::fft::fftfreq(n[2], vs[2]);
    let mut lap: Vec<Complex64> = p_k.clone();
    crate::maybe_par_iter_mut!(lap).enumerate().for_each(|(l, z)| {
        let (i, j, k) = (l % n[0], (l / n[0]) % n[1], l / (n[0] * n[1]));
        let k2 = if l == 0 { 1e6 } else { kx[i] * kx[i] + ky[j] * ky[j] + kz[k] * kz[k] };
        *z *= k2;
    });
    ifft(&mut lap[..]);
    let mut w: Vec<f64> = crate::maybe_par_iter!(lap).map(|z| z.re.abs()).collect(); // |L|, then W
    drop(lap);
    let mut ranked: Vec<f64> = inside.iter().map(|&l| w[l]).collect();
    let (t60, t999) = sti_percentiles(&mut ranked, 0.6, 0.999);
    drop(ranked);
    crate::maybe_par_iter_mut!(w).zip(crate::maybe_par_iter!(m)).for_each(|(v, &mv)| {
        *v = if mv == 0.0 {
            0.0
        } else if t999 > t60 {
            (1.0 - 0.8 * (*v - t60) / (t999 - t60)).clamp(0.2, 1.0)
        } else if *v <= t60 {
            1.0
        } else {
            0.2
        };
    });
    stage(trace, "Laplacian weights (incl. F phi)");

    // 3. initial solution: LSQR on D W D χ = D W φ, with D = F⁻¹ diag(dk) F (initial_solve)
    let (x1r, out1) = initial_solve(n, &p, &w, &dk, &crop_box, params.max_iter, &clock);
    trace.solves.push(out1);
    stage(trace, "solve 1 (initial LSQR)");
    progress(2, 4);

    // 4. FastQSM prior (sign(D) inverse blended with a k-space-smoothed copy where |D| is small)
    let xfs = sti_fastqsm(&p_k, &m, &inside, &dk, n, &fft, &ifft);
    drop(p_k);
    stage(trace, "FastQSM");
    progress(3, 4);

    // 5. gradient weights from the FastQSM map, pooled over the three axes in the mask
    let mut gw: Vec<Vec<f64>> = (0..3).map(|_| vec![0.0; nt]).collect();
    let mut pool = Vec::with_capacity(3 * inside.len());
    for (axis, a) in gw.iter_mut().enumerate() {
        // |∇_axis xfs| (circular forward difference), held in gw[axis] until the weights replace it
        let xfs = &xfs;
        crate::maybe_par_iter_mut!(a).enumerate().for_each(|(l, o)| *o = (xfs[circ_next(l, axis, n)] - xfs[l]).abs());
        pool.extend(inside.iter().map(|&l| a[l]));
    }
    let (t50, t70) = sti_percentiles(&mut pool, 0.5, 0.7);
    drop(pool);
    for a in gw.iter_mut() {
        crate::maybe_par_iter_mut!(a).zip(crate::maybe_par_iter!(m)).for_each(|(g, &mv)| {
            *g = if mv == 0.0 {
                0.0
            } else if t70 > t50 {
                ((t70 - *g) / (t70 - t50)).clamp(0.0, 1.0)
            } else if *g <= t50 {
                1.0
            } else {
                0.0
            };
        });
    }
    stage(trace, "gradient weights");
    progress(4, 4);

    // 6. streaking artefacts (artefact_solve), subtracted from the initial solution
    let sq = (nt as f64).sqrt();
    let (xsa, out2) = artefact_solve(n, &x1r, &m, &inside, gw, &dk, &crop_box1, params.tol, params.max_iter, &clock);
    trace.solves.push(out2);

    // 7. subtract, mask, crop back
    let mut chi = zeros();
    let mut sa = zeros();
    let mut fs = zeros();
    let mut x0 = zeros();
    for k in 0..c[2] {
        for j in 0..c[1] {
            for i in 0..c[0] {
                let dst = (i + lo[0]) + (j + lo[1]) * nx + (k + lo[2]) * nx * ny;
                let src = (i + pd[0]) + (j + pd[1]) * n[0] + (k + pd[2]) * n[0] * n[1];
                if m[src] > 0.0 {
                    let a = xsa[src] * sq;
                    chi[dst] = x1r[src] - a;
                    sa[dst] = a;
                    fs[dst] = xfs[src];
                    x0[dst] = x1r[src];
                }
            }
        }
    }
    stage(trace, "solve 2 (artefact LSQR, incl. setup) + output");
    trace.n_fft = clock.n.get();
    trace.fft_secs = clock.secs.get();
    (chi, sa, fs, x0)
}

/// FFT count and wall time of an iLSQR run (for [`IlsqrTrace`]).
#[derive(Default)]
struct FftClock {
    n: Cell<usize>,
    secs: Cell<f64>,
}

impl FftClock {
    fn time<R>(&self, f: impl FnOnce() -> R) -> R {
        let t = std::time::Instant::now();
        let r = f();
        self.n.set(self.n.get() + 1);
        self.secs.set(self.secs.get() + t.elapsed().as_secs_f64());
        r
    }
}

/// Step 3 of [`ilsqr_with_padding`]: LSQR (tol 0.01) on D W D χ = D W φ with
/// D = F⁻¹ diag(dk) F. Returns χ and how LSQR ended.
///
/// Solved in k-space, for x̃ = F χ: with the unitary U = F/√N, D W D = U* (dk·U W U*·dk) U,
/// so the LSQR iterates for (U A U*, U b) are U times those for (A, b) and the norms are equal
/// (exact arithmetic). The √N factors cancel in dk·F W F⁻¹·dk, and taking F b = √N·U b as the
/// right-hand side (LSQR is linear in b and its stopping tests are scale-free) leaves
/// χ = F⁻¹ x̃. That is 2 FFTs per operator call instead of 4; rounding differs only at the
/// level of the transforms' own rounding. The products with dk and W are fused into the
/// transforms' first and last passes, which only cover the crop box (W = 0 outside the mask).
#[allow(clippy::too_many_arguments)]
fn initial_solve(
    n: [usize; 3],
    p: &[f64],
    w: &[f64],
    dk: &[f64],
    crop_box: &Box3,
    max_iter: usize,
    clock: &FftClock,
) -> (Vec<f64>, LsqrOutcome) {
    let ws = RefCell::new(Fft3dWorkspace::new(n[0], n[1], n[2]));
    let tr = |a: &mut [Complex64], inverse: bool, o: &FftOpts| clock.time(|| transform(&mut ws.borrow_mut(), a, inverse, o));
    let apply1 = |v: &[Complex64], t: &mut [Complex64]| {
        let load = |l: usize, row: &mut [Complex64]| {
            row.iter_mut().zip(&v[l..]).zip(&dk[l..]).for_each(|((o, &z), &dv)| *o = z * dv);
        };
        tr(t, true, &FftOpts { needed: Some(crop_box), load: Some(&load), post: Some(w), ..Default::default() });
        tr(t, false, &FftOpts { support: Some(crop_box), post: Some(dk), ..Default::default() });
    };
    let mut b1: Vec<Complex64> = crate::maybe_par_iter!(p).zip(crate::maybe_par_iter!(w))
        .map(|(&pv, &wv)| Complex64::new(pv * wv, 0.0)).collect();
    tr(&mut b1, false, &FftOpts::default());
    crate::maybe_par_iter_mut!(b1).zip(crate::maybe_par_iter!(dk)).for_each(|(z, &dv)| *z *= dv);
    let dense = dense_parts(p.len());
    let (mut x1, out) = lsqr_matlab_into(apply1, apply1, b1, (&dense, &dense), 0.01, max_iter);
    tr(&mut x1, true, &FftOpts::default());
    (crate::maybe_par_iter!(x1).map(|z| z.re).collect(), out)
}

/// Step 6 of [`ilsqr_with_padding`]: the streaking artefacts, by LSQR (`tol`) over
/// the k-space cone |D| < 0.1 against the weighted gradient (weights `gw`, full grid,
/// zero off the mask) of the initial solution `x1`. Returns the real part of F⁻¹ of the
/// solution (the artefacts divided by √N) and how LSQR ended.
///
/// A y = G ∇ (√N F⁻¹(M_ic y)), Aᴴ r = M_ic F(∇ᴴ(G r)) / √N. Every vector LSQR forms in the
/// domain is exactly zero off the cone, and every one in the range is exactly zero off the
/// mask (G = 0 there), so both are stored compactly: the unknown on the cone positions, the
/// residual on the mask positions of each axis. With norms split as for the dense vectors
/// (`compact_parts`) the iterates have the same values as a dense solve. The image-space data
/// lives in `crop_box1` (the mask and its +e_a neighbours), so the transforms are pruned to it.
#[allow(clippy::too_many_arguments)]
fn artefact_solve(
    n: [usize; 3],
    x1: &[f64],
    m: &[f64],
    inside: &[usize],
    gw: Vec<Vec<f64>>,
    dk: &[f64],
    crop_box1: &Box3,
    tol: f64,
    max_iter: usize,
    clock: &FftClock,
) -> (Vec<f64>, LsqrOutcome) {
    let nt = dk.len();
    let sq = (nt as f64).sqrt();
    let cone: Vec<usize> = (0..nt).filter(|&l| dk[l].abs() < 0.1).collect();
    let nm = inside.len();
    assert!(nt < NONE as usize, "grid too large for 32-bit voxel indices");
    let index_of = |list: &[usize]| {
        let mut map = vec![NONE; nt];
        for (i, &l) in list.iter().enumerate() {
            map[l] = i as u32;
        }
        map
    };
    let cone_of = index_of(&cone);
    let mask_of = index_of(inside);
    let gwc: Vec<Vec<f64>> = gw.iter().map(|g| inside.iter().map(|&l| g[l]).collect()).collect();
    drop(gw);
    let u_parts = compact_parts((0..3).flat_map(|a| inside.iter().map(move |&l| a * nt + l)));
    let v_parts = compact_parts(cone.iter().copied());
    let zero = Complex64::new(0.0, 0.0);
    let ws = RefCell::new(Fft3dWorkspace::new(n[0], n[1], n[2]));
    let tr = |a: &mut [Complex64], inverse: bool, o: &FftOpts| clock.time(|| transform(&mut ws.borrow_mut(), a, inverse, o));
    let isq = 1.0 / sq;
    let t_cell = RefCell::new(vec![zero; nt]);
    let apply2 = |y: &[Complex64], out: &mut [Complex64]| {
        let mut t = t_cell.borrow_mut();
        // scatter y onto the cone in the transform's first pass; read on the mask and its
        // +e_a neighbours only
        let load = |l: usize, row: &mut [Complex64]| {
            row.iter_mut().zip(&cone_of[l..]).for_each(|(o, &ci)| *o = if ci == NONE { zero } else { y[ci as usize] });
        };
        tr(&mut t[..], true, &FftOpts { needed: Some(crop_box1), load: Some(&load), ..Default::default() });
        let t = &t[..];
        grad_masked(&|l| t[l] * sq, inside, &gwc, n, out);
    };
    let apply2h = |r: &[Complex64], out: &mut [Complex64]| {
        let mut acc = t_cell.borrow_mut();
        grad_masked_adj(r, &mask_of, &gwc, n, crop_box1, &mut acc);
        tr(&mut acc[..], false, &FftOpts { support: Some(crop_box1), ..Default::default() });
        let acc = &acc[..];
        crate::maybe_par_iter_mut!(out).zip(crate::maybe_par_iter!(cone)).for_each(|(o, &l)| *o = acc[l] * isq);
    };
    let mut b2 = vec![zero; 3 * nm];
    grad_masked(&|l| Complex64::new(x1[l] * m[l], 0.0), inside, &gwc, n, &mut b2);
    let (y, out) = lsqr_matlab_into(apply2, apply2h, b2, (&u_parts, &v_parts), tol, max_iter);
    drop(t_cell);
    let mut xsa = vec![zero; nt];
    for (&l, &z) in cone.iter().zip(&y) {
        xsa[l] = z;
    }
    drop(y);
    tr(&mut xsa, true, &FftOpts::default());
    (crate::maybe_par_iter!(xsa).map(|z| z.re).collect(), out)
}

/// STI's `FastQSM` on the padded grid, given `F φ` (unshifted k-space). Returns the masked
/// estimate, rescaled to the TKD (threshold 1/8) solution by least squares inside the mask.
fn sti_fastqsm(
    p_k: &[Complex64],
    m: &[f64],
    inside: &[usize],
    dk: &[f64],
    n: [usize; 3],
    fft: &dyn Fn(&mut [Complex64]),
    ifft: &dyn Fn(&mut [Complex64]),
) -> Vec<f64> {
    let nt = p_k.len();
    // kernel: ball3D(3) (centre excluded) / its count, closed along the 2nd axis with [1 1 1]
    // (out-of-array = −∞ for the dilation, +∞ for the erosion), not renormalised
    let r = 3i64;
    let s = 7usize;
    let mut ball = vec![0.0f64; s * s * s];
    let mut cnt = 0.0;
    for (l, bv) in ball.iter_mut().enumerate() {
        let (i, j, k) = ((l % s) as i64 - r, ((l / s) % s) as i64 - r, (l / (s * s)) as i64 - r);
        let d2 = (i * i + j * j + k * k) as f64;
        if d2 > 0.0 && d2 <= (r as f64 + 1.0 / 6.0).powi(2) {
            *bv = 1.0;
            cnt += 1.0;
        }
    }
    ball.iter_mut().for_each(|v| *v /= cnt);
    let line = |a: &[f64], dil: bool| -> Vec<f64> {
        let mut o = vec![0.0; a.len()];
        for (l, ov) in o.iter_mut().enumerate() {
            let j = (l / s) % s;
            let mut v = a[l];
            for jj in [j.wrapping_sub(1), j + 1] {
                if jj < s {
                    let q = a[l - j * s + jj * s];
                    v = if dil { v.max(q) } else { v.min(q) };
                }
            }
            *ov = v;
        }
        o
    };
    let closed = line(&line(&ball, true), false);
    // STI smooths the k-space of the *centred* image (fftnc: origin at the array centre); on
    // the k-space of the uncentred image that is the same convolution with the kernel
    // modulated by (−1)^(i+j+k) (all padded dimensions are even).
    let mut kern = vec![Complex64::new(0.0, 0.0); nt];
    for (l, &bv) in closed.iter().enumerate() {
        if bv != 0.0 {
            let off = [(l % s) as i64 - r, ((l / s) % s) as i64 - r, (l / (s * s)) as i64 - r];
            let idx = [0, 1, 2].map(|d| off[d].rem_euclid(n[d] as i64) as usize);
            let sign = if (off[0] + off[1] + off[2]).rem_euclid(2) == 0 { 1.0 } else { -1.0 };
            kern[idx[0] + idx[1] * n[0] + idx[2] * n[0] * n[1]] = Complex64::new(sign * bv, 0.0);
        }
    }
    fft(&mut kern[..]);
    let kh: Vec<f64> = kern.iter().map(|z| z.re).collect();
    drop(kern);
    // k-space blending weights from |D|^0.001, ranked over the whole grid (1st..30th pct)
    let mut wf: Vec<f64> = crate::maybe_par_iter!(dk).map(|&dv| dv.abs().powf(0.001)).collect();
    let mut zs = wf.clone();
    let (zlo, zhi) = sti_percentiles(&mut zs, 0.01, 0.3);
    drop(zs);
    crate::maybe_par_iter_mut!(wf).for_each(|v| *v = if zhi > zlo { ((*v - zlo) / (zhi - zlo)).clamp(0.0, 1.0) } else { 1.0 });
    // blend X with its k-space smoothing (a convolution over k-space indices)
    let blend = |x: &mut [Complex64]| {
        let mut sm = x.to_vec();
        fft(&mut sm[..]);
        crate::maybe_par_iter_mut!(sm).zip(crate::maybe_par_iter!(kh)).for_each(|(v, &h)| *v *= h);
        ifft(&mut sm[..]);
        crate::maybe_par_iter_mut!(x).zip(crate::maybe_par_iter!(sm)).zip(crate::maybe_par_iter!(wf))
            .for_each(|((xv, &sv), &w)| *xv = *xv * w + sv * (1.0 - w));
    };
    let mut x: Vec<Complex64> = crate::maybe_par_iter!(p_k).zip(crate::maybe_par_iter!(dk)).map(|(&pv, &dv)| pv * sign0(dv)).collect();
    blend(&mut x[..]);
    ifft(&mut x[..]);
    crate::maybe_par_iter_mut!(x).zip(crate::maybe_par_iter!(m)).for_each(|(v, &mv)| *v *= mv);
    fft(&mut x[..]);
    blend(&mut x[..]);
    ifft(&mut x[..]);
    let mut xfs: Vec<f64> = crate::maybe_par_iter!(x).zip(crate::maybe_par_iter!(m)).map(|(v, &mv)| v.re * mv).collect();
    drop(x);
    // TKD: 1/D clamped to ±8 (D = 0 → +8)
    let mut t: Vec<Complex64> = crate::maybe_par_iter!(p_k).zip(crate::maybe_par_iter!(dk)).map(|(&pv, &dv)| {
        let r = 1.0 / dv;
        pv * if r.abs() > 8.0 { 8.0f64.copysign(r) } else { r }
    }).collect();
    ifft(&mut t[..]);
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for &l in inside {
        sxy += xfs[l] * t[l].re;
        sxx += xfs[l] * xfs[l];
    }
    let a = if sxx > 0.0 { sxy / sxx } else { 1.0 };
    crate::maybe_par_iter_mut!(xfs).for_each(|v| *v *= a);
    xfs
}

fn sign0(v: f64) -> f64 {
    if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsqr_simple() {
        // Test LSQR on a simple diagonal system
        let n = 10;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let b: Vec<f64> = diag.iter().map(|&d| d * 2.0).collect();  // x = [2, 2, 2, ...]

        let apply_a = |x: &[f64]| -> Vec<f64> {
            x.iter().zip(diag.iter()).map(|(&xi, &di)| xi * di).collect()
        };

        let x = lsqr(apply_a, apply_a, &b, 1e-10, 100);

        for (i, &xi) in x.iter().enumerate() {
            assert!((xi - 2.0).abs() < 1e-6, "x[{}] = {}, expected 2.0", i, xi);
        }
    }

    // ---------------- STI Suite iLSQR ----------------

    fn c64(re: f64, im: f64) -> Complex64 { Complex64::new(re, im) }

    type Op = Box<dyn Fn(&[Complex64]) -> Vec<Complex64>>;

    /// Dense column-major matrix-vector products (A x, Aᴴ y) for the LSQR tests.
    fn dense(a: Vec<Complex64>, m: usize, n: usize) -> (Op, Op) {
        let a2 = a.clone();
        let fwd = move |x: &[Complex64]| (0..m).map(|i| (0..n).map(|j| a[i + j * m] * x[j]).sum()).collect();
        let adj = move |y: &[Complex64]| (0..n).map(|j| (0..m).map(|i| a2[i + j * m].conj() * y[i]).sum()).collect();
        (Box::new(fwd), Box::new(adj))
    }

    #[test]
    fn test_lsqr_matlab_matches_matlab_lsqr() {
        // Reference values from MATLAB R2023b `[x,flag,relres,iter] = lsqr(A, b, tol, maxit)`.
        // Inconsistent 3x2 system: stops on the normal-equations test after 2 iterations.
        let a: Vec<Complex64> = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0].iter().map(|&v| c64(v, 0.0)).collect();
        let (f, g) = dense(a, 3, 2);
        let b = [c64(1.0, 0.0), c64(2.0, 0.0), c64(4.0, 0.0)];
        let (x, o) = lsqr_matlab(f, g, &b, 1e-8, 10);
        assert_eq!((o.flag, o.iter), (0, 2));
        assert!((o.relres - 0.0890870806374748).abs() < 1e-12);
        assert!((x[0].re - 0.666666666666665).abs() < 1e-12 && (x[1].re - 0.083333333333331).abs() < 1e-12);

        // Ill-conditioned diagonal, iteration cap reached: x after exactly 5 iterations.
        let d: Vec<f64> = (0..8).map(|i| 10f64.powf(-3.0 * i as f64 / 7.0)).collect(); // logspace(0, -3, 8)
        let mut a = vec![c64(0.0, 0.0); 64];
        for i in 0..8 { a[i + 8 * i] = c64(d[i], 0.0); }
        let (f, g) = dense(a, 8, 8);
        let b = vec![c64(1.0, 0.0); 8];
        let (x, o) = lsqr_matlab(f, g, &b, 1e-10, 5);
        assert_eq!((o.flag, o.iter), (1, 5));
        assert!((o.relres - 0.573674508817809).abs() < 1e-12);
        let want = [0.999999943834766, 2.68269578689076, 7.1969177940897, 19.2460743622976,
                    59.9361124086567, 25.5070666956926, 9.67511787758301, 3.61516968204123];
        // (x(1) moves by ~1e-9 for a 1-ulp change in A here, so x is only checked to 1e-6)
        for (xi, wi) in x.iter().zip(want) { assert!((xi.re - wi).abs() < 1e-6 * wi, "{} vs {}", xi.re, wi); }

        // Complex 3x2 system.
        let a = vec![c64(1.0, 0.0), c64(2.0, 0.0), c64(0.0, 0.5), c64(0.0, 1.0), c64(-1.0, 0.0), c64(3.0, 0.0)];
        let (f, g) = dense(a, 3, 2);
        let b = [c64(1.0, 0.0), c64(0.0, 1.0), c64(2.0, 0.0)];
        let (x, o) = lsqr_matlab(f, g, &b, 1e-12, 20);
        assert_eq!((o.flag, o.iter), (0, 2));
        assert!((o.relres - 0.449991346403468).abs() < 1e-12);
        assert!((x[0] - c64(0.448598130841121, 0.186915887850467)).norm() < 1e-12);
        assert!((x[1] - c64(0.635514018691589, -0.168224299065421)).norm() < 1e-12);
    }

    #[test]
    fn test_lsqr_matlab_zero_rhs() {
        let (x, o) = lsqr_matlab(|v: &[Complex64]| v.to_vec(), |v: &[Complex64]| v.to_vec(), &[c64(0.0, 0.0); 4], 1e-6, 10);
        assert_eq!((o.flag, o.iter), (0, 0));
        assert!(x.iter().all(|z| z.norm() == 0.0));
    }

    #[test]
    fn test_sti_bounding_box() {
        // STI MaskBoundingBox outputs (1-based) from probing, here 0-based.
        type Case = ((usize, usize, usize), [usize; 3], [usize; 3], [[usize; 2]; 3]);
        let cases: [Case; 5] = [
            ((20, 20, 20), [4, 5, 6], [10, 11, 12], [[4, 11], [5, 12], [6, 13]]),
            ((20, 20, 20), [1, 1, 1], [5, 6, 7], [[1, 6], [1, 6], [1, 8]]),
            ((20, 20, 20), [14, 15, 16], [20, 20, 20], [[13, 20], [15, 20], [15, 20]]),
            ((21, 21, 21), [1, 2, 3], [21, 21, 21], [[1, 20], [2, 21], [2, 21]]),
            ((20, 20, 20), [7, 7, 7], [7, 8, 7], [[7, 8], [7, 8], [7, 8]]),
        ];
        for (dims, lo1, hi1, want) in cases {
            let mut m = vec![0u8; dims.0 * dims.1 * dims.2];
            for k in lo1[2] - 1..hi1[2] { for j in lo1[1] - 1..hi1[1] { for i in lo1[0] - 1..hi1[0] {
                m[i + j * dims.0 + k * dims.0 * dims.1] = 1;
            } } }
            let (lo, hi) = sti_bounding_box(&m, dims).unwrap();
            for d in 0..3 { assert_eq!([lo[d] + 1, hi[d] + 1], want[d], "dims {:?} box {:?}..{:?}", dims, lo1, hi1); }
        }
        assert!(sti_bounding_box(&[0u8; 8], (2, 2, 2)).is_none());
    }

    #[test]
    fn test_sti_padded_dims() {
        // (crop, voxel size, padsize) -> STI's padded size, from probing QSM_iLSQR
        type Case = ([usize; 3], [f64; 3], [f64; 3], [usize; 3]);
        let cases: [Case; 6] = [
            ([32, 30, 8], [0.8, 0.8, 3.0], [64.0; 3], [208, 192, 64]),
            ([48, 44, 12], [0.8, 0.8, 3.0], [0.0; 3], [64, 48, 16]),
            ([32, 34, 32], [1.3; 3], [10.0; 3], [64, 64, 64]),
            ([32, 30, 18], [1.3, 0.7, 1.1], [10.0, 3.0, 7.0], [64, 48, 32]),
            ([40, 42, 10], [1.05, 2.2, 3.3], [20.0; 3], [80, 64, 32]),
            ([46, 42, 10], [0.5, 0.5, 2.0], [64.0; 3], [304, 304, 80]),
        ];
        for (c, vs, pad, want) in cases { assert_eq!(sti_padded_dims(c, vs, pad), want); }
    }

    #[test]
    fn test_sti_percentile_is_rounded_index() {
        let v: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        assert_eq!(sti_percentile(&v, 0.25), 3.0); // round(2.5) = 3, no interpolation
        assert_eq!(sti_percentile(&v, 0.999), 10.0);
        assert_eq!(sti_percentile(&v, 0.0), 1.0);
    }

    /// `grad_masked` is the weighted circular forward difference on the mask voxels, and
    /// `grad_masked_adj` its adjoint.
    #[test]
    fn test_grad_masked_and_adjoint() {
        let n = [6, 5, 4];
        let nt = 120;
        let inside: Vec<usize> = (0..nt).filter(|&l| (l * 7) % 5 != 0).collect();
        let nm = inside.len();
        let mut mask_of = vec![NONE; nt];
        for (i, &l) in inside.iter().enumerate() { mask_of[l] = i as u32; }
        let u: Vec<Complex64> = (0..nt).map(|i| c64((i as f64 * 0.37).sin(), (i as f64 * 0.11).cos())).collect();
        let r: Vec<Complex64> = (0..3 * nm).map(|i| c64((i as f64 * 0.23).cos(), (i as f64 * 0.71).sin())).collect();
        let gw: Vec<Vec<f64>> = (0..3).map(|a| (0..nm).map(|i| ((i * 7 + a * 3) % 11) as f64 / 10.0).collect()).collect();
        let mut g = vec![c64(0.0, 0.0); 3 * nm];
        grad_masked(&|l| u[l], &inside, &gw, n, &mut g);
        for a in 0..3 {
            let (s, len) = ([1, 6, 30][a], n[a]);
            for (i, &l) in inside.iter().enumerate() {
                let c = (l / s) % len;
                let nb = if c + 1 == len { l - (len - 1) * s } else { l + s };
                assert_eq!(g[a * nm + i], (u[nb] - u[l]) * gw[a][i]);
            }
        }
        let mut h = vec![c64(9.0, 9.0); nt];
        grad_masked_adj(&r, &mask_of, &gw, n, &[0..6, 0..5, 0..4], &mut h);
        let lhs: Complex64 = g.iter().zip(&r).map(|(a, b)| a.conj() * b).sum();
        let rhs: Complex64 = u.iter().zip(&h).map(|(a, b)| a.conj() * b).sum();
        assert!((lhs - rhs).norm() < 1e-12);
    }

    /// LSQR on vectors stored compactly (`compact_parts`) gives exactly the iterates of the
    /// dense solve whose vectors are zero off the support.
    #[test]
    fn test_lsqr_compact_equals_dense() {
        let nt = 3 * NORM_CHUNK + 1234;
        let supp: Vec<usize> = (0..nt).filter(|&l| (l / 1000) % 3 != 1 && l % 7 != 3).collect();
        let w: Vec<f64> = (0..nt).map(|l| 0.5 + ((l * 13) % 17) as f64 / 7.0).collect();
        let b: Vec<Complex64> = (0..nt).map(|l| if supp.binary_search(&l).is_ok() { c64((l as f64 * 0.01).sin(), (l as f64 * 0.003).cos()) } else { c64(0.0, 0.0) }).collect();
        // diagonal operator restricted to the support
        let dense = |v: &[Complex64]| -> Vec<Complex64> { v.iter().zip(&w).map(|(&z, &wv)| z * wv).collect() };
        let (xd, od) = lsqr_matlab(dense, dense, &b, 1e-14, 15);
        let wc: Vec<f64> = supp.iter().map(|&l| w[l]).collect();
        let bc: Vec<Complex64> = supp.iter().map(|&l| b[l]).collect();
        let parts = compact_parts(supp.iter().copied());
        let op = |v: &[Complex64], out: &mut [Complex64]| out.iter_mut().zip(v).zip(&wc).for_each(|((o, &z), &wv)| *o = z * wv);
        let (xc, oc) = lsqr_matlab_into(op, op, bc, (&parts, &parts), 1e-14, 15);
        assert_eq!(od, oc);
        for (i, &l) in supp.iter().enumerate() { assert_eq!(xd[l], xc[i]); }
    }

    /// Sphere of susceptibility forward-modelled through the dipole kernel on a 2x grid.
    fn sphere_phantom(n: usize, vs: (f64, f64, f64), bdir: (f64, f64, f64)) -> (Vec<f64>, Vec<u8>, Vec<f64>) {
        let nt = n * n * n;
        let c = (n as f64 - 1.0) / 2.0;
        let mut chi = vec![0.0; nt];
        let mut mask = vec![0u8; nt];
        for k in 0..n { for j in 0..n { for i in 0..n {
            let (x, y, z) = (i as f64 - c, j as f64 - c, k as f64 - c);
            let r2 = x * x + y * y + z * z;
            let l = i + j * n + k * n * n;
            if r2 < (0.4 * n as f64).powi(2) { mask[l] = 1; }
            if (x - 2.0).powi(2) + y * y + z * z < 9.0 { chi[l] = 0.1; }
            if (x + 3.0).powi(2) + (y - 1.0).powi(2) + z * z < 4.0 { chi[l] = -0.05; }
        } } }
        let m2 = 2 * n;
        let g2 = Grid::new(m2, m2, m2, vs.0, vs.1, vs.2);
        let d = dipole_kernel(&g2, bdir);
        let mut big = vec![Complex64::new(0.0, 0.0); m2 * m2 * m2];
        for k in 0..n { for j in 0..n { for i in 0..n { big[i + j * m2 + k * m2 * m2].re = chi[i + j * n + k * n * n]; } } }
        let mut ws = Fft3dWorkspace::new(m2, m2, m2);
        ws.fft3d(&mut big);
        for (z, &dv) in big.iter_mut().zip(&d) { *z *= dv; }
        ws.ifft3d(&mut big);
        let mut field = vec![0.0; nt];
        for k in 0..n { for j in 0..n { for i in 0..n {
            let l = i + j * n + k * n * n;
            if mask[l] != 0 { field[l] = big[i + j * m2 + k * m2 * m2].re; }
        } } }
        (field, mask, chi)
    }

    fn corr_in(a: &[f64], b: &[f64], m: &[u8]) -> f64 {
        let idx: Vec<usize> = (0..a.len()).filter(|&i| m[i] != 0).collect();
        let n = idx.len() as f64;
        let (ma, mb) = (idx.iter().map(|&i| a[i]).sum::<f64>() / n, idx.iter().map(|&i| b[i]).sum::<f64>() / n);
        let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
        for &i in &idx { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma).powi(2); sbb += (b[i] - mb).powi(2); }
        sab / (saa * sbb).sqrt()
    }

    #[test]
    fn test_ilsqr_sti_recovers_spheres() {
        let n = 16;
        let bdir = (0.1, -0.2, 0.95f64.sqrt());
        let (field, mask, chi) = sphere_phantom(n, (1.0, 1.0, 1.0), bdir);
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let mut steps = Vec::new();
        let params = IlsqrParams { max_iter: 40, ..IlsqrParams::default() };
        let (x, xsa, xfs, x1) = ilsqr_with_padding(&field, &mask, &grid, bdir, &params, [4.0; 3], |s, t| steps.push((s, t)));
        assert_eq!(steps, vec![(1, 4), (2, 4), (3, 4), (4, 4)]);
        for v in [&x, &xsa, &xfs, &x1] {
            assert!(v.iter().all(|z| z.is_finite()));
            assert!(v.iter().zip(&mask).all(|(&z, &m)| m != 0 || z == 0.0), "output must be masked");
        }
        let (r, r1) = (corr_in(&x, &chi, &mask), corr_in(&x1, &chi, &mask));
        println!("corr with truth: chi {:.4}, initial LSQR {:.4}", r, r1);
        assert!(r > 0.9 && r1 > 0.8, "corr with truth {} / {}", r, r1);
        // chi = initial LSQR minus artefacts
        for i in 0..x.len() { assert!((x[i] - (x1[i] - xsa[i])).abs() < 1e-12); }
        // linear in the field (up to LSQR rounding: the artefact solve runs 100 unconverged steps)
        let f2: Vec<f64> = field.iter().map(|v| 3.0 * v).collect();
        let (y, _, _, _) = ilsqr_with_padding(&f2, &mask, &grid, bdir, &params, [4.0; 3], |_, _| {});
        let x3: Vec<f64> = x.iter().map(|v| 3.0 * v).collect();
        let worst = y.iter().zip(&x3).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        println!("scaling: max|chi(3f) - 3 chi(f)| = {:.2e}", worst);
        assert!(corr_in(&y, &x3, &mask) > 0.9999 && worst < 0.01 * 0.3, "not linear: {}", worst);
    }

    #[test]
    fn test_ilsqr_sti_empty_mask_and_zero_field() {
        let grid = Grid::new(8, 8, 8, 1.0, 1.0, 1.0);
        let (x, _, _, _) = ilsqr(&[1.0; 512], &[0u8; 512], &grid, (0.0, 0.0, 1.0), &IlsqrParams::default(), |_, _| {});
        assert!(x.iter().all(|&v| v == 0.0));
        let mut mask = vec![0u8; 512];
        for k in 2..6 { for j in 2..6 { for i in 2..6 { mask[i + 8 * j + 64 * k] = 1; } } }
        let (x, _, _, _) = ilsqr_with_padding(&[0.0; 512], &mask, &grid, (0.0, 0.0, 1.0), &IlsqrParams::default(), [4.0; 3], |_, _| {});
        assert!(x.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_ilsqr_params_defaults() {
        let p = IlsqrParams::default();
        assert_eq!((p.tol, p.max_iter), (1e-3, 100));
    }

}

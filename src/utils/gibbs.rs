//! Gibbs-ringing removal via local subvoxel shifts.
//!
//! Truncating k-space (finite acquisition matrix) produces oscillatory ringing near sharp edges.
//! For multi-echo data the ringing pattern differs per echo (image contrast changes with TE), so
//! it perturbs the mono-exponential decay and biases R2*/R2 maps — hence unringing is
//! recommended before relaxometry.
//!
//! Method (Kellner et al. 2016): resample the image at a set of subvoxel shifts using the Fourier
//! shift theorem; at each voxel pick the shift that minimises local total variation (the
//! "ring-free" sampling), then interpolate back to the grid. The 1-D correction runs along each
//! axis, each axis weighted where its ringing dominates (low spatial frequency in the other axes).
//!
//! # Which mode for which acquisition
//!
//! | [`GibbsMode`] | method | for |
//! |---|---|---|
//! | [`Bautista3d`](GibbsMode::Bautista3d) (**default**) | Bautista et al. 2021, MRtrix3 `mrdegibbs -dimensionality 3` | 3-D acquisitions (k-space truncated along all three axes) |
//! | [`Mrtrix2d`](GibbsMode::Mrtrix2d) | Kellner et al. 2016, MRtrix3 `mrdegibbs` (2-D) | 2-D multi-slice acquisitions (truncated in-plane only) |
//! | [`Dipy2d`](GibbsMode::Dipy2d) | Kellner et al. 2016, DIPY `gibbs_removal` | parity with DIPY, the reference QSM-CI runs |
//!
//! The default is the 3-D method because QSM's gradient-echo scans are typically 3-D. Kellner's
//! slice-wise method assumes the ringing runs only in-plane; on a 3-D acquisition it leaves the
//! ringing along the slice axis (measured below). On a 2-D multi-slice acquisition use
//! `Mrtrix2d` with `slice_axis` normal to the slices. The MRtrix modes are ported in
//! [`crate::utils::gibbs_mrtrix`]; DIPY's is in this module. Decided 2026-10-31
//! (astewartau/QSM.rs#327): MRtrix's implementation of Kellner's method for 2-D, Bautista's for
//! 3-D, DIPY's kept as a parity option.
//!
//! # Measured (2026-10-31, astewartau/QSM.rs#327)
//!
//! Phantoms: 96×96×64 voxels of 14 random rotated boxes and ellipsoids inside a head-like shell,
//! built on a 3× finer grid; ground truth its 3×3×3 voxel averages. A 3-D acquisition keeps the
//! central 96×96×64 of its k-space; a 2-D one averages each slice's 3 fine slices (the slice
//! profile) and keeps the central 96×96 in-plane. NRMSE to the ground truth (%), and the RMS
//! error over the flat voxels 2–8 voxels from any edge (pure ringing); mean of three phantoms
//! (seeds 0–2, no noise). The MRtrix rows were run with `mrdegibbs` itself (MRtrix3 `dev`
//! 9b0a2f1), which these ports reproduce to ~1e-16; the 3-D row has negatives set to 0, as here.
//!
//! | method | 3-D acq. NRMSE | 3-D acq. flat RMS | 2-D acq. NRMSE | 2-D acq. flat RMS |
//! |---|---|---|---|---|
//! | none | 2.81 | 0.0061 | 1.92 | 0.0044 |
//! | `Dipy2d` | 2.34 | 0.0037 | 1.61 | 0.0021 |
//! | `Mrtrix2d` | 2.05 | 0.0033 | **1.17** | **0.0009** |
//! | QSM.rs's former own 3-D generalisation (removed) | 2.03 | 0.0032 | 1.74 | 0.0022 |
//! | `Bautista3d` | **1.58** | **0.0013** | 1.37 | 0.0010 |
//!
//! - On 3-D-truncated data `Bautista3d` removes ~79 % of the flat-region ringing; the slice-wise
//!   methods remove ~40–45 %.
//! - On 2-D-truncated data `Mrtrix2d` is best, as expected.
//! - With noise (σ 0.01, seed 0) the order is the same (3-D acquisition NRMSE: none 3.28,
//!   `Dipy2d` 2.71, `Mrtrix2d` 2.49, `Bautista3d` 1.92).
//! - MRtrix's 2-D implementation beats DIPY's on both: it splits the slice into the two weighted
//!   parts before unringing each (Kellner's order), where DIPY unrings first and weights after.
//!
//! QSM.rs used to unring with its own 3-D generalisation of DIPY's steps (each axis unrung over
//! the whole volume, then combined with the 3-D cosine weights). It was not a published method,
//! and was no better than the slice-wise methods on 3-D data, so it was removed.
//!
//! # Sources and licences
//!
//! - Kellner, Dhital, Kiselev & Reisert, "Gibbs-ringing artifact removal based on local
//!   subvoxel-shifts", MRM 2016, 76, 1574–1581.
//! - Bautista, O'Muircheartaigh, Hajnal & Tournier, "Removal of Gibbs ringing artefacts for 3D
//!   acquisitions using subvoxel shifts", Proc. ISMRM 2021, 29, 3535.
//! - MRtrix3 `mrdegibbs` (`cpp/core/degibbs/unring{1d,2d,3d}.h`, `dev` 9b0a2f1), MPL-2.0:
//!   ported anew in Rust (no source copied), credited here and in `THIRD_PARTY_NOTICES.md`.
//!   Tournier, an author of the 3-D method, is an MRtrix3 author.
//! - DIPY 1.12.1 `dipy.denoise.gibbs.gibbs_removal`, BSD-3-Clause: ported anew in Rust.
//! - Kellner's own code (`unring`, bitbucket.org/reisert/unring) forbids reproduction or
//!   modification without the University Medical Center Freiburg's permission, so it is not
//!   ported; MRtrix3's 2-D implementation follows its structure (split before unringing, exact
//!   FFT-frequency cosines, a running TV over the complex line, linear interpolation).
//!
//! # `Dipy2d`
//!
//! DIPY's `gibbs_removal`, ported step for step: DIPY's 45 shifts in [0.02, 0.9], `n_points` 3,
//! each shift direction its own inverse transform and magnitude (`|ifft(c·exp(±ks))|`), the 2-D
//! combination `|ifft2(fftshift(C1)·G1 + fftshift(C0)·G0)|` with DIPY's `linspace(−π, π)` cosine
//! weights and edge rows. Its output is a magnitude, so it is never negative.
//!
//! **Agreement with DIPY 1.12.1** (verified QSM-CI `py-originals` image, QSMxT/QSM-CI#302;
//! measured 2026-10-31): NRMSE 2.6e-6 % on a 64×64×32 crop of a travelling head's five echoes
//! (`prisma-bridge-run1`; DIPY changes the input by 3.4 %), and to the fixture's float32 rounding
//! on the 24×24×16 phantom of `tests/gibbs_dipy_parity.rs`.

use crate::utils::gibbs_mrtrix::{gibbs_unring_2d, gibbs_unring_3d, MrtrixUnringParams};
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

const N_SHIFTS: usize = 45;
const PI: f64 = std::f64::consts::PI;

/// Local total variation along a 1-D line, minimum of the right- and left-side
/// TV over `n_points` neighbours, with periodic boundaries (matches DIPY
/// `_image_tv` reduced with `np.minimum`).
///
/// The neighbour-difference magnitudes are staged in `diff` (`diff[j] =
/// |row[j] - row[j+1 mod n]|`) so the sliding sums index it directly — the
/// interior needs no modulo at all, which matters in the per-shift hot loop.
fn tv_min_line(row: &[f64], n: usize, n_points: usize, out: &mut [f64], diff: &mut [f64]) {
    for j in 0..n - 1 {
        diff[j] = (row[j] - row[j + 1]).abs();
    }
    diff[n - 1] = (row[n - 1] - row[0]).abs();
    for i in 0..n {
        let mut ptv = 0.0;
        let mut ntv = 0.0;
        if i >= n_points && i + n_points <= n {
            // Interior: both windows lie within the array.
            for o in 0..n_points {
                ptv += diff[i + o];
                ntv += diff[i - 1 - o];
            }
        } else {
            for o in 0..n_points {
                ptv += diff[(i + o) % n];
                ntv += diff[(i + n - 1 - o) % n];
            }
        }
        out[i] = ptv.min(ntv);
    }
}

/// Which unringing [`gibbs_unring_volume`] and the functions built on it run. See the
/// [module docs](self) for which suits which acquisition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GibbsMode {
    /// Bautista et al.'s 3-D unringing (MRtrix3 `mrdegibbs -dimensionality 3`), for 3-D
    /// acquisitions, which truncate k-space along all three axes. The default: QSM's
    /// gradient-echo scans are typically 3-D. See [`gibbs_unring_3d`].
    #[default]
    Bautista3d,
    /// Kellner et al.'s 2-D unringing as MRtrix3's `mrdegibbs` implements it, for 2-D multi-slice
    /// acquisitions (k-space truncated in-plane only): each slice normal to `slice_axis`.
    /// See [`gibbs_unring_2d`].
    Mrtrix2d {
        /// The axis the slices are normal to (0, 1 or 2; MRtrix's default is 2).
        slice_axis: usize,
    },
    /// DIPY's `gibbs_removal` (Kellner et al.'s 2-D method as DIPY implements it): each slice
    /// normal to `slice_axis` is unrung along both of its axes, and the two results are combined
    /// in its 2-D Fourier domain with DIPY's cosine weights; the output is the magnitude of that
    /// combination. A parity option: DIPY is the reference QSM-CI runs.
    Dipy2d {
        /// The axis the slices are normal to (0, 1 or 2; DIPY's default is 2).
        slice_axis: usize,
    },
}


/// DIPY's TV window (`n_points`, its default).
const N_POINTS: usize = 3;

/// numpy's `linspace(start, stop, num)`: `i·step + start`, the last value exactly `stop`.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 1 {
        return vec![start];
    }
    let step = (stop - start) / (num - 1) as f64;
    let mut y: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    y[num - 1] = stop;
    y
}

/// Everything a line of length `n` needs: DIPY's 45 shifts `ssamp`, the shift factors
/// `exp(k·s)` for every shift (`k = fftfreq(n, 1/(2πi))`, computed as numpy does), and the FFT
/// plans.
struct LineCtx {
    n: usize,
    ssamp: Vec<f64>,
    /// `phases[s*n + m] = exp(i·θ)`, `θ = (m'·Im(1/(n·d)))·ssamp[s]` with `d = 1/(2πi)`.
    phases: Vec<Complex64>,
    fft: Arc<dyn Fft<f64>>,
    ifft: Arc<dyn Fft<f64>>,
}

impl LineCtx {
    fn new(n: usize, planner: &mut FftPlanner<f64>) -> Self {
        let ssamp = linspace(0.02, 0.9, N_SHIFTS);
        // numpy: d = 1/(2j·π) = −i/(2π); val = 1/(n·d) = i·(−1/(n·(−1/(2π)))) (Python complex
        // division: real parts 0); k[m] = m'·val with m' the fftfreq integers.
        let inv_2pi = 1.0 / (2.0 * PI);
        let b = n as f64 * -inv_2pi;
        let val_im = -1.0 / b;
        let mut phases = vec![Complex64::new(0.0, 0.0); ssamp.len() * n];
        for (s_i, &s) in ssamp.iter().enumerate() {
            for m in 0..n {
                let mi = if m < n.div_ceil(2) { m as f64 } else { m as f64 - n as f64 };
                let theta = (mi * val_im) * s;
                phases[s_i * n + m] = Complex64::new(theta.cos(), theta.sin());
            }
        }
        LineCtx { n, ssamp, phases, fft: planner.plan_fft_forward(n), ifft: planner.plan_fft_inverse(n) }
    }
}

/// Reusable per-thread scratch for [`unring_line`], sized to a line of length `n`.
struct LineBufs {
    c: Vec<Complex64>,
    buf: Vec<Complex64>,
    tvp: Vec<f64>,
    tvn: Vec<f64>,
    isp: Vec<f64>,
    isn: Vec<f64>,
    sp: Vec<f64>,
    sn: Vec<f64>,
    img: Vec<f64>,
    tvs: Vec<f64>,
    diff: Vec<f64>,
    scratch_f: Vec<Complex64>,
    scratch_i: Vec<Complex64>,
}

impl LineBufs {
    fn new(ctx: &LineCtx) -> Self {
        let (n, z) = (ctx.n, Complex64::new(0.0, 0.0));
        LineBufs {
            c: vec![z; n],
            buf: vec![z; n],
            tvp: vec![0.0; n],
            tvn: vec![0.0; n],
            isp: vec![0.0; n],
            isn: vec![0.0; n],
            sp: vec![0.0; n],
            sn: vec![0.0; n],
            img: vec![0.0; n],
            tvs: vec![0.0; n],
            diff: vec![0.0; n],
            scratch_f: vec![z; ctx.fft.get_inplace_scratch_len()],
            scratch_i: vec![z; ctx.ifft.get_inplace_scratch_len()],
        }
    }
}

/// DIPY's `_gibbs_removal_1d` on one line, in place: for each shift `s` in `ssamp`, the line
/// shifted by `+s` and by `−s` (`|ifft(c·exp(±k·s))|`, each its own inverse transform), and at
/// every voxel the shift of least local TV in each direction; the result interpolates the two.
fn unring_line(row: &mut [f64], b: &mut LineBufs, ctx: &LineCtx) {
    let n = ctx.n;
    let LineBufs { c, buf, tvp, tvn, isp, isn, sp, sn, img, tvs, diff, scratch_f, scratch_i } = b;
    for m in 0..n {
        c[m] = Complex64::new(row[m], 0.0);
    }
    ctx.fft.process_with_scratch(c, scratch_f);
    tv_min_line(row, n, N_POINTS, tvp, diff);
    tvn.copy_from_slice(tvp);
    isp.copy_from_slice(row);
    isn.copy_from_slice(row);
    sp.iter_mut().for_each(|x| *x = 0.0);
    sn.iter_mut().for_each(|x| *x = 0.0);
    let inv_n = 1.0 / n as f64;
    for (s_i, &s) in ctx.ssamp.iter().enumerate() {
        let ph = &ctx.phases[s_i * n..s_i * n + n];
        for (positive, (best_tv, best_img, best_s)) in [(true, (&mut *tvp, &mut *isp, &mut *sp)), (false, (&mut *tvn, &mut *isn, &mut *sn))] {
            for m in 0..n {
                buf[m] = if positive { c[m] * ph[m] } else { c[m] * ph[m].conj() };
            }
            ctx.ifft.process_with_scratch(buf, scratch_i);
            for m in 0..n {
                img[m] = (buf[m] * inv_n).norm();
            }
            tv_min_line(img, n, N_POINTS, tvs, diff);
            for i in 0..n {
                if best_tv[i] > tvs[i] {
                    best_img[i] = img[i];
                    best_s[i] = s;
                    best_tv[i] = tvs[i];
                }
            }
        }
    }
    for i in 0..n {
        let d = sp[i] + sn[i];
        if d != 0.0 {
            row[i] = (isp[i] - isn[i]) / d * sn[i] + isn[i];
        }
    }
}

#[cfg(test)]
fn fftfreq(n: usize) -> Vec<f64> {
    (0..n)
        .map(|m| {
            if m < n.div_ceil(2) {
                m as f64 / n as f64
            } else {
                (m as f64 - n as f64) / n as f64
            }
        })
        .collect()
}

/// Un-ring `vol` along a single axis (0=x, 1=y, 2=z) over the whole volume, returning a new
/// volume: DIPY's 1-D step, tested against DIPY's `_gibbs_removal_1d`.
#[cfg(test)]
fn unring_axis(vol: &[f64], (nx, ny, nz): (usize, usize, usize), axis: usize) -> Vec<f64> {
    let n = [nx, ny, nz][axis];
    if n < 2 {
        return vol.to_vec();
    }
    let mut planner = FftPlanner::<f64>::new();
    let ctx = LineCtx::new(n, &mut planner);
    let (stride, lines): (usize, Vec<usize>) = match axis {
        0 => (1, (0..ny * nz).map(|l| l * nx).collect()),
        1 => (nx, (0..nz).flat_map(|z| (0..nx).map(move |x| z * nx * ny + x)).collect()),
        _ => (nx * ny, (0..ny).flat_map(|y| (0..nx).map(move |x| y * nx + x)).collect()),
    };
    let mut mat = vec![0.0_f64; lines.len() * n];
    for (l, &start) in lines.iter().enumerate() {
        for (i, d) in mat[l * n..l * n + n].iter_mut().enumerate() {
            *d = vol[start + i * stride];
        }
    }
    #[cfg(feature = "parallel")]
    mat.par_chunks_mut(n).for_each_init(|| LineBufs::new(&ctx), |b, row| unring_line(row, b, &ctx));
    #[cfg(not(feature = "parallel"))]
    {
        let mut b = LineBufs::new(&ctx);
        for row in mat.chunks_mut(n) {
            unring_line(row, &mut b, &ctx);
        }
    }
    let mut out = vol.to_vec();
    for (l, &start) in lines.iter().enumerate() {
        for (i, &s) in mat[l * n..l * n + n].iter().enumerate() {
            out[start + i * stride] = s;
        }
    }
    out
}

/// DIPY's `_weights(shape)`: `G0`, `G1` (row-major `n0 × n1`) on the fft-shifted grid, from
/// `linspace(−π, π)` frequencies, with DIPY's edge rows and columns.
fn dipy_weights(n0: usize, n1: usize) -> (Vec<f64>, Vec<f64>) {
    let mut g0 = vec![0.0_f64; n0 * n1];
    let mut g1 = vec![0.0_f64; n0 * n1];
    let k0 = linspace(-PI, PI, n0);
    let k1 = linspace(-PI, PI, n1);
    for i in 1..n0.saturating_sub(1) {
        for j in 1..n1.saturating_sub(1) {
            let c0 = 1.0 + k0[i].cos();
            let c1 = 1.0 + k1[j].cos();
            g1[i * n1 + j] = c0 / (c0 + c1);
            g0[i * n1 + j] = c1 / (c0 + c1);
        }
    }
    if n0 >= 2 && n1 >= 1 {
        for i in 1..n0 - 1 {
            g1[i * n1] = 1.0;
            g1[i * n1 + n1 - 1] = 1.0;
        }
    }
    if n1 >= 2 && n0 >= 1 {
        for j in 1..n1 - 1 {
            g0[j] = 1.0;
            g0[(n0 - 1) * n1 + j] = 1.0;
        }
    }
    for &(i, j) in &[(0, 0), (n0 - 1, n1 - 1), (0, n1 - 1), (n0 - 1, 0)] {
        g1[i * n1 + j] = 0.5;
        g0[i * n1 + j] = 0.5;
    }
    (g0, g1)
}

/// Everything a slice of `n0 × n1` needs (DIPY's `_gibbs_removal_2d`).
struct SliceCtx {
    n0: usize,
    n1: usize,
    line0: LineCtx,
    line1: LineCtx,
    g0: Vec<f64>,
    g1: Vec<f64>,
}

/// In-place 2-D FFT of a row-major `n0 × n1` array with the line plans.
fn fft2_inplace(a: &mut [Complex64], n0: usize, n1: usize, along1: &Arc<dyn Fft<f64>>, along0: &Arc<dyn Fft<f64>>) {
    for row in a.chunks_mut(n1) {
        along1.process(row);
    }
    let mut col = vec![Complex64::new(0.0, 0.0); n0];
    for j in 0..n1 {
        for i in 0..n0 {
            col[i] = a[i * n1 + j];
        }
        along0.process(&mut col);
        for i in 0..n0 {
            a[i * n1 + j] = col[i];
        }
    }
}

/// DIPY's `_gibbs_removal_2d` on a row-major `n0 × n1` slice: unring along axis 1 (`img_c1`) and
/// axis 0 (`img_c0`), then `|ifft2(fftshift(fft2(img_c1))·G1 + fftshift(fft2(img_c0))·G0)|`.
fn unring_slice(img: &[f64], ctx: &SliceCtx) -> Vec<f64> {
    let (n0, n1) = (ctx.n0, ctx.n1);
    // Along axis 1: rows of length n1.
    let mut c1 = img.to_vec();
    let mut b1 = LineBufs::new(&ctx.line1);
    for row in c1.chunks_mut(n1) {
        unring_line(row, &mut b1, &ctx.line1);
    }
    // Along axis 0: columns of length n0.
    let mut c0 = img.to_vec();
    let mut b0 = LineBufs::new(&ctx.line0);
    let mut col = vec![0.0_f64; n0];
    for j in 0..n1 {
        for i in 0..n0 {
            col[i] = c0[i * n1 + j];
        }
        unring_line(&mut col, &mut b0, &ctx.line0);
        for i in 0..n0 {
            c0[i * n1 + j] = col[i];
        }
    }
    let mut s1: Vec<Complex64> = c1.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    let mut s0: Vec<Complex64> = c0.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    fft2_inplace(&mut s1, n0, n1, &ctx.line1.fft, &ctx.line0.fft);
    fft2_inplace(&mut s0, n0, n1, &ctx.line1.fft, &ctx.line0.fft);
    // fftshift (roll by n//2 on each axis), weight, sum.
    let (h0, h1) = (n0 / 2, n1 / 2);
    let mut sum = vec![Complex64::new(0.0, 0.0); n0 * n1];
    for i in 0..n0 {
        let si = (i + n0 - h0) % n0;
        for j in 0..n1 {
            let sj = (j + n1 - h1) % n1;
            let w = i * n1 + j;
            sum[w] = s1[si * n1 + sj] * ctx.g1[w] + s0[si * n1 + sj] * ctx.g0[w];
        }
    }
    fft2_inplace(&mut sum, n0, n1, &ctx.line1.ifft, &ctx.line0.ifft);
    let inv = 1.0 / (n0 * n1) as f64;
    sum.iter().map(|z| (z * inv).norm()).collect()
}

/// DIPY's `gibbs_removal(vol, slice_axis)` on one 3-D volume.
fn unring_slices(vol: &[f64], (nx, ny, nz): (usize, usize, usize), slice_axis: usize) -> Vec<f64> {
    // The slice's axes in order (DIPY moves the slice axis first and keeps the others' order).
    let (ns, n0, n1) = match slice_axis {
        0 => (nx, ny, nz),
        1 => (ny, nx, nz),
        _ => (nz, nx, ny),
    };
    let index = |s: usize, i: usize, j: usize| match slice_axis {
        0 => s + i * nx + j * nx * ny,
        1 => i + s * nx + j * nx * ny,
        _ => i + j * nx + s * nx * ny,
    };
    let mut planner = FftPlanner::<f64>::new();
    let (g0, g1) = dipy_weights(n0, n1);
    let ctx = SliceCtx { n0, n1, line0: LineCtx::new(n0, &mut planner), line1: LineCtx::new(n1, &mut planner), g0, g1 };
    let slice_out = |s: usize| -> Vec<f64> {
        let mut img = vec![0.0_f64; n0 * n1];
        for i in 0..n0 {
            for j in 0..n1 {
                img[i * n1 + j] = vol[index(s, i, j)];
            }
        }
        unring_slice(&img, &ctx)
    };
    #[cfg(feature = "parallel")]
    let slices: Vec<Vec<f64>> = (0..ns).into_par_iter().map(slice_out).collect();
    #[cfg(not(feature = "parallel"))]
    let slices: Vec<Vec<f64>> = (0..ns).map(slice_out).collect();
    let mut out = vec![0.0_f64; vol.len()];
    for (s, img) in slices.iter().enumerate() {
        for i in 0..n0 {
            for j in 0..n1 {
                out[index(s, i, j)] = img[i * n1 + j];
            }
        }
    }
    out
}

/// Gibbs-unring a single 3-D volume with `mode` ([`GibbsMode::default`] is Bautista et al.'s
/// 3-D unringing).
///
/// The output is never negative: DIPY's 2-D combination is a magnitude, and the MRtrix modes'
/// negative values are set to 0.
///
/// # Panics
/// If `vol` is not `nx·ny·nz` long or a slice axis is not 0, 1 or 2.
pub fn gibbs_unring_volume(vol: &[f64], dims: (usize, usize, usize), mode: GibbsMode) -> Vec<f64> {
    let (nx, ny, nz) = dims;
    assert_eq!(vol.len(), nx * ny * nz, "vol length must be nx*ny*nz");
    match mode {
        GibbsMode::Bautista3d => gibbs_unring_3d(vol, dims, MrtrixUnringParams::default()),
        GibbsMode::Mrtrix2d { slice_axis } => gibbs_unring_2d(vol, dims, slice_axis, MrtrixUnringParams::default()),
        GibbsMode::Dipy2d { slice_axis } => {
            assert!(slice_axis < 3, "slice_axis must be 0, 1 or 2");
            unring_slices(vol, dims, slice_axis)
        }
    }
}

/// Gibbs-unring multi-volume data (e.g. multi-echo magnitude), each volume in turn, as DIPY and
/// MRtrix do for a 4-D array.
///
/// # Arguments
/// * `data` - Interleaved `[voxel0_vol0, voxel0_vol1, ..., voxel1_vol0, ...]`
///   (row-major `(n_voxels, n_vols)`).
/// * `dims` - Volume dimensions `(nx, ny, nz)`.
/// * `n_vols` - Number of volumes.
/// * `mode` - See [`GibbsMode`]; the default is Bautista et al.'s 3-D unringing.
///
/// # Returns
/// Unrung data in the same layout (never negative).
pub fn gibbs_unring(data: &[f64], dims: (usize, usize, usize), n_vols: usize, mode: GibbsMode) -> Vec<f64> {
    let (nx, ny, nz) = dims;
    let n_vox = nx * ny * nz;
    assert_eq!(data.len(), n_vox * n_vols, "data length must be n_voxels * n_vols");
    let mut out = data.to_vec();
    let mut vol = vec![0.0_f64; n_vox];
    for e in 0..n_vols {
        for v in 0..n_vox {
            vol[v] = data[v * n_vols + e];
        }
        let un = gibbs_unring_volume(&vol, dims, mode);
        for v in 0..n_vox {
            out[v * n_vols + e] = un[v];
        }
    }
    out
}

/// Gibbs-unring multi-volume data within a mask bounding box (fast path).
///
/// Only the axis-aligned bounding box of `mask`, expanded by `margin` voxels, is
/// unrung; voxels outside are copied through unchanged. This skips air/background
/// lines and shrinks the per-line FFT length, which is much faster on data where
/// the region of interest fills a fraction of the FOV.
///
/// Because the FFT is taken over the cropped extent rather than the full FOV, the
/// result differs slightly from [`gibbs_unring`] near the crop border, but with a
/// sufficient `margin` the interior of the mask is essentially identical (the
/// ringing there is driven by edges inside the box). Intended for relaxometry
/// where downstream fitting is masked to the brain anyway.
///
/// # Arguments
/// * `data` - Interleaved `(n_voxels, n_vols)`, same layout as [`gibbs_unring`].
/// * `dims` - Volume dimensions `(nx, ny, nz)`.
/// * `n_vols` - Number of volumes.
/// * `mask` - Binary mask `[nx*ny*nz]`; the bounding box of `mask != 0` is unrung.
/// * `margin` - Voxels to expand the bounding box on every side.
/// * `mode` - See [`GibbsMode`].
pub fn gibbs_unring_masked(
    data: &[f64],
    dims: (usize, usize, usize),
    n_vols: usize,
    mask: &[u8],
    margin: usize,
    mode: GibbsMode,
) -> Vec<f64> {
    let (nx, ny, nz) = dims;
    let n_vox = nx * ny * nz;
    assert_eq!(data.len(), n_vox * n_vols, "data length must be n_voxels * n_vols");
    assert_eq!(mask.len(), n_vox, "mask length must be n_voxels");

    // Bounding box of the mask.
    let (mut x0, mut y0, mut z0) = (nx, ny, nz);
    let (mut x1, mut y1, mut z1) = (0usize, 0usize, 0usize);
    let mut any = false;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if mask[x + y * nx + z * nx * ny] != 0 {
                    any = true;
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    z0 = z0.min(z);
                    x1 = x1.max(x);
                    y1 = y1.max(y);
                    z1 = z1.max(z);
                }
            }
        }
    }
    if !any {
        return data.to_vec();
    }
    // Expand by margin, clamp to volume.
    x0 = x0.saturating_sub(margin);
    y0 = y0.saturating_sub(margin);
    z0 = z0.saturating_sub(margin);
    x1 = (x1 + margin + 1).min(nx);
    y1 = (y1 + margin + 1).min(ny);
    z1 = (z1 + margin + 1).min(nz);
    let (cx, cy, cz) = (x1 - x0, y1 - y0, z1 - z0);

    // If the crop is (almost) the whole volume, just do the full transform.
    if cx * cy * cz >= n_vox * 9 / 10 {
        return gibbs_unring(data, dims, n_vols, mode);
    }

    let n_crop = cx * cy * cz;
    let mut out = data.to_vec();
    let mut sub = vec![0.0_f64; n_crop];
    for e in 0..n_vols {
        // Extract the cropped sub-volume for this echo.
        for zz in 0..cz {
            for yy in 0..cy {
                for xx in 0..cx {
                    let full = (x0 + xx) + (y0 + yy) * nx + (z0 + zz) * nx * ny;
                    sub[xx + yy * cx + zz * cx * cy] = data[full * n_vols + e];
                }
            }
        }
        let un = gibbs_unring_volume(&sub, (cx, cy, cz), mode);
        // Write the corrected crop back.
        for zz in 0..cz {
            for yy in 0..cy {
                for xx in 0..cx {
                    let full = (x0 + xx) + (y0 + yy) * nx + (z0 + zz) * nx * ny;
                    out[full * n_vols + e] = un[xx + yy * cx + zz * cx * cy];
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::{fft3d_real, ifft3d_real};

    #[test]
    fn test_unring_reduces_edge_overshoot() {
        // 1-D step, band-limited by zeroing high k-space → Gibbs ringing. Unring
        // along x should reduce the overshoot just past the edge.
        let (nx, ny, nz) = (64usize, 3usize, 3usize);
        let mut clean = vec![0.0_f64; nx];
        for x in 0..nx {
            clean[x] = if x >= nx / 2 { 1.0 } else { 0.0 };
        }
        // Band-limit via FFT truncation.
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(nx);
        let ifft = planner.plan_fft_inverse(nx);
        let mut c: Vec<Complex64> = clean.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        fft.process(&mut c);
        let keep = nx / 4;
        for m in 0..nx {
            let fm = if m < nx / 2 { m } else { nx - m };
            if fm > keep {
                c[m] = Complex64::new(0.0, 0.0);
            }
        }
        ifft.process(&mut c);
        let rung: Vec<f64> = c.iter().map(|z| z.re / nx as f64).collect();

        // Broadcast the ringing line across a small volume.
        let mut vol = vec![0.0_f64; nx * ny * nz];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    vol[x + y * nx + z * nx * ny] = rung[x];
                }
            }
        }
        // Test the 1-D core directly (the 3-D Fourier combine assumes every axis
        // carries structure; a ring broadcast flat across y,z is a degenerate
        // case for it — the full 3-D path is validated on the phantom).
        let un = unring_axis(&vol, (nx, ny, nz), 0);
        // Extract the corrected centre line.
        let line: Vec<f64> = (0..nx).map(|x| un[x + nx + nx * ny]).collect();

        // Port check: values must match DIPY's `_gibbs_removal_1d` on the
        // identical ringing line (indices 30..=40).
        let dipy_ref = [
            0.06875, 0.25484, 0.67489, 1.06882, 1.02528, 0.96117, 0.98865, 1.02746, 1.00689,
            0.97820, 0.97820,
        ];
        for (k, &r) in dipy_ref.iter().enumerate() {
            let got = line[30 + k];
            assert!(
                (got - r).abs() < 5e-3,
                "index {}: Rust {:.5} vs DIPY {:.5}",
                30 + k,
                got,
                r
            );
        }
        // Total variation should drop like DIPY's (3.23 -> ~2.41).
        let tv = |a: &[f64]| a.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>();
        let tv_rung: f64 = rung.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
        assert!(tv(&line) < 0.85 * tv_rung, "unring should cut TV: {:.3} -> {:.3}", tv_rung, tv(&line));
    }

    /// Ringing around a block on a zero background undershoots below 0; the output never does,
    /// and the clamp leaves the block's interior and the far background untouched.
    #[test]
    fn test_unring_volume_is_never_negative() {
        let (nx, ny, nz) = (24usize, 24usize, 24usize);
        let n = nx * ny * nz;
        let mut block = vec![0.0_f64; n];
        for z in 8..16 {
            for y in 8..16 {
                for x in 8..16 {
                    block[x + y * nx + z * nx * ny] = 1.0;
                }
            }
        }
        // Band-limit in 3-D to make it ring.
        let mut spec = fft3d_real(&block, nx, ny, nz);
        let f = fftfreq(nx);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    if f[x].abs() > 0.2 || f[y].abs() > 0.2 || f[z].abs() > 0.2 {
                        spec[x + y * nx + z * nx * ny] = Complex64::new(0.0, 0.0);
                    }
                }
            }
        }
        let rung = ifft3d_real(&spec, nx, ny, nz);
        assert!(rung.iter().any(|&v| v < -1e-3), "the test input should undershoot");
        let un = gibbs_unring_volume(&rung, (nx, ny, nz), GibbsMode::default());
        assert!(un.iter().all(|&v| v >= 0.0), "unrung magnitude must not be negative");
        assert!(un.contains(&0.0));
        let c = 12 + 12 * nx + 12 * nx * ny;
        assert!(un[c] > 0.5, "block interior kept: {}", un[c]);
        for mode in [
            GibbsMode::Bautista3d,
            GibbsMode::Mrtrix2d { slice_axis: 2 },
            GibbsMode::Mrtrix2d { slice_axis: 0 },
            GibbsMode::Dipy2d { slice_axis: 2 },
            GibbsMode::Dipy2d { slice_axis: 0 },
        ] {
            let multi = gibbs_unring(&rung, (nx, ny, nz), 1, mode);
            assert!(multi.iter().all(|&v| v >= 0.0), "{mode:?}");
            assert!(multi[c] > 0.5, "{mode:?}: block interior kept: {}", multi[c]);
        }
    }

    /// Timing + numeric fingerprint on a synthetic volume (sphere + deterministic
    /// noise). Run manually with `--ignored --nocapture` to compare performance
    /// work against the baseline.
    #[test]
    #[ignore]
    fn bench_unring_volume() {
        let (nx, ny, nz) = (128usize, 128usize, 128usize);
        let mut vol = vec![0.0_f64; nx * ny * nz];
        let mut lcg: u64 = 42;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let dx = x as f64 - nx as f64 / 2.0;
                    let dy = y as f64 - ny as f64 / 2.0;
                    let dz = z as f64 - nz as f64 / 2.0;
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();
                    let base = if r < nx as f64 / 3.0 { 1.0 } else { 0.1 };
                    lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let noise = (lcg >> 33) as f64 / (1u64 << 31) as f64 * 0.02;
                    vol[x + y * nx + z * nx * ny] = base + noise;
                }
            }
        }
        let t = std::time::Instant::now();
        let out = gibbs_unring_volume(&vol, (nx, ny, nz), GibbsMode::default());
        let dt = t.elapsed();
        let sum: f64 = out.iter().sum();
        let sq: f64 = out.iter().map(|v| v * v).sum();
        let mx = out.iter().cloned().fold(f64::MIN, f64::max);
        println!(
            "gibbs_unring_volume 128^3: {:?}  sum {:.6}  sumsq {:.6}  max {:.6}",
            dt, sum, sq, mx
        );
    }

    #[test]
    fn test_fftfreq_matches_numpy() {
        // numpy fftfreq(4) = [0, .25, -.5, -.25]
        let f = fftfreq(4);
        assert!((f[0] - 0.0).abs() < 1e-12);
        assert!((f[1] - 0.25).abs() < 1e-12);
        assert!((f[2] + 0.5).abs() < 1e-12);
        assert!((f[3] + 0.25).abs() < 1e-12);
    }
}

//! MRtrix3's `mrdegibbs`, ported: Kellner et al.'s 2-D unringing for 2-D multi-slice acquisitions
//! ([`gibbs_unring_2d`], [`GibbsMode::Mrtrix2d`](crate::utils::gibbs::GibbsMode::Mrtrix2d)) and
//! Bautista et al.'s 3-D extension for 3-D acquisitions ([`gibbs_unring_3d`],
//! [`GibbsMode::Bautista3d`](crate::utils::gibbs::GibbsMode::Bautista3d), the default). Which mode
//! suits which acquisition, the measurements behind it, and the sources and licences are in
//! [`crate::utils::gibbs`].
//!
//! **Reference code:** MRtrix3 `dev` 9b0a2f1 (2026-09), `cpp/core/degibbs/unring1d.h`,
//! `unring2d.h`, `unring3d.h` and `cpp/cmd/mrdegibbs.cpp` (MPL-2.0; this is a new Rust
//! implementation of the same steps, credited here). The 3-D mode is not in a 3.0.x release.
//! Defaults ([`MrtrixUnringParams`]) are `mrdegibbs`'s: `nshifts` 20, `minW` 1, `maxW` 3.
//!
//! # 2-D (`Unring2D`, Kellner et al. 2016)
//!
//! Each slice (rows: the first in-plane axis, columns: the second; slices normal to z are
//! MRtrix's default `-axes 0,1`):
//! - its 2-D FFT is split into two parts with the weights `ck/(ck+cj)` and `cj/(ck+cj)`,
//!   `c = (1 + cos(2π·m/N))/2` at the exact FFT frequencies (both parts 0 where `ck+cj = 0`);
//! - each part is unrung by 1-D shifts along one axis (`Unring1D`): shifts `0, ±1, …, ±nshifts`
//!   in units of `1/(2·nshifts)` voxel, applied as `exp(iπ·shift·m/(N·nshifts))` on the positive
//!   bins and the conjugate on their mirrors, the even-length Nyquist bin zeroed in the shifted
//!   lines (kept in the unshifted one);
//! - per voxel, the shift of least oscillation: a running sum of `|ΔRe| + |ΔIm|` over neighbours
//!   `minW..=maxW` to the left and to the right (updated voxel by voxel as MRtrix updates it,
//!   left before right, the first shift winning ties), then linear interpolation to the voxel;
//! - the two parts are summed.
//!
//! # 3-D (`Unring3DFunctor`, Bautista et al. 2021)
//!
//! - The 3-D forward FFT of the input; per axis `i`, the filter
//!   `∏_{j≠i}(1+cos k_j) / Σ_m ∏_{j≠m}(1+cos k_j)`, `k_j = 2π·n_j/N_j` (0 where its denominator is
//!   0; the three sum to 1, Kellner's 2-D weights being the 2-D case), then the 3-D inverse FFT;
//! - along the axis, each line's FFT is shifted by `2·nshifts + 1` subvoxel shifts,
//!   `0, ±1/(2·nshifts + 1), …, ±nshifts/(2·nshifts + 1)` voxel, as `exp(2πi·n·s/N)` with the
//!   even-length Nyquist bin zeroed, each inverse transformed;
//! - per voxel, the shift of least oscillation: the sum of `|ΔRe| + |ΔIm|` between neighbours
//!   `minW..=maxW` to the left, and likewise to the right, the smaller of the two, the first
//!   shift winning ties;
//! - linear interpolation of the chosen shifted line back to the voxel
//!   (`a₁ − s·(a₁ − a₀)` for `s > 0`, else `a₁ + s·(a₁ − a₂)`), and the three axes' results
//!   summed.
//!
//! # Deviations
//!
//! [`gibbs_unring_2d`] and [`gibbs_unring_3d`] (real input) return the real part, as `mrdegibbs`
//! writes for a real input, with negative values set to 0 (a magnitude cannot be negative;
//! MRtrix keeps them, e.g. ringing in the background). The `_complex` functions are the
//! reference's output as is. Nothing else differs.
//!
//! # Agreement with `mrdegibbs` (MRtrix3 `dev` 9b0a2f1 built from source; measured 2026-10-31)
//!
//! - **2-D:** NRMSE 2.9e-16–3.7e-16, max |Δ| 1.6e-15, on the 96×96×64 phantoms of
//!   [`crate::utils::gibbs`] (values up to 1.85; slices normal to z and to y), and NRMSE
//!   3.4e-16–5.4e-16 on the four cases of `tests/gibbs_mrtrix2d_parity.rs` (24×24×16, and
//!   21×26×15 in slices normal to each axis).
//! - **3-D:** NRMSE 4.6e-16, max |Δ| 1.8e-15 on a 96×96×64 phantom, and NRMSE 4.5e-16 and 8.2e-16
//!   on the two small phantoms of `tests/gibbs_mrtrix3d_parity.rs` (24×24×16 and, for odd axes,
//!   21×26×15).

use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Parameters of [`gibbs_unring_2d`] and [`gibbs_unring_3d`], with `mrdegibbs`'s defaults.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MrtrixUnringParams {
    /// Subvoxel shifts on each side of 0 (`-nshifts`, default 20): `2·nshifts + 1` shifts in all.
    pub nshifts: usize,
    /// Nearest neighbour offset in the oscillation measure (`-minW`, default 1).
    pub min_w: usize,
    /// Farthest neighbour offset in the oscillation measure (`-maxW`, default 3).
    pub max_w: usize,
}

impl Default for MrtrixUnringParams {
    fn default() -> Self {
        Self { nshifts: 20, min_w: 1, max_w: 3 }
    }
}

/// MRtrix's `indexshift`: the signed Fourier index of bin `n` (`n > size/2` wraps negative).
fn indexshift(n: usize, size: usize) -> f64 {
    if n > size / 2 { n as f64 - size as f64 } else { n as f64 }
}

/// 3-D FFT of a complex volume in place (Fortran order, x fastest), unnormalised both ways, as
/// MRtrix's `Math::FFT` (FFTW) along axes 0, 1, 2.
fn fft3(data: &mut [Complex64], dims: [usize; 3], inverse: bool) {
    let mut planner = FftPlanner::<f64>::new();
    for axis in 0..3 {
        let n = dims[axis];
        if n < 2 {
            continue;
        }
        let plan = if inverse { planner.plan_fft_inverse(n) } else { planner.plan_fft_forward(n) };
        let (stride, starts) = line_starts(dims, axis);
        let mut mat = gather(data, n, stride, &starts);
        #[cfg(feature = "parallel")]
        mat.par_chunks_mut(n).for_each(|row| plan.process(row));
        #[cfg(not(feature = "parallel"))]
        mat.chunks_mut(n).for_each(|row| plan.process(row));
        scatter(&mat, data, n, stride, &starts, false);
    }
}

/// The stride along `axis` and the start index of every line along it.
fn line_starts(dims: [usize; 3], axis: usize) -> (usize, Vec<usize>) {
    let [nx, ny, nz] = dims;
    match axis {
        0 => (1, (0..ny * nz).map(|l| l * nx).collect()),
        1 => (nx, (0..nz).flat_map(|z| (0..nx).map(move |x| z * nx * ny + x)).collect()),
        _ => (nx * ny, (0..ny).flat_map(|y| (0..nx).map(move |x| y * nx + x)).collect()),
    }
}

fn gather(data: &[Complex64], n: usize, stride: usize, starts: &[usize]) -> Vec<Complex64> {
    let mut mat = vec![Complex64::new(0.0, 0.0); starts.len() * n];
    for (l, &s) in starts.iter().enumerate() {
        for i in 0..n {
            mat[l * n + i] = data[s + i * stride];
        }
    }
    mat
}

fn scatter(mat: &[Complex64], data: &mut [Complex64], n: usize, stride: usize, starts: &[usize], add: bool) {
    for (l, &s) in starts.iter().enumerate() {
        for i in 0..n {
            if add {
                data[s + i * stride] += mat[l * n + i];
            } else {
                data[s + i * stride] = mat[l * n + i];
            }
        }
    }
}

/// Everything a line of length `n` needs: the shifts, the shift factors and the FFT plans.
struct LineCtx {
    n: usize,
    shifts: Vec<f64>,
    /// `phases[f*n + m] = exp(2πi·indexshift(m)·shifts[f]/n)`, the Nyquist bin of an even `n` 0.
    phases: Vec<Complex64>,
    fft: Arc<dyn Fft<f64>>,
    ifft: Arc<dyn Fft<f64>>,
    params: MrtrixUnringParams,
}

impl LineCtx {
    fn new(n: usize, params: MrtrixUnringParams) -> Self {
        let nsh = params.nshifts;
        // MRtrix's zero-centred `shift_ind`: 0, then +j/(2·nsh+1), then −j/(2·nsh+1).
        let mut shifts = vec![0.0; 2 * nsh + 1];
        for j in 0..nsh {
            shifts[j + 1] = (j + 1) as f64 / (2.0 * nsh as f64 + 1.0);
            shifts[1 + nsh + j] = -shifts[j + 1];
        }
        let mut phases = vec![Complex64::new(0.0, 0.0); shifts.len() * n];
        for (f, &s) in shifts.iter().enumerate() {
            for m in 0..n {
                let arg = 2.0 * indexshift(m, n) * std::f64::consts::PI * s / n as f64;
                phases[f * n + m] = Complex64::new(0.0, arg).exp();
            }
            if n.is_multiple_of(2) {
                phases[f * n + n / 2] = Complex64::new(0.0, 0.0);
            }
        }
        let mut planner = FftPlanner::<f64>::new();
        LineCtx { n, shifts, phases, fft: planner.plan_fft_forward(n), ifft: planner.plan_fft_inverse(n), params }
    }

    /// MRtrix's `LineProcessor` on one line: returns the unrung line divided by `scale`.
    fn unring(&self, line: &[Complex64], scale: f64, shifted: &mut [Complex64], out: &mut [Complex64]) {
        let n = self.n;
        let nf = self.shifts.len();
        let mut spec = line.to_vec();
        self.fft.process(&mut spec);
        for f in 0..nf {
            let row = &mut shifted[f * n..f * n + n];
            for m in 0..n {
                row[m] = spec[m] * self.phases[f * n + m];
            }
            self.ifft.process(row);
        }
        let wrap = |i: isize| i.rem_euclid(n as isize) as usize;
        let (min_w, max_w) = (self.params.min_w as isize, self.params.max_w as isize);
        let d = |a: Complex64, b: Complex64| (a.re - b.re).abs() + (a.im - b.im).abs();
        for (p, o) in out.iter_mut().enumerate() {
            let pi = p as isize;
            let mut ind = 0;
            let mut opt = f64::MAX;
            for f in 0..nf {
                let row = &shifted[f * n..f * n + n];
                let (mut left, mut right) = (0.0, 0.0);
                for k in min_w..=max_w {
                    left += d(row[wrap(pi - k)], row[wrap(pi - k - 1)]);
                    right += d(row[wrap(pi + k)], row[wrap(pi + k + 1)]);
                }
                let tv = left.min(right);
                if tv < opt {
                    opt = tv;
                    ind = f;
                }
            }
            let s = self.shifts[ind];
            let row = &shifted[ind * n..ind * n + n];
            let (a0, a1, a2) = (row[wrap(pi - 1)], row[p], row[wrap(pi + 1)]);
            *o = if s > 0.0 { (a1 - (a1 - a0) * s) / scale } else { (a1 + (a1 - a2) * s) / scale };
        }
    }
}

/// `mrdegibbs -dimensionality 3` on one complex volume (Fortran order, x fastest): the
/// reference's output as is. See the [module docs](self).
///
/// # Panics
/// If `vol` is not `nx·ny·nz` long, or `min_w >= max_w` (as `mrdegibbs` refuses).
pub fn gibbs_unring_3d_complex(vol: &[Complex64], dims: (usize, usize, usize), params: MrtrixUnringParams) -> Vec<Complex64> {
    let dims = [dims.0, dims.1, dims.2];
    let n_vox = dims[0] * dims[1] * dims[2];
    assert_eq!(vol.len(), n_vox, "vol length must be nx*ny*nz");
    assert!(params.min_w < params.max_w, "min_w must be smaller than max_w");
    let mut ft = vol.to_vec();
    fft3(&mut ft, dims, false);
    let x: [Vec<f64>; 3] = std::array::from_fn(|d| {
        (0..dims[d]).map(|m| 1.0 + (2.0 * std::f64::consts::PI * indexshift(m, dims[d]) / dims[d] as f64).cos()).collect()
    });
    let scale_vol = n_vox as f64;
    let mut out = vec![Complex64::new(0.0, 0.0); n_vox];
    for axis in 0..3 {
        // filter for this axis, then back to the image domain
        let mut filt = ft.clone();
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let w = [x[1][j] * x[2][k], x[0][i] * x[2][k], x[0][i] * x[1][j]];
                    let den = w[0] + w[1] + w[2];
                    let idx = i + j * dims[0] + k * dims[0] * dims[1];
                    filt[idx] *= if den == 0.0 { 0.0 } else { w[axis] / den };
                }
            }
        }
        fft3(&mut filt, dims, true);
        // unring along the axis and add to the output
        let n = dims[axis];
        let ctx = LineCtx::new(n, params);
        let scale = scale_vol * n as f64;
        let (stride, starts) = line_starts(dims, axis);
        let src = gather(&filt, n, stride, &starts);
        let mut res = vec![Complex64::new(0.0, 0.0); src.len()];
        let nf = ctx.shifts.len();
        #[cfg(feature = "parallel")]
        res.par_chunks_mut(n).zip(src.par_chunks(n)).for_each_init(
            || vec![Complex64::new(0.0, 0.0); nf * n],
            |sh, (o, l)| ctx.unring(l, scale, sh, o),
        );
        #[cfg(not(feature = "parallel"))]
        {
            let mut sh = vec![Complex64::new(0.0, 0.0); nf * n];
            for (o, l) in res.chunks_mut(n).zip(src.chunks(n)) {
                ctx.unring(l, scale, &mut sh, o);
            }
        }
        scatter(&res, &mut out, n, stride, &starts, true);
    }
    out
}

/// 3-D Gibbs unringing (Bautista et al. 2021, MRtrix3's `mrdegibbs -dimensionality 3`) of one
/// real volume, e.g. a magnitude image from a 3-D acquisition: the real part of
/// [`gibbs_unring_3d_complex`], negative values set to 0.
///
/// For 2-D multi-slice acquisitions use Kellner's slice-wise method instead.
///
/// # Panics
/// As [`gibbs_unring_3d_complex`].
pub fn gibbs_unring_3d(vol: &[f64], dims: (usize, usize, usize), params: MrtrixUnringParams) -> Vec<f64> {
    let c: Vec<Complex64> = vol.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    gibbs_unring_3d_complex(&c, dims, params).iter().map(|v| v.re.max(0.0)).collect()
}

/// MRtrix's `Unring1D` on one line, in place: `data` is the line's spectrum (unnormalised FFT)
/// on input and the unrung line, scaled by `n` (the unnormalised inverse transform), on output.
///
/// Shifts `0, ±1, …, ±nsh` in units of `1/(2·nsh)` voxel, applied as `exp(iπ·shift·m/(n·nsh))`
/// on bins `m = 1…maxn` and the conjugate on their mirrors (the even-length Nyquist bin zeroed
/// in the shifted lines, kept in the unshifted one); the oscillation measure is a running sum
/// of `|ΔRe| + |ΔIm|` over the windows `minW..=maxW` left and right, updated per voxel as
/// MRtrix updates it.
fn unring1d_line(data: &mut [Complex64], ifft: &dyn Fft<f64>, p: MrtrixUnringParams, sh: &mut [Complex64], buf: &mut [Complex64]) {
    let n = data.len();
    let nsh = p.nshifts;
    let nf = 2 * nsh + 1;
    let (min_w, max_w) = (p.min_w as isize, p.max_w as isize);
    let shift = |j: usize| -> f64 {
        if j == 0 {
            0.0
        } else if j <= nsh {
            j as f64
        } else {
            -((j - nsh) as f64)
        }
    };
    let maxn = if n % 2 == 1 { (n - 1) / 2 } else { n / 2 - 1 };
    // the original line, then each shift
    buf.copy_from_slice(data);
    ifft.process(buf);
    sh[..n].copy_from_slice(buf);
    for j in 1..nf {
        let phi = std::f64::consts::PI * shift(j) / (n * nsh) as f64;
        let u = Complex64::new(phi.cos(), phi.sin());
        let mut e = Complex64::new(1.0, 0.0);
        buf[0] = data[0];
        if n % 2 == 0 {
            buf[n / 2] = Complex64::new(0.0, 0.0);
        }
        for l in 0..maxn {
            e = u * e;
            buf[l + 1] = e * data[l + 1];
            buf[n - 1 - l] = e.conj() * data[n - 1 - l];
        }
        ifft.process(buf);
        sh[j * n..j * n + n].copy_from_slice(buf);
    }
    let w = |i: isize| i.rem_euclid(n as isize) as usize;
    let at = |i: isize, j: usize| sh[j * n + w(i)];
    let mut tv1 = vec![0.0_f64; nf];
    let mut tv2 = vec![0.0_f64; nf];
    let ni = n as isize;
    for j in 0..nf {
        for t in min_w..=max_w {
            tv1[j] += (at(ni - t, j).re - at(ni - t - 1, j).re).abs();
            tv1[j] += (at(ni - t, j).im - at(ni - t - 1, j).im).abs();
            tv2[j] += (at(ni + t, j).re - at(ni + t + 1, j).re).abs();
            tv2[j] += (at(ni + t, j).im - at(ni + t + 1, j).im).abs();
        }
    }
    for l in 0..ni {
        let mut min_tv = f64::MAX;
        let mut min_idx = 0;
        for j in 0..nf {
            if tv1[j] < min_tv {
                min_tv = tv1[j];
                min_idx = j;
            }
            if tv2[j] < min_tv {
                min_tv = tv2[j];
                min_idx = j;
            }
            tv1[j] += (at(l - min_w + 1, j).re - at(l - min_w, j).re).abs();
            tv1[j] -= (at(l - max_w, j).re - at(l - (max_w + 1), j).re).abs();
            tv2[j] += (at(l + max_w + 1, j).re - at(l + (max_w + 2), j).re).abs();
            tv2[j] -= (at(l + min_w, j).re - at(l + (min_w + 1), j).re).abs();

            tv1[j] += (at(l - min_w + 1, j).im - at(l - min_w, j).im).abs();
            tv1[j] -= (at(l - max_w, j).im - at(l - (max_w + 1), j).im).abs();
            tv2[j] += (at(l + max_w + 1, j).im - at(l + (max_w + 2), j).im).abs();
            tv2[j] -= (at(l + min_w, j).im - at(l + (min_w + 1), j).im).abs();
        }
        let (a0, a1, a2) = (at(l - 1, min_idx), at(l, min_idx), at(l + 1, min_idx));
        let s = shift(min_idx) / (2.0 * nsh as f64);
        data[l as usize] = if s > 0.0 {
            Complex64::new(a1.re * (1.0 - s) + a0.re * s, a1.im * (1.0 - s) + a0.im * s)
        } else {
            Complex64::new(a1.re * (1.0 + s) - a2.re * s, a1.im * (1.0 + s) - a2.im * s)
        };
    }
}

/// FFT plans and scratch for one slice shape (`nr` rows along the first in-plane axis, `nc`
/// columns along the second).
struct SliceCtx {
    nr: usize,
    nc: usize,
    fwd_r: Arc<dyn Fft<f64>>,
    fwd_c: Arc<dyn Fft<f64>>,
    inv_r: Arc<dyn Fft<f64>>,
    inv_c: Arc<dyn Fft<f64>>,
    p: MrtrixUnringParams,
}

impl SliceCtx {
    fn new(nr: usize, nc: usize, p: MrtrixUnringParams) -> Self {
        let mut planner = FftPlanner::<f64>::new();
        SliceCtx {
            nr,
            nc,
            fwd_r: planner.plan_fft_forward(nr),
            fwd_c: planner.plan_fft_forward(nc),
            inv_r: planner.plan_fft_inverse(nr),
            inv_c: planner.plan_fft_inverse(nc),
            p,
        }
    }

    /// MRtrix's `Unring2D` on one slice, in place (`s[r + c·nr]`).
    fn unring(&self, s: &mut [Complex64]) {
        let (nr, nc) = (self.nr, self.nc);
        let mut row = vec![Complex64::new(0.0, 0.0); nc];
        // FFT along each row (over the columns), then along each column
        let along_row = |s: &mut [Complex64], f: &dyn Fft<f64>, row: &mut [Complex64]| {
            for r in 0..nr {
                for c in 0..nc {
                    row[c] = s[r + c * nr];
                }
                f.process(row);
                for c in 0..nc {
                    s[r + c * nr] = row[c];
                }
            }
        };
        let along_col = |s: &mut [Complex64], f: &dyn Fft<f64>| {
            for c in 0..nc {
                f.process(&mut s[c * nr..c * nr + nr]);
            }
        };
        along_row(s, &*self.fwd_c, &mut row);
        along_col(s, &*self.fwd_r);
        // split into the part unrung along the rows' axis (weight ck) and the other (cj)
        let mut s2 = vec![Complex64::new(0.0, 0.0); nr * nc];
        for k in 0..nc {
            let ck = (1.0 + (2.0 * std::f64::consts::PI * (k as f64 / nc as f64)).cos()) * 0.5;
            for j in 0..nr {
                let cj = (1.0 + (2.0 * std::f64::consts::PI * (j as f64 / nr as f64)).cos()) * 0.5;
                let i = j + k * nr;
                if ck + cj != 0.0 {
                    s2[i] = s[i] * cj / (ck + cj);
                    s[i] *= ck / (ck + cj);
                } else {
                    s[i] = Complex64::new(0.0, 0.0);
                    s2[i] = Complex64::new(0.0, 0.0);
                }
            }
        }
        along_row(s, &*self.inv_c, &mut row);
        along_col(&mut s2, &*self.inv_r);
        let nf = 2 * self.p.nshifts + 1;
        let mut sh = vec![Complex64::new(0.0, 0.0); nf * nr.max(nc)];
        let mut buf = vec![Complex64::new(0.0, 0.0); nr.max(nc)];
        // `s` is a spectrum along the columns' axis: unring each column
        for c in 0..nc {
            unring1d_line(&mut s[c * nr..c * nr + nr], &*self.inv_r, self.p, &mut sh[..nf * nr], &mut buf[..nr]);
        }
        // `s2` is a spectrum along the rows' axis: unring each row
        for r in 0..nr {
            for c in 0..nc {
                row[c] = s2[r + c * nr];
            }
            unring1d_line(&mut row, &*self.inv_c, self.p, &mut sh[..nf * nc], &mut buf[..nc]);
            for c in 0..nc {
                s2[r + c * nr] = row[c];
            }
        }
        let scale = (nr * nc) as f64;
        for (a, b) in s.iter_mut().zip(&s2) {
            *a = (*a + b) / scale;
        }
    }
}

/// `mrdegibbs` (2-D, Kellner et al. 2016, MRtrix3's default mode) on one complex volume (Fortran
/// order, x fastest): each slice normal to `slice_axis` is unrung in 2-D, its in-plane axes in
/// increasing order (`slice_axis` 2 is MRtrix's default `-axes 0,1`). The reference's output as
/// is. See the [module docs](self).
///
/// # Panics
/// If `vol` is not `nx·ny·nz` long, `slice_axis > 2`, or `min_w >= max_w`.
pub fn gibbs_unring_2d_complex(vol: &[Complex64], dims: (usize, usize, usize), slice_axis: usize, params: MrtrixUnringParams) -> Vec<Complex64> {
    let d = [dims.0, dims.1, dims.2];
    assert_eq!(vol.len(), d[0] * d[1] * d[2], "vol length must be nx*ny*nz");
    assert!(slice_axis < 3, "slice_axis must be 0, 1 or 2");
    assert!(params.min_w < params.max_w, "min_w must be smaller than max_w");
    let (a0, a1) = match slice_axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let st = [1, d[0], d[0] * d[1]];
    let (nr, nc, ns) = (d[a0], d[a1], d[slice_axis]);
    let ctx = SliceCtx::new(nr, nc, params);
    let idx = |r: usize, c: usize, z: usize| r * st[a0] + c * st[a1] + z * st[slice_axis];
    let mut slices = vec![Complex64::new(0.0, 0.0); nr * nc * ns];
    for z in 0..ns {
        for c in 0..nc {
            for r in 0..nr {
                slices[z * nr * nc + r + c * nr] = vol[idx(r, c, z)];
            }
        }
    }
    #[cfg(feature = "parallel")]
    slices.par_chunks_mut(nr * nc).for_each(|s| ctx.unring(s));
    #[cfg(not(feature = "parallel"))]
    slices.chunks_mut(nr * nc).for_each(|s| ctx.unring(s));
    let mut out = vec![Complex64::new(0.0, 0.0); vol.len()];
    for z in 0..ns {
        for c in 0..nc {
            for r in 0..nr {
                out[idx(r, c, z)] = slices[z * nr * nc + r + c * nr];
            }
        }
    }
    out
}

/// MRtrix3's 2-D `mrdegibbs` (Kellner et al. 2016) of one real volume, e.g. a magnitude image
/// from a 2-D multi-slice acquisition: the real part of [`gibbs_unring_2d_complex`], negative
/// values set to 0.
///
/// # Panics
/// As [`gibbs_unring_2d_complex`].
pub fn gibbs_unring_2d(vol: &[f64], dims: (usize, usize, usize), slice_axis: usize, params: MrtrixUnringParams) -> Vec<f64> {
    let c: Vec<Complex64> = vol.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    gibbs_unring_2d_complex(&c, dims, slice_axis, params).iter().map(|v| v.re.max(0.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A box truncated in k-space along all three axes rings along all three; the 3-D unringing
    /// takes out most of the ringing in the background.
    #[test]
    fn test_unring_3d_reduces_overshoot() {
        let (nx, ny, nz) = (24, 20, 16);
        let n = nx * ny * nz;
        let mut img = vec![0.0; n];
        for k in 4..12 {
            for j in 5..15 {
                for i in 6..18 {
                    img[i + j * nx + k * nx * ny] = 100.0;
                }
            }
        }
        let mut c: Vec<Complex64> = img.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        fft3(&mut c, [nx, ny, nz], false);
        let keep = |m: usize, len: usize| indexshift(m, len).abs() < len as f64 / 4.0;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if !(keep(i, nx) && keep(j, ny) && keep(k, nz)) {
                        c[i + j * nx + k * nx * ny] = Complex64::new(0.0, 0.0);
                    }
                }
            }
        }
        fft3(&mut c, [nx, ny, nz], true);
        let rung: Vec<f64> = c.iter().map(|v| v.re / n as f64).collect();
        let err = |a: &[f64]| a.iter().zip(&img).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt();
        let un = gibbs_unring_3d(&rung, (nx, ny, nz), MrtrixUnringParams::default());
        assert!(un.iter().all(|&v| v >= 0.0));
        // ringing in the background, 2+ voxels from the box
        let far = |a: &[f64]| {
            let mut s = 0.0;
            for k in 0..nz { for j in 0..ny { for i in 0..nx {
                let out = |x: usize, lo: usize, hi: usize| x + 2 <= lo || x >= hi + 2;
                if out(i, 6, 18) || out(j, 5, 15) || out(k, 4, 12) { s += a[i + j * nx + k * nx * ny].powi(2); }
            }}}
            s.sqrt()
        };
        assert!(far(&un) < 0.5 * far(&rung), "background ringing {} -> {}", far(&rung), far(&un));
        assert!(err(&un) < err(&rung), "error {} -> {}", err(&rung), err(&un));
    }

    /// Odd axes (no Nyquist bin) and a constant volume: the shifts change nothing.
    #[test]
    fn test_unring_3d_keeps_a_constant() {
        let dims = (7, 9, 5);
        let v = vec![3.0; 7 * 9 * 5];
        let out = gibbs_unring_3d(&v, dims, MrtrixUnringParams::default());
        assert!(out.iter().all(|&x| (x - 3.0).abs() < 1e-12), "{:?}", &out[..5]);
    }

    /// The 2-D mode on odd and even slices along every slice axis: a constant is unchanged.
    #[test]
    fn test_unring_2d_keeps_a_constant() {
        let dims = (7, 8, 5);
        let v = vec![3.0; 7 * 8 * 5];
        for axis in 0..3 {
            let out = gibbs_unring_2d(&v, dims, axis, MrtrixUnringParams::default());
            assert!(out.iter().all(|&x| (x - 3.0).abs() < 1e-12), "axis {axis}: {:?}", &out[..5]);
        }
    }
}

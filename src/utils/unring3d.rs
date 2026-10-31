//! 3-D Gibbs-ringing removal by subvoxel shifts for 3-D Fourier-encoded acquisitions
//! (Bautista, O'Muircheartaigh, Hajnal & Tournier, "Removal of Gibbs ringing artefacts for 3D
//! acquisitions using subvoxel shifts", Proc. ISMRM 2021, 29, 3535).
//!
//! Kellner et al.'s (2016) subvoxel-shift method unrings each 2-D slice: a slice-wise
//! acquisition truncates k-space only in-plane. A 3-D acquisition (e.g. QSM's 3-D GRE) truncates
//! it along all three axes, so the ringing runs along the slice axis too. Bautista et al.
//! extend the method to 3-D: for each axis the volume is Fourier-filtered to attenuate the high
//! frequencies of the two other axes, then unrung by 1-D subvoxel shifts along that axis, and
//! the three results are summed. The filter for axis `i` is
//! `∏_{j≠i}(1+cos k_j) / Σ_m ∏_{j≠m}(1+cos k_j)`, `k_j = 2π·n_j/N_j`, so the three filters sum
//! to 1 (Kellner's 2-D weights are the 2-D case).
//!
//! **Reference code.** The authors' implementation is MRtrix3's `mrdegibbs -dimensionality 3`
//! (J-Donald Tournier is an MRtrix3 author; `cpp/core/degibbs/unring3d.h`, MRtrix3 `dev`
//! 9b0a2f1, 2026-09; not in the 3.0.x releases), and this module ports it step for step:
//! - the 3-D forward FFT of the input; per axis, the filter above (0 where its denominator is
//!   0), then the 3-D inverse FFT;
//! - along the axis, each line's FFT is shifted by `2·nshifts + 1` subvoxel shifts,
//!   `0, ±1/(2·nshifts + 1), …, ±nshifts/(2·nshifts + 1)` voxel (`nshifts` 20), as
//!   `exp(2πi·n·s/N)` with the even-length Nyquist bin zeroed, each inverse transformed;
//! - per voxel, the shift of least oscillation: the sum of `|ΔRe| + |ΔIm|` between neighbours
//!   `minW..=maxW` (1..=3) to the left, and likewise to the right, the smaller of the two, the
//!   first shift winning ties;
//! - linear interpolation of the chosen shifted line back to the voxel
//!   (`a₁ − s·(a₁ − a₀)` for `s > 0`, else `a₁ + s·(a₁ − a₂)`), and the three axes' results
//!   summed.
//!
//! MRtrix3 is MPL-2.0; this is a port of its algorithm, written anew in Rust, and credited here.
//!
//! **Deviations.** [`gibbs_unring_3d`] (real input) returns the real part, as `mrdegibbs` writes
//! for a real input, with negative values set to 0 (a magnitude cannot be negative; MRtrix keeps
//! them, e.g. ringing in the background). [`gibbs_unring_3d_complex`] is the reference's output as
//! is. Nothing else differs.
//!
//! **Agreement with `mrdegibbs -dimensionality 3`** (MRtrix3 `dev` 9b0a2f1 built from source;
//! measured 2026-10-31): NRMSE 4.6e-16, max |Δ| 1.8e-15 on a 96×96×64 phantom (values up to 1.85),
//! and NRMSE 4.5e-16 and 8.2e-16 on the two small phantoms of `tests/gibbs_mrtrix3d_parity.rs`
//! (24×24×16 and, for odd axes, 21×26×15).
//!
//! **Why it exists** (astewartau/QSM.rs#327, measured 2026-10-31). Kellner's 2-D method is right
//! for 2-D multi-slice acquisitions; QSM's 3-D GRE truncates k-space along all three axes.
//! Phantoms: 96×96×64 voxels of 14 random rotated boxes and ellipsoids inside a head-like shell,
//! built on a 3× finer grid; ground truth its 3×3×3 voxel averages; a 3-D acquisition keeps the
//! central 96×96×64 of its k-space, a 2-D one averages each slice's 3 fine slices and keeps the
//! central 96×96 in-plane. NRMSE to the ground truth (%), and the RMS error over the flat voxels
//! 2–8 voxels from any edge (pure ringing), mean of three phantoms (seeds 0–2, no noise):
//!
//! | method | 3-D acq. NRMSE | 3-D acq. flat RMS | 2-D acq. NRMSE | 2-D acq. flat RMS |
//! |---|---|---|---|---|
//! | none | 2.81 | 0.0061 | 1.92 | 0.0044 |
//! | DIPY 2-D (QSM.rs `Slices`, #383) | 2.34 | 0.0037 | 1.61 | 0.0021 |
//! | MRtrix 2-D (Kellner's order) | 2.05 | 0.0033 | 1.17 | 0.0009 |
//! | QSM.rs's own 3-D generalisation (`Volume`, #383) | 2.03 | 0.0032 | 1.74 | 0.0022 |
//! | **this module (Bautista 3-D)** | **1.58** | **0.0013** | 1.37 | 0.0010 |
//!
//! On 3-D-truncated data this removes ~79 % of the flat-region ringing (DIPY's 2-D ~40 %); on
//! 2-D-truncated data the slice-wise method is best, as expected. With noise (σ 0.01, seed 0) the
//! order is the same (3-D acquisition NRMSE: none 3.28, DIPY 2.71, 3-D 1.92).

use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Parameters of [`gibbs_unring_3d`], with `mrdegibbs`'s defaults.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unring3dParams {
    /// Subvoxel shifts on each side of 0 (`-nshifts`, default 20): `2·nshifts + 1` shifts in all.
    pub nshifts: usize,
    /// Nearest neighbour offset in the oscillation measure (`-minW`, default 1).
    pub min_w: usize,
    /// Farthest neighbour offset in the oscillation measure (`-maxW`, default 3).
    pub max_w: usize,
}

impl Default for Unring3dParams {
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
    params: Unring3dParams,
}

impl LineCtx {
    fn new(n: usize, params: Unring3dParams) -> Self {
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
pub fn gibbs_unring_3d_complex(vol: &[Complex64], dims: (usize, usize, usize), params: Unring3dParams) -> Vec<Complex64> {
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
pub fn gibbs_unring_3d(vol: &[f64], dims: (usize, usize, usize), params: Unring3dParams) -> Vec<f64> {
    let c: Vec<Complex64> = vol.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    gibbs_unring_3d_complex(&c, dims, params).iter().map(|v| v.re.max(0.0)).collect()
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
        let un = gibbs_unring_3d(&rung, (nx, ny, nz), Unring3dParams::default());
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
        let out = gibbs_unring_3d(&v, dims, Unring3dParams::default());
        assert!(out.iter().all(|&x| (x - 3.0).abs() < 1e-12), "{:?}", &out[..5]);
    }
}

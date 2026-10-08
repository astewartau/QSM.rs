//! V-SHARP as implemented in STI Suite 3.0 (`V_SHARP.p`)
//!
//! STI Suite's V-SHARP is the background-field removal used by the UK Biobank QSM pipeline and
//! a frequent reference in QSM comparisons. It differs from the Wu et al. (2012) algorithm in
//! [`super::vsharp`] in ways that change the result substantially:
//!
//! 1. **No deconvolution.** The local field is `φ − S_r ∗ φ` with no division by `1 − Ŝ`, so
//!    the low spatial frequencies the SMV high-pass removes are not restored.
//! 2. **The background is estimated on a 1 mm isotropic grid.** The cropped, padded field and
//!    mask are linearly interpolated onto 1 mm points, the SMV radii are whole millimetres,
//!    and only the *background* (`S_r ∗ φ`) is interpolated back. It is subtracted from the
//!    native-resolution field, so the local field itself is never resampled.
//! 3. **Kernel.** A binary ball `{|x| ≤ r + 1/6} \ {0}` in 1 mm voxels: the centre voxel is
//!    excluded, and the `1/6` decides which lattice points on the shell are in.
//! 4. **Radii and mask.** Every voxel takes the largest radius (2..=`max_radius` mm) whose ball
//!    lies inside the mask; voxels where only 1 mm fits are dropped from the output mask.
//!
//! STI Suite ships only as protected P-code, so this was reverse-engineered by black-box
//! probing (call tracing, intermediate capture, and fitting each stage against STI's output).
//! It reproduces STI's local field and mask to single-precision rounding on synthetic inputs
//! over several voxel geometries, and on a UK Biobank SWI field (0.8×0.8×3 mm) to within one
//! mask voxel out of 755k (a tie at the 0.5 threshold, which STI evaluates in single precision).
//!
//! Details kept for parity, including ones that are arguably quirks:
//! - crop to the mask's bounding box, then zero-pad by a fixed 12 voxels per side whatever the
//!   voxel size (so for sub-millimetre voxels the circular SMV can wrap into the field);
//! - 1 mm sample points at `1..=ceil(n·vs)` mm, with native voxel `a` (1-based) at `a·vs` mm;
//! - interpolated mask `> 0.5`; odd 1 mm dimensions zero-padded to even before the FFTs;
//! - background initialised to the field (so the local field is zero wherever no ball fits);
//! - back-interpolated output mask `> 0.5`.
//!
//! One consequence worth knowing: on anisotropic grids, 1 mm points whose interpolated mask is just
//! above 0.5 lie up to half a native voxel outside the last brain voxel, so their phase is partly
//! blended with the zeros outside the mask. Balls that reach them pick up a small bias (≈1e-3 of a
//! linear background's range on a 0.8×0.8×3 mm test). STI does the same, so it is kept for parity.

use num_complex::Complex64;
use crate::Grid;
use crate::fft::{fft3d, ifft3d};

/// Fixed crop padding STI uses, in native voxels.
const STI_PAD: usize = 12;

/// Parameters for [`vsharp_sti`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct VsharpStiParams {
    /// Largest SMV radius in mm (STI's `smvsize`; default 12). Radii run 2..=max_radius in 1 mm steps.
    pub max_radius: usize,
}

impl Default for VsharpStiParams {
    fn default() -> Self {
        Self { max_radius: 12 }
    }
}

/// STI Suite-compatible V-SHARP background field removal.
///
/// `field` is the unwrapped total field (any linear unit: rad, Hz or ppm — the output is in the
/// same unit), `mask` the brain mask, both in Fortran order on `grid`.
///
/// Returns `(local_field, eroded_mask)` on the input grid.
pub fn vsharp_sti(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &VsharpStiParams,
    mut progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<u8>) {
    let (nx, ny, nz) = grid.dims;
    let n_total = nx * ny * nz;
    let vs = [grid.voxel_size.0, grid.voxel_size.1, grid.voxel_size.2];
    let mut local_out = vec![0.0; n_total];
    let mut mask_out = vec![0u8; n_total];

    // 1. crop to the mask's bounding box, pad by STI_PAD native voxels
    let Some((lo, hi)) = bounding_box(mask, grid.dims) else {
        return (local_out, mask_out);
    };
    let np = [hi[0] - lo[0] + 1 + 2 * STI_PAD, hi[1] - lo[1] + 1 + 2 * STI_PAD, hi[2] - lo[2] + 1 + 2 * STI_PAD];
    let mut p = vec![0.0; np[0] * np[1] * np[2]];
    let mut m = vec![0.0; p.len()];
    for k in lo[2]..=hi[2] {
        for j in lo[1]..=hi[1] {
            for i in lo[0]..=hi[0] {
                let src = i + j * nx + k * nx * ny;
                let dst = (i - lo[0] + STI_PAD) + (j - lo[1] + STI_PAD) * np[0] + (k - lo[2] + STI_PAD) * np[0] * np[1];
                p[dst] = field[src];
                m[dst] = if mask[src] != 0 { 1.0 } else { 0.0 };
            }
        }
    }

    // 2. onto 1 mm points: iso point q (1-based) sits at q mm, native voxel a (1-based) at a·vs mm
    let ni = [0, 1, 2].map(|d| (np[d] as f64 * vs[d]).ceil() as usize);
    let fwd = [0, 1, 2].map(|d| (1..=ni[d]).map(|q| q as f64 / vs[d]).collect::<Vec<f64>>());
    let pi = trilinear(&p, np, &fwd);
    let mi = trilinear(&m, np, &fwd);

    // 3. odd dimensions padded to even (zeros at the end), mask thresholded
    let ne = ni.map(|n| n + n % 2);
    let n_iso = ne[0] * ne[1] * ne[2];
    let mut pe = vec![0.0; n_iso];
    let mut me = vec![0.0; n_iso];
    for k in 0..ni[2] {
        for j in 0..ni[1] {
            for i in 0..ni[0] {
                let s = i + j * ni[0] + k * ni[0] * ni[1];
                let d = i + j * ne[0] + k * ne[0] * ne[1];
                pe[d] = pi[s];
                me[d] = if mi[s] > 0.5 { 1.0 } else { 0.0 };
            }
        }
    }

    // 4. per radius (ascending, so each voxel ends with the largest ball that fits)
    let to_c = |v: &[f64]| -> Vec<Complex64> { v.iter().map(|&x| Complex64::new(x, 0.0)).collect() };
    let mut p_fft = to_c(&pe);
    fft3d(&mut p_fft, ne[0], ne[1], ne[2]);
    let mut m_fft = to_c(&me);
    fft3d(&mut m_fft, ne[0], ne[1], ne[2]);
    let mut bg = pe.clone();
    let mut nm = vec![0.0; n_iso];
    let radii: Vec<usize> = (2..=params.max_radius.max(2)).collect();
    let mut buf = vec![Complex64::new(0.0, 0.0); n_iso];
    for (idx, &r) in radii.iter().enumerate() {
        progress(idx + 1, radii.len());
        let (mut k_fft, count) = sti_ball_fft(ne, r);
        // erosion: the ball (unnormalised) must lie entirely in the mask
        for i in 0..n_iso { buf[i] = m_fft[i] * k_fft[i]; }
        ifft3d(&mut buf, ne[0], ne[1], ne[2]);
        let fits: Vec<bool> = (0..n_iso).map(|i| me[i] > 0.0 && buf[i].re >= count - 0.5).collect();
        // background = mean over the ball
        for v in k_fft.iter_mut() { *v /= count; }
        for i in 0..n_iso { buf[i] = p_fft[i] * k_fft[i]; }
        ifft3d(&mut buf, ne[0], ne[1], ne[2]);
        for i in 0..n_iso {
            if fits[i] {
                bg[i] = buf[i].re;
                nm[i] = 1.0;
            }
        }
    }

    // 5. background and mask back onto the native (padded) grid; local = field − background
    let mut bg_c = vec![0.0; ni[0] * ni[1] * ni[2]];
    let mut nm_c = vec![0.0; bg_c.len()];
    for k in 0..ni[2] {
        for j in 0..ni[1] {
            for i in 0..ni[0] {
                let d = i + j * ni[0] + k * ni[0] * ni[1];
                let s = i + j * ne[0] + k * ne[0] * ne[1];
                bg_c[d] = bg[s];
                nm_c[d] = nm[s];
            }
        }
    }
    let back = [0, 1, 2].map(|d| (1..=np[d]).map(|a| a as f64 * vs[d]).collect::<Vec<f64>>());
    let bg_n = trilinear(&bg_c, ni, &back);
    let nm_n = trilinear(&nm_c, ni, &back);

    for k in lo[2]..=hi[2] {
        for j in lo[1]..=hi[1] {
            for i in lo[0]..=hi[0] {
                let s = (i - lo[0] + STI_PAD) + (j - lo[1] + STI_PAD) * np[0] + (k - lo[2] + STI_PAD) * np[0] * np[1];
                if nm_n[s] > 0.5 {
                    let d = i + j * nx + k * nx * ny;
                    local_out[d] = p[s] - bg_n[s];
                    mask_out[d] = 1;
                }
            }
        }
    }
    (local_out, mask_out)
}

/// Inclusive bounding box of the nonzero mask voxels, or `None` for an empty mask.
fn bounding_box(mask: &[u8], dims: (usize, usize, usize)) -> Option<([usize; 3], [usize; 3])> {
    let (nx, ny, nz) = dims;
    let mut lo = [usize::MAX; 3];
    let mut hi = [0usize; 3];
    let mut any = false;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if mask[i + j * nx + k * nx * ny] != 0 {
                    any = true;
                    for (d, v) in [i, j, k].into_iter().enumerate() {
                        lo[d] = lo[d].min(v);
                        hi[d] = hi[d].max(v);
                    }
                }
            }
        }
    }
    any.then_some((lo, hi))
}

/// MATLAB `interp3(..., 'linear')` on a separable grid: `coords[d]` are 1-based fractional source
/// indices along axis `d` (one output sample per entry). Samples outside `[1, n]` are 0 (MATLAB
/// returns NaN there and STI zeroes it).
fn trilinear(src: &[f64], n: [usize; 3], coords: &[Vec<f64>; 3]) -> Vec<f64> {
    let w = |d: usize| -> Vec<Option<(usize, f64)>> {
        coords[d].iter().map(|&u| {
            if !(1.0..=n[d] as f64).contains(&u) {
                return None;
            }
            let t = u - 1.0;
            let i0 = (t.floor() as usize).min(n[d].saturating_sub(2));
            Some((i0, t - i0 as f64))
        }).collect()
    };
    let (wx, wy, wz) = (w(0), w(1), w(2));
    let (ox, oy, oz) = (wx.len(), wy.len(), wz.len());
    let at = |i: usize, j: usize, k: usize| -> f64 {
        if i < n[0] && j < n[1] && k < n[2] { src[i + j * n[0] + k * n[0] * n[1]] } else { 0.0 }
    };
    let mut out = vec![0.0; ox * oy * oz];
    for (c, wzk) in wz.iter().enumerate() {
        let Some((k0, fz)) = *wzk else { continue };
        for (b, wyj) in wy.iter().enumerate() {
            let Some((j0, fy)) = *wyj else { continue };
            for (a, wxi) in wx.iter().enumerate() {
                let Some((i0, fx)) = *wxi else { continue };
                let mut v = 0.0;
                for (dk, gz) in [(0, 1.0 - fz), (1, fz)] {
                    if gz == 0.0 { continue; }
                    for (dj, gy) in [(0, 1.0 - fy), (1, fy)] {
                        if gy == 0.0 { continue; }
                        for (di, gx) in [(0, 1.0 - fx), (1, fx)] {
                            if gx == 0.0 { continue; }
                            v += gx * gy * gz * at(i0 + di, j0 + dj, k0 + dk);
                        }
                    }
                }
                out[a + b * ox + c * ox * oy] = v;
            }
        }
    }
    out
}

/// FFT of STI's ball of radius `r` (1 mm voxels) centred at the origin with wraparound, and its
/// voxel count. Membership: `|x|² ≤ (r + 1/6)²`, centre excluded — matches `ball3D.p` for r ≤ 30.
fn sti_ball_fft(n: [usize; 3], r: usize) -> (Vec<Complex64>, f64) {
    let mut k = vec![Complex64::new(0.0, 0.0); n[0] * n[1] * n[2]];
    let lim = (r as f64 + 1.0 / 6.0).powi(2);
    let ri = r as i64;
    let mut count = 0.0;
    for dz in -ri..=ri {
        for dy in -ri..=ri {
            for dx in -ri..=ri {
                let d2 = (dx * dx + dy * dy + dz * dz) as f64;
                if d2 == 0.0 || d2 > lim { continue; }
                let w = |d: i64, nd: usize| d.rem_euclid(nd as i64) as usize;
                k[w(dx, n[0]) + w(dy, n[1]) * n[0] + w(dz, n[2]) * n[0] * n[1]] = Complex64::new(1.0, 0.0);
                count += 1.0;
            }
        }
    }
    fft3d(&mut k, n[0], n[1], n[2]);
    (k, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(n: (usize, usize, usize), vs: (f64, f64, f64), r_mm: f64) -> Vec<u8> {
        let (nx, ny, nz) = n;
        let c = ((nx as f64 - 1.0) / 2.0, (ny as f64 - 1.0) / 2.0, (nz as f64 - 1.0) / 2.0);
        let mut m = vec![0u8; nx * ny * nz];
        for k in 0..nz { for j in 0..ny { for i in 0..nx {
            let d = (((i as f64 - c.0) * vs.0).powi(2) + ((j as f64 - c.1) * vs.1).powi(2) + ((k as f64 - c.2) * vs.2).powi(2)).sqrt();
            if d < r_mm { m[i + j * nx + k * nx * ny] = 1; }
        }}}
        m
    }

    #[test]
    fn ball_counts_match_sti_ball3d() {
        // sum(ball3D(r)) from STI Suite 3.0
        let sti = [(1, 6.0), (2, 32.0), (3, 146.0), (5, 586.0), (12, 7520.0), (14, 11836.0), (30, 115312.0)];
        for (r, want) in sti {
            assert_eq!(sti_ball_fft([64, 64, 64], r).1, want, "radius {r}");
        }
    }

    #[test]
    fn harmonic_field_is_removed() {
        // a linear field is harmonic: on a 1 mm grid (interpolation is the identity) V-SHARP
        // leaves nothing away from the mask edge. On anisotropic grids it is only approximately
        // zero: see the module note on interpolated-mask blending, which STI shares.
        let n = (64, 64, 64);
        let vs = (1.0, 1.0, 1.0);
        let mask = sphere(n, vs, 24.0);
        let grid = Grid::new(n.0, n.1, n.2, vs.0, vs.1, vs.2);
        let field: Vec<f64> = (0..n.0 * n.1 * n.2).map(|idx| {
            let (i, j, k) = (idx % n.0, (idx / n.0) % n.1, idx / (n.0 * n.1));
            0.01 * i as f64 * vs.0 - 0.02 * j as f64 * vs.1 + 0.03 * k as f64 * vs.2
        }).collect();
        let (local, m) = vsharp_sti(&field, &mask, &grid, &VsharpStiParams::default(), |_, _| {});
        let inner = sphere(n, vs, 16.0);
        let kept: Vec<f64> = local.iter().zip(&inner).filter(|(_, &b)| b != 0).map(|(v, _)| *v).collect();
        assert!(!kept.is_empty());
        let max = kept.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        assert!(max < 1e-9, "linear field leaves {max}");
        for (v, &b) in local.iter().zip(&m) {
            if b == 0 { assert_eq!(*v, 0.0); }
        }
    }

    #[test]
    fn no_deconvolution_point_response() {
        // 1 mm isotropic: a unit impulse far from the edge comes back as δ − ball/N (no 1/(1−Ŝ))
        let n = (72, 72, 72);
        let mask = sphere(n, (1.0, 1.0, 1.0), 34.0);
        let grid = Grid::new(n.0, n.1, n.2, 1.0, 1.0, 1.0);
        let mut field = vec![0.0; n.0 * n.1 * n.2];
        let c = 36 + 36 * n.0 + 36 * n.0 * n.1;
        field[c] = 1.0;
        let (local, _) = vsharp_sti(&field, &mask, &grid, &VsharpStiParams::default(), |_, _| {});
        assert!((local[c] - 1.0).abs() < 1e-9);
        assert!((local[c + 5] + 1.0 / 7520.0).abs() < 1e-9);
        assert!(local[c + 13].abs() < 1e-12);
    }

    #[test]
    fn empty_mask() {
        let grid = Grid::new(8, 8, 8, 1.0, 1.0, 1.0);
        let (l, m) = vsharp_sti(&vec![1.0; 512], &vec![0u8; 512], &grid, &VsharpStiParams::default(), |_, _| {});
        assert!(l.iter().all(|&v| v == 0.0) && m.iter().all(|&v| v == 0));
    }
}

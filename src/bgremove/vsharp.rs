//! V-SHARP background field removal
//!
//! Variable-kernel SHARP: each voxel uses the largest spherical mean value (SMV) ball that fits
//! inside the mask (radii 2–12 mm by default), so the brain edge keeps more tissue than
//! single-radius SHARP. The background (the mean of the field over the ball) is estimated on a
//! 1 mm isotropic grid and interpolated back; the local field is the native field minus the
//! background. Deconvolution by 1/(1 − Ŝ) is optional (`threshold > 0`).
//!
//! References:
//! Li, W., Wu, B., Liu, C. (2011).
//! "Quantitative susceptibility mapping of human brain reflects spatial variation in tissue composition."
//! NeuroImage, 55(4):1645-1656. https://doi.org/10.1016/j.neuroimage.2010.11.088
//!
//! Wu, B., Li, W., Guidon, A., Liu, C. (2012).
//! "Whole brain susceptibility mapping using compressed sensing."
//! Magnetic Resonance in Medicine, 67(1):137-147. https://doi.org/10.1002/mrm.23000

use num_complex::Complex64;
use crate::Grid;
use crate::fft::{fft3d, ifft3d};

/// Zero padding around the mask's bounding box, in native voxels.
const PAD: usize = 12;

/// V-SHARP algorithm parameters.
///
/// `threshold` defaults to 0 (no deconvolution), and `min_radius` is the smallest radius
/// (default 2); radii step by 1 mm.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct VsharpParams {
    /// Deconvolution (TSVD) threshold: 0 = no deconvolution (default). A positive value
    /// deconvolves the local field by 1/(1 − Ŝ) of the largest kernel, dropping frequencies
    /// where |1 − Ŝ| < threshold.
    pub threshold: f64,
    /// Largest SMV kernel radius in mm (default: 12.0); rounded down to a whole millimetre
    pub max_radius: f64,
    /// Smallest SMV kernel radius in mm (default: 2.0); rounded up to a whole millimetre.
    /// Radii run from here to `max_radius` in 1 mm steps (just `max_radius` if this is larger)
    pub min_radius: f64,
}

impl Default for VsharpParams {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            max_radius: 12.0,
            min_radius: 2.0,
        }
    }
}

/// V-SHARP background field removal
///
/// # Arguments
/// * `field` - Unwrapped total field (nx * ny * nz), any linear unit (rad, Hz, ppm); the local
///   field is returned in the same unit
/// * `mask` - Binary mask (nx * ny * nz), nonzero = inside ROI
/// * `grid` - Volume dimensions and voxel sizes
/// * `params` - V-SHARP parameters (deconvolution threshold, max/min radius in mm)
/// * `progress` - Progress callback (radius_index, total_radii)
///
/// # Returns
/// (local_field, eroded_mask), both on the input grid
pub fn vsharp(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &VsharpParams,
    mut progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<u8>) {
    let (nx, ny, nz) = grid.dims;
    let n_total = nx * ny * nz;
    let vs = [grid.voxel_size.0, grid.voxel_size.1, grid.voxel_size.2];
    let mut local_out = vec![0.0; n_total];
    let mut mask_out = vec![0u8; n_total];

    // 1. crop to the mask's bounding box, pad by PAD native voxels
    let Some((lo, hi)) = bounding_box(mask, grid.dims) else {
        return (local_out, mask_out);
    };
    let np = [0, 1, 2].map(|d| hi[d] - lo[d] + 1 + 2 * PAD);
    let mut p = vec![0.0; np[0] * np[1] * np[2]];
    let mut m = vec![0.0; p.len()];
    for k in lo[2]..=hi[2] {
        for j in lo[1]..=hi[1] {
            for i in lo[0]..=hi[0] {
                let src = i + j * nx + k * nx * ny;
                let dst = (i - lo[0] + PAD) + (j - lo[1] + PAD) * np[0] + (k - lo[2] + PAD) * np[0] * np[1];
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
    let mut p_fft = to_complex(&pe);
    fft3d(&mut p_fft, ne[0], ne[1], ne[2]);
    let mut m_fft = to_complex(&me);
    fft3d(&mut m_fft, ne[0], ne[1], ne[2]);
    let mut bg = pe.clone();
    let mut nm = vec![0.0; n_iso];
    let radii = vsharp_radii(params);
    let mut buf = vec![Complex64::new(0.0, 0.0); n_iso];
    for (idx, &r) in radii.iter().enumerate() {
        progress(idx + 1, radii.len());
        let (mut k_fft, count) = ball_fft(ne, r);
        if count == 0.0 {
            continue; // r < 5/6 mm: the ball minus its centre is empty
        }
        // erosion: the ball (unnormalised) must lie entirely in the mask
        for i in 0..n_iso { buf[i] = m_fft[i] * k_fft[i]; }
        ifft3d(&mut buf, ne[0], ne[1], ne[2]);
        let fits: Vec<bool> = (0..n_iso).map(|i| me[i] > 0.0 && buf[i].re >= count - 0.5).collect();
        // background = mean of the field over the ball
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

    // back to the unpadded 1 mm grid
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

    // optional deconvolution of the 1 mm local field by 1/(1 − Ŝ_rmax), folded back into the
    // background so the native-grid subtraction below is unchanged
    if params.threshold > 0.0 {
        if let Some(&r_max) = radii.last() {
            deconvolve_background(&pi, &mut bg_c, &nm_c, ni, r_max, params.threshold);
        }
    }

    // 5. background and mask onto the native (padded) grid; local = field − background
    let back = [0, 1, 2].map(|d| (1..=np[d]).map(|a| a as f64 * vs[d]).collect::<Vec<f64>>());
    let bg_n = trilinear(&bg_c, ni, &back);
    let nm_n = trilinear(&nm_c, ni, &back);

    for k in lo[2]..=hi[2] {
        for j in lo[1]..=hi[1] {
            for i in lo[0]..=hi[0] {
                let s = (i - lo[0] + PAD) + (j - lo[1] + PAD) * np[0] + (k - lo[2] + PAD) * np[0] * np[1];
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

/// Ascending list of whole-millimetre SMV radii: `ceil(min_radius)..=floor(max_radius)`, or just
/// `floor(max_radius)` if that range is empty (never below 1 mm).
fn vsharp_radii(params: &VsharpParams) -> Vec<f64> {
    let hi = params.max_radius.floor().max(1.0) as usize;
    let lo = params.min_radius.ceil().max(1.0) as usize;
    if lo > hi {
        return vec![hi as f64];
    }
    (lo..=hi).map(|r| r as f64).collect()
}

/// Deconvolution on the 1 mm grid: `L = (φ − B)·M`, then
/// `L' = F⁻¹[F(L) / (1 − Ŝ)]` with frequencies where `|1 − Ŝ| < threshold` dropped, and the
/// background becomes `φ − L'·M`. `Ŝ` is the normalised ball of the largest radius.
fn deconvolve_background(phi: &[f64], bg: &mut [f64], nm: &[f64], n: [usize; 3], r_max: f64, threshold: f64) {
    let (k_fft, count) = ball_fft(n, r_max);
    if count == 0.0 {
        return;
    }
    let mut l: Vec<Complex64> = (0..bg.len())
        .map(|i| Complex64::new((phi[i] - bg[i]) * nm[i], 0.0))
        .collect();
    fft3d(&mut l, n[0], n[1], n[2]);
    for (v, s) in l.iter_mut().zip(&k_fft) {
        // the ball is symmetric, so its transform is real
        let one_minus_s = 1.0 - s.re / count;
        *v = if one_minus_s.abs() < threshold { Complex64::new(0.0, 0.0) } else { *v / one_minus_s };
    }
    ifft3d(&mut l, n[0], n[1], n[2]);
    for i in 0..bg.len() {
        if nm[i] != 0.0 {
            bg[i] = phi[i] - l[i].re * nm[i];
        }
    }
}

fn to_complex(v: &[f64]) -> Vec<Complex64> {
    v.iter().map(|&x| Complex64::new(x, 0.0)).collect()
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
/// returns NaN there).
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

/// FFT of the ball of radius `r` mm (1 mm voxels) centred at the origin with wraparound, and
/// its voxel count. Membership: `|x|² ≤ (r + 1/6)²`, centre excluded.
fn ball_fft(n: [usize; 3], r: f64) -> (Vec<Complex64>, f64) {
    let mut k = vec![Complex64::new(0.0, 0.0); n[0] * n[1] * n[2]];
    let lim = (r + 1.0 / 6.0).powi(2);
    let h = (r + 1.0 / 6.0).floor().max(0.0) as i64;
    let wrap = |d: i64, nd: usize| d.rem_euclid(nd as i64) as usize;
    let mut count = 0.0;
    for dz in -h..=h {
        for dy in -h..=h {
            for dx in -h..=h {
                let d2 = (dx * dx + dy * dy + dz * dz) as f64;
                if d2 == 0.0 || d2 > lim { continue; }
                k[wrap(dx, n[0]) + wrap(dy, n[1]) * n[0] + wrap(dz, n[2]) * n[0] * n[1]] = Complex64::new(1.0, 0.0);
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

    fn linear_field(n: (usize, usize, usize), vs: (f64, f64, f64)) -> Vec<f64> {
        (0..n.0 * n.1 * n.2).map(|idx| {
            let (i, j, k) = (idx % n.0, (idx / n.0) % n.1, idx / (n.0 * n.1));
            0.01 * i as f64 * vs.0 - 0.02 * j as f64 * vs.1 + 0.03 * k as f64 * vs.2
        }).collect()
    }

    /// 72³ at 1 mm, sphere of 34 mm, unit impulse at the centre: (field, mask, grid, centre).
    fn impulse_setup() -> (Vec<f64>, Vec<u8>, Grid, usize) {
        let n = (72, 72, 72);
        let mask = sphere(n, (1.0, 1.0, 1.0), 34.0);
        let grid = Grid::new(n.0, n.1, n.2, 1.0, 1.0, 1.0);
        let mut field = vec![0.0; n.0 * n.1 * n.2];
        let c = 36 + 36 * n.0 + 36 * n.0 * n.1;
        field[c] = 1.0;
        (field, mask, grid, c)
    }

    #[test]
    fn default_radii() {
        let p = VsharpParams::default();
        assert_eq!((p.threshold, p.max_radius, p.min_radius), (0.0, 12.0, 2.0));
        assert_eq!(vsharp_radii(&p), (2..=12).map(|v| v as f64).collect::<Vec<_>>());
    }

    #[test]
    fn radii_from_params() {
        let p = |max_radius, min_radius| VsharpParams { threshold: 0.0, max_radius, min_radius };
        assert_eq!(vsharp_radii(&p(5.0, 3.0)), vec![3.0, 4.0, 5.0]);
        assert_eq!(vsharp_radii(&p(4.0, 1.0)), vec![1.0, 2.0, 3.0, 4.0]);
        // fractional radii round inwards to whole millimetres
        assert_eq!(vsharp_radii(&p(10.5, 2.0)), vsharp_radii(&p(10.0, 2.0)));
        assert_eq!(vsharp_radii(&p(4.0, 1.5)), vec![2.0, 3.0, 4.0]);
        // min above max: just the max radius; never below 1 mm
        assert_eq!(vsharp_radii(&p(3.0, 6.0)), vec![3.0]);
        assert_eq!(vsharp_radii(&p(0.4, 0.0)), vec![1.0]);
    }

    #[test]
    fn fractional_max_radius_matches_floor() {
        let n = (40, 40, 40);
        let mask = sphere(n, (1.0, 1.0, 1.0), 16.0);
        let grid = Grid::new(n.0, n.1, n.2, 1.0, 1.0, 1.0);
        let field: Vec<f64> = (0..n.0 * n.1 * n.2).map(|i| ((i * 7919) % 101) as f64 * 0.01).collect();
        let run = |max_radius| vsharp(&field, &mask, &grid, &VsharpParams { max_radius, ..Default::default() }, |_, _| {});
        assert_eq!(run(10.5), run(10.0));
    }

    #[test]
    fn ball_counts() {
        let expected = [(1, 6.0), (2, 32.0), (3, 146.0), (5, 586.0), (12, 7520.0), (14, 11836.0), (30, 115312.0)];
        for (r, want) in expected {
            assert_eq!(ball_fft([64, 64, 64], r as f64).1, want, "radius {r}");
        }
    }

    #[test]
    fn zero_field() {
        let n = (24, 24, 24);
        let mask = sphere(n, (1.0, 1.0, 1.0), 10.0);
        let grid = Grid::new(n.0, n.1, n.2, 1.0, 1.0, 1.0);
        let (local, m) = vsharp(&vec![0.0; 24 * 24 * 24], &mask, &grid, &VsharpParams::default(), |_, _| {});
        assert!(local.iter().all(|&v| v == 0.0));
        assert!(m.iter().any(|&v| v != 0));
    }

    #[test]
    fn harmonic_field_is_removed() {
        // a linear field is harmonic: on a 1 mm grid (interpolation is the identity) V-SHARP
        // leaves nothing away from the mask edge
        let n = (64, 64, 64);
        let vs = (1.0, 1.0, 1.0);
        let mask = sphere(n, vs, 24.0);
        let grid = Grid::new(n.0, n.1, n.2, vs.0, vs.1, vs.2);
        let field = linear_field(n, vs);
        let (local, m) = vsharp(&field, &mask, &grid, &VsharpParams::default(), |_, _| {});
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
    fn harmonic_field_is_nearly_removed_on_anisotropic_grid() {
        // only approximately zero off a 1 mm grid: 1 mm points at the mask edge blend in the
        // zeros outside it (≈1e-3 of the range)
        let n = (64, 64, 20);
        let vs = (0.8, 0.8, 3.0);
        let mask = sphere(n, vs, 24.0);
        let grid = Grid::new(n.0, n.1, n.2, vs.0, vs.1, vs.2);
        let field = linear_field(n, vs);
        let range = field.iter().fold(f64::MIN, |a, &v| a.max(v)) - field.iter().fold(f64::MAX, |a, &v| a.min(v));
        let (local, m) = vsharp(&field, &mask, &grid, &VsharpParams::default(), |_, _| {});
        let inner = sphere(n, vs, 14.0);
        let max = local.iter().zip(&inner).filter(|(_, &b)| b != 0).fold(0.0f64, |a, (v, _)| a.max(v.abs()));
        assert!(max < 5e-3 * range, "linear field leaves {max} of range {range}");
        assert!(local.iter().all(|v| v.is_finite()));
        for (v, &b) in local.iter().zip(&m) {
            if b == 0 { assert_eq!(*v, 0.0); }
        }
    }

    #[test]
    fn no_deconvolution_point_response() {
        // 1 mm isotropic: a unit impulse far from the edge comes back as δ − ball/N (no 1/(1−Ŝ))
        let (field, mask, grid, c) = impulse_setup();
        let (local, _) = vsharp(&field, &mask, &grid, &VsharpParams::default(), |_, _| {});
        assert!((local[c] - 1.0).abs() < 1e-9);
        assert!((local[c + 5] + 1.0 / 7520.0).abs() < 1e-9);
        assert!(local[c + 13].abs() < 1e-12);
    }

    #[test]
    fn deconvolution_is_opt_in_and_restores_the_impulse() {
        // with a threshold the local field is deconvolved by 1/(1 − Ŝ): the impulse response
        // moves from δ − ball/N back towards δ
        let (field, mask, grid, c) = impulse_setup();
        let (plain, m0) = vsharp(&field, &mask, &grid, &VsharpParams::default(), |_, _| {});
        let params = VsharpParams { threshold: 0.05, ..Default::default() };
        let (deconv, m1) = vsharp(&field, &mask, &grid, &params, |_, _| {});
        assert_eq!(m0, m1, "deconvolution must not change the mask");
        let err = |l: &[f64]| -> f64 {
            l.iter().enumerate().map(|(i, &v)| (v - if i == c { 1.0 } else { 0.0 }).powi(2)).sum()
        };
        assert!(deconv.iter().all(|v| v.is_finite()));
        assert!(err(&deconv) < 0.1 * err(&plain), "deconvolved {} vs plain {}", err(&deconv), err(&plain));
        for (v, &b) in deconv.iter().zip(&m1) {
            if b == 0 { assert_eq!(*v, 0.0); }
        }
    }

    #[test]
    fn larger_min_radius_erodes_more() {
        // the output mask is where at least the smallest ball fits, so raising min_radius
        // erodes deeper
        let n = (40, 40, 40);
        let mask = sphere(n, (1.0, 1.0, 1.0), 16.0);
        let grid = Grid::new(n.0, n.1, n.2, 1.0, 1.0, 1.0);
        let field = vec![0.0; n.0 * n.1 * n.2];
        let count = |p: &VsharpParams| vsharp(&field, &mask, &grid, p, |_, _| {}).1.iter().filter(|&&v| v != 0).count();
        let min2 = count(&VsharpParams::default());
        let deep = count(&VsharpParams { min_radius: 6.0, ..Default::default() });
        let total = mask.iter().filter(|&&v| v != 0).count();
        assert!(deep < min2 && min2 < total, "min 6 mm {deep}, min 2 mm {min2}, mask {total}");
    }

    #[test]
    fn progress_reports_each_radius() {
        let n = (16, 16, 16);
        let mask = sphere(n, (1.0, 1.0, 1.0), 6.0);
        let grid = Grid::new(n.0, n.1, n.2, 1.0, 1.0, 1.0);
        let mut calls = Vec::new();
        vsharp(&vec![0.1; 4096], &mask, &grid, &VsharpParams::default(), |i, t| calls.push((i, t)));
        assert_eq!(calls, (1..=11).map(|i| (i, 11)).collect::<Vec<_>>());
    }

    #[test]
    fn empty_mask() {
        let grid = Grid::new(8, 8, 8, 1.0, 1.0, 1.0);
        let (l, m) = vsharp(&vec![1.0; 512], &vec![0u8; 512], &grid, &VsharpParams::default(), |_, _| {});
        assert!(l.iter().all(|&v| v == 0.0) && m.iter().all(|&v| v == 0));
    }
}

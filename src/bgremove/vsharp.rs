//! V-SHARP background field removal
//!
//! Variable kernel SHARP uses multiple spherical mean value (SMV) kernel radii: each voxel takes
//! the largest kernel that fits inside the mask, so the brain edge keeps more tissue than
//! single-radius SHARP while the interior still gets a large kernel.
//!
//! Subtracting the spherical means leaves the SMV-high-passed local field `(δ − S_r) ∗ φ_local`
//! (the harmonic background cancels inside the mask). [`VsharpParams::deconvolve`] chooses what
//! happens next:
//!
//! - `deconvolve = true` (default) — the published V-SHARP (Wu et al. 2012, as implemented in
//!   QSM.jl): the high-passed field is deconvolved by `1/(1 − Ŝ)` of the largest kernel, with a
//!   TSVD threshold, and the kernels are built on the native grid. The deconvolution restores the
//!   low spatial frequencies the high-pass removed, so on a clean field it gets closer to the true
//!   local field.
//! - `deconvolve = false` — STI Suite 3.0's `V_SHARP`, reproduced step for step. STI omits the
//!   deconvolution step of Wu et al. 2012: its output is the high-passed field itself. The
//!   `1/(1 − Ŝ)` gain is largest at low spatial frequencies, which in vivo are where residual
//!   background and noise dominate, so leaving it out is more robust on real data (better
//!   test-retest reliability) at the cost of losing whole-structure offsets. To match STI exactly,
//!   this mode also follows STI's geometry, which differs from the deconvolving mode: the volume is
//!   cropped to the mask and zero-padded by 12 voxels, the background is estimated on a 1 mm
//!   isotropic grid and interpolated back (the local field is the native field minus it), kernels
//!   are balls of radius `r + 1/6` voxels without the centre, and radii are whole millimetres.
//!
//! References:
//! Li, W., Wu, B., Liu, C. (2011).
//! "Quantitative susceptibility mapping of human brain reflects spatial variation in tissue composition."
//! NeuroImage, 55(4):1645-1656. https://doi.org/10.1016/j.neuroimage.2010.11.088
//!
//! Wu, B., Li, W., Guidon, A., Liu, C. (2012).
//! "Whole brain susceptibility mapping using compressed sensing."
//! Magnetic Resonance in Medicine, 67(1):137-147. https://doi.org/10.1002/mrm.23000
//!
//! Reference implementations: https://github.com/kamesy/QSM.jl (`deconvolve = true`);
//! STI Suite 3.0 `V_SHARP` (`deconvolve = false`).

use num_complex::Complex64;
use crate::Grid;
use crate::fft::{fft3d, ifft3d};
use crate::kernels::smv::{smv_kernel, erode_mask_smv};

/// Zero padding around the mask's bounding box, in native voxels (`deconvolve = false` only).
const STI_PAD: usize = 12;

/// V-SHARP algorithm parameters
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct VsharpParams {
    /// Deconvolution (TSVD) threshold: k-space frequencies where |1 - S| < threshold are dropped
    /// rather than divided through. Higher = more regularisation, more robust to a noisy input field
    /// (default: 0.2). Used only when `deconvolve` is true.
    pub threshold: f64,
    /// Maximum (largest) SMV kernel radius in mm (default: 12.0). With `deconvolve = false` it is
    /// rounded down to a whole millimetre.
    pub max_radius: f64,
    /// Minimum (smallest) SMV kernel radius in mm (default: 1.0). The output mask is where the
    /// smallest kernel fits.
    ///
    /// With `deconvolve = true` it is also the step between successive radii
    /// (`max_radius, max_radius − min_radius, …`). With `deconvolve = false` radii run in 1 mm
    /// steps from `ceil(min_radius)` to `floor(max_radius)`, and the smallest radius used is 2 mm:
    /// STI Suite computes the 1 mm ball but drops voxels where only it fits, so values below 2
    /// behave as 2.
    pub min_radius: f64,
    /// Deconvolve the high-passed field by `1/(1 − Ŝ)` of the largest kernel (default: true).
    ///
    /// `true` is the published V-SHARP (Wu et al. 2012) on the native grid. `false` reproduces
    /// STI Suite 3.0's `V_SHARP`, which returns the SMV-high-passed local field
    /// `(δ − S_r) ∗ φ_local` without deconvolution and estimates the background on a 1 mm grid;
    /// see the module docs for the differences and when that is preferable.
    pub deconvolve: bool,
}

impl Default for VsharpParams {
    fn default() -> Self {
        Self {
            // 0.2 (was 0.05): the deconvolution amplifies structured error in a noisy estimated field,
            // and real pipelines always feed V-SHARP an estimated (not ground-truth) field. A stronger
            // regularisation is robust to that; 0.05 under-regularises and only wins on a clean field
            // (which BFR is never run on in practice). See the PR for the benchmark.
            threshold: 0.2,
            max_radius: 12.0,
            min_radius: 1.0,
            deconvolve: true,
        }
    }
}

/// V-SHARP background field removal
///
/// Uses multiple SMV kernel radii; each voxel takes the largest kernel that fits inside the mask.
/// See [`VsharpParams::deconvolve`] for the two modes.
///
/// # Arguments
/// * `field` - Unwrapped total field (nx * ny * nz), any linear unit (rad, Hz, ppm); the local
///   field is returned in the same unit
/// * `mask` - Binary mask (nx * ny * nz), nonzero = inside ROI
/// * `grid` - Volume dimensions and voxel sizes
/// * `params` - V-SHARP parameters (deconvolution switch and threshold, min/max radius in mm)
/// * `progress` - Progress callback (radius_index, total_radii)
///
/// # Returns
/// (local_field, eroded_mask), both on the input grid
pub fn vsharp(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &VsharpParams,
    progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<u8>) {
    if params.deconvolve {
        let radii = vsharp_radii(params);
        vsharp_with_radii(field, mask, grid, &radii, params.threshold, progress)
    } else {
        vsharp_sti(field, mask, grid, &sti_radii(params), progress)
    }
}

/// Build the descending list of SMV kernel radii (mm) from the params (`deconvolve = true`).
///
/// Starts at `max_radius` (mm) and steps down by `min_radius` (mm), which is
/// also the smallest kernel used.
fn vsharp_radii(params: &VsharpParams) -> Vec<f64> {
    let mut radii = Vec::new();
    let mut r = params.max_radius;
    let step = params.min_radius;
    while r >= step {
        radii.push(r);
        r -= step;
    }
    if radii.is_empty() {
        radii.push(params.max_radius);
    }
    radii
}

/// Ascending whole-millimetre radii for `deconvolve = false`:
/// `max(ceil(min_radius), 2) ..= floor(max_radius)`, or just `floor(max_radius)` (at least 1 mm)
/// if that range is empty.
fn sti_radii(params: &VsharpParams) -> Vec<f64> {
    let hi = params.max_radius.floor().max(1.0) as usize;
    let lo = params.min_radius.ceil().max(2.0) as usize;
    if lo > hi {
        return vec![hi as f64];
    }
    (lo..=hi).map(|r| r as f64).collect()
}

/// V-SHARP with an explicit list of SMV kernel radii (mm), with deconvolution.
///
/// Internal entry point; the public [`vsharp`] wrapper derives the radii
/// from [`VsharpParams`].
pub(crate) fn vsharp_with_radii(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    radii: &[f64],
    threshold: f64,
    mut progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<u8>) {
    let (nx, ny, nz) = grid.dims;

    if radii.is_empty() {
        return (vec![0.0; nx * ny * nz], mask.to_vec());
    }

    // If only one radius, use regular SHARP
    if radii.len() == 1 {
        progress(1, 1);
        return crate::bgremove::sharp::sharp_core(
            field, mask, grid, threshold, radii[0]
        );
    }

    let n_total = nx * ny * nz;
    let n_radii = radii.len();

    // Sort radii from largest to smallest
    let mut sorted_radii = radii.to_vec();
    sorted_radii.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // FFT of field
    let mut field_complex: Vec<Complex64> = field.iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    fft3d(&mut field_complex, nx, ny, nz);
    let field_fft = field_complex.clone();

    // Track which voxels have been processed and final mask
    let mut processed = vec![false; n_total];
    let mut local_field = vec![0.0; n_total];
    let mut final_mask = vec![0u8; n_total];

    let delta = 1.0 - 1e-7_f64.sqrt();
    let mut inverse_kernel: Option<Vec<f64>> = None;

    for (idx, &radius) in sorted_radii.iter().enumerate() {
        // Report progress
        progress(idx + 1, n_radii);

        // Generate SMV kernel
        let s_kernel = smv_kernel(grid, radius);

        // FFT of SMV kernel
        let mut s_complex: Vec<Complex64> = s_kernel.iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect();
        fft3d(&mut s_complex, nx, ny, nz);
        let s_fft: Vec<f64> = s_complex.iter().map(|c| c.re).collect();

        // Store inverse of first (largest) kernel
        if inverse_kernel.is_none() {
            inverse_kernel = Some(s_fft.iter().map(|&s| {
                let one_minus_s = 1.0 - s;
                if one_minus_s.abs() < threshold {
                    0.0
                } else {
                    1.0 / one_minus_s
                }
            }).collect());
        }

        // Erode mask for this radius
        let eroded = erode_mask_smv(mask, &s_fft, grid, delta);
        let current_mask: Vec<bool> = eroded.iter().map(|&m| m == 1).collect();

        // Apply high-pass filter
        let mut filtered = field_fft.clone();
        for i in 0..n_total {
            filtered[i] *= 1.0 - s_fft[i];
        }

        ifft3d(&mut filtered, nx, ny, nz);

        for i in 0..n_total {
            if current_mask[i] && !processed[i] {
                local_field[i] = filtered[i].re;
                processed[i] = true;
                final_mask[i] = 1;
            }
        }
    }

    // Deconvolution
    if let Some(inv_kernel) = inverse_kernel {
        let mut local_complex: Vec<Complex64> = local_field.iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect();

        fft3d(&mut local_complex, nx, ny, nz);

        for i in 0..n_total {
            local_complex[i] *= inv_kernel[i];
        }

        ifft3d(&mut local_complex, nx, ny, nz);

        for i in 0..n_total {
            local_field[i] = if final_mask[i] == 1 { local_complex[i].re } else { 0.0 };
        }
    }

    (local_field, final_mask)
}

/// V-SHARP without deconvolution, as STI Suite 3.0's `V_SHARP`.
///
/// 1. Crop to the mask's bounding box and zero-pad by [`STI_PAD`] native voxels.
/// 2. Resample field and mask to a 1 mm grid (trilinear; mask `> 0.5`), odd sizes padded to even.
/// 3. For each radius, ascending: where the ball fits in the 1 mm mask, background = mean of the
///    field over the ball (larger radii overwrite).
/// 4. Interpolate the background and mask back to the native grid; local = field − background
///    (the local field itself is never resampled).
///
/// `radii` are whole millimetres, ascending.
fn vsharp_sti(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    radii: &[f64],
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
    let np = [0, 1, 2].map(|d| hi[d] - lo[d] + 1 + 2 * STI_PAD);
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

    // odd dimensions padded to even (zeros at the end), mask thresholded
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

    // 3. per radius (ascending, so each voxel ends with the largest ball that fits)
    let mut p_fft = to_complex(&pe);
    fft3d(&mut p_fft, ne[0], ne[1], ne[2]);
    let mut m_fft = to_complex(&me);
    fft3d(&mut m_fft, ne[0], ne[1], ne[2]);
    let mut bg = pe.clone();
    let mut nm = vec![0.0; n_iso];
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

    // 4. background and mask onto the native (padded) grid; local = field − background
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

/// FFT of STI Suite's ball of radius `r` mm (1 mm voxels) centred at the origin with wraparound,
/// and its voxel count. Membership: `|x|² ≤ (r + 1/6)²`, centre excluded.
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

    // ---------------------------------------------------------------------------------------
    // deconvolve = true (default): the published V-SHARP
    // ---------------------------------------------------------------------------------------

    #[test]
    fn test_vsharp_default_params() {
        let p = VsharpParams::default();
        assert_eq!((p.threshold, p.max_radius, p.min_radius, p.deconvolve), (0.2, 12.0, 1.0, true));
        assert_eq!(vsharp_radii(&p), (1..=12).rev().map(|v| v as f64).collect::<Vec<_>>());
    }

    #[test]
    fn test_vsharp_default_is_deconvolving_vsharp() {
        // the public default runs the native-grid deconvolving path with radii 12, 11, …, 1 mm
        // and threshold 0.2, bit for bit
        let n = (24, 24, 24);
        let mask = sphere(n, (1.0, 1.0, 1.0), 10.0);
        let grid = Grid::new(n.0, n.1, n.2, 1.0, 1.0, 1.0);
        let field: Vec<f64> = (0..n.0 * n.1 * n.2).map(|i| ((i * 7919) % 101) as f64 * 0.01).collect();
        let radii: Vec<f64> = (1..=12).rev().map(|v| v as f64).collect();
        let a = vsharp(&field, &mask, &grid, &VsharpParams::default(), |_, _| {});
        let b = vsharp_with_radii(&field, &mask, &grid, &radii, 0.2, |_, _| {});
        assert_eq!(a, b);
        // and it differs from the no-deconvolution mode
        let c = vsharp(&field, &mask, &grid, &sti(), |_, _| {});
        assert_ne!(a.0, c.0);
    }

    #[test]
    fn test_vsharp_zero_field() {
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let radii = vec![4.0, 3.0, 2.0];
        let (local, _) = vsharp_with_radii(&field, &mask, &grid, &radii, 0.05, |_, _| {});

        for &val in local.iter() {
            assert!(val.abs() < 1e-10);
        }
    }

    #[test]
    fn test_vsharp_preserves_more_than_sharp() {
        let n = 16;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        // V-SHARP with multiple radii
        let radii = vec![5.0, 4.0, 3.0, 2.0];
        let (_, vsharp_mask) = vsharp_with_radii(&field, &mask, &grid, &radii, 0.05, |_, _| {});

        // SHARP with single large radius
        let (_, sharp_mask) = crate::bgremove::sharp::sharp_core(
            &field, &mask, &grid, 0.05, 5.0
        );

        let vsharp_count: usize = vsharp_mask.iter().map(|&m| m as usize).sum();
        let sharp_count: usize = sharp_mask.iter().map(|&m| m as usize).sum();

        // V-SHARP should preserve at least as many voxels as SHARP
        assert!(vsharp_count >= sharp_count,
            "V-SHARP {} should preserve at least as many as SHARP {}",
            vsharp_count, sharp_count);
    }

    #[test]
    fn test_vsharp_nonuniform_voxels() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 0.5, 1.0, 2.0);

        // Anisotropic voxel sizes
        let radii = vec![4.0, 3.0, 2.0];
        let (local, final_mask) = vsharp_with_radii(
            &field, &mask, &grid, &radii, 0.05, |_, _| {}
        );

        // All values should be finite
        for (i, &val) in local.iter().enumerate() {
            assert!(val.is_finite(), "V-SHARP nonuniform voxels: finite at index {}", i);
        }

        // Final mask should have some voxels
        let mask_count: usize = final_mask.iter().map(|&m| m as usize).sum();
        assert!(mask_count > 0, "V-SHARP nonuniform: final mask should have some voxels");
    }

    #[test]
    fn test_vsharp_single_radius() {
        let n = 8;
        let field: Vec<f64> = (0..n*n*n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        // Single radius should delegate to SHARP
        let radii = vec![3.0];
        let (local, final_mask) = vsharp_with_radii(
            &field, &mask, &grid, &radii, 0.05, |_, _| {}
        );

        // All values should be finite
        for (i, &val) in local.iter().enumerate() {
            assert!(val.is_finite(), "V-SHARP single radius: finite at index {}", i);
        }

        // Result should match SHARP with same radius
        let (sharp_local, sharp_mask) = crate::bgremove::sharp::sharp_core(
            &field, &mask, &grid, 0.05, 3.0
        );

        for i in 0..n*n*n {
            assert!(
                (local[i] - sharp_local[i]).abs() < 1e-10,
                "Single-radius V-SHARP should match SHARP at index {}", i
            );
        }

        assert_eq!(final_mask, sharp_mask, "Single-radius V-SHARP mask should match SHARP mask");
    }

    #[test]
    fn test_vsharp_empty_radii() {
        // Empty radii should return zeros and the original mask
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let (local, returned_mask) = vsharp_with_radii(
            &field, &mask, &grid, &[], 0.05, |_, _| {}
        );

        for &val in &local {
            assert_eq!(val, 0.0, "Empty radii should return zero local field");
        }
        assert_eq!(returned_mask, mask, "Empty radii should return original mask");
    }

    #[test]
    fn test_vsharp_larger_volume() {
        // 16x16x16 volume with spherical mask
        let n = 16;
        let mut field = vec![0.0; n * n * n];
        // Linear background field in z
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    field[x + y * n + z * n * n] = (z as f64) * 0.1;
                }
            }
        }

        // Spherical mask
        let mut mask = vec![0u8; n * n * n];
        let center = n / 2;
        let radius = n / 3;
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let dx = (x as i32) - (center as i32);
                    let dy = (y as i32) - (center as i32);
                    let dz = (z as i32) - (center as i32);
                    if dx * dx + dy * dy + dz * dz <= (radius * radius) as i32 {
                        mask[x + y * n + z * n * n] = 1;
                    }
                }
            }
        }

        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);
        let radii = vec![6.0, 4.0, 3.0, 2.0];
        let (local, final_mask) = vsharp_with_radii(
            &field, &mask, &grid, &radii, 0.05, |_, _| {}
        );

        assert_eq!(local.len(), n * n * n);
        for &val in &local {
            assert!(val.is_finite(), "V-SHARP larger volume values should be finite");
        }

        let mask_count: usize = final_mask.iter().map(|&m| m as usize).sum();
        assert!(mask_count > 0, "V-SHARP larger volume should have voxels in final mask");

        // Voxels outside the final mask should be zero
        for i in 0..n * n * n {
            if final_mask[i] == 0 {
                assert_eq!(local[i], 0.0, "Outside final mask should be zero");
            }
        }
    }

    #[test]
    fn test_vsharp_with_progress() {
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let radii = vec![4.0, 3.0, 2.0];
        let mut progress_calls = Vec::new();
        let (local, _) = vsharp_with_radii(
            &field, &mask, &grid, &radii, 0.05,
            |idx, total| { progress_calls.push((idx, total)); }
        );

        assert_eq!(local.len(), n * n * n);
        assert!(!progress_calls.is_empty(), "Progress should be called");
        for &val in &local {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_vsharp_with_progress_single_radius() {
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let radii = vec![3.0];
        let mut progress_calls = Vec::new();
        let (local, _) = vsharp_with_radii(
            &field, &mask, &grid, &radii, 0.05,
            |idx, total| { progress_calls.push((idx, total)); }
        );

        assert_eq!(local.len(), n * n * n);
        assert!(!progress_calls.is_empty(), "Progress should be called for single radius");
        for &val in &local {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_vsharp_with_progress_empty_radii() {
        let n = 8;
        let field = vec![0.0; n * n * n];
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let mut progress_calls = Vec::new();
        let (local, returned_mask) = vsharp_with_radii(
            &field, &mask, &grid, &[], 0.05,
            |idx, total| { progress_calls.push((idx, total)); }
        );

        for &val in &local {
            assert_eq!(val, 0.0);
        }
        assert_eq!(returned_mask, mask);
    }

    #[test]
    fn test_vsharp_unsorted_radii() {
        // Radii given in arbitrary order - should be sorted internally
        let n = 8;
        let field: Vec<f64> = (0..n * n * n).map(|i| (i as f64) * 0.001).collect();
        let mask = vec![1u8; n * n * n];
        let grid = Grid::new(n, n, n, 1.0, 1.0, 1.0);

        let radii_sorted = vec![4.0, 3.0, 2.0];
        let radii_unsorted = vec![2.0, 4.0, 3.0];

        let (local_sorted, mask_sorted) = vsharp_with_radii(
            &field, &mask, &grid, &radii_sorted, 0.05, |_, _| {}
        );
        let (local_unsorted, mask_unsorted) = vsharp_with_radii(
            &field, &mask, &grid, &radii_unsorted, 0.05, |_, _| {}
        );

        // Results should be the same regardless of input order
        assert_eq!(mask_sorted, mask_unsorted, "Sorted and unsorted radii should give same mask");
        for i in 0..n * n * n {
            assert!(
                (local_sorted[i] - local_unsorted[i]).abs() < 1e-10,
                "Results should match at index {}",
                i
            );
        }
    }

    // ---------------------------------------------------------------------------------------
    // deconvolve = false: STI Suite 3.0 V_SHARP
    // ---------------------------------------------------------------------------------------

    /// Default parameters with `deconvolve = false`.
    fn sti() -> VsharpParams {
        VsharpParams { deconvolve: false, ..Default::default() }
    }

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
    fn sti_default_radii() {
        // min_radius 1 (the default) behaves as 2: STI never uses the 1 mm ball
        assert_eq!(sti_radii(&sti()), (2..=12).map(|v| v as f64).collect::<Vec<_>>());
    }

    #[test]
    fn sti_radii_from_params() {
        let p = |max_radius, min_radius| VsharpParams { max_radius, min_radius, ..sti() };
        assert_eq!(sti_radii(&p(5.0, 3.0)), vec![3.0, 4.0, 5.0]);
        assert_eq!(sti_radii(&p(4.0, 1.0)), vec![2.0, 3.0, 4.0]);
        // fractional radii round inwards to whole millimetres
        assert_eq!(sti_radii(&p(10.5, 2.0)), sti_radii(&p(10.0, 2.0)));
        assert_eq!(sti_radii(&p(5.0, 2.5)), vec![3.0, 4.0, 5.0]);
        // min above max: just the max radius; never below 1 mm
        assert_eq!(sti_radii(&p(3.0, 6.0)), vec![3.0]);
        assert_eq!(sti_radii(&p(0.4, 0.0)), vec![1.0]);
    }

    #[test]
    fn sti_fractional_max_radius_matches_floor() {
        let n = (40, 40, 40);
        let mask = sphere(n, (1.0, 1.0, 1.0), 16.0);
        let grid = Grid::new(n.0, n.1, n.2, 1.0, 1.0, 1.0);
        let field: Vec<f64> = (0..n.0 * n.1 * n.2).map(|i| ((i * 7919) % 101) as f64 * 0.01).collect();
        let run = |max_radius| vsharp(&field, &mask, &grid, &VsharpParams { max_radius, ..sti() }, |_, _| {});
        assert_eq!(run(10.5), run(10.0));
    }

    #[test]
    fn sti_threshold_is_ignored() {
        let n = (32, 32, 32);
        let mask = sphere(n, (1.0, 1.0, 1.0), 13.0);
        let grid = Grid::new(n.0, n.1, n.2, 1.0, 1.0, 1.0);
        let field: Vec<f64> = (0..n.0 * n.1 * n.2).map(|i| ((i * 7919) % 101) as f64 * 0.01).collect();
        let a = vsharp(&field, &mask, &grid, &sti(), |_, _| {});
        let b = vsharp(&field, &mask, &grid, &VsharpParams { threshold: 0.0, ..sti() }, |_, _| {});
        let c = vsharp(&field, &mask, &grid, &VsharpParams { threshold: 0.05, ..sti() }, |_, _| {});
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn sti_ball_counts() {
        let expected = [(1, 6.0), (2, 32.0), (3, 146.0), (5, 586.0), (12, 7520.0), (14, 11836.0), (30, 115312.0)];
        for (r, want) in expected {
            assert_eq!(ball_fft([64, 64, 64], r as f64).1, want, "radius {r}");
        }
    }

    #[test]
    fn sti_zero_field() {
        let n = (24, 24, 24);
        let mask = sphere(n, (1.0, 1.0, 1.0), 10.0);
        let grid = Grid::new(n.0, n.1, n.2, 1.0, 1.0, 1.0);
        let (local, m) = vsharp(&vec![0.0; 24 * 24 * 24], &mask, &grid, &sti(), |_, _| {});
        assert!(local.iter().all(|&v| v == 0.0));
        assert!(m.iter().any(|&v| v != 0));
    }

    #[test]
    fn sti_harmonic_field_is_removed() {
        // a linear field is harmonic: on a 1 mm grid (interpolation is the identity) V-SHARP
        // leaves nothing away from the mask edge
        let n = (64, 64, 64);
        let vs = (1.0, 1.0, 1.0);
        let mask = sphere(n, vs, 24.0);
        let grid = Grid::new(n.0, n.1, n.2, vs.0, vs.1, vs.2);
        let field = linear_field(n, vs);
        let (local, m) = vsharp(&field, &mask, &grid, &sti(), |_, _| {});
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
    fn sti_harmonic_field_is_nearly_removed_on_anisotropic_grid() {
        // only approximately zero off a 1 mm grid: 1 mm points at the mask edge blend in the
        // zeros outside it (≈1e-3 of the range)
        let n = (64, 64, 20);
        let vs = (0.8, 0.8, 3.0);
        let mask = sphere(n, vs, 24.0);
        let grid = Grid::new(n.0, n.1, n.2, vs.0, vs.1, vs.2);
        let field = linear_field(n, vs);
        let range = field.iter().fold(f64::MIN, |a, &v| a.max(v)) - field.iter().fold(f64::MAX, |a, &v| a.min(v));
        let (local, m) = vsharp(&field, &mask, &grid, &sti(), |_, _| {});
        let inner = sphere(n, vs, 14.0);
        let max = local.iter().zip(&inner).filter(|(_, &b)| b != 0).fold(0.0f64, |a, (v, _)| a.max(v.abs()));
        assert!(max < 5e-3 * range, "linear field leaves {max} of range {range}");
        assert!(local.iter().all(|v| v.is_finite()));
        for (v, &b) in local.iter().zip(&m) {
            if b == 0 { assert_eq!(*v, 0.0); }
        }
    }

    #[test]
    fn sti_point_response_is_delta_minus_ball() {
        // 1 mm isotropic: a unit impulse far from the edge comes back as δ − S₁₂ (the 12 mm
        // ball without its centre, 7520 voxels, normalised), with no 1/(1 − Ŝ)
        let (field, mask, grid, c) = impulse_setup();
        let (local, _) = vsharp(&field, &mask, &grid, &sti(), |_, _| {});
        assert!((local[c] - 1.0).abs() < 1e-9);
        assert!((local[c + 5] + 1.0 / 7520.0).abs() < 1e-9);
        assert!((local[c + 12 * 72] + 1.0 / 7520.0).abs() < 1e-9);
        assert!(local[c + 13].abs() < 1e-12);
    }

    #[test]
    fn deconvolution_changes_the_point_response() {
        // with deconvolution (the default) the response is no longer δ − S
        let (field, mask, grid, c) = impulse_setup();
        let (plain, _) = vsharp(&field, &mask, &grid, &sti(), |_, _| {});
        let (deconv, m) = vsharp(&field, &mask, &grid, &VsharpParams::default(), |_, _| {});
        assert!(deconv.iter().all(|v| v.is_finite()));
        assert!((deconv[c + 5] - plain[c + 5]).abs() > 1e-6, "{} vs {}", deconv[c + 5], plain[c + 5]);
        for (v, &b) in deconv.iter().zip(&m) {
            if b == 0 { assert_eq!(*v, 0.0); }
        }
    }

    #[test]
    fn sti_larger_min_radius_erodes_more() {
        // the output mask is where at least the smallest ball fits, so raising min_radius
        // erodes deeper; min_radius 1 and 2 are the same
        let n = (40, 40, 40);
        let mask = sphere(n, (1.0, 1.0, 1.0), 16.0);
        let grid = Grid::new(n.0, n.1, n.2, 1.0, 1.0, 1.0);
        let field = vec![0.0; n.0 * n.1 * n.2];
        let run = |min_radius| vsharp(&field, &mask, &grid, &VsharpParams { min_radius, ..sti() }, |_, _| {}).1;
        let count = |m: &[u8]| m.iter().filter(|&&v| v != 0).count();
        assert_eq!(run(1.0), run(2.0));
        let min2 = count(&run(2.0));
        let deep = count(&run(6.0));
        let total = count(&mask);
        assert!(deep < min2 && min2 < total, "min 6 mm {deep}, min 2 mm {min2}, mask {total}");
    }

    #[test]
    fn sti_progress_reports_each_radius() {
        let n = (16, 16, 16);
        let mask = sphere(n, (1.0, 1.0, 1.0), 6.0);
        let grid = Grid::new(n.0, n.1, n.2, 1.0, 1.0, 1.0);
        let mut calls = Vec::new();
        vsharp(&vec![0.1; 4096], &mask, &grid, &sti(), |i, t| calls.push((i, t)));
        assert_eq!(calls, (1..=11).map(|i| (i, 11)).collect::<Vec<_>>());
    }

    #[test]
    fn sti_empty_mask() {
        let grid = Grid::new(8, 8, 8, 1.0, 1.0, 1.0);
        let (l, m) = vsharp(&vec![1.0; 512], &vec![0u8; 512], &grid, &sti(), |_, _| {});
        assert!(l.iter().all(|&v| v == 0.0) && m.iter().all(|&v| v == 0));
    }
}

/// Scratch: one-ingredient-at-a-time ablation between the published V-SHARP and STI's.
#[doc(hidden)]
pub mod ablation {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    pub struct Opts {
        /// background on a 1 mm grid, interpolated back (STI) vs native grid
        pub iso1mm: bool,
        /// background on an isotropic grid at the data's finest voxel size (min of the three
        /// voxel dims), interpolated back; same mechanics as `iso1mm` with spacing h = min(vs)
        /// instead of 1 mm. Mutually exclusive with `iso1mm`.
        pub iso_finest: bool,
        /// ball radius r + 1/6 (STI) vs r (main)
        pub ball_plus: bool,
        /// ball without its centre voxel (STI) vs with it (main)
        pub ball_no_centre: bool,
        /// radii 2..=12 whole mm (STI) vs 12, 11, ..., 1 (main)
        pub sti_radii: bool,
        /// crop to the mask bbox + this many native voxels of zeros (STI: 12) vs full FOV (None, main)
        pub pad: Option<usize>,
        /// ball must lie entirely in the mask (STI) vs normalised conv > 1 - sqrt(1e-7) (main)
        pub strict_fit: bool,
        pub deconvolve: bool,
        pub threshold: f64,
        pub max_radius: f64,
        pub min_radius: f64,
    }

    impl Opts {
        pub fn main() -> Self {
            Opts { iso1mm: false, iso_finest: false, ball_plus: false, ball_no_centre: false, sti_radii: false, pad: None, strict_fit: false, deconvolve: true, threshold: 0.2, max_radius: 12.0, min_radius: 1.0 }
        }
        /// Parse a compact spec such as `d1g0b0r0p0e0` (each letter then 0/1, starting from
        /// [`Opts::main`]): d deconvolve, g 1 mm grid, i isotropic grid at the finest voxel size
        /// (h = min voxel dim; `g1` and `i1` together are an error), b STI ball, r STI radii, p crop + 12-voxel pad,
        /// e strict edge fit. Optional: `q`/`c` the two halves of b (q: r + 1/6, c: no centre), `t<digits>` threshold 0.<digits>, `R<int>` max radius,
        /// `m<int>` min radius, `P<int>` crop + that pad. `main` / `sti` are presets.
        pub fn from_spec(spec: &str) -> Result<Self, String> {
            if spec == "main" { return Ok(Self::main()); }
            if spec == "sti" { return Ok(Self::sti()); }
            let mut o = Self::main();
            let c: Vec<char> = spec.chars().collect();
            let mut i = 0;
            while i < c.len() {
                let key = c[i];
                let mut j = i + 1;
                while j < c.len() && c[j].is_ascii_digit() { j += 1; }
                let num: String = c[i + 1..j].iter().collect();
                if num.is_empty() { return Err(format!("missing value after '{key}' in {spec}")); }
                let on = num == "1";
                match key {
                    'd' => o.deconvolve = on,
                    'g' => o.iso1mm = on,
                    'i' => o.iso_finest = on,
                    'b' => { o.ball_plus = on; o.ball_no_centre = on; }
                    'q' => o.ball_plus = on,
                    'c' => o.ball_no_centre = on,
                    'r' => o.sti_radii = on,
                    'p' => o.pad = if on { Some(STI_PAD) } else { None },
                    'e' => o.strict_fit = on,
                    't' => o.threshold = format!("0.{num}").parse().map_err(|e| format!("{e}"))?,
                    'R' => o.max_radius = num.parse().map_err(|e| format!("{e}"))?,
                    'm' => o.min_radius = num.parse().map_err(|e| format!("{e}"))?,
                    'P' => o.pad = Some(num.parse().map_err(|e| format!("{e}"))?),
                    _ => return Err(format!("unknown key '{key}' in {spec}")),
                }
                i = j;
            }
            if o.iso1mm && o.iso_finest { return Err(format!("g1 and i1 are mutually exclusive in {spec}")); }
            Ok(o)
        }
        pub fn sti() -> Self {
            Opts { iso1mm: true, iso_finest: false, ball_plus: true, ball_no_centre: true, sti_radii: true, pad: Some(STI_PAD), strict_fit: true, deconvolve: false, threshold: 0.2, max_radius: 12.0, min_radius: 1.0 }
        }
    }

    /// Unnormalised kernel on a grid of voxel size `vs` (wraparound at the origin) and its count.
    fn kernel_fft(n: [usize; 3], vs: [f64; 3], r: f64, plus: bool, no_centre: bool) -> (Vec<Complex64>, f64) {
        let mut k = vec![Complex64::new(0.0, 0.0); n[0] * n[1] * n[2]];
        let rr = if plus { r + 1.0 / 6.0 } else { r };
        let lim = rr * rr;
        let h = [0, 1, 2].map(|d| ((rr / vs[d]).floor() as i64).min(n[d] as i64 / 2));
        let wrap = |d: i64, nd: usize| d.rem_euclid(nd as i64) as usize;
        let mut count = 0.0;
        for dz in -h[2]..=h[2] {
            for dy in -h[1]..=h[1] {
                for dx in -h[0]..=h[0] {
                    let (x, y, z) = (dx as f64 * vs[0], dy as f64 * vs[1], dz as f64 * vs[2]);
                    let d2 = x * x + y * y + z * z;
                    if d2 > lim || (no_centre && d2 == 0.0) { continue; }
                    let idx = wrap(dx, n[0]) + wrap(dy, n[1]) * n[0] + wrap(dz, n[2]) * n[0] * n[1];
                    if k[idx].re == 0.0 {
                        k[idx] = Complex64::new(1.0, 0.0);
                        count += 1.0;
                    }
                }
            }
        }
        fft3d(&mut k, n[0], n[1], n[2]);
        (k, count)
    }

    pub fn run(field: &[f64], mask: &[u8], grid: &Grid, o: Opts) -> (Vec<f64>, Vec<u8>) {
        let (nx, ny, nz) = grid.dims;
        let vs = [grid.voxel_size.0, grid.voxel_size.1, grid.voxel_size.2];
        let mut local_out = vec![0.0; nx * ny * nz];
        let mut mask_out = vec![0u8; nx * ny * nz];
        let Some((bl, bh)) = bounding_box(mask, grid.dims) else { return (local_out, mask_out) };
        // native working volume (cropped + padded, or full FOV)
        let (lo, hi, pad) = match o.pad { Some(pad) => (bl, bh, pad), None => ([0; 3], [nx - 1, ny - 1, nz - 1], 0) };
        let np = [0, 1, 2].map(|d| hi[d] - lo[d] + 1 + 2 * pad);
        let mut p = vec![0.0; np[0] * np[1] * np[2]];
        let mut m = vec![0.0; p.len()];
        for k in lo[2]..=hi[2] { for j in lo[1]..=hi[1] { for i in lo[0]..=hi[0] {
            let src = i + j * nx + k * nx * ny;
            let dst = (i - lo[0] + pad) + (j - lo[1] + pad) * np[0] + (k - lo[2] + pad) * np[0] * np[1];
            p[dst] = field[src];
            m[dst] = if mask[src] != 0 { 1.0 } else { 0.0 };
        }}}

        // working grid W: native, or isotropic at spacing h (1 mm for g, finest voxel dim for i).
        // Axes already at h are left on the native lattice (identity, no extra plane); with h = 1
        // every expression reduces bitwise to the original 1 mm path.
        assert!(!(o.iso1mm && o.iso_finest), "iso1mm and iso_finest are mutually exclusive");
        let iso = o.iso1mm || o.iso_finest;
        let h = if o.iso_finest { vs[0].min(vs[1]).min(vs[2]) } else { 1.0 };
        let (ne, ni, vw, pe, me, pi) = if iso {
            let ni = [0, 1, 2].map(|d| if vs[d] == h { np[d] } else { (np[d] as f64 * vs[d] / h).ceil() as usize });
            let fwd = [0, 1, 2].map(|d| (1..=ni[d]).map(|q| if vs[d] == h { q as f64 } else { q as f64 * h / vs[d] }).collect::<Vec<f64>>());
            let pi = trilinear(&p, np, &fwd);
            let mi = trilinear(&m, np, &fwd);
            let ne = ni.map(|n| n + n % 2);
            let mut pe = vec![0.0; ne[0] * ne[1] * ne[2]];
            let mut me = vec![0.0; pe.len()];
            for k in 0..ni[2] { for j in 0..ni[1] { for i in 0..ni[0] {
                let s = i + j * ni[0] + k * ni[0] * ni[1];
                let d = i + j * ne[0] + k * ne[0] * ne[1];
                pe[d] = pi[s];
                me[d] = if mi[s] > 0.5 { 1.0 } else { 0.0 };
            }}}
            (ne, ni, [h; 3], pe, me, pi)
        } else {
            (np, np, vs, p.clone(), m.clone(), p.clone())
        };
        let _ = pi;
        let nw = ne[0] * ne[1] * ne[2];

        // radii, largest first; each voxel keeps the first (largest) that fits
        let rp = VsharpParams { threshold: o.threshold, max_radius: o.max_radius, min_radius: o.min_radius, deconvolve: o.deconvolve };
        let mut radii: Vec<f64> = if o.sti_radii { sti_radii(&rp) } else { vsharp_radii(&rp) };
        radii.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let mut p_fft = to_complex(&pe);
        fft3d(&mut p_fft, ne[0], ne[1], ne[2]);
        let mut m_fft = to_complex(&me);
        fft3d(&mut m_fft, ne[0], ne[1], ne[2]);
        let mut bg = pe.clone();
        let mut nm = vec![0.0; nw];
        let mut buf = vec![Complex64::new(0.0, 0.0); nw];
        let mut inv: Option<Vec<f64>> = None;
        let delta = 1.0 - 1e-7_f64.sqrt();
        for &r in &radii {
            let (mut k_fft, count) = kernel_fft(ne, vw, r, o.ball_plus, o.ball_no_centre);
            if count == 0.0 { continue; }
            for i in 0..nw { buf[i] = m_fft[i] * k_fft[i]; }
            ifft3d(&mut buf, ne[0], ne[1], ne[2]);
            let fits: Vec<bool> = (0..nw).map(|i| me[i] > 0.0 && if o.strict_fit { buf[i].re >= count - 0.5 } else { buf[i].re / count > delta }).collect();
            for v in k_fft.iter_mut() { *v /= count; }
            if inv.is_none() {
                inv = Some(k_fft.iter().map(|s| {
                    let oms = 1.0 - s.re;
                    if oms.abs() < o.threshold { 0.0 } else { 1.0 / oms }
                }).collect());
            }
            for i in 0..nw { buf[i] = p_fft[i] * k_fft[i]; }
            ifft3d(&mut buf, ne[0], ne[1], ne[2]);
            for i in 0..nw {
                if fits[i] && nm[i] == 0.0 {
                    bg[i] = buf[i].re;
                    nm[i] = 1.0;
                }
            }
        }
        // local on W, optional deconvolution on W
        let mut l: Vec<Complex64> = (0..nw).map(|i| Complex64::new((pe[i] - bg[i]) * nm[i], 0.0)).collect();
        if o.deconvolve {
            if let Some(inv) = inv {
                fft3d(&mut l, ne[0], ne[1], ne[2]);
                for i in 0..nw { l[i] *= inv[i]; }
                ifft3d(&mut l, ne[0], ne[1], ne[2]);
                for i in 0..nw { l[i] = Complex64::new(l[i].re * nm[i], 0.0); }
            }
        }

        // back to native
        let (loc_n, nm_n): (Vec<f64>, Vec<f64>) = if iso {
            let mut bg_c = vec![0.0; ni[0] * ni[1] * ni[2]];
            let mut nm_c = vec![0.0; bg_c.len()];
            for k in 0..ni[2] { for j in 0..ni[1] { for i in 0..ni[0] {
                let d = i + j * ni[0] + k * ni[0] * ni[1];
                let s = i + j * ne[0] + k * ne[0] * ne[1];
                bg_c[d] = if o.deconvolve { pe[s] - l[s].re } else { bg[s] };
                nm_c[d] = nm[s];
            }}}
            let back = [0, 1, 2].map(|d| (1..=np[d]).map(|a| if vs[d] == h { a as f64 } else { a as f64 * vs[d] / h }).collect::<Vec<f64>>());
            let bg_n = trilinear(&bg_c, ni, &back);
            let nm_n = trilinear(&nm_c, ni, &back);
            ((0..p.len()).map(|s| p[s] - bg_n[s]).collect(), nm_n)
        } else {
            ((0..nw).map(|i| l[i].re).collect(), nm)
        };
        for k in lo[2]..=hi[2] { for j in lo[1]..=hi[1] { for i in lo[0]..=hi[0] {
            let s = (i - lo[0] + pad) + (j - lo[1] + pad) * np[0] + (k - lo[2] + pad) * np[0] * np[1];
            if nm_n[s] > 0.5 {
                let d = i + j * nx + k * nx * ny;
                local_out[d] = loc_n[s];
                mask_out[d] = 1;
            }
        }}}
        (local_out, mask_out)
    }
}

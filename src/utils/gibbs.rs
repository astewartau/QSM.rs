//! Gibbs-ringing removal via local subvoxel shifts (Kellner et al. 2016).
//!
//! Truncating k-space (finite acquisition matrix) produces oscillatory ringing
//! near sharp edges. For multi-echo data the ringing pattern differs per echo
//! (image contrast changes with TE), so it perturbs the mono-exponential decay
//! and biases R2*/R2 maps — hence unringing is recommended before relaxometry.
//!
//! Method (Kellner 2016): resample the image at a set of subvoxel shifts using
//! the Fourier shift theorem; at each voxel pick the shift that minimises local
//! total variation (the "ring-free" sampling), then interpolate back to the
//! grid. The 1-D correction is applied along each axis and the axis-corrected
//! volumes are combined in the Fourier domain, weighting each axis where its
//! ringing dominates (low spatial frequency in the other axes).
//!
//! This is a faithful port of DIPY's `gibbs_removal` (Kellner 2016; Neto
//! Henriques 2018), generalised from 2-D slice-wise to full 3-D: the cosine
//! weight for axis *i* is `∏_{j≠i}(1+cos k_j) / Σ_m ∏_{j≠m}(1+cos k_j)`, which
//! reduces exactly to DIPY's 2-D weights.

use crate::fft::{fft3d_real, ifft3d_real};
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
fn tv_min_line(row: &[f64], n: usize, n_points: usize, out: &mut [f64]) {
    for i in 0..n {
        let mut ptv = 0.0;
        let mut ntv = 0.0;
        for o in 0..n_points {
            let a = (i + o) % n;
            let b = (i + o + 1) % n;
            ptv += (row[a] - row[b]).abs();
            let c = (i as isize - o as isize).rem_euclid(n as isize) as usize;
            let d = (i as isize - o as isize - 1).rem_euclid(n as isize) as usize;
            ntv += (row[c] - row[d]).abs();
        }
        out[i] = ptv.min(ntv);
    }
}

/// Un-ring a single 1-D line (Kellner subvoxel shift). `freq[m] = fftfreq(n)[m]`.
/// Reusable per-thread scratch for [`unring_row_inplace`], sized to a line of
/// length `n`. Allocated once per worker thread (via rayon `for_each_init`) so the
/// hot loop performs no allocation.
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
    scratch_f: Vec<Complex64>,
    scratch_i: Vec<Complex64>,
}

impl LineBufs {
    fn new(n: usize, sf: usize, si: usize) -> Self {
        let z = Complex64::new(0.0, 0.0);
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
            scratch_f: vec![z; sf],
            scratch_i: vec![z; si],
        }
    }
}

/// Un-ring a single 1-D line in place, reusing `b`'s buffers. `phases[s*n + m]`
/// is the precomputed positive-shift Fourier factor `exp(i·2π·freq[m]·ssamp[s])`
/// (the negative shift is its conjugate) — identical values to computing the
/// phase per line, just hoisted out of the hot loop.
#[allow(clippy::too_many_arguments)]
fn unring_row_inplace(
    row: &mut [f64],
    b: &mut LineBufs,
    n: usize,
    n_points: usize,
    ssamp: &[f64],
    phases: &[Complex64],
    fft: &Arc<dyn Fft<f64>>,
    ifft: &Arc<dyn Fft<f64>>,
) {
    // Skip near-constant/empty lines (nothing to unring).
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in row.iter() {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if hi - lo < 1e-12 {
        return;
    }

    let LineBufs {
        c,
        buf,
        tvp,
        tvn,
        isp,
        isn,
        sp,
        sn,
        img,
        tvs,
        scratch_f,
        scratch_i,
    } = b;

    for m in 0..n {
        c[m] = Complex64::new(row[m], 0.0);
    }
    fft.process_with_scratch(c, scratch_f);

    tv_min_line(row, n, n_points, tvp);
    tvn.copy_from_slice(tvp);
    isp.copy_from_slice(row);
    isn.copy_from_slice(row);
    sp.iter_mut().for_each(|x| *x = 0.0);
    sn.iter_mut().for_each(|x| *x = 0.0);
    let inv_n = 1.0 / n as f64;

    for (s_i, &s) in ssamp.iter().enumerate() {
        let ph = &phases[s_i * n..s_i * n + n];

        // Positive shift.
        for m in 0..n {
            buf[m] = c[m] * ph[m];
        }
        ifft.process_with_scratch(buf, scratch_i);
        for m in 0..n {
            img[m] = buf[m].norm() * inv_n;
        }
        tv_min_line(img, n, n_points, tvs);
        for i in 0..n {
            if tvp[i] > tvs[i] {
                isp[i] = img[i];
                sp[i] = s;
                tvp[i] = tvs[i];
            }
        }

        // Negative shift (conjugate phase).
        for m in 0..n {
            buf[m] = c[m] * ph[m].conj();
        }
        ifft.process_with_scratch(buf, scratch_i);
        for m in 0..n {
            img[m] = buf[m].norm() * inv_n;
        }
        tv_min_line(img, n, n_points, tvs);
        for i in 0..n {
            if tvn[i] > tvs[i] {
                isn[i] = img[i];
                sn[i] = s;
                tvn[i] = tvs[i];
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

/// Un-ring `vol` along a single axis (0=x, 1=y, 2=z), returning a new volume.
///
/// Lines are gathered into a contiguous `(num_lines × n)` matrix so they can be
/// processed in place with `par_chunks_mut` — each worker thread reuses one
/// [`LineBufs`] (no per-line allocation) and the shift phase factors are
/// precomputed once for the whole axis.
fn unring_axis(vol: &[f64], (nx, ny, nz): (usize, usize, usize), axis: usize) -> Vec<f64> {
    let n = [nx, ny, nz][axis];
    if n < 4 {
        return vol.to_vec(); // too short to unring
    }
    let freq = fftfreq(n);
    let ssamp: Vec<f64> = (0..N_SHIFTS)
        .map(|i| 0.02 + (0.9 - 0.02) * i as f64 / (N_SHIFTS - 1) as f64)
        .collect();
    // Precompute the positive-shift Fourier factors once (line-independent).
    let mut phases = vec![Complex64::new(0.0, 0.0); ssamp.len() * n];
    for (s_i, &s) in ssamp.iter().enumerate() {
        for m in 0..n {
            phases[s_i * n + m] = Complex64::from_polar(1.0, 2.0 * PI * freq[m] * s);
        }
    }
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);
    let sf = fft.get_inplace_scratch_len();
    let si = ifft.get_inplace_scratch_len();

    // Line start offsets + stride for this axis.
    let (stride, lines): (usize, Vec<usize>) = match axis {
        0 => (1, (0..ny * nz).map(|l| l * nx).collect()),
        1 => {
            let mut v = Vec::with_capacity(nx * nz);
            for z in 0..nz {
                for x in 0..nx {
                    v.push(z * nx * ny + x);
                }
            }
            (nx, v)
        }
        _ => {
            let mut v = Vec::with_capacity(nx * ny);
            for y in 0..ny {
                for x in 0..nx {
                    v.push(y * nx + x);
                }
            }
            (nx * ny, v)
        }
    };

    // Gather strided lines into a contiguous matrix (num_lines × n).
    let num_lines = lines.len();
    let mut mat = vec![0.0_f64; num_lines * n];
    for (l, &start) in lines.iter().enumerate() {
        let dst = &mut mat[l * n..l * n + n];
        for (i, d) in dst.iter_mut().enumerate() {
            *d = vol[start + i * stride];
        }
    }

    // Process each row in place (buffers reused per worker thread).
    #[cfg(feature = "parallel")]
    mat.par_chunks_mut(n).for_each_init(
        || LineBufs::new(n, sf, si),
        |b, row| unring_row_inplace(row, b, n, 3, &ssamp, &phases, &fft, &ifft),
    );
    #[cfg(not(feature = "parallel"))]
    {
        let mut b = LineBufs::new(n, sf, si);
        for row in mat.chunks_mut(n) {
            unring_row_inplace(row, &mut b, n, 3, &ssamp, &phases, &fft, &ifft);
        }
    }

    // Scatter back to a new volume.
    let mut out = vol.to_vec();
    for (l, &start) in lines.iter().enumerate() {
        let src = &mat[l * n..l * n + n];
        for (i, &s) in src.iter().enumerate() {
            out[start + i * stride] = s;
        }
    }
    out
}

/// Gibbs-unring a single 3-D volume.
pub fn gibbs_unring_volume(vol: &[f64], dims: (usize, usize, usize)) -> Vec<f64> {
    let (nx, ny, nz) = dims;
    let n_vox = nx * ny * nz;
    assert_eq!(vol.len(), n_vox, "vol length must be nx*ny*nz");

    let cx = unring_axis(vol, dims, 0);
    let cy = unring_axis(vol, dims, 1);
    let cz = unring_axis(vol, dims, 2);

    // Fourier-domain weighted combine (Kellner cosine weights, 3-D form).
    let fx = fftfreq(nx);
    let fy = fftfreq(ny);
    let fz = fftfreq(nz);
    let cosk = |f: f64| 1.0 + (2.0 * PI * f).cos();
    let ckx: Vec<f64> = fx.iter().map(|&f| cosk(f)).collect();
    let cky: Vec<f64> = fy.iter().map(|&f| cosk(f)).collect();
    let ckz: Vec<f64> = fz.iter().map(|&f| cosk(f)).collect();

    let sx = fft3d_real(&cx, nx, ny, nz);
    let sy = fft3d_real(&cy, nx, ny, nz);
    let sz = fft3d_real(&cz, nx, ny, nz);
    let mut spec = vec![Complex64::new(0.0, 0.0); n_vox];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = x + y * nx + z * nx * ny;
                let wx = cky[y] * ckz[z]; // axis-x correction weight
                let wy = ckx[x] * ckz[z];
                let wz = ckx[x] * cky[y];
                let den = wx + wy + wz;
                spec[idx] = if den > 1e-12 {
                    (sx[idx] * wx + sy[idx] * wy + sz[idx] * wz) / den
                } else {
                    (sx[idx] + sy[idx] + sz[idx]) / 3.0
                };
            }
        }
    }
    let mut out = ifft3d_real(&spec, nx, ny, nz);
    for v in out.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    out
}

/// Gibbs-unring multi-volume data (e.g. multi-echo magnitude).
///
/// # Arguments
/// * `data` - Interleaved `[voxel0_vol0, voxel0_vol1, ..., voxel1_vol0, ...]`
///   (row-major `(n_voxels, n_vols)`), same layout as [`r2star_arlo`].
/// * `dims` - Volume dimensions `(nx, ny, nz)`.
/// * `n_vols` - Number of volumes.
///
/// # Returns
/// Unrung data in the same layout.
pub fn gibbs_unring(data: &[f64], dims: (usize, usize, usize), n_vols: usize) -> Vec<f64> {
    let (nx, ny, nz) = dims;
    let n_vox = nx * ny * nz;
    assert_eq!(data.len(), n_vox * n_vols, "data length must be n_voxels * n_vols");
    let mut out = data.to_vec();
    let mut vol = vec![0.0_f64; n_vox];
    for e in 0..n_vols {
        for v in 0..n_vox {
            vol[v] = data[v * n_vols + e];
        }
        let un = gibbs_unring_volume(&vol, dims);
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
pub fn gibbs_unring_masked(
    data: &[f64],
    dims: (usize, usize, usize),
    n_vols: usize,
    mask: &[u8],
    margin: usize,
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
        return gibbs_unring(data, dims, n_vols);
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
        let un = gibbs_unring_volume(&sub, (cx, cy, cz));
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

        // Faithful-port check: values must match DIPY's `_gibbs_removal_1d` on the
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

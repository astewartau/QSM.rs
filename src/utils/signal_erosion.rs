//! Signal-gated mask erosion.
//!
//! Near the sinuses and skull base, T2* dropout leaves brain-mask voxels with no usable signal.
//! The field estimate there is unreliable and leaks background-field artefact into the χ map. A
//! global erosion suppresses it only by peeling healthy cortex just as hard. This erosion instead
//! strips mask **boundary** voxels whose (coil-debiased) magnitude is low, iteratively, so it eats
//! inward through dropout territory and stops as soon as it reaches real signal. Dark *interior*
//! structures (veins, iron-rich deep grey matter, haemorrhage) are never touched because they are
//! not reachable from the boundary through low-signal voxels within the depth cap.
//!
//! Works on any binary mask (BET, thresholding, a CNN mask, …) — it is a mask refinement, not
//! part of any one brain-extraction method.
//!
//! Port of `_signal_erode` from the QSM-CI `hd-bet-qsmci` submission (`extract.py`), used for the
//! QSM-CI harmonization track at `erode_global=1, erode_threshold=0.80, erode_depth_cap=5`
//! (the [`SignalErosionParams::default`]). Distances and smoothing scales are in **voxels**, as in
//! the reference implementation.

use crate::utils::bias_correction::fill_holes;
use crate::utils::mask::erode_mask;
use crate::Grid;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Signal-gated erosion parameters. The default is the QSM-CI harmonization setting.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SignalErosionParams {
    /// Signal gate as a fraction of the in-mask median of the coil-debiased magnitude
    /// (default 0.80). Above ~0.85 it starts over-carving.
    pub threshold: f64,
    /// Never remove a voxel more than this many voxels below the original mask surface
    /// (default 5). Bounds the erosion by construction: without it, the peel follows low-signal
    /// sulcal and interhemispheric CSF deep into healthy cortex. `0` disables the cap.
    pub depth_cap: usize,
    /// Plain 6-connected erosions applied before the signal gate (default 1). Trims the bright
    /// one-voxel skull/CSF sliver a mask often carries, which no signal gate can catch.
    pub global_erosions: usize,
    /// Gaussian scale in voxels of the receive-coil bias estimate divided out of the magnitude
    /// (default 12).
    pub bias_sigma: f64,
    /// Keep every connected component with at least this many voxels, not just the largest
    /// (default 1000): a capped peel can sever small real pieces off thin structures.
    pub min_component: usize,
}

impl Default for SignalErosionParams {
    fn default() -> Self {
        Self { threshold: 0.80, depth_cap: 5, global_erosions: 1, bias_sigma: 12.0, min_component: 1000 }
    }
}

/// Erode `mask` only where the signal is low (see the module docs).
///
/// `magnitude` should be a combined magnitude (e.g. the root-sum-of-squares over echoes) on the
/// same grid. Returns the refined mask (0/1).
pub fn signal_gated_erosion(
    mask: &[u8],
    magnitude: &[f64],
    grid: &Grid,
    params: &SignalErosionParams,
) -> Vec<u8> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    assert_eq!(mask.len(), n, "mask length does not match grid");
    assert_eq!(magnitude.len(), n, "magnitude length does not match grid");
    let inside: Vec<bool> = mask.iter().map(|&m| m != 0).collect();
    if !inside.iter().any(|&b| b) {
        return mask.to_vec();
    }

    // Divide out the receive-coil sensitivity profile: the magnitude over its own mask-normalised
    // smoothed self, so the estimate is not dragged down by the zeros outside the brain.
    let sig = gaussian_filter(magnitude, grid.dims, 1.0);
    let masked: Vec<f64> = sig.iter().zip(&inside).map(|(&s, &m)| if m { s } else { 0.0 }).collect();
    let num = gaussian_filter(&masked, grid.dims, params.bias_sigma);
    let mask_f: Vec<f64> = inside.iter().map(|&m| m as u8 as f64).collect();
    let den = gaussian_filter(&mask_f, grid.dims, params.bias_sigma);
    let rel: Vec<f64> = (0..n).map(|i| sig[i] / (num[i] / den[i].max(1e-6)).max(1e-6)).collect();

    let mut in_mask: Vec<f64> = (0..n).filter(|&i| inside[i]).map(|i| rel[i]).collect();
    let gate = params.threshold * median(&mut in_mask);

    // Depth below the ORIGINAL surface, on a padded mask so a brain touching the FOV face
    // (common inferiorly — exactly where this matters) is measured as surface, not interior.
    let depth = (params.depth_cap > 0).then(|| distance_transform_padded(&inside, grid.dims));
    let mut current = if params.global_erosions > 0 {
        erode_mask(mask, grid, params.global_erosions)
    } else {
        mask.iter().map(|&m| (m != 0) as u8).collect()
    };
    let candidates: Vec<usize> = (0..n)
        .filter(|&i| current[i] != 0 && rel[i] < gate)
        .filter(|&i| depth.as_ref().is_none_or(|d| d[i] <= params.depth_cap as f64))
        .collect();

    // Synchronous peel: each pass removes every low-signal candidate that is on the boundary of
    // the current mask (6-connected, volume edge counts as outside), exactly like
    // `eroded & ~binary_erosion(eroded) & low` per pass in the reference.
    let passes = (params.depth_cap + 2).max(40);
    let nxy = nx * ny;
    for _ in 0..passes {
        let drop: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|&i| current[i] != 0)
            .filter(|&i| {
                let (x, y, z) = (i % nx, (i / nx) % ny, i / nxy);
                x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1
                    || current[i - 1] == 0 || current[i + 1] == 0
                    || current[i - nx] == 0 || current[i + nx] == 0
                    || current[i - nxy] == 0 || current[i + nxy] == 0
            })
            .collect();
        if drop.is_empty() {
            break;
        }
        for i in drop {
            current[i] = 0;
        }
    }

    // Never leave an enclosed cavity, then keep every component of at least `min_component`
    // voxels (or the largest, if none is that big).
    let filled = fill_holes(&current, grid, n);
    keep_large_components(&filled, grid.dims, params.min_component)
}

/// Median with numpy semantics (mean of the two middle values for an even count).
fn median(v: &mut [f64]) -> f64 {
    let len = v.len();
    let mid = len / 2;
    let (_, &mut hi, _) = v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    if len % 2 == 1 {
        hi
    } else {
        let lo = v[..mid].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lo + hi)
    }
}

/// `scipy.ndimage` `reflect` boundary (half-sample symmetric: `d c b a | a b c d | d c b a`),
/// valid for any offset, including kernels longer than the axis.
#[inline]
fn reflect(i: isize, n: usize) -> usize {
    let n = n as isize;
    let m = i.rem_euclid(2 * n);
    (if m >= n { 2 * n - 1 - m } else { m }) as usize
}

/// Separable Gaussian filter matching `scipy.ndimage.gaussian_filter(x, sigma)` (voxel units,
/// `truncate=4.0`, `mode='reflect'`) on a column-major `(nx, ny, nz)` volume.
pub(crate) fn gaussian_filter(data: &[f64], dims: (usize, usize, usize), sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        return data.to_vec();
    }
    let radius = (4.0 * sigma + 0.5) as isize;
    let mut w: Vec<f64> = (-radius..=radius).map(|x| (-0.5 * (x * x) as f64 / (sigma * sigma)).exp()).collect();
    let s: f64 = w.iter().sum();
    w.iter_mut().for_each(|v| *v /= s);
    let (nx, ny, nz) = dims;
    let nxy = nx * ny;

    // x: along contiguous rows.
    let mut a = vec![0.0; data.len()];
    crate::maybe_par_chunks_mut!(a, nx).enumerate().for_each(|(r, row)| {
        let src = &data[r * nx..(r + 1) * nx];
        for (x, out) in row.iter_mut().enumerate() {
            *out = w.iter().enumerate().map(|(t, &wt)| wt * src[reflect(x as isize + t as isize - radius, nx)]).sum();
        }
    });
    // y: within each z-slice, weighted sum of whole rows.
    let mut b = vec![0.0; data.len()];
    crate::maybe_par_chunks_mut!(b, nxy).enumerate().for_each(|(z, slab)| {
        let src = &a[z * nxy..(z + 1) * nxy];
        for y in 0..ny {
            let out = &mut slab[y * nx..(y + 1) * nx];
            for (t, &wt) in w.iter().enumerate() {
                let sy = reflect(y as isize + t as isize - radius, ny);
                for (o, &v) in out.iter_mut().zip(&src[sy * nx..(sy + 1) * nx]) {
                    *o += wt * v;
                }
            }
        }
    });
    // z: each output slice is a weighted sum of whole input slices.
    let mut c = vec![0.0; data.len()];
    crate::maybe_par_chunks_mut!(c, nxy).enumerate().for_each(|(z, slab)| {
        for (t, &wt) in w.iter().enumerate() {
            let sz = reflect(z as isize + t as isize - radius, nz);
            for (o, &v) in slab.iter_mut().zip(&b[sz * nxy..(sz + 1) * nxy]) {
                *o += wt * v;
            }
        }
    });
    c
}

/// Exact Euclidean distance (voxel units) from each `inside` voxel to the nearest outside voxel,
/// with the volume padded by one outside voxel on every face — i.e.
/// `distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1, 1:-1]`. Outside voxels get 0.
pub(crate) fn distance_transform_padded(inside: &[bool], dims: (usize, usize, usize)) -> Vec<f64> {
    let (nx, ny, nz) = dims;
    let (px, py, pz) = (nx + 2, ny + 2, nz + 2);
    const FAR: f64 = 1e20;
    let mut f = vec![0.0; px * py * pz];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if inside[x + nx * (y + ny * z)] {
                    f[(x + 1) + px * ((y + 1) + py * (z + 1))] = FAR;
                }
            }
        }
    }
    // Felzenszwalb–Huttenlocher squared EDT, one axis at a time.
    let (mut line, mut out) = (Vec::new(), Vec::new());
    let mut pass = |f: &mut [f64], base: usize, stride: usize, len: usize| {
        line.clear();
        line.extend((0..len).map(|i| f[base + i * stride]));
        edt_1d(&line, &mut out);
        for (i, &v) in out.iter().enumerate() {
            f[base + i * stride] = v;
        }
    };
    for z in 0..pz {
        for y in 0..py {
            pass(&mut f, px * (y + py * z), 1, px);
        }
    }
    for z in 0..pz {
        for x in 0..px {
            pass(&mut f, x + px * py * z, px, py);
        }
    }
    for y in 0..py {
        for x in 0..px {
            pass(&mut f, x + px * y, px * py, pz);
        }
    }
    let mut d = vec![0.0; nx * ny * nz];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = x + nx * (y + ny * z);
                if inside[i] {
                    d[i] = f[(x + 1) + px * ((y + 1) + py * (z + 1))].sqrt();
                }
            }
        }
    }
    d
}

/// 1D squared distance transform of sampled function `f` (Felzenszwalb & Huttenlocher 2012).
fn edt_1d(f: &[f64], d: &mut Vec<f64>) {
    let n = f.len();
    let mut v = vec![0usize; n];
    let mut z = vec![0.0f64; n + 1];
    let mut k = 0usize;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    let inter = |q: usize, p: usize| {
        ((f[q] + (q * q) as f64) - (f[p] + (p * p) as f64)) / (2.0 * q as f64 - 2.0 * p as f64)
    };
    for q in 1..n {
        let mut s = inter(q, v[k]);
        while s <= z[k] {
            k -= 1;
            s = inter(q, v[k]);
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    d.clear();
    d.resize(n, 0.0);
    k = 0;
    for (q, dq) in d.iter_mut().enumerate() {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let dx = q as f64 - v[k] as f64;
        *dq = dx * dx + f[v[k]];
    }
}

/// Keep the 6-connected foreground components of at least `min(min_size, largest)` voxels.
fn keep_large_components(mask: &[u8], dims: (usize, usize, usize), min_size: usize) -> Vec<u8> {
    let (nx, ny, nz) = dims;
    let nxy = nx * ny;
    let mut label = vec![0u32; mask.len()];
    let mut sizes = vec![0usize]; // label 0 = background
    let mut stack = Vec::new();
    for seed in 0..mask.len() {
        if mask[seed] == 0 || label[seed] != 0 {
            continue;
        }
        let id = sizes.len() as u32;
        let mut size = 0;
        label[seed] = id;
        stack.push(seed);
        while let Some(i) = stack.pop() {
            size += 1;
            let (x, y, z) = (i % nx, (i / nx) % ny, i / nxy);
            let mut visit = |j: usize| {
                if mask[j] != 0 && label[j] == 0 {
                    label[j] = id;
                    stack.push(j);
                }
            };
            if x > 0 { visit(i - 1); }
            if x + 1 < nx { visit(i + 1); }
            if y > 0 { visit(i - nx); }
            if y + 1 < ny { visit(i + nx); }
            if z > 0 { visit(i - nxy); }
            if z + 1 < nz { visit(i + nxy); }
        }
        sizes.push(size);
    }
    let largest = sizes.iter().copied().max().unwrap_or(0);
    let keep = min_size.min(largest);
    label.iter().map(|&l| (l != 0 && sizes[l as usize] >= keep) as u8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: (usize, usize, usize)) -> Grid {
        Grid::new(n.0, n.1, n.2, 1.0, 1.0, 1.0)
    }

    fn idx(x: usize, y: usize, z: usize, n: (usize, usize, usize)) -> usize {
        x + n.0 * (y + n.1 * z)
    }

    #[test]
    fn edt_matches_brute_force() {
        let dims = (9, 7, 6);
        // Irregular blob that touches the x=0 face (so the padding matters).
        let inside: Vec<bool> = (0..dims.0 * dims.1 * dims.2)
            .map(|i| {
                let (x, y, z) = (i % dims.0, (i / dims.0) % dims.1, i / (dims.0 * dims.1));
                (x as f64 - 2.0).powi(2) / 9.0 + (y as f64 - 3.0).powi(2) / 6.0 + (z as f64 - 2.5).powi(2) / 5.0 < 1.6
                    && (x + y + z) % 7 != 3
            })
            .collect();
        let d = distance_transform_padded(&inside, dims);
        let (px, py, pz) = (dims.0 as isize + 2, dims.1 as isize + 2, dims.2 as isize + 2);
        let is_in = |x: isize, y: isize, z: isize| {
            x >= 1 && y >= 1 && z >= 1 && x <= dims.0 as isize && y <= dims.1 as isize && z <= dims.2 as isize
                && inside[idx((x - 1) as usize, (y - 1) as usize, (z - 1) as usize, dims)]
        };
        for z in 0..dims.2 {
            for y in 0..dims.1 {
                for x in 0..dims.0 {
                    let i = idx(x, y, z, dims);
                    let mut best = f64::INFINITY;
                    if inside[i] {
                        for qz in 0..pz {
                            for qy in 0..py {
                                for qx in 0..px {
                                    if !is_in(qx, qy, qz) {
                                        let dd = ((qx - x as isize - 1).pow(2) + (qy - y as isize - 1).pow(2)
                                            + (qz - z as isize - 1).pow(2)) as f64;
                                        best = best.min(dd);
                                    }
                                }
                            }
                        }
                        assert!((d[i] - best.sqrt()).abs() < 1e-9, "voxel {:?}: {} vs {}", (x, y, z), d[i], best.sqrt());
                    } else {
                        assert_eq!(d[i], 0.0);
                    }
                }
            }
        }
    }

    #[test]
    fn gaussian_matches_direct_reflect_convolution() {
        let dims = (7, 5, 4);
        let data: Vec<f64> = (0..140).map(|i| ((i * 37) % 11) as f64 - 3.0).collect();
        // sigma 1.5 → radius 6, longer than every axis, exercising repeated reflection.
        let sigma = 1.5;
        let got = gaussian_filter(&data, dims, sigma);
        let r = (4.0 * sigma + 0.5) as isize;
        let w: Vec<f64> = (-r..=r).map(|t| (-0.5 * (t * t) as f64 / (sigma * sigma)).exp()).collect();
        let s: f64 = w.iter().sum();
        for z in 0..dims.2 {
            for y in 0..dims.1 {
                for x in 0..dims.0 {
                    let mut acc = 0.0;
                    for (a, wa) in w.iter().enumerate() {
                        for (b, wb) in w.iter().enumerate() {
                            for (c, wc) in w.iter().enumerate() {
                                let sx = reflect(x as isize + a as isize - r, dims.0);
                                let sy = reflect(y as isize + b as isize - r, dims.1);
                                let sz = reflect(z as isize + c as isize - r, dims.2);
                                acc += wa * wb * wc * data[idx(sx, sy, sz, dims)];
                            }
                        }
                    }
                    let want = acc / (s * s * s);
                    assert!((got[idx(x, y, z, dims)] - want).abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn reflect_is_half_sample_symmetric() {
        let n = 4;
        let got: Vec<usize> = (-5..9).map(|i| reflect(i, n)).collect();
        assert_eq!(got, vec![3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0]);
    }

    #[test]
    fn median_matches_numpy() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
    }

    /// Ball with uniform signal except (a) a dark wedge reaching in from the -x surface (dropout)
    /// and (b) a dark enclosed core (a vein / iron-rich nucleus). The gate should peel the wedge,
    /// only down to the depth cap, and leave the dark core and the healthy surface alone.
    #[test]
    fn peels_boundary_dropout_but_not_interior_dark_structures() {
        let dims = (40, 40, 40);
        let g = grid(dims);
        let c = 19.5;
        let n = dims.0 * dims.1 * dims.2;
        let mut mask = vec![0u8; n];
        // Bright outside too (scalp), so a healthy edge isn't dimmed by partial-volume smoothing.
        let mut mag = vec![100.0; n];
        for z in 0..dims.2 {
            for y in 0..dims.1 {
                for x in 0..dims.0 {
                    let (dx, dy, dz) = (x as f64 - c, y as f64 - c, z as f64 - c);
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();
                    let i = idx(x, y, z, dims);
                    if r < 16.0 {
                        mask[i] = 1;
                        if dx < -6.0 && dy.abs() < 5.0 && dz.abs() < 5.0 {
                            mag[i] = 2.0; // dropout wedge from the -x surface, ~10 voxels deep
                        }
                        if r < 3.0 {
                            mag[i] = 2.0; // dark interior structure
                        }
                    }
                }
            }
        }
        let params = SignalErosionParams { global_erosions: 0, min_component: 10, ..Default::default() };
        let out = signal_gated_erosion(&mask, &mag, &g, &params);
        let depth = distance_transform_padded(&mask.iter().map(|&m| m != 0).collect::<Vec<_>>(), dims);

        // Interior dark core untouched.
        assert_eq!(out[idx(20, 20, 20, dims)], 1);
        // Healthy +x surface untouched.
        assert_eq!(out[idx(35, 20, 20, dims)], 1);
        // Dropout removed at the surface…
        assert_eq!(out[idx(5, 20, 20, dims)], 0);
        // …but never deeper than the cap.
        for i in 0..n {
            if mask[i] == 1 && out[i] == 0 {
                assert!(depth[i] <= params.depth_cap as f64 + 1e-9, "removed voxel at depth {}", depth[i]);
            }
        }
        // No enclosed cavities.
        assert_eq!(fill_holes(&out, &g, n), out);
    }

    #[test]
    fn uniform_signal_only_applies_global_erosion() {
        let dims = (20, 20, 20);
        let g = grid(dims);
        let mask: Vec<u8> = (0..8000)
            .map(|i| {
                let (x, y, z) = (i % 20, (i / 20) % 20, i / 400);
                (3..17).contains(&x) && (3..17).contains(&y) && (3..17).contains(&z)
            } as u8)
            .collect();
        let mag = vec![50.0; 8000];
        let params = SignalErosionParams { min_component: 1, ..Default::default() };
        let out = signal_gated_erosion(&mask, &mag, &g, &params);
        assert_eq!(out, erode_mask(&mask, &g, 1));
    }
}

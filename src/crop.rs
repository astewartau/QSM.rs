//! Cropping a volume to the region that actually carries signal, and putting it back.
//!
//! Every FFT-based stage — background removal, dipole inversion, Laplacian unwrapping — costs
//! `O(N log N)` in the *whole grid*, not in the brain. A brain typically fills a fifth of an
//! acquired volume and a tenth of one that has been resampled to a cardinal grid (resampling
//! wraps a box around a tilted slab, so half the result is empty corner). Reconstructing inside a
//! box around the mask and padding the answer back afterwards is therefore close to free, and on
//! a UK Biobank SWI it takes the resampled grid from 4.5× the cost of the acquired grid down to
//! parity.
//!
//! Two things make the difference, and they are worth separating because their risk differs:
//!
//! - **Fewer voxels** ([`crop_box_for_mask`]). Mask bounding box plus a margin, rather than the
//!   full field of view. This moves the FFT's periodic boundary *closer* to the object.
//! - **Sizes an FFT likes** ([`fft_pad_box`]). An extent with a large prime factor pushes
//!   `rustfft` onto Bluestein's algorithm. An axially-resampled UK Biobank grid comes out
//!   272×339×77 — that is 2⁴·17, 3·113 and 7·11, awkward on every axis, which is what resampling
//!   to a bounding box tends to produce. Padding to 280×343×80 costs 8% more voxels and takes the
//!   transform from 131 ms to 71 ms, a 1.85× speedup. Padding moves the boundary *further* from
//!   the object, so unlike cropping it carries no wrap-around risk.
//!
//! Both are expressed as a [`CropBox`], which may sit inside the grid (cropping), extend beyond
//! it (padding), or do both on different axes.
//!
//! ## This changes the numbers, not just the speed
//!
//! This caveat applies to **cropping**, not to padding. FFT-based reconstruction is periodic, so
//! moving the boundary closer to the object brings wrap-around with it, and the dipole kernel has
//! infinite support. Measured on a UK Biobank acquisition, a crop that actually removed voxels
//! changed χ by ~0.6% of its dynamic range at the median and ~4% at the 99th percentile.
//! [`margin_voxels`] takes the margin in **millimetres** so anisotropic voxels get a
//! geometrically equal margin on every side, but a caller should still validate a cropped
//! reconstruction against an uncropped one rather than assume the two agree.
//!
//! [`fft_pad_box`] has no such caveat: it discards nothing and only moves the boundary outward.

/// A box within a larger grid: where reconstruction actually happens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropBox {
    /// Index of the box's first voxel in the full grid. **May be negative**, which means the box
    /// extends past the edge and those voxels are padding rather than data.
    pub origin: (isize, isize, isize),
    /// Size of the box.
    pub dims: (usize, usize, usize),
    /// Size of the grid the box sits in.
    pub full_dims: (usize, usize, usize),
}

impl CropBox {
    /// A box covering the whole grid — cropping is then a no-op.
    pub fn full(full_dims: (usize, usize, usize)) -> Self {
        Self { origin: (0, 0, 0), dims: full_dims, full_dims }
    }

    /// Whether any axis reaches outside the grid, i.e. whether this box pads.
    pub fn pads(&self) -> bool {
        let (ox, oy, oz) = self.origin;
        let (dx, dy, dz) = self.dims;
        let (fx, fy, fz) = self.full_dims;
        ox < 0 || oy < 0 || oz < 0
            || ox + dx as isize > fx as isize
            || oy + dy as isize > fy as isize
            || oz + dz as isize > fz as isize
    }

    /// Whether this box is the whole grid.
    pub fn is_full(&self) -> bool {
        self.origin == (0, 0, 0) && self.dims == self.full_dims
    }

    pub fn voxels(&self) -> usize {
        self.dims.0 * self.dims.1 * self.dims.2
    }

    pub fn full_voxels(&self) -> usize {
        self.full_dims.0 * self.full_dims.1 * self.full_dims.2
    }

    /// How many times fewer voxels than the full grid.
    /// Full-grid voxels per box voxel. Above 1 the box is smaller than the grid (a net crop);
    /// below 1 it is larger (a net pad).
    pub fn reduction(&self) -> f64 {
        if self.voxels() == 0 { 1.0 } else { self.full_voxels() as f64 / self.voxels() as f64 }
    }

    /// The affine of the cropped volume: same orientation, origin shifted to the box corner.
    pub fn crop_affine(&self, affine: &[f64; 16]) -> [f64; 16] {
        let (ox, oy, oz) = (self.origin.0 as f64, self.origin.1 as f64, self.origin.2 as f64);
        let mut out = *affine;
        for row in 0..3 {
            out[4 * row + 3] = affine[4 * row + 3]
                + affine[4 * row] * ox
                + affine[4 * row + 1] * oy
                + affine[4 * row + 2] * oz;
        }
        out
    }
}

/// Smallest size `>= n` whose prime factors are all at most 7 — the radices `rustfft` has
/// dedicated butterflies for. Sizes with a large prime factor fall back to Bluestein's algorithm
/// and cost several times more despite holding the same data.
pub fn next_fft_friendly_size(n: usize) -> usize {
    if n <= 1 {
        return n.max(1);
    }
    let smooth = |mut m: usize| {
        for p in [2usize, 3, 5, 7] {
            while m.is_multiple_of(p) {
                m /= p;
            }
        }
        m == 1
    };
    let mut candidate = n;
    while !smooth(candidate) {
        candidate += 1;
    }
    candidate
}

/// Margin in voxels per axis for a margin given in millimetres, at least one voxel where the
/// margin is positive. Anisotropic voxels get a geometrically equal margin rather than an equal
/// voxel count — 8 voxels is 6.4 mm in-plane but 24 mm through-plane at 0.8 × 0.8 × 3 mm.
pub fn margin_voxels(margin_mm: f64, voxel_size: (f64, f64, f64)) -> (usize, usize, usize) {
    let per_axis = |mm: f64, vs: f64| -> usize {
        if mm <= 0.0 || vs <= 0.0 {
            0
        } else {
            ((mm / vs).ceil() as usize).max(1)
        }
    };
    (
        per_axis(margin_mm, voxel_size.0),
        per_axis(margin_mm, voxel_size.1),
        per_axis(margin_mm, voxel_size.2),
    )
}

/// The box to reconstruct in: the mask's bounding box, grown by `margin_mm` on every side, each
/// axis then rounded up to an FFT-friendly size and clamped to the grid.
///
/// Returns the full grid when the mask is empty, when it already fills the volume, or when an
/// axis cannot usefully shrink — so a caller can apply this unconditionally.
pub fn crop_box_for_mask(
    mask: &[u8],
    full_dims: (usize, usize, usize),
    voxel_size: (f64, f64, f64),
    margin_mm: f64,
) -> CropBox {
    let (nx, ny, nz) = full_dims;
    if mask.len() != nx * ny * nz {
        return fft_pad_box(full_dims);
    }

    let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
    let mut any = false;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if mask[x + y * nx + z * nx * ny] > 0 {
                    any = true;
                    for (d, v) in [x, y, z].iter().enumerate() {
                        lo[d] = lo[d].min(*v);
                        hi[d] = hi[d].max(*v);
                    }
                }
            }
        }
    }
    if !any {
        return fft_pad_box(full_dims);
    }

    let margin = margin_voxels(margin_mm, voxel_size);
    let margin = [margin.0, margin.1, margin.2];
    let full = [nx as isize, ny as isize, nz as isize];
    let (mut origin, mut dims) = ([0isize; 3], [0usize; 3]);

    for d in 0..3 {
        // Wanted extent: the mask plus its margin, clipped to what actually exists.
        let start = (lo[d] as isize - margin[d] as isize).max(0);
        let end = ((hi[d] + margin[d] + 1) as isize).min(full[d]);
        let wanted = next_fft_friendly_size((end - start) as usize) as isize;
        // Centre the FFT-friendly extent on that window. It may reach outside the grid, in which
        // case those voxels are padding — which is safe, since padding only moves the periodic
        // boundary further from the object.
        let extra = wanted - (end - start);
        let mut s0 = start - extra / 2;
        // Prefer to stay inside the grid where the extent allows it.
        if wanted <= full[d] {
            s0 = s0.clamp(0, full[d] - wanted);
        }
        origin[d] = s0;
        dims[d] = wanted as usize;
    }

    CropBox {
        origin: (origin[0], origin[1], origin[2]),
        dims: (dims[0], dims[1], dims[2]),
        full_dims,
    }
}

/// A box covering the whole grid, each axis grown outward to an FFT-friendly size.
///
/// The accuracy-neutral half of this module: no data is discarded and the periodic boundary moves
/// *away* from the object, so the only cost is the padded voxels. Worth doing whenever an axis
/// has an awkward length — 339 = 3·113 costs about a third more transform time than 343 = 7³
/// despite holding fewer voxels.
pub fn fft_pad_box(full_dims: (usize, usize, usize)) -> CropBox {
    let full = [full_dims.0, full_dims.1, full_dims.2];
    let (mut origin, mut dims) = ([0isize; 3], [0usize; 3]);
    for d in 0..3 {
        let wanted = next_fft_friendly_size(full[d]);
        // Centre the grid in the padded extent so the object stays central.
        origin[d] = -(((wanted - full[d]) / 2) as isize);
        dims[d] = wanted;
    }
    CropBox {
        origin: (origin[0], origin[1], origin[2]),
        dims: (dims[0], dims[1], dims[2]),
        full_dims,
    }
}

/// Copy the box out of a full-grid volume.
///
/// # Panics
/// If `data` does not match the box's `full_dims`.
pub fn crop_volume<T: Copy + Default>(data: &[T], b: &CropBox) -> Vec<T> {
    crop_volume_with(data, b, T::default())
}

/// Copy the box out of a full-grid volume, filling anything outside the grid with `fill`.
///
/// # Panics
/// If `data` does not match the box's `full_dims`.
pub fn crop_volume_with<T: Copy>(data: &[T], b: &CropBox, fill: T) -> Vec<T> {
    let (nx, ny, nz) = b.full_dims;
    assert_eq!(data.len(), b.full_voxels(), "crop_volume: data does not match the full grid");
    if b.is_full() {
        return data.to_vec();
    }
    let (cx, cy, cz) = b.dims;
    let (ox, oy, oz) = b.origin;
    let mut out = vec![fill; b.voxels()];
    for z in 0..cz {
        let sz = oz + z as isize;
        if sz < 0 || sz >= nz as isize {
            continue;
        }
        for y in 0..cy {
            let sy = oy + y as isize;
            if sy < 0 || sy >= ny as isize {
                continue;
            }
            // Clip the row to the part that exists in the source.
            let x0 = (-ox).max(0);
            let x1 = (nx as isize - ox).min(cx as isize);
            if x1 <= x0 {
                continue;
            }
            let src = (ox + x0) as usize + sy as usize * nx + sz as usize * nx * ny;
            let dst = x0 as usize + y * cx + z * cx * cy;
            let len = (x1 - x0) as usize;
            out[dst..dst + len].copy_from_slice(&data[src..src + len]);
        }
    }
    out
}

/// Put a cropped volume back into a full-grid volume, filling everything outside the box with
/// `fill` (zero for field maps and χ, which are undefined outside the mask anyway).
///
/// # Panics
/// If `data` does not match the box's `dims`.
pub fn uncrop_volume<T: Copy>(data: &[T], b: &CropBox, fill: T) -> Vec<T> {
    assert_eq!(data.len(), b.voxels(), "uncrop_volume: data does not match the crop box");
    if b.is_full() {
        return data.to_vec();
    }
    let (nx, ny, nz) = b.full_dims;
    let (cx, cy, cz) = b.dims;
    let (ox, oy, oz) = b.origin;
    let mut out = vec![fill; b.full_voxels()];
    for z in 0..cz {
        let dz = oz + z as isize;
        if dz < 0 || dz >= nz as isize {
            continue; // padding: nothing in the grid to write it to
        }
        for y in 0..cy {
            let dy = oy + y as isize;
            if dy < 0 || dy >= ny as isize {
                continue;
            }
            let x0 = (-ox).max(0);
            let x1 = (nx as isize - ox).min(cx as isize);
            if x1 <= x0 {
                continue;
            }
            let src = x0 as usize + y * cx + z * cx * cy;
            let dst = (ox + x0) as usize + dy as usize * nx + dz as usize * nx * ny;
            let len = (x1 - x0) as usize;
            out[dst..dst + len].copy_from_slice(&data[src..src + len]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_mask(full: (usize, usize, usize), lo: (usize, usize, usize), hi: (usize, usize, usize)) -> Vec<u8> {
        let (nx, ny, nz) = full;
        let mut m = vec![0u8; nx * ny * nz];
        for z in lo.2..=hi.2 {
            for y in lo.1..=hi.1 {
                for x in lo.0..=hi.0 {
                    m[x + y * nx + z * nx * ny] = 1;
                }
            }
        }
        m
    }

    #[test]
    fn fft_friendly_sizes() {
        assert_eq!(next_fft_friendly_size(339), 343); // 7^3; 339 = 3*113 would hit Bluestein
        assert_eq!(next_fft_friendly_size(340), 343);
        assert_eq!(next_fft_friendly_size(256), 256);
        assert_eq!(next_fft_friendly_size(188), 189); // 27*7
        assert_eq!(next_fft_friendly_size(1), 1);
        for n in 1..2000 {
            let s = next_fft_friendly_size(n);
            assert!(s >= n);
            let mut m = s;
            for p in [2, 3, 5, 7] {
                while m % p == 0 {
                    m /= p;
                }
            }
            assert_eq!(m, 1, "{s} (from {n}) is not 7-smooth");
        }
    }

    #[test]
    fn margin_is_geometric_not_voxel_count() {
        // 8 mm on 0.8 x 0.8 x 3 mm voxels: 10 in-plane, 3 through-plane.
        assert_eq!(margin_voxels(8.0, (0.8, 0.8, 3.0)), (10, 10, 3));
        assert_eq!(margin_voxels(0.0, (1.0, 1.0, 1.0)), (0, 0, 0));
    }

    /// With nothing to crop to, fall back to padding — never to discarding data.
    #[test]
    fn empty_or_unusable_mask_falls_back_to_padding() {
        let full = (16, 16, 8); // already 7-smooth, so the pad box is the grid itself
        for mask in [vec![0u8; 16 * 16 * 8], vec![1u8; 16 * 16 * 8], vec![1u8; 10]] {
            let b = crop_box_for_mask(&mask, full, (1.0, 1.0, 1.0), 4.0);
            assert!(b.is_full(), "{:?}+{:?}", b.origin, b.dims);
        }
        // On an awkward grid the fallback pads instead of leaving it alone.
        let awkward = (17, 16, 8);
        let b = crop_box_for_mask(&vec![0u8; 17 * 16 * 8], awkward, (1.0, 1.0, 1.0), 4.0);
        assert!(b.pads() && b.dims.0 == next_fft_friendly_size(17));
    }

    #[test]
    fn box_contains_the_mask_with_its_margin() {
        let full = (64, 64, 32);
        let mask = box_mask(full, (20, 24, 10), (40, 44, 20));
        let b = crop_box_for_mask(&mask, full, (1.0, 1.0, 1.0), 3.0);
        assert!(!b.is_full(), "should have cropped something");
        let lo = [b.origin.0, b.origin.1, b.origin.2];
        let hi = [
            b.origin.0 + b.dims.0 as isize,
            b.origin.1 + b.dims.1 as isize,
            b.origin.2 + b.dims.2 as isize,
        ];
        // Every mask voxel, and 3 mm around it, is inside the box.
        for (d, (mlo, mhi)) in [(20, 40), (24, 44), (10, 20)].iter().enumerate() {
            assert!(lo[d] <= mlo - 3, "axis {d}: box starts at {}, mask-3 at {}", lo[d], mlo - 3);
            assert!(hi[d] >= mhi + 1 + 3, "axis {d}: box ends at {}, mask+3 at {}", hi[d], mhi + 4);
        }
        // This one fits inside the grid, so it should not pad.
        assert!(!b.pads(), "box {:?}+{:?} should stay inside {:?}", b.origin, b.dims, full);
        for d in [b.dims.0, b.dims.1, b.dims.2] {
            assert_eq!(next_fft_friendly_size(d), d, "{d} should already be FFT-friendly");
        }
        assert!(b.reduction() > 1.0);
    }

    /// The realistic case this exists for.
    #[test]
    fn ukb_sized_volume_shrinks_substantially() {
        let full = (256, 288, 48);
        // A brain-ish box, roughly what HD-BET leaves on this acquisition.
        let mask = box_mask(full, (42, 36, 2), (213, 251, 45));
        let b = crop_box_for_mask(&mask, full, (0.8, 0.8, 3.0), 8.0);
        assert!(b.reduction() > 1.2, "expected a worthwhile reduction, got {:.2}x", b.reduction());
        assert!(b.voxels() < b.full_voxels());
    }

    // --- padding outward ---

    /// A real axially-resampled UK Biobank grid. Every axis is awkward — 272 = 2⁴·17,
    /// 339 = 3·113, 77 = 7·11 — which is exactly what resampling to a bounding box tends to
    /// produce, and exactly what a radix-2/3/5/7 FFT is worst at.
    #[test]
    fn fft_pad_box_grows_every_awkward_axis() {
        let b = fft_pad_box((272, 339, 77));
        assert_eq!(b.dims, (280, 343, 80), "each axis should reach the next 7-smooth size");
        assert!(b.pads());
        // The grid stays centred in the padded extent.
        assert_eq!(b.origin, (-4, -2, -1));
        // Padding costs voxels rather than saving them, and not many.
        assert!(b.reduction() < 1.0, "{}", b.reduction());
        let overhead = b.voxels() as f64 / b.full_voxels() as f64 - 1.0;
        assert!(overhead < 0.10, "padding overhead {overhead:.3} should stay under 10%");
    }

    #[test]
    fn fft_pad_box_is_a_no_op_on_a_friendly_grid() {
        let b = fft_pad_box((256, 288, 48));
        assert!(b.is_full(), "already 7-smooth: {:?}+{:?}", b.origin, b.dims);
        assert!(!b.pads());
    }

    /// Padding must not lose a single voxel: out and back is the identity.
    #[test]
    fn pad_then_unpad_is_lossless() {
        let full = (17, 13, 5); // every axis awkward
        let n = full.0 * full.1 * full.2;
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 + 1.0).collect();
        let b = fft_pad_box(full);
        assert!(b.pads() && b.voxels() > n);

        let padded = crop_volume(&data, &b);
        assert_eq!(padded.len(), b.voxels());
        // The original data is in there exactly once, and the rest is the fill.
        assert_eq!(padded.iter().filter(|v| **v != 0.0).count(), n, "all data present, nothing extra");

        let back = uncrop_volume(&padded, &b, f64::NAN);
        assert_eq!(back.len(), n);
        for (i, (got, want)) in back.iter().zip(data.iter()).enumerate() {
            assert_eq!(got, want, "voxel {i} changed across a pad round trip");
        }
    }

    #[test]
    fn pad_fill_value_is_respected() {
        let full = (11, 4, 2);
        let b = fft_pad_box(full);
        assert!(b.pads());
        let padded = crop_volume_with(&vec![7.0f64; full.0 * full.1 * full.2], &b, -1.0);
        assert!(padded.iter().any(|v| *v == -1.0), "padding should use the fill value");
        assert_eq!(padded.iter().filter(|v| **v == 7.0).count(), full.0 * full.1 * full.2);
    }

    /// A box that crops one axis and pads another — both at once, which is the realistic case
    /// once a margin is applied to an awkward grid.
    #[test]
    fn a_box_can_crop_and_pad_at_the_same_time() {
        let full = (64, 17, 16);
        let mask = box_mask(full, (20, 2, 4), (40, 14, 11));
        let b = crop_box_for_mask(&mask, full, (1.0, 1.0, 1.0), 2.0);
        // x has room to crop; y is awkward and the mask nearly fills it, so it pads.
        assert!(b.dims.0 < full.0, "x should crop, got {}", b.dims.0);
        assert!(b.pads(), "y should pad: {:?}+{:?}", b.origin, b.dims);
        for d in [b.dims.0, b.dims.1, b.dims.2] {
            assert_eq!(next_fft_friendly_size(d), d, "{d} not FFT-friendly");
        }
        // Round trip still restores every voxel the box covers.
        let n = full.0 * full.1 * full.2;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let back = uncrop_volume(&crop_volume(&data, &b), &b, -1.0);
        // Wherever the box covered the grid, the value is unchanged; elsewhere it is the fill.
        let mut restored = 0usize;
        for (i, v) in back.iter().enumerate() {
            if *v != -1.0 {
                assert_eq!(*v, data[i], "voxel {i} altered by the round trip");
                restored += 1;
            }
        }
        assert!(restored > 0 && restored < n, "expected a partial cover, got {restored} of {n}");
    }

    #[test]
    fn crop_then_uncrop_round_trips_inside_the_box() {
        let full = (12, 10, 6);
        let (nx, ny, _) = full;
        let data: Vec<f64> = (0..12 * 10 * 6).map(|i| i as f64).collect();
        let b = CropBox { origin: (2, 3, 1), dims: (6, 4, 3), full_dims: full };
        assert!(!b.pads());

        let cropped = crop_volume(&data, &b);
        assert_eq!(cropped.len(), b.voxels());
        // Spot-check the mapping.
        for z in 0..b.dims.2 {
            for y in 0..b.dims.1 {
                for x in 0..b.dims.0 {
                    let got = cropped[x + y * b.dims.0 + z * b.dims.0 * b.dims.1];
                    let want = data[(x + 2) + (y + 3) * nx + (z + 1) * nx * ny];
                    assert_eq!(got, want, "at ({x},{y},{z})");
                }
            }
        }

        let restored = uncrop_volume(&cropped, &b, 0.0);
        assert_eq!(restored.len(), data.len());
        for z in 0..full.2 {
            for y in 0..full.1 {
                for x in 0..full.0 {
                    let i = x + y * nx + z * nx * ny;
                    let inside = (2..8).contains(&x) && (3..7).contains(&y) && (1..4).contains(&z);
                    assert_eq!(restored[i], if inside { data[i] } else { 0.0 }, "at ({x},{y},{z})");
                }
            }
        }
    }

    #[test]
    fn full_box_is_a_no_op() {
        let full = (5, 4, 3);
        let data: Vec<f64> = (0..60).map(|i| i as f64).collect();
        let b = CropBox::full(full);
        assert_eq!(crop_volume(&data, &b), data);
        assert_eq!(uncrop_volume(&data, &b, 0.0), data);
        assert_eq!(b.reduction(), 1.0);
    }

    #[test]
    fn masks_crop_and_uncrop_too() {
        let full = (8, 8, 4);
        let mask = box_mask(full, (2, 2, 1), (5, 5, 2));
        let b = CropBox { origin: (1, 1, 0), dims: (6, 6, 4), full_dims: full };
        let c = crop_volume(&mask, &b);
        assert_eq!(c.iter().filter(|v| **v == 1).count(), mask.iter().filter(|v| **v == 1).count());
        let back = uncrop_volume(&c, &b, 0u8);
        assert_eq!(back, mask);
    }

    #[test]
    fn crop_affine_shifts_the_origin_only() {
        // 2 mm isotropic, origin at (-10, -20, -30).
        let affine = [
            2.0, 0.0, 0.0, -10.0,
            0.0, 2.0, 0.0, -20.0,
            0.0, 0.0, 2.0, -30.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let b = CropBox { origin: (3, 4, 5), dims: (4, 4, 4), full_dims: (16, 16, 16) };
        let out = b.crop_affine(&affine);
        // Rotation/scale untouched.
        for i in [0, 1, 2, 4, 5, 6, 8, 9, 10] {
            assert_eq!(out[i], affine[i]);
        }
        // The box corner keeps its world position: -10 + 2*3, -20 + 2*4, -30 + 2*5.
        assert_eq!((out[3], out[7], out[11]), (-4.0, -12.0, -20.0));
    }
}

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
//! Two things make the difference, and the second is easy to miss:
//!
//! - **Fewer voxels.** Mask bounding box plus a margin, rather than the full field of view.
//! - **Sizes an FFT likes.** A cropped extent of 339 has a prime factor of 113, which pushes
//!   `rustfft` onto Bluestein's algorithm; 340 factors as 2²·5·17 and is measurably faster
//!   despite being *larger*. [`next_fft_friendly_size`] rounds each axis up accordingly, which
//!   costs a few thousand voxels and buys back far more.
//!
//! ## This changes the numbers, not just the speed
//!
//! FFT-based reconstruction is periodic, so moving the boundary closer to the object brings
//! wrap-around with it. The dipole kernel in particular has infinite support. [`margin_voxels`]
//! takes the margin in **millimetres** so anisotropic voxels get a geometrically equal margin on
//! every side, and callers should validate a cropped reconstruction against an uncropped one
//! before trusting it, rather than assuming the two agree.

/// A box within a larger grid: where reconstruction actually happens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropBox {
    /// Index of the box's first voxel in the full grid.
    pub origin: (usize, usize, usize),
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
        return CropBox::full(full_dims);
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
        return CropBox::full(full_dims);
    }

    let margin = margin_voxels(margin_mm, voxel_size);
    let margin = [margin.0, margin.1, margin.2];
    let full = [nx, ny, nz];
    let (mut origin, mut dims) = ([0usize; 3], [0usize; 3]);

    for d in 0..3 {
        let start = lo[d].saturating_sub(margin[d]);
        let end = (hi[d] + margin[d] + 1).min(full[d]); // exclusive
        let wanted = next_fft_friendly_size(end - start);
        if wanted >= full[d] {
            // Rounding up covers the axis anyway; leave it alone rather than pay for a copy.
            origin[d] = 0;
            dims[d] = full[d];
            continue;
        }
        // Grow symmetrically around the box, then slide back inside the grid.
        let extra = wanted - (end - start);
        let mut s = start.saturating_sub(extra / 2);
        if s + wanted > full[d] {
            s = full[d] - wanted;
        }
        origin[d] = s;
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
    let (nx, ny, _) = b.full_dims;
    assert_eq!(data.len(), b.full_voxels(), "crop_volume: data does not match the full grid");
    if b.is_full() {
        return data.to_vec();
    }
    let (cx, cy, cz) = b.dims;
    let (ox, oy, oz) = b.origin;
    let mut out = Vec::with_capacity(b.voxels());
    for z in 0..cz {
        for y in 0..cy {
            let row = (ox) + (oy + y) * nx + (oz + z) * nx * ny;
            out.extend_from_slice(&data[row..row + cx]);
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
    let (nx, ny, _) = b.full_dims;
    let (cx, cy, cz) = b.dims;
    let (ox, oy, oz) = b.origin;
    let mut out = vec![fill; b.full_voxels()];
    for z in 0..cz {
        for y in 0..cy {
            let src = (y * cx) + z * cx * cy;
            let dst = ox + (oy + y) * nx + (oz + z) * nx * ny;
            out[dst..dst + cx].copy_from_slice(&data[src..src + cx]);
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

    #[test]
    fn empty_or_full_mask_gives_the_whole_grid() {
        let full = (16, 16, 8);
        assert!(crop_box_for_mask(&vec![0u8; 16 * 16 * 8], full, (1.0, 1.0, 1.0), 4.0).is_full());
        assert!(crop_box_for_mask(&vec![1u8; 16 * 16 * 8], full, (1.0, 1.0, 1.0), 4.0).is_full());
        // A mismatched mask is ignored rather than panicking.
        assert!(crop_box_for_mask(&vec![1u8; 10], full, (1.0, 1.0, 1.0), 4.0).is_full());
    }

    #[test]
    fn box_contains_the_mask_with_its_margin() {
        let full = (64, 64, 32);
        let mask = box_mask(full, (20, 24, 10), (40, 44, 20));
        let b = crop_box_for_mask(&mask, full, (1.0, 1.0, 1.0), 3.0);
        assert!(!b.is_full(), "should have cropped something");
        // Every mask voxel, and 3 mm around it, is inside the box.
        assert!(b.origin.0 + 3 <= 20 && b.origin.1 + 3 <= 24 && b.origin.2 + 3 <= 10);
        assert!(b.origin.0 + b.dims.0 >= 40 + 1 + 3);
        assert!(b.origin.1 + b.dims.1 >= 44 + 1 + 3);
        assert!(b.origin.2 + b.dims.2 >= 20 + 1 + 3);
        // Inside the grid, and FFT-friendly.
        assert!(b.origin.0 + b.dims.0 <= full.0 && b.origin.1 + b.dims.1 <= full.1 && b.origin.2 + b.dims.2 <= full.2);
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

    #[test]
    fn crop_then_uncrop_round_trips_inside_the_box() {
        let full = (12, 10, 6);
        let (nx, ny, _) = full;
        let data: Vec<f64> = (0..12 * 10 * 6).map(|i| i as f64).collect();
        let b = CropBox { origin: (2, 3, 1), dims: (6, 4, 3), full_dims: full };

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

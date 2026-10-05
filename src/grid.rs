//! Lightweight 3D volume grid descriptor.
//!
//! Contains only the geometric information needed by algorithm kernels:
//! dimensions and voxel sizes. This eliminates the need to pass 6 separate
//! parameters (nx, ny, nz, vsx, vsy, vsz) to every function.

/// A 3D volume grid with dimensions and voxel sizes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    /// Volume dimensions (nx, ny, nz)
    pub dims: (usize, usize, usize),
    /// Voxel sizes in mm (vsx, vsy, vsz)
    pub voxel_size: (f64, f64, f64),
}

impl Grid {
    /// Create a new Grid from dimensions and voxel sizes.
    #[inline]
    pub fn new(nx: usize, ny: usize, nz: usize, vsx: f64, vsy: f64, vsz: f64) -> Self {
        Self {
            dims: (nx, ny, nz),
            voxel_size: (vsx, vsy, vsz),
        }
    }

    #[inline]
    pub fn nx(&self) -> usize { self.dims.0 }
    #[inline]
    pub fn ny(&self) -> usize { self.dims.1 }
    #[inline]
    pub fn nz(&self) -> usize { self.dims.2 }
    #[inline]
    pub fn vsx(&self) -> f64 { self.voxel_size.0 }
    #[inline]
    pub fn vsy(&self) -> f64 { self.voxel_size.1 }
    #[inline]
    pub fn vsz(&self) -> f64 { self.voxel_size.2 }

    /// Total number of voxels.
    #[inline]
    pub fn n_total(&self) -> usize {
        self.dims.0 * self.dims.1 * self.dims.2
    }

    /// Spacing along one axis, in mm. `axis` is 0 = x, 1 = y, 2 = z.
    ///
    /// # Panics
    /// If `axis` is not 0, 1 or 2.
    #[inline]
    pub fn spacing(&self, axis: usize) -> f64 {
        match axis {
            0 => self.voxel_size.0,
            1 => self.voxel_size.1,
            2 => self.voxel_size.2,
            _ => panic!("axis must be 0, 1 or 2 (got {axis})"),
        }
    }

    /// Number of samples along one axis. `axis` is 0 = x, 1 = y, 2 = z.
    ///
    /// # Panics
    /// If `axis` is not 0, 1 or 2.
    #[inline]
    pub fn extent(&self, axis: usize) -> usize {
        match axis {
            0 => self.dims.0,
            1 => self.dims.1,
            2 => self.dims.2,
            _ => panic!("axis must be 0, 1 or 2 (got {axis})"),
        }
    }

    /// Check contiguity against slice geometry the acquisition may not have recorded.
    ///
    /// `None` means the acquisition did not say how thick its slices are, and a gap then
    /// **cannot be detected** — nothing in a NIfTI distinguishes 3 mm slices every 3 mm from
    /// 2 mm slices every 3 mm, because the spacing is the *pitch*. This returns `Ok` in that
    /// case, deliberately and in one place rather than at each call site: refusing every
    /// acquisition that failed to record a recommended BIDS field would block the vast
    /// majority of data, which has no gap. The missing information is in the input, not in
    /// this crate, and the right response is for the host to say so once.
    ///
    /// The name says `if_known` so no call site can read as though a gap had been ruled out.
    /// Where the thickness *is* known, this is [`Grid::require_contiguous_slices`].
    pub fn require_contiguous_slices_if_known(
        &self,
        slices: Option<SliceGeometry>,
    ) -> Result<(), SliceGapError> {
        match slices {
            Some(s) => self.require_contiguous_slices(s.thickness, s.axis),
            None => Ok(()),
        }
    }

    /// Check that the volume is contiguous along the slice axis, as the FFT dipole kernel
    /// requires.
    ///
    /// A 2D multi-slice acquisition often leaves a gap between slices: the excited slab is
    /// thinner than the slice-to-slice pitch, so the sampled volume has holes in it. A
    /// [`Grid`] cannot express that — its `voxel_size` is the pitch, and nothing records the
    /// thickness — so the gap has to be supplied, normally from the acquisition's
    /// `SliceThickness` sidecar field.
    ///
    /// It matters because the dipole kernel, and the spherical kernels of the SHARP family,
    /// are evaluated as FFTs on a uniform grid. That is not an approximation that degrades
    /// with the gap; the convolution being computed is simply not the one the physics calls
    /// for, and the resulting map is wrong rather than noisy — wrong in a way that still
    /// looks like a susceptibility map. So this refuses rather than warns.
    ///
    /// In-plane work is unaffected: a gap along z does not disturb a purely in-plane kernel,
    /// which is why [`vsharp_2d`](crate::bgremove::vsharp_2d) does not call this and
    /// [`vsharp_2d_pdf`](crate::bgremove::vsharp_2d_pdf), whose second stage is a 3D PDF,
    /// does.
    ///
    /// # Arguments
    /// * `slice_thickness` - Excited slice thickness in mm. For a contiguous acquisition this
    ///   equals the slice spacing, which is what a 3D acquisition should pass.
    /// * `slice_axis` - Axis the slices are stacked along: 0 = x, 1 = y, 2 = z.
    ///
    /// # Panics
    /// If `slice_axis` is not 0, 1 or 2.
    pub fn require_contiguous_slices(
        &self,
        slice_thickness: f64,
        slice_axis: usize,
    ) -> Result<(), SliceGapError> {
        let pitch = self.spacing(slice_axis);
        if !slice_thickness.is_finite() || slice_thickness <= 0.0 {
            return Err(SliceGapError { pitch, thickness: slice_thickness, axis: slice_axis });
        }
        // 1% of the pitch, so a thickness read back from a sidecar that rounded does not trip
        // it, while any gap worth the name does
        if slice_thickness < pitch * 0.99 {
            return Err(SliceGapError { pitch, thickness: slice_thickness, axis: slice_axis });
        }
        Ok(())
    }
}

/// How a 2D multi-slice acquisition laid its slices out, when that is recorded.
///
/// Only the thickness is missing from a [`Grid`]: the grid's spacing along the slice axis is
/// already the slice-to-slice *pitch*. The two together say whether the sampled volume has
/// holes in it. Carried as a unit because a thickness without an axis says nothing.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SliceGeometry {
    /// Excited slice thickness in mm, from the acquisition's `SliceThickness`.
    pub thickness: f64,
    /// Axis the slices are stacked along: 0 = x, 1 = y, 2 = z.
    pub axis: usize,
}

impl SliceGeometry {
    /// Slice geometry for an acquisition known to be contiguous: thickness equals the grid's
    /// spacing along `axis`.
    ///
    /// Only correct when the acquisition really has no gap. If it might, read `SliceThickness`
    /// from the sidecar instead — this constructor cannot discover a gap, by construction.
    pub fn contiguous(grid: &Grid, axis: usize) -> Self {
        Self { thickness: grid.spacing(axis), axis }
    }
}

/// The volume is not contiguous along the slice axis: the slices are thinner than the gap
/// between their centres, so there is unsampled tissue in between.
///
/// Returned by [`Grid::require_contiguous_slices`]. See that method for why this is refused
/// rather than approximated.
///
/// # Why a `Result` and not a panic
///
/// The crate uses both, and the split is by *whose* mistake it is. A length that disagrees with
/// the grid is the caller's — `rss_combine` and the bias correction panic on it, and so do the
/// entry points in [`crate::unwrap::slicewise`] and [`crate::bgremove::vsharp2d`] — and the only
/// useful audience is a developer reading a backtrace. A slice gap is the *acquisition's*: the
/// call is correct, the data simply cannot yield a right answer, and the audience is the person
/// who scanned it. Only the second can be surfaced to a user, so only the second is a `Result`.
///
/// One case sits across the line and is worth naming because it looks like a precondition:
/// a thickness that is zero, negative or not finite is refused here rather than asserted, since
/// it arrives from a sidecar rather than from code. A host that mis-parses `SliceThickness`
/// should be told its data is unusable, not handed a panic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SliceGapError {
    /// Slice-to-slice pitch in mm, taken from the grid.
    pub pitch: f64,
    /// Excited slice thickness in mm, as supplied by the caller.
    pub thickness: f64,
    /// Axis the slices are stacked along.
    pub axis: usize,
}

impl SliceGapError {
    /// Size of the unsampled gap between consecutive slices, in mm.
    pub fn gap(&self) -> f64 {
        self.pitch - self.thickness
    }
}

impl std::fmt::Display for SliceGapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.thickness.is_finite() || self.thickness <= 0.0 {
            return write!(
                f,
                "slice thickness must be a positive number of mm (got {})",
                self.thickness
            );
        }
        write!(
            f,
            "slices along axis {} are {:.4} mm thick but {:.4} mm apart, leaving a {:.4} mm gap: \
the volume is not contiguous, and the FFT dipole and spherical-mean kernels are defined on a \
uniform grid, so any field or susceptibility computed from it would be wrong rather than \
approximate",
            self.axis, self.thickness, self.pitch, self.gap()
        )
    }
}

impl std::error::Error for SliceGapError {}

/// How the slices of a volume sit inside it, for code that works one slice at a time.
///
/// A 2D multi-slice acquisition is processed slice by slice, and the operations that do so
/// — [`unwrap_slicewise`](crate::unwrap::unwrap_slicewise),
/// [`vsharp_2d`](crate::bgremove::vsharp_2d) — all want the same two things: the strides
/// that pick a slice out of the volume, and a [`Grid`] describing one slice. Collapsing the
/// slice axis to a length of 1 rather than writing 2D variants of each kernel is what keeps
/// those operations honest: the existing 3D code runs unmodified, its slice-axis neighbours
/// simply fall outside the volume, and a spherical kernel becomes the disc it should be.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SliceLayout {
    /// In-slice dimensions, in the order the extracted buffer uses.
    pub d0: usize,
    pub d1: usize,
    /// Number of slices along the slice axis.
    pub n_slices: usize,
    /// Stride between successive slices, in voxels.
    slice_stride: usize,
    /// Strides of the two in-slice axes, in voxels.
    stride0: usize,
    stride1: usize,
    /// Grid describing a single slice, with the slice axis collapsed to 1.
    pub grid: Grid,
}

impl SliceLayout {
    /// # Panics
    /// If `axis` is not 0, 1 or 2.
    pub fn new(grid: &Grid, axis: usize) -> Self {
        let (nx, ny, nz) = grid.dims;
        let (vsx, vsy, vsz) = grid.voxel_size;
        let (nxy, sy, sz) = (nx * ny, nx, nx * ny);
        match axis {
            0 => Self {
                d0: ny, d1: nz, n_slices: nx, slice_stride: 1, stride0: sy, stride1: sz,
                grid: Grid::new(ny, nz, 1, vsy, vsz, vsx),
            },
            1 => Self {
                d0: nx, d1: nz, n_slices: ny, slice_stride: sy, stride0: 1, stride1: sz,
                grid: Grid::new(nx, nz, 1, vsx, vsz, vsy),
            },
            2 => Self {
                d0: nx, d1: ny, n_slices: nz, slice_stride: nxy, stride0: 1, stride1: sy,
                grid: Grid::new(nx, ny, 1, vsx, vsy, vsz),
            },
            _ => panic!("slice_axis must be 0, 1 or 2 (got {axis})"),
        }
    }

    /// Number of voxels in one slice.
    #[inline]
    pub fn n_in_slice(&self) -> usize {
        self.d0 * self.d1
    }

    /// Volume index of in-slice position `p` on slice `s`.
    #[inline]
    pub fn index(&self, s: usize, p: usize) -> usize {
        let a = p % self.d0;
        let b = p / self.d0;
        s * self.slice_stride + a * self.stride0 + b * self.stride1
    }

    /// Copy slice `s` of `volume` into `out`. An empty `volume` leaves `out` alone, so
    /// callers can pass an optional input through without branching.
    pub fn gather<T: Copy>(&self, volume: &[T], s: usize, out: &mut [T]) {
        if volume.is_empty() {
            return;
        }
        for (p, o) in out.iter_mut().enumerate() {
            *o = volume[self.index(s, p)];
        }
    }

    /// Write `values` back into slice `s` of `volume`.
    pub fn scatter<T: Copy>(&self, volume: &mut [T], s: usize, values: &[T]) {
        for (p, &v) in values.iter().enumerate() {
            volume[self.index(s, p)] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_basic() {
        let g = Grid::new(64, 64, 32, 1.0, 1.0, 2.0);
        assert_eq!(g.nx(), 64);
        assert_eq!(g.ny(), 64);
        assert_eq!(g.nz(), 32);
        assert_eq!(g.n_total(), 64 * 64 * 32);
        assert_eq!(g.vsx(), 1.0);
        assert_eq!(g.vsz(), 2.0);
    }

    #[test]
    fn spacing_and_extent_follow_the_axis() {
        let g = Grid::new(5, 7, 3, 1.0, 2.0, 4.0);
        assert_eq!((g.extent(0), g.extent(1), g.extent(2)), (5, 7, 3));
        assert_eq!((g.spacing(0), g.spacing(1), g.spacing(2)), (1.0, 2.0, 4.0));
    }

    #[test]
    #[should_panic(expected = "axis must be 0, 1 or 2")]
    fn spacing_rejects_a_bad_axis() {
        Grid::new(2, 2, 2, 1.0, 1.0, 1.0).spacing(3);
    }

    #[test]
    #[should_panic(expected = "axis must be 0, 1 or 2")]
    fn extent_rejects_a_bad_axis() {
        Grid::new(2, 2, 2, 1.0, 1.0, 1.0).extent(3);
    }

    #[test]
    fn contiguous_slices_are_accepted_and_gapped_ones_refused() {
        let g = Grid::new(8, 8, 4, 1.0, 1.0, 3.0);

        // thickness == pitch: contiguous
        assert!(g.require_contiguous_slices(3.0, 2).is_ok());
        // and the slack absorbs a sidecar that rounded, but not a real gap
        assert!(g.require_contiguous_slices(2.99, 2).is_ok());
        assert!(g.require_contiguous_slices(2.9, 2).is_err());

        let err = g.require_contiguous_slices(2.0, 2).unwrap_err();
        assert_eq!(err.gap(), 1.0);
        assert_eq!((err.pitch, err.thickness, err.axis), (3.0, 2.0, 2));
        assert!(err.to_string().contains("not contiguous"));

        // thicker than the pitch is overlap, not a gap, and is not this check's business
        assert!(g.require_contiguous_slices(4.0, 2).is_ok());

        // the axis is honoured: 1 mm slices along x are contiguous on the same grid
        assert!(g.require_contiguous_slices(1.0, 0).is_ok());
        assert!(g.require_contiguous_slices(0.5, 0).is_err());
    }

    #[test]
    fn unrecorded_slice_geometry_cannot_refuse_and_does_not_pretend_to() {
        // Nothing in a grid distinguishes a gapped acquisition from a contiguous one, so with
        // no thickness there is nothing to check. The name of the method is what keeps a call
        // site from reading as though a gap had been ruled out.
        let g = Grid::new(8, 8, 4, 1.0, 1.0, 3.0);
        assert!(g.require_contiguous_slices_if_known(None).is_ok());
        assert!(g
            .require_contiguous_slices_if_known(Some(SliceGeometry { thickness: 3.0, axis: 2 }))
            .is_ok());
        let err = g
            .require_contiguous_slices_if_known(Some(SliceGeometry { thickness: 2.0, axis: 2 }))
            .unwrap_err();
        assert_eq!(err.gap(), 1.0);
    }

    #[test]
    fn slice_geometry_contiguous_reads_the_grid_spacing() {
        let g = Grid::new(8, 8, 4, 1.0, 2.0, 3.0);
        assert_eq!(SliceGeometry::contiguous(&g, 2), SliceGeometry { thickness: 3.0, axis: 2 });
        assert_eq!(SliceGeometry::contiguous(&g, 1), SliceGeometry { thickness: 2.0, axis: 1 });
        // and it always passes its own check, which is the point and the limitation
        for axis in 0..3 {
            assert!(g
                .require_contiguous_slices_if_known(Some(SliceGeometry::contiguous(&g, axis)))
                .is_ok());
        }
    }

    #[test]
    fn a_nonsense_slice_thickness_is_refused_and_says_so() {
        let g = Grid::new(8, 8, 4, 1.0, 1.0, 3.0);
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let err = g.require_contiguous_slices(bad, 2).unwrap_err();
            assert!(
                err.to_string().contains("must be a positive number"),
                "thickness {bad} gave: {err}"
            );
        }
    }

    #[test]
    fn slice_layout_visits_every_voxel_exactly_once() {
        let g = Grid::new(5, 7, 3, 1.0, 2.0, 3.0);
        for axis in 0..3 {
            let layout = SliceLayout::new(&g, axis);
            assert_eq!(layout.n_slices * layout.n_in_slice(), g.n_total(), "axis {axis}");
            let mut seen = vec![0u32; g.n_total()];
            for s in 0..layout.n_slices {
                for p in 0..layout.n_in_slice() {
                    seen[layout.index(s, p)] += 1;
                }
            }
            assert!(seen.iter().all(|&c| c == 1), "axis {axis} does not hit each voxel once");
        }
    }

    #[test]
    fn slice_layout_grid_keeps_the_two_in_plane_axes() {
        let g = Grid::new(5, 7, 3, 1.0, 2.0, 3.0);
        assert_eq!(SliceLayout::new(&g, 0).grid, Grid::new(7, 3, 1, 2.0, 3.0, 1.0));
        assert_eq!(SliceLayout::new(&g, 1).grid, Grid::new(5, 3, 1, 1.0, 3.0, 2.0));
        assert_eq!(SliceLayout::new(&g, 2).grid, Grid::new(5, 7, 1, 1.0, 2.0, 3.0));
    }

    #[test]
    fn slice_layout_buffer_order_matches_the_slice_grid() {
        // The gathered buffer is handed straight to kernels that read it through
        // `layout.grid`, so its two in-slice extents have to be that grid's own, in that
        // order. Transposing them is a bijection over the volume and leaves gather/scatter
        // round-tripping, so nothing else here would notice.
        let g = Grid::new(5, 7, 3, 1.0, 2.0, 3.0);
        for axis in 0..3 {
            let layout = SliceLayout::new(&g, axis);
            assert_eq!(
                (layout.d0, layout.d1),
                (layout.grid.nx(), layout.grid.ny()),
                "axis {axis}: buffer is {}x{} but its grid says {}x{}",
                layout.d0, layout.d1, layout.grid.nx(), layout.grid.ny()
            );
            // and stepping one along the buffer steps one along that grid's first axis
            assert_eq!(layout.index(0, 1) - layout.index(0, 0), match axis {
                0 => 5,            // x stride of the y axis
                1 => 1,            // x stride of the x axis
                _ => 1,
            });
        }
    }

    #[test]
    fn slice_layout_gather_and_scatter_round_trip() {
        let g = Grid::new(4, 5, 3, 1.0, 1.0, 1.0);
        let volume: Vec<f64> = (0..g.n_total()).map(|i| i as f64 * 0.25).collect();
        for axis in 0..3 {
            let layout = SliceLayout::new(&g, axis);
            let mut buf = vec![0.0; layout.n_in_slice()];
            let mut out = vec![-1.0; g.n_total()];
            for s in 0..layout.n_slices {
                layout.gather(&volume, s, &mut buf);
                layout.scatter(&mut out, s, &buf);
            }
            assert_eq!(out, volume, "axis {axis}");
        }
    }

    #[test]
    fn slice_layout_gather_of_an_empty_volume_leaves_the_buffer_alone() {
        let layout = SliceLayout::new(&Grid::new(4, 4, 2, 1.0, 1.0, 1.0), 2);
        let mut buf = vec![7.0; layout.n_in_slice()];
        layout.gather(&[] as &[f64], 0, &mut buf);
        assert!(buf.iter().all(|&v| v == 7.0));
    }

    #[test]
    #[should_panic(expected = "slice_axis must be 0, 1 or 2")]
    fn slice_layout_rejects_a_bad_axis() {
        SliceLayout::new(&Grid::new(2, 2, 2, 1.0, 1.0, 1.0), 3);
    }

    #[test]
    fn test_grid_copy() {
        let g = Grid::new(10, 20, 30, 0.5, 0.5, 1.0);
        let g2 = g;
        assert_eq!(g, g2);
    }
}

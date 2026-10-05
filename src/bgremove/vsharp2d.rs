//! 2D V-SHARP, and the 2D V-SHARP → 3D PDF chain for 2D multi-slice data.
//!
//! SHARP and its variants rest on the mean value property of harmonic functions: the average
//! of a harmonic field over a sphere equals its value at the centre, so convolving with a
//! normalised sphere and subtracting leaves only the non-harmonic — local — part. That
//! property is three-dimensional. A disc has no equivalent: the average of a harmonic field
//! over a disc is not its centre value, so a V-SHARP run slice by slice does not remove the
//! background field, it removes the part of it that varies in plane. The through-slice part
//! survives, and shows up in the EPI phase literature as persistent through-slice artifacts.
//!
//! So 2D V-SHARP is not a substitute for the 3D method, and [`vsharp_2d`] is not offered as
//! one. What makes it worth having is the second stage. [`vsharp_2d_pdf`] runs it first and
//! then a 3D PDF over the whole volume, which is the arrangement used in practice:
//!
//! 1. 2D V-SHARP removes the in-plane background component, eroding only in plane. On thick
//!    slices that matters — a 3D spherical kernel of any useful radius eats several slices
//!    off each end of a 20-slice acquisition, and a disc of the same radius eats none.
//! 2. 3D PDF removes what is left, which is the through-slice component the discs could not
//!    see. PDF projects the field onto the span of dipoles outside the ROI, needs no eroded
//!    boundary, and is a genuinely 3D model.
//!
//! # Slice gaps
//!
//! The two stages differ in whether a slice gap invalidates them, and the split is not a
//! detail — it is the reason they are separate functions.
//!
//! [`vsharp_2d`] is purely in plane. Its kernel never reaches along the slice axis, so it
//! neither knows nor cares whether the slices are contiguous, and it is valid on gapped data.
//!
//! [`vsharp_2d_pdf`] contains a 3D PDF, whose dipole kernel is an FFT on a uniform grid. On
//! gapped data that kernel is not an approximation of the right convolution, it is a
//! different convolution, and the field it attributes to the background is wrong in a way
//! that still looks like a field map. So the chain takes the acquisition's slice thickness
//! and refuses rather than computing it: see [`Grid::require_contiguous_slices`].
//!
//! The dipole inversion that follows is 3D for the same reason and carries the same
//! restriction. A 2D multi-slice acquisition still assembles into a 3D volume, and the dipole
//! kernel is inherently three-dimensional, so there is no 2D inversion to reach for — only
//! the same contiguity requirement, which the caller should check once before inverting.
//!
//! # References
//!
//! V-SHARP: Wu, B., Li, W., Guidon, A., Liu, C. (2012). "Whole brain susceptibility mapping
//! using compressed sensing." Magnetic Resonance in Medicine, 67(1):137-147.
//! <https://doi.org/10.1002/mrm.23000>
//!
//! PDF: Liu, T., Khalidov, I., de Rochefort, L., et al. (2011). "A novel background field
//! removal method for MRI using projection onto dipole fields." NMR in Biomedicine,
//! 24(9):1129-1136. <https://doi.org/10.1002/nbm.1670>

use crate::bgremove::pdf::{pdf, PdfParams};
use crate::bgremove::vsharp::{vsharp, VsharpParams};
use crate::grid::{SliceGapError, SliceLayout};
use crate::Grid;

/// Parameters for [`vsharp_2d_pdf`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug)]
pub struct Vsharp2dPdfParams {
    /// First stage: 2D V-SHARP. The radii are in mm and are measured in plane, so the
    /// defaults carry over from the 3D method unchanged.
    pub vsharp: VsharpParams,
    /// Second stage: 3D PDF over the volume.
    pub pdf: PdfParams,
    /// Axis the slices are stacked along: 0 = x, 1 = y, 2 = z (default: 2).
    pub slice_axis: usize,
    /// Excited slice thickness in mm, from the acquisition's `SliceThickness`.
    ///
    /// There is no default, and no way to skip it, because nothing in a [`Grid`] or a NIfTI
    /// records it — the spacing is the slice *pitch*, and a gapped acquisition is
    /// indistinguishable from a contiguous one without this. For a contiguous acquisition
    /// pass the slice spacing. See [`Grid::require_contiguous_slices`].
    pub slice_thickness: f64,
}

impl Vsharp2dPdfParams {
    /// Default parameters for an acquisition with the given slice geometry.
    ///
    /// `slice_thickness` is the excited slab thickness in mm, normally the sidecar's
    /// `SliceThickness`; it is what tells [`vsharp_2d_pdf`] whether the volume is contiguous.
    pub fn new(slice_axis: usize, slice_thickness: f64) -> Self {
        Self {
            vsharp: VsharpParams::default(),
            pdf: PdfParams::default(),
            slice_axis,
            slice_thickness,
        }
    }

    /// Parameters for a contiguous acquisition, whose slice thickness is its slice spacing.
    ///
    /// Only correct when the acquisition really has no slice gap. If it might, read
    /// `SliceThickness` from the sidecar and set [`slice_thickness`](Self::slice_thickness)
    /// from that instead.
    pub fn contiguous(grid: &Grid, slice_axis: usize) -> Self {
        Self::new(slice_axis, grid.spacing(slice_axis))
    }
}

/// V-SHARP applied to each slice independently, in plane.
///
/// Each slice is handed to [`vsharp`] on a grid with the slice axis collapsed to 1, so the
/// spherical mean value kernel becomes the disc of the same radius and the FFTs become 2D.
/// The radii in [`VsharpParams`] are in mm either way.
///
/// **This removes the in-plane background component only.** The mean value property SHARP
/// relies on is three-dimensional and a disc does not have it, so the through-slice component
/// of the background field survives. Follow this with a 3D method — [`vsharp_2d_pdf`] is that
/// chain — rather than treating it as background removal on its own.
///
/// Valid on data with a slice gap: the kernel is in plane and never crosses the gap.
///
/// # Arguments
/// * `field` - Unwrapped total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `grid` - Volume grid
/// * `params` - V-SHARP parameters; radii are in mm, measured in plane
/// * `slice_axis` - Axis the slices are stacked along: 0 = x, 1 = y, 2 = z
/// * `progress` - Progress callback, called once per slice as `(slice + 1, n_slices)`
///
/// # Returns
/// `(local_field, eroded_mask)`. The erosion is in plane only, so no slice is lost off either
/// end of the stack — which is the practical reason to prefer discs on thick slices.
///
/// # Panics
/// If `slice_axis` is not 0, 1 or 2, or if `field`/`mask` do not match `grid`.
pub fn vsharp_2d(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    params: &VsharpParams,
    slice_axis: usize,
    mut progress: impl FnMut(usize, usize),
) -> (Vec<f64>, Vec<u8>) {
    let n_total = grid.n_total();
    assert_eq!(field.len(), n_total, "field length must match grid dimensions");
    assert_eq!(mask.len(), n_total, "mask length must match grid dimensions");

    let layout = SliceLayout::new(grid, slice_axis);
    let n = layout.n_in_slice();
    let mut local = vec![0.0; n_total];
    let mut eroded = vec![0u8; n_total];
    let (mut f, mut m) = (vec![0.0; n], vec![0u8; n]);

    for s in 0..layout.n_slices {
        progress(s + 1, layout.n_slices);
        layout.gather(field, s, &mut f);
        layout.gather(mask, s, &mut m);
        if m.iter().all(|&v| v == 0) {
            // An optimisation, not a guard: vsharp on an empty mask already returns zeros
            // and an empty eroded mask. It is here because a 2D multi-slice stack routinely
            // has empty slices at each end and a wasted FFT per slice adds up.
            continue;
        }
        let (l, e) = vsharp(&f, &m, &layout.grid, params, |_, _| {});
        layout.scatter(&mut local, s, &l);
        layout.scatter(&mut eroded, s, &e);
    }
    (local, eroded)
}

/// 2D V-SHARP per slice, then 3D PDF over the volume.
///
/// The arrangement 2D multi-slice data needs: discs remove the in-plane background without
/// eroding along the slice axis, then PDF removes the through-slice part the discs are blind
/// to. See the module docs for why neither half is sufficient alone.
///
/// # Arguments
/// * `field` - Unwrapped total field (nx * ny * nz)
/// * `mask` - Binary mask (nx * ny * nz), 1 = inside ROI
/// * `grid` - Volume grid
/// * `bdir` - B0 direction as a unit vector, for the PDF stage
/// * `params` - Both stages' parameters, the slice axis, and the acquisition slice thickness
/// * `progress` - Progress callback. The two stages report in sequence over a combined total,
///   V-SHARP's slices first and then PDF's iterations.
///
/// # Returns
/// `(local_field, eroded_mask)`, or [`SliceGapError`] if the acquisition has a slice gap —
/// see [`Grid::require_contiguous_slices`] for why that is refused rather than computed.
///
/// # Panics
/// If `params.slice_axis` is not 0, 1 or 2, or if `field`/`mask` do not match `grid`.
pub fn vsharp_2d_pdf(
    field: &[f64],
    mask: &[u8],
    grid: &Grid,
    bdir: (f64, f64, f64),
    params: &Vsharp2dPdfParams,
    mut progress: impl FnMut(usize, usize),
) -> Result<(Vec<f64>, Vec<u8>), SliceGapError> {
    // Stage 2 is a 3D FFT dipole model, so the volume has to be contiguous. Checked before
    // any work, so a refusal costs nothing and cannot be mistaken for a result.
    grid.require_contiguous_slices(params.slice_thickness, params.slice_axis)?;

    let n_slices = grid.extent(params.slice_axis);
    let pdf_iterations = params.pdf.max_iter.unwrap_or_else(|| (grid.n_total() as f64).sqrt() as usize);
    let total = n_slices + pdf_iterations;

    let (in_plane_removed, eroded) = vsharp_2d(
        field, mask, grid, &params.vsharp, params.slice_axis,
        |done, _| progress(done, total),
    );
    let local = pdf(
        &in_plane_removed, &eroded, grid, bdir, &params.pdf,
        |done, _| progress((n_slices + done).min(total), total),
    );
    let local = local
        .iter()
        .zip(&eroded)
        .map(|(&v, &m)| if m != 0 { v } else { 0.0 })
        .collect();
    Ok((local, eroded))
}

#[cfg(test)]
mod tests {
    use super::*;

    const NX: usize = 32;
    const NY: usize = 32;
    const NZ: usize = 12;
    const VS: (f64, f64, f64) = (1.0, 1.0, 3.0);

    fn grid() -> Grid {
        Grid::new(NX, NY, NZ, VS.0, VS.1, VS.2)
    }

    fn world(i: usize, j: usize, k: usize) -> (f64, f64, f64) {
        (i as f64 * VS.0, j as f64 * VS.1, k as f64 * VS.2)
    }

    /// A ROI well inside the volume, with empty slices above and below it. This is what a 3D
    /// acquisition looks like, and the geometry a 3D spherical kernel is happiest on.
    fn mask_enclosed() -> Vec<u8> {
        let (cx, cy, cz) = world(NX / 2, NY / 2, NZ / 2);
        let mut m = vec![0u8; NX * NY * NZ];
        for k in 0..NZ {
            for j in 0..NY {
                for i in 0..NX {
                    let (x, y, z) = world(i, j, k);
                    let r = ((x - cx) / 11.0).powi(2)
                        + ((y - cy) / 11.0).powi(2)
                        + ((z - cz) / 13.0).powi(2);
                    m[i + j * NX + k * NX * NY] = u8::from(r <= 1.0);
                }
            }
        }
        m
    }

    /// A slab reaching the first and last slice, which is what 2D multi-slice coverage looks
    /// like: the stack is prescribed over a region of interest, not over the whole head, so
    /// the tissue runs out at the same place the volume does.
    fn mask_slab() -> Vec<u8> {
        let (cx, cy, _) = world(NX / 2, NY / 2, 0);
        let mut m = vec![0u8; NX * NY * NZ];
        for k in 0..NZ {
            for j in 0..NY {
                for i in 0..NX {
                    let (x, y, _) = world(i, j, k);
                    let r = ((x - cx) / 11.0).powi(2) + ((y - cy) / 11.0).powi(2);
                    m[i + j * NX + k * NX * NY] = u8::from(r <= 1.0);
                }
            }
        }
        m
    }

    /// Field of point dipoles placed outside the ROI, written out analytically rather than
    /// taken from the crate's own dipole kernel - a background field built with the code
    /// under test would be removed by construction and prove nothing.
    ///
    /// `(3*dz^2 - r^2) / r^5` is the second derivative of `1/r` along z, so it is harmonic
    /// everywhere away from its source. That is exactly what makes it background: inside the
    /// ROI there is no source, the field is harmonic, and a correct background removal has to
    /// take all of it away. The ground truth is therefore zero everywhere in the ROI, and a
    /// residual is an error outright rather than a difference from a reference.
    fn background_field() -> Vec<f64> {
        // off-axis and at different heights, so the field varies through slice as well as in
        // plane - a purely in-plane background would flatter the 2D stage
        let sources = [
            (-14.0, 4.0, 12.0, 1.2e6),
            (46.0, 40.0, 6.0, -0.9e6),
            (16.0, -12.0, 30.0, 0.7e6),
            (18.0, 44.0, -14.0, 1.0e6),
        ];
        let mut f = vec![0.0; NX * NY * NZ];
        for k in 0..NZ {
            for j in 0..NY {
                for i in 0..NX {
                    let (x, y, z) = world(i, j, k);
                    let mut v = 0.0;
                    for &(sx, sy, sz, m) in &sources {
                        let (dx, dy, dz) = (x - sx, y - sy, z - sz);
                        let r2 = dx * dx + dy * dy + dz * dz;
                        let r = r2.sqrt();
                        v += m * (3.0 * dz * dz - r2) / (r2 * r2 * r);
                    }
                    f[i + j * NX + k * NX * NY] = v;
                }
            }
        }
        f
    }

    fn rms(values: &[f64], mask: &[u8]) -> f64 {
        let (mut sum, mut n) = (0.0, 0usize);
        for (v, &m) in values.iter().zip(mask) {
            if m != 0 {
                sum += v * v;
                n += 1;
            }
        }
        if n == 0 { 0.0 } else { (sum / n as f64).sqrt() }
    }

    /// RMS over the given slices only.
    fn rms_slices(values: &[f64], mask: &[u8], slices: &[usize]) -> f64 {
        let (mut sum, mut n) = (0.0, 0usize);
        for &k in slices {
            for p in 0..NX * NY {
                let i = p + k * NX * NY;
                if mask[i] != 0 {
                    sum += values[i] * values[i];
                    n += 1;
                }
            }
        }
        if n == 0 { 0.0 } else { (sum / n as f64).sqrt() }
    }

    fn chain_params(g: &Grid) -> Vsharp2dPdfParams {
        Vsharp2dPdfParams {
            pdf: PdfParams { tol: 1e-6, max_iter: Some(60) },
            ..Vsharp2dPdfParams::contiguous(g, 2)
        }
    }

    /// A floor, not evidence about the disc. The mean of a linear function over *any* kernel
    /// symmetric about its centre is the centre value, so a linear ramp is removed exactly by a
    /// disc, a sphere, a square, or a disc of a quarter the radius. Checked by perturbation:
    /// shrinking the disc 4x and stretching it into an ellipse both leave this test passing.
    ///
    /// What it does pin is that the method high-passes at all, which together with
    /// `vsharp_2d_keeps_a_local_source_it_should_not_remove` rules out the two trivial
    /// implementations - returning the input, and returning zeros. The kernel's extent is
    /// pinned by `the_in_plane_voxel_sizes_reach_the_kernel`, its plane by
    /// `slice_axis_is_honoured`, and its behaviour against a real background by the slab tests,
    /// all of which use a quadratic or a dipole field rather than a ramp.
    #[test]
    fn vsharp_2d_removes_an_in_plane_linear_background() {
        let g = grid();
        let m = mask_enclosed();
        let mut field = vec![0.0; g.n_total()];
        for k in 0..NZ {
            for j in 0..NY {
                for i in 0..NX {
                    let (x, y, _) = world(i, j, k);
                    field[i + j * NX + k * NX * NY] = 3.0 + 0.7 * x - 0.4 * y;
                }
            }
        }
        let before = rms(&field, &m);
        let (local, eroded) = vsharp_2d(&field, &m, &g, &VsharpParams::default(), 2, |_, _| {});
        let after = rms(&local, &eroded);
        assert!(before > 1.0, "the test ramp is too small to prove anything: {before}");
        assert!(after < 0.01 * before, "ramp not removed: {before} -> {after}");
    }

    #[test]
    fn vsharp_2d_keeps_a_local_source_it_should_not_remove() {
        // The companion: a method that returned zero for everything would pass the test above.
        let g = grid();
        let m = mask_enclosed();
        let (cx, cy, cz) = world(NX / 2, NY / 2, NZ / 2);
        let mut field = vec![0.0; g.n_total()];
        for k in 0..NZ {
            for j in 0..NY {
                for i in 0..NX {
                    let (x, y, z) = world(i, j, k);
                    let r2 = (x - cx).powi(2) + (y - cy).powi(2) + (z - cz).powi(2);
                    field[i + j * NX + k * NX * NY] = 10.0 * (-r2 / 18.0).exp();
                }
            }
        }
        let (local, eroded) = vsharp_2d(&field, &m, &g, &VsharpParams::default(), 2, |_, _| {});
        assert!(
            rms(&local, &eroded) > 0.3 * rms(&field, &eroded),
            "the local source was removed along with the background"
        );
    }

    #[test]
    fn three_d_vsharp_is_the_better_method_when_the_roi_is_enclosed() {
        // Stated first, because it is the limit on everything below. 2D V-SHARP is not a
        // better V-SHARP: a disc has no mean value property, so wherever a sphere fits, the
        // sphere wins and 2D should not be reached for.
        let g = grid();
        let m = mask_enclosed();
        let field = background_field();
        let (l2, e2) = vsharp_2d(&field, &m, &g, &VsharpParams::default(), 2, |_, _| {});
        let (l3, e3) = vsharp(&field, &m, &g, &VsharpParams::default(), |_, _| {});
        let (r2, r3) = (rms(&l2, &e2), rms(&l3, &e3));
        assert!(
            r3 < 0.5 * r2,
            "3D V-SHARP ({r3:.4}) did not beat 2D ({r2:.4}) on an enclosed ROI - if a disc              matched a sphere where the sphere fits, the mean value property would not be              doing anything"
        );
    }

    #[test]
    fn vsharp_2d_wins_at_the_ends_of_slab_coverage() {
        // And why it is worth having anyway. Where the tissue runs out at the same slice the
        // volume does, a sphere has nothing above or below to average over, and V-SHARP falls
        // back to radii too small to remove much. A disc does not care: it is the same disc
        // on the end slices as in the middle.
        let g = grid();
        let m = mask_slab();
        let field = background_field();
        let ends = [0usize, NZ - 1];
        let middle = [4usize, 5, 6, 7];

        let (l2, e2) = vsharp_2d(&field, &m, &g, &VsharpParams::default(), 2, |_, _| {});
        let (l3, e3) = vsharp(&field, &m, &g, &VsharpParams::default(), |_, _| {});

        let (end_2d, end_3d) = (rms_slices(&l2, &e2, &ends), rms_slices(&l3, &e3, &ends));
        let (mid_2d, mid_3d) = (rms_slices(&l2, &e2, &middle), rms_slices(&l3, &e3, &middle));
        assert!(
            end_3d > 1.8 * end_2d,
            "3D V-SHARP ({end_3d:.4}) was not much worse than 2D ({end_2d:.4}) on the end              slices, which is the only place 2D is supposed to win"
        );
        assert!(
            mid_3d < 0.6 * mid_2d,
            "3D V-SHARP ({mid_3d:.4}) did not beat 2D ({mid_2d:.4}) in the interior - then              the advantage above is not specific to the ends and something else is going on"
        );
    }

    #[test]
    fn the_chain_beats_either_stage_alone_on_slab_coverage() {
        // The arrangement #75 asks for. 2D V-SHARP handles the ends but leaves the
        // through-slice background its discs cannot see; PDF takes that, and the pair beats
        // both 2D alone and plain 3D V-SHARP. Ground truth is zero - every source is outside
        // the ROI - so these residuals are errors, not differences from a reference.
        let g = grid();
        let m = mask_slab();
        let field = background_field();
        let input = rms(&field, &m);
        assert!(input > 1.0, "background field too weak to measure: {input}");

        let (l2, e2) = vsharp_2d(&field, &m, &g, &VsharpParams::default(), 2, |_, _| {});
        let (l3, e3) = vsharp(&field, &m, &g, &VsharpParams::default(), |_, _| {});
        let (lc, ec) =
            vsharp_2d_pdf(&field, &m, &g, (0.0, 0.0, 1.0), &chain_params(&g), |_, _| {}).unwrap();
        let (r2, r3, rc) = (rms(&l2, &e2), rms(&l3, &e3), rms(&lc, &ec));

        assert!(r2 < 0.1 * input, "2D V-SHARP removed almost nothing ({input} -> {r2})");
        assert!(
            rc < 0.92 * r2,
            "chaining PDF did not improve on 2D V-SHARP alone ({r2:.4} -> {rc:.4}); then the              second stage is not earning its place"
        );
        assert!(
            rc < 0.75 * r3,
            "the chain ({rc:.4}) did not beat plain 3D V-SHARP ({r3:.4}) on the geometry it              is for; then there is no reason to prefer it"
        );
    }

    #[test]
    fn the_chain_refuses_a_gapped_acquisition_and_accepts_a_contiguous_one() {
        let g = grid();
        let m = mask_enclosed();
        let field = background_field();

        // 2 mm slabs at the grid's 3 mm pitch: a 1 mm gap
        let gapped = Vsharp2dPdfParams::new(2, 2.0);
        let err = vsharp_2d_pdf(&field, &m, &g, (0.0, 0.0, 1.0), &gapped, |_, _| {})
            .expect_err("a gapped acquisition must be refused");
        assert_eq!(err.gap(), 1.0);
        assert!(err.to_string().contains("not contiguous"));

        let ok = Vsharp2dPdfParams {
            pdf: PdfParams { tol: 1e-4, max_iter: Some(5) },
            ..Vsharp2dPdfParams::contiguous(&g, 2)
        };
        assert!(vsharp_2d_pdf(&field, &m, &g, (0.0, 0.0, 1.0), &ok, |_, _| {}).is_ok());
    }

    #[test]
    fn the_chain_reports_progress_monotonically_to_its_total() {
        let g = grid();
        let m = mask_enclosed();
        let field = background_field();
        let params = Vsharp2dPdfParams {
            pdf: PdfParams { tol: 1e-4, max_iter: Some(5) },
            ..Vsharp2dPdfParams::contiguous(&g, 2)
        };
        let mut seen: Vec<(usize, usize)> = Vec::new();
        vsharp_2d_pdf(&field, &m, &g, (0.0, 0.0, 1.0), &params, |a, b| seen.push((a, b))).unwrap();
        assert!(!seen.is_empty());
        assert!(seen.iter().all(|&(a, b)| a <= b && b == NZ + 5));
        assert!(seen.windows(2).all(|w| w[0].0 <= w[1].0), "progress went backwards");
    }

    #[test]
    fn an_empty_slice_produces_no_output() {
        let g = grid();
        let mut m = mask_enclosed();
        for p in 0..NX * NY {
            m[p + 2 * NX * NY] = 0;
        }
        let field = background_field();
        let (local, eroded) = vsharp_2d(&field, &m, &g, &VsharpParams::default(), 2, |_, _| {});
        assert!(eroded[2 * NX * NY..3 * NX * NY].iter().all(|&v| v == 0));
        assert!(local[2 * NX * NY..3 * NX * NY].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn slice_axis_is_honoured() {
        // Running along the wrong axis is silently plausible, so check the axis is read. A
        // field that varies only along z is constant within every xy slice, so discs in xy
        // remove it outright; discs in xz see its curvature and cannot.
        let g = Grid::new(24, 24, 24, 1.0, 1.0, 1.0);
        let mut m = vec![0u8; g.n_total()];
        let mut field = vec![0.0; g.n_total()];
        for k in 0..24 {
            for j in 0..24 {
                for i in 0..24 {
                    let idx = i + j * 24 + k * 24 * 24;
                    let r = (i as f64 - 11.5).powi(2)
                        + (j as f64 - 11.5).powi(2)
                        + (k as f64 - 11.5).powi(2);
                    m[idx] = u8::from(r <= 9.0 * 9.0);
                    field[idx] = (k as f64 - 11.5).powi(2);
                }
            }
        }
        let residual = |axis: usize| {
            let (l, e) = vsharp_2d(&field, &m, &g, &VsharpParams::default(), axis, |_, _| {});
            rms(&l, &e)
        };
        let (across_z, along_z) = (residual(2), residual(1));
        assert!(
            across_z < 0.02 * rms(&field, &m),
            "discs in the plane the field is constant over did not remove it: {across_z}"
        );
        assert!(
            along_z > 10.0 * across_z,
            "stacking along a different axis gave the same answer ({along_z} vs {across_z}),              so slice_axis is not being read"
        );
    }

    #[test]
    fn vsharp_2d_handles_a_non_square_slice() {
        // Every other geometry here is square in plane, where swapping the slice grid's two
        // extents is a no-op. It is not a no-op in general: the buffer is still the right
        // length, so nothing panics - the slice is simply reshaped wrongly and the answer is
        // quietly scrambled. A ramp along the fast axis of a non-square slice catches it,
        // because reading it at the wrong stride turns the ramp into a sawtooth that no disc
        // can remove.
        //
        // The ramp is load-bearing here for that reason and not despite being linear: it is the
        // discontinuity the wrong stride introduces that this detects, not anything about the
        // kernel. Smoothing the field would make the test pass under the mutation.
        let (nx, ny, nz) = (32usize, 20usize, 6usize);
        let g = Grid::new(nx, ny, nz, 1.0, 1.0, 3.0);
        let mut m = vec![0u8; nx * ny * nz];
        let mut field = vec![0.0; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = i + j * nx + k * nx * ny;
                    let r = ((i as f64 - 15.5) / 13.0).powi(2) + ((j as f64 - 9.5) / 8.0).powi(2);
                    m[idx] = u8::from(r <= 1.0);
                    field[idx] = 0.6 * i as f64 - 0.3 * j as f64;
                }
            }
        }
        let before = rms(&field, &m);
        let (local, eroded) = vsharp_2d(&field, &m, &g, &VsharpParams::default(), 2, |_, _| {});
        let after = rms(&local, &eroded);
        assert!(eroded.iter().any(|&v| v != 0), "everything was eroded away");
        assert!(after < 0.01 * before, "non-square slice: ramp not removed, {before} -> {after}");
    }

    #[test]
    fn the_in_plane_voxel_sizes_reach_the_kernel() {
        // SMV radii are in mm, so the disc covers however many samples that many mm buys. A
        // slice grid built with the wrong spacings would still be a disc, and still produce a
        // plausible answer, just the wrong physical size - so check the spacings get through.
        // Stacking along y puts the 3 mm axis in plane, where changing it has to be visible.
        // the mask needs a boundary for erosion to mean anything: a mask of all ones
        // convolves to one everywhere and nothing is ever removed
        let mut m = vec![0u8; 24 * 24 * 24];
        let mut field = vec![0.0; 24 * 24 * 24];
        for k in 0..24 {
            for j in 0..24 {
                for i in 0..24 {
                    let idx = i + j * 24 + k * 24 * 24;
                    let r = (i as f64 - 11.5).powi(2)
                        + (j as f64 - 11.5).powi(2)
                        + (k as f64 - 11.5).powi(2);
                    m[idx] = u8::from(r <= 9.0 * 9.0);
                    field[idx] = (i % 7) as f64;
                }
            }
        }
        let eroded = |vsz: f64| {
            let g = Grid::new(24, 24, 24, 1.0, 1.0, vsz);
            let (_, e) = vsharp_2d(&field, &m, &g, &VsharpParams::default(), 1, |_, _| {});
            e.iter().filter(|&&v| v != 0).count()
        };
        let (thin, thick) = (eroded(1.0), eroded(3.0));
        assert!(thin > 0 && thin < m.iter().filter(|&&v| v != 0).count(), "nothing eroded: {thin}");
        assert!(
            thin != thick,
            "a 1 mm and a 3 mm through-plane spacing eroded identically ({thin} voxels), so \
             the slice grid is not carrying its voxel sizes into the kernel"
        );
    }

    #[test]
    #[should_panic(expected = "field length must match grid dimensions")]
    fn rejects_a_field_that_does_not_match_the_grid() {
        vsharp_2d(&[0.0; 10], &mask_enclosed(), &grid(), &VsharpParams::default(), 2, |_, _| {});
    }
}

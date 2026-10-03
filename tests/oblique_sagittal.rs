//! Integration test: does the oblique-resample pathway survive a sagittal anisotropic
//! acquisition?
//!
//! [`axial_grid_for`] lays its grid out on the *world* axes but takes voxel sizes from
//! [`voxel_sizes_from_affine`], which indexes them by *voxel* axis. Those two orders coincide only
//! when the acquisition is stored R-A-S, so until #132 the spacings were paired off by position and
//! a sagittal stack had its slice pitch assigned to the wrong world axis: resolution invented
//! across the slice direction, and most of what was acquired head-to-foot thrown away.
//!
//! That fix shipped with a unit test and nothing else, because no integration phantom has a
//! sagittal anisotropic affine — which is the only shape of data that exercises it. This builds one
//! out of the BIDS phantom: relabel it to sagittal storage order, average the slice direction down
//! so the voxels are genuinely anisotropic, and tilt the affine so something actually forces a
//! resample. Then run the pathway QSMxT runs — resample onto the cardinal grid, invert there, map
//! the result back to the acquisition's own voxel space — and check what comes out.
//!
//! Every threshold is pinned from both sides. The same reconstruction runs a second time on the
//! grid the position-paired spacings would have produced, and the test asserts the correct grid is
//! on the good side of each threshold *and* that the buggy one is not. A threshold that cannot tell
//! them apart fails here instead of passing quietly.
//!
//! Run with:
//!   cargo test --release --test oblique_sagittal -- --ignored --nocapture

mod common;

use common::{correlation, nrmse, TestData};
use qsm_core::geometry::{
    alignment_to, axial_grid_for, b0_direction_from_affine, obliquity_from_affine, reorient,
    resample_mask_onto, resample_onto, voxel_sizes_from_affine, AxisCode::*,
};
use qsm_core::inversion::{self, TkdParams};
use qsm_core::Grid;

/// How many 1 mm slices are averaged into one slice of the synthetic stack. The phantom's 164 L-R
/// voxels divide by 4 exactly, giving a 1 x 1 x 4 mm sagittal acquisition — anisotropic enough that
/// pairing the spacings off by position is unmistakable, and a ratio real protocols reach.
const SLICE_FACTOR: usize = 4;

/// Tilts of the slice stack: degrees about world S-I, then about world A-P. Both are *in-plane*
/// axes for a sagittal stack, which is the double-oblique case — the one where identifying the
/// slice axis from an anisotropic affine is hardest, and the one a tilt about the slice normal
/// never reaches. Enough obliquity that a resample is warranted at all, and far from the 45° where
/// `ras_axes` would stop calling the stack sagittal.
const TILTS: [(usize, f64); 2] = [(2, 18.0), (1, 12.0)];

/// Correlation the whole pathway has to clear on this acquisition. Measured 0.843 with the
/// spacings right and 0.789 with them paired off by position, so the floor sits between the two
/// and the test asserts that it does.
const CORR_FLOOR: f64 = 0.82;

/// How much better the correct grid has to be than the position-paired one, as a fraction of the
/// detail the correct grid keeps along S-I. Measured 0.159; a tenth leaves room for FFT jitter
/// across platforms without letting the two runs converge unnoticed.
const MIN_DETAIL_GAP: f64 = 0.10;

/// A sagittal anisotropic oblique acquisition synthesised from the phantom, with everything a
/// reconstruction needs expressed in its own voxel space.
struct Sagittal {
    dims: (usize, usize, usize),
    /// Row-major voxel→world. Voxel axis 0 runs A-P at 1 mm, axis 1 S-I at 1 mm, axis 2 L-R at the
    /// slice pitch — before the tilt, which rotates all three together.
    affine: [f64; 16],
    local_field: Vec<f64>,
    mask: Vec<u8>,
    /// Ground-truth χ, averaged down the same way, so it describes this grid's voxels.
    chi: Vec<f64>,
}

/// Relabel the phantom to sagittal storage order, then average `SLICE_FACTOR` slices together along
/// the new slice axis. Averaging is what a thick-slice acquisition does, so the result is a stack
/// that could have come off a scanner rather than a subsampled copy of a thin-slice one.
fn build_sagittal(data: &TestData) -> Sagittal {
    // The phantom is stored R-A-S with an identity affine.
    let ras = identity_affine(data.voxel_size);
    // Voxel axis 0 -> A-P, axis 1 -> S-I, axis 2 -> L-R: sagittal slices, stacked left to right.
    let (perm, flip) = alignment_to(&ras, [A, S, R]);
    let (field, sag_dims) = reorient(&data.fieldmap_local, data.dims, perm, flip);
    let (chi, _) = reorient(&data.chi, data.dims, perm, flip);
    let (mask, _) = reorient(&data.mask, data.dims, perm, flip);

    assert_eq!(
        sag_dims.2 % SLICE_FACTOR,
        0,
        "the slice axis ({}) has to divide by {SLICE_FACTOR} for the averaging to be clean",
        sag_dims.2,
    );
    let dims = (sag_dims.0, sag_dims.1, sag_dims.2 / SLICE_FACTOR);

    let field = average_slices(&field, sag_dims);
    let chi = average_slices(&chi, sag_dims);
    // A thick voxel is in the mask if most of what it covers was. Taking the middle sub-slice would
    // also work; a majority just keeps the mask from fraying at the edges.
    let mask_fraction = average_slices(&mask.iter().map(|&m| f64::from(m)).collect::<Vec<_>>(), sag_dims);
    let mask: Vec<u8> = mask_fraction.iter().map(|&v| u8::from(v > 0.5)).collect();

    let (vsx, vsy, vsz) = data.voxel_size;
    let slice_mm = vsx * SLICE_FACTOR as f64; // L-R came from the phantom's x axis.
    // Columns in voxel-axis order: A-P at vsy, S-I at vsz, L-R at the slice pitch.
    let cardinal = [
        0.0, 0.0, slice_mm, 0.0,
        vsy, 0.0, 0.0, 0.0,
        0.0, vsz, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    let tilted = TILTS.iter().fold(cardinal, |a, &(axis, deg)| tilt_about(&a, axis, deg));
    let affine = centre_on_origin(tilted, dims);

    Sagittal { dims, affine, local_field: field, mask, chi }
}

fn identity_affine(voxel_size: (f64, f64, f64)) -> [f64; 16] {
    let (vsx, vsy, vsz) = voxel_size;
    [
        vsx, 0.0, 0.0, 0.0,
        0.0, vsy, 0.0, 0.0,
        0.0, 0.0, vsz, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// Average blocks of `SLICE_FACTOR` along the last (column-major slowest) axis.
fn average_slices(data: &[f64], dims: (usize, usize, usize)) -> Vec<f64> {
    let plane = dims.0 * dims.1;
    let out_nz = dims.2 / SLICE_FACTOR;
    let mut out = vec![0.0f64; plane * out_nz];
    for k in 0..out_nz {
        for s in 0..SLICE_FACTOR {
            let src = (k * SLICE_FACTOR + s) * plane;
            for i in 0..plane {
                out[k * plane + i] += data[src + i];
            }
        }
        for v in &mut out[k * plane..(k + 1) * plane] {
            *v /= SLICE_FACTOR as f64;
        }
    }
    out
}

/// Rotate a voxel→world affine about a world axis. A left-multiply, so the voxel grid tilts and
/// the voxel sizes are untouched.
fn tilt_about(affine: &[f64; 16], world_axis: usize, degrees: f64) -> [f64; 16] {
    let (s, c) = degrees.to_radians().sin_cos();
    let rot = match world_axis {
        0 => [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        1 => [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        _ => [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
    };
    let mut out = *affine;
    for row in 0..3 {
        for col in 0..4 {
            out[4 * row + col] = (0..3).map(|k| rot[row][k] * affine[4 * k + col]).sum();
        }
    }
    out
}

/// Put the volume's centre at the world origin, so the translations are of a realistic size instead
/// of leaving one corner there.
fn centre_on_origin(mut affine: [f64; 16], dims: (usize, usize, usize)) -> [f64; 16] {
    let centre = [
        (dims.0 - 1) as f64 / 2.0,
        (dims.1 - 1) as f64 / 2.0,
        (dims.2 - 1) as f64 / 2.0,
    ];
    for row in 0..3 {
        affine[4 * row + 3] = -(0..3).map(|k| affine[4 * row + k] * centre[k]).sum::<f64>();
    }
    affine
}

/// Mean absolute first difference along one voxel axis, inside the mask — how much detail the
/// volume still carries in that direction.
///
/// This is the probe that matters here. Both grids put the χ back on the same voxels, and a
/// whole-volume error metric mostly reports the large-scale structure both get right; what the
/// wrong spacings actually destroy is fine detail along one specific direction, by sampling an axis
/// that was acquired at 1 mm at the 4 mm slice pitch. Measuring the gradient along that axis sees
/// it directly, where range-normalised NRMSE barely moves.
fn detail_along(data: &[f64], dims: (usize, usize, usize), mask: &[u8], axis: usize) -> f64 {
    let (nx, ny, nz) = dims;
    let stride = match axis {
        0 => 1,
        1 => nx,
        _ => nx * ny,
    };
    let len = [nx, ny, nz][axis];
    let mut sum = 0.0;
    let mut count = 0usize;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if [i, j, k][axis] + 1 >= len {
                    continue;
                }
                let idx = i + j * nx + k * nx * ny;
                if mask[idx] > 0 && mask[idx + stride] > 0 {
                    sum += (data[idx + stride] - data[idx]).abs();
                    count += 1;
                }
            }
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

/// A cardinal destination grid — diagonal affine, dimensions sized to cover `world_extent` — built
/// the way [`axial_grid_for`] builds one, so swapping in different spacings isolates that choice.
fn cardinal_grid(
    voxel_size: (f64, f64, f64),
    world_min: [f64; 3],
    world_extent: [f64; 3],
) -> ((usize, usize, usize), [f64; 16]) {
    let vs = [voxel_size.0, voxel_size.1, voxel_size.2];
    let dims = (
        (world_extent[0] / vs[0]).ceil() as usize + 1,
        (world_extent[1] / vs[1]).ceil() as usize + 1,
        (world_extent[2] / vs[2]).ceil() as usize + 1,
    );
    let affine = [
        vs[0], 0.0, 0.0, world_min[0],
        0.0, vs[1], 0.0, world_min[1],
        0.0, 0.0, vs[2], world_min[2],
        0.0, 0.0, 0.0, 1.0,
    ];
    (dims, affine)
}

/// One trip through the pathway: resample onto the destination grid, invert there, map the χ back
/// onto the acquisition's own grid.
fn reconstruct_via(
    sag: &Sagittal,
    dst_dims: (usize, usize, usize),
    dst_affine: &[f64; 16],
    dst_voxel: (f64, f64, f64),
) -> Vec<f64> {
    let field = resample_onto(&sag.local_field, sag.dims, &sag.affine, dst_dims, dst_affine)
        .expect("the acquisition's affine is invertible");
    let mask = resample_mask_onto(&sag.mask, sag.dims, &sag.affine, dst_dims, dst_affine)
        .expect("the acquisition's affine is invertible");

    // The destination grid is cardinal, so B0 is its +z by construction — that is what the resample
    // is for. Reading it off the affine rather than assuming it keeps this honest.
    let bdir = b0_direction_from_affine(dst_affine);
    assert!(bdir.2 > 1.0 - 1e-9, "the destination grid should be axial, got bdir {bdir:?}");

    let grid = Grid::new(dst_dims.0, dst_dims.1, dst_dims.2, dst_voxel.0, dst_voxel.1, dst_voxel.2);
    let chi = inversion::tkd(&field, &mask, &grid, bdir, &TkdParams { threshold: 0.2 });

    resample_onto(&chi, dst_dims, dst_affine, sag.dims, &sag.affine)
        .expect("a cardinal affine is invertible")
}

/// The spacing fix from #132, on the data that exercises it: a tilted sagittal 1 x 1 x 4 mm stack
/// reconstructed through the cardinal-resample pathway.
///
/// Checks both the geometry (what grid gets built) and the consequence (how good the χ is, back on
/// the acquisition's own voxels), each against the grid the pre-#132 code would have built.
#[test]
#[ignore]
fn sagittal_anisotropic_reconstruction_keeps_its_acquired_resolution() {
    let data = TestData::load().expect("Failed to load test data");
    let sag = build_sagittal(&data);
    let obliquity = obliquity_from_affine(&sag.affine);
    println!("[INFO] synthetic acquisition: dims {:?}, obliquity {obliquity:.1}°", sag.dims);
    assert!(
        obliquity > 10.0,
        "the stack has to be oblique enough that a resample is warranted at all, got {obliquity:.1}°",
    );

    // ── The grid itself ───────────────────────────────────────────────────────────────────────
    let g = axial_grid_for(sag.dims.0, sag.dims.1, sag.dims.2, &sag.affine);
    let slice_mm = data.voxel_size.0 * SLICE_FACTOR as f64;
    println!("[INFO] axial grid: dims {:?}, voxel size {:?}", g.dims, g.voxel_size);

    // Each spacing belongs on the world axis its voxel axis samples. L-R is the slice direction, so
    // L-R gets the thick pitch and the two in-plane axes keep 1 mm.
    for (axis, got, want) in [
        ("L-R", g.voxel_size.0, slice_mm),
        ("A-P", g.voxel_size.1, data.voxel_size.1),
        ("S-I", g.voxel_size.2, data.voxel_size.2),
    ] {
        assert!(
            (got - want).abs() < 1e-9,
            "{axis} spacing is {got}, expected {want} — spacings followed position, not geometry",
        );
    }

    // ── The grid the bug would have built ────────────────────────────────────────────────────
    // `voxel_sizes_from_affine` returns spacings per voxel axis; the old code handed that tuple
    // straight to the world axes.
    let by_position = voxel_sizes_from_affine(&sag.affine);
    let world_min = [g.affine[3], g.affine[7], g.affine[11]];
    let extent = [
        (g.dims.0 - 1) as f64 * g.voxel_size.0,
        (g.dims.1 - 1) as f64 * g.voxel_size.1,
        (g.dims.2 - 1) as f64 * g.voxel_size.2,
    ];
    let (buggy_dims, buggy_affine) = cardinal_grid(by_position, world_min, extent);
    println!("[INFO] position-paired grid: dims {buggy_dims:?}, voxel size {by_position:?}");

    // The two grids have to actually differ, or the rest of this proves nothing. S-I is where it
    // hurts: the acquisition resolved it at 1 mm and the position-paired grid samples it at the
    // slice pitch, while L-R gets 1 mm it never acquired.
    assert!(
        (by_position.2 - slice_mm).abs() < 1e-9,
        "the position-paired grid should put the slice pitch on S-I, got {by_position:?}",
    );
    assert!(
        buggy_dims.0 > g.dims.0 * 2,
        "position-paired L-R should blow up: {} vs {}",
        buggy_dims.0,
        g.dims.0,
    );
    assert!(
        buggy_dims.2 * 2 < g.dims.2,
        "position-paired S-I should collapse: {} vs {}",
        buggy_dims.2,
        g.dims.2,
    );

    // ── What that costs the reconstruction ───────────────────────────────────────────────────
    let chi_fixed = reconstruct_via(&sag, g.dims, &g.affine, g.voxel_size);
    let chi_buggy = reconstruct_via(&sag, buggy_dims, &buggy_affine, by_position);

    let r_fixed = correlation(&chi_fixed, &sag.chi, &sag.mask);
    let r_buggy = correlation(&chi_buggy, &sag.chi, &sag.mask);
    let e_fixed = nrmse(&chi_fixed, &sag.chi, &sag.mask);
    let e_buggy = nrmse(&chi_buggy, &sag.chi, &sag.mask);
    println!("[INFO] correct spacings: r = {r_fixed:.4}, NRMSE = {:.2}%", e_fixed * 100.0);
    println!("[INFO] position-paired:  r = {r_buggy:.4}, NRMSE = {:.2}%", e_buggy * 100.0);

    // Range-normalised NRMSE is ~1.5% for both runs and does not separate them: it is dominated
    // by the large-scale structure both grids get right, and the phantom's range is set by a few
    // extreme voxels. Stated here so nobody adds a threshold it cannot support — asserting
    // `< 20%` against values near 1% is a bound that can never fail, which is how #132's own
    // orientation test shipped with a vacuous assertion.
    assert!(r_fixed > CORR_FLOOR, "correlation {r_fixed:.4} below the {CORR_FLOOR} floor");
    assert!(
        r_buggy < CORR_FLOOR,
        "the position-paired grid scored r = {r_buggy:.4}, above the {CORR_FLOOR} floor this test \
         asserts — the threshold no longer separates the fix from the bug",
    );

    // What is actually lost, measured where it is lost: detail along source voxel axis 1, which
    // runs S-I. The acquisition resolved it at 1 mm; the position-paired grid resamples it at the
    // 4 mm slice pitch and the fine structure does not survive the round trip.
    let d_truth = detail_along(&sag.chi, sag.dims, &sag.mask, 1);
    let d_fixed = detail_along(&chi_fixed, sag.dims, &sag.mask, 1);
    let d_buggy = detail_along(&chi_buggy, sag.dims, &sag.mask, 1);
    println!(
        "[INFO] S-I detail retained: correct {:.3}, position-paired {:.3} (of truth {d_truth:.5})",
        d_fixed / d_truth,
        d_buggy / d_truth,
    );
    let gap = (d_fixed - d_buggy) / d_fixed;
    assert!(
        gap > MIN_DETAIL_GAP,
        "the correct grid kept only {:.1}% more S-I detail than the position-paired one, under the \
         {:.0}% this test asserts — either the fix regressed or the two grids stopped differing",
        gap * 100.0,
        MIN_DETAIL_GAP * 100.0,
    );

    common::save_center_slices(&chi_fixed, &sag.mask, sag.dims, "oblique_sagittal_chi");
    common::save_center_slices(&sag.chi, &sag.mask, sag.dims, "oblique_sagittal_truth");
}

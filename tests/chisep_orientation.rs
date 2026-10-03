//! Integration test: how much does the array orientation move χ-sepnet and R2PRIMEnet?
//!
//! The SNU-LIST networks take no B0 direction and no affine — they see a [`Grid`] and index the
//! volume column-major — so the orientation the caller hands them is the orientation they
//! reconstruct in. The authors' own guidance ("the B0 direction correction to `[0, 0, 1]` is
//! required"; "input data with the same orientation with trained data is recommended") says that
//! matters, but none of the reference implementations enforce it, and this test measures what it
//! actually costs. It relabels the volume's axes (a lossless gather — no interpolation), runs the
//! network, relabels the result back, and compares against the untouched run.
//!
//! This is a characterisation test, not a correctness one: the thresholds pin the sensitivity
//! that was measured when the recipe was written, so a change that makes either network much more
//! orientation-dependent — a transposed patch, a broken sliding window, a wrong channel order —
//! shows up here instead of silently shifting results for anyone whose data is not RAS.
//!
//! Run with:
//!   QSMCI_CHISEP=/path/to/chisep cargo test --release --features "onnx download" \
//!     --test chisep_orientation -- --ignored --nocapture
#![cfg(feature = "onnx")]

mod common;

use common::{chisep_score, correlation, load_chisep_phantom, nrmse};
use qsm_core::geometry::{alignment_to, invert_alignment, reorient, AxisCode::*};
use qsm_core::separation::{chisepnet, ChiSepNetNorm, ChiSepNetParams};
use qsm_core::Grid;

/// The phantom is stored RAS. Two relabellings: the quarter turn that separates RAS from the
/// training order, which is the case this test exists for, and one that takes the slice axis off
/// B0 entirely, which is the worst the cardinal group can do. A pure mirror used to sit between
/// them and scored within 0.001 of the quarter turn, so it earned nothing for the ~5 minutes it
/// cost on a CI runner. The 48-way algebra is covered far more cheaply by the unit tests on
/// [`reorient`].
const CASES: &[(&str, [qsm_core::geometry::AxisCode; 3])] = &[
    ("PRS in-plane quarter turn", [P, R, S]),
    ("SAR slice axis to L-R", [S, A, R]),
];

/// A cardinal RAS affine for the phantom's grid — the orientation the maps are stored in.
fn ras_affine(grid: &Grid) -> [f64; 16] {
    let mut a = [0.0; 16];
    a[0] = grid.vsx();
    a[5] = grid.vsy();
    a[10] = grid.vsz();
    a[15] = 1.0;
    a
}

#[test]
#[ignore]
fn chisepnet_is_insensitive_to_array_orientation() {
    let Some(ph) = load_chisep_phantom() else {
        println!("Skipping: chisep phantom not found");
        return;
    };
    let spec = qsm_core::models::find_model("chi-sepnet").expect("chi-sepnet in registry");
    let w = qsm_core::models::primary_weight_bytes(spec)
        .expect("chi-sepnet weights (set $QSM_MODEL_DIR or build with 'download')");
    let affine = ras_affine(&ph.grid);
    let patch = ChiSepNetParams::default().patch;

    let run = |field: &[f64], qsm: &[f64], r2p: &[f64], mask: &[u8], grid: &Grid| {
        chisepnet(field, qsm, r2p, mask, grid, &w, &ChiSepNetNorm::default(),
                  &ChiSepNetParams { patch })
            .expect("chisepnet")
    };
    let (ref_pos, ref_neg, _) =
        run(&ph.local_field_ppm, &ph.chi_total, &ph.r2prime, &ph.mask, &ph.grid);

    for (label, target) in CASES {
        let t = std::time::Instant::now();
        let (perm, flip) = alignment_to(&affine, *target);
        let (field, dims) = reorient(&ph.local_field_ppm, ph.dims, perm, flip);
        let (qsm, _) = reorient(&ph.chi_total, ph.dims, perm, flip);
        let (r2p, _) = reorient(&ph.r2prime, ph.dims, perm, flip);
        let (mask, _) = reorient(&ph.mask, ph.dims, perm, flip);
        let vs = [ph.grid.vsx(), ph.grid.vsy(), ph.grid.vsz()];
        let grid = Grid::new(dims.0, dims.1, dims.2, vs[perm[0]], vs[perm[1]], vs[perm[2]]);

        let (pos, neg, _) = run(&field, &qsm, &r2p, &mask, &grid);
        let (back, flip_back) = invert_alignment(perm, flip);
        let (pos, _) = reorient(&pos, dims, back, flip_back);
        let (neg, _) = reorient(&neg, dims, back, flip_back);

        // Scored against the untouched run rather than ground truth: the question is whether
        // the network answers the same way, not whether it answers well.
        let secs = t.elapsed().as_secs_f64();
        chisep_score(&format!("{label} χ+"), &pos, &ref_pos, &ph.mask, ph.dims, secs);
        chisep_score(&format!("{label} χ−"), &neg, &ref_neg, &ph.mask, ph.dims, secs);
        let (rp, ep) = (correlation(&pos, &ref_pos, &ph.mask),
                        nrmse(&pos, &ref_pos, &ph.mask) * 100.0);
        let (rn, en) = (correlation(&neg, &ref_neg, &ph.mask),
                        nrmse(&neg, &ref_neg, &ph.mask) * 100.0);

        // Measured band. Note [`nrmse`] normalises by the *range* of the reference, not its
        // norm, so these run far smaller than the relative errors in the module docs: the worst
        // case here (slice axis to L-R) is 1.6% range-normalised against roughly 12% relative.
        // Worst observed: χ+ r=0.989 / 1.6%, χ− r=0.965 / 1.4%. The field channel in this
        // phantom is a forward dipole of its own χ map, so it is cleaner than real data, which
        // moves roughly 2× further; the bounds leave headroom for that and no more. What they
        // catch is the network becoming genuinely orientation-bound: a transposed patch, a
        // broken sliding window, a permuted channel order.
        assert!(rp > 0.96, "{label}: χ+ correlation {rp:.4} collapsed");
        assert!(rn > 0.92, "{label}: χ− correlation {rn:.4} collapsed");
        assert!(ep < 6.0 && en < 6.0, "{label}: NRMSE {ep:.1}% / {en:.1}% too large");
    }
}

/// Relabelling the axes and relabelling back must be exactly the identity on the data — the
/// property the orientation test above relies on to compare runs voxel for voxel.
#[test]
fn relabelling_round_trips_on_the_phantom_grid() {
    let grid = Grid::new(5, 7, 9, 1.0, 1.2, 2.0);
    let n = grid.n_total();
    let data: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let affine = ras_affine(&grid);
    for (_, target) in CASES {
        let (perm, flip) = alignment_to(&affine, *target);
        let (moved, dims) = reorient(&data, grid.dims, perm, flip);
        let (back_perm, back_flip) = invert_alignment(perm, flip);
        let (back, back_dims) = reorient(&moved, dims, back_perm, back_flip);
        assert_eq!(back_dims, grid.dims);
        assert_eq!(back, data);
    }
}

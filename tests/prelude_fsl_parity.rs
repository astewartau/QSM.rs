//! `unwrap::unwrap_prelude` against FSL's `prelude` (fsl-fugue 2603.0), on small noisy phantoms.
//!
//! `tests/data/prelude_fsl_small.bin` is made by `tests/data/prelude_fsl_small.py` with QSM-CI's
//! `fsl-prelude` image (QSMxT/QSM-CI#302; built locally from its Dockerfile and lock file, as FSL's
//! licence keeps the image unpublished), so it can be regenerated in Actions
//! (`.github/workflows/regenerate-fixtures.yml`). Two phantoms, 72 and 64 voxels wide, each with
//! FSL's wrap counts for its defaults, `--labelslices`, `-f` and `-n 6`. The phantoms are noisy
//! enough that the slice and full-3D labellings disagree, so each run pins one mode, and the two
//! widths pin where FSL's default switches from full-3D to slice labelling
//! ([`PreludeLabelling::Auto`]).
//!
//! The comparison is of wrap counts up to one whole-turn offset over the (connected) mask: every
//! PRELUDE output is defined only up to that, and FSL's choice of it is not the median wrap count
//! that [`PreludeParams::correct_global`] takes, nor any other rule worked out (see the module
//! docs). Measured 2026-11-02: every voxel of every run takes FSL's wrap count, up to that offset.

use qsm_core::unwrap::{unwrap_prelude, PreludeLabelling, PreludeParams};
use qsm_core::Grid;

const CASES: [(usize, usize, usize); 2] = [(72, 24, 16), (64, 24, 16)];

/// One FSL run per entry: its options, and the settings here that should reproduce it.
fn runs() -> [(&'static str, PreludeParams); 4] {
    let p = |num_phase_partitions, labelling| PreludeParams {
        num_phase_partitions,
        labelling,
        ..PreludeParams::default()
    };
    [
        ("(defaults)", PreludeParams::default()),
        ("--labelslices", p(8, PreludeLabelling::Slices)),
        ("-f", p(8, PreludeLabelling::Volume)),
        ("-n 6", p(6, PreludeLabelling::Auto)),
    ]
}

/// One case: the mask, the wrapped phase on the grid, and FSL's wrap counts at the mask voxels
/// per run.
type Case = (Vec<u8>, Vec<f64>, Vec<Vec<i8>>);

fn load() -> Vec<Case> {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/prelude_fsl_small.bin")).expect("fixture");
    let mut o = 0;
    let out = CASES
        .iter()
        .map(|d| {
            let n = d.0 * d.1 * d.2;
            let mask = raw[o..o + n].to_vec();
            o += n;
            let inside: Vec<usize> = (0..n).filter(|&i| mask[i] != 0).collect();
            let m = inside.len();
            let mut phase = vec![0.0; n];
            for (k, &v) in inside.iter().enumerate() {
                phase[v] = f32::from_le_bytes(raw[o + 4 * k..o + 4 * k + 4].try_into().unwrap()) as f64;
            }
            o += 4 * m;
            let turns = (0..runs().len())
                .map(|_| {
                    let t = raw[o..o + m].iter().map(|&b| b as i8).collect();
                    o += m;
                    t
                })
                .collect();
            (mask, phase, turns)
        })
        .collect();
    assert_eq!(o, raw.len());
    out
}

/// In-mask voxels whose wrap count differs from FSL's beyond the most common offset, and that
/// offset.
fn differing(out: &[f64], phase: &[f64], mask: &[u8], fsl: &[i8]) -> (usize, i64) {
    let ours: Vec<i64> = (0..mask.len())
        .filter(|&i| mask[i] != 0)
        .map(|i| ((out[i] - phase[i]) / std::f64::consts::TAU).round() as i64)
        .collect();
    let diff: Vec<i64> = ours.iter().zip(fsl).map(|(&a, &b)| a - b as i64).collect();
    let mut counts = std::collections::BTreeMap::new();
    for &d in &diff {
        *counts.entry(d).or_insert(0usize) += 1;
    }
    let offset = *counts.iter().max_by_key(|(_, &c)| c).unwrap().0;
    (diff.iter().filter(|&&d| d != offset).count(), offset)
}

#[test]
fn prelude_matches_fsl_prelude() {
    for (d, (mask, phase, fsl)) in CASES.iter().zip(load()) {
        let grid = Grid::new(d.0, d.1, d.2, 1.0, 1.0, 1.0);
        let mut outputs = vec![];
        for ((opts, params), fsl) in runs().iter().zip(&fsl) {
            let out = unwrap_prelude(&phase, &mask, &grid, params);
            let (n, offset) = differing(&out, &phase, &mask, fsl);
            println!("{d:?} prelude {opts}: {n} voxels with another wrap count (offset {offset}·2π)");
            assert_eq!(n, 0, "{d:?} prelude {opts}: wrap counts differ");
            outputs.push(out);
        }
        // The fixture tells the labellings apart: slice and full-3D labelling disagree on both
        // phantoms, and FSL's default sides with slices on the wide one and full 3D on the other.
        let (slices_vs_volume, _) = differing(&outputs[1], &phase, &mask, &fsl[2]);
        assert!(slices_vs_volume > 0, "{d:?}: the phantom does not separate the labellings");
        let expect = if d.0 > 64 { PreludeLabelling::Slices } else { PreludeLabelling::Volume };
        assert_eq!(PreludeLabelling::Auto.resolve(*d), expect);
    }
}

//! `utils::n4_bias_correction` against ANTs 2.4.3's `N4BiasFieldCorrection` on a small synthetic
//! case.
//!
//! `tests/data/n4_ants_small.bin` (little-endian f32, column-major, 30x32x26 grid, voxel
//! 1x1x1.5 mm, identity orientation) holds:
//! - the input: three intensity classes (60, 100, 160) in a sphere, Gaussian noise (σ = 3) inside
//!   it, times the bias `exp(0.35x − 0.25y + 0.2z²)` (x, y, z in [−1, 1] across the grid), zero
//!   outside;
//! - the bias field from `N4BiasFieldCorrection -i in.nii -o [n4.nii,bias.nii]` (all defaults);
//! - the corrected image from `N4BiasFieldCorrection -i in.nii -x mask.nii -o n4m.nii` with the
//!   mask `input > 30` (so with ANTs' default rescaling to the input's range in the mask).
//!
//! ITK runs N4 in single precision and multithreaded, and the rounding is amplified over the
//! iterations: ANTs' own results here change by up to 2e-5 (bias field) and 3e-4 (masked,
//! rescaled output) relative between thread counts. The tolerances allow for that.

use qsm_core::utils::{n4_bias_correction, N4Params};
use qsm_core::Grid;

const DIMS: (usize, usize, usize) = (30, 32, 26);

fn load() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/n4_ants_small.bin")).expect("fixture");
    let n = DIMS.0 * DIMS.1 * DIMS.2;
    assert_eq!(raw.len(), 3 * 4 * n);
    let f = |a: usize| -> Vec<f64> {
        (0..n).map(|i| f32::from_le_bytes(raw[4 * (a * n + i)..4 * (a * n + i) + 4].try_into().unwrap()) as f64).collect()
    };
    (f(0), f(1), f(2))
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).filter(|(_, y)| **y != 0.0).map(|(x, y)| ((x - y) / y).abs()).fold(0.0, f64::max)
}

#[test]
fn n4_matches_ants_defaults() {
    let (input, ants_bias, _) = load();
    let grid = Grid::new(DIMS.0, DIMS.1, DIMS.2, 1.0, 1.0, 1.5);
    let out = n4_bias_correction(&input, None, &grid, &N4Params::default(), |_, _| {});
    let d = max_rel(&out.bias_field, &ants_bias);
    println!("bias field vs ANTs: max relative difference {d:.2e}");
    assert!(d < 2e-4, "bias field differs from ANTs by {d}");
}

#[test]
fn n4_matches_ants_with_mask() {
    let (input, _, ants_masked) = load();
    let grid = Grid::new(DIMS.0, DIMS.1, DIMS.2, 1.0, 1.0, 1.5);
    let mask: Vec<u8> = input.iter().map(|&v| (v > 30.0) as u8).collect();
    let out = n4_bias_correction(&input, Some(&mask), &grid, &N4Params::default(), |_, _| {});
    let d = max_rel(&out.corrected, &ants_masked);
    println!("masked, rescaled output vs ANTs: max relative difference {d:.2e}");
    assert!(d < 1e-3, "corrected image differs from ANTs by {d}");
}

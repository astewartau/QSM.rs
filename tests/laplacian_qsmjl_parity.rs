//! `unwrap::laplacian_unwrap` with [`LaplacianSolver::Dct`] against QSM.jl's `unwrap_laplacian`
//! with `solver=:dct` (kamesy/QSM.jl fde4b92).
//!
//! `tests/data/laplacian_qsmjl_small.bin` is made by `tests/data/laplacian_qsmjl_small.jl` in the
//! verified QSM-CI `julia-qsm` image (QSMxT/QSM-CI#299; the command is in its header, and
//! `.github/workflows/regenerate-fixtures.yml` runs it in Actions): per case, the wrapped phase
//! over the grid and QSM.jl's unwrapped phase (f64, column-major).

use qsm_core::unwrap::{laplacian_unwrap, LaplacianSolver};
use qsm_core::Grid;

/// (size, voxel size) of the generator's cases, in order.
const CASES: [([usize; 3], [f64; 3]); 2] = [([20, 18, 16], [1.0, 1.0, 1.0]), ([18, 20, 14], [0.9, 1.0, 1.3])];

#[test]
fn laplacian_dct_matches_qsmjl() {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/laplacian_qsmjl_small.bin")).expect("fixture");
    let mut o = 0;
    let mut f64s = |k: usize| -> Vec<f64> {
        let v = (0..k).map(|i| f64::from_le_bytes(raw[o + 8 * i..o + 8 * i + 8].try_into().unwrap())).collect();
        o += 8 * k;
        v
    };
    for (n, v) in CASES {
        let len = n.iter().product();
        let (phase, want) = (f64s(len), f64s(len));
        let grid = Grid::new(n[0], n[1], n[2], v[0], v[1], v[2]);
        let got = laplacian_unwrap(&phase, &vec![1u8; len], &grid, LaplacianSolver::Dct);
        let max = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let rms = (want.iter().map(|b| b * b).sum::<f64>() / len as f64).sqrt();
        println!("{n:?} at {v:?}: max |Δ| {max:.2e} rad (rms {rms:.2})");
        // Measured 2026-10-30: max |Δ| at double-precision rounding (see the assertion).
        assert!(max < 1e-10, "{n:?}: max |Δ| {max:e}");
    }
}

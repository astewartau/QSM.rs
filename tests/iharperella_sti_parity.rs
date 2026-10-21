//! `bgremove::iharperella` against STI Suite 3.0's `iHARPERELLA` on a small phantom.
//!
//! The input is rebuilt here exactly as the MATLAB script that made the references built it
//! (a smooth field in an ellipsoid mask, 32x32x24, 1-based `ndgrid` coordinates), so the
//! fixture only holds STI's output. `tests/data/iharperella_sti_small.bin` (little-endian): for
//! each case in [`CASES`] order, the number of mask voxels (u32) and STI's output there (f64,
//! column-major), from MATLAB R2023b
//! `tp = iHARPERELLA(f, mask, 'voxelsize', vs, 'padsize', pad, 'niter', niter)`.
//! STI's output is zero outside the mask.
//!
//! LSQR amplifies rounding after about ten iterations, so the 40-iteration case is compared
//! to a looser tolerance than the short ones.

use qsm_core::bgremove::{iharperella, HarperellaInput, IharperellaParams};
use qsm_core::Grid;

const DIMS: [usize; 3] = [32, 32, 24];

struct Case {
    vs: [f64; 3],
    pad: [usize; 3],
    niter: usize,
    /// Allowed max |difference| relative to max |STI|
    tol: f64,
}

const CASES: [Case; 3] = [
    Case { vs: [1.0, 1.0, 1.5], pad: [4, 5, 6], niter: 5, tol: 1e-12 },
    Case { vs: [1.0, 1.0, 1.0], pad: [0, 0, 0], niter: 8, tol: 1e-12 },
    Case { vs: [0.8, 0.9, 1.2], pad: [3, 3, 3], niter: 40, tol: TOL_40 },
];

/// STI's default 40 iterations: 3.9e-2 measured (rounding amplified by LSQR, see module docs).
const TOL_40: f64 = 0.1;

fn phantom() -> (Vec<f64>, Vec<u8>) {
    let n = DIMS;
    let mut f = vec![0.0; n[0] * n[1] * n[2]];
    let mut mask = vec![0u8; f.len()];
    for k in 0..n[2] {
        for j in 0..n[1] {
            for i in 0..n[0] {
                let (x, y, z) = ((i + 1) as f64, (j + 1) as f64, (k + 1) as f64);
                let l = i + j * n[0] + k * n[0] * n[1];
                f[l] = 0.3 * (0.4 * x + 0.2 * y).sin() * (0.3 * z).cos() + 0.002 * x * y;
                mask[l] = u8::from((x - 16.5).powi(2) + (y - 16.5).powi(2) + ((z - 12.5) * 1.2).powi(2) < 100.0);
            }
        }
    }
    (f, mask)
}

fn read_fixture() -> Vec<Vec<f64>> {
    let bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/iharperella_sti_small.bin")).unwrap();
    let mut pos = 0;
    let mut out = Vec::new();
    for _ in 0..CASES.len() {
        let n = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        out.push((0..n).map(|i| f64::from_le_bytes(bytes[pos + 8 * i..pos + 8 * i + 8].try_into().unwrap())).collect());
        pos += 8 * n;
    }
    assert_eq!(pos, bytes.len());
    out
}

#[test]
fn iharperella_matches_sti_suite() {
    let (f, mask) = phantom();
    let refs = read_fixture();
    for (case, sti) in CASES.iter().zip(&refs) {
        let grid = Grid::new(DIMS[0], DIMS[1], DIMS[2], case.vs[0], case.vs[1], case.vs[2]);
        let p = IharperellaParams { max_iter: case.niter, pad: case.pad, ..Default::default() };
        let (out, _) = iharperella(HarperellaInput::FieldPpm(&f), &mask, &grid, &p, |_, _| {}).unwrap();
        let ours: Vec<f64> = out.iter().zip(&mask).filter(|(_, &m)| m != 0).map(|(&v, _)| v).collect();
        assert_eq!(ours.len(), sti.len());
        let scale = sti.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let d = ours.iter().zip(sti).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / scale;
        println!("vs {:?} pad {:?} niter {}: max |QSM.rs - STI| / max |STI| = {d:e}", case.vs, case.pad, case.niter);
        assert!(d < case.tol, "vs {:?} niter {}: {d:e}", case.vs, case.niter);
    }
}

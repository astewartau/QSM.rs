//! `bgremove::vsharp` at its defaults against STI Suite 3.0's `V_SHARP`.
//!
//! The input is rebuilt here with the formulas of `tests/data/vsharp_sti_small.py`, which wrote it
//! for the verified `vsharp-sti` container (QSMxT/QSM-CI#304); the fixture holds only STI's
//! output: per case, the number of input-mask voxels (u32) and `V_SHARP`'s unmasked output there
//! (f64, column-major), NaN where its eroded mask is 0. STI filters in single precision, so the
//! two agree to float32 rounding (max |Δ| 1.4e-8 on a field of up to 0.15), with the same mask.

use qsm_core::bgremove::{vsharp, VsharpBall, VsharpParams};
use qsm_core::Grid;

const DIMS: [usize; 3] = [40, 44, 36];
/// `smvsize` of each case, in fixture order.
const SMVSIZE: [f64; 2] = [12.0, 6.0];

fn case() -> (Vec<f64>, Vec<u8>) {
    let [nx, ny, nz] = DIMS;
    let mut field = vec![0.0; nx * ny * nz];
    let mut mask = vec![0u8; field.len()];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let (x, y, z) = (i as f64, j as f64, k as f64);
                let l = i + j * nx + k * nx * ny;
                let inside = ((x - 19.5) / 15.0).powi(2) + ((y - 21.5) / 17.0).powi(2) + ((z - 17.5) / 13.0).powi(2) < 1.0;
                mask[l] = u8::from(inside && !(i > 28 && j < 14));
                field[l] = 0.05 * (0.21 * x + 0.3).sin() * (0.17 * y).cos() + 0.002 * (x - 20.0) * (z - 18.0) / 10.0
                    + 0.1 * (-((x - 15.0).powi(2) + (y - 22.0).powi(2) + (z - 18.0).powi(2)) / 8.0).exp();
            }
        }
    }
    (field, mask)
}

fn reference() -> Vec<Vec<f64>> {
    let bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/vsharp_sti_small.bin")).unwrap();
    let mut at = 0;
    let mut out = Vec::new();
    for _ in SMVSIZE {
        let n = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        at += 4;
        out.push((0..n).map(|v| f64::from_le_bytes(bytes[at + 8 * v..at + 8 * v + 8].try_into().unwrap())).collect());
        at += 8 * n;
    }
    assert_eq!(at, bytes.len());
    out
}

fn run(ball: VsharpBall, max_radius: f64) -> (Vec<f64>, Vec<u8>) {
    let (field, mask) = case();
    let grid = Grid::new(DIMS[0], DIMS[1], DIMS[2], 1.0, 1.0, 1.0);
    let p = VsharpParams { max_radius, ball, ..VsharpParams::default() };
    vsharp(&field, &mask, &grid, &p, |_, _| {}).unwrap()
}

/// (max |Δ|, eroded-mask voxels that differ) against STI on the input mask.
fn against_sti(ball: VsharpBall, case: usize, sti: &[f64]) -> (f64, usize) {
    let (_, mask) = self::case();
    let (local, out_mask) = run(ball, SMVSIZE[case]);
    let mut max = 0.0f64;
    let mut mask_diff = 0;
    for (v, l) in (0..mask.len()).filter(|&l| mask[l] != 0).enumerate() {
        let s = sti[v];
        if s.is_nan() != (out_mask[l] == 0) {
            mask_diff += 1;
        } else if !s.is_nan() {
            max = max.max((local[l] - s).abs());
        }
    }
    (max, mask_diff)
}

#[test]
fn vsharp_defaults_reproduce_sti_suites_v_sharp() {
    for (c, sti) in reference().iter().enumerate() {
        let (max, mask_diff) = against_sti(VsharpBall::default(), c, sti);
        assert_eq!(mask_diff, 0, "smvsize {}: eroded masks differ", SMVSIZE[c]);
        // STI filters in single precision: its output is float32-accurate (field up to 0.15).
        assert!(max < 5e-8, "smvsize {}: max |Δ| {max:e}", SMVSIZE[c]);
    }
}

#[test]
fn qsmjls_ball_is_not_stis() {
    // The QSM.jl ball (radius r, centre included) was the default before; it misses STI.
    let sti = &reference()[0];
    let (max, _) = against_sti(VsharpBall::QsmJl, 0, sti);
    assert!(max > 1e-3, "max |Δ| {max:e}");
}

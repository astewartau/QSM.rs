//! ROMEO multi-echo unwrapping (`unwrap::unwrap_romeo_multi_echo`) against ROMEO.jl's `unwrap`
//! (08776d4), and MCPC-3D-S (`utils::phase_offset_removal`) against MriResearchTools.jl's
//! `mcpc3ds` (771eadf), on a small noisy three-echo phantom in double precision.
//!
//! The inputs are rebuilt here exactly as `tests/data/romeo_romeojl_small.jl` built them (only
//! sums, products, quotients, round and integer xor, so bit for bit), so the fixture holds only
//! the references' output. The script ran in the QSM-CI `julia-qsm` image,
//! `sha256:b3c048acd8a88f23b7a04378f2b85dbac433bcc78b3d9cdf8ce58eb8a7478185` (Julia 1.10.11, the
//! Manifest committed in QSMxT/QSM-CI `algorithms/_julia/env`); the command is in the script's
//! header. `tests/data/romeo_romeojl_small.bin` (little-endian): for each unwrapping case in
//! [`CASES`] order and each echo, the number of turns ROMEO.jl added at each mask voxel
//! (column-major, i8); then `mcpc3ds`'s corrected phase at each mask voxel, per echo (f64).

use qsm_core::unwrap::{unwrap_romeo_multi_echo, RomeoParams, UnwrapMethod};
use qsm_core::utils::phase_offset_removal;
use qsm_core::Grid;
use std::f64::consts::PI;

const N: [usize; 3] = [28, 24, 18];
const TES: [f64; 3] = [4.0, 8.0, 12.0];

fn cases() -> [(&'static str, RomeoParams); 3] {
    [
        ("template, API defaults", RomeoParams::default()),
        ("template, the CLI's weights and temporal setting", RomeoParams::romeo_cli()),
        ("individual, global correction", RomeoParams { individual: true, correct_global: true, ..RomeoParams::default() }),
    ]
}

/// The generator's `phantom()`: wrapped phase and magnitude per echo, and the mask.
fn phantom() -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<u8>) {
    let n = N;
    let nv = n[0] * n[1] * n[2];
    let (a, b, c) = (0.46 * n[0] as f64, 0.44 * n[1] as f64, 0.42 * n[2] as f64);
    let mut phase = vec![vec![0.0; nv]; TES.len()];
    let mut mag = vec![vec![0.0; nv]; TES.len()];
    let mut mask = vec![0u8; nv];
    for (e, &te) in TES.iter().enumerate() {
        for k in 0..n[2] {
            for j in 0..n[1] {
                for i in 0..n[0] {
                    let l = i + j * n[0] + k * n[0] * n[1];
                    let x = i as f64 - (n[0] - 1) as f64 / 2.0 + 0.3;
                    let y = j as f64 - (n[1] - 1) as f64 / 2.0 + 0.3;
                    let z = k as f64 - (n[2] - 1) as f64 / 2.0 + 0.3;
                    let inside = x * x / (a * a) + y * y / (b * b) + z * z / (c * c) < 1.0;
                    mask[l] = u8::from(inside);
                    let omega = 0.08 * x + 0.05 * y - 0.03 * z + 0.004 * x * y + 0.002 * (x * x - z * z);
                    let phi0 = 0.5 + 0.02 * x - 0.03 * y;
                    let h = (i as u64 * 73856093) ^ (j as u64 * 19349663) ^ (k as u64 * 83492791) ^ (e as u64 * 2654435761);
                    let noise = ((h % 1000) as f64 / 1000.0 - 0.5) * if inside { 0.4 } else { 4.0 };
                    let raw = te * omega + phi0 + noise;
                    phase[e][l] = raw - 2.0 * PI * (raw / (2.0 * PI)).round_ties_even();
                    mag[e][l] = if inside {
                        (100.0 - 0.1 * (x * x + y * y + z * z)) / (1.0 + 0.1 * e as f64)
                    } else {
                        5.0 + (h % 7) as f64
                    };
                }
            }
        }
    }
    (phase, mag, mask)
}

#[test]
fn romeo_and_mcpc3ds_match_the_julia_packages() {
    let (phase, mag, mask) = phantom();
    let grid = Grid::new(N[0], N[1], N[2], 1.0, 1.0, 1.0);
    let inside: Vec<usize> = (0..mask.len()).filter(|&l| mask[l] != 0).collect();
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/romeo_romeojl_small.bin")).expect("fixture");
    let mut o = 0;
    let mut failed = Vec::new();
    for (name, params) in cases() {
        let uw = unwrap_romeo_multi_echo(&phase, &mag, &TES, &mask, &params, &grid);
        for e in 0..TES.len() {
            let want = &raw[o..o + inside.len()];
            o += inside.len();
            let differ = inside.iter().zip(want).filter(|(&l, &t)| ((uw[e][l] - phase[e][l]) / (2.0 * PI)).round() as i64 != t as i8 as i64).count();
            println!("{name}, echo {}: {differ} of {} voxels with a different number of turns", e + 1, inside.len());
            if differ != 0 {
                failed.push(format!("{name}, echo {}: {differ} voxels", e + 1));
            }
        }
    }
    let (corrected, _) = phase_offset_removal(&phase, &mag, &TES, [7.0; 3], [0, 1], UnwrapMethod::Romeo, &grid);
    for (e, c) in corrected.iter().enumerate() {
        let want: Vec<f64> = (0..inside.len()).map(|i| f64::from_le_bytes(raw[o + 8 * i..o + 8 * i + 8].try_into().unwrap())).collect();
        o += 8 * inside.len();
        let max_d = inside.iter().zip(&want).map(|(&l, w)| { let d = c[l] - w; (d - 2.0 * PI * (d / (2.0 * PI)).round()).abs() }).fold(0.0, f64::max);
        println!("mcpc3ds, echo {}: max |Δ| {max_d:.2e} rad (wrapped)", e + 1);
        if max_d.is_nan() || max_d >= 1e-9 {
            failed.push(format!("mcpc3ds echo {}: max |Δ| {max_d:e}", e + 1));
        }
    }
    assert_eq!(o, raw.len());
    assert!(failed.is_empty(), "{failed:#?}");
}

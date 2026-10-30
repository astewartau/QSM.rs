//! `bgremove::vsharp` at its defaults against STI Suite 3.0's `V_SHARP` on anisotropic voxels.
//!
//! `tests/data/vsharp_sti_aniso.bin` is made by `tests/data/vsharp_sti_aniso.py` with the verified
//! `vsharp-sti` container (QSMxT/QSM-CI#304); the phantoms are rebuilt here with its formulas, and
//! the layout is in the script. STI resamples these to a 1 mm working grid and back, and its crop
//! (`MaskBoundingBox`: the mask's bounding box made even) and its single-precision mask
//! interpolation decide which voxels reach that grid; astewartau/QSM.rs#329 was the crop. STI
//! filters in single precision, so the two agree to float32 rounding, with the same eroded mask.

use qsm_core::bgremove::{vsharp, VsharpParams};
use qsm_core::Grid;

/// The generator's CASES: dimensions and voxel size (mm).
const CASES: [([usize; 3], [f64; 3]); 4] = [
    ([32, 32, 22], [1.0, 1.0, 1.5]),
    ([30, 34, 22], [0.9, 0.9, 2.5]),
    ([28, 32, 24], [1.2, 0.8, 1.5]),
    ([40, 36, 18], [0.7, 0.7, 3.0]),
];

/// The generator's `phantom`: (field, mask).
fn phantom(n: [usize; 3], vs: [f64; 3]) -> (Vec<f64>, Vec<u8>) {
    let r = (n[0] as f64 * vs[0]).min(n[1] as f64 * vs[1]).min(n[2] as f64 * vs[2]) * 0.42;
    let mut field = vec![0.0; n[0] * n[1] * n[2]];
    let mut mask = vec![0u8; field.len()];
    for k in 0..n[2] {
        for j in 0..n[1] {
            for i in 0..n[0] {
                let x = (i as f64 - n[0] as f64 / 2.0 + 0.3) * vs[0];
                let y = (j as f64 - n[1] as f64 / 2.0 - 0.2) * vs[1];
                let z = (k as f64 - n[2] as f64 / 2.0 + 0.1) * vs[2];
                let l = i + n[0] * (j + n[1] * k);
                mask[l] = u8::from(x * x / ((r * 1.05) * (r * 1.05)) + y * y / (r * r) + z * z / ((r * 0.95) * (r * 0.95)) < 1.0);
                let d = (x - 3.0) * (x - 3.0) + (y + 2.0) * (y + 2.0) + z * z;
                field[l] = 0.05 * x - 0.03 * y + 0.02 * z + 0.001 * x * y - 0.0008 * z * z + 0.3 / (1.0 + d / 20.0);
            }
        }
    }
    (field, mask)
}

#[test]
fn vsharp_matches_sti_suite_on_anisotropic_voxels() {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/vsharp_sti_aniso.bin")).expect("fixture");
    let mut o = 0;
    for (n, vs) in CASES {
        let nv = n[0] * n[1] * n[2];
        let sti_mask: Vec<bool> = (0..nv).map(|l| raw[o + l / 8] >> (l % 8) & 1 == 1).collect();
        o += nv.div_ceil(8);
        let inside: Vec<usize> = (0..nv).filter(|&l| sti_mask[l]).collect();
        let sti: Vec<f64> = inside.iter().enumerate().map(|(q, _)| f64::from_le_bytes(raw[o + 8 * q..o + 8 * q + 8].try_into().unwrap())).collect();
        o += 8 * inside.len();

        let (field, mask) = phantom(n, vs);
        let grid = Grid::new(n[0], n[1], n[2], vs[0], vs[1], vs[2]);
        let (local, eroded) = vsharp(&field, &mask, &grid, &VsharpParams::default(), |_, _| {}).unwrap();
        let mask_diff = (0..nv).filter(|&l| (eroded[l] != 0) != sti_mask[l]).count();
        let ours: Vec<f64> = inside.iter().map(|&l| local[l]).collect();
        let max = ours.iter().zip(&sti).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let nrmse = (ours.iter().zip(&sti).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / sti.iter().map(|b| b * b).sum::<f64>()).sqrt();
        println!("{n:?} at {vs:?} mm: {} eroded voxels, {mask_diff} differ; NRMSE {nrmse:.2e}, max |Δ| {max:.2e}", inside.len());
        // Measured 2026-10-30: the same masks, NRMSE 7e-7 to 1.1e-6 (STI's float32). Before the
        // even crop and the single-precision mask interpolation: 6e-4 to 2 %, 22 mask voxels
        // apart at 1.2×0.8×1.5 mm.
        assert_eq!(mask_diff, 0, "{n:?} at {vs:?}: eroded masks");
        assert!(nrmse < 5e-6, "{n:?} at {vs:?}: NRMSE {nrmse:e}");
    }
    assert_eq!(o, raw.len(), "fixture length");
}

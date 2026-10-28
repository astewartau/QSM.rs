//! `bgremove::sharp` (the default, `SharpVariant::Sepia`) against SEPIA's SHARP: Tian Liu's
//! `SHARP.m` (kschan0214/sepia db939cc) with MEDI 2024.11.26's kernels, run through SEPIA's own
//! `BackgroundRemovalMacro` (which pads odd dimensions) in licensed MATLAB R2023b, on small
//! polynomial phantoms.
//!
//! The inputs are rebuilt here exactly as `tests/data/sharp_sepia_small.m` built them (the
//! phantom of `sharp_qsmjl_small.jl`, from + − × ÷ only, so bit for bit), so the fixture holds
//! only SEPIA's output; the command is in the script's header. `tests/data/sharp_sepia_small.bin`
//! (little-endian), per case in [`CASES`] order: the number of voxels in `SHARP.m`'s eroded mask
//! (u32), that mask as bits over the volume (column-major, least significant bit first), and
//! SEPIA's local field there (f64).

use qsm_core::bgremove::{sharp, SharpParams};
use qsm_core::Grid;

struct Case {
    name: &'static str,
    dims: [usize; 3],
    vs: [f64; 3],
    radius: f64,
    threshold: f64,
}

const CASES: [Case; 3] = [
    Case { name: "SEPIA defaults (4 mm, 0.03), 1 mm", dims: [40, 36, 32], vs: [1.0; 3], radius: 4.0, threshold: 0.03 },
    Case { name: "SEPIA defaults, anisotropic, odd dims", dims: [33, 29, 25], vs: [0.9, 1.0, 1.3], radius: 4.0, threshold: 0.03 },
    Case { name: "r 2.5 mm, thr 0.1, anisotropic", dims: [32, 28, 24], vs: [1.1, 0.8, 1.2], radius: 2.5, threshold: 0.1 },
];

/// The generators' `phantom(n, v)` (`sharp_sepia_small.m`, `sharp_qsmjl_small.jl`): field (zero outside the mask) and mask.
fn phantom(n: [usize; 3], v: [f64; 3]) -> (Vec<f64>, Vec<u8>) {
    let (a, b, c) = (0.46 * n[0] as f64 * v[0], 0.44 * n[1] as f64 * v[1], 0.42 * n[2] as f64 * v[2]);
    let mut f = vec![0.0; n[0] * n[1] * n[2]];
    let mut m = vec![0u8; f.len()];
    for k in 0..n[2] {
        for j in 0..n[1] {
            for i in 0..n[0] {
                let x = (i as f64 - (n[0] - 1) as f64 / 2.0 + 0.3) * v[0];
                let y = (j as f64 - (n[1] - 1) as f64 / 2.0 + 0.3) * v[1];
                let z = (k as f64 - (n[2] - 1) as f64 / 2.0 + 0.3) * v[2];
                if x * x / (a * a) + y * y / (b * b) + z * z / (c * c) < 1.0 {
                    let l = i + j * n[0] + k * n[0] * n[1];
                    m[l] = 1;
                    let mut val = 0.02 * z + 0.003 * (x * x - y * y) + 1.0e-4 * x * y * z;
                    let (p, q) = (x - 0.2 * a, 0.25 * a);
                    if p * p + y * y + z * z < q * q {
                        val += 0.05;
                    }
                    let (p, s, t, q) = (x + 0.3 * a, y - 0.2 * b, z + 0.1 * c, 0.2 * a);
                    if p * p + s * s + t * t < q * q {
                        val -= 0.03;
                    }
                    f[l] = val;
                }
            }
        }
    }
    (f, m)
}

/// SEPIA's eroded mask and local field per case.
fn load_sepia() -> Vec<(Vec<u8>, Vec<f64>)> {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/sharp_sepia_small.bin")).expect("fixture");
    let mut o = 0;
    let out = CASES.iter().map(|case| {
        let n = case.dims.iter().product::<usize>();
        let k = u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize;
        o += 4;
        let mask: Vec<u8> = (0..n).map(|l| (raw[o + l / 8] >> (l % 8)) & 1).collect();
        o += n.div_ceil(8);
        let vals = (0..k).map(|i| f64::from_le_bytes(raw[o + 8 * i..o + 8 * i + 8].try_into().unwrap())).collect();
        o += 8 * k;
        (mask, vals)
    }).collect();
    assert_eq!(o, raw.len());
    out
}

#[test]
fn sharp_matches_sepia_to_rounding() {
    assert_eq!((SharpParams::default().radius, SharpParams::default().threshold), (CASES[0].radius, CASES[0].threshold));
    for (case, (want_mask, want)) in CASES.iter().zip(load_sepia()) {
        let (field, mask) = phantom(case.dims, case.vs);
        let grid = Grid::new(case.dims[0], case.dims[1], case.dims[2], case.vs[0], case.vs[1], case.vs[2]);
        let p = SharpParams { radius: case.radius, threshold: case.threshold, ..SharpParams::default() };
        let (local, eroded) = sharp(&field, &mask, &grid, &p);
        assert!(!want.is_empty(), "{}: SEPIA's eroded mask is empty", case.name);
        assert_eq!(eroded, want_mask, "{}: eroded mask", case.name);
        assert!((0..local.len()).all(|l| eroded[l] != 0 || local[l] == 0.0), "{}: field outside the mask", case.name);
        let got: Vec<f64> = (0..local.len()).filter(|&l| eroded[l] != 0).map(|l| local[l]).collect();
        let max_d = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let rel = (got.iter().zip(&want).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / want.iter().map(|b| b * b).sum::<f64>()).sqrt();
        println!("{}: {} voxels, max|Δ| {:.2e} ppm, relative L2 {:.2e}", case.name, want.len(), max_d, rel);
        assert!(rel < 1e-12 && max_d < 1e-14, "{}: max|Δ| {max_d:e}, relative L2 {rel:e}", case.name);
    }
}

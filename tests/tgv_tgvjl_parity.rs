//! `inversion::tgv_qsm` against QuantitativeSusceptibilityMappingTGV.jl's `qsm_tgv` (korbinian90,
//! 6c0886a) on small polynomial phantoms whose masks touch the array's first and last z planes.
//!
//! The inputs are rebuilt here exactly as `tests/data/tgv_tgvjl_small.jl` built them (only
//! additions, subtractions, products and quotients, so bit for bit), so the fixture holds only the
//! package's output. The script ran
//! in the QSM-CI `julia-qsm` image,
//! `sha256:b3c048acd8a88f23b7a04378f2b85dbac433bcc78b3d9cdf8ce58eb8a7478185` (Julia 1.10.11, the
//! Manifest committed in QSMxT/QSM-CI `algorithms/_julia/env`, 4 threads); the command is in the
//! script's header. `tests/data/tgv_tgvjl_small.bin` (little-endian), per case in [`CASES`]
//! order: the number of voxels in the box the package returns (its eroded mask's bounding box
//! plus one voxel, u32), that box as bits over the volume (column-major, least significant bit
//! first), and the package's χ there (f64 of its Float32, ppm), including the rim outside the
//! eroded mask. Both run in single precision for hundreds of iterations, so they agree to
//! float32 rounding (relative L2 ~1e-6), not to f64's.

use qsm_core::inversion::{tgv_qsm, TgvInput, TgvParams};
use qsm_core::Grid;

const B0: f64 = 3.0;
const TE: f64 = 0.02;
const OBLIQUE: (f64, f64, f64) = (0.1, -0.2, 0.9746794344808963);

struct Case {
    name: &'static str,
    dims: [usize; 3],
    vs: [f64; 3],
    bdir: (f64, f64, f64),
    iterations: Option<usize>,
    /// `Some(scale)`: the phase `field · scale` rad as a wrapped echo (the package's default
    /// sin/cos Laplacian); `None`: the field in ppm (the discrete Laplacian).
    phase_scale: Option<f64>,
}

const CASES: [Case; 5] = [
    Case { name: "del, package defaults", dims: [32, 28, 24], vs: [1.0; 3], bdir: (0.0, 0.0, 1.0), iterations: None, phase_scale: Some(40.0) },
    Case {
        name: "field, anisotropic, oblique B0, 300 iterations",
        dims: [30, 26, 22],
        vs: [0.9, 1.0, 1.3],
        bdir: OBLIQUE,
        iterations: Some(300),
        phase_scale: None,
    },
    Case { name: "field, 300 iterations", dims: [30, 26, 22], vs: [1.0; 3], bdir: (0.0, 0.0, 1.0), iterations: Some(300), phase_scale: None },
    Case { name: "del, anisotropic, 300 iterations", dims: [30, 26, 22], vs: [0.9, 1.0, 1.3], bdir: (0.0, 0.0, 1.0), iterations: Some(300), phase_scale: Some(40.0) },
    Case { name: "del, oblique B0, 300 iterations", dims: [30, 26, 22], vs: [1.0; 3], bdir: OBLIQUE, iterations: Some(300), phase_scale: Some(40.0) },
];

/// The generator's `phantom(n, v)`: field in ppm (zero outside the mask) and mask.
fn phantom(n: [usize; 3], v: [f64; 3]) -> (Vec<f64>, Vec<u8>) {
    let (a, b, c) = (0.46 * n[0] as f64 * v[0], 0.44 * n[1] as f64 * v[1], 0.6 * n[2] as f64 * v[2]);
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
                    let mut val = 0.002 * z + 0.0003 * (x * x - y * y) + 1.0e-5 * x * y * z;
                    let (p, q) = (x - 0.2 * a, 0.25 * a);
                    if p * p + y * y + z * z < q * q {
                        val += 0.005;
                    }
                    let (p, s, t, q) = (x + 0.3 * a, y - 0.2 * b, z + 0.1 * c, 0.2 * a);
                    if p * p + s * s + t * t < q * q {
                        val -= 0.003;
                    }
                    f[l] = val;
                }
            }
        }
    }
    (f, m)
}

fn load_tgvjl() -> Vec<(Vec<u8>, Vec<f64>)> {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/tgv_tgvjl_small.bin")).expect("fixture");
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
fn tgv_matches_the_package_to_float32_rounding() {
    for (case, (want_mask, want)) in CASES.iter().zip(load_tgvjl()) {
        let (field, mask) = phantom(case.dims, case.vs);
        let grid = Grid::new(case.dims[0], case.dims[1], case.dims[2], case.vs[0], case.vs[1], case.vs[2]);
        // The package's own erosion, w-update divergence and output box
        // (`TgvParams::reference_edge_erosion`, `reference_divergence`, `reference_output_box`).
        let params = TgvParams {
            iterations: case.iterations,
            reference_divergence: true,
            reference_edge_erosion: true,
            reference_output_box: true,
            ..Default::default()
        };
        let chi = match case.phase_scale {
            Some(scale) => {
                let phase: Vec<f64> = field.iter().map(|v| v * scale).collect();
                let rpp = qsm_core::inversion::tgv::rad_per_ppm(TE, B0);
                let chi = tgv_qsm(TgvInput::WrappedPhase { phase: &phase, te_s: TE, b0_t: B0 }, &mask, &grid, &params, case.bdir, |_, _| {})
                    .unwrap();
                // As the generator does: χ as if the phase had been field · rad_per_ppm.
                chi.iter().map(|v| v * rpp / scale).collect::<Vec<f64>>()
            }
            None => tgv_qsm(TgvInput::FieldPpm { field: &field, echo_times: &[TE], b0_t: B0 }, &mask, &grid, &params, case.bdir, |_, _| {})
                .unwrap(),
        };
        // With `reference_output_box`, QSM.rs returns the package's whole box, rim included.
        assert!((0..chi.len()).all(|l| want_mask[l] != 0 || chi[l] == 0.0), "{}: χ outside the package's box", case.name);
        let got: Vec<f64> = (0..chi.len()).filter(|&l| want_mask[l] != 0).map(|l| chi[l]).collect();
        let max_d = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let rel = (got.iter().zip(&want).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / want.iter().map(|b| b * b).sum::<f64>()).sqrt();
        println!("{}: {} voxels, max|Δ| {max_d:.2e} ppm, relative L2 {rel:.2e}", case.name, want.len());
        assert!(rel < 1e-5, "{}: max|Δ| {max_d:e}, relative L2 {rel:e}", case.name);
    }
}

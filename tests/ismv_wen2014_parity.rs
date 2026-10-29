//! `bgremove::ismv` (the default, `IsmvVariant::Wen2014`) against the iSMV authors' lab code:
//! `iSMV.m` with its kernel `tdcsmv.m` (github.com/agr78/mSMV 42a1f18,
//! `code/dependencies/MEDI_functions/`), on small polynomial phantoms.
//!
//! `tests/data/ismv_wen2014_small.m` ran them in GNU Octave 7.3 (Debian bookworm); the command is
//! in its header. The cases: Wen et al. 2014's defaults (6 voxels, relative change < 5e-5, with
//! `iSMV.m`'s 0.001 replaced by a `tol` argument) on 1 mm and on 0.9 × 0.9 × 1.3 mm voxels,
//! `iSMV.m` unmodified (0.001) at 4 mm, and an odd-sized array zero-padded by one slice per odd
//! axis as SEPIA does (`iSMV.m` errors on odd sizes; QSM.rs pads the same way).
//!
//! The inputs are rebuilt here exactly as the generator built them (from + − × ÷ only, so bit
//! for bit), so the fixture holds only the lab code's output. `tests/data/ismv_wen2014_small.bin`
//! (little-endian), per case in [`cases`] order: the number of voxels in the eroded mask `M`
//! (u32), `M` as bits over the volume (column-major, least significant bit first), and
//! `real(L)` at those voxels (f64).

use qsm_core::bgremove::{ismv, IsmvParams, IsmvVariant};
use qsm_core::Grid;

struct Case {
    name: &'static str,
    dims: [usize; 3],
    vs: [f64; 3],
    params: IsmvParams,
}

fn cases() -> [Case; 4] {
    [
        Case { name: "Wen 2014 defaults (6 voxels, 5e-5), 1 mm", dims: [40, 36, 32], vs: [1.0; 3], params: IsmvParams::default() },
        Case { name: "Wen 2014 defaults, anisotropic 0.9 x 0.9 x 1.3 mm", dims: [36, 32, 26], vs: [0.9, 0.9, 1.3], params: IsmvParams::default() },
        Case {
            name: "iSMV.m as shipped (tol 0.001), r 4 mm",
            dims: [36, 32, 28],
            vs: [1.0; 3],
            params: IsmvParams { tol: 1e-3, radius: Some(4.0), ..IsmvParams::default() },
        },
        Case { name: "Wen 2014 defaults, odd dims (padded)", dims: [33, 29, 25], vs: [1.0; 3], params: IsmvParams::default() },
    ]
}

/// The generator's `phantom(n, v)` (`ismv_wen2014_small.m`, after `sharp_qsmjl_small.jl`): field (zero outside the mask) and mask.
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

/// `iSMV.m`'s eroded mask and real(L) (at the eroded voxels) per case.
fn load_lab(cases: &[Case]) -> Vec<(Vec<u8>, Vec<f64>)> {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/ismv_wen2014_small.bin")).expect("fixture");
    let mut o = 0;
    let out = cases.iter().map(|case| {
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
fn ismv_matches_the_lab_code() {
    let cases = cases();
    for (case, (want_mask, want)) in cases.iter().zip(load_lab(&cases)) {
        assert_eq!(case.params.variant, IsmvVariant::Wen2014);
        let (field, mask) = phantom(case.dims, case.vs);
        let grid = Grid::new(case.dims[0], case.dims[1], case.dims[2], case.vs[0], case.vs[1], case.vs[2]);
        let (local, eroded) = ismv(&field, &mask, &grid, &case.params, |_, _| {}).expect("ismv");
        assert_eq!(eroded, want_mask, "{}: eroded mask", case.name);
        assert!((0..local.len()).all(|l| eroded[l] != 0 || local[l] == 0.0), "{}: zero outside M", case.name);
        let got: Vec<f64> = (0..local.len()).filter(|&l| eroded[l] != 0).map(|l| local[l]).collect();
        assert_eq!(got.len(), want.len(), "{}: eroded voxels", case.name);
        let max_d = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let rel = (got.iter().zip(&want).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / want.iter().map(|b| b * b).sum::<f64>()).sqrt();
        println!("{}: {} voxels, max|Δ| {:.2e} ppm, relative L2 {:.2e}", case.name, want.len(), max_d, rel);
        assert!(rel < 1e-12 && max_d < 1e-14, "{}: max|Δ| {max_d:e}, relative L2 {rel:e}", case.name);
    }
}

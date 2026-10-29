//! QSM.rs's ports of QSM.jl's FFT dipole inversions (`tkd`, `tsvd`, `tikhonov`, `tv_admm`,
//! `rts`) against QSM.jl itself (kamesy/QSM.jl fde4b92) on small polynomial phantoms, at QSM.jl's
//! defaults (including its fast-FFT padding) and a few other settings.
//!
//! The inputs are rebuilt here exactly as `tests/data/inversion_qsmjl_small.jl` built them (an
//! ellipsoid mask holding a smooth field and two spheres, from + − × ÷ only, so bit for bit), so
//! the fixture holds only QSM.jl's output. The script ran in the QSM-CI `julia-qsm` image,
//! `sha256:b3c048acd8a88f23b7a04378f2b85dbac433bcc78b3d9cdf8ce58eb8a7478185` (Julia 1.10.11, the
//! Manifest committed in QSMxT/QSM-CI `algorithms/_julia/env`); the command is in the script's
//! header, and the script also says why the oblique cases use odd sizes, the iterative ones
//! 1 mm voxels, and why `nltv` is not here. `tests/data/inversion_qsmjl_small.bin`
//! (little-endian), per case in [`CASES`] order: the number of mask voxels (u32) and QSM.jl's χ
//! there (f64, ppm, column-major).

use qsm_core::inversion::{
    rts, tikhonov, tkd, tsvd, tv_admm, Regularization, RtsParams, TikhonovParams, TkdParams, TvParams,
};
use qsm_core::Grid;

const N: [usize; 3] = [24, 22, 20];
const N_ODD: [usize; 3] = [23, 21, 19];
const ANISO: [f64; 3] = [0.9, 1.0, 1.3];
const ISO: [f64; 3] = [1.0, 1.0, 1.0];
const AXIAL: (f64, f64, f64) = (0.0, 0.0, 1.0);
const OBLIQUE: (f64, f64, f64) = (0.1, -0.2, 0.9746794344808963);
const NOPAD: [i64; 3] = [-1; 3];

enum Method {
    Tkd(TkdParams),
    Tsvd(TkdParams),
    Tikh(TikhonovParams),
    Tv(TvParams),
    Rts(RtsParams),
}

struct Case {
    name: &'static str,
    dims: [usize; 3],
    vs: [f64; 3],
    bdir: (f64, f64, f64),
    method: Method,
}

const CASES: [Case; 8] = [
    Case { name: "tkd, QSM.jl defaults", dims: N, vs: ANISO, bdir: AXIAL, method: Method::Tkd(TkdParams { threshold: 0.15, pad: [0; 3] }) },
    Case {
        name: "tkd, oblique B0, odd sizes, no padding",
        dims: N_ODD,
        vs: ANISO,
        bdir: OBLIQUE,
        method: Method::Tkd(TkdParams { threshold: 0.15, pad: NOPAD }),
    },
    Case { name: "tsvd, QSM.jl defaults", dims: N, vs: ANISO, bdir: AXIAL, method: Method::Tsvd(TkdParams { threshold: 0.15, pad: [0; 3] }) },
    Case {
        name: "tikh, QSM.jl defaults",
        dims: N,
        vs: ANISO,
        bdir: AXIAL,
        method: Method::Tikh(TikhonovParams { lambda: 1e-2, reg: Regularization::Identity, pad: [0; 3] }),
    },
    Case {
        name: "tikh, gradient, lambda 1e-3, pad 4",
        dims: N,
        vs: ANISO,
        bdir: AXIAL,
        method: Method::Tikh(TikhonovParams { lambda: 1e-3, reg: Regularization::Gradient, pad: [4; 3] }),
    },
    Case {
        name: "tv, QSM.jl defaults",
        dims: N,
        vs: ISO,
        bdir: AXIAL,
        method: Method::Tv(TvParams { lambda: 1e-3, rho: None, tol: 1e-3, max_iter: 250, pad: [0; 3] }),
    },
    Case { name: "rts, QSM.jl defaults", dims: N, vs: ISO, bdir: AXIAL, method: Method::Rts(QSMJL_RTS) },
    Case {
        name: "rts, oblique B0, odd sizes, no padding",
        dims: N_ODD,
        vs: ISO,
        bdir: OBLIQUE,
        method: Method::Rts(RtsParams { pad: NOPAD, ..QSMJL_RTS }),
    },
];

/// QSM.jl `rts`'s defaults, spelled out so the test also pins `RtsParams::default()` below.
const QSMJL_RTS: RtsParams =
    RtsParams { delta: 0.15, mu: 1e5, rho: 10.0, tol: 1e-2, max_iter: 20, lsmr_iter: 4, tau: 0.7, gamma: 5.0, pad: [0; 3] };

/// The generator's `phantom(n, v)`: field (zero outside the mask) and mask.
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

fn load_qsmjl() -> Vec<Vec<f64>> {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/inversion_qsmjl_small.bin")).expect("fixture");
    let mut o = 0;
    let out = CASES.iter().map(|_| {
        let k = u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize;
        o += 4;
        let v = (0..k).map(|i| f64::from_le_bytes(raw[o + 8 * i..o + 8 * i + 8].try_into().unwrap())).collect();
        o += 8 * k;
        v
    }).collect();
    assert_eq!(o, raw.len());
    out
}

#[test]
fn defaults_are_qsm_jls() {
    let t = TkdParams::default();
    assert_eq!((t.threshold, t.pad), (0.15, [0; 3]));
    let t = TikhonovParams::default();
    assert!(matches!(t.reg, Regularization::Identity));
    assert_eq!((t.lambda, t.pad), (1e-2, [0; 3]));
    let t = TvParams::default();
    assert_eq!((t.lambda, t.rho, t.tol, t.max_iter, t.pad), (1e-3, None, 1e-3, 250, [0; 3]));
    let (r, q) = (RtsParams::default(), QSMJL_RTS);
    assert_eq!(
        (r.delta, r.mu, r.rho, r.tol, r.max_iter, r.lsmr_iter, r.tau, r.gamma, r.pad),
        (q.delta, q.mu, q.rho, q.tol, q.max_iter, q.lsmr_iter, q.tau, q.gamma, q.pad)
    );
}

#[test]
fn qsmjl_inversions_match_qsm_jl_to_rounding() {
    let mut failed = Vec::new();
    for (case, want) in CASES.iter().zip(load_qsmjl()) {
        let (f, m) = phantom(case.dims, case.vs);
        let g = Grid::new(case.dims[0], case.dims[1], case.dims[2], case.vs[0], case.vs[1], case.vs[2]);
        let (chi, tol) = match &case.method {
            Method::Tkd(p) => (tkd(&f, &m, &g, case.bdir, p), 1e-12),
            Method::Tsvd(p) => (tsvd(&f, &m, &g, case.bdir, p), 1e-12),
            Method::Tikh(p) => (tikhonov(&f, &m, &g, case.bdir, p), 1e-12),
            Method::Tv(p) => (tv_admm(&f, &m, &g, case.bdir, p, |_, _| {}), 1e-10),
            Method::Rts(p) => (rts(&f, &m, &g, case.bdir, p, |_, _| {}), 1e-10),
        };
        let got: Vec<f64> = (0..chi.len()).filter(|&l| m[l] != 0).map(|l| chi[l]).collect();
        assert_eq!(got.len(), want.len(), "{}: mask voxels", case.name);
        assert!((0..chi.len()).all(|l| m[l] != 0 || chi[l] == 0.0), "{}: χ outside the mask", case.name);
        let max_d = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let rel = (got.iter().zip(&want).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / want.iter().map(|b| b * b).sum::<f64>()).sqrt();
        println!("{}: max|Δ| {max_d:.2e} ppm, relative L2 {rel:.2e}", case.name);
        if rel.is_nan() || rel >= tol {
            failed.push(format!("{}: max|Δ| {max_d:e}, relative L2 {rel:e}", case.name));
        }
    }
    assert!(failed.is_empty(), "{failed:#?}");
}

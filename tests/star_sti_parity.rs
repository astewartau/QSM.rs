//! `inversion::star_qsm` against STI Suite 3.0's `QSM_star` on 48x56x40 phantoms.
//!
//! The inputs are rebuilt here exactly as the MATLAB script that made the references built them
//! (spheres of four susceptibilities in an ellipsoid mask, forward-modelled with STI's centred
//! dipole kernel, converted to phase with γ = 42.576 MHz/T), so the fixture only holds STI's
//! output. `tests/data/star_sti_small.bin` (little-endian): for each case in [`CASES`] order, the
//! number of mask voxels (u32) and STI's χ there (f64, ppm, column-major), from MATLAB R2023b
//! `chi = QSM_star(TP, mask, 'TE', TE, 'B0', B0, 'H', H, 'padsize', pad, 'voxelsize', vs)`.
//! The last case uses `StarParams`' default padding, `[8 8 20]` as STI Suite's `QSM_GUI` passes it.
//! STI's χ is zero outside the mask.

use num_complex::Complex64;
use qsm_core::fft::{fft3d, ifft3d};
use qsm_core::inversion::{star_qsm, StarParams, STAR_GUI_PAD};
use qsm_core::Grid;

const DIMS: [usize; 3] = [48, 56, 40];

struct Case {
    name: &'static str,
    b0: f64,
    te_ms: f64,
    h: [f64; 3],
    pad: [usize; 3],
    vs: [f64; 3],
    /// SpaRSA iterations STI's run took
    iterations: usize,
}

const CASES: [Case; 4] = [
    Case { name: "axial 3 T", b0: 3.0, te_ms: 20.0, h: [0.0, 0.0, 1.0], pad: [12; 3], vs: [1.0, 1.0, 1.0], iterations: 12 },
    Case { name: "anisotropic voxels", b0: 3.0, te_ms: 20.0, h: [0.0, 0.0, 1.0], pad: [12; 3], vs: [0.8, 0.9, 1.5], iterations: 5 },
    Case {
        name: "oblique 7 T, pad 8",
        b0: 7.0,
        te_ms: 5.0,
        h: [0.10101525445522108, -0.4040610178208843, 0.9091372900969897],
        pad: [8; 3],
        vs: [1.0, 1.0, 1.0],
        iterations: 7,
    },
    Case {
        name: "anisotropic voxels, GUI padding",
        b0: 3.0,
        te_ms: 20.0,
        h: [0.0, 0.0, 1.0],
        pad: STAR_GUI_PAD,
        vs: [0.8, 0.9, 1.5],
        iterations: 5,
    },
];

/// The reference script's phantom (`mk_phantom.m`): χ and mask.
fn phantom(vs: [f64; 3]) -> (Vec<f64>, Vec<u8>) {
    let n = DIMS;
    let r = 0.4 * (0..3).map(|a| n[a] as f64 * vs[a]).fold(f64::INFINITY, f64::min);
    let c = [[0.3, 0.2, 0.0], [-0.3, 0.1, 0.1], [0.0, -0.35, -0.2], [0.1, 0.1, 0.3]].map(|p| p.map(|v| v * r));
    let val = [0.1, -0.05, 0.2, 0.5];
    let rad = [0.15, 0.2, 0.12, 0.08].map(|v| v * r);
    let mut chi = vec![0.0; n[0] * n[1] * n[2]];
    let mut mask = vec![0u8; chi.len()];
    for k in 0..n[2] {
        for j in 0..n[1] {
            for i in 0..n[0] {
                let p = [i, j, k];
                let x: [f64; 3] = [0, 1, 2].map(|a| (((p[a] + 1) as f64 - n[a] as f64 / 2.0) - 0.5) * vs[a]);
                let l = i + j * n[0] + k * n[0] * n[1];
                if (x[0] / r).powi(2) + (x[1] / (0.9 * r)).powi(2) + (x[2] / (0.8 * r)).powi(2) < 1.0 {
                    mask[l] = 1;
                    for s in 0..4 {
                        if (x[0] - c[s][0]).powi(2) + (x[1] - c[s][1]).powi(2) + (x[2] - c[s][2]).powi(2) < rad[s].powi(2) {
                            chi[l] = val[s];
                        }
                    }
                }
            }
        }
    }
    (chi, mask)
}

/// STI's dipole kernel (`calcD2Matrix`), unshifted: `k = f/(n·vs)`, `D = 1/3 − (k·H)²/|k|²`.
fn sti_kernel(vs: [f64; 3], h: [f64; 3]) -> Vec<f64> {
    let n = DIMS;
    let k = |a: usize, i: usize| (if i < n[a] / 2 { i as f64 } else { i as f64 - n[a] as f64 }) / (n[a] as f64 * vs[a]);
    let mut d = vec![0.0; n[0] * n[1] * n[2]];
    for kk in 0..n[2] {
        for j in 0..n[1] {
            for i in 0..n[0] {
                let (x, y, z) = (k(0, i), k(1, j), k(2, kk));
                let k2 = x * x + y * y + z * z;
                if k2 != 0.0 {
                    let kh = x * h[0] + y * h[1] + z * h[2];
                    d[i + j * n[0] + kk * n[0] * n[1]] = 1.0 / 3.0 - kh * kh / k2;
                }
            }
        }
    }
    d
}

/// The reference script's tissue phase: `TP = (real(F⁻¹ D F χ)·mask)·1e-6·2π·42.576e6·B0·TE`.
fn tissue_phase(chi: &[f64], mask: &[u8], case: &Case) -> Vec<f64> {
    let d = sti_kernel(case.vs, case.h);
    let mut buf: Vec<Complex64> = chi.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    fft3d(&mut buf, DIMS[0], DIMS[1], DIMS[2]);
    buf.iter_mut().zip(&d).for_each(|(z, &dv)| *z *= dv);
    ifft3d(&mut buf, DIMS[0], DIMS[1], DIMS[2]);
    buf.iter().zip(mask)
        .map(|(z, &m)| z.re * m as f64 * 1e-6 * 2.0 * std::f64::consts::PI * 42.576e6 * case.b0 * case.te_ms / 1000.0)
        .collect()
}

fn load_sti() -> Vec<Vec<f64>> {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/star_sti_small.bin")).expect("fixture");
    let mut o = 0;
    let out: Vec<Vec<f64>> = CASES.iter().map(|_| {
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
fn star_matches_sti_suite_to_rounding() {
    let sti = load_sti();
    for (case, want) in CASES.iter().zip(&sti) {
        let (chi0, mask) = phantom(case.vs);
        let inside: Vec<usize> = (0..mask.len()).filter(|&l| mask[l] != 0).collect();
        assert_eq!(inside.len(), want.len(), "{}: phantom mask", case.name);
        let phase = tissue_phase(&chi0, &mask, case);

        let params = StarParams { pad: case.pad, te: case.te_ms * 1e-3, b0: case.b0 };
        let field: Vec<f64> = phase.iter().map(|&v| v / params.rad_per_ppm()).collect();
        let grid = Grid::new(DIMS[0], DIMS[1], DIMS[2], case.vs[0], case.vs[1], case.vs[2]);
        let mut iterations = 0;
        let chi = star_qsm(&field, &mask, &grid, (case.h[0], case.h[1], case.h[2]), &params, |i, _| iterations = i);

        assert!((0..chi.len()).all(|l| mask[l] != 0 || chi[l] == 0.0), "{}: χ outside the mask", case.name);
        let got: Vec<f64> = inside.iter().map(|&l| chi[l]).collect();
        let max_d = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let rel = (got.iter().zip(want).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / want.iter().map(|b| b * b).sum::<f64>()).sqrt();
        println!("{}: {} iterations, max|Δ| {:.2e} ppm, relative L2 {:.2e}", case.name, iterations, max_d, rel);
        assert_eq!(iterations, case.iterations, "{}: iterations", case.name);
        assert!(rel < 1e-12 && max_d < 1e-13, "{}: max|Δ| {max_d:e}, relative L2 {rel:e}", case.name);
    }
}

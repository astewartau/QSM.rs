//! `separation::chi_sep_ilsqr` against the SNU χ-separation toolbox's `chi_sep_iLSQR` (v1.1.3).
//!
//! `tests/data/chi_sep_ilsqr_snu_small.bin` holds only the toolbox's output, made by
//! `tests/data/chi_sep_ilsqr_snu_small.m` (its header names the toolbox files, their hashes and
//! the MATLAB release). The inputs are rebuilt here exactly as that script builds them (the
//! phantom of `chi_sep_medi_snu_small.m`): an ellipsoid with χ+ and χ− blocks, integer-LCG noise,
//! and a field from one FFT convolution (QSM.rs's FFT, equal to MATLAB's to rounding). Layout
//! (little-endian), per case in [`cases`] order: the number of mask voxels `m` (u32), the NNLS
//! iterations the toolbox ran (u32), then its χ+ (`x_para`) and |χ−| (`x_dia`) at the mask voxels
//! in column-major order (`2m` f64, ppm).

use qsm_core::fft::{fft3d, ifft3d};
use qsm_core::separation::{chi_sep_ilsqr, ChiSepIlsqrInputs, ChiSepIlsqrParams};
use qsm_core::Grid;
use num_complex::Complex64;

struct Case {
    name: &'static str,
    n: [usize; 3],
    vs: [f64; 3],
    b0: f64,
    h: [f64; 3],
    negative_r2p: bool,
    dr: Option<(f64, f64)>,
}

fn oblique() -> [f64; 3] {
    let v = [0.1f64, -0.4, 0.9];
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v.map(|c| c / norm)
}

fn cases() -> [Case; 3] {
    [
        Case { name: "3 T, 1 mm", n: [40, 36, 28], vs: [1.0; 3], b0: 3.0, h: [0.0, 0.0, 1.0], negative_r2p: false, dr: None },
        Case { name: "7 T, odd grid, 0.8x0.9x1.5 mm, oblique", n: [33, 29, 21], vs: [0.8, 0.9, 1.5], b0: 7.0, h: oblique(), negative_r2p: false, dr: None },
        Case { name: "negative R2', Dr 120/150", n: [40, 36, 28], vs: [1.0; 3], b0: 3.0, h: [0.0, 0.0, 1.0], negative_r2p: true, dr: Some((120.0, 150.0)) },
    ]
}

struct Phantom {
    mask: Vec<u8>,
    f_hz: Vec<f64>,
    r2p: Vec<f64>,
    qsm: Vec<f64>,
}

/// The script's `phantom(n, vs, B0, h, negative_r2p)` (its magnitude and N_std are not needed).
fn phantom(c: &Case) -> Phantom {
    let n = c.n;
    let nt = n[0] * n[1] * n[2];
    let u = |seed: usize, l: usize| ((1103515245.0 * ((l + seed * 7919) as f64) + 12345.0) % 2f64.powi(31)) / 2f64.powi(31);
    let r = [0, 1, 2].map(|a| 0.4 * n[a] as f64 * c.vs[a]);
    let mut mask = vec![0u8; nt];
    let mut csf = vec![0u8; nt];
    let mut cp = vec![0.0; nt];
    let mut cn = vec![0.0; nt];
    let mut xs = vec![[0.0; 3]; nt];
    for k in 0..n[2] {
        for j in 0..n[1] {
            for i in 0..n[0] {
                let l = i + n[0] * (j + n[1] * k);
                let p = [i, j, k];
                let x: [f64; 3] = [0, 1, 2].map(|a| ((p[a] + 1) as f64 - (n[a] as f64 + 1.0) / 2.0) * c.vs[a]);
                xs[l] = x;
                let inside = (x[0] / r[0]).powi(2) + (x[1] / r[1]).powi(2) + (x[2] / r[2]).powi(2) <= 1.0;
                let in_csf = inside
                    && (x[0] / (0.25 * r[0])).powi(2) + ((x[1] - 0.2 * r[1]) / (0.15 * r[1])).powi(2) + (x[2] / (0.3 * r[2])).powi(2) <= 1.0;
                mask[l] = inside as u8;
                csf[l] = in_csf as u8;
                let blk = |cx: f64, cy: f64, cz: f64, s: f64| {
                    ((x[0] - cx * r[0]).abs() <= s * r[0] && (x[1] - cy * r[1]).abs() <= s * r[1] && (x[2] - cz * r[2]).abs() <= s * r[2]) as u8 as f64
                };
                let keep = (inside as u8 as f64) * (!in_csf as u8 as f64);
                cp[l] = (0.08 * blk(0.4, -0.3, 0.1, 0.18) + 0.05 * blk(-0.35, -0.35, -0.2, 0.15) + 0.02 * ((x[1] < -0.5 * r[1]) as u8 as f64)) * keep;
                cn[l] = (0.04 * blk(-0.3, 0.45, 0.25, 0.2) + 0.03 * blk(0.35, 0.3, -0.3, 0.16) + 0.015 * ((x[0].abs() < 0.3 * r[0]) as u8 as f64)) * keep;
            }
        }
    }
    let cf = 42.58e6 * c.b0;
    // real(ifftn(D.*fftn(cp - cn))), D on k = f/(n*vs), unshifted, D(0) = 0
    let f = |a: usize, i: usize| {
        let ni = n[a] as i64;
        let fi = if (i as i64) < (ni + 1) / 2 { i as i64 } else { i as i64 - ni };
        fi as f64 / (n[a] as f64 * c.vs[a])
    };
    let mut buf: Vec<Complex64> = cp.iter().zip(&cn).map(|(a, b)| Complex64::new(a - b, 0.0)).collect();
    fft3d(&mut buf, n[0], n[1], n[2]);
    for k in 0..n[2] {
        for j in 0..n[1] {
            for i in 0..n[0] {
                let (kx, ky, kz) = (f(0, i), f(1, j), f(2, k));
                let k2 = kx * kx + ky * ky + kz * kz;
                let s = kx * c.h[0] + ky * c.h[1] + kz * c.h[2];
                let d = if k2 == 0.0 { 0.0 } else { 1.0 / 3.0 - (s * s) / k2 };
                buf[i + n[0] * (j + n[1] * k)] *= d;
            }
        }
    }
    ifft3d(&mut buf, n[0], n[1], n[2]);
    let dr = 137.0 / 2.9 * c.b0;
    let m = |l: usize| mask[l] as f64;
    let mut ph = Phantom { mask: mask.clone(), f_hz: vec![0.0; nt], r2p: vec![0.0; nt], qsm: vec![0.0; nt] };
    for l in 0..nt {
        ph.f_hz[l] = (buf[l].re * (cf * 1e-6) + 0.4 * (u(1, l) - 0.5)) * m(l);
        ph.r2p[l] = (dr * (cp[l] + cn[l]) + 3.0 + 2.0 * (u(2, l) - 0.5)) * m(l);
        if c.negative_r2p {
            ph.r2p[l] -= 4.0 * m(l) * ((xs[l][0] > 0.0) as u8 as f64);
        }
        ph.qsm[l] = (cp[l] - cn[l] + 0.004 * (u(3, l) - 0.5)) * m(l);
    }
    ph
}

fn load() -> Vec<(usize, Vec<f64>, Vec<f64>)> {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/chi_sep_ilsqr_snu_small.bin")).expect("fixture");
    let mut o = 0;
    let u32_at = |o: &mut usize| {
        let v = u32::from_le_bytes(raw[*o..*o + 4].try_into().unwrap()) as usize;
        *o += 4;
        v
    };
    let mut out = Vec::new();
    for _ in 0..3 {
        let m = u32_at(&mut o);
        let it = u32_at(&mut o);
        let f = |o: usize, k: usize| (0..k).map(|i| f64::from_le_bytes(raw[o + 8 * i..o + 8 * i + 8].try_into().unwrap())).collect::<Vec<_>>();
        let p = f(o, m);
        let q = f(o + 8 * m, m);
        o += 16 * m;
        out.push((it, p, q));
    }
    assert_eq!(o, raw.len());
    out
}

#[test]
fn chi_sep_ilsqr_matches_the_snu_toolbox_to_rounding() {
    let want = load();
    for (c, (iters, want_p, want_n)) in cases().iter().zip(&want) {
        let ph = phantom(c);
        let field_ppm: Vec<f64> = ph.f_hz.iter().map(|v| v / (42.58e6 * c.b0) * 1e6).collect();
        let inputs = ChiSepIlsqrInputs { local_field: &field_ppm, r2prime: &ph.r2p, qsm: &ph.qsm, mask: &ph.mask };
        let params = ChiSepIlsqrParams {
            cf: 42.58e6 * c.b0,
            dr_pos: c.dr.map(|d| d.0),
            dr_neg: c.dr.map(|d| d.1),
            ..ChiSepIlsqrParams::default()
        };
        let grid = Grid::new(c.n[0], c.n[1], c.n[2], c.vs[0], c.vs[1], c.vs[2]);
        let mut got_iters = 0;
        let (pos, neg, tot) = chi_sep_ilsqr(&inputs, &grid, (c.h[0], c.h[1], c.h[2]), &params, |i, _| got_iters = i).unwrap();
        let inside: Vec<usize> = (0..ph.mask.len()).filter(|&l| ph.mask[l] != 0).collect();
        assert_eq!(inside.len(), want_p.len(), "{}: mask", c.name);
        assert!((0..pos.len()).all(|l| ph.mask[l] != 0 || (pos[l] == 0.0 && neg[l] == 0.0)), "{}: χ outside the mask", c.name);
        assert!((0..pos.len()).all(|l| (tot[l] - (pos[l] + neg[l])).abs() == 0.0), "{}: χ_total", c.name);
        let stats = |got: &[f64], want: &[f64]| {
            let max_d = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            let rel = (got.iter().zip(want).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / want.iter().map(|b| b * b).sum::<f64>()).sqrt();
            (max_d, rel)
        };
        let gp: Vec<f64> = inside.iter().map(|&l| pos[l]).collect();
        let gn: Vec<f64> = inside.iter().map(|&l| -neg[l]).collect();
        let (dp, rp) = stats(&gp, want_p);
        let (dn, rn) = stats(&gn, want_n);
        println!("{}: {} NNLS iterations; χ+ max|Δ| {:.2e} ppm, rel L2 {:.2e}; |χ−| max|Δ| {:.2e}, rel L2 {:.2e}", c.name, got_iters, dp, rp, dn, rn);
        assert_eq!(got_iters, *iters, "{}: NNLS iterations", c.name);
        // Measured: 6.2e-12 / 4.0e-10 (case 1), 3.1e-14 / 2.3e-14 (case 2), 2.2e-11 / 1.3e-10 (case 3).
        // The toolbox's own result moves by up to 1.5e-9 between 8 and 1 threads on these phantoms.
        assert!(rp < 1e-8 && rn < 1e-8, "{}: rel L2 χ+ {rp:e}, |χ−| {rn:e}", c.name);
    }
}

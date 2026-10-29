//! `separation::decompose` against the authors' DECOMPOSE-QSM code.
//!
//! The reference is the QSM-CI `decompose-qsm` container (the authors' `utils/` and fit loop,
//! compiled from licensed MATLAB R2023b and verified identical to it, QSMxT/QSM-CI#309). The
//! inputs are rebuilt here with the formulas of `tests/data/decompose_qsm_small.py`, which wrote
//! them for it; the fixture holds only its output: `chi-para`, then `chi-dia` (|χ−|), every
//! voxel, column-major, f64 (from its float32 files). The container gives every echo the same
//! QSM, so the per-echo QSM here is that χ repeated.

use qsm_core::separation::{decompose, DecomposeParams};

const DIMS: [usize; 3] = [8, 8, 6];
const TE: [f64; 6] = [0.004, 0.009, 0.014, 0.019, 0.024, 0.029];
const B0: f64 = 3.0;

/// (per-echo χ, voxel-major magnitude, mask), as the generator builds them.
fn case() -> (Vec<f64>, Vec<f64>, Vec<u8>) {
    let g = 42.58 * 2.0 * std::f64::consts::PI;
    let a = 2.0 * std::f64::consts::PI * g * B0 / (9.0 * 3.0_f64.sqrt());
    let [nx, ny, nz] = DIMS;
    let ne = TE.len();
    let n = nx * ny * nz;
    let (mut chi, mut mag, mut mask) = (vec![0.0; n * ne], vec![0.0; n * ne], vec![1u8; n]);
    for kk in 0..nz {
        for jj in 0..ny {
            for ii in 0..nx {
                let (i, j, k) = (ii as f64, jj as f64, kk as f64);
                let l = ii + jj * nx + kk * nx * ny;
                let (chip, chim) = (0.01 + 0.012 * i + 0.004 * k, -(0.005 + 0.008 * j));
                let (cp, cm, c0, r0) = (0.2 + 0.03 * k, 0.25, 0.55 - 0.03 * k, 18.0 + 1.5 * i);
                for (e, &t) in TE.iter().enumerate() {
                    let term = |amp: f64, decay: f64, w: f64| {
                        let m = amp * (-decay * t).exp();
                        (m * (-w * t).cos(), m * (-w * t).sin())
                    };
                    let p = term(cp, a * chip + r0, (2.0 / 3.0) * chip * g * B0);
                    let q = term(cm, -a * chim + r0, (2.0 / 3.0) * chim * g * B0);
                    let z = (p.0 + q.0 + c0 * (-r0 * t).exp(), p.1 + q.1);
                    mag[l * ne + e] = (z.0 * z.0 + z.1 * z.1).sqrt();
                    chi[l * ne + e] = chip + chim;
                }
                if ii == 0 && jj == 0 {
                    mask[l] = 0;
                }
            }
        }
    }
    (chi, mag, mask)
}

fn reference() -> (Vec<f64>, Vec<f64>) {
    let bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/decompose_qsm_small.bin")).unwrap();
    let v: Vec<f64> = bytes.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c)).collect();
    let n = DIMS.iter().product::<usize>();
    assert_eq!(v.len(), 2 * n);
    (v[..n].to_vec(), v[n..].to_vec())
}

fn nrmse_r(a: &[f64], b: &[f64]) -> (f64, f64) {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut d, mut nb, mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        d += (x - y).powi(2);
        nb += y * y;
        sab += (x - ma) * (y - mb);
        saa += (x - ma).powi(2);
        sbb += (y - mb).powi(2);
    }
    ((d / nb).sqrt(), sab / (saa * sbb).sqrt())
}

#[test]
fn decompose_reproduces_the_authors_code() {
    let (chi, mag, mask) = case();
    let params = DecomposeParams { b0: B0, ..Default::default() };
    let (pos, neg, _) = decompose(&chi, &mag, &TE, &mask, &params, |_, _| {}).unwrap();
    let (ref_para, ref_dia) = reference();
    let inside: Vec<usize> = (0..mask.len()).filter(|&l| mask[l] != 0).collect();
    let pick = |v: &[f64], f: fn(f64) -> f64| inside.iter().map(|&l| f(v[l])).collect::<Vec<_>>();
    let (p, rp) = (pick(&pos, |x| x), pick(&ref_para, |x| x));
    let (d, rd) = (pick(&neg, f64::abs), pick(&ref_dia, |x| x));
    let (e_p, r_p) = nrmse_r(&p, &rp);
    let (e_d, r_d) = nrmse_r(&d, &rd);
    eprintln!("chi+ NRMSE {:.3} %, r {r_p:.5}; chi- NRMSE {:.3} %, r {r_d:.5}", 100.0 * e_p, 100.0 * e_d);
    // 0.24 % and 0.90 % with the lsqcurvefit port; what remains sits in near-degenerate voxels
    // (see the module docs, *Agreement with the authors' code*).
    assert!(e_p < 0.005 && r_p > 0.9999, "chi+ NRMSE {e_p}, r {r_p}");
    assert!(e_d < 0.015 && r_d > 0.9999, "chi- NRMSE {e_d}, r {r_d}");
    // Masked-out voxels stay zero on both sides.
    for l in (0..mask.len()).filter(|&l| mask[l] == 0) {
        assert_eq!((pos[l], neg[l], ref_para[l], ref_dia[l]), (0.0, 0.0, 0.0, 0.0));
    }
}

/// Writes the fixture's per-voxel conj(y) for `scripts/lsq_trr_matlab/probe_lsq_trr.m`, which
/// runs the authors' fit on it in MATLAB and records every `lsqcurvefit` call
/// (`DECOMPOSE_PROBE_IN=<file> cargo test --test decompose_qsm_parity -- --ignored`).
#[test]
#[ignore]
fn decompose_probe_input() {
    let Ok(path) = std::env::var("DECOMPOSE_PROBE_IN") else { return };
    let (chi, mag, mask) = case();
    let ne = TE.len();
    let g = 42.58 * 2.0 * std::f64::consts::PI;
    let gmax = mag.iter().cloned().fold(0.0_f64, f64::max);
    let vox: Vec<usize> = (0..mask.len()).filter(|&l| mask[l] != 0).collect();
    let mut out: Vec<f64> = vec![vox.len() as f64, ne as f64, B0];
    out.extend_from_slice(&TE);
    for &l in &vox {
        let (mut re, mut im) = (Vec::new(), Vec::new());
        for e in 0..ne {
            let m = mag[l * ne + e] / gmax;
            let ph = -(2.0 / 3.0) * chi[l * ne + e] * g * B0 * TE[e];
            re.push(m * ph.cos());
            im.push(-(m * ph.sin()));
        }
        out.extend(re);
        out.extend(im);
    }
    let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap();
}

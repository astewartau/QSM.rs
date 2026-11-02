//! `bgremove::lbv` against the MEDI toolbox 2024.11.26's `LBV.m` and its full-multigrid
//! `mexMGv6`, as the verified QSM-CI `lbv-cornell` container runs them (QSMxT/QSM-CI#300).
//!
//! `tests/data/lbv_cornell_small.bin` is made by `tests/data/lbv_cornell_small.py` with that
//! container, so it can be regenerated in Actions (`.github/workflows/regenerate-fixtures.yml`):
//! a small total field (19×23×17, anisotropic voxels, a background from two outside dipoles plus
//! local sources and noise) and, for LBV.m's defaults and two other settings, mexMGv6's local
//! field in MEDI's units, the container's local field in ppm and LBV's interior.
//!
//! [`LbvPrecision::Single`] is mexMGv6's arithmetic and must match it bit for bit; the default,
//! double precision, runs the same algorithm and differs by rounding only.

use qsm_core::bgremove::{lbv, lbv_multigrid, LbvParams, LbvPrecision};
use qsm_core::Grid;

const DIMS: (usize, usize, usize) = (19, 23, 17);
const VS: (f64, f64, f64) = (0.9, 1.0, 1.2);
const TE: [f64; 3] = [0.004, 0.009, 0.014];
const B0: f64 = 3.0;

struct Case {
    /// mexMGv6's fL (radians over the echo spacing)
    fl: Vec<f64>,
    /// the container's local field (ppm, float32)
    local: Vec<f32>,
    interior: Vec<u8>,
}

struct Fixture {
    field: Vec<f64>,
    mask: Vec<u8>,
    cases: Vec<Case>,
}

/// The generator's CASES, as `LbvParams`
fn cases() -> Vec<LbvParams> {
    let base = LbvParams { b0: B0, echo_times: TE.to_vec(), precision: LbvPrecision::Single, ..LbvParams::default() };
    vec![
        base.clone(),
        LbvParams { tol: 0.005, depth: Some(2), peel: 1, n1: 5, n2: 7, n3: 11, ..base.clone() },
        LbvParams { depth: Some(0), n3: 3, ..base },
    ]
}

fn fixture() -> Fixture {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/lbv_cornell_small.bin")).expect("fixture");
    let n = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    assert_eq!(n, DIMS.0 * DIMS.1 * DIMS.2);
    let mut o = 4;
    let field: Vec<f64> = raw[o..o + 8 * n].chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    o += 8 * n;
    let labels = raw[o..o + n].to_vec();
    o += n;
    let mut f32s = || {
        let v: Vec<f32> = raw[o..o + 4 * n].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        o += 4 * n;
        v
    };
    let cases = (0..cases().len())
        .map(|k| {
            let fl = f32s().into_iter().map(f64::from).collect();
            let local = f32s();
            Case { fl, local, interior: labels.iter().map(|&b| (b >> (1 + k)) & 1).collect() }
        })
        .collect();
    assert_eq!(o, raw.len(), "fixture length");
    Fixture { field, mask: labels.iter().map(|&b| b & 1).collect(), cases }
}

fn grid() -> Grid {
    Grid::new(DIMS.0, DIMS.1, DIMS.2, VS.0, VS.1, VS.2)
}

/// recon.m's conversion: field * 1e-6 * 2*pi * CF * delta_TE
fn ifreq(field: &[f64]) -> Vec<f64> {
    let cf = 42.576e6 * B0;
    field.iter().map(|&f| f * 1e-6 * 2.0 * std::f64::consts::PI * cf * (TE[1] - TE[0])).collect()
}

/// mexMGv6's arithmetic reproduces the compiled mex bit for bit: the multigrid output, the
/// interior, and the local field in ppm after the unit conversions.
#[test]
fn single_precision_is_bit_identical_to_mexmgv6() {
    let fx = fixture();
    let g = grid();
    for (k, (p, c)) in cases().iter().zip(&fx.cases).enumerate() {
        let (fl, interior) = lbv_multigrid(&ifreq(&fx.field), &fx.mask, &g, p, |_, _| {});
        let differ = fl.iter().zip(&c.fl).filter(|(a, b)| a != b).count();
        assert_eq!(differ, 0, "case {k}: {differ} voxels of fL differ from mexMGv6");
        assert_eq!(interior, c.interior, "case {k}: interior");
        assert!(c.interior.iter().any(|&b| b != 0), "case {k}: empty interior");

        let (local, interior) = lbv(&fx.field, &fx.mask, &g, p, |_, _| {}).unwrap();
        let differ = local.iter().zip(&c.local).filter(|(a, &b)| **a as f32 != b).count();
        assert_eq!(differ, 0, "case {k}: {differ} voxels of the ppm local field differ from lbv-cornell");
        assert_eq!(interior, c.interior, "case {k}: interior from lbv");
    }
}

/// The default, double precision: the same algorithm, so the same result up to single
/// precision's rounding (relative 2-norm over the mask).
#[test]
fn double_precision_matches_to_rounding() {
    let fx = fixture();
    let g = grid();
    for (k, (p, c)) in cases().iter().zip(&fx.cases).enumerate() {
        let p = LbvParams { precision: LbvPrecision::Double, ..p.clone() };
        let (local, interior) = lbv(&fx.field, &fx.mask, &g, &p, |_, _| {}).unwrap();
        assert_eq!(interior, c.interior, "case {k}: interior");
        let (mut num, mut den) = (0.0, 0.0);
        for i in (0..local.len()).filter(|&i| fx.mask[i] != 0) {
            num += (local[i] - c.local[i] as f64).powi(2);
            den += (c.local[i] as f64).powi(2);
        }
        let nrmse = (num / den).sqrt();
        assert!(nrmse < 1e-5, "case {k}: NRMSE {nrmse:e}");
    }
}

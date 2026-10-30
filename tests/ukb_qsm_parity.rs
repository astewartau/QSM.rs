//! The UK Biobank QSM pipeline's steps against `UKBiobank_QSM_core.m` as the verified QSM-CI
//! `ukb-qsm` container runs it (QSMxT/QSM-CI#305): STI Suite's `MRPhaseUnwrap` per echo
//! (`laplacian_unwrap` with [`LaplacianSolver::Fft`], pad 64), the echo combination with
//! `W = TE·exp(−TE/40 ms)` (`calculate_b0_weighted` with [`B0WeightType::AssumedDecay`]),
//! and `V_SHARP` (`vsharp` at its defaults). Its last step, `QSM_iLSQR` with 64 mm padding, is
//! checked against STI Suite by `ilsqr_sti_parity.rs` instead (at that padding it takes ~10 s
//! even here).
//!
//! `tests/data/ukb_qsm_small.bin` is made by `tests/data/ukb_qsm_small.py` with that container, so
//! it can be regenerated in Actions (`.github/workflows/regenerate-fixtures.yml`); the phase is
//! rebuilt here exactly as the script builds it. Each step is given the container's own input
//! (its mask, combined phase), so each is compared on its own. UK Biobank's
//! phase-variance mask trimming is not a QSM.rs port; its masks come from the fixture.

use qsm_core::bgremove::{vsharp, VsharpParams};
use qsm_core::unwrap::{laplacian_unwrap, LaplacianSolver};
use qsm_core::utils::{calculate_b0_weighted, B0WeightType};
use qsm_core::Grid;
use std::f64::consts::PI;

const N: [usize; 3] = [32, 32, 20];
const VS: [f64; 3] = [1.0, 1.0, 1.5];
const TE: [f64; 2] = [0.00942, 0.020];

/// The generator's wrapped phase, per echo.
fn phase() -> Vec<Vec<f64>> {
    let nv = N.iter().product();
    let mut out = vec![vec![0.0; nv]; TE.len()];
    for k in 0..N[2] {
        for j in 0..N[1] {
            for i in 0..N[0] {
                let l = i + N[0] * (j + N[1] * k);
                let (x, y, z) = (i as f64 - 15.6, j as f64 - 15.3, (k as f64 - 9.7) * 1.5);
                let d1 = (x - 4.0) * (x - 4.0) + (y + 3.0) * (y + 3.0) + z * z;
                let omega = 0.03 * x - 0.02 * y + 0.01 * z + 0.002 * x * y - 0.001 * z * z + 0.08 / (1.0 + d1 / 6.0);
                for (e, &t) in TE.iter().enumerate() {
                    let raw = omega * (t * 1000.0) + 0.3;
                    out[e][l] = raw - 2.0 * PI * (raw / (2.0 * PI)).round_ties_even();
                }
            }
        }
    }
    out
}

struct Fixture {
    mask: Vec<u8>,
    mask_vsharp: Vec<u8>,
    /// over the whole grid
    phs_comb: Vec<f64>,
    /// on the grid, zero outside V_SHARP's mask
    local: Vec<f64>,
}

fn load() -> Fixture {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/ukb_qsm_small.bin")).expect("fixture");
    let n: usize = N.iter().product();
    let nb = n.div_ceil(8);
    let bits = |b: usize| -> Vec<u8> { (0..n).map(|i| raw[b * nb + i / 8] >> (i % 8) & 1).collect() };
    let (mask, mask_vsharp) = (bits(0), bits(1));
    let mut o = 2 * nb;
    let mut spread = |m: &[u8]| -> Vec<f64> {
        let mut v = vec![0.0; n];
        for l in (0..n).filter(|&l| m[l] != 0) {
            v[l] = f64::from_le_bytes(raw[o..o + 8].try_into().unwrap());
            o += 8;
        }
        v
    };
    let all = vec![1u8; n];
    let (phs_comb, local) = (spread(&all), spread(&mask_vsharp));
    assert_eq!(o, raw.len(), "fixture length");
    Fixture { mask, mask_vsharp, phs_comb, local }
}

fn grid() -> Grid {
    Grid::new(N[0], N[1], N[2], VS[0], VS[1], VS[2])
}

/// The effective echo time UK Biobank divides by, `Σ W·TE / Σ W` (seconds).
fn te_eff() -> f64 {
    let w: Vec<f64> = TE.iter().map(|&t| t * 1000.0 * (-(t * 1000.0) / 40.0).exp()).collect();
    w.iter().zip(&TE).map(|(w, t)| w * t).sum::<f64>() / w.iter().sum::<f64>()
}

#[test]
fn ukb_unwrapping_and_echo_combination_match_the_ukb_qsm_container() {
    let fx = load();
    let g = grid();
    let ones = vec![1u8; fx.mask.len()];
    let unwrapped: Vec<Vec<f64>> = phase().iter().map(|p| laplacian_unwrap(p, &ones, &g, LaplacianSolver::Fft { pad: [64; 3] })).collect();
    // B0 in Hz, then back to UK Biobank's combined phase at its effective echo time
    let b0 = calculate_b0_weighted(&unwrapped, &vec![vec![1.0; fx.mask.len()]; TE.len()], &TE, &fx.mask, B0WeightType::assumed_decay_default(), &g);
    let scale = 2.0 * PI * te_eff();
    let inside: Vec<usize> = (0..fx.mask.len()).filter(|&l| fx.mask[l] != 0).collect();
    let max = inside.iter().map(|&l| (b0[l] * scale - fx.phs_comb[l]).abs()).fold(0.0, f64::max);
    let rms = (inside.iter().map(|&l| fx.phs_comb[l].powi(2)).sum::<f64>() / inside.len() as f64).sqrt();
    println!("MRPhaseUnwrap + echo combination: max |Δ| {max:.2e} rad (rms {rms:.2})");
    // Measured 2026-10-30: max |Δ| 2.5e-14 rad on a combined phase of rms 2.0 rad.
    assert!(max < 1e-9, "max |Δ| {max:e}");
}

#[test]
fn ukb_vsharp_matches_the_ukb_qsm_container() {
    let fx = load();
    let (local, mask) = vsharp(&fx.phs_comb, &fx.mask, &grid(), &VsharpParams::default(), |_, _| {}).unwrap();
    let mask_diff = (0..mask.len()).filter(|&l| (mask[l] != 0) != (fx.mask_vsharp[l] != 0)).count();
    let both: Vec<usize> = (0..mask.len()).filter(|&l| mask[l] != 0 && fx.mask_vsharp[l] != 0).collect();
    let max = both.iter().map(|&l| (local[l] - fx.local[l]).abs()).fold(0.0, f64::max);
    let nrmse = (both.iter().map(|&l| (local[l] - fx.local[l]).powi(2)).sum::<f64>() / both.iter().map(|&l| fx.local[l].powi(2)).sum::<f64>()).sqrt();
    println!("V-SHARP: {mask_diff} mask voxels differ; max |Δ| {max:.2e} rad, NRMSE {nrmse:.2e}");
    assert_eq!(mask_diff, 0, "V-SHARP masks");
    // Measured 2026-10-30: the same eroded mask, NRMSE 1.2e-6 (max |Δ| 7e-7 rad; STI's float32).
    // Before STI's even crop (astewartau/QSM.rs#329, `vsharp` module docs) it was 4.1 %, and this
    // fixture, then holding the combined phase only inside the mask, showed 4.5 %.
    assert!(nrmse < 5e-6, "V-SHARP NRMSE {nrmse:e}");
}

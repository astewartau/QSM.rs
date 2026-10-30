//! `inversion::medi` and the MEDI pipeline steps around it (`fit_ppm_complex_noise`, `arlo`,
//! `extract_csf`) against the MEDI toolbox 2024.11.26, as the verified QSM-CI `medi-cornell`
//! container runs them (QSMxT/QSM-CI#309).
//!
//! `tests/data/medi_cornell_small.bin` is made by `tests/data/medi_cornell_small.py` with that
//! container, so it can be regenerated in Actions (`.github/workflows/regenerate-fixtures.yml`),
//! on the phantom of `tests/support/medi_phantom.rs` (rebuilt bit for bit in Python): MEDI_L1's χ
//! (float32) after one Gauss-Newton step with each SMV option, and with all of MEDI_L1's defaults
//! (λ 1000, SMV 5 with mask shrinking, MERIT, MEDI+0 with λ_CSF 100, 10 steps) on the phantom with
//! three CSF-like blobs, plus that run's N_std, R2* and CSF mask, and the mask MEDI_L1 inverted on.

use qsm_core::inversion::{extract_csf, medi, EchoMagnitudes, MediParams};
use qsm_core::utils::{arlo, fit_ppm_complex_noise};

#[path = "support/medi_phantom.rs"]
mod medi_phantom;
use medi_phantom::{diff, grid, phantom, phantom_csf, ppm_to_rad, Phantom, B0, TE};

struct Fixture {
    /// χ over the mask voxels, per case: nosmv, smv with shrinking, smv without, defaults
    chi: Vec<Vec<f64>>,
    n_std: Vec<f64>,
    r2s: Vec<f64>,
    csf: Vec<bool>,
    inverted: Vec<bool>,
}

fn fixture() -> Fixture {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/medi_cornell_small.bin")).expect("fixture");
    let m = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let mut o = 4;
    let chi = (0..4).map(|_| {
        let v = (0..m).map(|i| f32::from_le_bytes(raw[o + 4 * i..o + 4 * i + 4].try_into().unwrap()) as f64).collect();
        o += 4 * m;
        v
    }).collect();
    let mut f64s = || {
        let v: Vec<f64> = (0..m).map(|i| f64::from_le_bytes(raw[o + 8 * i..o + 8 * i + 8].try_into().unwrap())).collect();
        o += 8 * m;
        v
    };
    let (n_std, r2s) = (f64s(), f64s());
    let labels = &raw[o..o + m];
    assert_eq!(o + m, raw.len(), "fixture length");
    Fixture { n_std, r2s, csf: labels.iter().map(|l| l & 1 != 0).collect(), inverted: labels.iter().map(|l| l & 2 != 0).collect(), chi }
}

fn run(p: &Phantom, prm: &MediParams) -> Vec<f64> {
    let g = grid();
    let local: Vec<f64> = p.rdf.iter().map(|v| v / ppm_to_rad()).collect();
    let chi = medi(&local, &p.mask, EchoMagnitudes { data: &p.mag, n_echoes: 4 }, &g, (0.0, 0.0, 1.0), prm, |_, _| {});
    (0..g.n_total()).filter(|&i| p.mask[i] != 0).map(|i| chi[i]).collect()
}

fn params() -> MediParams {
    MediParams { b0: B0, echo_times: TE.to_vec(), ..MediParams::default() }
}

/// One Gauss-Newton step, every SMV option: where SMV acts (field, mask, kernel, noise).
#[test]
fn medi_one_step_matches_the_medi_cornell_container() {
    let fx = fixture();
    let p = phantom();
    for (c, (name, smv, smv_shrink_mask)) in [("nosmv", false, false), ("smv, shrink", true, true), ("smv, no shrink", true, false)].into_iter().enumerate() {
        let prm = MediParams { csf: false, max_iter: 1, smv, smv_shrink_mask, ..params() };
        let (max, rel) = diff(&run(&p, &prm), &fx.chi[c]);
        println!("{name}: max|Δ| {max:.2e} ppm, relative L2 {rel:.2e}");
        // Measured 2026-10-30: max|Δ| 2.7e-7, 1.3e-8, 1.5e-7 ppm; relative L2 1.1e-6, 1.9e-6,
        // 5.7e-7. MEDI_L1 runs CG in single precision, and the container writes χ as float32.
        assert!(max < 1e-6 && rel < 1e-5, "{name}: max|Δ| {max:e}, relative L2 {rel:e}");
    }
}

/// MEDI_L1's defaults end to end, MEDI+0 included; and the pipeline steps the container ran
/// before it on the same input.
#[test]
fn medi_defaults_match_the_medi_cornell_container() {
    let fx = fixture();
    let p = phantom_csf();
    let g = grid();
    let inside: Vec<usize> = (0..g.n_total()).filter(|&i| p.mask[i] != 0).collect();

    // N_std (Fit_ppm_complex on the magnitudes) and R2* (arlo)
    let n_std = fit_ppm_complex_noise(&p.mag, 4);
    let (max, rel) = diff(&inside.iter().map(|&i| n_std[i]).collect::<Vec<_>>(), &fx.n_std);
    println!("N_std: max|Δ| {max:.2e}, relative L2 {rel:.2e}");
    assert!(rel < 1e-12, "N_std: relative L2 {rel:e}");
    let r2s = arlo(&p.mag, &TE, g.n_total()).unwrap();
    // QSM.rs clamps R2* at 0 (a documented deviation); arlo.m does not.
    let want: Vec<f64> = fx.r2s.iter().map(|v| v.max(0.0)).collect();
    let (max, rel) = diff(&inside.iter().map(|&i| r2s[i]).collect::<Vec<_>>(), &want);
    println!("R2*: max|Δ| {max:.2e} Hz, relative L2 {rel:.2e}");
    assert!(rel < 1e-12, "R2*: relative L2 {rel:e}");

    // The CSF mask, from that R2* (the fixture holds R2* in the mask only, and extract_CSF also
    // reads it outside, in its central sphere; there arlo.m's negative R2* and QSM.rs's 0 are both
    // below the threshold)
    let csf = extract_csf(&r2s, &p.mask, &g, 5.0, true);
    let csf_diff = inside.iter().zip(&fx.csf).filter(|(&i, &c)| (csf[i] != 0) != c).count();
    println!("CSF mask: {} voxels, {csf_diff} differ", fx.csf.iter().filter(|&&c| c).count());
    assert_eq!(csf_diff, 0, "CSF masks differ");

    let chi = run(&p, &params());
    // MEDI_L1 zeroes χ outside the mask it inverted on (the SMV-eroded mask)
    let outside = inside.iter().enumerate().filter(|&(v, _)| !fx.inverted[v] && chi[v] != 0.0).count();
    assert_eq!(outside, 0, "χ outside MEDI_L1's inverted mask");
    let (max, rel) = diff(&chi, &fx.chi[3]);
    println!("defaults (10 steps, MERIT, SMV, MEDI+0): max|Δ| {max:.2e} ppm, relative L2 {rel:.2e}");
    // Measured 2026-10-30: max|Δ| 2.9e-8 ppm, relative L2 6.0e-7, with every CG solve stopping
    // where MATLAB's did. Where single (MATLAB) and double (here) rounding move a CG stop at
    // cg_tol 0.01 by one iteration, χ moves by ~1e-3 (the `medi` module docs), so a failure here
    // that is near 1e-3 is that, not a different algorithm.
    assert!(max < 1e-6 && rel < 1e-5, "defaults: max|Δ| {max:e}, relative L2 {rel:e}");
}

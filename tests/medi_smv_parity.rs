//! MEDI's SMV options against the MEDI toolbox's MATLAB: `MEDI_L1` with `'nosmv'`, with
//! `'smv', 5` (mask shrinking on, the default) and with `'smv', 5, 'smv_shrink_mask', 0`, on the
//! small synthetic phantom of `tests/support/medi_phantom.rs`.
//!
//! Reference: MEDI toolbox release 2024.11.26 (`functions/`), the release `medi` ports, run in
//! MATLAB R2023b. The fixture also holds the mSMV prerelease's `MEDI_L1`
//! (agr78/mSMV 42a1f18, `prerelease/MEDI_toolbox-mSMV100125.zip`) with SMV on and shrinking off,
//! which differs from the release: it does not high-pass the field (see
//! [`MediParams::smv_shrink_mask`]), and `medi` follows the release.
//!
//! ```matlab
//! % d = dump directory (rdf.f64, mask.u8, mag.f64 from dump_inputs); one run per toolbox,
//! % with only that toolbox's functions/ on the path
//! N = [32 36 28]; vs = [1 1 1.2]; TE = [4 8 12 16]*1e-3; B0 = 3; n = prod(N);
//! rd = @(f,t,k) reshape(fread(fopen(fullfile(d,f)),k,t), [N k/n]);
//! RDF = rd('rdf.f64','double',n); Mask = rd('mask.u8','uint8',n) > 0; mag = rd('mag.f64','double',4*n);
//! [~, N_std] = Fit_ppm_complex(mag); iMag = sqrt(sum(mag.^2, 4));
//! matrix_size = N; voxel_size = vs; delta_TE = TE(2)-TE(1); CF = 42.576e6*B0; B0_dir = [0 0 1];
//! iFreq = [];
//! save RDF.mat RDF iFreq N_std iMag Mask matrix_size voxel_size delta_TE CF B0_dir
//! one = {'lambda', 1000, 'max_iter', 1};                    % no Mask_CSF: no CSF term
//! chi_nosmv    = MEDI_L1('filename', 'RDF.mat', one{:}, 'nosmv');
//! chi_shrink   = MEDI_L1('filename', 'RDF.mat', one{:}, 'smv', 5);
//! chi_noshrink = MEDI_L1('filename', 'RDF.mat', one{:}, 'smv', 5, 'smv_shrink_mask', 0);
//! ```
//!
//! One Gauss-Newton step is where the SMV options act (the field, the mask, the dipole kernel
//! and the noise); MERIT and the later steps are MEDI's own code. With all defaults (10 steps)
//! the agreement is set by where CG stops at `cg_tol` 0.01 in single (MATLAB) and double (here)
//! precision; the numbers are in the `medi` module docs (SMV options).

use qsm_core::inversion::{medi, EchoMagnitudes, MediParams};

#[path = "support/medi_phantom.rs"]
mod medi_phantom;
use medi_phantom::{diff, grid, phantom, ppm_to_rad, B0, TE};

/// Writes the phantom for the MATLAB script (`MEDI_PARITY_DUMP=dir cargo test --test
/// medi_smv_parity -- --ignored dump_inputs`).
#[test]
#[ignore]
fn dump_inputs() {
    medi_phantom::dump(&std::env::var("MEDI_PARITY_DUMP").expect("set MEDI_PARITY_DUMP"));
}

/// `tests/data/medi_smv_small.bin` (little-endian): the number of mask voxels `m` (u32), then
/// over the mask voxels in column-major order (f64 each, ppm): the release's χ with `'nosmv'`,
/// with `'smv', 5`, with `'smv', 5, 'smv_shrink_mask', 0`, and the mSMV prerelease's χ with
/// `'smv', 5, 'smv_shrink_mask', 0`.
struct Fixture {
    nosmv: Vec<f64>,
    shrink: Vec<f64>,
    noshrink: Vec<f64>,
    prerelease_noshrink: Vec<f64>,
}

fn fixture() -> Fixture {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/medi_smv_small.bin")).expect("fixture");
    let m = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let mut o = 4;
    let mut next = || {
        let v: Vec<f64> = raw[o..o + 8 * m].chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
        o += 8 * m;
        v
    };
    let (nosmv, shrink, noshrink, prerelease_noshrink) = (next(), next(), next(), next());
    assert_eq!(o, raw.len(), "fixture length");
    Fixture { nosmv, shrink, noshrink, prerelease_noshrink }
}

/// `medi` with `smv` and `smv_shrink_mask` as given, one Gauss-Newton step, no CSF term, over
/// the mask voxels
fn run(smv: bool, smv_shrink_mask: bool) -> Vec<f64> {
    let p = phantom();
    let g = grid();
    let local: Vec<f64> = p.rdf.iter().map(|v| v / ppm_to_rad()).collect();
    let prm = MediParams {
        b0: B0,
        echo_times: TE.to_vec(),
        csf: false,
        max_iter: 1,
        smv,
        smv_shrink_mask,
        ..MediParams::default()
    };
    let chi = medi(&local, &p.mask, EchoMagnitudes { data: &p.mag, n_echoes: 4 }, &g, (0.0, 0.0, 1.0), &prm, |_, _| {});
    (0..g.n_total()).filter(|&i| p.mask[i] != 0).map(|i| chi[i]).collect()
}

/// Every SMV combination of the 2024.11.26 `MEDI_L1` (single precision there, double here).
#[test]
fn medi_smv_options_match_medi_l1_release() {
    let fx = fixture();
    for (name, smv, shrink, want) in [
        ("nosmv", false, false, &fx.nosmv),
        ("smv, shrink", true, true, &fx.shrink),
        ("smv, no shrink", true, false, &fx.noshrink),
    ] {
        let (max, rel) = diff(&run(smv, shrink), want);
        println!("{name}: max|Δ| {max:.2e} ppm, relative L2 {rel:.2e}");
        assert!(max < 1e-6 && rel < 1e-5, "{name}: max|Δ| {max:e}, relative L2 {rel:e}");
    }
    // smv_shrink_mask is ignored without SMV, as in MEDI_L1
    assert_eq!(run(false, true), run(false, false));
}

/// The mSMV prerelease's `MEDI_L1` with SMV on and shrinking off leaves the field unfiltered
/// but still fits it with the SMV-filtered kernel; `medi` follows the release, which filters it.
#[test]
fn prerelease_without_shrinking_differs_from_the_release() {
    let fx = fixture();
    let (_, rel) = diff(&fx.prerelease_noshrink, &fx.noshrink);
    println!("prerelease vs release, smv without shrinking: relative L2 {rel:.2e}");
    assert!(rel > 0.1, "the two releases now agree: relative L2 {rel:e}");
}

/// All defaults (10 Gauss-Newton steps, MERIT) against MATLAB's runs in `$MEDI_PARITY_FULL`
/// (`rel_{nosmv,shrink,noshrink}_full.f64`, χ over the mask voxels, written by the MATLAB script
/// with `MEDI_L1('filename', 'RDF.mat', 'lambda', 1000, ...)`); prints the differences quoted in
/// the `medi` module docs (SMV options). Not run in CI: where CG stops makes it a report.
#[test]
#[ignore]
fn full_defaults_report() {
    let dir = std::env::var("MEDI_PARITY_FULL").expect("set MEDI_PARITY_FULL");
    let p = phantom();
    let g = grid();
    let local: Vec<f64> = p.rdf.iter().map(|v| v / ppm_to_rad()).collect();
    for (name, smv, shrink) in [("nosmv", false, false), ("shrink", true, true), ("noshrink", true, false)] {
        let raw = std::fs::read(format!("{dir}/rel_{name}_full.f64")).unwrap();
        let want: Vec<f64> = raw.chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
        let prm = MediParams { b0: B0, echo_times: TE.to_vec(), csf: false, smv, smv_shrink_mask: shrink, ..MediParams::default() };
        let chi = medi(&local, &p.mask, EchoMagnitudes { data: &p.mag, n_echoes: 4 }, &g, (0.0, 0.0, 1.0), &prm, |_, _| {});
        let got: Vec<f64> = (0..g.n_total()).filter(|&i| p.mask[i] != 0).map(|i| chi[i]).collect();
        let (max, rel) = diff(&got, &want);
        println!("{name}, all defaults: max|Δ| {max:.2e} ppm, relative L2 {rel:.2e}");
    }
}

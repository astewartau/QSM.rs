//! MEDI-mSMV against the authors' MATLAB: `msmv.m` alone (prefilter on and off, with its vessel
//! mask and threshold) and `MEDI_L1(..., 'msmv', 5)` end to end, on a small synthetic phantom.
//!
//! Reference: agr78/mSMV 42a1f18, `prerelease/MEDI_toolbox-mSMV100125.zip` (`functions/`),
//! MATLAB R2023b with the Image Processing Toolbox (`fibermetric`). The phantom
//! (`tests/support/medi_phantom.rs`) is built from `+ − × ÷` only, so it is the same on every
//! platform; `dump_inputs` writes it for the MATLAB script below, and
//! `tests/data/msmv_medi_small.bin` holds MATLAB's results (layout in [`Fixture`]).
//!
//! ```matlab
//! % d = dump directory; reads rdf.f64, mask.u8, r2s.f64, mag.f64 (column-major, little-endian)
//! N = [32 36 28]; vs = [1 1 1.2]; TE = [4 8 12 16]*1e-3; B0 = 3; n = prod(N);
//! rd = @(f,t,k) reshape(fread(fopen(fullfile(d,f)),k,t), [N k/n]);
//! RDF = rd('rdf.f64','double',n); Mask = rd('mask.u8','uint8',n) > 0;
//! R2s = rd('r2s.f64','double',n); mag = rd('mag.f64','double',4*n);
//! for pf = [1 0]                                      % msmv.m alone
//!   out{pf+1} = msmv(RDF, double(Mask), R2s, vs, 5, B0, pf);
//! end
//! t = kernel_lim(RDF, vs, N, double(Mask), B0);       % msmv.m's own lines for t, Me, Mv
//! SphereK = single(sphere_kernel(N, vs, 5)); Me = double(Mask) - (SMV(double(Mask), SphereK) > 0.999);
//! Mask_ev = double(Mask) - MaskErode(double(Mask), N, vs, 6);
//! Mv = imbinarize(fibermetric(Mask_ev.*R2s, 1:15, 'ObjectPolarity', 'bright'), 0);
//! % MEDI_L1 'msmv' as MEDI's pipeline feeds it (no CSF term)
//! [~, N_std] = Fit_ppm_complex(mag); iMag = sqrt(sum(mag.^2, 4)); R2s = arlo(TE, mag);
//! matrix_size = N; voxel_size = vs; delta_TE = TE(2)-TE(1); CF = 42.576e6*B0; B0_dir = [0 0 1];
//! B0_mag = B0; iFreq = []; prefilter = 1;
//! save RDF.mat RDF iFreq N_std iMag Mask matrix_size voxel_size delta_TE CF B0_dir R2s B0_mag prefilter
//! opt = {'lambda', 1000, 'msmv', 5, 'max_iter', 1};
//! chi_on = MEDI_L1('filename', 'RDF.mat', opt{:});
//! prefilter = 0; save('RDF.mat', 'prefilter', '-append');
//! chi_off = MEDI_L1('filename', 'RDF.mat', opt{:});
//! ```
//!
//! The MEDI runs take one Gauss-Newton step, which is where mSMV enters MEDI (the mSMV field,
//! the uneroded mask, the SMV-filtered dipole kernel and noise); MERIT and the later steps are
//! MEDI's own code, parity-tested with MEDI. Over more steps the agreement is set by CG: at
//! MEDI's default `cg_tol` 0.01, CG stops after a number of iterations that single (MATLAB) and
//! double (here) rounding can change by one, which moves χ by ~1e-3 relative (on this phantom,
//! all defaults, 10 steps: 6e-4 relative L2 with the prefilter, 1e-3 without). With each step
//! solved further the difference goes away: two steps at `cg_tol` 1e-4 agree to 6e-6, three at
//! 1e-5 to 1e-6.

use qsm_core::inversion::{medi, msmv, EchoMagnitudes, MediParams, MsmvParams};

#[path = "support/medi_phantom.rs"]
mod medi_phantom;
use medi_phantom::{diff, grid, phantom, ppm_to_rad, B0, TE};

/// Writes the phantom for the MATLAB script (`MSMV_PARITY_DUMP=dir cargo test --test
/// msmv_medi_parity -- --ignored dump_inputs`).
#[test]
#[ignore]
fn dump_inputs() {
    medi_phantom::dump(&std::env::var("MSMV_PARITY_DUMP").expect("set MSMV_PARITY_DUMP"));
}

fn medi_params(prefilter: bool) -> MediParams {
    MediParams {
        b0: B0,
        echo_times: TE.to_vec(),
        csf: false,
        msmv: Some(MsmvParams { prefilter, ..MsmvParams::default() }),
        ..MediParams::default()
    }
}

/// `tests/data/msmv_medi_small.bin` (little-endian): the number of mask voxels `m` (u32),
/// `kernel_lim`'s `t` (f64), one u8 per voxel (0 outside `Me`, 1 in `Me`, 2 in `Me` and `Mv`),
/// then over the mask voxels in column-major order (f64 each): `msmv.m`'s output with the
/// prefilter on, with it off, and `MEDI_L1`'s χ (ppm) with the prefilter on and off.
struct Fixture {
    t: f64,
    labels: Vec<u8>,
    field_pf: Vec<f64>,
    field_nopf: Vec<f64>,
    chi_pf: Vec<f64>,
    chi_nopf: Vec<f64>,
}

fn fixture(n: usize) -> Fixture {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/msmv_medi_small.bin")).expect("fixture");
    let m = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let t = f64::from_le_bytes(raw[4..12].try_into().unwrap());
    let labels = raw[12..12 + n].to_vec();
    let mut o = 12 + n;
    let mut next = || {
        let v: Vec<f64> = raw[o..o + 8 * m].chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
        o += 8 * m;
        v
    };
    let (field_pf, field_nopf, chi_pf, chi_nopf) = (next(), next(), next(), next());
    assert_eq!(o, raw.len(), "fixture length");
    Fixture { t, labels, field_pf, field_nopf, chi_pf, chi_nopf }
}

/// `msmv.m`: the threshold and the vessel mask exactly, the field to double rounding without the
/// prefilter and to single rounding with it (`msmv.m` stores the prefilter kernel, and so the
/// prefiltered field, as single).
#[test]
fn msmv_matches_msmv_m() {
    let p = phantom();
    let g = grid();
    let n = g.n_total();
    let fx = fixture(n);
    let inside: Vec<usize> = (0..n).filter(|&i| p.mask[i] != 0).collect();
    for (prefilter, want) in [(true, &fx.field_pf), (false, &fx.field_nopf)] {
        let out = msmv(&p.rdf, &p.mask, &p.r2s, &g, 5.0, B0, &MsmvParams { prefilter, ..MsmvParams::default() });
        assert_eq!(out.threshold, fx.t, "kernel_lim's t");
        for i in 0..n {
            let label = u8::from(out.boundary[i]) + u8::from(out.boundary[i] && out.vessel_mask[i]);
            assert_eq!(label, fx.labels[i], "Me/Mv at voxel {i}");
        }
        assert!((0..n).all(|i| p.mask[i] != 0 || out.field[i] == 0.0), "field outside the mask");
        let got: Vec<f64> = inside.iter().map(|&i| out.field[i]).collect();
        let (max, rel) = diff(&got, want);
        println!("prefilter {prefilter}: {} passes, max|Δ| {max:.2e} rad, relative L2 {rel:.2e}", out.passes);
        let (max_tol, rel_tol) = if prefilter { (1e-6, 1e-6) } else { (1e-14, 1e-14) };
        assert!(max < max_tol && rel < rel_tol, "prefilter {prefilter}: max|Δ| {max:e}, relative L2 {rel:e}");
    }
}

/// `MEDI_L1(..., 'msmv', 5)` end to end (R2* by ARLO from the echoes, SMV-filtered dipole kernel,
/// the uneroded mask), one Gauss-Newton step with MEDI's defaults (see the module docs).
#[test]
fn medi_msmv_matches_medi_l1() {
    let p = phantom();
    let g = grid();
    let n = g.n_total();
    let fx = fixture(n);
    let inside: Vec<usize> = (0..n).filter(|&i| p.mask[i] != 0).collect();
    let local: Vec<f64> = p.rdf.iter().map(|v| v / ppm_to_rad()).collect();
    let mags = EchoMagnitudes { data: &p.mag, n_echoes: 4 };
    for (prefilter, want) in [(true, &fx.chi_pf), (false, &fx.chi_nopf)] {
        let prm = MediParams { max_iter: 1, ..medi_params(prefilter) };
        let chi = medi(&local, &p.mask, mags, &g, (0.0, 0.0, 1.0), &prm, |_, _| {});
        let got: Vec<f64> = inside.iter().map(|&i| chi[i]).collect();
        let (max, rel) = diff(&got, want);
        println!("MEDI-mSMV prefilter {prefilter}: max|Δ| {max:.2e} ppm, relative L2 {rel:.2e}");
        assert!(max < 2e-6 && rel < 2e-6, "prefilter {prefilter}: max|Δ| {max:e}, relative L2 {rel:e}");
    }
}

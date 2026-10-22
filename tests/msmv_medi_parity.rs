//! MEDI-mSMV against the authors' MATLAB: `msmv.m` alone (prefilter on and off, with its vessel
//! mask and threshold) and `MEDI_L1(..., 'msmv', 5)` end to end, on a small synthetic phantom.
//!
//! Reference: agr78/mSMV 42a1f18, `prerelease/MEDI_toolbox-mSMV100125.zip` (`functions/`),
//! MATLAB R2023b with the Image Processing Toolbox (`fibermetric`). The phantom is built here
//! from `+ − × ÷` only (no transcendental functions, no FFT), so it is the same on every
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
use qsm_core::Grid;

const DIMS: [usize; 3] = [32, 36, 28];
const VS: [f64; 3] = [1.0, 1.0, 1.2];
const TE: [f64; 4] = [0.004, 0.008, 0.012, 0.016];
const B0: f64 = 3.0;

/// Deterministic value in [-1, 1) from a voxel index (xorshift, integer arithmetic only)
fn hash(i: usize, salt: u64) -> f64 {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt.wrapping_mul(0xD1B5_4A32_D192_ED03);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

/// A rational bump `a / (1 + d²/w)`
fn bump(a: f64, d2: f64, w: f64) -> f64 {
    a / (1.0 + d2 / w)
}

struct Phantom {
    /// Local field in radians over the echo spacing (zero outside the mask)
    rdf: Vec<f64>,
    mask: Vec<u8>,
    /// R2* (Hz) for `msmv.m` alone
    r2s: Vec<f64>,
    /// Four echo magnitudes, echo slowest
    mag: Vec<f64>,
}

/// An ellipsoidal "brain" with a harmonic residual background (largest at the edge), a vein-like
/// tube and blobs in the field, bright tubes in R2* near the edge, and hashed texture; the
/// magnitudes decay as `M0/(1 + R2*·TE)`.
fn phantom() -> Phantom {
    let [nx, ny, nz] = DIMS;
    let n = nx * ny * nz;
    let (mut rdf, mut mask, mut r2s, mut mag) = (vec![0.0; n], vec![0u8; n], vec![0.0; n], vec![0.0; 4 * n]);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let p = i + nx * (j + ny * k);
                let (x, y, z) = (i as f64 - 16.0, j as f64 - 18.0, (k as f64 - 14.0) * VS[2]);
                let inside = x * x / 169.0 + y * y / 225.0 + z * z / 144.0 < 1.0;
                let tube = (x - 9.0) * (x - 9.0) + (y - 6.0) * (y - 6.0);
                let tube2 = (y + 10.0) * (y + 10.0) + (z - 4.0) * (z - 4.0);
                let blob = (x + 4.0) * (x + 4.0) + (y - 3.0) * (y - 3.0) + z * z;
                let r2 = 20.0 + bump(45.0, tube, 1.5) + bump(35.0, tube2 + 0.25 * (x + 2.0) * (x + 2.0), 1.0)
                    + bump(25.0, blob, 6.0) + 3.0 * hash(p, 1);
                r2s[p] = r2;
                let m0 = if inside { 1.0 + 0.05 * hash(p, 2) } else { 0.05 * (1.0 + hash(p, 3)) };
                for (e, &te) in TE.iter().enumerate() {
                    mag[p + n * e] = m0 / (1.0 + r2 * te) * (1.0 + 0.01 * hash(p, 10 + e as u64));
                }
                if inside {
                    mask[p] = 1;
                    rdf[p] = 0.002 * (x * x - y * y) + 0.0012 * x * z + bump(0.3, tube, 2.0)
                        - bump(0.15, tube2, 1.5) + bump(0.1, blob, 9.0) + 0.004 * hash(p, 4);
                }
            }
        }
    }
    Phantom { rdf, mask, r2s, mag }
}

fn grid() -> Grid {
    Grid::new(DIMS[0], DIMS[1], DIMS[2], VS[0], VS[1], VS[2])
}

/// Writes the phantom for the MATLAB script (`MSMV_PARITY_DUMP=dir cargo test --test
/// msmv_medi_parity -- --ignored dump_inputs`).
#[test]
#[ignore]
fn dump_inputs() {
    let dir = std::env::var("MSMV_PARITY_DUMP").expect("set MSMV_PARITY_DUMP");
    let p = phantom();
    let f64s = |v: &[f64]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    std::fs::write(format!("{dir}/rdf.f64"), f64s(&p.rdf)).unwrap();
    std::fs::write(format!("{dir}/r2s.f64"), f64s(&p.r2s)).unwrap();
    std::fs::write(format!("{dir}/mag.f64"), f64s(&p.mag)).unwrap();
    std::fs::write(format!("{dir}/mask.u8"), &p.mask).unwrap();
}

fn ppm_to_rad() -> f64 {
    2.0 * std::f64::consts::PI * (TE[1] - TE[0]) * 42.576e6 * B0 * 1e-6
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

/// (max |a − b|, relative L2 of a − b to b)
fn diff(a: &[f64], b: &[f64]) -> (f64, f64) {
    let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    let rel = (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / b.iter().map(|y| y * y).sum::<f64>()).sqrt();
    (max, rel)
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

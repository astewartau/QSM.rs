//! The FANSI-family ports (`inversion::ndi`, `fansi` nlTV/nlTGV, `l1qsm`, `whqsm`) and
//! `inversion::hdqsm` against the verified QSM-CI containers of the originals (FANSI-toolbox
//! 93044a3, HD-QSM c456a94; each bit-identical to licensed MATLAB R2023b), QSM.rs#399–#403.
//!
//! `tests/data/fansi_qsmci_small.bin` is made by `tests/data/fansi_qsmci_small.py` with those
//! containers (regenerated in Actions by `.github/workflows/regenerate-fixtures.yml`), on the MEDI
//! parity phantom (`tests/support/medi_phantom.rs`) with an oblique B0 and 1 x 1 x 1.2 mm voxels,
//! so FANSI's discrete kernel (`dipole_kernel_angulated` mode 1, voxel size normalised by its
//! largest) is checked off-axis and anisotropic. Each original runs at its defaults, as its
//! wrapper calls it; the fixture holds the unmasked solution (radians, HD-QSM ppm) over the mask.
//!
//! The wrappers convert ppm to radians with FANSI's `gyro = 2*pi*42.58`; the ports use the
//! crate's 42.576 (`crate::constants`), so the ports are given the field scaled by 42.58/42.576
//! (the same phase in radians) and their χ is compared in radians.

use qsm_core::constants::GAMMA_BAR_MHZ_PER_T;
use qsm_core::inversion::admm::fansi_ppm_to_rad;
use qsm_core::inversion::{
    fansi, hdqsm, hdqsm_magnitude, l1qsm, ndi, whqsm, FansiParams, HdQsmParams, L1QsmParams, NdiParams, WhQsmParams,
};
use qsm_core::EchoTime;

#[path = "support/medi_phantom.rs"]
mod medi_phantom;
use medi_phantom::{diff, grid, phantom, ppm_to_rad, B0, TE};

/// The fixture's B0 direction (fansi_qsmci_small.py's `B0_DIR`)
const B0_DIR: (f64, f64, f64) = (0.1, -0.15, 0.98);
/// Relative L2 limit. Measured 2026-11-02 (relative L2 / max|Δ| in radians, HD-QSM ppm): NDI
/// 5.9e-7 / 9.6e-7, nlTV 2.8e-6 / 4.7e-6, nlTGV 2.3e-6 / 3.9e-6, L1-QSM 8.9e-7 / 1.3e-6, WH-QSM
/// 4.5e-6 / 1.6e-5, HD-QSM 3.5e-6 / 2.2e-6: the originals iterate in single precision (`zeros(N,
/// 'single')`), the ports in double. With the continuous kernel NDI is 15 % off (relative L2).
const TOL: f64 = 2e-5;
const CASES: [&str; 6] = ["ndi", "nltv", "nltgv", "l1qsm", "whqsm", "hdqsm"];

fn fixture() -> Vec<Vec<f64>> {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/fansi_qsmci_small.bin")).expect("fixture");
    let m = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    assert_eq!(raw.len(), 4 + CASES.len() * 8 * m, "fixture length");
    (0..CASES.len())
        .map(|c| (0..m).map(|i| f64::from_le_bytes(raw[4 + 8 * (c * m + i)..12 + 8 * (c * m + i)].try_into().unwrap())).collect())
        .collect()
}

struct Inputs {
    /// The local field in ppm (the fixture's), and as the ports must get it to see FANSI's phase
    field: Vec<f64>,
    field_fansi: Vec<f64>,
    mask: Vec<u8>,
    /// Every echo's magnitude (echo slowest), and their root sum of squares (the wrappers' iMag)
    mag: Vec<f64>,
    rss: Vec<f64>,
}

fn inputs() -> Inputs {
    let p = phantom();
    let n = grid().n_total();
    let field: Vec<f64> = p.rdf.iter().map(|v| v / ppm_to_rad()).collect();
    let field_fansi = field.iter().map(|v| v * 42.58 / GAMMA_BAR_MHZ_PER_T).collect();
    let rss = (0..n).map(|i| (0..TE.len()).map(|e| p.mag[e * n + i].powi(2)).sum::<f64>().sqrt()).collect();
    Inputs { field, field_fansi, mask: p.mask, mag: p.mag, rss }
}

fn in_mask(v: &[f64], mask: &[u8], scale: f64) -> Vec<f64> {
    v.iter().zip(mask).filter(|(_, &m)| m != 0).map(|(x, _)| x * scale).collect()
}

fn check(name: &str, got: &[f64], want: &[f64], tol: f64) {
    let (max, rel) = diff(got, want);
    println!("{name}: max|Δ| {max:.2e}, relative L2 {rel:.2e}");
    assert!(rel < tol, "{name}: relative L2 {rel:e} (max|Δ| {max:e})");
}

#[test]
fn ndi_nltv_nltgv_l1qsm_match_the_fansi_containers() {
    let fx = fixture();
    let x = inputs();
    let g = grid();
    let te = TE.to_vec();
    let first = fansi_ppm_to_rad(B0, TE[0]);
    let spacing = fansi_ppm_to_rad(B0, TE[1] - TE[0]);

    let p = NdiParams { b0: B0, echo_times: te.clone(), ..Default::default() };
    let chi = ndi(&x.field_fansi, &x.mask, &x.rss, &g, B0_DIR, &p, |_, _| {});
    check("ndi", &in_mask(&chi, &x.mask, first), &fx[0], TOL);

    for (c, is_tgv) in [(1, false), (2, true)] {
        let p = FansiParams { b0: B0, echo_times: te.clone(), is_tgv, ..Default::default() };
        assert_eq!(p.te, EchoTime::Spacing);
        let chi = fansi(&x.field_fansi, &x.mask, &x.rss, &g, B0_DIR, &p, |_, _| {});
        check(CASES[c], &in_mask(&chi, &x.mask, spacing), &fx[c], TOL);
    }

    let p = L1QsmParams { b0: B0, echo_times: te.clone(), ..Default::default() };
    let chi = l1qsm(&x.field_fansi, &x.mask, &x.rss, &g, B0_DIR, &p, |_, _| {});
    check("l1qsm", &in_mask(&chi, &x.mask, first), &fx[3], TOL);
}

#[test]
fn whqsm_and_hdqsm_match_their_containers() {
    let fx = fixture();
    let x = inputs();
    let g = grid();
    let p = WhQsmParams { b0: B0, echo_times: TE.to_vec(), ..Default::default() };
    let chi = whqsm(&x.field_fansi, &x.mask, None, &g, B0_DIR, &p, |_, _| {});
    check("whqsm", &in_mask(&chi, &x.mask, fansi_ppm_to_rad(B0, TE[0])), &fx[4], TOL);

    // HD-QSM works in ppm (no γ), weighted by the paper's Eq. 2 magnitude.
    let mag = hdqsm_magnitude(&x.mag, &TE);
    let chi = hdqsm(&x.field, &x.mask, Some(&mag), &g, B0_DIR, &HdQsmParams::default(), |_, _| {});
    check("hdqsm", &in_mask(&chi, &x.mask, 1.0), &fx[5], TOL);
}

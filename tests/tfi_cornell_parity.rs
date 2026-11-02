//! `inversion::tfi` against TFI as the MEDI toolbox 2024.11.26 implements it (`MEDI_L1.m` with a
//! preconditioner), as the verified QSM-CI `tfi-cornell` container runs it (astewartau/QSM.rs#407).
//!
//! `tests/data/tfi_cornell_small.bin` is made by `tests/data/tfi_cornell_small.py` with that
//! container, so it can be regenerated in Actions (`.github/workflows/regenerate-fixtures.yml`), on
//! the phantom of `tests/support/medi_phantom.rs` cropped to an odd 31×36×27 grid (its field as the
//! total field): χ after one Gauss-Newton step without MERIT, and with the container's defaults.
//! The parity options make the port `MEDI_L1.m`: [`TfiKernel::MediOddGrid`] (on an odd grid
//! `dipole_kernel.m` is half a step off the DFT grid) and [`TfiPrecision::Single`] (`MEDI_L1.m`
//! runs in single precision). The defaults (DFT-grid kernel, double precision) are printed too.

use qsm_core::inversion::{tfi, EchoMagnitudes, TfiKernel, TfiParams, TfiPrecision};

#[path = "support/medi_phantom.rs"]
mod medi_phantom;
use medi_phantom::{diff, phantom, ppm_to_rad, B0, DIMS, TE, VS};

const CROP: [usize; 3] = [31, 36, 27];

#[test]
fn tfi_matches_the_tfi_cornell_container() {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/tfi_cornell_small.bin")).expect("fixture");
    let [cx, cy, cz] = CROP;
    let n = cx * cy * cz;
    assert_eq!(raw.len(), 2 * 4 * n, "fixture length");
    let p = phantom();
    let full = DIMS[0] * DIMS[1] * DIMS[2];
    let src = |q: usize| {
        let (i, j, k) = (q % cx, (q / cx) % cy, q / (cx * cy));
        i + DIMS[0] * (j + DIMS[1] * k)
    };
    let k = ppm_to_rad();
    let field: Vec<f64> = (0..n).map(|q| p.rdf[src(q)] / k).collect();
    let mask: Vec<u8> = (0..n).map(|q| p.mask[src(q)]).collect();
    let mag: Vec<f64> = (0..4).flat_map(|e| (0..n).map(move |q| (e, q))).map(|(e, q)| p.mag[src(q) + full * e]).collect();
    let grid = qsm_core::Grid::new(cx, cy, cz, VS[0], VS[1], VS[2]);
    let base = TfiParams { b0: B0, echo_times: TE.to_vec(), kernel: TfiKernel::MediOddGrid, precision: TfiPrecision::Single, ..TfiParams::default() };
    let cases = [
        ("one Gauss-Newton step, no MERIT", TfiParams { max_iter: 1, merit: false, ..base.clone() }),
        ("defaults", base.clone()),
    ];
    let inside: Vec<usize> = (0..n).filter(|&q| mask[q] != 0).collect();
    for (c, (name, prm)) in cases.iter().enumerate() {
        let o = c * 4 * n;
        let want: Vec<f64> = inside.iter().map(|&q| f32::from_le_bytes(raw[o + 4 * q..o + 4 * q + 4].try_into().unwrap()) as f64).collect();
        let run = |prm: &TfiParams| {
            let chi = tfi(&field, &mask, EchoMagnitudes { data: &mag, n_echoes: 4 }, &grid, (0.0, 0.0, 1.0), prm, |_, _| {});
            inside.iter().map(|&q| chi[q]).collect::<Vec<f64>>()
        };
        let (max, rel) = diff(&run(prm), &want);
        let (_, rel_dft) = diff(&run(&TfiParams { kernel: TfiKernel::Dft, ..prm.clone() }), &want);
        let (_, rel_double) = diff(&run(&TfiParams { precision: TfiPrecision::Double, ..prm.clone() }), &want);
        println!("{name}: max|Δ| {max:.2e} ppm, relative L2 {rel:.2e} (default kernel: {rel_dft:.2e}, double precision: {rel_double:.2e})");
        // Measured 2026-11-02: one step 2.9e-6 (single-precision rounding), defaults 4.7e-4 (double
        // precision 4.7e-3). MATLAB itself moves 7e-4 on this case for 1e-7 relative noise on the
        // field, and the container differs from itself by 3.0e-4 on another CPU: the 100-iteration
        // CG solves at cg_tol 0.01 in single precision amplify rounding.
        assert!(rel < 2e-3, "{name}: max|Δ| {max:e}, relative L2 {rel:e}");
        assert!(rel_dft > 10.0 * rel, "{name}: the default kernel should not match on an odd grid");
    }
}

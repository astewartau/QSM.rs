//! `bgremove::pdf` against the MEDI toolbox 2024.11.26's `PDF.m`, as the verified QSM-CI
//! `pdf-cornell` container runs it (QSMxT/QSM-CI#322; astewartau/QSM.rs#405).
//!
//! `tests/data/pdf_cornell_small.bin` is made by `tests/data/pdf_cornell_small.py` with that
//! container, so it can be regenerated in Actions (`.github/workflows/regenerate-fixtures.yml`), on
//! the phantom of `tests/support/medi_phantom.rs` (its field as the total field, N_std from
//! `Fit_ppm_complex` on its magnitudes): `PDF.m`'s output with its defaults and B0 along z (the
//! k-space kernel), and an oblique B0 with the image-space and the k-space kernel.

use qsm_core::bgremove::{pdf, DipoleKernelSpace, PdfParams};
use qsm_core::utils::fit_ppm_complex_noise;

#[path = "support/medi_phantom.rs"]
mod medi_phantom;
use medi_phantom::{diff, grid, phantom, ppm_to_rad};

#[test]
fn pdf_matches_the_pdf_cornell_container() {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/pdf_cornell_small.bin")).expect("fixture");
    let g = grid();
    let n = g.n_total();
    let p = phantom();
    let k = ppm_to_rad();
    let field: Vec<f64> = p.rdf.iter().map(|v| v / k).collect();
    let n_std: Vec<f64> = fit_ppm_complex_noise(&p.mag, 4).into_iter().map(|v| v / k).collect();
    let oblique = (0.1, 0.2, 0.97);
    let cases = [
        ("defaults, B0 along z", (0.0, 0.0, 1.0), PdfParams::default()),
        ("oblique, imagespace", oblique, PdfParams::default()),
        ("oblique, kspace", oblique, PdfParams { space: DipoleKernelSpace::KSpace, ..PdfParams::default() }),
    ];
    assert_eq!(raw.len(), cases.len() * 8 * n, "fixture length");
    for (c, (name, bdir, prm)) in cases.iter().enumerate() {
        let o = c * 8 * n;
        let want: Vec<f64> = (0..n).map(|i| f64::from_le_bytes(raw[o + 8 * i..o + 8 * i + 8].try_into().unwrap())).collect();
        let local = pdf(&field, &n_std, &p.mask, &g, *bdir, prm, |_, _| {}).expect("pdf");
        let got: Vec<f64> = local.iter().map(|v| v * k).collect();
        let (max, rel) = diff(&got, &want);
        println!("{name}: max|Δ| {max:.2e} rad, relative L2 {rel:.2e}");
        // Measured 2026-11-02: relative L2 1.3e-15, 1.9e-15, 1.1e-15 (rounding only).
        assert!(rel < 1e-11, "{name}: max|Δ| {max:e}, relative L2 {rel:e}");
    }
}

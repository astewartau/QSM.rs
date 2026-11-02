//! `bgremove::resharp` against Hongfu Sun's `resharp.m` (github.com/sunhongfu/QSM c75b138), as the
//! verified QSM-CI `resharp` container runs it (QSMxT/QSM-CI#352; astewartau/QSM.rs#404).
//!
//! `tests/data/resharp_sunhongfu_small.bin` is made by `tests/data/resharp_sunhongfu_small.py` with
//! that container, so it can be regenerated in Actions (`.github/workflows/regenerate-fixtures.yml`),
//! on the phantom of `tests/support/medi_phantom.rs` (1×1×1.2 mm voxels, so the kernel is
//! `round(r/voxel)` per axis): the local field (float64) and the eroded mask for `resharp.m`'s
//! defaults (converged), a 2 mm kernel with λ 1e-3, and a 5-iteration cap (unconverged, where
//! MATLAB's `cgs` returns its smallest-residual iterate).

use qsm_core::bgremove::{resharp, ResharpParams};

#[path = "support/medi_phantom.rs"]
mod medi_phantom;
use medi_phantom::{diff, grid, phantom};

#[test]
fn resharp_matches_the_resharp_container() {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/resharp_sunhongfu_small.bin")).expect("fixture");
    let g = grid();
    let n = g.n_total();
    let p = phantom();
    let cases = [
        ("defaults", ResharpParams::default()),
        ("2 mm, λ 1e-3", ResharpParams { radius: 2.0, tik_reg: 1e-3, ..ResharpParams::default() }),
        ("5 iterations", ResharpParams { max_iter: 5, ..ResharpParams::default() }),
    ];
    assert_eq!(raw.len(), cases.len() * 9 * n, "fixture length");
    for (c, (name, prm)) in cases.iter().enumerate() {
        let o = c * 9 * n;
        let want: Vec<f64> = (0..n).map(|i| f64::from_le_bytes(raw[o + 8 * i..o + 8 * i + 8].try_into().unwrap())).collect();
        let want_mask = &raw[o + 8 * n..o + 9 * n];
        let (local, eroded) = resharp(&p.rdf, &p.mask, &g, prm, |_, _| {});
        assert_eq!(eroded.as_slice(), want_mask, "{name}: eroded mask");
        let (max, rel) = diff(&local, &want);
        println!("{name}: max|Δ| {max:.2e}, relative L2 {rel:.2e}");
        // Measured 2026-11-02: relative L2 5.6e-15, 1.3e-14, 5.4e-15 (rounding only).
        assert!(rel < 1e-11, "{name}: max|Δ| {max:e}, relative L2 {rel:e}");
    }
}

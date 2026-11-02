//! Inversions and background removal from the total field or the phase against their originals
//! (astewartau/QSM.rs#317): `tests/data/dl_field_qsmci_small.bin`, made by
//! `tests/data/dl_field_qsmci_small.py` with the `autoqsm`, `nextqsm` and `bfrnet` entries (given
//! the box's total field and mask), then `iqsm`, `iqsm-plus` and `iqfm` on the first and the third
//! echo, each alone (given that echo's phase and TE, iQSM+ also its magnitude, and the mask).
//!
//! The iQSM family runs on one echo at a time because the echo combination is checked separately:
//! for iQSM and iQFM QSM.rs defaults to (mag·TE)², as the authors' Python driver weights, and
//! offers their MATLAB echo fit's mag·TE² as an option (`src/inversion/iqsm_echo_weighting.rs`). The
//! per-echo outputs are the network and its pre- and post-processing; the multi-echo test then
//! checks QSM.rs's combinations against the authors' weights applied to the originals' per-echo
//! outputs.

use super::{assert_parity, grid, invivo, originals, weights, B0, B0_DIR, TES};
use qsm_core::inversion::IqsmEchoWeighting;

/// The generator's `ECHOES` (0-based).
const ECHOES: [usize; 2] = [0, 2];
/// iQSM's defaults, as its `run.py` passes them: phase sign −1, mask eroded by 3 voxels.
const PHASE_SIGN: f64 = -1.0;
const ERODED_RAD: i32 = 3;

/// The originals' outputs, in the generator's `RUNS` order: AutoQSM, NeXtQSM, BFRnet, then iQSM,
/// iQSM+ and iQFM at each of [`ECHOES`].
fn field_originals(mask: &[u8]) -> Vec<Vec<f64>> {
    originals("dl_field_qsmci_small.bin", mask, 3 + 3 * ECHOES.len())
}

/// The index in [`field_originals`] of `model` (0 iQSM, 1 iQSM+, 2 iQFM) at `ECHOES[k]`.
fn per_echo(model: usize, k: usize) -> usize {
    3 + model * ECHOES.len() + k
}

/// The authors' MATLAB echo fit (`echofit.m`), Σ mag·TE²·χ / Σ mag·TE², of the originals'
/// per-echo outputs, at the mask voxels.
fn echo_fit(per_echo: &[&Vec<f64>], magnitude: &[Vec<f64>], mask: &[u8], weighting: IqsmEchoWeighting) -> Vec<f64> {
    let inside: Vec<usize> = (0..mask.len()).filter(|&i| mask[i] != 0).collect();
    (0..inside.len())
        .map(|v| {
            let (mut num, mut den) = (0.0, 0.0);
            for (k, &e) in ECHOES.iter().enumerate() {
                let w = weighting.weight(magnitude[e][inside[v]], TES[e]);
                num += w * per_echo[k][v];
                den += w;
            }
            if den > 0.0 { num / den } else { 0.0 }
        })
        .collect()
}

#[test]
fn autoqsm_matches_its_original() {
    let d = invivo();
    let chi = qsm_core::inversion::autoqsm(&d.total_field, &d.mask, &grid(), &weights("autoqsm")).unwrap();
    assert_parity("AutoQSM", &chi, &d.mask, &field_originals(&d.mask)[0], 1e-3, 2e-6);
}

/// Not to rounding, but at the package's own spread: 0.12 % NRMSE, max |Δ| 0.058 ppm. The nextqsm
/// package itself moves by 0.10 % (max |Δ| 0.058 ppm) on this box between two CPUs, because its
/// regulariser goes through `sign(VarNet(x))` (`src/inversion/nextqsm.rs`, astewartau/QSM.rs#330).
/// The limits sit just above that floor.
#[test]
fn nextqsm_matches_its_original() {
    let d = invivo();
    let w = qsm_core::models::weights("nextqsm").expect("nextqsm weights");
    let chi = qsm_core::inversion::nextqsm(&d.total_field, &d.mask, &grid(), B0_DIR, &w[0], &w[1]).unwrap();
    assert_parity("NeXtQSM", &chi, &d.mask, &field_originals(&d.mask)[1], 0.14, 0.065);
}

#[test]
fn bfrnet_matches_its_original() {
    let d = invivo();
    let local = qsm_core::bgremove::bfrnet(&d.total_field, &d.mask, &grid(), &weights("bfrnet")).unwrap();
    assert_parity("BFRnet", &local, &d.mask, &field_originals(&d.mask)[2], 1e-3, 1e-6);
}

#[test]
fn iqsm_matches_its_original_per_echo() {
    let d = invivo();
    let (w, orig) = (weights("iqsm"), field_originals(&d.mask));
    for (k, &e) in ECHOES.iter().enumerate() {
        let chi = qsm_core::inversion::iqsm(&d.phase[e], &d.mask, &grid(), TES[e], B0, PHASE_SIGN, ERODED_RAD, &w).unwrap();
        assert_parity(&format!("iQSM echo {}", e + 1), &chi, &d.mask, &orig[per_echo(0, k)], 1e-2, 2e-5);
    }
}

#[test]
fn iqsm_plus_matches_its_original_per_echo() {
    let d = invivo();
    let (w, orig) = (weights("iqsm-plus"), field_originals(&d.mask));
    for (k, &e) in ECHOES.iter().enumerate() {
        let chi = qsm_core::inversion::iqsm_plus(
            &d.phase[e], &d.mask, &grid(), TES[e], B0, B0_DIR, PHASE_SIGN, ERODED_RAD, &w,
        )
        .unwrap();
        assert_parity(&format!("iQSM+ echo {}", e + 1), &chi, &d.mask, &orig[per_echo(1, k)], 2e-3, 5e-6);
    }
}

#[test]
fn iqfm_matches_its_original_per_echo() {
    let d = invivo();
    let (w, orig) = (weights("iqfm"), field_originals(&d.mask));
    for (k, &e) in ECHOES.iter().enumerate() {
        let local = qsm_core::inversion::iqfm(&d.phase[e], &d.mask, &grid(), TES[e], B0, PHASE_SIGN, ERODED_RAD, &w).unwrap();
        assert_parity(&format!("iQFM echo {}", e + 1), &local, &d.mask, &orig[per_echo(2, k)], 3e-2, 2e-5);
    }
}

/// The three networks' multi-echo entry points, against the combination of the originals'
/// per-echo outputs with the authors' weights: iQSM and iQFM with each [`IqsmEchoWeighting`]
/// (the Python's (mag·TE)², the default, and the MATLAB echo fit's mag·TE²), iQSM+ with (mag·TE)²
/// (both of its authors' implementations).
#[test]
fn iqsm_family_multi_echo_matches_the_echo_combination_of_its_originals() {
    use IqsmEchoWeighting::{MagnitudeSquaredTeSquared as Python, MagnitudeTeSquared as Matlab};
    let d = invivo();
    let orig = field_originals(&d.mask);
    let phases: Vec<&[f64]> = ECHOES.iter().map(|&e| d.phase[e].as_slice()).collect();
    let mags: Vec<&[f64]> = ECHOES.iter().map(|&e| d.magnitude[e].as_slice()).collect();
    let tes: Vec<f64> = ECHOES.iter().map(|&e| TES[e]).collect();
    let fit = |model: usize, w| {
        let per: Vec<&Vec<f64>> = (0..ECHOES.len()).map(|k| &orig[per_echo(model, k)]).collect();
        echo_fit(&per, &d.magnitude, &d.mask, w)
    };
    let g = grid();
    for (w, name) in [(Matlab, "mag·TE²"), (Python, "(mag·TE)²")] {
        let iqsm = qsm_core::inversion::iqsm_multi_echo(
            &phases, &mags, &d.mask, &g, &tes, B0, PHASE_SIGN, ERODED_RAD, &weights("iqsm"), None, w,
        )
        .unwrap();
        assert_parity(&format!("iQSM, echoes 1 and 3, {name}"), &iqsm, &d.mask, &fit(0, w), 5e-3, 1e-5);
        let iqfm = qsm_core::inversion::iqfm_multi_echo(
            &phases, &mags, &d.mask, &g, &tes, B0, PHASE_SIGN, ERODED_RAD, &weights("iqfm"), None, w,
        )
        .unwrap();
        assert_parity(&format!("iQFM, echoes 1 and 3, {name}"), &iqfm, &d.mask, &fit(2, w), 1.5e-2, 1e-5);
    }
    let iqsm_plus = qsm_core::inversion::iqsm_plus_multi_echo(
        &phases, &mags, &d.mask, &g, &tes, B0, B0_DIR, PHASE_SIGN, ERODED_RAD, &weights("iqsm-plus"), None,
    )
    .unwrap();
    assert_parity("iQSM+, echoes 1 and 3, (mag·TE)²", &iqsm_plus, &d.mask, &fit(1, Python), 2e-3, 5e-6);
}

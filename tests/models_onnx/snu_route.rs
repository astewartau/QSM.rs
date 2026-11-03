//! The χ-separation toolbox's route (`chi_sepnet_general`): QSMnetME and χ-sepnet against QSM-CI's
//! `qsmnetme` and `chisepnet` entries (astewartau/QSM.rs#377). `tests/data/dl_snu_route_small.bin`,
//! made by `tests/data/dl_snu_route_small.py` with the entries' `chisep_route.py` (a reproduction of
//! the toolbox's P-code, verified against it in licensed MATLAB R2023b), on the box as acquired (A:
//! 1 mm, B0 along z) and declared at 0.9×1.0×1.1 mm with B0 along (0.1, 0.2, 0.97) (B: the rotation
//! and anisotropic resampling).

use qsm_core::separation::{ChiSepNetR2, ChiSepNetToolboxModels};
use qsm_core::Grid;

use super::{assert_parity, data_file, f32s, invivo, weights, B0, DIMS};

/// `(voxel size mm, B0 direction)` of cases A and B.
type Geometry = ((f64, f64, f64), (f64, f64, f64));
const CASES: [Geometry; 2] = [((1.0, 1.0, 1.0), (0.0, 0.0, 1.0)), ((0.9, 1.0, 1.1), (0.1, 0.2, 0.97))];

/// Limits `(NRMSE %, max |Δ| ppm)` per case, about 5 × what was measured (2026-11-03: A 4e-5 to
/// 8e-5 %, max |Δ| ≤ 2.1e-7; B 1.2e-4 to 2.3e-4 %, max |Δ| ≤ 1e-5, where B's tilted frame
/// extrapolates the small box's edges to ±2 ppm): ONNX Runtime against tract, float32 rounding.
const LIMITS: [(f64, f64); 2] = [(5e-4, 1e-6), (1.2e-3, 5e-5)];

fn case(i: usize) -> (Grid, (f64, f64, f64)) {
    let ((x, y, z), (bx, by, bz)) = CASES[i];
    let n = (bx * bx + by * by + bz * bz).sqrt();
    (Grid::new(DIMS[0], DIMS[1], DIMS[2], x, y, z), (bx / n, by / n, bz / n))
}

/// The fixture: the R2* input (whole box) and the eight outputs at the mask voxels.
fn fixture(mask: &[u8]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let raw = data_file("dl_snu_route_small.bin");
    let n: usize = DIMS.iter().product();
    let m = mask.iter().filter(|&&v| v != 0).count();
    assert_eq!(raw.len(), 4 * n + 4 * m * 8);
    (f32s(&raw[..4 * n]), raw[4 * n..].chunks_exact(4 * m).map(f32s).collect())
}

#[test]
fn qsmnetme_matches_its_original() {
    let d = invivo();
    let (_, orig) = fixture(&d.mask);
    let w = weights("qsmnetme");
    for (i, name) in ["QSMnetME A", "QSMnetME B"].into_iter().enumerate() {
        let (g, b0) = case(i);
        let chi = qsm_core::inversion::qsmnetme(&d.local_field, &d.mask, &g, b0, &w, &Default::default(), &Default::default()).unwrap();
        assert_parity(name, &chi, &d.mask, &orig[i], LIMITS[i].0, LIMITS[i].1);
    }
}

fn route(d: &super::Invivo, r2: ChiSepNetR2, i: usize) -> (Vec<f64>, Vec<f64>) {
    let (g, b0) = case(i);
    let (cs, qn, rp) = (weights("chi-sepnet"), weights("qsmnetme"), weights("r2primenet"));
    let models = ChiSepNetToolboxModels { chisepnet: &cs, qsmnetme: &qn, r2primenet: Some(&rp) };
    let (pos, neg, _) = qsm_core::separation::chisepnet_toolbox(
        &d.local_field, r2, &d.mask, &g, b0, B0, &models, &Default::default(), &Default::default(),
    )
    .unwrap();
    // QSM.rs returns χ− signed (≤ 0); the original writes its magnitude.
    (pos, neg.iter().map(|v| -v).collect())
}

#[test]
fn chisepnet_toolbox_r2prime_matches_its_original() {
    let d = invivo();
    let (_, orig) = fixture(&d.mask);
    for (i, case) in ["A", "B"].into_iter().enumerate() {
        let (pos, neg) = route(&d, ChiSepNetR2::R2Prime(&d.r2prime), i);
        let (nrmse, dmax) = LIMITS[i];
        assert_parity(&format!("χ-sepnet-R2′ χ+ {case}"), &pos, &d.mask, &orig[2 + 2 * i], nrmse, dmax);
        assert_parity(&format!("χ-sepnet-R2′ χ− {case}"), &neg, &d.mask, &orig[3 + 2 * i], nrmse, dmax);
    }
}

#[test]
fn chisepnet_toolbox_r2star_matches_its_original() {
    let d = invivo();
    let (r2star, orig) = fixture(&d.mask);
    let (pos, neg) = route(&d, ChiSepNetR2::R2Star(&r2star), 0);
    let (nrmse, dmax) = LIMITS[0];
    assert_parity("χ-sepnet-R2* χ+ A", &pos, &d.mask, &orig[6], nrmse, dmax);
    assert_parity("χ-sepnet-R2* χ− A", &neg, &d.mask, &orig[7], nrmse, dmax);
}

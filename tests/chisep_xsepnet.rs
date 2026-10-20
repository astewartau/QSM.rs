//! Integration test: χ-sepnet (deep learning) on the QSM-CI chisep phantom,
//! scored with the QSM-CI metrics (correlation / XSIM / NRMSE per source map) so
//! results flow into the same summary comment as the classical χ-separation
//! methods. Weights are fetched from the registry at run time (needs the
//! `download` feature, or `$QSM_MODEL_DIR`). Ignored by default.
//!
//! Also here: the deep-learning methods (χ-sepnet, SUSEP-Net) at their 32-bit (WASM) patch,
//! run through the pipeline's `run_separation` and compared with their default.
//!
//! Run with:
//!   QSMCI_CHISEP=/home/ashley/repos/qsm/qsmci/qsmci/data/sim/chisep \
//!     cargo test --release --features "onnx download" --test chisep_xsepnet -- --ignored --nocapture
#![cfg(feature = "onnx")]

mod common;

use common::{
    assert_chisep_separated, chisep_score, correlation, load_chisep_phantom, save_center_slices,
    ChiSepPhantom,
};
use qsm_core::pipeline::{
    run_separation, ScanMetadata, SeparationAlgorithm, SeparationConfig, SeparationInputs,
};
use qsm_core::separation::{chisepnet, ChiSepNetNorm, ChiSepNetParams, SusepNetParams};
use std::time::Instant;

#[test]
#[ignore]
fn test_chisep_xsepnet() {
    let Some(ph) = load_chisep_phantom() else {
        common::skip_for_missing_data("chisep phantom not found");
        return;
    };
    let (nx, ny, nz) = ph.dims;
    println!("[INFO] phantom {}x{}x{}", nx, ny, nz);

    let spec = qsm_core::models::find_model("chi-sepnet").expect("chi-sepnet in registry");
    let w = qsm_core::models::primary_weight_bytes(spec)
        .expect("chi-sepnet weights (set $QSM_MODEL_DIR or build with 'download')");

    // Feed the provided conventional QSM χ_total (the network's QSM input channel),
    // the local field and R2′ — matching the SNU-LIST recon.py.
    let t = Instant::now();
    let (chi_pos, chi_neg, _tot) = chisepnet(
        &ph.local_field_ppm,
        &ph.chi_total,
        &ph.r2prime,
        &ph.mask,
        &ph.grid,
        &w,
        &ChiSepNetNorm::default(),
        &ChiSepNetParams::default(),
    )
    .expect("chisepnet");
    let secs = t.elapsed().as_secs_f64();

    // χ− is returned signed (≤0); the ground truth stores it as a positive magnitude.
    let dia_mag: Vec<f64> = chi_neg.iter().map(|&v| -v).collect();
    chisep_score("chisepnet χ+", &chi_pos, &ph.gt_para, &ph.mask, ph.dims, secs);
    chisep_score("chisepnet χ−", &dia_mag, &ph.gt_dia, &ph.mask, ph.dims, secs);
    save_center_slices(&chi_pos, &ph.mask, ph.dims, "chisep_xsepnet_para");
    save_center_slices(&dia_mag, &ph.mask, ph.dims, "chisep_xsepnet_dia");

    assert!(chi_pos.iter().all(|v| v.is_finite()), "chisepnet produced non-finite χ+");
    assert!(dia_mag.iter().all(|v| v.is_finite()), "chisepnet produced non-finite χ−");
    assert_chisep_separated("chisepnet", &chi_pos, &dia_mag, &ph.gt_para, &ph.gt_dia, &ph.mask, 0.5, 0.5);
}

/// The centred `dims` box of the phantom's maps (the magnitudes, which these tests do not read,
/// are left out): SUSEP-Net in patches costs several times its whole-volume pass, and the whole
/// phantom would take most of a CI job's time limit.
fn crop(ph: &ChiSepPhantom, dims: (usize, usize, usize)) -> ChiSepPhantom {
    let (nx, ny, nz) = ph.dims;
    let (cx, cy, cz) = dims;
    let (ox, oy, oz) = ((nx - cx) / 2, (ny - cy) / 2, (nz - cz) / 2);
    let cut = |v: &[f64]| -> Vec<f64> {
        let mut out = Vec::with_capacity(cx * cy * cz);
        for z in 0..cz {
            for y in 0..cy {
                for x in 0..cx {
                    out.push(v[(x + ox) + nx * ((y + oy) + ny * (z + oz))]);
                }
            }
        }
        out
    };
    let mask_f: Vec<f64> = ph.mask.iter().map(|&m| m as f64).collect();
    let (vx, vy, vz) = ph.grid.voxel_size;
    ChiSepPhantom {
        local_field_ppm: cut(&ph.local_field_ppm),
        chi_total: cut(&ph.chi_total),
        chi_echoes: None,
        r2prime: cut(&ph.r2prime),
        magnitude_rss: Vec::new(),
        mag_voxel_major: Vec::new(),
        tes: ph.tes.clone(),
        se_mag_voxel_major: None,
        se_tes: Vec::new(),
        mask: cut(&mask_f).iter().map(|&m| (m > 0.5) as u8).collect(),
        gt_para: cut(&ph.gt_para),
        gt_dia: cut(&ph.gt_dia),
        grid: qsm_core::Grid::new(cx, cy, cz, vx, vy, vz),
        dims,
        b0: ph.b0,
        cf: ph.cf,
        bdir: ph.bdir,
    }
}

/// `‖a − b‖ / ‖b‖` over the mask.
fn rel_err(a: &[f64], b: &[f64], mask: &[u8]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for i in (0..a.len()).filter(|&i| mask[i] != 0) {
        num += (a[i] - b[i]).powi(2);
        den += b[i].powi(2);
    }
    (num / den).sqrt()
}

/// Run one χ-separation config through the pipeline dispatcher on the phantom.
fn separate(ph: &ChiSepPhantom, cfg: &SeparationConfig) -> (Vec<f64>, Vec<f64>, f64) {
    let meta = ScanMetadata {
        dims: ph.dims,
        voxel_size: ph.grid.voxel_size,
        echo_times: ph.tes.clone(),
        field_strength: ph.b0,
        b0_direction: ph.bdir,
        slice_geometry: None,
        affine: None,
    };
    let inputs = SeparationInputs {
        local_field_ppm: &ph.local_field_ppm,
        qsm: &ph.chi_total,
        qsm_echoes: None,
        mask: &ph.mask,
        r2prime: Some(&ph.r2prime),
        r2star: None,
        magnitude_rss: None,
        magnitude_multi: None,
        se_magnitude_multi: None,
    };
    let t = Instant::now();
    let r = run_separation(inputs, &meta, cfg, &mut |_, _| {}).expect("run_separation");
    let secs = t.elapsed().as_secs_f64();
    (r.chi_pos, r.chi_neg, secs)
}

/// A deep-learning separation at its 32-bit (WASM) patch against the same method at its default
/// (χ-sepnet: the authors' 192×192×128 patch; SUSEP-Net: the whole volume). Tiling changes what
/// each voxel's receptive field sees, so the two are not identical; this pins how close they are
/// (`min_corr` per source map against the default, and `max_rel_err` = ‖low − default‖/‖default‖)
/// and scores the low-memory result against the ground truth like the default.
fn low_memory_matches_default(
    ph: &ChiSepPhantom,
    name: &str,
    default: SeparationConfig,
    low: SeparationConfig,
    min_corr: f64,
    max_rel_err: f64,
) {
    let (dp, dn, dsecs) = separate(ph, &default);
    let (lp, ln, lsecs) = separate(ph, &low);
    let dia: Vec<f64> = ln.iter().map(|&v| -v).collect();
    chisep_score(&format!("{name} χ+ (low-memory patch)"), &lp, &ph.gt_para, &ph.mask, ph.dims, lsecs);
    chisep_score(&format!("{name} χ− (low-memory patch)"), &dia, &ph.gt_dia, &ph.mask, ph.dims, lsecs);
    for (map, low, def) in [("χ+", &lp, &dp), ("χ−", &ln, &dn)] {
        let r = correlation(low, def, &ph.mask);
        let e = rel_err(low, def, &ph.mask);
        println!(
            "[INFO] {name} {map}: low-memory vs default r = {r:.5}, rel. err = {e:.4} \
             ({lsecs:.1} s vs {dsecs:.1} s)"
        );
        assert!(low.iter().all(|v| v.is_finite()), "{name} {map}: non-finite");
        assert!(r >= min_corr, "{name} {map}: r = {r:.5} < {min_corr}");
        assert!(e <= max_rel_err, "{name} {map}: rel. err = {e:.4} > {max_rel_err}");
    }
}

#[test]
#[ignore]
fn test_chisepnet_low_memory_patch() {
    let Some(ph) = load_chisep_phantom() else { return };
    let alg = SeparationAlgorithm::ChiSepNet;
    low_memory_matches_default(
        &ph,
        "chisepnet",
        SeparationConfig { algorithm: alg, ..Default::default() },
        SeparationConfig { algorithm: alg, chi_sepnet: ChiSepNetParams::low_memory(), ..Default::default() },
        // Measured: r = 0.9989 (χ+) / 0.9965 (χ−), rel. err. 0.036 / 0.047.
        0.995,
        0.06,
    );
}

#[test]
#[ignore]
fn test_susep_net_low_memory_patch() {
    let Some(ph) = load_chisep_phantom() else { return };
    // 2 × 2 × 2 patches at the low-memory 96×96×64.
    let ph = crop(&ph, (144, 144, 112));
    let alg = SeparationAlgorithm::SusepNet;
    low_memory_matches_default(
        &ph,
        "susep-net",
        SeparationConfig { algorithm: alg, ..Default::default() },
        SeparationConfig { algorithm: alg, susep_net: SusepNetParams::low_memory(), ..Default::default() },
        // Measured on this crop: r = 0.9996 (χ+) / 0.9991 (χ−), rel. err. 0.021 / 0.028 (on the
        // whole phantom 0.9994 / 0.9985, 0.027 / 0.036).
        0.995,
        0.06,
    );
}

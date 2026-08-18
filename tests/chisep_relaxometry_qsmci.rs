//! Integration test: relaxometry-based χ-separation on the QSM-CI chisep phantom,
//! scored with the QSM-CI metrics (correlation / XSIM / NRMSE per source map) so
//! results are directly comparable to the leaderboard entries.
//!
//! Two methods that separate χ+ / χ− from a *provided* conventional QSM
//! (χ_total) plus a relaxometry map, without a field-inversion of their own:
//!   - **R2\*-QSM** (Dimov 2022): closed form from χ_total + R2* (fit here from the
//!     multi-echo magnitude). No R2' needed.
//!   - **WaveSep** (Fang 2023): wavelet-L1 proximal-gradient from χ_total + R2'.
//!
//! Both consume the phantom's `inputs/chimap.nii.gz` (the provided QSM), matching
//! the QSM-CI chi-separation contract.
//!
//! Run with:
//!   QSMCI_CHISEP=/home/ashley/repos/qsm/qsmci/qsmci/data/sim/chisep \
//!     cargo test --release --features parallel --test chisep_relaxometry_qsmci -- --ignored --nocapture

mod common;

use common::{correlation, load_nifti_file, nrmse, save_center_slices, xsim};
use qsm_core::separation::{
    r2star_qsm_from_magnitude, wavesep, R2starQsmParams, WaveSepParams,
};
use qsm_core::Grid;
use std::path::Path;
use std::time::Instant;

fn base_dir() -> String {
    std::env::var("QSMCI_CHISEP")
        .unwrap_or_else(|_| "/home/ashley/repos/qsm/qsmci/qsmci/data/sim/chisep".to_string())
}

struct Phantom {
    chi_total: Vec<f64>,       // provided QSM (ppm)
    r2prime: Vec<f64>,         // Hz
    mag_voxel_major: Vec<f64>, // (n_voxels, n_echoes) row-major
    tes: Vec<f64>,             // seconds
    mask: Vec<u8>,
    gt_para: Vec<f64>,
    gt_dia: Vec<f64>, // positive magnitude
    grid: Grid,
    dims: (usize, usize, usize),
    b0: f64,
}

fn load_phantom() -> Option<Phantom> {
    let base = base_dir();
    let inputs = format!("{}/inputs", base);
    let gt = format!("{}/groundtruth", base);
    if !Path::new(&inputs).exists() {
        println!("Skipping: phantom not found at {}", inputs);
        return None;
    }
    // These methods take a provided QSM (χ_total) plus relaxometry inputs; skip
    // cleanly if the phantom variant on hand doesn't ship all of them.
    for req in ["chimap.nii.gz", "magnitude.nii.gz", "r2prime.nii.gz"] {
        if !Path::new(&format!("{}/{}", inputs, req)).exists() {
            println!("Skipping: {} not in phantom at {}", req, inputs);
            return None;
        }
    }

    let params: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{}/params.json", inputs)).unwrap()).unwrap();
    let b0 = params["B0"].as_f64().expect("params.json B0");
    let vs: Vec<f64> = params["voxel_size"]
        .as_array()
        .expect("voxel_size")
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let tes: Vec<f64> = params["TE"]
        .as_array()
        .expect("TE")
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let chi = load_nifti_file(&format!("{}/chimap.nii.gz", inputs)).expect("chimap");
    let (nx, ny, nz) = chi.dims;
    let n = nx * ny * nz;
    let r2p = load_nifti_file(&format!("{}/r2prime.nii.gz", inputs)).expect("r2prime");
    let mask: Vec<u8> = load_nifti_file(&format!("{}/mask.nii.gz", inputs))
        .expect("mask")
        .data
        .iter()
        .map(|&v| (v > 0.5) as u8)
        .collect();

    // Multi-echo GRE magnitude, transposed to voxel-major (v*ne + e).
    let bytes = std::fs::read(format!("{}/magnitude.nii.gz", inputs)).expect("magnitude");
    let (mag4, (mx, my, mz, ne), _, _) = qsm_core::io::load_nifti_4d(&bytes).expect("4D magnitude");
    assert_eq!((mx, my, mz), (nx, ny, nz), "magnitude dims mismatch");
    assert_eq!(ne, tes.len(), "echo count vs TE list");
    let mut mag_voxel_major = vec![0.0_f64; n * ne];
    for e in 0..ne {
        for v in 0..n {
            mag_voxel_major[v * ne + e] = mag4[e * n + v];
        }
    }

    let gt_para = load_nifti_file(&format!("{}/chi-para.nii.gz", gt)).expect("chi-para").data;
    let gt_dia = load_nifti_file(&format!("{}/chi-dia.nii.gz", gt)).expect("chi-dia").data;

    Some(Phantom {
        chi_total: chi.data,
        r2prime: r2p.data,
        mag_voxel_major,
        tes,
        mask,
        gt_para,
        gt_dia,
        grid: Grid::new(nx, ny, nz, vs[0], vs[1], vs[2]),
        dims: (nx, ny, nz),
        b0,
    })
}

fn score(label: &str, recon: &[f64], truth: &[f64], mask: &[u8], dims: (usize, usize, usize)) {
    let corr = correlation(recon, truth, mask);
    let xs = xsim(recon, truth, mask, dims);
    let nr = nrmse(recon, truth, mask);
    println!("  {:14}  corr {:.4}   xsim {:.4}   nrmse {:.1}%", label, corr, xs, nr * 100.0);
    println!("CHISEPMETRIC {}|correlation|{:.4}", label, corr);
    println!("CHISEPMETRIC {}|xsim|{:.4}", label, xs);
    println!("CHISEPMETRIC {}|nrmse_pct|{:.1}", label, nr * 100.0);
}

#[test]
#[ignore] // needs the qsmci phantom; run with --release (see module docs)
fn test_r2star_qsm_qsmci() {
    let Some(ph) = load_phantom() else { return };
    let (nx, ny, nz) = ph.dims;
    println!("[INFO] R2*-QSM on qsmci chisep phantom {}x{}x{}  B0 {} T", nx, ny, nz, ph.b0);

    let params = R2starQsmParams { b0: ph.b0, r_const_3t: 274.0 };
    let t = Instant::now();
    let (chi_pos, chi_neg, _chi_total) = r2star_qsm_from_magnitude(
        &ph.mag_voxel_major,
        &ph.tes,
        &ph.chi_total,
        &ph.mask,
        &params,
    );
    println!("[INFO] R2*-QSM done in {:.2?}", t.elapsed());
    println!("CHISEPMETRIC r2star_qsm|runtime_s|{:.2}", t.elapsed().as_secs_f64());

    // GT diamagnetic map is stored as a positive magnitude; χ− is returned signed.
    let dia_mag: Vec<f64> = chi_neg.iter().map(|&v| -v).collect();
    println!("[RESULT] r2star-qsm (closed-form from χ_total + R2*):");
    score("R2*-QSM χ+", &chi_pos, &ph.gt_para, &ph.mask, ph.dims);
    score("R2*-QSM χ−", &dia_mag, &ph.gt_dia, &ph.mask, ph.dims);

    // Figures for the CI PR comment (rendered by render_slices.py).
    save_center_slices(&chi_pos, &ph.mask, ph.dims, "relaxchisep_r2starqsm_para");
    save_center_slices(&dia_mag, &ph.mask, ph.dims, "relaxchisep_r2starqsm_dia");

    // Regression floors — R2*-QSM's single-constant model is a coarse separator,
    // so these are conservative (it carries the irreversible R2 baseline).
    let para_corr = correlation(&chi_pos, &ph.gt_para, &ph.mask);
    assert!(para_corr > 0.3, "R2*-QSM chi+ corr {:.3} too low", para_corr);
}

#[test]
#[ignore] // needs the qsmci phantom; run with --release (see module docs)
fn test_wavesep_qsmci() {
    let Some(ph) = load_phantom() else { return };
    let (nx, ny, nz) = ph.dims;
    println!("[INFO] WaveSep on qsmci chisep phantom {}x{}x{}  B0 {} T", nx, ny, nz, ph.b0);

    // Dr = 137 Hz/ppm is the qsm-forward phantom's single kernel (Shin 2022).
    let params = WaveSepParams::default();
    let t = Instant::now();
    let (chi_pos, chi_neg, _chi_total) = wavesep(
        &ph.chi_total,
        &ph.r2prime,
        &ph.mask,
        &ph.grid,
        &params,
        |i, tot| {
            if i % 20 == 0 || i == tot {
                println!("  iter {}/{}", i, tot);
            }
        },
    );
    println!("[INFO] WaveSep done in {:.2?}", t.elapsed());
    println!("CHISEPMETRIC wavesep|runtime_s|{:.2}", t.elapsed().as_secs_f64());

    // GT diamagnetic map is stored as a positive magnitude; χ− is returned signed.
    let dia_mag: Vec<f64> = chi_neg.iter().map(|&v| -v).collect();
    println!("[RESULT] wavesep (wavelet-L1 from χ_total + R2'):");
    score("WaveSep χ+", &chi_pos, &ph.gt_para, &ph.mask, ph.dims);
    score("WaveSep χ−", &dia_mag, &ph.gt_dia, &ph.mask, ph.dims);

    // Figures for the CI PR comment (rendered by render_slices.py).
    save_center_slices(&chi_pos, &ph.mask, ph.dims, "relaxchisep_wavesep_para");
    save_center_slices(&dia_mag, &ph.mask, ph.dims, "relaxchisep_wavesep_dia");

    let para_corr = correlation(&chi_pos, &ph.gt_para, &ph.mask);
    let dia_corr = correlation(&dia_mag, &ph.gt_dia, &ph.mask);
    assert!(para_corr > 0.5, "WaveSep chi+ corr {:.3} too low", para_corr);
    assert!(dia_corr > 0.4, "WaveSep chi- corr {:.3} too low", dia_corr);
}

//! Integration test: χ-separation on the QSM-CI chisep phantom, scored with the
//! QSM-CI metrics (correlation / XSIM / NRMSE per source map) so results are
//! directly comparable to the leaderboard entries:
//!
//! | entry (MATLAB, `-iso` runs)   | χ+ xsim/corr | χ− xsim/corr |
//! |-------------------------------|--------------|--------------|
//! | chi-sep-ilsqr (SNU-LIST)      | 0.590/0.862  | 0.480/0.807  |
//! | chi-sep-medi  (SNU-LIST)      | 0.731/0.904  | 0.580/0.911  |
//! | chi-sepnet    (deep learning) | 0.887/0.944  | 0.719/0.884  |
//!
//! Caveat: the phantom was regenerated after those runs (GT corr 0.935 between
//! versions), so posted numbers are a baseline zone, not an exact target.
//! The pipeline mirrors QSM-CI's chi-sep-ilsqr entry: the conventional QSM fed
//! to χ-separation is reconstructed in-house with our iLSQR from the same local
//! field (feeding an external χ_total is known to break the toolbox — see the
//! entry's recon.m).
//!
//! Run with:
//!   QSMCI_CHISEP=/home/ashley/repos/qsm/qsmci/qsmci/data/sim/chisep \
//!     cargo test --release --features parallel --test chisep_qsmci -- --ignored --nocapture

mod common;

use common::{correlation, load_nifti_file, nrmse, save_center_slices, xsim};
use qsm_core::inversion::{ilsqr, IlsqrParams};
use qsm_core::separation::{chi_sep_ilsqr, chi_sep_medi, ChiSepIlsqrParams, ChiSepParams};
use qsm_core::Grid;
use std::path::Path;
use std::time::Instant;

const GAMMA: f64 = 42.576e6; // Hz/T (proton gyromagnetic ratio, as used library-wide)

fn base_dir() -> String {
    std::env::var("QSMCI_CHISEP")
        .unwrap_or_else(|_| "/home/ashley/repos/qsm/qsmci/qsmci/data/sim/chisep".to_string())
}

struct Phantom {
    local_field_ppm: Vec<f64>,
    r2prime: Vec<f64>,
    magnitude: Vec<f64>, // RSS over echoes
    mask: Vec<u8>,
    gt_para: Vec<f64>,
    gt_dia: Vec<f64>, // stored as positive magnitude
    grid: Grid,
    dims: (usize, usize, usize),
    cf: f64,
    bdir: (f64, f64, f64),
}

fn load_phantom() -> Option<Phantom> {
    let base = base_dir();
    let inputs = format!("{}/inputs", base);
    let gt = format!("{}/groundtruth", base);
    if !Path::new(&inputs).exists() {
        println!("Skipping: phantom not found at {}", inputs);
        return None;
    }

    let params: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{}/params.json", inputs)).unwrap()).unwrap();
    let b0 = params["B0"].as_f64().expect("params.json B0");
    let cf = GAMMA * b0;
    let vs: Vec<f64> = params["voxel_size"]
        .as_array()
        .expect("voxel_size")
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let bd: Vec<f64> = params["B0_dir"]
        .as_array()
        .expect("B0_dir")
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let lf = load_nifti_file(&format!("{}/localfield.nii.gz", inputs)).expect("localfield");
    let (nx, ny, nz) = lf.dims;
    let n = nx * ny * nz;
    let r2p = load_nifti_file(&format!("{}/r2prime.nii.gz", inputs)).expect("r2prime");
    let mask: Vec<u8> = load_nifti_file(&format!("{}/mask.nii.gz", inputs))
        .expect("mask")
        .data
        .iter()
        .map(|&v| (v > 0.5) as u8)
        .collect();

    // Multi-echo GRE magnitude → RSS combine (as the MATLAB entry does).
    let bytes = std::fs::read(format!("{}/magnitude.nii.gz", inputs)).expect("magnitude");
    let (mag4, (mx, my, mz, ne), _, _) = qsm_core::io::load_nifti_4d(&bytes).expect("4D magnitude");
    assert_eq!((mx, my, mz), (nx, ny, nz), "magnitude dims mismatch");
    let mut magnitude = vec![0.0_f64; n];
    for e in 0..ne {
        for v in 0..n {
            magnitude[v] += mag4[e * n + v] * mag4[e * n + v];
        }
    }
    for v in magnitude.iter_mut() {
        *v = v.sqrt();
    }

    let gt_para = load_nifti_file(&format!("{}/chi-para.nii.gz", gt)).expect("chi-para").data;
    let gt_dia = load_nifti_file(&format!("{}/chi-dia.nii.gz", gt)).expect("chi-dia").data;

    Some(Phantom {
        local_field_ppm: lf.data,
        r2prime: r2p.data,
        magnitude,
        mask,
        gt_para,
        gt_dia,
        grid: Grid::new(nx, ny, nz, vs[0], vs[1], vs[2]),
        dims: (nx, ny, nz),
        cf,
        bdir: (bd[0], bd[1], bd[2]),
    })
}

/// Print human-readable scores plus `CHISEPMETRIC name|metric|value` lines that
/// the chisep-demo CI workflow parses into the PR comment's metrics table.
fn score(label: &str, recon: &[f64], truth: &[f64], mask: &[u8], dims: (usize, usize, usize)) {
    let corr = correlation(recon, truth, mask);
    let xs = xsim(recon, truth, mask, dims);
    let nr = nrmse(recon, truth, mask);
    println!(
        "  {:20}  corr {:.4}   xsim {:.4}   nrmse {:.1}%",
        label,
        corr,
        xs,
        nr * 100.0
    );
    println!("CHISEPMETRIC {}|correlation|{:.4}", label, corr);
    println!("CHISEPMETRIC {}|xsim|{:.4}", label, xs);
    println!("CHISEPMETRIC {}|nrmse_pct|{:.1}", label, nr * 100.0);
}

// Note: qsmci's local `results/chi-sep-ilsqr-iso/` volumes are from the broken
// pre-fix diagnostic run (recon corr 0.18 vs its own truth — verified); the real
// scored volumes are on HuggingFace and were produced from an older phantom
// version, so no voxelwise MATLAB comparison is made here. For apples-to-apples,
// rerun the qsmci MATLAB entry on the current phantom.

#[test]
#[ignore] // needs the qsmci phantom; run with --release (see module docs)
fn test_chi_sep_ilsqr_qsmci() {
    let Some(ph) = load_phantom() else { return };
    let (nx, ny, nz) = ph.dims;
    println!(
        "[INFO] phantom {}x{}x{}  cf {:.1} MHz",
        nx,
        ny,
        nz,
        ph.cf / 1e6
    );

    // Conventional QSM from the same local field with our iLSQR (ppm in → ppm out).
    let t = Instant::now();
    let (qsm, _, _, _) = ilsqr(
        &ph.local_field_ppm,
        &ph.mask,
        &ph.grid,
        ph.bdir,
        &IlsqrParams::default(),
        |_, _| {},
    );
    println!("[INFO] iLSQR QSM in {:.1?}", t.elapsed());
    println!("CHISEPMETRIC iLSQR QSM|runtime_s|{:.1}", t.elapsed().as_secs_f64());
    score("QSM vs GT para+dia", &qsm, &{
        let mut t: Vec<f64> = ph.gt_para.iter().zip(ph.gt_dia.iter()).map(|(&p, &d)| p - d).collect();
        for (v, &m) in t.iter_mut().zip(ph.mask.iter()) {
            if m == 0 { *v = 0.0; }
        }
        t
    }, &ph.mask, ph.dims);

    let lambda1: f64 = std::env::var("CHISEP_LAMBDA1")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let params = ChiSepIlsqrParams {
        cf: ph.cf,
        lambda1,
        ..ChiSepIlsqrParams::default()
    };
    println!("[INFO] chi_sep_ilsqr  lambda1 {}", lambda1);
    let t = Instant::now();
    let (chi_pos, chi_neg, _) = chi_sep_ilsqr(
        &ph.local_field_ppm,
        &ph.r2prime,
        &ph.magnitude,
        &qsm,
        &ph.mask,
        &ph.grid,
        ph.bdir,
        &params,
        |i, tot| {
            if i % 5 == 0 || i == tot {
                println!("  iter {}/{}", i, tot);
            }
        },
    );
    println!("[INFO] chi_sep_ilsqr done in {:.1?}", t.elapsed());
    println!("CHISEPMETRIC chi_sep_ilsqr|runtime_s|{:.1}", t.elapsed().as_secs_f64());

    let dia_mag: Vec<f64> = chi_neg.iter().map(|&v| -v).collect();
    println!("[RESULT] chi_sep_ilsqr (target zone: MATLAB chi-sep-ilsqr para 0.59/0.86, dia 0.48/0.81):");
    score("chi+ vs GT", &chi_pos, &ph.gt_para, &ph.mask, ph.dims);
    score("chi- vs GT", &dia_mag, &ph.gt_dia, &ph.mask, ph.dims);

    // Center-slice figures for the CI PR comment (rendered by render_slices.py).
    save_center_slices(&ph.gt_para, &ph.mask, ph.dims, "chisep_para_truth");
    save_center_slices(&chi_pos, &ph.mask, ph.dims, "chisep_para");
    save_center_slices(&ph.gt_dia, &ph.mask, ph.dims, "chisep_dia_truth");
    save_center_slices(&dia_mag, &ph.mask, ph.dims, "chisep_dia");

    // Conservative gates: the CI dataset (chisep-mc, multicompartment) is harder
    // than the local single-compartment phantom, so these are regression floors,
    // not the headline numbers (chisep phantom: para 0.95, dia 0.87).
    let para_corr = correlation(&chi_pos, &ph.gt_para, &ph.mask);
    let dia_corr = correlation(&dia_mag, &ph.gt_dia, &ph.mask);
    assert!(para_corr > 0.6, "chi+ corr {:.3} too low", para_corr);
    assert!(dia_corr > 0.4, "chi- corr {:.3} too low", dia_corr);
}

#[test]
#[ignore] // needs the qsmci phantom; run with --release (see module docs)
fn test_chi_sep_medi_qsmci() {
    let Some(ph) = load_phantom() else { return };
    println!("[INFO] chi_sep_medi on qsmci phantom (posted MATLAB chi-sep-medi: para 0.73/0.90, dia 0.58/0.91)");

    let params = ChiSepParams {
        cf: ph.cf,
        dr_pos: 137.0,
        dr_neg: 137.0,
        max_iter: 30,
        ..ChiSepParams::default()
    };
    let t = Instant::now();
    let (chi_pos, chi_neg, _) = chi_sep_medi(
        &ph.local_field_ppm,
        &ph.r2prime,
        &ph.magnitude,
        &ph.mask,
        &ph.grid,
        ph.bdir,
        &params,
        |i, tot| {
            if i % 5 == 0 || i == tot {
                println!("  iter {}/{}", i, tot);
            }
        },
    );
    println!("[INFO] chi_sep_medi done in {:.1?}", t.elapsed());
    println!("CHISEPMETRIC chi_sep_medi|runtime_s|{:.1}", t.elapsed().as_secs_f64());

    let dia_mag: Vec<f64> = chi_neg.iter().map(|&v| -v).collect();
    println!("[RESULT] chi_sep_medi:");
    score("chi+ vs GT", &chi_pos, &ph.gt_para, &ph.mask, ph.dims);
    score("chi- vs GT", &dia_mag, &ph.gt_dia, &ph.mask, ph.dims);
}

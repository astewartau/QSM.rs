//! QSMART Validation Tests — Stage-by-stage comparison against MATLAB reference
//!
//! These tests compare each stage of the Rust QSMART pipeline against intermediate
//! NIfTI outputs from Warda Syeda's MATLAB QSMART implementation on a 7T human brain
//! dataset.
//!
//! Test data location: QSMART_TEST/1.7.136.6.1.1.17/
//!
//! MATLAB parameters (from Warda's EMAIL.txt):
//!   Field: 7T, Gyro: 2.675e8
//!   SDF stage 1: sigma1=10, sigma2=0
//!   SDF stage 2: sigma1=8, sigma2=2
//!   SDF spatial radius: 8, lower_lim: 0.7, curv_constant: 500
//!   Frangi: scaleRange=[0.01, 0.05], scaleRatio=0.01, C=500
//!   Vasculature sphere radius: 8
//!   iLSQR: cgs_num=500, inv_num=500, smv_rad=0.1
//!
//! Run with: cargo test --test qsmart_validation -- --ignored --nocapture

mod common;

use std::path::Path;
use std::time::Instant;
use common::load_nifti_file;
use qsm_core::bgremove::{sdf, SdfParams};
use qsm_core::inversion::{ilsqr, ilsqr_simple};
use qsm_core::inversion::tkd;
use qsm_core::utils::{
    calculate_curvature_proximity,
    generate_vasculature_mask, VasculatureParams,
    adjust_offset,
};
use qsm_core::nifti_io::save_nifti_to_file;

/// Path to the QSMART test data directory
const QSMART_DIR: &str = "QSMART_TEST/1.7.136.6.1.1.17";

/// Output directory for Rust-generated intermediates
const OUTPUT_DIR: &str = "QSMART_TEST/rust_output";

// ============================================================================
// MATLAB parameters from Warda's EMAIL.txt
// ============================================================================

const FIELD_TESLA: f64 = 7.0;
const GYRO: f64 = 2.675e8; // rad/s/T

// SDF parameters
const SDF_SPATIAL_RADIUS: i32 = 8;
const SDF_LOWER_LIM: f64 = 0.7;
const SDF_CURV_CONSTANT: f64 = 500.0;

const SDF_SIGMA1_STAGE1: f64 = 10.0;
const SDF_SIGMA2_STAGE1: f64 = 0.0; // NOTE: Warda uses 0, not 10!

const SDF_SIGMA1_STAGE2: f64 = 8.0;
const SDF_SIGMA2_STAGE2: f64 = 2.0;

// Frangi parameters (Warda's tuned values for this dataset)
const FRANGI_SCALE_RANGE: [f64; 2] = [0.01, 0.05];
const FRANGI_SCALE_RATIO: f64 = 0.01;
const FRANGI_C: f64 = 500.0;

// Vasculature
const VASC_SPHERE_RADIUS: i32 = 8;

// iLSQR parameters
const ILSQR_MAX_ITER: usize = 500;
const ILSQR_TOL: f64 = 0.01; // MATLAB uses cgs_num=500, inv_num=500

// B0 direction (assumed axial)
const B0_DIR: (f64, f64, f64) = (0.0, 0.0, 1.0);

// ============================================================================
// Data loader for QSMART test data
// ============================================================================

struct QsmartTestData {
    // Dimensions and voxel sizes from the first loaded NIfTI
    dims: (usize, usize, usize),
    voxel_size: (f64, f64, f64),
    affine: [f64; 16],

    // Inputs
    tfs: Vec<f64>,          // Total field shift
    mask: Vec<f64>,         // Brain mask (f64 for SDF)
    mask_u8: Vec<u8>,       // Brain mask (u8 for iLSQR/vasculature)
    magnitude: Vec<f64>,    // Combined magnitude (mag1_sos)

    // MATLAB intermediate outputs (ground truth for comparison)
    prox_matlab: Option<Vec<f64>>,
    curvature_matlab: Option<Vec<f64>>,
    alpha_matlab: Option<Vec<f64>>,
    enhanced_matlab: Option<Vec<f64>>,
    vasc_only_matlab: Option<Vec<f64>>,
    lfs_sdf_1_matlab: Option<Vec<f64>>,
    lfs_sdf_2_matlab: Option<Vec<f64>>,
    qsm_1_matlab: Option<Vec<f64>>,
    qsm_2_matlab: Option<Vec<f64>>,
    removed_voxels_matlab: Option<Vec<f64>>,
    combined_chi_matlab: Option<Vec<f64>>,
    qsmart_final_matlab: Option<Vec<f64>>,
}

impl QsmartTestData {
    fn load() -> Result<Self, String> {
        let base = QSMART_DIR;

        if !Path::new(base).exists() {
            return Err(format!(
                "QSMART test data not found at '{}'. \
                 Please ensure the QSMART_TEST directory is available.",
                base
            ));
        }

        // Load required inputs
        let tfs_nii = load_nifti_file(&format!("{}/tfs.nii", base))?;
        let dims = tfs_nii.dims;
        let voxel_size = tfs_nii.voxel_size;
        let affine = tfs_nii.affine;
        let tfs = tfs_nii.data;

        let mask_nii = load_nifti_file(&format!("{}/mask_used.nii", base))?;
        let mask: Vec<f64> = mask_nii.data.iter()
            .map(|&v| if v > 0.5 { 1.0 } else { 0.0 })
            .collect();
        let mask_u8: Vec<u8> = mask.iter()
            .map(|&v| if v > 0.5 { 1 } else { 0 })
            .collect();

        let mag_nii = load_nifti_file(&format!("{}/mag1_sos.nii", base))?;
        let magnitude = mag_nii.data;

        // Load optional MATLAB intermediates (don't fail if missing)
        let load_opt = |name: &str| -> Option<Vec<f64>> {
            let path = format!("{}/{}", base, name);
            match load_nifti_file(&path) {
                Ok(nii) => {
                    println!("[INFO] Loaded MATLAB reference: {}", name);
                    Some(nii.data)
                }
                Err(e) => {
                    println!("[WARN] Could not load {}: {}", name, e);
                    None
                }
            }
        };

        println!("[INFO] Volume dimensions: {:?}", dims);
        println!("[INFO] Voxel size: {:?}", voxel_size);
        println!("[INFO] Mask voxels: {}", mask.iter().filter(|&&v| v > 0.5).count());

        Ok(QsmartTestData {
            dims,
            voxel_size,
            affine,
            tfs,
            mask,
            mask_u8,
            magnitude,
            prox_matlab: load_opt("prox.nii"),
            curvature_matlab: load_opt("curvature.nii"),
            alpha_matlab: load_opt("alpha.nii"),
            enhanced_matlab: load_opt("enhanced.nii"),
            vasc_only_matlab: load_opt("vasc_only.nii"),
            lfs_sdf_1_matlab: load_opt("lfs_sdf_1.nii"),
            lfs_sdf_2_matlab: load_opt("lfs_sdf_2.nii"),
            qsm_1_matlab: load_opt("QSM_1.nii"),
            qsm_2_matlab: load_opt("QSM_2.nii"),
            removed_voxels_matlab: load_opt("removed_voxels.nii"),
            combined_chi_matlab: load_opt("combined_chi.nii"),
            qsmart_final_matlab: load_opt("QSMART_adjusted_offset.nii"),
        })
    }
}

// ============================================================================
// Comparison utilities
// ============================================================================

/// Compute comparison metrics between Rust output and MATLAB reference within a mask
fn compare_volumes(
    name: &str,
    rust: &[f64],
    matlab: &[f64],
    mask: &[f64],
) -> ComparisonResult {
    let mut sum_sq = 0.0;
    let mut sum_abs = 0.0;
    let mut max_abs = 0.0_f64;
    let mut count = 0usize;

    let mut sum_r = 0.0;
    let mut sum_m = 0.0;
    let mut sum_rm = 0.0;
    let mut sum_r2 = 0.0;
    let mut sum_m2 = 0.0;

    let mut min_m = f64::INFINITY;
    let mut max_m = f64::NEG_INFINITY;

    for i in 0..rust.len().min(matlab.len()) {
        if mask[i] > 0.5 {
            let r = rust[i];
            let m = matlab[i];
            let diff = r - m;

            sum_sq += diff * diff;
            sum_abs += diff.abs();
            max_abs = max_abs.max(diff.abs());
            count += 1;

            sum_r += r;
            sum_m += m;
            sum_rm += r * m;
            sum_r2 += r * r;
            sum_m2 += m * m;

            if m < min_m { min_m = m; }
            if m > max_m { max_m = m; }
        }
    }

    let n = count as f64;
    let rmse = if count > 0 { (sum_sq / n).sqrt() } else { f64::NAN };
    let mae = if count > 0 { sum_abs / n } else { f64::NAN };
    let range = max_m - min_m;
    let nrmse = if range > 1e-10 { rmse / range } else { f64::NAN };

    let corr = if count > 2 {
        let num = n * sum_rm - sum_r * sum_m;
        let den = ((n * sum_r2 - sum_r * sum_r) * (n * sum_m2 - sum_m * sum_m)).sqrt();
        if den > 1e-30 { num / den } else { 0.0 }
    } else {
        f64::NAN
    };

    let result = ComparisonResult {
        name: name.to_string(),
        rmse,
        nrmse,
        mae,
        max_abs_diff: max_abs,
        correlation: corr,
        n_voxels: count,
        matlab_range: (min_m, max_m),
    };

    result.print();
    result
}

/// Compare binary masks using Dice coefficient and count differences
fn compare_masks(
    name: &str,
    rust: &[f64],
    matlab: &[f64],
    brain_mask: &[f64],
) -> f64 {
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_ = 0usize;
    let mut tn = 0usize;

    for i in 0..rust.len().min(matlab.len()) {
        if brain_mask[i] > 0.5 {
            let r = rust[i] > 0.5;
            let m = matlab[i] > 0.5;
            match (r, m) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => tn += 1,
            }
        }
    }

    let dice = if tp + fp + fn_ > 0 {
        2.0 * tp as f64 / (2.0 * tp as f64 + fp as f64 + fn_ as f64)
    } else {
        1.0
    };

    println!("[{}] Dice={:.4}  TP={}  FP={}  FN={}  TN={}",
        name, dice, tp, fp, fn_, tn);

    dice
}

#[derive(Debug)]
struct ComparisonResult {
    name: String,
    rmse: f64,
    nrmse: f64,
    mae: f64,
    max_abs_diff: f64,
    correlation: f64,
    n_voxels: usize,
    matlab_range: (f64, f64),
}

impl ComparisonResult {
    fn print(&self) {
        println!(
            "[{:<25}] RMSE={:<12.6} NRMSE={:<8.4} MAE={:<12.6} MaxDiff={:<12.6} r={:<8.4}  (n={}, range=[{:.4},{:.4}])",
            self.name, self.rmse, self.nrmse, self.mae, self.max_abs_diff,
            self.correlation, self.n_voxels, self.matlab_range.0, self.matlab_range.1
        );
    }
}

/// Save a volume as NIfTI for visual inspection
fn save_output(data: &[f64], dims: (usize, usize, usize), voxel_size: (f64, f64, f64), affine: &[f64; 16], name: &str) {
    std::fs::create_dir_all(OUTPUT_DIR).ok();
    let path = format!("{}/{}", OUTPUT_DIR, name);
    match save_nifti_to_file(Path::new(&path), data, dims, voxel_size, affine) {
        Ok(()) => println!("[INFO] Saved Rust output: {}", path),
        Err(e) => println!("[WARN] Failed to save {}: {}", path, e),
    }
}

// ============================================================================
// Stage 1: Proximity map
// ============================================================================

#[test]
#[ignore]
fn test_qsmart_01_proximity() {
    println!("\n{}", "=".repeat(80));
    println!("QSMART VALIDATION: Stage 1 — Proximity Map");
    println!("{}\n", "=".repeat(80));

    let data = QsmartTestData::load().expect("Failed to load QSMART test data");
    let (nx, ny, nz) = data.dims;

    let matlab_prox = match &data.prox_matlab {
        Some(p) => p,
        None => { println!("[SKIP] No MATLAB prox.nii available"); return; }
    };

    let start = Instant::now();

    // Compute proximity map: Gaussian smooth of mask with [sigma1, 2*sigma1, 2*sigma1]
    // This is the first step of SDF — we replicate it here to compare the intermediate
    let sigma1 = SDF_SIGMA1_STAGE1;
    let prox_rust = gaussian_smooth_3d_masked(
        &data.mask, &data.mask, nx, ny, nz,
        &[sigma1, 2.0 * sigma1, 2.0 * sigma1],
    );

    let elapsed = start.elapsed();
    println!("[INFO] Proximity map computed in {:.2?}", elapsed);

    save_output(&prox_rust, data.dims, data.voxel_size, &data.affine, "prox_rust.nii");

    let result = compare_volumes("Proximity", &prox_rust, matlab_prox, &data.mask);

    // Print stats about the proximity values
    let (mut r_min, mut r_max) = (f64::MAX, f64::MIN);
    let (mut m_min, mut m_max) = (f64::MAX, f64::MIN);
    for i in 0..prox_rust.len() {
        if data.mask[i] > 0.5 {
            r_min = r_min.min(prox_rust[i]);
            r_max = r_max.max(prox_rust[i]);
            m_min = m_min.min(matlab_prox[i]);
            m_max = m_max.max(matlab_prox[i]);
        }
    }
    println!("[INFO] Rust  prox range: [{:.4}, {:.4}]", r_min, r_max);
    println!("[INFO] MATLAB prox range: [{:.4}, {:.4}]", m_min, m_max);

    println!("\n[RESULT] Proximity NRMSE={:.4}, r={:.4}", result.nrmse, result.correlation);
}

/// Replicate the anisotropic Gaussian smoothing from SDF
/// (This is the same as sdf.rs::gaussian_smooth_3d_masked_f64 but exposed here for testing)
fn gaussian_smooth_3d_masked(
    data: &[f64],
    mask: &[f64],
    nx: usize, ny: usize, nz: usize,
    sigmas: &[f64; 3],
) -> Vec<f64> {
    let smoothed_x = convolve_1d(data, nx, ny, nz, sigmas[0], 'x');
    let smoothed_xy = convolve_1d(&smoothed_x, nx, ny, nz, sigmas[1], 'y');
    let smoothed_xyz = convolve_1d(&smoothed_xy, nx, ny, nz, sigmas[2], 'z');

    smoothed_xyz.iter()
        .zip(mask.iter())
        .map(|(&v, &m)| if m > 0.0 { v } else { 0.0 })
        .collect()
}

fn convolve_1d(data: &[f64], nx: usize, ny: usize, nz: usize, sigma: f64, dir: char) -> Vec<f64> {
    if sigma <= 0.0 {
        return data.to_vec();
    }
    let kr = (2.0 * sigma).ceil() as usize;
    let ks = 2 * kr + 1;
    let mut kernel = vec![0.0f64; ks];
    let mut sum = 0.0;
    for i in 0..ks {
        let x = i as f64 - kr as f64;
        kernel[i] = (-x * x / (2.0 * sigma * sigma)).exp();
        sum += kernel[i];
    }
    for k in kernel.iter_mut() { *k /= sum; }

    let n_total = nx * ny * nz;
    let mut result = vec![0.0f64; n_total];
    let idx = |i: usize, j: usize, k: usize| i + j * nx + k * nx * ny;

    match dir {
        'x' => {
            for k in 0..nz { for j in 0..ny { for i in 0..nx {
                let mut s = 0.0;
                for ki in 0..ks {
                    let ni = (i as isize + ki as isize - kr as isize).max(0).min(nx as isize - 1) as usize;
                    s += data[idx(ni, j, k)] * kernel[ki];
                }
                result[idx(i, j, k)] = s;
            }}}
        }
        'y' => {
            for k in 0..nz { for j in 0..ny { for i in 0..nx {
                let mut s = 0.0;
                for ki in 0..ks {
                    let nj = (j as isize + ki as isize - kr as isize).max(0).min(ny as isize - 1) as usize;
                    s += data[idx(i, nj, k)] * kernel[ki];
                }
                result[idx(i, j, k)] = s;
            }}}
        }
        'z' => {
            for k in 0..nz { for j in 0..ny { for i in 0..nx {
                let mut s = 0.0;
                for ki in 0..ks {
                    let nk = (k as isize + ki as isize - kr as isize).max(0).min(nz as isize - 1) as usize;
                    s += data[idx(i, j, nk)] * kernel[ki];
                }
                result[idx(i, j, k)] = s;
            }}}
        }
        _ => panic!("Invalid direction"),
    }
    result
}

// ============================================================================
// Stage 2: Curvature
// ============================================================================

#[test]
#[ignore]
fn test_qsmart_02_curvature() {
    println!("\n{}", "=".repeat(80));
    println!("QSMART VALIDATION: Stage 2 — Surface Curvature");
    println!("{}\n", "=".repeat(80));

    let data = QsmartTestData::load().expect("Failed to load QSMART test data");
    let (nx, ny, nz) = data.dims;

    let matlab_curv = match &data.curvature_matlab {
        Some(c) => c,
        None => { println!("[SKIP] No MATLAB curvature.nii available"); return; }
    };

    let start = Instant::now();

    // First compute prox1 (same as proximity test)
    let sigma1 = SDF_SIGMA1_STAGE1;
    let prox1 = gaussian_smooth_3d_masked(
        &data.mask, &data.mask, nx, ny, nz,
        &[sigma1, 2.0 * sigma1, 2.0 * sigma1],
    );

    // Then compute curvature proximity
    let (prox_curv, curv_i) = calculate_curvature_proximity(
        &data.mask_u8,
        &prox1,
        SDF_LOWER_LIM,
        SDF_CURV_CONSTANT,
        sigma1,
        nx, ny, nz,
    );

    let elapsed = start.elapsed();
    println!("[INFO] Curvature computed in {:.2?}", elapsed);

    save_output(&curv_i, data.dims, data.voxel_size, &data.affine, "curvature_rust.nii");
    save_output(&prox_curv, data.dims, data.voxel_size, &data.affine, "prox_curv_rust.nii");

    let result = compare_volumes("Curvature", &curv_i, matlab_curv, &data.mask);

    println!("\n[RESULT] Curvature NRMSE={:.4}, r={:.4}", result.nrmse, result.correlation);
}

// ============================================================================
// Stage 3: Alpha map (SDF filter radii)
// ============================================================================

#[test]
#[ignore]
fn test_qsmart_03_alpha() {
    println!("\n{}", "=".repeat(80));
    println!("QSMART VALIDATION: Stage 3 — Alpha Map (SDF filter radii)");
    println!("{}\n", "=".repeat(80));

    let data = QsmartTestData::load().expect("Failed to load QSMART test data");
    let (nx, ny, nz) = data.dims;

    let matlab_alpha = match &data.alpha_matlab {
        Some(a) => a,
        None => { println!("[SKIP] No MATLAB alpha.nii available"); return; }
    };

    let start = Instant::now();

    // Compute prox1
    let sigma1 = SDF_SIGMA1_STAGE1;
    let prox1 = gaussian_smooth_3d_masked(
        &data.mask, &data.mask, nx, ny, nz,
        &[sigma1, 2.0 * sigma1, 2.0 * sigma1],
    );

    // Compute curvature-weighted proximity
    let (prox_curv, _) = calculate_curvature_proximity(
        &data.mask_u8,
        &prox1,
        SDF_LOWER_LIM,
        SDF_CURV_CONSTANT,
        sigma1,
        nx, ny, nz,
    );

    // For stage 1: sigma2=0, so prox_final = prox_curv (no vasculature weighting)
    let sigma = (SDF_SIGMA1_STAGE1 * SDF_SIGMA1_STAGE1 + SDF_SIGMA2_STAGE1 * SDF_SIGMA2_STAGE1).sqrt();
    let n = if sigma > 0.0 { -sigma.ln() / 0.5_f64.ln() } else { 0.0 };

    let alpha_rust: Vec<f64> = prox_curv.iter()
        .zip(data.mask.iter())
        .map(|(&p, &m)| {
            if m > 0.0 {
                sigma * (p.powf(n) * 100.0).round() / 100.0
            } else {
                0.0
            }
        })
        .collect();

    let elapsed = start.elapsed();
    println!("[INFO] Alpha map computed in {:.2?}", elapsed);
    println!("[INFO] sigma={:.4}, n={:.4}", sigma, n);

    save_output(&alpha_rust, data.dims, data.voxel_size, &data.affine, "alpha_rust.nii");

    let result = compare_volumes("Alpha", &alpha_rust, matlab_alpha, &data.mask);

    println!("\n[RESULT] Alpha NRMSE={:.4}, r={:.4}", result.nrmse, result.correlation);
}

// ============================================================================
// Stage 4: Vasculature detection (bottom-hat + Frangi + Otsu)
// ============================================================================

#[test]
#[ignore]
fn test_qsmart_04_vasculature() {
    println!("\n{}", "=".repeat(80));
    println!("QSMART VALIDATION: Stage 4 — Vasculature Detection");
    println!("{}\n", "=".repeat(80));

    let data = QsmartTestData::load().expect("Failed to load QSMART test data");
    let (nx, ny, nz) = data.dims;

    let start = Instant::now();

    // Run vasculature mask generation with Warda's parameters
    let vasc_params = VasculatureParams {
        sphere_radius: VASC_SPHERE_RADIUS,
        frangi_scale_range: FRANGI_SCALE_RANGE,
        frangi_scale_ratio: FRANGI_SCALE_RATIO,
        frangi_c: FRANGI_C,
    };

    let vasc_only_rust = generate_vasculature_mask(
        &data.magnitude,
        &data.mask_u8,
        nx, ny, nz,
        &vasc_params,
    );

    let elapsed = start.elapsed();
    println!("[INFO] Vasculature mask generated in {:.2?}", elapsed);

    save_output(&vasc_only_rust, data.dims, data.voxel_size, &data.affine, "vasc_only_rust.nii");

    // Count vessel voxels
    let vasc_count_rust = vasc_only_rust.iter()
        .zip(data.mask.iter())
        .filter(|(&v, &m)| m > 0.5 && v < 0.5)
        .count();
    let mask_count = data.mask.iter().filter(|&&m| m > 0.5).count();
    println!("[INFO] Rust vessel voxels: {} / {} ({:.1}%)",
        vasc_count_rust, mask_count, 100.0 * vasc_count_rust as f64 / mask_count as f64);

    if let Some(ref matlab_vasc) = data.vasc_only_matlab {
        let vasc_count_matlab = matlab_vasc.iter()
            .zip(data.mask.iter())
            .filter(|(&v, &m)| m > 0.5 && v < 0.5)
            .count();
        println!("[INFO] MATLAB vessel voxels: {} / {} ({:.1}%)",
            vasc_count_matlab, mask_count, 100.0 * vasc_count_matlab as f64 / mask_count as f64);

        let dice = compare_masks("Vasculature", &vasc_only_rust, matlab_vasc, &data.mask);
        println!("\n[RESULT] Vasculature mask Dice={:.4}", dice);
    } else {
        println!("[SKIP] No MATLAB vasc_only.nii for comparison");
    }

    // Also compare Frangi vesselness if enhanced.nii is available
    if let Some(ref matlab_enhanced) = data.enhanced_matlab {
        println!("\n--- Frangi vesselness comparison ---");
        let _result = compare_volumes("Frangi vesselness", &vasc_only_rust, matlab_enhanced, &data.mask);
        println!("(Note: comparing vasc_only_rust to enhanced.nii is not apples-to-apples)");
        println!("(Need to save Frangi output separately for proper comparison)");
    }
}

// ============================================================================
// Stage 5: SDF Background Removal — Stage 1 (whole ROI)
// ============================================================================

#[test]
#[ignore]
fn test_qsmart_05_sdf_stage1() {
    println!("\n{}", "=".repeat(80));
    println!("QSMART VALIDATION: Stage 5 — SDF Background Removal (Stage 1)");
    println!("{}\n", "=".repeat(80));

    let data = QsmartTestData::load().expect("Failed to load QSMART test data");
    let (nx, ny, nz) = data.dims;

    let matlab_lfs1 = match &data.lfs_sdf_1_matlab {
        Some(l) => l,
        None => { println!("[SKIP] No MATLAB lfs_sdf_1.nii available"); return; }
    };

    let start = Instant::now();

    // Stage 1 SDF: no vasculature weighting (all ones)
    let ones = vec![1.0f64; data.tfs.len()];
    let params_stage1 = SdfParams {
        sigma1: SDF_SIGMA1_STAGE1,
        sigma2: SDF_SIGMA2_STAGE1, // 0.0 per Warda
        spatial_radius: SDF_SPATIAL_RADIUS,
        lower_lim: SDF_LOWER_LIM,
        curv_constant: SDF_CURV_CONSTANT,
        use_curvature: true,
    };

    let lfs_stage1_rust = sdf(
        &data.tfs,
        &data.mask,
        &ones,
        nx, ny, nz,
        &params_stage1,
    );

    let elapsed = start.elapsed();
    println!("[INFO] SDF Stage 1 completed in {:.2?}", elapsed);

    save_output(&lfs_stage1_rust, data.dims, data.voxel_size, &data.affine, "lfs_sdf_1_rust.nii");

    let result = compare_volumes("SDF Stage 1", &lfs_stage1_rust, matlab_lfs1, &data.mask);

    // Also test with sigma2=10 (our current Rust default) to show the difference
    println!("\n--- For comparison, SDF Stage 1 with sigma2=10 (current Rust default) ---");
    let params_stage1_old = SdfParams {
        sigma1: SDF_SIGMA1_STAGE1,
        sigma2: 10.0, // current Rust default
        spatial_radius: SDF_SPATIAL_RADIUS,
        lower_lim: SDF_LOWER_LIM,
        curv_constant: SDF_CURV_CONSTANT,
        use_curvature: true,
    };
    let lfs_stage1_old = sdf(&data.tfs, &data.mask, &ones, nx, ny, nz, &params_stage1_old);
    let result_old = compare_volumes("SDF S1 (sigma2=10)", &lfs_stage1_old, matlab_lfs1, &data.mask);

    println!("\n[RESULT] SDF Stage 1 (sigma2=0) NRMSE={:.4}, r={:.4}", result.nrmse, result.correlation);
    println!("[RESULT] SDF Stage 1 (sigma2=10) NRMSE={:.4}, r={:.4}", result_old.nrmse, result_old.correlation);
}

// ============================================================================
// Stage 5b: SDF Background Removal — Stage 1, using MATLAB alpha as input
// (isolates SDF filtering from curvature/proximity errors)
// ============================================================================

#[test]
#[ignore]
fn test_qsmart_05b_sdf_stage1_with_matlab_alpha() {
    println!("\n{}", "=".repeat(80));
    println!("QSMART VALIDATION: Stage 5b — SDF Stage 1 (using MATLAB alpha)");
    println!("{}\n", "=".repeat(80));

    let data = QsmartTestData::load().expect("Failed to load QSMART test data");
    let (nx, ny, nz) = data.dims;
    let n_total = nx * ny * nz;

    let matlab_lfs1 = match &data.lfs_sdf_1_matlab {
        Some(l) => l,
        None => { println!("[SKIP] No MATLAB lfs_sdf_1.nii available"); return; }
    };
    let matlab_alpha = match &data.alpha_matlab {
        Some(a) => a,
        None => { println!("[SKIP] No MATLAB alpha.nii available"); return; }
    };

    let start = Instant::now();

    // Use MATLAB's alpha directly to apply the SDF filtering
    // This isolates the filtering step from proximity/curvature computation
    let sigma = (SDF_SIGMA1_STAGE1 * SDF_SIGMA1_STAGE1 + SDF_SIGMA2_STAGE1 * SDF_SIGMA2_STAGE1).sqrt();
    let filter_size = 2 * (2.0 * sigma).ceil() as usize + 1;

    // Get unique alpha values
    let mut unique_alphas: Vec<f64> = matlab_alpha.iter()
        .filter(|&&a| a > 0.0)
        .copied()
        .collect();
    unique_alphas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_alphas.dedup();

    println!("[INFO] {} unique alpha values, filter_size={}", unique_alphas.len(), filter_size);

    // Build alpha index map
    let mut alpha_index = vec![0usize; n_total];
    for i in 0..n_total {
        if matlab_alpha[i] > 0.0 {
            let idx = unique_alphas.iter().position(|&a| (a - matlab_alpha[i]).abs() < 1e-10).unwrap_or(0);
            alpha_index[i] = idx + 1;
        }
    }

    // Apply SDF filtering with MATLAB's alpha
    let mut background = vec![0.0f64; n_total];
    for (alpha_idx, &current_alpha) in unique_alphas.iter().enumerate() {
        if current_alpha <= 0.0 { continue; }

        let weighted_tfs: Vec<f64> = data.tfs.iter()
            .zip(data.mask.iter())
            .map(|(&t, &m)| t * m)
            .collect();

        let num = gaussian_smooth_3d_isotropic(&weighted_tfs, nx, ny, nz, current_alpha, filter_size);
        let denom = gaussian_smooth_3d_isotropic(&data.mask, nx, ny, nz, current_alpha, filter_size);

        for i in 0..n_total {
            if alpha_index[i] == alpha_idx + 1 {
                background[i] = if denom[i].abs() > 1e-10 { num[i] / denom[i] } else { 0.0 };
            }
        }
    }

    let lfs_rust: Vec<f64> = data.tfs.iter()
        .zip(background.iter())
        .zip(data.mask.iter())
        .map(|((&t, &b), &m)| (t - b) * m)
        .collect();

    let elapsed = start.elapsed();
    println!("[INFO] SDF Stage 1 (MATLAB alpha) completed in {:.2?}", elapsed);

    save_output(&lfs_rust, data.dims, data.voxel_size, &data.affine, "lfs_sdf_1_matlab_alpha_rust.nii");

    let result = compare_volumes("SDF S1 (MATLAB alpha)", &lfs_rust, matlab_lfs1, &data.mask);

    println!("\n[RESULT] SDF Stage 1 using MATLAB alpha: NRMSE={:.4}, r={:.4}", result.nrmse, result.correlation);
}

/// Isotropic Gaussian smoothing (matching SDF's gaussian_smooth_3d_with_filter_size)
fn gaussian_smooth_3d_isotropic(data: &[f64], nx: usize, ny: usize, nz: usize, sigma: f64, filter_size: usize) -> Vec<f64> {
    if sigma <= 0.0 { return data.to_vec(); }

    let kr = (filter_size - 1) / 2;
    let mut kernel = vec![0.0f64; filter_size];
    let mut sum = 0.0;
    for i in 0..filter_size {
        let x = i as f64 - kr as f64;
        kernel[i] = (-x * x / (2.0 * sigma * sigma)).exp();
        sum += kernel[i];
    }
    for k in kernel.iter_mut() { *k /= sum; }

    let n_total = nx * ny * nz;
    let idx = |i: usize, j: usize, k: usize| i + j * nx + k * nx * ny;

    let conv_dir = |input: &[f64], dir: char| -> Vec<f64> {
        let mut out = vec![0.0f64; n_total];
        match dir {
            'x' => {
                for k in 0..nz { for j in 0..ny { for i in 0..nx {
                    let mut s = 0.0;
                    for ki in 0..filter_size {
                        let ni = (i as isize + ki as isize - kr as isize).max(0).min(nx as isize - 1) as usize;
                        s += input[idx(ni, j, k)] * kernel[ki];
                    }
                    out[idx(i, j, k)] = s;
                }}}
            }
            'y' => {
                for k in 0..nz { for j in 0..ny { for i in 0..nx {
                    let mut s = 0.0;
                    for ki in 0..filter_size {
                        let nj = (j as isize + ki as isize - kr as isize).max(0).min(ny as isize - 1) as usize;
                        s += input[idx(i, nj, k)] * kernel[ki];
                    }
                    out[idx(i, j, k)] = s;
                }}}
            }
            'z' => {
                for k in 0..nz { for j in 0..ny { for i in 0..nx {
                    let mut s = 0.0;
                    for ki in 0..filter_size {
                        let nk = (k as isize + ki as isize - kr as isize).max(0).min(nz as isize - 1) as usize;
                        s += input[idx(i, j, nk)] * kernel[ki];
                    }
                    out[idx(i, j, k)] = s;
                }}}
            }
            _ => panic!("Invalid direction"),
        }
        out
    };

    let smoothed_x = conv_dir(data, 'x');
    let smoothed_xy = conv_dir(&smoothed_x, 'y');
    conv_dir(&smoothed_xy, 'z')
}

// ============================================================================
// Stage 6: SDF Background Removal — Stage 2 (tissue only)
// ============================================================================

#[test]
#[ignore]
fn test_qsmart_06_sdf_stage2() {
    println!("\n{}", "=".repeat(80));
    println!("QSMART VALIDATION: Stage 6 — SDF Background Removal (Stage 2)");
    println!("{}\n", "=".repeat(80));

    let data = QsmartTestData::load().expect("Failed to load QSMART test data");
    let (nx, ny, nz) = data.dims;

    let matlab_lfs2 = match &data.lfs_sdf_2_matlab {
        Some(l) => l,
        None => { println!("[SKIP] No MATLAB lfs_sdf_2.nii available"); return; }
    };

    // Use MATLAB vasculature mask if available, otherwise compute our own
    let vasc_only = match &data.vasc_only_matlab {
        Some(v) => {
            println!("[INFO] Using MATLAB vasc_only as input (isolating SDF from vasculature errors)");
            v.clone()
        }
        None => {
            println!("[INFO] Computing vasculature mask (no MATLAB vasc_only.nii)");
            let vasc_params = VasculatureParams {
                sphere_radius: VASC_SPHERE_RADIUS,
                frangi_scale_range: FRANGI_SCALE_RANGE,
                frangi_scale_ratio: FRANGI_SCALE_RATIO,
                frangi_c: FRANGI_C,
            };
            generate_vasculature_mask(&data.magnitude, &data.mask_u8, nx, ny, nz, &vasc_params)
        }
    };

    let start = Instant::now();

    let params_stage2 = SdfParams {
        sigma1: SDF_SIGMA1_STAGE2,
        sigma2: SDF_SIGMA2_STAGE2,
        spatial_radius: SDF_SPATIAL_RADIUS,
        lower_lim: SDF_LOWER_LIM,
        curv_constant: SDF_CURV_CONSTANT,
        use_curvature: true,
    };

    let lfs_stage2_rust = sdf(
        &data.tfs,
        &data.mask,
        &vasc_only,
        nx, ny, nz,
        &params_stage2,
    );

    let elapsed = start.elapsed();
    println!("[INFO] SDF Stage 2 completed in {:.2?}", elapsed);

    save_output(&lfs_stage2_rust, data.dims, data.voxel_size, &data.affine, "lfs_sdf_2_rust.nii");

    let result = compare_volumes("SDF Stage 2", &lfs_stage2_rust, matlab_lfs2, &data.mask);

    println!("\n[RESULT] SDF Stage 2 NRMSE={:.4}, r={:.4}", result.nrmse, result.correlation);
}

// ============================================================================
// Stage 7: iLSQR Inversion — Stage 1
// (uses MATLAB lfs_sdf_1 as input to isolate iLSQR from SDF errors)
// ============================================================================

#[test]
#[ignore]
fn test_qsmart_07_ilsqr_stage1() {
    println!("\n{}", "=".repeat(80));
    println!("QSMART VALIDATION: Stage 7 — iLSQR Inversion (Stage 1)");
    println!("{}\n", "=".repeat(80));

    let data = QsmartTestData::load().expect("Failed to load QSMART test data");
    let (nx, ny, nz) = data.dims;
    let (vsx, vsy, vsz) = data.voxel_size;

    let matlab_lfs1 = match &data.lfs_sdf_1_matlab {
        Some(l) => l,
        None => { println!("[SKIP] No MATLAB lfs_sdf_1.nii for input"); return; }
    };
    let matlab_qsm1 = match &data.qsm_1_matlab {
        Some(q) => q,
        None => { println!("[SKIP] No MATLAB QSM_1.nii for comparison"); return; }
    };

    let start = Instant::now();

    // Use MATLAB local field as input to isolate iLSQR errors
    let chi_stage1_rust = ilsqr_simple(
        matlab_lfs1,
        &data.mask_u8,
        nx, ny, nz,
        vsx, vsy, vsz,
        B0_DIR,
        ILSQR_TOL,
        ILSQR_MAX_ITER,
    );

    let elapsed = start.elapsed();
    println!("[INFO] iLSQR Stage 1 completed in {:.2?}", elapsed);

    save_output(&chi_stage1_rust, data.dims, data.voxel_size, &data.affine, "QSM_1_rust.nii");

    let result = compare_volumes("iLSQR Stage 1", &chi_stage1_rust, matlab_qsm1, &data.mask);

    println!("\n[RESULT] iLSQR Stage 1 NRMSE={:.4}, r={:.4}", result.nrmse, result.correlation);
}

// ============================================================================
// Stage 8: iLSQR Inversion — Stage 2
// ============================================================================

#[test]
#[ignore]
fn test_qsmart_08_ilsqr_stage2() {
    println!("\n{}", "=".repeat(80));
    println!("QSMART VALIDATION: Stage 8 — iLSQR Inversion (Stage 2)");
    println!("{}\n", "=".repeat(80));

    let data = QsmartTestData::load().expect("Failed to load QSMART test data");
    let (nx, ny, nz) = data.dims;
    let (vsx, vsy, vsz) = data.voxel_size;

    let matlab_lfs2 = match &data.lfs_sdf_2_matlab {
        Some(l) => l,
        None => { println!("[SKIP] No MATLAB lfs_sdf_2.nii for input"); return; }
    };
    let matlab_qsm2 = match &data.qsm_2_matlab {
        Some(q) => q,
        None => { println!("[SKIP] No MATLAB QSM_2.nii for comparison"); return; }
    };

    // Build tissue-only mask: brain mask AND vasculature mask
    let vasc_only = match &data.vasc_only_matlab {
        Some(v) => v.clone(),
        None => {
            println!("[INFO] No MATLAB vasc_only.nii; using brain mask only for stage 2");
            data.mask.clone()
        }
    };

    let mask_stage2: Vec<u8> = data.mask.iter()
        .zip(vasc_only.iter())
        .map(|(&m, &v)| if m > 0.5 && v > 0.5 { 1 } else { 0 })
        .collect();

    let start = Instant::now();

    let chi_stage2_rust = ilsqr_simple(
        matlab_lfs2,
        &mask_stage2,
        nx, ny, nz,
        vsx, vsy, vsz,
        B0_DIR,
        ILSQR_TOL,
        ILSQR_MAX_ITER,
    );

    let elapsed = start.elapsed();
    println!("[INFO] iLSQR Stage 2 completed in {:.2?}", elapsed);

    save_output(&chi_stage2_rust, data.dims, data.voxel_size, &data.affine, "QSM_2_rust.nii");

    let result = compare_volumes("iLSQR Stage 2", &chi_stage2_rust, matlab_qsm2, &data.mask);

    println!("\n[RESULT] iLSQR Stage 2 NRMSE={:.4}, r={:.4}", result.nrmse, result.correlation);
}

// ============================================================================
// Stage 9: Removed voxels mask
// ============================================================================

#[test]
#[ignore]
fn test_qsmart_09_removed_voxels() {
    println!("\n{}", "=".repeat(80));
    println!("QSMART VALIDATION: Stage 9 — Removed Voxels Mask");
    println!("{}\n", "=".repeat(80));

    let data = QsmartTestData::load().expect("Failed to load QSMART test data");

    let matlab_removed = match &data.removed_voxels_matlab {
        Some(r) => r,
        None => { println!("[SKIP] No MATLAB removed_voxels.nii available"); return; }
    };

    let vasc_only = match &data.vasc_only_matlab {
        Some(v) => v,
        None => { println!("[SKIP] No MATLAB vasc_only.nii for input"); return; }
    };

    // removed_voxels = mask - vasc_only
    // (assuming R_0 is all ones / incorporated into mask_used)
    let removed_rust: Vec<f64> = data.mask.iter()
        .zip(vasc_only.iter())
        .map(|(&m, &v)| {
            let val = m - v;
            if val < 0.0 { 0.0 } else { val }
        })
        .collect();

    save_output(&removed_rust, data.dims, data.voxel_size, &data.affine, "removed_voxels_rust.nii");

    let result = compare_volumes("Removed voxels", &removed_rust, matlab_removed, &data.mask);
    let dice = compare_masks("Removed voxels", &removed_rust, matlab_removed, &data.mask);

    println!("\n[RESULT] Removed voxels Dice={:.4}, NRMSE={:.4}", dice, result.nrmse);
}

// ============================================================================
// Stage 10: Offset adjustment + final combination
// (uses MATLAB intermediates as inputs)
// ============================================================================

#[test]
#[ignore]
fn test_qsmart_10_offset_adjustment() {
    println!("\n{}", "=".repeat(80));
    println!("QSMART VALIDATION: Stage 10 — Offset Adjustment & Combination");
    println!("{}\n", "=".repeat(80));

    let data = QsmartTestData::load().expect("Failed to load QSMART test data");
    let (nx, ny, nz) = data.dims;
    let (vsx, vsy, vsz) = data.voxel_size;

    let matlab_qsm1 = match &data.qsm_1_matlab {
        Some(q) => q,
        None => { println!("[SKIP] No QSM_1.nii"); return; }
    };
    let matlab_qsm2 = match &data.qsm_2_matlab {
        Some(q) => q,
        None => { println!("[SKIP] No QSM_2.nii"); return; }
    };
    let matlab_lfs1 = match &data.lfs_sdf_1_matlab {
        Some(l) => l,
        None => { println!("[SKIP] No lfs_sdf_1.nii"); return; }
    };
    let matlab_removed = match &data.removed_voxels_matlab {
        Some(r) => r,
        None => { println!("[SKIP] No removed_voxels.nii"); return; }
    };
    let matlab_combined = &data.combined_chi_matlab;
    let matlab_final = match &data.qsmart_final_matlab {
        Some(f) => f,
        None => { println!("[SKIP] No QSMART_adjusted_offset.nii"); return; }
    };

    let start = Instant::now();

    // PPM factor (matching QSMART's gyro * field / 1e6)
    let ppm = GYRO * FIELD_TESLA / 1e6;
    println!("[INFO] PPM factor: {:.4}", ppm);

    // Scale LFS to ppm (matching qsmbly pipeline)
    let lfs_scaled: Vec<f64> = matlab_lfs1.iter().map(|&v| v * ppm).collect();

    let chi_qsmart_rust = adjust_offset(
        matlab_removed,
        &lfs_scaled,
        matlab_qsm1,
        matlab_qsm2,
        nx, ny, nz,
        vsx, vsy, vsz,
        B0_DIR,
        ppm,
    );

    let elapsed = start.elapsed();
    println!("[INFO] Offset adjustment completed in {:.2?}", elapsed);

    save_output(&chi_qsmart_rust, data.dims, data.voxel_size, &data.affine, "QSMART_adjusted_offset_rust.nii");

    // Compare combined chi (before offset adjustment)
    if let Some(ref matlab_comb) = matlab_combined {
        // Reconstruct combined_chi = chi_1_masked + chi_2
        let combined_rust: Vec<f64> = matlab_qsm1.iter()
            .zip(matlab_qsm2.iter())
            .zip(matlab_removed.iter())
            .map(|((&c1, &c2), &r)| {
                let c1_masked = if r > 0.0 { c1 } else { 0.0 };
                c1_masked + c2
            })
            .collect();

        save_output(&combined_rust, data.dims, data.voxel_size, &data.affine, "combined_chi_rust.nii");
        compare_volumes("Combined chi", &combined_rust, matlab_comb, &data.mask);
    }

    let result = compare_volumes("QSMART final", &chi_qsmart_rust, matlab_final, &data.mask);

    println!("\n[RESULT] QSMART final NRMSE={:.4}, r={:.4}", result.nrmse, result.correlation);
}

// ============================================================================
// FULL PIPELINE: Run complete QSMART with Warda's parameters
// ============================================================================

#[test]
#[ignore]
fn test_qsmart_full_pipeline() {
    println!("\n{}", "=".repeat(80));
    println!("QSMART VALIDATION: Full Pipeline (Warda's parameters)");
    println!("{}\n", "=".repeat(80));

    let data = QsmartTestData::load().expect("Failed to load QSMART test data");
    let (nx, ny, nz) = data.dims;
    let (vsx, vsy, vsz) = data.voxel_size;
    let n_total = nx * ny * nz;

    let pipeline_start = Instant::now();

    // ========================================================================
    // Step 1: Vasculature detection
    // ========================================================================
    println!("[INFO] Step 1: Vasculature detection...");
    let step_start = Instant::now();

    let vasc_params = VasculatureParams {
        sphere_radius: VASC_SPHERE_RADIUS,
        frangi_scale_range: FRANGI_SCALE_RANGE,
        frangi_scale_ratio: FRANGI_SCALE_RATIO,
        frangi_c: FRANGI_C,
    };
    let vasc_only = generate_vasculature_mask(
        &data.magnitude, &data.mask_u8, nx, ny, nz, &vasc_params,
    );

    println!("[INFO] Vasculature detection: {:.2?}", step_start.elapsed());

    // ========================================================================
    // Step 2: SDF Stage 1 (whole ROI)
    // ========================================================================
    println!("[INFO] Step 2: SDF Stage 1...");
    let step_start = Instant::now();

    let ones = vec![1.0f64; n_total];
    let params_stage1 = SdfParams {
        sigma1: SDF_SIGMA1_STAGE1,
        sigma2: SDF_SIGMA2_STAGE1,
        spatial_radius: SDF_SPATIAL_RADIUS,
        lower_lim: SDF_LOWER_LIM,
        curv_constant: SDF_CURV_CONSTANT,
        use_curvature: true,
    };

    let lfs_stage1 = sdf(&data.tfs, &data.mask, &ones, nx, ny, nz, &params_stage1);
    println!("[INFO] SDF Stage 1: {:.2?}", step_start.elapsed());

    // ========================================================================
    // Step 3: iLSQR Stage 1
    // ========================================================================
    println!("[INFO] Step 3: iLSQR Stage 1...");
    let step_start = Instant::now();

    let chi_stage1 = ilsqr_simple(
        &lfs_stage1, &data.mask_u8,
        nx, ny, nz, vsx, vsy, vsz,
        B0_DIR, ILSQR_TOL, ILSQR_MAX_ITER,
    );
    println!("[INFO] iLSQR Stage 1: {:.2?}", step_start.elapsed());

    // ========================================================================
    // Step 4: SDF Stage 2 (tissue only)
    // ========================================================================
    println!("[INFO] Step 4: SDF Stage 2...");
    let step_start = Instant::now();

    let params_stage2 = SdfParams {
        sigma1: SDF_SIGMA1_STAGE2,
        sigma2: SDF_SIGMA2_STAGE2,
        spatial_radius: SDF_SPATIAL_RADIUS,
        lower_lim: SDF_LOWER_LIM,
        curv_constant: SDF_CURV_CONSTANT,
        use_curvature: true,
    };

    let lfs_stage2 = sdf(&data.tfs, &data.mask, &vasc_only, nx, ny, nz, &params_stage2);
    println!("[INFO] SDF Stage 2: {:.2?}", step_start.elapsed());

    // ========================================================================
    // Step 5: iLSQR Stage 2
    // ========================================================================
    println!("[INFO] Step 5: iLSQR Stage 2...");
    let step_start = Instant::now();

    let mask_stage2: Vec<u8> = data.mask.iter()
        .zip(vasc_only.iter())
        .map(|(&m, &v)| if m > 0.5 && v > 0.5 { 1 } else { 0 })
        .collect();

    let chi_stage2 = ilsqr_simple(
        &lfs_stage2, &mask_stage2,
        nx, ny, nz, vsx, vsy, vsz,
        B0_DIR, ILSQR_TOL, ILSQR_MAX_ITER,
    );
    println!("[INFO] iLSQR Stage 2: {:.2?}", step_start.elapsed());

    // ========================================================================
    // Step 6: Offset adjustment
    // ========================================================================
    println!("[INFO] Step 6: Offset adjustment...");
    let step_start = Instant::now();

    let removed_voxels: Vec<f64> = data.mask.iter()
        .zip(vasc_only.iter())
        .map(|(&m, &v)| (m - v).max(0.0))
        .collect();

    let ppm = GYRO * FIELD_TESLA / 1e6;
    let lfs_scaled: Vec<f64> = lfs_stage1.iter().map(|&v| v * ppm).collect();

    let chi_qsmart = adjust_offset(
        &removed_voxels,
        &lfs_scaled,
        &chi_stage1,
        &chi_stage2,
        nx, ny, nz,
        vsx, vsy, vsz,
        B0_DIR,
        ppm,
    );
    println!("[INFO] Offset adjustment: {:.2?}", step_start.elapsed());

    let total_elapsed = pipeline_start.elapsed();
    println!("\n[INFO] Full pipeline completed in {:.2?}", total_elapsed);

    // ========================================================================
    // Save all outputs
    // ========================================================================
    save_output(&vasc_only, data.dims, data.voxel_size, &data.affine, "full_vasc_only.nii");
    save_output(&lfs_stage1, data.dims, data.voxel_size, &data.affine, "full_lfs_sdf_1.nii");
    save_output(&lfs_stage2, data.dims, data.voxel_size, &data.affine, "full_lfs_sdf_2.nii");
    save_output(&chi_stage1, data.dims, data.voxel_size, &data.affine, "full_QSM_1.nii");
    save_output(&chi_stage2, data.dims, data.voxel_size, &data.affine, "full_QSM_2.nii");
    save_output(&removed_voxels, data.dims, data.voxel_size, &data.affine, "full_removed_voxels.nii");
    save_output(&chi_qsmart, data.dims, data.voxel_size, &data.affine, "full_QSMART.nii");

    // ========================================================================
    // Compare against all MATLAB intermediates
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("COMPARISON SUMMARY");
    println!("{}\n", "=".repeat(80));

    if let Some(ref m) = data.vasc_only_matlab {
        compare_masks("Vasculature", &vasc_only, m, &data.mask);
    }
    if let Some(ref m) = data.lfs_sdf_1_matlab {
        compare_volumes("LFS Stage 1", &lfs_stage1, m, &data.mask);
    }
    if let Some(ref m) = data.lfs_sdf_2_matlab {
        compare_volumes("LFS Stage 2", &lfs_stage2, m, &data.mask);
    }
    if let Some(ref m) = data.qsm_1_matlab {
        compare_volumes("QSM Stage 1", &chi_stage1, m, &data.mask);
    }
    if let Some(ref m) = data.qsm_2_matlab {
        compare_volumes("QSM Stage 2", &chi_stage2, m, &data.mask);
    }
    if let Some(ref m) = data.removed_voxels_matlab {
        compare_volumes("Removed voxels", &removed_voxels, m, &data.mask);
    }
    if let Some(ref m) = data.qsmart_final_matlab {
        compare_volumes("QSMART Final", &chi_qsmart, m, &data.mask);
    }
}

// ============================================================================
// Diagnostic: Dump stats about all MATLAB intermediates
// ============================================================================

#[test]
#[ignore]
fn test_qsmart_00_inspect_data() {
    println!("\n{}", "=".repeat(80));
    println!("QSMART VALIDATION: Data Inspection");
    println!("{}\n", "=".repeat(80));

    let data = QsmartTestData::load().expect("Failed to load QSMART test data");
    let (nx, ny, nz) = data.dims;
    let n_total = nx * ny * nz;
    let mask_count = data.mask.iter().filter(|&&v| v > 0.5).count();

    println!("Volume: {}x{}x{} = {} voxels", nx, ny, nz, n_total);
    println!("Voxel size: {:?} mm", data.voxel_size);
    println!("Mask: {} voxels ({:.1}%)\n", mask_count, 100.0 * mask_count as f64 / n_total as f64);

    let print_stats = |name: &str, vol: &[f64], mask: &[f64]| {
        let vals: Vec<f64> = vol.iter().zip(mask.iter())
            .filter(|(_, &m)| m > 0.5)
            .map(|(&v, _)| v)
            .collect();
        if vals.is_empty() {
            println!("{:<30} (empty)", name);
            return;
        }
        let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
        let std: f64 = (vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64).sqrt();
        let nonzero = vals.iter().filter(|&&v| v.abs() > 1e-10).count();
        println!("{:<30} min={:<12.6} max={:<12.6} mean={:<12.6} std={:<12.6} nonzero={}",
            name, min, max, mean, std, nonzero);
    };

    println!("--- Inputs ---");
    print_stats("TFS", &data.tfs, &data.mask);
    print_stats("Magnitude (mag1_sos)", &data.magnitude, &data.mask);

    println!("\n--- MATLAB Intermediates ---");
    if let Some(ref v) = data.prox_matlab { print_stats("prox", v, &data.mask); }
    if let Some(ref v) = data.curvature_matlab { print_stats("curvature", v, &data.mask); }
    if let Some(ref v) = data.alpha_matlab { print_stats("alpha", v, &data.mask); }
    if let Some(ref v) = data.enhanced_matlab { print_stats("enhanced (Frangi)", v, &data.mask); }
    if let Some(ref v) = data.vasc_only_matlab {
        print_stats("vasc_only", v, &data.mask);
        let vessel_count = v.iter().zip(data.mask.iter())
            .filter(|(&val, &m)| m > 0.5 && val < 0.5).count();
        println!("{:<30} vessel_voxels={} ({:.1}%)", "", vessel_count,
            100.0 * vessel_count as f64 / mask_count as f64);
    }
    if let Some(ref v) = data.lfs_sdf_1_matlab { print_stats("lfs_sdf_1", v, &data.mask); }
    if let Some(ref v) = data.lfs_sdf_2_matlab { print_stats("lfs_sdf_2", v, &data.mask); }
    if let Some(ref v) = data.qsm_1_matlab { print_stats("QSM_1", v, &data.mask); }
    if let Some(ref v) = data.qsm_2_matlab { print_stats("QSM_2", v, &data.mask); }
    if let Some(ref v) = data.removed_voxels_matlab { print_stats("removed_voxels", v, &data.mask); }
    if let Some(ref v) = data.combined_chi_matlab { print_stats("combined_chi", v, &data.mask); }
    if let Some(ref v) = data.qsmart_final_matlab { print_stats("QSMART_adjusted_offset", v, &data.mask); }
}

// ============================================================================
// Diagnostic: iLSQR intermediate outputs + TKD comparison
// ============================================================================

#[test]
#[ignore]
fn test_qsmart_07b_ilsqr_diagnostics() {
    println!("\n{}", "=".repeat(80));
    println!("QSMART DIAGNOSTIC: iLSQR Intermediate Outputs");
    println!("{}\n", "=".repeat(80));

    let data = QsmartTestData::load().expect("Failed to load QSMART test data");
    let (nx, ny, nz) = data.dims;
    let (vsx, vsy, vsz) = data.voxel_size;

    let matlab_lfs1 = match &data.lfs_sdf_1_matlab {
        Some(l) => l,
        None => { println!("[SKIP] No MATLAB lfs_sdf_1.nii for input"); return; }
    };
    let matlab_qsm1 = match &data.qsm_1_matlab {
        Some(q) => q,
        None => { println!("[SKIP] No MATLAB QSM_1.nii for comparison"); return; }
    };

    // Run iLSQR with full output (chi, xsa, xfs, xlsqr)
    let start = Instant::now();
    let (chi, xsa, xfs, xlsqr) = ilsqr(
        matlab_lfs1,
        &data.mask_u8,
        nx, ny, nz,
        vsx, vsy, vsz,
        B0_DIR,
        ILSQR_TOL,
        ILSQR_MAX_ITER,
    );
    let elapsed = start.elapsed();
    println!("[INFO] iLSQR completed in {:.2?}", elapsed);

    // Save all intermediates
    save_output(&xlsqr, data.dims, data.voxel_size, &data.affine, "ilsqr_step1_xlsqr.nii");
    save_output(&xfs, data.dims, data.voxel_size, &data.affine, "ilsqr_step2_xfs.nii");
    save_output(&xsa, data.dims, data.voxel_size, &data.affine, "ilsqr_step3_xsa.nii");
    save_output(&chi, data.dims, data.voxel_size, &data.affine, "ilsqr_step4_chi.nii");

    // Compare each intermediate against MATLAB QSM_1
    println!("\n--- iLSQR intermediates vs MATLAB QSM_1 ---");
    let _ = compare_volumes("Step 1 (xlsqr)  ", &xlsqr, matlab_qsm1, &data.mask);
    let _ = compare_volumes("Step 2 (xfs)    ", &xfs, matlab_qsm1, &data.mask);
    let _ = compare_volumes("Step 3 (xsa)    ", &xsa, matlab_qsm1, &data.mask);
    let _ = compare_volumes("Step 4 (chi)    ", &chi, matlab_qsm1, &data.mask);

    // Also run TKD on the same input for reference
    let start = Instant::now();
    let chi_tkd = tkd(
        matlab_lfs1,
        &data.mask_u8,
        nx, ny, nz,
        vsx, vsy, vsz,
        B0_DIR,
        0.1, // threshold
    );
    let elapsed = start.elapsed();
    println!("\n[INFO] TKD completed in {:.2?}", elapsed);
    save_output(&chi_tkd, data.dims, data.voxel_size, &data.affine, "ilsqr_tkd_reference.nii");
    let _ = compare_volumes("TKD (reference) ", &chi_tkd, matlab_qsm1, &data.mask);

    // Print per-step statistics
    println!("\n--- Per-step statistics (inside mask) ---");
    let print_masked_stats = |name: &str, v: &[f64]| {
        let vals: Vec<f64> = v.iter().zip(data.mask.iter())
            .filter(|(_, &m)| m > 0.5)
            .map(|(&x, _)| x)
            .collect();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let std = (vals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / vals.len() as f64).sqrt();
        println!("{:<25} mean={:>12.6} std={:>10.6} min={:>12.6} max={:>12.6}", name, mean, std, min, max);
    };
    print_masked_stats("Input (lfs_sdf_1)", matlab_lfs1);
    print_masked_stats("MATLAB QSM_1", matlab_qsm1);
    print_masked_stats("xlsqr (step 1)", &xlsqr);
    print_masked_stats("xfs (step 2)", &xfs);
    print_masked_stats("xsa (step 3)", &xsa);
    print_masked_stats("chi (step 4)", &chi);
    print_masked_stats("TKD", &chi_tkd);
}

// ============================================================================
// Stage 4b: Frangi vesselness debug — intermediate outputs & statistics
// ============================================================================

#[test]
#[ignore]
fn test_qsmart_04b_frangi_debug() {
    println!("\n{}", "=".repeat(80));
    println!("QSMART DEBUG: Stage 4b — Frangi Vesselness Internals");
    println!("{}\n", "=".repeat(80));

    let data = QsmartTestData::load().expect("Failed to load QSMART test data");
    let (nx, ny, nz) = data.dims;
    let n_total = nx * ny * nz;

    // ========================================================================
    // Helper: print statistics of a volume within mask
    // ========================================================================
    let print_stats = |name: &str, vol: &[f64], mask: &[u8]| {
        let vals: Vec<f64> = vol.iter().zip(mask.iter())
            .filter(|(_, &m)| m > 0)
            .map(|(&v, _)| v)
            .collect();
        if vals.is_empty() {
            println!("[STATS] {:<35} (empty within mask)", name);
            return;
        }
        let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let std = (vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64).sqrt();
        let nonzero = vals.iter().filter(|&&v| v.abs() > 1e-10).count();
        let count = vals.len();
        println!("[STATS] {:<35} min={:<14.8} max={:<14.8} mean={:<14.8} std={:<14.8} nonzero={}/{}",
            name, min, max, mean, std, nonzero, count);
    };

    // ========================================================================
    // Step 1: Bottom-hat morphological filter (replicated from vasculature.rs)
    // Using a simple but correct implementation since the optimized one is private
    // ========================================================================
    println!("\n--- Step 1: Bottom-hat morphological filter ---");
    let step_start = Instant::now();

    let bottom_hat = morphological_bottom_hat_simple(
        &data.magnitude, nx, ny, nz, VASC_SPHERE_RADIUS,
    );

    println!("[INFO] Bottom-hat computed in {:.2?}", step_start.elapsed());
    print_stats("magnitude (input)", &data.magnitude, &data.mask_u8);
    print_stats("bottom_hat (raw)", &bottom_hat, &data.mask_u8);

    // ========================================================================
    // Step 2: Apply mask to bottom-hat
    // ========================================================================
    println!("\n--- Step 2: Masked bottom-hat ---");
    let masked_bottom_hat: Vec<f64> = bottom_hat.iter()
        .zip(data.mask_u8.iter())
        .map(|(&v, &m)| if m != 0 { v } else { 0.0 })
        .collect();

    print_stats("masked_bottom_hat", &masked_bottom_hat, &data.mask_u8);

    // Save masked bottom-hat
    save_output(&masked_bottom_hat, data.dims, data.voxel_size, &data.affine, "bottom_hat_rust.nii");

    // ========================================================================
    // Step 3: Frangi vesselness filter
    // ========================================================================
    println!("\n--- Step 3: Frangi vesselness filter ---");
    println!("[INFO] Parameters: scale_range={:?}, scale_ratio={}, c={}",
        FRANGI_SCALE_RANGE, FRANGI_SCALE_RATIO, FRANGI_C);

    // Enumerate sigma values
    let mut sigmas = Vec::new();
    let mut sigma = FRANGI_SCALE_RANGE[0];
    while sigma <= FRANGI_SCALE_RANGE[1] {
        sigmas.push(sigma);
        sigma += FRANGI_SCALE_RATIO;
    }
    if sigmas.is_empty() {
        sigmas.push(FRANGI_SCALE_RANGE[0]);
    }
    println!("[INFO] Sigma values ({} scales): {:?}", sigmas.len(), sigmas);

    // Print analysis of each sigma
    for &s in &sigmas {
        let kernel_radius = (3.0 * s).ceil() as usize;
        let kernel_size = 2 * kernel_radius + 1;
        let scale_factor = s * s;
        println!("[INFO]   sigma={:.4}: kernel_size={}, scale_factor(sigma^2)={:.8}",
            s, kernel_size, scale_factor);
    }

    // Print what the c parameter does
    let c2 = 2.0 * FRANGI_C * FRANGI_C;
    println!("[INFO] c={}, c^2={}, 2c^2={}", FRANGI_C, FRANGI_C * FRANGI_C, c2);
    println!("[INFO] For S^2 term: 1 - exp(-S^2 / 2c^2)");
    println!("[INFO]   If S=1:    1 - exp(-1/{}) = {:.8}", c2, 1.0 - (-1.0 / c2).exp());
    println!("[INFO]   If S=10:   1 - exp(-100/{}) = {:.8}", c2, 1.0 - (-100.0 / c2).exp());
    println!("[INFO]   If S=100:  1 - exp(-10000/{}) = {:.8}", c2, 1.0 - (-10000.0 / c2).exp());
    println!("[INFO]   If S=500:  1 - exp(-250000/{}) = {:.8}", c2, 1.0 - (-250000.0 / c2).exp());
    println!("[INFO]   If S=1000: 1 - exp(-1000000/{}) = {:.8}", c2, 1.0 - (-1000000.0 / c2).exp());

    let step_start = Instant::now();

    let frangi_params = qsm_core::utils::frangi::FrangiParams {
        scale_range: FRANGI_SCALE_RANGE,
        scale_ratio: FRANGI_SCALE_RATIO,
        alpha: 0.5,
        beta: 0.5,
        c: FRANGI_C,
        black_white: false,
    };

    let frangi_result = qsm_core::utils::frangi::frangi_filter_3d(
        &masked_bottom_hat, nx, ny, nz, &frangi_params,
    );
    let enhanced = &frangi_result.vesselness;

    println!("[INFO] Frangi filter computed in {:.2?}", step_start.elapsed());
    print_stats("frangi_vesselness (Rust)", enhanced, &data.mask_u8);

    // Save Frangi vesselness
    save_output(enhanced, data.dims, data.voxel_size, &data.affine, "frangi_vesselness_rust.nii");

    // Histogram of non-zero vesselness values within mask
    let nonzero_vessel: Vec<f64> = enhanced.iter()
        .zip(data.mask_u8.iter())
        .filter(|(&v, &m)| m > 0 && v > 1e-20)
        .map(|(&v, _)| v)
        .collect();
    println!("[INFO] Non-zero vesselness voxels within mask: {} / {}",
        nonzero_vessel.len(),
        data.mask_u8.iter().filter(|&&m| m > 0).count());
    if !nonzero_vessel.is_empty() {
        let min = nonzero_vessel.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = nonzero_vessel.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = nonzero_vessel.iter().sum::<f64>() / nonzero_vessel.len() as f64;
        println!("[INFO] Non-zero vesselness: min={:.10}, max={:.10}, mean={:.10}", min, max, mean);

        // Percentile distribution
        let mut sorted = nonzero_vessel.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pcts = [10, 25, 50, 75, 90, 95, 99];
        for &p in &pcts {
            let idx = (sorted.len() * p / 100).min(sorted.len() - 1);
            println!("[INFO]   P{:02}: {:.10}", p, sorted[idx]);
        }
    }

    // ========================================================================
    // Step 4: Otsu thresholding
    // ========================================================================
    println!("\n--- Step 4: Otsu thresholding ---");
    let threshold = qsm_core::utils::threshold::otsu_threshold(enhanced, 256);
    println!("[INFO] Otsu threshold = {:.10}", threshold);

    let above_threshold = enhanced.iter()
        .zip(data.mask_u8.iter())
        .filter(|(&v, &m)| m > 0 && v > threshold)
        .count();
    let mask_count = data.mask_u8.iter().filter(|&&m| m > 0).count();
    println!("[INFO] Voxels above threshold (within mask): {} / {} ({:.2}%)",
        above_threshold, mask_count, 100.0 * above_threshold as f64 / mask_count as f64);

    // ========================================================================
    // Step 5: Compare with MATLAB enhanced.nii
    // ========================================================================
    if let Some(ref matlab_enhanced) = data.enhanced_matlab {
        println!("\n--- Step 5: MATLAB enhanced.nii comparison ---");
        print_stats("MATLAB enhanced.nii", matlab_enhanced, &data.mask_u8);

        // Histogram of non-zero MATLAB vesselness
        let matlab_nonzero: Vec<f64> = matlab_enhanced.iter()
            .zip(data.mask_u8.iter())
            .filter(|(&v, &m)| m > 0 && v > 1e-20)
            .map(|(&v, _)| v)
            .collect();
        println!("[INFO] MATLAB non-zero vesselness voxels: {} / {}",
            matlab_nonzero.len(), mask_count);
        if !matlab_nonzero.is_empty() {
            let min = matlab_nonzero.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = matlab_nonzero.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mean = matlab_nonzero.iter().sum::<f64>() / matlab_nonzero.len() as f64;
            println!("[INFO] MATLAB non-zero vesselness: min={:.10}, max={:.10}, mean={:.10}", min, max, mean);

            let mut sorted = matlab_nonzero.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pcts = [10, 25, 50, 75, 90, 95, 99];
            for &p in &pcts {
                let idx = (sorted.len() * p / 100).min(sorted.len() - 1);
                println!("[INFO]   P{:02}: {:.10}", p, sorted[idx]);
            }
        }

        // MATLAB Otsu threshold
        let matlab_threshold = qsm_core::utils::threshold::otsu_threshold(matlab_enhanced, 256);
        println!("[INFO] MATLAB enhanced Otsu threshold = {:.10}", matlab_threshold);

        let matlab_above = matlab_enhanced.iter()
            .zip(data.mask_u8.iter())
            .filter(|(&v, &m)| m > 0 && v > matlab_threshold)
            .count();
        println!("[INFO] MATLAB voxels above threshold: {} / {} ({:.2}%)",
            matlab_above, mask_count, 100.0 * matlab_above as f64 / mask_count as f64);

        // Direct correlation between Rust and MATLAB vesselness
        let comp = compare_volumes("Frangi Rust vs MATLAB", enhanced, matlab_enhanced, &data.mask);
        println!("\n[RESULT] Frangi vesselness correlation: r={:.4}", comp.correlation);

        // Ratio analysis: where MATLAB is strong, what is Rust?
        println!("\n--- Ratio analysis (where MATLAB enhanced > 0.01) ---");
        let mut ratio_count = 0;
        let mut ratio_sum = 0.0;
        let mut ratio_min = f64::INFINITY;
        let mut ratio_max = f64::NEG_INFINITY;
        let mut rust_zero_where_matlab_nonzero = 0;
        for i in 0..n_total {
            if data.mask_u8[i] > 0 && matlab_enhanced[i] > 0.01 {
                if enhanced[i] > 1e-20 {
                    let ratio = enhanced[i] / matlab_enhanced[i];
                    ratio_sum += ratio;
                    ratio_count += 1;
                    ratio_min = ratio_min.min(ratio);
                    ratio_max = ratio_max.max(ratio);
                } else {
                    rust_zero_where_matlab_nonzero += 1;
                }
            }
        }
        if ratio_count > 0 {
            println!("[INFO] Rust/MATLAB ratio: min={:.8}, max={:.8}, mean={:.8} (n={})",
                ratio_min, ratio_max, ratio_sum / ratio_count as f64, ratio_count);
        }
        println!("[INFO] Rust=0 where MATLAB>0.01: {}", rust_zero_where_matlab_nonzero);
    } else {
        println!("\n[SKIP] No MATLAB enhanced.nii available for comparison");
    }

    // ========================================================================
    // Step 6: Diagnostic — what happens with different c values
    // ========================================================================
    println!("\n--- Step 6: Sensitivity to c parameter ---");
    for &test_c in &[1.0, 10.0, 50.0, 100.0, 500.0] {
        let test_params = qsm_core::utils::frangi::FrangiParams {
            scale_range: FRANGI_SCALE_RANGE,
            scale_ratio: FRANGI_SCALE_RATIO,
            alpha: 0.5,
            beta: 0.5,
            c: test_c,
            black_white: false,
        };

        let test_result = qsm_core::utils::frangi::frangi_filter_3d(
            &masked_bottom_hat, nx, ny, nz, &test_params,
        );

        let nz_count = test_result.vesselness.iter()
            .zip(data.mask_u8.iter())
            .filter(|(&v, &m)| m > 0 && v > 1e-20)
            .count();
        let max_v = test_result.vesselness.iter()
            .zip(data.mask_u8.iter())
            .filter(|(_, &m)| m > 0)
            .map(|(&v, _)| v)
            .fold(0.0_f64, f64::max);
        let mean_v: f64 = {
            let vals: Vec<f64> = test_result.vesselness.iter()
                .zip(data.mask_u8.iter())
                .filter(|(_, &m)| m > 0)
                .map(|(&v, _)| v)
                .collect();
            vals.iter().sum::<f64>() / vals.len() as f64
        };

        let test_thresh = qsm_core::utils::threshold::otsu_threshold(&test_result.vesselness, 256);
        let above = test_result.vesselness.iter()
            .zip(data.mask_u8.iter())
            .filter(|(&v, &m)| m > 0 && v > test_thresh)
            .count();

        println!("[INFO] c={:<6} max={:<14.8} mean={:<14.8} nonzero={:<8} otsu={:<14.8} above_otsu={}({:.1}%)",
            test_c, max_v, mean_v, nz_count, test_thresh, above, 100.0 * above as f64 / mask_count as f64);
    }

    // ========================================================================
    // Step 7: Noise floor fraction sweep
    // ========================================================================
    if let Some(ref matlab_vasc) = data.vasc_only_matlab {
        println!("\n--- Step 7: Noise floor fraction sweep ---");
        let max_v = enhanced.iter()
            .zip(data.mask_u8.iter())
            .filter(|(_, &m)| m > 0)
            .map(|(&v, _)| v)
            .fold(0.0f64, f64::max);

        let matlab_vessels: usize = matlab_vasc.iter()
            .zip(data.mask.iter())
            .filter(|(&v, &m)| m > 0.5 && v < 0.5)
            .count();
        println!("[INFO] MATLAB vessel count: {}", matlab_vessels);
        println!("[INFO] {:>10} {:>10} {:>10} {:>6} {:>6} {:>8}", "fraction", "threshold", "vessels", "TP", "FP+FN", "Dice");

        for &frac in &[0.0, 5e-4, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3, 5e-3, 1e-2] {
            let noise_floor = max_v * frac;
            let mut tp = 0usize;
            let mut fp = 0usize;
            let mut fn_ = 0usize;
            let mut tn = 0usize;
            let mut rust_vessels = 0usize;
            for i in 0..n_total {
                if data.mask_u8[i] == 0 { continue; }
                let rust_is_vessel = enhanced[i] > noise_floor;
                let matlab_is_vessel = matlab_vasc[i] < 0.5;
                if rust_is_vessel { rust_vessels += 1; }
                match (rust_is_vessel, matlab_is_vessel) {
                    (false, false) => tp += 1, // both tissue
                    (false, true) => fp += 1,  // Rust=tissue, MATLAB=vessel
                    (true, false) => fn_ += 1, // Rust=vessel, MATLAB=tissue
                    (true, true) => tn += 1,   // both vessel
                }
            }
            let dice = 2.0 * tp as f64 / (2.0 * tp as f64 + fp as f64 + fn_ as f64);
            println!("[INFO] {:>10.1e} {:>10.2e} {:>10} {:>6} {:>6} {:>8.4}",
                frac, noise_floor, rust_vessels, tn, fp + fn_, dice);
        }
    }

    println!("\n[INFO] Debug outputs saved to {}/", OUTPUT_DIR);
    println!("[INFO] Files: bottom_hat_rust.nii, frangi_vesselness_rust.nii");
}

// ============================================================================
// Simple morphological bottom-hat (for debug test — does not use private APIs)
// ============================================================================

/// Simple grayscale dilation with spherical structuring element
fn dilate_grayscale_simple(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    radius: i32,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut result = vec![f64::NEG_INFINITY; n_total];
    let r2 = (radius * radius) as f64;
    let stride_z = nx * ny;

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = i + j * nx + k * stride_z;
                let mut max_val = f64::NEG_INFINITY;

                for dk in -radius..=radius {
                    let nk = k as i32 + dk;
                    if nk < 0 || nk >= nz as i32 { continue; }
                    for dj in -radius..=radius {
                        let nj = j as i32 + dj;
                        if nj < 0 || nj >= ny as i32 { continue; }
                        for di in -radius..=radius {
                            let ni = i as i32 + di;
                            if ni < 0 || ni >= nx as i32 { continue; }
                            let dist2 = (di * di + dj * dj + dk * dk) as f64;
                            if dist2 <= r2 {
                                let nidx = ni as usize + nj as usize * nx + nk as usize * stride_z;
                                max_val = max_val.max(data[nidx]);
                            }
                        }
                    }
                }

                result[idx] = if max_val.is_finite() { max_val } else { data[idx] };
            }
        }
    }
    result
}

/// Simple grayscale erosion with spherical structuring element
fn erode_grayscale_simple(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    radius: i32,
) -> Vec<f64> {
    let n_total = nx * ny * nz;
    let mut result = vec![f64::INFINITY; n_total];
    let r2 = (radius * radius) as f64;
    let stride_z = nx * ny;

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = i + j * nx + k * stride_z;
                let mut min_val = f64::INFINITY;

                for dk in -radius..=radius {
                    let nk = k as i32 + dk;
                    if nk < 0 || nk >= nz as i32 { continue; }
                    for dj in -radius..=radius {
                        let nj = j as i32 + dj;
                        if nj < 0 || nj >= ny as i32 { continue; }
                        for di in -radius..=radius {
                            let ni = i as i32 + di;
                            if ni < 0 || ni >= nx as i32 { continue; }
                            let dist2 = (di * di + dj * dj + dk * dk) as f64;
                            if dist2 <= r2 {
                                let nidx = ni as usize + nj as usize * nx + nk as usize * stride_z;
                                min_val = min_val.min(data[nidx]);
                            }
                        }
                    }
                }

                result[idx] = if min_val.is_finite() { min_val } else { data[idx] };
            }
        }
    }
    result
}

/// Simple morphological bottom-hat = closing(data) - data
fn morphological_bottom_hat_simple(
    data: &[f64],
    nx: usize, ny: usize, nz: usize,
    radius: i32,
) -> Vec<f64> {
    let dilated = dilate_grayscale_simple(data, nx, ny, nz, radius);
    let closed = erode_grayscale_simple(&dilated, nx, ny, nz, radius);
    closed.iter()
        .zip(data.iter())
        .map(|(&c, &d)| (c - d).max(0.0))
        .collect()
}

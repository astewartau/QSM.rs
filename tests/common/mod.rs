//! Common test utilities for QSM-core integration tests

use std::fs;
use std::path::Path;
use serde::Deserialize;

/// BIDS JSON sidecar fields
#[derive(Deserialize)]
struct BidsSidecar {
    #[serde(rename = "EchoTime")]
    echo_time: f64,
    #[serde(rename = "MagneticFieldStrength")]
    magnetic_field_strength: Option<f64>,
}

/// Compute RMSE between two arrays, only within mask (non-zero values)
pub fn rmse(a: &[f64], b: &[f64], mask: &[u8]) -> f64 {
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    for i in 0..a.len() {
        if mask[i] > 0 {
            let diff = a[i] - b[i];
            sum_sq += diff * diff;
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    (sum_sq / count as f64).sqrt()
}

/// Compute NRMSE (normalized by range of ground truth within mask)
pub fn nrmse(a: &[f64], b: &[f64], mask: &[u8]) -> f64 {
    let rmse_val = rmse(a, b, mask);

    let mut min_b = f64::INFINITY;
    let mut max_b = f64::NEG_INFINITY;
    for i in 0..b.len() {
        if mask[i] > 0 {
            if b[i] < min_b { min_b = b[i]; }
            if b[i] > max_b { max_b = b[i]; }
        }
    }

    let range = max_b - min_b;
    if range == 0.0 {
        return 0.0;
    }
    rmse_val / range
}

/// Compute Pearson correlation coefficient within mask
pub fn correlation(a: &[f64], b: &[f64], mask: &[u8]) -> f64 {
    let mut sum_a = 0.0;
    let mut sum_b = 0.0;
    let mut sum_ab = 0.0;
    let mut sum_a2 = 0.0;
    let mut sum_b2 = 0.0;
    let mut n = 0usize;

    for i in 0..a.len() {
        if mask[i] > 0 {
            sum_a += a[i];
            sum_b += b[i];
            sum_ab += a[i] * b[i];
            sum_a2 += a[i] * a[i];
            sum_b2 += b[i] * b[i];
            n += 1;
        }
    }

    if n == 0 {
        return 0.0;
    }

    let n = n as f64;
    let numerator = n * sum_ab - sum_a * sum_b;
    let denominator = ((n * sum_a2 - sum_a * sum_a) * (n * sum_b2 - sum_b * sum_b)).sqrt();

    if denominator == 0.0 {
        return 0.0;
    }

    numerator / denominator
}

/// Compute XSIM (SSIM optimized for QSM) between two 3D volumes within mask.
///
/// Uses the same formula as SSIM but with QSM-optimized parameters:
///   L = 1.0 (native ppm range), K1 = 0.01, K2 = 0.001
/// Local statistics computed in 5×5×5 uniform windows.
///
/// Reference: Milovic et al., "XSIM: A structural similarity index measure
/// optimized for MRI QSM", Magn Reson Med. 2025;93:411-421.
pub fn xsim(a: &[f64], b: &[f64], mask: &[u8], dims: (usize, usize, usize)) -> f64 {
    let (nx, ny, nz) = dims;
    // XSIM parameters: L=1, K1=0.01, K2=0.001
    let c1: f64 = 1e-4;  // (K1 * L)² = (0.01)²
    let c2: f64 = 1e-6;  // (K2 * L)² = (0.001)²
    let half_w: usize = 2; // 5×5×5 window

    let mut sum_xsim = 0.0;
    let mut count = 0usize;

    for k in 0..nz {
        let k_lo = k.saturating_sub(half_w);
        let k_hi = (k + half_w + 1).min(nz);
        for j in 0..ny {
            let j_lo = j.saturating_sub(half_w);
            let j_hi = (j + half_w + 1).min(ny);
            for i in 0..nx {
                let idx = i + j * nx + k * nx * ny;
                if mask[idx] == 0 {
                    continue;
                }

                let i_lo = i.saturating_sub(half_w);
                let i_hi = (i + half_w + 1).min(nx);

                let mut sa = 0.0;
                let mut sb = 0.0;
                let mut sa2 = 0.0;
                let mut sb2 = 0.0;
                let mut sab = 0.0;
                let mut n = 0usize;

                for kk in k_lo..k_hi {
                    for jj in j_lo..j_hi {
                        for ii in i_lo..i_hi {
                            let nidx = ii + jj * nx + kk * nx * ny;
                            let va = a[nidx];
                            let vb = b[nidx];
                            sa += va;
                            sb += vb;
                            sa2 += va * va;
                            sb2 += vb * vb;
                            sab += va * vb;
                            n += 1;
                        }
                    }
                }

                let nf = n as f64;
                let mu_a = sa / nf;
                let mu_b = sb / nf;
                let var_a = sa2 / nf - mu_a * mu_a;
                let var_b = sb2 / nf - mu_b * mu_b;
                let cov_ab = sab / nf - mu_a * mu_b;

                let num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov_ab + c2);
                let den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2);

                if den > 0.0 {
                    sum_xsim += num / den;
                    count += 1;
                }
            }
        }
    }

    if count == 0 { 0.0 } else { sum_xsim / count as f64 }
}

/// Load a NIfTI file from disk and return the data
pub fn load_nifti_file(path: &str) -> Result<qsm_core::nifti_io::NiftiData, String> {
    let bytes = fs::read(path)
        .map_err(|e| format!("Failed to read file {}: {}", path, e))?;
    qsm_core::nifti_io::load_nifti(&bytes)
}

/// Test data structure holding all inputs and ground truth
pub struct TestData {
    /// Phase images for each echo (wrapped, radians)
    pub phase_echoes: Vec<Vec<f64>>,
    /// Magnitude images for each echo
    pub mag_echoes: Vec<Vec<f64>>,
    /// Brain mask
    pub mask: Vec<u8>,
    /// Ground truth B0 field map (ppm)
    pub fieldmap: Vec<f64>,
    /// Ground truth local field after background removal (ppm)
    pub fieldmap_local: Vec<f64>,
    /// Ground truth susceptibility map (ppm)
    pub chi: Vec<f64>,
    /// Volume dimensions (nx, ny, nz)
    pub dims: (usize, usize, usize),
    /// Voxel sizes in mm
    pub voxel_size: (f64, f64, f64),
    /// B0 direction
    pub b0_dir: (f64, f64, f64),
    /// Echo times in seconds
    pub echo_times: Vec<f64>,
    /// Field strength in Tesla
    pub field_strength: f64,
    /// Tissue segmentation labels (from dseg.nii)
    pub segmentation: Vec<u8>,
}

impl TestData {
    /// Load test data from the bids directory
    pub fn load() -> Result<Self, String> {
        let base = "bids/sub-1/anat";
        let deriv = "bids/derivatives/qsm-forward/sub-1/anat";

        // Check if test data exists
        if !Path::new(base).exists() {
            return Err(format!(
                "Test data not found at '{}'. Please ensure BIDS test data is available.",
                base
            ));
        }

        // Load phase echoes
        let mut phase_echoes = Vec::new();
        for e in 1..=4 {
            let path = format!("{}/sub-1_echo-{}_part-phase_MEGRE.nii", base, e);
            let nifti = load_nifti_file(&path)?;
            phase_echoes.push(nifti.data);
        }

        // Load magnitude echoes
        let mut mag_echoes = Vec::new();
        for e in 1..=4 {
            let path = format!("{}/sub-1_echo-{}_part-mag_MEGRE.nii", base, e);
            let nifti = load_nifti_file(&path)?;
            mag_echoes.push(nifti.data);
        }

        // Load mask
        let mask_nifti = load_nifti_file(&format!("{}/sub-1_mask.nii", deriv))?;
        let mask: Vec<u8> = mask_nifti.data.iter()
            .map(|&v| if v > 0.5 { 1 } else { 0 })
            .collect();
        let dims = mask_nifti.dims;
        let voxel_size = mask_nifti.voxel_size;

        // Load ground truth field maps
        let fieldmap = load_nifti_file(&format!("{}/sub-1_fieldmap.nii", deriv))?.data;
        let fieldmap_local = load_nifti_file(&format!("{}/sub-1_fieldmap-local.nii", deriv))?.data;
        let chi = load_nifti_file(&format!("{}/sub-1_Chimap.nii", deriv))?.data;

        // Load tissue segmentation
        let seg_path = format!("{}/sub-1_dseg.nii", deriv);
        let segmentation: Vec<u8> = if Path::new(&seg_path).exists() {
            let seg_nifti = load_nifti_file(&seg_path)?;
            seg_nifti.data.iter().map(|&v| v.round() as u8).collect()
        } else {
            vec![0u8; mask.len()]
        };

        // Parse echo times and field strength from BIDS JSON sidecars
        let mut echo_times = Vec::new();
        let mut field_strength = 7.0_f64;
        for e in 1..=4 {
            let json_path = format!("{}/sub-1_echo-{}_part-phase_MEGRE.json", base, e);
            let json_str = fs::read_to_string(&json_path)
                .map_err(|err| format!("Failed to read {}: {}", json_path, err))?;
            let sidecar: BidsSidecar = serde_json::from_str(&json_str)
                .map_err(|err| format!("Failed to parse {}: {}", json_path, err))?;
            echo_times.push(sidecar.echo_time);
            if let Some(fs_val) = sidecar.magnetic_field_strength {
                field_strength = fs_val;
            }
        }

        Ok(TestData {
            phase_echoes,
            mag_echoes,
            mask,
            fieldmap,
            fieldmap_local,
            chi,
            dims,
            voxel_size,
            b0_dir: (0.0, 0.0, 1.0),
            echo_times,
            field_strength,
            segmentation,
        })
    }
}

/// Result of running an algorithm test
#[derive(Debug)]
pub struct TestResult {
    pub name: String,
    pub rmse: f64,
    pub nrmse: f64,
    pub correlation: f64,
    pub xsim: f64,
}

impl TestResult {
    pub fn new(
        name: &str,
        output: &[f64],
        ground_truth: &[f64],
        mask: &[u8],
        dims: (usize, usize, usize),
    ) -> Self {
        TestResult {
            name: name.to_string(),
            rmse: rmse(output, ground_truth, mask),
            nrmse: nrmse(output, ground_truth, mask),
            correlation: correlation(output, ground_truth, mask),
            xsim: xsim(output, ground_truth, mask, dims),
        }
    }

    pub fn print(&self) {
        println!("{:<15} RMSE={:.6}  NRMSE={:.4}  r={:.4}  XSIM={:.4}",
            self.name, self.rmse, self.nrmse, self.correlation, self.xsim);
    }

    pub fn print_with_time(&self, elapsed: std::time::Duration) {
        println!("{:<15} {:>12.6} {:>10.4} {:>10.4} {:>10.4} {:>10.2?}",
            self.name, self.rmse, self.nrmse, self.correlation, self.xsim, elapsed);
    }

    /// Print machine-readable CSV line for CI metric collection
    pub fn print_ci_metrics(&self, elapsed: std::time::Duration) {
        println!("RESULT:{},{:.6},{:.4},{:.4},{:.4},{:.2}",
            self.name, self.rmse, self.nrmse, self.correlation, self.xsim, elapsed.as_secs_f64());
    }
}

/// Save center orthogonal slices of a 3D volume for CI visualization.
///
/// Writes a compact binary file containing the center axial, coronal, and
/// sagittal slices of both the result volume and mask. The Python script
/// `scripts/render_slices.py` reads these to produce matplotlib figures.
///
/// Binary format (all little-endian):
///   nx: u64, ny: u64, nz: u64
///   axial result:    f64 * (nx * ny)   -- z = nz/2
///   coronal result:  f64 * (nx * nz)   -- y = ny/2
///   sagittal result: f64 * (ny * nz)   -- x = nx/2
///   axial mask:      u8  * (nx * ny)
///   coronal mask:    u8  * (nx * nz)
///   sagittal mask:   u8  * (ny * nz)
pub fn save_center_slices(
    result: &[f64],
    mask: &[u8],
    dims: (usize, usize, usize),
    slug: &str,
) {
    let (nx, ny, nz) = dims;
    let dir = "slices";
    fs::create_dir_all(dir).ok();

    let mut buf: Vec<u8> = Vec::new();

    // Header
    buf.extend_from_slice(&(nx as u64).to_le_bytes());
    buf.extend_from_slice(&(ny as u64).to_le_bytes());
    buf.extend_from_slice(&(nz as u64).to_le_bytes());

    // Axial slice at z = nz/2, stored row-major (ny rows of nx pixels)
    let z_mid = nz / 2;
    for y in 0..ny {
        for x in 0..nx {
            buf.extend_from_slice(&result[x + y * nx + z_mid * nx * ny].to_le_bytes());
        }
    }

    // Coronal slice at y = ny/2, stored row-major (nz rows of nx pixels)
    let y_mid = ny / 2;
    for z in 0..nz {
        for x in 0..nx {
            buf.extend_from_slice(&result[x + y_mid * nx + z * nx * ny].to_le_bytes());
        }
    }

    // Sagittal slice at x = nx/2, stored row-major (nz rows of ny pixels)
    let x_mid = nx / 2;
    for z in 0..nz {
        for y in 0..ny {
            buf.extend_from_slice(&result[x_mid + y * nx + z * nx * ny].to_le_bytes());
        }
    }

    // Mask slices (same layout)
    for y in 0..ny {
        for x in 0..nx {
            buf.push(mask[x + y * nx + z_mid * nx * ny]);
        }
    }
    for z in 0..nz {
        for x in 0..nx {
            buf.push(mask[x + y_mid * nx + z * nx * ny]);
        }
    }
    for z in 0..nz {
        for y in 0..ny {
            buf.push(mask[x_mid + y * nx + z * nx * ny]);
        }
    }

    let path = format!("{}/{}.bin", dir, slug);
    fs::write(&path, &buf).unwrap_or_else(|e| {
        eprintln!("[WARN] Failed to save slices to {}: {}", path, e);
    });
    println!("[INFO] Saved center slices to {}", path);
}

/// Compute Dice coefficient between two binary masks
pub fn dice_coefficient(predicted: &[u8], ground_truth: &[u8]) -> f64 {
    let (mut tp, mut p_sum, mut gt_sum) = (0usize, 0usize, 0usize);
    for i in 0..predicted.len() {
        let p = (predicted[i] > 0) as usize;
        let g = (ground_truth[i] > 0) as usize;
        tp += p & g;
        p_sum += p;
        gt_sum += g;
    }
    if p_sum + gt_sum == 0 {
        1.0
    } else {
        2.0 * tp as f64 / (p_sum + gt_sum) as f64
    }
}

// ============================================================================
// Challenge Evaluation Metrics
// ============================================================================
//
// Ported from the QSM Reconstruction Challenge MATLAB evaluation code.
// Reference: challenge/functions4challenge/evaluation/
//
// Tissue segmentation labels:
//   1: Caudate, 2: Globus pallidus, 3: Putamen, 4: Red nucleus,
//   5: Dentate nucleus, 6: Substantia nigra & STN, 7: Thalamus,
//   8: White matter, 9: Gray matter, 10: CSF, 11: Blood,
//   12: Fat, 13: Bone, 14: Air, 15: Muscle, 16: Calcification

/// Linear least-squares fit: y = slope*x + intercept.
/// Returns (slope, intercept).
fn linear_fit(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y).map(|(xi, yi)| xi * yi).sum();
    let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-30 {
        return (0.0, 0.0);
    }
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    (slope, intercept)
}

/// Challenge-style NRMSE with demeaning and optional linear detrending.
///
/// Both the reconstruction and ground truth are demeaned within the mask,
/// then NRMSE is computed as `100 * ||recon - truth|| / ||truth||`.
/// The detrended variant additionally fits a linear model to remove
/// systematic scaling bias (common in regularized QSM methods).
///
/// Returns (nrmse_percent, nrmse_detrend_percent).
///
/// Reference: compute_rmse_detrend_v0.m
pub fn nrmse_challenge(a: &[f64], b: &[f64], mask: &[u8]) -> (f64, f64) {
    let mut recon_vals = Vec::new();
    let mut truth_vals = Vec::new();
    for i in 0..a.len() {
        if mask[i] > 0 {
            recon_vals.push(a[i]);
            truth_vals.push(b[i]);
        }
    }
    if recon_vals.is_empty() {
        return (f64::NAN, f64::NAN);
    }

    // Demean both within mask
    let mean_recon: f64 = recon_vals.iter().sum::<f64>() / recon_vals.len() as f64;
    let mean_truth: f64 = truth_vals.iter().sum::<f64>() / truth_vals.len() as f64;
    for v in recon_vals.iter_mut() { *v -= mean_recon; }
    for v in truth_vals.iter_mut() { *v -= mean_truth; }

    let norm_diff: f64 = recon_vals.iter().zip(&truth_vals)
        .map(|(r, t)| (r - t).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm_truth: f64 = truth_vals.iter().map(|t| t.powi(2)).sum::<f64>().sqrt();

    if norm_truth < 1e-30 {
        return (f64::NAN, f64::NAN);
    }

    let nrmse = 100.0 * norm_diff / norm_truth;

    // Linear detrending: fit chi_recon = slope * chi_true + intercept
    // Then invert to get corrected estimate of chi_true
    let (slope, intercept) = linear_fit(&truth_vals, &recon_vals);

    if slope.abs() < 1e-30 {
        return (nrmse, nrmse);
    }

    // Inverted: chi_true_est = (1/slope) * chi_recon + (-intercept/slope)
    let inv_slope = 1.0 / slope;
    let inv_intercept = -intercept / slope;
    let norm_diff_detrend: f64 = recon_vals.iter().zip(&truth_vals)
        .map(|(r, t)| {
            let corrected = inv_slope * r + inv_intercept;
            (corrected - t).powi(2)
        })
        .sum::<f64>()
        .sqrt();

    let nrmse_detrend = 100.0 * norm_diff_detrend / norm_truth;
    (nrmse, nrmse_detrend)
}

/// Compute the deep gray matter linearity metric.
///
/// For each of the 6 DGM regions (labels 1–6: Caudate, Globus pallidus,
/// Putamen, Red nucleus, Dentate nucleus, Substantia nigra & STN), compute
/// mean susceptibility in both reconstruction and ground truth. Fit a line
/// through these 6 points. Returns |1 - slope|, which is 0 for perfect
/// linear preservation.
///
/// Reference: compute_linearityDeepGM.m
pub fn dgm_linearity(recon: &[f64], truth: &[f64], seg: &[u8]) -> f64 {
    let mut truth_means = Vec::new();
    let mut recon_means = Vec::new();

    for label in 1..=6u8 {
        let mut sum_truth = 0.0;
        let mut sum_recon = 0.0;
        let mut count = 0usize;

        for i in 0..seg.len() {
            if seg[i] == label {
                sum_truth += truth[i];
                sum_recon += recon[i];
                count += 1;
            }
        }

        if count > 0 {
            truth_means.push(sum_truth / count as f64);
            recon_means.push(sum_recon / count as f64);
        }
    }

    if truth_means.len() < 2 {
        return f64::NAN;
    }

    let (slope, _) = linear_fit(&truth_means, &recon_means);
    (1.0 - slope).abs()
}

/// 3D binary dilation using a 3x3x3 cube structuring element (26-connected).
///
/// Reference: dilatemask.m (uses `ones(3,3,3)` kernel)
pub fn dilate_mask_3d(mask: &[u8], dims: (usize, usize, usize)) -> Vec<u8> {
    let (nx, ny, nz) = dims;
    let mut result = vec![0u8; mask.len()];

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if mask[i + j * nx + k * nx * ny] > 0 {
                    let k_lo = k.saturating_sub(1);
                    let k_hi = (k + 2).min(nz);
                    let j_lo = j.saturating_sub(1);
                    let j_hi = (j + 2).min(ny);
                    let i_lo = i.saturating_sub(1);
                    let i_hi = (i + 2).min(nx);

                    for kk in k_lo..k_hi {
                        for jj in j_lo..j_hi {
                            for ii in i_lo..i_hi {
                                result[ii + jj * nx + kk * nx * ny] = 1;
                            }
                        }
                    }
                }
            }
        }
    }
    result
}

/// Compute calcification metrics following the challenge protocol.
///
/// 1. Calcification moment: `volume * mean_susceptibility` within the
///    adaptively-thresholded calcification region. The deviation from
///    the ground truth moment is reported.
/// 2. Streak artifact level: standard deviation of residuals (after
///    linear detrending) in a rim region around the calcification,
///    normalized by the mean calcification susceptibility.
///
/// Returns (calc_moment_deviation, streak_artifact_level).
///
/// Reference: compute_calcification_metrics_v0.m
pub fn calcification_metrics(
    recon: &[f64],
    truth: &[f64],
    seg: &[u8],
    dims: (usize, usize, usize),
) -> (f64, f64) {
    let (nx, ny, nz) = dims;
    let calc_label = 16u8;

    // Ground truth moment from known segmentation
    let gt_vals: Vec<f64> = (0..seg.len())
        .filter(|&i| seg[i] == calc_label)
        .map(|i| truth[i])
        .collect();
    if gt_vals.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let gt_moment = gt_vals.len() as f64 * gt_vals.iter().sum::<f64>() / gt_vals.len() as f64;

    // Find bounding box of calcification voxels
    let mut x_min = nx;
    let mut x_max = 0usize;
    let mut y_min = ny;
    let mut y_max = 0usize;
    let mut z_min = nz;
    let mut z_max = 0usize;

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if seg[i + j * nx + k * nx * ny] == calc_label {
                    x_min = x_min.min(i);
                    x_max = x_max.max(i);
                    y_min = y_min.min(j);
                    y_max = y_max.max(j);
                    z_min = z_min.min(k);
                    z_max = z_max.max(k);
                }
            }
        }
    }

    // Build nested boxes: cube (N=3), rim (N=4 beyond cube), outer (N=4 beyond rim)
    let n1 = 3usize;
    let cube_x = (x_min.saturating_sub(n1), (x_max + n1 + 1).min(nx));
    let cube_y = (y_min.saturating_sub(n1), (y_max + n1 + 1).min(ny));
    let cube_z = (z_min.saturating_sub(n1), (z_max + n1 + 1).min(nz));

    let n2 = 4usize;
    let rim_x = (cube_x.0.saturating_sub(n2), (cube_x.1 + n2).min(nx));
    let rim_y = (cube_y.0.saturating_sub(n2), (cube_y.1 + n2).min(ny));
    let rim_z = (cube_z.0.saturating_sub(n2), (cube_z.1 + n2).min(nz));

    let n3 = 4usize;
    let outer_x = (rim_x.0.saturating_sub(n3), (rim_x.1 + n3).min(nx));
    let outer_y = (rim_y.0.saturating_sub(n3), (rim_y.1 + n3).min(ny));
    let outer_z = (rim_z.0.saturating_sub(n3), (rim_z.1 + n3).min(nz));

    // Classify and extract voxel values
    let mut qsm_cube = Vec::new();
    let mut qsm_no_cube = Vec::new();
    let mut rim_recon = Vec::new();
    let mut rim_truth = Vec::new();

    for k in outer_z.0..outer_z.1 {
        for j in outer_y.0..outer_y.1 {
            for i in outer_x.0..outer_x.1 {
                let idx = i + j * nx + k * nx * ny;
                let in_cube = i >= cube_x.0 && i < cube_x.1
                    && j >= cube_y.0 && j < cube_y.1
                    && k >= cube_z.0 && k < cube_z.1;

                if in_cube {
                    qsm_cube.push(recon[idx]);
                } else {
                    qsm_no_cube.push(recon[idx]);
                    let in_rim = i >= rim_x.0 && i < rim_x.1
                        && j >= rim_y.0 && j < rim_y.1
                        && k >= rim_z.0 && k < rim_z.1;
                    if in_rim {
                        rim_recon.push(recon[idx]);
                        rim_truth.push(truth[idx]);
                    }
                }
            }
        }
    }

    // Adaptive threshold: find the least-negative threshold where
    // no voxels outside the cube fall below it (calcification has
    // negative susceptibility)
    let mut threshold = -3.5_f64;
    for i in 0..=350 {
        let t = -(i as f64) * 0.01;
        let outside_count = qsm_no_cube.iter().filter(|&&v| v < t).count();
        if outside_count == 0 {
            threshold = t;
            break;
        }
    }

    // Segment calcification in reconstruction using threshold
    let calc_segmented: Vec<f64> = qsm_cube.iter()
        .filter(|&&v| v < threshold)
        .copied()
        .collect();

    if calc_segmented.is_empty() {
        return ((gt_moment - 0.0).abs(), f64::NAN);
    }

    let calc_volume = calc_segmented.len() as f64;
    let calc_mean = calc_segmented.iter().sum::<f64>() / calc_volume;
    let recon_moment = calc_volume * calc_mean;
    let moment_deviation = (gt_moment - recon_moment).abs();

    // Streak artifact: fit linear model in rim, compute normalized std of residuals
    if rim_recon.len() < 2 {
        return (moment_deviation, f64::NAN);
    }

    let (slope, intercept) = linear_fit(&rim_truth, &rim_recon);
    let residuals: Vec<f64> = rim_recon.iter()
        .zip(&rim_truth)
        .map(|(r, t)| r - (slope * t + intercept))
        .collect();

    let mean_res: f64 = residuals.iter().sum::<f64>() / residuals.len() as f64;
    let std_res: f64 = (residuals.iter()
        .map(|r| (r - mean_res).powi(2))
        .sum::<f64>() / residuals.len() as f64)
        .sqrt();

    let streak = if calc_mean.abs() > 1e-30 {
        std_res / calc_mean.abs()
    } else {
        f64::NAN
    };

    (moment_deviation, streak)
}

/// Challenge evaluation metrics for QSM susceptibility map quality.
///
/// These metrics match the QSM Reconstruction Challenge evaluation protocol,
/// providing tissue-specific and region-specific quality measures beyond
/// global RMSE/correlation.
#[derive(Debug)]
pub struct ChallengeMetrics {
    pub name: String,
    /// Demeaned NRMSE (%) — whole mask
    pub nrmse: f64,
    /// Detrended NRMSE (%) — whole mask, linear scaling bias removed
    pub nrmse_detrend: f64,
    /// Detrended NRMSE (%) — GM, WM, Thalamus only (labels 7–9)
    pub nrmse_tissue: f64,
    /// Detrended NRMSE (%) — blood vessels only (label 11, dilated)
    pub nrmse_blood: f64,
    /// Detrended NRMSE (%) — deep gray matter only (labels 1–6)
    pub nrmse_dgm: f64,
    /// DGM linearity: |1 - slope| of mean susceptibility per DGM region
    pub dgm_linearity: f64,
    /// Calcification moment deviation: |GT_moment - recon_moment|
    pub calc_moment_dev: f64,
    /// Calcification streak artifact level (normalized residual std)
    pub calc_streak: f64,
}

impl ChallengeMetrics {
    /// Compute all challenge metrics for a susceptibility map reconstruction.
    pub fn compute(
        name: &str,
        output: &[f64],
        ground_truth: &[f64],
        mask: &[u8],
        segmentation: &[u8],
        dims: (usize, usize, usize),
    ) -> Self {
        // Whole-mask demeaned/detrended NRMSE
        let (nrmse, nrmse_detrend) = nrmse_challenge(output, ground_truth, mask);

        // Tissue mask: Thalamus (7), WM (8), GM (9)
        let tissue_mask: Vec<u8> = mask.iter().zip(segmentation)
            .map(|(&m, &s)| if m > 0 && s >= 7 && s <= 9 { 1 } else { 0 })
            .collect();
        let (_, nrmse_tissue) = nrmse_challenge(output, ground_truth, &tissue_mask);

        // Blood mask: label 11, dilated by 1 voxel (3x3x3 kernel)
        let blood_base: Vec<u8> = mask.iter().zip(segmentation)
            .map(|(&m, &s)| if m > 0 && s == 11 { 1 } else { 0 })
            .collect();
        let blood_mask = dilate_mask_3d(&blood_base, dims);
        let (_, nrmse_blood) = nrmse_challenge(output, ground_truth, &blood_mask);

        // DGM mask: labels 1–6
        let dgm_mask: Vec<u8> = mask.iter().zip(segmentation)
            .map(|(&m, &s)| if m > 0 && s >= 1 && s <= 6 { 1 } else { 0 })
            .collect();
        let (_, nrmse_dgm) = nrmse_challenge(output, ground_truth, &dgm_mask);

        // DGM linearity
        let dgm_lin = dgm_linearity(output, ground_truth, segmentation);

        // Calcification metrics
        let (calc_moment_dev, calc_streak) = calcification_metrics(
            output, ground_truth, segmentation, dims,
        );

        ChallengeMetrics {
            name: name.to_string(),
            nrmse,
            nrmse_detrend,
            nrmse_tissue,
            nrmse_blood,
            nrmse_dgm,
            dgm_linearity: dgm_lin,
            calc_moment_dev,
            calc_streak,
        }
    }

    pub fn print(&self) {
        println!("{:<15} NRMSE={:.2}%  DT={:.2}%  Tissue={:.2}%  Blood={:.2}%  DGM={:.2}%  DGM_lin={:.4}  CalcDev={:.4}  Streak={:.4}",
            self.name, self.nrmse, self.nrmse_detrend, self.nrmse_tissue,
            self.nrmse_blood, self.nrmse_dgm, self.dgm_linearity,
            self.calc_moment_dev, self.calc_streak);
    }

    pub fn print_with_time(&self, elapsed: std::time::Duration) {
        println!("{:<15} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>10.4} {:>10.4} {:>10.4} {:>10.2?}",
            self.name, self.nrmse, self.nrmse_detrend, self.nrmse_tissue,
            self.nrmse_blood, self.nrmse_dgm, self.dgm_linearity,
            self.calc_moment_dev, self.calc_streak, elapsed);
    }

    /// Print machine-readable CSV line for CI metric collection
    pub fn print_ci_metrics(&self) {
        println!("CHALLENGE:{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.4},{:.4},{:.4}",
            self.name, self.nrmse, self.nrmse_detrend, self.nrmse_tissue,
            self.nrmse_blood, self.nrmse_dgm, self.dgm_linearity,
            self.calc_moment_dev, self.calc_streak);
    }
}

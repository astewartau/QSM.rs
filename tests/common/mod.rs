//! Common test utilities for QSM-core integration tests

use std::fs;
use std::path::Path;

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

        // Echo times from JSON (hardcoded for now, could parse JSON)
        let echo_times = vec![0.004, 0.008, 0.012, 0.016];

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
            field_strength: 7.0,
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
}

impl TestResult {
    pub fn new(name: &str, output: &[f64], ground_truth: &[f64], mask: &[u8]) -> Self {
        TestResult {
            name: name.to_string(),
            rmse: rmse(output, ground_truth, mask),
            nrmse: nrmse(output, ground_truth, mask),
            correlation: correlation(output, ground_truth, mask),
        }
    }

    pub fn print(&self) {
        println!("{:<15} RMSE={:.6}  NRMSE={:.4}  r={:.4}",
            self.name, self.rmse, self.nrmse, self.correlation);
    }

    pub fn print_with_time(&self, elapsed: std::time::Duration) {
        println!("{:<15} {:>12.6} {:>10.4} {:>10.4} {:>10.2?}",
            self.name, self.rmse, self.nrmse, self.correlation, elapsed);
    }
}

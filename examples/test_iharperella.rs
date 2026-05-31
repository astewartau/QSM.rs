//! Quick test of iHARPERELLA on real BIDS data
//!
//! Usage: cargo run --example test_iharperella --release

use std::path::Path;
use std::time::Instant;
use qsm_core::nifti_io;
use qsm_core::pipeline;
use qsm_core::Grid;

fn main() {
    let bids_dir = std::env::args().nth(1)
        .unwrap_or_else(|| "/home/ashley/bids".to_string());
    let sub = "sub-1";

    // Load first echo phase (wrapped, radians)
    let phase_path = format!("{}/{}/anat/{}_echo-1_part-phase_MEGRE.nii", bids_dir, sub, sub);
    println!("Loading phase: {}", phase_path);
    let phase_nii = nifti_io::read_nifti_file(Path::new(&phase_path)).expect("Failed to load phase");
    let (nx, ny, nz) = phase_nii.dims;
    let (vsx, vsy, vsz) = phase_nii.voxel_size;

    println!("Dims: {}x{}x{}, voxel: {:.2}x{:.2}x{:.2} mm", nx, ny, nz, vsx, vsy, vsz);

    // Load mask
    let mask_path = format!("{}/derivatives/qsm-forward/{}/anat/{}_mask.nii", bids_dir, sub, sub);
    println!("Loading mask: {}", mask_path);
    let mask_nii = nifti_io::read_nifti_file(Path::new(&mask_path)).expect("Failed to load mask");
    let mask: Vec<u8> = mask_nii.data.iter().map(|&v| if v > 0.5 { 1 } else { 0 }).collect();

    let mask_count: usize = mask.iter().map(|&m| m as usize).sum();
    println!("Mask voxels: {}", mask_count);

    // Scale phase to radians if needed (BIDS phase is often in [-4096, 4095] or [-pi, pi])
    let phase_min = phase_nii.data.iter().cloned().fold(f64::MAX, f64::min);
    let phase_max = phase_nii.data.iter().cloned().fold(f64::MIN, f64::max);
    println!("Phase range: [{:.4}, {:.4}]", phase_min, phase_max);

    let phase = if phase_max > 4.0 {
        // Likely integer-scaled, convert to radians
        println!("Scaling phase to radians (assuming range maps to [-pi, pi])");
        let scale = 2.0 * std::f64::consts::PI / (phase_max - phase_min);
        phase_nii.data.iter().map(|&v| v * scale).collect::<Vec<f64>>()
    } else {
        phase_nii.data.clone()
    };

    // Run iHARPERELLA
    println!("\nRunning iHARPERELLA (40 CG iterations, radius=10mm)...");
    let grid = Grid::new(nx, ny, nz, vsx, vsy, vsz);
    let params = pipeline::HarperellaParams::default();
    let start = Instant::now();
    let (tissue_phase, output_mask) = pipeline::iharperella(
        &phase, &mask,
        &grid,
        &params,
        |_iter, _total| {},
    );
    let elapsed = start.elapsed();
    println!("Completed in {:.2?}", elapsed);

    let output_count: usize = output_mask.iter().map(|&m| m as usize).sum();
    println!("Output mask voxels: {}", output_count);

    // Stats on tissue phase
    let mut vals: Vec<f64> = tissue_phase.iter()
        .zip(output_mask.iter())
        .filter(|(_, &m)| m == 1)
        .map(|(&v, _)| v)
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if !vals.is_empty() {
        let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
        let std: f64 = (vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64).sqrt();
        println!("Tissue phase stats (within eroded mask):");
        println!("  Mean: {:.6}", mean);
        println!("  Std:  {:.6}", std);
        println!("  Min:  {:.6}", vals[0]);
        println!("  Max:  {:.6}", vals[vals.len() - 1]);
    }

    // Save result
    let out_path = format!("{}/derivatives/qsmxt.rs/{}/anat/{}_iharperella.nii", bids_dir, sub, sub);
    println!("\nSaving result to: {}", out_path);
    if let Err(e) = nifti_io::save_nifti_to_file(
        Path::new(&out_path),
        &tissue_phase,
        (nx, ny, nz),
        (vsx, vsy, vsz),
        &phase_nii.affine,
    ) {
        eprintln!("Failed to save: {}", e);
    }
    println!("Done!");
}

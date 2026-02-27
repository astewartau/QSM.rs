//! QSMART pipeline test on real multi-echo NIfTI data
//!
//! Usage: cargo run --release --example qsmart_nifti

use std::path::Path;
use std::time::Instant;

use qsm_core::nifti_io::{read_nifti_file, save_nifti_to_file};
use qsm_core::bet::run_bet;
use qsm_core::unwrap::laplacian_unwrap;
use qsm_core::bgremove::{sdf, SdfParams};
use qsm_core::inversion::ilsqr_simple;
use qsm_core::utils::{
    multi_echo_linear_fit, field_to_hz,
    generate_vasculature_mask, VasculatureParams,
    adjust_offset,
    majority_filter_3d,
    morphological_close,
};

fn main() -> Result<(), String> {
    let total_start = Instant::now();

    let input_dir = Path::new("/home/ashley/nifti");
    let output_dir = Path::new("/home/ashley/nifti/out");

    // Create output directory
    std::fs::create_dir_all(output_dir).map_err(|e| format!("Failed to create output dir: {e}"))?;

    // ========================================================================
    // Load multi-echo data
    // ========================================================================
    println!("[INFO] Loading NIfTI data...");
    let start = Instant::now();

    // Magnitude echoes
    let mag1_nii = read_nifti_file(&input_dir.join("_QSM_p2_1mmIso_TE20_20170705134507_5.nii"))?;
    let mag2_nii = read_nifti_file(&input_dir.join("_QSM_p2_1mmIso_TE20_20170705134507_5a.nii"))?;

    // Phase echoes
    let pha1_nii = read_nifti_file(&input_dir.join("_QSM_p2_1mmIso_TE20_20170705134507_6_ph.nii"))?;
    let pha2_nii = read_nifti_file(&input_dir.join("_QSM_p2_1mmIso_TE20_20170705134507_6_pha.nii"))?;

    let (nx, ny, nz) = mag1_nii.dims;
    let (vsx, vsy, vsz) = mag1_nii.voxel_size;
    let affine = mag1_nii.affine;
    let n_total = nx * ny * nz;

    println!("[INFO] Loaded in {:.2?}", start.elapsed());
    println!("[INFO] Volume: {}x{}x{}, Voxel: {:.2}x{:.2}x{:.2} mm", nx, ny, nz, vsx, vsy, vsz);

    // Echo times from JSON sidecars (EchoTime1=0.02s, EchoTime2=0.015s)
    let echo_times = [0.02_f64, 0.015_f64];
    let field_strength = 3.0_f64; // Tesla
    let b0_dir = (0.0, 0.0, 1.0);

    // Rescale phase from Siemens integer range [0, 4095] to [-pi, pi]
    let pi = std::f64::consts::PI;
    let rescale_phase = |data: &[f64]| -> Vec<f64> {
        // Siemens stores phase as integers scaled to [0, 4095] or [-4096, 4095]
        // Find actual range to auto-detect scaling
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("[INFO]   Phase range: [{:.1}, {:.1}]", min, max);

        if min >= -pi * 1.1 && max <= pi * 1.1 {
            // Already in radians
            println!("[INFO]   Phase appears to be in radians, no rescaling needed");
            data.to_vec()
        } else if min >= 0.0 {
            // Unsigned integer range [0, max] -> [-pi, pi]
            println!("[INFO]   Rescaling unsigned phase [{:.0}, {:.0}] -> [-pi, pi]", min, max);
            data.iter().map(|&v| (v / max) * 2.0 * pi - pi).collect()
        } else {
            // Signed integer range [min, max] -> [-pi, pi]
            println!("[INFO]   Rescaling signed phase [{:.0}, {:.0}] -> [-pi, pi]", min, max);
            let range = max - min;
            data.iter().map(|&v| ((v - min) / range) * 2.0 * pi - pi).collect()
        }
    };

    println!("[INFO] Rescaling phase echo 1...");
    let phase1 = rescale_phase(&pha1_nii.data);
    println!("[INFO] Rescaling phase echo 2...");
    let phase2 = rescale_phase(&pha2_nii.data);

    let phases: Vec<&[f64]> = vec![&phase1, &phase2];
    let mags: Vec<&[f64]> = vec![&mag1_nii.data, &mag2_nii.data];

    // ========================================================================
    // Step 1: Brain extraction (BET) + mask post-processing
    // Matching QSMART brainmask.m: BET f=0.2, magnitude threshold,
    // majority filter, morphological closing with sphere radius 2
    // ========================================================================
    println!("\n[STEP 1] Brain extraction (BET + post-processing)...");
    let start = Instant::now();
    let bet_mask = run_bet(
        &mag1_nii.data,
        nx, ny, nz,
        vsx, vsy, vsz,
        0.2,   // fractional intensity threshold (matching QSMART)
        1.0,   // smoothness factor
        0.0,   // gradient threshold
        1000,  // iterations
        4,     // icosphere subdivisions
    );
    println!("[INFO] BET completed in {:.2?}", start.elapsed());

    // Magnitude thresholding: mask(BET_map < mag_threshold) = 0
    // QSMART default: mag_threshold = 100
    let mag_threshold = 100.0;
    let mut mask: Vec<u8> = bet_mask.iter()
        .enumerate()
        .map(|(i, &m)| {
            if m > 0 && mag1_nii.data[i] >= mag_threshold { 1 } else { 0 }
        })
        .collect();
    let pre_morph_count: usize = mask.iter().map(|&v| v as usize).sum();
    println!("[INFO] After magnitude threshold (>= {:.0}): {} voxels", mag_threshold, pre_morph_count);

    // Majority filter: bwmorph3(mask, 'majority') — removes isolated voxels
    mask = majority_filter_3d(&mask, nx, ny, nz);
    let post_majority_count: usize = mask.iter().map(|&v| v as usize).sum();
    println!("[INFO] After majority filter: {} voxels", post_majority_count);

    // Morphological closing: imclose(mask, strel('sphere', sph_radius1))
    // QSMART default: sph_radius1 = 2
    mask = morphological_close(&mask, nx, ny, nz, 2);
    let mask_voxels: usize = mask.iter().map(|&v| v as usize).sum();
    println!("[INFO] After morphological closing (r=2): {} voxels ({:.1}%)",
        mask_voxels, 100.0 * mask_voxels as f64 / n_total as f64);
    println!("[INFO] Brain mask generation completed in {:.2?}", start.elapsed());

    // Save mask
    let mask_f64: Vec<f64> = mask.iter().map(|&v| v as f64).collect();
    save_nifti_to_file(
        &output_dir.join("mask.nii.gz"),
        &mask_f64, (nx, ny, nz), (vsx, vsy, vsz), &affine,
    )?;
    println!("[INFO] Saved mask.nii.gz");

    // ========================================================================
    // Step 2: Phase unwrapping (Laplacian) per echo
    // ========================================================================
    println!("\n[STEP 2] Phase unwrapping (Laplacian)...");
    let start = Instant::now();
    let unwrapped_phases: Vec<Vec<f64>> = phases.iter()
        .enumerate()
        .map(|(i, phase)| {
            println!("[INFO]   Unwrapping echo {}...", i + 1);
            laplacian_unwrap(phase, &mask, nx, ny, nz, vsx, vsy, vsz)
        })
        .collect();
    println!("[INFO] Phase unwrapping completed in {:.2?}", start.elapsed());

    // Save unwrapped phases
    for (i, uwp) in unwrapped_phases.iter().enumerate() {
        save_nifti_to_file(
            &output_dir.join(format!("unwrapped_echo{}.nii.gz", i + 1)),
            uwp, (nx, ny, nz), (vsx, vsy, vsz), &affine,
        )?;
    }
    println!("[INFO] Saved unwrapped phase echoes");

    // ========================================================================
    // Step 3: Multi-echo linear fit with reliability masking
    // ========================================================================
    println!("\n[STEP 3] Multi-echo linear fit...");
    let start = Instant::now();
    let fit_result = multi_echo_linear_fit(
        &unwrapped_phases,
        &mags,
        &echo_times,
        &mask,
        false, // no intercept (through-origin fit)
        0.0,   // static threshold (QSMART default: 40)
        (nx, ny, nz),
        (vsx, vsy, vsz),
    );
    println!("[INFO] Linear fit completed in {:.2?}", start.elapsed());

    // Convert field from rad/s to Hz
    let field_hz = field_to_hz(&fit_result.field);

    // Save field map, reliability mask, and fit residual
    save_nifti_to_file(
        &output_dir.join("field_hz.nii.gz"),
        &field_hz, (nx, ny, nz), (vsx, vsy, vsz), &affine,
    )?;

    let reliability_f64: Vec<f64> = fit_result.reliability_mask.iter().map(|&v| v as f64).collect();
    save_nifti_to_file(
        &output_dir.join("reliability_mask.nii.gz"),
        &reliability_f64, (nx, ny, nz), (vsx, vsy, vsz), &affine,
    )?;

    save_nifti_to_file(
        &output_dir.join("fit_residual.nii.gz"),
        &fit_result.fit_residual, (nx, ny, nz), (vsx, vsy, vsz), &affine,
    )?;

    let reliable_voxels: usize = fit_result.reliability_mask.iter().map(|&v| v as usize).sum();
    println!("[INFO] Reliable voxels: {} ({:.1}% of brain)",
        reliable_voxels, 100.0 * reliable_voxels as f64 / mask_voxels as f64);
    println!("[INFO] Saved field_hz.nii.gz, reliability_mask.nii.gz, fit_residual.nii.gz");

    // ========================================================================
    // Step 4: Vasculature mask (Frangi filter)
    // ========================================================================
    println!("\n[STEP 4] Vasculature mask generation (Frangi filter)...");
    let start = Instant::now();
    let vasc_mask = generate_vasculature_mask(
        &mag1_nii.data,
        &mask,
        nx, ny, nz,
        &VasculatureParams::default(),
    );
    println!("[INFO] Vasculature mask completed in {:.2?}", start.elapsed());

    let vessel_voxels: usize = vasc_mask.iter().filter(|&&v| v < 0.5).count();
    println!("[INFO] Vessel voxels: {} ({:.1}% of brain)",
        vessel_voxels, 100.0 * vessel_voxels as f64 / mask_voxels as f64);

    save_nifti_to_file(
        &output_dir.join("vasculature_mask.nii.gz"),
        &vasc_mask, (nx, ny, nz), (vsx, vsy, vsz), &affine,
    )?;
    println!("[INFO] Saved vasculature_mask.nii.gz");

    // ========================================================================
    // Prepare weighted mask and PPM factors
    // ========================================================================
    let weighted_mask: Vec<f64> = mask_f64.iter()
        .zip(fit_result.reliability_mask.iter())
        .map(|(&m, &r)| if m > 0.0 && r > 0 { 1.0 } else { 0.0 })
        .collect();

    save_nifti_to_file(
        &output_dir.join("weighted_mask.nii.gz"),
        &weighted_mask, (nx, ny, nz), (vsx, vsy, vsz), &affine,
    )?;
    println!("[INFO] Saved weighted_mask.nii.gz");

    let gyro_rad: f64 = 2.675e8; // rad/s/T
    let ppm_factor = gyro_rad * field_strength / 1e6;
    let scale_to_ppm = 1e6 / (42.576e6 * field_strength);

    // ========================================================================
    // Step 5: QSMART Stage 1 - SDF + iLSQR (whole ROI)
    // ========================================================================
    println!("\n[STEP 5] QSMART Stage 1: SDF + iLSQR (full mask)...");
    let start = Instant::now();

    let ones_vasc: Vec<f64> = vec![1.0; n_total];

    let lfs_stage1 = sdf(
        &field_hz,
        &weighted_mask,
        &ones_vasc,
        nx, ny, nz,
        &SdfParams::stage1(),
    );
    println!("[INFO] SDF Stage 1 completed in {:.2?}", start.elapsed());

    save_nifti_to_file(
        &output_dir.join("lfs_stage1.nii.gz"),
        &lfs_stage1, (nx, ny, nz), (vsx, vsy, vsz), &affine,
    )?;

    let start = Instant::now();
    let mask_stage1_u8: Vec<u8> = weighted_mask.iter()
        .map(|&v| if v > 0.1 { 1 } else { 0 })
        .collect();

    let chi_stage1 = ilsqr_simple(
        &lfs_stage1,
        &mask_stage1_u8,
        nx, ny, nz,
        vsx, vsy, vsz,
        b0_dir,
        0.01,
        50,
    );
    println!("[INFO] iLSQR Stage 1 completed in {:.2?}", start.elapsed());

    save_nifti_to_file(
        &output_dir.join("chi_stage1.nii.gz"),
        &chi_stage1, (nx, ny, nz), (vsx, vsy, vsz), &affine,
    )?;
    println!("[INFO] Saved lfs_stage1.nii.gz, chi_stage1.nii.gz");

    // ========================================================================
    // Step 6: QSMART Stage 2 - SDF + iLSQR (tissue only)
    // ========================================================================
    println!("\n[STEP 6] QSMART Stage 2: SDF + iLSQR (tissue only)...");
    let start = Instant::now();

    // Weight field by reliability mask
    let field_hz_weighted: Vec<f64> = field_hz.iter()
        .zip(weighted_mask.iter())
        .map(|(&f, &m)| f * m)
        .collect();

    let lfs_stage2 = sdf(
        &field_hz_weighted,
        &weighted_mask,
        &vasc_mask,
        nx, ny, nz,
        &SdfParams::stage2(),
    );
    println!("[INFO] SDF Stage 2 completed in {:.2?}", start.elapsed());

    save_nifti_to_file(
        &output_dir.join("lfs_stage2.nii.gz"),
        &lfs_stage2, (nx, ny, nz), (vsx, vsy, vsz), &affine,
    )?;

    let start = Instant::now();
    let mask_stage2_u8: Vec<u8> = weighted_mask.iter()
        .zip(vasc_mask.iter())
        .map(|(&wm, &v)| if wm > 0.1 && v > 0.5 { 1 } else { 0 })
        .collect();

    let chi_stage2 = ilsqr_simple(
        &lfs_stage2,
        &mask_stage2_u8,
        nx, ny, nz,
        vsx, vsy, vsz,
        b0_dir,
        0.01,
        50,
    );
    println!("[INFO] iLSQR Stage 2 completed in {:.2?}", start.elapsed());

    save_nifti_to_file(
        &output_dir.join("chi_stage2.nii.gz"),
        &chi_stage2, (nx, ny, nz), (vsx, vsy, vsz), &affine,
    )?;
    println!("[INFO] Saved lfs_stage2.nii.gz, chi_stage2.nii.gz");

    // ========================================================================
    // Step 7: Offset adjustment and final PPM scaling
    // ========================================================================
    println!("\n[STEP 7] Offset adjustment and PPM scaling...");
    let start = Instant::now();

    let removed_voxels: Vec<f64> = weighted_mask.iter()
        .zip(vasc_mask.iter())
        .map(|(&wm, &v)| wm - v)
        .collect();

    // Scale lfs_stage1 to ppm for adjust_offset
    let lfs_stage1_ppm: Vec<f64> = lfs_stage1.iter()
        .map(|&v| v * ppm_factor)
        .collect();

    let chi_qsmart_raw = adjust_offset(
        &removed_voxels,
        &lfs_stage1_ppm,
        &chi_stage1,
        &chi_stage2,
        nx, ny, nz,
        vsx, vsy, vsz,
        b0_dir,
        ppm_factor,
    );

    // Scale to ppm
    let chi_qsmart: Vec<f64> = chi_qsmart_raw.iter()
        .enumerate()
        .map(|(i, &v)| if mask[i] > 0 { v * scale_to_ppm } else { 0.0 })
        .collect();
    println!("[INFO] Offset adjustment completed in {:.2?}", start.elapsed());

    save_nifti_to_file(
        &output_dir.join("chi_qsmart.nii.gz"),
        &chi_qsmart, (nx, ny, nz), (vsx, vsy, vsz), &affine,
    )?;

    // Also save unscaled raw for reference
    save_nifti_to_file(
        &output_dir.join("chi_qsmart_raw.nii.gz"),
        &chi_qsmart_raw, (nx, ny, nz), (vsx, vsy, vsz), &affine,
    )?;
    println!("[INFO] Saved chi_qsmart.nii.gz, chi_qsmart_raw.nii.gz");

    // ========================================================================
    // Summary
    // ========================================================================
    println!("\n{}", "=".repeat(60));
    println!("QSMART Pipeline Complete!");
    println!("Total time: {:.2?}", total_start.elapsed());
    println!("{}", "=".repeat(60));
    println!("\nOutputs saved to {}:", output_dir.display());
    println!("  mask.nii.gz              - Brain mask (BET)");
    println!("  unwrapped_echo1.nii.gz   - Unwrapped phase echo 1");
    println!("  unwrapped_echo2.nii.gz   - Unwrapped phase echo 2");
    println!("  field_hz.nii.gz          - B0 field map (Hz)");
    println!("  reliability_mask.nii.gz  - Reliability mask (R_0)");
    println!("  fit_residual.nii.gz      - Multi-echo fit residual");
    println!("  weighted_mask.nii.gz     - Weighted mask (mask * R_0)");
    println!("  vasculature_mask.nii.gz  - Vasculature mask (1=tissue, 0=vessel)");
    println!("  lfs_stage1.nii.gz        - Local field (Stage 1, SDF)");
    println!("  chi_stage1.nii.gz        - Susceptibility (Stage 1, iLSQR)");
    println!("  lfs_stage2.nii.gz        - Local field (Stage 2, SDF tissue)");
    println!("  chi_stage2.nii.gz        - Susceptibility (Stage 2, iLSQR tissue)");
    println!("  chi_qsmart_raw.nii.gz    - QSMART susceptibility (raw, Hz)");
    println!("  chi_qsmart.nii.gz        - QSMART susceptibility (ppm)");

    Ok(())
}

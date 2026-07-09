//! High-level pipeline: describe the scan, then run the stages.
//!
//! This is the recommended entry point. You build a [`ScanMetadata`] and the
//! per-stage config structs, then call the `run_*` stage functions, which
//! dispatch to the right algorithm and handle unit conversions internally.
//!
//! Run with: `cargo run --release --example pipeline_highlevel`
//!
//! In real use, load `phase`/`magnitude` from NIfTI with [`qsm_core::io`].

use qsm_core::pipeline::{
    ScanMetadata, FieldMappingConfig, BgRemovalConfig, InversionConfig, QsmReference,
    run_field_mapping, run_bg_removal, run_dipole_inversion, apply_reference,
};

fn main() {
    let (nx, ny, nz) = (32, 32, 32);
    let n = nx * ny * nz;

    // Synthetic single-echo inputs: a bright sphere with smooth wrapped phase.
    let (mut magnitude, mut phase, mut mask) = (vec![0.0f64; n], vec![0.0f64; n], vec![0u8; n]);
    let center = [nx as f64 / 2.0, ny as f64 / 2.0, nz as f64 / 2.0];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = i + j * nx + k * nx * ny;
                let r2 = (i as f64 - center[0]).powi(2)
                    + (j as f64 - center[1]).powi(2)
                    + (k as f64 - center[2]).powi(2);
                if r2 < 100.0 {
                    magnitude[idx] = 100.0;
                    phase[idx] = 0.5 * (i as f64 * 0.3).sin();
                    mask[idx] = 1;
                }
            }
        }
    }

    // Describe the acquisition once; every stage reads from this.
    let meta = ScanMetadata {
        dims: (nx, ny, nz),
        voxel_size: (1.0, 1.0, 1.0),
        echo_times: vec![0.020], // seconds
        field_strength: 3.0,     // Tesla
        b0_direction: (0.0, 0.0, 1.0),
    };

    let phases: Vec<&[f64]> = vec![&phase];
    let magnitudes: Vec<&[f64]> = vec![&magnitude];

    // 1. Multi-echo phase -> B0 field map (ppm)
    let field = run_field_mapping(
        &phases, Some(&magnitudes), &mask, &meta,
        &FieldMappingConfig::default(), &mut |_, _| {},
    ).expect("field mapping failed");

    // 2. Total field -> local field (ppm); also returns the eroded mask
    let bg = run_bg_removal(
        &field.b0_field_ppm, &mask, &meta,
        &BgRemovalConfig::default(), &mut |_, _| {},
    ).expect("background removal failed");

    // 3. Local field -> susceptibility (ppm)
    let chi = run_dipole_inversion(
        &bg.local_field_ppm, &bg.eroded_mask, &meta,
        &InversionConfig::default(), Some(&magnitude), &mut |_, _| {},
    ).expect("dipole inversion failed");

    // 4. Reference the map (subtract the mean inside the mask)
    let chi = apply_reference(&chi, &bg.eroded_mask, QsmReference::Mean);

    let mean = chi.iter().sum::<f64>() / n as f64;
    println!("Reconstructed susceptibility map: {} voxels, mean = {:.6} ppm", chi.len(), mean);
}

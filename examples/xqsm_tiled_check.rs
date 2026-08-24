//! Compare whole-volume xQSM against overlap-tiled xQSM on a real local field, to quantify
//! the tile-boundary approximation error AND the speed of the tiled path across tile configs.
//!
//! Run:
//!   cargo run --release --example xqsm_tiled_check --features onnx,download -- <localfield.nii> <mask.nii>
//!
//! Weights (`xqsm.onnx`) are taken from the model cache / `$QSM_MODEL_DIR` (or downloaded).
//! Writes chi_full.nii + chi_tiled_<cfg>.nii + chi_diff_<cfg>.nii to /tmp/xqsm_check.

use qsm_core::inversion::{xqsm, xqsm_tiled, TileConfig};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::models::onnx::{OnnxModel, Tensor};
use qsm_core::Grid;
use std::path::Path;
use std::time::Instant;

/// Pearson correlation + NRMSE (normalized by the reference std) over masked voxels.
fn compare(reference: &[f64], other: &[f64], mask: &[u8]) -> (f64, f64) {
    let idx: Vec<usize> = (0..reference.len()).filter(|&i| mask[i] != 0).collect();
    let n = idx.len() as f64;
    let mr = idx.iter().map(|&i| reference[i]).sum::<f64>() / n;
    let mo = idx.iter().map(|&i| other[i]).sum::<f64>() / n;
    let (mut sro, mut srr, mut soo, mut sse) = (0.0, 0.0, 0.0, 0.0);
    for &i in &idx {
        let dr = reference[i] - mr;
        let dv = other[i] - mo;
        sro += dr * dv;
        srr += dr * dr;
        soo += dv * dv;
        sse += (reference[i] - other[i]).powi(2);
    }
    let r = sro / (srr.sqrt() * soo.sqrt());
    let nrmse = (sse / n).sqrt() / (srr / n).sqrt();
    (r, nrmse)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lf_path = args.get(1).expect("usage: xqsm_tiled_check <localfield.nii> <mask.nii>");
    let mask_path = args.get(2).expect("usage: xqsm_tiled_check <localfield.nii> <mask.nii>");

    let lf = read_nifti_file(Path::new(lf_path)).expect("load local field");
    let m = read_nifti_file(Path::new(mask_path)).expect("load mask");
    assert_eq!(lf.dims, m.dims, "field/mask dims differ");
    let (nx, ny, nz) = lf.dims;
    let grid = Grid::new(nx, ny, nz, lf.voxel_size.0, lf.voxel_size.1, lf.voxel_size.2);
    let mask: Vec<u8> = m.data.iter().map(|&v| if v > 0.5 { 1 } else { 0 }).collect();
    let nmask = mask.iter().filter(|&&x| x != 0).count();
    eprintln!("volume {nx}x{ny}x{nz} = {} voxels, mask = {nmask} voxels", nx * ny * nz);

    let weights = qsm_core::models::primary_weight("xqsm")
        .expect("xqsm.onnx not found (set $QSM_MODEL_DIR or enable download)");

    let out = Path::new("/tmp/xqsm_check");
    std::fs::create_dir_all(out).ok();

    // --- Probe: isolate one-time graph optimization vs per-patch run cost (144^3 patch). ---
    {
        let p = 144usize;
        let model = OnnxModel::load(&weights).expect("load");
        let t = Instant::now();
        let plan = model.plan_for(&[&[1, 1, p, p, p]]).expect("plan");
        let opt = t.elapsed().as_secs_f64();
        let zero = Tensor::new(vec![1, 1, p, p, p], vec![0.0f32; p * p * p]);
        let t = Instant::now();
        let _ = plan.run_single(&zero).expect("run");
        let run1 = t.elapsed().as_secs_f64();
        eprintln!("probe(144^3): optimize(once)={opt:.1}s  run/patch={run1:.1}s");
    }

    // --- Whole-volume reference. ---
    let t = Instant::now();
    let chi_full = xqsm(&lf.data, &mask, &grid, &weights).expect("whole-volume xqsm");
    eprintln!("whole-volume xQSM: {:.1}s", t.elapsed().as_secs_f64());
    save_nifti_to_file(&out.join("chi_full.nii"), &chi_full, lf.dims, lf.voxel_size, &lf.affine).expect("save");

    // --- Config sweep: (core, halo). Bigger core + smaller halo = less overlap = faster;
    //     smaller halo = less global context = more boundary error. ---
    println!("{:>14} {:>7} {:>9} {:>8} {:>8}", "config", "patch", "time", "r", "NRMSE");
    for &(core, halo) in &[(128usize, 8usize), (128, 16), (96, 16)] {
        let cfg = TileConfig { core, halo };
        let t = Instant::now();
        let chi_tiled = xqsm_tiled(&lf.data, &mask, &grid, &weights, &cfg, |_, _| {}).expect("tiled xqsm");
        let secs = t.elapsed().as_secs_f64();
        let (r, nrmse) = compare(&chi_full, &chi_tiled, &mask);
        let p = (core + 2 * halo).div_ceil(8) * 8;
        println!(
            "  C={core:>3},H={halo:>2} {:>4}^3 {secs:>7.0}s {r:>8.4} {:>7.1}%",
            p,
            nrmse * 100.0
        );
        let tag = format!("c{core}h{halo}");
        let diff: Vec<f64> = chi_full.iter().zip(&chi_tiled).map(|(a, b)| a - b).collect();
        save_nifti_to_file(&out.join(format!("chi_tiled_{tag}.nii")), &chi_tiled, lf.dims, lf.voxel_size, &lf.affine).expect("save");
        save_nifti_to_file(&out.join(format!("chi_diff_{tag}.nii")), &diff, lf.dims, lf.voxel_size, &lf.affine).expect("save");
    }
    println!("Saved chi_full.nii + chi_tiled_*/chi_diff_* to {}", out.display());
}

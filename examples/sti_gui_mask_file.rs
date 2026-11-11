//! STI Suite 3.0 QSM_GUI's mask steps (QSM.rs `sti_gui_threshold`, `MaskOp::LargestComponent`,
//! `MaskOp::Imfill`, `pipeline::bg_removal::polish_mask`) on NIfTI files, for parity checks
//! against the GUI's logged MATLAB calls.
//!
//! usage: sti_gui_mask_file threshold <last_echo_magnitude> <mask_out>
//!        sti_gui_mask_file polish <mask> <mask_out>
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::pipeline::{apply_mask_ops, MaskOp, MaskThresholdMethod, ScanMetadata};
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: {} threshold|polish <input> <mask_out>", a[0]);
        std::process::exit(2);
    }
    let input = read_nifti_file(Path::new(&a[2])).expect("read input");
    let (nx, ny, nz) = input.dims;
    let (vx, vy, vz) = input.voxel_size;
    let grid = qsm_core::Grid::new(nx, ny, nz, vx, vy, vz);
    let mask: Vec<u8> = match a[1].as_str() {
        "threshold" => {
            let meta = ScanMetadata {
                dims: input.dims, voxel_size: input.voxel_size, echo_times: vec![], field_strength: 3.0,
                b0_direction: (0.0, 0.0, 1.0), slice_geometry: None, affine: None,
            };
            let ops = [
                MaskOp::Threshold { method: MaskThresholdMethod::StiGui, value: None },
                MaskOp::LargestComponent,
                MaskOp::Imfill,
            ];
            apply_mask_ops(vec![0; input.data.len()], &ops, &input.data, None, &meta).expect("mask")
        }
        "polish" => {
            let m: Vec<u8> = input.data.iter().map(|&v| u8::from(v != 0.0)).collect();
            qsm_core::pipeline::bg_removal::polish_mask(&m, &grid)
        }
        other => panic!("unknown step {other}"),
    };
    eprintln!("{} voxels", mask.iter().filter(|&&m| m != 0).count());
    let out: Vec<f64> = mask.iter().map(|&m| m as f64).collect();
    save_nifti_to_file(Path::new(&a[3]), &out, input.dims, input.voxel_size, &input.affine).expect("write");
}

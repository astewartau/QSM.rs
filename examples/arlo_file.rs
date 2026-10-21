//! Run ARLO (`arlo.m`, MEDI toolbox 2024.11.26) on NIfTI echo magnitudes, for parity checks
//! against MATLAB.
//!
//! usage: arlo_file <echo1,echo2,...> <TE1_ms,TE2_ms,...> <r2star_out>
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::r2star::arlo;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: {} <echo1,echo2,..> <TE1_ms,TE2_ms,..> <r2star_out>", a[0]);
        std::process::exit(2);
    }
    let first = read_nifti_file(Path::new(a[1].split(',').next().unwrap())).expect("read magnitude");
    let mags: Vec<f64> = a[1].split(',')
        .flat_map(|p| read_nifti_file(Path::new(p)).expect("read magnitude").data)
        .collect();
    let te: Vec<f64> = a[2].split(',').map(|t| t.parse::<f64>().expect("TE") * 1e-3).collect();
    let r2 = arlo(&mags, &te, first.data.len()).expect("arlo");
    save_nifti_to_file(Path::new(&a[3]), &r2, first.dims, first.voxel_size, &first.affine).expect("write");
}

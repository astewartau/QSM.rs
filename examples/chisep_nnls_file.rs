//! Run the χ-separation toolbox's `r2star_nnls` (QSM.rs `r2star::chisep_nnls`) on NIfTI echo
//! magnitudes and a mask, for parity checks against MATLAB.
//!
//! usage: chisep_nnls_file <echo1,echo2,...> <TE1_ms,TE2_ms,...> <mask> <r2star_out>
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::r2star::chisep_nnls;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!("usage: {} <echo1,echo2,..> <TE1_ms,TE2_ms,..> <mask> <r2star_out>", a[0]);
        std::process::exit(2);
    }
    let first = read_nifti_file(Path::new(a[1].split(',').next().unwrap())).expect("read magnitude");
    let mags: Vec<f64> = a[1].split(',')
        .flat_map(|p| read_nifti_file(Path::new(p)).expect("read magnitude").data)
        .collect();
    // Chisep_script.m passes TE*1000 (ms); QSM.rs takes seconds and multiplies back.
    let te: Vec<f64> = a[2].split(',').map(|t| t.parse::<f64>().expect("TE") / 1000.0).collect();
    let mask: Vec<bool> = read_nifti_file(Path::new(&a[3])).expect("read mask").data.iter().map(|&m| m != 0.0).collect();
    let t = std::time::Instant::now();
    let r2 = chisep_nnls(&mags, &te, &mask).expect("chisep_nnls");
    eprintln!("chisep_nnls: {} voxels in the mask, {:.1} s", mask.iter().filter(|&&m| m).count(), t.elapsed().as_secs_f64());
    save_nifti_to_file(Path::new(&a[4]), &r2, first.dims, first.voxel_size, &first.affine).expect("write");
}

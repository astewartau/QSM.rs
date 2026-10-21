//! Run HD-QSM (HDQSM.m) on NIfTI files, for parity checks against HDQSM.m in MATLAB.
//! Takes the local field in ppm and writes susceptibility in ppm, as the pipeline does.
//!
//! usage: hdqsm_file <field_ppm.nii[.gz]> <mask.nii[.gz]> <magnitude.nii[.gz]|-> <chi_out.nii.gz> <Hx> <Hy> <Hz> [alpha_l2 mu1_l2 tol_update]
use qsm_core::inversion::{hdqsm, HdQsmParams};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 8 && a.len() != 11 {
        eprintln!("usage: {} <field_ppm> <mask> <magnitude|-> <chi_out> <Hx> <Hy> <Hz> [alpha_l2 mu1_l2 tol_update]", a[0]);
        std::process::exit(2);
    }
    let num = |i: usize| -> f64 { a[i].parse().unwrap_or_else(|_| panic!("not a number: {}", a[i])) };
    let f = read_nifti_file(Path::new(&a[1])).expect("read field");
    let m = read_nifti_file(Path::new(&a[2])).expect("read mask");
    let mask: Vec<u8> = m.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let magnitude = (a[3] != "-").then(|| read_nifti_file(Path::new(&a[3])).expect("read magnitude").data);
    let grid = Grid::new(f.dims.0, f.dims.1, f.dims.2, f.voxel_size.0, f.voxel_size.1, f.voxel_size.2);
    let mut params = HdQsmParams::default();
    if a.len() == 11 {
        params.alpha_l2 = num(8);
        params.mu1_l2 = num(9);
        params.tol_update = num(10);
    }
    let mut last = 0;
    let t = std::time::Instant::now();
    let chi = hdqsm(&f.data, &mask, magnitude.as_deref(), &grid, (num(5), num(6), num(7)), &params, |i, _| last = i);
    eprintln!("hdqsm: {} iterations, {:.1}s", last, t.elapsed().as_secs_f64());
    save_nifti_to_file(Path::new(&a[4]), &chi, f.dims, f.voxel_size, &f.affine).expect("write chi");
}

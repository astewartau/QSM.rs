//! Run NLTV (QSM.jl nltv) on NIfTI files, for parity checks against QSM.jl in Julia.
//! Takes the local field in ppm and writes susceptibility in ppm, as the pipeline does.
//!
//! usage: nltv_file <field_ppm.nii[.gz]> <mask.nii[.gz]> <magnitude.nii[.gz]> <chi_out.nii.gz> <TE_ms> <B0_T> <Hx> <Hy> <Hz> [lambda] [max_iter]
use qsm_core::inversion::{nltv, NltvParams};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if !(10..=12).contains(&a.len()) {
        eprintln!("usage: {} <field_ppm> <mask> <magnitude> <chi_out> <TE_ms> <B0_T> <Hx> <Hy> <Hz> [lambda] [max_iter]", a[0]);
        std::process::exit(2);
    }
    let num = |i: usize| -> f64 { a[i].parse().unwrap_or_else(|_| panic!("not a number: {}", a[i])) };
    let f = read_nifti_file(Path::new(&a[1])).expect("read field");
    let m = read_nifti_file(Path::new(&a[2])).expect("read mask");
    let mask: Vec<u8> = m.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let magnitude = read_nifti_file(Path::new(&a[3])).expect("read magnitude").data;
    let grid = Grid::new(f.dims.0, f.dims.1, f.dims.2, f.voxel_size.0, f.voxel_size.1, f.voxel_size.2);
    let mut params = NltvParams { te: num(5) * 1e-3, b0: num(6), ..NltvParams::default() };
    if a.len() >= 11 {
        params.lambda = num(10);
    }
    if a.len() == 12 {
        params.max_iter = num(11) as usize;
    }
    eprintln!("voxel size {:?}", grid.voxel_size);
    let mut last = 0;
    let t = std::time::Instant::now();
    let chi = nltv(&f.data, &mask, &magnitude, &grid, (num(7), num(8), num(9)), &params, |i, _| last = i);
    eprintln!("nltv: {} iterations, {:.1}s", last, t.elapsed().as_secs_f64());
    save_nifti_to_file(Path::new(&a[4]), &chi, f.dims, f.voxel_size, &f.affine).expect("write chi");
}

//! Run MEDI (MEDI toolbox MEDI_L1.m) on NIfTI files, for parity checks against MEDI_L1 in MATLAB.
//! Takes the local field in ppm and writes susceptibility in ppm, as the pipeline does.
//!
//! usage: medi_file <field_ppm> <mask> <magnitude> <n_std|-> <chi_out> <delta_TE_ms> <B0_T> <Hx> <Hy> <Hz> [smv_radius_mm, 0 = no SMV] [merit|nomerit]
use qsm_core::inversion::{medi, MediParams};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 11 {
        eprintln!("usage: {} <field_ppm> <mask> <magnitude> <n_std|-> <chi_out> <delta_TE_ms> <B0_T> <Hx> <Hy> <Hz> [smv_radius_mm, 0 = no SMV] [merit|nomerit]", a[0]);
        std::process::exit(2);
    }
    let num = |i: usize| -> f64 { a[i].parse().unwrap_or_else(|_| panic!("not a number: {}", a[i])) };
    let f = read_nifti_file(Path::new(&a[1])).expect("read field");
    let m = read_nifti_file(Path::new(&a[2])).expect("read mask");
    let mask: Vec<u8> = m.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let mag = read_nifti_file(Path::new(&a[3])).expect("read magnitude").data;
    let n_std = (a[4] != "-").then(|| read_nifti_file(Path::new(&a[4])).expect("read n_std").data);
    let grid = Grid::new(f.dims.0, f.dims.1, f.dims.2, f.voxel_size.0, f.voxel_size.1, f.voxel_size.2);
    let mut params = MediParams { te: num(6) * 1e-3, b0: num(7), ..MediParams::default() };
    if a.len() > 11 {
        params.smv = num(11) > 0.0;
        params.smv_radius = if params.smv { num(11) } else { params.smv_radius };
    }
    if a.len() > 12 {
        params.merit = a[12] == "merit";
    }
    let t = std::time::Instant::now();
    let chi = medi(&f.data, &mask, &mag, n_std.as_deref(), &grid, (num(8), num(9), num(10)), &params, |_, _| {});
    eprintln!("medi: {:.1}s", t.elapsed().as_secs_f64());
    save_nifti_to_file(Path::new(&a[5]), &chi, f.dims, f.voxel_size, &f.affine).expect("write chi");
}

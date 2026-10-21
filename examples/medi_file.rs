//! Run MEDI (MEDI toolbox 2024.11.26 MEDI_L1.m) on NIfTI files, for parity checks against MEDI_L1
//! in MATLAB. Takes the local field in ppm and writes susceptibility in ppm, as the pipeline does.
//!
//! usage: medi_file <field_ppm> <mask> <magnitude> <chi_out> <delta_TE_ms> <B0_T> <Hx> <Hy> <Hz>
//!                  [smv_radius_mm, 0 = no SMV] [merit|nomerit] [zeropad: even|none]
//!
//! `<magnitude>` is `<echo1>,<echo2>,...`, the echo magnitudes (RSS edge mask and the
//! Fit_ppm_complex noise map).
use qsm_core::inversion::{medi, EchoMagnitudes, MediParams};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 10 {
        eprintln!("usage: {} <field_ppm> <mask> <echo1,echo2,..> <chi_out> <delta_TE_ms> <B0_T> <Hx> <Hy> <Hz> [smv_radius_mm] [merit|nomerit] [even|none]", a[0]);
        std::process::exit(2);
    }
    let num = |i: usize| -> f64 { a[i].parse().unwrap_or_else(|_| panic!("not a number: {}", a[i])) };
    let f = read_nifti_file(Path::new(&a[1])).expect("read field");
    let m = read_nifti_file(Path::new(&a[2])).expect("read mask");
    let mask: Vec<u8> = m.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let files = a[3].as_str();
    let mags: Vec<f64> = files.split(',')
        .flat_map(|p| read_nifti_file(Path::new(p)).expect("read magnitude").data)
        .collect();
    let n_echoes = files.split(',').count();
    let magnitude = EchoMagnitudes { data: &mags, n_echoes };
    let grid = Grid::new(f.dims.0, f.dims.1, f.dims.2, f.voxel_size.0, f.voxel_size.1, f.voxel_size.2);
    let mut params = MediParams { te: num(5) * 1e-3, b0: num(6), ..MediParams::default() };
    if a.len() > 10 {
        params.smv = num(10) > 0.0;
        params.smv_radius = if params.smv { num(10) } else { params.smv_radius };
    }
    if a.len() > 11 {
        params.merit = a[11] == "merit";
    }
    if a.len() > 12 && a[12] == "even" {
        params.zeropad = (f.dims.0 % 2, f.dims.1 % 2, f.dims.2 % 2);
    }
    let t = std::time::Instant::now();
    let chi = medi(&f.data, &mask, magnitude, &grid, (num(7), num(8), num(9)), &params, |_, _| {});
    eprintln!("medi: {:.1}s", t.elapsed().as_secs_f64());
    save_nifti_to_file(Path::new(&a[4]), &chi, f.dims, f.voxel_size, &f.affine).expect("write chi");
}

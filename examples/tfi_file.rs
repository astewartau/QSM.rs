//! Run TFI (the 2024 MEDI toolbox's MEDI_L1 with a preconditioner) on NIfTI files, for parity
//! checks against MEDI_L1 in MATLAB. Takes the total field in ppm and writes susceptibility in
//! ppm, as the pipeline does.
//!
//! usage: tfi_file <total_field_ppm> <mask> <magnitude> <chi_out> <delta_TE_ms> <B0_T> <Hx> <Hy> <Hz>
//!                 [merit|nomerit] [linear|exp]
//!
//! `<magnitude>` is `<echo1>,<echo2>,...`, the echo magnitudes (RSS edge mask and the
//! Fit_ppm_complex noise map).
use qsm_core::inversion::{tfi, EchoMagnitudes, TfiParams};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 10 {
        eprintln!("usage: {} <total_field_ppm> <mask> <echo1,echo2,..> <chi_out> <delta_TE_ms> <B0_T> <Hx> <Hy> <Hz> [merit|nomerit] [linear|exp]", a[0]);
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
    let mut params = TfiParams { te: num(5) * 1e-3, b0: num(6), ..TfiParams::default() };
    if a.len() > 10 {
        params.merit = a[10] == "merit";
    }
    if a.len() > 11 {
        params.linear = a[11] == "linear";
    }
    let t = std::time::Instant::now();
    let chi = tfi(&f.data, &mask, magnitude, &grid, (num(7), num(8), num(9)), &params, |_, _| {});
    eprintln!("tfi: {:.1}s", t.elapsed().as_secs_f64());
    save_nifti_to_file(Path::new(&a[4]), &chi, f.dims, f.voxel_size, &f.affine).expect("write chi");
}

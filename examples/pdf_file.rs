//! Run PDF (MEDI toolbox 2024.11.26 PDF.m) on NIfTI files, for parity checks against PDF.m in
//! MATLAB. Takes the total field in ppm and writes the local field in ppm, with the noise map
//! made from the echo magnitudes as the pipeline makes it.
//!
//! usage: pdf_file <field_ppm> <mask> <echo1,echo2,..> <local_out> <delta_TE_ms> <B0_T> <Hx> <Hy> <Hz>
use qsm_core::bgremove::{pdf, PdfParams};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 10 {
        eprintln!("usage: {} <field_ppm> <mask> <echo1,echo2,..> <local_out> <delta_TE_ms> <B0_T> <Hx> <Hy> <Hz>", a[0]);
        std::process::exit(2);
    }
    let num = |i: usize| -> f64 { a[i].parse().unwrap_or_else(|_| panic!("not a number: {}", a[i])) };
    let f = read_nifti_file(Path::new(&a[1])).expect("read field");
    let m = read_nifti_file(Path::new(&a[2])).expect("read mask");
    let mask: Vec<u8> = m.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let mags: Vec<f64> = a[3].split(',')
        .flat_map(|p| read_nifti_file(Path::new(p)).expect("read magnitude").data)
        .collect();
    let n_echoes = a[3].split(',').count();
    // N_std from Fit_ppm_complex (radians over delta_TE), in ppm like the field
    let rad_per_ppm = 2.0 * std::f64::consts::PI * 42.576e6 * num(6) * num(5) * 1e-3 * 1e-6;
    let n_std: Vec<f64> = qsm_core::utils::fit_ppm_complex_noise(&mags, n_echoes)
        .into_iter().map(|v| v / rad_per_ppm).collect();
    let grid = Grid::new(f.dims.0, f.dims.1, f.dims.2, f.voxel_size.0, f.voxel_size.1, f.voxel_size.2);
    let t = std::time::Instant::now();
    let mut iters = 0;
    let local = pdf(&f.data, &n_std, &mask, &grid, (num(7), num(8), num(9)), &PdfParams::default(), |i, _| iters = i)
        .expect("pdf");
    eprintln!("pdf: {:.1}s, {iters} CG iterations", t.elapsed().as_secs_f64());
    save_nifti_to_file(Path::new(&a[4]), &local, f.dims, f.voxel_size, &f.affine).expect("write");
}

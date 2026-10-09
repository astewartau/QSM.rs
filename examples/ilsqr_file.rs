//! Run iLSQR (STI Suite 3.0 algorithm) on NIfTI files, e.g. for parity checks against STI
//! Suite's own `QSM_iLSQR` in MATLAB. Takes the tissue phase in radians, as STI does, and
//! writes susceptibility in ppm (STI's scaling, gamma = 42.575 MHz/T).
//!
//! usage: ilsqr_file <phase_rad.nii[.gz]> <mask.nii[.gz]> <chi_out.nii.gz> <TE_ms> <B0_T> <Hx> <Hy> <Hz> [pad_mm]
//! (H = B0 direction in voxel axes; pad_mm = STI's 'padsize', default 64)
use qsm_core::inversion::{ilsqr_with_padding, IlsqrParams, STI_PAD_MM};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 9 && a.len() != 10 {
        eprintln!("usage: {} <phase_rad.nii[.gz]> <mask.nii[.gz]> <chi_out.nii.gz> <TE_ms> <B0_T> <Hx> <Hy> <Hz> [pad_mm]", a[0]);
        std::process::exit(2);
    }
    let num = |i: usize| -> f64 { a[i].parse().unwrap_or_else(|_| panic!("not a number: {}", a[i])) };
    let f = read_nifti_file(Path::new(&a[1])).expect("read phase");
    let m = read_nifti_file(Path::new(&a[2])).expect("read mask");
    let (te, b0, h) = (num(4), num(5), (num(6), num(7), num(8)));
    let pad = if a.len() == 10 { num(9) } else { STI_PAD_MM };
    let grid = Grid::new(f.dims.0, f.dims.1, f.dims.2, f.voxel_size.0, f.voxel_size.1, f.voxel_size.2);
    let mask: Vec<u8> = m.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let t = std::time::Instant::now();
    let (chi, _, _, _) = ilsqr_with_padding(&f.data, &mask, &grid, h, &IlsqrParams::default(), [pad; 3], |s, n| eprintln!("  step {}/{}", s, n));
    let to_ppm = 1.0 / (2.0 * std::f64::consts::PI * 42.575 * b0 * te * 1e-3);
    let chi: Vec<f64> = chi.iter().map(|v| v * to_ppm).collect();
    eprintln!("ilsqr: {:?} vs {:?}, {:.1}s", f.dims, f.voxel_size, t.elapsed().as_secs_f64());
    save_nifti_to_file(Path::new(&a[3]), &chi, f.dims, f.voxel_size, &f.affine).expect("write chi");
}

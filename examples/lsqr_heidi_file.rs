//! Run LSQR (Schweser 2010) and HEIDI (Schweser 2012) on NIfTI files, for parity checks against
//! the authors' qsm_heidi code in MATLAB. Takes the local field in ppm and writes ppm.
//!
//! usage: lsqr_heidi_file <lsqr|heidi> <field_ppm> <mask> <magnitude|-> <chi_out.nii.gz> <B0_T> <Hx> <Hy> <Hz> [TE_ms]
//! (H = B0 direction in voxel axes; `LSQR_MAX_ITER` / `LSQR_TOL` override the LSQR defaults,
//! `HEIDI_FLOOR` HEIDI's `gradient_mask_floor`)
use qsm_core::inversion::{heidi, lsqr_qsm, HeidiParams, LsqrQsmParams};
use qsm_core::EchoTime;
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 10 && a.len() != 11 {
        eprintln!("usage: {} <lsqr|heidi> <field_ppm> <mask> <magnitude|-> <chi_out> <B0_T> <Hx> <Hy> <Hz> [TE_ms]", a[0]);
        std::process::exit(2);
    }
    let num = |i: usize| -> f64 { a[i].parse().unwrap_or_else(|_| panic!("not a number: {}", a[i])) };
    let f = read_nifti_file(Path::new(&a[2])).expect("read field");
    let m = read_nifti_file(Path::new(&a[3])).expect("read mask");
    let mask: Vec<u8> = m.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let magnitude = (a[4] != "-").then(|| read_nifti_file(Path::new(&a[4])).expect("read magnitude").data);
    let grid = Grid::new(f.dims.0, f.dims.1, f.dims.2, f.voxel_size.0, f.voxel_size.1, f.voxel_size.2);
    let bdir = (num(7), num(8), num(9));
    let t = std::time::Instant::now();
    let mut lsqr = LsqrQsmParams { b0: num(6), mask_output: a[1] != "heidi", ..Default::default() };
    if let Ok(v) = std::env::var("LSQR_MAX_ITER") {
        lsqr.max_iter = v.parse().expect("LSQR_MAX_ITER");
    }
    if let Ok(v) = std::env::var("LSQR_TOL") {
        lsqr.tol = v.parse().expect("LSQR_TOL");
    }
    let mut heidi_params = HeidiParams { b0: num(6), ..HeidiParams::default() };
    if a.len() == 11 {
        heidi_params.te = EchoTime::Seconds(num(10) * 1e-3);
    }
    if let Ok(v) = std::env::var("HEIDI_FLOOR") {
        heidi_params.gradient_mask_floor = v.parse().expect("HEIDI_FLOOR");
    }
    let chi = if a[1] == "heidi" {
        let mut last = 0;
        let chi = heidi(&f.data, &mask, magnitude.as_deref(), &grid, bdir, &lsqr, &heidi_params, |i, _| last = i);
        eprintln!("lsqr + heidi: {} iterations, {:.1}s", last, t.elapsed().as_secs_f64());
        chi
    } else {
        let chi = lsqr_qsm(&f.data, &mask, magnitude.as_deref(), &grid, bdir, &lsqr, |_, _| {});
        eprintln!("lsqr: {:.1}s", t.elapsed().as_secs_f64());
        chi
    };
    save_nifti_to_file(Path::new(&a[5]), &chi, f.dims, f.voxel_size, &f.affine).expect("write chi");
}

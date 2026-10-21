//! Run FANSI nlTV / nlTGV or L1-QSM (nlL1TV) on NIfTI files, for parity checks against FANSI in
//! MATLAB. Takes the local field in ppm and writes susceptibility in ppm, as the pipeline does.
//!
//! usage: fansi_file <nltv|nltgv|l1qsm> <field_ppm.nii[.gz]> <mask.nii[.gz]> <magnitude.nii[.gz]|-> <chi_out.nii.gz> <TE_ms> <B0_T> <Hx> <Hy> <Hz> [alpha1] [lambda]
use qsm_core::inversion::{fansi, l1qsm, FansiParams, L1QsmParams};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if !(11..=13).contains(&a.len()) {
        eprintln!("usage: {} <nltv|nltgv|l1qsm> <field_ppm> <mask> <magnitude|-> <chi_out> <TE_ms> <B0_T> <Hx> <Hy> <Hz> [alpha1] [lambda]", a[0]);
        std::process::exit(2);
    }
    let num = |i: usize| -> f64 { a[i].parse().unwrap_or_else(|_| panic!("not a number: {}", a[i])) };
    let f = read_nifti_file(Path::new(&a[2])).expect("read field");
    let m = read_nifti_file(Path::new(&a[3])).expect("read mask");
    let mask: Vec<u8> = m.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let magnitude = (a[4] != "-").then(|| read_nifti_file(Path::new(&a[4])).expect("read magnitude").data);
    let grid = Grid::new(f.dims.0, f.dims.1, f.dims.2, f.voxel_size.0, f.voxel_size.1, f.voxel_size.2);
    let (te, b0, bdir) = (num(6) * 1e-3, num(7), (num(8), num(9), num(10)));
    let mut last = 0;
    let t = std::time::Instant::now();
    let chi = match a[1].as_str() {
        "l1qsm" => {
            let mut p = L1QsmParams { te, b0, ..Default::default() };
            if a.len() > 11 { p.alpha1 = num(11); }
            if a.len() > 12 { p.lambda = num(12); }
            l1qsm(&f.data, &mask, magnitude.as_deref(), &grid, bdir, &p, |i, _| last = i)
        }
        method @ ("nltv" | "nltgv") => {
            let mut p = FansiParams { te, b0, is_tgv: method == "nltgv", ..Default::default() };
            if a.len() > 11 { p.alpha1 = num(11); }
            fansi(&f.data, &mask, magnitude.as_deref(), &grid, bdir, &p, |i, _| last = i)
        }
        other => panic!("unknown method {other}"),
    };
    eprintln!("{}: {} iterations, {:.1}s", a[1], last, t.elapsed().as_secs_f64());
    save_nifti_to_file(Path::new(&a[5]), &chi, f.dims, f.voxel_size, &f.affine).expect("write chi");
}

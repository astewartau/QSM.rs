//! Run STAR-QSM (STI Suite 3.0's `QSM_star`) on NIfTI files, e.g. for parity checks against STI
//! Suite in MATLAB. Takes the tissue phase in radians, as STI does, and writes susceptibility in
//! ppm.
//!
//! usage: star_file <phase_rad.nii[.gz]> <mask.nii[.gz]> <chi_out.nii.gz> <TE_ms> <B0_T> <Hx> <Hy> <Hz> [px,py,pz [vx vy vz]]
//! (H = B0 direction in voxel axes; px,py,pz = STI's 'padsize' in voxels per axis, default
//! 8,8,20 as STI Suite's GUI passes it; vx vy vz = voxel size in mm, overriding the header's,
//! e.g. for files written by MATLAB's `niftiwrite`)
//!
//! Prints the iteration count and run time. With `STAR_RAW=<path>` it also writes χ (ppm, f64,
//! little-endian) for bit-level comparisons, since the NIfTI output may be single precision.
use qsm_core::inversion::{star_qsm, StarParams};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if ![9, 10, 13].contains(&a.len()) {
        eprintln!("usage: {} <phase_rad.nii[.gz]> <mask.nii[.gz]> <chi_out.nii.gz> <TE_ms> <B0_T> <Hx> <Hy> <Hz> [px,py,pz [vx vy vz]]", a[0]);
        std::process::exit(2);
    }
    let num = |i: usize| -> f64 { a[i].parse().unwrap_or_else(|_| panic!("not a number: {}", a[i])) };
    let f = read_nifti_file(Path::new(&a[1])).expect("read phase");
    let m = read_nifti_file(Path::new(&a[2])).expect("read mask");
    let mut params = StarParams { te: num(4) * 1e-3, b0: num(5), ..StarParams::default() };
    if a.len() >= 10 {
        let p: Vec<usize> = a[9].split(',').map(|v| v.parse().expect("pad must be whole numbers of voxels")).collect();
        params.pad = p.try_into().expect("pad must be three values, px,py,pz");
    }
    let h = (num(6), num(7), num(8));
    let vs = if a.len() == 13 { (num(10), num(11), num(12)) } else { f.voxel_size };
    let grid = Grid::new(f.dims.0, f.dims.1, f.dims.2, vs.0, vs.1, vs.2);
    let mask: Vec<u8> = m.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let field: Vec<f64> = f.data.iter().map(|&v| v / params.rad_per_ppm()).collect();
    let t = std::time::Instant::now();
    let mut iters = 0;
    let chi = star_qsm(&field, &mask, &grid, h, &params, |i, _| iters = i);
    eprintln!("star: {:?} vs {:?} (pad {:?}), {} iterations, {:.2}s", f.dims, vs, params.pad, iters, t.elapsed().as_secs_f64());
    if let Ok(raw) = std::env::var("STAR_RAW") {
        let bytes: Vec<u8> = chi.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&raw, bytes).expect("write raw");
    }
    save_nifti_to_file(Path::new(&a[3]), &chi, f.dims, vs, &f.affine).expect("write chi");
}

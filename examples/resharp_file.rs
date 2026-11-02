//! Run RESHARP (sunhongfu/QSM resharp.m) on NIfTI files at resharp.m's defaults, for parity
//! checks against resharp.m in MATLAB.
//!
//! usage: resharp_file <field> <mask> <local_out> [radius_mm] [lambda] [max_iter]
use qsm_core::bgremove::{resharp, ResharpParams};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: {} <field> <mask> <local_out> [radius_mm] [lambda] [max_iter]", a[0]);
        std::process::exit(2);
    }
    let f = read_nifti_file(Path::new(&a[1])).expect("read field");
    let m = read_nifti_file(Path::new(&a[2])).expect("read mask");
    let mask: Vec<u8> = m.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let mut p = ResharpParams::default();
    if let Some(v) = a.get(4) { p.radius = v.parse().unwrap(); }
    if let Some(v) = a.get(5) { p.tik_reg = v.parse().unwrap(); }
    if let Some(v) = a.get(6) { p.max_iter = v.parse().unwrap(); }
    let grid = Grid::new(f.dims.0, f.dims.1, f.dims.2, f.voxel_size.0, f.voxel_size.1, f.voxel_size.2);
    let t = std::time::Instant::now();
    let mut iters = 0;
    let (local, _) = resharp(&f.data, &mask, &grid, &p, |i, _| iters = i + 1);
    eprintln!("resharp: {:.1}s, {iters} cgs iterations", t.elapsed().as_secs_f64());
    save_nifti_to_file(Path::new(&a[3]), &local, f.dims, f.voxel_size, &f.affine).expect("write");
}

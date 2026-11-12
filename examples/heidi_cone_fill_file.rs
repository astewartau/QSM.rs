//! HEIDI's cone fill on any seed χ map (streaking reduction after a dipole inversion), on NIfTI
//! files: the measurement behind the `heidi.seed-*` rules (QSM.rs heidi.rs docs, *Any seed*).
//! Takes the local field in ppm and the seed in ppm, and writes ppm. HEIDI runs at its defaults
//! with the scan's field strength and echo times (the cone at the last echo time, as the
//! pipeline's `heidi.te-last`).
//!
//! usage:
//!   heidi_cone_fill_file fill <field_ppm> <mask> <seed_chi> <chi_out> <B0_T> <Hx> <Hy> <Hz> <TE_s>...
//!   heidi_cone_fill_file <tkd|tsvd> <field_ppm> <mask> <threshold> <chi_out> <B0_T> <Hx> <Hy> <Hz>
//! (H = B0 direction in voxel axes; `tkd`/`tsvd` write the seed at the given threshold, at its
//! other defaults, e.g. 0.1 so that HEIDI accepts it)
use qsm_core::inversion::{heidi_cone_fill, tkd, tsvd, HeidiParams, TkdParams, TsvdParams};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 10 {
        eprintln!("usage: {} <fill|tkd|tsvd> <field_ppm> <mask> <seed|threshold> <chi_out> <B0_T> <Hx> <Hy> <Hz> [TE_s...]", a[0]);
        std::process::exit(2);
    }
    let num = |s: &str| -> f64 { s.parse().unwrap_or_else(|_| panic!("not a number: {s}")) };
    let f = read_nifti_file(Path::new(&a[2])).expect("read field");
    let m = read_nifti_file(Path::new(&a[3])).expect("read mask");
    let mask: Vec<u8> = m.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let grid = Grid::new(f.dims.0, f.dims.1, f.dims.2, f.voxel_size.0, f.voxel_size.1, f.voxel_size.2);
    let bdir = (num(&a[7]), num(&a[8]), num(&a[9]));
    let t = std::time::Instant::now();
    let chi = match a[1].as_str() {
        "fill" => {
            let seed = read_nifti_file(Path::new(&a[4])).expect("read seed");
            let seed: Vec<f64> = seed.data.iter().map(|&v| if v.is_finite() { v } else { 0.0 }).collect();
            let echo_times: Vec<f64> = a[10..].iter().map(|s| num(s)).collect();
            assert!(!echo_times.is_empty(), "fill needs the echo times (s)");
            let params = HeidiParams { b0: num(&a[6]), echo_times, ..HeidiParams::default() };
            heidi_cone_fill(&f.data, &mask, &seed, &grid, bdir, &params, |_, _| {})
        }
        "tkd" | "tsvd" => {
            if a[1] == "tkd" {
                tkd(&f.data, &mask, &grid, bdir, &TkdParams { threshold: num(&a[4]), ..TkdParams::default() })
            } else {
                tsvd(&f.data, &mask, &grid, bdir, &TsvdParams { threshold: num(&a[4]), ..TsvdParams::default() })
            }
        }
        other => panic!("unknown mode {other}"),
    };
    eprintln!("{}: {:.1}s", a[1], t.elapsed().as_secs_f64());
    save_nifti_to_file(Path::new(&a[5]), &chi, f.dims, f.voxel_size, &f.affine).expect("write chi");
}

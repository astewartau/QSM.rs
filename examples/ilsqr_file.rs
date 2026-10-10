//! Run iLSQR (STI Suite 3.0 algorithm) on NIfTI files, e.g. for parity checks against STI
//! Suite's own `QSM_iLSQR` in MATLAB. Takes the tissue phase in radians, as STI does, and
//! writes susceptibility in ppm (STI's scaling, gamma = 42.575 MHz/T).
//!
//! usage: ilsqr_file <phase_rad.nii[.gz]> <mask.nii[.gz]> <chi_out.nii.gz> <TE_ms> <B0_T> <Hx> <Hy> <Hz> [pad_mm]
//! (H = B0 direction in voxel axes; pad_mm = STI's 'padsize', default 64)
//!
//! Prints the time per stage, the FFT count and how both LSQR solves ended. With
//! `ILSQR_RAW=<path>` it also writes χ and the initial solution (ppm, f64, little-endian, χ
//! first) for bit-level comparisons, since the NIfTI output may be single precision.
use qsm_core::inversion::{ilsqr_with_padding_traced, IlsqrParams, IlsqrTrace, STI_PAD_MM};
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
    let mut tr = IlsqrTrace::default();
    let (chi, _, _, x0) = ilsqr_with_padding_traced(&f.data, &mask, &grid, h, &IlsqrParams::default(), [pad; 3], |_, _| {}, &mut tr);
    let to_ppm = 1.0 / (2.0 * std::f64::consts::PI * 42.575 * b0 * te * 1e-3);
    let chi: Vec<f64> = chi.iter().map(|v| v * to_ppm).collect();
    eprintln!("ilsqr: {:?} vs {:?} (padded {:?}), {:.2}s", f.dims, f.voxel_size, tr.dims, t.elapsed().as_secs_f64());
    for (name, s, nf, tf) in &tr.stages {
        eprintln!("  {:>7.3}s  {:<34} ({} FFTs, {:.3}s)", s, name, nf, tf);
    }
    eprintln!("  FFTs: {} in {:.2}s ({:.1} ms each)", tr.n_fft, tr.fft_secs, 1e3 * tr.fft_secs / tr.n_fft.max(1) as f64);
    for (i, o) in tr.solves.iter().enumerate() {
        eprintln!("  solve {}: flag {} iter {} relres {:.17e}", i + 1, o.flag, o.iter, o.relres);
    }
    if let Ok(raw) = std::env::var("ILSQR_RAW") {
        let bytes: Vec<u8> = chi.iter().chain(x0.iter().map(|v| v * to_ppm).collect::<Vec<_>>().iter()).flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&raw, bytes).expect("write raw");
    }
    save_nifti_to_file(Path::new(&a[3]), &chi, f.dims, f.voxel_size, &f.affine).expect("write chi");
}

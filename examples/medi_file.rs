//! Run MEDI (MEDI toolbox 2024.11.26 MEDI_L1.m) on NIfTI files, for parity checks against MEDI_L1
//! in MATLAB. Takes the local field in ppm and writes susceptibility in ppm, as the pipeline does.
//!
//! usage: medi_file <field_ppm> <mask> <magnitude> <chi_out> <TE1_ms,TE2_ms,..> <B0_T> <Hx> <Hy> <Hz>
//!                  [smv_radius_mm, 0 = no SMV] [merit|nomerit] [zeropad: even|none]
//!                  [csf|nocsf] [csf_mask_out]
//!
//! `<magnitude>` is `<echo1>,<echo2>,...`, the echo magnitudes (RSS edge mask, the
//! Fit_ppm_complex noise map and, for MEDI+0, the arlo R2* behind the CSF mask).
//!
//! Flags (anywhere): `--msmv` runs MEDI-mSMV (`MEDI_L1(..., 'msmv', radius)`, radius 5 mm unless
//! the SMV radius argument is given), `--msmv-noprefilter` the same with `msmv.m`'s prefilter
//! off; `--max-iter=N`, `--cg-tol=X` and `--cg-max-iter=N` set MEDI's Gauss-Newton iterations,
//! CG tolerance and CG iterations.
use qsm_core::inversion::{extract_csf, medi, EchoMagnitudes, MediParams, MsmvParams};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let all: Vec<String> = std::env::args().collect();
    let flags: Vec<&String> = all.iter().filter(|x| x.starts_with("--")).collect();
    let flag_value = |name: &str| -> Option<f64> {
        flags.iter().find_map(|f| f.strip_prefix(name).and_then(|v| v.strip_prefix('=')).map(|v| v.parse().expect("flag value")))
    };
    let a: Vec<String> = all.iter().filter(|x| !x.starts_with("--")).cloned().collect();
    if a.len() < 10 {
        eprintln!("usage: {} <field_ppm> <mask> <echo1,echo2,..> <chi_out> <delta_TE_ms> <B0_T> <Hx> <Hy> <Hz> [smv_radius_mm] [merit|nomerit] [even|none] [csf|nocsf] [csf_mask_out]", a[0]);
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
    let echo_times: Vec<f64> = a[5].split(',').map(|t| t.parse::<f64>().expect("TE") * 1e-3).collect();
    let mut params = MediParams { echo_times, b0: num(6), ..MediParams::default() };
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
    if a.len() > 13 {
        params.csf = a[13] == "csf";
    }
    if a.len() > 14 {
        let n = grid.n_total();
        let r2s = qsm_core::r2star::arlo(&mags, &params.echo_times, n).expect("two or more echoes");
        let csf = extract_csf(&r2s, &mask, &grid, params.csf_r2s_threshold, params.csf_erode);
        let csf: Vec<f64> = csf.iter().map(|&v| v as f64).collect();
        save_nifti_to_file(Path::new(&a[14]), &csf, f.dims, f.voxel_size, &f.affine).expect("write csf");
        let r2s_path = a[14].replace(".nii", "_r2s.nii");
        save_nifti_to_file(Path::new(&r2s_path), &r2s, f.dims, f.voxel_size, &f.affine).expect("write r2s");
    }
    if flags.iter().any(|f| f.starts_with("--msmv")) {
        let prefilter = !flags.iter().any(|f| *f == "--msmv-noprefilter");
        params.msmv = Some(MsmvParams { prefilter, ..MsmvParams::default() });
    }
    if let Some(v) = flag_value("--max-iter") {
        params.max_iter = v as usize;
    }
    if let Some(v) = flag_value("--cg-tol") {
        params.cg_tol = v;
    }
    if let Some(v) = flag_value("--cg-max-iter") {
        params.cg_max_iter = v as usize;
    }
    let t = std::time::Instant::now();
    let chi = medi(&f.data, &mask, magnitude, &grid, (num(7), num(8), num(9)), &params, |_, _| {});
    eprintln!("medi: {:.1}s", t.elapsed().as_secs_f64());
    save_nifti_to_file(Path::new(&a[4]), &chi, f.dims, f.voxel_size, &f.affine).expect("write chi");
}

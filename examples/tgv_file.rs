//! Run TGV-QSM on NIfTI files, for parity checks against QuantitativeSusceptibilityMappingTGV.jl.
//! `field` input is a field in ppm, `phase` input one wrapped echo in rad; TE and B0 are those
//! of the echo (for a field, the TE it is expressed at). Writes χ in ppm.
//!
//! usage: tgv_file <field|phase> <input.nii[.gz]> <mask.nii[.gz]> <chi_out.nii.gz> <TE_s> <B0_T> <Hx> <Hy> <Hz> [iterations]
use qsm_core::inversion::{tgv_qsm, TgvInput, TgvParams};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 10 && a.len() != 11 {
        eprintln!("usage: {} <field|phase> <input> <mask> <chi_out> <TE_s> <B0_T> <Hx> <Hy> <Hz> [iterations]", a[0]);
        std::process::exit(2);
    }
    let num = |i: usize| -> f64 { a[i].parse().unwrap_or_else(|_| panic!("not a number: {}", a[i])) };
    let f = read_nifti_file(Path::new(&a[2])).expect("read input");
    let m = read_nifti_file(Path::new(&a[3])).expect("read mask");
    let mask: Vec<u8> = m.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let grid = Grid::new(f.dims.0, f.dims.1, f.dims.2, f.voxel_size.0, f.voxel_size.1, f.voxel_size.2);
    let (te_s, b0_t) = (num(5), num(6));
    let input = match a[1].as_str() {
        "field" => TgvInput::FieldPpm { field: &f.data, echo_times: &[te_s], b0_t },
        "phase" => TgvInput::WrappedPhase { phase: &f.data, te_s, b0_t },
        other => panic!("input kind must be field or phase, got {other}"),
    };
    let iterations = (a.len() == 11).then(|| num(10) as usize);
    let params = TgvParams { iterations, ..TgvParams::default() };
    eprintln!("tgv params: {params:?}");
    let t = std::time::Instant::now();
    let chi = tgv_qsm(input, &mask, &grid, &params, (num(7), num(8), num(9)), |_, _| {}).expect("tgv");
    eprintln!("tgv: {:.1}s", t.elapsed().as_secs_f64());
    save_nifti_to_file(Path::new(&a[4]), &chi, f.dims, f.voxel_size, &f.affine).expect("write chi");
}

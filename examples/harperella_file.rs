//! Run HARPERELLA or iHARPERELLA on a total field in ppm, e.g. for parity checks.
//!
//! usage: harperella_file <h|ih> <field_ppm.nii[.gz]> <mask.nii[.gz]> <local_out.nii[.gz]>
//!
//! Default parameters, except `HARPERELLA_ITER=<n>` (max iterations) and `HARPERELLA_PAD=<x,y,z>`
//! (zero padding in voxels per side). With `HARPERELLA_RAW=<path>` it also writes the local field as
//! little-endian f64, since the NIfTI output may be single precision.
use qsm_core::bgremove::{harperella, iharperella, HarperellaInput, HarperellaParams, IharperellaParams};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 5 {
        eprintln!("usage: {} <h|ih> <field_ppm.nii[.gz]> <mask.nii[.gz]> <local_out.nii[.gz]>", a[0]);
        std::process::exit(2);
    }
    let f = read_nifti_file(Path::new(&a[2])).expect("read field");
    let m = read_nifti_file(Path::new(&a[3])).expect("read mask");
    assert_eq!(f.dims, m.dims, "field and mask dimensions differ");
    let grid = Grid::new(f.dims.0, f.dims.1, f.dims.2, f.voxel_size.0, f.voxel_size.1, f.voxel_size.2);
    let mask: Vec<u8> = m.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let input = HarperellaInput::FieldPpm(&f.data);
    let t = std::time::Instant::now();
    let iter = std::env::var("HARPERELLA_ITER").ok().map(|v| v.parse::<usize>().expect("HARPERELLA_ITER"));
    let pad = std::env::var("HARPERELLA_PAD").ok().map(|v| {
        let p: Vec<usize> = v.split(',').map(|x| x.parse().expect("HARPERELLA_PAD")).collect();
        [p[0], p[1], p[2]]
    });
    let mut hp = HarperellaParams::default();
    let mut ip = IharperellaParams::default();
    if let Some(n) = iter { hp.max_iter = n; ip.max_iter = n; }
    if let Some(p) = pad { hp.pad = p; ip.pad = p; }
    let (local, _) = match a[1].as_str() {
        "h" => harperella(input, &mask, &grid, &hp, |_, _| {}),
        "ih" => iharperella(input, &mask, &grid, &ip, |_, _| {}),
        s => panic!("method must be h or ih, got {s}"),
    }
    .expect("harperella");
    eprintln!("{}: {:?} {:?}, {:.2}s", a[1], f.dims, f.voxel_size, t.elapsed().as_secs_f64());
    if let Ok(raw) = std::env::var("HARPERELLA_RAW") {
        let bytes: Vec<u8> = local.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&raw, bytes).expect("write raw");
    }
    save_nifti_to_file(Path::new(&a[4]), &local, f.dims, f.voxel_size, &f.affine).expect("write local field");
}

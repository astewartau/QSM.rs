//! Run iSMV on NIfTI files, e.g. for parity checks against other implementations.
//!
//! usage: ismv_file <field.nii[.gz]> <mask.nii[.gz]> <local_out.nii[.gz]> [wen2014|qsmjl]
//!
//! The variant's own defaults (`IsmvParams::default()` or `IsmvParams::qsmjl()`). The local field
//! is in the units of the input field; the voxel size is the header's, as read. Prints the radius,
//! the eroded voxel count and the run time.
use qsm_core::bgremove::{ismv, IsmvParams};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if !(4..=5).contains(&a.len()) {
        eprintln!("usage: {} <field.nii[.gz]> <mask.nii[.gz]> <local_out.nii[.gz]> [wen2014|qsmjl]", a[0]);
        std::process::exit(2);
    }
    let params = match a.get(4).map(String::as_str) {
        None | Some("wen2014") => IsmvParams::default(),
        Some("qsmjl") => IsmvParams::qsmjl(),
        Some(s) => panic!("variant must be wen2014 or qsmjl, got {s}"),
    };
    let f = read_nifti_file(Path::new(&a[1])).expect("read field");
    let m = read_nifti_file(Path::new(&a[2])).expect("read mask");
    assert_eq!(f.dims, m.dims, "field and mask dimensions differ");
    let grid = Grid::new(f.dims.0, f.dims.1, f.dims.2, f.voxel_size.0, f.voxel_size.1, f.voxel_size.2);
    let mask: Vec<u8> = m.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let t = std::time::Instant::now();
    let (local, eroded) = ismv(&f.data, &mask, &grid, &params, |_, _| {}).unwrap();
    eprintln!(
        "ismv ({}): {:?} vs {:?}, radius {} mm, {} eroded voxels, {:.2}s",
        params.variant.id(), f.dims, f.voxel_size, params.radius_mm(&grid),
        eroded.iter().filter(|&&v| v != 0).count(), t.elapsed().as_secs_f64()
    );
    save_nifti_to_file(Path::new(&a[3]), &local, f.dims, f.voxel_size, &f.affine).expect("write local field");
}

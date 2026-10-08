//! Run V-SHARP (STI Suite 3.0 algorithm, default parameters) on NIfTI files, e.g. for parity
//! checks against STI Suite's own `V_SHARP` in MATLAB.
//!
//! usage: vsharp_file <field.nii[.gz]> <mask.nii[.gz]> <local_out.nii.gz> <mask_out.nii.gz>
use qsm_core::bgremove::{vsharp, VsharpParams};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 5 {
        eprintln!("usage: {} <field.nii[.gz]> <mask.nii[.gz]> <local_out.nii.gz> <mask_out.nii.gz>", a[0]);
        std::process::exit(2);
    }
    let f = read_nifti_file(Path::new(&a[1])).expect("read field");
    let m = read_nifti_file(Path::new(&a[2])).expect("read mask");
    let grid = Grid::new(f.dims.0, f.dims.1, f.dims.2, f.voxel_size.0, f.voxel_size.1, f.voxel_size.2);
    let mask: Vec<u8> = m.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let t = std::time::Instant::now();
    let (local, out) = vsharp(&f.data, &mask, &grid, &VsharpParams::default(), |_, _| {});
    eprintln!("vsharp: {:?} {:.1}s, mask {} -> {}", f.dims, t.elapsed().as_secs_f64(),
              mask.iter().filter(|&&v| v != 0).count(), out.iter().filter(|&&v| v != 0).count());
    save_nifti_to_file(Path::new(&a[3]), &local, f.dims, f.voxel_size, &f.affine).expect("write local field");
    let o: Vec<f64> = out.iter().map(|&v| v as f64).collect();
    save_nifti_to_file(Path::new(&a[4]), &o, f.dims, f.voxel_size, &f.affine).expect("write mask");
}

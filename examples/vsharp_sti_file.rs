//! Run STI Suite-compatible V-SHARP on NIfTI files (parity checks against MATLAB).
//! usage: vsharp_sti_file <field.nii[.gz]> <mask.nii[.gz]> <local_out.nii.gz> <mask_out.nii.gz>
use qsm_core::bgremove::{vsharp_sti, VsharpStiParams};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let f = read_nifti_file(Path::new(&a[1])).expect("field");
    let m = read_nifti_file(Path::new(&a[2])).expect("mask");
    let grid = Grid::new(f.dims.0, f.dims.1, f.dims.2, f.voxel_size.0, f.voxel_size.1, f.voxel_size.2);
    let mask: Vec<u8> = m.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let t = std::time::Instant::now();
    let (local, out) = vsharp_sti(&f.data, &mask, &grid, &VsharpStiParams::default(), |_, _| {});
    eprintln!("vsharp_sti: {:?} {:.1}s, mask {} -> {}", f.dims, t.elapsed().as_secs_f64(),
              mask.iter().filter(|&&v| v != 0).count(), out.iter().filter(|&&v| v != 0).count());
    save_nifti_to_file(Path::new(&a[3]), &local, f.dims, f.voxel_size, &f.affine).unwrap();
    let o: Vec<f64> = out.iter().map(|&v| v as f64).collect();
    save_nifti_to_file(Path::new(&a[4]), &o, f.dims, f.voxel_size, &f.affine).unwrap();
}

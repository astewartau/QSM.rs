//! Run AMP-PE on NIfTI files. usage: amppe_file <field_ppm> <mask> <magnitude|-> <out> <B0_T> <Hx> <Hy> <Hz> [norm|raw] [morphology_magnitude]
use qsm_core::inversion::{amp_pe, AmpPeParams};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::Grid;
use std::path::Path;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let num = |i: usize| -> f64 { a[i].parse().unwrap() };
    let f = read_nifti_file(Path::new(&a[1])).unwrap();
    let mask: Vec<u8> = read_nifti_file(Path::new(&a[2])).unwrap().data.iter().map(|&v| (v > 0.5) as u8).collect();
    let mag = (a[3] != "-").then(|| read_nifti_file(Path::new(&a[3])).unwrap().data);
    let grid = Grid::new(f.dims.0, f.dims.1, f.dims.2, f.voxel_size.0, f.voxel_size.1, f.voxel_size.2);
    let morph = a.get(10).map(|f| read_nifti_file(Path::new(f)).unwrap().data);
    let p = AmpPeParams { b0: num(5), normalize_weight: a.get(9).is_some_and(|s| s == "norm"), ..Default::default() };
    let t = std::time::Instant::now();
    let chi = amp_pe(&f.data, &mask, mag.as_deref(), morph.as_deref(), &grid, (num(6), num(7), num(8)), &p, |_, _| {}).unwrap();
    eprintln!("amp_pe {:.1}s", t.elapsed().as_secs_f64());
    save_nifti_to_file(Path::new(&a[4]), &chi, f.dims, f.voxel_size, &f.affine).unwrap();
}

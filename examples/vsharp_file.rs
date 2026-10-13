//! Run V-SHARP on NIfTI files, e.g. for parity checks against other implementations.
//!
//! usage: vsharp_file <field.nii[.gz]> <mask.nii[.gz]> <local_out.nii[.gz]> <support_out.nii[.gz]> [grid] [deconv]
//! - grid: `auto` (default; 1 mm isotropic for anisotropic voxels, else native), `native`, or
//!   `iso:<mm>` (isotropic working grid of that spacing)
//! - deconv: `deconv` or `nodeconv` (default: `VsharpParams::default()`)
//!
//! The local field is in the units of the input field. Prints the resolved working grid and the
//! run time. With `VSHARP_RAW=<path>` it also writes the local field as little-endian f64 for
//! bit-level comparisons, since the NIfTI output may be single precision.
use qsm_core::bgremove::{vsharp, VsharpGrid, VsharpParams};
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if !(5..=7).contains(&a.len()) {
        eprintln!("usage: {} <field.nii[.gz]> <mask.nii[.gz]> <local_out.nii[.gz]> <support_out.nii[.gz]> [auto|native|iso:<mm>] [deconv|nodeconv]", a[0]);
        std::process::exit(2);
    }
    let mut params = VsharpParams::default();
    if let Some(g) = a.get(5) {
        params.grid = match g.as_str() {
            "auto" => None,
            "native" => Some(VsharpGrid::Native),
            s => match s.strip_prefix("iso:").and_then(|v| v.parse().ok()) {
                Some(spacing_mm) => Some(VsharpGrid::Isotropic { spacing_mm }),
                None => panic!("grid must be auto, native or iso:<mm>, got {s}"),
            },
        };
    }
    if let Some(d) = a.get(6) {
        params.deconvolve = match d.as_str() {
            "deconv" => true,
            "nodeconv" => false,
            s => panic!("deconv must be deconv or nodeconv, got {s}"),
        };
    }
    let f = read_nifti_file(Path::new(&a[1])).expect("read field");
    let m = read_nifti_file(Path::new(&a[2])).expect("read mask");
    assert_eq!(f.dims, m.dims, "field and mask dimensions differ");
    let grid = Grid::new(f.dims.0, f.dims.1, f.dims.2, f.voxel_size.0, f.voxel_size.1, f.voxel_size.2);
    let mask: Vec<u8> = m.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let t = std::time::Instant::now();
    let (local, support) = vsharp(&f.data, &mask, &grid, &params, |_, _| {});
    eprintln!(
        "vsharp: {:?} vs {:?}, grid {:?}, deconvolve {}, {} support voxels, {:.2}s",
        f.dims, f.voxel_size, params.working_grid(&grid), params.deconvolve,
        support.iter().filter(|&&v| v != 0).count(), t.elapsed().as_secs_f64()
    );
    if let Ok(raw) = std::env::var("VSHARP_RAW") {
        let bytes: Vec<u8> = local.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&raw, bytes).expect("write raw");
    }
    let support: Vec<f64> = support.iter().map(|&v| v as f64).collect();
    save_nifti_to_file(Path::new(&a[3]), &local, f.dims, f.voxel_size, &f.affine).expect("write local field");
    save_nifti_to_file(Path::new(&a[4]), &support, f.dims, f.voxel_size, &f.affine).expect("write support");
}

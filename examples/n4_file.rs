//! Run N4 bias field correction (ANTs' `N4BiasFieldCorrection` with its default options) on a
//! NIfTI file, e.g. for parity checks against ANTs.
//!
//! usage: n4_file <in.nii[.gz]> <corrected_out.nii[.gz]> [bias_out.nii[.gz]] [mask.nii[.gz]]
//!
//! The mask (voxels > 0.5) is optional, as ANTs' `-x`. Prints the iteration count and run
//! time. `N4_ITERS=50x50x50x50`, `N4_THRESHOLD=0`, `N4_SHRINK=4` and `N4_ORDER=3` override
//! the iterations per level and convergence threshold (ANTs' `-c`), the shrink factor (`-s`)
//! and the spline order (`-b [1x1x1,order]`).
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::utils::{n4_bias_correction, N4Params};
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if !(3..=5).contains(&a.len()) {
        eprintln!("usage: {} <in.nii[.gz]> <corrected_out.nii[.gz]> [bias_out.nii[.gz]] [mask.nii[.gz]]", a[0]);
        std::process::exit(2);
    }
    let f = read_nifti_file(Path::new(&a[1])).expect("read input");
    let vs = f.voxel_size;
    let grid = Grid::new(f.dims.0, f.dims.1, f.dims.2, vs.0, vs.1, vs.2);
    let mask: Option<Vec<u8>> = a.get(4).map(|p| {
        let m = read_nifti_file(Path::new(p)).expect("read mask");
        m.data.iter().map(|&v| (v > 0.5) as u8).collect()
    });
    let mut params = N4Params::default();
    if let Ok(c) = std::env::var("N4_ITERS") {
        params.max_iterations = c.split('x').map(|v| v.parse().expect("iterations")).collect();
    }
    if let Ok(s) = std::env::var("N4_SHRINK") {
        params.shrink_factor = s.parse().expect("shrink factor");
    }
    if let Ok(t) = std::env::var("N4_THRESHOLD") {
        params.convergence_threshold = t.parse().expect("convergence threshold");
    }
    if let Ok(o) = std::env::var("N4_ORDER") {
        params.spline_order = o.parse().expect("spline order");
    }
    let t = std::time::Instant::now();
    let iterations = std::cell::Cell::new(0);
    let out = n4_bias_correction(&f.data, mask.as_deref(), &grid, &params, |i, _| iterations.set(i)).expect("n4");
    eprintln!("n4: {:?}, {} iterations, {:.2}s", f.dims, iterations.get(), t.elapsed().as_secs_f64());
    save_nifti_to_file(Path::new(&a[2]), &out.corrected, f.dims, vs, &f.affine).expect("write corrected");
    if let Some(p) = a.get(3) {
        save_nifti_to_file(Path::new(p), &out.bias_field, f.dims, vs, &f.affine).expect("write bias");
    }
}

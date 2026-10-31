//! Gibbs-unring a NIfTI volume (3-D, or 4-D one volume at a time) with one of QSM.rs's modes,
//! e.g. for parity checks against MRtrix3's `mrdegibbs` or DIPY's `gibbs_removal`.
//!
//! usage: gibbs_file <in.nii[.gz]> <out.nii.gz> <bautista3d|mrtrix2d|dipy2d> [slice_axis]
//!
//! `slice_axis` (default 2) applies to the 2-D modes. With `GIBBS_RAW=<path>` it also writes the
//! output as little-endian f64, since the NIfTI output may be single precision.
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::unring::{gibbs_unring_volume, GibbsMode};
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if !(4..=5).contains(&a.len()) {
        eprintln!("usage: {} <in.nii[.gz]> <out.nii.gz> <bautista3d|mrtrix2d|dipy2d> [slice_axis]", a[0]);
        std::process::exit(2);
    }
    let slice_axis = a.get(4).map_or(2, |s| s.parse().expect("slice_axis"));
    let mode = match a[3].as_str() {
        "bautista3d" => GibbsMode::Bautista3d,
        "mrtrix2d" => GibbsMode::Mrtrix2d { slice_axis },
        "dipy2d" => GibbsMode::Dipy2d { slice_axis },
        m => panic!("unknown mode {m}"),
    };
    let f = read_nifti_file(Path::new(&a[1])).expect("read");
    let t = std::time::Instant::now();
    let out = gibbs_unring_volume(&f.data, f.dims, mode);
    eprintln!("{mode:?} {:?}: {:.2}s", f.dims, t.elapsed().as_secs_f64());
    if let Ok(raw) = std::env::var("GIBBS_RAW") {
        std::fs::write(&raw, out.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).expect("write raw");
    }
    save_nifti_to_file(Path::new(&a[2]), &out, f.dims, f.voxel_size, &f.affine).expect("write");
}

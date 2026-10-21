//! Run `fit_ppm_complex` (MEDI toolbox 2024.11.26 Fit_ppm_complex.m) on 4D NIfTI echoes, for
//! parity checks against Fit_ppm_complex in MATLAB.
//!
//! usage: fit_ppm_complex_file <mag1,mag2,...> <phase1,phase2,...> <TE1,TE2,...(s)> <out_prefix>
//!
//! One 3D file per echo for the magnitude and the phase (radians).
//!
//! Writes `<out_prefix>_field` (wrapped, radians per echo spacing, QSM.rs sign: MATLAB's p1 is
//! minus this), `_noise` (dp1), `_phase0` (minus MATLAB's p0) and `_relres`, each as a NIfTI
//! (`.nii.gz`, float32) and as raw little-endian f64 in NIfTI voxel order (`.f64`), so the parity
//! check is not limited by float32.
use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::fieldmap::{fit_ppm_complex, FitPpmComplexParams};
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 5 {
        eprintln!("usage: {} <mag1,mag2,..> <phase1,phase2,..> <TE1,TE2,..> <out_prefix>", a[0]);
        std::process::exit(2);
    }
    let tes: Vec<f64> = a[3].split(',').map(|t| t.parse().expect("echo time")).collect();
    let read = |files: &str| -> Vec<qsm_core::io::NiftiData> {
        files.split(',').map(|p| read_nifti_file(Path::new(p)).expect("read echo")).collect()
    };
    let (mag, ph) = (read(&a[1]), read(&a[2]));
    assert!(mag.len() == tes.len() && ph.len() == tes.len(), "one magnitude and one phase file per echo");
    let mr: Vec<&[f64]> = mag.iter().map(|v| v.data.as_slice()).collect();
    let pr: Vec<&[f64]> = ph.iter().map(|v| v.data.as_slice()).collect();
    let t = std::time::Instant::now();
    let r = fit_ppm_complex(&mr, &pr, &tes, &FitPpmComplexParams::default()).expect("fit");
    eprintln!("fit_ppm_complex: {:.1}s, {} iterations", t.elapsed().as_secs_f64(), r.iterations);
    for (name, v) in [("field", &r.field), ("noise", &r.noise), ("phase0", &r.phase0), ("relres", &r.relres)] {
        let p = format!("{}_{}.nii.gz", a[4], name);
        save_nifti_to_file(Path::new(&p), v, mag[0].dims, mag[0].voxel_size, &mag[0].affine).expect("write");
        let raw: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(format!("{}_{}.f64", a[4], name), raw).expect("write raw");
    }
}

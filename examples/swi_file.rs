//! CLEAR-SWI of NIfTI echoes, with the intermediate images.
//!
//! ```text
//! cargo run --release --features parallel --example swi_file -- \
//!     --out out/sub-1 --te 0.0094,0.0197 --mag e1_mag.nii.gz,e2_mag.nii.gz \
//!     --phase e1_phase.nii.gz,e2_phase.nii.gz \
//!     [--unwrap laplacian|laplacianslice|romeo] [--weighting phase_snr|phase_var|...] \
//!     [--mask mask.nii.gz] [--mip 7]
//! ```
//!
//! Phase is rescaled to [-π, π] if it is not already (`--raw-phase` skips that). Writes `<out>_swi.nii.gz`,
//! `<out>_minIP.nii.gz` and the steps `<out>_{swimag,swiphase,filteredphase,mask}.nii.gz`.
//! Without `--mask` the phase is filtered within CLEAR-SWI's own mask (robust mask of the
//! first echo's magnitude), as CLEARSWI.jl does.

use qsm_core::io::{read_nifti_file, save_nifti_to_file};
use qsm_core::pipeline::scale_phase_to_pi;
use qsm_core::swi::{calculate_swi, create_mip, SwiParams, SwiUnwrap};
use qsm_core::utils::B0WeightType;
use qsm_core::Grid;
use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut out, mut tes, mut mags, mut phases, mut mask_path) = (None, None, None, None, None);
    let mut params = SwiParams::default();
    let mut raw_phase = false;
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--out" => out = Some(val()),
            "--te" => tes = Some(val().split(',').map(|v| v.parse::<f64>().expect("TE")).collect::<Vec<_>>()),
            "--mag" => mags = Some(val()),
            "--phase" => phases = Some(val()),
            "--mask" => mask_path = Some(val()),
            "--mip" => params.mip_window = val().parse().expect("mIP window"),
            "--unwrap" => {
                params.unwrap = match val().as_str() {
                    "laplacian" => SwiUnwrap::Laplacian,
                    "laplacianslice" => SwiUnwrap::LaplacianSlice,
                    "romeo" => SwiUnwrap::Romeo,
                    other => panic!("unknown unwrapping {other}"),
                }
            }
            "--raw-phase" => raw_phase = true,
            "--weighting" => params.echo_weighting = B0WeightType::from_str(&val()),
            other => panic!("unknown argument {other}"),
        }
    }
    let out = out.expect("--out is required");
    let tes = tes.expect("--te is required");
    let read = |list: String| -> Vec<qsm_core::io::NiftiData> {
        list.split(',').map(|p| read_nifti_file(Path::new(p)).expect("read NIfTI")).collect()
    };
    let mags = read(mags.expect("--mag is required"));
    let mut phases = read(phases.expect("--phase is required"));
    if !raw_phase {
        for p in &mut phases {
            scale_phase_to_pi(&mut p.data);
        }
    }
    let h = &mags[0];
    let grid = Grid::new(h.dims.0, h.dims.1, h.dims.2, h.voxel_size.0, h.voxel_size.1, h.voxel_size.2);
    let mask: Option<Vec<u8>> = mask_path.map(|p| {
        read_nifti_file(Path::new(&p)).expect("read mask").data.iter().map(|&x| u8::from(x > 0.5)).collect()
    });

    let t = std::time::Instant::now();
    let mag_data: Vec<&[f64]> = mags.iter().map(|n| n.data.as_slice()).collect();
    let phase_data: Vec<&[f64]> = phases.iter().map(|n| n.data.as_slice()).collect();
    let r = calculate_swi(&phase_data, &mag_data, &tes, mask.as_deref(), &grid, &params).expect("CLEAR-SWI");
    let mip = create_mip(&r.swi, &grid, &h.affine, params.mip_window).expect("mIP");
    println!("{} echoes, grid {:?}, {params:?}: {:.1} s", tes.len(), grid.dims, t.elapsed().as_secs_f64());

    let save = |suffix: &str, data: &[f64]| {
        let p = format!("{out}_{suffix}.nii.gz");
        save_nifti_to_file(Path::new(&p), data, h.dims, h.voxel_size, &h.affine).expect("write");
        println!("wrote {p}");
    };
    save("swi", &r.swi);
    save("swimag", &r.magnitude);
    save("swiphase", &r.phase_mask);
    save("filteredphase", &r.filtered_phase);
    save("mask", &r.mask.iter().map(|&m| f64::from(m)).collect::<Vec<_>>());
    let p = format!("{out}_minIP.nii.gz");
    save_nifti_to_file(Path::new(&p), &mip.data, mip.grid.dims, mip.grid.voxel_size, &mip.affine).expect("write");
    println!("wrote {p}");
}

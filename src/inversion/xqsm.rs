//! xQSM deep-learning dipole inversion (`onnx` feature).
//!
//! xQSM is an octave-convolution U-Net (with a global residual) that maps a
//! local tissue field (ppm) to susceptibility (ppm). It is orientation-agnostic
//! and takes no dipole kernel — the field goes in, χ comes out. This mirrors the
//! authors' pure-Python inference (`sunhongfu/xQSM`): centered zero-pad each
//! dimension to a multiple of 8, run the net, crop back, and multiply the output
//! by the mask. No normalization.
//!
//! Weights are not bundled; the caller passes the exported `xqsm.onnx` bytes
//! (see [`crate::models`]).
//!
//! # Reference pipeline
//!
//! - **Training data:** local fields simulated from 1 mm isotropic χ maps (`xQSM_invivo`: 90
//!   LN-QSM subjects, 48³ patches) by the forward dipole model; no field strength or TE enters
//!   (Gao et al., NMR Biomed 2021, "Data Preparation").
//! - **Inputs fed in the paper's in-vivo tests:** best-path unwrapping, then RESHARP background
//!   removal (paper, in-vivo experiments paragraph); tested at 1 mm (3T) and 0.6 mm (7T) isotropic.
//!   The paper's resolution study found all networks degrade on anisotropic voxels (NRMSE 22–47 %)
//!   and on voxels finer than the training data ("Generalization Evaluation").
//! - **Resampling:** the authors' MATLAB wrapper (`sunhongfu/iQSM_Plus`
//!   `legacy/matlab/xQSM.m:104-127,170-176`) interpolates anisotropic data to an isotropic grid at
//!   the finest voxel size (`imresize3`) before the net and back afterwards. Their Python
//!   `sunhongfu/xQSM` `inference.py` does not resample. This crate does not resample either; the
//!   caller must supply an isotropic grid to match the MATLAB wrapper.
//! - **Orientation:** axial (B0 along the third axis; `matlab/eval/run_demo.m:8`).
//!
//! # Pipeline rules
//!
//! The facts above that a configuration can depart from are published as rules in
//! [`crate::pipeline::rules::RULES`] (each with its strength, basis, reason and source): `xqsm.bg-resharp`.

use crate::grid::Grid;
use crate::models::onnx::{OnnxModel, OnnxError, Tensor};

/// Run xQSM dipole inversion.
///
/// * `local_field_ppm` — local tissue field (ppm), column-major `(nx,ny,nz)`.
/// * `mask` — binary brain mask (same layout); applied to the output.
/// * `model_onnx` — bytes of the exported `xqsm.onnx`.
///
/// Returns susceptibility (ppm), masked, in the same layout.
pub fn xqsm(
    local_field_ppm: &[f64],
    mask: &[u8],
    grid: &Grid,
    model_onnx: &[u8],
) -> Result<Vec<f64>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    assert_eq!(local_field_ppm.len(), n, "field length must match grid");
    assert_eq!(mask.len(), n, "mask length must match grid");

    // Centered padding offsets so each dim is a multiple of 8 (matches the
    // Python reference's `_zero_pad`).
    let pad = |s: usize| -> (usize, usize) {
        let total = (8 - s % 8) % 8;
        let before = total / 2;
        (before, s + total) // (offset, padded size)
    };
    let (bx, px) = pad(nx);
    let (by, py) = pad(ny);
    let (bz, pz) = pad(nz);

    // Repack column-major (nx,ny,nz) f64 → row-major (px,py,pz) f32, centered.
    // The field is NOT masked before inference (unlike BFRnet).
    let mut input = vec![0.0f32; px * py * pz];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let src = x + nx * (y + ny * z);
                let dst = (z + bz) + pz * ((y + by) + py * (x + bx));
                input[dst] = local_field_ppm[src] as f32;
            }
        }
    }

    let model = OnnxModel::load(model_onnx)?;
    let out = model.run_single(&Tensor::new(vec![1, 1, px, py, pz], input))?;
    if out.shape != [1, 1, px, py, pz] {
        return Err(OnnxError::Run(format!(
            "unexpected output shape {:?}, expected [1,1,{px},{py},{pz}]",
            out.shape
        )));
    }

    // Crop back (centered), mask, and unpack row-major → column-major.
    let mut chi = vec![0.0f64; n];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let dst = x + nx * (y + ny * z);
                if mask[dst] != 0 {
                    let src = (z + bz) + pz * ((y + by) + py * (x + bx));
                    chi[dst] = out.data[src] as f64;
                }
            }
        }
    }
    Ok(chi)
}

/// Memory-bounded xQSM via overlap-tiling — the same fully-convolutional net run patch-by-
/// patch so peak memory is bounded by one patch (for 32-bit WASM, where the whole-volume
/// [`xqsm`] overflows the 4 GB heap on clinical-size data). Results approximate whole-volume
/// xQSM up to tile-boundary error; see [`crate::inversion::tiled`].
pub fn xqsm_tiled(
    local_field_ppm: &[f64],
    mask: &[u8],
    grid: &Grid,
    model_onnx: &[u8],
    cfg: &super::tiled::TileConfig,
    progress: impl FnMut(usize, usize),
) -> Result<Vec<f64>, OnnxError> {
    let model = OnnxModel::load(model_onnx)?;
    // xQSM feeds the raw field (no normalization) and reads χ directly; pad each patch to /8.
    super::tiled::tiled_field_inversion(
        local_field_ppm, mask, grid, &model, &super::tiled::tile_cost_of("xqsm"), cfg, |v| v as f32, |o| o as f64, progress,
    )
}

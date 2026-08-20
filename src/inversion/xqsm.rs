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

//! BFRnet deep-learning background field removal (`onnx` feature).
//!
//! BFRnet is a 3D dual-frequency octave-convolution U-Net that predicts the
//! **background** field from a masked total field; the local tissue field is
//! `total − background`, re-masked. Everything is in ppm — the network never
//! sees TE/B0/B0-direction. This mirrors the QSM-CI ONNX port (faithful to the
//! authors' MATLAB `predict` to |Δ| ≈ 1e-7).
//!
//! Weights are not bundled; the caller passes the exported `bfrnet.onnx` bytes
//! (see [`crate::models`]).

use crate::grid::Grid;
use crate::models::onnx::{OnnxModel, OnnxError, Tensor};

/// Run BFRnet on a total field map.
///
/// * `field_ppm` — total field (ppm), column-major `(nx,ny,nz)`.
/// * `mask` — binary brain mask (same layout).
/// * `model_onnx` — bytes of the exported `bfrnet.onnx`.
///
/// Returns the local tissue field (ppm), masked, in the same layout.
///
/// The whole volume is run in one pass. BFRnet is fully convolutional with three
/// pooling levels, so each spatial dimension is zero-padded up to a multiple of
/// 8 and cropped back. (Memory-bounded Hann-blended tiling — as the Python port
/// uses for very large volumes — is a future refinement.)
pub fn bfrnet(
    field_ppm: &[f64],
    mask: &[u8],
    grid: &Grid,
    model_onnx: &[u8],
) -> Result<Vec<f64>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    assert_eq!(field_ppm.len(), n, "field length must match grid");
    assert_eq!(mask.len(), n, "mask length must match grid");

    // The net was trained on masked total field.
    let masked: Vec<f64> = field_ppm
        .iter()
        .zip(mask)
        .map(|(&f, &m)| if m != 0 { f } else { 0.0 })
        .collect();

    // Padded spatial dims (multiple of 8).
    let (px, py, pz) = (
        nx.div_ceil(8) * 8,
        ny.div_ceil(8) * 8,
        nz.div_ceil(8) * 8,
    );

    // Repack column-major (nx,ny,nz) f64 → row-major (px,py,pz) f32, zero-padded.
    // ONNX/tract tensors are row-major (last axis fastest), so the axis order
    // fed to the net matches the Python port's (X,Y,Z).
    let mut input = vec![0.0f32; px * py * pz];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let src = x + nx * (y + ny * z);
                let dst = z + pz * (y + py * x);
                input[dst] = masked[src] as f32;
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

    // Crop back to (nx,ny,nz), form local = masked − background, re-mask,
    // and unpack row-major → column-major.
    let mut local = vec![0.0f64; n];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let dst = x + nx * (y + ny * z);
                if mask[dst] != 0 {
                    let src = z + pz * (y + py * x);
                    local[dst] = masked[dst] - out.data[src] as f64;
                }
            }
        }
    }
    Ok(local)
}

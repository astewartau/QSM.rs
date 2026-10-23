//! BFRnet deep-learning background field removal (`onnx` feature).
//!
//! BFRnet is a 3D dual-frequency octave-convolution U-Net that predicts the
//! **background** field from a masked total field; the local tissue field is
//! `total − background`, re-masked. Everything is in ppm — the network never
//! sees TE/B0/B0-direction. This mirrors the QSM-CI ONNX port (faithful to the
//! authors' MATLAB `predict` to |Δ| ≈ 1e-7).
//!
//! The whole volume runs in one pass, as the authors run it. Its activations scale with the
//! volume: a 184×256×128 scan needs more than a 32-bit (WASM) heap holds. [`bfrnet_tiled`] runs
//! it patch by patch instead, with memory bounded by the patch; see it for what that costs.
//!
//! Weights are not bundled; the caller passes the exported `bfrnet.onnx` bytes
//! (see [`crate::models`]).
//!
//! # Reference pipeline
//!
//! From Zhu et al., Z Med Phys 2023 (doi:10.1016/j.zemedi.2022.08.001), Methods:
//!
//! - **Training data:** 1 mm isotropic 3T (96 subjects, 64³ patches) with simulated healthy and
//!   hemorrhage sources.
//! - **In-vivo test inputs:** best-path unwrapping (Laplacian unwrapping for the hemorrhage
//!   patient), FSL `bet2 -f 0.2` (`-w 2`) mask eroded by 3 voxels, total field from a fit of the
//!   unwrapped phase against TE; iLSQR after BFRnet.
//! - **Resolution:** "we have only evaluated testing datasets with 1 mm³ isotropic image resolution
//!   since this matches the training datasets. BFRnet's application on total field maps of other
//!   image resolutions may need retraining" (Discussion). The authors' `BFRnet_demo.m` does not
//!   resample.
//!
//! # Pipeline rules
//!
//! The facts above that a configuration can depart from are published as rules in
//! [`crate::pipeline::rules::RULES`] (each with its strength, basis, reason and source): `bfrnet.mask-bet`.

use crate::grid::Grid;
use crate::inversion::TileConfig;
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
/// 8 and cropped back. For bounded memory see [`bfrnet_tiled`].
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

/// BFRnet patch by patch: each `core` block of the background field is predicted from a patch
/// with `halo` voxels of context on every side (zero outside the volume), padded to a multiple of
/// 8, so peak memory follows the patch, not the volume.
///
/// An approximation of [`bfrnet`]: each patch sees only its own part of the masked field, while
/// the background field it predicts comes from sources outside the brain. Tiles that do not touch
/// `mask` are skipped. Returns the local field (ppm), masked, like [`bfrnet`].
pub fn bfrnet_tiled(
    field_ppm: &[f64],
    mask: &[u8],
    grid: &Grid,
    model_onnx: &[u8],
    cfg: &TileConfig,
    progress: impl FnMut(usize, usize),
) -> Result<Vec<f64>, OnnxError> {
    let n = grid.n_total();
    assert_eq!(field_ppm.len(), n, "field length must match grid");
    assert_eq!(mask.len(), n, "mask length must match grid");
    let masked: Vec<f64> = field_ppm.iter().zip(mask).map(|(&f, &m)| if m != 0 { f } else { 0.0 }).collect();
    let model = OnnxModel::load(model_onnx)?;
    let background = crate::inversion::tiled::tiled_field_inversion(
        &masked, mask, grid, &model, &crate::inversion::tiled::tile_cost_of("bfrnet"), cfg, |v| v as f32, |o| o as f64, progress,
    )?;
    Ok(masked
        .iter()
        .zip(&background)
        .zip(mask)
        .map(|((&f, &b), &m)| if m != 0 { f - b } else { 0.0 })
        .collect())
}

//! QSMnet deep-learning dipole inversion (`onnx` feature).
//!
//! QSMnet is a 3D U-Net (SNU-LIST) that maps a local field (ppm) to
//! susceptibility (ppm). The upstream weights are TensorFlow, but we ship a
//! *clean* PyTorch re-export (`scripts/onnx-export/export_qsmnet.py`): the plain
//! U-Net rebuilt in PyTorch with the TF weights ported in, giving a tidy NCDHW
//! ONNX (`[1, 1, X, Y, Z]`) that the pure-Rust `tract` engine runs — unlike the
//! `tf2onnx` graph, whose NHWC↔NCHW Reshape/Transpose ops tract can't analyse.
//! This mirrors the authors' inference (`Code/inference.py`): normalize by the
//! dataset mean/std shipped with the checkpoint, centered zero-pad each dim to a
//! multiple of 16 (four pool/deconv levels), run, crop, de-normalize, and mask.
//!
//! Weights are not bundled; the caller passes the exported `qsmnet.onnx` bytes
//! (see [`crate::models`]).

use crate::grid::Grid;
use crate::models::onnx::{OnnxModel, OnnxError, Tensor};

/// Dataset normalization constants stored beside a QSMnet checkpoint
/// (`norm_factor_<name>.mat`): `field_n = (field - in_mean)/in_std`, and
/// `chi = out_std*pred + out_mean`.
#[derive(Clone, Copy, Debug)]
pub struct QsmnetNorm {
    pub in_mean: f64,
    pub in_std: f64,
    pub out_mean: f64,
    pub out_std: f64,
}

impl QsmnetNorm {
    /// Constants for the `QSMnet_64` checkpoint.
    pub fn qsmnet() -> Self {
        Self { in_mean: 0.0, in_std: 0.01, out_mean: 0.0, out_std: 0.0317 }
    }

    /// Constants for the `QSMnet+_64` checkpoint.
    pub fn qsmnet_plus() -> Self {
        Self { in_mean: 0.0, in_std: 0.0205, out_mean: 0.0, out_std: 0.0734 }
    }
}

impl Default for QsmnetNorm {
    /// Constants for the `QSMnet_64` checkpoint.
    fn default() -> Self {
        Self::qsmnet()
    }
}

/// Run QSMnet dipole inversion.
///
/// * `local_field_ppm` — local tissue field (ppm), column-major `(nx,ny,nz)`.
/// * `mask` — binary brain mask (same layout); applied to the output.
/// * `model_onnx` — bytes of the exported `qsmnet.onnx` (NDHWC input/output).
/// * `norm` — dataset normalization constants for this checkpoint.
///
/// Returns susceptibility (ppm), masked, in the same layout.
pub fn qsmnet(
    local_field_ppm: &[f64],
    mask: &[u8],
    grid: &Grid,
    model_onnx: &[u8],
    norm: &QsmnetNorm,
) -> Result<Vec<f64>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    assert_eq!(local_field_ppm.len(), n, "field length must match grid");
    assert_eq!(mask.len(), n, "mask length must match grid");

    // Centered padding to a multiple of 16 (matches the Python `pad_to_multiple`).
    let pad = |s: usize| -> (usize, usize) {
        let total = (16 - s % 16) % 16;
        (total / 2, s + total)
    };
    let (bx, px) = pad(nx);
    let (by, py) = pad(ny);
    let (bz, pz) = pad(nz);

    // Normalize + repack column-major (nx,ny,nz) f64 → row-major NCDHW
    // [1,1,px,py,pz] f32, centered.
    let inv_std = 1.0 / norm.in_std;
    let mut input = vec![0.0f32; px * py * pz];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let src = x + nx * (y + ny * z);
                let dst = ((x + bx) * py + (y + by)) * pz + (z + bz);
                input[dst] = ((local_field_ppm[src] - norm.in_mean) * inv_std) as f32;
            }
        }
    }

    let model = OnnxModel::load(model_onnx)?;
    let out = model.run_single(&Tensor::new(vec![1, 1, px, py, pz], input))?;
    let expected = [1, 1, px, py, pz];
    if out.shape != expected {
        return Err(OnnxError::Run(format!(
            "unexpected output shape {:?}, expected {expected:?}",
            out.shape
        )));
    }

    // De-normalize, crop (centered), mask, unpack → column-major.
    let mut chi = vec![0.0f64; n];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let dst = x + nx * (y + ny * z);
                if mask[dst] != 0 {
                    let src = ((x + bx) * py + (y + by)) * pz + (z + bz);
                    chi[dst] = norm.out_std * out.data[src] as f64 + norm.out_mean;
                }
            }
        }
    }
    Ok(chi)
}

/// Memory-bounded QSMnet via overlap-tiling — the fully-convolutional U-Net run patch-by-patch
/// (for 32-bit WASM, where whole-volume [`qsmnet`] overflows the heap on clinical data). The
/// dataset normalization is applied per value inside the tile loop and the net's pool depth
/// requires a `/16` patch. Approximates whole-volume up to tile-boundary error; see
/// [`crate::inversion::tiled`].
pub fn qsmnet_tiled(
    local_field_ppm: &[f64],
    mask: &[u8],
    grid: &Grid,
    model_onnx: &[u8],
    norm: &QsmnetNorm,
    cfg: &super::tiled::TileConfig,
    progress: impl FnMut(usize, usize),
) -> Result<Vec<f64>, OnnxError> {
    let model = OnnxModel::load(model_onnx)?;
    let (in_mean, inv_std, out_std, out_mean) =
        (norm.in_mean, 1.0 / norm.in_std, norm.out_std, norm.out_mean);
    super::tiled::tiled_field_inversion(
        local_field_ppm, mask, grid, &model, 16, cfg,
        move |v| ((v - in_mean) * inv_std) as f32,
        move |o| out_std * o as f64 + out_mean,
        progress,
    )
}

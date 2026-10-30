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
//!
//! # Reference pipeline
//!
//! From the authors' README (`SNU-LIST/QSMnet` `README.md:34-72`) unless noted:
//!
//! - **Training data:** 3T (Siemens), single-echo 3D GRE, 1 mm isotropic, TE = 25 ms; COSMOS
//!   labels.
//! - **Phase processing:** FSL BET brain extraction, STI Suite Laplacian unwrapping, STI Suite
//!   3D V-SHARP background removal.
//! - **Resolution:** data at another resolution "need to" be interpolated to 1 mm isotropic
//!   (e.g. Fourier zero-padding/truncation) *before* phase processing. This crate does not
//!   resample; the caller must.
//! - **Units:** the V-SHARP tissue phase is converted to ppm as `phase / (ΣTE · B0 · γ)`
//!   (`save_input_data_for_QSMnet.m:22`), i.e. the field in ppm this module takes.
//! - **Output:** `Code/inference.py` crops but does not multiply by the mask; this crate does.
//! - **Orientation:** inputs must match the orientation/polarity of the shipped `val_input.mat`
//!   (`README.md:86`): axial, B0 along the third axis.
//!
//! # Activation and parity
//!
//! The two checkpoints differ in more than their weights: each one's `network_info_<name>.npy`
//! names its activation, and the authors' inference builds the U-Net with it. `QSMnet_64` uses
//! plain ReLU, `QSMnet+_64` LeakyReLU (α 0.1). The first `qsmnet.onnx` export used LeakyReLU for
//! both, and was checked against a TF graph built the same way, so it went unnoticed: χ came out
//! 1.34× too large, 81 % NRMSE from the original (QSM.rs#330). `export_qsmnet.py` now takes the
//! activation from `network_info`.
//!
//! Measured against QSM-CI's `qsmnet` and `qsmnet-plus` entries (TensorFlow 1.14, SNU-LIST/QSMnet
//! f8782ed, the authors' checkpoints) on a 36×32×28 in-vivo box of `prisma-bridge-run1`:
//! [`qsmnet`] is 0.0002 % NRMSE from the original (max |Δ| 2.9e-7 ppm; before the re-export,
//! 81 %). Run through ONNX Runtime with the same normalisation and padding, the exported graph is
//! 0.00006 % from it and QSMnet+'s is 0.0001 %. The exported graph matches the TF graph to 4.4e-6
//! on the export's own check.
//!
//! # Pipeline rules
//!
//! The facts above that a configuration can depart from are published as rules in
//! [`crate::pipeline::rules::RULES`] (each with its strength, basis, reason and source): `qsmnet.mask-bet`, `qsmnet.unwrap-laplacian`, `qsmnet.echo-combination`, `qsmnet.echo-weights`, `qsmnet.bg-vsharp`.

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
        // QSMnet and QSMnet+ are one architecture, so one tile costs the same (a registry test
        // checks the two figures agree).
        local_field_ppm, mask, grid, &model, &super::tiled::tile_cost_of("qsmnet"), cfg,
        move |v| ((v - in_mean) * inv_std) as f32,
        move |o| out_std * o as f64 + out_mean,
        progress,
    )
}

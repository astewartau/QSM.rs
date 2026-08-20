//! iQFM single-step deep-learning tissue-field mapping (`onnx` feature).
//!
//! iQFM (Gao 2022) is the **local (tissue) field** output of the *same network as
//! iQSM* — a learnable-Laplacian "LoT" front-end + U-Net that maps raw wrapped MRI
//! **phase** (radians) straight to the background-removed local field (ppm),
//! folding phase unwrapping and background-field removal into one network. It is
//! the `iQFM` head of the authors' repo (weights `iQFM_40_v2.pth` +
//! `LoTLayer_lfs_40_v2.pth`), the sibling of the χ head used by [`super::iqsm`].
//!
//! The exported graph, the four+border inputs, the preprocessing (phase sign,
//! sphere erosion, centre-pad to /16, mask, crop) and the multi-echo magnitude·TE²
//! combination are **identical** to iQSM — only the trained weights and the meaning
//! of the output (local field vs susceptibility) differ. So the Rust glue simply
//! runs iQSM's exact pipeline with the `iqfm.onnx` bytes.
//!
//! Weights are not bundled; the caller passes the exported `iqfm.onnx` bytes.

use crate::grid::Grid;
use crate::models::onnx::OnnxError;

/// Run iQFM on a single echo of wrapped phase; returns the local field (ppm),
/// masked, column-major `(nx,ny,nz)`. See [`super::iqsm::iqsm`] for the argument
/// semantics — they are shared verbatim (only the network and its output differ).
#[allow(clippy::too_many_arguments)]
pub fn iqfm(
    phase_rad: &[f64],
    mask: &[u8],
    grid: &Grid,
    te: f64,
    b0: f64,
    phase_sign: f64,
    eroded_rad: i32,
    model_onnx: &[u8],
) -> Result<Vec<f64>, OnnxError> {
    super::iqsm::iqsm(phase_rad, mask, grid, te, b0, phase_sign, eroded_rad, model_onnx)
}

/// Multi-echo iQFM: reconstruct each echo and combine with magnitude·TE² weights,
/// exactly as [`super::iqsm::iqsm_multi_echo`] (the authors' `--echo_4d` path).
#[allow(clippy::too_many_arguments)]
pub fn iqfm_multi_echo(
    phases: &[&[f64]],
    magnitudes: &[&[f64]],
    mask: &[u8],
    grid: &Grid,
    tes: &[f64],
    b0: f64,
    phase_sign: f64,
    eroded_rad: i32,
    model_onnx: &[u8],
) -> Result<Vec<f64>, OnnxError> {
    super::iqsm::iqsm_multi_echo(
        phases, magnitudes, mask, grid, tes, b0, phase_sign, eroded_rad, model_onnx,
    )
}

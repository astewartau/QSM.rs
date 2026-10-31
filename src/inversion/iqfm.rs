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
//! sphere erosion, centre-pad to /16, mask, crop) and the multi-echo combination
//! ([`IqsmEchoWeighting`]) are **identical** to iQSM — only the trained weights and the meaning
//! of the output (local field vs susceptibility) differ. So the Rust glue simply
//! runs iQSM's exact pipeline with the `iqfm.onnx` bytes.
//!
//! Weights are not bundled; the caller passes the exported `iqfm.onnx` bytes.
//!
//! # Reference pipeline
//!
//! Identical to [iQSM](mod@crate::inversion::iqsm#reference-pipeline): the authors run iQFM through the
//! same `run_iqsm` (`sunhongfu/iQSM` `inference.py:139-237`), which requires a brain mask for the
//! iQFM output (README).
//!
//! # Agreement with the original
//!
//! `tests/models_onnx/field.rs` runs it on a 36×32×28 box of in-vivo data (1 mm, 2.89 T, across the
//! brain's edge) against its original's verified QSM-CI entry, iQFM bb163ef (`iqfm`)
//! (QSMxT/QSM-CI#316; the fixture is regenerated in Actions by
//! `tests/data/dl_field_qsmci_small.py`): per echo, NRMSE 2.8e-3 to 6.8e-3 %, max |Δ| up to 4.0e-6
//! ppm: float32 rounding.
//!
//! The original runs one echo at a time and combines the echoes afterwards. As for iQSM, the
//! authors' MATLAB echo fit (`sunhongfu/iQSM_Plus` `legacy/matlab/iQFM.m:263-270`, and the iQFM
//! output of `sunhongfu/iQSM` `legacy/matlab/demo_multi_echo.m:152-160`) weights by mag·TE², an
//! option here, and their Python driver (`sunhongfu/iQSM` `run.py:170`, `app.py:416`) by
//! (mag·TE)², the inverse-variance weight and the default ([`IqsmEchoWeighting`]). On the box's first and third echoes the driver's
//! combination is (mag·TE)² of its per-echo outputs to 5e-6 %, and the two weightings differ by
//! 3.4 % NRMSE; [`iqfm_multi_echo`] matches the MATLAB fit of the original's per-echo outputs to
//! 2.7e-3 % (measured 2026-10-30), and with the Python weighting their (mag·TE)² combination to
//! 2.7e-3 % (measured 2026-10-31).

use crate::grid::Grid;
use crate::inversion::{IqsmEchoWeighting, TileConfig};
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

/// Multi-echo iQFM: reconstruct each echo and combine them with `weighting` (default
/// (magnitude·TE)², the inverse-variance weight; see [`IqsmEchoWeighting`]), exactly as
/// [`super::iqsm::iqsm_multi_echo`], whole-volume or tiled (`tile`).
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
    tile: Option<&TileConfig>,
    weighting: IqsmEchoWeighting,
) -> Result<Vec<f64>, OnnxError> {
    super::iqsm::lot_multi_echo(
        "iqfm", phases, magnitudes, mask, grid, tes, b0, phase_sign, eroded_rad, model_onnx, tile, weighting,
    )
}

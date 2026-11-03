//! QSMnetME deep-learning dipole inversion (`onnx` feature).
//!
//! QSMnetME is the QSM network of the SNU-LIST χ-separation toolbox v1.1.3
//! (`models/240904_QSMnet.onnx`): a QSMnet-style 3D U-Net (Conv/ReLU/MaxPool/ConvTranspose/Concat,
//! 1 channel in and out) trained on multi-echo data (Kim et al. 2025, χ-sepnet, §2.2.2), which the
//! toolbox runs to make χ-sepnet's QSM channel. It is not the QSMnet of Yoon et al. 2018
//! ([`super::qsmnet`]): its own weights, normalisation and input route. The hosted
//! `qsmnetme.onnx` is the toolbox's graph with its fixed 192×192×128 input re-declared dynamic
//! (`scripts/onnx-export/redeclare_dynamic_axes.py`; weights and nodes untouched, identical output
//! at the declared shape).
//!
//! # The authors' route
//!
//! The toolbox never runs QSMnetME on its own. It runs it inside `chi_sepnet_general`
//! (`Chisep_script.m`:194), which was probed as a black box in licensed MATLAB R2023b (QSM.rs#377;
//! method and the resampling in [`crate::utils::chisep_frame`]). The QSM channel there is:
//!
//! 1. the local field (Hz) resampled into the training frame (B0 along the third axis, 1 mm,
//!    padded to multiples of 16) with `interp3 'spline'`, the mask with `'linear'` > 0.5;
//! 2. converted to ppm with the central frequency (`field_hz / (CF·1e-6)`), z-scored with the
//!    *field* statistics of the χ-sepnet normalisation file
//!    (`xsepnet_train_patch_norm_factor_inplane_largedegree_romeo_arlo.mat`: `field_mean`,
//!    `field_std`), and set to 0 outside the frame mask;
//! 3. one pass of the network over the whole frame volume (no tiling);
//! 4. de-normalised with the COSMOS susceptibility statistics of the same file
//!    (`cosmos_sus_std·y + cosmos_sus_mean`), and set to 0 outside the frame mask.
//!
//! Every constant and step was read off the toolbox's workspaces: the network input matched
//! `(field/(CF·1e-6) − field_mean)/field_std` inside the mask to 7.7e-7 (float32 rounding) and
//! 0 outside, and the QSM matched `cosmos_sus_std·y + cosmos_sus_mean` exactly.
//!
//! [`qsmnetme`] takes the local field in ppm, as every inversion in this crate does, so step 2's
//! conversion is the identity. It then does what the toolbox does to the maps it returns:
//! resample the frame QSM back onto the acquired grid (`interp3 'spline'`, see
//! [`crate::utils::chisep_frame`]) and mask it. That last step is the toolbox's treatment of
//! χ-sepnet's χ+ and χ−, applied here to the QSM, which the toolbox keeps internal; the QSM is
//! signed, so nothing is clipped.
//!
//! # Agreement with the original
//!
//! The toolbox's QSM channel (MATLAB R2023b, `chi_sepnet_general` with the network computed by
//! ONNX Runtime) against the Python reproduction QSM-CI's `qsmnetme` entry runs
//! (`algorithms/qsmnetme/recon.py`): 1.7e-5 % NRMSE on 64×60×48 at 0.8×0.9×1.1 mm with B0 along
//! (0.1, 0.2, 0.97), 1.9e-5 % at 1 mm with B0 along z (float32 rounding, ONNX Runtime's thread
//! count). `tests/models_onnx/dipole.rs` checks [`qsmnetme`] against that entry on the in-vivo box
//! (`tests/data/dl_snu_route_small.py`).
//!
//! # Memory
//!
//! The authors run the whole frame in one pass. For a 1 mm head that is ~192×224×176 voxels with
//! 64 channels at full resolution: about 1.1 GB per activation. [`QsmnetmeParams::patch`] runs an
//! overlapping sliding window instead (a smaller result near patch seams: not the authors'), for
//! hosts that cannot hold that.
//!
//! Reference: Kim, M., …, Shin, H.-G., Lee, J. (2025). "χ-sepnet: Deep neural network for magnetic
//! susceptibility source separation." Human Brain Mapping, 46(4):e70136. doi:10.1002/hbm.70136.

#![cfg_attr(not(feature = "onnx"), allow(dead_code))]

#[cfg(feature = "onnx")]
use crate::grid::Grid;
#[cfg(feature = "onnx")]
use crate::models::onnx::{OnnxError, OnnxModel};

/// QSMnetME's normalisation: `(mean, std)` of the field (ppm) input and the χ (ppm) output, from
/// `xsepnet_train_patch_norm_factor_inplane_largedegree_romeo_arlo.mat` (`field_*`,
/// `cosmos_sus_*`; the same values χ-sepnet's [`ChiSepNetNorm`](crate::separation::ChiSepNetNorm)
/// uses for its field and QSM channels).
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QsmnetmeNorm {
    pub field: (f64, f64),
    pub chi: (f64, f64),
}

impl Default for QsmnetmeNorm {
    fn default() -> Self {
        Self {
            field: (-0.0026045702397823334, 0.010849659331142902),
            chi: (-0.0013402086915448308, 0.031504031270742416),
        }
    }
}

/// Inference parameters for [`qsmnetme`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QsmnetmeParams {
    /// `None` (default): the whole frame in one pass, as the toolbox runs it. `Some(patch)`: an
    /// overlapping sliding window of that patch (each dimension a multiple of 16), for hosts that
    /// cannot hold the whole volume's activations.
    pub patch: Option<(usize, usize, usize)>,
}

impl QsmnetmeParams {
    /// Check the patch: every dimension a positive multiple of 16.
    pub fn validate(&self) -> Result<(), String> {
        if let Some((d, h, w)) = self.patch {
            for (axis, v) in [("D", d), ("H", h), ("W", w)] {
                if v == 0 || v % 16 != 0 {
                    return Err(format!("QSMnetME patch {axis} = {v} must be a positive multiple of 16"));
                }
            }
        }
        Ok(())
    }
}

/// QSMnetME in the training frame (steps 2-4 of the module docs, ppm in): the frame QSM (ppm,
/// masked) and the z-scored field channel, which χ-sepnet takes too.
#[cfg(feature = "onnx")]
pub(crate) fn qsmnetme_in_frame(
    field_ppm: &[f64],
    mask: &[u8],
    dims: (usize, usize, usize),
    model: &OnnxModel,
    norm: &QsmnetmeNorm,
    patch: Option<(usize, usize, usize)>,
) -> Result<(Vec<f64>, Vec<f32>), OnnxError> {
    let x: Vec<f32> = field_ppm
        .iter()
        .zip(mask)
        .map(|(&f, &m)| if m != 0 { ((f - norm.field.0) / norm.field.1) as f32 } else { 0.0 })
        .collect();
    let y = crate::utils::chisep_frame::run_net(model, std::slice::from_ref(&x), dims, 1, patch)?;
    let qsm = y[0]
        .iter()
        .zip(mask)
        .map(|(&v, &m)| if m != 0 { v as f64 * norm.chi.1 + norm.chi.0 } else { 0.0 })
        .collect();
    Ok((qsm, x))
}

/// Run QSMnetME dipole inversion by the authors' route (module docs).
///
/// * `local_field_ppm` — local field (ppm), column-major `(nx, ny, nz)`.
/// * `mask` — binary brain mask (same layout); the output is 0 outside it.
/// * `grid` — dimensions and voxel sizes (mm).
/// * `b0_dir` — B0 direction in the array's axes.
/// * `model_onnx` — bytes of the hosted `qsmnetme.onnx`.
///
/// Returns χ (ppm) on the acquired grid.
#[cfg(feature = "onnx")]
pub fn qsmnetme(
    local_field_ppm: &[f64],
    mask: &[u8],
    grid: &Grid,
    b0_dir: (f64, f64, f64),
    model_onnx: &[u8],
    norm: &QsmnetmeNorm,
    params: &QsmnetmeParams,
) -> Result<Vec<f64>, OnnxError> {
    let n = grid.n_total();
    assert_eq!(local_field_ppm.len(), n, "field length must match grid");
    assert_eq!(mask.len(), n, "mask length must match grid");
    params.validate().map_err(OnnxError::Domain)?;
    let frame = crate::utils::chisep_frame::TrainingFrame::new(grid.dims, grid.voxel_size, b0_dir);
    let field = frame.to_frame(local_field_ppm);
    let fmask = frame.mask_to_frame(mask);
    let model = OnnxModel::load(model_onnx)?;
    let (qsm, _) = qsmnetme_in_frame(&field, &fmask, frame.dims(), &model, norm, params.patch)?;
    let back = frame.to_native(&qsm);
    Ok(back.into_iter().zip(mask).map(|(v, &m)| if m != 0 { v } else { 0.0 }).collect())
}

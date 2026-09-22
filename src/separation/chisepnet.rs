//! χ-sepnet deep-learning χ-separation (`onnx` feature).
//!
//! χ-sepnet (SNU-LIST) is a 3D U-Net that maps a 3-channel patch — [QSM (χ_total,
//! ppm), local field (ppm), R2′/Dr] each z-scored by training statistics — to
//! paramagnetic (χ+) and diamagnetic (χ−) source magnitudes (z-scored). The network
//! is a fixed **192×192×128** patch; we run it as an overlapping sliding window over
//! the whole volume and average the overlaps, then de-normalize.
//!
//! Recipe (mirrors the SNU-LIST / QSM-CI `recon.py`): z-score each channel, zero
//! outside the mask, end-pad each dim up to the patch size, tile with a 0.75 stride
//! plus a final flush patch, average overlaps, de-normalize, crop, and mask. `Dr`
//! (114 Hz/ppm, the network's COSMOS-referenced relaxivity) scales R2′ into the
//! network's ppm-equivalent input channel. We return χ− as a signed (≤ 0) value to
//! match the crate's separation convention `(chi_pos ≥ 0, chi_neg ≤ 0, chi_total)`.
//!
//! **Patch size.** The published graph declares that patch as a fixed input shape, but it is
//! fully convolutional (Conv/Relu/MaxPool/ConvTranspose/Concat only), so the hosted
//! `chi-sepnet.onnx` has its spatial axes re-declared as dynamic — bit-identical at the
//! authors' patch, and able to run smaller ones. [`ChiSepNetParams::patch`] defaults to the
//! authors' 192×192×128; a 32-bit host (WASM) cannot afford it — one 64-channel activation at
//! that size is 1.2 GB — and uses [`WASM_PATCH`] instead. Patch dimensions must be multiples
//! of 16 (four pooling levels) or the skip-connection concatenations misalign.
//!
//! Weights are not bundled; the caller passes the exported `chi-sepnet.onnx` bytes.

use crate::grid::Grid;
use crate::models::onnx::{OnnxError, OnnxModel, Tensor};
use crate::utils::sliding::starts;

/// The authors' patch: `(D, H, W)` = `(x, y, z)`.
pub const AUTHORS_PATCH: (usize, usize, usize) = (192, 192, 128);

/// A patch that fits a 32-bit (WASM) heap; see the module docs for what it costs.
pub const WASM_PATCH: (usize, usize, usize) = (128, 128, 64);

/// Four pooling levels, so every patch dimension must be a multiple of this.
const POOL_MULTIPLE: usize = 16;

/// Inference parameters for [`chisepnet`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug)]
pub struct ChiSepNetParams {
    /// Sliding-window patch `(D, H, W)`; each dimension a multiple of 16. Defaults to
    /// [`AUTHORS_PATCH`]; WASM hosts pass [`WASM_PATCH`].
    pub patch: (usize, usize, usize),
}

impl Default for ChiSepNetParams {
    fn default() -> Self {
        Self { patch: AUTHORS_PATCH }
    }
}

/// Training z-score constants for χ-sepnet (`xsepnet_train_patch_norm_factor…mat`):
/// each field is `(mean, std)`. Inputs are normalized `(x-mean)/std`; outputs
/// de-normalized `y*std + mean`. `dr` scales R2′ (Hz) into the ppm-equivalent
/// channel `(r2prime/dr - mean)/std`.
#[derive(Clone, Copy, Debug)]
pub struct ChiSepNetNorm {
    pub qsm: (f64, f64),
    pub field: (f64, f64),
    pub r2prime: (f64, f64),
    pub chi_pos: (f64, f64),
    pub chi_neg: (f64, f64),
    pub dr: f64,
}

impl Default for ChiSepNetNorm {
    /// Constants from `xsepnet_train_patch_norm_factor_inplane_largedegree_romeo_arlo.mat`.
    fn default() -> Self {
        Self {
            qsm: (-0.0013402086915448308, 0.031504031270742416),
            field: (-0.0026045702397823334, 0.010849659331142902),
            r2prime: (0.05141879618167877, 0.06977531313896179),
            chi_pos: (0.025912819430232048, 0.03526417911052704),
            chi_neg: (0.026700690388679504, 0.027936099097132683),
            dr: 114.0,
        }
    }
}

/// Run χ-sepnet χ-separation.
///
/// * `local_field_ppm`, `qsm` (χ_total, ppm), `r2prime` (Hz) — column-major `(nx,ny,nz)`.
/// * `mask` — binary brain mask (same layout).
/// * `model_onnx` — bytes of the exported `chi-sepnet.onnx` (192×192×128, 3→2 chan).
/// * `norm` — training normalization constants.
/// * `params` — patch size; see [`ChiSepNetParams`].
///
/// Returns `(chi_pos ≥ 0, chi_neg ≤ 0, chi_total = chi_pos + chi_neg)` in ppm,
/// masked, in the same layout.
///
/// # Panics
/// If a patch dimension is not a multiple of 16 (the network's four pooling levels).
#[allow(clippy::too_many_arguments)]
pub fn chisepnet(
    local_field_ppm: &[f64],
    qsm: &[f64],
    r2prime: &[f64],
    mask: &[u8],
    grid: &Grid,
    model_onnx: &[u8],
    norm: &ChiSepNetNorm,
    params: &ChiSepNetParams,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), OnnxError> {
    let (nx, ny, nz) = grid.dims;
    let (pd, ph, pw) = params.patch;
    for (axis, d) in [("D", pd), ("H", ph), ("W", pw)] {
        assert!(
            d > 0 && d % POOL_MULTIPLE == 0,
            "patch {axis} = {d} must be a positive multiple of {POOL_MULTIPLE} (four pooling levels)"
        );
    }
    let n = nx * ny * nz;
    for (name, v) in [("field", local_field_ppm), ("qsm", qsm), ("r2prime", r2prime)] {
        assert_eq!(v.len(), n, "{name} length must match grid");
    }
    assert_eq!(mask.len(), n, "mask length must match grid");

    // End-pad each dim up to at least the patch size (col-major padded volume).
    let (px, py, pz) = (nx.max(pd), ny.max(ph), nz.max(pw));
    let mut ch = [vec![0.0f32; px * py * pz], vec![0.0f32; px * py * pz], vec![0.0f32; px * py * pz]];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = x + nx * (y + ny * z);
                if mask[i] == 0 {
                    continue;
                }
                let p = x + px * (y + py * z);
                ch[0][p] = ((qsm[i] - norm.qsm.0) / norm.qsm.1) as f32;
                ch[1][p] = ((local_field_ppm[i] - norm.field.0) / norm.field.1) as f32;
                ch[2][p] = ((r2prime[i] / norm.dr - norm.r2prime.0) / norm.r2prime.1) as f32;
            }
        }
    }

    let model = OnnxModel::load(model_onnx)?;
    let plane = pd * ph * pw;
    let mut acc0 = vec![0.0f64; px * py * pz];
    let mut acc1 = vec![0.0f64; px * py * pz];
    let mut wsum = vec![0.0f64; px * py * pz];
    for &x0 in &starts(px, pd) {
        for &y0 in &starts(py, ph) {
            for &z0 in &starts(pz, pw) {
                // Build the [1,3,pd,ph,pw] NCDHW patch (D=x, H=y, W=z).
                let mut buf = vec![0.0f32; 3 * plane];
                for i in 0..pd {
                    for j in 0..ph {
                        for k in 0..pw {
                            let p = (x0 + i) + px * ((y0 + j) + py * (z0 + k));
                            let o = (i * ph + j) * pw + k;
                            buf[o] = ch[0][p];
                            buf[plane + o] = ch[1][p];
                            buf[2 * plane + o] = ch[2][p];
                        }
                    }
                }
                let out = model.run_single(&Tensor::new(vec![1, 3, pd, ph, pw], buf))?;
                if out.data.len() < 2 * plane {
                    return Err(OnnxError::Run("expected 2-channel output".into()));
                }
                for i in 0..pd {
                    for j in 0..ph {
                        for k in 0..pw {
                            let p = (x0 + i) + px * ((y0 + j) + py * (z0 + k));
                            let o = (i * ph + j) * pw + k;
                            acc0[p] += out.data[o] as f64;
                            acc1[p] += out.data[plane + o] as f64;
                            wsum[p] += 1.0;
                        }
                    }
                }
            }
        }
    }

    // Average overlaps, de-normalize, crop to (nx,ny,nz), mask. χ− → signed (≤ 0).
    let mut chi_pos = vec![0.0f64; n];
    let mut chi_neg = vec![0.0f64; n];
    let mut chi_total = vec![0.0f64; n];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = x + nx * (y + ny * z);
                if mask[i] == 0 {
                    continue;
                }
                let p = x + px * (y + py * z);
                let w = wsum[p].max(1.0);
                let pos = (acc0[p] / w) * norm.chi_pos.1 + norm.chi_pos.0;
                let neg = (acc1[p] / w) * norm.chi_neg.1 + norm.chi_neg.0;
                chi_pos[i] = pos;
                chi_neg[i] = -neg;
                chi_total[i] = pos - neg;
            }
        }
    }
    Ok((chi_pos, chi_neg, chi_total))
}

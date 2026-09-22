//! SUSEP-Net deep-learning χ-separation (`onnx` feature).
//!
//! SUSEP-Net (Li/Gao/Sun 2025) is a dual-branch 3D U-Net that maps three guidance
//! maps — QSM (χ_total, ppm), R2′ (Hz), local field (ppm) — to paramagnetic (χ+)
//! and diamagnetic (χ−) source magnitudes. Clean NCDHW ONNX export (three inputs
//! `qsm`,`r2prime`,`lfs`; two outputs `chi_pos`,`chi_neg`).
//!
//! Recipe (mirrors the authors' `recon.py`): z-score each input by the training
//! stats, zero outside the mask, post-pad each dim to a multiple of 8, run,
//! de-normalize the outputs, crop, and mask. The network's ReLU makes both
//! outputs non-negative magnitudes; we return χ− as a signed (≤ 0) value to match
//! the crate's separation convention `(chi_pos ≥ 0, chi_neg ≤ 0, chi_total)`.
//!
//! **Whole volume vs patches.** The authors run the whole volume in one forward pass, and
//! [`SusepNetParams::patch`] defaults to that (`None`). Activations then scale with the volume,
//! which a 32-bit host cannot afford — at 1 mm whole-brain the first conv block alone runs to
//! several GB against a 4 GB WASM heap — so such a host passes a patch and gets an overlapping
//! sliding window (0.75 stride, overlaps averaged) with memory bounded by the patch. Tiling a
//! net trained on whole volumes is an **approximation**: each patch is blind to structure
//! outside it. Patch dimensions must be multiples of 8 (three pooling levels).
//!
//! Weights are not bundled; the caller passes the exported `susep-net.onnx` bytes
//! (see [`crate::models`]).

use crate::grid::Grid;
use crate::models::onnx::{OnnxModel, OnnxError, Tensor};
use crate::utils::sliding::starts;

/// Three 2× pooling levels, so every patch dimension must be a multiple of this.
const POOL_MULTIPLE: usize = 8;

/// A patch that fits a 32-bit (WASM) heap; see the module docs for what it costs.
pub const WASM_PATCH: (usize, usize, usize) = (128, 128, 64);

/// Inference parameters for [`susep_net`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, Default)]
pub struct SusepNetParams {
    /// `None` (the default) runs the whole volume in one pass, as the authors do. `Some(patch)`
    /// runs an overlapping sliding window instead — bounded memory, at the cost of each patch
    /// seeing less context. Each dimension must be a multiple of 8.
    pub patch: Option<(usize, usize, usize)>,
}

/// Training z-score constants for SUSEP-Net (`all_mean_std.mat`): each field is
/// `(mean, std)`. Inputs are normalized `(x-mean)/std`; outputs de-normalized
/// `y*std + mean`.
#[derive(Clone, Copy, Debug)]
pub struct SusepNetNorm {
    pub qsm: (f64, f64),
    pub lfs: (f64, f64),
    pub r2prime: (f64, f64),
    pub chi_pos: (f64, f64),
    pub chi_neg: (f64, f64),
}

impl Default for SusepNetNorm {
    /// Constants shipped with the released `SUSEPNet.pth`.
    fn default() -> Self {
        Self {
            qsm: (-6.0663105e-05, 0.023533047),
            lfs: (-4.8702253e-05, 0.012554166),
            r2prime: (4.7629275, 10.889079),
            chi_pos: (0.0089528897, 0.025519046),
            chi_neg: (0.0090135528, 0.019603666),
        }
    }
}

/// Run SUSEP-Net χ-separation.
///
/// * `local_field_ppm`, `qsm` (χ_total, ppm), `r2prime` (Hz) — column-major `(nx,ny,nz)`.
/// * `mask` — binary brain mask (same layout).
/// * `model_onnx` — bytes of the exported `susep-net.onnx`.
/// * `norm` — training normalization constants.
/// * `params` — whole volume or sliding-window patch; see [`SusepNetParams`].
/// * `progress` — progress callback `(patches_done, patches_total)`; a whole-volume run
///   reports a single patch.
///
/// Returns `(chi_pos ≥ 0, chi_neg ≤ 0, chi_total = chi_pos + chi_neg)` in ppm,
/// masked, in the same layout.
#[allow(clippy::too_many_arguments)]
pub fn susep_net(
    local_field_ppm: &[f64],
    qsm: &[f64],
    r2prime: &[f64],
    mask: &[u8],
    grid: &Grid,
    model_onnx: &[u8],
    norm: &SusepNetNorm,
    params: &SusepNetParams,
    mut progress: impl FnMut(usize, usize),
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), OnnxError> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    for (name, v) in [("field", local_field_ppm), ("qsm", qsm), ("r2prime", r2prime)] {
        assert_eq!(v.len(), n, "{name} length must match grid");
    }
    assert_eq!(mask.len(), n, "mask length must match grid");

    // Whole-volume (the authors' default) is the degenerate case of one patch covering the
    // padded volume, so both paths share the code below. Either way every extent is a
    // multiple of 8, the network's three pooling levels.
    let round_up = |v: usize| v.div_ceil(POOL_MULTIPLE) * POOL_MULTIPLE;
    let (pd, ph, pw) = match params.patch {
        Some((d, h, w)) => {
            for (axis, v) in [("D", d), ("H", h), ("W", w)] {
                assert!(
                    v > 0 && v % POOL_MULTIPLE == 0,
                    "patch {axis} = {v} must be a positive multiple of {POOL_MULTIPLE} \
                     (three pooling levels)"
                );
            }
            (d, h, w)
        }
        None => (round_up(nx), round_up(ny), round_up(nz)),
    };
    let (px, py, pz) = (nx.max(pd), ny.max(ph), nz.max(pw));

    // z-score + mask, in the padded volume's row-major NCDHW layout.
    let plane = px * py * pz;
    let zscore = |src: &[f64], (mean, std): (f64, f64)| -> Vec<f32> {
        let inv = 1.0 / std;
        let mut buf = vec![0.0f32; plane];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let i = x + nx * (y + ny * z);
                    if mask[i] != 0 {
                        buf[(x * py + y) * pz + z] = ((src[i] - mean) * inv) as f32;
                    }
                }
            }
        }
        buf
    };
    // Channel order must match the exported graph: qsm, r2prime, lfs.
    let channels = [
        zscore(qsm, norm.qsm),
        zscore(r2prime, norm.r2prime),
        zscore(local_field_ppm, norm.lfs),
    ];

    let model = OnnxModel::load(model_onnx)?;
    let patch_voxels = pd * ph * pw;
    let mut acc_pos = vec![0.0f64; plane];
    let mut acc_neg = vec![0.0f64; plane];
    let mut wsum = vec![0.0f64; plane];

    let (sx, sy, sz) = (starts(px, pd), starts(py, ph), starts(pz, pw));
    let total = sx.len() * sy.len() * sz.len();
    let mut done = 0usize;
    progress(0, total);
    for &x0 in &sx {
        for &y0 in &sy {
            for &z0 in &sz {
                let tensors: Vec<Tensor> = channels
                    .iter()
                    .map(|ch| {
                        let mut buf = vec![0.0f32; patch_voxels];
                        for i in 0..pd {
                            for j in 0..ph {
                                let row = ((x0 + i) * py + (y0 + j)) * pz + z0;
                                let dst = (i * ph + j) * pw;
                                buf[dst..dst + pw].copy_from_slice(&ch[row..row + pw]);
                            }
                        }
                        Tensor::new(vec![1, 1, pd, ph, pw], buf)
                    })
                    .collect();
                let outs = model.run(&tensors)?;
                if outs.len() < 2 {
                    return Err(OnnxError::Run(format!("expected 2 outputs, got {}", outs.len())));
                }
                for i in 0..pd {
                    for j in 0..ph {
                        for k in 0..pw {
                            let dst = ((x0 + i) * py + (y0 + j)) * pz + (z0 + k);
                            let src = (i * ph + j) * pw + k;
                            acc_pos[dst] += outs[0].data[src] as f64;
                            acc_neg[dst] += outs[1].data[src] as f64;
                            wsum[dst] += 1.0;
                        }
                    }
                }
                done += 1;
                progress(done, total);
            }
        }
    }

    // Average overlaps, de-normalize, crop, mask, unpack. χ− is a magnitude → return signed.
    let mut chi_pos = vec![0.0f64; n];
    let mut chi_neg = vec![0.0f64; n];
    let mut chi_total = vec![0.0f64; n];
    let (pm, ps) = norm.chi_pos;
    let (nm, ns) = norm.chi_neg;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = x + nx * (y + ny * z);
                if mask[i] == 0 {
                    continue;
                }
                let src = (x * py + y) * pz + z;
                let w = wsum[src].max(1.0);
                let pos = (acc_pos[src] / w) * ps + pm;
                let neg_mag = (acc_neg[src] / w) * ns + nm;
                chi_pos[i] = pos;
                chi_neg[i] = -neg_mag;
                chi_total[i] = pos - neg_mag;
            }
        }
    }
    Ok((chi_pos, chi_neg, chi_total))
}

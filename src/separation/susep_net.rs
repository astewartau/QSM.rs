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
//! Weights are not bundled; the caller passes the exported `susep-net.onnx` bytes
//! (see [`crate::models`]).

use crate::grid::Grid;
use crate::models::onnx::{OnnxModel, OnnxError, Tensor};

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
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), OnnxError> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    for (name, v) in [("field", local_field_ppm), ("qsm", qsm), ("r2prime", r2prime)] {
        assert_eq!(v.len(), n, "{name} length must match grid");
    }
    assert_eq!(mask.len(), n, "mask length must match grid");

    // Post-pad each dim to a multiple of 8 (three 2× pools).
    let (px, py, pz) = (nx.div_ceil(8) * 8, ny.div_ceil(8) * 8, nz.div_ceil(8) * 8);

    // z-score + mask + repack column-major (nx,ny,nz) → row-major NCDHW [1,1,px,py,pz].
    let pack = |src: &[f64], (mean, std): (f64, f64)| -> Tensor {
        let inv = 1.0 / std;
        let mut buf = vec![0.0f32; px * py * pz];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let i = x + nx * (y + ny * z);
                    if mask[i] != 0 {
                        let dst = (x * py + y) * pz + z;
                        buf[dst] = ((src[i] - mean) * inv) as f32;
                    }
                }
            }
        }
        Tensor::new(vec![1, 1, px, py, pz], buf)
    };

    // Input order must match the exported graph: qsm, r2prime, lfs.
    let inputs = [
        pack(qsm, norm.qsm),
        pack(r2prime, norm.r2prime),
        pack(local_field_ppm, norm.lfs),
    ];
    let model = OnnxModel::load(model_onnx)?;
    let outs = model.run(&inputs)?;
    if outs.len() < 2 {
        return Err(OnnxError::Run(format!("expected 2 outputs, got {}", outs.len())));
    }

    // De-normalize, crop, mask, unpack. χ− is a magnitude → return signed (≤ 0).
    let mut chi_pos = vec![0.0f64; n];
    let mut chi_neg = vec![0.0f64; n];
    let mut chi_total = vec![0.0f64; n];
    let (pm, ps) = norm.chi_pos;
    let (nm, ns) = norm.chi_neg;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = x + nx * (y + ny * z);
                if mask[i] != 0 {
                    let src = (x * py + y) * pz + z;
                    let pos = outs[0].data[src] as f64 * ps + pm;
                    let neg_mag = outs[1].data[src] as f64 * ns + nm;
                    chi_pos[i] = pos;
                    chi_neg[i] = -neg_mag;
                    chi_total[i] = pos - neg_mag;
                }
            }
        }
    }
    Ok((chi_pos, chi_neg, chi_total))
}

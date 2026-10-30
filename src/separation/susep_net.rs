//! SUSEP-Net deep-learning χ-separation (`onnx` feature).
//!
//! SUSEP-Net (Li/Gao/Sun 2025) is a dual-branch 3D U-Net that maps three guidance
//! maps — QSM (χ_total, ppm), R2′ (Hz), local field (ppm) — to paramagnetic (χ+)
//! and diamagnetic (χ−) source magnitudes. Clean NCDHW ONNX export (three inputs
//! `qsm`,`r2prime`,`lfs`; two outputs `chi_pos`,`chi_neg`).
//!
//! Recipe (mirrors the authors' `recon.py`): z-score each input by the training
//! stats, zero outside the mask, post-pad each dim to a multiple of 8, run,
//! de-normalize the outputs, crop, and mask. We return χ− as a signed (≤ 0) value to match
//! the crate's separation convention `(chi_pos ≥ 0, chi_neg ≤ 0, chi_total)`.
//!
//! **Deviation:** both decoders end in a 1×1 convolution with no activation (`SQNet.py`), so a
//! de-normalised magnitude can be negative, and `test_demo.py` saves it as it is. Here negative
//! χ+ and |χ−| are set to 0, since a negative source magnitude has no physical meaning, and
//! χ_total is χ+ + χ− after that. Every non-negative value is the network's unchanged.
//!
//! **Array orientation.** SUSEP-Net is a different group's network from the χ-sepnet family and
//! its training orientation has not been established here; the authors' `recon.py` does not
//! reorient, and neither does this function (it is handed a [`Grid`], not an affine). Treat it as
//! assuming the usual axial convention — slice axis along B0 — and see
//! [`chisepnet`](crate::separation::chisepnet) for how little that turned out to be worth on the
//! χ-sepnet side; no equivalent measurement has been made for this network.
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
//!
//! # Reference pipeline
//!
//! Li M., …, Sun H., Gao Y. 2025 (arXiv:2506.13293):
//!
//! - **Training (§2.3):** 3 T (Philips Ingenia Elition X), 0.9 × 0.9 × 1 mm, 8 echoes,
//!   TE1/ΔTE = 4.1/4.0 ms, with 64³ patches. The training inputs are *simulated* by the forward
//!   model (Eq. 1) from APART-QSM-reconstructed χ+/χ− and decay-kernel maps, so the network was
//!   never fit to one particular field-mapping chain.
//! - **In-vivo inputs (§3.2):** the local field from iQFM and the QSM from iQSM+, both computed
//!   directly from the multi-echo GRE phase. R2* and R2 come from mono-exponential fits in
//!   STI Suite (multi-echo GRE and multi-echo SE), and R2' = R2* − R2.
//! - R2' enters in Hz with no D_r scaling (the input normalisation in [`SusepNetNorm`]).
//!
//! Reference:
//! Li, M., Chen, C., Li, Z., …, Sun, H., Gao, Y. (2025). "SUSEP-Net: Simulation-Supervised and
//! Contrastive Learning-based Deep Neural Networks for Susceptibility Source Separation."
//! arXiv:2506.13293.
//! Code: <https://github.com/YangGaoUQ/SUSEP-Net>
//!
//! # Agreement with the original
//!
//! `tests/models_onnx/separation.rs` runs it on a 36×32×28 box of in-vivo data (1 mm, 2.89 T,
//! across the brain's edge) against its original's verified QSM-CI entry, SUSEP-Net (`susep-net`)
//! (QSMxT/QSM-CI#316; the fixture is regenerated in Actions by
//! `tests/data/dl_separation_qsmci_small.py`): χ+ and χ− NRMSE 2.4e-5 %, max |Δ| 1.7e-7 ppm
//! (measured 2026-10-30): float32 rounding. The original moves by about 1e-7 relative from run to
//! run, since its wrapper uses every CPU.
//!
//! # Pipeline rules
//!
//! The facts above that a configuration can depart from are published as rules in
//! [`crate::pipeline::rules::RULES`] (each with its strength, basis, reason and source): `susep-net.qsm-iqsm-plus`.

// The parameters are plain Rust, so configs can carry them; only inference needs `onnx`.
#![cfg_attr(not(feature = "onnx"), allow(dead_code))]

#[cfg(feature = "onnx")]
use crate::grid::Grid;
#[cfg(feature = "onnx")]
use crate::models::onnx::{OnnxModel, OnnxError, Tensor};
#[cfg(feature = "onnx")]
use crate::utils::sliding::starts;

/// Three 2× pooling levels, so every patch dimension must be a multiple of this. Public (and
/// built without `onnx`) so a host can check a patch, or state the rule, without restating the
/// number.
pub const POOL_MULTIPLE: usize = 8;

/// A patch that fits a 32-bit (WASM) heap alongside a browser pipeline's other volumes. Peak
/// memory, measured natively (single-threaded): about 1.8 KB per patch voxel plus ~0.9 GB for the
/// parsed model and buffers, so 2.0 GB at this patch, 2.8 GB at 128×128×64 and 12.5 GB for a
/// whole 184×256×128 volume. On the QSM-CI χ-separation phantom it agrees with the whole volume
/// to r ≥ 0.998 on both maps (`tests/chisep_xsepnet.rs`).
pub const WASM_PATCH: (usize, usize, usize) = (96, 96, 64);

/// Inference parameters for [`susep_net`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SusepNetParams {
    /// `None` (the default) runs the whole volume in one pass, as the authors do. `Some(patch)`
    /// runs an overlapping sliding window instead — bounded memory, at the cost of each patch
    /// seeing less context. Each dimension must be a multiple of 8.
    pub patch: Option<(usize, usize, usize)>,
}

impl SusepNetParams {
    /// Settings for a 32-bit (WASM) heap: a [`WASM_PATCH`] sliding window.
    pub fn low_memory() -> Self {
        Self { patch: Some(WASM_PATCH) }
    }

    /// Check the patch, if any: every dimension a positive multiple of 8 (the network's three
    /// pooling levels). [`susep_net`] panics on what this refuses.
    pub fn validate(&self) -> Result<(), String> {
        let Some((d, h, w)) = self.patch else { return Ok(()) };
        for (axis, v) in [("D", d), ("H", h), ("W", w)] {
            if v == 0 || v % POOL_MULTIPLE != 0 {
                return Err(format!(
                    "SUSEP-Net patch {axis} = {v} must be a positive multiple of {POOL_MULTIPLE} \
                     (three pooling levels)"
                ));
            }
        }
        Ok(())
    }
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
#[cfg(feature = "onnx")]
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
) -> Result<super::ChiSepMaps, OnnxError> {
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
    if let Err(e) = params.validate() {
        panic!("{e}");
    }
    let (pd, ph, pw) = params.patch.unwrap_or((round_up(nx), round_up(ny), round_up(nz)));
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
                // deviation: no negative source magnitudes (see the module docs)
                (chi_pos[i], chi_neg[i], chi_total[i]) = super::sources_from_magnitudes(pos, neg_mag);
            }
        }
    }
    Ok((chi_pos, chi_neg, chi_total))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whole volume needs no patch; a given patch must sit on the 8-grid.
    #[test]
    fn patch_validation() {
        assert_eq!(SusepNetParams::default().patch, None);
        assert_eq!(SusepNetParams::low_memory().patch, Some(WASM_PATCH));
        assert!(SusepNetParams::default().validate().is_ok());
        assert!(SusepNetParams::low_memory().validate().is_ok());
        for bad in [(0, 128, 64), (128, 124, 64), (128, 128, 4)] {
            let e = SusepNetParams { patch: Some(bad) }.validate().unwrap_err();
            assert!(e.contains("multiple of 8"), "{e}");
        }
    }
}

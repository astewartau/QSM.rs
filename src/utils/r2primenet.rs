//! R2PRIMEnet: learned R2* → R2′ conversion (`onnx` feature).
//!
//! When no spin-echo acquisition provides a measured R2, [`r2prime`](super::epg::r2prime)
//! (R2′ = R2* − R2) has nothing to subtract. R2PRIMEnet — the R2*→R2′ conversion network of
//! the χ-sepnet pipeline (SNU-LIST) — fills that gap for the **GRE-only** condition: a 3D
//! U-Net trained on paired R2*/R2′ data predicts the reversible component from the
//! GRE-derived R2* alone, so the R2′-consuming χ-separation methods
//! ([`chisepnet`](crate::separation::chisepnet), [`susep_net`](crate::separation::susep_net),
//! [`hc_chisep`](crate::separation::hc_chisep), [`wavesep`](crate::separation::wavesep),
//! [`chi_sep_medi`](crate::separation::chi_sep_medi),
//! [`chi_sep_ilsqr`](crate::separation::chi_sep_ilsqr)) can run on a plain multi-echo GRE.
//!
//! The authors run the network on **192×192×128** single-channel patches as an overlapping
//! sliding window over the whole volume, averaging the overlaps. Recipe (mirrors the
//! authors' inference and the QSM-CI `r2primenet` submission): scale R2* by `Dr` into the
//! network's ppm-equivalent channel, z-score by the training statistics, zero outside the
//! mask, end-pad each dim up to the patch size, tile with a 0.75 stride plus a final flush
//! patch, average overlaps, de-normalize, scale back by `Dr`, clip negatives to zero (the
//! paper's convention), crop, and mask.
//!
//! `Dr` (114 Hz/ppm) is the network family's COSMOS-referenced relaxivity — the same constant
//! [`ChiSepNetNorm`](crate::separation::ChiSepNetNorm) uses. The network was trained at that
//! value, so changing it feeds off-distribution inputs.
//!
//! **R2\* fitting.** The training pipeline fitted R2* with ARLO, so
//! [`r2primenet_from_magnitude`] uses [`r2star_arlo`](super::r2star::r2star_arlo). Supplying
//! an R2* map fitted some other way is allowed but moves the input off-distribution.
//!
//! **Patch size.** The published graph declares a fixed 192×192×128 input, but it is fully
//! convolutional (Conv/Relu/MaxPool/ConvTranspose/Concat only), so the hosted `r2primenet.onnx`
//! has its spatial axes re-declared as dynamic — bit-identical at the authors' patch, and able
//! to run smaller ones. [`R2PrimeNetParams::patch`] defaults to the authors' 192×192×128 and
//! should stay there natively; a 32-bit host (WASM) cannot afford it — one 64-channel
//! activation at that size is 1.2 GB — and runs 128×128×64 instead, which on the reference
//! volume agrees with the authors' patch at corr 0.998 / NRMSE 2.9%. Patch dimensions must be
//! multiples of 16 (four pooling levels) or the skip-connection concatenations misalign.
//!
//! Weights are not bundled; the caller passes the exported `r2primenet.onnx` bytes (see
//! [`crate::models`]).
//!
//! Reference:
//! Kim, M., Ji, S., Lee, J., et al. (2025). "χ-sepnet: Deep neural network for magnetic
//! susceptibility source separation." Human Brain Mapping, 46(4):e70136 — R2PRIMEnet is its
//! R2*→R2′ conversion network.

use crate::grid::Grid;
use crate::models::onnx::{OnnxError, OnnxModel, Tensor};
use super::sliding::starts;

/// The authors' patch: `(D, H, W)` = `(x, y, z)`.
pub const AUTHORS_PATCH: (usize, usize, usize) = (192, 192, 128);

/// A patch that fits a 32-bit (WASM) heap; see the module docs for what it costs.
pub const WASM_PATCH: (usize, usize, usize) = (128, 128, 64);

/// Four pooling levels, so every patch dimension must be a multiple of this.
const POOL_MULTIPLE: usize = 16;

/// Training z-score constants for R2PRIMEnet
/// (`xsepnet_train_patch_norm_factor_inplane_largedegree_romeo_arlo.mat`): each field is
/// `(mean, std)`, in `Dr`-scaled (ppm-equivalent) units. The input is normalized
/// `(r2star/dr - mean)/std`; the output de-normalized `(y*std + mean)*dr`.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug)]
pub struct R2PrimeNetNorm {
    /// `(mean, std)` of the `Dr`-scaled R2* input channel.
    pub r2star: (f64, f64),
    /// `(mean, std)` of the `Dr`-scaled R2′ output channel.
    pub r2prime: (f64, f64),
    /// Relaxivity (Hz/ppm) scaling R2*/R2′ into the network's ppm-equivalent units.
    pub dr: f64,
}

/// Inference parameters for [`r2primenet`].
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Copy, Debug)]
pub struct R2PrimeNetParams {
    /// Sliding-window patch `(D, H, W)`; each dimension a multiple of 16. Defaults to
    /// [`AUTHORS_PATCH`]; WASM hosts pass [`WASM_PATCH`].
    pub patch: (usize, usize, usize),
}

impl Default for R2PrimeNetParams {
    fn default() -> Self {
        Self { patch: AUTHORS_PATCH }
    }
}

impl Default for R2PrimeNetNorm {
    /// Constants from `xsepnet_train_patch_norm_factor_inplane_largedegree_romeo_arlo.mat`
    /// — the same normalization file χ-sepnet uses.
    fn default() -> Self {
        Self {
            r2star: (0.15865366160869598, 0.07501979172229767),
            r2prime: (0.05141879618167877, 0.06977531313896179),
            dr: 114.0,
        }
    }
}

/// Run R2PRIMEnet: predict R2′ (Hz) from an R2* map (Hz).
///
/// # Arguments
/// * `r2star` — R2* map in **Hz**, column-major `(nx,ny,nz)`.
/// * `mask` — binary brain mask (same layout).
/// * `grid` — volume dimensions and voxel sizes.
/// * `model_onnx` — bytes of the exported `r2primenet.onnx` (1→1 chan, dynamic spatial axes).
/// * `norm` — training normalization constants.
/// * `params` — patch size; see [`R2PrimeNetParams`].
/// * `progress` — progress callback `(patches_done, patches_total)`.
///
/// # Returns
/// R2′ in **Hz**, clipped at zero and restricted to `mask`, in the same layout.
///
/// # Panics
/// If a patch dimension is not a multiple of 16 (the network's four pooling levels).
pub fn r2primenet(
    r2star: &[f64],
    mask: &[u8],
    grid: &Grid,
    model_onnx: &[u8],
    norm: &R2PrimeNetNorm,
    params: &R2PrimeNetParams,
    mut progress: impl FnMut(usize, usize),
) -> Result<Vec<f64>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    assert_eq!(r2star.len(), n, "r2star length must match grid");
    assert_eq!(mask.len(), n, "mask length must match grid");
    let (pd, ph, pw) = params.patch;
    for (axis, d) in [("D", pd), ("H", ph), ("W", pw)] {
        assert!(
            d > 0 && d % POOL_MULTIPLE == 0,
            "patch {axis} = {d} must be a positive multiple of {POOL_MULTIPLE} (four pooling levels)"
        );
    }

    // End-pad each dim up to at least the patch size (col-major padded volume).
    let (px, py, pz) = (nx.max(pd), ny.max(ph), nz.max(pw));
    let mut vol = vec![0.0f32; px * py * pz];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = x + nx * (y + ny * z);
                if mask[i] == 0 {
                    continue;
                }
                let p = x + px * (y + py * z);
                vol[p] = ((r2star[i] / norm.dr - norm.r2star.0) / norm.r2star.1) as f32;
            }
        }
    }

    let model = OnnxModel::load(model_onnx)?;
    let plane = pd * ph * pw;
    let mut acc = vec![0.0f64; px * py * pz];
    let mut wsum = vec![0.0f64; px * py * pz];

    let (sx, sy, sz) = (starts(px, pd), starts(py, ph), starts(pz, pw));
    let total = sx.len() * sy.len() * sz.len();
    let mut done = 0usize;
    progress(0, total);
    for &x0 in &sx {
        for &y0 in &sy {
            for &z0 in &sz {
                // Build the [1,1,pd,ph,pw] NCDHW patch (D=x, H=y, W=z).
                let mut buf = vec![0.0f32; plane];
                for i in 0..pd {
                    for j in 0..ph {
                        for k in 0..pw {
                            let p = (x0 + i) + px * ((y0 + j) + py * (z0 + k));
                            buf[(i * ph + j) * pw + k] = vol[p];
                        }
                    }
                }
                let out = model.run_single(&Tensor::new(vec![1, 1, pd, ph, pw], buf))?;
                if out.data.len() < plane {
                    return Err(OnnxError::Run("expected 1-channel output".into()));
                }
                for i in 0..pd {
                    for j in 0..ph {
                        for k in 0..pw {
                            let p = (x0 + i) + px * ((y0 + j) + py * (z0 + k));
                            acc[p] += out.data[(i * ph + j) * pw + k] as f64;
                            wsum[p] += 1.0;
                        }
                    }
                }
                done += 1;
                progress(done, total);
            }
        }
    }

    // Average overlaps, de-normalize to Hz, clip negatives, crop to (nx,ny,nz), mask.
    let mut r2prime = vec![0.0f64; n];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = x + nx * (y + ny * z);
                if mask[i] == 0 {
                    continue;
                }
                let p = x + px * (y + py * z);
                let w = wsum[p].max(1.0);
                let v = ((acc[p] / w) * norm.r2prime.1 + norm.r2prime.0) * norm.dr;
                r2prime[i] = v.max(0.0);
            }
        }
    }
    Ok(r2prime)
}

/// Convenience: fit R2* from multi-echo magnitude with ARLO, then run [`r2primenet`].
///
/// `magnitude` is voxel-major `(n_voxels, n_echoes)` and `echo_times` are in **seconds**
/// (ARLO requires ≥3 equi-spaced echoes). Returns `(r2prime_hz, r2star_hz)` so the caller
/// can keep the intermediate R2* map.
pub fn r2primenet_from_magnitude(
    magnitude: &[f64],
    echo_times: &[f64],
    mask: &[u8],
    grid: &Grid,
    model_onnx: &[u8],
    norm: &R2PrimeNetNorm,
    params: &R2PrimeNetParams,
    progress: impl FnMut(usize, usize),
) -> Result<(Vec<f64>, Vec<f64>), OnnxError> {
    let (r2star, _s0) = super::r2star::r2star_arlo(magnitude, mask, echo_times, grid);
    let r2prime = r2primenet(&r2star, mask, grid, model_onnx, norm, params, progress)?;
    Ok((r2prime, r2star))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both shipped patches are legal for a four-pool-level network.
    #[test]
    fn patches_are_pool_aligned() {
        for (d, h, w) in [AUTHORS_PATCH, WASM_PATCH] {
            for v in [d, h, w] {
                assert_eq!(v % POOL_MULTIPLE, 0, "{v} must be a multiple of {POOL_MULTIPLE}");
            }
        }
        assert_eq!(R2PrimeNetParams::default().patch, AUTHORS_PATCH);
    }

    /// The normalization round-trips: de-normalizing a normalized R2* with the *input*
    /// constants must return the original, which pins the Dr scaling on both sides.
    #[test]
    fn norm_round_trips() {
        let norm = R2PrimeNetNorm::default();
        for hz in [0.0, 12.5, 40.0, 137.0] {
            let z = (hz / norm.dr - norm.r2star.0) / norm.r2star.1;
            let back = (z * norm.r2star.1 + norm.r2star.0) * norm.dr;
            assert!((back - hz).abs() < 1e-9, "{hz} -> {z} -> {back}");
        }
    }
}

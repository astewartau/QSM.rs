//! RS2-Net rodent brain extraction (`onnx` feature).
//!
//! RS2-Net is a Swin-UNETR trained within nnU-Net v2 on 1,142 rat and mouse MRIs from 89 centres
//! (3–17.2 T). It masks rodent brains, where BET's human-scale surface model and HD-BET (trained on
//! humans) both fail. This module ports the nnU-Net pipeline RS2-Net ships around the exported
//! network (`rs2-net.onnx`, see [`crate::models`]), sharing HD-BET's sliding-window and
//! resampling helpers ([`super::hdbet`]):
//!
//! 1. transpose `(z, y, x)` → `(y, z, x)` (the plans' `transpose_forward = [1, 0, 2]`);
//! 2. crop to the bounding box of non-zero voxels, z-score normalise over the crop;
//! 3. resample linearly to 0.25 × 0.2 × 0.16 mm (nnU-Net's "separate z" rule for anisotropic
//!    voxels: linear in-plane, nearest-neighbour through-plane);
//! 4. pad to at least one patch, then Gaussian-weighted sliding-window inference with 50 %
//!    overlap (optionally with 8-fold mirroring test-time augmentation);
//! 5. resample the logits back the same way, threshold the sigmoid at 0.5 (logit > 0), un-crop
//!    and transpose back.
//!
//! One deliberate deviation: RS2-Net's export resamples the logits back with the *untransposed*
//! original spacing (upstream nnU-Net v2 transposes it), so on thick-slice data it picks the wrong
//! axis for nearest-neighbour. This port uses the transposed spacing, as nnU-Net intends; on an
//! in-vivo mouse GRE (0.17 × 0.20 × 0.8 mm) the two masks agree at Dice 0.986. Isotropic data is
//! unaffected.
//!
//! Swin's window attention uses a relative position bias only, so the network runs at any patch
//! that is a multiple of 32 per axis — but the exported graph is traced at a fixed one,
//! [`RS2_NET_PATCH`]: 128 × 96 × 128 instead of RS2-Net's 128 × 128 × 160, which peaks at
//! ≈4.5 GB and does not fit wasm32's address space (this one peaks at ≈2.7 GB). On the in-vivo
//! mouse GRE the two patches agree at Dice 0.987; larger volumes are tiled.
//!
//! The input should be a magnitude image (the first echo of a multi-echo GRE works well), in the
//! scanner's native orientation. RS2-Net applies no post-processing.
//!
//! Reference:
//! Lin, Y., Ding, Y., Chang, S., Ge, X., Sui, X., Jiang, Y. (2024). "RS2-Net: An end-to-end deep
//! learning framework for rodent skull stripping in multi-center brain MRI." NeuroImage,
//! 298:120769. https://doi.org/10.1016/j.neuroimage.2024.120769
//!
//! Reference implementation: https://github.com/VitoLin21/Rodent-Skull-Stripping (GPL-3.0).

// The pre/post-processing is plain Rust; only inference needs `onnx`.
#![cfg_attr(not(feature = "onnx"), allow(dead_code))]

use super::hdbet::{nonzero_bbox, resample_nnunet};
#[cfg(feature = "onnx")]
use super::hdbet::predict_logits;
use crate::grid::Grid;
#[cfg(feature = "onnx")]
use crate::models::onnx::{OnnxError, OnnxModel};

/// nnU-Net target spacing (mm), in the transposed `(y, z, x)` order.
const TARGET_SPACING: [f64; 3] = [0.25, 0.2, 0.16];

/// Patch the exported graph was traced at, in the transposed `(y, z, x)` order.
pub const RS2_NET_PATCH: [usize; 3] = [128, 96, 128];

/// RS2-Net inference parameters.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Rs2NetParams {
    /// Sliding-window step as a fraction of the patch (default 0.5, nnU-Net's).
    pub tile_step: f64,
    /// 8-fold mirroring test-time augmentation (default off: 8× slower, and on an in-vivo mouse
    /// GRE the mask agrees with the mirrored one at Dice 0.987).
    pub mirror_tta: bool,
}

impl Default for Rs2NetParams {
    fn default() -> Self {
        Self { tile_step: 0.5, mirror_tta: false }
    }
}

/// Rodent brain mask via RS2-Net.
///
/// * `magnitude` — magnitude image, column-major `(nx, ny, nz)`.
/// * `model_onnx` — bytes of the exported `rs2-net.onnx` (see [`crate::models`]).
/// * `progress(done, total)` — called after each network evaluation.
///
/// Returns a binary mask (0/1) on the input grid.
#[cfg(feature = "onnx")]
pub fn rs2_net(
    magnitude: &[f64],
    grid: &Grid,
    model_onnx: &[u8],
    params: &Rs2NetParams,
    progress: impl FnMut(usize, usize),
) -> Result<Vec<u8>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    assert_eq!(magnitude.len(), nx * ny * nz, "magnitude length must match grid");
    if !(params.tile_step > 0.0 && params.tile_step <= 1.0) {
        return Err(OnnxError::Shape(format!("tile_step {} must be in (0, 1]", params.tile_step)));
    }
    let Some(pre) = preprocess(magnitude, grid) else {
        return Ok(vec![0; magnitude.len()]);
    };
    let model = OnnxModel::load(model_onnx)?;
    let logits = predict_logits(
        &pre.data, pre.dims, &model, RS2_NET_PATCH, 1, params.tile_step, params.mirror_tta, progress,
    )?;
    Ok(postprocess(&logits, &pre))
}

/// nnU-Net-preprocessed volume plus what is needed to map predictions back.
struct Preprocessed {
    /// Cropped, normalised, resampled volume, C-order `(y, z, x)`.
    data: Vec<f32>,
    dims: [usize; 3],
    /// Crop box `[start, end)` per axis in the transposed `(y, z, x)` volume.
    bbox: [(usize, usize); 3],
    /// Transposed volume shape `(y, z, x)`.
    full_dims: [usize; 3],
    /// Transposed spacing `(y, z, x)`.
    spacing: [f64; 3],
}

/// Column-major `(nx, ny, nz)` — i.e. C-order `(z, y, x)` — to C-order `(y, z, x)`.
fn transpose_forward(data: &[f64], (nx, ny, nz): (usize, usize, usize)) -> Vec<f64> {
    let mut out = vec![0.0; data.len()];
    for z in 0..nz {
        for y in 0..ny {
            let (src, dst) = ((z * ny + y) * nx, (y * nz + z) * nx);
            out[dst..dst + nx].copy_from_slice(&data[src..src + nx]);
        }
    }
    out
}

/// Transpose, crop to non-zero, z-score, resample. `None` if the image is entirely zero.
fn preprocess(magnitude: &[f64], grid: &Grid) -> Option<Preprocessed> {
    let (nx, ny, nz) = grid.dims;
    let (vx, vy, vz) = grid.voxel_size;
    let full_dims = [ny, nz, nx];
    let spacing = [vy, vz, vx];
    let data = transpose_forward(magnitude, grid.dims);

    let bbox = nonzero_bbox(&data, full_dims)?;
    let cdims: [usize; 3] = std::array::from_fn(|a| bbox[a].1 - bbox[a].0);
    let mut img = Vec::with_capacity(cdims.iter().product());
    for i in bbox[0].0..bbox[0].1 {
        for j in bbox[1].0..bbox[1].1 {
            let row = (i * full_dims[1] + j) * full_dims[2];
            img.extend_from_slice(&data[row + bbox[2].0..row + bbox[2].1]);
        }
    }

    // ZScoreNormalization without a mask (RS2-Net's plans: use_mask_for_norm = false).
    let n = img.len() as f64;
    let mean = img.iter().sum::<f64>() / n;
    let std = (img.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
    let s = std.max(1e-8);
    img.iter_mut().for_each(|v| *v = (*v - mean) / s);

    // compute_new_shape: round(spacing / target * shape), Python round (half to even).
    let rdims: [usize; 3] =
        std::array::from_fn(|a| (spacing[a] / TARGET_SPACING[a] * cdims[a] as f64).round_ties_even() as usize);
    // RS2-Net's plans: linear (order 1), nearest through-plane.
    let data = resample_nnunet(&img, cdims, rdims, spacing, TARGET_SPACING, 1)
        .into_iter()
        .map(|v| v as f32)
        .collect();
    Some(Preprocessed { data, dims: rdims, bbox, full_dims, spacing })
}

/// Resample logits back to the cropped grid, threshold, un-crop, transpose back. Returns the
/// column-major `(nx, ny, nz)` mask.
fn postprocess(logits: &[f32], pre: &Preprocessed) -> Vec<u8> {
    let bbox = pre.bbox;
    let cdims: [usize; 3] = std::array::from_fn(|a| bbox[a].1 - bbox[a].0);
    let ch: Vec<f64> = logits.iter().map(|&v| v as f64).collect();
    let back = resample_nnunet(&ch, pre.dims, cdims, TARGET_SPACING, pre.spacing, 1);
    let [ny, nz, nx] = pre.full_dims;
    let mut mask = vec![0u8; ny * nz * nx];
    let mut idx = 0;
    for y in bbox[0].0..bbox[0].1 {
        for z in bbox[1].0..bbox[1].1 {
            for x in bbox[2].0..bbox[2].1 {
                // sigmoid(logit) > 0.5, written into the column-major (z, y, x) layout
                mask[(z * ny + y) * nx + x] = (back[idx] > 0.0) as u8;
                idx += 1;
            }
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_swaps_z_and_y() {
        let (nx, ny, nz) = (2, 3, 4);
        let v: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let t = transpose_forward(&v, (nx, ny, nz));
        // column-major voxel (x=1, y=2, z=3) lands at C-order (y=2, z=3, x=1) of [ny, nz, nx]
        assert_eq!(t[(2 * nz + 3) * nx + 1], v[(3 * ny + 2) * nx + 1]);
    }

    #[test]
    fn patch_is_swin_compatible() {
        // Swin-UNETR downsamples five times: every patch axis must be a multiple of 32.
        assert!(RS2_NET_PATCH.iter().all(|&p| p > 0 && p % 32 == 0));
    }

    #[test]
    fn empty_image_gives_no_preprocessing() {
        assert!(preprocess(&[0.0; 8], &Grid::new(2, 2, 2, 0.1, 0.1, 0.1)).is_none());
    }

    #[test]
    fn postprocess_inverts_preprocess_geometry() {
        // A bright box in a thick-slice volume: thresholding the preprocessed (z-scored,
        // resampled) intensities at the box/background midpoint, as if they were logits, must
        // give the box back exactly — i.e. crop, transpose and both resamplings round-trip.
        let (nx, ny, nz) = (40, 30, 8);
        let mut v = vec![0.001; nx * ny * nz];
        for z in 2..6 {
            for y in 8..22 {
                for x in 10..30 {
                    v[(z * ny + y) * nx + x] = 1.0;
                }
            }
        }
        let pre = preprocess(&v, &Grid::new(nx, ny, nz, 0.2, 0.2, 0.8)).unwrap();
        // 0.2 x 0.8 x 0.2 mm (transposed) is anisotropic past 3x: separate-z resampling
        assert_eq!(super::super::hdbet::separate_z_axis(pre.spacing, TARGET_SPACING), Some(1));
        let (lo, hi) = pre.data.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        let logits: Vec<f32> = pre.data.iter().map(|v| v - (lo + hi) / 2.0).collect();
        let mask = postprocess(&logits, &pre);
        let want: Vec<u8> = v.iter().map(|&x| (x > 0.5) as u8).collect();
        assert_eq!(mask, want);
    }

    // ---- parity against RS2-Net's own pipeline: see tests/models_onnx.rs (rs2net_*) ----

    /// Minimal little-endian f32 C-order `.npy` reader.
    fn read_npy_f32(path: &str) -> (Vec<usize>, Vec<f32>) {
        let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        assert_eq!(&b[..6], b"\x93NUMPY");
        let (hlen, off) = if b[6] == 1 {
            (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
        } else {
            (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
        };
        let header = std::str::from_utf8(&b[off..off + hlen]).unwrap();
        assert!(header.contains("'<f4'") && header.contains("'fortran_order': False"), "{header}");
        let shape_str = header.split("'shape': (").nth(1).unwrap().split(')').next().unwrap();
        let shape: Vec<usize> = shape_str.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        let data = b[off + hlen..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        (shape, data)
    }

    /// `RS2_REF_DIR` holds `mag.nii` plus `ref_preprocessed.npy`, `ref_logits.npy` and
    /// `ref_mask.nii.gz` written by `scripts/onnx-export/ref_rs2net.py` on it.
    fn load_ref() -> Option<(String, Vec<f64>, Grid)> {
        let dir = std::env::var("RS2_REF_DIR").ok()?;
        let nii = crate::io::read_nifti_file(std::path::Path::new(&format!("{dir}/mag.nii"))).unwrap();
        let (nx, ny, nz) = nii.dims;
        let (vx, vy, vz) = nii.voxel_size;
        Some((dir, nii.data, Grid::new(nx, ny, nz, vx, vy, vz)))
    }

    /// Transpose + crop + z-score + resample vs RS2-Net's `DefaultPreprocessor.run_case`.
    #[test]
    #[ignore]
    fn preprocessing_matches_rs2net() {
        let Some((dir, mag, grid)) = load_ref() else { return eprintln!("RS2_REF_DIR not set; skipping") };
        let pre = preprocess(&mag, &grid).unwrap();
        let (shape, want) = read_npy_f32(&format!("{dir}/ref_preprocessed.npy"));
        assert_eq!(pre.dims.to_vec(), shape);
        let err = pre.data.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("dims {:?}, bbox {:?}, max |d| {err:e}", pre.dims, pre.bbox);
        assert!(err < 1e-3, "preprocessed max |d| = {err}");
    }

    /// Resample-back + threshold + un-crop applied to RS2-Net's own logits reproduces its mask
    /// (exported with the spacing transposed; see the module docs).
    #[test]
    #[ignore]
    fn postprocessing_matches_rs2net() {
        let Some((dir, mag, grid)) = load_ref() else { return eprintln!("RS2_REF_DIR not set; skipping") };
        let pre = preprocess(&mag, &grid).unwrap();
        let (shape, logits) = read_npy_f32(&format!("{dir}/ref_logits.npy"));
        // RS2-Net broadcasts its one output channel into two identical ones; take the first.
        let n: usize = shape[1..].iter().product();
        let mask = postprocess(&logits[..n], &pre);
        let want = crate::io::read_nifti_file(std::path::Path::new(&format!("{dir}/ref_mask.nii.gz"))).unwrap();
        let diff = mask.iter().zip(&want.data).filter(|(&m, &w)| (m != 0) != (w > 0.5)).count();
        eprintln!("{diff} voxels differ from RS2-Net's mask");
        assert_eq!(diff, 0, "{diff} differing voxels");
    }
}

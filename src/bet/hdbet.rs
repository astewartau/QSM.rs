//! HD-BET deep-learning brain extraction (`onnx` feature).
//!
//! HD-BET v2 is an nnU-Net v2 3D U-Net (`PlainConvUNet`, 1 mm isotropic, 96×192×192 patches)
//! trained on 11,751 multi-sequence clinical MRIs; it segments the brain from a single magnitude
//! image. This module is a faithful port of nnU-Net's inference pipeline around the exported
//! network (`hd-bet.onnx`, see [`crate::models`]):
//!
//! 1. crop to the bounding box of non-zero voxels;
//! 2. z-score normalise over the cropped image;
//! 3. resample to 1 mm with cubic splines (nnU-Net's "separate z" rule for anisotropic voxels:
//!    cubic in-plane, nearest-neighbour through-plane);
//! 4. pad to at least one patch, then Gaussian-weighted sliding-window inference with 50 %
//!    overlap (optionally with 8-fold mirroring test-time augmentation);
//! 5. resample the logits back (linear; nearest through-plane for anisotropic voxels), take the
//!    argmax and un-crop.
//!
//! nnU-Net works on SimpleITK arrays in `(z, y, x)` order, which is exactly the memory layout of
//! the crate's column-major `(nx, ny, nz)` volumes — no transposition is needed.
//!
//! The input should be a magnitude image (for multi-echo GRE, the root-sum-of-squares over
//! echoes works well) in roughly standard radiological orientation (axial slices along z), as
//! HD-BET was trained on MNI-aligned data. HD-BET applies no post-processing; combine with
//! [`crate::utils::fill_holes`] / erosion refinements as needed.
//!
//! Reference:
//! Isensee, F., Schell, M., Pflueger, I., et al. (2019). "Automated brain extraction of
//! multisequence MRI using artificial neural networks." Human Brain Mapping, 40(17):4952-4964.
//! https://doi.org/10.1002/hbm.24750
//!
//! Reference implementation: https://github.com/MIC-DKFZ/HD-BET (Apache-2.0; weights CC-BY-NC-4.0).

// The pre/post-processing is plain Rust; only inference needs `onnx`.
#![cfg_attr(not(feature = "onnx"), allow(dead_code))]

use crate::grid::Grid;
#[cfg(feature = "onnx")]
use crate::models::onnx::{OnnxError, OnnxModel, Tensor};
use crate::utils::resample::{resize, resize_axis};

/// HD-BET inference parameters.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct HdBetParams {
    /// Sliding-window patch size `(px, py, pz)` in voxels at 1 mm (default `(192, 192, 96)`, the
    /// size HD-BET was trained with). Must be multiples of `(32, 32, 16)`. Smaller patches bound
    /// memory (peak ≈1.9 GB for `(128, 128, 64)` vs ≈4.5 GB native, on a 164×205×205 volume) at a
    /// small accuracy cost; below ~`(128, 128, 64)` tiles that lie wholly inside the brain start
    /// to be labelled background.
    pub patch: (usize, usize, usize),
    /// Sliding-window step as a fraction of the patch (default 0.5, nnU-Net's).
    pub tile_step: f64,
    /// 8-fold mirroring test-time augmentation (default off, as in QSM-CI; 8× slower for a
    /// marginal gain).
    pub mirror_tta: bool,
}

impl Default for HdBetParams {
    fn default() -> Self {
        Self { patch: (192, 192, 96), tile_step: 0.5, mirror_tta: false }
    }
}

impl HdBetParams {
    /// Memory-bounded setting for constrained hosts (e.g. 32-bit WASM): `(128, 128, 64)` patches,
    /// peak ≈1.9 GB on a 164×205×205 volume. Agreed with the native patch at Dice ≈0.99 in
    /// validation (and with nnU-Net run at the same patch size to 2 voxels).
    pub fn low_memory() -> Self {
        Self { patch: (128, 128, 64), ..Self::default() }
    }
}

/// nnU-Net target spacing for HD-BET (mm).
const TARGET_SPACING: [f64; 3] = [1.0, 1.0, 1.0];
/// Required divisibility of the patch in nnU-Net `(z, y, x)` order (5 down-samplings, the last
/// one in-plane only).
const PATCH_DIVISOR: [usize; 3] = [16, 32, 32];

/// Brain mask via HD-BET.
///
/// * `magnitude` — magnitude image, column-major `(nx, ny, nz)`.
/// * `model_onnx` — bytes of the exported `hd-bet.onnx` (see [`crate::models`]).
/// * `progress(done, total)` — called after each network evaluation.
///
/// Returns a binary mask (0/1) on the input grid.
#[cfg(feature = "onnx")]
pub fn hd_bet(
    magnitude: &[f64],
    grid: &Grid,
    model_onnx: &[u8],
    params: &HdBetParams,
    progress: impl FnMut(usize, usize),
) -> Result<Vec<u8>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    assert_eq!(magnitude.len(), nx * ny * nz, "magnitude length must match grid");
    let patch = [params.patch.2, params.patch.1, params.patch.0];
    if patch.iter().zip(PATCH_DIVISOR).any(|(&p, d)| p == 0 || p % d != 0) {
        return Err(OnnxError::Shape(format!(
            "HD-BET patch {:?} must be a non-zero multiple of (32, 32, 16)",
            params.patch
        )));
    }
    if !(params.tile_step > 0.0 && params.tile_step <= 1.0) {
        return Err(OnnxError::Shape(format!("tile_step {} must be in (0, 1]", params.tile_step)));
    }
    let Some(pre) = preprocess(magnitude, grid) else {
        return Ok(vec![0; magnitude.len()]);
    };
    let model = OnnxModel::load(model_onnx)?;
    let logits = predict_logits(&pre.data, pre.dims, &model, patch, params, progress)?;
    Ok(postprocess(&logits, &pre))
}

/// nnU-Net-preprocessed volume plus what is needed to map predictions back.
struct Preprocessed {
    /// Cropped, normalised, resampled volume, C-order `(z, y, x)`.
    data: Vec<f32>,
    dims: [usize; 3],
    /// Crop box `[start, end)` per axis in the original `(z, y, x)` volume.
    bbox: [(usize, usize); 3],
    /// Original volume shape `(z, y, x)`.
    full_dims: [usize; 3],
    /// Original spacing `(z, y, x)`.
    spacing: [f64; 3],
}

/// Crop to non-zero, z-score, resample to 1 mm. `None` if the image is entirely zero.
fn preprocess(magnitude: &[f64], grid: &Grid) -> Option<Preprocessed> {
    let (nx, ny, nz) = grid.dims;
    let (vx, vy, vz) = grid.voxel_size;
    let full_dims = [nz, ny, nx];
    let spacing = [vz, vy, vx];

    let bbox = nonzero_bbox(magnitude, full_dims)?;
    let cdims = [bbox[0].1 - bbox[0].0, bbox[1].1 - bbox[1].0, bbox[2].1 - bbox[2].0];
    let mut img = Vec::with_capacity(cdims.iter().product());
    for z in bbox[0].0..bbox[0].1 {
        for y in bbox[1].0..bbox[1].1 {
            let row = (z * ny + y) * nx;
            img.extend_from_slice(&magnitude[row + bbox[2].0..row + bbox[2].1]);
        }
    }

    // ZScoreNormalization without a mask (HD-BET's plans: use_mask_for_norm = false).
    let n = img.len() as f64;
    let mean = img.iter().sum::<f64>() / n;
    let std = (img.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
    let s = std.max(1e-8);
    img.iter_mut().for_each(|v| *v = (*v - mean) / s);

    // compute_new_shape: round(spacing / target * shape), Python round (half to even).
    let rdims: [usize; 3] =
        std::array::from_fn(|a| (spacing[a] / TARGET_SPACING[a] * cdims[a] as f64).round_ties_even() as usize);
    let data = resample_nnunet(&img, cdims, rdims, spacing, TARGET_SPACING, 3)
        .into_iter()
        .map(|v| v as f32)
        .collect();
    Some(Preprocessed { data, dims: rdims, bbox, full_dims, spacing })
}

/// Bounding box `[start, end)` of the non-zero voxels of a C-order `(z, y, x)` volume.
fn nonzero_bbox(data: &[f64], dims: [usize; 3]) -> Option<[(usize, usize); 3]> {
    let mut lo = [usize::MAX; 3];
    let mut hi = [0usize; 3];
    let mut any = false;
    for z in 0..dims[0] {
        for y in 0..dims[1] {
            let row = (z * dims[1] + y) * dims[2];
            for x in 0..dims[2] {
                if data[row + x] != 0.0 {
                    any = true;
                    for (a, c) in [z, y, x].into_iter().enumerate() {
                        lo[a] = lo[a].min(c);
                        hi[a] = hi[a].max(c + 1);
                    }
                }
            }
        }
    }
    any.then(|| std::array::from_fn(|a| (lo[a], hi[a])))
}

/// nnU-Net `resample_data_or_seg_to_shape` for one channel (non-segmentation, `order_z = 0`,
/// `force_separate_z = None`): a plain 3D spline resize, unless the voxels are anisotropic
/// (max/min spacing > 3), in which case each slice is resized in-plane with `order` and the
/// low-resolution axis with nearest-neighbour.
fn resample_nnunet(
    data: &[f64],
    dims: [usize; 3],
    new_dims: [usize; 3],
    current_spacing: [f64; 3],
    new_spacing: [f64; 3],
    order: usize,
) -> Vec<f64> {
    if dims == new_dims {
        return data.to_vec();
    }
    let Some(axis) = separate_z_axis(current_spacing, new_spacing) else {
        return resize(data, dims, new_dims, order);
    };
    let inplane: Vec<usize> = (0..3).filter(|&a| a != axis).collect();
    let mut out = data.to_vec();
    let mut cur = dims;
    if inplane.iter().any(|&a| dims[a] != new_dims[a]) {
        // skimage.resize on each 2D slice: both in-plane axes (a same-length axis still goes
        // through the spline for order 3, as in scipy), then clip each slice to its own range.
        for &a in &inplane {
            if order == 3 || cur[a] != new_dims[a] {
                (out, cur) = resize_axis(&out, cur, a, new_dims[a], order);
            }
        }
        if order > 1 {
            let range = slice_ranges(data, dims, axis);
            for_each_slice(&mut out, cur, axis, |k, v| *v = v.clamp(range[k].0, range[k].1));
        }
    }
    if cur[axis] != new_dims[axis] {
        (out, _) = resize_axis(&out, cur, axis, new_dims[axis], 0);
    }
    out
}

/// nnU-Net `determine_do_sep_z_and_axis(force_separate_z=None, ...)`: the low-resolution axis if
/// either spacing is anisotropic by more than 3×, and that axis is unique.
fn separate_z_axis(current: [f64; 3], new: [f64; 3]) -> Option<usize> {
    let aniso = |s: [f64; 3]| {
        let (mn, mx) = s.iter().fold((f64::INFINITY, 0.0f64), |(a, b), &v| (a.min(v), b.max(v)));
        mx / mn > 3.0
    };
    let lowres = |s: [f64; 3]| {
        let mx = s.iter().copied().fold(0.0f64, f64::max);
        let axes: Vec<usize> = (0..3).filter(|&a| mx / s[a] == 1.0).collect();
        (axes.len() == 1).then(|| axes[0])
    };
    if aniso(current) {
        lowres(current)
    } else if aniso(new) {
        lowres(new)
    } else {
        None
    }
}

/// Per-slice `(min, max)` along `axis` of a C-order volume.
fn slice_ranges(data: &[f64], dims: [usize; 3], axis: usize) -> Vec<(f64, f64)> {
    let mut r = vec![(f64::INFINITY, f64::NEG_INFINITY); dims[axis]];
    let mut idx = 0;
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            for k in 0..dims[2] {
                let s = [i, j, k][axis];
                let v = data[idx];
                r[s] = (r[s].0.min(v), r[s].1.max(v));
                idx += 1;
            }
        }
    }
    r
}

/// Apply `f(slice_index_along_axis, &mut value)` to every voxel of a C-order volume.
fn for_each_slice(data: &mut [f64], dims: [usize; 3], axis: usize, mut f: impl FnMut(usize, &mut f64)) {
    let mut idx = 0;
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            for k in 0..dims[2] {
                f([i, j, k][axis], &mut data[idx]);
                idx += 1;
            }
        }
    }
}

/// nnU-Net `compute_steps_for_sliding_window` for one axis.
fn window_steps(size: usize, tile: usize, step: f64) -> Vec<usize> {
    let n = ((size - tile) as f64 / (tile as f64 * step)).ceil() as usize + 1;
    if n == 1 {
        return vec![0];
    }
    let actual = (size - tile) as f64 / (n - 1) as f64;
    (0..n).map(|i| (actual * i as f64).round_ties_even() as usize).collect()
}

/// nnU-Net `compute_gaussian(tile, sigma_scale=1/8, value_scaling_factor=10)`: a Gaussian
/// centred on the patch (σ = tile/8 per axis), peak 10.
fn gaussian_importance(tile: [usize; 3]) -> Vec<f32> {
    let w: [Vec<f64>; 3] = std::array::from_fn(|a| {
        let (c, s) = ((tile[a] / 2) as f64, tile[a] as f64 / 8.0);
        (0..tile[a]).map(|i| (-0.5 * ((i as f64 - c) / s).powi(2)).exp()).collect()
    });
    let mut g = Vec::with_capacity(tile.iter().product());
    for &a in &w[0] {
        for &b in &w[1] {
            for &c in &w[2] {
                g.push((10.0 * a * b * c) as f32);
            }
        }
    }
    g
}

/// Gaussian-weighted sliding-window logits (2 channels, C-order `(z, y, x)`, shape `dims`).
#[cfg(feature = "onnx")]
fn predict_logits(
    data: &[f32],
    dims: [usize; 3],
    model: &OnnxModel,
    patch: [usize; 3],
    params: &HdBetParams,
    mut progress: impl FnMut(usize, usize),
) -> Result<Vec<f32>, OnnxError> {
    // Pad (centred, extra voxel on the high side) to at least one patch, with zeros (= the mean
    // after normalisation), as pad_nd_image does.
    let pdims: [usize; 3] = std::array::from_fn(|a| dims[a].max(patch[a]));
    let lo: [usize; 3] = std::array::from_fn(|a| (pdims[a] - dims[a]) / 2);
    let np: usize = pdims.iter().product();
    let mut padded = vec![0.0f32; np];
    for z in 0..dims[0] {
        for y in 0..dims[1] {
            let src = (z * dims[1] + y) * dims[2];
            let dst = ((z + lo[0]) * pdims[1] + y + lo[1]) * pdims[2] + lo[2];
            padded[dst..dst + dims[2]].copy_from_slice(&data[src..src + dims[2]]);
        }
    }

    let steps: [Vec<usize>; 3] = std::array::from_fn(|a| window_steps(pdims[a], patch[a], params.tile_step));
    let gauss = gaussian_importance(patch);
    let pp: usize = patch.iter().product();
    let shape = [1, 1, patch[0], patch[1], patch[2]];
    let plan = model.plan_for(&[&shape])?;
    let flips: &[[bool; 3]] = if params.mirror_tta {
        &[[false, false, false], [true, false, false], [false, true, false], [false, false, true],
          [true, true, false], [true, false, true], [false, true, true], [true, true, true]]
    } else {
        &[[false, false, false]]
    };
    let total = steps.iter().map(Vec::len).product::<usize>() * flips.len();
    let mut done = 0;

    let mut acc = vec![0.0f32; 2 * np];
    let mut weight = vec![0.0f32; np];
    let mut input = vec![0.0f32; pp];
    let mut pred = vec![0.0f32; 2 * pp];
    for &s0 in &steps[0] {
        for &s1 in &steps[1] {
            for &s2 in &steps[2] {
                for_each_patch_row(pdims, patch, [s0, s1, s2], |src, dst| {
                    input[dst..dst + patch[2]].copy_from_slice(&padded[src..src + patch[2]]);
                });
                pred.iter_mut().for_each(|v| *v = 0.0);
                for flip in flips {
                    let x = flip3(&input, patch, *flip);
                    let out = plan.run_single(&Tensor::new(shape.to_vec(), x))?;
                    if out.shape != [1, 2, patch[0], patch[1], patch[2]] {
                        return Err(OnnxError::Run(format!("unexpected HD-BET output shape {:?}", out.shape)));
                    }
                    for c in 0..2 {
                        let back = flip3(&out.data[c * pp..(c + 1) * pp], patch, *flip);
                        pred[c * pp..(c + 1) * pp].iter_mut().zip(back).for_each(|(p, v)| *p += v);
                    }
                    done += 1;
                    progress(done, total);
                }
                let inv = 1.0 / flips.len() as f32;
                for_each_patch_row(pdims, patch, [s0, s1, s2], |dst, src| {
                    for k in 0..patch[2] {
                        let g = gauss[src + k];
                        weight[dst + k] += g;
                        for c in 0..2 {
                            acc[c * np + dst + k] += pred[c * pp + src + k] * inv * g;
                        }
                    }
                });
            }
        }
    }

    // Normalise and crop the padding back off.
    let n: usize = dims.iter().product();
    let mut logits = vec![0.0f32; 2 * n];
    for c in 0..2 {
        for z in 0..dims[0] {
            for y in 0..dims[1] {
                let src = ((z + lo[0]) * pdims[1] + y + lo[1]) * pdims[2] + lo[2];
                let dst = (z * dims[1] + y) * dims[2];
                for x in 0..dims[2] {
                    logits[c * n + dst + x] = acc[c * np + src + x] / weight[src + x];
                }
            }
        }
    }
    Ok(logits)
}

/// Visit the rows of a patch at `origin` inside a volume: `f(volume_row_start, patch_row_start)`.
fn for_each_patch_row(vdims: [usize; 3], patch: [usize; 3], origin: [usize; 3], mut f: impl FnMut(usize, usize)) {
    for z in 0..patch[0] {
        for y in 0..patch[1] {
            let v = ((origin[0] + z) * vdims[1] + origin[1] + y) * vdims[2] + origin[2];
            f(v, (z * patch[1] + y) * patch[2]);
        }
    }
}

/// Mirror a C-order patch along the flagged axes (an involution).
fn flip3(src: &[f32], dims: [usize; 3], flip: [bool; 3]) -> Vec<f32> {
    if flip == [false; 3] {
        return src.to_vec();
    }
    let mut out = vec![0.0f32; src.len()];
    let m = |i: usize, a: usize| if flip[a] { dims[a] - 1 - i } else { i };
    for z in 0..dims[0] {
        for y in 0..dims[1] {
            for x in 0..dims[2] {
                out[(m(z, 0) * dims[1] + m(y, 1)) * dims[2] + m(x, 2)] = src[(z * dims[1] + y) * dims[2] + x];
            }
        }
    }
    out
}

/// Resample logits back to the cropped grid, argmax, un-crop. Returns the column-major mask.
fn postprocess(logits: &[f32], pre: &Preprocessed) -> Vec<u8> {
    let n = pre.data.len();
    let bbox = pre.bbox;
    let cdims = [bbox[0].1 - bbox[0].0, bbox[1].1 - bbox[1].0, bbox[2].1 - bbox[2].0];
    let back = |c: usize| -> Vec<f64> {
        let ch: Vec<f64> = logits[c * n..(c + 1) * n].iter().map(|&v| v as f64).collect();
        resample_nnunet(&ch, pre.dims, cdims, TARGET_SPACING, pre.spacing, 1)
    };
    let (bg, fg) = (back(0), back(1));
    let fd = pre.full_dims;
    let mut mask = vec![0u8; fd.iter().product()];
    let mut i = 0;
    for z in bbox[0].0..bbox[0].1 {
        for y in bbox[1].0..bbox[1].1 {
            let row = (z * fd[1] + y) * fd[2];
            for x in bbox[2].0..bbox[2].1 {
                // argmax over (background, brain); a tie goes to background like torch.argmax
                mask[row + x] = (fg[i] > bg[i]) as u8;
                i += 1;
            }
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_steps_match_nnunet() {
        // nnU-Net docstring example: image 110, patch 64, step 0.5 -> 0, 23, 46
        assert_eq!(window_steps(110, 64, 0.5), vec![0, 23, 46]);
        assert_eq!(window_steps(205, 96, 0.5), vec![0, 36, 73, 109]);
        assert_eq!(window_steps(192, 192, 0.5), vec![0]);
        assert_eq!(window_steps(205, 192, 1.0), vec![0, 13]);
    }

    #[test]
    fn gaussian_peaks_at_centre_with_value_10() {
        let g = gaussian_importance([16, 32, 32]);
        let centre = (8 * 32 + 16) * 32 + 16;
        assert!((g[centre] - 10.0).abs() < 1e-6);
        assert!(g.iter().all(|&v| v > 0.0 && v <= 10.0));
    }

    #[test]
    fn separate_z_rule() {
        assert_eq!(separate_z_axis([4.0, 0.9, 0.9], [1.0; 3]), Some(0));
        assert_eq!(separate_z_axis([1.0; 3], [0.9, 0.9, 4.0]), Some(2));
        assert_eq!(separate_z_axis([1.2, 0.9, 0.9], [1.0; 3]), None);
        // two equally coarse axes -> no separate-z (nnU-Net: len(axis) == 2)
        assert_eq!(separate_z_axis([4.0, 4.0, 1.0], [1.0; 3]), None);
    }

    #[test]
    fn flip_is_an_involution() {
        let d = [2, 3, 4];
        let v: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let f = flip3(&v, d, [true, false, true]);
        assert_ne!(f, v);
        assert_eq!(flip3(&f, d, [true, false, true]), v);
        // voxel (0,0,0) of the flipped patch is source voxel (z=1, y=0, x=3)
        assert_eq!(f[0], v[(3 * 4) + 3]);
    }

    #[test]
    fn bbox_of_nonzero() {
        let mut d = vec![0.0; 3 * 4 * 5];
        let at = |z: usize, y: usize, x: usize| (z * 4 + y) * 5 + x;
        d[at(1, 2, 3)] = 1.0;
        d[at(2, 1, 1)] = -2.0;
        assert_eq!(nonzero_bbox(&d, [3, 4, 5]), Some([(1, 3), (1, 3), (1, 4)]));
        assert_eq!(nonzero_bbox(&[0.0; 8], [2, 2, 2]), None);
    }

    // ---- parity against nnU-Net intermediates (ref_hdbet.py in qsm-ci scripts/onnx-export) ----

    /// Minimal little-endian f32 C-order `.npy` reader.
    fn read_npy_f32(path: &str) -> (Vec<usize>, Vec<f32>) {
        let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        assert_eq!(&b[..6], b"\x93NUMPY");
        let (hlen, off) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else {
            (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
        };
        let header = std::str::from_utf8(&b[off..off + hlen]).unwrap();
        assert!(header.contains("'<f4'") && header.contains("'fortran_order': False"), "{header}");
        let shape_str = header.split("'shape': (").nth(1).unwrap().split(')').next().unwrap();
        let shape: Vec<usize> = shape_str.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        let data = b[off + hlen..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        (shape, data)
    }

    fn ref_dir() -> Option<String> {
        std::env::var("HDBET_REF_DIR").ok()
    }

    fn load_case(dir: &str, case: &str) -> (Vec<f64>, Grid) {
        let nii = crate::io::read_nifti_file(std::path::Path::new(&format!("{dir}/{case}_mag.nii.gz"))).unwrap();
        let (nx, ny, nz) = nii.dims;
        let (vx, vy, vz) = nii.voxel_size;
        (nii.data, Grid::new(nx, ny, nz, vx, vy, vz))
    }

    /// Crop + z-score + resample vs nnU-Net's `run_case_npy` for the three reference cases.
    #[test]
    #[ignore]
    fn preprocessing_matches_nnunet() {
        let Some(dir) = ref_dir() else { return eprintln!("HDBET_REF_DIR not set; skipping") };
        for case in ["A", "B", "C"] {
            let (mag, grid) = load_case(&dir, case);
            let pre = preprocess(&mag, &grid).unwrap();
            let (shape, want) = read_npy_f32(&format!("{dir}/{case}_preprocessed.npy"));
            assert_eq!(pre.dims.to_vec(), shape, "case {case}: shape");
            let err = pre.data.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            eprintln!("case {case}: dims {:?}, bbox {:?}, max |d| {err:e}", pre.dims, pre.bbox);
            assert!(err < 1e-3, "case {case}: preprocessed max |d| = {err}");
        }
    }

    /// Resample-back + argmax + un-crop applied to nnU-Net's own logits reproduces the CLI mask.
    #[test]
    #[ignore]
    fn postprocessing_matches_nnunet() {
        let Some(dir) = ref_dir() else { return eprintln!("HDBET_REF_DIR not set; skipping") };
        for case in ["A", "B", "C"] {
            let (mag, grid) = load_case(&dir, case);
            let pre = preprocess(&mag, &grid).unwrap();
            let (_, logits) = read_npy_f32(&format!("{dir}/{case}_logits.npy"));
            let mask = postprocess(&logits, &pre);
            let cli = crate::io::read_nifti_file(std::path::Path::new(&format!("{dir}/{case}_mask_cli.nii.gz"))).unwrap();
            let diff = mask.iter().zip(&cli.data).filter(|(&m, &c)| (m != 0) != (c > 0.5)).count();
            eprintln!("case {case}: {diff} voxels differ from the CLI mask");
            assert!(diff * 100_000 < mask.len(), "case {case}: {diff} differing voxels");
        }
    }
}

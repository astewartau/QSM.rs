//! MindGrab deep-learning brain extraction (`onnx` feature).
//!
//! MindGrab is a 146 k-parameter MeshNet — 25 dilated 3×3×3 convolutions (15 channels,
//! dilations 16-8-4-2-1 five times over, each followed by instance normalisation and GELU) and a
//! 1×1×1 two-class head — trained only on synthetic, modality-agnostic data, so it skull-strips
//! head images of any contrast. This module ports the pipeline brainchop-cli runs around the
//! network (`mindgrab.onnx`, see [`crate::models`]) for `brainchop --skull-strip`:
//!
//! 1. conform the image as niimath's `-conform` does (a port of FastSurfer's `conform.py`):
//!    scale intensities to 0–255 over the robust range (0 to the 98th percentile of the
//!    histogram), then reslice trilinearly to a 256³, 1 mm, LIA grid centred on the image, and
//!    round to `u8`;
//! 2. normalise to the unit interval between the 2nd and 98th percentiles, clipping outside;
//! 3. run the network on the whole 256³ volume (optionally also on its left–right mirror,
//!    summing the logits, as brainchop's `--tta` does);
//! 4. take the argmax, keep the largest 26-connected component (niimath `-bwlabel 26`);
//! 5. reslice the mask back onto the input grid with nearest neighbour (niimath `-reslice_nn`).
//!
//! The conform step needs the image's voxel→world affine: the network expects its input in a
//! fixed anatomical orientation, and that is the only way to find it. Oblique acquisitions are
//! resliced, not just permuted, as niimath does.
//!
//! Instance normalisation runs over the whole volume, so the network cannot be tiled: one pass
//! holds a few 15-channel 256³ activations (≈1 GB each in `f32`).
//!
//! Reference:
//! Fani, A., Masoud, M., Rorden, C., Plis, S. (2025). "MindGrab for BrainChop: Fast and
//! Accurate Skull Stripping for Command Line and Browser." arXiv:2506.11860.
//! https://doi.org/10.48550/arXiv.2506.11860
//!
//! Reference implementation: https://github.com/neuroneural/brainchop-cli (MIT), with niimath
//! (https://github.com/rordenlab/niimath, BSD-2-Clause) for conform, labelling and reslicing.

// The pre/post-processing is plain Rust; only inference needs `onnx`.
#![cfg_attr(not(feature = "onnx"), allow(dead_code))]

#[cfg(feature = "onnx")]
use crate::grid::Grid;
#[cfg(feature = "onnx")]
use crate::models::onnx::{OnnxError, OnnxModel, Tensor};
#[cfg(feature = "onnx")]
use crate::utils::connected::{largest_component, Connectivity};

/// Side of the conformed cube, in voxels (1 mm each).
pub const CONFORM_DIM: usize = 256;

/// Upper fraction of the histogram kept when scaling to `u8` (niimath `-conform`; FreeSurfer's
/// `mri_convert` uses 0.999).
const CONFORM_F_HIGH: f64 = 0.98;

/// MindGrab inference parameters.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Clone, Debug, PartialEq, Default)]
pub struct MindGrabParams {
    /// Left–right flip test-time augmentation: also run the mirrored volume and sum the logits
    /// (brainchop's `--tta`). Doubles the run time. Default off, as brainchop runs it.
    pub flip_tta: bool,
}

/// Brain mask via MindGrab.
///
/// * `image` — head image of any contrast (for GRE, a magnitude), column-major `(nx, ny, nz)`.
/// * `affine` — the image's voxel→world affine (row-major 4×4, as in a NIfTI sform/qform).
/// * `model_onnx` — bytes of the exported `mindgrab.onnx` (see [`crate::models`]).
/// * `progress(done, total)` — called after each network evaluation.
///
/// Returns a binary mask (0/1) on the input grid.
#[cfg(feature = "onnx")]
pub fn mindgrab(
    image: &[f64],
    grid: &Grid,
    affine: &[f64; 16],
    model_onnx: &[u8],
    params: &MindGrabParams,
    mut progress: impl FnMut(usize, usize),
) -> Result<Vec<u8>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    assert_eq!(image.len(), nx * ny * nz, "image length must match grid");
    let conformed = Conformed::new(image, grid.dims, affine)
        .ok_or_else(|| OnnxError::Shape("MindGrab needs a non-singular affine".into()))?;
    let input = normalize(&conformed.data);

    let model = OnnxModel::load(model_onnx)?;
    let n = CONFORM_DIM;
    let shape = vec![1, 1, n, n, n];
    let passes = if params.flip_tta { 2 } else { 1 };
    let mut logits = model.run_single(&Tensor::new(shape.clone(), input.clone()))?.data;
    progress(1, passes);
    if params.flip_tta {
        let flipped = model.run_single(&Tensor::new(shape, flip_axis0(&input, 1)))?.data;
        logits.iter_mut().zip(flip_axis0(&flipped, 2)).for_each(|(l, f)| *l += f);
        progress(2, passes);
    }

    let mask = argmax_mask(&logits);
    let mask = largest_component(&mask, (n, n, n), Connectivity::Full);
    Ok(conformed.reslice_back(&mask))
}

/// The input conformed to the 256³ 1 mm LIA grid, plus what is needed to map a mask back.
struct Conformed {
    /// Conformed `u8` intensities, column-major `(256, 256, 256)`.
    data: Vec<u8>,
    /// Input dimensions.
    dims: (usize, usize, usize),
    /// Input voxel → conformed voxel (row-major 3×4).
    to_conformed: [[f64; 4]; 3],
}

impl Conformed {
    /// niimath `-conform`: scale to `u8`'s range, reslice to 256³ 1 mm LIA. `None` if the affine
    /// is singular.
    fn new(image: &[f64], dims: (usize, usize, usize), affine: &[f64; 16]) -> Option<Self> {
        let out_affine = conform_affine(dims, affine);
        // conformed voxel → input voxel
        let from_conformed = compose(&invert(affine)?, &out_affine);
        let to_conformed = invert(&from_conformed).map(|m| top3(&m))?;
        let scaled = scale_to_u8_range(image);
        let data = reslice_linear(&scaled, dims, &top3(&from_conformed));
        Some(Self { data, dims, to_conformed })
    }

    /// niimath `-reslice_nn`: nearest-neighbour from the conformed grid back onto the input grid
    /// (rounding half away from zero; 0 outside the conformed cube).
    fn reslice_back(&self, mask: &[u8]) -> Vec<u8> {
        let (nx, ny, nz) = self.dims;
        let n = CONFORM_DIM as i64;
        let m = &self.to_conformed;
        let mut out = vec![0u8; nx * ny * nz];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let c = apply(m, x as f64, y as f64, z as f64).map(|v| v.round() as i64);
                    if c.iter().all(|&v| (0..n).contains(&v)) {
                        out[x + nx * (y + ny * z)] = mask[(c[0] + n * (c[1] + n * c[2])) as usize];
                    }
                }
            }
        }
        out
    }
}

/// The conformed grid's voxel→world affine (niimath's `conformVox2Vox` with `toRAS = false`):
/// 1 mm voxels along LIA (+i left, +j inferior, +k anterior), with the cube's centre voxel
/// `(128, 128, 128)` on the world position of the input's voxel `dims / 2`.
fn conform_affine(dims: (usize, usize, usize), affine: &[f64; 16]) -> [f64; 16] {
    let half = [dims.0 as f64 / 2.0, dims.1 as f64 / 2.0, dims.2 as f64 / 2.0];
    let centre: [f64; 3] = std::array::from_fn(|r| {
        affine[4 * r] * half[0] + affine[4 * r + 1] * half[1] + affine[4 * r + 2] * half[2] + affine[4 * r + 3]
    });
    let c = CONFORM_DIM as f64 / 2.0;
    // rotation rows: x = -i, y = k, z = -j
    #[rustfmt::skip]
    let out = [
        -1.0, 0.0, 0.0, centre[0] + c,
         0.0, 0.0, 1.0, centre[1] - c,
         0.0, -1.0, 0.0, centre[2] + c,
         0.0, 0.0, 0.0, 1.0,
    ];
    out
}

/// niimath's `voxelIntensityScale` (FastSurfer `getscale` + `scalecrop`) to `[0, 255]`, with
/// `f_low = 0` and `f_high = 0.98`.
///
/// The robust maximum is read from a 1000-bin histogram: the first bin whose cumulative count
/// reaches all voxels but 2 % of the *non-zero* ones. The minimum is the image minimum (or 0, if
/// fewer than 2 % of voxels are negative). niimath works in `f32`, and so does this.
fn scale_to_u8_range(image: &[f64]) -> Vec<f32> {
    let img: Vec<f32> = image.iter().map(|&v| v as f32).collect();
    let n = img.len();
    let (mut src_min, src_max) = img.iter().fold((img[0], img[0]), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let nz = img.iter().filter(|v| v.abs() >= 1e-15).count();
    let nltz = img.iter().filter(|&&v| v < 0.0).count();
    if src_min < 0.0 && 100.0 * (nltz as f64 / n as f64) < 2.0 {
        src_min = 0.0;
    }

    const BINS: usize = 1000;
    let bin_size = (src_max - src_min) / BINS as f32;
    let mut cs = vec![0usize; BINS];
    for &v in &img {
        // C's (int) cast truncates toward zero; then clamp into range
        let b = ((v - src_min) / bin_size) as i64;
        cs[b.clamp(0, BINS as i64 - 1) as usize] += 1;
    }
    for i in 1..BINS {
        cs[i] += cs[i - 1];
    }
    // f_low = 0: the lower limit is the first bin with a cumulative count >= 0, i.e. bin 0
    let nth = n - ((1.0 - CONFORM_F_HIGH as f32) * nz as f32) as usize;
    let mut idx = 0;
    while idx < BINS - 1 && cs[idx + 1] < nth {
        idx += 1;
    }
    let robust_max = idx as f32 * bin_size + src_min;
    let scale = if src_min != robust_max { 255.0 / (robust_max - src_min) } else { 1.0 };
    img.iter().map(|&v| (scale * (v - src_min)).clamp(0.0, 255.0)).collect()
}

/// niimath's linear `doReslice` onto the 256³ grid, rounded to `u8` (`-odt char`).
///
/// A conformed voxel whose 2×2×2 input neighbourhood is not wholly inside the input keeps the
/// image minimum, as niimath fills it.
fn reslice_linear(img: &[f32], (nx, ny, nz): (usize, usize, usize), m: &[[f64; 4]; 3]) -> Vec<u8> {
    let n = CONFORM_DIM;
    let fill = img.iter().copied().fold(f32::INFINITY, f32::min);
    let mut out = vec![fill; n * n * n];
    let (sx, sy) = (1, nx);
    let sz = nx * ny;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let [ix, iy, iz] = apply(m, x as f64, y as f64, z as f64);
                let (fx, fy, fz) = (ix.floor(), iy.floor(), iz.floor());
                if fx < 0.0 || fy < 0.0 || fz < 0.0 {
                    continue;
                }
                let (fx, fy, fz) = (fx as usize, fy as usize, fz as usize);
                if fx + 1 >= nx || fy + 1 >= ny || fz + 1 >= nz {
                    continue;
                }
                let (dx, dy, dz) = ((ix - fx as f64) as f32, (iy - fy as f64) as f32, (iz - fz as f64) as f32);
                let i000 = fx * sx + fy * sy + fz * sz;
                let at = |o: usize| img[i000 + o];
                let v = at(0) * (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
                    + at(sz) * (1.0 - dx) * (1.0 - dy) * dz
                    + at(sy) * (1.0 - dx) * dy * (1.0 - dz)
                    + at(sy + sz) * (1.0 - dx) * dy * dz
                    + at(sx) * dx * (1.0 - dy) * (1.0 - dz)
                    + at(sx + sz) * dx * (1.0 - dy) * dz
                    + at(sx + sy) * dx * dy * (1.0 - dz)
                    + at(sx + sy + sz) * dx * dy * dz;
                out[x + n * (y + n * z)] = v;
            }
        }
    }
    // C round(): half away from zero; values are already in [0, 255]
    out.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect()
}

/// brainchop's `qnormalize`: map the 2nd..98th percentile (numpy's linear interpolation) to
/// `[0, 1]` and clip. Returns the network input in C order `(x, y, z)` — brainchop feeds the
/// conformed volume to the network as `(x, y, z)`, the transpose of its column-major layout.
fn normalize(data: &[u8]) -> Vec<f32> {
    let mut hist = [0usize; 256];
    data.iter().for_each(|&v| hist[v as usize] += 1);
    // value at a sorted position, from the histogram
    let at = |pos: usize| {
        let mut cum = 0;
        for (v, &c) in hist.iter().enumerate() {
            cum += c;
            if cum > pos {
                return v as f64;
            }
        }
        255.0
    };
    let quantile = |q: f64| {
        let p = q * (data.len() - 1) as f64;
        let (lo, frac) = (p.floor() as usize, p - p.floor());
        let a = at(lo);
        if frac == 0.0 { a } else { a + (at(lo + 1) - a) * frac }
    };
    let (qlow, qhigh) = (quantile(0.02), quantile(0.98));
    let denom = qhigh - qlow + 1e-3;
    let lut: Vec<f32> = (0..256).map(|v| ((v as f64 - qlow) / denom).clamp(0.0, 1.0) as f32).collect();

    let n = CONFORM_DIM;
    let mut out = vec![0.0f32; n * n * n];
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                out[(x * n + y) * n + z] = lut[data[x + n * (y + n * z)] as usize];
            }
        }
    }
    out
}

/// Two-class logits `(1, 2, x, y, z)` (C order) → column-major mask: class 1 where its logit
/// beats class 0 (ties go to class 0, as an argmax does).
fn argmax_mask(logits: &[f32]) -> Vec<u8> {
    let n = CONFORM_DIM;
    let n3 = n * n * n;
    assert_eq!(logits.len(), 2 * n3, "MindGrab must output two 256³ channels");
    let (bg, fg) = logits.split_at(n3);
    let mut mask = vec![0u8; n3];
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                let i = (x * n + y) * n + z;
                mask[x + n * (y + n * z)] = (fg[i] > bg[i]) as u8;
            }
        }
    }
    mask
}

/// Reverse the first spatial axis of `channels` C-order `(x, y, z)` 256³ volumes (the
/// sagittal flip of brainchop's TTA).
fn flip_axis0(data: &[f32], channels: usize) -> Vec<f32> {
    let n = CONFORM_DIM;
    let plane = n * n;
    let mut out = vec![0.0f32; data.len()];
    for c in 0..channels {
        let base = c * n * plane;
        for x in 0..n {
            let (src, dst) = (base + x * plane, base + (n - 1 - x) * plane);
            out[dst..dst + plane].copy_from_slice(&data[src..src + plane]);
        }
    }
    out
}

// ---- 4×4 affine helpers (row-major) ----

fn compose(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    std::array::from_fn(|i| {
        let (r, c) = (i / 4, i % 4);
        (0..4).map(|k| a[4 * r + k] * b[4 * k + c]).sum()
    })
}

fn invert(a: &[f64; 16]) -> Option<[f64; 16]> {
    let inv = crate::geometry::world_to_voxel(a)?;
    let mut out = [0.0; 16];
    for r in 0..3 {
        out[4 * r..4 * r + 4].copy_from_slice(&inv[r]);
    }
    out[15] = 1.0;
    Some(out)
}

fn top3(a: &[f64; 16]) -> [[f64; 4]; 3] {
    std::array::from_fn(|r| std::array::from_fn(|c| a[4 * r + c]))
}

fn apply(m: &[[f64; 4]; 3], x: f64, y: f64, z: f64) -> [f64; 3] {
    std::array::from_fn(|r| m[r][0] * x + m[r][1] * y + m[r][2] * z + m[r][3])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    const RAS_1MM: [f64; 16] = [
        1.0, 0.0, 0.0, -10.0,
        0.0, 1.0, 0.0, -20.0,
        0.0, 0.0, 1.0, -30.0,
        0.0, 0.0, 0.0, 1.0,
    ];

    #[test]
    fn conform_affine_is_lia_centred_on_the_input() {
        let a = conform_affine((40, 60, 80), &RAS_1MM);
        // input voxel dims/2 = (20, 30, 40) sits at world (10, 10, 10)
        let centre = [a[0] * 128.0 + a[1] * 128.0 + a[2] * 128.0 + a[3],
                      a[4] * 128.0 + a[5] * 128.0 + a[6] * 128.0 + a[7],
                      a[8] * 128.0 + a[9] * 128.0 + a[10] * 128.0 + a[11]];
        assert_eq!(centre, [10.0, 10.0, 10.0]);
        // +i → left (-x), +j → inferior (-z), +k → anterior (+y)
        assert_eq!([a[0], a[4], a[8]], [-1.0, 0.0, 0.0]);
        assert_eq!([a[1], a[5], a[9]], [0.0, 0.0, -1.0]);
        assert_eq!([a[2], a[6], a[10]], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn mask_round_trips_through_the_conformed_grid() {
        // A 1 mm RAS box resliced into the conformed cube and back lands on the same voxels.
        let dims = (40, 60, 80);
        let n = dims.0 * dims.1 * dims.2;
        let mut img = vec![0.0; n];
        for z in 20..60 {
            for y in 15..45 {
                for x in 10..30 {
                    img[x + dims.0 * (y + dims.1 * z)] = 100.0;
                }
            }
        }
        let c = Conformed::new(&img, dims, &RAS_1MM).unwrap();
        let mask: Vec<u8> = c.data.iter().map(|&v| (v > 127) as u8).collect();
        let back = c.reslice_back(&mask);
        let want: Vec<u8> = img.iter().map(|&v| (v > 50.0) as u8).collect();
        assert_eq!(back, want);
    }

    #[test]
    fn intensity_scaling_maps_robust_range_to_u8() {
        // 98 % of the voxels at 1..=98, a 2 % bright tail: the tail saturates at 255.
        let img: Vec<f64> = (0..10_000).map(|i| if i < 9_800 { (i % 98 + 1) as f64 } else { 1000.0 }).collect();
        let s = scale_to_u8_range(&img);
        assert_eq!(s.iter().cloned().fold(f32::MIN, f32::max), 255.0);
        assert!(s[..9_800].iter().all(|&v| v < 255.0));
        assert_eq!(s.iter().cloned().fold(f32::MAX, f32::min), 0.0);
    }

    #[test]
    fn normalize_matches_numpy_quantiles() {
        // 0..=255 repeated evenly: np.quantile at 0.02 / 0.98 by linear interpolation.
        let n = CONFORM_DIM;
        let data: Vec<u8> = (0..n * n * n).map(|i| (i % 256) as u8).collect();
        let out = normalize(&data);
        // sorted position q*(N-1): value = floor(pos / (N/256)) with linear interpolation
        let total = (n * n * n) as f64;
        let per = total / 256.0;
        let q = |q: f64| {
            let p = q * (total - 1.0);
            let lo = (p.floor() / per).floor();
            let hi = ((p.floor() + 1.0) / per).floor();
            lo + (hi - lo) * (p - p.floor())
        };
        let (ql, qh) = (q(0.02), q(0.98));
        let want = |v: f64| ((v - ql) / (qh - ql + 1e-3)).clamp(0.0, 1.0) as f32;
        // voxel x=200, y=0, z=0 has value 200, lands at C-order index 200 * n * n
        assert_eq!(out[200 * n * n], want(200.0));
        assert_eq!(out[0], 0.0);
        assert_eq!(out[255 * n * n], 1.0);
    }

    #[test]
    fn flip_is_an_involution() {
        let n = CONFORM_DIM;
        let v: Vec<f32> = (0..n * n * n).map(|i| (i % 977) as f32).collect();
        assert_eq!(flip_axis0(&flip_axis0(&v, 1), 1), v);
        assert_eq!(flip_axis0(&v, 1)[0], v[(n - 1) * n * n]);
    }

    // ---- parity against brainchop-cli's own pipeline: see tests/models_onnx.rs (mindgrab_*) ----

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
        let data = b[off + hlen..].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        (shape, data)
    }

    /// Fixture directory plus its `mag.nii`.
    type Ref = (String, crate::io::NiftiData);

    /// `MINDGRAB_REF_DIR` holds the fixtures `scripts/onnx-export/ref_mindgrab.py` writes.
    fn load_ref() -> Option<Ref> {
        let dir = std::env::var("MINDGRAB_REF_DIR").ok()?;
        let nii = crate::io::read_nifti_file(std::path::Path::new(&format!("{dir}/mag.nii"))).unwrap();
        Some((dir, nii))
    }

    fn read_conformed(dir: &str, name: &str) -> Vec<u8> {
        let nii = crate::io::read_nifti_file(std::path::Path::new(&format!("{dir}/{name}"))).unwrap();
        assert_eq!(nii.dims, (CONFORM_DIM, CONFORM_DIM, CONFORM_DIM));
        nii.data.iter().map(|&v| v as u8).collect()
    }

    /// Conform vs niimath `-conform`.
    #[test]
    #[ignore]
    fn conform_matches_niimath() {
        let Some((dir, mag)) = load_ref() else { return eprintln!("MINDGRAB_REF_DIR not set; skipping") };
        let c = Conformed::new(&mag.data, mag.dims, &mag.affine).unwrap();
        let want = read_conformed(&dir, "conformed.nii");
        let diff: Vec<i32> = c.data.iter().zip(&want).map(|(&a, &b)| a as i32 - b as i32).collect();
        let n_diff = diff.iter().filter(|&&d| d != 0).count();
        let max = diff.iter().map(|d| d.abs()).max().unwrap();
        eprintln!("{n_diff} conformed voxels differ, max |d| {max}");
        // niimath composes the affines in f32; allow off-by-one rounding at a few voxels
        assert!(max <= 1 && n_diff < 1000, "{n_diff} voxels differ, max |d| = {max}");
    }

    /// Normalisation of niimath's conformed volume vs brainchop's `qnormalize`.
    #[test]
    #[ignore]
    fn normalization_matches_brainchop() {
        let Some((dir, ..)) = load_ref() else { return eprintln!("MINDGRAB_REF_DIR not set; skipping") };
        let got = normalize(&read_conformed(&dir, "conformed.nii"));
        let (shape, want) = read_npy_f32(&format!("{dir}/ref_input.npy"));
        assert_eq!(shape, vec![1, 1, CONFORM_DIM, CONFORM_DIM, CONFORM_DIM]);
        let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(err < 1e-6, "max |d| = {err}");
    }

    /// Argmax + largest component + reslice back, applied to the reference logits, reproduces
    /// brainchop's mask.
    #[test]
    #[ignore]
    #[cfg(feature = "onnx")]
    fn postprocessing_matches_brainchop() {
        let Some((dir, mag)) = load_ref() else { return eprintln!("MINDGRAB_REF_DIR not set; skipping") };
        let (_, logits) = read_npy_f32(&format!("{dir}/ref_logits.npy"));
        let n = CONFORM_DIM;
        let mask = largest_component(&argmax_mask(&logits), (n, n, n), Connectivity::Full);
        let want_c = read_conformed(&dir, "ref_mask_conformed.nii");
        let diff_c = mask.iter().zip(&want_c).filter(|(&m, &w)| (m != 0) != (w != 0)).count();
        eprintln!("{diff_c} conformed voxels differ from brainchop's (tinygrad) mask");

        // the reslice itself, on brainchop's own conformed mask
        let c = Conformed::new(&mag.data, mag.dims, &mag.affine).unwrap();
        let back = c.reslice_back(&want_c.iter().map(|&w| (w != 0) as u8).collect::<Vec<_>>());
        let want = crate::io::read_nifti_file(std::path::Path::new(&format!("{dir}/ref_mask.nii.gz"))).unwrap();
        let diff = back.iter().zip(&want.data).filter(|(&m, &w)| (m != 0) != (w > 0.5)).count();
        eprintln!("{diff} native voxels differ from niimath -reslice_nn");
        assert_eq!(diff, 0, "reslice: {diff} differing voxels");
        let brain = want_c.iter().filter(|&&w| w != 0).count();
        assert!((diff_c as f64) < 1e-3 * brain as f64, "{diff_c} of {brain} conformed voxels differ");
    }
}

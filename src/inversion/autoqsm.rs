//! AutoQSM single-step deep-learning reconstruction (`onnx` feature).
//!
//! AutoQSM (Wei 2019) is a patch V-Net that maps a **total** field (ppm) directly
//! to susceptibility (ppm) — no brain extraction, no separate background removal.
//! The network takes a fixed 64³ input patch and returns the central 32³. Whole
//! volumes are reconstructed by overlap-tiled sliding-window inference: 32³ output
//! patches at stride 24 (8-voxel overlap), each from a 64³ input patch with a
//! 16-voxel context margin, edge-padded and linearly blended across the overlap.
//! This mirrors the authors' `util.data_predict` / `patch_process`.
//!
//! Upstream weights are Keras; QSM.rs ships a clean PyTorch re-export
//! (`scripts/onnx-export/export_autoqsm.py`) so `tract` runs it. The network is
//! all Conv3D+ReLU with no normalization, so no input scaling is applied.
//!
//! **Field sign.** AutoQSM was trained on fields of the opposite sign to this
//! crate's (`field = D * chi` with `D = 1/3 - kz^2/k^2`, so a paramagnetic source
//! raises the field along B0). On the authors' own `test_data/0.mat` input the
//! network returns bright deep grey matter, while the same field read in this
//! crate's convention (SHARP + TKD) anticorrelates with that output (r = -0.42);
//! fed this crate's field unchanged it returns `-chi`-like maps (r = -0.50 on the
//! test phantom). The field is therefore negated on the way in: the network sees
//! its own training convention and its output is already susceptibility in ppm
//! with the usual sign. The network is non-linear, so negating the output instead
//! would not be equivalent.
//!
//! Weights are not bundled; the caller passes the exported `autoqsm.onnx` bytes.

use crate::grid::Grid;
use crate::models::onnx::{OnnxModel, OnnxError, Tensor};

const IN: usize = 64;
const OUT: usize = 32;
const MARGIN: usize = (IN - OUT) / 2; // 16
const SHIFT: usize = 24;
const OVERLAP: usize = OUT - SHIFT; // 8

#[inline]
fn ri(x: usize, y: usize, z: usize, dy: usize, dz: usize) -> usize {
    (x * dy + y) * dz + z // row-major (X,Y,Z), matches numpy C-order
}

/// The output-grid extent for an axis of `d` voxels: the smallest whole number of output
/// patches (`OUT`, then one more per `SHIFT`) that covers it. An axis shorter than one patch gets
/// one patch; the samples past `d` are edge-padded and cropped away.
fn output_extent(d: usize) -> usize {
    OUT + d.saturating_sub(OUT).div_ceil(SHIFT) * SHIFT
}

/// Run AutoQSM on a total field map.
///
/// * `field_ppm` — total field (ppm) in this crate's sign convention,
///   column-major `(nx,ny,nz)`. It is negated internally to match the network's
///   training convention (see the module docs).
/// * `mask` — binary brain mask; applied to the output (AutoQSM has no brain
///   extraction, so the raw output is whole-head).
/// * `model_onnx` — bytes of the exported `autoqsm.onnx` (fixed 64³→32³).
///
/// Returns susceptibility (ppm), masked, in the same layout.
pub fn autoqsm(
    field_ppm: &[f64],
    mask: &[u8],
    grid: &Grid,
    model_onnx: &[u8],
) -> Result<Vec<f64>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    assert_eq!(field_ppm.len(), n, "field length must match grid");
    assert_eq!(mask.len(), n, "mask length must match grid");

    // Column-major (crate) -> row-major (X,Y,Z) working volume, negated into the
    // network's training field convention (module docs).
    let mut vol = vec![0.0f64; n];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                vol[ri(x, y, z, ny, nz)] = -field_ppm[x + nx * (y + ny * z)];
            }
        }
    }

    let (xp, yp, zp) = (output_extent(nx), output_extent(ny), output_extent(nz)); // output-grid size

    // Edge-pad: MARGIN before, (pad + MARGIN) after, per axis (replicate).
    let (bx, by, bz) = (MARGIN, MARGIN, MARGIN);
    let (dx, dy, dz) = (xp + 2 * MARGIN, yp + 2 * MARGIN, zp + 2 * MARGIN);
    let clamp = |v: isize, lo: usize, hi: usize| (v.clamp(lo as isize, hi as isize - 1)) as usize;
    let mut padded = vec![0.0f64; dx * dy * dz];
    for x in 0..dx {
        let sx = clamp(x as isize - bx as isize, 0, nx);
        for y in 0..dy {
            let sy = clamp(y as isize - by as isize, 0, ny);
            for z in 0..dz {
                let sz = clamp(z as isize - bz as isize, 0, nz);
                padded[ri(x, y, z, dy, dz)] = vol[ri(sx, sy, sz, ny, nz)];
            }
        }
    }

    let num_i = (xp - OUT) / SHIFT + 1;
    let num_j = (yp - OUT) / SHIFT + 1;
    let num_k = (zp - OUT) / SHIFT + 1;

    let model = OnnxModel::load(model_onnx)?;
    let mut output = vec![0.0f64; xp * yp * zp];
    let mut patches: Vec<Vec<f64>> = Vec::with_capacity(num_i * num_j * num_k);

    for k in 0..num_k {
        for j in 0..num_j {
            for i in 0..num_i {
                // Extract the 64³ input patch (already row-major = NCDHW buffer).
                let mut buf = vec![0.0f32; IN * IN * IN];
                for px in 0..IN {
                    for py in 0..IN {
                        for pz in 0..IN {
                            let src = ri(SHIFT * i + px, SHIFT * j + py, SHIFT * k + pz, dy, dz);
                            buf[ri(px, py, pz, IN, IN)] = padded[src] as f32;
                        }
                    }
                }
                let out = model.run_single(&Tensor::new(vec![1, 1, IN, IN, IN], buf))?;
                let mut patch: Vec<f64> = out.data.iter().map(|&v| v as f64).collect();

                // Linear-ramp blend against already-placed neighbors.
                if i != 0 {
                    blend(&mut patch, &patches[patches.len() - 1], 0);
                }
                if j != 0 {
                    blend(&mut patch, &patches[patches.len() - num_i], 1);
                }
                if k != 0 {
                    blend(&mut patch, &patches[patches.len() - num_i * num_j], 2);
                }

                for ox in 0..OUT {
                    for oy in 0..OUT {
                        for oz in 0..OUT {
                            output[ri(SHIFT * i + ox, SHIFT * j + oy, SHIFT * k + oz, yp, zp)] =
                                patch[ri(ox, oy, oz, OUT, OUT)];
                        }
                    }
                }
                patches.push(patch);
            }
        }
    }

    // Crop to the input grid, mask, and repack row-major -> column-major.
    let mut chi = vec![0.0f64; n];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = x + nx * (y + ny * z);
                if mask[i] != 0 {
                    chi[i] = output[ri(x, y, z, yp, zp)];
                }
            }
        }
    }
    Ok(chi)
}

/// Blend the leading `OVERLAP` slabs of `patch` with the trailing slabs of a
/// previously-placed `neighbor` along `dir` (0=x,1=y,2=z), ramping neighbor→patch
/// (matches `util.patch_process`).
fn blend(patch: &mut [f64], neighbor: &[f64], dir: usize) {
    let denom = (OVERLAP - 1) as f64;
    for t in 0..OVERLAP {
        let w = 1.0 - t as f64 / denom; // weight on the neighbor
        for a in 0..OUT {
            for b in 0..OUT {
                let (pi, ni) = match dir {
                    0 => (ri(t, a, b, OUT, OUT), ri(OUT - OVERLAP + t, a, b, OUT, OUT)),
                    1 => (ri(a, t, b, OUT, OUT), ri(a, OUT - OVERLAP + t, b, OUT, OUT)),
                    _ => (ri(a, b, t, OUT, OUT), ri(a, b, OUT - OVERLAP + t, OUT, OUT)),
                };
                patch[pi] = w * neighbor[ni] + (1.0 - w) * patch[pi];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An axis shorter than one output patch used to underflow `d - OUT` (a panic in debug
    /// builds, a wrapped size and a "capacity overflow" in release ones).
    #[test]
    fn output_extent_covers_every_axis_in_whole_patches() {
        for (d, want) in [(1, 32), (16, 32), (32, 32), (33, 56), (56, 56), (57, 80), (184, 200)] {
            assert_eq!(output_extent(d), want, "d = {d}");
        }
        for d in 1..300 {
            let e = output_extent(d);
            assert!(e >= d.max(OUT) && (e - OUT).is_multiple_of(SHIFT), "d = {d}: {e}");
            assert!(e < d.max(OUT) + SHIFT, "d = {d}: {e} is a patch too many");
        }
    }
}

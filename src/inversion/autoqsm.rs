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

/// Run AutoQSM on a total field map.
///
/// * `field_ppm` — total field (ppm), column-major `(nx,ny,nz)`.
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

    // Column-major (crate) -> row-major (X,Y,Z) working volume.
    let mut vol = vec![0.0f64; n];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                vol[ri(x, y, z, ny, nz)] = field_ppm[x + nx * (y + ny * z)];
            }
        }
    }

    // Intermediate padded size so (dim - OUT) is a multiple of SHIFT.
    let pad_to = |d: usize| -> usize {
        let e = d - OUT;
        e.div_ceil(SHIFT) * SHIFT - e // extra samples appended
    };
    let (padx, pady, padz) = (pad_to(nx), pad_to(ny), pad_to(nz));
    let (xp, yp, zp) = (nx + padx, ny + pady, nz + padz); // output-grid size

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

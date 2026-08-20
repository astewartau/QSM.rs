//! QSMGAN dipole inversion (`onnx` feature).
//!
//! QSMGAN (Chen 2020) inverts the dipole with a 3D U-Net **generator** refined by a
//! Wasserstein GAN. Inference is patch-based with an increased receptive field: a
//! 64³ local-field patch maps to a 48³ susceptibility patch (a centre crop baked
//! into the exported graph — the "i64o48" scheme). The network itself is the only
//! part in ONNX; this glue reproduces the authors' `recon.py` pipeline:
//!
//! * sign flip — the QSM-CI runner negates the local field before the net;
//! * tile by non-overlapping 48³ **output** patches, each fed a 64³ **input** patch
//!   (same centre, 8-voxel context margin, zero-padded at the volume edges);
//! * `input_scale = 100`; output head `tanh`, so `χ = atanh(clip(out)) / 10`;
//! * mask the result.
//!
//! The input is the SEPIA-style **local field in ppm** (already background-removed).
//! Weights are not bundled; the caller passes the exported `qsmgan.onnx` bytes.

use crate::grid::Grid;
use crate::models::onnx::{OnnxError, OnnxModel, Tensor};

const IPS: usize = 64; // input patch size
const OPS: usize = 48; // output patch size
const INPUT_SCALE: f64 = 100.0;
const OUTPUT_SCALE: f64 = 10.0;

/// Run QSMGAN on a background-removed local field (ppm), column-major `(nx,ny,nz)`.
/// Returns susceptibility (ppm), masked, in the same layout.
pub fn qsmgan(
    local_field_ppm: &[f64],
    mask: &[u8],
    grid: &Grid,
    model_onnx: &[u8],
) -> Result<Vec<f64>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    assert_eq!(local_field_ppm.len(), n, "field length must match grid");
    assert_eq!(mask.len(), n, "mask length must match grid");

    let model = OnnxModel::load(model_onnx)?;
    let (half_i, half_o) = (IPS as i64 / 2, OPS as i64 / 2); // 32, 24
    let (nxi, nyi, nzi) = (nx as i64, ny as i64, nz as i64);
    // Negated local field (the runner's sign flip), zero outside the volume.
    let field = |x: i64, y: i64, z: i64| -> f32 {
        if x >= 0 && x < nxi && y >= 0 && y < nyi && z >= 0 && z < nzi {
            (-INPUT_SCALE * local_field_ppm[x as usize + nx * (y as usize + ny * z as usize)]) as f32
        } else {
            0.0
        }
    };

    let mut predict = vec![0.0f64; n];
    // Output-patch centres: range(half_o, dim + half_o + 1, OPS) per axis.
    let mut cx = half_o;
    while cx <= nxi + half_o {
        let mut cy = half_o;
        while cy <= nyi + half_o {
            let mut cz = half_o;
            while cz <= nzi + half_o {
                // In-volume extent of this 48³ output patch.
                let (x0, y0, z0) = (cx - half_o, cy - half_o, cz - half_o); // ≥ 0
                let xe = (x0 + OPS as i64).min(nxi);
                let ye = (y0 + OPS as i64).min(nyi);
                let ze = (z0 + OPS as i64).min(nzi);
                let (px, py, pz) = (xe - x0, ye - y0, ze - z0);
                if px <= 0 || py <= 0 || pz <= 0 {
                    cz += OPS as i64;
                    continue;
                }

                // 64³ input patch centred at (cx,cy,cz), row-major NCDHW.
                let mut inp = vec![0.0f32; IPS * IPS * IPS];
                for i in 0..IPS {
                    for j in 0..IPS {
                        for k in 0..IPS {
                            let v = field(
                                cx - half_i + i as i64,
                                cy - half_i + j as i64,
                                cz - half_i + k as i64,
                            );
                            inp[(i * IPS + j) * IPS + k] = v;
                        }
                    }
                }
                let out = model.run_single(&Tensor::new(vec![1, 1, IPS, IPS, IPS], inp))?;

                // Place the in-bounds part of the 48³ output patch: χ = atanh(clip)/10.
                for oi in 0..px as usize {
                    for oj in 0..py as usize {
                        for ok in 0..pz as usize {
                            let o = (out.data[(oi * OPS + oj) * OPS + ok] as f64)
                                .clamp(-0.999999, 0.999999);
                            let dst = (x0 as usize + oi)
                                + nx * ((y0 as usize + oj) + ny * (z0 as usize + ok));
                            predict[dst] = o.atanh() / OUTPUT_SCALE;
                        }
                    }
                }
                cz += OPS as i64;
            }
            cy += OPS as i64;
        }
        cx += OPS as i64;
    }

    for i in 0..n {
        if mask[i] == 0 {
            predict[i] = 0.0;
        }
    }
    Ok(predict)
}

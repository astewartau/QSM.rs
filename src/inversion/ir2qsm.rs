//! IR2QSM dipole inversion (`onnx` feature).
//!
//! IR2QSM (Li et al., Med. Phys. 2025; arXiv:2406.12300) maps the **local (tissue)
//! field in ppm** directly to **susceptibility in ppm** with a single "IR2U-net": a
//! 3D U-net (`depth=4`) run for `iterations=4` unrolled passes with reverse
//! concatenations and a recurrent SRU middle module, then a learned integration of
//! the four per-iteration residual estimates (`latest_out`, the network's final
//! product). Unlike LPCNN there is no separate physics/unroll here — the **entire**
//! network is the exported ONNX graph, and this glue only does the surrounding I/O
//! pipeline:
//!
//! 1. **No normalization.** IR2QSM consumes the ppm local field directly and emits
//!    ppm susceptibility — no dataset mean/std, no `norm_factor` (there are no baked
//!    scalar constants). Matches `IR2QSM/Evaluate/test_util.py`.
//! 2. **Zero-pad to a multiple of 8.** The U-net has 3 pool/deconv levels, so each
//!    spatial dim must be divisible by `2³ = 8`; we center-pad exactly as the
//!    reference `zero_padding(image, 8)` (low offset `= ceil((target − shape)/2)`),
//!    run the net, then crop back.
//! 3. **Mask.** The result is multiplied by the supplied brain mask on the original
//!    grid.
//!
//! **Determinism.** `IR2Unet.forward` has an ungated inference-time `AddNoise` in the
//! decoder (`torch.rand(1) > 0.3` per iteration) that makes plain PyTorch inference
//! mildly stochastic. The ONNX was exported with that call pinned to its noise-free
//! branch (identity), so this net is deterministic. See `export_ir2qsm.py`.
//!
//! Input is the background-removed local field in ppm (single orientation; the net is
//! orientation-agnostic — no dipole kernel, no `b_vec`). Weights are not bundled; the
//! caller passes `ir2qsm.onnx`.

use crate::grid::Grid;
use crate::models::onnx::{OnnxError, OnnxModel, Tensor};

/// U-net pool/deconv depth requirement: spatial dims are padded to a multiple of this.
const SIZE_DIVISOR: usize = 8;

/// Run IR2QSM on a background-removed local field (ppm), column-major `(nx,ny,nz)`.
/// Returns χ (ppm), masked, on the original grid.
pub fn ir2qsm(
    local_field_ppm: &[f64],
    mask: &[u8],
    grid: &Grid,
    onnx: &[u8],
) -> Result<Vec<f64>, OnnxError> {
    let (nx, ny, nz) = grid.dims;
    let n = nx * ny * nz;
    assert_eq!(local_field_ppm.len(), n, "field length must match grid");
    assert_eq!(mask.len(), n, "mask length must match grid");

    let model = OnnxModel::load(onnx)?;

    // Center zero-pad each dim up to a multiple of 8 (low offset = ceil(pad/2)).
    let pad = |d: usize| -> (usize, usize) {
        let target = d.div_ceil(SIZE_DIVISOR) * SIZE_DIVISOR;
        let total = target - d;
        let lo = total.div_ceil(2); // ceil((target - d)/2), matching the reference
        (lo, total - lo)
    };
    let (px0, _px1) = pad(nx);
    let (py0, _py1) = pad(ny);
    let (pz0, _pz1) = pad(nz);
    let (pnx, pny, pnz) =
        (nx.div_ceil(SIZE_DIVISOR) * SIZE_DIVISOR,
         ny.div_ceil(SIZE_DIVISOR) * SIZE_DIVISOR,
         nz.div_ceil(SIZE_DIVISOR) * SIZE_DIVISOR);

    // Build the padded NCDHW (row-major) input from the column-major volume.
    let mut input = vec![0.0f32; pnx * pny * pnz];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let src = x + nx * (y + ny * z);
                let (px, py, pz) = (x + px0, y + py0, z + pz0);
                input[((px * pny) + py) * pnz + pz] = local_field_ppm[src] as f32;
            }
        }
    }

    let out = model.run_single(&Tensor::new(vec![1, 1, pnx, pny, pnz], input))?;

    // Crop back to the original grid, repack to column-major, and mask.
    let mut chi = vec![0.0f64; n];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let dst = x + nx * (y + ny * z);
                let (px, py, pz) = (x + px0, y + py0, z + pz0);
                let v = out.data[((px * pny) + py) * pnz + pz] as f64;
                chi[dst] = if mask[dst] != 0 { v } else { 0.0 };
            }
        }
    }
    Ok(chi)
}

/// Memory-bounded IR2QSM via overlap-tiling — the IR2U-net run patch-by-patch (for 32-bit WASM,
/// where whole-volume [`ir2qsm`] overflows the heap on clinical data). No normalization; the
/// net's 3 pool levels require a `/8` patch. Approximates whole-volume up to tile-boundary
/// error; see [`crate::inversion::tiled`].
pub fn ir2qsm_tiled(
    local_field_ppm: &[f64],
    mask: &[u8],
    grid: &Grid,
    onnx: &[u8],
    cfg: &super::tiled::TileConfig,
    progress: impl FnMut(usize, usize),
) -> Result<Vec<f64>, OnnxError> {
    let model = OnnxModel::load(onnx)?;
    super::tiled::tiled_field_inversion(
        local_field_ppm, mask, grid, &model, SIZE_DIVISOR, cfg, |v| v as f32, |o| o as f64, progress,
    )
}

//! The input tensors each registry network file takes at a spatial shape, as its wrapper builds
//! them, filled with values that look like data (the memory a run takes does not depend on them).
//!
//! Shared by `examples/inference_memory.rs` (native measurements) and the `browser_memory` cases
//! in `tests/wasm_pipeline.rs` (the same inferences in WASM), so both run the same graphs on the
//! same tensors.

use qsm_core::models::onnx::Tensor;

fn wave(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| scale * ((i as f32) * 0.37).sin()).collect()
}

/// Inputs of network file `file` (a [`qsm_core::models::WeightFile::name`]) at spatial shape `s`,
/// in graph order.
pub fn net_inputs(file: &str, s: [usize; 3]) -> Vec<Tensor> {
    let n = s[0] * s[1] * s[2];
    let vol = |c: usize| Tensor::new(vec![1, c, s[0], s[1], s[2]], wave(c * n, 0.05));
    let ones = || Tensor::new(vec![1, 1, s[0], s[1], s[2]], vec![1.0; n]);
    let scalar = |v: f32| Tensor::new(vec![1], vec![v]);
    match file {
        // Channels-last.
        "synthseg.onnx" | "synthseg-2.0.onnx" => vec![Tensor::new(vec![1, s[0], s[1], s[2], 1], wave(n, 1.0))],
        "modl-qsm.onnx" => vec![vol(2)],
        "chi-sepnet.onnx" => vec![vol(3)],
        "susep-net.onnx" => vec![vol(1), vol(1), vol(1)],
        // phase, mask, TE, B0, border
        "iqsm.onnx" | "iqfm.onnx" => vec![vol(1), ones(), scalar(0.02), scalar(3.0), ones()],
        // phase, mask, TE, B0, B0 direction, border
        "iqsm-plus.onnx" => vec![
            vol(1),
            ones(),
            scalar(0.02),
            scalar(3.0),
            Tensor::new(vec![1, 1, 3], vec![0.0, 0.0, 1.0]),
            ones(),
        ],
        _ => vec![vol(1)],
    }
}

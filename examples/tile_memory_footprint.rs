//! Peak live activation bytes one ONNX patch costs, for a given model and patch size.
//!
//! This is where `inversion::tiled::ACTIVATION_BYTES_PER_PATCH_VOXEL` comes from, and how to
//! re-derive it after a `tract` upgrade or for a new tiled net. It walks the **optimized** plan in
//! tract's own evaluation order and tracks which outlets are still live at each step, so the
//! figure is the real simultaneous peak rather than the sum of every intermediate (for xQSM at
//! 64³ those differ by 14×: 270 MB live against 3.7 GB total).
//!
//! That per-patch figure is what bounds how many tiles a 32-bit WASM host can run at once — see
//! [`qsm_core::inversion::tile_concurrency`].
//!
//! Run:
//!   cargo run --release --example tile_memory_footprint --features onnx -- <model.onnx> [patch]
//!
//! Measured at a 64³ patch (tract 0.23.7): xQSM 269.5 MB, QSMnet 147.6 MB, NeXtQSM 134.9 MB
//! (VJP net), LPCNN 103.9 MB, IR2QSM 93.6 MB. It is linear in patch voxels — xQSM is 1028
//! B/voxel at both 64³ and 144³.

use std::collections::HashMap;
use tract_onnx::prelude::*;

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .expect("usage: tile_memory_footprint <model.onnx> [patch size, default 64]");
    let p: usize = args.next().map_or(64, |s| s.parse().expect("patch size must be a number"));

    let bytes = std::fs::read(&model_path).expect("read model");
    let mut model = tract_onnx::onnx()
        .model_for_read(&mut std::io::Cursor::new(&bytes))
        .expect("parse onnx");
    model
        .set_input_fact(0, f32::fact([1, 1, p, p, p]).into())
        .expect("set input shape");
    let opt = model.into_optimized().expect("optimize");
    let order = opt.eval_order().expect("eval order");

    // Where each node runs, and the last step that still reads each outlet. Graph outputs live
    // past the end of the plan.
    let step: HashMap<usize, usize> = order.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let mut last_use: HashMap<(usize, usize), usize> = HashMap::new();
    for &nid in &order {
        for inlet in &opt.node(nid).inputs {
            let e = last_use.entry((inlet.node, inlet.slot)).or_insert(0);
            *e = (*e).max(step[&nid]);
        }
    }
    for out in &opt.outputs {
        last_use.insert((out.node, out.slot), order.len());
    }

    let outlet_bytes = |nid: usize, slot: usize| -> usize {
        let fact = &opt.node(nid).outputs[slot].fact;
        let n: usize = fact.shape.as_concrete().map_or(0, |s| s.iter().product());
        n * fact.datum_type.size_of()
    };
    // Folded constants (conv weights) are allocated once for the plan, not per patch.
    let is_const =
        |nid: usize| opt.node(nid).inputs.is_empty() && opt.node(nid).op().name().contains("Const");

    let mut live: HashMap<(usize, usize), usize> = HashMap::new();
    let (mut cur, mut peak, mut peak_step) = (0usize, 0usize, 0usize);
    for (i, &nid) in order.iter().enumerate() {
        if !is_const(nid) {
            for slot in 0..opt.node(nid).outputs.len() {
                let b = outlet_bytes(nid, slot);
                live.insert((nid, slot), b);
                cur += b;
            }
        }
        if cur > peak {
            peak = cur;
            peak_step = i;
        }
        let dead: Vec<(usize, usize)> = live
            .keys()
            .filter(|k| last_use.get(*k).copied().unwrap_or(0) <= i)
            .copied()
            .collect();
        for k in dead {
            cur -= live.remove(&k).unwrap();
        }
    }
    let consts: usize = order
        .iter()
        .filter(|&&n| is_const(n))
        .map(|&n| (0..opt.node(n).outputs.len()).map(|s| outlet_bytes(n, s)).sum::<usize>())
        .sum();

    let voxels = p * p * p;
    println!("{model_path}  patch {p}³ ({voxels} voxels), {} nodes", opt.nodes().len());
    println!(
        "  peak live activations : {:.1} MB  ({} B/patch voxel), at eval step {peak_step}/{}",
        peak as f64 / 1e6,
        peak / voxels,
        order.len()
    );
    println!(
        "  folded constants      : {:.1} MB (once per plan, shared by every patch)",
        consts as f64 / 1e6
    );
}

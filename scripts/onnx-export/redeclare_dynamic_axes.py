#!/usr/bin/env python3
"""Re-declare a fixed-patch SNU-LIST net's spatial axes as dynamic.

The χ-sepnet toolbox ships its networks already in ONNX, but with a **fixed** 192×192×128
input: `240531_R2PRIMEnet.onnx` (1 channel in, 1 out) and `240904_xsepnet.onnx` (3 in, 2 out).
That patch is fine natively and impossible on a 32-bit host — one 64-channel activation at that
size is 1.2 GB, against a 4 GB WASM heap.

Both graphs are fully convolutional — Conv / Relu / MaxPool / ConvTranspose / Concat, nothing
that reads a shape — so the only thing pinning the patch is the declared input. This rewrites
the spatial axes of every input and output to dynamic dim_params, leaving the weights and every
node untouched: identical output at the authors' patch, and smaller patches now runnable. Patch
dims must stay multiples of 16 (four pooling levels) or the skip-connection concatenations
misalign.

    python redeclare_dynamic_axes.py <toolbox>/models/240531_R2PRIMEnet.onnx r2primenet.onnx
    python redeclare_dynamic_axes.py <toolbox>/models/240904_xsepnet.onnx   chi-sepnet.onnx

Needs: numpy, onnx, onnxruntime.
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort

FIXED_PATCH = (192, 192, 128)
SMALL_PATCH = (128, 128, 64)
ALLOWED_OPS = {"Conv", "Relu", "MaxPool", "ConvTranspose", "Concat"}


def main():
    src, dst = sys.argv[1], sys.argv[2]
    model = onnx.load(src)

    ops = {n.op_type for n in model.graph.node}
    unexpected = ops - ALLOWED_OPS
    if unexpected:
        raise SystemExit(
            f"refusing to re-declare axes: graph contains shape-sensitive ops {sorted(unexpected)}"
        )

    for values in (model.graph.input, model.graph.output):
        for v in values:
            dims = v.type.tensor_type.shape.dim
            if len(dims) != 5:
                raise SystemExit(f"expected NCDHW, got {len(dims)} dims on '{v.name}'")
            for k, name in zip((2, 3, 4), ("D", "H", "W")):
                dims[k].ClearField("dim_value")
                dims[k].dim_param = name
    onnx.checker.check_model(model)
    onnx.save(model, dst)

    # The re-declared graph must be bit-identical at the authors' patch, and must run a smaller one.
    rng = np.random.default_rng(0)
    a, b = (ort.InferenceSession(p, providers=["CPUExecutionProvider"]) for p in (src, dst))

    def feed(sess, patch):
        """One random tensor per graph input, at `patch`, with that input's channel count."""
        return {
            i.name: rng.standard_normal((1, i.shape[1], *patch), dtype=np.float32)
            for i in sess.get_inputs()
        }

    fixed = feed(a, FIXED_PATCH)
    oa, ob = a.run(None, fixed), b.run(None, {k: v for k, v in fixed.items()})
    for k, (x, y) in enumerate(zip(oa, ob)):
        d = np.abs(x - y).max()
        if d != 0.0:
            raise SystemExit(f"output {k} differs at {FIXED_PATCH}: max|d| {d}")
    for k, out in enumerate(b.run(None, feed(b, SMALL_PATCH))):
        if out.shape[2:] != SMALL_PATCH or not np.isfinite(out).all():
            raise SystemExit(f"small-patch run failed on output {k}: shape {out.shape}")
    print(f"wrote {dst}: {len(oa)} output(s) identical at {FIXED_PATCH}, runs {SMALL_PATCH}")


if __name__ == "__main__":
    main()

# ONNX export provenance

Scripts that turn published upstream weights into the `.onnx` files listed in
`src/models/registry.rs`. Weights are never vendored here; these scripts document
exactly how each hosted artifact was produced so it can be regenerated and re-verified.

## SynthSeg

`export_synthseg.py` rebuilds the segmentation U-Net exactly as
`SynthSeg/predict_synthseg.py:build_model()` does for the non-robust path, loads the
published Keras weights by name, and converts with `tf2onnx` at opset 14. Spatial
dimensions are left dynamic, so one graph serves any padded input size.

```
git clone https://github.com/BBillot/SynthSeg      # Apache-2.0; ships models/synthseg_1.0.h5
python -m venv .venv && .venv/bin/pip install "tensorflow-cpu==2.15.1" "tf2onnx==1.16.1" onnx
.venv/bin/python export_synthseg.py --version 1.0
```

Environment notes: TensorFlow 2.15 (Keras 2.15) on Python 3.11. The vendored
`ext/lab2im/utils.py` uses `np.int`/`np.float`, removed in NumPy ≥ 1.24 — either pin
NumPy 1.23 (with a correspondingly old SciPy) or drop those two aliases from the
`isinstance` check on line 335.

SynthSeg 2.0's weights are not in the repository; fetch `synthseg_2.0.h5` from a
FreeSurfer install or the UCL link in the SynthSeg README, then pass
`--version 2.0 --weights <path>`.

Verified export: `synthseg_1.0.onnx`, 52,998,326 bytes,
sha256 `c2821a74e8a03d4073896b5c2e359b3ef86f9776bacd78e00da1c757aa97bbcc`.
Against the Keras graph on a real 192×256×128 input the posteriors agree to
max |diff| 1.0e-5 and the argmax is identical at every voxel.

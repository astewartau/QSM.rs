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

Hosted as `synthseg.onnx` on `qsmxt/qsm-onnx-weights`. Verified export: 52,998,326 bytes,
sha256 `c2821a74e8a03d4073896b5c2e359b3ef86f9776bacd78e00da1c757aa97bbcc`.
Against the Keras graph on a real 192×256×128 input the posteriors agree to
max |diff| 1.0e-5 and the argmax is identical at every voxel.

## RS2-Net

`export_rs2net.py` builds `RSSNet` as `RS2/inference/predict.py:load_what_we_need()` does, loads
the released `RS2_pretrained_model.pt`, and exports at opset 17 with a **fixed** `[1,1,128,96,128]`
input — the Swin encoder computes its window padding with Python ints, which tracing bakes in, so
the graph only runs at the traced size. 128×96×128 (nnU-Net's transposed `(y, z, x)` order) peaks at
≈2.7 GB in tract, inside wasm32's 4 GB; the released 128×128×160 needs ≈4.5 GB.

```
git clone https://github.com/VitoLin21/Rodent-Skull-Stripping      # GPL-3.0
curl -LO https://github.com/VitoLin21/Rodent-Skull-Stripping/releases/download/main-tag/RS2_pretrained_model.pt
# SwinTransformer.proj_out: F.layer_norm(x, [ch]) -> [int(ch)] (ch comes from x.size(), which
# the exporter cannot make static)
sed -i 's/F.layer_norm(x, \[ch\])/F.layer_norm(x, [int(ch)])/' Rodent-Skull-Stripping/RS2/network/RSSNet.py
python -m venv .venv && .venv/bin/pip install torch==2.3.0 "monai[einops]==1.3.0" onnx onnxruntime
.venv/bin/python export_rs2net.py --rs2-dir Rodent-Skull-Stripping --weights RS2_pretrained_model.pt
```

The checkpoint was saved from a `torch.compile`d model; the script strips the `_orig_mod.` prefix
from its keys. Environment: Python 3.10, torch 2.3.0 (CPU), onnx 1.23.

To be hosted as `rs2-net.onnx` on `qsmxt/qsm-onnx-weights` (registry status `Pending` until then;
bring-your-own via `$QSM_MODEL_DIR` works meanwhile). Verified export (reproducible — re-running
gives identical bytes): 63,456,162 bytes, sha256
`a120ce43b06a9f3ddecf85bf18f281cbb54c354ef15b34a6bcaaffa4e3c4a570`. Against PyTorch the logits
agree to max |diff| 2.2e-3 with the same sign at 99.9998 % of voxels.

`ref_rs2net.py` runs RS2-Net's own nnU-Net pipeline around this graph (ONNX Runtime) and writes the
fixtures for `bet::rs2net`'s parity tests and `tests/models_onnx.rs::rs2net_matches_python_reference`
(`RS2_REF_DIR`, `RS2NET_ONNX`). It needs RS2-Net's requirements plus `blosc2` and
`acvl_utils==0.2.1`; set `TORCHDYNAMO_DISABLE=1`, as RS2-Net calls `torch.compile`.

## R2PRIMEnet and χ-sepnet

The SNU-LIST χ-sepnet toolbox ships both networks already in ONNX — `240531_R2PRIMEnet.onnx`
(single channel, R2*→R2′) and `240904_xsepnet.onnx` (3 channels → 2, χ-separation), each z-scored
in `Dr`-scaled units — but with a **fixed** `[1,·,192,192,128]` input.
`redeclare_dynamic_axes.py` re-declares the spatial axes as dynamic — both graphs are
Conv/Relu/MaxPool/ConvTranspose/Concat only, so nothing reads a shape — and verifies every
output is bit-identical at the authors' patch and runnable at a smaller one. Weights and nodes
are untouched.

That matters because a 32-bit host cannot afford the authors' patch: one 64-channel activation
at 192×192×128 is 1.2 GB, and the WASM heap tops out at 4 GB. `relaxometry::r2primenet` and
`separation::chisepnet` each take the patch as a parameter (`AUTHORS_PATCH` natively,
`WASM_PATCH` = 128×128×64 in the browser); patch dims must be multiples of 16, the networks'
four pooling levels.

```
python redeclare_dynamic_axes.py <toolbox>/models/240531_R2PRIMEnet.onnx r2primenet.onnx
python redeclare_dynamic_axes.py <toolbox>/models/240904_xsepnet.onnx   chi-sepnet.onnx
```

Hosted on `qsmxt/qsm-onnx-weights`: `r2primenet.onnx` 90,307,128 bytes, sha256
`44cb5e67d1c68dae87a5f532e501dc0d4bf627d91f9c8ba35a56df5ac7ec3cd0`; `chi-sepnet.onnx`
90,314,172 bytes, sha256 `5b442fdfdb88f50ec9149b384dabd0d2d9adb6983947d9cfa58382b78676bed9`. Redistribution permission
was obtained from the authors; the toolbox itself is behind their Google form. The normalisation
constants baked into `R2PrimeNetNorm` come from the toolbox's
`xsepnet_train_patch_norm_factor_inplane_largedegree_romeo_arlo.mat` — the same file χ-sepnet
uses (`r2star_mean/std`, `r2prime_mean/std`, Dr = 114 Hz/ppm).

`ref_r2primenet.py` is the parity reference: it runs the authors' recipe in Python/onnxruntime
on a deterministic synthetic R2* volume over a real brain mask and writes both the input and the
output, so `tests/models_onnx.rs::r2primenet_matches_python_reference` compares against
identical input bytes.

```
python ref_r2primenet.py --onnx <toolbox>/models/240531_R2PRIMEnet.onnx \
    --norm <toolbox>/models/xsepnet_train_patch_norm_factor_inplane_largedegree_romeo_arlo.mat \
    --mask ../../TEST_DATA/QSM_Dat08c_Mask.nii.gz --out /tmp/r2primenet_ref
R2PRIMENET_ONNX=<hosted>/r2primenet.onnx \
  cargo test --release --features onnx --test models_onnx r2primenet_matches -- --ignored --nocapture
```

Verified on the 205×164×205 challenge mask: the authors' patch (2×1×2 tiles) matches the Python
reference at corr 1.000000, max |Δ| 2.0e-5 Hz, and the WASM patch (16 tiles) agrees with the
authors' patch at corr 0.9980, NRMSE 2.92% — the price of the smaller context, and the test's
regression guard.

## SUSEP-Net

No export or edit: the authors' `susep-net.onnx` already has dynamic spatial axes (three
`[1,1,D,H,W]` inputs — qsm, r2prime, lfs — and two outputs). Hosted verbatim as
`susep-net.onnx`, 205,743,584 bytes.

`ref_susep_net.py` is its parity reference — the authors' whole-volume recipe in
Python/onnxruntime on deterministic synthetic inputs over a real brain mask. The Rust test
checks two things against it: that the whole-volume path reproduces it, and what
`SusepNetParams::patch` (the sliding window a 32-bit host needs, since whole-volume activations
do not fit) costs relative to it.

```
python ref_susep_net.py --onnx susep-net.onnx \
    --mask ../../TEST_DATA/QSM_Dat08c_Mask.nii.gz --out /tmp/susep_net_ref
SUSEPNET_ONNX=<hosted>/susep-net.onnx \
  cargo test --release --features onnx --test models_onnx susep_net_matches_ref -- --ignored --nocapture
```

On the 208×168×208 padded volume the whole-volume path matches the reference to corr > 0.999999
and max |Δ| < 5e-5 ppm, and the 128×128×64 sliding window agrees with it at χ+ corr 0.9999 /
NRMSE 0.72%, χ− corr 0.9981 / NRMSE 1.02%.

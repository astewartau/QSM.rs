"""Export RS2-Net (rodent brain extraction) from its released PyTorch checkpoint to ONNX.

Builds `RS2.network.RSSNet.RSSNet` exactly as `RS2/inference/predict.py:load_what_we_need()`
does, loads the released `RS2_pretrained_model.pt`, and exports it at opset 17 with a **fixed**
`[1, 1, *PATCH]` input: the Swin encoder's window padding is computed with Python ints, which
tracing bakes in, so the graph only runs at the size it was traced at. The default 128×96×128
patch (in nnU-Net's transposed `(y, z, x)` order) peaks at ≈2.7 GB in tract; RS2-Net's own
128×128×160 needs ≈4.5 GB, more than wasm32 can address.

Two fixes to the released artefacts, neither of which changes the network:

* the checkpoint was saved from a `torch.compile`d model, so every state-dict key carries an
  `_orig_mod.` prefix — stripped here;
* `SwinTransformer.proj_out` calls `F.layer_norm(x, [ch])` with `ch` taken from `x.size()`,
  which the ONNX exporter cannot make static — patch that line to `[int(ch)]` in the RS2
  checkout before running (see README).

    python export_rs2net.py --rs2-dir Rodent-Skull-Stripping --weights RS2_pretrained_model.pt

Needs: torch 2.3 (CPU), monai 1.3, einops, onnx, onnxruntime (for the parity check).
"""
import argparse
import hashlib
import os
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--rs2-dir', default='Rodent-Skull-Stripping',
               help='checkout of https://github.com/VitoLin21/Rodent-Skull-Stripping')
p.add_argument('--weights', default='RS2_pretrained_model.pt',
               help='RS2_pretrained_model.pt from the repository releases')
p.add_argument('--patch', default='128x96x128', help='fixed input patch, (y, z, x) order')
p.add_argument('--out', default='rs2-net.onnx')
a = p.parse_args()

sys.path.insert(0, os.path.abspath(a.rs2_dir))
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import torch  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from RS2.network.RSSNet import RSSNet  # noqa: E402

patch = tuple(int(v) for v in a.patch.split('x'))
assert all(v % 32 == 0 for v in patch), 'Swin-UNETR needs every patch axis to be a multiple of 32'

net = RSSNet(img_size=(128, 128, 160), in_channels=1, out_channels=1, feature_size=48)
state = torch.load(a.weights, map_location='cpu')['state_dict']
net.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in state.items()})
net.eval()

x = torch.from_numpy(np.random.default_rng(0).standard_normal((1, 1, *patch)).astype(np.float32))
with torch.no_grad():
    ref = net(x).numpy()
torch.onnx.export(net, x, a.out, opset_version=17, input_names=['input'], output_names=['logits'],
                  do_constant_folding=True)
onnx.checker.check_model(onnx.load(a.out))

y = ort.InferenceSession(a.out, providers=['CPUExecutionProvider']).run(None, {'input': x.numpy()})[0]
print(f'max |ort - torch| {np.abs(y - ref).max():.2e}, sign agreement {((y > 0) == (ref > 0)).mean():.7f}')
data = open(a.out, 'rb').read()
print(f'{a.out}: {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()}')

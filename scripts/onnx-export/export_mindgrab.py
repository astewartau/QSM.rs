"""Export MindGrab (brainchop's skull-stripping MeshNet) to ONNX.

MindGrab is a 25-layer dilated-convolution MeshNet (15 channels, dilations 16-8-4-2-1 five
times over) with a 1x1x1 two-class head. brainchop-cli builds it in tinygrad from
`model.json` + `model.pth` (neuroneural/brainchop-models, meshnet/mindgrab); this script
rebuilds the same network in PyTorch, loads the same state dict, and exports it.

Every hidden layer is Conv3d (no bias) -> GroupNorm(15 groups, 15 channels, no affine,
eps 1e-5) -> GELU. Two details follow tinygrad rather than torch's defaults, because the
released weights are run through tinygrad:
  * GELU is the tanh approximation (tinygrad's `Tensor.gelu` default);
  * GroupNorm with one channel per group is instance normalisation; it is written out as
    mean / variance reductions so the graph uses only ReduceMean, Sub, Mul, Sqrt, Div.

The network is fully convolutional (padding = dilation), so the spatial axes are exported as
dynamic; it is run on brainchop's 256^3 conformed volume.

    python export_mindgrab.py --config model.json --weights model.pth --out mindgrab.onnx
"""
import argparse
import json

import numpy as np
import torch
from torch import nn


class InstanceNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        y = x - x.mean(dim=(2, 3, 4), keepdim=True)
        return y / torch.sqrt((y * y).mean(dim=(2, 3, 4), keepdim=True) + self.eps)


class MindGrab(nn.Module):
    def __init__(self, config, channels=15, n_classes=2):
        super().__init__()
        layers = config["layers"]
        blocks = []
        for i, l in enumerate(layers[:-1]):
            cin = 1 if i == 0 else channels
            blocks.append(nn.Sequential(
                nn.Conv3d(cin, channels, l["kernel_size"], stride=l["stride"],
                          padding=l["padding"], dilation=l["dilation"], bias=False),
                InstanceNorm(),
                nn.GELU(approximate="tanh"),
            ))
        last = layers[-1]
        blocks.append(nn.Conv3d(channels, n_classes, last["kernel_size"], padding=last["padding"],
                                dilation=last["dilation"], bias=True))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


def load(config_path, weights_path):
    with open(config_path) as f:
        config = json.load(f)
    # brainchop's config uses -1 placeholders; load_meshnet then fills channels=15, classes=2
    assert config["layers"][0]["in_channels"] == -1 and config["layers"][-1]["out_channels"] == -1
    assert config["bnorm"] and config.get("gelu") and not config.get("affine", False)
    net = MindGrab(config)
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    # keys are model.<i>.0.weight (conv in each block) and model.25.{weight,bias} -- same layout
    net.load_state_dict(sd, strict=True)
    return net.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="mindgrab.onnx")
    a = ap.parse_args()

    net = load(a.config, a.weights)
    print("parameters:", sum(p.numel() for p in net.parameters()))
    x = torch.from_numpy(np.random.default_rng(0).random((1, 1, 64, 64, 64), dtype=np.float32))
    torch.onnx.export(
        net, x, a.out, input_names=["image"], output_names=["logits"], opset_version=17,
        dynamic_axes={"image": {2: "d", 3: "h", 4: "w"}, "logits": {2: "d", 3: "h", 4: "w"}},
        dynamo=False,
    )

    import onnxruntime as ort
    with torch.no_grad():
        want = net(x).numpy()
    got = ort.InferenceSession(a.out).run(None, {"image": x.numpy()})[0]
    print("torch vs onnxruntime max |diff|:", float(np.abs(want - got).max()))


if __name__ == "__main__":
    main()

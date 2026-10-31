#!/usr/bin/env python3
"""Rebuild QSMnet (SNU-LIST) cleanly in PyTorch, port the TF1.14 checkpoint weights,
validate against a TF reference output, and export a tract-friendly ONNX.

Why not tf2onnx directly: tf2onnx emits NHWC↔NCHW Reshape/Transpose and dynamic
deconv-shape ops that the pure-Rust `tract` engine can't shape-analyse. Rebuilding
the plain 3D U-Net in PyTorch and exporting gives a clean NCDHW graph tract runs
(like xQSM), keeping QSMnet tract-only + wasm-capable.

Inputs (written by dump_qsmnet_tf.py inside QSM-CI's original image):
  qn_weights.npz  – all trained TF variables (no optimizer slots)
  qn_in.npy       – a fixed NDHWC input
  qn_tfout.npy    – the TF graph's output for that input (ground truth)
  qn_act.txt      – the checkpoint's activation, from its network_info_<name>.npy

Layer recipe (from Code/network_model.py + utils.py):
  conv block : Conv3d(k5,pad2) + bias -> BatchNorm(eps=1e-3, inference stats) -> activation
               activation: ReLU for QSMnet_64, LeakyReLU(alpha=0.1) for QSMnet+_64 (network_info)
  deconv     : ConvTranspose3d(k2,s2) + bias   (no BN, no activation)
  final      : Conv3d(k1) + bias                (linear)
  pool       : MaxPool3d(2)
"""
import argparse
import numpy as np
import torch
import torch.nn as nn


ACTS = {
    # utils.py: tf.nn.relu, or tf.nn.leaky_relu(alpha=0.1) (not TF's 0.2 default)
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
}


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, act):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, 5, padding=2)
        self.bn = nn.BatchNorm3d(cout, eps=1e-3)
        self.act = ACTS[act]()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class QSMnet(nn.Module):
    def __init__(self, act):
        super().__init__()
        def block(cin, cout):
            return ConvBlock(cin, cout, act)

        self.c11, self.c12 = block(1, 32), block(32, 32)
        self.c21, self.c22 = block(32, 64), block(64, 64)
        self.c31, self.c32 = block(64, 128), block(128, 128)
        self.c41, self.c42 = block(128, 256), block(256, 256)
        self.lc1, self.lc2 = block(256, 512), block(512, 512)
        self.d4 = nn.ConvTranspose3d(512, 256, 2, stride=2)
        self.c51, self.c52 = block(512, 256), block(256, 256)
        self.d3 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.c61, self.c62 = block(256, 128), block(128, 128)
        self.d2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.c71, self.c72 = block(128, 64), block(64, 64)
        self.d1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.c81, self.c82 = block(64, 32), block(32, 32)
        self.out = nn.Conv3d(32, 1, 1)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        c11 = self.c11(x); c12 = self.c12(c11); p1 = self.pool(c12)
        c21 = self.c21(p1); c22 = self.c22(c21); p2 = self.pool(c22)
        c31 = self.c31(p2); c32 = self.c32(c31); p3 = self.pool(c32)
        c41 = self.c41(p3); c42 = self.c42(c41); p4 = self.pool(c42)
        lc1 = self.lc1(p4); lc2 = self.lc2(lc1)
        d4 = self.d4(lc2); c51 = self.c51(torch.cat([c42, d4], 1)); c52 = self.c52(c51)
        d3 = self.d3(c52); c61 = self.c61(torch.cat([c32, d3], 1)); c62 = self.c62(c61)
        d2 = self.d2(c62); c71 = self.c71(torch.cat([c22, d2], 1)); c72 = self.c72(c71)
        d1 = self.d1(c72); c81 = self.c81(torch.cat([c12, d1], 1)); c82 = self.c82(c81)
        return self.out(c82)


def _conv_w(w):  # TF [kd,kh,kw,in,out] -> torch [out,in,kd,kh,kw]
    return torch.from_numpy(np.transpose(w, (4, 3, 0, 1, 2)).copy())


def _deconv_w(w):  # TF conv3d_transpose [kd,kh,kw,out,in] -> torch [in,out,kd,kh,kw]
    return torch.from_numpy(np.transpose(w, (4, 3, 0, 1, 2)).copy())


def load_weights(net: QSMnet, W: dict):
    def block(mod: ConvBlock, name: str):
        mod.conv.weight.data = _conv_w(W[f"qsmnet/{name}/conv_weights"])
        mod.conv.bias.data = torch.from_numpy(W[f"qsmnet/{name}/biases"].copy())
        mod.bn.weight.data = torch.from_numpy(W[f"qsmnet/{name}/bn/gamma"].copy())
        mod.bn.bias.data = torch.from_numpy(W[f"qsmnet/{name}/bn/beta"].copy())
        mod.bn.running_mean.data = torch.from_numpy(W[f"qsmnet/{name}/bn/mean_sh"].copy())
        mod.bn.running_var.data = torch.from_numpy(W[f"qsmnet/{name}/bn/var_sh"].copy())

    def deconv(mod, name):
        mod.weight.data = _deconv_w(W[f"qsmnet/{name}/deconv_weights"])
        mod.bias.data = torch.from_numpy(W[f"qsmnet/{name}/biases"].copy())

    for mod, name in [
        (net.c11, "conv11"), (net.c12, "conv12"), (net.c21, "conv21"), (net.c22, "conv22"),
        (net.c31, "conv31"), (net.c32, "conv32"), (net.c41, "conv41"), (net.c42, "conv42"),
        (net.lc1, "l_conv1"), (net.lc2, "l_conv2"),
        (net.c51, "conv51"), (net.c52, "conv52"), (net.c61, "conv61"), (net.c62, "conv62"),
        (net.c71, "conv71"), (net.c72, "conv72"), (net.c81, "conv81"), (net.c82, "conv82"),
    ]:
        block(mod, name)
    for mod, name in [(net.d4, "deconv4"), (net.d3, "deconv3"), (net.d2, "deconv2"), (net.d1, "deconv1")]:
        deconv(mod, name)
    net.out.weight.data = _conv_w(W["qsmnet/out/weights"])
    net.out.bias.data = torch.from_numpy(W["qsmnet/out/biases"].copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/qsmnet_export")
    ap.add_argument("--out", default="/tmp/qsmnet_export/qsmnet.onnx")
    args = ap.parse_args()

    W = dict(np.load(f"{args.dir}/qn_weights.npz"))
    act = open(f"{args.dir}/qn_act.txt").read().strip()
    print(f"activation (network_info): {act}")
    net = QSMnet(act).eval()
    load_weights(net, W)

    # Validate against the TF reference (NDHWC -> NCDHW).
    x_ndhwc = np.load(f"{args.dir}/qn_in.npy")
    tf_out = np.load(f"{args.dir}/qn_tfout.npy")  # NDHWC
    x = torch.from_numpy(np.transpose(x_ndhwc, (0, 4, 1, 2, 3)).copy())
    with torch.inference_mode():
        y = net(x).numpy()  # NCDHW
    y_ndhwc = np.transpose(y, (0, 2, 3, 4, 1))
    md = float(np.max(np.abs(y_ndhwc - tf_out)))
    print(f"torch-vs-TF max|Δ| = {md:.3e}")
    assert md < 1e-3, "reimplementation does not match TF"

    # Export clean NCDHW ONNX (input 'field' [1,1,D,H,W]).
    example = torch.from_numpy(x.numpy())
    dyn = {2: "D", 3: "H", 4: "W"}
    torch.onnx.export(
        net, example, args.out,
        input_names=["field"], output_names=["chi"],
        dynamic_axes={"field": dyn, "chi": dyn},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

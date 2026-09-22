#!/usr/bin/env python3
"""Reference implementation of SUSEP-Net inference, for the Rust parity test.

Runs the authors' recipe in Python/onnxruntime — the one `separation::susep_net` ports — on
deterministic synthetic inputs over a real brain mask, and writes the inputs and both outputs so
`tests/models_onnx.rs::susep_net_matches_python_reference` compares against identical bytes.

Whole volume, as the authors run it: z-score each input, zero outside the mask, post-pad each
dim to a multiple of 8, one forward pass, de-normalise, crop, mask. The Rust port additionally
supports a sliding-window patch for 32-bit hosts; the test measures that against this.

    python ref_susep_net.py --onnx susep-net.onnx \
        --mask ../../TEST_DATA/QSM_Dat08c_Mask.nii.gz --out /tmp/susep_net_ref

Needs: numpy, nibabel, onnxruntime.
"""
import argparse
import os

import numpy as np
import nibabel as nib
import onnxruntime as ort

# Training z-score constants shipped with SUSEPNet.pth (all_mean_std.mat), as in SusepNetNorm.
NORM = {
    "qsm": (-6.0663105e-05, 0.023533047),
    "lfs": (-4.8702253e-05, 0.012554166),
    "r2prime": (4.7629275, 10.889079),
    "chi_pos": (0.0089528897, 0.025519046),
    "chi_neg": (0.0090135528, 0.019603666),
}


def synth(shape, kind):
    """Deterministic, structured stand-ins in the right physical ranges."""
    nx, ny, nz = shape
    x = np.arange(nx)[:, None, None] / nx
    y = np.arange(ny)[None, :, None] / ny
    z = np.arange(nz)[None, None, :] / nz
    wave = np.sin(6 * np.pi * x) * np.cos(5 * np.pi * y) * np.cos(3 * np.pi * z)
    blob = np.exp(-(((x - 0.45) ** 2 + (y - 0.55) ** 2 + (z - 0.5) ** 2) / 0.004))
    if kind == "qsm":         # ppm, +/- 0.1 with a paramagnetic core
        return 0.03 * wave + 0.12 * blob
    if kind == "lfs":         # ppm, smaller than chi_total
        return 0.012 * wave + 0.04 * blob
    return np.clip(6 + 10 * (0.5 + 0.5 * wave) + 25 * blob, 0, None)   # r2prime, Hz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", default="/tmp/susep_net_ref")
    args = ap.parse_args()

    mask_img = nib.load(args.mask)
    mask = np.asarray(mask_img.get_fdata()) > 0.5
    fields = {k: synth(mask.shape, k) * mask for k in ("qsm", "lfs", "r2prime")}

    X, Y, Z = mask.shape
    pad = [(0, (-d) % 8) for d in (X, Y, Z)]

    def zs(name):
        mean, std = NORM[name]
        return np.pad(((fields[name] - mean) / std * mask).astype(np.float32), pad)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    feed = {i.name: zs(i.name)[None, None] for i in sess.get_inputs()}
    print("inputs:", {k: v.shape for k, v in feed.items()}, flush=True)
    outs = sess.run(None, feed)

    pos = (outs[0][0, 0] * NORM["chi_pos"][1] + NORM["chi_pos"][0])[:X, :Y, :Z] * mask
    neg = (outs[1][0, 0] * NORM["chi_neg"][1] + NORM["chi_neg"][0])[:X, :Y, :Z] * mask

    os.makedirs(args.out, exist_ok=True)
    aff, hdr = mask_img.affine, mask_img.header
    for name, data in (("qsm", fields["qsm"]), ("lfs", fields["lfs"]), ("r2prime", fields["r2prime"]),
                       ("chi_pos_ref", pos), ("chi_neg_ref", neg)):
        nib.save(nib.Nifti1Image(data.astype(np.float32), aff, hdr), os.path.join(args.out, f"{name}.nii.gz"))
    print(f"chi+ mean {pos[mask].mean():.4f} ppm, |chi-| mean {neg[mask].mean():.4f} ppm; wrote {args.out}")


if __name__ == "__main__":
    main()

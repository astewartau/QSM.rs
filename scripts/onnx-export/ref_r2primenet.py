#!/usr/bin/env python3
"""Reference implementation of R2PRIMEnet inference, for the Rust parity test.

Runs the authors' recipe in Python/onnxruntime — the same one `utils::r2primenet` ports —
on a deterministic synthetic R2* volume over a real brain mask, and writes both the input
and the output so `tests/models_onnx.rs::r2primenet_matches_python_reference` can compare
against identical bytes.

The input is synthetic on purpose: the port's risk is the sliding-window tiling, the
Dr-scaled z-scoring and the de-normalisation, none of which care whether the R2* map came
from a real fit. Using a written-out volume rather than a formula evaluated twice keeps
the two implementations bit-identical on input.

    python ref_r2primenet.py \
        --onnx  ~/repos/qsm/chi-separation/Chisep_Toolbox_v1.1.3/models/240531_R2PRIMEnet.onnx \
        --norm  ~/repos/qsm/chi-separation/Chisep_Toolbox_v1.1.3/models/xsepnet_train_patch_norm_factor_inplane_largedegree_romeo_arlo.mat \
        --mask  ~/repos/qsm/QSM.rs/TEST_DATA/QSM_Dat08c_Mask.nii.gz \
        --out   /tmp/r2primenet_ref

Needs: numpy, scipy, nibabel, onnxruntime.
"""
import argparse
import os

import numpy as np
import nibabel as nib
import scipy.io as sio
import onnxruntime as ort

PATCH = (192, 192, 128)
DEFAULT_DR = 114.0


def synthetic_r2star(shape):
    """A deterministic, structured R2* map in Hz (roughly 0-60, in vivo-like range)."""
    nx, ny, nz = shape
    x = np.arange(nx)[:, None, None]
    y = np.arange(ny)[None, :, None]
    z = np.arange(nz)[None, None, :]
    base = 20.0 + 12.0 * np.sin(2 * np.pi * x / 37.0) * np.cos(2 * np.pi * y / 29.0)
    ramp = 8.0 * (z / max(nz - 1, 1))
    # A couple of iron-like hot spots so the map is not purely smooth.
    def blob(cx, cy, cz, r, amp):
        d2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
        return amp * np.exp(-d2 / (2.0 * r * r))
    hot = blob(nx * 0.42, ny * 0.55, nz * 0.5, 9.0, 25.0) + blob(nx * 0.58, ny * 0.55, nz * 0.5, 9.0, 25.0)
    return np.clip(base + ramp + hot, 0.0, None)


def starts(size, patch):
    if size <= patch:
        return [0]
    st = list(range(0, size - patch + 1, int(patch * 0.75)))
    if st[-1] != size - patch:
        st.append(size - patch)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--norm", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", default="/tmp/r2primenet_ref")
    ap.add_argument("--dr", type=float, default=DEFAULT_DR)
    args = ap.parse_args()

    mask_img = nib.load(args.mask)
    mask = np.asarray(mask_img.get_fdata()) > 0.5
    r2s = synthetic_r2star(mask.shape) * mask

    N = sio.loadmat(args.norm)
    s = lambda k: float(N[k].ravel()[0])

    vol = (((r2s / args.dr) - s("r2star_mean")) / s("r2star_std") * mask).astype(np.float32)
    X, Y, Z = vol.shape
    pad = [max(0, PATCH[k] - (X, Y, Z)[k]) for k in range(3)]
    volp = np.pad(vol, ((0, pad[0]), (0, pad[1]), (0, pad[2])))
    Xp, Yp, Zp = volp.shape

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    acc = np.zeros((Xp, Yp, Zp), np.float64)
    wsum = np.zeros((Xp, Yp, Zp), np.float64)
    tiles = [(x0, y0, z0) for x0 in starts(Xp, PATCH[0])
             for y0 in starts(Yp, PATCH[1]) for z0 in starts(Zp, PATCH[2])]
    for n, (x0, y0, z0) in enumerate(tiles, 1):
        sl = (slice(x0, x0 + PATCH[0]), slice(y0, y0 + PATCH[1]), slice(z0, z0 + PATCH[2]))
        out = sess.run(None, {iname: volp[sl][None, None].astype(np.float32)})[0][0, 0]
        acc[sl] += out
        wsum[sl] += 1.0
        print(f"  patch {n}/{len(tiles)}", flush=True)
    acc /= np.maximum(wsum, 1.0)

    r2p = np.clip((acc * s("r2prime_std") + s("r2prime_mean")) * args.dr, 0, None)[:X, :Y, :Z] * mask

    os.makedirs(args.out, exist_ok=True)
    aff, hdr = mask_img.affine, mask_img.header
    nib.save(nib.Nifti1Image(r2s.astype(np.float32), aff, hdr), os.path.join(args.out, "r2star.nii.gz"))
    nib.save(nib.Nifti1Image(r2p.astype(np.float32), aff, hdr), os.path.join(args.out, "r2prime_ref.nii.gz"))
    print(f"R2* mean {r2s[mask].mean():.2f} Hz -> R2' mean {r2p[mask].mean():.2f} Hz "
          f"(Dr={args.dr:g}); wrote {args.out}")


if __name__ == "__main__":
    main()

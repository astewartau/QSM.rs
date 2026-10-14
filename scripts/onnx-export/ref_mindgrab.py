"""Reference fixtures for `bet::mindgrab`'s parity tests, from brainchop-cli's own pipeline.

Runs MindGrab exactly as `brainchop --skull-strip` does — niimath `-conform` (256^3, 1 mm,
LIA, intensities robustly scaled to uint8), quantile normalisation, the tinygrad MeshNet,
largest 26-connected component (niimath `-bwlabel 26`), nearest-neighbour reslice back to
the input grid (niimath `-reslice_nn`) — and the exported ONNX graph on the same input.

Writes to --out:
  mag.nii                   the 3D input (first volume of a 4D input)
  conformed.nii             niimath's uint8 conformed volume (its header carries the affine)
  ref_input.npy             normalised network input, float32 C-order (x, y, z) = (1,1,256,256,256)
  ref_logits.npy            ONNX Runtime logits on ref_input (1,2,256,256,256)
  ref_mask_conformed.nii    brainchop's mask in conformed space (after the largest component)
  ref_mask.nii.gz           brainchop's mask on the input grid

    pip install brainchop onnxruntime nibabel     # brainchop pulls in tinygrad and niimath
    python ref_mindgrab.py --mag magnitude.nii.gz --onnx mindgrab.onnx --out /tmp/mindgrab_ref
"""
import argparse
import os
import subprocess

import nibabel as nib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mag", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    img = nib.load(a.mag)
    data = np.asanyarray(img.dataobj).astype(np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    mag = os.path.join(a.out, "mag.nii")
    nib.save(nib.Nifti1Image(data, img.affine, img.header), mag)

    from brainchop import load, segment
    from brainchop.niimath import conform, truncate_header_bytes
    from brainchop.tiny_meshnet import qnormalize
    from tinygrad import Tensor

    vol = load(mag)
    conf_u8, conf_hdr = conform(mag)
    with open(os.path.join(a.out, "conformed.nii"), "wb") as f:
        f.write(truncate_header_bytes(conf_hdr) + conf_u8.tobytes())

    # (z, y, x) uint8 -> network input (1, 1, x, y, z), as brainchop.api.segment does
    x = vol.data.permute(2, 1, 0).cast("float32").rearrange("... -> 1 1 ...")
    x = qnormalize(x).numpy().astype(np.float32)
    np.save(os.path.join(a.out, "ref_input.npy"), x)

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count() or 1
    logits = ort.InferenceSession(a.onnx, so).run(None, {"image": x})[0]
    np.save(os.path.join(a.out, "ref_logits.npy"), logits.astype(np.float32))

    result = segment(vol, "mindgrab")
    m = result.data.numpy().astype(np.uint8)  # (z, y, x) C-order == NIfTI (x, y, z) Fortran
    onnx_argmax = (logits[0, 1] > logits[0, 0]).astype(np.uint8).transpose(2, 1, 0)
    print("ONNX argmax vs tinygrad (before largest component): differing voxels",
          int(((onnx_argmax > 0) != (m > 0)).sum()), "of", int((m > 0).sum()), "brain")

    mc = os.path.join(a.out, "ref_mask_conformed.nii")
    with open(mc, "wb") as f:
        f.write(truncate_header_bytes(conf_hdr) + m.tobytes())
    subprocess.run(["niimath", mc, "-reslice_nn", mag, "-gz", "1",
                    os.path.join(a.out, "ref_mask.nii.gz"), "-odt", "char"], check=True)


if __name__ == "__main__":
    main()

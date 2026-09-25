#!/usr/bin/env python3
"""Reference RS2-Net inference, for the Rust parity tests.

Runs RS2-Net's own nnU-Net pipeline (its `DefaultPreprocessor`, sliding window and export) on a
magnitude image, with the exported `rs2-net.onnx` under ONNX Runtime standing in for the PyTorch
network — so the Rust port (`bet::rs2net`) and this reference see the same graph and patch. Writes
the fixtures `bet::rs2net`'s unit tests and `tests/models_onnx.rs::rs2net_matches_python_reference`
read:

* `mag.nii` — the input, copied, so both sides see identical bytes;
* `ref_preprocessed.npy` — the preprocessed volume (transposed, cropped, z-scored, resampled);
* `ref_logits.npy` — the sliding-window logits;
* `ref_mask.nii.gz` — the final mask.

One deliberate difference from RS2-Net's `export_prediction_from_sigmoid`: the logits are
resampled back with the original spacing **transposed** (`transpose_forward`), as upstream
nnU-Net v2 does. RS2-Net passes it untransposed, which on anisotropic data applies the
nearest-neighbour pass to the wrong axis; the Rust port follows nnU-Net.

    python ref_rs2net.py --rs2-dir Rodent-Skull-Stripping --onnx rs2-net.onnx \\
        --input mouse_mag.nii --out /tmp/rs2net_ref

Needs: RS2-Net's Python dependencies (see its requirements.txt), onnxruntime, nibabel.
"""
import argparse
import os
import shutil
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--rs2-dir', default='Rodent-Skull-Stripping')
p.add_argument('--onnx', default='rs2-net.onnx')
p.add_argument('--patch', default='128x96x128', help='the patch the ONNX graph was traced at')
p.add_argument('--input', required=True, help='magnitude NIfTI (uncompressed .nii)')
p.add_argument('--out', default='/tmp/rs2net_ref')
a = p.parse_args()

sys.path.insert(0, os.path.abspath(a.rs2_dir))
import nibabel as nib  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
import RS2  # noqa: E402
from batchgenerators.utilities.file_and_folder_operations import load_json  # noqa: E402
from RS2.utilities.plans_handling.plans_handler import PlansManager  # noqa: E402
from RS2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor  # noqa: E402
from RS2.inference.sliding_window_prediction import (  # noqa: E402
    compute_gaussian, predict_sliding_window_return_logits)

os.makedirs(a.out, exist_ok=True)
shutil.copy(a.input, os.path.join(a.out, 'mag.nii'))
patch = [int(v) for v in a.patch.split('x')]

plans = PlansManager(load_json(os.path.join(RS2.__path__[0], 'jsons/plans.json')))
dataset = load_json(os.path.join(RS2.__path__[0], 'jsons/dataset.json'))
cm = plans.get_configuration('3d_fullres')
data, _, props = DefaultPreprocessor(verbose=False).run_case([a.input], None, plans, cm, dataset)
np.save(os.path.join(a.out, 'ref_preprocessed.npy'), data[0].astype(np.float32))

sess = ort.InferenceSession(a.onnx, providers=['CPUExecutionProvider'])


class OnnxNet(torch.nn.Module):
    def forward(self, x):
        return torch.from_numpy(sess.run(None, {'input': x.numpy().astype(np.float32)})[0])


# RS2-Net sizes the logits by the label manager's two heads and broadcasts its one output
# channel into both; mirror that, so the fixture has the layout RS2-Net itself produces.
logits = predict_sliding_window_return_logits(
    OnnxNet(), torch.from_numpy(data), 2, patch, mirror_axes=None, tile_step_size=0.5,
    use_gaussian=True, precomputed_gaussian=torch.from_numpy(compute_gaussian(patch)).half(),
    perform_everything_on_gpu=False, verbose=False, device=torch.device('cpu')).numpy()
np.save(os.path.join(a.out, 'ref_logits.npy'), logits.astype(np.float32))

spacing = [props['spacing'][i] for i in plans.transpose_forward]
back = cm.resampling_fn(logits[:1].astype(np.float32), props['shape_after_cropping_and_before_resampling'],
                        cm.spacing, spacing)
seg = np.zeros(props['shape_before_cropping'], dtype=np.uint8)
(b0, b1, b2) = props['bbox_used_for_cropping']
seg[b0[0]:b0[1], b1[0]:b1[1], b2[0]:b2[1]] = (back[0] > 0).astype(np.uint8)   # sigmoid > 0.5
seg = seg.transpose(plans.transpose_backward)                                 # (z, y, x) array order
img = nib.load(a.input)
nib.save(nib.Nifti1Image(seg.transpose(2, 1, 0), img.affine), os.path.join(a.out, 'ref_mask.nii.gz'))
print(f'preprocessed {data.shape}, bbox {props["bbox_used_for_cropping"]}, '
      f'mask {int(seg.sum())} voxels -> {a.out}')

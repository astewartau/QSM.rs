# Generator of tests/data/vsharp_sti_small.bin: STI Suite 3.0 V_SHARP's output on the case
# tests/vsharp_sti_parity.rs builds (the same formulas). Original: the verified QSM-CI container
# ghcr.io/astewartau/qsm-ci/matlab-sti-vsharp@sha256:715c565758a5a64c6df8e73d34a0a62bf7afeb5d6d7feae5f748322cba2504d6
# (vsharp-sti v2, bit-identical to licensed MATLAB R2023b; QSMxT/QSM-CI#304).
#
#   python vsharp_sti_small.py inputs      # writes totalfield (float64), mask, params.json
#   for r in 12 6: cp inputs/* in$r/; echo "{\"smvsize\": $r}" > in$r/config.json
#     podman run --rm --network none --userns=keep-id -e QSMCI_INTERMEDIATES=1 \
#       -v $PWD/in$r:/input:ro -v $PWD/out$r:/output -v <QSM-CI>/algorithms/vsharp-sti:/algo:ro \
#       <image above> bash /algo/run.sh /input /output
#
# The .bin (little-endian) holds, for smvsize 12 then 6: the number of input-mask voxels (u32),
# then V_SHARP's unmasked output (out$r/intermediates/vsharp_unmasked.nii.gz, float64) at those
# voxels in column-major order, NaN where its eroded mask (newmask) is 0.
import sys, json, numpy as np, nibabel as nib
out = sys.argv[1]
n = (40, 44, 36)
i, j, k = np.meshgrid(*[np.arange(v, dtype=float) for v in n], indexing="ij")
mask = ((i - 19.5) / 15) ** 2 + ((j - 21.5) / 17) ** 2 + ((k - 17.5) / 13) ** 2 < 1
mask &= ~((i > 28) & (j < 14))
field = (0.05 * np.sin(0.21 * i + 0.3) * np.cos(0.17 * j) + 0.002 * (i - 20) * (k - 18) / 10
         + 0.1 * np.exp(-((i - 15) ** 2 + (j - 22) ** 2 + (k - 18) ** 2) / 8))
aff = np.eye(4)
nib.save(nib.Nifti1Image(field.astype(np.float64), aff), f"{out}/totalfield.nii.gz")
nib.save(nib.Nifti1Image(mask.astype(np.uint8), aff), f"{out}/mask.nii.gz")
json.dump({"B0": 3.0, "B0_dir": [0, 0, 1], "TE": [0.02], "voxel_size": [1.0, 1.0, 1.0]}, open(f"{out}/params.json", "w"))

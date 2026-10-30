#!/usr/bin/env python3
"""Writes tests/data/dl_invivo_small.bin, the in-vivo inputs of the deep-learning parity fixtures.

The deep-learning originals' fixtures (`tests/data/dl_*_qsmci_small.py`, astewartau/QSM.rs#317)
run on a 36x32x28 box of real data, so the networks see the values they were trained on. The box
is cut from the case QSMxT/QSM-CI#316 verified the originals on (`fidelity/dl/README.md`, "Case"):
the public travelling-head acquisition `prisma-bridge-run1` (Siemens Prisma, 2.89 T, 5 echoes,
1 mm), cut by `fidelity/dl/harness.py prepare` to 128x160x96, with its total field from
`romeo-qsmrs`, its local field from `vsharp-qsmrs` (both the pinned QSMxT v10.0.0 entries), and
χ and R2' from the `xqsm` and `r2primenet` originals. This script cuts the box at voxel
(88, 116, 56) of that case, where the brain's edge crosses it (62 % of the box is in the mask), so
the ports' masking, cropping and padding are exercised on a size no network divides.

    python3 scripts/dl_fixture_inputs.py <case dir> tests/data/dl_invivo_small.bin

Layout (little-endian, column-major over the 36x32x28 grid): the mask (u8), then as f32 the total
field (ppm), the local field (ppm), χ (ppm), R2' (s⁻¹), the five echoes' magnitudes, and the five
echoes' phases (rad). TE = 5, 11, 17, 23, 29 ms; B0 = 2.8946 T along z; 1 mm isotropic.
"""
import json, sys
from pathlib import Path
import numpy as np, nibabel as nib

START, SIZE = (88, 116, 56), (36, 32, 28)

case, out = Path(sys.argv[1]), sys.argv[2]
box = tuple(slice(s, s + n) for s, n in zip(START, SIZE))
p = json.loads((case / "params.json").read_text())
assert p["TE"] == [0.005, 0.011, 0.017, 0.023, 0.029] and p["B0"] == 2.8946 and p["B0_dir"] == [0.0, 0.0, 1.0]
vol = lambda n: np.asanyarray(nib.load(str(case / f"{n}.nii.gz")).dataobj)[box]
mask = vol("mask") > 0
with open(out, "wb") as f:
    f.write(mask.astype(np.uint8).tobytes(order="F"))
    for n in ("totalfield", "localfield", "chimap", "r2prime"):
        f.write(vol(n).astype("<f4").tobytes(order="F"))
    for n in ("magnitude", "phase"):
        a = vol(n)
        assert a.shape == SIZE + (5,)
        for e in range(5):
            f.write(a[..., e].astype("<f4").tobytes(order="F"))
print(f"{out}: {SIZE} at {START}, {mask.sum()} mask voxels ({100 * mask.mean():.0f} %)")

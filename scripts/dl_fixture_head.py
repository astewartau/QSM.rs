#!/usr/bin/env python3
"""Writes tests/data/dl_head_small.bin, the input of the HD-BET parity fixture.

HD-BET segments a whole head, so it cannot run on the small box of tests/data/dl_invivo_small.bin
(on a 64³ box its masks agree with the original's to Dice 0.09; QSMxT/QSM-CI#316). This is the
128x160x96 box of `prisma-bridge-run1` that QSM-CI#316 verified the originals on (see
scripts/dl_fixture_inputs.py), as #316's HD-BET pair ran it: the root-sum-of-squares of the five
echoes' magnitudes. To keep the fixture small it is averaged over 2x2x2 voxels (to 64x80x48 at 2
mm; HD-BET resamples to 1 mm itself, both the original and QSM.rs's port) and rounded to integers
(2 to 2787 after averaging), so it is stored exactly as u16.

    python3 scripts/dl_fixture_head.py <case dir> tests/data/dl_head_small.bin

<case dir> holds #316's magnitude.nii.gz (128x160x96x5). Layout: the 64x80x48 RSS magnitude
(u16, little-endian, column-major), 2 mm isotropic.
"""
import sys
from pathlib import Path
import numpy as np, nibabel as nib

case, out = Path(sys.argv[1]), sys.argv[2]
mag = np.asanyarray(nib.load(str(case / "magnitude.nii.gz")).dataobj).astype(np.float64)
assert mag.shape == (128, 160, 96, 5)
rss = np.sqrt((mag ** 2).sum(-1))
head = np.rint(rss.reshape(64, 2, 80, 2, 48, 2).mean((1, 3, 5)))
assert head.max() < 65536
open(out, "wb").write(head.astype("<u2").tobytes(order="F"))
print(f"{out}: {head.shape} at 2 mm, RSS in [{head.min():.0f}, {head.max():.0f}]")

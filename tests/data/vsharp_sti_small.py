# Generator of tests/data/vsharp_sti_small.bin: STI Suite 3.0 V_SHARP's output on the case
# tests/vsharp_sti_parity.rs builds (the same formulas). Original: the verified QSM-CI container
# ghcr.io/astewartau/qsm-ci/matlab-sti-vsharp@sha256:715c565758a5a64c6df8e73d34a0a62bf7afeb5d6d7feae5f748322cba2504d6
# (vsharp-sti v2, bit-identical to licensed MATLAB R2023b; QSMxT/QSM-CI#304), run with
# QSMCI_INTERMEDIATES=1 at smvsize 12 and 6.
#
#   python3 tests/data/vsharp_sti_small.py tests/data/vsharp_sti_small.bin
#
# Needs numpy, nibabel and podman or docker (CONTAINER_ENGINE overrides; the regenerate-fixtures
# workflow runs it in Actions). The .bin (little-endian) holds, for smvsize 12 then 6: the number
# of input-mask voxels (u32), then V_SHARP's unmasked output (intermediates/vsharp_unmasked.nii.gz,
# float64) at those voxels in column-major order, NaN where its eroded mask (newmask) is 0.
import json, os, shutil, struct, subprocess, sys, tempfile
import numpy as np, nibabel as nib

IMAGE = "ghcr.io/astewartau/qsm-ci/matlab-sti-vsharp@sha256:715c565758a5a64c6df8e73d34a0a62bf7afeb5d6d7feae5f748322cba2504d6"

n = (40, 44, 36)
i, j, k = np.meshgrid(*[np.arange(v, dtype=float) for v in n], indexing="ij")
mask = ((i - 19.5) / 15) ** 2 + ((j - 21.5) / 17) ** 2 + ((k - 17.5) / 13) ** 2 < 1
mask &= ~((i > 28) & (j < 14))
field = (0.05 * np.sin(0.21 * i + 0.3) * np.cos(0.17 * j) + 0.002 * (i - 20) * (k - 18) / 10
         + 0.1 * np.exp(-((i - 15) ** 2 + (j - 22) ** 2 + (k - 18) ** 2) / 8))
params = {"B0": 3.0, "B0_dir": [0, 0, 1], "TE": [0.02], "voxel_size": [1.0, 1.0, 1.0]}


def run(smvsize):
    """V_SHARP's unmasked output and eroded mask from the container at this smvsize."""
    engine = os.environ.get("CONTAINER_ENGINE") or shutil.which("podman") or "docker"
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        os.makedirs(inp), os.makedirs(out)
        os.chmod(out, 0o777)
        nib.save(nib.Nifti1Image(field.astype(np.float64), np.eye(4)), f"{inp}/totalfield.nii.gz")
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)), f"{inp}/mask.nii.gz")
        json.dump(params, open(f"{inp}/params.json", "w"))
        json.dump({"smvsize": smvsize}, open(f"{inp}/config.json", "w"))
        user = ["--userns=keep-id"] if os.path.basename(engine) == "podman" else ["--user", f"{os.getuid()}:{os.getgid()}"]
        subprocess.run([engine, "run", "--rm", "--network", "none", *user, "-e", "MCR_CACHE_ROOT=/tmp/mcr",
                        "-e", "QSMCI_INTERMEDIATES=1", "-v", f"{inp}:/input:ro", "-v", f"{out}:/output", IMAGE,
                        "/opt/qsm-ci/recon", "/input", "/output"], check=True)
        ld = lambda f: np.asarray(nib.load(f"{out}/intermediates/{f}.nii.gz").dataobj)
        return ld("vsharp_unmasked"), ld("newmask")


m = mask.flatten(order="F")
with open(sys.argv[1], "wb") as f:
    for r in (12, 6):
        local, newmask = run(r)
        v = np.where(newmask.flatten(order="F") != 0, local.flatten(order="F"), np.nan)[m]
        f.write(struct.pack("<I", int(m.sum())))
        f.write(v.astype("<f8").tobytes())

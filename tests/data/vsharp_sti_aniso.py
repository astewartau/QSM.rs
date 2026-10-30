# Generator of tests/data/vsharp_sti_aniso.bin: STI Suite 3.0 V_SHARP (smvsize 12) on four small
# phantoms with anisotropic voxels, for tests/vsharp_sti_aniso_parity.rs. Original: the verified
# QSM-CI container
#   ghcr.io/astewartau/qsm-ci/matlab-sti-vsharp@sha256:715c565758a5a64c6df8e73d34a0a62bf7afeb5d6d7feae5f748322cba2504d6
# (vsharp-sti v2, bit-identical to licensed MATLAB R2023b; QSMxT/QSM-CI#304), running QSM-CI's
# algorithms/vsharp-sti adapter as baked into the image.
#
#   python3 tests/data/vsharp_sti_aniso.py tests/data/vsharp_sti_aniso.bin
#
# Needs numpy, nibabel and podman or docker (CONTAINER_ENGINE overrides). The field is nonzero
# outside the mask on purpose: on a non-1 mm axis STI's resampling carries the phase just outside
# the mask (inside its even bounding box) onto its 1 mm working grid. The phantoms are built from
# + - * / only, so the Rust test rebuilds them bit for bit. Layout (little-endian, column-major),
# per case in CASES order: V_SHARP's eroded mask as bits over the grid (least significant bit
# first, ceil(n/8) bytes), then its unmasked output (vsharp_unmasked, float64) at those voxels.
import json, os, shutil, subprocess, sys, tempfile
import numpy as np, nibabel as nib

IMAGE = "ghcr.io/astewartau/qsm-ci/matlab-sti-vsharp@sha256:715c565758a5a64c6df8e73d34a0a62bf7afeb5d6d7feae5f748322cba2504d6"
CASES = [
    ((32, 32, 22), (1.0, 1.0, 1.5)),   # UK Biobank's voxel; an odd z extent (17 slices)
    ((30, 34, 22), (0.9, 0.9, 2.5)),
    ((28, 32, 24), (1.2, 0.8, 1.5)),   # odd working-grid size along x, mask ties at 0.5
    ((40, 36, 18), (0.7, 0.7, 3.0)),
]


def phantom(n, vs):
    i, j, k = np.meshgrid(*[np.arange(v, dtype=float) for v in n], indexing="ij")
    x = (i - n[0] / 2 + 0.3) * vs[0]
    y = (j - n[1] / 2 - 0.2) * vs[1]
    z = (k - n[2] / 2 + 0.1) * vs[2]
    r = min(n[0] * vs[0], n[1] * vs[1], n[2] * vs[2]) * 0.42
    mask = x * x / ((r * 1.05) * (r * 1.05)) + y * y / (r * r) + z * z / ((r * 0.95) * (r * 0.95)) < 1.0
    d = (x - 3.0) * (x - 3.0) + (y + 2.0) * (y + 2.0) + z * z
    field = 0.05 * x - 0.03 * y + 0.02 * z + 0.001 * x * y - 0.0008 * z * z + 0.3 / (1.0 + d / 20.0)
    return field, mask


def run(image, inp, out):
    engine = os.environ.get("CONTAINER_ENGINE") or shutil.which("podman") or "docker"
    user = ["--userns=keep-id"] if os.path.basename(engine) == "podman" else ["--user", f"{os.getuid()}:{os.getgid()}"]
    os.chmod(out, 0o777)
    subprocess.run([engine, "run", "--rm", "--network", "none", *user, "-e", "MCR_CACHE_ROOT=/tmp/mcr", "-e", "HOME=/tmp",
                    "-e", "QSMCI_INTERMEDIATES=1", "-v", f"{inp}:/input:ro", "-v", f"{out}:/output", image,
                    "/opt/qsm-ci/recon", "/input", "/output"], check=True, stdout=subprocess.DEVNULL)


with open(sys.argv[1], "wb") as f:
    for n, vs in CASES:
        field, mask = phantom(n, vs)
        with tempfile.TemporaryDirectory() as tmp:
            inp, out = os.path.join(tmp, "in"), os.path.join(tmp, "out")
            os.makedirs(inp), os.makedirs(out)
            aff = np.diag(list(vs) + [1.0])
            nib.save(nib.Nifti1Image(field, aff), os.path.join(inp, "totalfield.nii.gz"))
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), aff), os.path.join(inp, "mask.nii.gz"))
            json.dump({"B0": 3.0, "B0_dir": [0, 0, 1], "TE": [0.02], "voxel_size": list(vs)}, open(os.path.join(inp, "params.json"), "w"))
            run(IMAGE, inp, out)
            ld = lambda name: np.asarray(nib.load(os.path.join(out, "intermediates", name + ".nii.gz")).dataobj).ravel(order="F")
            em, local = ld("newmask") > 0.5, ld("vsharp_unmasked")
        f.write(np.packbits(em, bitorder="little").tobytes() + local[em].astype("<f8").tobytes())
        print(n, vs, "mask", int(mask.sum()), "eroded", int(em.sum()))

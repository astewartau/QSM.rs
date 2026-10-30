# Generator of tests/data/dl_hdbet_qsmci_small.bin: HD-BET's brain mask, as its verified QSM-CI entry
# makes it, of the head in tests/data/dl_head_small.bin, for tests/models_onnx/separation.rs.
#
# Original: ghcr.io/astewartau/qsm-ci/hd-bet@sha256:0a0e31375fbb6c5526eacdcd5d639d4eab52bc01f7cef2cba3bd74d0dc3b038d
# (`hd-bet-qsmci`: HD-BET 2.0.1, torch 2.13.0+cpu, checkpoint SHA-256 pinned; bit-identical run to
# run at 4 threads; QSMxT/QSM-CI#316). Its `extract.py` runs the HD-BET v2 CLI,
#   hd-bet -i mag.nii.gz -o brain.nii.gz -device cpu --disable_tta --save_bet_mask --no_bet_image
# that is, nnU-Net's sliding window (192x192x96 patches at 1 mm, step 0.5, Gaussian weights) with no
# mirroring, and writes the mask on the input grid; its signal-gated erosion is off by default. The
# entry's `run.sh` and `extract.py` live in QSM-CI's `algorithms/hd-bet-qsmci/`, fetched here from
# QSM-CI at QSMCI_COMMIT and checked against their SHA-256s; it runs as `fidelity/dl/harness.py`
# runs it (threads fixed at 4, no network).
#
# The input is the 2 mm RSS magnitude of dl_head_small.bin (scripts/dl_fixture_head.py), as a 3-D
# float32 NIfTI with a 2 mm diagonal affine (the entry combines echoes only for a 4-D input).
#
#   python3 tests/data/dl_hdbet_qsmci_small.py tests/data/dl_hdbet_qsmci_small.bin
#
# machine-dependent: PyTorch's CPU kernels round by instruction set (see
# tests/data/dl_dipole_qsmci_small.py), which can move a voxel at the mask's edge.
#
# Needs numpy, nibabel, network access (GitHub) and podman or docker (CONTAINER_ENGINE overrides; the
# regenerate-fixtures workflow runs it in Actions). Layout: the mask over the 64x80x48 grid,
# column-major, as bits (numpy.packbits, little-endian bit order).
import hashlib, os, shutil, subprocess, sys, tempfile, urllib.request
import numpy as np, nibabel as nib

QSMCI_COMMIT = "271a49bbc24d70c8e31e1642d79f6d5ac74a368d"  # QSMxT/QSM-CI#316
IMAGE = "ghcr.io/astewartau/qsm-ci/hd-bet@sha256:0a0e31375fbb6c5526eacdcd5d639d4eab52bc01f7cef2cba3bd74d0dc3b038d"
FILES = {"run.sh": "73e24d309943475f8d7a165a6b7589858ef08cb9e22d9a9fa8707ce10ced0db1",
         "extract.py": "fe1551391c7946aa5775c554db499be7437ee52d3106d68bbdd30703a69e2d26"}
DIMS, VOXEL = (64, 80, 48), 2.0

raw = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dl_head_small.bin"), "rb").read()
head = np.frombuffer(raw, "<u2").reshape(DIMS, order="F").astype(np.float32)
engine = os.environ.get("CONTAINER_ENGINE") or shutil.which("podman") or "docker"
with tempfile.TemporaryDirectory() as tmp:
    algo, inp, out = (os.path.join(tmp, d) for d in ("algo", "in", "out"))
    os.makedirs(algo), os.makedirs(inp), os.makedirs(out)
    os.chmod(out, 0o777)
    for name, sha in FILES.items():
        url = f"https://raw.githubusercontent.com/QSMxT/QSM-CI/{QSMCI_COMMIT}/algorithms/hd-bet-qsmci/{name}"
        data = urllib.request.urlopen(url, timeout=60).read()
        assert hashlib.sha256(data).hexdigest() == sha, f"{url}: SHA-256 mismatch"
        open(os.path.join(algo, name), "wb").write(data)
    nib.save(nib.Nifti1Image(head, np.diag([VOXEL, VOXEL, VOXEL, 1.0])), os.path.join(inp, "magnitude.nii.gz"))
    env = {k: "4" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    user = ["--userns=keep-id"] if os.path.basename(engine) == "podman" else ["--user", f"{os.getuid()}:{os.getgid()}"]
    subprocess.run([engine, "run", "--rm", "--network", "none", *user,
                    *[x for k, v in env.items() for x in ("-e", f"{k}={v}")],
                    "-v", f"{algo}:/algo:ro", "-v", f"{inp}:/input:ro", "-v", f"{out}:/output",
                    IMAGE, "bash", "/algo/run.sh"], check=True)
    mask = np.asarray(nib.load(os.path.join(out, "mask.nii.gz")).dataobj) > 0
assert mask.shape == DIMS
open(sys.argv[1], "wb").write(np.packbits(mask.flatten(order="F"), bitorder="little").tobytes())
print(f"HD-BET: {mask.sum()} mask voxels of {mask.size} ({100 * mask.mean():.1f} %)")

# Generator of tests/data/decompose_qsm_small.bin: the QSM-CI decompose-qsm container (the
# authors' DECOMPOSE-QSM utils and fit loop, compiled from licensed MATLAB R2023b and verified
# identical to it, QSMxT/QSM-CI#309) on a small synthetic case, for tests/decompose_qsm_parity.rs:
#   ghcr.io/astewartau/qsm-ci/decompose-qsm@sha256:a273a9b34f2863e4dac7e0c99f2c90e0dc220c56bf2e0f892c671a16be40112f
#
#   python3 tests/data/decompose_qsm_small.py tests/data/decompose_qsm_small.bin
#
# Needs numpy, nibabel and podman or docker (CONTAINER_ENGINE overrides; the regenerate-fixtures
# workflow runs it in Actions). It writes the inputs (magnitude (4-D), chimap, mask, params.json),
# runs the container's compiled recon on them, and packs its output.
#
# The container gives every echo the same QSM (chimap), so the per-echo QSM the port takes is
# chimap repeated. tests/decompose_qsm_parity.rs rebuilds these inputs with the same formulas.
# The .bin (little-endian f64, column-major, every voxel): out/chi-para.nii.gz, then
# out/chi-dia.nii.gz (|χ−|), as the container wrote them (float32, widened).
import json, os, shutil, subprocess, sys, tempfile
import numpy as np, nibabel as nib

IMAGE = "ghcr.io/astewartau/qsm-ci/decompose-qsm@sha256:a273a9b34f2863e4dac7e0c99f2c90e0dc220c56bf2e0f892c671a16be40112f"
tmp = tempfile.TemporaryDirectory()
out = os.path.join(tmp.name, "in")
os.makedirs(out)
n = (8, 8, 6)
TE = [0.004, 0.009, 0.014, 0.019, 0.024, 0.029]
B0 = 3.0
g = 42.58 * 2 * np.pi
a = 2 * np.pi * g * B0 / (9 * np.sqrt(3))
i, j, k = np.meshgrid(*[np.arange(v, dtype=float) for v in n], indexing="ij")
chip = 0.01 + 0.012 * i + 0.004 * k
chim = -(0.005 + 0.008 * j)
cp, cm, c0, r0 = 0.2 + 0.03 * k, 0.25, 0.55 - 0.03 * k, 18.0 + 1.5 * i
mag = np.stack([np.abs(cp * np.exp(-(a * chip + r0 + 1j * (2 / 3) * chip * g * B0) * t)
                       + cm * np.exp(-(-a * chim + r0 + 1j * (2 / 3) * chim * g * B0) * t)
                       + c0 * np.exp(-r0 * t)) for t in TE], axis=3)
chi = chip + chim
mask = np.ones(n, dtype=np.uint8); mask[0, 0, :] = 0
aff = np.eye(4)
nib.save(nib.Nifti1Image(mag.astype(np.float64), aff), f"{out}/magnitude.nii.gz")
nib.save(nib.Nifti1Image(chi.astype(np.float64), aff), f"{out}/chimap.nii.gz")
nib.save(nib.Nifti1Image(mask, aff), f"{out}/mask.nii.gz")
json.dump({"B0": B0, "B0_dir": [0, 0, 1], "TE": TE, "voxel_size": [1, 1, 1]}, open(f"{out}/params.json", "w"))

res = os.path.join(tmp.name, "out")
os.makedirs(res)
os.chmod(res, 0o777)
engine = os.environ.get("CONTAINER_ENGINE") or shutil.which("podman") or "docker"
user = ["--userns=keep-id"] if os.path.basename(engine) == "podman" else ["--user", f"{os.getuid()}:{os.getgid()}"]
subprocess.run([engine, "run", "--rm", "--network", "none", *user, "-e", "MCR_CACHE_ROOT=/tmp/mcr", "-e", "HOME=/tmp",
                "-v", f"{out}:/input:ro", "-v", f"{res}:/output", IMAGE, "/opt/qsm-ci/recon", "/input", "/output"],
               check=True, stdout=subprocess.DEVNULL)
ld = lambda f: np.asarray(nib.load(os.path.join(res, f)).dataobj, dtype=np.float64).ravel(order="F")
np.concatenate([ld("chi-para.nii.gz"), ld("chi-dia.nii.gz")]).astype("<f8").tofile(sys.argv[1])
tmp.cleanup()

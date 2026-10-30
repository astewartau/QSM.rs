# Generator of tests/data/mppca_dipy_small.bin: DIPY 1.12.1's `mppca` (Veraart et al. 2016) on a
# small noisy multi-volume phantom, for tests/mppca_dipy_parity.rs.
#
# Original: ghcr.io/astewartau/qsm-ci/py-originals@sha256:7654140b9e7ff00552590d945db9b8d52c526fd01b8e36d5e1ad524b02b510ed
# (py-originals v1: DIPY 1.12.1, hash-locked wheels, one thread; bit-reproducible, and identical
# outside the container; QSMxT/QSM-CI#302, fidelity/cpp-py), running QSM-CI's call script
# algorithms/_python-originals/mppca.py (fetched at the commit below, checked by hash):
# `mppca(data, mask, patch_radius=2, pca_method='eig')`, overlapping patches averaged.
#
#   python3 tests/data/mppca_dipy_small.py tests/data/mppca_dipy_small.bin
#
# Needs numpy, nibabel, network access (the call script) and podman or docker. The phantom: six
# echoes of a mono-exponential decay with smoothly varying S0 and R2* in an ellipsoid, plus seeded
# Gaussian noise (σ 4), stored as float32. Layout (little-endian, column-major, 14x14x10, volumes
# last): the mask (u8), the input (f32), then DIPY's output (f32).
import os, sys, tempfile
import numpy as np, nibabel as nib

IMAGE = "ghcr.io/astewartau/qsm-ci/py-originals@sha256:7654140b9e7ff00552590d945db9b8d52c526fd01b8e36d5e1ad524b02b510ed"
QSMCI_COMMIT = "b72e7409c6be35942d06def8fbec33fa0d27d848"


def qsmci_script(path, sha256):
    """A call script from QSMxT/QSM-CI at QSMCI_COMMIT, checked against its SHA-256."""
    import hashlib, urllib.request
    url = f"https://raw.githubusercontent.com/QSMxT/QSM-CI/{QSMCI_COMMIT}/{path}"
    data = urllib.request.urlopen(url).read()
    assert hashlib.sha256(data).hexdigest() == sha256, f"{path}: unexpected content"
    return data


def run(image, work, cmd, env=()):
    """Runs cmd in the image with `work` (a host directory) mounted at /work, as the invoking user."""
    import os, shutil, subprocess
    engine = os.environ.get("CONTAINER_ENGINE") or shutil.which("podman") or "docker"
    user = ["--userns=keep-id"] if os.path.basename(engine) == "podman" else ["--user", f"{os.getuid()}:{os.getgid()}"]
    os.chmod(work, 0o777)
    subprocess.run([engine, "run", "--rm", "--network", "none", *user, *sum((["-e", e] for e in env), []),
                    "-v", f"{work}:/work", "-w", "/work", image, *cmd], check=True)


n = (14, 14, 10)
i, j, k = np.meshgrid(*[np.arange(v, dtype=float) for v in n], indexing="ij")
c = [(v - 1) / 2 for v in n]
mask = ((i - c[0]) / 6.5) ** 2 + ((j - c[1]) / 6.5) ** 2 + ((k - c[2]) / 4.6) ** 2 < 1
s0 = 200 + 40 * np.sin(0.4 * i) * np.cos(0.3 * j) + 30 * (k - c[2]) / 5
r2s = 30 + 10 * np.cos(0.5 * i + 0.2 * k)
te = 0.004 + 0.005 * np.arange(6)
data = s0[..., None] * np.exp(-r2s[..., None] * te) + 4.0 * np.random.default_rng(11).standard_normal(n + (6,))
data = (data * mask[..., None]).astype(np.float32)

script = qsmci_script("algorithms/_python-originals/mppca.py",
                      "ec541d7363277ed7ca9b34d5ab0486418de1770ad43b8c4df72ed69c0b144582")
with tempfile.TemporaryDirectory() as w:
    open(f"{w}/mppca.py", "wb").write(script)
    nib.save(nib.Nifti1Image(data, np.eye(4)), f"{w}/in.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)), f"{w}/mask.nii.gz")
    run(IMAGE, w, ["python", "mppca.py", "in.nii.gz", "out.nii.gz", "--mask", "mask.nii.gz"])
    out = np.asarray(nib.load(f"{w}/out.nii.gz").dataobj, dtype=np.float64)
with open(sys.argv[1], "wb") as f:
    f.write(mask.ravel(order="F").astype(np.uint8).tobytes())
    f.write(data.ravel(order="F").astype("<f4").tobytes() + out.ravel(order="F").astype("<f4").tobytes())
print("mask", int(mask.sum()), "change rms in mask", np.sqrt(np.mean((out - data)[mask] ** 2)))

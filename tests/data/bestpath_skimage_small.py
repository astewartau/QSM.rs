# Generator of tests/data/bestpath_skimage_small.bin: scikit-image 0.26.0's 3-D best-path
# unwrapping (`unwrap_phase`, Abdul-Rahman et al. 2007) on small wrapped phantoms, for
# tests/bestpath_skimage_parity.rs.
#
# Original: ghcr.io/astewartau/qsm-ci/py-originals@sha256:7654140b9e7ff00552590d945db9b8d52c526fd01b8e36d5e1ad524b02b510ed
# (py-originals v1: scikit-image 0.26.0, hash-locked wheels, one thread; bit-reproducible, and
# identical outside the container; QSMxT/QSM-CI#302, fidelity/cpp-py), running QSM-CI's call
# script algorithms/_python-originals/bestpath.py (fetched at the commit below, checked by hash):
# `unwrap_phase(masked phase, wrap_around=False, rng=0)`, masked-out voxels 0.
#
#   python3 tests/data/bestpath_skimage_small.py tests/data/bestpath_skimage_small.bin
#
# Needs numpy, nibabel, network access (the call script) and podman or docker. Layout
# (little-endian), per case: the number of mask voxels m (u32), the mask (u8, column-major), the
# wrapped phase at the mask voxels (f64, rad), then the unwrapped phase there (f64, rad).
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


def case(n, noise):
    """A smooth phase spanning several wraps in an ellipsoid with a notch, plus seeded noise."""
    i, j, k = np.meshgrid(*[np.arange(v, dtype=float) for v in n], indexing="ij")
    c = [(v - 1) / 2 for v in n]
    mask = ((i - c[0]) / (0.45 * n[0])) ** 2 + ((j - c[1]) / (0.42 * n[1])) ** 2 + ((k - c[2]) / (0.4 * n[2])) ** 2 < 1
    mask &= ~((i > c[0]) & (np.abs(j - c[1]) < 2))
    true = (0.08 * (i - c[0]) ** 2 - 0.06 * (j - c[1]) ** 2 + 0.7 * (k - c[2])
            + 6 * np.exp(-((i - c[0] + 3) ** 2 + (j - c[1] - 2) ** 2 + (k - c[2]) ** 2) / 15))
    true += noise * np.random.default_rng(3).standard_normal(n)
    return np.angle(np.exp(1j * true)), mask


script = qsmci_script("algorithms/_python-originals/bestpath.py",
                      "7327da1ca864ba5e0fd05ab81cdc60bb52a49e08bfe4f1cc84a21499934f5d56")
with open(sys.argv[1], "wb") as f:
    for n, noise in [((28, 26, 20), 0.0), ((26, 28, 18), 0.3)]:
        phase, mask = case(n, noise)
        with tempfile.TemporaryDirectory() as w:
            open(f"{w}/bestpath.py", "wb").write(script)
            nib.save(nib.Nifti1Image(phase, np.eye(4)), f"{w}/phase.nii.gz")
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)), f"{w}/mask.nii.gz")
            run(IMAGE, w, ["python", "bestpath.py", "phase.nii.gz", "out.nii.gz", "--mask", "mask.nii.gz"])
            out = np.asarray(nib.load(f"{w}/out.nii.gz").dataobj, dtype=np.float64)
        m = mask.ravel(order="F")
        f.write(np.uint32(m.sum()).astype("<u4").tobytes() + m.astype(np.uint8).tobytes())
        f.write(phase.ravel(order="F")[m].astype("<f8").tobytes() + out.ravel(order="F")[m].astype("<f8").tobytes())
        print(n, noise, int(m.sum()), "voxels, unwrapped range", out[mask].min(), out[mask].max())

# Generator of tests/data/gibbs_dipy_small.bin: DIPY 1.12.1's `gibbs_removal` (Kellner et al. 2016)
# on a small ringing phantom, for tests/gibbs_dipy_parity.rs.
#
# Original: ghcr.io/astewartau/qsm-ci/py-originals@sha256:7654140b9e7ff00552590d945db9b8d52c526fd01b8e36d5e1ad524b02b510ed
# (py-originals v1: DIPY 1.12.1, hash-locked wheels, one process; bit-reproducible, and identical
# outside the container; QSMxT/QSM-CI#302, fidelity/cpp-py), running QSM-CI's call script
# algorithms/_python-originals/gibbs.py (fetched at the commit below, checked by hash): DIPY's
# defaults, 2-D unringing of each slice normal to z (`slice_axis=2`), `n_points=3`.
#
#   python3 tests/data/gibbs_dipy_small.py tests/data/gibbs_dipy_small.bin
#
# Needs numpy, nibabel, network access (the call script) and podman or docker. The phantom:
# ellipsoids of three intensities, ringing from truncating its k-space to the central half in x
# and y (and 3/4 in z), stored as float32. Layout (little-endian f32, column-major, 24x24x16): the
# input, then DIPY's output.
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


n = (24, 24, 16)
i, j, k = np.meshgrid(*[np.arange(v, dtype=float) for v in n], indexing="ij")
c = [(v - 1) / 2 for v in n]
img = 100.0 * ((((i - c[0]) / 9.5) ** 2 + ((j - c[1]) / 8.5) ** 2 + ((k - c[2]) / 6.5) ** 2) < 1)
img += 60.0 * ((((i - c[0] - 2) / 4) ** 2 + ((j - c[1] + 1) / 5) ** 2 + ((k - c[2]) / 4) ** 2) < 1)
img -= 40.0 * ((((i - c[0] + 4) / 2.5) ** 2 + ((j - c[1] - 3) / 2.5) ** 2 + ((k - c[2] - 1) / 3) ** 2) < 1)
# Truncate k-space (keep |f| below a quarter of the samples in x and y, 3/8 in z): Gibbs ringing.
F = np.fft.fftn(img)
keep = [np.abs(np.fft.fftfreq(v)) < q for v, q in zip(n, (0.25, 0.25, 0.375))]
F *= keep[0][:, None, None] & keep[1][None, :, None] & keep[2][None, None, :]
img = np.real(np.fft.ifftn(F)).astype(np.float32)

script = qsmci_script("algorithms/_python-originals/gibbs.py",
                      "2026de17bf80fa063d6d11931f9410cfb9ba1fb4ec0a16bfc775881998c1ef6d")
with tempfile.TemporaryDirectory() as w:
    open(f"{w}/gibbs.py", "wb").write(script)
    nib.save(nib.Nifti1Image(img, np.eye(4)), f"{w}/in.nii.gz")
    run(IMAGE, w, ["python", "gibbs.py", "in.nii.gz", "out.nii.gz"])
    out = np.asarray(nib.load(f"{w}/out.nii.gz").dataobj, dtype=np.float64)
with open(sys.argv[1], "wb") as f:
    f.write(img.ravel(order="F").astype("<f4").tobytes() + out.ravel(order="F").astype("<f4").tobytes())
print("input range", img.min(), img.max(), "change rms", np.sqrt(np.mean((out - img) ** 2)))

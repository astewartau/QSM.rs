# Generator of tests/data/bet_bet2_small.bin: FSL bet2 (Bostrix/FSL-BET2 d5acd7f) on a small
# synthetic head, for tests/bet_bet2_parity.rs.
#
# Original: QSM-CI's `bet2` image (algorithms/_bet2, QSMxT/QSM-CI#302: bet2 built from source at
# d5acd7f; bit-reproducible, and identical to a second build with another toolchain). bet2 is
# under the FSL licence, so that image is never published: this script builds it locally from
# QSM-CI's Dockerfile (fetched at the commit below, checked by hash) unless BET2_IMAGE names one,
# which works in Actions too. It runs bet2 as QSM-CI's bet.sh does (`bet2 in out -f F`).
#
#   python3 tests/data/bet_bet2_small.py tests/data/bet_bet2_small.bin
#
# Needs numpy, nibabel, network access and podman or docker. The head is built from comparisons
# only, so tests/bet_bet2_parity.rs rebuilds it exactly; the .bin holds bet2's masks, for -f 0.5
# (bet2's default) then -f 0.2 (QSMART's), each as bits over the grid (column-major, least
# significant bit first, ceil(n/8) bytes).
import os, shutil, subprocess, sys, tempfile
import numpy as np, nibabel as nib

DOCKERFILE_SHA256 = "7ae496e78718fb9710b545cb6fce0cd37627b4df12583e935c6a8e6409305e0d"
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


def head():
    """56x64x48 at 1.5 mm: brain (white 110, grey 80 shell), CSF 20, skull 5, scalp 70."""
    n = (56, 64, 48)
    i, j, k = np.meshgrid(*[np.arange(v, dtype=float) for v in n], indexing="ij")
    x, y, z = i - 27.5, j - 31.5, k - 23.5
    e = lambda a, b, c: (x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2
    img = np.zeros(n, np.float32)
    img[e(26, 30, 22) < 1] = 70
    img[e(23, 27, 19) < 1] = 5
    img[e(21, 25, 17) < 1] = 20
    img[e(19.5, 23.5, 15.5) < 1] = 80
    img[e(16, 20, 12.5) < 1] = 110
    return img


img = head()
engine = os.environ.get("CONTAINER_ENGINE") or shutil.which("podman") or "docker"
image = os.environ.get("BET2_IMAGE")
if not image:
    image = "localhost/qsm-ci/bet2:d5acd7f"
    with tempfile.TemporaryDirectory() as ctx:
        open(f"{ctx}/Dockerfile", "wb").write(qsmci_script("algorithms/_bet2/Dockerfile", DOCKERFILE_SHA256))
        subprocess.run([engine, "build", "-t", image, ctx], check=True)
with open(sys.argv[1], "wb") as f:
    for frac in ("0.5", "0.2"):
        with tempfile.TemporaryDirectory() as w:
            nib.save(nib.Nifti1Image(img, np.diag([1.5, 1.5, 1.5, 1.0])), f"{w}/in.nii.gz")
            run(image, w, ["bet2", "in.nii.gz", "out", "-f", frac], ["FSLOUTPUTTYPE=NIFTI_GZ"])
            mask = np.asarray(nib.load(f"{w}/out_mask.nii.gz").dataobj) > 0
        f.write(np.packbits(mask.ravel(order="F"), bitorder="little").tobytes())
        print(f"-f {frac}: {int(mask.sum())} voxels")

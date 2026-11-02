# Generator of tests/data/prelude_fsl_small.bin: FSL's `prelude` (fsl-fugue 2603.0, Jenkinson's
# own implementation) on small noisy wrapped phantoms, for tests/prelude_fsl_parity.rs.
#
# Original: QSM-CI's `fsl-prelude` image (algorithms/_fsl-prelude, QSMxT/QSM-CI#302/#376: FSL's
# own binaries from FSL's conda channel, every package pinned by MD5). FSL's licence keeps that
# image unpublished, so this script builds it locally from QSM-CI's Dockerfile and lock file
# (fetched at the commit below, checked by hash) unless FSL_PRELUDE_IMAGE names one, which works in
# Actions too. FSL is run as a black box; its source is not read (THIRD_PARTY_NOTICES.md).
#
#   python3 tests/data/prelude_fsl_small.py tests/data/prelude_fsl_small.bin
#
# Needs numpy, nibabel, network access and podman or docker. Two noisy wrapped phantoms, the
# first 72 voxels wide (over the 64 above which FSL's default labels its regions slice by slice)
# and the second 64 (labelled in full 3D by default). Each is run through `prelude` with the mask
# given (`-m`) and a constant magnitude (`-a`, which FSL needs but uses only to make a mask when
# none is given), with the options in RUNS: the defaults (8 partitions), `--labelslices`, `-f` and
# `-n 6`. Layout (little-endian), per case: the mask (u8, column-major), the wrapped phase at the
# mask voxels (f32, rad, exactly what FSL read), then per run the wrap count at the mask voxels
# (i8: round((unwrapped - wrapped) / 2π)).
import os, shutil, subprocess, sys, tempfile
import numpy as np, nibabel as nib

QSMCI_COMMIT = "fd74c5efceeacb3702af44010fe7656e191a0ca6"
FILES = {
    "algorithms/_fsl-prelude/Dockerfile": "01970495b4c687ef4422428550a706c94864fb69ed9a9d3226db58afc4a1b9f0",
    "algorithms/_fsl-prelude/fsl-explicit.lock": "1c73557e40b0a0e81da6e29855f1ace26e4763e0cc65136cbbd1bbac762ec8a4",
}
CASES = [(72, 24, 16), (64, 24, 16)]
RUNS = [[], ["--labelslices"], ["-f"], ["-n", "6"]]


def qsmci_file(path, sha256):
    """A file from QSMxT/QSM-CI at QSMCI_COMMIT, checked against its SHA-256."""
    import hashlib, urllib.request
    url = f"https://raw.githubusercontent.com/QSMxT/QSM-CI/{QSMCI_COMMIT}/{path}"
    data = urllib.request.urlopen(url).read()
    assert hashlib.sha256(data).hexdigest() == sha256, f"{path}: unexpected content"
    return data


def run(image, work, cmd, env=()):
    """Runs cmd in the image with `work` (a host directory) mounted at /work, as the invoking user."""
    engine = os.environ.get("CONTAINER_ENGINE") or shutil.which("podman") or "docker"
    user = ["--userns=keep-id"] if os.path.basename(engine) == "podman" else ["--user", f"{os.getuid()}:{os.getgid()}"]
    os.chmod(work, 0o777)
    subprocess.run([engine, "run", "--rm", "--network", "none", *user, *sum((["-e", e] for e in env), []),
                    "-v", f"{work}:/work", "-w", "/work", image, *cmd], check=True)


def phantom(dims):
    """An ellipsoid filling the grid; a smooth field of about a turn per centimetre with two
    point-like sources, plus Gaussian phase noise (σ 1.2 rad) so that some initial regions
    straddle wraps and the labellings disagree."""
    i, j, k = np.meshgrid(*[np.arange(v, dtype=float) for v in dims], indexing="ij")
    c = [(v - 1) / 2 for v in dims]
    x, y, z = i - c[0], j - c[1], k - c[2]
    mask = (x / (c[0] + 0.5)) ** 2 + (y / (c[1] + 0.5)) ** 2 + (z / (c[2] + 0.5)) ** 2 < 1
    field = 0.6 * x + 0.5 * y + 0.7 * z + 0.01 * (x * x - y * y) + 0.03 * x * z
    for (cx, cy, cz, s) in [(12, -3, 1, 40.0), (-15, 4, -2, -30.0)]:
        r2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 + 2.0
        field += s * (3 * (z - cz) ** 2 / r2 - 1) / r2 ** 1.5
    noise = np.random.default_rng(0).normal(0, 1.2, dims)
    wrapped = np.angle(np.exp(1j * (field + noise))).astype(np.float32)
    return mask, np.where(mask, wrapped, np.float32(0))


engine = os.environ.get("CONTAINER_ENGINE") or shutil.which("podman") or "docker"
image = os.environ.get("FSL_PRELUDE_IMAGE")
if not image:
    image = "localhost/qsm-ci/fsl-prelude:2603.0"
    with tempfile.TemporaryDirectory() as ctx:
        for path, sha in FILES.items():
            open(f"{ctx}/{os.path.basename(path)}", "wb").write(qsmci_file(path, sha))
        subprocess.run([engine, "build", "-t", image, ctx], check=True)
affine = np.diag([1.0, 1.0, 1.0, 1.0])
with open(sys.argv[1], "wb") as f:
    for dims in CASES:
        mask, phase = phantom(dims)
        inside = mask.ravel(order="F")
        f.write(inside.astype(np.uint8).tobytes())
        f.write(phase.ravel(order="F")[inside].astype("<f4").tobytes())
        for opts in RUNS:
            with tempfile.TemporaryDirectory() as w:
                nib.save(nib.Nifti1Image(phase, affine), f"{w}/phase.nii.gz")
                nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), f"{w}/mask.nii.gz")
                nib.save(nib.Nifti1Image(mask.astype(np.float32) * 100, affine), f"{w}/abs.nii.gz")
                run(image, w, ["prelude", "-p", "phase.nii.gz", "-a", "abs.nii.gz", "-m", "mask.nii.gz",
                               "-o", "out", *opts], ["FSLOUTPUTTYPE=NIFTI_GZ"])
                out = np.asarray(nib.load(f"{w}/out.nii.gz").dataobj, dtype=np.float64)
            d = (out - phase.astype(np.float64)).ravel(order="F")[inside]
            turns = np.round(d / (2 * np.pi))
            assert np.abs(d - 2 * np.pi * turns).max() < 1e-4, "FSL's output is not the input plus whole turns"
            f.write(turns.astype("<i1").tobytes())
            print(f"{dims} prelude {' '.join(opts) or '(defaults)'}: {int(inside.sum())} voxels, "
                  f"wrap counts {int(turns.min())}..{int(turns.max())}")

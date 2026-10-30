# Generator of tests/data/wavesep_small.bin: WaveSep (ZhenghanFang/WaveSep f3a192e, unmodified) on a
# small synthetic χ-separation case, for tests/wavesep_parity.rs.
#
# Original: ghcr.io/astewartau/qsm-ci/wavesep@sha256:c96094a98049ccfb9d660a2463ef772502b30dcb412095e45641d67edeb8791a
# (wavesep v1; bit-reproducible, and identical to a rebuild from its Dockerfile; QSMxT/QSM-CI#302),
# running QSM-CI's submission script algorithms/wavesep/recon.py (fetched at the commit below,
# checked by hash) at its defaults (WaveSep's: db4, α 0.2, λ 0.02, 100 iterations; Dr 137 Hz/ppm),
# one BLAS thread as its run.sh sets. recon.py zero-pads each axis to a multiple of 2^L and writes
# χ+ and |χ−| as float32.
#
#   python3 tests/data/wavesep_small.py tests/data/wavesep_small.bin
#
# Needs numpy, nibabel, network access (the script) and podman or docker. Layout (little-endian,
# column-major, 24x24x24): the number of mask voxels m (u32), the mask (u8), then at the mask
# voxels χ_total (f32, ppm), R2' (f32, Hz), WaveSep's χ+ and |χ−| (f32, ppm).
import os, sys, tempfile
import numpy as np, nibabel as nib

IMAGE = "ghcr.io/astewartau/qsm-ci/wavesep@sha256:c96094a98049ccfb9d660a2463ef772502b30dcb412095e45641d67edeb8791a"
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


n = (24, 24, 24)
i, j, k = np.meshgrid(*[np.arange(v, dtype=float) for v in n], indexing="ij")
c = [(v - 1) / 2 for v in n]
mask = ((i - c[0]) / 10.5) ** 2 + ((j - c[1]) / 10) ** 2 + ((k - c[2]) / 9.5) ** 2 < 1
blob = lambda a, b, d, r: np.exp(-((i - c[0] - a) ** 2 + (j - c[1] - b) ** 2 + (k - c[2] - d) ** 2) / r)
para = 0.04 + 0.08 * blob(3, -2, 1, 6) + 0.03 * (np.abs(i - c[0] + 4) < 2)
dia = 0.03 + 0.06 * blob(-4, 3, -1, 8) + 0.02 * np.cos(0.3 * j) ** 2
chi = ((para - dia) * mask).astype(np.float32)
r2p = (137.0 * (para + dia) * mask).astype(np.float32)

script = qsmci_script("algorithms/wavesep/recon.py",
                      "677b6198fbb9a7a0edb417c7aba9d55327b52021fce6ac573491aba9eb71598b")
with tempfile.TemporaryDirectory() as w:
    os.makedirs(f"{w}/in"), os.makedirs(f"{w}/out")
    open(f"{w}/recon.py", "wb").write(script)
    for name, a in [("chimap", chi), ("r2prime", r2p), ("mask", mask.astype(np.uint8))]:
        nib.save(nib.Nifti1Image(a, np.eye(4)), f"{w}/in/{name}.nii.gz")
    open(f"{w}/in/params.json", "w").write('{"B0": 3.0, "B0_dir": [0, 0, 1], "TE": [0.02], "voxel_size": [1, 1, 1]}')
    run(IMAGE, w, ["python", "recon.py", "in", "out"],
        ["OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "MPLCONFIGDIR=/tmp/mpl"])
    ld = lambda f: np.asarray(nib.load(f"{w}/out/{f}.nii.gz").dataobj)
    xp, xn = ld("chi-para"), ld("chi-dia")
m = mask.ravel(order="F")
with open(sys.argv[1], "wb") as f:
    f.write(np.uint32(m.sum()).astype("<u4").tobytes() + m.astype(np.uint8).tobytes())
    for a in (chi, r2p, xp, xn):
        f.write(a.ravel(order="F")[m].astype("<f4").tobytes())
print(int(m.sum()), "mask voxels; χ+", xp[mask].min(), xp[mask].max(), "|χ−|", xn[mask].min(), xn[mask].max())

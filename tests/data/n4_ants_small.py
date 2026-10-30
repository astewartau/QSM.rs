# Generator of tests/data/n4_ants_small.bin: ANTs 2.4.3's N4BiasFieldCorrection on a small
# synthetic case, for tests/n4_ants_parity.rs.
#
# Original: ghcr.io/astewartau/qsm-ci/ants-n4@sha256:621644e172c4872f1d181c9c18dc8717c40aece19de8d28c31fce93028017c49
# (ants-n4 v1: the official ANTs 2.4.3 release binary, ITK 4 threads; bit-reproducible, and
# identical to the same binary outside the container; QSMxT/QSM-CI#302, fidelity/cpp-py).
#
#   python3 tests/data/n4_ants_small.py tests/data/n4_ants_small.bin
#
# Needs numpy, nibabel and podman or docker (CONTAINER_ENGINE overrides; the regenerate-fixtures
# workflow runs it in Actions). The case: three intensity classes (60, 100, 160) in a sphere,
# Gaussian noise (σ = 3, seeded) inside it, times the bias exp(0.35x − 0.25y + 0.2z²), 30x32x26 at
# 1x1x1.5 mm. Layout (little-endian f32, column-major): the input; the bias field of
# `N4BiasFieldCorrection -i in.nii -o [n4.nii,bias.nii]` (all defaults); the corrected image of
# the same with `-x mask.nii`, the mask `input > 30`.
import os, sys, tempfile
import numpy as np, nibabel as nib

IMAGE = "ghcr.io/astewartau/qsm-ci/ants-n4@sha256:621644e172c4872f1d181c9c18dc8717c40aece19de8d28c31fce93028017c49"

def run(image, work, cmd, env=()):
    """Runs cmd in the image with `work` (a host directory) mounted at /work, as the invoking user."""
    import os, shutil, subprocess
    engine = os.environ.get("CONTAINER_ENGINE") or shutil.which("podman") or "docker"
    user = ["--userns=keep-id"] if os.path.basename(engine) == "podman" else ["--user", f"{os.getuid()}:{os.getgid()}"]
    os.chmod(work, 0o777)
    subprocess.run([engine, "run", "--rm", "--network", "none", *user, *sum((["-e", e] for e in env), []),
                    "-v", f"{work}:/work", "-w", "/work", image, *cmd], check=True)


nx, ny, nz = 30, 32, 26
rng = np.random.default_rng(7)
i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
x, y, z = (i - 14.5) / 15, (j - 15.5) / 16, (k - 12.5) / 13
r = np.sqrt(x**2 + y**2 + z**2)
tissue = np.where(r < 0.85, np.where(np.abs(x + 0.3 * y) < 0.35, 160.0, 100.0), 0.0)
tissue[(r < 0.85) & (r > 0.7)] = 60.0
noise = rng.normal(0, 3.0, tissue.shape) * (tissue > 0)
bias = np.exp(0.35 * x - 0.25 * y + 0.2 * z * z)
img = ((tissue + noise) * bias).astype(np.float32)
img[img < 0] = 0
aff = np.diag([1.0, 1.0, 1.5, 1.0])
with tempfile.TemporaryDirectory() as w:
    nib.save(nib.Nifti1Image(img, aff), f"{w}/in.nii")
    nib.save(nib.Nifti1Image((img > 30).astype(np.float32), aff), f"{w}/mask.nii")
    n4 = ["N4BiasFieldCorrection", "-i", "in.nii"]
    env = ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4"]
    run(IMAGE, w, n4 + ["-o", "[n4.nii,n4_bias.nii]"], env)
    run(IMAGE, w, n4 + ["-x", "mask.nii", "-o", "[n4m.nii,n4m_bias.nii]"], env)
    ld = lambda f: np.asarray(nib.load(f"{w}/{f}").dataobj, dtype=np.float32).ravel(order="F")
    out = np.concatenate([img.ravel(order="F"), ld("n4_bias.nii"), ld("n4m.nii")]).astype("<f4")
out.tofile(sys.argv[1])

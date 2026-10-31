# Generator of tests/data/gibbs_mrtrix3d_small.bin: MRtrix3's `mrdegibbs -dimensionality 3`, the
# 3-D subvoxel-shift unringing of Bautista et al. (Proc. ISMRM 2021, 29, 3535; the authors' code),
# on two small phantoms ringing along all three axes, for tests/gibbs_mrtrix3d_parity.rs.
#
# Original: MRtrix3 `dev` at the commit below (the 3-D mode is not in a 3.0.x release, and there is
# no image of `dev`), built from source in ubuntu:24.04 (pinned by digest) with only `mrdegibbs`
# (no GUI, no GPU), run at its defaults (`nshifts` 20, `minW` 1, `maxW` 3), writing float64.
#
#   python3 tests/data/gibbs_mrtrix3d_small.py tests/data/gibbs_mrtrix3d_small.bin
#
# machine-dependent: FFTW's plans round differently on different CPUs (~1e-16 relative)
#
# Needs numpy, nibabel, network access (apt and the MRtrix3 sources, inside the container) and
# podman or docker; the build takes ~10 min. The phantoms: case 1 is the 24x24x16 phantom of
# tests/data/gibbs_dipy_small.py (its k-space cut to the central half in x and y and 3/4 in z), case
# 2 a 21x26x15 one (odd axes; cut to 0.6, 0.5 and 0.7). Layout (little-endian f64, column-major):
# for each case its input, then mrdegibbs's output.
import os, sys, tempfile
import numpy as np, nibabel as nib

IMAGE = "docker.io/library/ubuntu@sha256:78d4744ead1f5b19b04d0df558ab182a5300903900755e515b2ca4d10eb2044d"
MRTRIX_COMMIT = "9b0a2f1f8778ed7f72d4fe0148738380a6162bc9"
BUILD = f"""set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null
apt-get install -y -qq --no-install-recommends ca-certificates git cmake ninja-build g++ libeigen3-dev libfftw3-dev zlib1g-dev python3 >/dev/null
git init -q /src && cd /src && git fetch -q --depth 1 https://github.com/MRtrix3/mrtrix3.git {MRTRIX_COMMIT} && git checkout -q FETCH_HEAD
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DMRTRIX_BUILD_GUI=OFF -DMRTRIX_ENABLE_GPU=OFF -DMRTRIX_USE_PCH=OFF -DMRTRIX_IGNORE_VERSION_MISMATCH=ON >/dev/null
ninja -C build mrdegibbs >/dev/null
for c in 1 2; do build/bin/mrdegibbs -quiet -nthreads 1 -dimensionality 3 -datatype float64 /work/in$c.nii /work/out$c.nii; done
"""


def run(image, work, cmd):
    """Runs cmd in the image with `work` (a host directory) mounted at /work."""
    import shutil, subprocess
    engine = os.environ.get("CONTAINER_ENGINE") or shutil.which("podman") or "docker"
    os.chmod(work, 0o777)
    subprocess.run([engine, "run", "--rm", "-v", f"{work}:/work", image, *cmd], check=True)


def phantom(n, cut, shift):
    i, j, k = np.meshgrid(*[np.arange(v, dtype=float) for v in n], indexing="ij")
    c = [(v - 1) / 2 + s for v, s in zip(n, shift)]
    img = 100.0 * ((((i - c[0]) / (0.4 * n[0])) ** 2 + ((j - c[1]) / (0.35 * n[1])) ** 2 + ((k - c[2]) / (0.4 * n[2])) ** 2) < 1)
    img += 60.0 * ((((i - c[0] - 2) / 4) ** 2 + ((j - c[1] + 1) / 5) ** 2 + ((k - c[2]) / 4) ** 2) < 1)
    img -= 40.0 * ((((i - c[0] + 4) / 2.5) ** 2 + ((j - c[1] - 3) / 2.5) ** 2 + ((k - c[2] - 1) / 3) ** 2) < 1)
    F = np.fft.fftn(img)
    keep = [np.abs(np.fft.fftfreq(v)) < q / 2 for v, q in zip(n, cut)]
    F *= keep[0][:, None, None] & keep[1][None, :, None] & keep[2][None, None, :]
    return np.real(np.fft.ifftn(F))


cases = [phantom((24, 24, 16), (0.5, 0.5, 0.75), (0, 0, 0)), phantom((21, 26, 15), (0.6, 0.5, 0.7), (0.3, -0.4, 0.2))]
out = []
with tempfile.TemporaryDirectory() as w:
    for c, img in enumerate(cases, 1):
        nib.save(nib.Nifti1Image(img, np.eye(4)), f"{w}/in{c}.nii")
    run(IMAGE, w, ["bash", "-c", BUILD])
    for c, img in enumerate(cases, 1):
        res = np.asarray(nib.load(f"{w}/out{c}.nii").dataobj, dtype=np.float64)
        out.append(img.ravel(order="F").astype("<f8").tobytes() + res.ravel(order="F").astype("<f8").tobytes())
        print(f"case {c} {img.shape}: change rms {np.sqrt(np.mean((res - img) ** 2)):.3f}, min {res.min():.3f}")
with open(sys.argv[1], "wb") as f:
    f.write(b"".join(out))

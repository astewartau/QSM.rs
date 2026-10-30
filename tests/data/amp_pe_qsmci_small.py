# Generator of tests/data/amp_pe_qsmci_small.bin: AMP-PE (Huang et al., MRM 2023; EmoryCN2L/QSM_AMP_PE),
# as the verified QSM-CI container runs it, on two small phantoms, for tests/amppe_qsmci.rs.
#
# Original: ghcr.io/astewartau/qsm-ci/amp-pe@sha256:0e1f44758045229f3297926a573e5664e6de5d2a895dca2a30d2f2339d507fc1
# (amp-pe v2: bit-identical to licensed MATLAB R2023b with the real Wavelet Toolbox; QSMxT/QSM-CI#309).
# Its recon.m is the dipole stage: it turns the local field (ppm) into a simulated single-echo
# phase at simulated_TE with γ = 42.58 MHz/T, zero-pads to a multiple of 2^nlevel, weights the data
# by the magnitude normalised to unit mean in the mask (uniform without one), builds the wavelet
# morphology mask from the magnitude (all ones without one), seeds with chiL2 (β 2e-2), then runs
# the two AMP-PE stages (Gaussian, then the two-component mixture), each for max_linearization_ite
# linearisations (GAMP damping 0.01, κ 0.1, 5 + 5 inner iterations, threshold 1e-6).
#
#   python3 tests/data/amp_pe_qsmci_small.py tests/data/amp_pe_qsmci_small.bin
#
# Needs numpy, nibabel and podman or docker (CONTAINER_ENGINE overrides; the regenerate-fixtures
# workflow runs it in Actions). Layout (little-endian), per case in CASES order: the mask over the
# grid (u8, column-major), the local field at the mask voxels (f64, ppm; zero outside, as the
# container is given it), the magnitude over the grid (f32) if the case has one, then the
# container's χ at the mask voxels (f32, ppm).
import json, os, shutil, subprocess, sys, tempfile
import numpy as np, nibabel as nib

IMAGE = "ghcr.io/astewartau/qsm-ci/amp-pe@sha256:0e1f44758045229f3297926a573e5664e6de5d2a895dca2a30d2f2339d507fc1"

_H = (0.1, -0.4, 0.9)
_H = tuple(v / sum(w * w for w in _H) ** 0.5 for v in _H)
# (grid, voxel size mm, B0 direction, magnitude?, config.json)
# The first is padded (30x28x22 -> 32x32x24 at nlevel 3) and weighted by a magnitude; the second
# uses db2 at nlevel 2 with an oblique B0 and no magnitude. 6 linearisations per stage (the default
# 25 only takes longer).
CASES = [
    ((30, 28, 22), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0), True,
     {"wave_idx": 1, "nlevel": 3, "max_linearization_ite": 6}),
    ((32, 32, 24), (1.0, 1.0, 1.2), _H, False,
     {"wave_idx": 2, "nlevel": 2, "max_linearization_ite": 6}),
]


def phantom(n, vs, h, with_mag):
    """Spheres of four susceptibilities in an ellipsoid mask, forward modelled with the dipole kernel
    on the grid; returns (local field ppm, mask, magnitude or None)."""
    x = [((np.arange(n[a]) + 1) - n[a] / 2 - 0.5) * vs[a] for a in range(3)]
    X, Y, Z = np.meshgrid(*x, indexing="ij")
    r = 0.42 * min(n[a] * vs[a] for a in range(3))
    mask = (X / r) ** 2 + (Y / (0.9 * r)) ** 2 + (Z / (0.8 * r)) ** 2 < 1
    chi = np.zeros(n)
    for c, v, rad in zip([(0.3, 0.2, 0), (-0.3, 0.1, 0.1), (0, -0.35, -0.2), (0.1, 0.1, 0.3)],
                         [0.1, -0.05, 0.2, 0.5], [0.15, 0.2, 0.12, 0.08]):
        chi[(X - c[0] * r) ** 2 + (Y - c[1] * r) ** 2 + (Z - c[2] * r) ** 2 < (rad * r) ** 2] = v
    chi *= mask
    k = np.meshgrid(*[np.fft.fftfreq(n[a], vs[a]) for a in range(3)], indexing="ij")
    k2 = k[0] ** 2 + k[1] ** 2 + k[2] ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        d = 1 / 3 - (k[0] * h[0] + k[1] * h[1] + k[2] * h[2]) ** 2 / k2
    d[0, 0, 0] = 0
    field = np.real(np.fft.ifftn(d * np.fft.fftn(chi))) * mask
    mag = None
    if with_mag:
        mag = ((900 - 1200 * chi + 40 * X / r) * mask + 30).astype(np.float32)
    return field, mask, mag


def run(inputs, params, config):
    engine = os.environ.get("CONTAINER_ENGINE") or shutil.which("podman") or "docker"
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        os.makedirs(inp), os.makedirs(out)
        os.chmod(out, 0o777)
        for name, arr in inputs.items():
            nib.save(nib.Nifti1Image(arr, np.eye(4)), os.path.join(inp, name + ".nii.gz"))
        json.dump(params, open(os.path.join(inp, "params.json"), "w"))
        json.dump(config, open(os.path.join(inp, "config.json"), "w"))
        user = ["--userns=keep-id"] if os.path.basename(engine) == "podman" else ["--user", f"{os.getuid()}:{os.getgid()}"]
        subprocess.run([engine, "run", "--rm", "--network", "none", *user, "-e", "MCR_CACHE_ROOT=/tmp/mcr",
                        "-v", f"{inp}:/input:ro", "-v", f"{out}:/output", IMAGE,
                        "/opt/qsm-ci/recon", "/input", "/output"], check=True)
        return np.asarray(nib.load(os.path.join(out, "chimap.nii.gz")).dataobj)


with open(sys.argv[1], "wb") as f:
    for n, vs, h, with_mag, config in CASES:
        field, mask, mag = phantom(n, vs, h, with_mag)
        inputs = {"localfield": field, "mask": mask.astype(np.uint8)}
        if mag is not None:
            inputs["magnitude"] = mag
        params = {"B0": 3.0, "B0_dir": list(h), "TE": [0.02], "voxel_size": list(vs)}
        chi = run(inputs, params, config)
        assert chi.dtype == np.float32 and chi.shape == n
        m = mask.flatten(order="F")
        f.write(m.astype(np.uint8).tobytes())
        f.write(field.flatten(order="F")[m].astype("<f8").tobytes())
        if mag is not None:
            f.write(mag.astype("<f4").tobytes(order="F"))
        f.write(chi.flatten(order="F")[m].astype("<f4").tobytes())
        print(f"{n} {config}: {m.sum()} mask voxels, χ in [{chi.min():.4f}, {chi.max():.4f}] ppm")

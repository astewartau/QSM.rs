# Generator of tests/data/ilsqr_sti_qsmci_small.bin: STI Suite 3.0's QSM_iLSQR, as the verified
# QSM-CI container runs it, on two small phantoms, for tests/ilsqr_sti_parity.rs.
#
# Original: ghcr.io/astewartau/qsm-ci/matlab-sti-ilsqr@sha256:fa2660014473dbff51a6ddd6aba0840074349ce6d97be9efd0cb8afe6c1c597c
# (ilsqr-sti v2, bit-identical to licensed MATLAB R2023b; QSMxT/QSM-CI#304). Its recon.m turns the
# local field (ppm) into phase with γ = 42.576 MHz/T and calls
#   QSM_iLSQR(phase, mask, 'TE', TE, 'B0', B0, 'H', H, 'padsize', [p p p], 'voxelsize', vs)
# without 'niter' (so 100), then writes χ (ppm) as float32. STI computes χ in single precision, so
# the float32 output is STI's χ exactly.
#
# machine-dependent: STI's single-precision LSQR rounds differently on different CPUs. The
# committed .bin was made on an Intel i5-11500 (12 threads); a GitHub-hosted runner gives χ 3.6e-4
# and 1.2e-4 relative L2 away (max 3.7e-5 ppm), the same on every run on either machine (MKL_CBWR
# does not change it). That is far inside tests/ilsqr_sti_parity.rs's tolerance, so the
# regenerate-fixtures workflow reports a mismatch here as a warning, not a failure.
#
#   python3 tests/data/ilsqr_sti_qsmci_small.py tests/data/ilsqr_sti_qsmci_small.bin
#
# Needs numpy, nibabel and podman or docker (CONTAINER_ENGINE overrides; the regenerate-fixtures
# workflow runs it in Actions). Layout (little-endian), per case in CASES order: the number of
# mask voxels m (u32), the mask over the grid (u8, column-major), the local field at the mask
# voxels (f64, ppm; zero outside the mask, as the container is given it), then the container's χ
# at the mask voxels (f32, ppm).
import json, os, shutil, struct, subprocess, sys, tempfile
import numpy as np, nibabel as nib

IMAGE = "ghcr.io/astewartau/qsm-ci/matlab-sti-ilsqr@sha256:fa2660014473dbff51a6ddd6aba0840074349ce6d97be9efd0cb8afe6c1c597c"

# (grid, voxel size mm, B0 direction, B0 T, TE s, padsize)
CASES = [
    ((32, 32, 24), (1.0, 1.0, 1.5), (0.1, -0.2, 0.95 ** 0.5), 3.0, 0.020, 12),
    ((32, 28, 28), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0), 7.0, 0.010, 8),
]


def phantom(n, vs, h):
    """Spheres of four susceptibilities in an ellipsoid mask (STAR's fixture phantom), forward
    modelled with STI's dipole kernel on the grid itself; returns (local field ppm, mask)."""
    x = [((np.arange(n[a]) + 1) - n[a] / 2 - 0.5) * vs[a] for a in range(3)]
    X, Y, Z = np.meshgrid(*x, indexing="ij")
    r = 0.4 * min(n[a] * vs[a] for a in range(3))
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
    return field, mask


def run(inputs, params, config, out_names):
    """Runs the container's compiled recon on the given NIfTI inputs; returns its outputs."""
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
        return [np.asarray(nib.load(os.path.join(out, o + ".nii.gz")).dataobj) for o in out_names]


with open(sys.argv[1], "wb") as f:
    for n, vs, h, b0, te, pad in CASES:
        field, mask = phantom(n, vs, h)
        params = {"B0": b0, "B0_dir": list(h), "TE": [te], "voxel_size": list(vs)}
        (chi,) = run({"localfield": field, "mask": mask.astype(np.uint8)}, params, {"padsize": pad}, ["chimap"])
        assert chi.dtype == np.float32
        m = mask.flatten(order="F")
        f.write(struct.pack("<I", int(m.sum())))
        f.write(m.astype(np.uint8).tobytes())
        f.write(field.flatten(order="F")[m].astype("<f8").tobytes())
        f.write(chi.flatten(order="F")[m].astype("<f4").tobytes())
        print(f"{n} {vs} pad {pad}: {m.sum()} mask voxels, χ in [{chi.min():.4f}, {chi.max():.4f}] ppm")

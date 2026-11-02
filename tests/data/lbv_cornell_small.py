# Generator of tests/data/lbv_cornell_small.bin: the MEDI toolbox 2024.11.26's LBV.m and its
# full-multigrid mexMGv6, as the verified QSM-CI container runs them, on a small synthetic total
# field, for tests/lbv_cornell_parity.rs.
#
# Original: ghcr.io/astewartau/qsm-ci/matlab-medi@sha256:aceec8964177fbce20e46635632be6ae167d6cd395c5e65a3030198ea25fff1f
# (lbv-cornell lbv-v1, identical to licensed MATLAB R2023b in the local field, LBV's interior and
# every intermediate; QSMxT/QSM-CI#300). Its recon.m converts the total field from ppm to MEDI's
# units (field * 1e-6 * 2*pi * CF * delta_TE, CF = 42.576e6 * B0, delta_TE = TE(2) - TE(1)), runs
# [fL, interior] = LBV(iFreq, Mask, matrix_size, voxel_size, tol, depth, peel, N1, N2, N3) with the
# options in config.json, and writes fL in ppm as float32 (localfield) and, with
# QSMCI_INTERMEDIATES=1, iFreq, fL and the interior as float64.
#
#   python3 tests/data/lbv_cornell_small.py tests/data/lbv_cornell_small.bin
#
# Needs numpy, nibabel and podman or docker (CONTAINER_ENGINE overrides; the regenerate-fixtures
# workflow runs it in Actions). The field is built with + − × ÷ and sqrt only, so it is the same
# bits everywhere, and is stored in the fixture with the mask. Layout (little-endian, column-major
# over all n voxels): n (u32); the total field (f64, ppm); one u8 per voxel (bit 0: the mask, bit
# 1 + k: LBV's interior in case k); then per case in CASES order, mexMGv6's fL (f32: it computes
# in single precision, so its double output is exactly a float) and recon's localfield (f32, ppm).
import json, os, shutil, struct, subprocess, sys, tempfile
import numpy as np, nibabel as nib

IMAGE = "ghcr.io/astewartau/qsm-ci/matlab-medi@sha256:aceec8964177fbce20e46635632be6ae167d6cd395c5e65a3030198ea25fff1f"

# config.json of each case: LBV.m's defaults (tol 0.01, depth -1 = automatic, which is 1 here,
# peel 0, N1/N2/N3 30/100/100); three levels with a peeled mask, another tol and other counts;
# and depth 0 (one grid, relaxed to convergence).
CASES = [
    {},
    {"tol": 0.005, "depth": 2, "peel": 1, "n1": 5, "n2": 7, "n3": 11},
    {"depth": 0, "n3": 3},
]

# Odd sizes (the coarse grids round up), anisotropic voxels.
DIMS, VS, TE, B0 = (19, 23, 17), (0.9, 1.0, 1.2), (0.004, 0.009, 0.014), 3.0
M64 = (1 << 64) - 1


def hsh(i, salt):
    """xorshift on u64, in [-1, 1) (tests/support/medi_phantom.rs's hash)."""
    x = ((i * 0x9E3779B97F4A7C15) & M64) ^ ((salt * 0xD1B54A32D192ED03) & M64)
    x ^= (x << 13) & M64
    x ^= x >> 7
    x ^= (x << 17) & M64
    return (x >> 11) / float(1 << 52) - 1.0


def phantom():
    """Total field (ppm): two dipoles outside the volume (background) plus local bumps and
    noise inside an ellipsoid mask."""
    nx, ny, nz = DIMS
    n = nx * ny * nz
    field, mask = np.zeros(n), np.zeros(n, np.uint8)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                p = i + nx * (j + ny * k)
                x, y, z = (i - 9.0) * VS[0], (j - 11.0) * VS[1], (k - 8.0) * VS[2]
                bg = 0.0
                for (cx, cy, cz, s) in ((2.0, -3.0, -16.0, 400.0), (-14.0, 5.0, 6.0, -250.0)):
                    dx, dy, dz = x - cx, y - cy, z - cz
                    r2 = dx * dx + dy * dy + dz * dz
                    bg += s * (3.0 * dz * dz - r2) / (r2 * r2 * np.sqrt(r2))
                inside = x * x / 64.0 + y * y / 90.0 + z * z / 64.0 < 1.0
                tube = (x - 3.0) * (x - 3.0) + (y + 2.0) * (y + 2.0)
                blob = (x + 2.0) * (x + 2.0) + (y - 3.0) * (y - 3.0) + (z - 1.0) * (z - 1.0)
                local = 0.08 / (1.0 + tube / 2.0) - 0.05 / (1.0 + blob / 4.0)
                field[p] = bg + local + 0.003 * hsh(p, 7)
                mask[p] = 1 if inside else 0
    vol = lambda v: v.reshape(DIMS, order="F")
    return vol(field), vol(mask)


def run(inputs, params, config):
    """Runs the container's compiled recon; returns its output directory's files as arrays."""
    engine = os.environ.get("CONTAINER_ENGINE") or shutil.which("podman") or "docker"
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        os.makedirs(inp), os.makedirs(out)
        os.chmod(out, 0o777)
        aff = np.diag(list(VS) + [1.0])
        for name, arr in inputs.items():
            img = nib.Nifti1Image(arr, aff)
            img.header.set_slope_inter(1.0, 0.0)   # recon.m would apply nibabel's NaN slope
            nib.save(img, os.path.join(inp, name + ".nii.gz"))
        json.dump(params, open(os.path.join(inp, "params.json"), "w"))
        json.dump(config, open(os.path.join(inp, "config.json"), "w"))
        user = ["--userns=keep-id"] if os.path.basename(engine) == "podman" else ["--user", f"{os.getuid()}:{os.getgid()}"]
        # -w /tmp: with peel > 0, mexMGv6 writes the peeled mask to mask_p<peel>.bin in the
        # working directory (MG::preProcessing) and crashes MATLAB (fclose of a NULL FILE*) where
        # it cannot; the image's own working directory is read-only.
        subprocess.run([engine, "run", "--rm", "--network", "none", *user, "-w", "/tmp", "-e", "MCR_CACHE_ROOT=/tmp/mcr",
                        "-e", "QSMCI_INTERMEDIATES=1", "-v", f"{inp}:/input:ro", "-v", f"{out}:/output", IMAGE,
                        "/opt/qsm-ci/recon", "/input", "/output"], check=True, stdout=subprocess.DEVNULL)
        ld = lambda f: np.asarray(nib.load(f).dataobj)
        res = {"localfield": ld(os.path.join(out, "localfield.nii.gz"))}
        for f in os.listdir(os.path.join(out, "intermediates")):
            if f.endswith(".nii.gz"):
                res[f[:-7]] = ld(os.path.join(out, "intermediates", f))
        return res


field, mask = phantom()
params = {"B0": B0, "B0_dir": [0, 0, 1], "TE": list(TE), "voxel_size": list(VS)}
flat = lambda a: a.ravel(order="F")
labels = flat(mask).astype(np.uint8)
outs = []
for k, cfg in enumerate(CASES):
    res = run({"totalfield": field, "mask": mask}, params, cfg)
    fl = flat(res["fl"])
    assert np.array_equal(fl, fl.astype(np.float32)), "mexMGv6's output is single precision"
    assert res["localfield"].dtype == np.float32
    labels |= ((flat(res["lbv_interior"]) > 0).astype(np.uint8) << (1 + k))
    outs.append(fl.astype("<f4").tobytes() + flat(res["localfield"]).astype("<f4").tobytes())
    print(cfg, "interior", int((flat(res["lbv_interior"]) > 0).sum()), "of", int(mask.sum()),
          "fL in", fl.min(), fl.max())
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<I", field.size))
    f.write(flat(field).astype("<f8").tobytes())
    f.write(labels.tobytes())
    for o in outs:
        f.write(o)

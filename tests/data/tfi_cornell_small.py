# Generator of tests/data/tfi_cornell_small.bin: TFI as the MEDI toolbox 2024.11.26 implements it
# (MEDI_L1.m with a preconditioner), as the verified QSM-CI `tfi-cornell` container runs it, on the
# phantom of tests/support/medi_phantom.rs cropped to an odd 31×36×27 grid, for
# tests/tfi_cornell_parity.rs.
#
# Original: ghcr.io/astewartau/qsm-ci/matlab-medi@sha256:ba6b3eb2c7c410cd1f66544dd455060a2da32ab76b9d9029e0b7b9345fb69565
# (tfi-v1, identical to licensed MATLAB R2023b in χ and every intermediate). Its recon.m converts the
# field (ppm) to radians over the echo spacing, takes iMag (root sum of squares) and N_std
# (Fit_ppm_complex) from the magnitudes, sets P = 1 in the mask and 30 outside, and runs MEDI_L1
# ('nosmv', 'linear', λ 1000, MERIT, 10 Gauss-Newton / 100 CG iterations, unless config.json says
# otherwise); it writes χ (ppm) as float32.
#
#   python3 tests/data/tfi_cornell_small.py tests/data/tfi_cornell_small.bin
#
# Needs numpy, nibabel and podman or docker (CONTAINER_ENGINE overrides; the regenerate-fixtures
# workflow runs it in Actions). The phantom is rebuilt with the Rust code's integer hash and + − × ÷
# only, so it is the same bits; its field (radians) divided by medi_phantom.rs's ppm_to_rad is the
# total field. The odd grid is what exercises dipole_kernel.m's half-step k grid. Layout
# (little-endian, column-major over the 31×36×27 grid, n voxels): per case in CASES order, χ (f32).
import json, os, shutil, struct, subprocess, sys, tempfile
import numpy as np, nibabel as nib

IMAGE = "ghcr.io/astewartau/qsm-ci/matlab-medi@sha256:ba6b3eb2c7c410cd1f66544dd455060a2da32ab76b9d9029e0b7b9345fb69565"

# config.json of each case: one Gauss-Newton step without MERIT; and the container's defaults.
CASES = [{"max_iter": 1, "merit": 0}, {}]
CROP = (31, 36, 27)

DIMS, VS, TE, B0 = (32, 36, 28), (1.0, 1.0, 1.2), (0.004, 0.008, 0.012, 0.016), 3.0
M64 = (1 << 64) - 1


def hsh(i, salt):
    """tests/support/medi_phantom.rs's hash: xorshift on u64, in [-1, 1)."""
    x = ((i * 0x9E3779B97F4A7C15) & M64) ^ ((salt * 0xD1B54A32D192ED03) & M64)
    x ^= (x << 13) & M64
    x ^= x >> 7
    x ^= (x << 17) & M64
    return (x >> 11) / float(1 << 52) - 1.0


def bump(a, d2, w):
    return a / (1.0 + d2 / w)


def phantom(csf=False):
    nx, ny, nz = DIMS
    n = nx * ny * nz
    rdf, mask, mag = np.zeros(n), np.zeros(n, np.uint8), np.zeros((4, n))
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                p = i + nx * (j + ny * k)
                x, y, z = i - 16.0, j - 18.0, (k - 14.0) * VS[2]
                inside = x * x / 169.0 + y * y / 225.0 + z * z / 144.0 < 1.0
                tube = (x - 9.0) * (x - 9.0) + (y - 6.0) * (y - 6.0)
                tube2 = (y + 10.0) * (y + 10.0) + (z - 4.0) * (z - 4.0)
                blob = (x + 4.0) * (x + 4.0) + (y - 3.0) * (y - 3.0) + z * z
                r2 = (20.0 + bump(45.0, tube, 1.5) + bump(35.0, tube2 + 0.25 * (x + 2.0) * (x + 2.0), 1.0)
                      + bump(25.0, blob, 6.0) + 3.0 * hsh(p, 1))
                if csf and min((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz)
                               for cx, cy, cz in ((0.0, -5.0, 0.0), (0.0, 5.0, 0.0), (-3.0, 0.0, 4.0))) < 6.0:
                    r2 = 2.0 + 0.5 * hsh(p, 5)
                m0 = 1.0 + 0.05 * hsh(p, 2) if inside else 0.05 * (1.0 + hsh(p, 3))
                for e, te in enumerate(TE):
                    mag[e, p] = m0 / (1.0 + r2 * te) * (1.0 + 0.01 * hsh(p, 10 + e))
                if inside:
                    mask[p] = 1
                    rdf[p] = (0.002 * (x * x - y * y) + 0.0012 * x * z + bump(0.3, tube, 2.0)
                              - bump(0.15, tube2, 1.5) + bump(0.1, blob, 9.0) + 0.004 * hsh(p, 4))
    vol = lambda v: v.reshape(DIMS, order="F")
    return vol(rdf), vol(mask), np.stack([vol(m) for m in mag], axis=-1)



def run(inputs, params, config):
    """Runs the container's compiled recon; returns χ (ppm)."""
    engine = os.environ.get("CONTAINER_ENGINE") or shutil.which("podman") or "docker"
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        os.makedirs(inp), os.makedirs(out)
        os.chmod(out, 0o777)
        aff = np.diag(list(VS) + [1.0])
        for name, arr in inputs.items():
            nib.save(nib.Nifti1Image(arr, aff), os.path.join(inp, name + ".nii.gz"))
        json.dump(params, open(os.path.join(inp, "params.json"), "w"))
        json.dump(config, open(os.path.join(inp, "config.json"), "w"))
        user = ["--userns=keep-id"] if os.path.basename(engine) == "podman" else ["--user", f"{os.getuid()}:{os.getgid()}"]
        subprocess.run([engine, "run", "--rm", "--network", "none", *user, "-e", "MCR_CACHE_ROOT=/tmp/mcr",
                        "-v", f"{inp}:/input:ro", "-v", f"{out}:/output", IMAGE,
                        "/opt/qsm-ci/recon", "/input", "/output"], check=True, stdout=subprocess.DEVNULL)
        return np.asarray(nib.load(os.path.join(out, "chimap.nii.gz")).dataobj)


ppm_to_rad = 2.0 * np.pi * (TE[1] - TE[0]) * 42.576e6 * B0 * 1e-6      # medi_phantom.rs's
rdf, mask, mag = phantom()
c = tuple(slice(0, s) for s in CROP)
inputs = {"totalfield": rdf[c] / ppm_to_rad, "mask": mask[c], "magnitude": mag[c]}
params = {"B0": B0, "B0_dir": [0, 0, 1], "TE": list(TE), "voxel_size": list(VS)}
with open(sys.argv[1], "wb") as f:
    for cfg in CASES:
        chi = run(inputs, params, cfg)
        assert chi.dtype == np.float32 and chi.shape == CROP
        f.write(chi.ravel(order="F").astype("<f4").tobytes())
        print(cfg, "χ in", chi.min(), chi.max())

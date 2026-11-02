# Generator of tests/data/resharp_sunhongfu_small.bin: Hongfu Sun's resharp.m (github.com/sunhongfu/QSM
# c75b138), as the verified QSM-CI `resharp` container runs it, on the phantom of
# tests/support/medi_phantom.rs (its field as the total field), for tests/resharp_sunhongfu_parity.rs.
#
# Original: ghcr.io/astewartau/qsm-ci/resharp@sha256:b147f71e433555c8be550195d16451caa191dacbcbccd3e392e200db8e0d7e2d
# (v1, identical to licensed MATLAB R2023b in vivo; QSMxT/QSM-CI#352). Its recon.m calls
# resharp(field, mask, voxel_size, radius, lambda, iterations) and, with QSMCI_INTERMEDIATES=1,
# writes the local field in float64 and the eroded mask.
#
#   python3 tests/data/resharp_sunhongfu_small.py tests/data/resharp_sunhongfu_small.bin
#
# Needs numpy, nibabel and podman or docker (CONTAINER_ENGINE overrides; the regenerate-fixtures
# workflow runs it in Actions). The phantom is rebuilt with the Rust code's integer hash and + − × ÷
# only, so it is the same bits. Voxels are 1×1×1.2 mm, so the 3 mm kernel is 3×3×3 voxels
# (round(3/1.2) = 3) and the 2 mm one 2×2×2 (round(1.67)). Layout (little-endian, column-major over
# the whole 32×36×28 grid, n voxels): per case in CASES order, the local field (f64), then the
# eroded mask (u8).
import json, os, shutil, struct, subprocess, sys, tempfile
import numpy as np, nibabel as nib

IMAGE = "ghcr.io/astewartau/qsm-ci/resharp@sha256:b147f71e433555c8be550195d16451caa191dacbcbccd3e392e200db8e0d7e2d"

# config.json of each case: resharp.m's defaults (3 mm, λ 1e-4, at most 200 cgs iterations, which
# converges), a smaller kernel and larger λ, and a cap of 5 iterations (cgs stops unconverged and
# returns its smallest-residual iterate).
CASES = [{}, {"radius": 2, "lambda": 1e-3}, {"iterations": 5}]

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
    """Runs the container's compiled recon; returns its intermediates as arrays."""
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
                        "-e", "QSMCI_INTERMEDIATES=1", "-v", f"{inp}:/input:ro", "-v", f"{out}:/output", IMAGE,
                        "/opt/qsm-ci/recon", "/input", "/output"], check=True, stdout=subprocess.DEVNULL)
        ld = lambda f: np.asarray(nib.load(os.path.join(out, "intermediates", f + ".nii.gz")).dataobj)
        return ld("localfield_f64"), ld("resharp_mask")


rdf, mask, _ = phantom()
params = {"B0": B0, "B0_dir": [0, 0, 1], "TE": list(TE), "voxel_size": list(VS)}
inputs = {"totalfield": rdf, "mask": mask}
with open(sys.argv[1], "wb") as f:
    for cfg in CASES:
        local, ero = run(inputs, params, cfg)
        assert local.dtype == np.float64
        f.write(local.ravel(order="F").astype("<f8").tobytes())
        f.write((ero.ravel(order="F") > 0).astype(np.uint8).tobytes())
        print(cfg, "eroded mask", int((ero > 0).sum()), "of", int(mask.sum()), "local field in", local.min(), local.max())

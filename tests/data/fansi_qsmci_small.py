# Generator of tests/data/fansi_qsmci_small.bin: the FANSI-family originals (FANSI-toolbox
# 93044a3: ndi.m, nlTV.m, nlTGV.m, nlL1TV.m, WH_nlTV.m) and HD-QSM (HDQSM.m, mglambert/HD-QSM
# c456a94), as the verified QSM-CI containers run them, on the MEDI parity phantom of
# tests/support/medi_phantom.rs with an oblique B0 and anisotropic voxels (1 x 1 x 1.2 mm), for
# tests/fansi_qsmci_parity.rs.
#
# Originals (each bit-identical to licensed MATLAB R2023b; QSMxT/QSM-CI#327, #372): see IMAGES.
# Each recon.m builds the kernel with FANSI's dipole_kernel_angulated(N, voxel_size, B0_dir)
# (mode 1, the discrete kernel), converts the field to radians with gyro = 2*pi*42.58 (FANSI's)
# at the echo time each method's wrapper uses (NDI, L1-QSM, WH-QSM: the first echo; nlTV/nlTGV:
# TE(2) - TE(1)), weights by min(2*mask*iMag/max(iMag), 1) of the echoes' root sum of squares
# (NDI, nlTV, nlTGV; L1-QSM times lambda 0.8; WH-QSM: WH_nlTV's own ones), and runs each function
# at its defaults. HD-QSM works in ppm, weights by mask.*mag_use/max(mag_use) with the paper's
# Eq. 2 magnitude, and masks its input. With QSMCI_INTERMEDIATES=1 the recons write the
# unmasked solution in float64 (x_rad, or HD-QSM's x_unmasked).
#
#   python3 tests/data/fansi_qsmci_small.py tests/data/fansi_qsmci_small.bin
#
# Needs numpy, nibabel and podman or docker (CONTAINER_ENGINE overrides; the regenerate-fixtures
# workflow runs it in Actions). The phantom is medi_cornell_small.py's, the same bits as the Rust
# one. Layout (little-endian, column-major over the phantom's mask voxels, m of them): m (u32);
# then per case in IMAGES order, the solution (f64): radians for the FANSI methods, ppm for HD-QSM.
# machine-dependent: the originals iterate in single precision through the MATLAB Runtime's MKL
# FFTs, which may round differently on another CPU; the test's limit (relative L2 2e-5) is ~4x the
# largest difference to the ports measured on the VM (2026-11-02).
import json, os, shutil, struct, subprocess, sys, tempfile
import numpy as np, nibabel as nib

IMAGES = [
    ("ndi", "ghcr.io/astewartau/qsm-ci/ndi@sha256:98f820be521d171c729f96e1deef733ebfad7d8bbb4996693a12187a59719f28", "x_rad"),
    ("nltv", "ghcr.io/astewartau/qsm-ci/fansi@sha256:e606064f1d980cddf65ad7cc51cea68a759e26593f0e161531ab21b2d9399995", "x_rad"),
    ("nltgv", "ghcr.io/astewartau/qsm-ci/fansi@sha256:e34728c1a66c2ab7654bfcf5308ed507db79da9d7be599936a77156cd4df8730", "x_rad"),
    ("l1qsm", "ghcr.io/astewartau/qsm-ci/l1-qsm@sha256:edb8c46ae324f34ce337eea4f7179f8d73267f5c21f1517a9041e28c449b2177", "x_rad"),
    ("whqsm", "ghcr.io/astewartau/qsm-ci/wh-qsm@sha256:f0e2678680d195b13a2de71b180b3177171a2a79840c134f68310cb665fca451", "x_rad"),
    ("hdqsm", "ghcr.io/astewartau/qsm-ci/hd-qsm@sha256:79d3ecefcd9ad766502c8ef079bcb1a3408d658cfe27f50bfc6f0b61cba8cb92", "x_unmasked"),
]
B0_DIR = [0.1, -0.15, 0.98]   # tests/fansi_qsmci_parity.rs's; recon.m normalises it

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


def run(image, inputs, params):
    """Runs the container's compiled recon; returns its output directory's files as arrays."""
    engine = os.environ.get("CONTAINER_ENGINE") or shutil.which("podman") or "docker"
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        os.makedirs(inp), os.makedirs(out)
        os.chmod(out, 0o777)
        aff = np.diag(list(VS) + [1.0])
        for name, arr in inputs.items():
            nib.save(nib.Nifti1Image(arr, aff), os.path.join(inp, name + ".nii.gz"))
        json.dump(params, open(os.path.join(inp, "params.json"), "w"))
        user = ["--userns=keep-id"] if os.path.basename(engine) == "podman" else ["--user", f"{os.getuid()}:{os.getgid()}"]
        subprocess.run([engine, "run", "--rm", "--network", "none", *user, "-e", "MCR_CACHE_ROOT=/tmp/mcr",
                        "-e", "QSMCI_INTERMEDIATES=1", "-v", f"{inp}:/input:ro", "-v", f"{out}:/output", image,
                        "/opt/qsm-ci/recon", "/input", "/output"], check=True, stdout=subprocess.DEVNULL)
        ld = lambda f: np.asarray(nib.load(f).dataobj)
        res = {"chimap": ld(os.path.join(out, "chimap.nii.gz"))}
        for f in os.listdir(os.path.join(out, "intermediates")):
            if f.endswith(".nii.gz"):
                res[f[:-7]] = ld(os.path.join(out, "intermediates", f))
        return res


ppm_to_rad = 2.0 * np.pi * (TE[1] - TE[0]) * 42.576e6 * B0 * 1e-6      # medi_phantom.rs's
rdf, mask, mag = phantom()
inputs = {"localfield": rdf / ppm_to_rad, "mask": mask, "magnitude": mag}
params = {"B0": B0, "B0_dir": B0_DIR, "TE": list(TE), "voxel_size": list(VS)}
m = mask.ravel(order="F") > 0
at = lambda a: a.ravel(order="F")[m]
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<I", int(m.sum())))
    for name, image, key in IMAGES:
        res = run(image, inputs, params)
        assert res[key].dtype == np.float64, (name, res[key].dtype)
        f.write(at(res[key]).astype("<f8").tobytes())
        print(name, key, "in", res[key][mask > 0].min(), res[key][mask > 0].max())

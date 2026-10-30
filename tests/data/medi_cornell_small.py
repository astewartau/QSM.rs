# Generator of tests/data/medi_cornell_small.bin: the MEDI toolbox 2024.11.26's pipeline around
# MEDI_L1, as the verified QSM-CI container runs it, on the phantom of
# tests/support/medi_phantom.rs, for tests/medi_cornell_parity.rs.
#
# Original: ghcr.io/astewartau/qsm-ci/matlab-medi@sha256:b81527b0c5a49a39378700358dd3b0ada3a4a0ed5742715d0d699c99846f0278
# (medi-cornell v4, bit-identical to licensed MATLAB R2023b in χ and every intermediate;
# QSMxT/QSM-CI#309). Its recon.m runs README.m's steps on the given local field: N_std from
# Fit_ppm_complex(magnitudes), iMag = sqrt(sum(mag.^2, 4)), R2* = arlo(TE, mag),
# Mask_CSF = extract_CSF(R2*, Mask, voxel_size), then MEDI_L1 with the options in config.json,
# and writes χ (ppm) as float32 and, with QSMCI_INTERMEDIATES=1, the intermediates as float64.
#
#   python3 tests/data/medi_cornell_small.py tests/data/medi_cornell_small.bin
#
# Needs numpy, nibabel and podman or docker (CONTAINER_ENGINE overrides; the regenerate-fixtures
# workflow runs it in Actions). The phantom is rebuilt here with the Rust code's integer hash and
# + − × ÷ only, so it is the same bits. Layout (little-endian, column-major over the phantom's
# mask voxels, m of them): m (u32); then per case in CASES order, χ (f32, ppm); then, from the
# last case (MEDI_L1's defaults), N_std and R2* (f64), and the CSF mask and the mask MEDI_L1
# inverted on, one u8 each (bit 0: CSF, bit 1: inverted).
import json, os, shutil, struct, subprocess, sys, tempfile
import numpy as np, nibabel as nib

IMAGE = "ghcr.io/astewartau/qsm-ci/matlab-medi@sha256:b81527b0c5a49a39378700358dd3b0ada3a4a0ed5742715d0d699c99846f0278"

# config.json of each case: one Gauss-Newton step without the CSF term, with each SMV option
# (where SMV acts), on the phantom as it is; then MEDI_L1's defaults (λ 1000, SMV 5 with mask
# shrinking, MERIT, MEDI+0 with λ_CSF 100, 10 steps) on the phantom with three CSF-like
# (R2* ≈ 2 Hz) blobs near the centre, so extract_CSF finds ventricles.
CASES = [
    {"smv": 0, "csf": 0, "max_iter": 1},
    {"smv": 1, "smv_shrink_mask": 1, "csf": 0, "max_iter": 1},
    {"smv": 1, "smv_shrink_mask": 0, "csf": 0, "max_iter": 1},
    {},
]

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
        json.dump(config, open(os.path.join(inp, "config.json"), "w"))
        user = ["--userns=keep-id"] if os.path.basename(engine) == "podman" else ["--user", f"{os.getuid()}:{os.getgid()}"]
        subprocess.run([engine, "run", "--rm", "--network", "none", *user, "-e", "MCR_CACHE_ROOT=/tmp/mcr",
                        "-e", "QSMCI_INTERMEDIATES=1", "-v", f"{inp}:/input:ro", "-v", f"{out}:/output", IMAGE,
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
rdf, _, mag_csf = phantom(csf=True)
inputs_csf = {"localfield": rdf / ppm_to_rad, "mask": mask, "magnitude": mag_csf}
params = {"B0": B0, "B0_dir": [0, 0, 1], "TE": list(TE), "voxel_size": list(VS)}
m = mask.ravel(order="F") > 0
at = lambda a: a.ravel(order="F")[m]
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<I", int(m.sum())))
    for cfg in CASES:
        res = run(inputs if cfg else inputs_csf, params, cfg)
        assert res["chimap"].dtype == np.float32
        f.write(at(res["chimap"]).astype("<f4").tobytes())
        print(cfg, "χ in", res["chimap"].min(), res["chimap"].max())
    f.write(at(res["n_std"]).astype("<f8").tobytes() + at(res["r2s"]).astype("<f8").tobytes())
    labels = (at(res["mask_csf"]) > 0).astype(np.uint8) | 2 * (at(res["mask_inverted"]) > 0).astype(np.uint8)
    f.write(labels.tobytes())
    print("CSF voxels", int((labels & 1).sum()), "inverted on", int((labels >> 1).sum()), "of", int(m.sum()))

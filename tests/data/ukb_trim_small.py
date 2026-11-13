# Generator of tests/data/ukb_trim_small.bin: UK Biobank's phase-reliability maps and mask trims
# (phasevariance_nonlin_v2.m and UKBiobank_QSM.m, Wang, Tendler & Miller), as the verified QSM-CI
# ukb-qsm container runs them, on three small two-echo phantoms, for tests/ukb_qsm_parity.rs:
#   ghcr.io/astewartau/qsm-ci/matlab-ukb-qsm@sha256:c8a50375ab0b69e8a928a07b1ecef51e533f7948c67b706434284ec6f1e3b9cf
# (ukb-qsm v2, bit-identical to licensed MATLAB R2023b in every intermediate; QSMxT/QSM-CI#305).
#
#   python3 tests/data/ukb_trim_small.py tests/data/ukb_trim_small.bin
#
# The phantoms are 1 mm (where the kernel's placement is a half-integer, which MATLAB rounds up),
# 1 x 1 x 1.5 mm, and UK Biobank's own 0.8 x 0.8 x 3 mm, all with even dimensions (the container
# pads odd ones, which UK Biobank's code itself never does). Each has a smooth field, a patch of
# fast-varying phase (a "sinus") that the trims remove, and a little noise.
#
# Needs numpy, nibabel and podman or docker (CONTAINER_ENGINE overrides). Layout (little-endian,
# column-major), per case in the order of CASES: the brain mask as bits over the grid (least
# significant bit first, ceil(n/8) bytes); the first and the last echo's wrapped phase over the
# grid (f32, rad; the container is given these values); at the mask's voxels the container's
# reliability maps `map` and `map2` (f64); then as bits its trimmed mask (`mask`, before V_SHARP),
# V_SHARP's mask and the final trimmed mask.
# pip: numpy nibabel
import json, os, shutil, subprocess, sys, tempfile
import numpy as np, nibabel as nib

IMAGE = "ghcr.io/astewartau/qsm-ci/matlab-ukb-qsm@sha256:c8a50375ab0b69e8a928a07b1ecef51e533f7948c67b706434284ec6f1e3b9cf"
TE, B0 = (0.00942, 0.020), 3.0
CASES = [((32, 30, 24), (1.0, 1.0, 1.0)), ((32, 32, 20), (1.0, 1.0, 1.5)), ((40, 36, 12), (0.8, 0.8, 3.0))]


def phantom(N, VS):
    i, j, k = np.meshgrid(*[np.arange(v, dtype=float) for v in N], indexing="ij")
    c = [(n - 1) / 2 + 0.4 for n in N]
    x, y, z = (i - c[0]) * VS[0], (j - c[1]) * VS[1], (k - c[2]) * VS[2]
    R = [0.42 * n * v for n, v in zip(N, VS)]
    mask = (x / R[0]) ** 2 + (y / R[1]) ** 2 + (z / R[2]) ** 2 < 1.0
    d1 = (x - 4.0) ** 2 + (y + 3.0) ** 2 + z ** 2
    omega = 0.03 * x - 0.02 * y + 0.01 * z + 0.002 * x * y - 0.001 * z * z + 0.08 / (1.0 + d1 / 6.0)
    ds = (x - 0.5 * R[0]) ** 2 + (y - 0.3 * R[1]) ** 2 + (z + 0.3 * R[2]) ** 2
    omega += 3.0 * np.exp(-ds / (0.15 * R[0]) ** 2) * np.sin(1.7 * x + 2.3 * y)
    noise = np.random.default_rng(0).normal(0, 0.05, N)
    phase = np.zeros(tuple(N) + (len(TE),))
    for e, t in enumerate(TE):
        raw = omega * (t * 1000.0) + 0.3 + noise * (e + 1)
        phase[..., e] = raw - 2.0 * np.pi * np.round(raw / (2.0 * np.pi))
    return phase.astype(np.float32).astype(np.float64), mask


def run(phase, mask, VS):
    engine = os.environ.get("CONTAINER_ENGINE") or shutil.which("podman") or "docker"
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        os.makedirs(inp), os.makedirs(out)
        os.chmod(out, 0o777)
        aff = np.diag(list(VS) + [1.0])
        nib.save(nib.Nifti1Image(phase, aff), os.path.join(inp, "phase.nii.gz"))
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), aff), os.path.join(inp, "mask.nii.gz"))
        json.dump({"B0": B0, "B0_dir": [0, 0, 1], "TE": list(TE), "voxel_size": list(VS)}, open(os.path.join(inp, "params.json"), "w"))
        user = ["--userns=keep-id"] if os.path.basename(engine) == "podman" else ["--user", f"{os.getuid()}:{os.getgid()}"]
        subprocess.run([engine, "run", "--rm", "--network", "none", *user, "-e", "MCR_CACHE_ROOT=/tmp/mcr", "-e", "HOME=/tmp",
                        "-e", "QSMCI_INTERMEDIATES=1", "-v", f"{inp}:/input:ro", "-v", f"{out}:/output", IMAGE,
                        "/opt/qsm-ci/recon", "/input", "/output"], check=True, stdout=subprocess.DEVNULL)
        ld = lambda f: np.asarray(nib.load(os.path.join(out, "intermediates", f + ".nii.gz")).dataobj).ravel(order="F")
        return {f: ld(f) for f in ("map", "map2", "mask", "mask_vsharp", "mask_final")}


bits = lambda m: np.packbits(m, bitorder="little").tobytes()
with open(sys.argv[1], "wb") as f:
    for N, VS in CASES:
        phase, mask = phantom(N, VS)
        r = run(phase, mask, VS)
        m = mask.ravel(order="F")
        f.write(bits(m))
        for e in (0, -1):
            f.write(phase[..., e].ravel(order="F").astype("<f4").tobytes())
        f.write(r["map"][m].astype("<f8").tobytes() + r["map2"][m].astype("<f8").tobytes())
        for name in ("mask", "mask_vsharp", "mask_final"):
            f.write(bits(r[name] > 0))
        print(N, VS, "mask", int(m.sum()), "trimmed", int((r["mask"] > 0).sum()), "final", int((r["mask_final"] > 0).sum()))

# Generator of tests/data/ukb_qsm_small.bin: the UK Biobank QSM pipeline (UKBiobank_QSM_core.m,
# Wang, Tendler & Miller), as the verified QSM-CI ukb-qsm container runs it, on a small two-echo
# phantom, for tests/ukb_qsm_parity.rs:
#   ghcr.io/astewartau/qsm-ci/matlab-ukb-qsm@sha256:c8a50375ab0b69e8a928a07b1ecef51e533f7948c67b706434284ec6f1e3b9cf
# (ukb-qsm v2, bit-identical to licensed MATLAB R2023b in χ and every intermediate;
# QSMxT/QSM-CI#305). The pipeline: phase-variance mask trimming, STI Suite's MRPhaseUnwrap per echo
# (padsize 64), the echoes combined with W = TE·exp(−TE/40 ms), V_SHARP (smvsize 12), more
# trimming, QSM_iLSQR (padsize 64).
#
#   python3 tests/data/ukb_qsm_small.py tests/data/ukb_qsm_small.bin
#
# The final step, QSM_iLSQR with UK Biobank's 64 mm padding, is not in the fixture: at that
# padding it takes ~10 s even on this phantom, and the port is checked against STI Suite by
# tests/ilsqr_sti_parity.rs.
#
# Needs numpy, nibabel and podman or docker (CONTAINER_ENGINE overrides). The phase is built from
# + - * / and round only, so the Rust test rebuilds it bit for bit. Layout (little-endian,
# column-major, 32x32x20 at 1x1x1.5 mm): the trimmed mask and V_SHARP's mask, as bits over the
# grid (least significant bit first, ceil(n/8) bytes each); then the combined unwrapped phase over
# the whole grid (f64, rad: V_SHARP uses the phase just outside the mask, which its even crop and
# its resampling of the 1.5 mm axis carry onto its 1 mm grid), and at V_SHARP's mask's voxels its
# local field (f64).
import json, os, shutil, subprocess, sys, tempfile
import numpy as np, nibabel as nib

IMAGE = "ghcr.io/astewartau/qsm-ci/matlab-ukb-qsm@sha256:c8a50375ab0b69e8a928a07b1ecef51e533f7948c67b706434284ec6f1e3b9cf"
N, VS, TE, B0 = (32, 32, 20), (1.0, 1.0, 1.5), (0.00942, 0.020), 3.0


def phantom():
    """Wrapped two-echo phase (rad) and the brain mask."""
    i, j, k = np.meshgrid(*[np.arange(v, dtype=float) for v in N], indexing="ij")
    x, y, z = i - 15.6, j - 15.3, (k - 9.7) * 1.5
    mask = x * x / 196.0 + y * y / 169.0 + z * z / 144.0 < 1.0
    d1 = (x - 4.0) * (x - 4.0) + (y + 3.0) * (y + 3.0) + z * z
    # rad/ms: a background spanning several wraps at 20 ms, and a local source
    omega = 0.03 * x - 0.02 * y + 0.01 * z + 0.002 * x * y - 0.001 * z * z + 0.08 / (1.0 + d1 / 6.0)
    phase = np.zeros(N + (len(TE),))
    for e, t in enumerate(TE):
        raw = omega * (t * 1000.0) + 0.3
        phase[..., e] = raw - 2.0 * np.pi * np.round(raw / (2.0 * np.pi))
    return phase, mask


phase, mask = phantom()
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
    m1, mv = ld("mask") > 0, ld("mask_vsharp") > 0
    comb, local = ld("phs_comb"), ld("dB_vsf")
with open(sys.argv[1], "wb") as f:
    for m in (m1, mv):
        f.write(np.packbits(m, bitorder="little").tobytes())
    f.write(comb.astype("<f8").tobytes() + local[mv].astype("<f8").tobytes())
print("masks", int(m1.sum()), int(mv.sum()), "of", int(mask.sum()))

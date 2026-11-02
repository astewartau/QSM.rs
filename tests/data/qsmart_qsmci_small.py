# Generator of tests/data/qsmart_qsmci_small.bin: QSMART v1.0, the authors' MATLAB toolbox, as the
# verified QSM-CI container runs it, on a small multi-echo vessel phantom, with every intermediate,
# for tests/qsmart_qsmci_parity.rs.
#
# Original: ghcr.io/astewartau/qsm-ci/matlab-qsmart@sha256:01f36e868bcacf494914b498b8f1cfc39480e533b0b654519e4476c19e532493
# (qsmart-matlab v4: bit-identical to licensed MATLAB R2023b with the real Image Processing Toolbox
# in the final χ and every intermediate, in silico and in vivo; QSMxT/QSM-CI#296). v4 runs on the
# matrix as given, as the authors' QSMART.m does; v3 zero-padded odd dimensions to even. This grid is
# even in every dimension, so the two give the same output here, and the offset adjustment's
# k-grid choice (QSM.rs's `QsmartOffsetGrid`, which only differs on odd grids) does not arise. Its recon.m runs
# QSMART.m's pipeline at Demo_QSMART.m's settings (γ = 2.675e8 rad/s/T, Frangi scales 0.5-6 in
# steps of 0.5, C 500, bottom-hat radius 8, fit_threshold 40, SDF radius 8, σ (10, 0) then (8, 2),
# lower limit 0.6, curvature constant 500, STI Suite's QSM_iLSQR at 'niter' 50):
#   vasculature_mask (mean magnitude, N4, bottom-hat, Frangi, Otsu) -> unwrap_phase (lapunwrap) ->
#   echofit (tfs, R_0) -> QSMART_SDF stage 1 -> QSM_iLSQR -> QSMART_SDF stage 2 -> QSM_iLSQR ->
#   adjust_offset
# and with QSMCI_INTERMEDIATES=1 writes each stage's output in float64.
#
#   python3 tests/data/qsmart_qsmci_small.py tests/data/qsmart_qsmci_small.bin
#
# machine-dependent: STI Suite's QSM_iLSQR is single-precision LSQR and rounds differently on
# different CPUs (see tests/data/ilsqr_sti_qsmci_small.py), and the stages after it carry that. The
# committed .bin was made with v3 on an Intel i5-11500. v4 on a GitHub-hosted runner
# (regenerate-fixtures run 41384288424, 2026-11-02) reproduces the phantom and every intermediate
# up to lfs1 and lfs2 exactly, and moves chi1, chi2 and chimap by 1.2e-6, 1.8e-5 and 1.8e-5
# relative (v3 on a runner, 2026-10-30: the same, with lfs1 and lfs2 at 4e-14).
#
# Needs numpy, nibabel and podman or docker (CONTAINER_ENGINE overrides; the regenerate-fixtures
# workflow runs it in Actions). The phantom (built here, seeded): a 36x36x28 grid at 1 mm, B0 3 T
# along z, TE 4, 8, 12 ms; an ellipsoidal brain (the mask) inside a head, in air; in the brain,
# three iron-rich spheres and three veins (tubes 1.5 mm in radius, χ 0.45 ppm, short T2*); the field
# from the head's and brain's χ by the dipole kernel on the grid; complex noise (SNR ~60 in tissue).
# Layout (little-endian, column-major over the grid): the mask (u8), the three echoes' wrapped
# phases (f32, rad), their magnitudes (f32); then the container's intermediates: vasc_only and R_0
# over the grid (u8), and tfs, lfs1, lfs2 (ppm), chi1, chi2 and the final chimap (ppm), each f32 at
# the mask voxels.
import json, os, shutil, subprocess, sys, tempfile
import numpy as np, nibabel as nib

IMAGE = "ghcr.io/astewartau/qsm-ci/matlab-qsmart@sha256:01f36e868bcacf494914b498b8f1cfc39480e533b0b654519e4476c19e532493"
DIMS = (36, 36, 28)
B0, TE = 3.0, [0.004, 0.008, 0.012]
GYRO = 2.675e8  # QSMART's γ (rad/s/T), as recon.m's params.gyro


def phantom():
    """(phase per echo, magnitude per echo, mask), float32 / uint8."""
    rng = np.random.default_rng(317)
    x = [np.arange(n) - (n - 1) / 2 for n in DIMS]
    X, Y, Z = np.meshgrid(*x, indexing="ij")
    head = (X / 17) ** 2 + (Y / 16.5) ** 2 + (Z / 13) ** 2 < 1
    brain = (X / 14) ** 2 + (Y / 13.5) ** 2 + (Z / 10.5) ** 2 < 1
    chi = np.where(head, 0.0, 9.4)  # air outside the head (relative to tissue), ppm
    r2s = np.where(brain, 25.0, 30.0)
    m0 = np.where(brain, 1000.0, np.where(head, 600.0, 0.0))
    for c, v, rad in [((-5, 4, 2), 0.15, 3.0), ((5, -3, -1), 0.10, 2.5), ((0, 6, -4), 0.20, 2.0)]:
        s = (X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2 < rad ** 2
        chi[s] += v
        r2s[s] += 200 * v
    # Veins: tubes along z, along x (oblique in y) and along y.
    veins = [((X - 3) ** 2 + (Y + 7) ** 2 < 1.5 ** 2) & (np.abs(Z) < 9),
             ((Y - 0.3 * X - 1) ** 2 + (Z - 5) ** 2 < 1.5 ** 2) & (np.abs(X) < 11),
             ((X + 8) ** 2 + (Z + 2) ** 2 < 1.5 ** 2) & (np.abs(Y) < 10)]
    for v in veins:
        chi[v & brain] += 0.45
        r2s[v & brain] = 90.0
        m0[v & brain] = 700.0
    k = np.meshgrid(*[np.fft.fftfreq(n) for n in DIMS], indexing="ij")
    k2 = k[0] ** 2 + k[1] ** 2 + k[2] ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        d = 1 / 3 - k[2] ** 2 / k2
    d[0, 0, 0] = 0
    field = np.real(np.fft.ifftn(d * np.fft.fftn(chi)))  # ppm
    field -= field[brain].mean()
    phases, mags = [], []
    for te in TE:
        s = m0 * np.exp(-r2s * te) * np.exp(1j * field * GYRO * B0 * 1e-6 * te)
        s = s + rng.normal(0, 10, DIMS) + 1j * rng.normal(0, 10, DIMS)
        phases.append(np.angle(s).astype(np.float32))
        mags.append(np.abs(s).astype(np.float32))
    return phases, mags, brain.astype(np.uint8)


phases, mags, mask = phantom()
engine = os.environ.get("CONTAINER_ENGINE") or shutil.which("podman") or "docker"
names = ["vasc_only", "R_0", "tfs", "lfs1", "lfs2", "chi1", "chi2", "chimap"]
with tempfile.TemporaryDirectory() as tmp:
    inp, out = os.path.join(tmp, "in"), os.path.join(tmp, "out")
    os.makedirs(inp), os.makedirs(out)
    os.chmod(out, 0o777)
    for name, arr in (("phase", np.stack(phases, -1)), ("magnitude", np.stack(mags, -1)), ("mask", mask)):
        nib.save(nib.Nifti1Image(arr, np.eye(4)), os.path.join(inp, name + ".nii.gz"))
    json.dump({"B0": B0, "B0_dir": [0.0, 0.0, 1.0], "TE": TE, "voxel_size": [1.0, 1.0, 1.0]},
              open(os.path.join(inp, "params.json"), "w"))
    user = ["--userns=keep-id"] if os.path.basename(engine) == "podman" else ["--user", f"{os.getuid()}:{os.getgid()}"]
    subprocess.run([engine, "run", "--rm", "--network", "none", *user, "-e", "MCR_CACHE_ROOT=/tmp/mcr",
                    "-e", "QSMCI_INTERMEDIATES=1", "-v", f"{inp}:/input:ro", "-v", f"{out}:/output", IMAGE,
                    "/opt/qsm-ci/recon", "/input", "/output"], check=True)
    res = {n: np.asarray(nib.load(os.path.join(out, "intermediates", n + ".nii.gz")).dataobj) for n in names}
    print(open(os.path.join(out, "intermediates", "intermediates.json")).read()[:300])

m = mask.flatten(order="F") != 0
with open(sys.argv[1], "wb") as f:
    f.write(mask.tobytes(order="F"))
    for a in phases + mags:
        f.write(a.astype("<f4").tobytes(order="F"))
    for n in ("vasc_only", "R_0"):
        v = res[n]
        assert set(np.unique(v)) <= {0.0, 1.0}, n
        f.write(v.astype(np.uint8).tobytes(order="F"))
    for n in names[2:]:
        v = res[n].astype(np.float64).flatten(order="F")[m]
        f.write(v.astype("<f4").tobytes())
        print(f"{n}: [{v.min():.4f}, {v.max():.4f}] at {m.sum()} mask voxels")
print(f"vessels (vasc_only = 0 in the mask): {int(((res['vasc_only'] == 0) & (mask != 0)).sum())}, "
      f"unreliable (R_0 = 0 in the mask): {int(((res['R_0'] == 0) & (mask != 0)).sum())}")

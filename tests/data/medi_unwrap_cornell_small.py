# Generator of tests/data/medi_unwrap_cornell_small.bin: MEDI's field map, the MEDI toolbox
# 2024.11.26's Fit_ppm_complex followed by its region-growing unwrapPhase (mexUnwrap.c, Tian Liu),
# as the verified QSM-CI `fit-ppm-complex-cornell` container runs them, on small multi-echo
# phantoms, for tests/medi_unwrap_cornell_parity.rs.
#
# Original: ghcr.io/astewartau/qsm-ci/matlab-medi@sha256:7efa4bd641b8311ac97a8a09dd438290f998f8163219b2f5d7ecae777a59811c
# (fit-ppm-complex-cornell fitppm-v1, identical to licensed MATLAB R2023b in the total field and
# every intermediate; QSMxT/QSM-CI#300). Its recon.m builds iField = mag .* exp(-1i*phase), runs
#   iMag = sqrt(sum(abs(iField).^2, 4));
#   [iFreq_raw, N_std] = Fit_ppm_complex(iField);
#   iFreq = unwrapPhase(iMag, iFreq_raw, matrix_size);
# and, with QSMCI_INTERMEDIATES=1, writes iMag, iFreq_raw, iFreq and the total field
# (iFreq / (2*pi*delta_TE*CF) * 1e6 ppm) as float64.
#
#   python3 tests/data/medi_unwrap_cornell_small.py tests/data/medi_unwrap_cornell_small.bin
#
# Needs numpy, nibabel and podman or docker (CONTAINER_ENGINE overrides; the regenerate-fixtures
# workflow runs it in Actions). The fixture stores the phantoms' inputs as well, so the test
# never rebuilds them.
#
# What the phantoms exercise in mexUnwrap.c:
# - magnitude plateaus (iMag takes few distinct values), so the max-heap meets many ties and its
#   exact sift order decides which neighbour a voxel is unwrapped against;
# - a field that wraps several times per echo spacing, plus noise, so the result depends on the
#   order of growth;
# - background below the noise level (0.5 % of the brightest iMag), which is never planned and
#   keeps its wrapped phase, and a bright island cut off from the seed by that background, which
#   stays wrapped too;
# - even and odd matrix sizes, for the seed at (dim/2) of every axis (integer division).
#
# Layout (little-endian, column-major): number of cases (u32); per case: nx, ny, nz, n_echoes
# (u32 each), the echo times (f64, s), B0 (f64, T); per echo the magnitude then the phase (f64,
# the inputs); then iMag (f64), iFreq_raw (f64), iFreq (f32: mexUnwrap computes in single
# precision, so its double output is exactly a float) and the total field (f64, ppm).
import json, os, shutil, struct, subprocess, sys, tempfile
import numpy as np, nibabel as nib

IMAGE = "ghcr.io/astewartau/qsm-ci/matlab-medi@sha256:7efa4bd641b8311ac97a8a09dd438290f998f8163219b2f5d7ecae777a59811c"

# (dims, voxel size, echo times, B0): even/odd/odd and odd/even/even sizes.
CASES = [
    ((18, 15, 11), (1.0, 1.0, 1.0), (0.004, 0.009, 0.014), 3.0),
    ((13, 16, 10), (0.9, 1.1, 1.3), (0.003, 0.008, 0.013, 0.018), 7.0),
]
M64 = (1 << 64) - 1


def hsh(i, salt):
    """xorshift on u64, in [-1, 1) (tests/support/medi_phantom.rs's hash)."""
    x = ((i * 0x9E3779B97F4A7C15) & M64) ^ ((salt * 0xD1B54A32D192ED03) & M64)
    x ^= (x << 13) & M64
    x ^= x >> 7
    x ^= (x << 17) & M64
    return (x >> 11) / float(1 << 52) - 1.0


def phantom(dims, tes, salt):
    """Per-echo magnitude and wrapped phase. The head is an ellipsoid whose magnitude takes five
    plateau levels (so iMag ties), the background is far below the noise level, and a bright cube
    in one corner is cut off from the head. The field (Hz) is a gradient plus a bump, so that
    2*pi*f*dTE spans several turns."""
    nx, ny, nz = dims
    n = nx * ny * nz
    cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
    dte = tes[1] - tes[0]
    level = np.zeros(n)
    freq = np.zeros(n)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                p = i + nx * (j + ny * k)
                x, y, z = (i - cx) / (nx / 2.0), (j - cy) / (ny / 2.0), (k - cz) / (nz / 2.0)
                r2 = x * x / 0.8 + y * y / 0.8 + z * z / 0.9
                if r2 < 1.0:
                    level[p] = float(1 + int(5.0 * (1.0 - r2)) % 5)   # plateaus 1..5
                elif i <= 1 and j <= 1 and k <= 1:
                    level[p] = 4.0                                    # the island
                else:
                    level[p] = 0.001 * (1.0 + hsh(p, salt + 1))       # background
                b = (x - 0.2) * (x - 0.2) + (y + 0.1) * (y + 0.1) + z * z
                freq[p] = (3.5 * x + 2.0 * y - 1.5 * z + 2.5 / (1.0 + 8.0 * b)) / dte \
                          + 0.04 / dte * hsh(p, salt + 2)
    mags, phases = [], []
    for e, te in enumerate(tes):
        mags.append(level * np.exp(-25.0 * te))
        ph = 2.0 * np.pi * freq * te + 0.3 + 0.05 * np.array([hsh(p, salt + 10 + e) for p in range(n)])
        phases.append(np.angle(np.exp(1j * ph)))
    return mags, phases


def run(dims, vs, inputs, params):
    """Runs the container's compiled recon; returns its output directory's files as arrays."""
    engine = os.environ.get("CONTAINER_ENGINE") or shutil.which("podman") or "docker"
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        os.makedirs(inp), os.makedirs(out)
        os.chmod(out, 0o777)
        aff = np.diag(list(vs) + [1.0])
        for name, arr in inputs.items():
            img = nib.Nifti1Image(arr, aff)
            img.header.set_slope_inter(1.0, 0.0)   # recon.m would apply nibabel's NaN slope
            nib.save(img, os.path.join(inp, name + ".nii.gz"))
        json.dump(params, open(os.path.join(inp, "params.json"), "w"))
        user = ["--userns=keep-id"] if os.path.basename(engine) == "podman" else ["--user", f"{os.getuid()}:{os.getgid()}"]
        subprocess.run([engine, "run", "--rm", "--network", "none", *user, "-w", "/tmp", "-e", "MCR_CACHE_ROOT=/tmp/mcr",
                        "-e", "QSMCI_INTERMEDIATES=1", "-v", f"{inp}:/input:ro", "-v", f"{out}:/output", IMAGE,
                        "/opt/qsm-ci/recon", "/input", "/output"], check=True, stdout=subprocess.DEVNULL)
        ld = lambda f: np.asarray(nib.load(f).dataobj)
        return {f[:-7]: ld(os.path.join(out, "intermediates", f))
                for f in os.listdir(os.path.join(out, "intermediates")) if f.endswith(".nii.gz")}


flat = lambda a: np.asarray(a).ravel(order="F")
blob = struct.pack("<I", len(CASES))
for c, (dims, vs, tes, b0) in enumerate(CASES):
    mags, phases = phantom(dims, tes, 100 * c)
    vol4 = lambda vs_: np.stack([v.reshape(dims, order="F") for v in vs_], axis=3)
    params = {"B0": b0, "B0_dir": [0, 0, 1], "TE": list(tes), "voxel_size": list(vs)}
    res = run(dims, vs, {"phase": vol4(phases), "magnitude": vol4(mags)}, params)
    ifreq = flat(res["ifreq"])
    assert np.array_equal(ifreq, ifreq.astype(np.float32)), "mexUnwrap's output is single precision"
    raw = flat(res["ifreq_raw"])
    imag = flat(res["imag"])
    noise = np.float32(imag.astype(np.float32).max()) * np.float32(5e-3)
    grown = int((imag.astype(np.float32) > noise).sum())
    moved = int((np.abs(ifreq - raw.astype(np.float32)) > 1.0).sum())
    print(dims, "voxels above the noise level", grown, "of", imag.size, "; unwrapped by a turn or more", moved)
    blob += struct.pack("<4I", *dims, len(tes)) + struct.pack(f"<{len(tes)}d", *tes) + struct.pack("<d", b0)
    for m, p in zip(mags, phases):
        blob += m.astype("<f8").tobytes() + p.astype("<f8").tobytes()
    blob += imag.astype("<f8").tobytes() + raw.astype("<f8").tobytes()
    blob += ifreq.astype("<f4").tobytes() + flat(res["totalfield_f64"]).astype("<f8").tobytes()
with open(sys.argv[1], "wb") as f:
    f.write(blob)

# Generator of tests/data/dl_field_qsmci_small.bin: the originals of the deep-learning ports that
# start from the total field or the phase, as their verified QSM-CI entries run them, on the in-vivo
# box tests/data/dl_invivo_small.bin, for tests/models_onnx/field.rs.
#
# Originals (QSMxT/QSM-CI#316: pinned source, framework versions and weight SHA-256s; each image's
# output is bit-identical run to run at 4 threads):
#   autoqsm    ghcr.io/astewartau/qsm-ci/autoqsm@sha256:8b5e8552...    AutoQSM b59f5e7 (total field)
#   nextqsm    ghcr.io/astewartau/qsm-ci/nextqsm@sha256:015ff441...    nextqsm 1.0.6 (total field)
#   bfrnet     ghcr.io/astewartau/qsm-ci/bfrnet@sha256:ab939d37...     BFRnet (total -> local field)
#   iqsm       ghcr.io/astewartau/qsm-ci/iqsm@sha256:1777cace...       iQSM bb163ef (phase)
#   iqsm-plus  ghcr.io/astewartau/qsm-ci/iqsm-plus@sha256:0541a1ba...  iQSM+ a534d94 (phase, magnitude)
#   iqfm       ghcr.io/astewartau/qsm-ci/iqfm@sha256:980ba446...       iQFM bb163ef (phase -> local field)
# (full digests in ENTRIES). An entry's image holds the authors' code and weights; its `run.sh` (and
# the wrapper scripts next to it) live in QSM-CI's `algorithms/<slug>/`, fetched here from QSM-CI at
# QSMCI_COMMIT and checked against their SHA-256s. Each runs as `fidelity/dl/harness.py` runs it
# (as `qsm-ci run` does, with OMP/MKL/OpenBLAS/TensorFlow threads fixed at 4 and no network).
#
# BFRnet's wrapper tiles a volume larger than 128³ into patches; this box is one patch, so it runs
# whole, as QSM.rs's `bfrnet` does. The iQSM family runs on one echo at a time (ECHOES: the first
# and the third), with that echo's TE: the authors' Python driver combines echoes with weights
# (mag·TE)², where QSM.rs follows their MATLAB echo fit, mag·TE² (`src/inversion/iqsm.rs`), so the
# per-echo outputs are what the two have in common, and tests/models_onnx.rs combines them itself.
# iQSM and iQFM are given no magnitude (with one echo it only scales the weight); iQSM+'s wrapper
# requires one.
#
#   python3 tests/data/dl_field_qsmci_small.py tests/data/dl_field_qsmci_small.bin
#
# machine-dependent: the PyTorch and TensorFlow CPU kernels round by instruction set (see
# tests/data/dl_dipole_qsmci_small.py; measured there at up to 1.3e-6 relative).
#
# Needs numpy, nibabel, network access (GitHub) and podman or docker (CONTAINER_ENGINE overrides; the
# regenerate-fixtures workflow runs it in Actions, where each image is removed after its last use;
# REMOVE_IMAGES=1 does that elsewhere). The inputs come from the box (B0 = 2.8946 T along z, 1 mm
# voxels, identity affine). Layout (little-endian): for each run in RUNS order, its output (f32;
# χ in ppm, or the local field in ppm for bfrnet and iqfm) at the mask voxels of
# dl_invivo_small.bin, column-major.
import hashlib, json, os, shutil, subprocess, sys, tempfile, urllib.request
import numpy as np, nibabel as nib

QSMCI_COMMIT = "271a49bbc24d70c8e31e1642d79f6d5ac74a368d"  # QSMxT/QSM-CI#316
ENTRIES = {
    "autoqsm": ("ghcr.io/astewartau/qsm-ci/autoqsm@sha256:8b5e8552a6795a59add02f9ddb392f45c7a8dcb98f465520ecda89a2fa055d54",
                {"run.sh": "fee24cfc57dbf872e3c135e76330c0d4bd60914caf7aee4be017424064d9d200",
                 "predict.py": "6fdce0feaec9b87319f72a08ab0f10f039fbde231130450b3158c39305fd2f70"}),
    "nextqsm": ("ghcr.io/astewartau/qsm-ci/nextqsm@sha256:015ff4413ee0b8b2f665194b1dc722aca6523f63cf9c5ce1c2fc5ec27891a4e4",
                {"run.sh": "fd1ed4415d29bcf8d9095ff42d4f3473e099336bfae5ebeae05b645d5f32b99a"}),
    "bfrnet": ("ghcr.io/astewartau/qsm-ci/bfrnet@sha256:ab939d3759104300dcca11421b020221dcc682bd397d77c656862cc02b5c9891",
               {"run.sh": "e5e7fe9f70347e8506d1afa5027bbde8a573c68c0ca872094c95582286d62158",
                "recon.py": "95047d705e9723c065871efe860d4c94b058291ada008c945b2bb9a43b15a3be"}),
    "iqsm": ("ghcr.io/astewartau/qsm-ci/iqsm@sha256:1777cace0a0207b9433f54e53c53cb02c8b52a33629a1aa18682c8d94e04287a",
             {"run.sh": "e71783a96263a323a189ec807f18594b3b4a6cb2eaf5decd3f75d6a7d3031276"}),
    "iqsm-plus": ("ghcr.io/astewartau/qsm-ci/iqsm-plus@sha256:0541a1ba778023a315708be3d6695425bbd31e5ce6711c3b198197256cc11429",
                  {"run.sh": "55440393c86d864e6b2d4b413cb408898f34acc5e0a8941b474419e1feba25d2"}),
    "iqfm": ("ghcr.io/astewartau/qsm-ci/iqfm@sha256:980ba446e62fb9c864c231d180ec32ed732e39528e9de8d3a44fc6f9e08b8de6",
             {"run.sh": "066d3665ef928cd787419e0b1920b5981771b2903d548887602e01300a9f8579"}),
}
# The iQSM family's echoes (0-based, of dl_invivo_small.bin's five).
ECHOES = [0, 2]
# (slug, inputs, echo or None for every echo, output)
RUNS = ([("autoqsm", ("totalfield", "mask"), None, "chimap"),
         ("nextqsm", ("totalfield", "mask"), None, "chimap"),
         ("bfrnet", ("totalfield", "mask"), None, "localfield")]
        + [("iqsm", ("phase", "mask"), e, "chimap") for e in ECHOES]
        + [("iqsm-plus", ("phase", "magnitude", "mask"), e, "chimap") for e in ECHOES]
        + [("iqfm", ("phase", "mask"), e, "localfield") for e in ECHOES])

# tests/data/dl_invivo_small.bin (scripts/dl_fixture_inputs.py)
DIMS = (36, 32, 28)
PARAMS = {"TE": [0.005, 0.011, 0.017, 0.023, 0.029], "B0": 2.8946, "B0_dir": [0.0, 0.0, 1.0],
          "voxel_size": [1.0, 1.0, 1.0]}


def invivo():
    """The box's volumes, as in scripts/dl_fixture_inputs.py's layout."""
    raw = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dl_invivo_small.bin"), "rb").read()
    n = int(np.prod(DIMS))
    vol = lambda o: np.frombuffer(raw, "<f4", n, o).reshape(DIMS, order="F")
    mask = np.frombuffer(raw, np.uint8, n, 0).reshape(DIMS, order="F")
    names = ["totalfield", "localfield", "chimap", "r2prime"]
    v = {k: vol(n + 4 * n * i) for i, k in enumerate(names)}
    v["magnitude"] = np.stack([vol(n + 4 * n * (4 + e)) for e in range(5)], -1)
    v["phase"] = np.stack([vol(n + 4 * n * (9 + e)) for e in range(5)], -1)
    assert len(raw) == n + 4 * n * 14
    v["mask"] = mask
    return v


def fetch(slug, files, dest):
    """QSM-CI's algorithms/<slug>/ wrapper files at QSMCI_COMMIT, checked against their SHA-256s."""
    os.makedirs(dest)
    for name, sha in files.items():
        url = f"https://raw.githubusercontent.com/QSMxT/QSM-CI/{QSMCI_COMMIT}/algorithms/{slug}/{name}"
        data = urllib.request.urlopen(url, timeout=60).read()
        assert hashlib.sha256(data).hexdigest() == sha, f"{url}: SHA-256 mismatch"
        open(os.path.join(dest, name), "wb").write(data)


def run(slug, inputs, params, out_names, last_use):
    """Runs one entry as fidelity/dl/harness.py does; returns its outputs."""
    (image, files), sets = ENTRIES[slug], {}
    engine = os.environ.get("CONTAINER_ENGINE") or shutil.which("podman") or "docker"
    with tempfile.TemporaryDirectory() as tmp:
        algo, inp, out = (os.path.join(tmp, d) for d in ("algo", "in", "out"))
        fetch(slug, files, algo)
        os.makedirs(inp), os.makedirs(out)
        os.chmod(out, 0o777)
        for name, arr in inputs.items():
            nib.save(nib.Nifti1Image(arr, np.eye(4)), os.path.join(inp, name + ".nii.gz"))
        json.dump(params, open(os.path.join(inp, "params.json"), "w"))
        if sets:
            json.dump(sets, open(os.path.join(inp, "config.json"), "w"))
        # QSMCI_* as qsm_ci.containers._param_env sets them, and the harness's fixed thread counts.
        env = {"QSMCI_TE": " ".join(str(t) for t in params["TE"]), "QSMCI_TE0": str(params["TE"][0]),
               "QSMCI_B0": str(params["B0"]), "QSMCI_B0_DIR": " ".join(str(x) for x in params["B0_dir"]),
               "QSMCI_VOXEL_SIZE": " ".join(str(x) for x in params["voxel_size"]),
               **{f"QSMCI_SET_{k.upper()}": str(v) for k, v in sets.items()},
               **{k: "4" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                   "RAYON_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")}}
        user = ["--userns=keep-id"] if os.path.basename(engine) == "podman" else ["--user", f"{os.getuid()}:{os.getgid()}"]
        subprocess.run([engine, "run", "--rm", "--network", "none", *user,
                        *[x for k, v in env.items() for x in ("-e", f"{k}={v}")],
                        "-v", f"{algo}:/algo:ro", "-v", f"{inp}:/input:ro", "-v", f"{out}:/output",
                        image, "bash", "/algo/run.sh"], check=True)
        res = [np.asarray(nib.load(os.path.join(out, o + ".nii.gz")).dataobj) for o in out_names]
    if last_use and (os.environ.get("REMOVE_IMAGES") or os.environ.get("GITHUB_ACTIONS") == "true"):
        subprocess.run([engine, "rmi", image], check=False)
    return res


v = invivo()
m = v["mask"].flatten(order="F") != 0
with open(sys.argv[1], "wb") as f:
    for i, (slug, names, echo, out) in enumerate(RUNS):
        inputs, params = {n: v[n] for n in names}, dict(PARAMS)
        if echo is not None:
            inputs.update({n: v[n][..., echo:echo + 1] for n in ("phase", "magnitude") if n in names})
            params["TE"] = [PARAMS["TE"][echo]]
        (res,) = run(slug, inputs, params, [out], all(r[0] != slug for r in RUNS[i + 1:]))
        assert res.shape == DIMS, (slug, res.shape)
        res = res.astype(np.float32).flatten(order="F")[m]
        assert np.isfinite(res).all(), slug
        f.write(res.astype("<f4").tobytes())
        print(f"{slug}{'' if echo is None else f' echo {echo + 1}'}: {out} in [{res.min():.4f}, {res.max():.4f}] ppm")

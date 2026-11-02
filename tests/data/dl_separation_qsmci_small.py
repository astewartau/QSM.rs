# Generator of tests/data/dl_separation_qsmci_small.bin: the deep-learning χ-separation and R2′
# originals, as their verified QSM-CI entries run them, on the in-vivo box
# tests/data/dl_invivo_small.bin, for tests/models_onnx/separation.rs.
#
# Originals (QSMxT/QSM-CI#316: the authors' ONNX exports and normalisation files, pinned by digest;
# each image's output is bit-identical run to run at 4 threads except SUSEP-Net's, whose recon.py
# uses every CPU: on this box it moves by 1e-7 relative run to run):
#   chisepnet   ghcr.io/astewartau/qsm-ci/chi-sepnet@sha256:9588a43b...  SNU-LIST 240904_xsepnet.onnx
#   susep-net   ghcr.io/astewartau/qsm-ci/susep-net@sha256:b4d17556...   SUSEP-Net
#   r2primenet  ghcr.io/astewartau/qsm-ci/r2primenet@sha256:21a051be...  SNU-LIST 240531_R2PRIMEnet.onnx
# (full digests in ENTRIES). An entry's image holds the network and its normalisation; its `run.sh`
# and `recon.py` (the sliding window around the network) live in QSM-CI's `algorithms/<slug>/`,
# fetched here from QSM-CI at QSMCI_COMMIT and checked against their SHA-256s. Each runs as
# `fidelity/dl/harness.py` runs it (as `qsm-ci run` does, with OMP/MKL/OpenBLAS threads fixed at 4
# and no network).
#
# χ-sepnet and SUSEP-Net are given the box's local field (ppm), χ (ppm), R2′ (s⁻¹) and mask.
# R2PRIMEnet's entry is given the five echoes' magnitudes and fits R2* itself, with a weighted
# log-linear least-squares fit that QSM-CI's recon.py uses as a stand-in for ARLO (QSM.rs's
# `r2primenet_from_magnitude` uses ARLO); tests/models_onnx/separation.rs repeats that fit to
# compare the network.
#
#   python3 tests/data/dl_separation_qsmci_small.py tests/data/dl_separation_qsmci_small.bin
#
# machine-dependent: ONNX Runtime's CPU kernels round by instruction set, as the PyTorch and
# TensorFlow originals of tests/data/dl_dipole_qsmci_small.py do (measured there at up to 1.3e-6
# relative), and SUSEP-Net moves by 1e-7 relative even on one machine (above).
#
# Needs numpy, nibabel, network access (GitHub) and podman or docker (CONTAINER_ENGINE overrides; the
# regenerate-fixtures workflow runs it in Actions, where each image is removed after its last use;
# REMOVE_IMAGES=1 does that elsewhere). B0 = 2.8946 T along z, 1 mm voxels, identity affine. Layout
# (little-endian): χ-sepnet's χ+ and χ− (ppm; χ− as a magnitude), SUSEP-Net's χ+ and χ−, then
# R2PRIMEnet's R2′ (s⁻¹), each f32 at the mask voxels of dl_invivo_small.bin, column-major.
import hashlib, json, os, shutil, subprocess, sys, tempfile, urllib.request
import numpy as np, nibabel as nib

QSMCI_COMMIT = "a6057fceb60b3e2833772afc0ba430a146fda700"  # QSMxT/QSM-CI#316
ENTRIES = {
    "chisepnet": ("ghcr.io/astewartau/qsm-ci/chi-sepnet@sha256:9588a43b6f99e310ef4c16bc26a8dc59751d7527bf62a1d914dfba82c788bc88",
                  {"run.sh": "3428ffd3d0758f0f8cb8a8eb59bf75de6b6f53d9f5a937cd9c37539a32ba6c8f",
                   "recon.py": "5765de017a1b4c38357f668e9689f57ac64bbee2be1567ad78882c591ad5d7ec"}),
    "susep-net": ("ghcr.io/astewartau/qsm-ci/susep-net@sha256:b4d175563f6b32f84b3af163a1ac5975cc8397393457c2ef83367db11056d975",
                  {"run.sh": "a11690bb80e2d68d2f2e867d7d65c328fe4d51bb44d209cfb72f5cea3cc9230b",
                   "recon.py": "b0fe56e0ff2b409fa3d7cd784039bae5019ab71db38c41e1b286cea63f342d91",
                   "susepnet_model.py": "03beb18035cf6a6948c586051a21742aaa404b01c4ceffb651ae1cf5988074ec"}),
    "r2primenet": ("ghcr.io/astewartau/qsm-ci/r2primenet@sha256:21a051bec5387524e262a1015b84f0d74de2404eec747bae2460617a6f7b2bea",
                   {"run.sh": "3060f86f002c39ed9e914854776662e8ac4a3f2594e91c8b91c811b3726e5acb",
                    "recon.py": "d2d48451890b2769fd3168ea8dbd68d5fa6f5d2285fb42a104696c3adbe07842"}),
}
# (slug, inputs, outputs)
RUNS = [("chisepnet", ("localfield", "chimap", "r2prime", "mask"), ("chi-para", "chi-dia")),
        ("susep-net", ("localfield", "chimap", "r2prime", "mask"), ("chi-para", "chi-dia")),
        ("r2primenet", ("magnitude", "mask"), ("r2prime",))]

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
    for i, (slug, names, outs) in enumerate(RUNS):
        res = run(slug, {n: v[n] for n in names}, PARAMS, list(outs), all(r[0] != slug for r in RUNS[i + 1:]))
        for o, r in zip(outs, res):
            assert r.shape == DIMS, (slug, o, r.shape)
            r = r.astype(np.float32).flatten(order="F")[m]
            assert np.isfinite(r).all(), (slug, o)
            f.write(r.astype("<f4").tobytes())
            print(f"{slug}: {o} in [{r.min():.4f}, {r.max():.4f}]")

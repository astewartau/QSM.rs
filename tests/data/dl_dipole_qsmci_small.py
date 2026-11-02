# Generator of tests/data/dl_dipole_qsmci_small.bin: the deep-learning dipole inversions' originals,
# as their verified QSM-CI entries run them, on the in-vivo box tests/data/dl_invivo_small.bin, for
# tests/models_onnx/dipole.rs.
#
# Originals (QSMxT/QSM-CI#316: pinned source, framework versions and weight SHA-256s; each image's
# output is bit-identical run to run at 4 threads), in SLUGS order:
#   xqsm         ghcr.io/astewartau/qsm-ci/xqsm@sha256:f28dabe6...  xQSM 5d13b36
#   qsmnet       ghcr.io/astewartau/qsm-ci/qsmnet@sha256:7081b078...  SNU-LIST/QSMnet f8782ed, QSMnet_64
#   qsmnet-plus  ghcr.io/astewartau/qsm-ci/qsmnet-plus@sha256:d63e1a6a...  the same code, QSMnet+_64
#   lpcnn        ghcr.io/astewartau/qsm-ci/lpcnn@sha256:d0878b2e...  Sulam-Group/LPCNN dd1ef2b
#   qsmgan       ghcr.io/astewartau/qsm-ci/qsmgan@sha256:c7c83e39...  QSMGAN-LupoLab 521d202
#   modl-qsm     ghcr.io/astewartau/qsm-ci/modl-qsm@sha256:75ada9f0...  Ruimin-Feng/MoDL-QSM 111a493
#   ir2qsm       ghcr.io/astewartau/qsm-ci/ir2qsm@sha256:7df8887a...  IR2QSM 1ddb147, add_noise=false
# (full digests in ENTRIES). An entry's image holds the authors' code and weights; its `run.sh` (and
# the wrapper scripts next to it) live in QSM-CI's `algorithms/<slug>/`, fetched here from QSM-CI at
# QSMCI_COMMIT and checked against their SHA-256s. Each runs as `fidelity/dl/harness.py` runs it
# (as `qsm-ci run` does, with OMP/MKL/OpenBLAS/TensorFlow threads fixed at 4 and no network).
#
# IR2QSM runs with `add_noise=false`. The authors' `IR2Unet.forward` adds noise in the decoder at
# inference (12.6 % NRMSE run to run on QSM-CI's case); with it off the network is deterministic,
# and that is the branch QSM.rs's export pins (`src/inversion/ir2qsm.rs`).
#
#   python3 tests/data/dl_dipole_qsmci_small.py tests/data/dl_dipole_qsmci_small.bin
#
# machine-dependent: the PyTorch and TensorFlow CPU kernels round by instruction set. With oneDNN and
# MKL held to AVX2 (DNNL_MAX_CPU_ISA, MKL_ENABLE_INSTRUCTIONS) instead of the AVX-512 of the Intel
# i5-11500 that made the committed .bin, xqsm, lpcnn, qsmgan and ir2qsm move by 4e-7 to 1.3e-6
# relative (max |Δ| up to 8.5e-7 ppm); qsmnet, qsmnet-plus and modl-qsm do not move. On a
# GitHub-hosted runner (regenerate-fixtures, 2026-10-30) all seven move, by 2e-7 to 9e-7 relative
# (max |Δ| up to 5.2e-7 ppm), well inside the tests' limits.
#
# Needs numpy, nibabel, network access (GitHub) and podman or docker (CONTAINER_ENGINE overrides; the
# regenerate-fixtures workflow runs it in Actions, where each image is removed after use, since
# together they take ~25 GB; REMOVE_IMAGES=1 does that elsewhere). The inputs are the box's local
# field (ppm) and mask, with B0 = 2.8946 T along z, 1 mm voxels and an identity affine. Layout
# (little-endian): for each entry in SLUGS order, its χ (f32, ppm) at the mask voxels of
# dl_invivo_small.bin, column-major.
import hashlib, json, os, shutil, subprocess, sys, tempfile, urllib.request
import numpy as np, nibabel as nib

QSMCI_COMMIT = "a6057fceb60b3e2833772afc0ba430a146fda700"  # QSMxT/QSM-CI#316
ENTRIES = {
    "xqsm": ("ghcr.io/astewartau/qsm-ci/xqsm@sha256:f28dabe6f7f41db967133a022f3eb5205d2a2bbb5fab78117d58b1fb84b26444",
             {"run.sh": "637a9bfbfaea4d6c7b365a6e8b9baf8c1921ee4b02587e132cafca0990fad6a2"}, {}),
    "qsmnet": ("ghcr.io/astewartau/qsm-ci/qsmnet@sha256:7081b0782c7cb23ec8a9fc509af44f7f3a4960eec886299af63dc950c6d0b2c3",
               {"run.sh": "0f88ebec54dfc964c7e92508796935d0f023df5ece52d6b00182abb40c93a7bc",
                "qsmnet_infer.py": "102716a56a932645bb2b286eabad65792f91a1271f2cd7d4d7104a22fefcdcaf"}, {}),
    "qsmnet-plus": ("ghcr.io/astewartau/qsm-ci/qsmnet-plus@sha256:d63e1a6ad3f0e208c9fb45e510c715fe4669cf806c52e47814e80898a4c65ea7",
                    {"run.sh": "0f88ebec54dfc964c7e92508796935d0f023df5ece52d6b00182abb40c93a7bc",
                     "qsmnet_infer.py": "102716a56a932645bb2b286eabad65792f91a1271f2cd7d4d7104a22fefcdcaf"}, {}),
    "lpcnn": ("ghcr.io/astewartau/qsm-ci/lpcnn@sha256:d0878b2e361437c9f2adc86a45949344fd19b0150e474267bafb18dbb57b2b51",
              {"run.sh": "3273d90b4d522ba6305001db6f4b678c0cfd146ca41fde57494bf1abef76bc1b",
               "recon.py": "2b2edd2e7f874f85dec2b0b24b97032ea499d143bcafa93b0c7f3a9ea1cb628a"}, {}),
    "qsmgan": ("ghcr.io/astewartau/qsm-ci/qsmgan@sha256:c7c83e392af271ed21e9a63062fbdc2a4ed2e46547798d81cc1e7e527700d380",
               {"run.sh": "21acf575f2114df3da20153215684b4b650cbbaf12ce05a1bafec914e62ae95b",
                "recon.py": "611330c14d2e8d2336ce4772feed6a5b24b9a93b16ecd40d472c656becd37a98"}, {}),
    "modl-qsm": ("ghcr.io/astewartau/qsm-ci/modl-qsm@sha256:75ada9f042b203a3398731efd1a552a974b1176dec8e1e30c683bffdeb73063e",
                 {"run.sh": "0342835e5fb88efc7ed8b5825e04817862c9099695c9e46a260a6c0e9ab68464",
                  "recon.py": "3069986eaeeefa9f1a97124100fdadd9914e6e08a09b3c467067e8e784191ede"}, {}),
    "ir2qsm": ("ghcr.io/astewartau/qsm-ci/ir2qsm@sha256:7df8887acc7cdfac4ea3ec845377c3d0b595975c3d240cd1ee63e826908c98ca",
               {"run.sh": "0d98845bfefde941ba3ca6890b90194b1c0a25abb3a9d5037e4125a31f55c822",
                "ir2qsm_infer.py": "b8606150df8f19443062ef4e14044a426b6932dfc0dc7e3151295335fd538db4"},
               {"add_noise": False}),
}
SLUGS = ["xqsm", "qsmnet", "qsmnet-plus", "lpcnn", "qsmgan", "modl-qsm", "ir2qsm"]

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


def run(slug, inputs, out_names):
    """Runs one entry as fidelity/dl/harness.py does; returns its outputs."""
    image, files, sets = ENTRIES[slug]
    engine = os.environ.get("CONTAINER_ENGINE") or shutil.which("podman") or "docker"
    with tempfile.TemporaryDirectory() as tmp:
        algo, inp, out = (os.path.join(tmp, d) for d in ("algo", "in", "out"))
        fetch(slug, files, algo)
        os.makedirs(inp), os.makedirs(out)
        os.chmod(out, 0o777)
        for name, arr in inputs.items():
            nib.save(nib.Nifti1Image(arr, np.eye(4)), os.path.join(inp, name + ".nii.gz"))
        json.dump(PARAMS, open(os.path.join(inp, "params.json"), "w"))
        if sets:
            json.dump(sets, open(os.path.join(inp, "config.json"), "w"))
        # QSMCI_* as qsm_ci.containers._param_env sets them, and the harness's fixed thread counts.
        env = {"QSMCI_TE": " ".join(str(t) for t in PARAMS["TE"]), "QSMCI_TE0": str(PARAMS["TE"][0]),
               "QSMCI_B0": str(PARAMS["B0"]), "QSMCI_B0_DIR": " ".join(str(x) for x in PARAMS["B0_dir"]),
               "QSMCI_VOXEL_SIZE": " ".join(str(x) for x in PARAMS["voxel_size"]),
               **{f"QSMCI_SET_{k.upper()}": str(v) for k, v in sets.items()},
               **{k: "4" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                   "RAYON_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")}}
        user = ["--userns=keep-id"] if os.path.basename(engine) == "podman" else ["--user", f"{os.getuid()}:{os.getgid()}"]
        subprocess.run([engine, "run", "--rm", "--network", "none", *user,
                        *[x for k, v in env.items() for x in ("-e", f"{k}={v}")],
                        "-v", f"{algo}:/algo:ro", "-v", f"{inp}:/input:ro", "-v", f"{out}:/output",
                        image, "bash", "/algo/run.sh"], check=True)
        res = [np.asarray(nib.load(os.path.join(out, o + ".nii.gz")).dataobj) for o in out_names]
    if os.environ.get("REMOVE_IMAGES") or os.environ.get("GITHUB_ACTIONS") == "true":
        subprocess.run([engine, "rmi", image], check=False)
    return res


v = invivo()
m = v["mask"].flatten(order="F") != 0
with open(sys.argv[1], "wb") as f:
    for slug in SLUGS:
        (chi,) = run(slug, {"localfield": v["localfield"], "mask": v["mask"]}, ["chimap"])
        assert chi.shape == DIMS, (slug, chi.shape)
        chi = chi.astype(np.float32).flatten(order="F")[m]
        f.write(chi.astype("<f4").tobytes())
        print(f"{slug}: χ in [{chi.min():.4f}, {chi.max():.4f}] ppm at {m.sum()} mask voxels")

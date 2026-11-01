# Generator of tests/data/chi_sep_medi_qsmci_small.bin: the SNU χ-separation toolbox v1.1.3's
# chi_sep_MEDI, as the verified QSM-CI chi-sep-medi container runs it (QSMxT/QSM-CI#351), on a
# small phantom, for tests/chi_sep_medi_qsmci_parity.rs:
#   ghcr.io/astewartau/qsm-ci/chi-sep-medi@sha256:c3afb14130929cb73fd41d9011afcce69634969b73216e996341e9c946fc8d7c
# (v4, identical to licensed MATLAB R2023b). Its recon.m: CF = 42.5774e6·B0, the local field
# ppm → Hz, option_data.qsm = chimap, N_std = MEDI Fit_ppm_complex's dp1 from the multi-echo
# magnitude, zero outside the mask (written as an intermediate), mag = the echoes'
# root-sum-of-squares, mask_CSF = [] (no CSF term), lambda = lambda_CSF = 1, no time limit; it
# writes χ+ and |χ−| as float32. The phantom is chi_sep_ilsqr_qsmci_small.py's: its magnitude
# decays at a rate that varies along z, so dp1 is not proportional to 1/RSS (QSM.rs#382).
#
#   python3 tests/data/chi_sep_medi_qsmci_small.py tests/data/chi_sep_medi_qsmci_small.bin
#
# Needs numpy, nibabel and podman or docker (CONTAINER_ENGINE overrides). The phantom is built from
# + - * / only, so the Rust test rebuilds it bit for bit. Layout (little-endian, column-major over
# the mask voxels, m of them): m (u32); the container's N_std (f64, its intermediate); χ+ and |χ−|
# (f32).
import json, os, shutil, struct, subprocess, sys, tempfile
import numpy as np, nibabel as nib

MEDI = "ghcr.io/astewartau/qsm-ci/chi-sep-medi@sha256:c3afb14130929cb73fd41d9011afcce69634969b73216e996341e9c946fc8d7c"
N, TE, B0 = (24, 22, 18), (0.01, 0.02, 0.03), 3.0


def phantom():
    """Ellipsoid with a paramagnetic and a diamagnetic blob; returns (field ppm, R2' Hz, χ_total,
    magnitudes (x, y, z, echo), mask)."""
    i, j, k = np.meshgrid(*[np.arange(v, dtype=float) for v in N], indexing="ij")
    x, y, z = i - 11.6, j - 10.4, k - 8.7
    mask = x * x / 110.0 + y * y / 90.0 + z * z / 60.0 < 1.0
    d1 = (x - 3.0) * (x - 3.0) + (y + 2.0) * (y + 2.0) + z * z
    d2 = (x + 4.0) * (x + 4.0) + (y - 3.0) * (y - 3.0) + (z - 1.0) * (z - 1.0)
    para = 0.03 + 0.06 / (1.0 + d1 / 8.0)
    dia = 0.02 + 0.05 / (1.0 + d2 / 10.0)
    field = 0.002 * z + 0.02 / (1.0 + d1 / 6.0) - 0.015 / (1.0 + d2 / 8.0) + 0.0001 * x * y
    r2p = 137.0 * (para + dia) + 0.5 * x / 10.0
    chi = para - dia
    m0 = 1.0 + 0.01 * x - 0.005 * y
    mag = np.stack([m0 / (1.0 + (20.0 + 0.5 * z) * t) for t in TE], axis=-1)
    f = lambda a: a * mask
    return f(field), f(r2p), f(chi), mag * mask[..., None], mask


def run(image, inputs, params):
    engine = os.environ.get("CONTAINER_ENGINE") or shutil.which("podman") or "docker"
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        os.makedirs(inp), os.makedirs(out)
        os.chmod(out, 0o777)
        for name, arr in inputs.items():
            nib.save(nib.Nifti1Image(arr, np.eye(4)), os.path.join(inp, name + ".nii.gz"))
        json.dump(params, open(os.path.join(inp, "params.json"), "w"))
        user = ["--userns=keep-id"] if os.path.basename(engine) == "podman" else ["--user", f"{os.getuid()}:{os.getgid()}"]
        subprocess.run([engine, "run", "--rm", "--network", "none", *user, "-e", "MCR_CACHE_ROOT=/tmp/mcr", "-e", "HOME=/tmp",
                        "-e", "QSMCI_INTERMEDIATES=1", "-v", f"{inp}:/input:ro", "-v", f"{out}:/output", image,
                        "/opt/qsm-ci/recon", "/input", "/output"], check=True, stdout=subprocess.DEVNULL)
        ld = lambda f: np.asarray(nib.load(os.path.join(out, f)).dataobj)
        res = {"para": ld("chi-para.nii.gz"), "dia": ld("chi-dia.nii.gz")}
        res["n_std"] = ld(os.path.join("intermediates", "n_std.nii.gz"))
        return res




field, r2p, chi, mag, mask = phantom()
inputs = {"localfield": field, "r2prime": r2p, "chimap": chi, "magnitude": mag, "mask": mask.astype(np.uint8)}
params = {"B0": B0, "B0_dir": [0, 0, 1], "TE": list(TE), "voxel_size": [1, 1, 1]}
m = mask.ravel(order="F")
at = lambda a, t: a.ravel(order="F")[m].astype(t).tobytes()
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<I", int(m.sum())))
    r = run(MEDI, inputs, params)
    f.write(at(r["n_std"], "<f8") + at(r["para"], "<f4") + at(r["dia"], "<f4"))
    print("chi-sep-medi: χ+", r["para"][mask].mean(), "|χ−|", r["dia"][mask].mean())

# Generator of tests/data/dl_snu_route_small.bin: the χ-separation toolbox's route (QSMnetME and
# χ-sepnet by `chi_sepnet_general`), as QSM-CI's `qsmnetme` and `chisepnet` entries run it, on the
# in-vivo box tests/data/dl_invivo_small.bin, for tests/models_onnx/{dipole,separation}.rs.
#
# The reference is QSM-CI's algorithms/chisepnet/chisep_route.py (identical to
# algorithms/qsmnetme/chisep_route.py) at QSMCI_COMMIT, checked against its SHA-256: a NumPy + ONNX
# Runtime reproduction of the toolbox v1.1.3's chi_sepnet_general P-code, recovered by black-box
# probing in licensed MATLAB R2023b and matching it to 1.4e-5 to 2.1e-5 % NRMSE (astewartau/QSM.rs#377).
# The entries' image holds the toolbox's own files, which are not public; this generator runs the same
# code on the registry's copies of the networks instead (Hugging Face qsmxt/qsm-onnx-weights at
# HF_REVISION, SHA-256 checked): the toolbox's graphs with their spatial axes re-declared dynamic,
# which give the same output (the route re-declares them in memory anyway), and the normalisation
# file's values, which chisep_route.NORM holds.
#
# Cases (CF = 42.5774806e6·B0, B0 = 2.8946 T, the box's echoes and mask):
#   A  the box as acquired: 1 mm, B0 along z;
#   B  the same arrays declared at 0.9×1.0×1.1 mm with B0 along (0.1, 0.2, 0.97)/|·|, which checks the
#      rotation and the anisotropic resampling (a test of the computation, not a real acquisition).
# Outputs: QSMnetME χ (A, B); χ-sepnet-R2′ χ+ and |χ−| from the box's R2′ (A, B); χ-sepnet-R2* χ+ and
# |χ−| (A) from an R2* fitted here (log-linear least squares over the five echoes, weighted by
# magnitude², not clipped: the route takes R2* as it is).
#
#   python3 tests/data/dl_snu_route_small.py tests/data/dl_snu_route_small.bin
#
# pip: scipy==1.18.1 onnx==1.23.2 onnxruntime==1.31.1
#
# machine-dependent: ONNX Runtime's CPU kernels round by instruction set (as in
# tests/data/dl_separation_qsmci_small.py).
#
# Needs numpy, scipy, onnx, onnxruntime and network access (GitHub, Hugging Face); measured with
# Python 3.14, NumPy 2.5.4, SciPy 1.18.1, ONNX 1.23.2, ONNX Runtime 1.31.1. Layout
# (little-endian): the R2* input as f32 over the whole box (column-major), then as f32 at the mask
# voxels of dl_invivo_small.bin: QSMnetME A, QSMnetME B, χ-sepnet-R2′ χ+ A, |χ−| A, χ+ B, |χ−| B,
# χ-sepnet-R2* χ+ A, |χ−| A.
import hashlib, importlib.util, os, sys, tempfile, urllib.request
import numpy as np

QSMCI_COMMIT = "d43555925c8e957717e0cfdeaa81a5ac4dca4735"  # QSMxT/QSM-CI parity/qsmnetme-377
ROUTE_SHA = "ee244927ac0f7e723d1fc1cbd85543f220316c8ac7a27c8d0a3439642450dbcc"
HF_REVISION = "e7116c8b8be0ad0aa4155d68cb2f9c23363be362"
NETS = {"qsmnet": ("qsmnetme.onnx", "c2f030b625b2fc83f3ae45b38c996e006a069444d2da32d2b6022045c92c01e5"),
        "xsepnet": ("chi-sepnet.onnx", "5b442fdfdb88f50ec9149b384dabd0d2d9adb6983947d9cfa58382b78676bed9"),
        "r2primenet": ("r2primenet.onnx", "44cb5e67d1c68dae87a5f532e501dc0d4bf627d91f9c8ba35a56df5ac7ec3cd0")}
DIMS = (36, 32, 28)
TES = np.array([0.005, 0.011, 0.017, 0.023, 0.029])
B0 = 2.8946
CF = 42.5774806e6 * B0
DR = 114.0
CASES = {"A": ((1.0, 1.0, 1.0), (0.0, 0.0, 1.0)), "B": ((0.9, 1.0, 1.1), (0.1, 0.2, 0.97))}


def get(url, sha):
    data = urllib.request.urlopen(url, timeout=300).read()
    assert hashlib.sha256(data).hexdigest() == sha, f"{url}: SHA-256 mismatch"
    return data


def invivo():
    raw = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dl_invivo_small.bin"), "rb").read()
    n = int(np.prod(DIMS))
    vol = lambda i: np.frombuffer(raw, "<f4", n, n + 4 * n * i).reshape(DIMS, order="F").astype(np.float64)
    assert len(raw) == n + 4 * n * 14
    mask = np.frombuffer(raw, np.uint8, n, 0).reshape(DIMS, order="F") != 0
    return mask, vol(1), vol(3), np.stack([vol(4 + e) for e in range(5)], -1)


def r2star_fit(mag):
    m = np.maximum(mag, 1e-9)
    w = m * m
    lm = np.log(m)
    sw = w.sum(-1)
    tbar = (w * TES).sum(-1) / sw
    lbar = (w * lm).sum(-1) / sw
    cov = (w * (TES - tbar[..., None]) * (lm - lbar[..., None])).sum(-1)
    var = (w * (TES - tbar[..., None]) ** 2).sum(-1)
    return -np.where(var > 0, cov / np.where(var > 0, var, 1), 0.0)


with tempfile.TemporaryDirectory() as tmp:
    src = get(f"https://raw.githubusercontent.com/QSMxT/QSM-CI/{QSMCI_COMMIT}/algorithms/chisepnet/chisep_route.py", ROUTE_SHA)
    open(os.path.join(tmp, "chisep_route.py"), "wb").write(src)
    spec = importlib.util.spec_from_file_location("chisep_route", os.path.join(tmp, "chisep_route.py"))
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)
    for key, (name, sha) in NETS.items():
        open(os.path.join(tmp, name), "wb").write(
            get(f"https://huggingface.co/qsmxt/qsm-onnx-weights/resolve/{HF_REVISION}/{name}", sha))
    nets = {k: cr.load_net(os.path.join(tmp, name), threads=4) for k, (name, _) in NETS.items()}

    mask, field_ppm, r2prime, mag = invivo()
    r2star = r2star_fit(mag).astype(np.float32).astype(np.float64)
    hz = field_ppm * CF * 1e-6
    outs = []
    for case in ("A", "B"):
        vs, b0 = CASES[case]
        b = np.array(b0) / np.linalg.norm(b0)
        outs.append(("qsmnetme " + case, cr.qsmnetme(hz, mask, b, CF, np.array(vs), nets)))
    for case in ("A", "B"):
        vs, b0 = CASES[case]
        b = np.array(b0) / np.linalg.norm(b0)
        xp, xn, *_ = cr.route(hz, r2prime, mask, DR, b, CF, np.array(vs), True, nets)
        outs += [("chisepnet-R2' chi+ " + case, xp), ("chisepnet-R2' |chi-| " + case, xn)]
    vs, b0 = CASES["A"]
    xp, xn, *_ = cr.route(hz, r2star, mask, DR, np.array(b0), CF, np.array(vs), False, nets)
    outs += [("chisepnet-R2* chi+ A", xp), ("chisepnet-R2* |chi-| A", xn)]

m = mask.flatten(order="F")
with open(sys.argv[1], "wb") as f:
    f.write(r2star.astype("<f4").tobytes(order="F"))
    for name, r in outs:
        r = r.astype(np.float32).flatten(order="F")[m]
        assert np.isfinite(r).all(), name
        f.write(r.astype("<f4").tobytes())
        print(f"{name}: [{r.min():.4f}, {r.max():.4f}]")

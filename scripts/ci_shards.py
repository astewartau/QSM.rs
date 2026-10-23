#!/usr/bin/env python3
"""Pack the selected integration tests into a few CI jobs, and run one job's share.

integration-tests.yml builds the test binaries once per feature set (the `build-plain` and
`build-onnx` jobs) and runs them in a handful of shard jobs, rather than one job per test that
each compiles the crate. This script holds what used to be spread over one job definition per
test family: which binary a test is in, its arguments and environment, whether it gates the
merge, its time limit, its retries and the metrics row it leaves when it fails.

A unit is what one matrix entry ran before: a test name in tests/integration_tests.rs or
tests/dl_models.rs, or a whole test binary (the χ-separation binaries, `oblique_sagittal`,
`multislice_2d`). Each unit still runs as its own process with the same command line, and still
leaves `metrics/<unit>.csv` (its RESULT: rows, then a STATUS: line) and whatever it writes under
`slices/`, so the summary job reads exactly what it read before.

Usage:
  ci_shards.py plan     read the selection from the environment (the JSON lists detect-changes
                        computes: BGREMOVE, INVERSION, COMBINED, PIPELINE, CHISEP, DL; and
                        RUN_BET, RUN_GEOMETRY, RUN_MULTISLICE = true/false) and print, or with
                        --github-output write, `plain_shards` and `onnx_shards`: JSON lists of
                        {"id", "units"} for the two shard matrices
  ci_shards.py run UNIT...
                        run these units from ./bins (the build job's binaries), writing
                        metrics/ and slices/ in the current directory. Exits non-zero when a
                        gating unit failed; a non-gating one (deep-learning models, QSMART) is
                        reported as failed in its metrics and does not fail the job.
                        CI_SHARDS_BACKOFF (seconds, default 20) scales the pause between
                        attempts of a retried unit
  ci_shards.py spec UNIT
                        print the command line, environment and gating a unit runs with
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# Median seconds each unit spent in its test binary over three full runs on GitHub's hosted
# runners (2026-10-23: runs 39888473649, 39902014692, 39895862373). Only the packing reads them;
# a runner can be several times slower or faster than this on the day. A unit missing here
# (a new test) is packed as DEFAULT_SECONDS.
SECONDS = {
    "chisep_orientation": 377, "chisep_qsmci": 1027, "chisep_r2primenet": 100,
    "chisep_relaxometry_qsmci": 253, "chisep_xsepnet": 151,
    "multislice_2d": 1, "oblique_sagittal": 1,
    "test_bet": 2, "test_bgremove_ismv": 11, "test_bgremove_lbv": 12, "test_bgremove_msmv": 8,
    "test_bgremove_pdf": 20, "test_bgremove_resharp": 80, "test_bgremove_sharp": 2,
    "test_bgremove_vsharp": 6, "test_bgremove_vsharp_no_deconvolve": 6,
    "test_bias_correction": 4, "test_combined_tfi": 100, "test_combined_tgv": 195,
    "test_dl_autoqsm": 295, "test_dl_bfrnet": 281, "test_dl_hdbet": 58, "test_dl_iqfm": 21,
    "test_dl_iqsm": 79, "test_dl_iqsm_plus": 108, "test_dl_ir2qsm": 112, "test_dl_lpcnn": 170,
    "test_dl_mindgrab": 104, "test_dl_modl_qsm": 162, "test_dl_nextqsm": 233,
    "test_dl_qsmgan": 46, "test_dl_qsmnet": 122, "test_dl_qsmnet_plus": 138,
    "test_dl_r2primenet": 60, "test_dl_rs2net": 51, "test_dl_synthseg": 86,
    "test_dl_synthseg_window": 428, "test_dl_xqsm": 125,
    "test_inversion_amp_pe": 98, "test_inversion_fansi": 105, "test_inversion_fansi_tgv": 292,
    "test_inversion_hdqsm": 58, "test_inversion_heidi": 160, "test_inversion_ilsqr": 17,
    "test_inversion_l1qsm": 86, "test_inversion_lsqr": 20, "test_inversion_medi": 216,
    "test_inversion_ndi": 27, "test_inversion_nltv": 79, "test_inversion_rts": 3,
    "test_inversion_star": 4, "test_inversion_tikhonov": 1, "test_inversion_tkd": 2,
    "test_inversion_tkd_cropped": 2, "test_inversion_tsvd": 2, "test_inversion_tv": 14,
    "test_inversion_whqsm": 125, "test_motion_correction": 20, "test_multiorient_cosmos": 3,
    "test_multiorient_coverage": 1, "test_multiorient_noise": 4, "test_multiorient_register": 9,
    "test_multiorient_register_capture_range": 17, "test_multiorient_sti": 6,
    "test_n4_bias_correction": 6, "test_pipeline_harperella": 44,
    "test_pipeline_iharperella": 37, "test_pipeline_masking": 4, "test_pipeline_qsmart": 528,
    "test_pipeline_qsmart_tikhonov": 442, "test_pipeline_romeo_b0": 4, "test_pipeline_tgv": 457,
    "test_pipeline_tgv_single_echo": 378, "test_pipeline_two_pass": 11,
    "test_pipeline_unwrap_bfr_bestpath": 9, "test_pipeline_unwrap_bfr_laplacian_neumann": 10,
    "test_pipeline_unwrap_bfr_prelude": 6, "test_pipeline_unwrap_bfr_romeo": 9,
    "test_pipeline_unwrap_bfr_stage_laplacian": 9, "test_r2star_t2star": 1,
    "test_robust_echo_combination": 12, "test_smwi": 2, "test_swi": 9,
}
DEFAULT_SECONDS = 120

# A shard's share of test time, in seconds. A job costs about a minute before its first test
# (checkout, artifacts, data), so shards much shorter than this mostly pay for that; much longer
# and the run waits on them. A unit longer than this gets a shard to itself, and its group's
# shards are filled up to FILL of its length.
TARGET_SECONDS = 480
FILL = 0.75


def family(unit):
    if unit == "test_bet":
        return "bet"
    if unit.startswith("test_dl_"):
        return "dl"
    if unit.startswith("chisep_"):
        return "chisep"
    if unit in ("oblique_sagittal", "multislice_2d"):
        return unit
    return "integration"


def model_cache(cwd):
    """The job's weights cache when it set one (integration-tests.yml does), else ./.model-cache."""
    return os.environ.get("QSM_MODEL_CACHE") or str(cwd / ".model-cache")


def spec(unit, cwd):
    """How a unit runs: the same command line, environment and outcome rules its job had."""
    fam = family(unit)
    env = {}
    if fam == "integration":
        binary, args = "integration_tests", [unit, "--ignored", "--exact", "--nocapture"]
        fallback = f"RESULT:{unit},-,-,-,-,-"
        # The old pipeline and combined jobs stopped a test at 30 min; bgremove and inversion had
        # no limit of their own (the job's six hours).
        timeout = 60 if unit.startswith(("test_bgremove_", "test_inversion_", "test_multiorient_")) else 30
        # QSMART still runs and reports metrics/figures, but does not gate merges: it has a known
        # ~0.39 corr (below the 0.5 target) pending accuracy work.
        gate = "qsmart" not in unit
        attempts = 1
    elif fam == "bet":
        # Not --exact: the old bet-test job ran `test_bet` as a name filter.
        binary, args = "integration_tests", ["test_bet", "--ignored", "--nocapture"]
        fallback, timeout, gate, attempts = "RESULT:BET,-,-,-,-,-", 60, True, 1
    elif fam == "chisep":
        binary, args = unit, ["--ignored", "--nocapture"]
        env = {"QSMCI_CHISEP": str(cwd / "qsmci-chisep"), "QSM_MODEL_CACHE": model_cache(cwd)}
        fallback, timeout, gate, attempts = f"RESULT:{unit},-,-,-,-", 40, True, 1
    elif fam == "dl":
        # Informational: weight downloads are network-dependent and the deep-learning metrics do
        # not gate merges. Three attempts, as weight hosts throttle concurrent fetches with
        # spurious 403s; the model cache keeps what an earlier attempt fetched.
        binary, args = "dl_models", [unit, "--ignored", "--exact", "--nocapture"]
        env = {"QSM_MODEL_CACHE": model_cache(cwd)}
        fallback, timeout, gate, attempts = f"RESULT:{unit},-,-,-,-,-", 40, False, 3
    else:  # oblique_sagittal, multislice_2d: pass/fail sections, no fallback row
        binary, args = unit, ["--ignored", "--nocapture"]
        if unit == "multislice_2d":
            env = {"QSM_BIDS_2D": "bids"}
        fallback, timeout, gate, attempts = None, 30, True, 1
    return {"binary": binary, "args": args, "env": env, "fallback": fallback,
            "timeout_min": timeout, "gate": gate, "attempts": attempts}


def pack(units, target=TARGET_SECONDS):
    """Longest-first onto the least-loaded shard; as many shards as the total needs."""
    if not units:
        return []
    units = sorted(set(units), key=lambda u: (-SECONDS.get(u, DEFAULT_SECONDS), u))
    total = sum(SECONDS.get(u, DEFAULT_SECONDS) for u in units)
    # The longest unit bounds the group's wall time whatever the packing, so shards nearly as
    # long cost no wall time and save jobs. Not quite as long: a runner can be twice as slow as
    # the table on the day, and a full shard then outlasts the longest unit (seen: the other
    # χ-separation binaries packed to chisep_qsmci's 17 min took 20).
    target = max(target, FILL * SECONDS.get(units[0], DEFAULT_SECONDS))
    n = max(1, min(len(units), math.ceil(total / target)))
    shards = [[0, []] for _ in range(n)]
    for u in units:
        s = min(shards, key=lambda s: s[0])
        s[0] += SECONDS.get(u, DEFAULT_SECONDS)
        s[1].append(u)
    shards.sort(key=lambda s: -s[0])
    return shards


def plan(args):
    def lst(name):
        return json.loads(os.environ.get(name) or "[]")

    def flag(name):
        return os.environ.get(name) == "true"

    plain = lst("BGREMOVE") + lst("INVERSION") + lst("COMBINED") + lst("PIPELINE")
    plain += [u for u, f in (("test_bet", "RUN_BET"), ("oblique_sagittal", "RUN_GEOMETRY"),
                             ("multislice_2d", "RUN_MULTISLICE")) if flag(f)]
    # χ-separation and deep-learning units share the onnx build; they need different data
    # (the chisep phantom vs the BIDS phantom), so they are packed apart.
    chisep, dl = lst("CHISEP"), lst("DL")

    out = {}
    for key, groups in (("plain_shards", [("p", plain)]),
                        ("onnx_shards", [("c", chisep), ("d", dl)])):
        shards = []
        for prefix, units in groups:
            packed = pack(units)
            for i, (secs, us) in enumerate(packed, 1):
                shards.append({"id": f"{prefix}{i}", "units": " ".join(us),
                               "data": "chisep" if prefix == "c" else "bids",
                               "est_min": round(secs / 60, 1)})
        out[key] = shards
        print(f"{key}:", file=sys.stderr)
        for s in shards:
            print(f"  {s['id']:4} ~{s['est_min']:5.1f} min  {s['units']}", file=sys.stderr)
    if args.github_output:
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            for k, v in out.items():
                fh.write(f"{k}={json.dumps(v, separators=(',', ':'))}\n")
    else:
        print(json.dumps(out, indent=1))


def run_once(cmd, env, timeout_s, log):
    """Run cmd, echoing its combined output and keeping it; returns (ok, output lines)."""
    lines = []
    # Its own process group, so a time-out kills whatever it started too, and the pipe closes.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            bufsize=1, text=True, errors="replace", start_new_session=True)
    timed_out = threading.Event()

    def kill():
        timed_out.set()
        os.killpg(proc.pid, signal.SIGKILL)

    timer = threading.Timer(timeout_s, kill)
    timer.start()
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            lines.append(line.rstrip("\n"))
            log.write(line)
        proc.wait()
    finally:
        timer.cancel()
    sys.stdout.flush()
    if timed_out.is_set():
        print(f"::error::{Path(cmd[0]).name} timed out after {timeout_s / 60:g} min")
        return False, lines
    return proc.returncode == 0, lines


def run(args):
    cwd = Path.cwd()
    Path("metrics").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    failed_gating, report = [], []
    for unit in args.units:
        s = spec(unit, cwd)
        cmd = [str(cwd / "bins" / s["binary"])] + s["args"]
        env = dict(os.environ, **s["env"])
        t0 = time.monotonic()
        ok, lines = False, []
        with open(f"logs/{unit}.log", "w") as log:
            for attempt in range(1, s["attempts"] + 1):
                print(f"::group::{unit}" + (f" attempt {attempt}" if s["attempts"] > 1 else ""))
                print("+ " + " ".join([f"{k}={v}" for k, v in s["env"].items()] + cmd), flush=True)
                ok, lines = run_once(cmd, env, s["timeout_min"] * 60, log)
                print("::endgroup::", flush=True)
                if ok or attempt == s["attempts"]:
                    break
                print(f"{unit}: attempt {attempt} failed; backing off before retry", flush=True)
                time.sleep(attempt * int(os.environ.get("CI_SHARDS_BACKOFF", "20")))
        results = [l for l in lines if l.startswith("RESULT:")]
        if not results and s["fallback"]:
            results = [s["fallback"]]
        status = "success" if ok else "failure"
        with open(f"metrics/{unit}.csv", "w") as fh:
            for r in results:
                fh.write(r + "\n")
            fh.write(f"STATUS:{status}\n")
        mins = (time.monotonic() - t0) / 60
        report.append((unit, status, s["gate"], mins))
        if not ok:
            if s["gate"]:
                failed_gating.append(unit)
                print(f"::error::{unit} failed")
            else:
                print(f"::warning::{unit} failed (informational, does not gate)")
    print("\nunit                                        status   gates  minutes")
    for unit, status, gate, mins in report:
        print(f"{unit:43} {status:8} {'yes' if gate else 'no':5} {mins:7.1f}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write("| Unit | Status | Gates | Minutes |\n|---|---|---|---|\n")
            for unit, status, gate, mins in report:
                fh.write(f"| `{unit}` | {status} | {'yes' if gate else 'no'} | {mins:.1f} |\n")
    if failed_gating:
        print(f"Gating units failed: {' '.join(failed_gating)}")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("plan")
    pp.add_argument("--github-output", action="store_true")
    pr = sub.add_parser("run")
    pr.add_argument("units", nargs="+")
    ps = sub.add_parser("spec")
    ps.add_argument("unit")
    a = p.parse_args()
    if a.cmd == "plan":
        plan(a)
    elif a.cmd == "run":
        run(a)
    else:
        print(json.dumps(spec(a.unit, Path.cwd()), indent=1))


if __name__ == "__main__":
    main()

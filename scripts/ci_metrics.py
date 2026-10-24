#!/usr/bin/env python3
"""Record an integration-test run's metrics as JSON, and compare a run against a main record.

    ci_metrics.py record <metrics-dir> > run.json
    ci_metrics.py compare <run.json> [<baseline.json>] > delta.md

`record` reads the metrics/<test>.csv files the jobs in .github/workflows/integration-tests.yml
upload (RESULT: rows and a STATUS: line) and names each RESULT column. The column layouts are the
ones scripts/integration_report.sh tables; a test this file does not know keeps its values under
v1, v2, ... so nothing is dropped, and adding it to LAYOUTS below gives them names. Run context
comes from the environment (see context()). The records live on the `ci-figures` branch under
metrics/ and are never pruned.

`compare` writes the "Change vs main" section of the PR comment: every metric that moved, with
those past the thresholds in THRESHOLDS listed first, and a warning when the two runs did not use
the same test data or test code (then the numbers are not like-for-like).
"""
import json
import os
import re
import sys
from datetime import datetime, timezone

SCHEMA = 1

# Column names per test, after the leading result name. Mirrors scripts/integration_report.sh.
FIELD_MAP = ["rmse", "nrmse_range", "nrmse_chi_pct", "corr", "xsim", "time_s"]
CHI = ["nrmse_pct", "detrend_pct", "tissue_pct", "blood_pct", "dgm_pct", "dgm_lin", "calc_dev",
       "streak", "corr", "xsim", "time_s"]
MASK = ["dice", None, None, "time_s"]
CHISEP = ["corr", "xsim", "nrmse_pct", "time_s"]
FIELD_SHORT = ["nrmse_range", "corr", "xsim", "time_s"]

LAYOUTS = {}
for t in ["test_bgremove_sharp", "test_bgremove_resharp", "test_bgremove_vsharp",
          "test_bgremove_vsharp_deconvolve", "test_bgremove_pdf", "test_bgremove_ismv",
          "test_bgremove_lbv", "test_pipeline_harperella",
          "test_pipeline_iharperella", "test_pipeline_romeo_b0", "test_pipeline_unwrap_bfr_romeo",
          "test_pipeline_unwrap_bfr_laplacian_neumann", "test_pipeline_unwrap_bfr_bestpath",
          "test_pipeline_unwrap_bfr_prelude", "test_pipeline_unwrap_bfr_stage_laplacian",
          "test_dl_iqfm", "test_dl_bfrnet"]:
    LAYOUTS[t] = FIELD_MAP
for t in ["test_bet", "test_dl_hdbet", "test_dl_mindgrab", "test_dl_rs2net",
          "test_pipeline_masking"]:
    LAYOUTS[t] = MASK
for t in ["chisep_qsmci", "chisep_relaxometry_qsmci", "chisep_xsepnet", "chisep_r2primenet",
          "chisep_orientation"]:
    LAYOUTS[t] = CHISEP
for t in ["test_motion_correction", "test_robust_echo_combination"]:
    LAYOUTS[t] = FIELD_SHORT
for t in ["test_dl_synthseg", "test_dl_synthseg_window"]:
    LAYOUTS[t] = ["dice", "structures_found", "structures_total", "time_s"]
LAYOUTS["test_multiorient_register_capture_range"] = [
    "capture_range_deg", "first_failure_deg", "worst_rotation_err_deg", "time_s"]
# Result rows whose layout differs from the rest of their test's.
ROW_LAYOUTS = {
    "Motion recovery": ["rotation_err_deg", "translation_err_mm", "b0_direction_err_deg",
                        "tissue_displacement_mm", "time_s"],
    "Echo quality": ["score_spread", "score_spread_clean", "echo_flagged", "fraction_downweighted"],
    "SynthSeg window vs whole": ["mean_structure_dice", "min_structure_dice"],
}
# Utilities: only the last column (time) is a measurement.
UTILITY_TESTS = {"test_bias_correction", "test_n4_bias_correction", "test_swi", "test_smwi",
                 "test_r2star_t2star", "test_dl_r2primenet"}
# Every other RESULT row of 12 values is a susceptibility map (inversion, combined, pipeline,
# multi-orientation and DL dipole-inversion tests).


def value(s):
    s = s.strip()
    if s in ("", "-"):
        return None
    try:
        return float(s.rstrip("%"))
    except ValueError:
        return s


def parse_row(test, payload):
    """One RESULT payload -> (row name, {metric: value})."""
    if re.match(r"\w+=", payload):
        # key=value results (multislice_2d), several to a line; all go into one row. A value
        # that starts with a number is that number; anything else (a refusal message) is kept.
        parts = re.split(r"(?:^|\s)(\w+)=", payload)[1:]
        out = {}
        for k, v in zip(parts[::2], parts[1::2]):
            num = re.match(r"-?\d+(\.\d+)?(?=\s|$)", v.strip())
            out[k] = float(num.group(0)) if num else v.strip()
        return "values", out
    name, *vals = payload.split(",")
    if test in UTILITY_TESTS:
        names = [None] * (len(vals) - 1) + ["time_s"]
    else:
        names = ROW_LAYOUTS.get(name) or LAYOUTS.get(test)
        if names is None and len(vals) == 11:
            names = CHI
        if names is None or len(names) != len(vals):
            names = [f"v{i + 1}" for i in range(len(vals))]
    return name, {k: x for k, v in zip(names, vals) if k and (x := value(v)) is not None}


def read_metrics(d):
    tests = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".csv"):
            continue
        test = f[:-4]
        status, rows = None, {}
        with open(os.path.join(d, f), encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("STATUS:"):
                    status = line[7:].strip() or None
                elif line.startswith("RESULT:"):
                    name, metrics = parse_row(test, line[7:])
                    if name in rows:  # multislice_2d: its key=value lines share one row
                        rows[name].update(metrics)
                    else:
                        rows[name] = metrics
        tests[test] = {"status": status, "rows": rows}
    return tests


def context():
    env = os.environ.get
    pr = env("PR_NUMBER") or None
    return {
        "commit": env("HEAD_SHA") or env("GITHUB_SHA"),      # the branch tip that was tested
        "tested_commit": env("TESTED_SHA"),                  # PRs: the merge with the base
        "base_commit": env("BASE_SHA") or None,
        "ref": env("GITHUB_REF"),
        "event": env("GITHUB_EVENT_NAME"),
        "pr": int(pr) if pr else None,
        "run_id": int(env("GITHUB_RUN_ID", "0")),
        "run_attempt": int(env("GITHUB_RUN_ATTEMPT", "1")),
        "run_url": env("RUN_URL"),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        # What makes two records comparable: the same data in and the same tests measuring.
        "test_data": {"cache_key": env("TEST_DATA_KEY") or None,
                      "sha256": env("TEST_DATA_SHA256") or None},
        "chisep_data": {"cache_key": env("CHISEP_DATA_KEY") or None,
                        "sha256": env("CHISEP_DATA_SHA256") or None},
        "tests_tree": env("TESTS_TREE") or None,          # git tree of tests/
        "src_tree": env("SRC_TREE") or None,              # git tree of src/
        "unit_tests": {"passed": env("TESTS_PASSED") or None,
                       "coverage_pct": value(env("COVERAGE") or "")},
    }


def record(d):
    out = {"schema": SCHEMA, "context": context(), "tests": read_metrics(d)}
    json.dump(out, sys.stdout, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    sys.stdout.write("\n")


# --- compare ----------------------------------------------------------------------------------

# (kind, threshold, better): "abs" flags |Δ| >= threshold, "rel" flags |Δ|/|main| >= threshold.
# better is +1 when higher is better, -1 when lower is, 0 when the metric has no direction.
def threshold(metric):
    m = metric.lower()
    if m == "time_s":
        return ("rel", 0.5, -1)  # runner speed varies; only a halving/doubling is flagged
    if m in ("corr", "xsim", "dice", "mean_structure_dice", "min_structure_dice") or m.endswith("_corr"):
        return ("abs", 0.005, +1)
    if "nrmse" in m or m == "rmse" or "err" in m:
        return ("rel", 0.02, -1)
    return ("rel", 0.02, 0)


THRESHOLD_NOTE = ("correlation, XSIM and Dice: an absolute change of 0.005 or more; NRMSE, RMSE "
                  "and error measures: 2% relative; other metrics: 2% relative; and any change of "
                  "pass/fail. Times are listed apart, when they change by 50% and at least 1 s")


def fmt(v):
    if v is None:
        return "–"
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e6:
            return str(int(v))
        return f"{v:.4g}"
    return str(v)


def compare(cur_path, base_path):
    cur = json.load(open(cur_path, encoding="utf-8"))
    print("### Change vs main")
    if not base_path or not os.path.exists(base_path):
        print("")
        print("_No main record to compare with yet; this run's metrics are recorded for the next._")
        print("")
        return
    base = json.load(open(base_path, encoding="utf-8"))
    bc, cc = base["context"], cur["context"]
    sha = (bc.get("commit") or "")[:7]
    link = f"[{sha}]({bc['run_url']})" if bc.get("run_url") else f"`{sha}`"
    print("")
    print(f"_Against the latest main record: {link}, {bc.get('date', '?')[:10]}. "
          f"Flagged: {THRESHOLD_NOTE}._")
    print("")
    warn = []
    for key, label in (("test_data", "test data"), ("chisep_data", "χ-separation data")):
        b, c = (bc.get(key) or {}).get("sha256"), (cc.get(key) or {}).get("sha256")
        if b and c and b != c:
            warn.append(f"the {label} differ (sha256 `{b[:12]}` on main, `{c[:12]}` here)")
    if bc.get("tests_tree") and cc.get("tests_tree") and bc["tests_tree"] != cc["tests_tree"]:
        warn.append("the test code (`tests/`) differs from main's, so a test may measure "
                    "something different")
    if warn:
        print("> [!WARNING]")
        print("> Not like-for-like: " + "; ".join(warn) + ".")
        print("")

    flagged, moved, timing, unchanged = [], [], [], 0
    new_tests, gone_tests = [], []
    for test in sorted(set(cur["tests"]) | set(base["tests"])):
        if test not in base["tests"]:
            new_tests.append(test)
            continue
        if test not in cur["tests"]:
            gone_tests.append(test)  # not selected for this PR, usually
            continue
        bt, ct = base["tests"][test], cur["tests"][test]
        if bt.get("status") != ct.get("status"):
            flagged.append((":warning:", test, "", "status", bt.get("status"), ct.get("status"), ""))
        for row in ct["rows"]:
            if row not in bt["rows"]:
                continue
            for metric, cv in ct["rows"][row].items():
                bv = bt["rows"][row].get(metric)
                if not isinstance(cv, float) or not isinstance(bv, float):
                    if cv != bv and metric in bt["rows"][row]:
                        moved.append(("", test, row, metric, bv, cv, ""))
                    continue
                d = cv - bv
                if d == 0:
                    unchanged += 1
                    continue
                kind, thr, better = threshold(metric)
                size = abs(d) if kind == "abs" else (abs(d) / abs(bv) if bv else float("inf"))
                rel = f" ({d / bv:+.1%})" if bv else ""
                delta = f"{d:+.4g}{rel}"
                if metric == "time_s":
                    # Runner speed varies by tens of percent between identical runs: only a
                    # halving or doubling of at least a second is reported, apart from the rest.
                    if size >= thr and abs(d) >= 1.0:
                        timing.append(("", test, row, metric, bv, cv, delta))
                    continue
                if size >= thr:
                    icon = ":white_circle:" if better == 0 else (
                        ":green_circle:" if d * better > 0 else ":red_circle:")
                    flagged.append((icon, test, row, metric, bv, cv, delta))
                else:
                    moved.append(("", test, row, metric, bv, cv, delta))

    def table(rows):
        print("| | Test | Result | Metric | main | this PR | Δ |")
        print("|---|---|---|---|---|---|---|")
        for icon, test, row, metric, bv, cv, delta in rows:
            print(f"| {icon} | `{test}` | {row} | {metric} | {fmt(bv)} | {fmt(cv)} | {delta} |")
        print("")

    if flagged:
        print(f"**{len(flagged)} change(s) past the thresholds** "
              "(:green_circle: better, :red_circle: worse, :white_circle: no better/worse direction):")
        print("")
        table(flagged)
    else:
        print("**No metric moved past the thresholds.**")
        print("")
    cap = 200  # per list; keeps the comment under GitHub's size limit
    for rows, summary in ((moved, f"{len(moved)} smaller change(s)"),
                          (timing, f"{len(timing)} timing change(s) of 50% or more")):
        if not rows:
            continue
        print(f"<details><summary>{summary}</summary>")
        print("")
        table(rows[:cap])
        if len(rows) > cap:
            print(f"_{len(rows) - cap} more not shown; the full records are on the `ci-figures` branch._")
            print("")
        print("</details>")
        print("")
    notes = [f"{unchanged} metric(s) identical to main"]
    if new_tests:
        notes.append("no main record for " + ", ".join(f"`{t}`" for t in new_tests))
    if gone_tests:
        notes.append(f"{len(gone_tests)} test(s) on main not run here")
    print("_" + "; ".join(notes) + "._")
    print("")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "record":
        record(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        sys.exit(__doc__)

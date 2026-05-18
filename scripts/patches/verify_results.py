#!/usr/bin/env python3
"""
verify_results.py — HypatiaX JMLR result verification

Checks that benchmark outputs match the expected values from the paper.
Tolerances are generous to allow for hardware/seed variation.
Exit 0 = pass, Exit 1 = fail.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ROOT is the repo root.  __file__ lives at scripts/patches/verify_results.py,
# so .parent = scripts/patches/, .parent.parent = scripts/, .parent.parent.parent = repo root.
ROOT = Path(__file__).resolve().parent.parent.parent

# verify_results.sh exports VERIFY_RESULTS_DIR and PATCHED_DATA_DIR pointing at
# the actual results tree produced by merge_shards.py.  ci_experiment.yml also
# exports RESULTS_BASE for the same purpose.  Fall back to the canonical
# repo-relative path when running locally without those env vars set.
_env_results = (
    os.environ.get("VERIFY_RESULTS_DIR")
    or os.environ.get("RESULTS_BASE")
    or os.environ.get("PATCHED_DATA_DIR")
)
RESULTS_DIR = Path(_env_results) if _env_results else ROOT / "hypatiax" / "data" / "results"

# PATCHED_DIR is a post-processed copy used for paper-final verification.
# Falls back to RESULTS_DIR so the verifier works on raw CI outputs when the
# patching pipeline has not run yet (all CI runs before paper submission).
_env_patched = os.environ.get("PATCHED_DATA_DIR")
PATCHED_DIR = Path(_env_patched) if _env_patched else RESULTS_DIR

# ── Expected values from paper (v3.0) ─────────────────────────────────────────
EXPECTED = {
    "defi": {
        "accuracy":     0.892,  # §10.2  — 89.2% discovery rate
        "total_cases":  74,     # §10.3  — 74 DeFi cases
        "easy_cases":   24,
        "medium_cases": 29,
        "hard_cases":   21,
    },
    "feynman": {
        "successes":    9,      # §10.7  — 9/30 full extrapolation success
        "total":        30,
    },
    "core15": {
        "mw_p":         0.2948, # §10.6  — Mann-Whitney p=0.2948
        "mw_u":         126.0,
    },
    "instability": {
        "total_tasks":  70,     # §10.9  — 70 tasks, K=30 runs
    },
}

TOL = {
    "accuracy":     0.03,   # ±3% allowed
    "mw_p":         0.10,   # ±10% for statistical tests
    "mw_u":         20.0,
    "successes":    2,      # ±2 count
}

PASS_COUNT = 0
FAIL_COUNT = 0
WARN_COUNT = 0

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_json(path):
    with open(path) as f:
        return json.load(f)

def find_result(subdir, filename_glob):
    """Find the most recent matching result file.

    Search order:
      1. PATCHED_DIR / subdir  — post-processed / staged files (preferred)
      2. RESULTS_DIR / subdir  — raw CI merge outputs
    Within each directory, `filename_glob` is tried first; if it returns
    nothing, two fallback globs are tried in order:
      · _stats.json    (written by merge_shards.py for every experiment)
      · _merged.json   (written by merge_shards.py for JSON experiments)
    This lets check_*() functions work on raw CI outputs even when the full
    patching pipeline has not yet produced a canonical benchmark_results.json.
    """
    _FALLBACKS = ("_stats.json", "_merged.json")

    for base in (PATCHED_DIR, RESULTS_DIR):
        d = base / subdir
        if not d.exists():
            continue
        # Primary glob
        candidates = sorted(d.glob(filename_glob), key=os.path.getmtime, reverse=True)
        if candidates:
            return candidates[0]
        # Fallback globs
        for fb in _FALLBACKS:
            fb_candidates = sorted(d.glob(fb), key=os.path.getmtime, reverse=True)
            if fb_candidates:
                return fb_candidates[0]

    return None

def check(name, actual, expected, tol, fmt=".4f"):
    global PASS_COUNT, FAIL_COUNT
    diff = abs(actual - expected)
    symbol = "✅" if diff <= tol else "❌"
    print(f"  {symbol} {name:40s}  got={actual:{fmt}}  expected={expected:{fmt}}  tol=±{tol:{fmt}}")
    if diff <= tol:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1

def check_int(name, actual, expected, tol=0):
    global PASS_COUNT, FAIL_COUNT, WARN_COUNT
    diff = abs(actual - expected)
    if diff <= tol:
        print(f"  ✅ {name:40s}  got={actual}  expected={expected}")
        PASS_COUNT += 1
    else:
        print(f"  ❌ {name:40s}  got={actual}  expected={expected}  diff={diff}")
        FAIL_COUNT += 1

def warn_missing(name, path):
    global WARN_COUNT
    print(f"  ⚠  {name:40s}  file not found: {path}")
    WARN_COUNT += 1

# ── DeFi checks ───────────────────────────────────────────────────────────────
def check_defi():
    print("\n── DeFi Benchmark (§10.2–10.4) ─────────────────────────────────────")
    f = find_result("defi", "benchmark_results*.json")
    if not f:
        warn_missing("DeFi results", PATCHED_DIR / "defi/benchmark_results.json")
        return
    data = load_json(f)

    # Accuracy — paper field names first, then merge_shards.py _stats.json aliases
    acc = (
        data.get("accuracy")
        or data.get("success_rate")
        or data.get("discovery_rate")
        or data.get("solve_rate")   # _stats.json from merge_shards.py
    )
    if acc is not None:
        check("DeFi accuracy (§10.2)", acc, EXPECTED["defi"]["accuracy"], TOL["accuracy"])
    else:
        warn_missing("DeFi accuracy field", f)

    # Case counts — paper field names first, then merge_shards.py aliases
    total = data.get("total_cases") or data.get("n_cases") or data.get("n_tasks")
    if total is not None:
        check_int("DeFi total cases (§10.3)", total, EXPECTED["defi"]["total_cases"])

    def _difficulty_count(data, simple_key, nested_key):
        """Return case count whether stored as int or {"count": N} dict."""
        v = data.get(simple_key)
        if isinstance(v, int):
            return v
        nested = data.get(nested_key)
        if isinstance(nested, int):
            return nested
        if isinstance(nested, dict):
            return nested.get("count")
        return None

    easy   = _difficulty_count(data, "easy_cases",   "easy")
    medium = _difficulty_count(data, "medium_cases", "medium")
    hard   = _difficulty_count(data, "hard_cases",   "hard")
    if easy is not None:
        check_int("DeFi easy cases",   easy,   EXPECTED["defi"]["easy_cases"])
    if medium is not None:
        check_int("DeFi medium cases", medium, EXPECTED["defi"]["medium_cases"])
    if hard is not None:
        check_int("DeFi hard cases",   hard,   EXPECTED["defi"]["hard_cases"])

# ── Feynman checks ────────────────────────────────────────────────────────────
def check_feynman():
    print("\n── Feynman Benchmark (§10.7) ────────────────────────────────────────")
    f = find_result("feynman", "benchmark_results*.json")
    if not f:
        warn_missing("Feynman results", PATCHED_DIR / "feynman/benchmark_results.json")
        return
    data = load_json(f)

    successes = (
        data.get("successes")
        or data.get("n_success")
        or data.get("full_extrapolation_success")
        or data.get("n_solved")   # _stats.json from merge_shards.py
    )
    total = (
        data.get("total")
        or data.get("n_cases")
        or data.get("n_tasks")    # _stats.json from merge_shards.py
    )
    if successes is not None:
        check_int("Feynman successes (§10.7)", successes, EXPECTED["feynman"]["successes"], tol=TOL["successes"])
    if total is not None:
        check_int("Feynman total cases",       total,     EXPECTED["feynman"]["total"])

# ── Core-15 / Ablation checks ─────────────────────────────────────────────────
def check_core15():
    print("\n── Core-15 Ablation (§10.6) ─────────────────────────────────────────")
    f = find_result("exp1_ablation", "*.json")
    if not f:
        warn_missing("Core-15 results", PATCHED_DIR / "exp1_ablation/")
        return
    data = load_json(f)

    mw_p = data.get("mw_p") or data.get("mann_whitney_p") or data.get("p_value")
    mw_u = data.get("mw_u") or data.get("mann_whitney_u") or data.get("u_statistic")
    if mw_p is not None:
        check("Core-15 MW p-value (§10.6)", mw_p, EXPECTED["core15"]["mw_p"], TOL["mw_p"])
    if mw_u is not None:
        check("Core-15 MW U statistic",     mw_u, EXPECTED["core15"]["mw_u"], TOL["mw_u"])

# ── Instability checks ────────────────────────────────────────────────────────
def check_instability():
    print("\n── Instability Benchmark (§10.9) ────────────────────────────────────")

    # instability is a CSV-only experiment (array_key=None in merge_shards.py).
    # merge_shards._merge_instability_csvs() writes:
    #   _stats.json   → {"n_tasks": N, "n_shard_files": M}   ← preferred
    #   _merged.csv   → concatenated CSV rows                 ← count fallback
    # A canonical benchmark_results.json may also exist in the patched tree.
    f = find_result("instability", "*.json")
    if f:
        data = load_json(f)
        tasks = (
            data.get("total_tasks")
            or data.get("n_tasks")
            or len(data.get("tasks", data.get("results", [])))
        )
        if tasks:
            check_int("Instability total tasks (§10.9)", tasks, EXPECTED["instability"]["total_tasks"])
        else:
            warn_missing("Instability task count field", f)
        return

    # JSON not found — fall back to counting rows in _merged.csv
    import csv as _csv
    for base in (PATCHED_DIR, RESULTS_DIR):
        csv_path = base / "instability" / "_merged.csv"
        if csv_path.exists():
            try:
                with open(csv_path, newline="", encoding="utf-8") as fh:
                    row_count = sum(1 for _ in _csv.DictReader(fh))
                check_int("Instability total tasks (§10.9)", row_count,
                          EXPECTED["instability"]["total_tasks"])
                return
            except Exception as exc:
                print(f"  ⚠  Could not read {csv_path.name}: {exc}", file=sys.stderr)

    warn_missing("Instability results", PATCHED_DIR / "instability/")

# ── Duplicate case check ───────────────────────────────────────────────────────
def check_defi_duplicates():
    """FIX-C1: verify no duplicate case names in DeFi benchmark source."""
    global PASS_COUNT, FAIL_COUNT
    import re
    from collections import Counter

    print("\n── DeFi Duplicate Case Check (FIX-C1) ──────────────────────────────")
    # Search from repo ROOT so the script is found regardless of cwd.
    bench_files = list(ROOT.rglob("hypatiax_defi_benchmark_v3c.py"))
    if not bench_files:
        warn_missing("hypatiax_defi_benchmark_v3c.py", "not found in repo")
        return
    src = bench_files[0].read_text()
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', src)
    dupes = {n: c for n, c in Counter(names).items() if c > 1}
    if dupes:
        print("  ❌ Duplicate case names found (FIX-C1 not applied):")
        for name, count in dupes.items():
            print(f"       '{name}' appears {count}×")
        FAIL_COUNT += 1
    else:
        print("  ✅ No duplicate case names")
        PASS_COUNT += 1


def build_summary():
    return {
        "pass": PASS_COUNT,
        "fail": FAIL_COUNT,
        "warn": WARN_COUNT,
        "status": (
            "failed"
            if FAIL_COUNT > 0
            else "warning"
            if WARN_COUNT > 0
            else "passed"
        )
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main(report=False, report_file=None, results_dir=None):
    global RESULTS_DIR, PATCHED_DIR
    global PASS_COUNT, FAIL_COUNT, WARN_COUNT

    # Reset counters so repeated calls (tests, multiple invocations) don't
    # accumulate across runs.
    PASS_COUNT = FAIL_COUNT = WARN_COUNT = 0

    # --results-dir CLI flag (or RESULTS_BASE env var set by ci_experiment.yml)
    # overrides module-level paths so all check_*() functions see the correct dir.
    if results_dir:
        RESULTS_DIR = Path(results_dir)
        PATCHED_DIR = Path(results_dir)

    print("═" * 65)
    print("  HypatiaX JMLR Result Verification")
    print(f"  RESULTS_DIR : {RESULTS_DIR}")
    print(f"  PATCHED_DIR : {PATCHED_DIR}")
    print("═" * 65)

    check_defi()
    check_feynman()
    check_core15()
    check_instability()
    check_defi_duplicates()

    summary = build_summary()

    print("\n" + "═" * 65)
    print(
        f"  PASS: {summary['pass']}   "
        f"FAIL: {summary['fail']}   "
        f"WARN: {summary['warn']}"
    )
    print("═" * 65)

    if report:
        print("\n── Verification report ─────────────────────────────")
        print(json.dumps(summary, indent=2))

    if report_file:
        Path(report_file).write_text(
            json.dumps(summary, indent=2)
        )
        print(f"\nReport written → {report_file}")

    if FAIL_COUNT > 0:
        print("\n❌ Verification FAILED")
        sys.exit(1)

    print(
        "\n⚠ Verification passed with warnings"
        if WARN_COUNT > 0
        else "\n✅ All checks passed"
    )
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify HypatiaX benchmark outputs"
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Print JSON summary report"
    )

    parser.add_argument(
        "--report-file",
        type=str,
        default=None,
        help="Write report to file"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Override the results directory used by all checks.  "
            "Equivalent to setting VERIFY_RESULTS_DIR in the environment.  "
            "When omitted, the env var VERIFY_RESULTS_DIR / RESULTS_BASE / "
            "PATCHED_DATA_DIR is used, falling back to the repo-relative "
            "hypatiax/data/results/ path."
        ),
    )

    args = parser.parse_args()

    main(
        report=args.report,
        report_file=args.report_file,
        results_dir=args.results_dir,
    )

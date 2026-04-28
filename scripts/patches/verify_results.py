#!/usr/bin/env python3
"""
verify_results.py — HypatiaX JMLR result verification

Checks that benchmark outputs match the expected values from the paper.
Tolerances are generous to allow for hardware/seed variation.
Exit 0 = pass, Exit 1 = fail.
"""
import json
import os
import sys
from pathlib import Path

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "hypatiax" / "data" / "results"
PATCHED_DIR = ROOT / "hypatiax" / "data" / "patched"

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
    """Find the most recent matching result file."""
    d = PATCHED_DIR / subdir
    if not d.exists():
        d = RESULTS_DIR / subdir
    if not d.exists():
        return None
    candidates = sorted(d.glob(filename_glob), key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None

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

    # Accuracy
    acc = data.get("accuracy") or data.get("success_rate") or data.get("discovery_rate")
    if acc is not None:
        check("DeFi accuracy (§10.2)", acc, EXPECTED["defi"]["accuracy"], TOL["accuracy"])
    else:
        warn_missing("DeFi accuracy field", f)

    # Case counts
    total = data.get("total_cases") or data.get("n_cases")
    if total is not None:
        check_int("DeFi total cases (§10.3)", total, EXPECTED["defi"]["total_cases"])

    easy   = data.get("easy_cases")   or data.get("easy",   {}).get("count")
    medium = data.get("medium_cases") or data.get("medium", {}).get("count")
    hard   = data.get("hard_cases")   or data.get("hard",   {}).get("count")
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
    )
    total = data.get("total") or data.get("n_cases")
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
    f = find_result("instability", "*.json")
    if not f:
        warn_missing("Instability results", PATCHED_DIR / "instability/")
        return
    data = load_json(f)

    tasks = (
        data.get("total_tasks")
        or data.get("n_tasks")
        or len(data.get("tasks", data.get("results", [])))
    )
    if tasks:
        check_int("Instability total tasks (§10.9)", tasks, EXPECTED["instability"]["total_tasks"])

# ── Duplicate case check ───────────────────────────────────────────────────────
def check_defi_duplicates():
    """FIX-C1: verify no duplicate case names in DeFi benchmark source."""
    print("\n── DeFi Duplicate Case Check (FIX-C1) ──────────────────────────────")
    bench_files = list(Path(ROOT).rglob("hypatiax_defi_benchmark_v3c.py"))
    if not bench_files:
        warn_missing("hypatiax_defi_benchmark_v3c.py", "not found in repo")
        return
    src = bench_files[0].read_text()
    import re
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', src)
    from collections import Counter
    dupes = {n: c for n, c in Counter(names).items() if c > 1}
    if dupes:
        global FAIL_COUNT
        print("  ❌ Duplicate case names found (FIX-C1 not applied):")
        for name, count in dupes.items():
            print(f"       '{name}' appears {count}×")
        FAIL_COUNT += 1
    else:
        global PASS_COUNT
        print("  ✅ No duplicate case names")
        PASS_COUNT += 1

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("═" * 65)
    print("  HypatiaX JMLR Result Verification")
    print("═" * 65)

    check_defi()
    check_feynman()
    check_core15()
    check_instability()
    check_defi_duplicates()

    print("\n" + "═" * 65)
    print(f"  PASS: {PASS_COUNT}   FAIL: {FAIL_COUNT}   WARN (missing): {WARN_COUNT}")
    print("═" * 65)

    if FAIL_COUNT > 0:
        print("\n❌ Verification FAILED — check logs above")
        sys.exit(1)
    elif WARN_COUNT > 0:
        print("\n⚠  Verification passed with warnings (some files not yet generated)")
        sys.exit(0)
    else:
        print("\n✅ All checks passed")
        sys.exit(0)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
check_run_health.py
====================
Post-run diagnostic: checks whether the bugs identified in the §10.7 / PCA
investigation are still present in the latest result files.

Checks:
  [1] Bug 1  — split_protocol_disclosure.json missing random_split_used in 15_pca/
  [2] Bug 2  — EnhancedHybridSystemDeFi: success=True with r2=NaN (Black-Scholes, Theta)
  [3] Bug 3  — HYBRID_V50_2_AVAILABLE was False: HDS rows all "not available"
  [4] exp1_ablation still on engine v5.1 (not yet rerun with v5.4)
  [5] exp1_ablation NaN RMSE on Henderson-Hasselbalch / Rate Law
  [6] Mann-Whitney: TOO_FEW_MW_PAIRS (ablation_paired.json insufficient)
  [7] exp2_pca_4060_summary.json n_pass provenance still unresolved (>110 divergence)
  [8] pca_test_r2 all None (Bug 6 regression check — was silent NameError)
  [9] fixc3_baseline.json solve_rate is null (written before exp2 results existed)
 [10] protocol_core_noiseless_pca_*.json: HypatiaX-only recount needed vs pooled count

Usage:
  python check_run_health.py                        # auto-discover under default paths
  python check_run_health.py --results-dir /path/to/hypatiax/data/results
  python check_run_health.py --results-dir /path/to/results --verbose

Exit code:
  0 — all checks passed (no issues found)
  1 — one or more checks failed
"""

import argparse
import json
import math
import sys
from pathlib import Path


# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}[OK   ]{RESET} {msg}")
def warn(msg):  print(f"  {YELLOW}[WARN ]{RESET} {msg}")
def fail(msg):  print(f"  {RED}[FAIL ]{RESET} {msg}")
def info(msg):  print(f"  {BOLD}[INFO ]{RESET} {msg}")


# ── helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception as e:
        return None


def latest_glob(directory: Path, pattern: str):
    """Return the most-recently-modified file matching pattern, or None."""
    files = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
    return files[0] if files else None


def iter_results(data):
    """Yield individual result dicts from any known JSON schema."""
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        for key in ("tests", "results", "equation_results", "data"):
            v = data.get(key)
            if v:
                yield from iter_results(v)
                return
        yield data


# ── Check implementations ─────────────────────────────────────────────────────

def check_1_disclosure_random_split_used(results_dir: Path, verbose: bool) -> bool:
    """Bug 1: 15_pca/split_protocol_disclosure.json must have random_split_used key."""
    disc_path = results_dir / "comparison_results/noise-noiseless/15_pca/split_protocol_disclosure.json"
    if not disc_path.exists():
        fail(f"[1] 15_pca/split_protocol_disclosure.json not found: {disc_path}")
        info("    exp1b_pca has not yet run, or run_all.sh patch not deployed.")
        return False

    data = load_json(disc_path)
    if data is None:
        fail("[1] 15_pca/split_protocol_disclosure.json could not be parsed.")
        return False

    if "random_split_used" not in data:
        fail("[1] 15_pca/split_protocol_disclosure.json is missing key 'random_split_used'.")
        if verbose:
            info(f"    Keys present: {list(data.keys())}")
        return False

    val = data["random_split_used"]
    if val is not False:
        fail(f"[1] random_split_used = {val!r} (expected False).")
        return False

    ok("[1] 15_pca/split_protocol_disclosure.json has random_split_used=False.")
    return True


def check_2_hybrid_defi_nan_r2(results_dir: Path, verbose: bool) -> bool:
    """Bug 2: hypatiax_defi_benchmark_pca_results.json must not have success=True + r2=NaN."""
    defi_dir = results_dir / "comparison_results/noise-noiseless/noiseless/defi_pca"
    result_file = latest_glob(defi_dir, "hypatiax_defi_benchmark_pca_results*.json")
    if result_file is None:
        warn("[2] No hypatiax_defi_benchmark_pca_results*.json found — exp1_pca not yet run.")
        return True  # not a failure, just not run yet

    data = load_json(result_file)
    if data is None:
        fail(f"[2] Could not parse {result_file.name}.")
        return False

    bad_cases = []
    cases = data if isinstance(data, list) else []
    for case in cases:
        if not isinstance(case, dict):
            continue
        hybrid = case.get("results", {}).get("hybrid", {})
        if not hybrid.get("success", False):
            continue
        train_r2 = hybrid.get("train_r2")
        if train_r2 is None or (isinstance(train_r2, float) and math.isnan(train_r2)):
            bad_cases.append(case.get("equation_id", "?"))

    if bad_cases:
        fail(f"[2] {len(bad_cases)} case(s) have success=True but train_r2=NaN: {bad_cases}")
        info("    Fix: NaN fallback in HybridDeFiMethod.run() (Bug 2 patch).")
        return False

    ok(f"[2] No success=True + r2=NaN cases in {result_file.name}.")
    return True


def check_3_hds_not_available(results_dir: Path, verbose: bool) -> bool:
    """Bug 3: HDS rows must not all be 'not available' in PCA Feynman results."""
    pca_dir = results_dir / "comparison_results/feynman-tests/exp2_pca_4060"
    pca_files = sorted(pca_dir.glob("protocol_core_*.json")) if pca_dir.exists() else []
    if not pca_files:
        warn("[3] No protocol_core_*.json in exp2_pca_4060/ — exp2_feynman_pca not yet run.")
        return True

    hds_name = "HybridDiscoverySystem v50_2 (tools)"
    total_hds = 0
    unavailable_hds = 0

    for fp in pca_files:
        data = load_json(fp)
        if data is None:
            continue
        for rec in iter_results(data):
            results = rec.get("results", {})
            if hds_name in results:
                total_hds += 1
                res = results[hds_name]
                err = (res.get("error") or "").lower()
                if "not available" in err or res.get("success") is False and "not available" in err:
                    unavailable_hds += 1

    if total_hds == 0:
        warn(f"[3] No HDS rows found in exp2_pca_4060/ — method may not have been included.")
        return True

    unavail_pct = 100 * unavailable_hds / total_hds
    if unavailable_hds == total_hds:
        fail(f"[3] ALL {total_hds} HDS rows are 'not available' — Bug 3 sys.path fix not effective.")
        info("    Check that run_comparative_suite_benchmark_pca.py patch was deployed.")
        return False
    elif unavailable_hds > 0:
        warn(f"[3] {unavailable_hds}/{total_hds} HDS rows ({unavail_pct:.0f}%) are 'not available'.")
        if verbose:
            info("    Some HDS runs failed — may be Julia/PySR timeouts, not Bug 3.")
        return True  # partial availability is not the same bug
    else:
        ok(f"[3] All {total_hds} HDS rows ran (not 'not available').")
        return True


def check_4_ablation_engine_version(results_dir: Path, verbose: bool) -> bool:
    """exp1_ablation: engine_version must be 5.4, not 5.1."""
    ablation_dir = results_dir / "ablation/exp1_ablation"
    ablation_file = latest_glob(ablation_dir, "exp1_ablation_results*.json")
    if ablation_file is None:
        warn("[4] exp1_ablation_results*.json not found — exp1_ablation not yet run.")
        return True

    data = load_json(ablation_file)
    if data is None:
        fail(f"[4] Could not parse {ablation_file.name}.")
        return False

    # exp1_ablation_results.json schema: {"0": {"name": ..., "hypatia": {...}, "pysr_only": {...}}, ...}
    # Entries are keyed by index strings, not a "results"/"tests" list, so iter_results
    # would yield the entire outer dict as one blob.  Walk entries explicitly instead.
    entries = data.values() if isinstance(data, dict) else iter_results(data)

    v51_cases = []
    for rec in entries:
        if not isinstance(rec, dict):
            continue
        for method_key in ("hypatia", "hypatia_x", "HypatiaX"):
            # Method block is a direct child of the entry, not nested under "results"
            block = rec.get(method_key) or rec.get("results", {}).get(method_key) or {}
            ver = block.get("engine_version") or block.get("version") or ""
            if "5.1" in str(ver):
                v51_cases.append(rec.get("equation_name", rec.get("name", "?")))
                break

    if v51_cases:
        fail(f"[4] {len(v51_cases)} case(s) still show engine_version 5.1: {v51_cases[:5]}")
        info("    exp1_ablation has not been rerun with v5.4 yet.")
        return False

    ok(f"[4] No engine_version 5.1 entries in {ablation_file.name}.")
    return True


def check_5_ablation_nan_rmse(results_dir: Path, verbose: bool) -> bool:
    """exp1_ablation: Henderson-Hasselbalch and Rate Law must not have NaN RMSE."""
    ablation_dir = results_dir / "ablation/exp1_ablation"
    ablation_file = latest_glob(ablation_dir, "exp1_ablation_results*.json")
    if ablation_file is None:
        warn("[5] exp1_ablation_results*.json not found — skipping NaN RMSE check.")
        return True

    data = load_json(ablation_file)
    if data is None:
        fail(f"[5] Could not parse {ablation_file.name}.")
        return False

    TARGET_CASES = {"henderson-hasselbalch", "rate law", "henderson_hasselbalch", "rate_law"}
    nan_cases = []

    for rec in iter_results(data):
        name = (rec.get("equation_name") or rec.get("name") or "").lower().replace(" ", "_")
        if not any(t in name for t in TARGET_CASES):
            continue
        for method_key, block in rec.get("results", {}).items():
            rmse = block.get("train_rmse") or block.get("rmse")
            extrap_r2 = block.get("extrap_r2_far")
            if rmse is not None and (str(rmse).lower() == "nan" or
                    (isinstance(rmse, float) and math.isnan(rmse))):
                nan_cases.append(f"{name} / {method_key}")
            if extrap_r2 is None and block.get("success"):
                nan_cases.append(f"{name} / {method_key} extrap_r2_far=null")

    if nan_cases:
        fail(f"[5] NaN RMSE / null extrap_r2_far on ablation target cases: {nan_cases}")
        info("    RC-1–RC-5 fixes in hybrid_system_v50_2.py v5.4 not yet in effect.")
        return False

    ok("[5] No NaN RMSE on Henderson-Hasselbalch / Rate Law in ablation results.")
    return True


def check_6_mann_whitney_pairs(results_dir: Path, verbose: bool) -> bool:
    """ablation_paired.json must have >= 5 pairs for a valid Mann-Whitney test."""
    # Common locations
    candidates = [
        results_dir / "ablation/exp1_ablation/ablation_paired.json",
        results_dir / "ablation/ablation_paired.json",
        results_dir / "comparison_results/ablation_paired.json",
    ]
    paired_file = next((p for p in candidates if p.exists()), None)

    if paired_file is None:
        warn("[6] ablation_paired.json not found — merge step may not have run yet.")
        return True

    data = load_json(paired_file)
    if data is None:
        fail(f"[6] Could not parse ablation_paired.json.")
        return False

    pairs = data if isinstance(data, list) else data.get("pairs", data.get("results", []))
    n_pairs = len(pairs) if isinstance(pairs, list) else 0

    if n_pairs < 5:
        fail(f"[6] ablation_paired.json has only {n_pairs} pair(s) — Mann-Whitney needs ≥5.")
        info("    Rerun exp2_feynman_extrap after exp1_ablation v5.4 completes.")
        return False

    ok(f"[6] ablation_paired.json has {n_pairs} pairs (≥5 — Mann-Whitney is runnable).")
    return True


def check_7_pca_summary_provenance(results_dir: Path, verbose: bool) -> bool:
    """exp2_pca_4060_summary.json: n_pass must be stable and match shard files."""
    summary_path = results_dir / "comparison_results/feynman-tests/exp2_pca_4060/exp2_pca_4060_summary.json"
    if not summary_path.exists():
        warn("[7] exp2_pca_4060_summary.json not found — exp2_feynman_pca not yet run.")
        return True

    data = load_json(summary_path)
    if data is None:
        fail("[7] Could not parse exp2_pca_4060_summary.json.")
        return False

    n_pass  = data.get("n_pass")
    n_total = data.get("n_total")
    rate    = data.get("solve_rate")

    if n_pass is None or n_total is None:
        fail("[7] exp2_pca_4060_summary.json missing n_pass or n_total.")
        return False

    # Count passes directly from the shard files for cross-check
    pca_dir = summary_path.parent
    shard_pass = shard_total = 0
    PREFERRED = {"hypatia", "enhancedhybrid", "hypatiaX"}
    THRESHOLD = 0.995

    for fp in sorted(pca_dir.glob("protocol_core_*.json")):
        d = load_json(fp)
        if d is None:
            continue
        for rec in iter_results(d):
            for mname, mres in rec.get("results", {}).items():
                m_lower = mname.lower().replace(" ", "").replace("-", "").replace("_", "")
                if not any(p.lower() in m_lower for p in PREFERRED):
                    continue
                if not mres.get("success"):
                    continue
                r2 = mres.get("r2")
                if r2 is None:
                    continue
                try:
                    r2f = float(r2)
                except (TypeError, ValueError):
                    continue
                if r2f > 1.01:
                    continue
                shard_total += 1
                if r2f >= THRESHOLD:
                    shard_pass += 1

    info(f"[7] Summary reports {n_pass}/{n_total} (rate={rate}). "
         f"Direct shard count (HypatiaX-only, R²≥{THRESHOLD}): {shard_pass}/{shard_total}.")

    if shard_total > 0 and abs(n_pass - shard_pass) > 5:
        warn(f"[7] Summary n_pass={n_pass} diverges from shard recount={shard_pass} by "
             f"{abs(n_pass-shard_pass)} — pooled vs HypatiaX-only mismatch still present.")
        info("    Manual filter-to-HypatiaX recount required before citing §10.7 figure.")
        return False

    if shard_total == 0:
        warn("[7] No HypatiaX rows found in shard files for cross-check.")
        return True

    ok(f"[7] Summary n_pass={n_pass} matches shard recount={shard_pass} (within tolerance).")
    return True


def check_8_pca_test_r2_not_all_none(results_dir: Path, verbose: bool) -> bool:
    """Bug 6 regression: pca_test_r2 must not be all-None in protocol_core_*_pca_*.json."""
    pca_dir = results_dir / "comparison_results/feynman-tests/exp2_pca_4060"
    pca_files = sorted(pca_dir.glob("protocol_core_*.json")) if pca_dir.exists() else []
    if not pca_files:
        warn("[8] No protocol_core_*.json in exp2_pca_4060/ — skipping pca_test_r2 check.")
        return True

    total_records = 0
    records_with_pca_r2 = 0
    records_all_none = 0

    for fp in pca_files:
        data = load_json(fp)
        if data is None:
            continue
        for rec in iter_results(data):
            pca_r2 = rec.get("pca_test_r2")
            if pca_r2 is None:
                continue  # non-PCA record, skip
            total_records += 1
            values = list(pca_r2.values()) if isinstance(pca_r2, dict) else []
            non_none = [v for v in values if v is not None]
            if non_none:
                records_with_pca_r2 += 1
            else:
                records_all_none += 1

    if total_records == 0:
        warn("[8] No records with pca_test_r2 field found — split_protocol may not be 'pca_40_60'.")
        return True

    if records_all_none == total_records:
        fail(f"[8] ALL {total_records} record(s) have pca_test_r2 all-None — Bug 6 still active.")
        info("    _RUNNER_EVAL_FORMULA NameError fix in run_comparative_suite_benchmark_pca.py not deployed.")
        return False

    if records_all_none > 0:
        warn(f"[8] {records_all_none}/{total_records} records have pca_test_r2 all-None "
             f"(partial — may be NN-only methods which cannot be evaluated symbolically).")
        return True

    ok(f"[8] {records_with_pca_r2}/{total_records} records have non-None pca_test_r2 values.")
    return True


def check_9_fixc3_baseline_not_null(results_dir: Path, verbose: bool) -> bool:
    """fixc3_baseline.json must exist and have a non-null solve_rate."""
    baseline_path = results_dir / "comparison_results/feynman-tests/exp2_pca_4060/fixc3_baseline.json"
    if not baseline_path.exists():
        # Also try root of results_dir
        baseline_path = results_dir / "fixc3_baseline.json"
    if not baseline_path.exists():
        warn("[9] fixc3_baseline.json not found — Gate C baseline lock has not run yet.")
        return True

    data = load_json(baseline_path)
    if data is None:
        fail("[9] fixc3_baseline.json could not be parsed.")
        return False

    rate = data.get("solve_rate")
    n_pass  = data.get("n_pass")
    n_total = data.get("n_total")

    if rate is None:
        fail(f"[9] fixc3_baseline.json has solve_rate=null — was written before exp2 results existed.")
        info("    Delete fixc3_baseline.json and re-run after exp2_feynman results are present.")
        return False

    ok(f"[9] fixc3_baseline.json has solve_rate={rate:.3f} ({n_pass}/{n_total}) — valid baseline locked.")
    return True


def check_10_hypatia_only_recount(results_dir: Path, verbose: bool) -> bool:
    """
    Protocol_core_noiseless_pca_*.json: check whether the published count is
    pooled (all 6 methods) or HypatiaX-only. If pooled, the §10.7 number is wrong.
    """
    pca_dir = results_dir / "comparison_results/feynman-tests/exp2_pca_4060"
    pca_files = sorted(pca_dir.glob("protocol_core_noiseless_pca_*.json")) if pca_dir.exists() else []
    if not pca_files:
        warn("[10] No protocol_core_noiseless_pca_*.json found — recount not yet possible.")
        return True

    HYPATIA_KEYWORDS = {"hypatia", "enhancedhybrid", "hypatiaX"}
    ALL_METHODS: set = set()
    hypatia_pass = hypatia_total = 0
    pooled_pass  = pooled_total  = 0
    THRESHOLD = 0.9999

    for fp in pca_files:
        data = load_json(fp)
        if data is None:
            continue
        for rec in iter_results(data):
            for mname, mres in rec.get("results", {}).items():
                ALL_METHODS.add(mname)
                if not mres.get("success"):
                    continue
                r2 = mres.get("r2")
                try:
                    r2f = float(r2) if r2 is not None else None
                except (TypeError, ValueError):
                    r2f = None
                if r2f is None or r2f > 1.01:
                    continue
                pooled_total += 1
                if r2f >= THRESHOLD:
                    pooled_pass += 1
                m_lower = mname.lower().replace(" ","").replace("-","").replace("_","")
                if any(k.lower() in m_lower for k in HYPATIA_KEYWORDS):
                    hypatia_total += 1
                    if r2f >= THRESHOLD:
                        hypatia_pass += 1

    if pooled_total == 0:
        warn("[10] No result rows found in noiseless_pca files.")
        return True

    info(f"[10] Methods seen: {len(ALL_METHODS)} — {sorted(ALL_METHODS)[:4]}{'…' if len(ALL_METHODS)>4 else ''}")
    info(f"[10] Pooled (all methods):      {pooled_pass}/{pooled_total} pass R²≥{THRESHOLD}")
    info(f"[10] HypatiaX-only:             {hypatia_pass}/{hypatia_total} pass R²≥{THRESHOLD}")

    if hypatia_total == 0:
        warn("[10] No HypatiaX rows found — HDS still not running (check Bug 3 fix).")
        return False

    if pooled_pass != hypatia_pass:
        warn(f"[10] Pooled count ({pooled_pass}) ≠ HypatiaX-only count ({hypatia_pass}) — "
             f"§10.7 must cite {hypatia_pass}/{hypatia_total}, NOT {pooled_pass}/{pooled_total}.")
        info("    Update paper_targets.json and fixc3_baseline.json with HypatiaX-only figure.")
        return False

    ok(f"[10] HypatiaX-only count ({hypatia_pass}/{hypatia_total}) matches pooled — recount consistent.")
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Post-run health check for HypatiaX §10.7 / PCA investigation bugs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_run_health.py
  python check_run_health.py --results-dir hypatiax/data/results
  python check_run_health.py --results-dir /abs/path/to/results --verbose
  python check_run_health.py --check 1 3 6          # run only specific checks
        """,
    )
    parser.add_argument(
        "--results-dir", "-r",
        default=None,
        metavar="DIR",
        help="Path to results directory (default: auto-discover from cwd)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print extra diagnostic detail on each check",
    )
    parser.add_argument(
        "--check", "-c",
        nargs="+",
        type=int,
        metavar="N",
        help="Run only specific check numbers (1–10). Default: all.",
    )
    args = parser.parse_args()

    # ── Locate results dir ────────────────────────────────────────────────────
    if args.results_dir:
        results_dir = Path(args.results_dir).resolve()
    else:
        # Auto-discover: walk up from cwd looking for hypatiax/data/results
        candidates = [
            Path.cwd() / "hypatiax/data/results",
            Path.cwd() / "data/results",
            Path.cwd() / "results",
            Path.cwd(),
        ]
        results_dir = next((p for p in candidates if p.exists()), Path.cwd())

    print(f"\n{BOLD}=== HypatiaX §10.7 / PCA Bug Health Check ==={RESET}")
    print(f"Results dir : {results_dir}")
    print(f"Checks      : {args.check or 'all (1–10)'}")
    print()

    if not results_dir.exists():
        print(f"{RED}ERROR: results directory not found: {results_dir}{RESET}")
        sys.exit(1)

    # ── Run checks ────────────────────────────────────────────────────────────
    ALL_CHECKS = [
        (1,  check_1_disclosure_random_split_used,  "Bug 1  — 15_pca split_protocol_disclosure random_split_used"),
        (2,  check_2_hybrid_defi_nan_r2,            "Bug 2  — HybridDeFi success=True + r2=NaN"),
        (3,  check_3_hds_not_available,             "Bug 3  — HDS all 'not available' in exp2_pca"),
        (4,  check_4_ablation_engine_version,       "Rerun  — exp1_ablation engine still v5.1"),
        (5,  check_5_ablation_nan_rmse,             "Rerun  — NaN RMSE on Henderson-Hasselbalch / Rate Law"),
        (6,  check_6_mann_whitney_pairs,            "Rerun  — ablation_paired.json too few pairs for Mann-Whitney"),
        (7,  check_7_pca_summary_provenance,        "Data   — exp2_pca_4060_summary n_pass provenance divergence"),
        (8,  check_8_pca_test_r2_not_all_none,      "Bug 6  — pca_test_r2 all-None regression check"),
        (9,  check_9_fixc3_baseline_not_null,       "Data   — fixc3_baseline.json solve_rate null"),
        (10, check_10_hypatia_only_recount,         "§10.7  — pooled vs HypatiaX-only count"),
    ]

    active = set(args.check) if args.check else {n for n, *_ in ALL_CHECKS}

    results = {}
    for num, fn, label in ALL_CHECKS:
        if num not in active:
            continue
        print(f"{BOLD}Check {num:2d}: {label}{RESET}")
        try:
            passed = fn(results_dir, args.verbose)
        except Exception as e:
            fail(f"[{num}] Unexpected error: {e}")
            passed = False
        results[num] = passed
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    n_total   = len(results)
    n_passed  = sum(1 for v in results.values() if v)
    n_failed  = n_total - n_passed

    print(f"{BOLD}{'='*50}{RESET}")
    print(f"{BOLD}Summary: {n_passed}/{n_total} checks passed{RESET}")
    if n_failed == 0:
        print(f"{GREEN}All checks passed — no known issues remain.{RESET}")
    else:
        failed_nums = [str(n) for n, v in results.items() if not v]
        print(f"{RED}Failed checks: {', '.join(failed_nums)}{RESET}")
        print(f"{YELLOW}Re-read the check output above for remediation steps.{RESET}")
    print()

    sys.exit(0 if n_failed == 0 else 1)


if __name__ == "__main__":
    main()

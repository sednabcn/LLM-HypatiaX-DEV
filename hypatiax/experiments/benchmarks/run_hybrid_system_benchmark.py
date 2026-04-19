#!/usr/bin/env python3
"""
run_hybrid_system_benchmark.py

Master script to run the full benchmark suite:
  1. Hybrid system evaluation (in-distribution, batch mode)
  2. Extrapolation tests (73 cases)
  3. Performance analysis
  4. Final report

RESUME SUPPORT
──────────────
Run with  --resume  to pick up exactly where the previous run stopped:

  python run_hybrid_system_benchmark.py --resume

The script inspects the results directory to determine what is already done:

  Step 1 done    →  hybrid_defi_<ts>.json exists with ≥1 record  → skip
  Step 2 done    →  extrapolation_73cases_enhanced.json has 73 entries → skip
  Step 2 partial →  file has < 73 entries → pass --resume to the test script
                    so it continues from the last saved checkpoint case
  Step 3 done    →  report_hybrid_*.json exists → skip

Without --resume all four steps run from scratch (original behaviour).

OTHER FIXES (vs original)
─────────────────────────
  - Module-level random seeds for reproducibility
  - shlex.split + shell=False (no shell-injection risk)
  - --verbose forwarded to all sub-scripts
  - Fixed stray double-space in Step 3 command
  - Guarded division-by-zero (total==0, empty lists)
  - Added results_dir existence check before glob
  - json.load calls wrapped in try/except
  - Unknown decisions counted and shown (not silently dropped)
  - extrap_files glob corrected: *.csv → *.json
"""

import json
import random
import shlex
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# ── Reproducibility ────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Constants ──────────────────────────────────────────────────────────────
RESULTS_DIR        = Path("hypatiax/data/results")
EXTRAP_JSON        = RESULTS_DIR / "extrapolation_73cases_enhanced.json"
TOTAL_EXTRAP_CASES = 73


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint detection helpers
# ══════════════════════════════════════════════════════════════════════════════

def _hybrid_step_done() -> bool:
    """True if Step 1 produced a non-empty hybrid results file."""
    if not RESULTS_DIR.exists():
        return False
    files = sorted(RESULTS_DIR.glob("hybrid_defi_*.json"))
    if not files:
        return False
    try:
        with open(files[-1]) as f:
            data = json.load(f)
        raw = data.get("results", data) if isinstance(data, dict) else data
        return isinstance(raw, list) and len(raw) > 0
    except (json.JSONDecodeError, OSError):
        return False


def _extrap_cases_done() -> int:
    """Number of extrapolation cases saved in the checkpoint file (0 if none)."""
    if not EXTRAP_JSON.exists():
        return 0
    try:
        with open(EXTRAP_JSON) as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else 0
    except (json.JSONDecodeError, OSError):
        return 0


def _analysis_done() -> bool:
    """True if the performance-analysis output file already exists."""
    if not RESULTS_DIR.exists():
        return False
    return bool(sorted(RESULTS_DIR.glob("report_hybrid_*.json")))


# ══════════════════════════════════════════════════════════════════════════════
# Subprocess runner
# ══════════════════════════════════════════════════════════════════════════════

def run_command(cmd: str, description: str, verbose: bool = False) -> bool:
    """
    Run a subprocess safely (shlex + shell=False).
    Returns True on exit-code 0, False otherwise.
    """
    print("\n" + "=" * 80)
    print(f"▶️  {description}".center(80))
    print("=" * 80 + "\n")

    if verbose:
        print(f"   CMD: {cmd}\n")

    result = subprocess.run(shlex.split(cmd), shell=False, text=True)

    if result.returncode != 0:
        print(f"\n❌ Failed (exit {result.returncode}): {description}")
        return False

    print(f"\n✅ Completed: {description}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    start_time   = datetime.now()
    verbose_flag = "--verbose" in sys.argv
    resume_flag  = "--resume"  in sys.argv
    verbose_str  = "--verbose" if verbose_flag else ""

    print("=" * 80)
    print("🚀 FULL BENCHMARK SUITE — HYPATIAX DEFI 🚀".center(80))
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'RESUME (skip completed steps)' if resume_flag else 'FULL RUN'}")
    print()
    print("  1. Hybrid System  — in-distribution tests")
    print("  2. Extrapolation  — 73 diverse cases, aggressive splits")
    print("  3. Performance Analysis")
    print("  4. Final Report")
    print("=" * 80)

    # ── Step 1: Hybrid system ─────────────────────────────────────────────
    if resume_flag and _hybrid_step_done():
        print("\n[Step 1/4] ⏭  Hybrid System Evaluation — already done, skipping")
    else:
        run_command(
            f"python hypatiax/core/generation/hybrid_defi_system/"
            f"hybrid_system_nn_defi_domain.py --batch {verbose_str}".strip(),
            "Step 1/4: Hybrid System Evaluation",
            verbose=verbose_flag,
        ) or print("\n⚠️  Continuing despite errors in Step 1…")

    # ── Step 2: Extrapolation tests ────────────────────────────────────────
    n_done = _extrap_cases_done() if resume_flag else 0

    if resume_flag and n_done >= TOTAL_EXTRAP_CASES:
        print(f"\n[Step 2/4] ⏭  Extrapolation Tests — "
              f"all {TOTAL_EXTRAP_CASES} cases done, skipping")
    else:
        if resume_flag and n_done > 0:
            # Partial run: tell the test script to continue from the checkpoint
            print(f"\n[Step 2/4] ▶  Extrapolation Tests — "
                  f"resuming from case {n_done + 1}/{TOTAL_EXTRAP_CASES} "
                  f"({n_done} already done)")
            extrap_flags = f"--resume {verbose_str}".strip()
        else:
            extrap_flags = verbose_str

        run_command(
            f"python hypatiax/experiments/tests/test_enhanced_defi_extrapolation.py "
            f"{extrap_flags}".strip(),
            "Step 2/4: Extrapolation Tests — 73 cases"
            + (f" (resuming from case {n_done + 1})" if n_done > 0 else ""),
            verbose=verbose_flag,
        ) or print("\n⚠️  Continuing despite errors in Step 2…")

    # ── Step 3: Performance analysis ──────────────────────────────────────
    if resume_flag and _analysis_done():
        print("\n[Step 3/4] ⏭  Performance Analysis — already done, skipping")
    else:
        run_command(
            f"python hypatiax/analysis/analyze_hybrid_performance.py "
            f"--results-dir {RESULTS_DIR} {verbose_str}".strip(),
            "Step 3/4: Performance Analysis",
            verbose=verbose_flag,
        ) or print("\n⚠️  Continuing despite errors in Step 3…")

    # ── Step 4: Final report ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("▶️  Step 4/4: Generating Final Report".center(80))
    print("=" * 80 + "\n")
    generate_final_report()

    end_time = datetime.now()
    print("\n" + "=" * 80)
    print("🎉 BENCHMARK COMPLETE 🎉".center(80))
    print("=" * 80)
    print(f"Duration:  {end_time - start_time}")
    print(f"End time:  {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📊 Results saved in: {RESULTS_DIR}/")
    print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# Final report generator
# ══════════════════════════════════════════════════════════════════════════════


# ── Known structurally-intractable case names (mirrors test catalogue) ────────
# These are excluded from the standard aggregate just as the test script does.
_INTRACTABLE_CASES = {
    "AMM arbitrage profit",
    "Optimal LP Position (Kelly)",
    "Options Delta",
    "Black-Scholes Call Price",
    "Put option intrinsic",        # piecewise max(K-S,0) — test set collapses to one side
    "Impermanent loss breakeven",
}

# ── Clip bounds for robust R² stats ──────────────────────────────────────────
_CLIP_LO, _CLIP_HI = -10.0, 1.0


def _robust_stats(scores):
    """
    Compute robust statistics on a list of R² floats.
    Filters NaN, then reports median, clipped mean, >0.9 %, >0.99 %, catastrophic.
    """
    valid = [s for s in scores if not np.isnan(s)]
    if not valid:
        return dict(n=0, median=float("nan"), mean_clipped=float("nan"),
                    pct_09=0.0, pct_099=0.0, n_catastrophic=0)
    arr     = np.array(valid)
    clipped = np.clip(arr, _CLIP_LO, _CLIP_HI)
    n_cat   = int(np.sum(arr < -1.0))
    return dict(
        n            = len(valid),
        median       = float(np.median(arr)),
        mean_clipped = float(np.mean(clipped)),
        pct_09       = float(np.mean(arr > 0.9)  * 100),
        pct_099      = float(np.mean(arr > 0.99) * 100),
        n_catastrophic = n_cat,
    )


def _fmt(s):
    """Format a robust-stats dict for one-line printing."""
    if s["n"] == 0:
        return "n/a"
    return (
        f"median={s['median']:+.4f}  "
        f"clipped_mean={s['mean_clipped']:+.4f}  "
        f">0.9: {s['pct_09']:5.1f}%  "
        f">0.99: {s['pct_099']:5.1f}%  "
        f"catastrophic: {s['n_catastrophic']}"
    )


def generate_final_report():
    """Generate a comprehensive report from saved result files."""

    if not RESULTS_DIR.exists():
        print(f"\u274c Results directory not found: {RESULTS_DIR}")
        print("   Run Steps 1\u20133 first to generate result files.")
        return

    hybrid_files = sorted(RESULTS_DIR.glob("hybrid_defi_*.json"))
    report_files = sorted(RESULTS_DIR.glob("report_hybrid_*.json"))
    extrap_files = sorted(RESULTS_DIR.glob("extrapolation_*.json"))

    # ── Section header ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("\U0001f4ca FINAL REPORT".center(80))
    print("=" * 80)

    # ── 1. Extrapolation comparison table ─────────────────────────────────
    extrap_73  = [f for f in extrap_files if "73cases" in f.name]
    extrap_src = extrap_73[-1] if extrap_73 else (extrap_files[-1] if extrap_files else None)

    if extrap_src:
        try:
            with open(extrap_src) as f:
                extrap_data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"\u26a0\ufe0f  Could not read extrapolation file: {exc}")
            extrap_data = []

        if extrap_data:
            # Deduplicate by test_case name (same logic as _load_checkpoint)
            seen = {}
            for r in extrap_data:
                seen[r.get("test_case", id(r))] = r
            extrap_data = list(seen.values())

            # Partition standard vs intractable
            standard    = [r for r in extrap_data
                           if r.get("test_case") not in _INTRACTABLE_CASES]
            intractable = [r for r in extrap_data
                           if r.get("test_case") in _INTRACTABLE_CASES]

            n_saved  = len(extrap_data)
            n_std    = len(standard)
            n_intr   = len(intractable)
            status   = (
                f"complete ({n_saved}/{TOTAL_EXTRAP_CASES})"
                if n_saved >= TOTAL_EXTRAP_CASES
                else f"\u26a0\ufe0f  PARTIAL \u2014 {n_saved}/{TOTAL_EXTRAP_CASES} cases"
            )
            print(f"\n\U0001f4d0 Extrapolation Results  [{extrap_src.name}]  [{status}]")
            print(f"   Standard cases: {n_std}   |   Intractable (excluded): {n_intr}")

            def _collect(results_list, method_key):
                out = []
                for r in results_list:
                    v = r.get("results", {}).get(method_key, {}).get("test_r2")
                    if v is not None:
                        out.append(float(v))
                return out

            methods = [
                ("Pure LLM",       "pure_llm"),
                ("Neural Network", "neural_network"),
                ("Hybrid",         "hybrid"),
            ]

            # Full comparison table
            print()
            hdr = f"  {'Method':<18} {'n':>4}  {'Median R²':>10}  {'Clip-Mean':>10}  "
            hdr += f"{'> 0.9 %':>8}  {'> 0.99 %':>9}  {'Catastro.':>9}"
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for label, key in methods:
                sc = _collect(standard, key)
                st = _robust_stats(sc)
                med  = f"{st['median']:+.4f}"  if st["n"] else "  n/a  "
                cm   = f"{st['mean_clipped']:+.4f}" if st["n"] else "  n/a  "
                p9   = f"{st['pct_09']:5.1f}%" if st["n"] else "  n/a "
                p99  = f"{st['pct_099']:5.1f}%" if st["n"] else "  n/a "
                cat  = str(st["n_catastrophic"]) if st["n"] else "-"
                print(f"  {label:<18} {st['n']:>4}  {med:>10}  {cm:>10}  "
                      f"{p9:>8}  {p99:>9}  {cat:>9}")

            # Intractable summary
            if intractable:
                print(f"\n  Intractable cases ({n_intr}) — reported separately, "
                      "excluded from above:")
                for r in intractable:
                    name = r.get("test_case", "?")
                    h_r2 = r.get("results", {}).get("hybrid", {}).get("test_r2")
                    l_r2 = r.get("results", {}).get("pure_llm", {}).get("test_r2")
                    h_s  = f"{h_r2:+.3f}" if h_r2 is not None else "  n/a"
                    l_s  = f"{l_r2:+.3f}" if l_r2 is not None else "  n/a"
                    print(f"    * {name:<40}  LLM={l_s}  Hybrid={h_s}")

            if n_saved < TOTAL_EXTRAP_CASES:
                remaining = TOTAL_EXTRAP_CASES - n_saved
                print(f"\n  \u2139\ufe0f  {remaining} case(s) still to run.  Resume with:")
                print(f"     python run_hybrid_system_benchmark.py --resume")

    # ── 2. In-distribution hybrid results (Step 1) ────────────────────────
    if hybrid_files:
        try:
            with open(hybrid_files[-1]) as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"\u26a0\ufe0f  Could not read hybrid results: {exc}")
            raw = {}

        hybrid_data = raw.get("results", []) if isinstance(raw, dict) else raw

        if hybrid_data:
            print(f"\n\U0001f3e2 In-Distribution Hybrid Results  [{hybrid_files[-1].name}]")

            # Decision breakdown
            decisions: dict = {}
            for r in hybrid_data:
                dec = r.get("decision", "unknown")
                decisions[dec] = decisions.get(dec, 0) + 1
            total = sum(decisions.values())

            print(f"   Decision breakdown (n={total}):")
            for key in sorted(decisions):
                cnt = decisions[key]
                print(f"     {key.capitalize():<12}: {cnt:>3}  ({cnt / total * 100:.1f}%)")

            # R² by decision type
            print(f"   R\u00b2 by decision (in-distribution):")
            for label in ("llm", "nn", "ensemble"):
                group = [r for r in hybrid_data if r.get("decision") == label]
                if not group:
                    continue
                r2s = []
                for r in group:
                    v = r.get("evaluation", {}).get("r2")
                    if v is None:
                        v = r.get("hybrid_train_r2")
                    if v is not None:
                        r2s.append(float(v))
                st = _robust_stats(r2s)
                print(f"     {label.upper():<12}: {_fmt(st)}")

            # Poor performers
            def _get_r2(r):
                v = r.get("evaluation", {}).get("r2")
                return v if v is not None else r.get("hybrid_train_r2")

            poor = [r for r in hybrid_data if (_get_r2(r) or 1.0) < 0.80]
            if poor:
                print(f"\n   \u26a0\ufe0f  {len(poor)} in-distribution case(s) with R\u00b2 < 0.80:")
                for r in poor[:5]:
                    print(f"     • {r.get('description', 'Unknown')[:65]}\u2026")
            else:
                print("\n   \u2705 All in-distribution cases: R\u00b2 \u2265 0.80")

    # ── 3. Performance analysis report ────────────────────────────────────
    if report_files:
        try:
            with open(report_files[-1]) as f:
                report_data = json.load(f)
            overall = report_data.get("overall", {})
            print(f"\n\U0001f4c8 Performance Analysis Report  [{report_files[-1].name}]")
            print(f"   Total cases:    {overall.get('total_cases', 0)}")
            print(f"   Success rate:   {overall.get('success_rate', 0) * 100:.1f}%")
            # Use median if available, fall back to mean with a warning
            med_r2 = overall.get("median_r2")
            if med_r2 is not None:
                print(f"   Median R\u00b2:      {med_r2:.6f}")
            else:
                raw_mean = overall.get("mean_r2", 0)
                print(f"   Mean R\u00b2 (raw):  {raw_mean:.6f}  "
                      "\u26a0\ufe0f use median for extrapolation benchmarks")
            print(f"   By domain:")
            for domain, dstats in report_data.get("by_domain", {}).items():
                dm = dstats.get("median_r2", dstats.get("mean_r2", 0))
                print(f"     {domain:<24}: R\u00b2 = {dm:.4f}  "
                      f"({dstats.get('total', 0)} cases)")
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            print(f"\u26a0\ufe0f  Could not read report file: {exc}")

    # ── 4. Output file listing ─────────────────────────────────────────────
    print(f"\n\U0001f4c1 Result files in {RESULTS_DIR}:")
    if hybrid_files:
        print(f"   Hybrid results:        {hybrid_files[-1].name}")
    if report_files:
        print(f"   Performance report:    {report_files[-1].name}")
    if extrap_src:
        print(f"   Extrapolation data:    {extrap_src.name}")
    print()


if __name__ == "__main__":
    main()

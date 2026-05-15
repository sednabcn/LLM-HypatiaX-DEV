#!/usr/bin/env python3
"""
scripts/run_analysis.py
=======================
HypatiaX post-consolidation statistical analysis.

Called exclusively by ci_analysis.yml after _merged.json has been committed.
NEVER called by workers or the consolidate job.

Input
-----
_merged.json  — produced by scripts/merge_shards.py
    List of records (one per equation / task), each with shape:

    {
        "equation_id":              str,
        "difficulty":               str,          # "easy" | "medium" | "hard"
        "formula_type":             str,          # "rational" | "transcendental" | ...
        "extrapolation_intractable": bool,
        "results": {
            "pure_llm":       { "train_r2": float|null, "test_r2": float|null,
                                "success": bool, "time_s": float,
                                "extrapolation_gap": float|null,
                                "stability_score":   float|null },
            "neural_network": { ... , "timed_out": bool },
            "hybrid":         { ... , "decision": str }
        }
    }

    Records with "extrapolation_intractable": true are excluded from
    primary method comparisons (counted separately).

Outputs written to --output-dir
--------------------------------
_analysis.json  Structured results (machine-readable).
                Includes "fatal_conditions" list — non-empty list causes
                ci_analysis.yml to fail the workflow after committing.
_report.md      Human-readable Markdown report.

Fatal conditions (any → workflow aborted after commit)
------------------------------------------------------
- n_merged == 0          : empty dataset
- all methods 0% success : complete failure across the board
- hybrid never beats NN  : hybrid strictly worse than NN on ALL equations
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.stats import mannwhitneyu
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHODS = ["pure_llm", "neural_network", "hybrid"]
METHOD_LABELS = {
    "pure_llm":       "Pure LLM",
    "neural_network": "Neural Net",
    "hybrid":         "Hybrid",
}

# R² threshold above which a result counts as a "success" for coverage tables.
R2_SUCCESS_THRESHOLD = 0.80

# R² clip range for Mann-Whitney (avoids -∞ distorting rank sums).
R2_CLIP_LO = -10.0
R2_CLIP_HI = 1.0

# Fatal-condition thresholds.
MIN_RECORDS_FOR_STATS = 3   # below this, flag fatal
HYBRID_MUST_WIN_FRACTION = 0.0  # hybrid must beat NN on >0% of equations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_finite(v: Any) -> bool:
    if v is None:
        return False
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _safe_float(v: Any, fallback: float = float("nan")) -> float:
    if v is None:
        return fallback
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


def _r2_values(records: list[dict], method: str) -> list[float]:
    """Clipped, finite test_r2 values for a method across all records."""
    out = []
    for r in records:
        v = _safe_float(r.get("results", {}).get(method, {}).get("test_r2"))
        if math.isfinite(v):
            out.append(max(R2_CLIP_LO, min(R2_CLIP_HI, v)))
    return out


def _success_rate(records: list[dict], method: str) -> tuple[int, int, float]:
    """Returns (n_success, n_total, rate) using explicit 'success' flag."""
    n_total = 0
    n_success = 0
    for r in records:
        res = r.get("results", {}).get(method)
        if res is None:
            continue
        n_total += 1
        if res.get("success", False):
            n_success += 1
    rate = n_success / n_total if n_total else 0.0
    return n_success, n_total, rate


def _r2_success_rate(records: list[dict], method: str,
                     threshold: float = R2_SUCCESS_THRESHOLD) -> tuple[int, int, float]:
    """Success = test_r2 >= threshold (R²-based, independent of 'success' flag)."""
    n_total = 0
    n_above = 0
    for r in records:
        v = _safe_float(r.get("results", {}).get(method, {}).get("test_r2"))
        if math.isfinite(v):
            n_total += 1
            if v >= threshold:
                n_above += 1
    rate = n_above / n_total if n_total else 0.0
    return n_above, n_total, rate


def _median(vals: list[float]) -> float | None:
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return None
    return float(np.median(finite))


def _mean(vals: list[float]) -> float | None:
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return None
    return float(np.mean(finite))


def _mann_whitney(a: list[float], b: list[float]) -> dict:
    """Two-sided Mann-Whitney U test. Returns stat, p, direction."""
    if not _SCIPY_OK:
        return {"available": False, "reason": "scipy not installed"}
    if len(a) < 2 or len(b) < 2:
        return {"available": False, "reason": "insufficient samples"}
    try:
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        direction = "a_greater" if float(np.median(a)) > float(np.median(b)) else "b_greater"
        return {
            "available": True,
            "statistic": round(float(stat), 4),
            "p_value":   round(float(p), 6),
            "significant_05": float(p) < 0.05,
            "significant_01": float(p) < 0.01,
            "direction": direction,
            "n_a": len(a),
            "n_b": len(b),
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyse(records: list[dict], experiment: str) -> dict:
    """
    Run full statistical analysis on a list of merged records.
    Returns a dict that is written verbatim to _analysis.json.
    """

    # -- Partition: standard vs intractable ------------------------------------
    standard     = [r for r in records if not r.get("extrapolation_intractable", False)]
    intractable  = [r for r in records if r.get("extrapolation_intractable", False)]

    n_total      = len(records)
    n_standard   = len(standard)
    n_intractable= len(intractable)

    # -- Per-method summary ----------------------------------------------------
    method_summary: dict[str, dict] = {}
    for m in METHODS:
        r2_vals  = _r2_values(standard, m)
        suc_n, suc_d, suc_r   = _success_rate(standard, m)
        r2s_n, r2s_d, r2s_r   = _r2_success_rate(standard, m)
        method_summary[m] = {
            "n_records":          suc_d,
            "n_success_flag":     suc_n,
            "success_rate_flag":  round(suc_r, 4),
            "n_r2_above_80":      r2s_n,
            "r2_above_80_rate":   round(r2s_r, 4),
            "median_test_r2":     _median(r2_vals),
            "mean_test_r2":       _mean(r2_vals),
            "n_finite_r2":        len(r2_vals),
        }

    # -- Coverage gaps ---------------------------------------------------------
    # Equations where NO method achieves R² >= threshold.
    coverage_gaps: list[dict] = []
    for r in standard:
        eq_id = r.get("equation_id", "?")
        best  = max(
            (_safe_float(r.get("results", {}).get(m, {}).get("test_r2"))
             for m in METHODS),
            default=float("nan"),
        )
        if not math.isfinite(best) or best < R2_SUCCESS_THRESHOLD:
            coverage_gaps.append({
                "equation_id": eq_id,
                "difficulty":  r.get("difficulty"),
                "formula_type": r.get("formula_type"),
                "best_test_r2": None if not math.isfinite(best) else round(best, 4),
                "per_method": {
                    m: round(_safe_float(r.get("results", {}).get(m, {}).get("test_r2")), 4)
                    if math.isfinite(_safe_float(r.get("results", {}).get(m, {}).get("test_r2")))
                    else None
                    for m in METHODS
                },
            })

    # -- Mann-Whitney pairwise comparisons (on standard records only) ----------
    r2_llm  = _r2_values(standard, "pure_llm")
    r2_nn   = _r2_values(standard, "neural_network")
    r2_hyb  = _r2_values(standard, "hybrid")

    mann_whitney = {
        "hybrid_vs_llm": _mann_whitney(r2_hyb, r2_llm),
        "hybrid_vs_nn":  _mann_whitney(r2_hyb, r2_nn),
        "nn_vs_llm":     _mann_whitney(r2_nn,  r2_llm),
    }

    # -- Per-difficulty breakdown -----------------------------------------------
    difficulties = sorted({r.get("difficulty", "unknown") for r in standard})
    by_difficulty: dict[str, dict] = {}
    for diff in difficulties:
        sub = [r for r in standard if r.get("difficulty") == diff]
        by_difficulty[diff] = {
            m: {
                "n": len([r for r in sub if r.get("results", {}).get(m) is not None]),
                "median_test_r2": _median(_r2_values(sub, m)),
                "r2_above_80_rate": round(_r2_success_rate(sub, m)[2], 4),
            }
            for m in METHODS
        }

    # -- Per-formula-type breakdown ---------------------------------------------
    ftypes = sorted({r.get("formula_type", "unknown") for r in standard})
    by_formula_type: dict[str, dict] = {}
    for ft in ftypes:
        sub = [r for r in standard if r.get("formula_type") == ft]
        by_formula_type[ft] = {
            m: {
                "n": len([r for r in sub if r.get("results", {}).get(m) is not None]),
                "median_test_r2": _median(_r2_values(sub, m)),
                "r2_above_80_rate": round(_r2_success_rate(sub, m)[2], 4),
            }
            for m in METHODS
        }

    # -- Extrapolation gap analysis --------------------------------------------
    # extrapolation_gap = train_r2 - test_r2 (higher = more overfit / poor extrapolation)
    gap_summary: dict[str, dict] = {}
    for m in METHODS:
        gaps = []
        for r in standard:
            g = _safe_float(r.get("results", {}).get(m, {}).get("extrapolation_gap"))
            if math.isfinite(g):
                gaps.append(g)
        gap_summary[m] = {
            "mean_gap":   _mean(gaps),
            "median_gap": _median(gaps),
            "n":          len(gaps),
        }

    # -- Timing summary --------------------------------------------------------
    timing: dict[str, dict] = {}
    for m in METHODS:
        times = [
            _safe_float(r.get("results", {}).get(m, {}).get("time_s"))
            for r in standard
            if math.isfinite(_safe_float(r.get("results", {}).get(m, {}).get("time_s")))
        ]
        timing[m] = {
            "mean_s":   _mean(times),
            "median_s": _median(times),
            "total_s":  round(sum(times), 2) if times else None,
            "n":        len(times),
        }

    # -- Hybrid decision breakdown ---------------------------------------------
    decisions: dict[str, int] = {}
    for r in standard:
        dec = r.get("results", {}).get("hybrid", {}).get("decision")
        if dec:
            decisions[dec] = decisions.get(dec, 0) + 1

    # -- Hybrid vs NN head-to-head (equation level) ----------------------------
    hyb_beats_nn   = 0
    nn_beats_hyb   = 0
    tied           = 0
    n_both_finite  = 0
    for r in standard:
        hyb_r2 = _safe_float(r.get("results", {}).get("hybrid",         {}).get("test_r2"))
        nn_r2  = _safe_float(r.get("results", {}).get("neural_network", {}).get("test_r2"))
        if math.isfinite(hyb_r2) and math.isfinite(nn_r2):
            n_both_finite += 1
            if hyb_r2 > nn_r2 + 1e-6:
                hyb_beats_nn += 1
            elif nn_r2 > hyb_r2 + 1e-6:
                nn_beats_hyb += 1
            else:
                tied += 1

    hybrid_vs_nn_headtohead = {
        "n_equations_both_finite": n_both_finite,
        "hybrid_wins": hyb_beats_nn,
        "nn_wins":     nn_beats_hyb,
        "tied":        tied,
        "hybrid_win_rate": round(hyb_beats_nn / n_both_finite, 4) if n_both_finite else None,
    }

    # -- Fatal conditions ------------------------------------------------------
    fatal: list[str] = []

    if n_total == 0:
        fatal.append("EMPTY_DATASET: _merged.json contains 0 records.")

    if n_standard == 0 and n_total > 0:
        fatal.append(
            f"ALL_INTRACTABLE: all {n_total} records are marked extrapolation_intractable; "
            "no standard equations to analyse."
        )

    if n_standard < MIN_RECORDS_FOR_STATS and n_standard > 0:
        fatal.append(
            f"TOO_FEW_RECORDS: only {n_standard} standard records (need ≥ {MIN_RECORDS_FOR_STATS}) "
            "for meaningful statistics."
        )

    # All methods 0% success (success flag, not R²-based)
    all_zero_success = all(
        method_summary.get(m, {}).get("success_rate_flag", 0.0) == 0.0
        for m in METHODS
        if method_summary.get(m, {}).get("n_records", 0) > 0
    )
    if all_zero_success and n_standard > 0:
        fatal.append(
            "TOTAL_FAILURE: all methods report 0% success across all standard equations. "
            "Check experiment scripts for systematic errors."
        )

    # Hybrid never beats NN (on any equation with both finite)
    if n_both_finite >= MIN_RECORDS_FOR_STATS and hyb_beats_nn == 0 and nn_beats_hyb > 0:
        fatal.append(
            f"HYBRID_NEVER_BEATS_NN: hybrid ≤ neural_network on all {n_both_finite} "
            "equations where both produced finite R². Possible routing or fix regression."
        )

    # -- Assemble output -------------------------------------------------------
    result = {
        "experiment": experiment,
        "n_total":      n_total,
        "n_standard":   n_standard,
        "n_intractable": n_intractable,
        "r2_success_threshold": R2_SUCCESS_THRESHOLD,
        "method_summary":       method_summary,
        "mann_whitney":         mann_whitney,
        "coverage_gaps":        coverage_gaps,
        "n_coverage_gaps":      len(coverage_gaps),
        "by_difficulty":        by_difficulty,
        "by_formula_type":      by_formula_type,
        "extrapolation_gap_summary": gap_summary,
        "timing":               timing,
        "hybrid_decisions":     decisions,
        "hybrid_vs_nn_headtohead": hybrid_vs_nn_headtohead,
        "fatal_conditions":     fatal,
    }

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _pct(rate: float | None) -> str:
    if rate is None:
        return "N/A"
    return f"{rate * 100:.1f}%"


def _r2f(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v:.4f}"


def _mw_row(mw: dict) -> str:
    if not mw.get("available"):
        return f"  N/A ({mw.get('reason', '?')})"
    sig = "**" if mw.get("significant_05") else ""
    return (
        f"  U={mw['statistic']},  p={mw['p_value']:.4f}{sig},  "
        f"direction={mw['direction']},  n=({mw['n_a']}, {mw['n_b']})"
    )


def write_report(analysis: dict, path: Path) -> None:
    exp   = analysis["experiment"]
    lines = []

    def h(level: int, text: str):
        lines.append(f"\n{'#' * level} {text}\n")

    def p(*args):
        lines.append(" ".join(str(a) for a in args))

    h(1, f"HypatiaX Analysis Report — `{exp}`")
    p(f"N total: {analysis['n_total']} "
      f"| N standard: {analysis['n_standard']} "
      f"| N intractable: {analysis['n_intractable']}")
    p(f"R² success threshold: {analysis['r2_success_threshold']}")

    # -- Fatal conditions ------------------------------------------------------
    fatal = analysis.get("fatal_conditions", [])
    if fatal:
        h(2, "⚠️ Fatal Conditions")
        for fc in fatal:
            lines.append(f"- **{fc}**")
    else:
        h(2, "✅ No Fatal Conditions")

    # -- Method summary table --------------------------------------------------
    h(2, "Method Summary (standard equations only)")
    lines.append(
        "| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |"
    )
    lines.append(
        "|--------|---|-----------------|----------|----------------|--------------|"
    )
    for m in METHODS:
        s = analysis["method_summary"].get(m, {})
        lines.append(
            f"| {METHOD_LABELS[m]} "
            f"| {s.get('n_records', 0)} "
            f"| {_pct(s.get('success_rate_flag'))} "
            f"| {_pct(s.get('r2_above_80_rate'))} "
            f"| {_r2f(s.get('median_test_r2'))} "
            f"| {_r2f(s.get('mean_test_r2'))} |"
        )

    # -- Mann-Whitney ----------------------------------------------------------
    h(2, "Mann-Whitney U Tests (two-sided, clipped R², standard equations)")
    mw = analysis.get("mann_whitney", {})
    for pair, label in [
        ("hybrid_vs_llm", "Hybrid vs Pure LLM"),
        ("hybrid_vs_nn",  "Hybrid vs Neural Net"),
        ("nn_vs_llm",     "Neural Net vs Pure LLM"),
    ]:
        h(3, label)
        p(_mw_row(mw.get(pair, {})))
    p("_** = p < 0.05_")

    # -- Hybrid vs NN head-to-head ---------------------------------------------
    h(2, "Hybrid vs Neural Net (head-to-head, equation level)")
    hh = analysis.get("hybrid_vs_nn_headtohead", {})
    p(f"Equations with both finite R²: {hh.get('n_equations_both_finite', 0)}")
    p(f"Hybrid wins:  {hh.get('hybrid_wins', 0)}  ({_pct(hh.get('hybrid_win_rate'))})")
    p(f"NN wins:      {hh.get('nn_wins', 0)}")
    p(f"Tied:         {hh.get('tied', 0)}")

    # -- Coverage gaps ---------------------------------------------------------
    gaps = analysis.get("coverage_gaps", [])
    h(2, f"Coverage Gaps ({len(gaps)} equations with best R² < {analysis['r2_success_threshold']})")
    if gaps:
        lines.append("| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |")
        lines.append("|----------|------------|------|---------|-----|----|----|")
        for g in gaps:
            pm = g.get("per_method", {})
            lines.append(
                f"| {g['equation_id']} "
                f"| {g.get('difficulty', '?')} "
                f"| {g.get('formula_type', '?')} "
                f"| {_r2f(g.get('best_test_r2'))} "
                f"| {_r2f(pm.get('pure_llm'))} "
                f"| {_r2f(pm.get('neural_network'))} "
                f"| {_r2f(pm.get('hybrid'))} |"
            )
    else:
        p("_None — all standard equations have at least one method achieving R² ≥ threshold._")

    # -- By difficulty ---------------------------------------------------------
    h(2, "R²≥0.80 Rate by Difficulty")
    by_diff = analysis.get("by_difficulty", {})
    if by_diff:
        lines.append("| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |")
        lines.append("|------------|---|-------------|------------|----------------|")
        for diff, data in sorted(by_diff.items()):
            n = data.get("pure_llm", {}).get("n", "?")
            lines.append(
                f"| {diff} | {n} "
                f"| {_pct(data.get('pure_llm', {}).get('r2_above_80_rate'))} "
                f"| {_pct(data.get('neural_network', {}).get('r2_above_80_rate'))} "
                f"| {_pct(data.get('hybrid', {}).get('r2_above_80_rate'))} |"
            )

    # -- By formula type -------------------------------------------------------
    h(2, "Median Test R² by Formula Type")
    by_ft = analysis.get("by_formula_type", {})
    if by_ft:
        lines.append("| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |")
        lines.append("|--------------|---|---------------|--------------|------------------|")
        for ft, data in sorted(by_ft.items()):
            n = data.get("pure_llm", {}).get("n", "?")
            lines.append(
                f"| {ft} | {n} "
                f"| {_r2f(data.get('pure_llm', {}).get('median_test_r2'))} "
                f"| {_r2f(data.get('neural_network', {}).get('median_test_r2'))} "
                f"| {_r2f(data.get('hybrid', {}).get('median_test_r2'))} |"
            )

    # -- Extrapolation gap -----------------------------------------------------
    h(2, "Extrapolation Gap (train R² − test R²)")
    gap_s = analysis.get("extrapolation_gap_summary", {})
    lines.append("| Method | Mean gap | Median gap | N |")
    lines.append("|--------|----------|------------|---|")
    for m in METHODS:
        g = gap_s.get(m, {})
        lines.append(
            f"| {METHOD_LABELS[m]} "
            f"| {_r2f(g.get('mean_gap'))} "
            f"| {_r2f(g.get('median_gap'))} "
            f"| {g.get('n', 0)} |"
        )

    # -- Timing ----------------------------------------------------------------
    h(2, "Wall-clock Timing (standard equations)")
    timing = analysis.get("timing", {})
    lines.append("| Method | Mean (s) | Median (s) | Total (s) | N |")
    lines.append("|--------|----------|------------|-----------|---|")
    for m in METHODS:
        t = timing.get(m, {})
        lines.append(
            f"| {METHOD_LABELS[m]} "
            f"| {_r2f(t.get('mean_s'))} "
            f"| {_r2f(t.get('median_s'))} "
            f"| {t.get('total_s', 'N/A')} "
            f"| {t.get('n', 0)} |"
        )

    # -- Hybrid decisions ------------------------------------------------------
    h(2, "Hybrid Routing Decisions")
    decisions = analysis.get("hybrid_decisions", {})
    if decisions:
        lines.append("| Decision | Count |")
        lines.append("|----------|-------|")
        for dec, cnt in sorted(decisions.items(), key=lambda x: -x[1]):
            lines.append(f"| {dec} | {cnt} |")
    else:
        p("_No hybrid decision data available._")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="HypatiaX post-consolidation statistical analysis."
    )
    ap.add_argument("--experiment",  required=True,  help="Experiment ID (e.g. exp1)")
    ap.add_argument("--merged-json", required=True,  help="Path to _merged.json")
    ap.add_argument("--output-dir",  required=True,  help="Directory to write _report.md and _analysis.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    merged_path = Path(args.merged_json)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not merged_path.exists():
        print(f"::error::_merged.json not found at {merged_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {merged_path} …")
    with open(merged_path, encoding="utf-8") as f:
        raw = json.load(f)

    # _merged.json is either a list of records or a dict keyed by task_id.
    if isinstance(raw, dict):
        records = list(raw.values())
    elif isinstance(raw, list):
        records = raw
    else:
        print(f"::error::Unexpected _merged.json top-level type: {type(raw)}", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(records)} records loaded.")

    if not _SCIPY_OK:
        print("WARNING: scipy not available — Mann-Whitney tests will be skipped.", file=sys.stderr)

    print("Running analysis …")
    analysis = analyse(records, experiment=args.experiment)

    # Write outputs.
    analysis_path = output_dir / "_analysis.json"
    report_path   = output_dir / "_report.md"

    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"✅ _analysis.json → {analysis_path}")

    write_report(analysis, report_path)
    print(f"✅ _report.md     → {report_path}")

    fatal = analysis.get("fatal_conditions", [])
    if fatal:
        print(f"\n⚠️  {len(fatal)} fatal condition(s) detected:", file=sys.stderr)
        for fc in fatal:
            print(f"  - {fc}", file=sys.stderr)
        print("\nReport committed. ci_analysis.yml will abort the workflow.", file=sys.stderr)
        # Exit 0 here — the CI abort step in ci_analysis.yml reads fatal_conditions
        # from _analysis.json and calls sys.exit(1) itself, AFTER the commit step.
        sys.exit(0)

    print("\nAnalysis complete. No fatal conditions.")


if __name__ == "__main__":
    main()

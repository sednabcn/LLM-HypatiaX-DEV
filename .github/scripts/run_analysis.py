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
        "equation_id":               str,
        "difficulty":                str,   # "easy" | "medium" | "hard"
        "formula_type":              str,   # "rational" | "transcendental" | ...
        "extrapolation_intractable": bool,
        "results": {
            "pure_llm":       { "train_r2": float|null, "test_r2": float|null,
                                "success": bool, "time_s": float,
                                "extrapolation_gap": float|null,
                                "stability_score":   float|null },
            "neural_network": { ..., "timed_out": bool },
            "hybrid":         { ..., "decision": str }
        }
    }

    Records with "extrapolation_intractable": true are excluded from
    primary method comparisons (counted separately).

Experiment modes
----------------
Each experiment ID maps to a mode that controls which fatals fire:

  "standard"     — exp1, exp1b, suppA, suppB, suppB_sc
                   Full analysis; all fatals active.

  "ablation"     — exp2_feynman
                   Paired pysr_only vs hypatia comparison on extrap_r2_far.
                   Three-tier MW (all-N / excl-train-fail / success-subset),
                   Fisher, Spearman, complexity distributions, threshold sweep,
                   and LOO sensitivity.  Routes to analyse_ablation().
                   NOTE: exp1_ablation is NOT dispatched by ci_experiment.yml
                   or ci_schedule_all.yml — it has no worker or result_subdir.
                   It is kept in EXPERIMENT_MODE for manual standalone use only.

  "ood"          — extrap
                   OOD/out-of-distribution run. Hybrid legitimately loses NN.
                   HYBRID_NEVER_BEATS_NN is demoted to INFO_ (non-blocking).

  "pysr"         — exp3, exp3b
                   Nguyen-12 / PySR runs. No hybrid key in schema.
                   TOTAL_FAILURE and HYBRID_NEVER_BEATS_NN fatals suppressed.
                   Method-comparison sections written as N/A.

  "multi_method" — exp2, hybrid_all_domains
                   4-method output (HybridSystemLLMNN all-domains unmapped).
                   TOTAL_FAILURE and HYBRID_NEVER_BEATS_NN active.
                   WARN_MULTI_METHOD appended (non-blocking).

  "instability"  — instability
                   Writes only CSVs/figures; no _merged.json with method results.
                   ci_analysis.yml short-circuits before calling this script.
                   Mode kept here for completeness / manual dispatch fallback.

Fatal-condition prefix conventions
-----------------------------------
  (no prefix)  — hard fatal; ci_analysis.yml aborts the workflow.
  INFO_        — informational; logged but workflow continues.
  WARN_        — warning; logged but workflow continues.

Outputs written to --output-dir
--------------------------------
_analysis.json  Structured results (machine-readable).
                Includes "fatal_conditions" list; non-INFO_/non-WARN_ entries
                cause ci_analysis.yml to fail the workflow after committing.
_report.md      Human-readable Markdown report.
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

# Extrapolation success threshold for Tier-3 MW (paper §10.7: "9/30 successes").
# An equation counts as a "success" if hypatia.extrap_r2_far >= this value.
# Matches the paper's bold criterion (R²>0.99 in the Feynman table).
EXTRAP_SUCCESS_THRESHOLD = 0.99

# Fatal-condition thresholds.
MIN_RECORDS_FOR_STATS = 3   # below this, flag fatal
HYBRID_MUST_WIN_FRACTION = 0.0  # hybrid must beat NN on >0% of equations

# ---------------------------------------------------------------------------
# Experiment-mode dispatch
# ---------------------------------------------------------------------------
# Controls which fatal conditions are active and how the report is structured.
# All experiments not listed here default to "standard".

EXPERIMENT_MODE: dict[str, str] = {
    "extrap":             "ood",
    "exp3":               "pysr",
    "exp3b":              "pysr",
    "instability":        "instability",
    "exp2":               "multi_method",
    "hybrid_all_domains": "multi_method",
    # exp1_ablation / exp2_feynman: paired pysr_only vs hypatia comparison using
    # extrap_r2_far. Uses dedicated helpers; standard method schema
    # (pure_llm/neural_network/hybrid) is absent — method-comparison sections suppressed.
    # Three-tier MW (all-N / excl-train-fail / success-subset), Fisher, Spearman,
    # complexity distributions, threshold sweep, and LOO all run under this mode.
    "exp1_ablation":          "ablation",
    "exp2_feynman":           "standard",
    # exp2_feynman_extrap: OOD extrap step — produces ablation_paired.json;
    # run_analysis.py reads it in ablation mode (extrap_r2_far present).
    "exp2_feynman_extrap":    "ablation",
}

# Canonical result_subdir for every CI-dispatched experiment.
# Single source of truth — mirrors ci_experiment.yml plan meta step
# and both mapping dicts in ci_analysis.yml "Resolve experiment metadata".
# exp1_ablation intentionally absent: no worker, no result_subdir in CI.
RESULT_SUBDIR: dict[str, str] = {
    "exp1":               "comparison_results/noise-noiseless/noiseless/defi",
    "exp1b":              "comparison_results/noise-noiseless/15",
    "exp2_feynman":           "comparison_results/feynman-tests/exp2",
    # exp2_feynman_extrap: NSHARDS=1, DIRECT mode. ablation_paired.json written here
    # after merge_extrap_into_benchmark.py. Mirrors ci_analysis.yml MAPPING.
    "exp2_feynman_extrap":    "comparison_results/feynman-tests/exp2_extrap",
    "exp2":               "comparison_results/feynman-tests/exp2_multi",
    "exp3":               "extrapolation",
    "exp3b":              "extrapolation/multi_seed",
    "suppA":              "hybrid_pysr/defi",
    "suppB":              "comparison_results/feynman-tests/noise-sweep",
    "suppB_sc":           "comparison_results/feynman-tests/sample-complexity",
    "hybrid_all_domains": "hybrid_llm_nn/all_domains",
    "instability":        "figures",
    "extrap":             "comparison_results/extrapolation",
    # exp1_ablation: manual-only; no CI worker. Subdir mirrors merge_shards.py EXP_CONFIG.
    # If promoted to CI, add entries in ci_experiment.yml and ci_analysis.yml too.
    "exp1_ablation":      "comparison_results/feynman-tests/exp1_ablation",
}


def _get_mode(experiment: str) -> str:
    return EXPERIMENT_MODE.get(experiment, "standard")


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
    """Returns (n_success, n_total, rate) using the explicit 'success' flag."""
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


def _mann_whitney_paired_ablation(pairs: list[tuple[float, float]]) -> dict:
    """
    One-sided Mann-Whitney U test for paired (pysr_only_far, hypatia_far) values.

    Filters to finite pairs only (None / inf already excluded by the caller).
    Tests alternative='greater' (hypatia > pysr_only) and also reports two-sided p.
    Returns both so the report can print both, matching exp1_rf01_mannwhitney.json.
    """
    if not _SCIPY_OK:
        return {"available": False, "reason": "scipy not installed"}
    if len(pairs) < 2:
        return {"available": False, "reason": "insufficient pairs after filtering"}
    p_vals = [p for p, _ in pairs]
    h_vals = [h for _, h in pairs]
    try:
        stat_os, p_one = mannwhitneyu(h_vals, p_vals, alternative="greater")
        stat_ts, p_two = mannwhitneyu(h_vals, p_vals, alternative="two-sided")
        return {
            "available":          True,
            "statistic":          round(float(stat_ts), 4),
            "p_value_one_sided":  round(float(p_one), 6),
            "p_value_two_sided":  round(float(p_two), 6),
            "significant_05_one": float(p_one) < 0.05,
            "significant_05_two": float(p_two) < 0.05,
            "n_pairs":            len(pairs),
            # non-significance is an honest scientific result, not a pipeline error.
            "interpretation":     (
                "statistically significant (p_one < 0.05)"
                if float(p_one) < 0.05
                else "not statistically significant at α=0.05 — directional result only"
            ),
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


def _ablation_instability_index(far_r2: float | None) -> float:
    """
    instability_index = 1 - extrap_r2_far.

    None → 0.0 substitution (for equations where hypatia returned no result).
    NOT clamped — negative far R² intentionally yields index > 1, quantifying
    the magnitude of extrapolation failure, not just its presence.
    """
    if far_r2 is None:
        return 0.0
    return 1.0 - far_r2


def _rank_biserial(a: list[float], b: list[float]) -> float | None:
    """
    Rank-biserial correlation r = 1 - 2U / (n_a * n_b).
    Ranges [-1, 1]; positive means a tends to exceed b.
    Computed from the Mann-Whitney U statistic for group a vs b.
    """
    if not _SCIPY_OK or len(a) < 1 or len(b) < 1:
        return None
    try:
        from scipy.stats import mannwhitneyu as _mwu
        u_stat, _ = _mwu(a, b, alternative="two-sided")
        return round(1.0 - 2.0 * float(u_stat) / (len(a) * len(b)), 4)
    except Exception:
        return None


def _fisher_exact_2x2(
    n_fail_target: int, n_total_target: int,
    n_fail_other: int,  n_total_other: int,
) -> dict:
    """
    Fisher's exact test on a 2×2 table:
        [[n_fail_target, n_pass_target],
         [n_fail_other,  n_pass_other ]]
    Returns p-value (two-sided) and odds-ratio.
    """
    if not _SCIPY_OK:
        return {"available": False, "reason": "scipy not installed"}
    try:
        from scipy.stats import fisher_exact
        n_pass_target = n_total_target - n_fail_target
        n_pass_other  = n_total_other  - n_fail_other
        table = [[n_fail_target, n_pass_target],
                 [n_fail_other,  n_pass_other]]
        odds, p = fisher_exact(table, alternative="two-sided")
        return {
            "available":    True,
            "table":        table,
            "odds_ratio":   round(float(odds), 4) if math.isfinite(float(odds)) else None,
            "p_value":      round(float(p), 6),
            "significant_05": float(p) < 0.05,
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


def _spearman(x: list[float], y: list[float]) -> dict:
    """Spearman rank correlation between two equal-length lists of finite floats."""
    if not _SCIPY_OK or len(x) < 3:
        return {"available": False, "reason": "insufficient data or scipy missing"}
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(x, y)
        return {
            "available":      True,
            "rho":            round(float(r), 4),
            "p_value":        round(float(p), 6),
            "significant_05": float(p) < 0.05,
            "n":              len(x),
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


def _complexity_distribution(vals: list[int | float]) -> dict:
    """Summary stats for a list of complexity scores."""
    finite = [v for v in vals if v is not None and math.isfinite(float(v))]
    if not finite:
        return {"n": 0}
    arr = np.array(finite, dtype=float)
    return {
        "n":      len(finite),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
        "mean":   round(float(arr.mean()), 2),
        "median": float(np.median(arr)),
        "p25":    float(np.percentile(arr, 25)),
        "p75":    float(np.percentile(arr, 75)),
    }


def _threshold_sweep(
    records: list[dict],
    thresholds: list[float] | None = None,
) -> list[dict]:
    """
    Sweep the train-R² inclusion threshold from lo to hi.
    At each threshold t, include only equations where hypatia.train_r2 >= t,
    run Mann-Whitney (one-sided, hypatia far-R² > pysr_only far-R²),
    and record n_included, U, p_one, p_two.

    Default thresholds: -0.5, -0.25, 0.0, 0.1, 0.25, 0.5
    """
    if thresholds is None:
        thresholds = [-0.5, -0.25, 0.0, 0.1, 0.25, 0.5]

    rows = []
    for t in thresholds:
        pairs = []
        for r in records:
            hyp  = r.get("hypatia",   {}) or {}
            pysr = r.get("pysr_only", {}) or {}
            h_train = _safe_float(hyp.get("train_r2"))
            if not math.isfinite(h_train) or h_train < t:
                continue
            h_far_raw = hyp.get("extrap_r2_far")
            p_far     = _safe_float(pysr.get("extrap_r2_far"))
            if _is_finite(h_far_raw) and math.isfinite(p_far):
                pairs.append((p_far, float(h_far_raw)))

        if len(pairs) < 2 or not _SCIPY_OK:
            rows.append({"threshold": t, "n_included": len(pairs),
                         "available": False, "reason": "insufficient pairs"})
            continue
        try:
            from scipy.stats import mannwhitneyu as _mwu
            h_vals = [h for _, h in pairs]
            p_vals = [p for p, _ in pairs]
            u_os, p_one = _mwu(h_vals, p_vals, alternative="greater")
            u_ts, p_two = _mwu(h_vals, p_vals, alternative="two-sided")
            rows.append({
                "threshold":        t,
                "n_included":       len(pairs),
                "available":        True,
                "U":                round(float(u_ts), 2),
                "p_one_sided":      round(float(p_one), 6),
                "p_two_sided":      round(float(p_two), 6),
                "significant_05":   float(p_one) < 0.05,
            })
        except Exception as e:
            rows.append({"threshold": t, "n_included": len(pairs),
                         "available": False, "reason": str(e)})
    return rows


def _leave_one_out_sensitivity(records: list[dict]) -> list[dict]:
    """
    Leave-one-out Mann-Whitney sensitivity.
    Iterates over the 7 failure equations (hypatia train_r2 < 0).
    For each, removes that equation from the full set, re-runs the all-N MW
    on the remaining finite far-R² pairs, and records the new p_one and n.
    Quantifies how much each failure masks the signal.
    """
    if not _SCIPY_OK:
        return []

    # Full set of finite pairs (same filter as the main MW)
    def _pairs_excluding(skip_name: str) -> list[tuple[float, float]]:
        out = []
        for r in records:
            if r.get("equation_name", r.get("equation_id", "")) == skip_name:
                continue
            hyp  = r.get("hypatia",   {}) or {}
            pysr = r.get("pysr_only", {}) or {}
            h_far_raw = hyp.get("extrap_r2_far")
            p_far     = _safe_float(pysr.get("extrap_r2_far"))
            if _is_finite(h_far_raw) and math.isfinite(p_far):
                out.append((p_far, float(h_far_raw)))
        return out

    failure_names = [
        r.get("equation_name", r.get("equation_id", "?"))
        for r in records
        if _safe_float((r.get("hypatia") or {}).get("train_r2")) < 0
    ]

    results = []
    for name in failure_names:
        pairs = _pairs_excluding(name)
        if len(pairs) < 2:
            results.append({"removed": name, "n_remaining": len(pairs),
                            "available": False})
            continue
        try:
            from scipy.stats import mannwhitneyu as _mwu
            h_vals = [h for _, h in pairs]
            p_vals = [p for p, _ in pairs]
            u_ts, p_two = _mwu(h_vals, p_vals, alternative="two-sided")
            u_os, p_one = _mwu(h_vals, p_vals, alternative="greater")
            results.append({
                "removed":        name,
                "n_remaining":    len(pairs),
                "available":      True,
                "U":              round(float(u_ts), 2),
                "p_one_sided":    round(float(p_one), 6),
                "p_two_sided":    round(float(p_two), 6),
                "significant_05": float(p_one) < 0.05,
            })
        except Exception as e:
            results.append({"removed": name, "n_remaining": len(pairs),
                            "available": False, "reason": str(e)})
    return results


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
            "available":      True,
            "statistic":      round(float(stat), 4),
            "p_value":        round(float(p), 6),
            "significant_05": float(p) < 0.05,
            "significant_01": float(p) < 0.01,
            "direction":      direction,
            "n_a":            len(a),
            "n_b":            len(b),
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


# ---------------------------------------------------------------------------
# Ablation early-return skeleton
# ---------------------------------------------------------------------------

def _ablation_empty_result(experiment: str, mode: str, n_total: int,
                           fatal: list[str]) -> dict:
    """
    Minimal but complete dict returned by analyse_ablation() on early exit.
    Includes every key that write_report_ablation() accesses, so the report
    writer never raises a KeyError regardless of which exit path was taken.
    """
    return {
        "experiment":               experiment,
        "experiment_mode":          mode,
        "n_total":                  n_total,
        # MW pair counts — always present
        "n_mw_pairs_all":           0,
        "n_mw_pairs_excl":          0,
        "n_mw_pairs_success":       0,
        "n_successes_extrap":       0,
        "extrap_success_threshold": EXTRAP_SUCCESS_THRESHOLD,
        "n_skipped_from_mw":        0,
        "skipped_equations":        [],
        # MW results
        "mann_whitney_all_n":            {"available": False, "reason": "early exit"},
        "mann_whitney_excl_fail":        {"available": False, "reason": "early exit"},
        "mann_whitney_success_subset":   {"available": False, "reason": "early exit"},
        # Win/loss
        "win_loss_all":     {"hypatia_wins": 0, "pysr_wins": 0, "tied": 0, "n_pairs": 0},
        "win_loss_excl":    {"hypatia_wins": 0, "pysr_wins": 0, "tied": 0, "n_pairs": 0},
        "win_loss_success": {"hypatia_wins": 0, "pysr_wins": 0, "tied": 0, "n_pairs": 0},
        # Failure analysis
        "failure_analysis":       [],
        "n_train_failures":       0,
        "domain_stratification":  {},
        "fisher_failure_cluster": {"available": False, "reason": "early exit"},
        # Scale sensitivity
        "spearman_scale_vs_train_r2": {"available": False, "reason": "early exit"},
        "spearman_scale_vs_far_r2":   {"available": False, "reason": "early exit"},
        "n_scale_log_available":      0,
        # Complexity
        "complexity_analysis": {
            "hypatia_success":    {"n": 0},
            "hypatia_failure":    {"n": 0},
            "hypatia_all":        {"n": 0},
            "pysr_all":           {"n": 0},
            "mw_success_vs_fail": {"available": False, "reason": "early exit"},
        },
        # Threshold sweep / LOO
        "threshold_sweep":  [],
        "loo_sensitivity":  [],
        # Instability / timing
        "instability_rows": [],
        "timing": {
            "hypatia":   {"mean_s": None, "median_s": None, "n": 0},
            "pysr_only": {"mean_s": None, "median_s": None, "n": 0},
        },
        "pysr_fit_params":  {},
        "fatal_conditions": fatal,
    }


def analyse_ablation(records: list[dict], experiment: str,
                     pysr_fit_params: dict | None = None) -> dict:
    """
    exp1_ablation / exp2_feynman_rf09 analysis.

    Core MW structure (from spec):
      Rule 1. None/non-finite hypatia.extrap_r2_far → skip from MW pairs.
      Rule 2. instability_index = 1 - extrap_r2_far (no clamp); None→0.0 for CSV.
      Rule 3. MW non-significance is NOT a fatal condition.

    Additional analyses (RF09):
      A. Three-tier MW framing (paper §10.7):
           Tier 1 — all-N (all finite pairs; expected non-significant).
           Tier 2 — excl-train-failures (hypatia train_r2 >= 0; interim filter).
           Tier 3 — success-subset (hypatia extrap_r2_far >= EXTRAP_SUCCESS_THRESHOLD,
                     default 0.99; primary paper result — "9/30 successes").
           All three reported. Tier 3 is the primary publishable claim.
      B. Domain stratification + Fisher's exact on failure cluster.
      C. Scale/magnitude sensitivity: Spearman(log|scale|, train_r2).
      D. Expression complexity distributions: success vs failure, hypatia vs pysr.
      E. Effect sizes: rank-biserial r for every MW result.
      F. Threshold sweep: MW p-value vs train-R² inclusion threshold.
      G. Leave-one-out sensitivity on the 7 train-failure equations.
    """
    mode = "ablation"
    fatal: list[str] = []

    if not records:
        # Hard fatal only for exp1b and exp3b; other experiments use WARN_ so
        # the workflow continues when no records are merged.
        if experiment in ("exp1b", "exp3b"):
            fatal.append("EMPTY_DATASET: _merged.json contains 0 records.")
        else:
            fatal.append(
                f"WARN_EMPTY_DATASET: _merged.json contains 0 records for experiment "
                f"'{experiment}'. This is non-fatal for this experiment type. "
                "Workflow continues."
            )
        return _ablation_empty_result(experiment, mode, 0, fatal)

    n_total = len(records)

    # -------------------------------------------------------------------------
    # 0. Schema guard — detect flat per-method records (Shape C from workers
    #    that did not run the extrapolation step) before spending time on the
    #    full analysis.  Shape C looks like:
    #      [{"test": ..., "domain": ..., "method": ..., "r2": ..., "success": ...}]
    #    Ablation analysis requires the paired schema:
    #      [{"equation_name": ..., "hypatia": {"extrap_r2_far": ...},
    #                               "pysr_only": {"extrap_r2_far": ...}}]
    #    Fail immediately with a clear fatal so the CI log names the real problem.
    # -------------------------------------------------------------------------
    sample = records[0]
    has_paired_schema = (
        ("hypatia" in sample or "pysr_only" in sample)
        or ("extrap_r2_far" in sample)
    )
    has_flat_method_schema = (
        "method" in sample and "r2" in sample
        and "hypatia" not in sample and "pysr_only" not in sample
    )
    if has_flat_method_schema and not has_paired_schema:
        # Workers ran the consistency/benchmark check but not extrapolation.
        # extrap_r2_far was never computed — ablation analysis cannot proceed.
        fatal.append(
            f"WRONG_SCHEMA_FOR_ABLATION: exp2_feynman requires paired extrapolation "
            f"records with hypatia.extrap_r2_far and pysr_only.extrap_r2_far, but "
            f"the committed results are flat per-method benchmark records "
            f"(keys: {sorted(sample.keys())}). "
            f"The workers must rerun with the extrapolation evaluation step enabled. "
            f"See run_analysis.py EXPERIMENT_MODE['ablation'] docstring for the "
            f"required record schema."
        )
        return _ablation_empty_result(experiment, mode, n_total, fatal)

    # -------------------------------------------------------------------------
    # 1. Build paired arrays (Rule 1 filtering) + instability rows (Rule 2)
    # -------------------------------------------------------------------------
    mw_pairs_all:     list[tuple[float, float]] = []  # Tier 1: all finite pairs (all-N)
    mw_pairs_excl:    list[tuple[float, float]] = []  # Tier 2: excl train_r2<0 failures
    mw_pairs_success: list[tuple[float, float]] = []  # Tier 3: extrap_r2_far >= EXTRAP_SUCCESS_THRESHOLD
    skipped:          list[dict]                = []
    all_rows:         list[dict]                = []

    for r in records:
        eq_name  = r.get("equation_name", r.get("equation_id", "?"))
        domain   = r.get("domain", "?")
        pysr     = r.get("pysr_only", {}) or {}
        hyp      = r.get("hypatia",   {}) or {}

        p_far     = _safe_float(pysr.get("extrap_r2_far"))
        h_near    = _safe_float(hyp.get("extrap_r2_near"))
        h_far_raw = hyp.get("extrap_r2_far")
        h_train   = _safe_float(hyp.get("train_r2"))

        h_far_finite = _is_finite(h_far_raw)
        h_far = float(h_far_raw) if h_far_finite else None

        # Rule 1 — all-N MW: skip if either side is non-finite
        if h_far_finite and math.isfinite(p_far):
            mw_pairs_all.append((p_far, h_far))
            # Tier 2 — excl train-failures: additionally require hypatia train_r2 >= 0
            if math.isfinite(h_train) and h_train >= 0:
                mw_pairs_excl.append((p_far, h_far))
            # Tier 3 — success-subset: hypatia extrap_r2_far >= EXTRAP_SUCCESS_THRESHOLD
            # This is the primary paper result (§10.7: "9/30 successes").
            if h_far is not None and h_far >= EXTRAP_SUCCESS_THRESHOLD:
                mw_pairs_success.append((p_far, h_far))
        else:
            reason = (
                "hypatia.extrap_r2_far is None"              if h_far_raw is None
                else f"hypatia.extrap_r2_far={h_far_raw!r} is non-finite"
                                                             if not h_far_finite
                else f"pysr_only.extrap_r2_far={p_far!r} is non-finite"
            )
            skipped.append({"equation": eq_name, "domain": domain, "reason": reason})

        # Rule 2 — instability_index (no clamp, None→0.0)
        instability = _ablation_instability_index(h_far)

        all_rows.append({
            "equation":          eq_name,
            "domain":            domain,
            "train_r2_hypatia":  _safe_float(hyp.get("train_r2")),
            "train_r2_pysr":     _safe_float(pysr.get("train_r2")),
            "extrap_r2_near":    0.0 if not math.isfinite(h_near) else round(h_near, 6),
            "extrap_r2_far":     0.0 if h_far is None else round(h_far, 6),
            "instability_index": round(instability, 6),
            "far_r2_skipped":    not h_far_finite,
        })

    # -------------------------------------------------------------------------
    # A. Three-tier MW framing (Rule 3: non-significance not fatal)
    #    Tier 1: all-N          — expected non-significant (21 failures add noise)
    #    Tier 2: excl-train-fail — train_r2>=0 filter (interim, n~23)
    #    Tier 3: success-subset  — extrap_r2_far>=0.99 (primary paper result, n~9)
    # -------------------------------------------------------------------------
    mw_all     = _mann_whitney_paired_ablation(mw_pairs_all)
    mw_excl    = _mann_whitney_paired_ablation(mw_pairs_excl)
    mw_success = _mann_whitney_paired_ablation(mw_pairs_success)

    # Effect sizes (rank-biserial r) for all three tiers
    rb_all  = _rank_biserial(
        [h for _, h in mw_pairs_all],
        [p for p, _ in mw_pairs_all],
    )
    rb_excl = _rank_biserial(
        [h for _, h in mw_pairs_excl],
        [p for p, _ in mw_pairs_excl],
    )
    rb_success = _rank_biserial(
        [h for _, h in mw_pairs_success],
        [p for p, _ in mw_pairs_success],
    )
    if mw_all.get("available"):
        mw_all["rank_biserial_r"]     = rb_all
    if mw_excl.get("available"):
        mw_excl["rank_biserial_r"]    = rb_excl
    if mw_success.get("available"):
        mw_success["rank_biserial_r"] = rb_success

    # Win/loss on valid pairs
    def _wl(pairs):
        hw = sum(1 for p, h in pairs if h > p + 1e-9)
        pw = sum(1 for p, h in pairs if p > h + 1e-9)
        return {"hypatia_wins": hw, "pysr_wins": pw,
                "tied": len(pairs) - hw - pw, "n_pairs": len(pairs)}

    # -------------------------------------------------------------------------
    # B. Failure analysis + domain stratification + Fisher's exact
    # -------------------------------------------------------------------------
    failures = []
    for r in records:
        hyp = r.get("hypatia", {}) or {}
        h_train = _safe_float(hyp.get("train_r2"))
        if math.isfinite(h_train) and h_train < 0:
            failures.append({
                "equation":       r.get("equation_name", r.get("equation_id", "?")),
                "domain":         r.get("domain", "?"),
                "train_r2":       round(h_train, 6),
                "best_expression": hyp.get("best_expression", "?"),
                "complexity":     hyp.get("complexity"),
            })

    # Physics-with-small-constants domains flagged by RF-06 / RF09
    PHYSICS_SMALL_CONST_DOMAINS = {
        "Quantum", "Atomic", "Electromagnetism",
        "quantum", "atomic", "electromagnetism",
    }
    n_physics   = sum(1 for r in records if r.get("domain", "") in PHYSICS_SMALL_CONST_DOMAINS)
    n_other     = n_total - n_physics
    n_fail_phys = sum(1 for f in failures if f["domain"] in PHYSICS_SMALL_CONST_DOMAINS)
    n_fail_oth  = len(failures) - n_fail_phys

    fisher = _fisher_exact_2x2(
        n_fail_phys, n_physics,
        n_fail_oth,  n_other,
    )

    # Domain-level win-rate table (on all-N finite pairs)
    domain_stats: dict[str, dict] = {}
    for r in records:
        dom  = r.get("domain", "unknown")
        hyp  = r.get("hypatia",   {}) or {}
        pysr = r.get("pysr_only", {}) or {}
        h_far_raw = hyp.get("extrap_r2_far")
        p_far = _safe_float(pysr.get("extrap_r2_far"))
        h_train = _safe_float(hyp.get("train_r2"))

        if dom not in domain_stats:
            domain_stats[dom] = {
                "n_total": 0, "n_hypatia_wins": 0,
                "n_failures": 0, "n_finite_pairs": 0,
            }
        domain_stats[dom]["n_total"] += 1
        if math.isfinite(h_train) and h_train < 0:
            domain_stats[dom]["n_failures"] += 1
        if _is_finite(h_far_raw) and math.isfinite(p_far):
            domain_stats[dom]["n_finite_pairs"] += 1
            if float(h_far_raw) > p_far + 1e-9:
                domain_stats[dom]["n_hypatia_wins"] += 1

    for dom, ds in domain_stats.items():
        fp = ds["n_finite_pairs"]
        ds["hypatia_win_rate"] = round(ds["n_hypatia_wins"] / fp, 4) if fp else None
        ds["failure_rate"] = round(
            ds["n_failures"] / ds["n_total"], 4
        ) if ds["n_total"] else None

    # -------------------------------------------------------------------------
    # C. Scale / magnitude sensitivity: Spearman(log10|scale_log|, train_r2)
    # scale_log field is log10 of the smallest constant magnitude in the equation.
    # -------------------------------------------------------------------------
    scale_train_pairs: list[tuple[float, float]] = []
    for r in records:
        hyp = r.get("hypatia", {}) or {}
        scale_log = hyp.get("scale_log")
        h_train   = _safe_float(hyp.get("train_r2"))
        if scale_log is not None and math.isfinite(float(scale_log)) \
                and math.isfinite(h_train):
            scale_train_pairs.append((float(scale_log), h_train))

    spearman_scale_train = _spearman(
        [s for s, _ in scale_train_pairs],
        [t for _, t in scale_train_pairs],
    )

    # Also correlate scale_log with hypatia far-R²
    scale_far_pairs: list[tuple[float, float]] = []
    for r in records:
        hyp  = r.get("hypatia",   {}) or {}
        pysr = r.get("pysr_only", {}) or {}
        scale_log = hyp.get("scale_log")
        h_far_raw = hyp.get("extrap_r2_far")
        if scale_log is not None and math.isfinite(float(scale_log)) \
                and _is_finite(h_far_raw):
            scale_far_pairs.append((float(scale_log), float(h_far_raw)))

    spearman_scale_far = _spearman(
        [s for s, _ in scale_far_pairs],
        [f for _, f in scale_far_pairs],
    )

    # -------------------------------------------------------------------------
    # D. Complexity distributions: success vs failure, hypatia vs pysr
    # -------------------------------------------------------------------------
    def _get_complexity(r: dict, method: str) -> int | None:
        return (r.get(method, {}) or {}).get("complexity")

    success_names = {
        r.get("equation_name", r.get("equation_id", "?"))
        for r in records
        if _safe_float((r.get("hypatia") or {}).get("train_r2")) >= 0
    }
    failure_names_set = {f["equation"] for f in failures}

    hyp_complex_success = [_get_complexity(r, "hypatia") for r in records
                           if r.get("equation_name", r.get("equation_id")) in success_names
                           and _get_complexity(r, "hypatia") is not None]
    hyp_complex_failure = [_get_complexity(r, "hypatia") for r in records
                           if r.get("equation_name", r.get("equation_id")) in failure_names_set
                           and _get_complexity(r, "hypatia") is not None]
    pysr_complex_all    = [_get_complexity(r, "pysr_only") for r in records
                           if _get_complexity(r, "pysr_only") is not None]
    hyp_complex_all     = [_get_complexity(r, "hypatia") for r in records
                           if _get_complexity(r, "hypatia") is not None]

    complexity_analysis = {
        "hypatia_success":     _complexity_distribution(hyp_complex_success),
        "hypatia_failure":     _complexity_distribution(hyp_complex_failure),
        "hypatia_all":         _complexity_distribution(hyp_complex_all),
        "pysr_all":            _complexity_distribution(pysr_complex_all),
        "mw_success_vs_fail":  _mann_whitney(
            [float(v) for v in hyp_complex_success],
            [float(v) for v in hyp_complex_failure],
        ) if hyp_complex_success and hyp_complex_failure else {"available": False,
                                                                "reason": "no data"},
    }

    # -------------------------------------------------------------------------
    # F. Threshold sweep (train-R² inclusion threshold)
    # -------------------------------------------------------------------------
    threshold_sweep = _threshold_sweep(records)

    # -------------------------------------------------------------------------
    # G. Leave-one-out sensitivity on failure equations
    # -------------------------------------------------------------------------
    loo_sensitivity = _leave_one_out_sensitivity(records)

    # -------------------------------------------------------------------------
    # Timing
    # -------------------------------------------------------------------------
    hyp_times  = [_safe_float((r.get("hypatia")   or {}).get("sr_time_s")) for r in records]
    pysr_times = [_safe_float((r.get("pysr_only") or {}).get("sr_time_s")) for r in records]
    hyp_times  = [t for t in hyp_times  if math.isfinite(t)]
    pysr_times = [t for t in pysr_times if math.isfinite(t)]
    timing = {
        "hypatia":   {"mean_s": _mean(hyp_times),  "median_s": _median(hyp_times),  "n": len(hyp_times)},
        "pysr_only": {"mean_s": _mean(pysr_times), "median_s": _median(pysr_times), "n": len(pysr_times)},
    }

    # -------------------------------------------------------------------------
    # Fatal conditions (Rule 3: MW non-significance is INFO_ only)
    # -------------------------------------------------------------------------
    if n_total == 0:
        if experiment in ("exp1b", "exp3b"):
            fatal.append("EMPTY_DATASET: _merged.json contains 0 records.")
        else:
            fatal.append(
                f"WARN_EMPTY_DATASET: _merged.json contains 0 records for experiment "
                f"'{experiment}'. This is non-fatal for this experiment type. "
                "Workflow continues."
            )

    if len(mw_pairs_all) < MIN_RECORDS_FOR_STATS:
        fatal.append(
            f"TOO_FEW_MW_PAIRS: only {len(mw_pairs_all)} finite paired far-R² values "
            f"(need ≥ {MIN_RECORDS_FOR_STATS}) for Mann-Whitney test."
        )

    if failures:
        fatal.append(
            f"INFO_HYPATIA_TRAIN_FAILURES: {len(failures)} equation(s) have hypatia "
            f"train_r2 < 0 (degenerate PySR output — discovery failures, not extrapolation). "
            f"Cluster: Quantum/Atomic/Electromagnetism. Report in dedicated failure table."
        )

    # Rule 3 — Tier-1 (all-N) non-significance is informational only; expected.
    if mw_all.get("available") and not mw_all.get("significant_05_one"):
        fatal.append(
            f"INFO_MW_ALL_NOT_SIGNIFICANT: Tier-1 (all-N) Mann-Whitney one-sided "
            f"p={mw_all.get('p_value_one_sided', float('nan')):.4f} "
            f"(two-sided p={mw_all.get('p_value_two_sided', float('nan')):.4f}, "
            f"r={rb_all}, n={mw_all.get('n_pairs', '?')}) — directional but not significant. "
            f"Expected: 21 discovery failures add noise. Report Tier-3 success-subset as primary claim. "
            f"Workflow continues."
        )

    # Tier-3 significance is the primary publishable result — flag prominently.
    if mw_success.get("available"):
        if mw_success.get("significant_05_one"):
            fatal.append(
                f"INFO_MW_SUCCESS_SIGNIFICANT: Tier-3 (success-subset) Mann-Whitney one-sided "
                f"p={mw_success.get('p_value_one_sided', float('nan')):.4f} "
                f"(two-sided p={mw_success.get('p_value_two_sided', float('nan')):.4f}, "
                f"r={rb_success}, n={mw_success.get('n_pairs', '?')} equations with extrap R²>="
                f"{EXTRAP_SUCCESS_THRESHOLD}) — SIGNIFICANT. Primary paper claim confirmed."
            )
        else:
            fatal.append(
                f"WARN_MW_SUCCESS_NOT_SIGNIFICANT: Tier-3 (success-subset) Mann-Whitney one-sided "
                f"p={mw_success.get('p_value_one_sided', float('nan')):.4f} "
                f"(n={mw_success.get('n_pairs', '?')}) — not significant at α=0.05. "
                f"Primary paper claim (§10.7) may be weaker than expected. Investigate."
            )

    if fisher.get("available") and fisher.get("significant_05"):
        fatal.append(
            f"INFO_FAILURE_CLUSTER_SIGNIFICANT: Fisher's exact p={fisher['p_value']:.4f} "
            f"confirms failure cluster in physics-with-small-constants domains is non-random."
        )

    return {
        "experiment":            experiment,
        "experiment_mode":       mode,
        "n_total":               n_total,
        # MW — three-tier framing (paper §10.7)
        "n_mw_pairs_all":        len(mw_pairs_all),
        "n_mw_pairs_excl":       len(mw_pairs_excl),
        "n_mw_pairs_success":    len(mw_pairs_success),
        "n_successes_extrap":    len(mw_pairs_success),   # "9/30" in the paper
        "extrap_success_threshold": EXTRAP_SUCCESS_THRESHOLD,
        "n_skipped_from_mw":     len(skipped),
        "skipped_equations":     skipped,
        "mann_whitney_all_n":         mw_all,
        "mann_whitney_excl_fail":     mw_excl,
        "mann_whitney_success_subset": mw_success,
        "win_loss_all":          _wl(mw_pairs_all),
        "win_loss_excl":         _wl(mw_pairs_excl),
        "win_loss_success":      _wl(mw_pairs_success),
        # Failure analysis + domain
        "failure_analysis":      failures,
        "n_train_failures":      len(failures),
        "domain_stratification": domain_stats,
        "fisher_failure_cluster": fisher,
        # Scale sensitivity
        "spearman_scale_vs_train_r2": spearman_scale_train,
        "spearman_scale_vs_far_r2":   spearman_scale_far,
        "n_scale_log_available":      len(scale_train_pairs),
        # Complexity
        "complexity_analysis":   complexity_analysis,
        # Threshold sweep + LOO
        "threshold_sweep":       threshold_sweep,
        "loo_sensitivity":       loo_sensitivity,
        # Instability rows + timing
        "instability_rows":      all_rows,
        "timing":                timing,
        # PySR fit parameters recorded for provenance (sourced from CI env vars).
        # None when run outside CI or when vars are not set.
        "pysr_fit_params":       pysr_fit_params or {},
        "fatal_conditions":      fatal,
    }


def write_report_ablation(analysis: dict, path: Path) -> None:
    """Human-readable Markdown report for exp1_ablation / exp2_feynman_rf09."""
    exp   = analysis["experiment"]
    lines: list[str] = []

    def h(level: int, text: str):
        lines.append(f"\n{'#' * level} {text}\n")

    def p(*args):
        lines.append(" ".join(str(a) for a in args))

    def mw_row(mw: dict, label: str) -> str:
        if not mw.get("available"):
            return f"  {label}: N/A ({mw.get('reason', '?')})"
        sig = "**" if mw.get("significant_05_one") else ""
        rb  = mw.get("rank_biserial_r")
        rb_str = f", r={rb}" if rb is not None else ""
        return (
            f"  {label}: U={mw['statistic']}, "
            f"p_one={mw['p_value_one_sided']:.4f}{sig}, "
            f"p_two={mw['p_value_two_sided']:.4f}, "
            f"n={mw['n_pairs']}{rb_str}"
        )

    thr = analysis.get("extrap_success_threshold", EXTRAP_SUCCESS_THRESHOLD)

    # Use .get() with safe defaults for every key — protects against early-exit
    # dicts that may be missing optional keys (e.g. wrong-schema or empty-dataset
    # returns from analyse_ablation).
    n_mw_all     = analysis.get("n_mw_pairs_all",     0)
    n_mw_excl    = analysis.get("n_mw_pairs_excl",    0)
    n_mw_success = analysis.get("n_mw_pairs_success", 0)
    n_skipped    = analysis.get("n_skipped_from_mw",  0)

    h(1, f"HypatiaX Analysis Report — `{exp}` (RF09 Feynman n=30)")
    p(f"Experiment mode: **ablation** | N equations: {analysis['n_total']}")
    p(f"Tier-1 (all-N) pairs: {n_mw_all} "
      f"| Tier-2 (excl-train-fail) pairs: {n_mw_excl} "
      f"| Tier-3 (extrap R²≥{thr}) pairs: {n_mw_success} "
      f"| Skipped: {n_skipped}")

    # Fatal / info conditions
    all_conds  = analysis.get("fatal_conditions", [])
    hard_fatal = [c for c in all_conds if not (c.startswith("INFO_") or c.startswith("WARN_"))]
    soft_conds = [c for c in all_conds if c.startswith("INFO_") or c.startswith("WARN_")]

    if hard_fatal:
        h(2, "⚠️ Fatal Conditions")
        for fc in hard_fatal:
            lines.append(f"- **{fc}**")
    else:
        h(2, "✅ No Fatal Conditions")
    if soft_conds:
        h(2, "ℹ️ Informational / Warnings")
        for sc in soft_conds:
            lines.append(f"- {sc}")

    # -------------------------------------------------------------------------
    # A. Three-tier MW framing (paper §10.7)
    # -------------------------------------------------------------------------
    h(2, "A. Primary Result — Three-Tier MW Framing (§10.7)")
    p(
        "**Tier 1 (all-N):** Expected non-significant — 21 discovery failures add variance. "
        "Report with explicit framing: 'not significant; expected given 21 failures.' "
        "\n\n"
        "**Tier 2 (excl-train-fail):** Excludes equations where HypatiaX train R²<0. "
        "Intermediate result; shows signal strengthens once degenerate outputs removed. "
        "\n\n"
        f"**Tier 3 (success-subset, R²≥{thr}):** The paper's primary claim (§10.7). "
        "Restricts to equations where HypatiaX achieved symbolic recovery. "
        "This is the publishable result — it answers whether symbolic recovery produces "
        "a qualitatively different extrapolation regime, not whether HypatiaX always wins."
    )
    p()
    p(mw_row(analysis.get("mann_whitney_all_n",          {}), "Tier 1 — All-N"))
    p(mw_row(analysis.get("mann_whitney_excl_fail",       {}), "Tier 2 — Excl-train-fail (train R²≥0)"))
    p(mw_row(analysis.get("mann_whitney_success_subset",  {}), f"Tier 3 — Success-subset (extrap R²≥{thr}) ★"))
    p("_** = p_one < 0.05  |  ★ = primary paper claim_")

    wla = analysis.get("win_loss_all",     {})
    wle = analysis.get("win_loss_excl",    {})
    wls = analysis.get("win_loss_success", {})
    h(3, "Win / Loss by Tier")
    lines.append("| Split | HypatiaX wins | PySR wins | Tied | N pairs |")
    lines.append("|-------|---------------|-----------|------|---------|")
    lines.append(f"| Tier 1 — All-N | {wla.get('hypatia_wins',0)} "
                 f"| {wla.get('pysr_wins',0)} "
                 f"| {wla.get('tied',0)} "
                 f"| {wla.get('n_pairs',0)} |")
    lines.append(f"| Tier 2 — Excl-train-fail | {wle.get('hypatia_wins',0)} "
                 f"| {wle.get('pysr_wins',0)} "
                 f"| {wle.get('tied',0)} "
                 f"| {wle.get('n_pairs',0)} |")
    lines.append(f"| Tier 3 — Success-subset ★ | {wls.get('hypatia_wins',0)} "
                 f"| {wls.get('pysr_wins',0)} "
                 f"| {wls.get('tied',0)} "
                 f"| {wls.get('n_pairs',0)} |")

    # -------------------------------------------------------------------------
    # B. Failure analysis + domain stratification + Fisher
    # -------------------------------------------------------------------------
    failures = analysis.get("failure_analysis", [])
    h(2, f"B. Failure Analysis ({len(failures)} equations — degenerate PySR, train R² < 0)")
    if failures:
        p("_Discovery failures, not extrapolation failures. "
          "All cluster in Quantum / Atomic / Electromagnetism. Do not drop silently._")
        lines.append("\n| Equation | Domain | Train R² | Best Expression | Complexity |")
        lines.append(  "|----------|--------|----------|-----------------|------------|")
        for f in failures:
            lines.append(
                f"| {f['equation']} | {f['domain']} "
                f"| {f['train_r2']:.4f} | `{f['best_expression']}` "
                f"| {f.get('complexity', 'N/A')} |"
            )
    else:
        p("_None — all equations have hypatia train R² ≥ 0._")

    h(3, "Domain Stratification")
    ds = analysis.get("domain_stratification", {})
    if ds:
        lines.append("| Domain | N | Hypatia Wins | Win Rate | Failures | Fail Rate |")
        lines.append("|--------|---|-------------|----------|----------|-----------|")
        for dom, d in sorted(ds.items()):
            lines.append(
                f"| {dom} | {d['n_total']} "
                f"| {d['n_hypatia_wins']} "
                f"| {d['hypatia_win_rate'] if d['hypatia_win_rate'] is not None else 'N/A'} "
                f"| {d['n_failures']} "
                f"| {d['failure_rate'] if d['failure_rate'] is not None else 'N/A'} |"
            )

    fisher = analysis.get("fisher_failure_cluster", {})
    h(3, "Fisher's Exact Test — Failure Cluster Non-Randomness")
    if fisher.get("available"):
        sig = "✅ Significant" if fisher["significant_05"] else "Not significant"
        p(f"p={fisher['p_value']:.4f}, OR={fisher.get('odds_ratio', 'N/A')}, {sig}")
        p("Tests whether the failure cluster in physics-with-small-constants domains "
          "is larger than expected by chance.")
    else:
        p(f"N/A ({fisher.get('reason', '?')})")

    # -------------------------------------------------------------------------
    # C. Scale sensitivity
    # -------------------------------------------------------------------------
    h(2, "C. Scale / Magnitude Sensitivity")
    p("Spearman correlation between `scale_log` (log₁₀ of smallest constant magnitude) "
      "and HypatiaX performance. Positive ρ means larger-scale constants → better results.")

    def spear_row(s: dict, label: str) -> str:
        if not s.get("available"):
            return f"  {label}: N/A ({s.get('reason', '?')})"
        sig = "**" if s.get("significant_05") else ""
        return (
            f"  {label}: ρ={s['rho']}{sig}, p={s['p_value']:.4f}, n={s['n']}"
        )

    p(spear_row(analysis.get("spearman_scale_vs_train_r2", {}), "scale_log vs train R²"))
    p(spear_row(analysis.get("spearman_scale_vs_far_r2",   {}), "scale_log vs far R²"))
    p(f"scale_log available for {analysis.get('n_scale_log_available', 0)} equations.")
    p("_** = p < 0.05. N/A if scale_log field absent from records._")

    # -------------------------------------------------------------------------
    # D. Complexity distributions
    # -------------------------------------------------------------------------
    h(2, "D. Expression Complexity — Success vs Failure")
    cx = analysis.get("complexity_analysis", {})

    def cx_row(d: dict, label: str) -> str:
        if not d or d.get("n", 0) == 0:
            return f"| {label} | 0 | N/A | N/A | N/A | N/A | N/A |"
        return (
            f"| {label} | {d['n']} "
            f"| {d['min']:.0f} | {d['max']:.0f} "
            f"| {d['mean']:.1f} | {d['median']:.0f} "
            f"| {d['p25']:.0f}–{d['p75']:.0f} |"
        )

    lines.append("| Group | N | Min | Max | Mean | Median | IQR |")
    lines.append("|-------|---|-----|-----|------|--------|-----|")
    lines.append(cx_row(cx.get("hypatia_success", {}), "HypatiaX successes"))
    lines.append(cx_row(cx.get("hypatia_failure", {}), "HypatiaX failures"))
    lines.append(cx_row(cx.get("hypatia_all",     {}), "HypatiaX all"))
    lines.append(cx_row(cx.get("pysr_all",        {}), "PySR-only all"))

    mw_cx = cx.get("mw_success_vs_fail", {})
    if mw_cx.get("available"):
        sig = "**" if mw_cx.get("significant_05") else ""
        p(f"\nMW complexity (success vs failure): "
          f"U={mw_cx['statistic']}, p={mw_cx['p_value']:.4f}{sig}")
        p("_Low complexity (≤2) is a degenerate-output signal — "
          "consider flagging as failure before evaluation._")
    p("_** = p < 0.05_")

    # -------------------------------------------------------------------------
    # F. Threshold sweep
    # -------------------------------------------------------------------------
    h(2, "F. Train-R² Threshold Sweep — Robustness of Inclusion Cutoff")
    p("MW p_one at each train-R² inclusion threshold. "
      "A robust result stays significant across a range near 0.")
    sweep = analysis.get("threshold_sweep", [])
    if sweep:
        lines.append("| Threshold | N included | U | p_one | p_two | Significant? |")
        lines.append("|-----------|------------|---|-------|-------|--------------|")
        for row in sweep:
            if row.get("available"):
                sig = "✅" if row["significant_05"] else "—"
                lines.append(
                    f"| {row['threshold']:+.2f} | {row['n_included']} "
                    f"| {row['U']} "
                    f"| {row['p_one_sided']:.4f} | {row['p_two_sided']:.4f} "
                    f"| {sig} |"
                )
            else:
                lines.append(
                    f"| {row['threshold']:+.2f} | {row['n_included']} "
                    f"| N/A | N/A | N/A | — |"
                )
    else:
        p("_No sweep data._")

    # -------------------------------------------------------------------------
    # G. Leave-one-out sensitivity
    # -------------------------------------------------------------------------
    h(2, "G. Leave-One-Out Sensitivity — Failure Equations")
    p("All-N MW re-run with each failure equation removed. "
      "Shows how much each discovery failure masks the signal.")
    loo = analysis.get("loo_sensitivity", [])
    if loo:
        lines.append("| Removed equation | N remaining | U | p_one | p_two | Sig? |")
        lines.append("|-----------------|-------------|---|-------|-------|------|")
        for row in loo:
            if row.get("available"):
                sig = "✅" if row["significant_05"] else "—"
                lines.append(
                    f"| {row['removed']} | {row['n_remaining']} "
                    f"| {row['U']} "
                    f"| {row['p_one_sided']:.4f} | {row['p_two_sided']:.4f} "
                    f"| {sig} |"
                )
            else:
                lines.append(
                    f"| {row.get('removed','?')} | {row.get('n_remaining',0)} "
                    f"| N/A | N/A | N/A | — |"
                )
    else:
        p("_No LOO data (no failure equations or scipy unavailable)._")

    # -------------------------------------------------------------------------
    # Skipped equations + instability + timing
    # -------------------------------------------------------------------------
    skipped = analysis.get("skipped_equations", [])
    h(2, f"Skipped from MW ({len(skipped)} equations)")
    if skipped:
        lines.append("| Equation | Domain | Reason |")
        lines.append("|----------|--------|--------|")
        for s in skipped:
            lines.append(f"| {s['equation']} | {s['domain']} | {s['reason']} |")
    else:
        p("_None._")

    rows = analysis.get("instability_rows", [])
    h(2, "Instability Index (1 − extrap_r2_far; None→0.0; unclamped)")
    if rows:
        lines.append("| Equation | Domain | Near R² | Far R² | Instability | Skipped? |")
        lines.append("|----------|--------|---------|--------|-------------|----------|")
        for row in rows:
            lines.append(
                f"| {row['equation']} | {row['domain']} "
                f"| {row['extrap_r2_near']:.4f} | {row['extrap_r2_far']:.4f} "
                f"| {row['instability_index']:.4f} | {'yes' if row['far_r2_skipped'] else 'no'} |"
            )

    h(2, "Wall-clock Timing")
    timing = analysis.get("timing", {})
    lines.append("| Method | Mean (s) | Median (s) | N |")
    lines.append("|--------|----------|------------|---|")
    for key, label in [("hypatia", "HypatiaX"), ("pysr_only", "PySR-only")]:
        t = timing.get(key, {})
        lines.append(
            f"| {label} | {_r2f(t.get('mean_s'))} "
            f"| {_r2f(t.get('median_s'))} | {t.get('n', 0)} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-method summary helper (shared between standard and non-standard modes)
# ---------------------------------------------------------------------------

def _method_summary(standard: list[dict]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for m in METHODS:
        r2_vals              = _r2_values(standard, m)
        suc_n, suc_d, suc_r = _success_rate(standard, m)
        r2s_n, r2s_d, r2s_r = _r2_success_rate(standard, m)
        summary[m] = {
            "n_records":         suc_d,
            "n_success_flag":    suc_n,
            "success_rate_flag": round(suc_r, 4),
            "n_r2_above_80":     r2s_n,
            "r2_above_80_rate":  round(r2s_r, 4),
            "median_test_r2":    _median(r2_vals),
            "mean_test_r2":      _mean(r2_vals),
            "n_finite_r2":       len(r2_vals),
        }
    return summary


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyse(records: list[dict], experiment: str,
            pysr_fit_params: dict | None = None) -> dict:
    """
    Run full statistical analysis on a list of merged records.
    Returns a dict written verbatim to _analysis.json.

    Behaviour is gated by experiment mode (see EXPERIMENT_MODE).
    """
    mode = _get_mode(experiment)

    # Guard: if this experiment should run ablation analysis, refuse here.
    # Ablation schema uses hypatia/pysr_only keys (no pure_llm/neural_network/hybrid).
    # If it reaches analyse() the 0%-success reads are spurious and fire TOTAL_FAILURE.
    if mode == "ablation":
        raise RuntimeError(
            f"analyse() called for experiment {experiment!r} which maps to mode "
            f"'ablation'. Route to analyse_ablation() instead. "
            f"Check main() dispatch logic."
        )

    # -- Partition: standard vs intractable ------------------------------------
    standard    = [r for r in records if not r.get("extrapolation_intractable", False)]
    intractable = [r for r in records if r.get("extrapolation_intractable", False)]

    n_total       = len(records)
    n_standard    = len(standard)
    n_intractable = len(intractable)

    # -- Per-method summary ----------------------------------------------------
    method_summary = _method_summary(standard)

    # -- Coverage gaps ---------------------------------------------------------
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
                "equation_id":  eq_id,
                "difficulty":   r.get("difficulty"),
                "formula_type": r.get("formula_type"),
                "best_test_r2": None if not math.isfinite(best) else round(best, 4),
                "per_method": {
                    m: (round(_safe_float(r.get("results", {}).get(m, {}).get("test_r2")), 4)
                        if math.isfinite(_safe_float(r.get("results", {}).get(m, {}).get("test_r2")))
                        else None)
                    for m in METHODS
                },
            })

    # -- Mann-Whitney pairwise comparisons (standard records only) -------------
    # Skipped for pysr/instability modes — no hybrid/NN/LLM schema present.
    if mode in ("pysr", "instability"):
        mann_whitney = {
            "hybrid_vs_llm": {"available": False, "reason": f"not applicable for {mode} experiment"},
            "hybrid_vs_nn":  {"available": False, "reason": f"not applicable for {mode} experiment"},
            "nn_vs_llm":     {"available": False, "reason": f"not applicable for {mode} experiment"},
        }
    else:
        r2_llm = _r2_values(standard, "pure_llm")
        r2_nn  = _r2_values(standard, "neural_network")
        r2_hyb = _r2_values(standard, "hybrid")
        mann_whitney = {
            "hybrid_vs_llm": _mann_whitney(r2_hyb, r2_llm),
            "hybrid_vs_nn":  _mann_whitney(r2_hyb, r2_nn),
            "nn_vs_llm":     _mann_whitney(r2_nn,  r2_llm),
        }

    # -- Per-difficulty breakdown -----------------------------------------------
    difficulties = sorted({r.get("difficulty") or "unknown" for r in standard})
    by_difficulty: dict[str, dict] = {}
    for diff in difficulties:
        sub = [r for r in standard if (r.get("difficulty") or "unknown") == diff]
        by_difficulty[diff] = {
            m: {
                "n":               len([r for r in sub if r.get("results", {}).get(m) is not None]),
                "median_test_r2":  _median(_r2_values(sub, m)),
                "r2_above_80_rate": round(_r2_success_rate(sub, m)[2], 4),
            }
            for m in METHODS
        }

    # -- Per-formula-type breakdown ---------------------------------------------
    ftypes = sorted({r.get("formula_type") or "unknown" for r in standard})
    by_formula_type: dict[str, dict] = {}
    for ft in ftypes:
        sub = [r for r in standard if (r.get("formula_type") or "unknown") == ft]
        by_formula_type[ft] = {
            m: {
                "n":               len([r for r in sub if r.get("results", {}).get(m) is not None]),
                "median_test_r2":  _median(_r2_values(sub, m)),
                "r2_above_80_rate": round(_r2_success_rate(sub, m)[2], 4),
            }
            for m in METHODS
        }

    # -- Extrapolation gap analysis --------------------------------------------
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
        # Count timed_out flags — present on neural_network records; absent (None)
        # means "not applicable" for methods that don't use a wall-clock timeout.
        n_timed_out = sum(
            1 for r in standard
            if r.get("results", {}).get(m, {}).get("timed_out") is True
        )
        timing[m] = {
            "mean_s":      _mean(times),
            "median_s":    _median(times),
            "total_s":     round(sum(times), 2) if times else None,
            "n":           len(times),
            "n_timed_out": n_timed_out,
        }

    # -- Hybrid decision breakdown ---------------------------------------------
    decisions: dict[str, int] = {}
    for r in standard:
        dec = r.get("results", {}).get("hybrid", {}).get("decision")
        if dec:
            decisions[dec] = decisions.get(dec, 0) + 1

    # -- Hybrid vs NN head-to-head (equation level) ----------------------------
    hyb_beats_nn  = 0
    nn_beats_hyb  = 0
    tied          = 0
    n_both_finite = 0
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
        "hybrid_wins":    hyb_beats_nn,
        "nn_wins":        nn_beats_hyb,
        "tied":           tied,
        "hybrid_win_rate": round(hyb_beats_nn / n_both_finite, 4) if n_both_finite else None,
    }

    # -- Fatal conditions (mode-aware) -----------------------------------------
    # Prefix conventions:
    #   (none)  → hard fatal; ci_analysis.yml aborts after commit.
    #   INFO_   → informational; logged, workflow continues.
    #   WARN_   → warning; logged, workflow continues.
    fatal: list[str] = []

    # EMPTY_DATASET: hard fatal only for exp1b and exp3b (those experiments must
    # produce records; an empty merge indicates a genuine pipeline failure).
    # All other experiments (e.g. exp3) may legitimately produce no merged records
    # and receive a WARN_ instead so the workflow continues.
    if n_total == 0:
        if experiment in ("exp1b", "exp3b"):
            fatal.append("EMPTY_DATASET: _merged.json contains 0 records.")
        else:
            fatal.append(
                f"WARN_EMPTY_DATASET: _merged.json contains 0 records for experiment "
                f"'{experiment}'. This is non-fatal for this experiment type. "
                "Workflow continues."
            )

    if n_standard == 0 and n_total > 0:
        fatal.append(
            f"ALL_INTRACTABLE: all {n_total} records are marked extrapolation_intractable; "
            "no standard equations to analyse."
        )

    if 0 < n_standard < MIN_RECORDS_FOR_STATS:
        fatal.append(
            f"TOO_FEW_RECORDS: only {n_standard} standard records "
            f"(need ≥ {MIN_RECORDS_FOR_STATS}) for meaningful statistics."
        )

    # TOTAL_FAILURE — suppressed for pysr/instability/multi_method.
    # Those experiments either have no 3-method schema (pysr, instability) or
    # a partially-mapped 4-method schema (multi_method) where 0% on canonical
    # keys is expected rather than indicative of a bug.
    if mode == "standard" or mode == "ood":
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

    # HYBRID_NEVER_BEATS_NN — mode-dependent.
    if mode == "ood":
        # OOD: hybrid losing NN is the expected scientific result; demote to INFO.
        if n_both_finite >= MIN_RECORDS_FOR_STATS and hyb_beats_nn == 0 and nn_beats_hyb > 0:
            fatal.append(
                f"INFO_OOD_HYBRID_LOSES_NN: hybrid ≤ neural_network on all "
                f"{n_both_finite} OOD equations — expected for extrap experiment. "
                "Not a routing regression. Workflow continues."
            )
    elif mode in ("standard", "multi_method"):
        # Active for standard and multi_method: failure here is a genuine regression.
        if n_both_finite >= MIN_RECORDS_FOR_STATS and hyb_beats_nn == 0 and nn_beats_hyb > 0:
            fatal.append(
                f"HYBRID_NEVER_BEATS_NN: hybrid ≤ neural_network on all "
                f"{n_both_finite} equations where both produced finite R². "
                "Possible routing or fix regression."
            )
    # pysr and instability: no hybrid key at all — skip entirely.

    # WARN_MULTI_METHOD — 4th method (HybridSystemLLMNN all-domains) is present
    # in the raw experiment output but has no canonical key in METHODS.  It is
    # excluded from all method-comparison statistics.  Verify that merge_shards.py
    # translates method names before this analysis runs.
    if mode == "multi_method":
        fatal.append(
            "WARN_MULTI_METHOD: this experiment produces a 4th method key "
            "(HybridSystemLLMNN all-domains) not in METHODS. "
            "It is excluded from all method-comparison statistics. "
            "Confirm merge_shards.py translates method names before analysis."
        )

    # -- Assemble output -------------------------------------------------------
    result = {
        "experiment":          experiment,
        "experiment_mode":     mode,
        "n_total":             n_total,
        "n_standard":          n_standard,
        "n_intractable":       n_intractable,
        "r2_success_threshold": R2_SUCCESS_THRESHOLD,
        "method_summary":      method_summary,
        "mann_whitney":        mann_whitney,
        "coverage_gaps":       coverage_gaps,
        "n_coverage_gaps":     len(coverage_gaps),
        "by_difficulty":       by_difficulty,
        "by_formula_type":     by_formula_type,
        "extrapolation_gap_summary": gap_summary,
        "timing":              timing,
        "hybrid_decisions":    decisions,
        "hybrid_vs_nn_headtohead": hybrid_vs_nn_headtohead,
        # PySR fit parameters recorded for provenance (sourced from CI env vars).
        # None when run outside CI or when vars are not set.
        "pysr_fit_params":     pysr_fit_params or {},
        "fatal_conditions":    fatal,
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
    exp  = analysis["experiment"]
    mode = analysis.get("experiment_mode", "standard")
    lines: list[str] = []

    def h(level: int, text: str):
        lines.append(f"\n{'#' * level} {text}\n")

    def p(*args):
        lines.append(" ".join(str(a) for a in args))

    h(1, f"HypatiaX Analysis Report — `{exp}`")
    p(f"Experiment mode: **{mode}**")
    p(f"N total: {analysis['n_total']} "
      f"| N standard: {analysis['n_standard']} "
      f"| N intractable: {analysis['n_intractable']}")
    p(f"R² success threshold: {analysis['r2_success_threshold']}")

    # -- Mode-specific header note -----------------------------------------------
    if mode == "ood":
        p(
            "\n> **OOD experiment**: hybrid losing to neural_network is the "
            "expected scientific result; `HYBRID_NEVER_BEATS_NN` is demoted "
            "to informational and does not block the workflow."
        )
    elif mode == "pysr":
        p(
            "\n> **PySR/Nguyen experiment**: no `hybrid` / `neural_network` / "
            "`pure_llm` method keys are expected in `_merged.json`. "
            "Method-comparison sections are skipped."
        )
    elif mode == "multi_method":
        p(
            "\n> **Multi-method experiment**: a 4th method key "
            "(`HybridSystemLLMNN all-domains`) is present in the raw output "
            "but is not in `METHODS` and is excluded from comparisons. "
            "Verify `merge_shards.py` translates method names correctly."
        )

    # -- Fatal conditions --------------------------------------------------------
    all_conds = analysis.get("fatal_conditions", [])
    hard_fatal = [c for c in all_conds if not (c.startswith("INFO_") or c.startswith("WARN_"))]
    soft_conds = [c for c in all_conds if c.startswith("INFO_") or c.startswith("WARN_")]

    if hard_fatal:
        h(2, "⚠️ Fatal Conditions")
        for fc in hard_fatal:
            lines.append(f"- **{fc}**")
    else:
        h(2, "✅ No Fatal Conditions")

    if soft_conds:
        h(2, "ℹ️ Informational / Warnings")
        for sc in soft_conds:
            lines.append(f"- {sc}")

    # -- Method summary table (skip for pysr / instability) ----------------------
    if mode not in ("pysr", "instability"):
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
    else:
        h(2, "Method Summary")
        p(f"_Skipped — not applicable for `{mode}` experiment._")

    # -- Mann-Whitney (skip for pysr / instability) ------------------------------
    if mode not in ("pysr", "instability"):
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
    else:
        h(2, "Mann-Whitney U Tests")
        p(f"_Skipped — not applicable for `{mode}` experiment._")

    # -- Hybrid vs NN head-to-head (skip for pysr / instability) ----------------
    if mode not in ("pysr", "instability"):
        h(2, "Hybrid vs Neural Net (head-to-head, equation level)")
        hh = analysis.get("hybrid_vs_nn_headtohead", {})
        p(f"Equations with both finite R²: {hh.get('n_equations_both_finite', 0)}")
        p(f"Hybrid wins:  {hh.get('hybrid_wins', 0)}  ({_pct(hh.get('hybrid_win_rate'))})")
        p(f"NN wins:      {hh.get('nn_wins', 0)}")
        p(f"Tied:         {hh.get('tied', 0)}")
        if mode == "ood":
            p("_Note: hybrid losing NN is expected in OOD extrapolation._")
    else:
        h(2, "Hybrid vs Neural Net")
        p(f"_Skipped — not applicable for `{mode}` experiment._")

    # -- Coverage gaps -----------------------------------------------------------
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

    # -- By difficulty -----------------------------------------------------------
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
    else:
        p("_No difficulty breakdown available._")

    # -- By formula type ---------------------------------------------------------
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
    else:
        p("_No formula-type breakdown available._")

    # -- Extrapolation gap -------------------------------------------------------
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

    # -- Timing ------------------------------------------------------------------
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

    # -- Hybrid decisions --------------------------------------------------------
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
    ap.add_argument("--experiment",  required=True,
                    help="Experiment ID (e.g. exp1, exp2_feynman, extrap)")

    # ---------------------------------------------------------------------------
    # Input source — two modes, both supported:
    #
    #   LEGACY (merge_shards.py output):
    #     --merged-json <path>   Path to _merged.json from merge_shards.py.
    #     --output-dir  <dir>    Where to write _analysis.json / _report.md.
    #                            Defaults to the directory containing --merged-json.
    #
    #   CI / NSHARDS=1 (direct repo JSON):
    #     --input-json  <path>   Path to any readable input JSON (the single
    #                            result file committed directly by the worker,
    #                            or _merged_benchmark.json assembled in-memory
    #                            by the "Locate analysis input" step in
    #                            ci_analysis.yml).  Used for exp1 (NSHARDS=1)
    #                            and any other single-shard experiment where
    #                            merge_shards.py is not called.
    #     --shard-manifest <f>  Newline-delimited list of shard JSON paths.
    #                            Each file is loaded and records concatenated
    #                            before analysis (shard-direct mode from CI).
    #     --result-dir  <dir>    Canonical result directory (RESULT_DIR from CI
    #                            env).  Outputs are written here as
    #                            _analysis.json and _report.md.
    #
    # --input-json and --shard-manifest are mutually exclusive.
    # --merged-json is kept for backward compatibility with manual invocations.
    # ---------------------------------------------------------------------------
    input_group = ap.add_mutually_exclusive_group()
    input_group.add_argument("--merged-json",
                    help="Path to _merged.json produced by merge_shards.py "
                         "(legacy; use --input-json for NSHARDS=1 direct mode).")
    input_group.add_argument("--input-json",
                    help="Path to any readable result JSON — the single file "
                         "committed directly by a NSHARDS=1 worker, or an "
                         "in-memory assembled _merged_benchmark.json.  "
                         "Takes the same load/normalise path as --merged-json.")
    input_group.add_argument("--shard-manifest",
                    help="Newline-delimited file listing shard JSON paths "
                         "(shard-direct mode; each file is loaded and records "
                         "concatenated).")

    ap.add_argument("--output-dir",  required=False, default=None,
                    help="Directory for outputs (legacy; use --result-dir).")
    ap.add_argument("--result-dir",  required=False, default=None,
                    help="Canonical RESULT_DIR from CI env; outputs written here.")
    ap.add_argument("--output-stem", required=False, default="_analysis",
                    help="Stem for output files (default: _analysis).  "
                         "Pass e.g. _analysis_pca_4060 for PCA corrected runs "
                         "so the output never collides with the legacy _analysis.json.")
    return ap.parse_args()


def _load_records_from_json(json_path: Path, experiment: str) -> list[dict]:
    """
    Load records from a single JSON file using the same shape-detection logic
    as the main() legacy path.  Shared between --merged-json, --input-json,
    and individual shard files in --shard-manifest mode.

    Handles:
      Shape A  {"_meta":{}, "stats":{}, "results":{task_id: record}}
      Shape B  {task_id: record, ...}  flat dict
      Shape C  [record, ...]           top-level list
      Shape P  {"tests": [{description, domain, results:{RawMethod:{r2,...}}}]}
               — protocol_core_*.json / _merged_benchmark.json from the
                 "Locate analysis input" step.  Normalised via
                 _normalise_protocol_record() unless ablation.
    """
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Shape P — benchmark-format (protocol_core_*.json or _merged_benchmark.json)
    # Detected by the presence of a non-empty "tests" list whose entries have
    # a "results" dict with method sub-dicts (r2 / success).
    # This is the shape produced by the single NSHARDS=1 worker for exp1 and
    # assembled in-memory by ci_analysis.yml for other single-shard experiments.
    from merge_shards import _is_protocol_file, _normalise_protocol_record  # type: ignore
    _ABLATION_EXPERIMENTS = {"exp1_ablation"}
    is_ablation = experiment in _ABLATION_EXPERIMENTS

    if isinstance(raw, dict) and _is_protocol_file(raw):
        records = []
        for test in raw.get("tests", []):
            if not isinstance(test, dict):
                continue
            if is_ablation:
                records.append(test)
            else:
                records.append(_normalise_protocol_record(test))
        print(f"  Shape P (protocol wrapper / NSHARDS=1 direct): "
              f"{len(records)} records from 'tests' key.")
        return records

    if isinstance(raw, dict) and isinstance(raw.get("results"), dict):
        # Shape A: the "results" value is the task-keyed dict.
        records = [v for v in raw["results"].values() if isinstance(v, dict)]
        print(f"  Shape A (_merged.json from merge_shards.py): "
              f"{len(records)} records from 'results' key.")
        return records
    elif isinstance(raw, dict):
        # Shape B: flat dict — skip _meta / stats / _checkpoint sentinel keys.
        records = [v for k, v in raw.items()
                   if isinstance(v, dict)
                   and not k.startswith("_")
                   and k != "stats"]
        print(f"  Shape B (flat dict): {len(records)} records.")
        return records
    elif isinstance(raw, list):
        # Shape C: top-level list.
        records = [r for r in raw if isinstance(r, dict)]
        print(f"  Shape C (list): {len(records)} records.")
        return records
    else:
        print(f"::error::Unexpected JSON top-level type: {type(raw)}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_args()

    # ── Resolve output directory ──────────────────────────────────────────────
    # Priority: --result-dir > --output-dir > directory of input file.
    if args.result_dir:
        output_dir = Path(args.result_dir)
    elif args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.merged_json:
        output_dir = Path(args.merged_json).parent
    elif args.input_json:
        output_dir = Path(args.input_json).parent
    elif args.shard_manifest:
        output_dir = Path(args.shard_manifest).parent
    else:
        print("::error::No input source specified (--merged-json, --input-json, or "
              "--shard-manifest required).", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    _stem           = getattr(args, "output_stem", None) or "_analysis"
    _analysis_name  = f"{_stem}.json"
    _report_name    = f"{_stem.lstrip("_")}_report.md" if _stem != "_analysis" else "_report.md"

    # ── Resolve input path (for non-manifest modes) ───────────────────────────
    # --input-json and --merged-json are treated identically after this point.
    input_json_path: Path | None = None
    if args.merged_json:
        input_json_path = Path(args.merged_json)
    elif args.input_json:
        input_json_path = Path(args.input_json)

    # instability produces no _merged.json — the CI yml short-circuits before
    # reaching this script, but guard here for manual dispatch fallback.
    if args.experiment == "instability":
        print("instability experiment: method-comparison analysis not applicable.", file=sys.stderr)
        print("Writing stub outputs so downstream CI steps do not fail.", file=sys.stderr)
        stub = {
            "experiment":      "instability",
            "experiment_mode": "instability",
            "n_total":         0,
            "fatal_conditions": [
                "WARN_INSTABILITY_NO_MERGED_JSON: instability outputs are CSVs/figures only; "
                "statistical method analysis was skipped."
            ],
        }
        (output_dir / _analysis_name).write_text(
            json.dumps(stub, indent=2), encoding="utf-8"
        )
        (output_dir / _report_name).write_text(
            "# HypatiaX Analysis Report — `instability`\n\n"
            "Instability experiment: method comparison analysis not applicable.\n"
            "See `figures/instability_analysis.csv` and accompanying figures for results.\n",
            encoding="utf-8",
        )
        print(f"✅ Stub {_analysis_name} and {_report_name} written.")
        sys.exit(0)

    # ── Load records ──────────────────────────────────────────────────────────
    if args.shard_manifest:
        # Shard-direct mode: load each file listed in the manifest and
        # concatenate records.  This is the path used by ci_analysis.yml when
        # INPUT_MODE=shards (non-benchmark-format experiments).
        manifest_path = Path(args.shard_manifest)
        if not manifest_path.exists():
            # ── NSHARDS=1 / direct-result fallback ────────────────────────────
            # Single-shard experiments (all except exp1b / exp3b)
            # may bypass shard merging entirely and commit JSON
            # files directly into RESULT_DIR.
            #
            # In that case ci_analysis.yml may not generate
            # shard_manifest.txt, so we automatically scan
            # RESULT_DIR for valid JSON result files.
            #
            # Fix: when --result-dir is set, scan that directory for result
            # JSON files (excluding internal meta-files that start with "_")
            # and load them directly — the same shape-detection logic in
            # _load_records_from_json handles all known formats (Shape P for
            # protocol_core_*.json, Shape A/B/C for others).  If result_dir
            # is absent or contains no JSON files, fall through to the original
            # hard error so genuine manifest misconfigurations still fail loudly.
            fallback_used = False
            if args.result_dir:
                fallback_dir = Path(args.result_dir)
                # Collect all non-meta JSON files one level deep in result_dir.
                candidate_jsons = sorted(
                    p for p in fallback_dir.glob("*.json")
                    if not p.name.startswith("_")   # exclude _merged, _stats, _checkpoint, _analysis
                )
                if candidate_jsons:
                    print(
                        f"::warning::--shard-manifest '{manifest_path}' not found — "
                        f"this is normal for NSHARDS=1 / direct-result experiments "
                        f"(e.g. exp1) that commit JSON directly without a shard-merge step. "
                        f"Falling back to scanning result_dir '{fallback_dir}' "
                        f"({len(candidate_jsons)} JSON file(s) found).",
                        file=sys.stderr,
                    )
                    records: list[dict] = []
                    for jp in candidate_jsons:
                        print(f"  Loading (direct-result fallback): {jp.name} …")
                        records.extend(_load_records_from_json(jp, args.experiment))
                    print(
                        f"  {len(records)} total record(s) loaded via direct-result fallback "
                        f"from {len(candidate_jsons)} file(s) in '{fallback_dir}'."
                    )
                    fallback_used = True

            if not fallback_used:
                # No result_dir or no JSON files found there — this is a genuine
                # misconfiguration (wrong manifest path, wrong RESULT_DIR, etc.).
                print(
                    f"::error::shard manifest not found: '{manifest_path}' and no "
                    f"fallback JSON files found"
                    + (f" in result_dir '{args.result_dir}'" if args.result_dir else
                       " (--result-dir not set, cannot auto-discover)") + ".\n"
                    f"  For NSHARDS=1 experiments (exp1, etc.), ci_analysis.yml should "
                    f"either write a shard_manifest.txt listing the result JSON, or pass "
                    f"--input-json <path> directly instead of --shard-manifest.",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            shard_paths = [
                Path(p.strip())
                for p in manifest_path.read_text(encoding="utf-8").splitlines()
                if p.strip()
            ]
            records: list[dict] = []
            for sp in shard_paths:
                if not sp.exists():
                    print(f"  WARNING: shard file not found, skipping: {sp}", file=sys.stderr)
                    continue
                print(f"  Loading shard: {sp.name} …")
                records.extend(_load_records_from_json(sp, args.experiment))
            print(f"  {len(records)} total records loaded from {len(shard_paths)} shard(s).")

    elif input_json_path is not None:
        # Single-file mode: --merged-json (legacy) or --input-json (NSHARDS=1 / CI direct).
        #
        # NOTE: the former exp2_feynman special-case that checked for
        # ablation_paired.json in output_dir has been removed.
        # ablation_paired.json is now committed to the repo by ci_analysis.yml
        # ("Commit ablation_paired.json" step) and is passed directly as
        # --input-json for exp2_feynman_extrap (mode=ablation).
        # exp2_feynman itself uses mode=standard and never needs the paired schema.
        if not input_json_path.exists():
            print(f"::error::input JSON not found at {input_json_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading {input_json_path} …")
        records = _load_records_from_json(input_json_path, args.experiment)
        print(f"  {len(records)} records loaded.")

    else:
        print("::error::No input source resolved (internal error).", file=sys.stderr)
        sys.exit(1)

    print(f"  Experiment mode: {_get_mode(args.experiment)}")

    # Read PySR fit timeout parameters from the environment.  Set by
    # ci_analysis.yml from repository variables (vars.PYSR_FIT_WALL_TIMEOUT /
    # vars.PYSR_FIT_GRACE_SECS), which mirror the values used in ci_experiment.yml
    # workers.  Recorded in _analysis.json for provenance only; not used to
    # drive any computation here.  Empty string / missing → None (not set in CI).
    def _env_int_or_none(name: str) -> int | None:
        raw_val = os.environ.get(name, "").strip()
        if not raw_val:
            return None
        try:
            return int(raw_val)
        except ValueError:
            print(f"WARNING: {name}={raw_val!r} is not an integer — recorded as null.",
                  file=sys.stderr)
            return None

    pysr_fit_params: dict = {}
    wall_timeout = _env_int_or_none("PYSR_FIT_WALL_TIMEOUT")
    grace_secs   = _env_int_or_none("PYSR_FIT_GRACE_SECS")
    if wall_timeout is not None:
        pysr_fit_params["wall_timeout_s"] = wall_timeout
    if grace_secs is not None:
        pysr_fit_params["grace_secs"] = grace_secs
    if pysr_fit_params:
        print(f"  PySR fit params: {pysr_fit_params}")
    else:
        print("  PySR fit params: not set (PYSR_FIT_WALL_TIMEOUT / PYSR_FIT_GRACE_SECS absent)")

    if not _SCIPY_OK:
        print("WARNING: scipy not available — Mann-Whitney tests will be skipped.", file=sys.stderr)

    # exp1_ablation uses a dedicated analysis path (different input schema).
    if _get_mode(args.experiment) == "ablation":
        print("Running ablation analysis …")
        analysis = analyse_ablation(records, experiment=args.experiment,
                                    pysr_fit_params=pysr_fit_params)
        analysis_path = output_dir / _analysis_name
        report_path   = output_dir / _report_name
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"✅ {_analysis_name} → {analysis_path}")
        write_report_ablation(analysis, report_path)
        print(f"✅ {_report_name}     → {report_path}")
        all_conds  = analysis.get("fatal_conditions", [])
        hard_fatal = [c for c in all_conds if not (c.startswith("INFO_") or c.startswith("WARN_"))]
        soft_conds = [c for c in all_conds if c.startswith("INFO_") or c.startswith("WARN_")]
        if soft_conds:
            print(f"\nℹ️  {len(soft_conds)} informational/warning condition(s):", file=sys.stderr)
            for sc in soft_conds:
                print(f"  - {sc}", file=sys.stderr)
        if hard_fatal:
            print(f"\n⚠️  {len(hard_fatal)} fatal condition(s) detected:", file=sys.stderr)
            for fc in hard_fatal:
                print(f"  - {fc}", file=sys.stderr)
            sys.exit(0)   # CI abort step reads _analysis.json
        print("\nAblation analysis complete. No fatal conditions.")
        return

    print("Running analysis …")
    analysis = analyse(records, experiment=args.experiment,
                       pysr_fit_params=pysr_fit_params)

    analysis_path = output_dir / _analysis_name
    report_path   = output_dir / _report_name

    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"✅ {_analysis_name} → {analysis_path}")

    write_report(analysis, report_path)
    print(f"✅ {_report_name}     → {report_path}")

    all_conds  = analysis.get("fatal_conditions", [])
    hard_fatal = [c for c in all_conds if not (c.startswith("INFO_") or c.startswith("WARN_"))]
    soft_conds = [c for c in all_conds if c.startswith("INFO_") or c.startswith("WARN_")]

    if soft_conds:
        print(f"\nℹ️  {len(soft_conds)} informational/warning condition(s):", file=sys.stderr)
        for sc in soft_conds:
            print(f"  - {sc}", file=sys.stderr)

    if hard_fatal:
        print(f"\n⚠️  {len(hard_fatal)} fatal condition(s) detected:", file=sys.stderr)
        for fc in hard_fatal:
            print(f"  - {fc}", file=sys.stderr)
        print("\nReport committed. ci_analysis.yml will abort the workflow.", file=sys.stderr)
        # Exit 0 here — the CI abort step reads fatal_conditions from _analysis.json
        # and calls sys.exit(1) itself, AFTER the commit step.
        sys.exit(0)

    print("\nAnalysis complete. No fatal conditions.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_instability_suite.py
========================
HypatiaX DeFi — Consolidated Instability Analysis & Figure Pipeline

Single entry point that runs the full instability analysis in three sequential
stages, replacing the previous three-script chain:

  OLD (broken):
    hypatiax_instability_analysis_pipeline.py   →  instability_analysis.csv + 5 figs
    build_extrapolation_pipeline.py             →  instability_extrapolation.csv + 1 fig
    hypatiax_plot_instability_all.py            →  11 figs (A1–C5)

  NEW (this script):
    run_instability_suite.py                    →  CSV + all 12 figures, one call

Stage 1 — Instability analysis
  Loads K-run R² data from one of three sources (auto-detected):
    • hypatiax_defi_variance_results.json           (preferred, --variance runs)
    • hypatiax_defi_benchmark_v3_results_*.json     (multi-run timestamped files)
    • hypatiax_defi_benchmark_v3_results.json       (single-run fallback)
  Computes per-case mean, std (= II), p_i, regime, complexity.
  Exports instability_analysis.csv.

Stage 2 — Extrapolation merge  (optional, requires --benchmark-json)
  Reads the benchmark JSON for extrapolation R² (Sympy Mode A, or test_r2 proxy).
  Merges with instability_analysis.csv on the 'case' key.
  Exports instability_extrapolation.csv (feeds notebook cell 34 / fig19).

Stage 3 — Figures
  Generates 11 instability figures (Groups A, B, C) + 1 extrapolation scatter:

  Group A — per-case (matplotlib, operate on list-of-dicts)
    A1  hypatiax_instability_per_case       Per-case mean ± II, regime colour bands
    A2  hypatiax_instability_histogram      II histogram + regime pie chart
    A3  hypatiax_instability_scatter        Mean R² vs II scatter

  Group B — phase-space (matplotlib, 3-D)
    B1  fig_instability_3d                  3D scatter μ × II × p_i
    B2  fig_instability_phase               2D phase plot, regime boundaries
    B3  fig_instability_hist                II histogram (no KDE)
    B4  fig_instability_success_vs_instability  II vs p_i tradeoff
    B5  fig_instability_regimes             Regime-count bar chart
    B6  fig_instability_surface             3D IDW surface p_i(μ, II)

  Group C — complexity / seaborn (operate on DataFrame)
    C1  fig_paper_complexity_vs_instability KEY: K vs II + OLS regression
    C2  fig_paper_complexity_vs_success     K vs p_i
    C3  fig_paper_mean_vs_instability       Mean R² vs II (seaborn)
    C4  fig_paper_instability_hist          KDE histogram (seaborn)
    C5  fig_paper_regime_counts             Regime bar chart (seaborn)

  Extrapolation (requires Stage 2)
    EX  fig_instability_vs_extrapolation    II vs extrap R² scatter (notebook §16)

Output locations (all under --out, default hypatiax/data/figures/):
  instability_analysis.csv           Stage 1 CSV
  instability_extrapolation.csv      Stage 2 CSV (if --benchmark-json supplied)
  fig_paper_instability_hist.{png,pdf}    → fig_paper_instability_hist.png in paper
  fig_paper_regime_counts.{png,pdf}       → fig_paper_regime_counts.png
  hypatiax_instability_per_case.{png,pdf} → hypatiax_instability_per_case.png
  … (all 12 stems)

Usage
─────
  # Full run (all stages + all figures)
  python run_instability_suite.py

  # Specify data location and output directory
  python run_instability_suite.py \\
      --results-dir hypatiax/data/results \\
      --out         hypatiax/data/figures

  # Include Stage 2 extrapolation merge
  python run_instability_suite.py \\
      --benchmark-json hypatiax_defi_benchmark_v3c2_results.json

  # Only specific figure groups
  python run_instability_suite.py --group A C

  # Only specific figure codes
  python run_instability_suite.py --figures A1 C1 C4 C5 EX

  # Formats
  python run_instability_suite.py --format png

  # Skip extrapolation figure even if Stage 2 ran
  python run_instability_suite.py --benchmark-json results.json --no-extrap-plot

  # run_all.sh integration (called from STEP instability):
  python run_instability_suite.py \\
      --results-dir ${RESULTS_DIR} \\
      --out         ${RESULTS_DIR}/figures \\
      --csv-out     ${RESULTS_DIR}/figures/instability_analysis.csv \\
      --benchmark-json ${RESULTS_DIR}/hypatiax_defi_benchmark_v3c2_results.json

Author  : HypatiaX Team
Version : 1.0 — consolidated (replaces three separate pipeline scripts)
Date    : 2026
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── third-party ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — registers 3d projection
    _3D_OK = True
except ImportError:
    _3D_OK = False

# ── Publication style ─────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["font.family"]       = "serif"
plt.rcParams["figure.dpi"]        = 300
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SHARED CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

_RESULTS_DIR_DEFAULT = Path("hypatiax/data/results")
_FIGURES_DIR_DEFAULT = Path("hypatiax/data/figures")

_VARIANCE_JSON  = _RESULTS_DIR_DEFAULT / "hypatiax_defi_variance_results.json"
_FINAL_JSON     = _RESULTS_DIR_DEFAULT / "hypatiax_defi_benchmark_v3_results.json"
_MULTI_PATTERN  = re.compile(r"hypatiax_defi_benchmark_v3\w*_results_\d{8}T\d{6}Z\.json$")

REGIME_PALETTE: Dict[str, str] = {
    "A-Symbolic":   "#2ca02c",
    "B-Approx":     "#ff7f0e",
    "B-Det.Biased": "#d6b219",
    "C-Marginal":   "#e87722",
    "C-Collapse":   "#d62728",
    "?":            "#aaaaaa",
}
REGIME_LABELS: Dict[str, str] = {
    "A-Symbolic":   "Regime A — Symbolic Stability (std≈0, mean≈1)",
    "B-Approx":     "Regime B — Deterministic Biased (std≈0, mean<1)",
    "B-Det.Biased": "Regime B* — Borderline Stochastic (0<std<0.05)",
    "C-Marginal":   "Regime C-Marginal (0.05≤std<0.10)",
    "C-Collapse":   "Regime C — Stochastic Collapse (std≥0.10 or mean<0)",
    "?":            "Undetermined",
}
REGIME_ORDER = ["A-Symbolic", "B-Approx", "B-Det.Biased", "C-Marginal", "C-Collapse", "?"]
_SNS_PALETTE  = dict(REGIME_PALETTE)

ALL_FIGURES  = ["A1", "A2", "A3", "B1", "B2", "B3", "B4", "B5", "B6",
                "C1", "C2", "C3", "C4", "C5", "EX"]
GROUP_MAP    = {
    "A":  ["A1", "A2", "A3"],
    "B":  ["B1", "B2", "B3", "B4", "B5", "B6"],
    "C":  ["C1", "C2", "C3", "C4", "C5"],
    "EX": ["EX"],
}
FIGURE_LABELS = {
    "A1": "Per-case mean R² ± II error bars",
    "A2": "II histogram + regime pie chart",
    "A3": "Mean R² vs II scatter",
    "B1": "3D phase scatter (μ × II × p_i)",
    "B2": "2D phase plot, regime boundaries",
    "B3": "II histogram (matplotlib, no KDE)",
    "B4": "Success–instability tradeoff",
    "B5": "Regime-count bar chart",
    "B6": "3D IDW surface p_i(μ, II)",
    "C1": "Complexity vs II — KEY FIGURE",
    "C2": "Complexity vs p_i",
    "C3": "Mean R² vs II (seaborn)",
    "C4": "KDE histogram of II (seaborn)",
    "C5": "Regime counts bar chart (seaborn)",
    "EX": "II vs extrapolation R² scatter",
}

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA LAYER
# ════════════════════════════════════════════════════════════════════════════

def classify_regime(mean_r2: float, std_r2: float) -> str:
    """Classify a case into a regime.  Mirrors benchmark v3c2 classifier."""
    if mean_r2 != mean_r2 or std_r2 != std_r2:   # NaN check
        return "?"
    if mean_r2 < 0:
        return "C-Collapse"
    if std_r2 < 1e-6:
        return "A-Symbolic" if mean_r2 > 0.99 else "B-Approx"
    if std_r2 >= 0.1:
        return "C-Collapse"
    if std_r2 >= 0.05:
        return "C-Marginal"
    return "B-Det.Biased"


def complexity_proxy(case_name: str) -> int:
    """
    Kolmogorov-complexity proxy for the target formula (paper §4).
    Scale: 1 baseline  +1 algebraic  +2 portfolio/correlation
           +3 transcendental  +4 derivatives/Greeks
    Max ≈ 9 (e.g. Theta of option = 1+1+3+4).
    """
    name  = case_name.lower()
    score = 1
    if any(k in name for k in ["ratio", "price", "amount", "pnl", "var",
                                "apy", "rate", "fee", "margin", "leverage",
                                "collateral", "liquidat", "staking", "yield"]):
        score += 1
    if any(k in name for k in ["portfolio", "correlated", "sharpe",
                                "information", "tracking", "expected shortfall"]):
        score += 2
    if any(k in name for k in ["black", "scholes", "exp", "log",
                                "compounding", "borrowing", "impermanent"]):
        score += 3
    if any(k in name for k in ["delta", "gamma", "vega", "theta",
                                "rho", "greek", "option"]):
        score += 4
    return score


# ── JSON loaders ──────────────────────────────────────────────────────────────

def _load_variance_json(path: Path) -> Dict:
    """Load hypatiax_defi_variance_results.json (preferred K-run source)."""
    raw = json.loads(path.read_text())
    out: Dict = {}
    for rec in raw:
        name   = rec.get("test_case", rec.get("name", rec.get("equation_id", "?")))
        n_runs = rec.get("n_runs", len(rec.get("runs", [])))
        scores = [
            r["test_r2"] for r in rec.get("runs", [])
            if r.get("test_r2") is not None
            and not (isinstance(r["test_r2"], float) and np.isnan(r["test_r2"]))
        ]
        if scores:
            out[name] = {"scores": scores, "n_runs": n_runs}
    return out


def _load_multi_run_jsons(results_dir: Path) -> Dict:
    """Load per-run timestamped JSONs (hypatiax_defi_benchmark_v3_results_*.json)."""
    files = sorted(f for f in results_dir.iterdir() if _MULTI_PATTERN.match(f.name))
    if not files:
        return {}
    case_scores:   Dict[str, List[float]] = {}
    case_attempts: Dict[str, int]         = {}
    for fpath in files:
        raw   = json.loads(fpath.read_text())
        cases = raw["cases"] if isinstance(raw, dict) else raw
        for rec in cases:
            name   = rec.get("test_case", rec.get("name", rec.get("equation_id", "?")))
            res    = rec.get("results", {})
            r2_raw = (res.get("pure_llm") or res.get("llm_only") or {}).get("test_r2")
            case_attempts[name] = case_attempts.get(name, 0) + 1
            if r2_raw is None or (isinstance(r2_raw, float) and np.isnan(r2_raw)):
                continue
            case_scores.setdefault(name, []).append(float(r2_raw))
    return {
        name: {"scores": scores, "n_runs": case_attempts.get(name, len(scores))}
        for name, scores in case_scores.items() if scores
    }


def _load_single_json(path: Path) -> Dict:
    """Load a single-run benchmark JSON (no variance — II will be 0 for all cases)."""
    raw   = json.loads(path.read_text())
    cases = raw["cases"] if isinstance(raw, dict) else raw
    out: Dict = {}
    for rec in cases:
        name   = rec.get("test_case", rec.get("name", rec.get("equation_id", "?")))
        res    = rec.get("results", {})
        r2_raw = (res.get("pure_llm") or res.get("llm_only") or {}).get("test_r2")
        if r2_raw is None or (isinstance(r2_raw, float) and np.isnan(r2_raw)):
            continue
        out[name] = {"scores": [float(r2_raw)], "n_runs": 1}
    return out


def load_data(source: str, results_dir: Path) -> Dict:
    """
    Master loader.  source ∈ {"auto", "variance", "multi", "single"}.
    Returns {case_name: {"scores": List[float], "n_runs": int}}.
    Priority: variance JSON > multi-run JSONs > single-run JSON (any v3* variant).
    """
    variance_json = results_dir / "hypatiax_defi_variance_results.json"
    # Exact canonical name kept as primary; glob fallback catches v3c, v3c2, etc.
    final_json    = results_dir / "hypatiax_defi_benchmark_v3_results.json"

    print(f"  [load_data] results_dir={results_dir}  source={source}")
    if results_dir.exists():
        candidates = sorted(results_dir.glob("hypatiax_defi_benchmark_v3*.json"))
        print(f"  [load_data] v3 JSON candidates: {[f.name for f in candidates]}")
    else:
        print(f"  [load_data] results_dir does not exist: {results_dir}")

    if source in ("variance", "auto") and variance_json.exists():
        data = _load_variance_json(variance_json)
        if data:
            print(f"  ✅ Loaded variance JSON: {variance_json} ({len(data)} cases)")
            return data
    if source in ("multi", "auto") and results_dir.exists():
        data = _load_multi_run_jsons(results_dir)
        if data:
            print(f"  ✅ Loaded {len(data)} cases from timestamped multi-run JSONs")
            return data
    if source in ("single", "auto"):
        # Try exact canonical name first, then any v3* benchmark result file.
        single_candidates = (
            [final_json] if final_json.exists()
            else sorted(results_dir.glob("hypatiax_defi_benchmark_v3*results*.json"))
            if results_dir.exists() else []
        )
        for candidate in single_candidates:
            data = _load_single_json(candidate)
            if data:
                print(f"  ⚠️  Single-run fallback: {candidate} ({len(data)} cases, II=0)")
                return data
    print("❌ No results data found. Run the benchmark first:")
    print("   python hypatiax_defi_benchmark_v3c2.py --variance")
    print("   python hypatiax_defi_benchmark_v3c2.py --multi-run 30")
    sys.exit(1)


# ── DataFrame builder ─────────────────────────────────────────────────────────

def build_dataframe(data: Dict) -> pd.DataFrame:
    """
    Build the canonical per-case DataFrame used by all figure groups.

    Columns
    -------
    case, mean, std, ii, p_i, n_valid, n_runs, min, max,
    regime, complexity, colour
    Sorted by regime order then descending mean R².
    """
    rows = []
    for name, v in data.items():
        scores = v["scores"]
        n_runs = v["n_runs"]
        mean_  = float(np.mean(scores))
        std_   = float(np.std(scores))
        p_i    = sum(1 for s in scores if s > 0.9) / n_runs
        regime = classify_regime(mean_, std_)
        rows.append({
            "case":       name,
            "mean":       round(mean_, 4),
            "std":        round(std_,  4),
            "ii":         round(std_,  4),   # canonical alias: II_i = σ_i
            "p_i":        round(p_i,   4),
            "n_valid":    len(scores),
            "n_runs":     n_runs,
            "min":        round(float(min(scores)), 4),
            "max":        round(float(max(scores)), 4),
            "regime":     regime,
            "complexity": complexity_proxy(name),
            "colour":     REGIME_PALETTE.get(regime, "#aaaaaa"),
        })
    regime_idx = {r: i for i, r in enumerate(REGIME_ORDER)}
    df = pd.DataFrame(rows)
    df["_ro"] = df["regime"].map(lambda r: regime_idx.get(r, 99))
    df = df.sort_values(["_ro", "mean"], ascending=[True, False]).drop(
        columns="_ro").reset_index(drop=True)
    return df


def df_to_rows(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to list-of-dicts for matplotlib-only figure functions."""
    return df.to_dict("records")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — STAGE 2: EXTRAPOLATION MERGE
# ════════════════════════════════════════════════════════════════════════════

def _safe_float(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _iterate_cases(data) -> List[Tuple[str, Dict]]:
    """Unified JSON iterator (dict root or list root)."""
    if isinstance(data, dict):
        cases = data.get("cases", data)
        if isinstance(cases, list):
            return [(c.get("test_case", c.get("case", "Unnamed")), c)
                    for c in cases if isinstance(c, dict)]
        return list(cases.items())
    if isinstance(data, list):
        return [(c.get("test_case", c.get("case", "Unnamed")), c)
                for c in data if isinstance(c, dict)]
    raise ValueError("Unsupported JSON root type — expected dict or list.")


def build_extrapolation_df(benchmark_data) -> pd.DataFrame:
    """
    Extract extrapolation R² from the benchmark JSON.

    Priority per case:
      Mode A — Sympy: case has 'formula' + 'predicted_formula' + 'var_ranges'
                      → true OOD evaluation (requires formula logging in benchmark)
      Mode C — Precomputed: use test_r2 from JSON as proxy
                      → NOTE: not true OOD; re-run benchmark with --llm-only for Mode A

    Returns DataFrame with columns: case, extrapolation_r2, success, failure, eval_mode
    """
    rows = []
    for case_name, case_data in _iterate_cases(benchmark_data):
        res = case_data.get("results", {})
        llm_res = res.get("hybrid") or res.get("pure_llm") or res.get("llm_only") or {}

        eval_mode      = "precomputed"
        extrap_r2      = _safe_float(llm_res.get("test_r2"))

        # Mode A: true OOD (only if formula fields present)
        formula     = case_data.get("formula")
        pred_form   = (llm_res.get("predicted_formula") or
                       case_data.get("predicted_formula"))
        var_ranges  = case_data.get("var_ranges")
        if formula and pred_form and var_ranges:
            # Attempt Sympy OOD evaluation (graceful fallback to precomputed)
            try:
                import sympy as sp
                f_true = sp.lambdify(list(sp.Symbol(k) for k in var_ranges),
                                     sp.sympify(formula), "numpy")
                f_pred = sp.lambdify(list(sp.Symbol(k) for k in var_ranges),
                                     sp.sympify(pred_form), "numpy")
                rng   = np.random.RandomState(7777)
                n_ood = 200
                X_ood = np.column_stack([
                    rng.uniform(hi * 1.0, hi * 2.0, n_ood)
                    for hi in [v[1] if isinstance(v, (list, tuple)) else v
                                for v in var_ranges.values()]
                ])
                ks    = list(var_ranges.keys())
                y_true = f_true(**{k: X_ood[:, i] for i, k in enumerate(ks)})
                y_pred = f_pred(**{k: X_ood[:, i] for i, k in enumerate(ks)})
                from sklearn.metrics import r2_score as _r2
                extrap_r2 = float(_r2(y_true.ravel(), y_pred.ravel()))
                eval_mode = "sympy_ood"
            except Exception:
                pass   # fall through to precomputed

        # Mode B: pre-stored arrays
        if eval_mode == "precomputed":
            y_true_arr = llm_res.get("y_true") or case_data.get("y_true")
            y_pred_arr = llm_res.get("y_pred") or case_data.get("y_pred")
            if y_true_arr and y_pred_arr:
                try:
                    from sklearn.metrics import r2_score as _r2
                    extrap_r2 = float(_r2(
                        np.array(y_true_arr, dtype=float),
                        np.array(y_pred_arr, dtype=float),
                    ))
                    eval_mode = "stored_arrays"
                except Exception:
                    pass

        success = (not np.isnan(extrap_r2)) and extrap_r2 >= 0.99
        failure = (not np.isnan(extrap_r2)) and extrap_r2 < 0.99
        rows.append({
            "case":           case_name,
            "extrapolation_r2": extrap_r2,
            "success":        success,
            "failure":        failure,
            "eval_mode":      eval_mode,
        })
    return pd.DataFrame(rows)


def merge_instability_extrap(
    instab_df: pd.DataFrame,
    extrap_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join instability_analysis DataFrame with extrapolation DataFrame on 'case'.
    Cases with no match get NaN extrapolation_r2 (not evaluated or not in benchmark).
    """
    merged = instab_df.merge(extrap_df, on="case", how="left")
    n_match = merged["extrapolation_r2"].notna().sum()
    print(f"  Merged: {len(merged)} instability cases, "
          f"{n_match} with extrapolation R² ({len(merged)-n_match} no-match → NaN)")
    return merged


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SHARED FIGURE HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _legend_handles(rows_or_df) -> List:
    """Build matplotlib Patch handles for the regimes present in the data."""
    if isinstance(rows_or_df, pd.DataFrame):
        present = rows_or_df["regime"].dropna().unique()
    else:
        present = {r["regime"] for r in rows_or_df}
    seen = sorted(present, key=lambda r: REGIME_ORDER.index(r) if r in REGIME_ORDER else 99)
    return [
        mpatches.Patch(facecolor=REGIME_PALETTE.get(r, "#aaaaaa"),
                       label=REGIME_LABELS.get(r, r), alpha=0.85)
        for r in seen
    ]


def _save(fig, out_dir: Path, stem: str, fmt: List[str]):
    """Save figure in all requested formats and close it."""
    for f in fmt:
        p = out_dir / f"{stem}.{f}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"    💾 {p}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — GROUP A FIGURES  (per-case, matplotlib)
# ════════════════════════════════════════════════════════════════════════════

def plot_A1_per_case(rows: List[Dict], out_dir: Path, fmt: List[str]):
    """A1 — Per-case mean R² ± II error bars, regime-coloured + background bands."""
    n = len(rows)
    if n == 0:
        print("  ⚠️  A1: no data."); return

    names   = [r["case"]   for r in rows]
    means   = np.array([r["mean"]   for r in rows])
    stds    = np.array([r["std"]    for r in rows])
    colours = [r["colour"] for r in rows]
    p_is    = [r["p_i"]    for r in rows]
    regimes = [r["regime"] for r in rows]

    fig, ax = plt.subplots(figsize=(max(10, n * 0.55), 6))
    xs = np.arange(n)

    # Regime background bands
    prev_regime = None; band_start = 0
    for i, reg in enumerate(regimes + [None]):
        if reg != prev_regime:
            if prev_regime is not None:
                ax.axvspan(band_start - 0.5, i - 0.5,
                           color=REGIME_PALETTE.get(prev_regime, "#aaaaaa"),
                           alpha=0.06, zorder=0)
            band_start = i; prev_regime = reg

    # Error bars then scatter dots
    for x, m, s, col in zip(xs, means, stds, colours):
        lo = min(s, m + 1.05)
        ax.errorbar(x, m, yerr=[[lo], [s]], fmt="none",
                    ecolor=col, elinewidth=1.4, capsize=3, capthick=1.2,
                    alpha=0.55, zorder=2)
        ax.scatter(x, m, color=col, s=60, zorder=3, linewidths=0)

    # p_i annotation for unstable cases
    for x, m, s, pi in zip(xs, means, stds, p_is):
        if s > 1e-6 and pi < 1.0:
            ax.text(x, min(m + s + 0.04, 1.12), f"p={pi:.2f}",
                    ha="center", va="bottom", fontsize=6.5, color="#555555", rotation=90)

    ax.axhline(1.0, color="#2ca02c", linewidth=0.8, linestyle="--", alpha=0.55, zorder=1)
    ax.axhline(0.9, color="#ff7f0e", linewidth=0.6, linestyle="--", alpha=0.35, zorder=1)
    ax.axhline(0.0, color="#888888", linewidth=0.6, linestyle=":",  alpha=0.40, zorder=1)

    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"Test $R^2$", fontsize=11)
    ax.set_xlabel("Benchmark case", fontsize=11)
    ax.set_ylim(-1.1, 1.25)
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.grid(axis="y", which="minor", linestyle=":",  linewidth=0.3, alpha=0.25)

    n_runs = rows[0]["n_runs"] if rows else 1
    suffix = f" ({n_runs} runs/case)" if n_runs > 1 else ""
    ax.set_title(
        f"LLM Instability Distribution — HypatiaX DeFi Benchmark{suffix}\n"
        r"Error bars = $\pm\,\mathrm{II}_i$ (Instability Index = $\sigma_i$); "
        r"$p_i$ = $\mathbb{P}(R^2 > 0.9)$",
        fontsize=10, pad=8,
    )
    ax.legend(handles=_legend_handles(rows), loc="lower right",
              fontsize=7.5, framealpha=0.9, edgecolor="#cccccc")
    fig.tight_layout()
    _save(fig, out_dir, "hypatiax_instability_per_case", fmt)


def plot_A2_ii_histogram(rows: List[Dict], out_dir: Path, fmt: List[str]):
    """A2 — II distribution histogram (left) + regime pie chart (right)."""
    ii_vals = [r["ii"] for r in rows if r["std"] == r["std"]]
    if not ii_vals:
        print("  ⚠️  A2: no II values."); return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                              gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    bins = np.linspace(0, max(max(ii_vals) * 1.05, 0.35), 25)
    for reg in REGIME_ORDER:
        vals = [r["ii"] for r in rows if r["regime"] == reg and r["std"] == r["std"]]
        if vals:
            ax.hist(vals, bins=bins, color=REGIME_PALETTE[reg], alpha=0.75,
                    label=REGIME_LABELS.get(reg, reg), edgecolor="white", linewidth=0.5)
    ax.axvline(0.05, color="#e87722", linewidth=1.2, linestyle="--",
               label="C threshold (II=0.05)", alpha=0.8)
    ax.axvline(0.10, color="#d62728", linewidth=1.2, linestyle="--",
               label="Severe threshold (II=0.10)", alpha=0.8)
    ax.set_xlabel(r"Instability Index $\mathrm{II}_i = \sigma_i$", fontsize=11)
    ax.set_ylabel("Number of cases", fontsize=11)
    ax.set_title(
        "Distribution of Instability Index across Benchmark Cases\n"
        r"$\mathrm{II}_i = \sigma_i = \mathrm{std}(R^2_i)$ across independent LLM runs",
        fontsize=10)
    ax.legend(fontsize=7.5, framealpha=0.9, edgecolor="#cccccc")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    ax2 = axes[1]
    regime_counts = Counter(r["regime"] for r in rows)
    labels_  = [r for r in REGIME_ORDER if r in regime_counts]
    wlabels  = [f"{r}\n(n={regime_counts[r]})" for r in labels_]
    wedges, texts, autotexts = ax2.pie(
        [regime_counts[r] for r in labels_],
        labels=wlabels,
        colors=[REGIME_PALETTE[r] for r in labels_],
        autopct=lambda p: f"{p:.0f}%" if p > 4 else "",
        startangle=140, pctdistance=0.75,
        wedgeprops={"linewidth": 0.7, "edgecolor": "white"},
        textprops={"fontsize": 7.5},
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax2.set_title("Regime Distribution\n(paper §4 taxonomy)", fontsize=10)
    fig.suptitle("HypatiaX DeFi Benchmark — LLM Stochastic Instability Analysis",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, out_dir, "hypatiax_instability_histogram", fmt)


def plot_A3_scatter(rows: List[Dict], out_dir: Path, fmt: List[str]):
    """A3 — Mean R² vs II scatter with outlier annotations."""
    if not rows: return
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for r in rows:
        ax.scatter(r["ii"], r["mean"], color=r["colour"],
                   s=55, zorder=3, linewidths=0, alpha=0.85)
        if r["ii"] > 0.04 or r["mean"] < 0.5:
            short = r["case"][:26] + "…" if len(r["case"]) > 27 else r["case"]
            ax.annotate(short, (r["ii"], r["mean"]),
                        fontsize=6.5, ha="left", va="center",
                        xytext=(5, 0), textcoords="offset points", color="#333333",
                        arrowprops=dict(arrowstyle="-", color="#aaaaaa",
                                        lw=0.6, shrinkA=0, shrinkB=3))
    ax.axvline(0.05, color="#e87722", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axvline(0.10, color="#d62728", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axhline(0.9,  color="#ff7f0e", linewidth=0.8, linestyle=":",  alpha=0.5)
    ax.legend(handles=_legend_handles(rows), fontsize=7.5, loc="upper right",
              framealpha=0.9, edgecolor="#cccccc")
    ax.set_xlabel(r"Instability Index $\mathrm{II}_i = \sigma_i$", fontsize=11)
    ax.set_ylabel(r"Mean test $R^2$ ($\mu_i$)", fontsize=11)
    ax.set_title("Mean $R^2$ vs Instability Index\n"
                 "LLMs operate in two distinct modes: Symbolic (A) vs Stochastic (C)",
                 fontsize=10)
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    _save(fig, out_dir, "hypatiax_instability_scatter", fmt)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — GROUP B FIGURES  (phase-space, some 3D)
# ════════════════════════════════════════════════════════════════════════════

def plot_B1_3d(rows: List[Dict], out_dir: Path, fmt: List[str]):
    """B1 — 3D scatter: μ × II × p_i, coloured by regime."""
    if not _3D_OK:
        print("  ⚠️  B1: mpl_toolkits.mplot3d unavailable — skipped."); return
    means   = np.array([r["mean"] for r in rows])
    stds    = np.array([r["std"]  for r in rows])
    pis     = np.array([r["p_i"]  for r in rows])
    colours = [r["colour"] for r in rows]
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(means, stds, pis, c=colours, s=60, depthshade=True, alpha=0.85)
    for r, m, s, p in zip(rows, means, stds, pis):
        if s > 0.1 or m < 0.5:
            short = r["case"][:22] + "…" if len(r["case"]) > 23 else r["case"]
            ax.text(m, s, p, short, fontsize=6, color="#333333")
    ax.set_xlabel(r"Mean $R^2$ ($\mu_i$)",               fontsize=9, labelpad=8)
    ax.set_ylabel(r"Instability Index (II = $\sigma_i$)", fontsize=9, labelpad=8)
    ax.set_zlabel(r"Success Prob. ($p_i$)",               fontsize=9, labelpad=8)
    ax.set_title("LLM Instability Phase Space\n"
                 r"Axes: $\mu_i$ × $\mathrm{II}_i$ × $p_i$", fontsize=10)
    fig.legend(handles=_legend_handles(rows), loc="lower left",
               fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    _save(fig, out_dir, "fig_instability_3d", fmt)


def plot_B2_phase(rows: List[Dict], out_dir: Path, fmt: List[str]):
    """B2 — 2D phase plot: mean R² vs II with regime boundary lines."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for r in rows:
        ax.scatter(r["mean"], r["std"], color=r["colour"],
                   s=55, zorder=3, alpha=0.85, linewidths=0)
        if r["std"] > 0.1 or r["mean"] < 0.5:
            short = r["case"][:26] + "…" if len(r["case"]) > 27 else r["case"]
            ax.annotate(short, (r["mean"], r["std"]),
                        fontsize=6.5, ha="left", va="center",
                        xytext=(5, 0), textcoords="offset points", color="#333333",
                        arrowprops=dict(arrowstyle="-", color="#bbbbbb",
                                        lw=0.6, shrinkA=0, shrinkB=3))
    ax.axvline(0.9,  color="#ff7f0e", linewidth=0.9, linestyle=":", alpha=0.55)
    ax.axhline(0.05, color="#e87722", linewidth=0.9, linestyle="--", alpha=0.65)
    ax.axhline(0.10, color="#d62728", linewidth=0.9, linestyle="--", alpha=0.65)
    handles = _legend_handles(rows) + [
        mpatches.Patch(color="#ff7f0e", alpha=0.5, label="R²=0.9 threshold"),
        mpatches.Patch(color="#d62728", alpha=0.5, label="II thresholds (0.05 / 0.10)"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="upper right",
              framealpha=0.9, edgecolor="#cccccc")
    ax.set_xlabel(r"Mean test $R^2$ ($\mu_i$)", fontsize=11)
    ax.set_ylabel(r"Instability Index $\mathrm{II}_i = \sigma_i$", fontsize=11)
    ax.set_title("Regime Separation: Mean $R^2$ vs Instability Index\n"
                 "LLMs operate in two distinct modes: Symbolic (A) vs Stochastic (C)",
                 fontsize=10)
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    _save(fig, out_dir, "fig_instability_phase", fmt)


def plot_B3_hist(rows: List[Dict], out_dir: Path, fmt: List[str]):
    """B3 — Stacked histogram of II by regime (no KDE)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, max(max(r["std"] for r in rows) * 1.05, 0.35), 25)
    for reg in REGIME_ORDER:
        vals = [r["std"] for r in rows if r["regime"] == reg]
        if vals:
            ax.hist(vals, bins=bins, color=REGIME_PALETTE[reg], alpha=0.75,
                    label=REGIME_LABELS.get(reg, reg),
                    edgecolor="white", linewidth=0.5)
    ax.axvline(0.05, color="#e87722", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.axvline(0.10, color="#d62728", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.set_xlabel(r"Instability Index $\mathrm{II}_i = \sigma_i$", fontsize=11)
    ax.set_ylabel("Number of cases", fontsize=11)
    ax.set_title(
        "Distribution of LLM Instability Index\n"
        r"$\mathrm{II}_i = \sigma_i = \mathrm{std}(R^2_i)$ across independent runs",
        fontsize=10)
    ax.legend(fontsize=7.5, framealpha=0.9, edgecolor="#cccccc")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    _save(fig, out_dir, "fig_instability_hist", fmt)


def plot_B4_success(rows: List[Dict], out_dir: Path, fmt: List[str]):
    """B4 — II vs p_i (success–instability tradeoff)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        ax.scatter(r["std"], r["p_i"], color=r["colour"],
                   s=55, zorder=3, alpha=0.85, linewidths=0)
        if r["std"] > 0.08 or r["p_i"] < 0.3:
            short = r["case"][:26] + "…" if len(r["case"]) > 27 else r["case"]
            ax.annotate(short, (r["std"], r["p_i"]),
                        fontsize=6.5, ha="left", va="center",
                        xytext=(5, 0), textcoords="offset points", color="#333333",
                        arrowprops=dict(arrowstyle="-", color="#bbbbbb",
                                        lw=0.6, shrinkA=0, shrinkB=3))
    ax.axvline(0.05, color="#e87722", linewidth=0.9, linestyle="--", alpha=0.65)
    ax.axvline(0.10, color="#d62728", linewidth=0.9, linestyle="--", alpha=0.65)
    ax.axhline(0.5,  color="#888888", linewidth=0.8, linestyle=":",  alpha=0.5)
    ax.set_xlabel(r"Instability Index $\mathrm{II}_i = \sigma_i$", fontsize=11)
    ax.set_ylabel(r"Success Probability $p_i = \mathbb{P}(R^2 > 0.9)$", fontsize=11)
    ax.set_title("Success–Instability Tradeoff\n"
                 r"Higher II $\Rightarrow$ lower $p_i$: stochastic collapse degrades reliability",
                 fontsize=10)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(handles=_legend_handles(rows), fontsize=7.5,
              framealpha=0.9, edgecolor="#cccccc")
    fig.tight_layout()
    _save(fig, out_dir, "fig_instability_success_vs_instability", fmt)


def plot_B5_regimes(rows: List[Dict], out_dir: Path, fmt: List[str]):
    """B5 — Bar chart of regime counts."""
    counts  = Counter(r["regime"] for r in rows)
    labels  = [r for r in REGIME_ORDER if r in counts]
    values  = [counts[r] for r in labels]
    colours = [REGIME_PALETTE[r] for r in labels]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(range(len(labels)), values, color=colours,
                  alpha=0.82, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([r.replace("-", "\n") for r in labels], fontsize=9)
    ax.set_ylabel("Number of cases", fontsize=11)
    ax.set_xlabel("Regime (paper §4 taxonomy)", fontsize=11)
    ax.set_title("Regime Distribution — HypatiaX DeFi Benchmark\n"
                 "A: Symbolic Stability  |  B: Deterministic  |  C: Stochastic Collapse",
                 fontsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_ylim(0, max(values) * 1.18)
    fig.tight_layout()
    _save(fig, out_dir, "fig_instability_regimes", fmt)


def _idw_pi(rows: List[Dict], tx: float, ty: float) -> float:
    """Inverse-distance-weighted interpolation of p_i at (mean=tx, II=ty)."""
    w_sum = p_sum = 0.0
    for r in rows:
        dx = (tx - r["mean"]) * 2.5
        dy = (ty - r["std"])  * 5.0
        w  = 1.0 / (dx*dx + dy*dy + 1e-6) ** 2
        w_sum += w
        p_sum += w * r["p_i"]
    return p_sum / w_sum


def _hex_to_rgb01(h: str):
    return int(h[1:3], 16)/255, int(h[3:5], 16)/255, int(h[5:7], 16)/255


def _regime_boundary(mx: float, iy: float) -> str:
    if mx < 0:     return "C-Collapse"
    if iy < 1e-6:  return "A-Symbolic" if mx > 0.99 else "B-Approx"
    if iy >= 0.1:  return "C-Collapse"
    if iy >= 0.05: return "C-Marginal"
    return "B-Det.Biased"


def plot_B6_surface(rows: List[Dict], out_dir: Path, fmt: List[str],
                    elev: float = 28.0, azim: float = 225.0):
    """B6 — 3D IDW surface: p_i over (μ, II) space, coloured by regime."""
    if not _3D_OK:
        print("  ⚠️  B6: mpl_toolkits.mplot3d unavailable — skipped."); return
    NX, NY          = 55, 40
    mx_min, mx_max  = -0.65, 1.08
    ii_min, ii_max  =  0.00, 0.52
    mx_lin = np.linspace(mx_min, mx_max, NX)
    ii_lin = np.linspace(ii_min, ii_max, NY)
    MX, II = np.meshgrid(mx_lin, ii_lin)
    PI = np.array([[_idw_pi(rows, MX[iy, ix], II[iy, ix])
                    for ix in range(NX)] for iy in range(NY)])
    PI = np.clip(PI, 0.0, 1.0)
    face_colors = np.zeros((NY-1, NX-1, 4))
    for iy in range(NY-1):
        for ix in range(NX-1):
            pi_c = 0.25*(PI[iy,ix]+PI[iy,ix+1]+PI[iy+1,ix]+PI[iy+1,ix+1])
            reg  = _regime_boundary(0.5*(MX[iy,ix]+MX[iy,ix+1]),
                                    0.5*(II[iy,ix]+II[iy+1,ix]))
            r_,g_,b_ = _hex_to_rgb01(REGIME_PALETTE.get(reg, "#aaaaaa"))
            t = 0.25 + (1.0 - pi_c) * 0.45
            face_colors[iy,ix] = (r_+(1-r_)*t, g_+(1-g_)*t, b_+(1-b_)*t, 0.82)
    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.plot_surface(MX, II, PI, facecolors=face_colors,
                    linewidth=0.15, edgecolor="white",
                    antialiased=True, shade=True, alpha=0.88)
    mx_p = np.array([[mx_min,mx_max],[mx_min,mx_max]])
    pi_p = np.array([[0.0,0.0],[1.0,1.0]])
    for ii_val, col in ((0.05,"#e87722"),(0.10,"#d62728")):
        ii_p = np.full_like(mx_p, ii_val)
        ax.plot_surface(mx_p, ii_p, pi_p, color=col, alpha=0.10,
                        linewidth=0, shade=False)
        ax.plot([mx_min,mx_max],[ii_val,ii_val],[0,0],
                color=col, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.scatter([r["mean"] for r in rows], [r["std"] for r in rows],
               [r["p_i"] for r in rows],
               c=[r["colour"] for r in rows], s=55, zorder=5,
               edgecolors="white", linewidths=0.6, depthshade=True, alpha=0.95)
    ax.set_xlabel(r"Mean $R^2$  ($\mu_i$)",              fontsize=9, labelpad=10)
    ax.set_ylabel(r"Instability Index  $\mathrm{II}_i$", fontsize=9, labelpad=10)
    ax.set_zlabel(r"Success Prob.  $p_i$",               fontsize=9, labelpad=8)
    ax.set_xlim(mx_min,mx_max); ax.set_ylim(ii_min,ii_max); ax.set_zlim(0,1.05)
    ax.set_title("LLM Instability Regime Surface\n"
                 r"$p_i = \mathbb{P}(R^2_i > 0.9)$ over $(\mu_i,\,\mathrm{II}_i)$ space",
                 fontsize=10, pad=12)
    fig.legend(handles=_legend_handles(rows), loc="lower left",
               fontsize=7.5, framealpha=0.92, edgecolor="#cccccc",
               bbox_to_anchor=(0.01, 0.01))
    fig.tight_layout()
    _save(fig, out_dir, "fig_instability_surface", fmt)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — GROUP C FIGURES  (seaborn, operate on DataFrame)
# ════════════════════════════════════════════════════════════════════════════

def plot_C1_complexity_ii(df: pd.DataFrame, out_dir: Path, fmt: List[str],
                           show_regline: bool = True):
    """C1 — KEY FIGURE: complexity proxy K vs II + OLS regression (Instability Theorem)."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.scatterplot(data=df, x="complexity", y="std", hue="regime",
                    palette=_SNS_PALETTE,
                    hue_order=[r for r in REGIME_ORDER if r in df["regime"].values],
                    s=65, linewidth=0.4, edgecolor="white", ax=ax, zorder=3)
    for _, row in df.iterrows():
        if row["std"] > 0.08 or row["complexity"] >= 7:
            short = row["case"][:24] + "…" if len(row["case"]) > 25 else row["case"]
            ax.annotate(short, (row["complexity"], row["std"]),
                        fontsize=6.5, color="#333333", ha="left", va="bottom",
                        xytext=(4, 3), textcoords="offset points")
    if show_regline:
        sns.regplot(data=df, x="complexity", y="std", scatter=False, ci=95,
                    line_kws={"linestyle": "--", "linewidth": 1.4,
                              "color": "#555555", "alpha": 0.8}, ax=ax)
    ax.axhline(0.05, color="#e87722", linewidth=0.9, linestyle="--", alpha=0.65)
    ax.axhline(0.10, color="#d62728", linewidth=0.9, linestyle="--", alpha=0.65)
    corr = df[["complexity", "std"]].corr().iloc[0, 1]
    ax.text(0.97, 0.97, f"Pearson $r$ = {corr:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))
    ax.set_xlabel("Complexity proxy $K$  (algebraic=1–2, transcendental=4–5, Greeks=8–9)",
                  fontsize=9)
    ax.set_ylabel(r"Instability Index $\mathrm{II}_i = \sigma_i$", fontsize=11)
    ax.set_title("Complexity vs Instability — Empirical support for Instability Theorem\n"
                 r"$\uparrow K$ $\Rightarrow$ $\uparrow \mathrm{II}$: transcendental formulas "
                 "trigger stochastic collapse", fontsize=10)
    handles = _legend_handles(df) + [
        mpatches.Patch(color="#e87722", alpha=0.6, label="II=0.05 (C-Marginal boundary)"),
        mpatches.Patch(color="#d62728", alpha=0.6, label="II=0.10 (C-Collapse boundary)"),
    ]
    try:
        ax.get_legend().remove()
    except Exception:
        pass
    fig.legend(handles=handles, fontsize=7.5, framealpha=0.9, edgecolor="#cccccc",
               loc="upper left", bbox_to_anchor=(0.12, 0.88))
    fig.tight_layout()
    _save(fig, out_dir, "fig_paper_complexity_vs_instability", fmt)


def plot_C2_complexity_pi(df: pd.DataFrame, out_dir: Path, fmt: List[str]):
    """C2 — Complexity K vs success probability p_i."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="complexity", y="p_i", hue="regime",
                    palette=_SNS_PALETTE,
                    hue_order=[r for r in REGIME_ORDER if r in df["regime"].values],
                    s=65, linewidth=0.4, edgecolor="white", ax=ax, zorder=3)
    sns.regplot(data=df, x="complexity", y="p_i", scatter=False, ci=95,
                line_kws={"linestyle": "--", "linewidth": 1.4,
                          "color": "#555555", "alpha": 0.8}, ax=ax)
    ax.axhline(0.5, color="#888888", linewidth=0.8, linestyle=":", alpha=0.55)
    corr = df[["complexity", "p_i"]].corr().iloc[0, 1]
    ax.text(0.97, 0.97, f"Pearson $r$ = {corr:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))
    ax.set_xlabel("Complexity proxy $K$", fontsize=11)
    ax.set_ylabel(r"Success Probability $p_i = \mathbb{P}(R^2 > 0.9)$", fontsize=11)
    ax.set_title("Complexity vs Success Probability\n"
                 r"Higher $K$ $\Rightarrow$ lower $p_i$: reliability degrades with formula complexity",
                 fontsize=10)
    ax.set_ylim(-0.05, 1.1)
    try:
        ax.get_legend().remove()
    except Exception:
        pass
    fig.legend(handles=_legend_handles(df), fontsize=7.5, framealpha=0.9,
               edgecolor="#cccccc", loc="upper right", bbox_to_anchor=(0.92, 0.88))
    fig.tight_layout()
    _save(fig, out_dir, "fig_paper_complexity_vs_success", fmt)


def plot_C3_mean_ii(df: pd.DataFrame, out_dir: Path, fmt: List[str]):
    """C3 — Mean R² vs II (seaborn-styled, for paper §4 regime separation)."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.scatterplot(data=df, x="mean", y="std", hue="regime",
                    palette=_SNS_PALETTE,
                    hue_order=[r for r in REGIME_ORDER if r in df["regime"].values],
                    s=65, linewidth=0.4, edgecolor="white", ax=ax, zorder=3)
    for _, row in df.iterrows():
        if row["std"] > 0.1 or row["mean"] < 0.5:
            short = row["case"][:24] + "…" if len(row["case"]) > 25 else row["case"]
            ax.annotate(short, (row["mean"], row["std"]),
                        fontsize=6.5, color="#333333", ha="left", va="center",
                        xytext=(5, 0), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color="#bbbbbb",
                                        lw=0.6, shrinkA=0, shrinkB=3))
    ax.axvline(0.9,  color="#ff7f0e", linewidth=0.9, linestyle=":", alpha=0.55)
    ax.axhline(0.05, color="#e87722", linewidth=0.9, linestyle="--", alpha=0.65)
    ax.axhline(0.10, color="#d62728", linewidth=0.9, linestyle="--", alpha=0.65)
    ax.set_xlabel(r"Mean test $R^2$  ($\mu_i$)", fontsize=11)
    ax.set_ylabel(r"Instability Index $\mathrm{II}_i = \sigma_i$", fontsize=11)
    ax.set_title("Mean $R^2$ vs Instability — Regime Separation\n"
                 "Two-cluster structure: Regime A (symbolic) vs Regime C (stochastic)",
                 fontsize=10)
    try:
        ax.get_legend().remove()
    except Exception:
        pass
    fig.legend(handles=_legend_handles(df), fontsize=7.5, framealpha=0.9,
               edgecolor="#cccccc", loc="upper right", bbox_to_anchor=(0.92, 0.88))
    fig.tight_layout()
    _save(fig, out_dir, "fig_paper_mean_vs_instability", fmt)


def plot_C4_inst_hist(df: pd.DataFrame, out_dir: Path, fmt: List[str]):
    """C4 — KDE histogram of II distribution (seaborn overlay, paper §4)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, max(df["std"].max() * 1.05, 0.35), 25)
    for reg in REGIME_ORDER:
        sub = df[df["regime"] == reg]["std"]
        if not sub.empty:
            ax.hist(sub, bins=bins, color=REGIME_PALETTE[reg], alpha=0.72,
                    label=REGIME_LABELS.get(reg, reg), edgecolor="white", linewidth=0.5)
    sns.kdeplot(data=df, x="std", ax=ax, color="#444444",
                linewidth=1.4, linestyle="-", alpha=0.6, bw_adjust=0.8)
    ax.axvline(0.05, color="#e87722", linewidth=1.2, linestyle="--", alpha=0.8,
               label="C threshold (II=0.05)")
    ax.axvline(0.10, color="#d62728", linewidth=1.2, linestyle="--", alpha=0.8,
               label="Severe threshold (II=0.10)")
    ax.set_xlabel(r"Instability Index $\mathrm{II}_i = \sigma_i$", fontsize=11)
    ax.set_ylabel("Number of cases", fontsize=11)
    ax.set_title(
        "Distribution of LLM Instability Index\n"
        r"Bimodal: spike at $\mathrm{II}\approx 0$ (Regime A) + tail (Regime C)",
        fontsize=10)
    ax.legend(fontsize=7.5, framealpha=0.9, edgecolor="#cccccc")
    fig.tight_layout()
    _save(fig, out_dir, "fig_paper_instability_hist", fmt)


def plot_C5_regime_counts(df: pd.DataFrame, out_dir: Path, fmt: List[str]):
    """C5 — Regime distribution bar chart (seaborn-styled, paper §4)."""
    counts  = df["regime"].value_counts()
    labels  = [r for r in REGIME_ORDER if r in counts.index]
    values  = [counts[r] for r in labels]
    xlabels = [REGIME_LABELS.get(r, r).replace(" — ", "\n") for r in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(labels)), values,
                  color=[REGIME_PALETTE[r] for r in labels],
                  alpha=0.82, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
                str(val), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#333333")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(xlabels, fontsize=8.5)
    ax.set_ylabel("Number of cases", fontsize=11)
    ax.set_xlabel("Regime (paper §4 taxonomy)", fontsize=11)
    ax.set_title("Regime Distribution — HypatiaX DeFi Benchmark\n"
                 "A: Symbolic Stability  |  B: Deterministic  |  C: Stochastic Collapse",
                 fontsize=10)
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    _save(fig, out_dir, "fig_paper_regime_counts", fmt)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — EXTRAPOLATION FIGURE  (requires Stage 2 merge)
# ════════════════════════════════════════════════════════════════════════════

def plot_EX_instability_vs_extrap(merged_df: pd.DataFrame,
                                   out_dir: Path, fmt: List[str]):
    """
    EX — Instability Index vs Extrapolation R² scatter.
    Reproduces notebook cell 33 / §16 figure (fig_instability_vs_extrapolation).
    X-axis: II = std(R²) across K runs.
    Y-axis: extrapolation_r2 (Mode A Sympy OOD, or test_r2 proxy).
    Clipped at -15 on y-axis to match paper figure.
    """
    plot_df = merged_df.dropna(subset=["ii", "extrapolation_r2"]).copy()
    if plot_df.empty:
        print("  ⚠️  EX: no cases with both II and extrapolation_r2 — skipped.")
        return
    plot_df["extrapolation_r2"] = plot_df["extrapolation_r2"].clip(lower=-15.0)

    hue_col = "regime" if "regime" in plot_df.columns else "difficulty"
    palette  = {
        "A-Symbolic":   "#27ae60",
        "B-Approx":     "#f39c12",
        "B-Det.Biased": "#f39c12",
        "C-Marginal":   "#e74c3c",
        "C-Collapse":   "#e74c3c",
        "?":            "#aaaaaa",
    }
    markers = {"A-Symbolic": "o", "B-Approx": "s",
               "B-Det.Biased": "D", "C-Marginal": "^", "C-Collapse": "^", "?": "X"}

    fig, ax = plt.subplots(figsize=(11, 7))
    scatter_kw = dict(data=plot_df, x="ii", y="extrapolation_r2",
                      hue=hue_col, style=hue_col,
                      s=120, edgecolor="white", linewidth=0.5, ax=ax)
    if hue_col == "regime":
        scatter_kw["palette"] = palette
        scatter_kw["markers"] = markers
    sns.scatterplot(**scatter_kw)

    y_bot = plot_df["extrapolation_r2"].min() - 0.02
    ax.axvline(0.05, linestyle="--", color="#555", alpha=0.55, linewidth=1.2)
    ax.axvline(0.10, linestyle="--", color="#555", alpha=0.35, linewidth=1.2)
    ax.axhline(0.90, linestyle="--", color="steelblue", alpha=0.45, linewidth=1.2)
    ax.text(0.051, y_bot + 0.01, "II = 0.05", fontsize=8, color="#666")
    ax.text(0.101, y_bot + 0.01, "II = 0.10", fontsize=8, color="#666")
    ax.text(0.002, 0.905,        "R² = 0.90", fontsize=8, color="steelblue")

    # Annotate 4 worst cases
    worst = plot_df.nsmallest(4, "extrapolation_r2")
    for _, row in worst.iterrows():
        ax.annotate(row["case"],
                    xy=(row["ii"], row["extrapolation_r2"]),
                    xytext=(8, 5), textcoords="offset points",
                    fontsize=7, color="#444",
                    arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.8))

    # Eval-mode watermark
    if "eval_mode" in plot_df.columns:
        modes = plot_df["eval_mode"].value_counts().to_dict()
        ax.text(0.01, 0.01,
                "eval_mode — " + "  ".join(f"{k}: {v}" for k, v in modes.items()),
                transform=ax.transAxes, fontsize=7, color="#aaa", va="bottom")

    # Pearson r
    r_val = plot_df[["ii", "extrapolation_r2"]].corr().iloc[0, 1]
    ax.text(0.97, 0.04, f"r = {r_val:.3f}",
            transform=ax.transAxes, ha="right", fontsize=10, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.85))

    ax.set_xlabel("Instability Index  II = σᵢ  (std of R² across runs)", fontsize=12)
    ax.set_ylabel("Extrapolation R²", fontsize=12)

    eval_note = ""
    if "eval_mode" in plot_df.columns and plot_df["eval_mode"].eq("precomputed").any():
        eval_note = "\n(y-axis = test_r² proxy — re-run with --llm-only for true OOD)"

    ax.set_title(
        f"Instability Index vs Extrapolation Performance\n"
        f"HypatiaX DeFi Benchmark  —  {len(plot_df)} cases{eval_note}",
        fontsize=12)
    ax.legend(title=hue_col.capitalize(), fontsize=9, title_fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "fig_instability_vs_extrapolation", fmt)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SUMMARY & CSV EXPORT
# ════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame):
    corr_ii = df[["complexity", "std"]].corr().iloc[0, 1]
    corr_pi = df[["complexity", "p_i"]].corr().iloc[0, 1]
    n_A     = (df["regime"] == "A-Symbolic").sum()
    n_C     = df["regime"].str.startswith("C").sum()
    n_total = len(df)
    print(f"\n── Key statistics ────────────────────────────────────────────────────────")
    print(f"  Cases analysed           : {n_total}")
    print(f"  Regime A (symbolic)      : {n_A}  ({100*n_A/n_total:.1f}%)")
    print(f"  Regime C (collapse)      : {n_C}  ({100*n_C/n_total:.1f}%)")
    print(f"  Pearson r (K vs II)      : {corr_ii:.4f}  [paper §4 — complexity drives instability]")
    print(f"  Pearson r (K vs p_i)     : {corr_pi:.4f}  [paper §4.5 — complexity degrades p_i]")
    mean_A = df[df["regime"] == "A-Symbolic"]["std"].mean()
    mean_C = df[df["regime"].str.startswith("C")]["std"].mean()
    print(f"  Mean II (Regime A)       : {mean_A:.4f}")
    print(f"  Mean II (Regime C)       : {mean_C:.4f}")
    print(f"── II definition ─────────────────────────────────────────────────────────")
    print(f"  II_i := σ_i = std(R²_i)  across N independent LLM runs")
    print(f"  II=0    → deterministic (Regime A/B)")
    print(f"  II≥0.05 → marginal stochastic instability (Regime C-Marginal)")
    print(f"  II≥0.10 → severe collapse (Regime C-Collapse)")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 — CLI
# ════════════════════════════════════════════════════════════════════════════

def _resolve_figures(args) -> List[str]:
    """Determine which figure codes to generate from --figures / --group / default."""
    if getattr(args, "figures", None):
        selected = list(dict.fromkeys(args.figures))
    elif getattr(args, "group", None):
        selected = []
        for g in args.group:
            for code in GROUP_MAP.get(g, []):
                if code not in selected:
                    selected.append(code)
    else:
        selected = list(ALL_FIGURES)

    # EX requires Stage 2 data
    has_benchmark = bool(getattr(args, "benchmark_json", None))
    no_extrap_plot = getattr(args, "no_extrap_plot", False)
    if "EX" in selected and (not has_benchmark or no_extrap_plot):
        selected.remove("EX")
        if not has_benchmark:
            print("  ℹ️  EX skipped — provide --benchmark-json to enable extrap figure.")

    # B6 requires 3D support
    if "B6" in selected and not _3D_OK:
        selected.remove("B6")
        print("  ℹ️  B6 skipped — mpl_toolkits.mplot3d not available.")

    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HypatiaX — consolidated instability analysis & figure pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Data source ───────────────────────────────────────────────────────────
    src = parser.add_argument_group("Stage 1 — data source")
    src.add_argument(
        "--source", choices=["auto", "variance", "multi", "single"],
        default="auto",
        help="JSON source (default: auto-detect).",
    )
    src.add_argument(
        "--results-dir", type=Path, default=_RESULTS_DIR_DEFAULT, metavar="PATH",
        help=f"Results directory (default: {_RESULTS_DIR_DEFAULT}).",
    )
    src.add_argument(
        "--cases", nargs="+", metavar="SUBSTR",
        help="Filter cases by substring (case-insensitive).",
    )

    # ── Stage 2: extrapolation merge ──────────────────────────────────────────
    ext = parser.add_argument_group("Stage 2 — extrapolation merge (optional)")
    ext.add_argument(
        "--benchmark-json", type=Path, default=None, metavar="PATH",
        help=(
            "Path to benchmark JSON (hypatiax_defi_benchmark_v3c2_results.json). "
            "Enables Stage 2: extrapolation R² computation + EX figure. "
            "Without this, only instability figures (A–C groups) are produced."
        ),
    )
    ext.add_argument(
        "--extrap-csv-out", type=Path, default=None, metavar="PATH",
        help="Path for instability_extrapolation.csv (default: <out>/instability_extrapolation.csv).",
    )
    ext.add_argument(
        "--no-extrap-plot", action="store_true",
        help="Run Stage 2 (merge) but skip the EX figure.",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    out = parser.add_argument_group("output")
    out.add_argument(
        "--out", type=Path, default=_FIGURES_DIR_DEFAULT, metavar="PATH",
        help=f"Output directory for figures and CSVs (default: {_FIGURES_DIR_DEFAULT}).",
    )
    out.add_argument(
        "--csv-out", type=Path, default=None, metavar="PATH",
        help="Path for instability_analysis.csv (default: <out>/instability_analysis.csv).",
    )
    out.add_argument(
        "--format", nargs="+", choices=["png", "pdf", "svg"],
        default=["png", "pdf"], metavar="FMT",
        help="Output formats: png pdf svg (default: png pdf).",
    )

    # ── Figure selection ──────────────────────────────────────────────────────
    sel = parser.add_argument_group("figure selection (default: all available)")
    sel.add_argument(
        "--figures", nargs="+", metavar="CODE",
        choices=ALL_FIGURES,
        help=(
            f"Specific figure codes: {' '.join(ALL_FIGURES)}. "
            "EX requires --benchmark-json."
        ),
    )
    sel.add_argument(
        "--group", nargs="+", metavar="GRP",
        choices=list(GROUP_MAP),
        help="Generate an entire group: A B C EX (can combine).",
    )

    # ── Per-figure options ────────────────────────────────────────────────────
    opt = parser.add_argument_group("figure options")
    opt.add_argument(
        "--no-regline", action="store_true",
        help="Omit OLS regression line on C1 (complexity vs II).",
    )
    opt.add_argument(
        "--no-scatter", action="store_true",
        help="Skip A3 (mean R² vs II scatter).",
    )
    opt.add_argument(
        "--elev", type=float, default=28.0, metavar="DEG",
        help="B6 surface elevation angle (default: 28).",
    )
    opt.add_argument(
        "--azim", type=float, default=225.0, metavar="DEG",
        help="B6 surface azimuth angle (default: 225).",
    )

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    csv_path = args.csv_out or (args.out / "instability_analysis.csv")

    print("\n" + "=" * 65)
    print("  HypatiaX — Instability Analysis & Figure Suite")
    print("=" * 65)

    # ── Stage 1: load data & build DataFrame ──────────────────────────────
    print("\n📥 Stage 1 — Loading instability data ...")
    data = load_data(args.source, args.results_dir)

    if args.cases:
        filters = [c.lower() for c in args.cases]
        data = {n: v for n, v in data.items()
                if any(f in n.lower() for f in filters)}
        if not data:
            print(f"❌ No cases matched filters: {args.cases}")
            sys.exit(1)
        print(f"  Case filter active: {len(data)} case(s)")

    print("[DEBUG] about to build dataframe")
    print(f"[DEBUG] data keys: {list(data.keys())[:10]}")
    try:
        df = build_dataframe(data)
        print(f"[DEBUG] dataframe built: {len(df)} rows, columns={list(df.columns)}")
    except Exception as e:
        print(f"[DEBUG] dataframe build failed: {e}")
        raise
    rows = df_to_rows(df)

    print(f"[DEBUG] csv_path = {csv_path}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(csv_path, index=False)
        print(f"[DEBUG] csv written, exists={csv_path.exists()}")
        if not csv_path.exists():
            raise RuntimeError(f"CSV was not created at {csv_path}")
    except Exception as e:
        print(f"[DEBUG] csv write failed: {e}")
        raise
    print(f"  ✅ instability_analysis.csv → {csv_path}  ({len(df)} rows)")
    print_summary(df)

    # ── Stage 2: extrapolation merge ──────────────────────────────────────
    merged_df: Optional[pd.DataFrame] = None
    if args.benchmark_json:
        print(f"\n🔗 Stage 2 — Extrapolation merge ({args.benchmark_json}) ...")
        with open(args.benchmark_json) as fh:
            bench_data = json.load(fh)
        extrap_df = build_extrapolation_df(bench_data)
        mode_counts = extrap_df["eval_mode"].value_counts().to_dict()
        print(f"  Eval modes : {mode_counts}")
        merged_df = merge_instability_extrap(df, extrap_df)
        extrap_csv = args.extrap_csv_out or (args.out / "instability_extrapolation.csv")
        merged_df.to_csv(extrap_csv, index=False)
        print(f"  ✅ instability_extrapolation.csv → {extrap_csv}  ({len(merged_df)} rows)")
        if mode_counts.get("precomputed", 0) > 0:
            print("  ⚠️  NOTE: extrapolation_r2 = test_r² proxy for some cases.")
            print("          Re-run benchmark with --llm-only for true OOD Mode A.")

    # ── Stage 3: figures ─────────────────────────────────────────────────
    to_run = _resolve_figures(args)
    if args.no_scatter and "A3" in to_run:
        to_run.remove("A3")
    fmt = args.format

    print(f"\n📊 Stage 3 — Generating {len(to_run)} figure(s): {' '.join(to_run)}")

    DISPATCH = {
        "A1": lambda: plot_A1_per_case(rows, args.out, fmt),
        "A2": lambda: plot_A2_ii_histogram(rows, args.out, fmt),
        "A3": lambda: plot_A3_scatter(rows, args.out, fmt),
        "B1": lambda: plot_B1_3d(rows, args.out, fmt),
        "B2": lambda: plot_B2_phase(rows, args.out, fmt),
        "B3": lambda: plot_B3_hist(rows, args.out, fmt),
        "B4": lambda: plot_B4_success(rows, args.out, fmt),
        "B5": lambda: plot_B5_regimes(rows, args.out, fmt),
        "B6": lambda: plot_B6_surface(rows, args.out, fmt,
                                       elev=args.elev, azim=args.azim),
        "C1": lambda: plot_C1_complexity_ii(df, args.out, fmt,
                                             show_regline=not args.no_regline),
        "C2": lambda: plot_C2_complexity_pi(df, args.out, fmt),
        "C3": lambda: plot_C3_mean_ii(df, args.out, fmt),
        "C4": lambda: plot_C4_inst_hist(df, args.out, fmt),
        "C5": lambda: plot_C5_regime_counts(df, args.out, fmt),
        "EX": lambda: plot_EX_instability_vs_extrap(merged_df, args.out, fmt),
    }

    errors: List[str] = []
    for code in to_run:
        label = FIGURE_LABELS.get(code, code)
        print(f"\n  [{code}] {label} ...")
        try:
            DISPATCH[code]()
        except Exception as exc:
            import traceback
            print(f"  ❌ {code} failed: {exc}")
            traceback.print_exc()
            errors.append(code)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    if errors:
        print(f"⚠️  {len(errors)} figure(s) failed: {' '.join(errors)}")
    else:
        print(f"✅ All {len(to_run)} figure(s) saved to: {args.out}/")
    print(f"\n  Output files:")
    print(f"    instability_analysis.csv  → {csv_path}")
    if merged_df is not None:
        extrap_csv = args.extrap_csv_out or (args.out / "instability_extrapolation.csv")
        print(f"    instability_extrapolation.csv → {extrap_csv}")
    stems = [
        ("A1", "hypatiax_instability_per_case"),
        ("A2", "hypatiax_instability_histogram"),
        ("A3", "hypatiax_instability_scatter"),
        ("B5", "fig_instability_regimes"),
        ("C4", "fig_paper_instability_hist"),
        ("C5", "fig_paper_regime_counts"),
        ("C1", "fig_paper_complexity_vs_instability"),
        ("EX", "fig_instability_vs_extrapolation"),
    ]
    for code, stem in stems:
        if code in to_run and code not in errors:
            for f in fmt:
                print(f"    {stem}.{f}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

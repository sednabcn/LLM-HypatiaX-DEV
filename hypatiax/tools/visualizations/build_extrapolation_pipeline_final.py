#!/usr/bin/env python3
"""
build_extrapolation_pipeline_final.py
=====================================

HypatiaX DeFi — Extrapolation Evaluation Pipeline  (final, paper-ready)

Inputs
------
  --input              Path to benchmark JSON  (hypatiax_defi_benchmark_v3c2_results.json)
  --instability-csv    Path to instability_analysis.csv  (ii, regime, p_i, mean_r2)
  --output             Output CSV  (default: instability_extrapolation.csv)
  --plot               Output PNG  (default: fig_instability_vs_extrapolation.
  --n-samples          OOD samples per case in Mode A  (default: 200)
  --no-plot            Skip figure generation

Evaluation modes (auto-selected per case)
-----------------------------------------
  A — Sympy
      case has "formula" + "predicted_formula" + "var_ranges"
      → OOD data via generate_data(..., extrapolation=True)
      → R² via evaluate_model(func_true, func_pred, data)

  B — Stub
      case has pre-stored "y_pred" + "y_true"
      → R² = sklearn r2_score(y_true, y_pred)

  C — Precomputed
      fallback: use test_r2 already in the JSON as proxy
      → eval_mode = "precomputed"
      → NOTE: not true OOD — re-run benchmark with formula logging
        (hypatiax_defi_benchmark_v3c2.py --llm-only) to get Mode A.

Instability axis (x)
--------------------
  Always taken from instability_analysis.csv:
    ii  = std(R²) across 30 independent LLM runs  [paper Def. 4.1]
  NOT from the single-run JSON field "instability" = test_r2 - train_r2.

Output CSV columns
------------------
  case, regime, difficulty, formula_type, complexity,
  mean_r2, std_r2, ii, p_i, n_valid, n_runs,
  extrapolation_r2, success, failure, eval_mode

Usage
-----
  # Minimal (uses precomputed mode - no formula logging yet)
  python build_extrapolation_pipeline_final.py \
      --input  hypatiax_defi_benchmark_v3c2_results.json \
      --instability-csv instability_analysis.csv

  # Full Mode A (after re-running benchmark with formula logging)
  python build_extrapolation_pipeline_final.py \
      --input  hypatiax_defi_benchmark_v3c2_results.json \
      --instability-csv instability_analysis.csv \
      --n-samples 300
"""

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sympy as sp
from sklearn.metrics import r2_score

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False

try:
    from evaluate_extrapolation import evaluate_model as _eval_model
    from extrapolation_generator import generate_data as _gen_data
    _LOCAL_MODULES = True
except ImportError:
    _LOCAL_MODULES = False


# =============================================================================
# Utilities
# =============================================================================

def iterate_cases(data) -> List[Tuple[str, Dict]]:
    if isinstance(data, dict):
        cases = data.get("cases", data)
        if isinstance(cases, list):
            return [
                (c.get("test_case", c.get("case", "Unnamed")), c)
                for c in cases if isinstance(c, dict)
            ]
        return list(cases.items())
    if isinstance(data, list):
        return [
            (c.get("test_case", c.get("case", "Unnamed")), c)
            for c in data if isinstance(c, dict)
        ]
    raise ValueError("Unsupported JSON root type - expected dict or list.")


def safe_float(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


# =============================================================================
# Symbol mapping + sympy callable
# =============================================================================

def _resolve_symbol_map(
    free_syms: set,
    var_names: List[str],
) -> Dict[str, str]:
    """
    Build a bijection from formula free-symbol names to var_ranges keys.

    Priority passes:
      1. Exact match
      2. Suffix
      3. Prefix
      4. Tail segment
      5. Digit-in-var
      6. Compound prefix
      7. Finance alias (rho→corr, sigma→vol, mu→ret, T→maturity, etc.)
      8. Case-insensitive substring
      9. Single remaining
     10. Positional fallback
    """
    remaining_syms = [str(s) for s in free_syms]
    remaining_vars = list(var_names)
    mapping: Dict[str, str] = {}

    def pick(sym: str, candidates: List[str]) -> None:
        best = min(candidates, key=len)
        mapping[sym] = best
        remaining_syms.remove(sym)
        if best in remaining_vars:
            remaining_vars.remove(best)

    # 1. Exact
    for sym in list(remaining_syms):
        if sym in remaining_vars:
            pick(sym, [sym])

    # 2. Suffix
    for sym in list(remaining_syms):
        c = [v for v in remaining_vars if v.endswith("_" + sym) or v.endswith(sym)]
        if c:
            pick(sym, c)

    # 3. Prefix
    for sym in list(remaining_syms):
        c = [v for v in remaining_vars if v.startswith(sym + "_") or v.startswith(sym)]
        if c:
            pick(sym, c)

    # 4. Tail segment
    for sym in list(remaining_syms):
        tail = sym.split("_")[-1]
        if len(tail) >= 3:
            c = [v for v in remaining_vars if v.endswith("_" + tail) or v.endswith(tail)]
            if len(c) == 1:
                pick(sym, c)

    # 5. Digit-in-var
    for sym in list(remaining_syms):
        if sym and sym[-1].isdigit():
            digit = sym[-1]
            c = [v for v in remaining_vars if digit in v]
            if len(c) == 1:
                pick(sym, c)

    # 6. Compound prefix
    for sym in list(remaining_syms):
        if "_" in sym:
            root, sfx = sym.split("_", 1)
            if len(root) >= 4:
                c = [v for v in remaining_vars
                     if v.startswith(root) and v.endswith("_" + sfx)]
                if len(c) == 1:
                    pick(sym, c)

    # 7. Finance aliases
    _ALIASES: Dict[str, List[str]] = {
        "rho":   ["corr"],
        "\u03c1": ["corr"],
        "sigma": ["vol", "implied"],
        "mu":    ["ret", "return"],
        "vol":   ["volatility"],
        "T":     ["time_to", "expir", "matur", "tenor", "time"],
        "n":     ["num_period", "compound", "n_period", "periods", "num"],
        "r":     ["risk_free", "rate"],
    }
    for sym in list(remaining_syms):
        keywords = _ALIASES.get(sym, [])
        for keyword in keywords:
            c = [v for v in remaining_vars if keyword in v.lower()]
            if len(c) == 1:
                pick(sym, c)
                break

    # 8. Case-insensitive substring
    for sym in list(remaining_syms):
        sym_lower = sym.lower()
        single_upper = (len(sym) == 1 and sym.isupper())
        if len(sym_lower) >= 2 or single_upper:
            c = [v for v in remaining_vars if sym_lower in v.lower()]
            if len(c) == 1:
                pick(sym, c)

    # 9. Single remaining
    if len(remaining_syms) == 1 and remaining_vars:
        mapping[remaining_syms[0]] = remaining_vars[0]
        remaining_syms.clear()

    # 10. Positional fallback
    if remaining_syms and len(remaining_syms) == len(remaining_vars):
        for sym, var in zip(sorted(remaining_syms), remaining_vars):
            mapping[sym] = var

    return mapping


def _preprocess_formula(expr_str: str) -> str:
    import re
    expr_str = re.sub(r"N'\s*\(([^)]+)\)", r"norm_pdf(\1)", expr_str)
    expr_str = re.sub(r"\breturn\b", "r_return", expr_str)
    expr_str = re.sub(r"\bmax\s*\(", "Max(", expr_str)
    expr_str = re.sub(r"\bmin\s*\(", "Min(", expr_str)
    return expr_str


def _make_sympy_callable(
    expr_str: str,
    var_names: List[str],
    constants: Optional[Dict[str, float]] = None,
) -> Callable:
    import re as _re

    from scipy.stats import norm as _scipy_norm

    expr_str = _preprocess_formula(expr_str)

    _N_sym        = sp.Function("N")
    _norm_pdf_sym = sp.Function("norm_pdf")
    sympify_locals = {
        "N":        _N_sym,
        "norm_pdf": _norm_pdf_sym,
        "True":  sp.Integer(1),
        "False": sp.Integer(0),
        "S": sp.Symbol("S"),
        "K": sp.Symbol("K"),
        "E": sp.Symbol("E"),
        "I": sp.Symbol("I"),
    }

    expr = sp.sympify(expr_str, locals=sympify_locals)

    free_syms   = expr.free_symbols
    sym_to_var  = _resolve_symbol_map(free_syms, var_names)
    unresolved  = {str(s) for s in free_syms} - set(sym_to_var)

    _DEFI_CONSTANT_DEFAULTS: Dict[str, float] = {
        "K":  100.0,
        "r":  0.05,
        "n":  12.0,
        "q":  0.0,
    }
    if unresolved and constants is not None:
        merged_constants = {**_DEFI_CONSTANT_DEFAULTS, **constants}
    else:
        merged_constants = _DEFI_CONSTANT_DEFAULTS

    if unresolved:
        const_subs = {}
        still_unresolved = set()
        for sym_str in unresolved:
            if sym_str in merged_constants:
                const_subs[sp.Symbol(sym_str)] = sp.Float(merged_constants[sym_str])
            else:
                still_unresolved.add(sym_str)
        if const_subs:
            expr = expr.subs(const_subs)
            free_syms  = expr.free_symbols
            sym_to_var = _resolve_symbol_map(free_syms, var_names)
            unresolved = {str(s) for s in free_syms} - set(sym_to_var)

    if unresolved:
        raise ValueError(
            f"Unresolved formula symbols after mapping: {unresolved}. "
            f"var_names={var_names}"
        )

    subs = {sp.Symbol(s): sp.Symbol(v)
            for s, v in sym_to_var.items() if s != v}
    expr_renamed = expr.subs(subs)

    var_symbols = sp.symbols(var_names)
    f = sp.lambdify(
        var_symbols, expr_renamed,
        modules=[
            {
                "norm_pdf": _scipy_norm.pdf,
                "norm_cdf": _scipy_norm.cdf,
                "N":        _scipy_norm.cdf,
            },
            "numpy",
        ],
    )

    def caller(**kwargs):
        args = [kwargs[v] for v in var_names]
        return f(*args)

    return caller


# =============================================================================
# Mode A / B / C evaluation
# =============================================================================

def _evaluate_mode_a(case_data: Dict[str, Any], n_samples: int) -> float:
    if not _LOCAL_MODULES:
        print("  [Mode A] extrapolation_generator / evaluate_extrapolation not found - skipping.")
        return np.nan

    formula      = case_data.get("formula")
    pred_formula = case_data.get("predicted_formula")
    raw_ranges   = case_data.get("var_ranges")

    if not formula or not pred_formula or not raw_ranges:
        missing = [k for k, v in [("formula", formula),
                                   ("predicted_formula", pred_formula),
                                   ("var_ranges", raw_ranges)] if not v]
        print(f"  [Mode A] Missing fields: {missing} - skipping.")
        return np.nan

    var_ranges = {k: tuple(v) for k, v in raw_ranges.items()}
    var_names  = list(var_ranges.keys())

    metadata  = case_data.get("metadata") or {}
    constants = metadata.get("constants") or {}

    try:
        func_true = _make_sympy_callable(formula, var_names, constants=constants)
    except Exception as exc:
        print(f"  [Mode A] ground-truth sympy parse error: {exc}")
        return np.nan

    _globals: Dict = {"np": np, "numpy": np}
    _locals:  Dict = {}
    try:
        exec(pred_formula, _globals, _locals)  # noqa: S102
        func_pred_raw = next((v for v in _locals.values() if callable(v)), None)
        if func_pred_raw is None:
            raise ValueError("No callable found in predicted_formula code block.")

        def func_pred(**kwargs):
            args = [kwargs[v] for v in var_names]
            return func_pred_raw(*args)

    except Exception as exc:
        print(f"  [Mode A] predicted_formula exec error: {exc}")
        return np.nan

    data = _gen_data(func_true, var_ranges, n=n_samples, extrapolation=True)
    if not data:
        print("  [Mode A] generate_data returned no valid samples.")
        return np.nan

    train_ys = []
    for _ in range(50):
        s = {v: np.random.uniform(lo, hi) for v, (lo, hi) in var_ranges.items()}
        try:
            y = func_true(**s)
            if np.isfinite(y):
                train_ys.append(float(y))
        except Exception:
            pass
    if train_ys:
        y_scale = max(np.std(train_ys) * 100, 1e6)
        data = [d for d in data if np.isfinite(d.get("y", np.nan))
                and abs(d["y"]) < y_scale]

    if not data:
        print("  [Mode A] All OOD samples filtered (singularity in extrap region).")
        return np.nan

    return float(_eval_model(func_true, func_pred, data))


def _evaluate_mode_b(case_data: Dict[str, Any]) -> float:
    if "y_pred" not in case_data or "y_true" not in case_data:
        return np.nan
    y_true = np.asarray(case_data["y_true"], dtype=float)
    y_pred = np.asarray(case_data["y_pred"], dtype=float)
    if len(y_true) < 5:
        return np.nan
    return float(r2_score(y_true, y_pred))


def _evaluate_mode_c(case_data: Dict[str, Any]) -> float:
    res = (case_data.get("results") or {})
    llm_res = res.get("pure_llm") or res.get("llm_only") or res.get("llm") or {}
    return safe_float(llm_res.get("test_r2"))


def evaluate_case(
    case_data: Dict[str, Any],
    n_samples: int,
) -> Tuple[float, str]:
    if (case_data.get("formula")
            and case_data.get("predicted_formula")
            and case_data.get("var_ranges")):
        return _evaluate_mode_a(case_data, n_samples), "sympy"

    if "y_pred" in case_data and "y_true" in case_data:
        return _evaluate_mode_b(case_data), "stub"

    return _evaluate_mode_c(case_data), "precomputed"


# =============================================================================
# Core pipeline
# =============================================================================

def build_extrapolation_df(data, n_samples: int) -> pd.DataFrame:
    rows = []
    for name, case in iterate_cases(data):
        if not isinstance(case, dict):
            continue

        results = case.get("results") or {}
        res = (results.get("pure_llm")
               or results.get("llm_only")
               or results.get("llm")
               or {})

        r2_ext, mode = evaluate_case(case, n_samples)

        rows.append({
            "case":             name,
            "difficulty":       case.get("difficulty"),
            "formula_type":     case.get("formula_type"),
            "train_r2":         safe_float(res.get("train_r2")),
            "test_r2":          safe_float(res.get("test_r2")),
            "extrapolation_r2": r2_ext,
            "success":          bool(res.get("success", False)),
            "failure":          res.get("failure") or "",
            "time_s":           safe_float(res.get("time_s")),
            "eval_mode":        mode,
        })

    return pd.DataFrame(rows)


def merge_with_instability(
    extrap_df: pd.DataFrame,
    ia_path: Optional[str],
) -> pd.DataFrame:
    if ia_path is None or not Path(ia_path).exists():
        print("  Warning: instability_analysis.csv not provided / not found.")
        print("           Returning extrapolation-only DataFrame (no ii/regime).")
        return extrap_df

    ia = pd.read_csv(ia_path)
    ia = ia.rename(columns={"mean": "mean_r2", "std": "std_r2"})

    extrap_dedup = extrap_df.drop_duplicates(subset="case", keep="first")

    merged = ia.merge(
        extrap_dedup[[
            "case", "difficulty", "formula_type",
            "test_r2", "extrapolation_r2",
            "success", "failure", "eval_mode",
        ]],
        on="case",
        how="left",
    )

    preferred_cols = [
        "case", "regime", "difficulty", "formula_type", "complexity",
        "mean_r2", "std_r2", "ii", "p_i", "n_valid", "n_runs",
        "extrapolation_r2", "success", "failure", "eval_mode",
    ]
    cols = [c for c in preferred_cols if c in merged.columns]
    merged = (merged[cols]
              .sort_values(["regime", "ii"], ascending=[True, True])
              .reset_index(drop=True))

    return merged


# =============================================================================
# Plot
# =============================================================================

def plot_instability_vs_extrapolation(df: pd.DataFrame, output_path: str) -> None:
    if not _PLOT_AVAILABLE:
        print("  Warning: matplotlib/seaborn not available - skipping plot.")
        return

    CLIP_LO = -15.0

    plot_df = df.dropna(subset=["ii", "extrapolation_r2"]).copy()

    if plot_df.empty:
        print("  Warning: no rows with both ii and extrapolation_r2 - skipping plot.")
        return

    plot_df["extrap_clipped"] = plot_df["extrapolation_r2"].clip(lower=CLIP_LO)
    n_clipped = int((plot_df["extrapolation_r2"] < CLIP_LO).sum())

    sns.set(style="whitegrid", context="talk")

    palette = {
        "A-Symbolic":   "#2ecc71",
        "B-Approx":     "#3498db",
        "B-Det.Biased": "#f39c12",
        "C-Collapse":   "#e74c3c",
    }

    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(
        data=plot_df,
        x="ii",
        y="extrap_clipped",
        hue="regime",
        palette=palette,
        s=60,
        edgecolor="k",
        alpha=0.9,
    )

    from scipy.stats import pearsonr
    valid = plot_df[["ii", "extrap_clipped"]].dropna()
    if len(valid) >= 3:
        r, p = pearsonr(valid["ii"], valid["extrap_clipped"])
        ax.text(
            0.02, 0.98,
            f"Pearson r = {r:.3f} (p = {p:.3f})\n"
            f"n = {len(valid)} cases",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel("Instability Index II (std of R² across 30 runs)")
    ax.set_ylabel("Extrapolation $R^2$ (clipped at -15)")
    ax.set_title("Instability vs Extrapolation Performance (HypatiaX DeFi)")

    # Label only interesting points
    for _, row in plot_df.iterrows():
        label = row["case"]
        cond = (
            row["regime"] == "C-Collapse"
            or abs(row["extrap_clipped"]) > 1.0
            or row["ii"] > 0.05
        )
        if cond:
            ax.text(
                row["ii"], row["extrap_clipped"],
                f" {label}",
                fontsize=8,
                ha="left", va="center",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Plot saved : {output_path}  ({len(valid)} cases plotted, {n_clipped} clipped to {CLIP_LO})")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HypatiaX DeFi — Extrapolation Evaluation Pipeline (paper-ready)."
    )
    parser.add_argument("--input", required=True,
                        help="Path to hypatiax_defi_benchmark_v3c2_results.json")
    parser.add_argument("--instability-csv", required=False, default=None,
                        help="Path to instability_analysis.csv")
    parser.add_argument("--output", default="instability_extrapolation.csv",
                        help="Output CSV path.")
    parser.add_argument("--plot", default="fig_instability_vs_extrapolation.png",
                        help="Output PNG path for scatter plot.")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="OOD samples per case in Mode A (default: 200).")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip figure generation.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input JSON not found: {input_path}")

    print(f"Loading benchmark JSON : {input_path}")
    with open(input_path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        print(f"  benchmark : {data.get('benchmark', 'Unknown')}")
        print(f"  version   : {data.get('version', 'Unknown')}")
        print(f"  saved_at  : {data.get('saved_at', 'Unknown')}")
        print(f"  total     : {data.get('total_cases', len(data.get('cases', [])))} cases")

    print(f"\nEvaluating extrapolation R²  (n_samples={args.n_samples} for Mode A) ...")
    extrap_df = build_extrapolation_df(data, args.n_samples)
    modes = extrap_df["eval_mode"].value_counts().to_dict()
    n_nan = int(extrap_df["extrapolation_r2"].isna().sum())
    print(f"  Eval modes : {modes}")
    print(f"  NaN R²     : {n_nan}")

    print("\nMerging with instability_analysis.csv ...")
    merged = merge_with_instability(extrap_df, args.instability_csv)

    merged.to_csv(args.output, index=False)
    print(f"\nPipeline complete")
    print(f"CSV saved  : {args.output}  ({merged.shape[0]} rows x {merged.shape[1]} cols)")

    if "regime" in merged.columns:
        by_regime = merged.groupby("regime").agg(
            mean_r2=("mean_r2", ["count", "mean", "min"]),
            ii=("ii", ["count", "mean", "min"]),
            extrapolation_r2=("extrapolation_r2", ["count", "mean", "min"]),
            p_i=("p_i", ["count", "mean", "min"]),
        )
        print("\nBy regime:")
        print(by_regime)

        hi = merged[(merged["ii"] > 0.05)]
        if not hi.empty:
            print("\nHigh-instability cases  (ii > 0.05):")
            print(hi[["case", "regime", "difficulty", "ii", "extrapolation_r2", "success", "failure"]])

    if not args.no-plot:
        print("\nGenerating plot ...")
        plot_instability_vs_extrapolation(merged, args.plot)

    print("\nNOTE: extrapolation_r2 may be test_r2 (precomputed proxy) for Mode C cases.")
    print("For true OOD R², re-run the benchmark with formula logging:")
    print("  python hypatiax_defi_benchmark_v3c2.py --llm-only")
    print("Then re-run this pipeline - Mode A will activate automatically.")


if __name__ == "__main__":
    main()

"""
How to run (commentary)
-----------------------

# Minimal (current v3c2 JSON, mostly Mode C)
python build_extrapolation_pipeline_final.py \
    --input hypatiax_defi_benchmark_v3c2_results.json \
    --instability-csv instability_analysis.csv

# Full Mode A (after re-running benchmark with formula logging)
python hypatiax_defi_benchmark_v3c2.py --llm-only

python build_extrapolation_pipeline_final.py \
    --input hypatiax_defi_benchmark_v3c2_results.json \
    --instability-csv instability_analysis.csv \
    --n-samples 300

# Skip plot generation
python build_extrapolation_pipeline_final.py \
    --input hypatiax_defi_benchmark_v3c2_results.json \
    --instability-csv instability_analysis.csv \
    --no-plot
"""

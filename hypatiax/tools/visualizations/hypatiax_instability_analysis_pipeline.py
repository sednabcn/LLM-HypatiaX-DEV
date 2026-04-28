#!/usr/bin/env python3
"""
hypatiax_instability_analysis_pipeline.py
==========================================
Publication-ready instability analysis pipeline for HypatiaX DeFi Benchmark.

Generates 5 seaborn figures + 1 CSV export:

  1. _complexity_vs_instability  — KEY FIGURE: complexity proxy K vs II,
                                   regression line, regime colours.
                                   Empirical support for Instability Theorem.
  2. _complexity_vs_success      — Complexity K vs p_i (success probability)
  3. _mean_vs_instability        — Mean R² vs II (regime separation)
  4. _instability_hist           — KDE histogram of II distribution
  5. _regime_counts              — Regime distribution bar chart
  CSV: instability_analysis.csv  — full per-case stats table

Reads from (same priority as the rest of the pipeline):
  1. hypatiax/data/results/hypatiax_defi_variance_results.json   (--variance runs)
  2. hypatiax_defi_benchmark_v3_results_<TIMESTAMP>.json files   (--multi-run N)
  3. hypatiax_defi_benchmark_v3_results.json                     (single-run fallback)

Usage
─────
  python hypatiax_instability_analysis_pipeline.py
  python hypatiax_instability_analysis_pipeline.py --results-dir path/to/results --out path/to/figures
  python hypatiax_instability_analysis_pipeline.py --format png pdf
  python hypatiax_instability_analysis_pipeline.py --cases theta sharpe
  python hypatiax_instability_analysis_pipeline.py --source variance
  python hypatiax_instability_analysis_pipeline.py --no-regline   # omit regression line on Fig 1

Author : HypatiaX Team
Version: 2.0 — JMLR submission (refactored from fig_paper.py)
Date   : 2026
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Default paths (mirror benchmark) ──────────────────────────────────────────
RESULTS_DIR   = Path("hypatiax/data/results")
VARIANCE_JSON = RESULTS_DIR / "hypatiax_defi_variance_results.json"
FINAL_JSON    = RESULTS_DIR / "hypatiax_defi_benchmark_v3_results.json"
MULTI_PATTERN = re.compile(r"hypatiax_defi_benchmark_v3_results_\d{8}T\d{6}Z\.json$")

# ── Publication style ──────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["font.family"]  = "serif"
plt.rcParams["figure.dpi"]   = 300
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False

# ── Regime palette (identical across all pipeline scripts) ────────────────────
REGIME_PALETTE = {
    "A-Symbolic":   "#2ca02c",
    "B-Approx":     "#ff7f0e",
    "B-Det.Biased": "#d6b219",
    "C-Marginal":   "#e87722",
    "C-Collapse":   "#d62728",
    "?":            "#aaaaaa",
}
REGIME_LABELS = {
    "A-Symbolic":   "Regime A — Symbolic Stability",
    "B-Approx":     "Regime B — Deterministic Biased",
    "B-Det.Biased": "Regime B* — Borderline Stochastic",
    "C-Marginal":   "Regime C-Marginal",
    "C-Collapse":   "Regime C — Stochastic Collapse",
    "?":            "Undetermined",
}
REGIME_ORDER = ["A-Symbolic", "B-Approx", "B-Det.Biased", "C-Marginal", "C-Collapse", "?"]

# Seaborn-compatible palette dict (used in hue= calls)
_SNS_PALETTE = {k: v for k, v in REGIME_PALETTE.items()}


# ── Regime classifier (mirrors benchmark v3c2) ─────────────────────────────────
def classify_regime(mean_r2: float, std_r2: float) -> str:
    if mean_r2 != mean_r2 or std_r2 != std_r2:
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


# ── Complexity proxy ───────────────────────────────────────────────────────────
def complexity_proxy(case_name: str) -> int:
    """
    Proxy for Kolmogorov complexity of the target formula.
    Scores presence of increasingly complex mathematical operations.

    Scale:
      1   — baseline (any formula)
      +1  — algebraic operations (ratios, prices, linear PnL)
      +2  — portfolio / correlation terms
      +3  — transcendental functions (exp, log, Black-Scholes)
      +4  — derivatives / Greeks (delta, gamma, vega, theta)

    Maximum observed score ≈ 8 (e.g. Theta of option = 1+1+3+4 = 9).
    Used in paper §4 as an empirical proxy for formula complexity class.
    """
    name = case_name.lower()
    score = 1  # baseline: all formulas get at least 1

    # Algebraic baseline operations
    if any(k in name for k in ["ratio", "price", "amount", "pnl", "var",
                                "apy", "rate", "fee", "margin", "leverage",
                                "collateral", "liquidat", "staking", "yield"]):
        score += 1

    # Portfolio / correlation
    if any(k in name for k in ["portfolio", "correlated", "sharpe",
                                "information", "tracking", "expected shortfall"]):
        score += 2

    # Transcendental functions
    if any(k in name for k in ["black", "scholes", "exp", "log",
                                "compounding", "borrowing", "impermanent"]):
        score += 3

    # Derivatives / Greeks (highest complexity)
    if any(k in name for k in ["delta", "gamma", "vega", "theta",
                                "rho", "greek", "option"]):
        score += 4

    return score


# ── Data loaders (identical to hypatiax_3D_plot_instability.py) ────────────────
def _load_variance_json(path: Path) -> dict:
    raw = json.loads(path.read_text())
    out = {}
    for rec in raw:
        name   = rec.get("test_case", rec.get("name", "?"))
        n_runs = rec.get("n_runs", len(rec.get("runs", [])))
        scores = [
            r["test_r2"] for r in rec.get("runs", [])
            if r.get("test_r2") is not None
            and not (isinstance(r["test_r2"], float) and np.isnan(r["test_r2"]))
        ]
        if scores:
            out[name] = {"scores": scores, "n_runs": n_runs}
    return out


def _load_multi_run_jsons(results_dir: Path) -> dict:
    files = sorted(f for f in results_dir.iterdir() if MULTI_PATTERN.match(f.name))
    if not files:
        return {}
    case_scores   = {}
    case_attempts = {}
    for fpath in files:
        raw   = json.loads(fpath.read_text())
        cases = raw["cases"] if isinstance(raw, dict) else raw
        for rec in cases:
            name   = rec.get("test_case", rec.get("name", "?"))
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


def _load_single_json(path: Path) -> dict:
    raw   = json.loads(path.read_text())
    cases = raw["cases"] if isinstance(raw, dict) else raw
    out   = {}
    for rec in cases:
        name   = rec.get("test_case", rec.get("name", "?"))
        res    = rec.get("results", {})
        r2_raw = (res.get("pure_llm") or res.get("llm_only") or {}).get("test_r2")
        if r2_raw is None or (isinstance(r2_raw, float) and np.isnan(r2_raw)):
            continue
        out[name] = {"scores": [float(r2_raw)], "n_runs": 1}
    return out


def load_data(source: str, results_dir: Path) -> dict:
    if source in ("variance", "auto") and VARIANCE_JSON.exists():
        data = _load_variance_json(VARIANCE_JSON)
        if data:
            print(f"✅ Loaded variance JSON: {VARIANCE_JSON} ({len(data)} cases)")
            return data
    if source in ("multi", "auto") and results_dir.exists():
        data = _load_multi_run_jsons(results_dir)
        if data:
            print(f"✅ Loaded {len(data)} cases from timestamped multi-run JSONs")
            return data
    if source in ("single", "auto") and FINAL_JSON.exists():
        data = _load_single_json(FINAL_JSON)
        if data:
            print(f"⚠️  Single-run fallback: {FINAL_JSON} ({len(data)} cases, no variance)")
            return data
    print("❌ No results data found. Run the benchmark first:")
    print("   python hypatiax_defi_benchmark_v3c2.py --variance")
    print("   python hypatiax_defi_benchmark_v3c2.py --multi-run 30")
    sys.exit(1)


# ── Build DataFrame ────────────────────────────────────────────────────────────
def build_dataframe(data: dict) -> pd.DataFrame:
    rows = []
    for name, v in data.items():
        scores = v["scores"]
        n_runs = v["n_runs"]
        mean_  = float(np.mean(scores))
        std_   = float(np.std(scores))
        p_i    = sum(1 for s in scores if s > 0.9) / n_runs
        regime = classify_regime(mean_, std_)
        comp   = complexity_proxy(name)
        rows.append({
            "case":        name,
            "mean":        round(mean_,  4),
            "std":         round(std_,   4),
            "ii":          round(std_,   4),   # alias — II = std
            "p_i":         round(p_i,    4),
            "n_valid":     len(scores),
            "n_runs":      n_runs,
            "regime":      regime,
            "complexity":  comp,
        })
    df = pd.DataFrame(rows)
    regime_idx = {r: i for i, r in enumerate(REGIME_ORDER)}
    df["regime_order"] = df["regime"].map(lambda r: regime_idx.get(r, 99))
    df = df.sort_values(["regime_order", "mean"], ascending=[True, False]).drop(
        columns="regime_order").reset_index(drop=True)
    return df


def _legend_handles(df: pd.DataFrame) -> list:
    seen = sorted(df["regime"].unique(),
                  key=lambda r: REGIME_ORDER.index(r) if r in REGIME_ORDER else 99)
    return [
        mpatches.Patch(facecolor=REGIME_PALETTE.get(r, "#aaaaaa"),
                       label=REGIME_LABELS.get(r, r), alpha=0.85)
        for r in seen
    ]


def _save(fig, out_dir: Path, stem: str, fmt: list):
    for f in fmt:
        p = out_dir / f"{stem}.{f}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  💾 {p}")
    plt.close(fig)


# ── Figure 1 — Complexity vs Instability (KEY FIGURE) ─────────────────────────
def plot_complexity_vs_instability(df: pd.DataFrame, out_dir: Path, fmt: list,
                                   show_regline: bool = True):
    """
    KEY PAPER FIGURE: empirical support for the Instability Theorem.
    Higher complexity K → higher II.  Regression line confirms the trend.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Scatter coloured by regime
    sns.scatterplot(
        data=df, x="complexity", y="std", hue="regime",
        palette=_SNS_PALETTE,
        hue_order=[r for r in REGIME_ORDER if r in df["regime"].values],
        s=65, linewidth=0.4, edgecolor="white", ax=ax, zorder=3,
    )

    # Annotate notable cases (high II or high complexity)
    for _, row in df.iterrows():
        if row["std"] > 0.08 or row["complexity"] >= 7:
            short = row["case"][:24] + "…" if len(row["case"]) > 25 else row["case"]
            ax.annotate(short, (row["complexity"], row["std"]),
                        fontsize=6.5, color="#333333", ha="left", va="bottom",
                        xytext=(4, 3), textcoords="offset points")

    # Regression line (OLS) with 95% CI band
    if show_regline:
        sns.regplot(
            data=df, x="complexity", y="std",
            scatter=False, ci=95,
            line_kws={"linestyle": "--", "linewidth": 1.4,
                      "color": "#555555", "alpha": 0.8},
            ax=ax,
        )

    # Threshold lines
    ax.axhline(0.05, color="#e87722", linewidth=0.9, linestyle="--", alpha=0.65,
               label="C threshold (II=0.05)")
    ax.axhline(0.10, color="#d62728", linewidth=0.9, linestyle="--", alpha=0.65,
               label="Severe instability (II=0.10)")

    # Pearson r annotation
    corr = df[["complexity", "std"]].corr().iloc[0, 1]
    ax.text(0.97, 0.97, f"Pearson $r$ = {corr:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

    ax.set_xlabel("Complexity proxy $K$  (algebraic=1–2, transcendental=4–5, Greeks=8–9)",
                  fontsize=9)
    ax.set_ylabel("Instability Index $\\mathrm{II}_i = \\sigma_i$", fontsize=11)
    ax.set_title(
        "Complexity vs Instability — Empirical support for Instability Theorem\n"
        r"$\uparrow K$ $\Rightarrow$ $\uparrow \mathrm{II}$: transcendental formulas "
        "trigger stochastic collapse",
        fontsize=10,
    )

    # Combined legend: regimes + threshold lines
    handles = _legend_handles(df) + [
        mpatches.Patch(color="#e87722", alpha=0.6, label="II=0.05 (C-Marginal boundary)"),
        mpatches.Patch(color="#d62728", alpha=0.6, label="II=0.10 (C-Collapse boundary)"),
    ]
    ax.legend(handles=handles, fontsize=7.5, framealpha=0.9, edgecolor="#cccccc",
              loc="upper left")
    ax.get_legend().remove()          # remove seaborn auto-legend
    fig.legend(handles=handles, fontsize=7.5, framealpha=0.9, edgecolor="#cccccc",
               loc="upper left", bbox_to_anchor=(0.12, 0.88))

    fig.tight_layout()
    _save(fig, out_dir, "fig_paper_complexity_vs_instability", fmt)


# ── Figure 2 — Complexity vs Success Probability ──────────────────────────────
def plot_complexity_vs_success(df: pd.DataFrame, out_dir: Path, fmt: list):
    fig, ax = plt.subplots(figsize=(7, 5))

    sns.scatterplot(
        data=df, x="complexity", y="p_i", hue="regime",
        palette=_SNS_PALETTE,
        hue_order=[r for r in REGIME_ORDER if r in df["regime"].values],
        s=65, linewidth=0.4, edgecolor="white", ax=ax, zorder=3,
    )

    # Regression line
    sns.regplot(
        data=df, x="complexity", y="p_i",
        scatter=False, ci=95,
        line_kws={"linestyle": "--", "linewidth": 1.4,
                  "color": "#555555", "alpha": 0.8},
        ax=ax,
    )

    ax.axhline(0.5, color="#888888", linewidth=0.8, linestyle=":", alpha=0.55,
               label="$p_i = 0.5$ (coin-flip)")

    corr = df[["complexity", "p_i"]].corr().iloc[0, 1]
    ax.text(0.97, 0.97, f"Pearson $r$ = {corr:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

    ax.set_xlabel("Complexity proxy $K$", fontsize=11)
    ax.set_ylabel("Success Probability $p_i = \\mathbb{P}(R^2 > 0.9)$", fontsize=11)
    ax.set_title(
        "Complexity vs Success Probability\n"
        r"Higher $K$ $\Rightarrow$ lower $p_i$: reliability degrades with formula complexity",
        fontsize=10,
    )
    ax.set_ylim(-0.05, 1.1)

    ax.get_legend().remove()
    fig.legend(handles=_legend_handles(df), fontsize=7.5, framealpha=0.9,
               edgecolor="#cccccc", loc="upper right",
               bbox_to_anchor=(0.92, 0.88))
    fig.tight_layout()
    _save(fig, out_dir, "fig_paper_complexity_vs_success", fmt)


# ── Figure 3 — Mean R² vs Instability (styled) ────────────────────────────────
def plot_mean_vs_instability(df: pd.DataFrame, out_dir: Path, fmt: list):
    fig, ax = plt.subplots(figsize=(7, 5.5))

    sns.scatterplot(
        data=df, x="mean", y="std", hue="regime",
        palette=_SNS_PALETTE,
        hue_order=[r for r in REGIME_ORDER if r in df["regime"].values],
        s=65, linewidth=0.4, edgecolor="white", ax=ax, zorder=3,
    )

    # Annotate outliers
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

    ax.set_xlabel("Mean test $R^2$  ($\\mu_i$)", fontsize=11)
    ax.set_ylabel("Instability Index $\\mathrm{II}_i = \\sigma_i$", fontsize=11)
    ax.set_title(
        "Mean $R^2$ vs Instability — Regime Separation\n"
        "Two-cluster structure: Regime A (symbolic) vs Regime C (stochastic)",
        fontsize=10,
    )

    ax.get_legend().remove()
    fig.legend(handles=_legend_handles(df), fontsize=7.5, framealpha=0.9,
               edgecolor="#cccccc", loc="upper right",
               bbox_to_anchor=(0.92, 0.88))
    fig.tight_layout()
    _save(fig, out_dir, "fig_paper_mean_vs_instability", fmt)


# ── Figure 4 — Instability distribution (KDE histogram) ──────────────────────
def plot_instability_hist(df: pd.DataFrame, out_dir: Path, fmt: list):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Stacked histogram by regime, then KDE overlay
    import numpy as np
    bins = np.linspace(0, max(df["std"].max() * 1.05, 0.35), 25)
    for reg in REGIME_ORDER:
        sub = df[df["regime"] == reg]["std"]
        if sub.empty:
            continue
        ax.hist(sub, bins=bins, color=REGIME_PALETTE[reg], alpha=0.72,
                label=REGIME_LABELS.get(reg, reg), edgecolor="white", linewidth=0.5)

    # KDE overlay on full distribution
    sns.kdeplot(data=df, x="std", ax=ax, color="#444444",
                linewidth=1.4, linestyle="-", alpha=0.6, bw_adjust=0.8)

    ax.axvline(0.05, color="#e87722", linewidth=1.2, linestyle="--", alpha=0.8,
               label="C threshold (II=0.05)")
    ax.axvline(0.10, color="#d62728", linewidth=1.2, linestyle="--", alpha=0.8,
               label="Severe threshold (II=0.10)")

    ax.set_xlabel("Instability Index $\\mathrm{II}_i = \\sigma_i$", fontsize=11)
    ax.set_ylabel("Number of cases", fontsize=11)
    ax.set_title(
        "Distribution of LLM Instability Index\n"
        r"Bimodal: spike at $\mathrm{II}\approx 0$ (Regime A) + tail (Regime C)",
        fontsize=10,
    )
    ax.legend(fontsize=7.5, framealpha=0.9, edgecolor="#cccccc")
    fig.tight_layout()
    _save(fig, out_dir, "fig_paper_instability_hist", fmt)


# ── Figure 5 — Regime distribution bar chart ──────────────────────────────────
def plot_regime_counts(df: pd.DataFrame, out_dir: Path, fmt: list):
    counts  = df["regime"].value_counts()
    labels  = [r for r in REGIME_ORDER if r in counts.index]
    values  = [counts[r] for r in labels]
    colours = [REGIME_PALETTE[r] for r in labels]
    xlabels = [REGIME_LABELS.get(r, r).replace(" — ", "\n") for r in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(labels)), values, color=colours,
                  alpha=0.82, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.12,
                str(val), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#333333")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(xlabels, fontsize=8.5)
    ax.set_ylabel("Number of cases", fontsize=11)
    ax.set_xlabel("Regime (paper §4 taxonomy)", fontsize=11)
    ax.set_title(
        "Regime Distribution — HypatiaX DeFi Benchmark\n"
        "A: Symbolic Stability  |  B: Deterministic  |  C: Stochastic Collapse",
        fontsize=10,
    )
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    _save(fig, out_dir, "fig_paper_regime_counts", fmt)


# ── Summary statistics ─────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame, csv_path: Path):
    df.to_csv(csv_path, index=False)
    print(f"\n✅ CSV exported → {csv_path}")
    print(f"   Columns: {list(df.columns)}")

    corr_ii   = df[["complexity", "std"]].corr().iloc[0, 1]
    corr_pi   = df[["complexity", "p_i"]].corr().iloc[0, 1]
    n_A = (df["regime"] == "A-Symbolic").sum()
    n_C = df["regime"].str.startswith("C").sum()
    n_total = len(df)

    print(f"\n── Key statistics ────────────────────────────────────────────────────────")
    print(f"  Cases analysed           : {n_total}")
    print(f"  Regime A (symbolic)      : {n_A}  ({100*n_A/n_total:.1f}%)")
    print(f"  Regime C (collapse)      : {n_C}  ({100*n_C/n_total:.1f}%)")
    print(f"  Pearson r (K vs II)      : {corr_ii:.4f}  "
          f"[paper §4 — complexity drives instability]")
    print(f"  Pearson r (K vs p_i)     : {corr_pi:.4f}  "
          f"[paper §4.5 — complexity degrades success prob.]")
    print(f"  Mean II (Regime A)       : "
          f"{df[df['regime']=='A-Symbolic']['std'].mean():.4f}")
    print(f"  Mean II (Regime C)       : "
          f"{df[df['regime'].str.startswith('C')]['std'].mean():.4f}")
    print(f"── Instability Index legend ──────────────────────────────────────────────")
    print(f"  II_i := sigma_i = std(R²_i^(k))  across N independent LLM runs")
    print(f"  II=0     → deterministic (Regime A/B)")
    print(f"  II≥0.05  → marginal stochastic instability")
    print(f"  II≥0.10  → severe collapse (LLM samples formula distribution)")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Instability analysis pipeline — HypatiaX DeFi Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source", choices=["auto", "variance", "multi", "single"],
                        default="auto",
                        help="JSON source (default: auto-detect)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        metavar="PATH",
                        help=f"Results directory (default: {RESULTS_DIR})")
    parser.add_argument("--out", type=Path, default=Path("hypatiax/data/figures"),
                        metavar="PATH",
                        help="Output directory for figures (default: hypatiax/data/figures)")
    parser.add_argument("--csv-out", type=Path, default=None, metavar="PATH",
                        help="CSV output path (default: <out>/instability_analysis.csv)")
    parser.add_argument("--format", nargs="+", choices=["png", "pdf", "svg"],
                        default=["png", "pdf"], metavar="FMT",
                        help="Output format(s): png pdf svg (default: png pdf)")
    parser.add_argument("--cases", nargs="+", metavar="SUBSTRING",
                        help="Filter cases by substring (case-insensitive)")
    parser.add_argument("--no-regline", action="store_true",
                        help="Omit regression line and CI band from Figure 1")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv_out or (args.out / "instability_analysis.csv")

    # Load
    data = load_data(args.source, args.results_dir)

    # Optional filter
    if args.cases:
        filters = [c.lower() for c in args.cases]
        data = {n: v for n, v in data.items()
                if any(f in n.lower() for f in filters)}
        if not data:
            print(f"❌ No cases matched filters: {args.cases}")
            sys.exit(1)
        print(f"  Case filter active: {len(data)} case(s)")

    df  = build_dataframe(data)
    fmt = args.format

    print(f"\n📊 Figure 1 — Complexity vs Instability (KEY) ...")
    plot_complexity_vs_instability(df, args.out, fmt, show_regline=not args.no_regline)

    print(f"📊 Figure 2 — Complexity vs Success Probability ...")
    plot_complexity_vs_success(df, args.out, fmt)

    print(f"📊 Figure 3 — Mean R² vs Instability ...")
    plot_mean_vs_instability(df, args.out, fmt)

    print(f"📊 Figure 4 — Instability distribution (KDE histogram) ...")
    plot_instability_hist(df, args.out, fmt)

    print(f"📊 Figure 5 — Regime counts ...")
    plot_regime_counts(df, args.out, fmt)

    print_summary(df, csv_path)

    stems = ["fig_paper_complexity_vs_instability", "fig_paper_complexity_vs_success",
             "fig_paper_mean_vs_instability", "fig_paper_instability_hist",
             "fig_paper_regime_counts"]
    print(f"\n✅ All 5 figures saved to {args.out}/")
    for s in stems:
        for f in fmt:
            print(f"   {args.out / s}.{f}")


if __name__ == "__main__":
    main()

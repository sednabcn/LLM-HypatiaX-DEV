#!/usr/bin/env python3
"""
hypatiax_3D_plot_instability.py
================================
Extended instability figures for HypatiaX DeFi Benchmark — 6 plots:

  1. _3d                    — 3D scatter: mean R² x II x p_i, coloured by regime
  2. _phase                 — 2D phase plot: mean R² vs II, regime colours + annotations
  3. _hist                  — Histogram of Instability Index distribution
  4. _success_vs_instability — Scatter: II vs p_i (success probability)
  5. _regimes               — Bar chart of regime counts
  6. _surface               — 3D surface: p_i interpolated over (mean R², II) grid,
                              coloured by regime, actual data points overlaid

Reads from (same priority as hypatiax_plot_instability.py):
  1. hypatiax/data/results/hypatiax_defi_variance_results.json   (--variance runs)
  2. hypatiax_defi_benchmark_v3_results_<TIMESTAMP>.json files   (--multi-run N)
  3. hypatiax_defi_benchmark_v3_results.json                     (single-run fallback)

Usage
─────
  python hypatiax_3D_plot_instability.py                          # all 6 figures
  python hypatiax_3D_plot_instability.py --results-dir path/to/results --out path/to/figures
  python hypatiax_3D_plot_instability.py --format png pdf
  python hypatiax_3D_plot_instability.py --cases theta sharpe     # filter cases
  python hypatiax_3D_plot_instability.py --source variance        # variance JSON only
  python hypatiax_3D_plot_instability.py --elev 30 --azim 220     # custom surface view angle

Author : HypatiaX Team
Version: 1.1 — JMLR submission figures (+ 3D regime surface)
Date   : 2026
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

# ── Default paths (mirror benchmark) ──────────────────────────────────────────
RESULTS_DIR   = Path("hypatiax/data/results")
VARIANCE_JSON = RESULTS_DIR / "hypatiax_defi_variance_results.json"
FINAL_JSON    = RESULTS_DIR / "hypatiax_defi_benchmark_v3_results.json"
MULTI_PATTERN = re.compile(r"hypatiax_defi_benchmark_v3_results_\d{8}T\d{6}Z\.json$")

# ── Regime palette (identical to hypatiax_plot_instability.py) ─────────────────
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


# ── Data loaders (identical logic to hypatiax_plot_instability.py) ─────────────
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


# ── Stats ──────────────────────────────────────────────────────────────────────
def compute_stats(data: dict) -> list:
    rows = []
    for name, v in data.items():
        scores = v["scores"]
        n_runs = v["n_runs"]
        mean_  = float(np.mean(scores))
        std_   = float(np.std(scores))
        p_i    = sum(1 for s in scores if s > 0.9) / n_runs
        regime = classify_regime(mean_, std_)
        rows.append({
            "name":   name,
            "mean":   mean_,
            "std":    std_,
            "p_i":    p_i,
            "regime": regime,
            "colour": REGIME_PALETTE.get(regime, "#aaaaaa"),
        })
    regime_idx = {r: i for i, r in enumerate(REGIME_ORDER)}
    rows.sort(key=lambda r: (regime_idx.get(r["regime"], 99), -r["mean"]))
    return rows


def _legend_handles(rows):
    seen = sorted({r["regime"] for r in rows},
                  key=lambda r: REGIME_ORDER.index(r) if r in REGIME_ORDER else 99)
    return [
        mpatches.Patch(facecolor=REGIME_PALETTE.get(r, "#aaaaaa"),
                       label=REGIME_LABELS.get(r, r), alpha=0.8)
        for r in seen
    ]


def _save(fig, out_dir: Path, stem: str, fmt: list):
    for f in fmt:
        p = out_dir / f"{stem}.{f}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  💾 {p}")
    plt.close(fig)


# ── Figure 1 — 3D scatter ──────────────────────────────────────────────────────
def plot_3d(rows, out_dir, fmt):
    means   = np.array([r["mean"] for r in rows])
    stds    = np.array([r["std"]  for r in rows])
    pis     = np.array([r["p_i"]  for r in rows])
    colours = [r["colour"] for r in rows]

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    ax.scatter(means, stds, pis, c=colours, s=60, depthshade=True, alpha=0.85)

    # Annotate extreme points
    for r, m, s, p in zip(rows, means, stds, pis):
        if s > 0.1 or m < 0.5:
            short = r["name"][:22] + "…" if len(r["name"]) > 23 else r["name"]
            ax.text(m, s, p, short, fontsize=6, color="#333333")

    ax.set_xlabel("Mean $R^2$ ($\\mu_i$)",        fontsize=9, labelpad=8)
    ax.set_ylabel("Instability Index (II = $\\sigma_i$)", fontsize=9, labelpad=8)
    ax.set_zlabel("Success Prob. ($p_i$)",         fontsize=9, labelpad=8)
    ax.set_title("LLM Instability Phase Space\n"
                 r"Axes: $\mu_i$ × $\mathrm{II}_i$ × $p_i$", fontsize=10)

    fig.legend(handles=_legend_handles(rows), loc="lower left",
               fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    _save(fig, out_dir, "fig_instability_3d", fmt)


# ── Figure 2 — Phase plot (mean vs II) ────────────────────────────────────────
def plot_phase(rows, out_dir, fmt):
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for r in rows:
        ax.scatter(r["mean"], r["std"], color=r["colour"],
                   s=55, zorder=3, alpha=0.85, linewidths=0)
        if r["std"] > 0.1 or r["mean"] < 0.5:
            short = r["name"][:26] + "…" if len(r["name"]) > 27 else r["name"]
            ax.annotate(short, (r["mean"], r["std"]),
                        fontsize=6.5, ha="left", va="center",
                        xytext=(5, 0), textcoords="offset points",
                        color="#333333",
                        arrowprops=dict(arrowstyle="-", color="#bbbbbb",
                                        lw=0.6, shrinkA=0, shrinkB=3))

    ax.axvline(0.9,  color="#ff7f0e", linewidth=0.9, linestyle=":",  alpha=0.55,
               label="R²=0.9 success threshold")
    ax.axhline(0.05, color="#e87722", linewidth=0.9, linestyle="--", alpha=0.65,
               label="C threshold (II=0.05)")
    ax.axhline(0.10, color="#d62728", linewidth=0.9, linestyle="--", alpha=0.65,
               label="Severe instability (II=0.10)")

    ax.set_xlabel("Mean test $R^2$ ($\\mu_i$)", fontsize=11)
    ax.set_ylabel("Instability Index $\\mathrm{II}_i = \\sigma_i$", fontsize=11)
    ax.set_title("Regime Separation: Mean $R^2$ vs Instability Index\n"
                 "LLMs operate in two distinct modes: Symbolic (A) vs Stochastic (C)",
                 fontsize=10)
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.35)

    handles = _legend_handles(rows) + [
        mpatches.Patch(color="#ff7f0e", alpha=0.5, label="R²=0.9 threshold"),
        mpatches.Patch(color="#d62728", alpha=0.5, label="II thresholds (0.05 / 0.10)"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="upper right",
              framealpha=0.9, edgecolor="#cccccc")
    fig.tight_layout()
    _save(fig, out_dir, "fig_instability_phase", fmt)


# ── Figure 3 — Histogram of II ────────────────────────────────────────────────
def plot_hist(rows, out_dir, fmt):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    bins = np.linspace(0, max(max(r["std"] for r in rows) * 1.05, 0.35), 25)
    for reg in REGIME_ORDER:
        vals = [r["std"] for r in rows if r["regime"] == reg]
        if vals:
            ax.hist(vals, bins=bins, color=REGIME_PALETTE[reg], alpha=0.75,
                    label=REGIME_LABELS.get(reg, reg),
                    edgecolor="white", linewidth=0.5)

    ax.axvline(0.05, color="#e87722", linewidth=1.2, linestyle="--", alpha=0.8,
               label="C threshold (II=0.05)")
    ax.axvline(0.10, color="#d62728", linewidth=1.2, linestyle="--", alpha=0.8,
               label="Severe threshold (II=0.10)")

    ax.set_xlabel("Instability Index $\\mathrm{II}_i = \\sigma_i$", fontsize=11)
    ax.set_ylabel("Number of cases", fontsize=11)
    ax.set_title("Distribution of LLM Instability Index\n"
                 r"$\mathrm{II}_i = \sigma_i = \mathrm{std}(R^2_i)$ across independent runs",
                 fontsize=10)
    ax.legend(fontsize=7.5, framealpha=0.9, edgecolor="#cccccc")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    _save(fig, out_dir, "fig_instability_hist", fmt)


# ── Figure 4 — Success probability vs II ─────────────────────────────────────
def plot_success_vs_instability(rows, out_dir, fmt):
    fig, ax = plt.subplots(figsize=(7, 5))

    for r in rows:
        ax.scatter(r["std"], r["p_i"], color=r["colour"],
                   s=55, zorder=3, alpha=0.85, linewidths=0)
        if r["std"] > 0.08 or r["p_i"] < 0.3:
            short = r["name"][:26] + "…" if len(r["name"]) > 27 else r["name"]
            ax.annotate(short, (r["std"], r["p_i"]),
                        fontsize=6.5, ha="left", va="center",
                        xytext=(5, 0), textcoords="offset points",
                        color="#333333",
                        arrowprops=dict(arrowstyle="-", color="#bbbbbb",
                                        lw=0.6, shrinkA=0, shrinkB=3))

    ax.axvline(0.05, color="#e87722", linewidth=0.9, linestyle="--", alpha=0.65,
               label="C threshold (II=0.05)")
    ax.axvline(0.10, color="#d62728", linewidth=0.9, linestyle="--", alpha=0.65,
               label="Severe instability (II=0.10)")
    ax.axhline(0.5,  color="#888888", linewidth=0.8, linestyle=":",  alpha=0.5,
               label="$p_i = 0.5$ (coin-flip)")

    ax.set_xlabel("Instability Index $\\mathrm{II}_i = \\sigma_i$", fontsize=11)
    ax.set_ylabel("Success Probability $p_i = \\mathbb{P}(R^2 > 0.9)$", fontsize=11)
    ax.set_title("Success–Instability Tradeoff\n"
                 r"Higher II $\Rightarrow$ lower $p_i$: stochastic collapse degrades reliability",
                 fontsize=10)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(handles=_legend_handles(rows), fontsize=7.5,
              framealpha=0.9, edgecolor="#cccccc")
    fig.tight_layout()
    _save(fig, out_dir, "fig_instability_success_vs_instability", fmt)


# ── Figure 5 — Regime counts bar ──────────────────────────────────────────────
def plot_regimes(rows, out_dir, fmt):
    counts = Counter(r["regime"] for r in rows)
    labels = [r for r in REGIME_ORDER if r in counts]
    values = [counts[r] for r in labels]
    colours = [REGIME_PALETTE[r] for r in labels]
    short_labels = [r.replace("-", "\n") for r in labels]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(range(len(labels)), values, color=colours, alpha=0.82,
                  edgecolor="white", linewidth=0.8)

    # Count labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylabel("Number of cases", fontsize=11)
    ax.set_xlabel("Regime (paper §4 taxonomy)", fontsize=11)
    ax.set_title("Regime Distribution — HypatiaX DeFi Benchmark\n"
                 "A: Symbolic Stability  |  B: Deterministic  |  C: Stochastic Collapse",
                 fontsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_ylim(0, max(values) * 1.18)
    fig.tight_layout()
    _save(fig, out_dir, "fig_instability_regimes", fmt)


# ── Figure 6 — 3D regime surface ──────────────────────────────────────────────

def _idw_pi(rows: list, tx: float, ty: float) -> float:
    """
    Inverse-distance-weighted interpolation of p_i at query point (mean=tx, II=ty).
    Distances are scaled so both axes contribute equally despite different ranges.
    """
    w_sum = p_sum = 0.0
    for r in rows:
        dx = (tx - r["mean"]) * 2.5   # scale mean R² axis
        dy = (ty - r["std"])  * 5.0   # scale II axis (compressed range)
        d2 = dx*dx + dy*dy + 1e-6
        w  = 1.0 / (d2 * d2)          # w ~ 1/d^4 for sharp localisation
        w_sum += w
        p_sum += w * r["p_i"]
    return p_sum / w_sum


def _regime_at(mx: float, iy: float) -> str:
    """Regime of a grid cell — pure boundary function, no interpolation."""
    if mx < 0:
        return "C-Collapse"
    if iy < 1e-6:
        return "A-Symbolic" if mx > 0.99 else "B-Approx"
    if iy >= 0.1:
        return "C-Collapse"
    if iy >= 0.05:
        return "C-Marginal"
    return "B-Det.Biased"


def _hex_to_rgb01(h: str):
    r = int(h[1:3], 16) / 255
    g = int(h[3:5], 16) / 255
    b = int(h[5:7], 16) / 255
    return r, g, b


def plot_3d_surface(rows: list, out_dir: Path, fmt: list,
                    elev: float = 28.0, azim: float = 225.0):
    """
    3D surface plot: axes are mean R² (X), Instability Index II (Y), p_i (Z).

    Surface height  = p_i interpolated via IDW from the empirical data.
    Surface colour  = regime classification at each (mean, II) grid cell,
                      shaded by surface normal (Lambertian diffuse).
    Scatter overlay = actual empirical cases plotted as coloured spheres on top.

    Regime boundary planes rendered as semi-transparent vertical sheets:
      - II = 0.05  (Regime C-Marginal threshold)
      - II = 0.10  (Regime C-Collapse threshold)
      - mean R² = 0.99  (Regime A lower boundary)
    """
    from matplotlib.colors import to_rgba
    from scipy.interpolate import RBFInterpolator

    # ── Grid ──────────────────────────────────────────────────────────────────
    NX, NY      = 55, 40
    mx_min, mx_max = -0.65, 1.08
    ii_min, ii_max =  0.00, 0.52

    mx_lin = np.linspace(mx_min, mx_max, NX)
    ii_lin = np.linspace(ii_min, ii_max, NY)
    MX, II = np.meshgrid(mx_lin, ii_lin)   # (NY, NX)

    # IDW p_i surface
    PI = np.zeros((NY, NX))
    for iy in range(NY):
        for ix in range(NX):
            PI[iy, ix] = _idw_pi(rows, MX[iy, ix], II[iy, ix])
    PI = np.clip(PI, 0.0, 1.0)

    # Per-cell face colour from regime + slight p_i brightness modulation
    face_colors = np.zeros((NY-1, NX-1, 4))
    for iy in range(NY-1):
        for ix in range(NX-1):
            mx_c = 0.5 * (MX[iy, ix] + MX[iy, ix+1])
            ii_c = 0.5 * (II[iy, ix] + II[iy+1, ix])
            pi_c = 0.25 * (PI[iy,ix] + PI[iy,ix+1] + PI[iy+1,ix] + PI[iy+1,ix+1])
            regime = _regime_at(mx_c, ii_c)
            r, g, b = _hex_to_rgb01(REGIME_PALETTE.get(regime, "#aaaaaa"))
            # lighten toward white as p_i decreases (low success = washed out)
            t = 0.25 + (1.0 - pi_c) * 0.45
            face_colors[iy, ix] = (r+(1-r)*t, g+(1-g)*t, b+(1-b)*t, 0.82)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    # Surface
    surf = ax.plot_surface(
        MX, II, PI,
        facecolors=face_colors,
        linewidth=0.15,
        edgecolor="white",
        antialiased=True,
        shade=True,
        alpha=0.88,
    )

    # ── Regime boundary planes (vertical sheets) ───────────────────────────────
    # Plane: II = 0.05 (C-Marginal threshold)
    mx_plane = np.array([[mx_min, mx_max], [mx_min, mx_max]])
    pi_plane = np.array([[0.0, 0.0],       [1.0,    1.0]])
    ii_plane_05 = np.full_like(mx_plane, 0.05)
    ax.plot_surface(mx_plane, ii_plane_05, pi_plane,
                    color="#e87722", alpha=0.10, linewidth=0, shade=False)
    ax.plot([mx_min, mx_max], [0.05, 0.05], [0.0, 0.0],
            color="#e87722", linewidth=1.2, linestyle="--", alpha=0.7)

    # Plane: II = 0.10 (C-Collapse threshold)
    ii_plane_10 = np.full_like(mx_plane, 0.10)
    ax.plot_surface(mx_plane, ii_plane_10, pi_plane,
                    color="#d62728", alpha=0.10, linewidth=0, shade=False)
    ax.plot([mx_min, mx_max], [0.10, 0.10], [0.0, 0.0],
            color="#d62728", linewidth=1.2, linestyle="--", alpha=0.7)

    # Plane: mean R² = 0.99 (Regime A lower boundary)
    ii_v = np.array([[ii_min, ii_min], [ii_max, ii_max]])
    pi_v = np.array([[0.0,    1.0],    [0.0,    1.0]])
    mx_v = np.full_like(ii_v, 0.99)
    ax.plot_surface(mx_v, ii_v, pi_v,
                    color="#2ca02c", alpha=0.08, linewidth=0, shade=False)
    ax.plot([0.99, 0.99], [ii_min, ii_max], [0.0, 0.0],
            color="#2ca02c", linewidth=1.0, linestyle=":", alpha=0.6)

    # ── Scatter: actual data points overlaid on surface ────────────────────────
    means_arr = np.array([r["mean"] for r in rows])
    stds_arr  = np.array([r["std"]  for r in rows])
    pis_arr   = np.array([r["p_i"]  for r in rows])
    cols_arr  = [r["colour"] for r in rows]

    # Slightly raise points above the surface so they're always visible
    pi_offset = pis_arr + 0.02
    ax.scatter(means_arr, stds_arr, pi_offset,
               c=cols_arr, s=55, zorder=5,
               edgecolors="white", linewidths=0.6, depthshade=True, alpha=0.95)

    # Label only the most notable cases (Theta + cases where pi < 0.1 or ii > 0.15)
    for r in rows:
        if r["std"] > 0.15 or (r["p_i"] < 0.1 and r["std"] > 0.08):
            short = r["name"][:20] + "…" if len(r["name"]) > 21 else r["name"]
            ax.text(r["mean"], r["std"], r["p_i"] + 0.06,
                    short, fontsize=6.5, color="#222222",
                    ha="center", va="bottom")

    # ── Axes & labels ─────────────────────────────────────────────────────────
    ax.set_xlabel(r"Mean $R^2$  ($\mu_i$)",              fontsize=9, labelpad=10)
    ax.set_ylabel(r"Instability Index  $\mathrm{II}_i$", fontsize=9, labelpad=10)
    ax.set_zlabel(r"Success Prob.  $p_i$",               fontsize=9, labelpad=8)

    ax.set_xlim(mx_min, mx_max)
    ax.set_ylim(ii_min, ii_max)
    ax.set_zlim(0.0,    1.05)

    ax.set_xticks([-0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_zticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=8)

    ax.set_title(
        "LLM Instability Regime Surface\n"
        r"$p_i = \mathbb{P}(R^2_i > 0.9)$ over $(\mu_i,\,\mathrm{II}_i)$ space — "
        "coloured by regime",
        fontsize=10, pad=12,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = _legend_handles(rows) + [
        mpatches.Patch(color="#e87722", alpha=0.55, label="II=0.05 boundary (C-Marginal)"),
        mpatches.Patch(color="#d62728", alpha=0.55, label="II=0.10 boundary (C-Collapse)"),
        mpatches.Patch(color="#2ca02c", alpha=0.35, label=r"$\mu$=0.99 boundary (Regime A)"),
    ]
    fig.legend(handles=legend_handles, loc="lower left",
               fontsize=7.5, framealpha=0.92, edgecolor="#cccccc",
               bbox_to_anchor=(0.01, 0.01))

    fig.tight_layout()
    _save(fig, out_dir, "fig_instability_surface", fmt)



def main():
    parser = argparse.ArgumentParser(
        description="Generate 6 instability figures for HypatiaX DeFi Benchmark",
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
    parser.add_argument("--format", nargs="+", choices=["png", "pdf", "svg"],
                        default=["png", "pdf"], metavar="FMT",
                        help="Output format(s): png pdf svg (default: png pdf)")
    parser.add_argument("--cases", nargs="+", metavar="SUBSTRING",
                        help="Filter cases by substring (case-insensitive)")
    parser.add_argument("--elev", type=float, default=28.0, metavar="DEG",
                        help="Surface elevation angle in degrees (default: 28)")
    parser.add_argument("--azim", type=float, default=225.0, metavar="DEG",
                        help="Surface azimuth angle in degrees (default: 225)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

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

    rows = compute_stats(data)
    fmt  = args.format

    print(f"\n📊 Figure 1 — 3D phase scatter ...")
    plot_3d(rows, args.out, fmt)

    print(f"📊 Figure 2 — Phase plot (mean R² vs II) ...")
    plot_phase(rows, args.out, fmt)

    print(f"📊 Figure 3 — II histogram ...")
    plot_hist(rows, args.out, fmt)

    print(f"📊 Figure 4 — Success vs Instability ...")
    plot_success_vs_instability(rows, args.out, fmt)

    print(f"📊 Figure 5 — Regime counts ...")
    plot_regimes(rows, args.out, fmt)

    print(f"📊 Figure 6 — 3D regime surface (elev={args.elev}°, azim={args.azim}°) ...")
    plot_3d_surface(rows, args.out, fmt, elev=args.elev, azim=args.azim)

    stems = ["fig_instability_3d", "fig_instability_phase", "fig_instability_hist",
             "fig_instability_success_vs_instability", "fig_instability_regimes",
             "fig_instability_surface"]
    print(f"\n✅ All 6 figures saved to {args.out}/")
    for s in stems:
        for f in fmt:
            print(f"   {args.out / s}.{f}")


if __name__ == "__main__":
    main()

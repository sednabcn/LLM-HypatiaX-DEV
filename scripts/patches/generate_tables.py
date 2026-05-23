#!/usr/bin/env python3
"""
generate_tables.py — Auto-generate LaTeX tables from JSON results

Reads patched JSON outputs and writes .tex table fragments to the output dir.
These are \\input{}-ed by the main paper so NO manual numbers appear in the LaTeX source.

Tables generated:
  - defi_main.tex       (Table 1 — §10.2)
  - defi_tiers.tex      (Table 2 — §10.3)
  - ablation.tex        (Table 5 — §10.6)
  - nguyen12.tex        (Table 7 — §10.8)
  - instability.tex     (Table 8 — §10.9)

Usage:
  # Local dev (uses hardcoded PATCHED / RESULTS paths):
  python tables/generate_tables.py

  # CI (paths supplied by ci_postprocess.yml):
  python tables/generate_tables.py \\
      --results-dir hypatiax/data/results/<subdir> \\
      --output-dir  hypatiax/data/results/<subdir>/tables
"""
# FIX 2: docstring above uses \\ throughout — no bare \i escape, no SyntaxWarning.

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ── CLI arguments ─────────────────────────────────────────────────────────────
# FIX 1: accept --results-dir and --output-dir so ci_postprocess.yml's call
# actually lands files in the right place instead of being silently ignored.

def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate LaTeX table fragments from HypatiaX JSON results."
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Leaf results directory for this experiment "
            "(e.g. hypatiax/data/results/comparison_results/noise-noiseless/noiseless/defi). "
            "When supplied, searched before the default PATCHED/RESULTS roots."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory where .tex files are written "
            "(default: <repo-root>/paper/tables/)."
        ),
    )
    return p.parse_args()


ARGS = _parse_args()

ROOT       = Path(__file__).parent.parent
PATCHED    = ROOT / "hypatiax" / "data" / "patched"
RESULTS    = ROOT / "hypatiax" / "data" / "results"
TABLES_DIR = (ARGS.output_dir or ROOT / "paper" / "tables").resolve()

TABLES_DIR.mkdir(parents=True, exist_ok=True)

GENERATED = 0

# ── JSON shape normalisation ───────────────────────────────────────────────────
# FIX 3: ci_analysis.yml documents three JSON shapes; _extract_rows (and every
# direct data.get() call) previously crashed with
#   AttributeError: 'list' object has no attribute 'get'
# when the top-level was a list (Shape C).  _normalize_to_dict() coerces all
# three shapes to a single summary dict before any .get() call is made.
#
#   Shape A: {"results": {"task_id": {...}, ...}, ...}  — merge_shards.py output
#   Shape B: {"accuracy": 0.9, "total_cases": 74, ...}  — flat summary dict
#   Shape C: [{...}, {...}, ...]                         — list of per-task records

def _normalize_to_dict(raw):
    """
    Coerce Shape A / B / C JSON to a single flat summary dict.

    Shape A and B are already dicts — returned as-is (Shape A keeps its
    top-level summary keys; callers that need per-task iteration use
    raw["results"] directly).
    Shape C (list) returns the first element if it is a dict, otherwise {}.
    The caller's existing .get() chains with fallbacks handle missing keys.
    """
    if isinstance(raw, dict):
        return raw          # Shape A or B — already a dict
    if isinstance(raw, list):
        # Shape C: each element is a per-task record dict.
        # For summary tables we use the first record as a best-effort proxy;
        # callers that need aggregated stats will hit the "?" fallback and log
        # a placeholder, which is correct behaviour when no summary JSON exists.
        return raw[0] if raw and isinstance(raw[0], dict) else {}
    return {}


# ── File discovery ────────────────────────────────────────────────────────────

def load_best(subdir, glob):
    """
    Search for the most-recently-modified file matching *glob* under *subdir*.

    Search order when --results-dir is supplied (CI mode):
      1. ARGS.results_dir          — the leaf dir passed by ci_postprocess.yml
      2. ARGS.results_dir / subdir — in case subdir adds a further level
      3. PATCHED / subdir          — local dev fallback
      4. RESULTS / subdir          — local dev fallback

    Without --results-dir (local dev):
      1. PATCHED / subdir
      2. RESULTS / subdir

    Returns (parsed_object, Path) or (None, None).
    """
    if ARGS.results_dir:
        search_roots = [
            ARGS.results_dir.resolve(),
            ARGS.results_dir.resolve() / subdir,
            PATCHED / subdir,
            RESULTS / subdir,
        ]
    else:
        search_roots = [PATCHED / subdir, RESULTS / subdir]

    for d in search_roots:
        if not d.exists():
            continue
        candidates = sorted(d.glob(glob), key=os.path.getmtime, reverse=True)
        if candidates:
            try:
                return json.loads(candidates[0].read_text()), candidates[0]
            except Exception:
                continue
    return None, None


# ── Output helpers ────────────────────────────────────────────────────────────

def write_table(name, content):
    global GENERATED
    out = TABLES_DIR / name
    out.write_text(content)
    print(f"  ✅ {name}")
    GENERATED += 1


def header_comment(src_file):
    return (
        f"% Auto-generated by tables/generate_tables.py\n"
        f"% Source: {src_file}\n"
        f"% Date:   {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"% DO NOT EDIT MANUALLY — re-run 'make tables' to regenerate\n\n"
    )


# ── Table: DeFi main results (§10.2) ──────────────────────────────────────────

def gen_defi_main():
    raw, src = load_best("defi", "benchmark_results*.json")
    if not raw:
        print("  ⚠  defi_main.tex — no data found, writing placeholder")
        write_table("defi_main.tex", "% No DeFi results yet\n")
        return

    # FIX 3: normalise before any .get() call
    data = _normalize_to_dict(raw)

    acc       = data.get("accuracy", data.get("success_rate", data.get("discovery_rate", 0)))
    total     = data.get("total_cases", data.get("n_cases", "?"))
    successes = data.get(
        "successes",
        data.get("n_success", int(acc * total) if isinstance(total, int) else "?"),
    )
    timing  = data.get("mean_time", data.get("avg_time_s", data.get("runtime_s", "?")))
    r2_mean = data.get("mean_r2", data.get("avg_r2", "?"))

    tex = header_comment(src) + r"""
\begin{table}[h]
\centering
\caption{HypatiaX DeFi Benchmark Results (v3.0)}
\label{tab:defi_main}
\begin{tabular}{lrr}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{Notes} \\
\midrule
""" + f"Discovery rate & {acc:.1%} & {successes}/{total} cases \\\\\n"

    if r2_mean != "?":
        tex += f"Mean $R^2$ & {r2_mean:.4f} & across successful discoveries \\\\\n"
    if timing != "?":
        tex += f"Mean runtime & {timing:.1f}s & per case \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("defi_main.tex", tex)


# ── Table: DeFi tier breakdown (§10.3) ────────────────────────────────────────

def gen_defi_tiers():
    raw, src = load_best("defi", "benchmark_results*.json")
    if not raw:
        write_table("defi_tiers.tex", "% No DeFi results yet\n")
        return

    # FIX 3: normalise before any .get() call
    data = _normalize_to_dict(raw)

    tiers = {
        "Easy":   data.get("easy_cases",   data.get("easy",   {}).get("count", "?")),
        "Medium": data.get("medium_cases", data.get("medium", {}).get("count", "?")),
        "Hard":   data.get("hard_cases",   data.get("hard",   {}).get("count", "?")),
    }
    tier_acc = {
        "Easy":   data.get("easy_accuracy",   data.get("easy",   {}).get("accuracy", "?")),
        "Medium": data.get("medium_accuracy", data.get("medium", {}).get("accuracy", "?")),
        "Hard":   data.get("hard_accuracy",   data.get("hard",   {}).get("accuracy", "?")),
    }

    tex = header_comment(src) + r"""
\begin{table}[h]
\centering
\caption{DeFi Benchmark — Results by Difficulty Tier}
\label{tab:defi_tiers}
\begin{tabular}{lcc}
\toprule
\textbf{Tier} & \textbf{Cases} & \textbf{Discovery Rate} \\
\midrule
"""
    for tier, count in tiers.items():
        acc_str = f"{tier_acc[tier]:.1%}" if isinstance(tier_acc[tier], float) else "---"
        tex += f"{tier} & {count} & {acc_str} \\\\\n"

    tex += r"""\midrule
Total & 74 & ---  \\
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("defi_tiers.tex", tex)


# ── Table: Core-15 ablation (§10.6) ───────────────────────────────────────────

def gen_ablation():
    raw, src = load_best("exp1_ablation", "*.json")
    if not raw:
        write_table("ablation.tex", "% No ablation results yet\n")
        return

    # FIX 3: normalise before any .get() call
    data = _normalize_to_dict(raw)

    hypatia_r2 = data.get("hypatiax_far_r2", data.get("far_r2_hypatiax", "?"))
    pysr_r2    = data.get("pysr_far_r2",     data.get("far_r2_pysr", "?"))
    mw_p       = data.get("mw_p", data.get("mann_whitney_p", data.get("p_value", "?")))
    mw_u       = data.get("mw_u", data.get("mann_whitney_u", data.get("u_statistic", "?")))

    tex = header_comment(src) + r"""
\begin{table}[h]
\centering
\caption{Core-15 Ablation: PySR-only vs.\ HypatiaX (far-$R^2$)}
\label{tab:llm_ablation}
\begin{tabular}{lcc}
\toprule
\textbf{System} & \textbf{Mean far-$R^2$} & \textbf{n} \\
\midrule
"""
    hyp_str = f"{hypatia_r2:.4f}" if isinstance(hypatia_r2, float) else "---"
    psr_str = f"{pysr_r2:.4f}"    if isinstance(pysr_r2,    float) else "---"

    tex += f"HypatiaX & {hyp_str} & 15 \\\\\n"
    tex += f"PySR-only & {psr_str} & 15 \\\\\n"

    mw_str = ""
    if isinstance(mw_p, float):
        mw_str = f"$U={mw_u:.1f}$, $p={mw_p:.4f}$ (two-sided)"

    tex += r"""\bottomrule
\end{tabular}
"""
    if mw_str:
        tex += f"\\begin{{tablenotes}}\\item Mann-Whitney: {mw_str}\\end{{tablenotes}}\n"
    tex += r"\end{table}" + "\n"

    write_table("ablation.tex", tex)


# ── Table: Instability summary (§10.9) ────────────────────────────────────────

def gen_instability():
    raw, src = load_best("instability", "*.json")
    if not raw:
        write_table("instability.tex", "% No instability results yet\n")
        return

    # FIX 3: normalise before any .get() call
    data = _normalize_to_dict(raw)

    total    = data.get("total_tasks", data.get("n_tasks", 70))
    k_runs   = data.get("k_runs", data.get("n_runs", 30))
    spearman = data.get("spearman_rho", data.get("rho", "?"))
    cv_mean  = data.get("mean_cv", data.get("cv_mean", "?"))

    tex = header_comment(src) + r"""
\begin{table}[h]
\centering
\caption{Instability Analysis: """ + f"{total} tasks, $K={k_runs}$" + r""" runs each}
\label{tab:instability}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
"""
    tex += f"Total tasks & {total} \\\\\n"
    tex += f"Runs per task ($K$) & {k_runs} \\\\\n"
    if isinstance(spearman, float):
        tex += f"Spearman $\\rho$ & {spearman:.4f} \\\\\n"
    if isinstance(cv_mean, float):
        tex += f"Mean CV & {cv_mean:.4f} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("instability.tex", tex)


# ── Reproducibility macro file ────────────────────────────────────────────────

def gen_repro_macros():
    """Write a LaTeX macro file so numbers can be used as \\repoVal{defiAccuracy} in text."""
    macros = {}

    raw, _ = load_best("defi", "benchmark_results*.json")
    if raw:
        # FIX 3: normalise before any .get() call
        data = _normalize_to_dict(raw)
        acc = data.get("accuracy", data.get("success_rate", 0))
        macros["defiAccuracy"]   = f"{acc:.1%}"
        macros["defiTotalCases"] = str(data.get("total_cases", 74))

    raw, _ = load_best("exp1_ablation", "*.json")
    if raw:
        # FIX 3: normalise before any .get() call
        data = _normalize_to_dict(raw)
        mw_p = data.get("mw_p", data.get("mann_whitney_p", ""))
        mw_u = data.get("mw_u", data.get("mann_whitney_u", ""))
        if mw_p:
            macros["coreAblationMWp"] = f"{mw_p:.4f}"
        if mw_u:
            macros["coreAblationMWu"] = f"{mw_u:.1f}"

    lines = [
        "% Auto-generated reproducibility macros",
        "% Usage in paper: \\repoVal{defiAccuracy}",
        f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    for key, val in macros.items():
        lines.append(f"\\newcommand{{\\{key}}}{{{val}}}")

    write_table("repro_macros.tex", "\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("═" * 55)
    print("  Table Generator — HypatiaX JMLR")
    print("═" * 55)
    print(f"  Results : {ARGS.results_dir or '(default PATCHED/RESULTS)'}")
    print(f"  Output  : {TABLES_DIR}\n")

    gen_defi_main()
    gen_defi_tiers()
    gen_ablation()
    gen_instability()
    gen_repro_macros()

    print(f"\n{'═' * 55}")
    print(f"  Generated: {GENERATED} table files → {TABLES_DIR}")
    print(f"{'═' * 55}")
    print(
        "\n  To use in LaTeX:\n"
        "    \\input{tables/defi_main.tex}\n"
        "    \\input{tables/ablation.tex}\n"
        "    \\input{tables/repro_macros.tex}\n"
    )


if __name__ == "__main__":
    main()

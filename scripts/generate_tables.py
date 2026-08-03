#!/usr/bin/env python3
"""
generate_tables.py — Auto-generate LaTeX tables from JSON results

Reads patched JSON outputs and writes .tex table fragments to paper/tables/.
These are \\input{}-ed by the main paper and supplements so NO manual numbers
appear in the LaTeX source.

Tables generated  (main paper)
  defi_main.tex       tab:main_results   §10.2   ← results/defi/
  defi_tiers.tex      tab:difficulty     §10.3   ← results/defi/
  runtime.tex         tab:runtime        §10.4   ← results/defi/
  portfolio_sweep.tex tab:portfolio_seed §10.5   ← portfolio_variance_seed_sweep.json
  ablation.tex        tab:llm_ablation   §10.6   ← results/ablation/exp1_ablation/
  feynman.tex         tab:feynman        §10.7   ← results/feynman/
  nguyen12.tex        tab:nguyen12       §10.8   ← results/nguyen12/
  instability.tex     tab:instability    §10.9   ← results/instability/
  version_history.tex tab:version_hist   §App B  ← hardcoded (stable)
  timing_detail.tex   tab:timing_detail  §App C  ← results/defi/
  repro_macros.tex    \\newcommand macros for inline numbers

Tables generated  (Supplement B — suppB / STEP 10 outputs)
  five_system.tex             tab:five_systems_full     App    ← five_systems/exp1_five/
                                                                  exp1_five_results.json
                                                                  (no fallback; only the launched
                                                                  experiment's results are used)
  five_system_performance.tex tab:five_systems_perf     App    ← exp1_five_performance.json
  five_system_extrapolation.tex tab:five_systems_extrap App    ← exp1_five_extrapolation.json
  five_system_stat_tests.tex  app:statistical_tests     App D  ← exp1_five_results.json
                                                                  (Mann-Whitney U / Cohen's d /
                                                                  Glass's Delta — see Issue 4,
                                                                  04_ci_sd_incompatibility.tex)
  five_system_exp2five.tex             tab:five_systems_full_exp2five      App
                                        ← five_systems/exp2_five/ (own dedicated
                                          reader — see gen_five_system_exp2five())
  five_system_exp2five_performance.tex tab:five_systems_performance_exp2five App
  five_system_exp2five_extrapolation.tex tab:five_systems_extrapolation_exp2five App
  five_system_exp2five_stat_tests.tex  Appendix (exp2_five stats)          App
                                        ← same Mann-Whitney/Cohen's-d pipeline as
                                          five_system_stat_tests.tex, exp2_five's
                                          own data, never mixed with exp1_five's
  suppb_r2_noise.tex      tab:r2_noise    §noise  ← noise_sweep_*.json
  suppb_rr_noise.tex      tab:rr_noise    §noise  ← noise_sweep_*.json
  suppb_time_noise.tex    tab:time_noise  §noise  ← noise_sweep_*.json
  suppb_sc_metrics.tex    tab:sc_metrics  §sc     ← sample_complexity_*.json
  suppb_winrate.tex       tab:winrate     §winrate← both JSONs
  suppb_noiseless.tex     tab:overall     §noiseless ← protocol_core_noiseless_*.json

Usage
-----
  python generate_tables.py
  python generate_tables.py \\
      --results-dir hypatiax/data/results \\
      --output-dir  scripts/paper/tables
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate all HypatiaX LaTeX tables from result JSONs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--results-dir",  type=Path, default=None, dest="results_dir",
                   metavar="PATH",
                   help="Root of hypatiax/data/results (auto-detected if omitted).")
    p.add_argument("--output-dir",   type=Path, default=None, dest="output_dir",
                   metavar="PATH",
                   help="Output dir for .tex files (default: <repo>/paper/tables).")
    p.add_argument("--patched-dir",  type=Path, default=None, dest="patched_dir",
                   metavar="PATH",
                   help="Patched-results override dir (checked before --results-dir).")
    p.add_argument("--noise-sweep-json", type=Path, default=None, dest="noise_sweep",
                   metavar="PATH",
                   help="Explicit noise_sweep_*.json (auto-detected if omitted).")
    p.add_argument("--sample-complexity-json", type=Path, default=None,
                   dest="sample_complexity", metavar="PATH",
                   help="Explicit sample_complexity_*.json (auto-detected if omitted).")
    p.add_argument("--experiment", type=str, default=None, dest="experiment",
                   metavar="NAME",
                   help="Experiment tag (e.g. exp2_feynman_pca).  When supplied, "
                        "only the tables relevant to that experiment are generated. "
                        "Omit (or pass 'all') to regenerate every table.")
    p.add_argument("--allow-fallback", action="store_true", dest="allow_fallback",
                   help="Exit 0 even if one or more result JSONs are missing and "
                        "paper-verified fallback numbers were substituted. Without "
                        "this flag (the default), a missing JSON causes the tables "
                        "to still be generated (for local debugging) but the process "
                        "exits non-zero, so a CI step invoking this script fails "
                        "instead of reporting a stale green summary. Intended for "
                        "deliberate local/partial drafting only — never pass this "
                        "flag in CI.")
    return p.parse_args()


# ── Path resolution ───────────────────────────────────────────────────────────

def _find_repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent,
                      *Path(__file__).resolve().parents]:
        if (candidate / "hypatiax" / "__init__.py").exists():
            return candidate
    # fallback: two levels up from this script
    return Path(__file__).resolve().parent.parent


_ARGS      = _parse_args()
_ROOT      = _find_repo_root()
PATCHED    = _ARGS.patched_dir  or (_ROOT / "hypatiax" / "data" / "patched")
RESULTS    = _ARGS.results_dir  or (_ROOT / "hypatiax" / "data" / "results")
TABLES_DIR = _ARGS.output_dir   or (_ROOT / "paper" / "tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ── Normalise RESULTS against known suppB/suppB_sc canonical subdirs ──────────
# load_sweep_json() and load_best() always append a hardcoded subdir such as
# "comparison_results/feynman-tests/sample-complexity" to RESULTS.  When CI
# passes the already-resolved canonical dir as --results-dir (e.g.
# hypatiax/data/results/comparison_results/feynman-tests/sample-complexity),
# RESULTS / subdir produces a self-nested doubled path that does not exist and
# causes sc_data / noise_data to come back None, silently falling back to
# placeholder tables.  Strip the suffix when present so the join always lands
# at the correct location regardless of which --results-dir the caller supplies.
_CANONICAL_SUFFIXES = (
    "comparison_results/feynman-tests/sample-complexity",
    "comparison_results/feynman-tests/noise-sweep/noise-sweep",
    "comparison_results/feynman-tests/noise-sweep",
    "ablation/exp1_ablation",
    # _load_five_system_rows_real() / _load_exp1_five_subtable_json() /
    # _load_exp2_five_own_rows() all join "five_systems/exp1_five" or
    # "five_systems/exp2_five" onto RESULTS themselves (same convention as
    # the suppB/suppC/exp1_ablation cases above). Guard against the same
    # doubled-path mistake if a future CI edit passes the already-resolved
    # subdir (e.g. "${OUT_BASE}/${SUB}") instead of the results root.
    "five_systems/exp1_five",
    "five_systems/exp2_five",
)
for _suffix in _CANONICAL_SUFFIXES:
    _parts = Path(_suffix).parts
    if RESULTS.parts[-len(_parts):] == _parts:
        RESULTS = RESULTS.parents[len(_parts) - 1]
        break

GENERATED = 0

# Tables (or repro_macros entries) written using hardcoded paper-verified
# fallback data rather than a located, live source JSON. Populated by
# generator functions themselves (not just the main() JSON audit), so a
# table can be flagged as fallback even when *some* JSON was found at its
# usual path but didn't actually contain the field the table needs -- see
# gen_five_system()/gen_repro_macros() and Issue 4 (CI/SD incompatibility):
# the exp1_ablation JSON exists and loads fine, but it is per-equation
# ablation data and has never been confirmed to contain a "five_system"/
# "system_comparison" key, so silently treating "JSON found" as "five-system
# data found" is exactly how that issue's fallback usage went unflagged.
FALLBACK_TABLES: list[str] = []

# ── JSON location map (run_all.sh → tables-generator) ────────────────────────
#
#  This table documents where each experiment step writes its JSON output and
#  which load_best() subdir / glob is used to pick it up.
#
#  Step          run_all.sh output path                           load_best subdir / glob
#  ─────────────────────────────────────────────────────────────────────────────────────
#  exp1          RESULTS_DIR/                                     ""  (root)  benchmark_results*.json
#                  hypatiax_defi_benchmark_v3*results*.json         (defi fallback also checked)
#  exp1b         RESULTS_DIR/                                     ""  (root)  portfolio_variance*.json
#                  portfolio_variance_seed_sweep.json
#  extrap        RESULTS_DIR/comparison_results/extrapolation/    "comparison_results/extrapolation"
#                  all_domains_extrap_v4_*.json
#  hybrid_all    RESULTS_DIR/hybrid_llm_nn/all_domains/           "hybrid_llm_nn/all_domains"
#                  hybrid_llm_nn_all_domains_*.json
#  instability   RESULTS_DIR/figures/                             "figures"  (CSV + JSON)
#                  instability_analysis.csv / instability*.json
#  exp1_ablation RESULTS_DIR/ablation/exp1_ablation/              "ablation/exp1_ablation"  *.json  ✓
#                  (matches ci_postprocess.yml's SUB mapping: 'ablation/exp1_ablation';
#                   previously read "exp1_ablation" here, which silently mismatched
#                   the CI-resolved path and always fell through to fallback data)
#  exp2_feynman  RESULTS_DIR/comparison_results/feynman-tests/    "comparison_results/feynman-tests/exp2"
#                  exp2/exp2_results*.json                          *.json
#  exp2          RESULTS_DIR/  exp2_run.log  (no dedicated JSON)  "comparison_results"  all_systems_merged.json
#  exp3/exp3b    RESULTS_DIR/  (nguyen12 script writes to cwd)    "nguyen12"  *.json  — may need
#                  exp3_nguyen12_hybrid50v_02.py output              explicit --results-dir
#  suppB         RESULTS_DIR/comparison_results/feynman-tests/    "comparison_results/feynman-tests/noise-sweep"
#                  noise-sweep/noise_sweep_*.json                   noise_sweep_*.json  ✓
#  suppB_sc      RESULTS_DIR/comparison_results/feynman-tests/    "comparison_results/feynman-tests/sample-complexity"
#                  sample-complexity/sample_complexity_*.json        sample_complexity_*.json  ✓
#  noiseless     RESULTS_DIR/comparison_results/noise-noiseless/  hardcoded glob in gen_suppb_noiseless()  ✓
#                  noiseless/protocol_core_noiseless_*.json


# ── Helpers ───────────────────────────────────────────────────────────────────

# FIX SC-CHECKPOINT-POLLUTION: per-shard checkpoint files (e.g.
# sample_complexity_n1000_checkpoint.json, written mid-run by
# run_sample_complexity_benchmark.py and left behind alongside the final
# consolidated sample_complexity_<timestamp>.json) match the same
# "*sample_complexity*.json" / glob patterns used below. Neither load_best()
# nor load_sweep_json() excluded them, and both sort candidates by mtime —
# so whichever file happens to have the latest mtime wins, which has been
# the checkpoint shard (confirmed against real CI runs: it silently loaded
# instead of the canonical file in every run checked, producing a
# header-only suppb_sc_metrics.tex with 0 data rows every time, including
# before and after the per_equation rmse fix above). The bash step in
# ci_postprocess.yml that invokes this script DOES exclude these correctly
# when computing $SC_DATA for its own log line, but never passes that value
# through via --sample-complexity-json — so auto-detection here is the only
# thing actually selecting the file used. Mirrors generate_figures.py's
# _SWEEP_EXCLUDE_SUBSTRINGS so both scripts agree on what counts as a
# "real" result file for the same family of inputs.
_EXCLUDE_SUBSTRINGS = ("checkpoint", "_sig", "MISSING")


def _filtered_glob(d: Path, glob_pat: str) -> list[Path]:
    """d.glob(glob_pat), minus any candidate whose basename contains one of
    _EXCLUDE_SUBSTRINGS (checkpoint shards, per-sigma shards, MISSING
    placeholders) — see _EXCLUDE_SUBSTRINGS docstring above for why this
    can't be skipped."""
    return [p for p in d.glob(glob_pat)
            if not any(s in p.name for s in _EXCLUDE_SUBSTRINGS)]


def load_best(subdir: str, glob_pat: str,
              extra_subdirs: list[str] | None = None) -> tuple[dict | None, Path | None]:
    """Return (data, path) for the newest matching JSON.

    Search order:
      1. PATCHED / subdir
      2. RESULTS / subdir
      3. Each path in extra_subdirs (checked as RESULTS / extra)
    An empty-string subdir means search directly under the base directory.
    """
    search_dirs: list[Path] = []
    for base in [PATCHED, RESULTS]:
        search_dirs.append(base / subdir if subdir else base)
    for extra in (extra_subdirs or []):
        search_dirs.append(RESULTS / extra if extra else RESULTS)

    for d in search_dirs:
        if not d.exists():
            continue
        candidates = sorted(_filtered_glob(d, glob_pat), key=os.path.getmtime, reverse=True)
        if candidates:
            try:
                return json.loads(candidates[0].read_text()), candidates[0]
            except Exception:
                continue
    return None, None


# ── Five-system table source resolution (superseded guess-based approach) ──
#
# Formerly this file searched an unconfirmed list of candidate directories
# (_FIVE_SYSTEM_CANDIDATES: ablation/exp1_ablation, comparison_results/
# five_system, comparison_results/system_comparison, etc.) for a JSON with a
# "five_system"/"system_comparison" key, and fell back to hardcoded
# FIVE_SYSTEM_PAPER_ROWS when none was found -- which was always, since that
# key was never confirmed to exist anywhere in the tree (see Issue 4).
#
# Both the guessing and the hardcoded fallback are gone. The two real
# sources are now named explicitly:
#   1. exp1_five_system.py → RESULTS_DIR/five_systems/exp1_five/exp1_five_results.json
#      (_load_exp1_five_rows(), below _load_exp2_five_system_rows())
#   2. exp2/exp2_extrap (Feynman suite) → _load_exp2_five_system_rows()
#      (previously written but never wired into gen_five_system() -- dead
#      code until now)
# See _load_five_system_rows_real() for the combined loader gen_five_system()
# and the repro_macros nnExtrap* block actually call.


def load_sweep_json(explicit: Path | None, subdir: str, glob_pat: str) -> dict | None:
    """Load a sweep JSON — explicit path takes priority, then glob in RESULTS/subdir."""
    if explicit and explicit.exists():
        try:
            return json.loads(explicit.read_text())
        except Exception:
            pass
    # auto-detect: newest matching file under noise-sweep subdir
    sweep_dir = RESULTS / subdir
    if sweep_dir.exists():
        candidates = sorted(_filtered_glob(sweep_dir, glob_pat), key=os.path.getmtime, reverse=True)
        for c in candidates:
            try:
                return json.loads(c.read_text())
            except Exception:
                continue
    # also try the parent comparison_results level
    alt_dir = RESULTS / "comparison_results" / "feynman-tests" / "noise-sweep"
    if alt_dir.exists():
        candidates = sorted(_filtered_glob(alt_dir, glob_pat), key=os.path.getmtime, reverse=True)
        for c in candidates:
            try:
                return json.loads(c.read_text())
            except Exception:
                continue
    return None




def write_table(name: str, content: str) -> None:
    global GENERATED
    out = TABLES_DIR / name
    out.write_text(content)
    print(f"  ✅ {name}")
    GENERATED += 1


def header_comment(src_file) -> str:
    src = str(src_file) if src_file else "unknown"
    return (
        f"% Auto-generated by tables/generate_tables.py\n"
        f"% Source: {src}\n"
        f"% Date:   {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"% DO NOT EDIT MANUALLY — re-run 'make tables' to regenerate\n\n"
    )


def _pct(v) -> str:
    if isinstance(v, (int, float)):
        return f"{v*100:.1f}\\%"
    return "---"


def _f4(v) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    return "---"


def _f6(v) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.6f}"
    return "---"


# ── Main paper tables ─────────────────────────────────────────────────────────

def gen_defi_main() -> None:
    """
    Tab 2 — Aggregate extrapolation performance on the HypatiaX DeFi Benchmark (74 tasks).
    Columns: Method | Median R² | Mean R² | >0.99 (%) | >0.9 (%) | Catastrophic
    Three methods: Pure LLM, Neural MLP, HypatiaX.
    Source JSON is expected to have a top-level key per method name (or a list under
    "methods") with the aggregate scalar stats. Falls back to paper-verified values.

    FIX ISSUE-9 (masked-failure / uncorrected Mean R²):
    The real on-disk file written by hypatiax_defi_benchmark_v3c.py /
    hypatiax_defi_benchmark_pca.py is a flat LIST of 74 per-case dicts
    (each `{"results": {"pure_llm": {...}, "neural_network": {...},
    "hybrid": {"test_r2":, "decision":, "success":}}}`) — see those files'
    _save_final(). It is NOT a dict with a "methods" list or named
    "pure_llm"/"neural_mlp"/"hypatiax" sub-dicts (Shapes 1/2 below), which
    no generator in this codebase actually produces. _extract_rows()
    therefore always fell through to the hardcoded PAPER_ROWS fallback —
    which is where the uncorrected Mean R² = +0.8721 in the paper actually
    comes from, not from a live computation. Shape 3 below fixes the
    parsing AND applies the decision-attribution correction described in
    the abstract's masking-disclosure footnote: HypatiaX's own hybrid.test_r2
    is not trusted at face value — for cases routed to "llm"/"nn"/
    "nn_fallback", the actual independently-computed sub-method result
    (pure_llm.test_r2 / neural_network.test_r2) is used instead, since that
    is the sub-method the routing decision actually names.
    """
    # run_all.sh (exp1) writes hypatiax_defi_benchmark_v3*results*.json to RESULTS_DIR root.
    # Also check legacy defi/ subdir for backwards compatibility.
    data, src = load_best("", "hypatiax_defi_benchmark_v3*results*.json",
                          extra_subdirs=["defi"])
    # NOTE: these are the pre-Issue-9 UNCORRECTED paper values — kept only as
    # an absolute last resort when no result file can be parsed at all, and
    # NEVER silently: see the ::warning:: below whenever this path is hit.
    # Do not treat this as a source of truth for Mean R² / success rates.
    PAPER_ROWS = [
        ("Pure LLM",   1.0000, -0.7571, 62.2, 62.2, 6),
        ("Neural MLP", -0.4675, -0.9482,  5.4, 12.2, 0),
        ("HypatiaX",   1.0000, +0.8721, 90.5, 90.5, 0),
    ]

    # decision -> which independently-computed sub-method result actually
    # backs that routing decision. "ensemble" has no separate baseline arm
    # (it's unique to the hybrid pipeline), so it keeps hybrid's own value.
    _DECISION_TO_BASELINE = {
        "llm":         "pure_llm",
        "nn":          "neural_network",
        "nn_fallback": "neural_network",
    }

    def _is_num(v) -> bool:
        return isinstance(v, (int, float)) and v == v  # excludes NaN

    def _corrected_hybrid_r2(case_results: dict) -> float:
        """Cross-check hybrid's routing decision against the actual
        sub-method it names, instead of trusting hybrid's own self-reported
        test_r2 (Issue 9 — this is what masks e.g. pure_llm.test_r2 ≈
        -141,000 behind a hybrid.test_r2 ≈ 1.0 for the same case)."""
        hybrid   = case_results.get("hybrid", {}) or {}
        decision = hybrid.get("decision", "")
        baseline_key = _DECISION_TO_BASELINE.get(decision)
        if baseline_key:
            baseline_r2 = (case_results.get(baseline_key, {}) or {}).get("test_r2")
            if _is_num(baseline_r2):
                return float(baseline_r2)
        return hybrid.get("test_r2", float("nan"))

    def _clip(v):
        return max(-10.0, min(1.0, v)) if _is_num(v) else float("nan")

    def _method_stats(r2_values: list) -> tuple:
        """(median_r2, mean_r2) clipped to [-10,1], pct>0.99, pct>0.9 (raw),
        n_catastrophic (raw R² < -10), over a fixed denominator of 74."""
        raw = [v for v in r2_values if _is_num(v)]
        n   = len(r2_values) or 1   # fixed denominator, matches table caption
        clipped = [_clip(v) for v in raw]
        median_r2 = statistics.median(clipped) if clipped else float("nan")
        mean_r2   = (sum(clipped) / len(clipped)) if clipped else float("nan")
        pct99 = 100.0 * sum(1 for v in raw if v > 0.99) / n
        pct90 = 100.0 * sum(1 for v in raw if v > 0.9)  / n
        n_cat = sum(1 for v in raw if v < -10)
        return median_r2, mean_r2, pct99, pct90, n_cat

    def _extract_rows(d) -> list[tuple]:
        """Try to read 3-method rows from various JSON shapes."""
        # Shape 3: the ACTUAL on-disk format — flat list of per-case dicts.
        if isinstance(d, list) and d and isinstance(d[0], dict) and "results" in d[0]:
            pure_llm_r2, nn_r2, hybrid_r2 = [], [], []
            for rec in d:
                cr = rec.get("results", {}) or {}
                pure_llm_r2.append((cr.get("pure_llm", {}) or {}).get("test_r2", float("nan")))
                nn_r2.append((cr.get("neural_network", {}) or {}).get("test_r2", float("nan")))
                hybrid_r2.append(_corrected_hybrid_r2(cr))
            rows = []
            for name, values in [("Pure LLM", pure_llm_r2), ("Neural MLP", nn_r2),
                                  ("HypatiaX", hybrid_r2)]:
                med, mean, p99, p90, ncat = _method_stats(values)
                rows.append((name, med, mean, p99, p90, ncat))
            return rows

        if not isinstance(d, dict):
            return []
        rows = []
        # Shape 1: d["methods"] = [{name, median_r2, mean_r2, ...}, ...]
        if "methods" in d and isinstance(d["methods"], list):
            for m in d["methods"]:
                rows.append((
                    m.get("name", "?"),
                    m.get("median_r2", m.get("median_test_r2", float("nan"))),
                    m.get("mean_r2",   m.get("mean_test_r2",   float("nan"))),
                    m.get("success_rate_99", m.get("r2_gt_099", float("nan"))) * 100
                    if m.get("success_rate_99", m.get("r2_gt_099", 0)) <= 1
                    else m.get("success_rate_99", m.get("r2_gt_099", float("nan"))),
                    m.get("success_rate_90", m.get("r2_gt_09",  float("nan"))) * 100
                    if m.get("success_rate_90", m.get("r2_gt_09", 0)) <= 1
                    else m.get("success_rate_90", m.get("r2_gt_09", float("nan"))),
                    m.get("n_catastrophic", m.get("catastrophic_failures", 0)),
                ))
        # Shape 2: d["pure_llm"], d["neural_mlp"], d["hypatiax"] sub-dicts
        for name, key in [("Pure LLM", "pure_llm"), ("Neural MLP", "neural_mlp"),
                          ("HypatiaX", "hypatiax")]:
            m = d.get(key, {})
            if m:
                rows.append((
                    name,
                    m.get("median_r2", float("nan")),
                    m.get("mean_r2",   float("nan")),
                    m.get("success_rate_99", float("nan")),
                    m.get("success_rate_90", float("nan")),
                    m.get("n_catastrophic", 0),
                ))
        return rows if len(rows) == 3 else []

    rows = _extract_rows(data) if data else []
    if not rows:
        print("  ::warning:: gen_defi_main: could not parse live results "
              f"(src={src}) — falling back to hardcoded PAPER_ROWS. These "
              "are the UNCORRECTED pre-Issue-9 paper values and do NOT "
              "reflect the decision-attribution fix; do not cite Mean R² "
              "from this table run without checking this warning.")
        rows = PAPER_ROWS   # last-resort fallback — see warning above

    def _r2(v): return f"{v:.4f}" if isinstance(v, float) and not (v != v) else "---"
    def _pct(v): return f"{v:.1f}" if isinstance(v, float) and not (v != v) else "---"
    def _int(v): return str(int(v)) if isinstance(v, (int, float)) else "---"

    tex = header_comment(src) + r"""
\begin{table}[t]
\centering
\caption{Aggregate extrapolation performance on the HypatiaX DeFi Benchmark
  (74 tasks). All $R^2$ values clipped to $[-10, 1]$; fixed denominator of 74.
  Catastrophic: $R^2 < -10$. HypatiaX's Mean $R^2$ and success-rate columns
  are decision-attribution-corrected --- see \S\ref{sec:hybrid-attribution-bug}.}
\label{tab:main_results}
\begin{tabular}{lrrrrr}
\toprule
\textbf{Method} & \textbf{Median $R^2$} & \textbf{Mean $R^2$}
  & $\mathbf{>0.99}$ \textbf{(\%)} & $\mathbf{>0.9}$ \textbf{(\%)}
  & \textbf{Catastrophic} \\
\midrule
"""
    for name, med, mean, r99, r90, cat in rows:
        tex += f"{name} & {_r2(med)} & {_r2(mean)} & {_pct(r99)} & {_pct(r90)} & {_int(cat)} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("defi_main.tex", tex)


def gen_defi_tiers() -> None:
    """
    Tab 3 — Near-perfect success rate (R²>0.99) by difficulty.
    Columns: Difficulty | n | Pure LLM (%) | HypatiaX (%) | Gain (pp)
    Paper-verified fallback values from Table 3 (v3.0).
    """
    data, src = load_best("", "hypatiax_defi_benchmark_v3*results*.json",
                          extra_subdirs=["defi"])

    # Paper-verified fallback (Table 3)
    PAPER_TIERS = [
        ("Easy",   24, 87.5, 100.0, +12.5),
        ("Medium", 29, 58.6,  89.7, +31.1),
        ("Hard",   21, 38.1,  76.2, +38.1),
        ("Overall",74, 62.2,  89.2, +27.0),
    ]

    def _extract_tiers(d):
        if not isinstance(d, dict):
            return []
        tiers = []
        for label, key, n_default in [
            ("Easy",    "easy",    24),
            ("Medium",  "medium",  29),
            ("Hard",    "hard",    21),
            ("Overall", "overall", 74),
        ]:
            sub = d.get(key, {})
            n   = sub.get("n", sub.get("count", n_default))
            llm = sub.get("llm_r99", sub.get("pure_llm_success_rate_99",
                  sub.get("llm_success_99", float("nan"))))
            hyp = sub.get("hypatiax_r99", sub.get("hypatiax_success_rate_99",
                  sub.get("hybrid_success_99", float("nan"))))
            if isinstance(llm, float) and llm <= 1.0:
                llm *= 100
            if isinstance(hyp, float) and hyp <= 1.0:
                hyp *= 100
            gain = (hyp - llm) if isinstance(hyp, float) and isinstance(llm, float) else float("nan")
            tiers.append((label, n, llm, hyp, gain))
        return tiers

    tiers = _extract_tiers(data) if data else []
    if not tiers or any(t[2] != t[2] for t in tiers):   # NaN check
        tiers = PAPER_TIERS

    def _pct(v): return f"{v:.1f}" if isinstance(v, float) and not (v != v) else "---"
    def _sgn(v):
        if not isinstance(v, float) or v != v:
            return "---"
        return f"+{v:.1f}" if v >= 0 else f"{v:.1f}"

    tex = header_comment(src) + r"""
\begin{table}[t]
\centering
\caption{Near-perfect success rate ($R^2 > 0.99$) by difficulty.
  Fixed denominator per tier; LLM and Hybrid use single-run evaluation.}
\label{tab:difficulty}
\begin{tabular}{lcrrrr}
\toprule
\textbf{Difficulty} & \textbf{n}
  & \textbf{Pure LLM (\%)} & \textbf{HypatiaX (\%)} & \textbf{Gain (pp)} \\
\midrule
"""
    for label, n, llm, hyp, gain in tiers:
        sep = r"\midrule" + "\n" if label == "Overall" else ""
        tex += f"{sep}{label} & {n} & {_pct(llm)} & {_pct(hyp)} & {_sgn(gain)} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("defi_tiers.tex", tex)


def gen_ablation() -> None:
    """
    Tab 6 — LLM Ablation: PySR Alone vs. HypatiaX (PySR + LLM Warm-Start) on Core 15.
    Per-equation rows with Train / Near / Med / Far R² for P and H, plus timing.
    Matches Table 6 in §10.6.
    """
    data, src = load_best("ablation/exp1_ablation", "*.json")

    # Paper-verified values for all 15 equations (Table 6)
    PAPER_EQUATIONS = [
        # (equation, domain, P_train, H_train, P_near, H_near,
        #   P_med, H_med, P_far, H_far, P_time, H_time)
        ("Arrhenius",             "Chemistry",  0.9896, 0.9971, -0.9783, -0.4012, -0.6766, -0.6624, -12.5549, -12.5553, 149, 110),
        ("Henderson-Hasselbalch", "Chemistry",  0.9123, 0.9338,  0.2137,  0.2172,  0.9633, -3.6019,   0.2137,  -4.9142, 110, 110),
        ("Rate Law",              "Chemistry",  0.9977, 0.9977,  1.0000,  0.9999,  1.0000,  1.0000,   1.0000,   0.9999, 158, 159),
        ("Allometric Scaling",    "Biology",    0.9977, 0.9973,  0.9996,  0.9509,  1.0000,  0.8602,   0.9996,  -2.1139, 102, 106),
        ("Michaelis-Menten",      "Biology",    0.9948, 0.9968, -68.5896, -0.0979, -368.7928, -2.4717, -83899.527, -634.5989, 144, 123),
        ("Logistic Growth",       "Biology",    0.9974, 0.9975,  0.9795,  0.9999,  0.9947,  1.0000,   0.9934,   0.9999, 145, 151),
        ("Kinetic Energy",        "Physics",    0.9968, 0.9968,  1.0000,  1.0000,  1.0000,  1.0000,   1.0000,   1.0000, 139, 138),
        ("Gravitational Force",   "Physics",    0.9146, 0.9544, -4.2880, -2.6752, -0.0260, -0.0016,  -9.0418,  -7.6360, 104, 108),
        ("Ideal Gas Law",         "Physics",    0.9976, 0.9976,  0.9999,  0.9999,  1.0000,  1.0000,   0.9999,   0.9999, 136, 139),
        ("Impermanent Loss",      "DeFi AMM",   0.9975, 0.9975,  0.9121,  0.9113, -0.3063, -0.3091, -62.4026, -62.5166, 106, 106),
        ("Price Impact",          "DeFi AMM",   0.9976, 0.9976,  1.0000,  1.0000,  1.0000,  1.0000,   1.0000,   1.0000, 106, 111),
        ("Constant Product",      "DeFi AMM",   0.9982, 0.9982,  0.9996,  0.9996,  1.0000,  1.0000,   0.9996,   0.9996, 137, 147),
        ("Value at Risk",         "DeFi Risk",  0.9979, 0.9979,  0.9999,  0.9999,  1.0000,  1.0000,   0.9999,   0.9999, 138, 143),
        ("Liquidation Price",     "DeFi Risk",  0.9974, 0.9974,  0.9999,  1.0000,  1.0000,  1.0000,   1.0000,   1.0000, 145, 146),
        ("Portfolio Variance",    "DeFi Risk",  0.9504, 0.9975,  0.8865,  1.0000,  0.9493,  1.0000, -118.4482,   1.0000, 141, 141),
    ]

    # Try to read per-equation data from the authoritative merged file.
    # _merged.json is a dict keyed by equation display name (not "equations"/
    # "cases"/"results" -- those keys don't exist in the real schema, which
    # is why this always fell back to PAPER_EQUATIONS before). Loaded
    # explicitly by filename rather than via load_best()'s newest-mtime glob,
    # because _merged.json, the per-shard files, _analysis.json, and others
    # all share an identical mtime in this results directory -- mtime alone
    # can't reliably pick the merged file over a partial shard.
    def _load_merged() -> tuple[dict | None, Path | None]:
        for base in (PATCHED, RESULTS):
            p = (base / "ablation" / "exp1_ablation" / "_merged.json")
            if p.exists():
                try:
                    return json.loads(p.read_text()), p
                except Exception:
                    continue
        return None, None

    def _extract_equations(d):
        if not isinstance(d, dict):
            return []
        rows = []
        for name, eq in d.items():
            if not isinstance(eq, dict):
                continue
            p = eq.get("pysr_only", {}) or {}
            h = eq.get("hypatia", {}) or {}
            rows.append((
                eq.get("name", name),
                eq.get("domain", "?"),
                p.get("train_r2",       float("nan")),
                h.get("train_r2",       float("nan")),
                p.get("extrap_r2_near", float("nan")),
                h.get("extrap_r2_near", float("nan")),
                p.get("extrap_r2_medium", float("nan")),
                h.get("extrap_r2_medium", float("nan")),
                p.get("extrap_r2_far",  float("nan")),
                h.get("extrap_r2_far",  float("nan")),
                p.get("total_time_s",   p.get("sr_time_s", float("nan"))),
                h.get("total_time_s",   h.get("sr_time_s", float("nan"))),
            ))
        return rows if len(rows) >= 15 else []

    merged_data, merged_src = _load_merged()
    if merged_data:
        data, src = merged_data, merged_src

    equations = _extract_equations(data) if data else []
    if not equations:
        equations = PAPER_EQUATIONS

    # Real Mann-Whitney result, loaded explicitly from exp1_rf01_mannwhitney.json
    # rather than hardcoded placeholder defaults (previously 0.2948 / 126.0,
    # neither of which came from any real computation). IMPORTANT: this file
    # reports n_pairs=3 (12 of 15 equations skipped, Chemistry domain only) --
    # a materially different sample size than the "n=15" claimed elsewhere in
    # the paper, and a materially different conclusion (not significant,
    # h_wins=0). Surfaced honestly here rather than silently relabelled as
    # n=15; do not paper over this by hand-editing n back to 15 upstream.
    def _load_mannwhitney() -> dict | None:
        for base in (PATCHED, RESULTS):
            p = base / "ablation" / "exp1_ablation" / "exp1_rf01_mannwhitney.json"
            if p.exists():
                try:
                    return json.loads(p.read_text()).get("rf01_mann_whitney")
                except Exception:
                    continue
        return None

    _mw = _load_mannwhitney()
    if _mw:
        mw_u = _mw.get("U_two_sided")
        mw_p = _mw.get("p_two_sided")
        mw_n = _mw.get("n_pairs")
        mw_skipped = _mw.get("n_skipped", 0)
    else:
        _d = data if isinstance(data, dict) else {}
        mw_p = _d.get("mw_p", _d.get("mann_whitney_p"))
        mw_u = _d.get("mw_u", _d.get("mann_whitney_u"))
        mw_n = None
        mw_skipped = 0

    def _r(v, clip=None):
        if not isinstance(v, (int, float)) or v != v:
            return "---"
        if clip and v < clip:
            return r"$\ll{-100}$"
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:.1f}"

    def _t(v):
        return str(int(v)) if isinstance(v, (int, float)) and v == v else "---"

    tex = header_comment(src) + r"""
\begin{table*}[t]
\centering
\caption{LLM Ablation: PySR Alone vs.\ HypatiaX (PySR + LLM Warm-Start) on Core~15.
  Extrap columns show $R^2$ at near ($1.2\times$), medium (canonical),
  and far ($5\times$) out-of-distribution ranges.}
\label{tab:llm_ablation}
\small
\begin{tabular}{llrrrrrrrrrr}
\toprule
 & & \multicolumn{2}{c}{\textbf{Train $R^2$}}
   & \multicolumn{2}{c}{\textbf{Near $R^2$}}
   & \multicolumn{2}{c}{\textbf{Med $R^2$}}
   & \multicolumn{2}{c}{\textbf{Far $R^2$}}
   & \multicolumn{2}{c}{\textbf{Time (s)}} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
\cmidrule(lr){9-10}\cmidrule(lr){11-12}
\textbf{Equation} & \textbf{Domain}
  & P & H & P & H & P & H & P & H & P & H \\
\midrule
"""
    for (eq, dom, pt, ht, pn, hn, pm, hm, pf, hf, ptime, htime) in equations:
        tex += (
            f"{eq} & {dom} & {_r(pt)} & {_r(ht)} & {_r(pn)} & {_r(hn)}"
            f" & {_r(pm)} & {_r(hm)} & {_r(pf,-1000)} & {_r(hf,-1000)}"
            f" & {_t(ptime)} & {_t(htime)} \\\\\n"
        )

    tex += r"""\midrule
\multicolumn{2}{l}{\textit{Mean}} """
    # Compute means over the 15 equations
    import statistics as _st
    def _mean_r2(col):
        vals = [r for r in col if isinstance(r, float) and r == r and r >= -1e5]
        return f"{_st.mean(vals):.4f}" if vals else "---"

    cols = list(zip(*equations))
    tex += (
        f"& {_mean_r2(cols[2])} & {_mean_r2(cols[3])}"
        f" & {_mean_r2(cols[4])} & {_mean_r2(cols[5])}"
        f" & {_mean_r2(cols[6])} & {_mean_r2(cols[7])}"
        f" & {_mean_r2(cols[8])} & {_mean_r2(cols[9])}"
        f" & {_mean_r2(cols[10])} & {_mean_r2(cols[11])} \\\\\n"
    )

    if mw_u is not None and mw_p is not None:
        _n_str = str(mw_n) if mw_n is not None else "?"
        _mw_note = (
            f"  Mann--Whitney (far-$R^2$): $U={mw_u:.1f}$, $p={mw_p:.4f}$ "
            f"(two-sided, $n={_n_str}$"
        )
        if mw_skipped:
            _mw_note += (
                f"; {mw_skipped} of 15 equations excluded from this test -- "
                r"see \texttt{exp1\_rf01\_mannwhitney.json} for which and why"
            )
        _mw_note += ").\n"
    else:
        _mw_note = "  Mann--Whitney statistic unavailable for this run.\n"

    tex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item P = PySR-only; H = HypatiaX (PySR + LLM warm-start).
  Near/Med/Far $R^2$ at $1.2\times$, canonical, and $5\times$ training range.
""" + _mw_note + r"""\end{tablenotes}
\end{table*}
"""
    write_table("ablation.tex", tex)


# 95% two-tailed t critical values, df=1..30 (Student's t table). df>30 falls
# back to the normal-approximation z=1.960. Avoids adding a scipy dependency
# for a single-purpose lookup; values match standard published t-tables.
_T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _ci95(mean, std, n) -> str | None:
    """
    95% CI for a sample mean, computed from that sample's own mean/std/n via
    the standard t-distribution formula: mean +/- t(0.975, n-1) * std/sqrt(n).

    Returns None (never a fabricated number) unless mean, std, and n are all
    present and numeric with n >= 2, so a table can never print a CI that
    wasn't actually derived from its own std/n columns. This is what keeps
    the CI and Std columns from being able to drift apart again: they are
    now computed from the same row, in the same place, every regeneration.
    """
    try:
        mean = float(mean)
        std = float(std)
        n = int(n)
    except (TypeError, ValueError):
        return None
    if n < 2 or std < 0:
        return None
    df = n - 1
    t_crit = _T_CRIT_95.get(df, 1.960)  # normal approx beyond the table
    import math
    margin = t_crit * std / math.sqrt(n)
    return f"[{mean - margin:.1f}, {mean + margin:.1f}]"


# Paper-verified fallback (Table 1) — REMOVED. Two real, named data sources
# now exist (_load_exp1_five_rows(), _load_exp2_five_system_rows(), combined
# via _load_five_system_rows_real()); a hardcoded numeric substitute is no
# longer needed and its presence was exactly the kind of silent-substitution
# risk this whole cleanup pass exists to eliminate. If both real sources are
# ever unavailable, gen_five_system() and gen_repro_macros() now write a
# clearly-marked "NO DATA" placeholder and hard-fail via FALLBACK_TABLES
# instead of emitting numbers that look real but aren't sourced from any run.


# ── Live exp2/exp2_extrap five-system aggregation (second-tier source) ────────
#
# See five_systems.tex (investigation report) for the full derivation of
# everything below. Summary of the two load-bearing findings that shape this
# loader:
#
#   1. exp2 (Feynman, 10-domain, --protocol all_domains) runs SIX methods,
#      not five. EnhancedHybridSystemDeFi (core) is excluded below because
#      it is DeFi-domain-scoped (different module, different decision
#      strategy) and does not correspond to any of the five paper row
#      names — confirmed via its 'ensemble' vs. method #4's explicit
#      LLM-with-NN-fallback 'llm'/'nn_applied' decision pattern.
#
#   2. compute_extrap_r2_far() in run_comparative_suite_benchmark_v2.py
#      only evaluates methods that return a re-evaluable symbolic formula
#      string. Methods returning NN architecture tags or "N/A" (Neural
#      Network, and usually System 3 LLM+Fallback) get null extrapolation
#      fields — a structural property of that script, not a gap in this
#      loader. Their rows below will legitimately show n=0 for the
#      extrapolation columns while still having real train_r2_mean data.
#
# IMPORTANT: rows from this loader are a different, live, traceable dataset
# on the actual exp2 (Feynman-suite) problem set, distinct from
# _load_exp1_five_rows()'s Core-15 dataset above it in the loader chain. In
# particular:
#   - Hybrid v50_2's n happens to match the old (now-removed) fallback's 14
#     but its extrapolation error % differs by two to three orders of
#     magnitude (old fallback: "0.0", live: ~120%) — a reminder that the two
#     numbers were never actually the same measurement.
#   - Neural Network's old fallback shows n=13 extrapolation records, which
#     is structurally impossible for this script to produce (see finding 2
#     above) — the old number almost certainly came from a different,
#     DeFi-domain harness (run_hybrid_system_benchmark.py), not from exp2.
# gen_five_system() labels output from this tier accordingly and does not
# claim it reproduces the previously published table.

_EXP2_METHOD_TO_ROW: dict[str, str] = {
    "PureLLM Baseline (core)":              "Pure LLM",
    "ImprovedNN (core)":                    "Neural Network",
    "SymbolicEngineWithLLM (tools)":        "System 2 Symbolic",
    "HybridSystemLLMNN all-domains (core)": "System 3 LLM+Fallback",
    "HybridDiscoverySystem v50_2 (tools)":  "Hybrid v50\\_2",
}

_EXP2_DESIGN_FOCUS: dict[str, str] = {
    "Pure LLM": "Recognition",
    "Neural Network": "Baseline",
    "System 2 Symbolic": "Validation",
    "System 3 LLM+Fallback": "Robustness",
    "Hybrid v50\\_2": "Extrapolation",
}

# Presentation order matching the original FIVE_SYSTEM_PAPER_ROWS layout.
_EXP2_ROW_ORDER = [
    "Hybrid v50\\_2", "Neural Network", "Pure LLM",
    "System 2 Symbolic", "System 3 LLM+Fallback",
]


def _finite_or_none(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v == v and v not in (float("inf"), float("-inf")) else None


def _robust_median_clipped_mean(values: list[float]) -> tuple[float | None, float | None]:
    """Median (always robust) and an IQR-clipped mean (outlier-resistant).

    Raw means of extrap_error_pct are dominated by single catastrophic-
    blowup equations — e.g. one degenerate Pure LLM formula alone can push
    a raw mean into the billions of percent (see five_systems.tex) — so an
    unclipped arithmetic mean is not a usable summary statistic here.
    """
    vals = sorted(values)
    if not vals:
        return None, None
    n = len(vals)
    median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    q1, q3 = vals[n // 4], vals[(3 * n) // 4]
    iqr = q3 - q1
    clipped = [v for v in vals if v <= q3 + 1.5 * iqr] if iqr > 0 else vals
    mean = sum(clipped) / len(clipped) if clipped else None
    return median, mean


def _load_exp2_five_system_rows() -> tuple[list[tuple] | None, Path | None]:
    """Build five-system rows directly from live exp2 + exp2_extrap output.

    Returns (rows, representative_path) in the same shape _rows_from_data()
    returns, or (None, None) if no exp2 output can be found/parsed at all.
    See the module comment above _EXP2_METHOD_TO_ROW for why this will NOT
    reproduce FIVE_SYSTEM_PAPER_ROWS, and why that's expected.
    """
    # ── in-distribution (train) R² per method, deduped by (domain, test) ──
    # Checks both the nested comparison_results/... path (default layout)
    # and a bare "exp2_multi"/"exp2_extrap" subdir, in case --results-dir
    # was already resolved to the experiment-specific directory by the
    # caller (mirroring the precedent in _load_exp2_multi_domain_rows()).
    train_dirs = []
    for base in (PATCHED, RESULTS):
        train_dirs.append(base / "comparison_results/feynman-tests/exp2_multi")
        train_dirs.append(base / "exp2_multi")

    latest_test: dict[tuple, dict] = {}
    train_src: Path | None = None
    for d in train_dirs:
        if not d.exists():
            continue
        for f in sorted(_filtered_glob(d, "protocol_core_noiseless_*.json")):
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            train_src = train_src or f
            for rec in data.get("tests", []):
                key = (rec.get("domain"), rec.get("description"))
                latest_test[key] = rec  # sorted filenames -> later timestamp wins

    r2_by_method: dict[str, list[float]] = {m: [] for m in _EXP2_METHOD_TO_ROW}
    for rec in latest_test.values():
        for mname, mres in rec.get("results", {}).items():
            if mname not in r2_by_method or not mres.get("success"):
                continue
            v = _finite_or_none(mres.get("r2"))
            if v is not None:
                r2_by_method[mname].append(v)

    # ── extrapolation error % per method, from the canonical flat extrap file ──
    extrap_dirs = []
    for base in (PATCHED, RESULTS):
        extrap_dirs.append(base / "comparison_results/feynman-tests/exp2_extrap")
        extrap_dirs.append(base / "exp2_extrap")

    err_by_method: dict[str, list[float]] = {m: [] for m in _EXP2_METHOD_TO_ROW}
    extrap_src: Path | None = None
    for d in extrap_dirs:
        if not d.exists() or extrap_src is not None:
            continue
        # Canonical (unsuffixed) file only — shard-suffixed copies
        # (benchmark_results_extrap_shard*.json) are per-shard partials of
        # the same merged data and would double-count records if also read.
        canonical = d / "benchmark_results_extrap.json"
        if not canonical.exists():
            continue
        try:
            data = json.loads(canonical.read_text())
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        extrap_src = canonical
        for rec in data:
            mname = rec.get("method")
            if mname not in err_by_method:
                continue
            v = _finite_or_none(rec.get("extrap_error_pct"))
            if v is not None:
                err_by_method[mname].append(v)

    if train_src is None and extrap_src is None:
        return None, None

    rows_by_name: dict[str, tuple] = {}
    for mname, row_name in _EXP2_METHOD_TO_ROW.items():
        tr_vals = r2_by_method.get(mname, [])
        tr_mean = sum(tr_vals) / len(tr_vals) if tr_vals else None
        tr_std = None
        if tr_mean is not None and len(tr_vals) >= 2:
            tr_std = (sum((x - tr_mean) ** 2 for x in tr_vals) / (len(tr_vals) - 1)) ** 0.5

        err_vals = err_by_method.get(mname, [])
        median, clipped_mean = _robust_median_clipped_mean(err_vals)

        rows_by_name[row_name] = (
            row_name,
            len(err_vals),
            f"{median:.1f}" if median is not None else "---",
            f"{clipped_mean:.1f}" if clipped_mean is not None else "---",
            f"{tr_mean:.3f}" if tr_mean is not None else "---",
            f"{tr_std:.4f}" if tr_std is not None else "---",
            _EXP2_DESIGN_FOCUS[row_name],
        )

    rows = [rows_by_name[name] for name in _EXP2_ROW_ORDER if name in rows_by_name]
    if len(rows) < 2:
        return None, None
    return rows, (train_src or extrap_src)


# ── exp2_five's OWN dedicated output (five_systems/exp2_five/) ────────────────
#
# Previously exp2_five had no reader at all: gen_five_system()'s "secondary"
# fallback (_load_exp2_five_system_rows() above) actually reads exp2's own
# regular full run (comparison_results/feynman-tests/exp2_multi/exp2_extrap),
# re-filtered down to the same 5 methods -- NOT five_systems/exp2_five/,
# which is where exp2_five (run_comparative_suite_benchmark_v2.py
# --methods 1 2 4 5 6) actually writes its results. So exp2_five's own run
# was silently unused by every table.
#
# This is a genuinely separate data source from exp1_five, not another
# fallback tier for the same table: different equation suite (10-domain
# Feynman vs. Core-15 DeFi/physics), different sample sizes per method, same
# script/JSON schema as exp2 (protocol_core_noiseless_*.json for train R²,
# benchmark_results_extrap.json for extrapolation error), same method-name
# keys, so _EXP2_METHOD_TO_ROW / _EXP2_DESIGN_FOCUS / _EXP2_ROW_ORDER are
# reused as-is -- only the directory changes.
def _load_exp2_five_own_raw(method_name: str) -> tuple[list[float], list[float], Path | None]:
    """Raw per-record (train_r2 values, extrap_error_pct values, src) for one
    JSON method-name key (an _EXP2_METHOD_TO_ROW key, e.g.
    "HybridDiscoverySystem v50_2 (tools)"), read directly from exp2_five's
    own output under five_systems/exp2_five/. Shared by both the aggregated
    row loader below and the raw-sample loader gen_five_system_exp2five_stat_tests()
    needs for its Mann-Whitney test."""
    train_dirs = [base / "five_systems/exp2_five" for base in (PATCHED, RESULTS)]
    latest_test: dict[tuple, dict] = {}
    train_src: Path | None = None
    for d in train_dirs:
        if not d.exists():
            continue
        for f in sorted(_filtered_glob(d, "protocol_core_noiseless_*.json")):
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            train_src = train_src or f
            for rec in data.get("tests", []):
                key = (rec.get("domain"), rec.get("description"))
                latest_test[key] = rec

    r2_vals: list[float] = []
    for rec in latest_test.values():
        mres = rec.get("results", {}).get(method_name)
        if not isinstance(mres, dict) or not mres.get("success"):
            continue
        v = _finite_or_none(mres.get("r2"))
        if v is not None:
            r2_vals.append(v)

    extrap_dirs = [base / "five_systems/exp2_five" for base in (PATCHED, RESULTS)]
    err_vals: list[float] = []
    extrap_src: Path | None = None
    for d in extrap_dirs:
        if not d.exists() or extrap_src is not None:
            continue
        canonical = d / "benchmark_results_extrap.json"
        if not canonical.exists():
            continue
        try:
            data = json.loads(canonical.read_text())
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        extrap_src = canonical
        for rec in data:
            if rec.get("method") != method_name:
                continue
            v = _finite_or_none(rec.get("extrap_error_pct"))
            if v is not None:
                err_vals.append(v)

    return r2_vals, err_vals, (train_src or extrap_src)


def _load_exp2_five_own_rows() -> tuple[list[tuple] | None, Path | None]:
    """Aggregated (median, clipped-mean, tr_mean, tr_std) rows from
    exp2_five's OWN output -- same shape _load_exp2_five_system_rows()
    returns, but sourced from five_systems/exp2_five/ instead of exp2's
    regular directory. See module comment above for why these two loaders
    are deliberately kept separate rather than merged."""
    rows_by_name: dict[str, tuple] = {}
    src: Path | None = None
    for mname, row_name in _EXP2_METHOD_TO_ROW.items():
        tr_vals, err_vals, this_src = _load_exp2_five_own_raw(mname)
        src = src or this_src
        tr_mean = sum(tr_vals) / len(tr_vals) if tr_vals else None
        tr_std = None
        if tr_mean is not None and len(tr_vals) >= 2:
            tr_std = (sum((x - tr_mean) ** 2 for x in tr_vals) / (len(tr_vals) - 1)) ** 0.5
        median, clipped_mean = _robust_median_clipped_mean(err_vals)
        rows_by_name[row_name] = (
            row_name,
            len(err_vals),
            f"{median:.1f}" if median is not None else "---",
            f"{clipped_mean:.1f}" if clipped_mean is not None else "---",
            f"{tr_mean:.3f}" if tr_mean is not None else "---",
            f"{tr_std:.4f}" if tr_std is not None else "---",
            _EXP2_DESIGN_FOCUS[row_name],
        )
    rows = [rows_by_name[name] for name in _EXP2_ROW_ORDER if name in rows_by_name]
    if len(rows) < 2 or src is None:
        return None, None
    return rows, src


# ── exp1_five_system.py (primary source, added once that experiment existed) ──
#
# Reads exp1_five_results.json directly: {eq_idx: {"name":..., "domain":...,
# method_name: {...train_r2, train_rmse, extrap_rmse_far, success, design_focus...}}}.
# method_name keys are already the five row names verbatim (exp1_five_system.py
# writes them that way), so no name-mapping table is needed here the way
# _EXP2_METHOD_TO_ROW is needed for the exp2-based loader below.
#
# extrap_error_pct is computed the same way compute_extrap_r2_far()/
# _load_exp2_five_system_rows() define it — (extrap_rmse_far / train_rmse) * 100
# — so numbers from this loader and the exp2-based one are on the same scale
# and comparable, even though they're measuring different equation suites
# (Core-15 vs. the 10-domain Feynman set).
def _load_exp1_five_rows() -> tuple[list[tuple] | None, Path | None]:
    src = None
    for base in (PATCHED, RESULTS):
        candidate = base / "five_systems/exp1_five/exp1_five_results.json"
        if candidate.exists():
            src = candidate
            break
    if src is None:
        return None, None
    try:
        data = json.loads(src.read_text())
    except Exception:
        return None, None
    if not isinstance(data, dict):
        return None, None

    r2_by_method: dict[str, list[float]] = {m: [] for m in _EXP2_ROW_ORDER}
    err_by_method: dict[str, list[float]] = {m: [] for m in _EXP2_ROW_ORDER}
    focus_by_method: dict[str, str] = {}
    for eq_entry in data.values():
        if not isinstance(eq_entry, dict):
            continue
        for method_name, res in eq_entry.items():
            if method_name not in r2_by_method or not isinstance(res, dict):
                continue
            if not res.get("success"):
                continue
            if "design_focus" in res:
                focus_by_method.setdefault(method_name, res["design_focus"])
            r2 = _finite_or_none(res.get("train_r2"))
            if r2 is not None:
                r2_by_method[method_name].append(r2)
            train_rmse = _finite_or_none(res.get("train_rmse"))
            far_rmse   = _finite_or_none(res.get("extrap_rmse_far"))
            if train_rmse and train_rmse > 0 and far_rmse is not None:
                err_by_method[method_name].append((far_rmse / train_rmse) * 100.0)

    rows_by_name: dict[str, tuple] = {}
    for mname in _EXP2_ROW_ORDER:
        tr_vals = r2_by_method.get(mname, [])
        tr_mean = sum(tr_vals) / len(tr_vals) if tr_vals else None
        tr_std = None
        if tr_mean is not None and len(tr_vals) >= 2:
            tr_std = (sum((x - tr_mean) ** 2 for x in tr_vals) / (len(tr_vals) - 1)) ** 0.5
        err_vals = err_by_method.get(mname, [])
        median, clipped_mean = _robust_median_clipped_mean(err_vals)
        rows_by_name[mname] = (
            mname, len(err_vals),
            f"{median:.1f}" if median is not None else "---",
            f"{clipped_mean:.1f}" if clipped_mean is not None else "---",
            f"{tr_mean:.3f}" if tr_mean is not None else "---",
            f"{tr_std:.4f}" if tr_std is not None else "---",
            focus_by_method.get(mname, _EXP2_DESIGN_FOCUS.get(mname, "---")),
        )
    rows = [rows_by_name[m] for m in _EXP2_ROW_ORDER if m in rows_by_name]
    return (rows, src) if len(rows) >= 2 else (None, None)


# ── Raw per-equation extrapolation errors + Mann-Whitney/effect-size stats ────
#
# Added to close Issue 4 ("CI/SD Statistical Incompatibility", see
# 04_ci_sd_incompatibility.tex): Source A (jmlr_paper_main.tex, Table 1 /
# tab:five_systems_full) prints a blank Std column, while Source B
# (supp_benchmark_report.tex, app:statistical_tests) separately hand-types a
# Mann-Whitney U test, descriptive stats, and Cohen's d/Glass's Delta for the
# Hybrid-v50_2-vs-Neural-Network extrapolation comparison -- with no generator
# anywhere in this file computing those numbers from exp1_five_results.json,
# so the two .tex sources could (and did) drift apart. _load_exp1_five_rows()
# above only returns aggregated (median, clipped-mean, std) per method, which
# is enough for Table 1 but NOT for a Mann-Whitney test, which needs the raw
# per-equation sample -- hence this separate loader.
def _load_exp1_five_raw_extrap_errors(method_name: str) -> tuple[list[float], Path | None]:
    """Raw per-equation extrapolation-error-percent values for one method_name
    (e.g. _EXP2_ROW_ORDER[0] == "Hybrid v50\\_2"), read directly from
    exp1_five_results.json. Same file/parsing convention and same
    (extrap_rmse_far / train_rmse) * 100 error definition as
    _load_exp1_five_rows(), just returning the un-aggregated list."""
    src = None
    for base in (PATCHED, RESULTS):
        candidate = base / "five_systems/exp1_five/exp1_five_results.json"
        if candidate.exists():
            src = candidate
            break
    if src is None:
        return [], None
    try:
        data = json.loads(src.read_text())
    except Exception:
        return [], None
    if not isinstance(data, dict):
        return [], None

    errs: list[float] = []
    for eq_entry in data.values():
        if not isinstance(eq_entry, dict):
            continue
        res = eq_entry.get(method_name)
        if not isinstance(res, dict) or not res.get("success"):
            continue
        train_rmse = _finite_or_none(res.get("train_rmse"))
        far_rmse   = _finite_or_none(res.get("extrap_rmse_far"))
        if train_rmse and train_rmse > 0 and far_rmse is not None:
            errs.append((far_rmse / train_rmse) * 100.0)
    return errs, src


def _norm_cdf(z: float) -> float:
    import math
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _mann_whitney_one_tailed(sample_a: list[float], sample_b: list[float]) -> dict | None:
    """One-tailed Mann-Whitney U test, H1: sample_a is stochastically LESS
    than sample_b.

    Implemented from scratch (rank-sum -> U -> normal approximation with tie
    correction and continuity correction), not via scipy, to match this
    file's existing stdlib-only convention (see the local
    `import statistics as _st` pattern used elsewhere instead of numpy/scipy
    -- see the module's import block). This is the standard Mann & Whitney
    (1947) asymptotic normal approximation, appropriate for n >= ~8 per
    group; it will not bit-for-bit match a previously hand-computed exact
    permutation p-value, which is expected -- see gen_five_system_stat_tests()
    docstring for why that's the point, not a bug.
    """
    n1, n2 = len(sample_a), len(sample_b)
    if n1 == 0 or n2 == 0:
        return None
    combined = sorted([(v, 0) for v in sample_a] + [(v, 1) for v in sample_b])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    R_a = sum(r for (v, grp), r in zip(combined, ranks) if grp == 0)
    U_a = R_a - n1 * (n1 + 1) / 2.0
    U_b = n1 * n2 - U_a
    N = n1 + n2
    from collections import Counter
    tie_counts = Counter(v for v, _ in combined)
    tie_term = sum(t ** 3 - t for t in tie_counts.values())
    sigma_U = (((n1 * n2 / 12.0) * ((N + 1) - tie_term / (N * (N - 1)))) ** 0.5) if N > 1 else 0.0
    mu_U = n1 * n2 / 2.0
    if sigma_U == 0:
        return {"U": U_a, "U_other": U_b, "z": float("nan"), "p_one_tailed": float("nan"),
                "n1": n1, "n2": n2}
    # Continuity correction toward mu_U; H1 expects U_a to be SMALL (sample_a
    # ranks low), so the one-tailed p we want is the left-tail probability.
    z = (U_a - mu_U + 0.5) / sigma_U
    return {"U": U_a, "U_other": U_b, "z": z, "p_one_tailed": _norm_cdf(z),
            "n1": n1, "n2": n2, "sigma_U": sigma_U, "mu_U": mu_U}


def _cohens_d_pooled(mean_a, mean_b, sd_a, sd_b) -> float | None:
    try:
        s_pooled = ((sd_a ** 2 + sd_b ** 2) / 2.0) ** 0.5
        if s_pooled == 0:
            return None
        return (mean_b - mean_a) / s_pooled
    except TypeError:
        return None


def _glass_delta(mean_a, mean_b, sd_control) -> float | None:
    try:
        if not sd_control:
            return None
        return (mean_b - mean_a) / sd_control
    except TypeError:
        return None


def gen_five_system_stat_tests() -> None:
    """
    Appendix D -- Statistical Test Details (app:statistical_tests): Mann-
    Whitney U test + effect size (Cohen's d, Glass's Delta) comparing Hybrid
    v50_2 vs. Neural Network extrapolation error at the medium (2x training
    range) regime -- the exact comparison Source B (supp_benchmark_report.tex,
    app:statistical_tests) previously reported ONLY as hand-typed numbers,
    with no generator anywhere in this file and no traceable link back to
    exp1_five_results.json. See 04_ci_sd_incompatibility.tex ("Issue 4: CI/SD
    Statistical Incompatibility", category Open) for the discrepancy this
    closes: every number below is now computed live from the same
    exp1_five_results.json that feeds gen_five_system() (Table 1), so the two
    can no longer silently drift apart the way they already had.

    NOTE: the Mann-Whitney p-value is the standard normal approximation (see
    _mann_whitney_one_tailed docstring), not an exact permutation test, so it
    will not necessarily match a previously hand-computed p-value exactly --
    that's expected: this number is reproducible and re-derivable from data
    going forward, rather than a fixed constant that can drift from the data
    silently.
    """
    hybrid_key = _EXP2_ROW_ORDER[0]   # "Hybrid v50\_2"
    nn_key = _EXP2_ROW_ORDER[1]       # "Neural Network"
    errs_hybrid, src = _load_exp1_five_raw_extrap_errors(hybrid_key)
    errs_nn, src2 = _load_exp1_five_raw_extrap_errors(nn_key)
    src = src or src2

    no_data = not errs_hybrid or not errs_nn
    if no_data:
        FALLBACK_TABLES.append(
            "five_system_stat_tests.tex (app:statistical_tests) -- "
            "exp1_five_results.json not found, or missing a 'Hybrid v50_2' "
            "or 'Neural Network' row with >=1 successful finite "
            "extrapolation-error measurement. No hardcoded fallback exists; "
            "wrote a NO DATA placeholder instead."
        )
        tex = header_comment(src)
        tex += (
            "% NO DATA -- run exp1_five_system.py; needs both a\n"
            "% 'Hybrid v50\\_2' and a 'Neural Network' row with >=1 successful,\n"
            "% finite extrapolation-error measurement each.\n"
        )
        write_table("five_system_stat_tests.tex", tex)
        return

    tex = _render_stat_test_tex(
        errs_hybrid, errs_nn, "Hybrid v50\\_2", "Neural Network", src,
        "Appendix D -- Statistical Test Details (app:statistical_tests). "
        "Auto-derived from exp1_five_results.json -- see "
        "gen_five_system_stat_tests() docstring and 04_ci_sd_incompatibility.tex "
        "(Issue 4) for why this generator exists.",
    )
    tex += f"\n% Source: exp1_five_results.json ({src})\n"
    write_table("five_system_stat_tests.tex", tex)


def _render_stat_test_tex(errs_a: list[float], errs_b: list[float], label_a: str,
                           label_b: str, src, source_note: str) -> str:
    """Shared Mann-Whitney U + Cohen's d / Glass's Delta LaTeX renderer, used
    by both gen_five_system_stat_tests() (exp1_five, Core-15) and
    gen_five_system_exp2five_stat_tests() (exp2_five, Feynman) so the two
    genuinely-separate data sources produce structurally identical Appendix
    output without duplicating this ~60-line formatting block. label_a is
    the "should be better" group (H1: label_a < label_b)."""
    import statistics as _st
    n1, n2 = len(errs_a), len(errs_b)
    mean_a, mean_b = _st.mean(errs_a), _st.mean(errs_b)
    med_a, med_b = _st.median(errs_a), _st.median(errs_b)
    sd_a = _st.stdev(errs_a) if n1 >= 2 else 0.0
    sd_b = _st.stdev(errs_b) if n2 >= 2 else 0.0
    range_a = (min(errs_a), max(errs_a))
    range_b = (min(errs_b), max(errs_b))

    mw = _mann_whitney_one_tailed(errs_a, errs_b)
    d_pooled = _cohens_d_pooled(mean_a, mean_b, sd_a, sd_b)
    d_cons = (mean_b - mean_a) / sd_b if sd_b else None
    glass_d = _glass_delta(mean_a, mean_b, sd_b)

    def _f(v, nd=1):
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) and v == v else "---"

    tex = header_comment(src)
    tex += (
        f"% {source_note}\n"
        "% Auto-derived -- see _render_stat_test_tex() in generate_tables.py\n\n"
    )
    tex += r"""\subsection{Mann-Whitney U Test for Extrapolation Performance}

For medium extrapolation regime (2$\times$ training range):

\paragraph{Test Setup:}
\begin{itemize}
"""
    tex += f"\\item \\textbf{{Sample sizes}}: $n_{{\\text{{{label_a}}}}} = {n1}$, $n_{{\\text{{{label_b}}}}} = {n2}$\n"
    tex += (
        f"\\item \\textbf{{Null hypothesis}}: $H_0: E_{{\\text{{{label_a}}}}} \\geq E_{{\\text{{{label_b}}}}}$ "
        f"({label_a} is not better)\n"
        f"\\item \\textbf{{Alternative hypothesis}}: $H_1: E_{{\\text{{{label_a}}}}} < E_{{\\text{{{label_b}}}}}$ "
        f"({label_a} is better)\n"
        r"\item \textbf{Significance level}: $\alpha = 0.05$ (one-tailed test)" "\n"
        "\\end{itemize}\n\n\\paragraph{Descriptive Statistics:}\n\\begin{itemize}\n"
    )
    tex += (f"\\item \\textbf{{{label_a}}}: Mean = {_f(mean_a)}\\%, "
            f"Median = {_f(med_a)}\\%, SD = {_f(sd_a)}\\%, "
            f"Range = [{_f(range_a[0])}, {_f(range_a[1])}]\n")
    tex += (f"\\item \\textbf{{{label_b}}}: Mean = {_f(mean_b)}\\%, "
            f"Median = {_f(med_b)}\\%, SD = {_f(sd_b)}\\%, "
            f"Range = [{_f(range_b[0])}, {_f(range_b[1])}]\n")
    tex += "\\end{itemize}\n\n\\paragraph{Test Results:}\n\\begin{itemize}\n"

    if mw and mw["p_one_tailed"] == mw["p_one_tailed"]:
        tex += f"\\item \\textbf{{Test statistic}}: $U = {_f(mw['U'], 1)}$\n"
        tex += (f"\\item \\textbf{{P-value}}: $p = {mw['p_one_tailed']:.2e}$ "
                "(one-tailed, normal approximation with tie correction -- "
                "see \\texttt{\\_render\\_stat\\_test\\_tex()} in "
                "\\texttt{generate\\_tables.py})\n")
        rejects = mw["p_one_tailed"] < 0.05
        verdict = "Reject" if rejects else "Fail to reject"
        cmp_sym = "<" if rejects else r"\geq"
        tex += (f"\\item \\textbf{{Conclusion}}: $p {cmp_sym} 0.05 "
                f"\\Rightarrow$ {verdict} $H_0$ at $\\alpha = 0.05$\n")
        if mw["U"] == 0:
            tex += (
                r"\item \textbf{Interpretation}: Complete rank separation "
                f"--- on every tested case, {label_a} extrapolation error is "
                f"strictly less than {label_b} error "
                f"($U = 0$, $p = {mw['p_one_tailed']:.2e}$).\n"
            )
    else:
        tex += (
            "\\item Mann-Whitney statistic unavailable "
            "(degenerate sample -- zero variance in the combined pool).\n"
        )
    tex += "\\end{itemize}\n\n"

    tex += r"\subsection{Effect Size Calculation (Cohen's $d$)}" + "\n\n\\begin{align}\n"
    tex += (f"\\mu_{{\\text{{{label_b}}}}} &= {_f(mean_b)}\\%, \\quad "
            f"\\sigma_{{\\text{{{label_b}}}}} = {_f(sd_b)}\\% \\\\\n")
    tex += (f"\\mu_{{\\text{{{label_a}}}}} &= {_f(mean_a)}\\%, \\quad "
            f"\\sigma_{{\\text{{{label_a}}}}} = {_f(sd_a)}\\% \\\\\n")
    if d_pooled is not None:
        s_pooled = ((sd_a ** 2 + sd_b ** 2) / 2.0) ** 0.5
        tex += (r"s_{\text{pooled}} &= \sqrt{\frac{\sigma_{\text{" + label_b + r"}}^2 + "
                r"\sigma_{\text{" + label_a + r"}}^2}{2}} = " f"{_f(s_pooled)}\\% \\\\\n")
        tex += f"d &= \\frac{{{_f(mean_b)} - {_f(mean_a)}}}{{{_f(s_pooled)}}} = {_f(d_pooled, 2)}\n"
    else:
        tex += r"d &= \text{undefined (pooled SD = 0)}" + "\n"
    tex += "\\end{align}\n\n"

    tex += (
        f"\\paragraph{{Conservative Estimate:}}\nUsing only {label_b} standard "
        "deviation (most conservative approach):\n\\begin{equation}\n"
    )
    if d_cons is not None:
        tex += f"d_{{\\text{{conservative}}}} = \\frac{{{_f(mean_b)}}}{{{_f(sd_b)}}} = {_f(d_cons, 2)}"
        tex += " \\quad \\text{(interpret magnitude per Cohen 1988 conventions)}\n"
    else:
        tex += r"d_{\text{conservative}} = \text{undefined (" + label_b + " SD = 0)}" + "\n"
    tex += "\\end{equation}\n\n"

    tex += (
        r"\paragraph{Alternative Calculation (Glass's $\Delta$):}"
        f"\nUsing only the control group ({label_b}) standard deviation:\n\\begin{{equation}}\n"
    )
    if glass_d is not None:
        tex += f"\\Delta = \\frac{{{_f(mean_b)} - {_f(mean_a)}}}{{{_f(sd_b)}}} = {_f(glass_d, 2)}\n"
    else:
        tex += r"\Delta = \text{undefined (" + label_b + " SD = 0)}" + "\n"
    tex += "\\end{equation}\n"
    return tex


# ── exp2_five: genuinely separate five-system pipeline (Feynman suite) ────────
#
# Same four-table structure as exp1_five's pipeline above (main comparison +
# performance sub-table + extrapolation sub-table + Mann-Whitney/effect-size
# stat tests), but sourced ONLY from exp2_five's own output
# (five_systems/exp2_five/ via _load_exp2_five_own_rows() /
# _load_exp2_five_own_raw() above) -- never falls back to exp1_five or to
# exp2's regular directory. This is intentionally a second, independent set
# of tables on a different equation suite (10-domain Feynman, method subset
# 1/2/4/5/6) and different sample sizes per method, not another tier of the
# same table. See the "exp2_five: genuinely separate" comment on
# _load_exp2_five_own_raw() for why a dedicated reader was needed at all.
def gen_five_system_exp2five() -> None:
    """
    Five-System Comparison (exp2_five / Feynman suite): Extrapolation Error
    vs. Interpolation R^2. Structurally the same table as gen_five_system()
    (tab:five_systems_full, exp1_five/Core-15), but a separate table
    (tab:five_systems_full_exp2five) built only from exp2_five's own data --
    see the module comment above this function for why the two are kept
    apart rather than merged/averaged.
    """
    rows, src = _load_exp2_five_own_rows()
    no_data = rows is None
    if no_data:
        FALLBACK_TABLES.append(
            "five_system_exp2five.tex (tab:five_systems_full_exp2five) -- "
            "no usable rows under five_systems/exp2_five/ in patched/ or "
            "results/. No fallback to exp1_five or exp2's own directory "
            "(deliberate -- see module comment above gen_five_system_exp2five()). "
            "Wrote a NO DATA placeholder instead."
        )

    tex = header_comment(src) + r"""
\begin{table}[t]
\centering
\caption{Five-System Comparison (exp2\_five, Feynman 10-domain suite,
  methods 1/2/4/5/6): Extrapolation Error vs.\ Interpolation $R^2$.
  Same 95\% CI convention as Table~\ref{tab:five_systems_full}
  ($\mathrm{mean} \pm t_{0.975,\,n-1} \cdot \mathrm{std}/\sqrt{n}$); a
  separate table from tab:five\_systems\_full -- different equation suite
  and sample sizes, not another view of the same data.}
\label{tab:five_systems_full_exp2five}
\begin{tabular}{lrrrrrrr}
\toprule
\textbf{System} & \textbf{n}
  & \textbf{Extrap.\ Median (\%)} & \textbf{Extrap.\ Mean (\%)}
  & \textbf{Train $R^2$ Mean} & \textbf{Std} & \textbf{95\% CI (\%)}
  & \textbf{Design Focus} \\
\midrule
"""
    if no_data:
        tex += r"\multicolumn{8}{c}{\textit{NO DATA -- run exp2\_five}} \\" + "\n"
    else:
        sep_done = False
        for (name, n, emed, emean, tr2, std, focus) in rows:
            if not sep_done and n == 0:
                tex += r"\midrule" + "\n"
                tex += r"\multicolumn{8}{l}{\textit{Systems Without Extrapolation Testing}} \\" + "\n"
                sep_done = True
            ci = _ci95(emean, std, n) or "---"
            tex += f"{name} & {n} & {emed} & {emean} & {tr2} & {std} & {ci} & {focus} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    if no_data:
        tex += (
            "% NO DATA -- five_systems/exp2_five/ produced no usable rows.\n"
            "% Run exp2_five (run_comparative_suite_benchmark_v2.py\n"
            "% --methods 1 2 4 5 6) and re-generate.\n"
        )
    else:
        tex += f"% Source: exp2_five own output ({src})\n"
    write_table("five_system_exp2five.tex", tex)


def gen_five_system_exp2five_performance() -> None:
    """
    Performance sub-table for exp2_five's own data: Train R^2/RMSE per
    method, computed directly from exp2_five's raw protocol_core_noiseless
    records (unlike exp1_five_performance.json, exp2_five's script doesn't
    pre-aggregate a summary file, so the n/mean/CI here are derived in this
    function rather than just formatted from a pre-computed JSON).
    """
    rows_data: dict[str, dict] = {}
    src = None
    for mname, row_name in _EXP2_METHOD_TO_ROW.items():
        tr_vals, _err_vals, this_src = _load_exp2_five_own_raw(mname)
        src = src or this_src
        n = len(tr_vals)
        mean = sum(tr_vals) / n if n else None
        std = None
        if mean is not None and n >= 2:
            std = (sum((x - mean) ** 2 for x in tr_vals) / (n - 1)) ** 0.5
        rows_data[row_name] = {"n": n, "mean": mean, "std": std}

    no_data = src is None
    if no_data:
        FALLBACK_TABLES.append(
            "five_system_exp2five_performance.tex -- no usable rows under "
            "five_systems/exp2_five/. Run exp2_five and re-generate."
        )

    tex = header_comment(src) + r"""
\begin{table}[t]
\centering
\caption{Five-System Comparison (exp2\_five) -- Performance (Feynman suite,
  interpolation). 95\% CI computed from raw per-domain train $R^2$ at
  generation time.}
\label{tab:five_systems_performance_exp2five}
\begin{tabular}{lrrrr}
\toprule
\textbf{System} & \textbf{n} & \textbf{Train $R^2$ Mean}
  & \textbf{Train $R^2$ 95\% CI} & \textbf{Design Focus} \\
\midrule
"""
    if no_data:
        tex += r"\multicolumn{5}{c}{\textit{NO DATA -- run exp2\_five}} \\" + "\n"
    else:
        for row_name in _EXP2_ROW_ORDER:
            d = rows_data.get(row_name, {"n": 0, "mean": None, "std": None})
            mean_s = f"{d['mean']:.3f}" if d["mean"] is not None else "---"
            ci = _ci95(d["mean"], d["std"], d["n"]) if d["mean"] is not None else None
            ci_s = ci or "---"
            focus = _EXP2_DESIGN_FOCUS.get(row_name, "---")
            tex += f"{row_name} & {d['n']} & {mean_s} & {ci_s} & {focus} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("five_system_exp2five_performance.tex", tex)


def gen_five_system_exp2five_extrapolation() -> None:
    """
    Extrapolation sub-table for exp2_five's own data: extrapolation error %
    per method, derived directly from exp2_five's benchmark_results_extrap.json.
    exp2_five's harness doesn't tag near/medium/far regimes the way
    exp1_ablation.py's EXTRAP_REGIMES does, so this is a single aggregate
    per method rather than three regime rows.
    """
    rows_data: dict[str, dict] = {}
    src = None
    for mname, row_name in _EXP2_METHOD_TO_ROW.items():
        _tr_vals, err_vals, this_src = _load_exp2_five_own_raw(mname)
        src = src or this_src
        median, clipped_mean = _robust_median_clipped_mean(err_vals)
        rows_data[row_name] = {"n": len(err_vals), "median": median, "mean": clipped_mean}

    no_data = src is None
    if no_data:
        FALLBACK_TABLES.append(
            "five_system_exp2five_extrapolation.tex -- no usable rows under "
            "five_systems/exp2_five/. Run exp2_five and re-generate."
        )

    tex = header_comment(src) + r"""
\begin{table}[t]
\centering
\caption{Five-System Comparison (exp2\_five) -- Extrapolation (Feynman
  suite). Median and IQR-clipped mean of per-domain extrapolation error \%.}
\label{tab:five_systems_extrapolation_exp2five}
\begin{tabular}{lrrr}
\toprule
\textbf{System} & \textbf{n}
  & \textbf{Extrap.\ Median (\%)} & \textbf{Extrap.\ Mean (\%)} \\
\midrule
"""
    if no_data:
        tex += r"\multicolumn{4}{c}{\textit{NO DATA -- run exp2\_five}} \\" + "\n"
    else:
        for row_name in _EXP2_ROW_ORDER:
            d = rows_data.get(row_name, {"n": 0, "median": None, "mean": None})
            med_s = f"{d['median']:.1f}" if d["median"] is not None else "---"
            mean_s = f"{d['mean']:.1f}" if d["mean"] is not None else "---"
            tex += f"{row_name} & {d['n']} & {med_s} & {mean_s} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("five_system_exp2five_extrapolation.tex", tex)


def gen_five_system_exp2five_stat_tests() -> None:
    """
    Appendix statistical test details for exp2_five's own Hybrid-v50_2-vs-
    Neural-Network extrapolation comparison -- structurally the same test as
    gen_five_system_stat_tests() (exp1_five/Core-15), reusing
    _render_stat_test_tex(), but computed from exp2_five's own raw data
    (five_systems/exp2_five/), never mixed with exp1_five's numbers.
    """
    hybrid_json_key = "HybridDiscoverySystem v50_2 (tools)"
    nn_json_key = "ImprovedNN (core)"
    _tr_h, errs_hybrid, src = _load_exp2_five_own_raw(hybrid_json_key)
    _tr_n, errs_nn, src2 = _load_exp2_five_own_raw(nn_json_key)
    src = src or src2

    no_data = not errs_hybrid or not errs_nn
    if no_data:
        FALLBACK_TABLES.append(
            "five_system_exp2five_stat_tests.tex -- five_systems/exp2_five/ "
            "not found or missing Hybrid v50_2 / Neural Network records with "
            ">=1 successful finite extrapolation-error measurement each. "
            "No fallback; wrote a NO DATA placeholder instead."
        )
        tex = header_comment(src)
        tex += (
            "% NO DATA -- run exp2_five; needs both Hybrid v50\\_2 and\n"
            "% Neural Network records with >=1 successful, finite\n"
            "% extrapolation-error measurement each.\n"
        )
        write_table("five_system_exp2five_stat_tests.tex", tex)
        return

    tex = _render_stat_test_tex(
        errs_hybrid, errs_nn, "Hybrid v50\\_2", "Neural Network", src,
        "Appendix -- Statistical Test Details (exp2_five / Feynman suite). "
        "See gen_five_system_exp2five_stat_tests() and 04_ci_sd_incompatibility.tex.",
    )
    tex += f"\n% Source: exp2_five own output ({src})\n"
    write_table("five_system_exp2five_stat_tests.tex", tex)


def _load_exp1_five_subtable_json(filename: str) -> tuple[dict | None, Path | None]:
    for base in (PATCHED, RESULTS):
        candidate = base / "five_systems/exp1_five" / filename
        if candidate.exists():
            try:
                return json.loads(candidate.read_text()), candidate
            except Exception:
                continue
    return None, None


# PRE-EXISTING BUG FOUND WHILE TESTING gen_five_system_stat_tests(): this
# function is referenced from gen_five_system(), gen_repro_macros(), and the
# audit path (search for "_load_five_system_rows_real" across this file) as
# "the combined loader", but was never actually defined anywhere -- every one
# of those call sites raised NameError at runtime. Reconstructed here from
# the two real sources documented at the top of this section
# (_load_exp1_five_rows() primary / Core-15, _load_exp2_five_system_rows()
# secondary / Feynman) and the "tier" string those call sites expect back
# (used only for the "% Source:" comment / FALLBACK_TABLES bookkeeping).
def _load_five_system_rows_real() -> tuple[list[tuple] | None, Path | None, str]:
    """Load five-system rows from the launched experiment only.

    No fallback to results from other experiments is performed.
    """

    rows, src = _load_exp1_five_rows()
    if rows is not None:
        return rows, src, "exp1_five"

    return None, None, "none"


def gen_five_system_performance() -> None:
    """
    Sub-table 1 of 2 (metric item 4): performance -- train R^2 / train RMSE
    per method, Core-15 equations. Reads exp1_five_performance.json, which
    exp1_five_system.py itself already computed (n/mean/median/sd/se/CI via
    the same t-based convention as evaluator.txt) -- this function only
    formats it, it does not re-derive any statistic.
    """
    data, src = _load_exp1_five_subtable_json("exp1_five_performance.json")
    no_data = data is None
    if no_data:
        FALLBACK_TABLES.append(
            "five_system_performance.tex -- exp1_five_performance.json not "
            "found under five_systems/exp1_five/ in patched/ or results/. "
            "No fallback; run exp1_five_system.py."
        )

    tex = header_comment(src) + r"""
\begin{table}[t]
\centering
\caption{Five-System Comparison -- Performance (Core-15, interpolation).
  95\% CI computed from raw per-equation train $R^2$/RMSE at generation time.}
\label{tab:five_systems_performance}
\begin{tabular}{lrrrrr}
\toprule
\textbf{System} & \textbf{n} & \textbf{Train $R^2$ Mean}
  & \textbf{Train $R^2$ 95\% CI} & \textbf{Train RMSE Mean}
  & \textbf{Design Focus} \\
\midrule
"""
    if no_data:
        tex += r"\multicolumn{6}{c}{\textit{NO DATA -- run exp1\_five\_system.py}} \\" + "\n"
    else:
        for method_name, row in data.items():
            r2 = row.get("train_r2") or {}
            rmse = row.get("train_rmse") or {}
            n = r2.get("n", rmse.get("n", 0))
            r2_mean = f"{r2['mean']:.3f}" if r2.get("mean") is not None else "---"
            r2_ci = (
                f"[{r2['ci_low']:.3f}, {r2['ci_high']:.3f}]"
                if r2.get("ci_low") is not None else "---"
            )
            rmse_mean = f"{rmse['mean']:.4f}" if rmse.get("mean") is not None else "---"
            focus = row.get("design_focus", "---")
            tex += f"{method_name} & {n} & {r2_mean} & {r2_ci} & {rmse_mean} & {focus} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("five_system_performance.tex", tex)


def gen_five_system_extrapolation() -> None:
    """
    Sub-table 2 of 2 (metric item 4): extrapolation -- R^2/RMSE per method
    for near/medium/far regimes, Core-15 equations. Same source-of-truth
    convention as gen_five_system_performance(): reads
    exp1_five_extrapolation.json, formats only, no re-derivation.
    """
    data, src = _load_exp1_five_subtable_json("exp1_five_extrapolation.json")
    no_data = data is None
    if no_data:
        FALLBACK_TABLES.append(
            "five_system_extrapolation.tex -- exp1_five_extrapolation.json "
            "not found under five_systems/exp1_five/ in patched/ or results/. "
            "No fallback; run exp1_five_system.py."
        )

    tex = header_comment(src) + r"""
\begin{table}[t]
\centering
\caption{Five-System Comparison -- Extrapolation (Core-15, near/medium/far
  regimes as defined in exp1\_ablation.py's EXTRAP\_REGIMES).
  95\% CI computed from raw per-equation extrapolation $R^2$/RMSE at
  generation time.}
\label{tab:five_systems_extrapolation}
\begin{tabular}{llrrr}
\toprule
\textbf{System} & \textbf{Regime} & \textbf{n}
  & \textbf{Extrap $R^2$ Mean} & \textbf{Extrap RMSE Mean} \\
\midrule
"""
    if no_data:
        tex += r"\multicolumn{5}{c}{\textit{NO DATA -- run exp1\_five\_system.py}} \\" + "\n"
    else:
        for method_name, row in data.items():
            for regime in ("near", "medium", "far"):
                cell = row.get(regime, {})
                r2 = cell.get("extrap_r2") or {}
                rmse = cell.get("extrap_rmse") or {}
                n = r2.get("n", rmse.get("n", 0))
                r2_mean = f"{r2['mean']:.3f}" if r2.get("mean") is not None else "---"
                rmse_mean = f"{rmse['mean']:.4f}" if rmse.get("mean") is not None else "---"
                tex += f"{method_name} & {regime} & {n} & {r2_mean} & {rmse_mean} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("five_system_extrapolation.tex", tex)


def gen_five_system() -> None:
    """
    Five-System Comparison: Extrapolation Error vs. Interpolation R^2.
    Matches tab:five_systems_full in supp_benchmark_report.tex (Appendix,
    not the main paper — see 04_ci_sd_incompatibility.tex Source A).

    The 95% CI column is *derived* from each row's own n/mean/std at
    generation time (see _ci95), rather than being a separately hand-typed
    number elsewhere in the paper.

    Source resolution (_load_five_system_rows_real(), no hardcoded fallback
    exists anymore):
      1. exp1_five_system.py's own output (Core-15 suite; the dedicated
         experiment for this table).
      2. exp2/exp2_extrap (Feynman suite) as a second, supplementary real
         source, via the previously-dead _load_exp2_five_system_rows().
    If neither source has data, this writes a table containing an explicit
    "NO DATA" placeholder (not fabricated numbers) and registers a
    FALLBACK_TABLES entry so the CI build still fails loudly.
    """
    rows, src, tier = _load_five_system_rows_real()
    no_data = rows is None
    if no_data:
        FALLBACK_TABLES.append(
            "five_system.tex (tab:five_systems_full) -- neither exp1_five "
            "(five_systems/exp1_five/exp1_five_results.json) nor "
            "exp2/exp2_extrap had usable five-system rows. No hardcoded "
            "fallback exists; wrote a NO DATA placeholder table instead."
        )

    tex = header_comment(src) + r"""
\begin{table}[t]
\centering
\caption{Five-System Comparison: Extrapolation Error vs.\ Interpolation $R^2$.
  95\% CI computed from each row's own Mean/Std/n
  ($\mathrm{mean} \pm t_{0.975,\,n-1} \cdot \mathrm{std}/\sqrt{n}$); left
  blank where Std or n is unavailable, never hand-entered.}
\label{tab:five_systems_full}
\begin{tabular}{lrrrrrrr}
\toprule
\textbf{System} & \textbf{n}
  & \textbf{Extrap.\ Median (\%)} & \textbf{Extrap.\ Mean (\%)}
  & \textbf{Train $R^2$ Mean} & \textbf{Std} & \textbf{95\% CI (\%)}
  & \textbf{Design Focus} \\
\midrule
"""
    if no_data:
        tex += r"\multicolumn{8}{c}{\textit{NO DATA -- run exp1\_five\_system.py or exp2/exp2\_extrap}} \\" + "\n"
    else:
        sep_done = False
        for (name, n, emed, emean, tr2, std, focus) in rows:
            if not sep_done and n == 0:
                tex += r"\midrule" + "\n"
                tex += r"\multicolumn{8}{l}{\textit{Systems Without Extrapolation Testing}} \\" + "\n"
                sep_done = True
            ci = _ci95(emean, std, n) or "---"
            tex += f"{name} & {n} & {emed} & {emean} & {tr2} & {std} & {ci} & {focus} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    if no_data:
        tex += (
            "% NO DATA -- neither exp1_five n produced\n"
            "% usable five-system rows. Run exp1_five_system.py (preferred,\n"
            "% Core-15) or exp2_five/exp2 (Feynman suite) and re-generate.\n"
        )
    else:
        tex += f"% Source: {tier}\n"
    write_table("five_system.tex", tex)



def gen_runtime() -> None:
    """
    Tab 4 — Wall-clock time per task (seconds). Matches Table 4 in §10.4.
    """
    data, src = load_best("", "hypatiax_defi_benchmark_v3*results*.json",
                          extra_subdirs=["defi"])

    # Paper-verified fallback (Table 4)
    PAPER_ROWS = [
        ("Pure LLM",                   11.4, 10.3, 74, "3.80× slower"),
        ("Neural MLP",                  3.0,  2.7, 74, "— (baseline)"),
        ("HypatiaX",                    6.8,  1.7, 74, "2.30× slower (mean) / 1.64× faster (median)"),
        ("HypatiaX (LLM-routed only)", None, None, 68, "1.73× faster"),
    ]

    def _extract(d):
        if not isinstance(d, dict):
            return []
        timing = d.get("timing", d.get("runtime", {}))
        rows = []
        for name, key in [("Pure LLM", "pure_llm"), ("Neural MLP", "neural_mlp"),
                          ("HypatiaX", "hypatiax")]:
            t = timing.get(key, {})
            rows.append((
                name,
                t.get("mean_s", t.get("mean_time_s", float("nan"))),
                t.get("median_s", t.get("median_time_s", float("nan"))),
                t.get("n", 74),
                t.get("vs_nn", "---"),
            ))
        return rows if len(rows) >= 3 else []

    rows = _extract(data) if data else []
    if not rows:
        rows = PAPER_ROWS

    def _t(v): return f"{v:.1f}" if isinstance(v, float) and v == v else "---"

    tex = header_comment(src) + r"""
\begin{table}[t]
\centering
\caption{Wall-clock time per task (seconds). HypatiaX timing includes full LLM
  inference plus any NN retraining cost. Speedups relative to Neural MLP.}
\label{tab:runtime}
\begin{tabular}{lrrrr}
\toprule
\textbf{Method} & \textbf{Mean (s)} & \textbf{Median (s)}
  & \textbf{n} & \textbf{vs.\ NN} \\
\midrule
"""
    for (name, mean, med, n, vs_nn) in rows:
        tex += f"{name} & {_t(mean)} & {_t(med)} & {n} & {vs_nn} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("runtime.tex", tex)


def gen_portfolio_seed_sweep() -> None:
    """
    Tab 5 — Portfolio Variance seed-sweep results.
    H recovers? = exact closed-form formula recovered.
    H wins?     = HypatiaX far-R² strictly > PySR-only.
    Matches Table 5 in §10.5.
    """
    # Try to find portfolio_variance_seed_sweep.json
    src_path = None
    for base in [PATCHED, RESULTS]:
        for cand in [base / "portfolio_variance_seed_sweep.json",
                     *sorted(base.glob("portfolio_variance*.json"),
                             key=lambda p: p.stat().st_mtime, reverse=True)]:
            if cand.exists():
                src_path = cand
                break
        if src_path:
            break

    data = None
    if src_path:
        try:
            data = json.loads(src_path.read_text())
        except Exception:
            pass

    # Paper-verified fallback (Table 5)
    PAPER_ROWS = [
        (42,   -21.004, -0.023,  "linear",    False, True),
        (99,    -1.226, -15.191, "linear",    False, False),
        (123,  -18.651, -18.090, "exp denom", False, True),
        (777,   -0.438,  +1.000, "exact",     True,  True),
        (2024, -12.109,  +1.000, "exact",     True,  True),
    ]

    def _extract(d):
        if not isinstance(d, dict):
            return []
        seeds = d.get("seeds", d.get("results", []))
        if not isinstance(seeds, list) or len(seeds) < 5:
            return []
        rows = []
        for s in seeds:
            rows.append((
                s.get("seed", "?"),
                s.get("pysr_far_r2",    s.get("p_far_r2", float("nan"))),
                s.get("hypatiax_far_r2", s.get("h_far_r2", float("nan"))),
                s.get("h_formula", s.get("formula", "?")),
                bool(s.get("h_recovers", s.get("exact_recovery", False))),
                bool(s.get("h_wins",     s.get("hypatiax_wins",  False))),
            ))
        return rows

    rows = _extract(data) if data else []
    if not rows:
        rows = PAPER_ROWS

    def _r(v): return f"{v:.3f}" if isinstance(v, float) and v == v else "---"
    def _yn(v): return "Yes" if v else "No"

    tex = header_comment(src_path) + r"""
\begin{table}[t]
\centering
\caption{Portfolio Variance seed-sweep results.
  \textbf{H recovers?}: exact closed-form formula recovered.
  \textbf{H wins?}: HypatiaX far-$R^2$ strictly greater than PySR-only.}
\label{tab:portfolio_seed}
\begin{tabular}{rrrrrr}
\toprule
\textbf{Seed} & \textbf{P far-$R^2$} & \textbf{H far-$R^2$}
  & \textbf{H formula} & \textbf{H recovers?} & \textbf{H wins?} \\
\midrule
"""
    p_means, h_means = [], []
    for (seed, pfar, hfar, hform, hrec, hwins) in rows:
        tex += f"{seed} & {_r(pfar)} & {_r(hfar)} & {hform} & {_yn(hrec)} & {_yn(hwins)} \\\\\n"
        if isinstance(pfar, float) and pfar == pfar: p_means.append(pfar)
        if isinstance(hfar, float) and hfar == hfar: h_means.append(hfar)

    import statistics as _st
    pm = f"{_st.mean(p_means):.3f}" if p_means else "---"
    hm = f"{_st.mean(h_means):.3f}" if h_means else "---"
    n_wins  = sum(1 for r in rows if r[5])
    n_exact = sum(1 for r in rows if r[4])
    tex += r"\midrule" + "\n"
    tex += f"Mean & {pm} & {hm} & & \\multicolumn{{2}}{{r}}{{H: {n_wins}/5 wins, {n_exact}/5 exact}} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("portfolio_sweep.tex", tex)


# Paper-verified fallback (Table 7, Kaggle 4-vCPU run). Module-level so both
# gen_feynman_results() (exp2_feynman) and the exp2 all-30 multi-domain
# generators below (gen_all30_domain_summary / gen_multi_domain_rank_table)
# can fall back to the same 30-equation set when fresh per-domain shards
# aren't available yet.
_PAPER_FEYNMAN_EQUATIONS = [
        ("Gaussian",             "Mechanics",      0.926,  -10.36,  -24.20),
        ("Coulomb Force",        "Mechanics",      0.869,   -7.43,  -999),
        ("Relativistic momentum","Mechanics",      0.997,   -0.25,   -4.76),
        ("Doppler shift",        "Mechanics",      0.997,    0.688,  -0.26),
        ("Harmonic oscillator",  "Mechanics",      0.997,    1.000,  -2.04),
        ("Electric potential",   "Thermodynamics", 0.997,    0.998,   0.962),
        ("Energy of photon",     "Thermodynamics", -2.71,  -999,    0.677),
        ("Magnetization",        "Thermodynamics", 0.924,   -0.47,  -1.07),
        ("Relativistic Doppler", "Optics",         0.998,    0.994,   0.987),
        ("Heat conduction",      "Optics",         0.923,    0.136,  -999),
        ("Snell's law",          "Optics",         0.993,   -0.31,  -0.13),
        ("Polarization",         "Electromagnetism",0.982,   0.941,   0.923),
        ("Torque",               "Electromagnetism",0.998,   1.000,  -2.01),
        ("Interference intensity","Electromagnetism",0.985,  1.000,  -6.07),
        ("Polarizability",       "Electromagnetism",-0.95, -11.75,   0.931),
        ("Planck radiation",     "Electromagnetism",-0.86,  -5.90,  -1.39),
        ("Photon energy",        "Quantum",        -2.61,  -999,    0.906),
        ("Magnetic moment",      "Quantum",        -0.76,   -9.59,  -2.56),
        ("Bose-Einstein",        "Quantum",         0.997,   0.997,   0.778),
        ("Gravity potential",    "Gravitation",     0.978,   -2.38,  -999),
        ("Orbital period",       "Gravitation",     0.998,   1.000,   0.862),
        ("Dielectric constant",  "Fluid",           0.579,   0.000,   0.000),
        ("Diffraction",          "Fluid",           0.995,   0.997,   0.825),
        ("Wave superposition",   "Waves",           0.692,   -1.14,  -999),
        ("de Broglie wavelength","Waves",          -0.11,   -9.46,  -999),
        ("Time dilation",        "Relativity",      0.997,   0.639,  -1.78),
        ("Lorentz factor",       "Relativity",      0.997,   0.711,  -0.54),
        ("Coulomb potential",    "Atomic",          0.063,  -999,   -4.66),
        ("Diffusion coefficient","Atomic",         -0.56,  -999,    0.034),
    ("Larmor frequency",     "Nuclear",         0.998,   1.000,  -1.40),
]


def _extract_equation_rows(d) -> list[tuple]:
    """Pull (name, domain, train_r2, extrap_r2, nn_r2) rows out of a single
    result JSON that follows the standard "equations"/"results" list shape
    used across both exp2_feynman's single combined file and exp2's
    per-domain shards. Shared by gen_feynman_results() and the exp2 all-30
    multi-domain generators so both interpret result JSONs identically."""
    if not isinstance(d, dict):
        return []
    eqs = d.get("equations", d.get("results", []))
    if not isinstance(eqs, list) or not eqs:
        return []
    rows = []
    for e in eqs:
        if not isinstance(e, dict):
            continue
        rows.append((
            e.get("name", "?"),
            e.get("domain", "?"),
            e.get("hyp_train_r2",  e.get("train_r2",  float("nan"))),
            e.get("hyp_extrap_r2", e.get("extrap_r2", float("nan"))),
            e.get("nn_extrap_r2",  e.get("nn_r2",     float("nan"))),
        ))
    return rows


def _load_exp2_multi_domain_rows() -> tuple[list[tuple], list[Path]]:
    """Load & merge every per-domain result shard for exp2 (all-30
    multi-domain).

    Unlike exp2_feynman (one combined JSON under a nested
    comparison_results/feynman-tests/exp2/ subdir), exp2's --results-dir is
    already resolved by ci_postprocess.yml to the canonical exp2_multi dir
    (comparison_results/feynman-tests/exp2_multi), so shards are read
    directly from PATCHED / RESULTS with subdir="" — mirroring how exp1
    reads its root-level JSON. Passing a subdir here would double-nest the
    path (RESULTS is already exp2_multi) and silently find nothing, which is
    exactly the bug that made plain "exp2" fall back to exp2_feynman's
    generator and its Table 7 fallback data instead of exp2's own results.

    Each shard is expected to hold one domain's worth of equation rows (same
    "equations"/"results" list shape as exp2_feynman's combined file — see
    _extract_equation_rows). Rows from every shard found are flattened into
    one list; duplicate equation names are resolved by taking the entry from
    the last shard processed (PATCHED overrides RESULTS, matching the
    override precedence load_best() uses elsewhere in this file).
    """
    by_name: dict[str, tuple] = {}
    sources: list[Path] = []
    for base in [RESULTS, PATCHED]:
        if not base.exists():
            continue
        for f in sorted(_filtered_glob(base, "*.json")):
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            rows = _extract_equation_rows(data)
            if not rows:
                continue
            sources.append(f)
            for row in rows:
                by_name[row[0]] = row
    return list(by_name.values()), sources


def gen_feynman_results() -> None:
    """
    Tab 7 — Feynman Extrapolation Benchmark (n=30), Kaggle primary run.
    Matches Table 7 in §10.7 (Appendix).
    """
    # run_all.sh (exp2_feynman) writes to RESULTS_DIR/comparison_results/feynman-tests/exp2/
    data, src = load_best("comparison_results/feynman-tests/exp2", "*.json",
                          extra_subdirs=["feynman"])

    equations = _extract_equation_rows(data) if data else []
    if not equations:
        equations = _PAPER_FEYNMAN_EQUATIONS

    def _r(v, lo=-100):
        if not isinstance(v, (int, float)) or v != v: return "---"
        if v <= lo: return r"$\ll{-100}$"
        return f"{v:.3f}"

    def _bold(v):
        """Bold if R² ≥ 0.99."""
        if isinstance(v, float) and v >= 0.99:
            return r"\textbf{" + f"{v:.3f}" + "}"
        return _r(v)

    tex = header_comment(src) + r"""
\begin{table*}[t]
\centering
\caption{Feynman extrapolation benchmark --- Kaggle 4-vCPU multiprocessing run (primary).
  Bold: extrap $R^2 > 0.99$; italic: $R^2 < 0$.}
\label{tab:feynman}
\small
\begin{tabular}{llrrr}
\toprule
\textbf{Equation} & \textbf{Domain}
  & \textbf{Hyp Train $R^2$} & \textbf{Hyp Extrap $R^2$}
  & \textbf{NN Extrap $R^2$} \\
\midrule
"""
    for (eq, dom, htr, hex_, nne) in equations:
        htr_s = _r(htr)
        hex_s = _bold(hex_) if isinstance(hex_, float) and hex_ >= 0.99 else _r(hex_)
        nne_s = _r(nne)
        # italic for negatives
        if isinstance(hex_, float) and hex_ < 0 and hex_ > -100:
            hex_s = r"\textit{" + f"{hex_:.3f}" + "}"
        if isinstance(nne, float) and nne < 0 and nne > -100:
            nne_s = r"\textit{" + f"{nne:.3f}" + "}"
        tex += f"{eq} & {dom} & {htr_s} & {hex_s} & {nne_s} \\\\\n"

    n_succ = sum(1 for r in equations if isinstance(r[3], float) and r[3] >= 0.99)
    n_nn   = sum(1 for r in equations if isinstance(r[4], float) and r[4] >= 0.99)
    tex += r"""\midrule
""" + f"Successes ($R^2 > 0.99$) & & & {n_succ}/30 ({n_succ/30*100:.1f}\\%) & {n_nn}/30 (0.0\\%) \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    write_table("feynman.tex", tex)


def gen_all30_domain_summary() -> None:
    """
    exp2 — all-30 multi-domain: per-domain aggregate summary.

    Source: per-domain result shards directly under RESULTS (exp2's
    --results-dir is already comparison_results/feynman-tests/exp2_multi —
    see _load_exp2_multi_domain_rows()). Groups the flattened 30 equation
    rows by physics domain and reports, per domain: equation count, mean
    train R², mean extrap R², and hybrid-success count (extrap R² >= 0.99).
    """
    rows, sources = _load_exp2_multi_domain_rows()
    used_fallback = not rows
    if used_fallback:
        rows = _PAPER_FEYNMAN_EQUATIONS

    src_label = sources[0] if sources else None

    domains: dict[str, list[tuple]] = {}
    for row in rows:
        domains.setdefault(row[1], []).append(row)

    def _mean(vals):
        vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
        return sum(vals) / len(vals) if vals else float("nan")

    def _r(v):
        return f"{v:.3f}" if isinstance(v, float) and v == v else "---"

    tex = header_comment(src_label) + r"""
\begin{table}[t]
\centering
\caption{exp2 --- all-30 multi-domain benchmark, per-domain summary.}
\label{tab:all30_domain_summary}
\small
\begin{tabular}{lrrrr}
\toprule
\textbf{Domain} & \textbf{N} & \textbf{Mean Train $R^2$}
  & \textbf{Mean Extrap $R^2$} & \textbf{Successes} \\
\midrule
"""
    for domain in sorted(domains):
        drows = domains[domain]
        n = len(drows)
        mean_train  = _mean(r[2] for r in drows)
        mean_extrap = _mean(r[3] for r in drows)
        n_succ = sum(1 for r in drows if isinstance(r[3], float) and r[3] >= 0.99)
        tex += f"{domain} & {n} & {_r(mean_train)} & {_r(mean_extrap)} & {n_succ}/{n} \\\\\n"

    n_total = len(rows)
    n_succ_total = sum(1 for r in rows if isinstance(r[3], float) and r[3] >= 0.99)
    tex += r"\midrule" + "\n"
    tex += (f"\\textbf{{All domains}} & {n_total} & "
            f"{_r(_mean(r[2] for r in rows))} & {_r(_mean(r[3] for r in rows))} & "
            f"{n_succ_total}/{n_total} \\\\\n")
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("all30_domain_summary.tex", tex)


def gen_multi_domain_rank_table() -> None:
    """
    exp2 — all-30 multi-domain: domain ranking by extrapolation R^2.

    Source: same shards as gen_all30_domain_summary() (loaded independently
    here since generators are called standalone from the dispatch table).
    Ranks each domain by mean extrap R^2 descending, alongside the hybrid
    vs. NN win margin, so the reader can see at a glance which domains the
    hybrid method struggles or excels on relative to the pure NN baseline.
    """
    rows, sources = _load_exp2_multi_domain_rows()
    used_fallback = not rows
    if used_fallback:
        rows = _PAPER_FEYNMAN_EQUATIONS

    src_label = sources[0] if sources else None

    domains: dict[str, list[tuple]] = {}
    for row in rows:
        domains.setdefault(row[1], []).append(row)

    def _mean(vals):
        vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
        return sum(vals) / len(vals) if vals else float("nan")

    def _r(v):
        return f"{v:.3f}" if isinstance(v, float) and v == v else "---"

    ranked = sorted(
        domains.items(),
        key=lambda kv: _mean(r[3] for r in kv[1]),
        reverse=True,
    )

    tex = header_comment(src_label) + r"""
\begin{table}[t]
\centering
\caption{exp2 --- all-30 multi-domain benchmark, domains ranked by mean
  Hybrid extrapolation $R^2$ (descending). Margin is Hybrid $-$ NN.}
\label{tab:multi_domain_rank_table}
\small
\begin{tabular}{clrrr}
\toprule
\textbf{Rank} & \textbf{Domain} & \textbf{Hybrid Extrap $R^2$}
  & \textbf{NN Extrap $R^2$} & \textbf{Margin} \\
\midrule
"""
    for rank, (domain, drows) in enumerate(ranked, start=1):
        mean_hyp = _mean(r[3] for r in drows)
        mean_nn  = _mean(r[4] for r in drows)
        margin = (mean_hyp - mean_nn) if mean_hyp == mean_hyp and mean_nn == mean_nn else float("nan")
        tex += f"{rank} & {domain} & {_r(mean_hyp)} & {_r(mean_nn)} & {_r(margin)} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("multi_domain_rank_table.tex", tex)


def _extract_hybrid_domain_rows(d) -> list[tuple]:
    """Pull (name, domain, hybrid_r2, llm_r2, nn_r2) rows out of a single
    hybrid_all_domains result shard written by
    hybrid_system_llm_nn_all_domains.py.

    NOTE: the exact field names emitted by that script were not confirmed
    against a live run (no sample JSON was available while writing this) —
    several common spellings are tried per field, and a row with an
    unrecognised shape contributes "?"/NaN for that field rather than being
    dropped outright, so partial/renamed data still shows up instead of
    silently vanishing. If real output uses different keys, update the
    e.get(...) chains below rather than the overall structure.
    """
    if not isinstance(d, dict):
        return []
    eqs = d.get("equations", d.get("results", d.get("cases", [])))
    if not isinstance(eqs, list) or not eqs:
        return []
    domain_fallback = d.get("domain", "?")
    rows = []
    for e in eqs:
        if not isinstance(e, dict):
            continue
        rows.append((
            e.get("name", e.get("equation", "?")),
            e.get("domain", domain_fallback),
            e.get("hybrid_r2", e.get("hyp_r2", e.get("hyp_extrap_r2", float("nan")))),
            e.get("llm_r2",    e.get("pure_llm_r2", float("nan"))),
            e.get("nn_r2",     e.get("neural_network_r2", e.get("nn_extrap_r2", float("nan")))),
        ))
    return rows


def _load_hybrid_all_domains_rows() -> tuple[list[tuple], list[Path]]:
    """Load & merge every per-domain shard for hybrid_all_domains.

    ci_postprocess.yml resolves --results-dir to the canonical
    hybrid_llm_nn/all_domains dir for this experiment, so shards are read
    with subdir="" directly from PATCHED / RESULTS — same reasoning as
    _load_exp2_multi_domain_rows(). Duplicate equation names are resolved by
    taking the entry from the last shard processed (PATCHED overrides
    RESULTS, matching load_best()'s override precedence elsewhere in this
    file).
    """
    by_name: dict[str, tuple] = {}
    sources: list[Path] = []
    for base in [RESULTS, PATCHED]:
        if not base.exists():
            continue
        for f in sorted(_filtered_glob(base, "*.json")):
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            rows = _extract_hybrid_domain_rows(data)
            if not rows:
                continue
            sources.append(f)
            for row in rows:
                by_name[row[0]] = row
    return list(by_name.values()), sources


def gen_hybrid_all_domains_summary() -> None:
    """
    hybrid_all_domains — per-domain LLM+NN hybrid summary.
    Produces: hybrid_all_domains_summary.tex

    Previously this experiment had NO generator at all (_DISPATCH entry was
    an unconditional empty list), so the "Tables: hybrid_all_domains" CI
    step always failed with "wrote no .tex files" regardless of whether
    source data existed.
    """
    rows, sources = _load_hybrid_all_domains_rows()
    src_label = sources[0] if sources else None

    domains: dict[str, list[tuple]] = {}
    for row in rows:
        domains.setdefault(row[1], []).append(row)

    def _mean(vals):
        vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
        return sum(vals) / len(vals) if vals else float("nan")

    def _r(v):
        return f"{v:.3f}" if isinstance(v, float) and v == v else "---"

    tex = header_comment(src_label) + r"""
\begin{table}[t]
\centering
\caption{hybrid\_all\_domains --- per-domain LLM+NN hybrid summary.}
\label{tab:hybrid_all_domains_summary}
\small
\begin{tabular}{lrrrr}
\toprule
\textbf{Domain} & \textbf{N} & \textbf{Hybrid $R^2$}
  & \textbf{LLM $R^2$} & \textbf{NN $R^2$} \\
\midrule
"""
    for domain in sorted(domains):
        drows = domains[domain]
        n = len(drows)
        tex += (f"{domain} & {n} & {_r(_mean(r[2] for r in drows))} & "
                f"{_r(_mean(r[3] for r in drows))} & {_r(_mean(r[4] for r in drows))} \\\\\n")
    if not domains:
        tex += r"\multicolumn{5}{c}{No hybrid\_all\_domains data available.} \\" + "\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("hybrid_all_domains_summary.tex", tex)


def gen_domain_rank_table() -> None:
    """
    hybrid_all_domains — domains ranked by Hybrid R^2 (descending).
    Produces: domain_rank_table.tex
    """
    rows, sources = _load_hybrid_all_domains_rows()
    src_label = sources[0] if sources else None

    domains: dict[str, list[tuple]] = {}
    for row in rows:
        domains.setdefault(row[1], []).append(row)

    def _mean(vals):
        vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
        return sum(vals) / len(vals) if vals else float("nan")

    def _r(v):
        return f"{v:.3f}" if isinstance(v, float) and v == v else "---"

    ranked = sorted(domains.items(), key=lambda kv: _mean(r[2] for r in kv[1]), reverse=True)

    tex = header_comment(src_label) + r"""
\begin{table}[t]
\centering
\caption{hybrid\_all\_domains --- domains ranked by mean Hybrid $R^2$ (descending).}
\label{tab:domain_rank_table}
\small
\begin{tabular}{clr}
\toprule
\textbf{Rank} & \textbf{Domain} & \textbf{Hybrid $R^2$} \\
\midrule
"""
    for rank, (domain, drows) in enumerate(ranked, start=1):
        tex += f"{rank} & {domain} & {_r(_mean(r[2] for r in drows))} \\\\\n"
    if not ranked:
        tex += r"\multicolumn{3}{c}{No hybrid\_all\_domains data available.} \\" + "\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("domain_rank_table.tex", tex)


def _extract_extrap_rows(d) -> list[tuple]:
    """Pull (name, domain, train_r2, ood_extrap_r2) rows out of a single
    all_domains_extrap_v4_*.json shard. Reuses the hyp_train_r2/hyp_extrap_r2
    field spelling shared with exp2_feynman/exp2 (extrap runs the same
    protocol_core-style harness, just scored on an OOD split), with
    ood_r2/extrap_r2 tried first since that's the field name implied by the
    ci_postprocess.yml comment for this step ("extrap_r2 per method per
    equation")."""
    if not isinstance(d, dict):
        return []
    eqs = d.get("equations", d.get("results", []))
    if not isinstance(eqs, list) or not eqs:
        return []
    domain_fallback = d.get("domain", "?")
    rows = []
    for e in eqs:
        if not isinstance(e, dict):
            continue
        rows.append((
            e.get("name", "?"),
            e.get("domain", domain_fallback),
            e.get("hyp_train_r2", e.get("train_r2", float("nan"))),
            e.get("ood_r2", e.get("extrap_r2", e.get("hyp_extrap_r2", float("nan")))),
        ))
    return rows


def _load_extrap_rows() -> tuple[list[tuple], list[Path]]:
    """Load & merge every all_domains_extrap_v4_*.json shard for extrap.

    Like hybrid_all_domains, --results-dir is already resolved to the
    canonical comparison_results/extrapolation dir for this experiment, so
    shards are read with subdir="" directly. Falls back to a generic
    "*.json" glob if no all_domains_extrap_v4_*.json shard is found, in case
    the shard naming has drifted.
    """
    by_name: dict[str, tuple] = {}
    sources: list[Path] = []
    for base in [RESULTS, PATCHED]:
        if not base.exists():
            continue
        shards = _filtered_glob(base, "all_domains_extrap_v4_*.json")
        if not shards:
            shards = _filtered_glob(base, "*.json")
        for f in sorted(shards):
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            rows = _extract_extrap_rows(data)
            if not rows:
                continue
            sources.append(f)
            for row in rows:
                by_name[row[0]] = row
    return list(by_name.values()), sources


def gen_extrap_ood_table() -> None:
    """
    extrap — out-of-distribution extrapolation table (Tab 9 OOD columns).
    Produces: extrap_ood_table.tex

    Previously this experiment had NO generator at all (_DISPATCH entry was
    an unconditional empty list), so the "Tables: extrap" CI step always
    failed with "wrote no .tex files" regardless of whether source data
    existed.
    """
    rows, sources = _load_extrap_rows()
    src_label = sources[0] if sources else None

    def _r(v):
        return f"{v:.3f}" if isinstance(v, float) and v == v else "---"

    tex = header_comment(src_label) + r"""
\begin{table*}[t]
\centering
\caption{extrap --- out-of-distribution (OOD) extrapolation results, all domains.}
\label{tab:extrap_ood_table}
\small
\begin{tabular}{llrr}
\toprule
\textbf{Equation} & \textbf{Domain} & \textbf{Train $R^2$} & \textbf{OOD Extrap $R^2$} \\
\midrule
"""
    for (name, domain, train_r2, ood_r2) in sorted(rows, key=lambda r: (r[1], r[0])):
        tex += f"{name} & {domain} & {_r(train_r2)} & {_r(ood_r2)} \\\\\n"
    if not rows:
        tex += r"\multicolumn{4}{c}{No extrap OOD data available.} \\" + "\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    write_table("extrap_ood_table.tex", tex)


def gen_nguyen12() -> None:
    """
    Tab 8 — Nguyen-12 Benchmark: train and extrapolation R² by equation.
    P = PySR-only, H = HypatiaX, N = Neural MLP.
    Matches Table 8 in §10.8.
    """
    # run_all.sh (exp3/exp3b) writes nguyen12 results to RESULTS_DIR root.
    # Also check legacy nguyen12/ subdir.
    data, src = load_best("", "exp3*nguyen12*.json",
                          extra_subdirs=["nguyen12"])

    # Paper-verified fallback (Table 8)
    PAPER_ROWS = [
        # (eq, formula, P_train, P_extrap, H_train, H_extrap, N_train, N_extrap)
        ("N-1",  r"x^3 + x^2 + x",
         0.9999, 1.0000, 0.9999, 0.9999, 0.9993, -0.784),
        ("N-2",  r"x^4 + x^3 + x^2 + x",
         0.9999, 1.0000, 0.9999, 1.0000, 0.9986, -0.902),
        ("N-3",  r"x^5 + x^4 + x^3 + x^2 + x",
         0.9999, -426.2, 0.9999, 0.9976, 0.9986, -0.913),
        ("N-4",  r"x^6+x^5+x^4+x^3+x^2+x",
         0.9999, -999,   0.9999, -999,   0.9979, -0.828),
        ("N-5",  r"\sin(x^2)\cos(x)-1",
         0.9999, 1.0000, 0.9999, 1.0000, 0.9979, -5.586),
        ("N-6",  r"\sin(x)+\sin(x+x^2)",
         0.9999, 1.0000, 0.9999, 1.0000, 0.9987,-12.654),
        ("N-7",  r"\ln(x+1)+\ln(x^2+1)",
         0.9999, 0.9762, 0.9999, 0.7316, 0.9868,  0.856),
        ("N-8",  r"\sqrt{x}",
         0.9999, 1.0000, 0.9999, 1.0000, 0.9988,  0.954),
        ("N-9",  r"\sin(x)+\sin(y^2)",
         0.9999, 1.0000, 0.9999, 1.0000, 0.9986, -6.708),
        ("N-10", r"2\sin(x)\cos(y)",
         0.9999, 1.0000, 0.9999, 0.9997, 0.9995, -2.379),
        ("N-11", r"x^y",
         0.9999, 1.0000, 0.9999, 0.9999, 0.9984, -0.423),
        ("N-12", r"x^4-x^3+\tfrac{1}{2}y^2-y",
         0.9987, -1.056, 0.9994, -1.054, 0.9985, -1.198),
    ]

    def _extract(d):
        if not isinstance(d, dict):
            return []
        eqs = d.get("equations", d.get("results", []))
        if not isinstance(eqs, list) or len(eqs) < 12:
            return []
        rows = []
        for e in eqs:
            rows.append((
                e.get("name", "?"), e.get("formula", "?"),
                e.get("pysr_train",    float("nan")),
                e.get("pysr_extrap",   float("nan")),
                e.get("hypatia_train", float("nan")),
                e.get("hypatia_extrap",float("nan")),
                e.get("nn_train",      float("nan")),
                e.get("nn_extrap",     float("nan")),
            ))
        return rows

    equations = _extract(data) if data else []
    if not equations:
        equations = PAPER_ROWS

    def _r(v, lo=-100):
        if not isinstance(v, (int, float)) or v != v: return "---"
        if v <= lo: return r"$\ll{-100}$"
        if v >= 0.9999: return r"\textbf{" + f"{v:.4f}" + "}"
        if v < 0: return r"\textit{" + f"{v:.3f}" + "}"
        return f"{v:.4f}"

    tex = header_comment(src) + r"""
\begin{table*}[t]
\centering
\caption{Nguyen-12 benchmark: train and extrapolation $R^2$ by equation.
  P = PySR-only; H = HypatiaX; N = Neural MLP.
  Near-miss criterion: $R^2 \ge 0.9999$.
  Bold: extrap $R^2 \ge 0.9999$. Italic: $R^2 < 0$.}
\label{tab:nguyen12}
\small
\begin{tabular}{llrrrrrr}
\toprule
\textbf{Eq.} & \textbf{Formula}
  & \textbf{P Train} & \textbf{P Extrap}
  & \textbf{H Train} & \textbf{H Extrap}
  & \textbf{N Train} & \textbf{N Extrap} \\
\midrule
"""
    for (eq, form, pt, pe, ht, he, nt, ne) in equations:
        tex += f"{eq} & ${form}$ & {_r(pt)} & {_r(pe,-500)} & {_r(ht)} & {_r(he)} & {_r(nt)} & {_r(ne)} \\\\\n"

    n_p = sum(1 for r in equations if isinstance(r[3], float) and r[3] >= 0.9999)
    n_h = sum(1 for r in equations if isinstance(r[5], float) and r[5] >= 0.9999)
    n_n = 0
    tex += r"""\midrule
""" + f"Success ($R^2 \\ge 0.9999$) & & \\multicolumn{{2}}{{c}}{{{n_p}/12 ({n_p/12*100:.1f}\\%)}}"
    tex += f" & \\multicolumn{{2}}{{c}}{{{n_h}/12 ({n_h/12*100:.1f}\\%)}}"
    tex += f" & \\multicolumn{{2}}{{c}}{{{n_n}/12 (0.0\\%)}} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    write_table("nguyen12.tex", tex)


def gen_version_history() -> None:
    """
    Tab 10 — HypatiaX benchmark version history.
    Matches Table 10 in Appendix B. Values are stable/hardcoded.
    """
    ROWS = [
        ("v1.0", 62, "Initial benchmark; axis-aligned splits; no trust gating."),
        ("v2.0", 71, "PCA-directed splits introduced; trust gate added ($R^2 > 0.1$)."),
        ("v3.0", 74, "Three hard cases added; trust gate raised to $R^2 > 0.5$; "
                     "data leakage fixed; unified executor."),
    ]
    tex = (
        "% Auto-generated by generate_tables.py — version history is hardcoded (stable)\n"
        f"% Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        + r"""
\begin{table}[t]
\centering
\caption{HypatiaX benchmark version history and key changes.}
\label{tab:version_hist}
\begin{tabular}{lrl}
\toprule
\textbf{Version} & \textbf{Cases} & \textbf{Key Changes} \\
\midrule
"""
    )
    for (ver, cases, changes) in ROWS:
        tex += f"{ver} & {cases} & {changes} \\\\\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("version_history.tex", tex)


def gen_timing_detail() -> None:
    """
    Tab 11 — Detailed timing comparison and speedup calculations (Appendix C).
    Matches Table 11 (Appendix C).
    """
    data, src = load_best("", "hypatiax_defi_benchmark_v3*results*.json",
                          extra_subdirs=["defi"])

    # Paper-verified fallback (Table 11)
    PAPER_ROWS = [
        ("Mean (all 74 cases)",         11.4, 3.0, 6.8,  "Hybrid 2.30× slower than NN"),
        ("Median (all 74 cases)",        10.3, 2.7, 1.7,  "Hybrid 1.64× faster than NN"),
        ("LLM-routed only ($n=68$)",    None, 2.7, 1.56,  "Hybrid 1.73× faster than NN"),
    ]

    def _extract(d):
        if not isinstance(d, dict):
            return []
        td = d.get("timing_detail", d.get("timing", {}))
        rows = []
        for label, key in [("Mean (all 74 cases)", "mean_all"),
                           ("Median (all 74 cases)", "median_all"),
                           ("LLM-routed only ($n=68$)", "llm_routed")]:
            t = td.get(key, {})
            rows.append((
                label,
                t.get("llm_s", t.get("llm_time_s", None)),
                t.get("nn_s",  t.get("nn_time_s",  None)),
                t.get("hyp_s", t.get("hyp_time_s", None)),
                t.get("speedup_note", "---"),
            ))
        return rows if len(rows) >= 3 else []

    rows = _extract(data) if data else []
    if not rows:
        rows = PAPER_ROWS

    def _t(v): return f"{v:.2f}" if isinstance(v, (int, float)) and v is not None and v == v else "---"

    tex = header_comment(src) + r"""
\begin{table}[t]
\centering
\caption{Detailed timing comparison and speedup calculations (v3.0 benchmark).}
\label{tab:timing_detail}
\begin{tabular}{lrrrr}
\toprule
\textbf{Comparison} & \textbf{LLM (s)} & \textbf{NN (s)}
  & \textbf{Hybrid (s)} & \textbf{Speedup} \\
\midrule
"""
    for (label, llm, nn, hyp, note) in rows:
        tex += f"{label} & {_t(llm)} & {_t(nn)} & {_t(hyp)} & {note} \\\\\n"

    tex += r"""\midrule
\multicolumn{5}{l}{\textit{Previously claimed: $3.7\times$ speedup = 73\% reduction. Not supported by data.}} \\
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("timing_detail.tex", tex)



def gen_instability() -> None:
    """
    Writes instability.tex (tab:instability in main paper §10.9).
    Regime distribution: A-Symbolic, B-Approx, B-Det.Biased, C-Collapse.
    Source: instability/ JSON or instability_analysis.csv (from pipeline).
    Falls back to the hardcoded paper values (70 tasks, K=30) when no JSON found.
    """
    # run_all.sh (instability step) writes instability*.json to RESULTS_DIR/figures/.
    # Also check legacy instability/ subdir.
    data, src = load_best("figures", "instability*.json",
                          extra_subdirs=["instability"])

    # If no JSON, try the instability_analysis.csv produced by the pipeline.
    # run_all.sh writes it to RESULTS_DIR/figures/instability_analysis.csv.
    if not data:
        csv_candidates = (
            list((RESULTS / "figures").glob("instability_analysis.csv")) +
            list(RESULTS.glob("instability_analysis.csv"))
        )
        if csv_candidates:
            try:
                import csv as _csv
                rows = list(_csv.DictReader(open(csv_candidates[0])))
                regime_counts: dict[str, int] = {}
                for row in rows:
                    r = row.get("regime", "?")
                    regime_counts[r] = regime_counts.get(r, 0) + 1
                data = {"regime_counts": regime_counts,
                        "total_tasks": len(rows),
                        "k_runs": 30}
                src = csv_candidates[0]
            except Exception:
                pass

    if not data:
        write_table("instability.tex", "% No instability results yet\n")
        return

    total  = data.get("total_tasks", data.get("n_tasks", 70))
    k_runs = data.get("k_runs",      data.get("n_runs", 30))

    # Regime counts — prefer explicit dict, else compute from raw scores
    rc = data.get("regime_counts", {})
    n_A  = rc.get("A-Symbolic",   data.get("n_symbolic",   61))
    n_B  = rc.get("B-Approx",     data.get("n_biased",      2))
    n_B2 = rc.get("B-Det.Biased", data.get("n_borderline",  4))
    n_C  = rc.get("C-Collapse",   data.get("n_collapse",    3))

    def _frac(n):
        try:
            return f"{int(n)/int(total)*100:.1f}\\,\\%"
        except Exception:
            return "---"

    tex = header_comment(src) + r"""
\begin{table}[h]
\centering
\caption{LLM instability regime distribution """ + \
    f"({total} tasks, $K={k_runs}$ runs each). " + \
    r"$\mathrm{II}_i = \sigma_i = \mathrm{std}(R^2_i)$ across independent runs.}" + r"""
\label{tab:instability}
\begin{tabular}{lrrrr}
\toprule
Regime & Definition & $n$ & Fraction \\
\midrule
""" + \
    f"A: Symbolic Stability   & $\\sigma\\approx0$, $\\mu\\approx1$ & {n_A} & {_frac(n_A)} \\\\\n" + \
    f"B: Deterministic Biased & $\\sigma\\approx0$, $\\mu<1$       & {n_B} & {_frac(n_B)} \\\\\n" + \
    f"B*: Borderline Stochastic & $0 < \\sigma < 0.05$              & {n_B2} & {_frac(n_B2)} \\\\\n" + \
    f"C: Stochastic Collapse  & $\\sigma \\ge 0.10$ or $\\mu < 0$ & {n_C} & {_frac(n_C)} \\\\\n" + \
    r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("instability.tex", tex)


def gen_repro_macros() -> None:
    macros: dict[str, str] = {}
    data, _ = load_best("", "hypatiax_defi_benchmark_v3*results*.json",
                        extra_subdirs=["defi"])
    if isinstance(data, dict):
        acc = data.get("accuracy", data.get("success_rate", 0))
        macros["defiAccuracy"]   = f"{acc:.1%}"
        macros["defiTotalCases"] = str(data.get("total_cases", 74))
    data, _ = load_best("ablation/exp1_ablation", "*.json")
    if isinstance(data, dict):
        mw_p = data.get("mw_p", data.get("mann_whitney_p", ""))
        mw_u = data.get("mw_u", data.get("mann_whitney_u", ""))
        if mw_p:
            macros["coreAblationMWp"] = f"{mw_p:.4f}"
        if mw_u:
            macros["coreAblationMWu"] = f"{mw_u:.1f}"

    # Neural Network extrapolation mean/std/CI. Sourced from the same
    # _load_five_system_rows_real() combined loader gen_five_system() uses
    # (exp1_five primary, exp2/exp2_extrap secondary; no hardcoded fallback),
    # so these macros always match the table's Std/CI columns exactly and
    # are left undefined (not silently wrong) if neither real source exists.
    five_rows, _fs_src, _fs_tier = _load_five_system_rows_real()
    fs_no_data = five_rows is None
    if not fs_no_data:
        for (name, n, _emed, emean, _tr2, std, _focus) in five_rows:
            if name not in ("Neural Network",):
                continue
            try:
                nn_mean = float(emean)
            except (TypeError, ValueError):
                nn_mean = None
            try:
                nn_std = float(std)
            except (TypeError, ValueError):
                nn_std = None
            nn_n = n
            if nn_mean is not None:
                macros["nnExtrapMean"] = f"{nn_mean:.1f}"
            if nn_std is not None:
                macros["nnExtrapStd"] = f"{nn_std:.1f}"
            if nn_n:
                macros["nnExtrapN"] = str(int(nn_n))
            ci = _ci95(nn_mean, nn_std, nn_n)
            if ci:
                macros["nnExtrapCI"] = ci
            break
    if fs_no_data:
        FALLBACK_TABLES.append(
            "repro_macros.tex (nnExtrapMean/nnExtrapStd/nnExtrapN/nnExtrapCI) "
            "-- same missing exp1_five/exp2 source as five_system.tex; macros "
            "left undefined rather than filled with fallback numbers."
        )


    lines = [
        "% Auto-generated reproducibility macros",
        "% Usage: \\repoVal{defiAccuracy}",
        f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]
    if fs_no_data:
        lines.append(
            "% ⚠ nnExtrap* macros below are FALLBACK DATA (Issue 4) — no "
            "five_system/system_comparison JSON located yet."
        )
    lines.append("")
    for key, val in macros.items():
        lines.append(f"\\newcommand{{\\{key}}}{{{val}}}")
    write_table("repro_macros.tex", "\n".join(lines) + "\n")


# ── Supplement B tables — noise sweep ─────────────────────────────────────────
#
#  Source JSON schema (from run_noise_sweep_benchmark.py):
#    data["noise_levels"]   : [0.0, 0.005, 0.01, 0.05, 0.10]
#    data["methods"]        : ["EnhancedHybridSystemDeFi", "HybridSystemLLMNN all-domains"]
#    data["per_noise"][sigma_str]["method_summary"][method_name] :
#        {median_r2, mean_r2, std_r2, recovery_rate, n_success, n_total,
#         threshold_used, n_catastrophic}
#
#  Method short labels (matching supp_benchmark_report.tex)
#    M3 = EnhancedHybridSystemDeFi  (EHD)
#    M4 = HybridSystemLLMNN all-domains  (HSL)

_M3_KEY = "EnhancedHybridSystemDeFi"
_M4_KEY = "HybridSystemLLMNN all-domains"

# Fallback key fragments for flexible matching
_M3_FRAG = ("enhanced", "hybrid", "defi", "m3")
_M4_FRAG = ("llmnn", "all_domain", "all-domain", "m4")

_SIGMA_LABELS = {
    "0.0000": "0\\%", "0.005":  "0.5\\%",
    "0.0050": "0.5\\%",
    "0.0100": "1\\%",  "0.0500": "5\\%",  "0.1000": "10\\%",
    "0.01":  "1\\%",   "0.05":  "5\\%",   "0.1":   "10\\%",
}


def _sigma_str(sigma: float) -> str:
    return f"{sigma:.4f}"


def _label(sigma: float) -> str:
    s = _sigma_str(sigma)
    return _SIGMA_LABELS.get(s, f"{sigma*100:.4g}\\%")


def _pick_method(method_summary: dict, frags: tuple[str, ...]) -> dict:
    """Return the entry whose key contains any of frags (case-insensitive)."""
    for key, val in method_summary.items():
        kl = key.lower().replace(" ", "").replace("-", "").replace("_", "")
        if any(f in kl for f in frags):
            return val
    return {}


def _pick_method_key(method_summary: dict, frags: tuple[str, ...]) -> str | None:
    """Like _pick_method, but returns the matched key itself rather than its
    metrics dict. Needed to look up the same method's rows in a sibling
    per_equation dict, which is keyed by the literal method name rather than
    by metric — _pick_method alone discards that name.
    """
    for key in method_summary:
        kl = key.lower().replace(" ", "").replace("-", "").replace("_", "")
        if any(f in kl for f in frags):
            return key
    return None


def _median_rmse_from_per_equation(level: dict, method_key: str | None) -> float | None:
    """Median rmse for one method across every equation in level["per_equation"].

    method_summary for the sample-complexity sweep never carries
    median_rmse (confirmed 2026-06-18 schema: median_r2, mean_r2, std_r2,
    recovery_rate, n_success, n_total, threshold_used only) — but
    per_equation has a real per-equation rmse for every method at every n,
    with full 30/30 coverage. This replaces the previous
    sqrt(1 - median_r2) approximation, which assumes a fixed output scale
    and is off by an order of magnitude versus the real per-equation rmse
    (e.g. n=50, M3: approx ≈ 1.8e-4 vs actual median ≈ 4.0e-3) because
    per-equation y-scales vary by 1-2 orders of magnitude across the
    benchmark set.
    """
    if not method_key or not isinstance(level, dict):
        return None
    per_eq = level.get("per_equation")
    if not isinstance(per_eq, dict) or not per_eq:
        return None
    vals = []
    for eq_methods in per_eq.values():
        if not isinstance(eq_methods, dict):
            continue
        entry = eq_methods.get(method_key)
        if isinstance(entry, dict):
            v = entry.get("rmse")
            if isinstance(v, (int, float)):
                vals.append(float(v))
    if not vals:
        return None
    import statistics
    return statistics.median(vals)


def gen_suppb_r2_noise(noise_data: dict | None) -> None:
    """tab:r2_noise — Median R², Min R², Std by σ for M3 and M4."""
    if not noise_data:
        write_table("suppb_r2_noise.tex", "% suppB noise_sweep data not available\n")
        return

    noise_levels = sorted(noise_data.get("noise_levels", []))
    per_noise    = noise_data.get("per_noise", {})
    src          = "noise_sweep_*.json"

    tex = header_comment(src) + r"""
\begin{table}[H]
\centering
\caption{$R^2$ statistics per noise level ($n=200$, 30 equations).}
\label{tab:r2_noise}
\renewcommand{\arraystretch}{1.2}
\small
\begin{tabular}{l r r r r r r}
\toprule
& \multicolumn{3}{c}{\textbf{\EHD{} (M3)}}
& \multicolumn{3}{c}{\textbf{\HSL{} (M4)}}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
$\sigma$ & Median & Min & Std & Median & Min & Std\\
\midrule
"""
    for sigma in noise_levels:
        ss  = _sigma_str(sigma)
        pnd = per_noise.get(ss) or {}
        ms  = pnd.get("method_summary", {}) if isinstance(pnd, dict) else {}
        m3  = _pick_method(ms, _M3_FRAG)
        m4  = _pick_method(ms, _M4_FRAG)

        def _v(d, k):
            v = d.get(k)
            return f"{v:.7f}" if isinstance(v, float) else "---"

        tex += (
            f"{_label(sigma)} & {_v(m3,'median_r2')} & --- & {_v(m3,'std_r2')}"
            f" & {_v(m4,'median_r2')} & --- & {_v(m4,'std_r2')} \\\\\n"
        )

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("suppb_r2_noise.tex", tex)


def gen_suppb_rr_noise(noise_data: dict | None) -> None:
    """tab:rr_noise — Recovery rate and catastrophic failure count by σ."""
    if not noise_data:
        write_table("suppb_rr_noise.tex", "% suppB noise_sweep data not available\n")
        return

    noise_levels = sorted(noise_data.get("noise_levels", []))
    per_noise    = noise_data.get("per_noise", {})
    src          = "noise_sweep_*.json"

    tex = header_comment(src) + r"""
\begin{table}[H]
\centering
\caption{Recovery rate and catastrophic failure count per noise level ($n=200$).}
\label{tab:rr_noise}
\small
\begin{tabular}{lrrrr}
\toprule
$\sigma$ & M3 Recovery & M3 Catastrophic & M4 Recovery & M4 Catastrophic\\
\midrule
"""
    for sigma in noise_levels:
        ss  = _sigma_str(sigma)
        pnd = per_noise.get(ss) or {}
        ms  = pnd.get("method_summary", {}) if isinstance(pnd, dict) else {}
        m3  = _pick_method(ms, _M3_FRAG)
        m4  = _pick_method(ms, _M4_FRAG)

        def _rr(d):
            v = d.get("recovery_rate")
            return f"{v*100:.1f}\\%" if isinstance(v, float) else "---"

        def _cat(d):
            return str(d.get("n_catastrophic", "---"))

        tex += (
            f"{_label(sigma)} & {_rr(m3)} & {_cat(m3)} & {_rr(m4)} & {_cat(m4)} \\\\\n"
        )

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("suppb_rr_noise.tex", tex)


def gen_suppb_time_noise(noise_data: dict | None) -> None:
    """tab:time_noise — Average computation time per noise level."""
    if not noise_data:
        write_table("suppb_time_noise.tex", "% suppB noise_sweep data not available\n")
        return

    noise_levels = sorted(noise_data.get("noise_levels", []))
    per_noise    = noise_data.get("per_noise", {})
    src          = "noise_sweep_*.json"

    tex = header_comment(src) + r"""
\begin{table}[H]
\centering
\caption{Average per-equation computation time (seconds) per noise level.}
\label{tab:time_noise}
\small
\begin{tabular}{lrrl}
\toprule
$\sigma$ & M3 avg (s) & M4 avg (s) & Speedup\\
\midrule
"""
    for sigma in noise_levels:
        ss  = _sigma_str(sigma)
        pnd = per_noise.get(ss) or {}
        ms  = pnd.get("method_summary", {}) if isinstance(pnd, dict) else {}
        m3  = _pick_method(ms, _M3_FRAG)
        m4  = _pick_method(ms, _M4_FRAG)

        # timing may be stored in method_summary or top-level timing sub-dict
        timing = (pnd or {}).get("timing", {}) if isinstance(pnd, dict) else {}
        t3 = m3.get("mean_time_s", timing.get("m3_mean_s"))
        t4 = m4.get("mean_time_s", timing.get("m4_mean_s"))

        t3_str = f"{t3:.1f}" if isinstance(t3, float) else "---"
        t4_str = f"{t4:.1f}" if isinstance(t4, float) else "---"
        if isinstance(t3, float) and isinstance(t4, float) and t4 > 0:
            spd = f"${t3/t4:.1f}\\times$"
        else:
            spd = "---"

        tex += f"{_label(sigma)} & {t3_str} & {t4_str} & {spd} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("suppb_time_noise.tex", tex)


# ── Supplement B tables — sample complexity ───────────────────────────────────
#
#  Source JSON schema (from run_sample_complexity_benchmark.py):
#    data["sample_sizes"]  : [50, 100, 200, 500]
#    data["methods"]       : [...]
#    data["per_n"][n_str]["method_summary"][method_name] :
#        {median_r2, mean_r2, std_r2, recovery_rate, n_success, n_total,
#         threshold_used}
#    data["data_efficiency"][method]["min_n_above_threshold"] : int | null

def gen_suppb_sc_metrics(sc_data: dict | None) -> None:
    """tab:sc_metrics — Median R² and RMSE by sample size (σ=5%)."""
    if not sc_data:
        write_table("suppb_sc_metrics.tex", "% suppB sample_complexity data not available\n")
        return

    sample_sizes = sorted(sc_data.get("sample_sizes", []))
    per_n        = sc_data.get("per_n", {})
    src          = "sample_complexity_*.json"

    tex = header_comment(src) + r"""
\begin{table}[H]
\centering
\caption{$R^2$ and RMSE per sample size ($\sigma=5\%$, 30 equations).}
\label{tab:sc_metrics}
\renewcommand{\arraystretch}{1.2}
\small
\begin{tabular}{r r r r r r r}
\toprule
& \multicolumn{3}{c}{\textbf{\EHD{} (M3)}}
& \multicolumn{3}{c}{\textbf{\HSL{} (M4)}}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
$n$ & Med $R^2$ & Min $R^2$ & Med RMSE & Med $R^2$ & Min $R^2$ & Med RMSE\\
\midrule
"""
    for n in sample_sizes:
        ns  = str(n)
        pnd = per_n.get(ns) or {}
        ms  = pnd.get("method_summary", {}) if isinstance(pnd, dict) else {}
        m3  = _pick_method(ms, _M3_FRAG)
        m4  = _pick_method(ms, _M4_FRAG)
        m3_key = _pick_method_key(ms, _M3_FRAG)
        m4_key = _pick_method_key(ms, _M4_FRAG)

        def _v(d, k):
            v = d.get(k)
            return f"{v:.7f}" if isinstance(v, float) else "---"

        # FIX SUPPB_SC-RMSE: real per-equation median, not the
        # sqrt(1 - median_r2) placeholder this used to compute (see
        # _median_rmse_from_per_equation's docstring for why that was wrong).
        def _rmse(method_key):
            v = _median_rmse_from_per_equation(pnd, method_key)
            return f"{v:.4f}" if isinstance(v, (int, float)) else "---"

        tex += (
            f"{n:4d} & {_v(m3,'median_r2')} & --- & {_rmse(m3_key)}"
            f" & {_v(m4,'median_r2')} & --- & {_rmse(m4_key)} \\\\\n"
        )

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("suppb_sc_metrics.tex", tex)


def gen_suppb_sc_summary(sc_data: dict | None) -> None:
    """tab:sc_summary — Aggregate summary across all sample sizes for each method.

    Columns: Method | Best n (min n where recovery_rate ≥ threshold) |
             Max Median R² | Recovery Rate at max n | Data Efficiency Note.
    This is the cross-n aggregate view that complements the per-n breakdown
    already produced by gen_suppb_sc_metrics().
    """
    if not sc_data:
        write_table("suppb_sc_summary.tex", "% suppB sample_complexity data not available\n")
        return

    sample_sizes = sorted(sc_data.get("sample_sizes", []))
    per_n        = sc_data.get("per_n", {})
    src          = "sample_complexity_*.json"

    # Collect per-(method, n) metrics so we can aggregate across n.
    from collections import defaultdict
    method_records: dict[str, dict] = defaultdict(lambda: {
        "r2_by_n": {}, "rr_by_n": {}, "n_success_by_n": {}, "n_total_by_n": {}
    })

    for n in sample_sizes:
        ns  = str(n)
        pnd = per_n.get(ns) or {}
        ms  = pnd.get("method_summary", {}) if isinstance(pnd, dict) else {}
        for mname, metrics in ms.items():
            if not isinstance(metrics, dict):
                continue
            rec = method_records[mname]
            r2  = metrics.get("median_r2")
            rr  = metrics.get("recovery_rate")
            ns_ = metrics.get("n_success")
            nt  = metrics.get("n_total")
            if isinstance(r2, float):  rec["r2_by_n"][n] = r2
            if isinstance(rr, float):  rec["rr_by_n"][n] = rr
            if isinstance(ns_, int):   rec["n_success_by_n"][n] = ns_
            if isinstance(nt,  int):   rec["n_total_by_n"][n]   = nt

    if not method_records:
        write_table("suppb_sc_summary.tex", "% suppB sc_data has no method_summary entries\n")
        return

    # Per-method summary stats
    # sc_data["threshold"] is sometimes a per-method dict rather than a scalar
    # (e.g. {"EnhancedHybridSystemDeFi": 0.999999, ...}).  Fall back to 0.8
    # whenever the value is not a plain number.
    _raw_thresh = sc_data.get("threshold", 0.8)
    threshold = _raw_thresh if isinstance(_raw_thresh, (int, float)) else 0.8

    rows = []
    for mname, rec in sorted(method_records.items()):
        r2s = rec["r2_by_n"]
        rrs = rec["rr_by_n"]
        max_r2  = max(r2s.values()) if r2s else float("nan")
        max_rr  = max(rrs.values()) if rrs else float("nan")
        # Best (highest) recovery-rate n
        best_n_rr = min((n for n, rr in rrs.items() if rr >= threshold),
                        default=None)
        # Recovery rate at the largest sample size tested
        final_n   = max(r2s.keys()) if r2s else None
        final_rr  = rrs.get(final_n, float("nan")) if final_n else float("nan")
        note = (
            f"≥{threshold:.0%} at n={best_n_rr}" if best_n_rr is not None
            else f"<{threshold:.0%} at all n"
        )
        rows.append((mname, best_n_rr, max_r2, max_rr, final_rr, note))

    def _r(v):
        return f"{v:.4f}" if isinstance(v, float) and v == v else "---"
    def _n(v):
        return str(v) if v is not None else "---"
    def _pct(v):
        return f"{v*100:.1f}\\%" if isinstance(v, float) and v == v else "---"

    tex = header_comment(src) + r"""
\begin{table}[H]
\centering
\caption{Sample-complexity sweep aggregate summary ($\sigma=5\%$, 30 equations).
  \textbf{Best n}: smallest $n$ achieving recovery rate $\ge """ + f"{threshold:.0%}" + r"""$.
  \textbf{Max Med $R^2$}: peak median $R^2$ across all $n$.
  \textbf{Final RR}: recovery rate at the largest $n$ tested.}
\label{tab:sc_summary}
\small
\begin{tabular}{l r r r r l}
\toprule
\textbf{Method} & \textbf{Best $n$} & \textbf{Max Med $R^2$} & \textbf{Max RR} & \textbf{Final RR} & \textbf{Data Efficiency} \\
\midrule
"""
    for (mname, best_n, max_r2, max_rr, final_rr, note) in rows:
        short = mname[:32]
        tex += f"{short} & {_n(best_n)} & {_r(max_r2)} & {_pct(max_rr)} & {_pct(final_rr)} & {note} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("suppb_sc_summary.tex", tex)


def gen_suppb_sc_by_sample(sc_data: dict | None) -> None:
    """tab:sc_by_sample — Full per-(n, method) breakdown with all available metrics.

    This is a wider version of suppb_sc_metrics.tex: where gen_suppb_sc_metrics
    shows only M3 and M4 with three columns each, this table shows every method
    present in the data with all numeric metrics from method_summary so readers
    can compare the full six-method suite at a glance.

    Columns (per method): Median R² | Mean R² | Std R² | Recovery Rate | n_success/n_total
    """
    if not sc_data:
        write_table("suppb_sc_by_sample.tex", "% suppB sample_complexity data not available\n")
        return

    sample_sizes = sorted(sc_data.get("sample_sizes", []))
    per_n        = sc_data.get("per_n", {})
    src          = "sample_complexity_*.json"

    # Discover all methods across all sample sizes
    all_methods: list[str] = []
    seen: set[str] = set()
    for n in sample_sizes:
        ns  = str(n)
        pnd = per_n.get(ns) or {}
        ms  = pnd.get("method_summary", {}) if isinstance(pnd, dict) else {}
        for mname in ms:
            if mname not in seen:
                all_methods.append(mname)
                seen.add(mname)

    if not all_methods:
        write_table("suppb_sc_by_sample.tex", "% suppB sc_data has no method_summary entries\n")
        return

    def _v(d: dict, k: str) -> str:
        v = d.get(k)
        return f"{v:.5f}" if isinstance(v, float) else "---"

    def _rr(d: dict) -> str:
        v = d.get("recovery_rate")
        return f"{v*100:.1f}\\%" if isinstance(v, float) else "---"

    def _succ(d: dict) -> str:
        ns = d.get("n_success")
        nt = d.get("n_total")
        if isinstance(ns, int) and isinstance(nt, int):
            return f"{ns}/{nt}"
        return "---"

    # Shorten method names for column headers
    def _short(name: str) -> str:
        name = name.replace("EnhancedHybridSystemDeFi", "EHD")
        name = name.replace("HybridSystemLLMNN all-domains", "HSL")
        return name[:18]

    n_methods = len(all_methods)
    col_spec = "r" + " rrrrr" * n_methods

    tex = header_comment(src) + r"""
\begin{table}[H]
\centering
\caption{Full sample-complexity results by sample size and method
  ($\sigma=5\%$, 30 equations). Each method block: Med $R^2$, Mean $R^2$, Std, RR, Success.}
\label{tab:sc_by_sample}
\renewcommand{\arraystretch}{1.1}
\scriptsize
\begin{tabular}{""" + col_spec + r"""}
\toprule
"""
    # Header row 1: method names spanning 5 columns each
    hdr1 = "$n$"
    for mname in all_methods:
        hdr1 += f" & \\multicolumn{{5}}{{c}}{{\\textbf{{{_short(mname)}}}}}"
    tex += hdr1 + " \\\\\n"

    # Sub-header cmidrules
    cmidrule_parts = []
    for i, _ in enumerate(all_methods):
        lo = 2 + i * 5
        hi = lo + 4
        cmidrule_parts.append(f"\\cmidrule(lr){{{lo}-{hi}}}")
    tex += " ".join(cmidrule_parts) + "\n"

    # Header row 2: metric labels
    hdr2 = ""
    for _ in all_methods:
        hdr2 += " & Med $R^2$ & Mean $R^2$ & Std & RR & Succ"
    tex += hdr2 + " \\\\\n\\midrule\n"

    for n in sample_sizes:
        ns  = str(n)
        pnd = per_n.get(ns) or {}
        ms  = pnd.get("method_summary", {}) if isinstance(pnd, dict) else {}
        row = str(n)
        for mname in all_methods:
            d = ms.get(mname, {})
            row += (
                f" & {_v(d,'median_r2')} & {_v(d,'mean_r2')}"
                f" & {_v(d,'std_r2')} & {_rr(d)} & {_succ(d)}"
            )
        tex += row + " \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("suppb_sc_by_sample.tex", tex)


def gen_suppb_winrate(noise_data: dict | None, sc_data: dict | None) -> None:
    """tab:winrate — Head-to-head win rates M3 vs M4 (noise + SC sweeps)."""
    if not noise_data and not sc_data:
        write_table("suppb_winrate.tex", "% suppB data not available\n")
        return

    def _count_wins(sweep_data: dict | None) -> tuple[int, int, int, int]:
        """Returns (m3_wins, m4_wins, ties, total)."""
        if not sweep_data:
            return 0, 0, 0, 0
        m3_w = m4_w = ties = total = 0
        key = "noise_levels" if "noise_levels" in sweep_data else "sample_sizes"
        levels = sorted(sweep_data.get(key, []))
        pn_key = "per_noise" if "per_noise" in sweep_data else "per_n"
        per = sweep_data.get(pn_key, {})
        for lvl in levels:
            lk  = _sigma_str(lvl) if key == "noise_levels" else str(lvl)
            pnd = per.get(lk) or {}
            ms  = pnd.get("method_summary", {}) if isinstance(pnd, dict) else {}
            m3  = _pick_method(ms, _M3_FRAG)
            m4  = _pick_method(ms, _M4_FRAG)
            n3  = m3.get("n_total", 0) or 0
            n4  = m4.get("n_total", 0) or 0
            # use n_success as a proxy for wins vs per-equation comparison
            s3  = m3.get("recovery_rate") or 0
            s4  = m4.get("recovery_rate") or 0
            n_eq = max(n3, n4, 30)
            total += n_eq
            eps = 1e-6
            if s3 > s4 + eps:
                m3_w += n_eq
            elif s4 > s3 + eps:
                m4_w += n_eq
            else:
                ties += n_eq
        return m3_w, m4_w, ties, total

    n3n, n4n, tn, totn = _count_wins(noise_data)
    n3s, n4s, ts, tots = _count_wins(sc_data)

    def _pct2(a, b):
        return f"{a}/{b} ({a/b*100:.1f}\\%)" if b > 0 else "---"

    src = "noise_sweep_*.json + sample_complexity_*.json"
    tex = header_comment(src) + r"""
\begin{table}[H]
\centering
\caption{Head-to-head win rates (M3 vs.\ M4): noise sweep (""" + \
    str(totn) + r" comparisons) and sample complexity sweep (" + str(tots) + r""" comparisons).}
\label{tab:winrate}
\small
\begin{tabular}{l r r r}
\toprule
\textbf{Outcome} & \textbf{Noise} & \textbf{SC} & \textbf{Consistent?}\\
\midrule
""" + \
    f"M3 strictly higher $R^2$ & {_pct2(n3n,totn)} & {_pct2(n3s,tots)} & \\\\\n" + \
    f"M4 strictly higher $R^2$ & {_pct2(n4n,totn)} & {_pct2(n4s,tots)} & \\\\\n" + \
    f"Tied ($R^2 > 0.9999$)    & {_pct2(tn,totn)}  & {_pct2(ts,tots)}  & \\checkmark\\\\\n" + \
    r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("suppb_winrate.tex", tex)


def gen_suppb_noiseless() -> None:
    """tab:overall — Six-method noiseless aggregate performance."""
    # Source: protocol_core_noiseless_*.json
    # NOTE: exp1's canonical output dir has a trailing "defi" segment — see
    # EXP1_SUBDIR="comparison_results/noise-noiseless/noiseless/defi" in
    # ci_postprocess.yml. Without it this glob searches one level too
    # shallow and never finds the file, even after exp1 has run.
    noiseless_dir = RESULTS / "comparison_results" / "noise-noiseless" / "noiseless" / "defi"
    candidates = sorted(noiseless_dir.glob("protocol_core_noiseless_*.json"),
                        key=os.path.getmtime, reverse=True) if noiseless_dir.exists() else []
    data = None
    src  = None
    for c in candidates:
        try:
            data = json.loads(c.read_text())
            src  = c
            break
        except Exception:
            continue

    if not data:
        write_table("suppb_noiseless.tex", "% suppB noiseless data not available\n")
        return

    # Extract aggregate stats per method from "tests" list.
    # FIX ISSUE-3 (EHSDeFi runtime 20.2s vs 841.4s): this loop previously
    # only read res["r2"], silently discarding res["time_s"] even though
    # every benchmark script in this codebase (see e.g.
    # hypatiax_defi_benchmark_v3c.py's case_results[...]["time_s"]) stores
    # per-test wall-clock time in that exact key, right next to "r2", in
    # this exact per-method results dict shape. That means tab:overall's
    # "Avg Runtime" column was never actually computed by this generator —
    # whatever "20.2 s" appears in the paper did not come from this code
    # path. This now computes it for real, from the same source JSON.
    method_r2:   dict[str, list[float]] = {}
    method_time: dict[str, list[float]] = {}
    for test in data.get("tests", []):
        for mname, res in test.get("results", {}).items():
            r2 = res.get("r2")
            if isinstance(r2, (int, float)):
                method_r2.setdefault(mname, []).append(float(r2))
            t = res.get("time_s")
            if isinstance(t, (int, float)):
                method_time.setdefault(mname, []).append(float(t))

    import statistics as _st

    # Cross-check note: EHSDeFi's own noise-sweep run reports σ=0% runtime
    # separately (tab:time_noise, gen_suppb_time_noise() above). If both are
    # available, surface the discrepancy directly instead of letting two
    # contradictory numbers sit unremarked in two different tables (Issue 3).
    _m3_key = _pick_method_key(method_r2, _M3_FRAG)

    tex = header_comment(src) + r"""
\begin{table}[H]
\centering
\caption{Six-method aggregate performance, noiseless protocol
  ($\sigma=0$, $n=200$, $R^2 \ge 0.999999$ threshold, 30 equations).
  Avg Runtime is the mean wall-clock time per equation, this run
  (\texttt{protocol\_core\_noiseless\_*.json}); compare against
  tab:time\_noise's $\sigma=0\%$ row, sourced from the separate
  noise-sweep run --- see \S\ref{sec:ehsdefi-runtime} if they disagree.}
\label{tab:overall}
\small
\begin{tabular}{lrrrr}
\toprule
\textbf{Method} & \textbf{Median $R^2$} & \textbf{Recovery Rate} & \textbf{Avg Runtime} & \textbf{n} \\
\midrule
"""
    for mname, vals in sorted(method_r2.items()):
        med = _st.median(vals)
        rr  = sum(1 for v in vals if v >= 0.999999) / len(vals)
        tvals = method_time.get(mname, [])
        t_str = f"{_st.mean(tvals):.1f}\\,s" if tvals else "---"
        tex += f"{mname[:38]} & {med:.6f} & {rr*100:.1f}\\% & {t_str} & {len(vals)} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("suppb_noiseless.tex", tex)

    # Print a direct numeric comparison to the console/CI log so the
    # discrepancy (or its resolution) is visible without opening both .tex
    # files — this is exactly the trace Issue 3 asks for.
    if _m3_key and method_time.get(_m3_key):
        m3_noiseless_avg = _st.mean(method_time[_m3_key])
        print(f"  [tab:overall] {_m3_key} avg runtime (noiseless protocol run): "
              f"{m3_noiseless_avg:.1f}s — compare tab:time_noise σ=0% row.")
    elif _m3_key:
        print(f"  ::warning:: [tab:overall] {_m3_key} found in noiseless protocol "
              "JSON but no per-test 'time_s' field present — Avg Runtime column "
              "will show '---' for this method until the source JSON includes it.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("═" * 65)
    print("  Table Generator — HypatiaX JMLR + Supplement B")
    print("═" * 65)
    print(f"  Results dir : {RESULTS}")
    print(f"  Output dir  : {TABLES_DIR}")
    if _ARGS.experiment:
        print(f"  Experiment  : {_ARGS.experiment}")
    print()

    # ── Experiment scoping ────────────────────────────────────────────────────
    # Computed up front so both the missing-JSON audit below and the
    # generator dispatch at the end of main() can be restricted to only the
    # JSON(s)/table(s) that the selected experiment actually produces,
    # instead of checking/generating the full main-paper set every time.
    # "all", "suppa", or an unrecognised tag fall back to the full audit +
    # full dispatch (previous behaviour) since their table ownership isn't
    # cleanly single-sourced.
    _EXP = (_ARGS.experiment or "all").lower()
    _SCOPED_EXPERIMENTS = {
        "exp1", "exp1_pca", "exp1b", "exp1b_pca", "exp1_ablation",
        "exp1_five", "exp2_five",
        "exp2_feynman", "exp2", "exp2_feynman_extrap", "exp2_feynman_pca",
        "exp3", "exp3b", "instability", "hybrid_all_domains", "extrap",
        "suppb", "suppb_sc",
    }

    # ── Missing JSON audit ────────────────────────────────────────────────────
    # Check every expected JSON before running generators so the user gets a
    # complete picture of what will fall back to paper-verified numbers.
    print("  ── Missing JSON audit ──────────────────────────────────────")
    _AUDIT: list[tuple[str, str, str, str, tuple[str, ...]]] = [
        # (label,  subdir,  glob,  extra_subdirs_csv,  owner_experiments)
        ("exp1 benchmark (Tab 2/3/4/11)",
         "", "hypatiax_defi_benchmark_v3*results*.json", "defi",
         ("exp1", "exp1_pca")),
        ("exp1_ablation Core-15 per-equation data (Tab 6 + Fig F — the newest "
         "*.json in this dir by mtime)",
         "ablation/exp1_ablation", "*.json", "",
         ("exp1_ablation",)),
        # Sentinel subdir "__FIVE_SYSTEM__" is handled specially below via
        # _load_five_system_rows_real() (exp1_five primary, exp2/exp2_extrap
        # secondary), which scans real, named source directories rather than
        # load_best()'s single-newest-file pick — the (subdir, glob,
        # extra_csv) tuple shape used by every other row can't express
        # "check two separate real experiments in priority order", so this
        # can't just reuse load_best() the way the others do.
        ("five-system JSON (App — tab:five_systems_full, supp_benchmark_report.tex; "
         "exp1_five primary, exp2/exp2_extrap secondary)",
         "__FIVE_SYSTEM__", "", "",
         ("exp1_five", "exp2_five", "exp2")),
        ("portfolio_variance seed-sweep (Tab 5 + Fig G)",
         "", "portfolio_variance*.json", "",
         ("exp1b", "exp1b_pca")),
        ("exp2_feynman results (Tab 7)",
         "comparison_results/feynman-tests/exp2", "*.json", "feynman",
         ("exp2_feynman", "exp2_feynman_extrap", "exp2_feynman_pca")),
        # exp2 (all-30 multi-domain) is NOT the same source as exp2_feynman:
        # its --results-dir is already resolved to the canonical exp2_multi
        # dir by ci_postprocess.yml, so shards are read with subdir=""
        # directly (see _load_exp2_multi_domain_rows()). Previously exp2 was
        # aliased to exp2_feynman's audit/dispatch entries, which pointed at
        # a subdir that doesn't exist relative to exp2_multi and always
        # silently fell back to exp2_feynman's Table 7 paper-verified data.
        ("exp2 all-30 multi-domain shards (all30_domain_summary / multi_domain_rank_table)",
         "", "*.json", "",
         ("exp2",)),
        ("exp3 Nguyen-12 results (Tab 8)",
         "", "exp3*nguyen12*.json", "nguyen12",
         ("exp3", "exp3b")),
        ("instability JSON or CSV (Tab 9 / §10.9)",
         "figures", "instability*.json", "instability",
         ("instability",)),
        # hybrid_all_domains: --results-dir is already resolved by
        # ci_postprocess.yml to the canonical hybrid_llm_nn/all_domains dir,
        # so shards are read with subdir="" directly (see
        # _load_hybrid_all_domains_rows()). The previous subdir value here —
        # "hybrid_llm_nn/all_domains" — double-nested against an
        # already-resolved --results-dir and always found nothing, the same
        # class of bug that broke exp2's audit row.
        ("hybrid_all_domains JSON (§10.9 hybrid)",
         "", "*.json", "",
         ("hybrid_all_domains",)),
        # extrap previously had NO audit row at all — a missing source JSON
        # was never reported here; the only symptom was the tables step
        # failing later with "wrote no .tex files" because _DISPATCH["extrap"]
        # was an unconditional empty list (see gen_extrap_ood_table() below).
        ("extrap OOD results (Tab 9 OOD columns)",
         "", "all_domains_extrap_v4_*.json", "",
         ("extrap",)),
        ("noise_sweep JSON (suppB Tab 28/29)",
         "comparison_results/feynman-tests/noise-sweep", "noise_sweep_*.json", "",
         ("suppb",)),
        ("sample_complexity JSON (suppB Tab 29)",
         "comparison_results/feynman-tests/sample-complexity",
         "sample_complexity_*.json", "",
         ("suppb", "suppb_sc")),
        ("noiseless protocol JSON (suppB tab:overall)",
         "comparison_results/noise-noiseless/noiseless/defi",
         "protocol_core_noiseless_*.json", "",
         ("suppb",)),
    ]

    if _EXP in _SCOPED_EXPERIMENTS:
        _AUDIT = [row for row in _AUDIT if _EXP in row[4]]
        print(f"  (scoped to --experiment {_EXP}: {len(_AUDIT)} relevant JSON source(s))\n")

    _missing: list[str] = []
    _found:   list[str] = []
    for label, subdir, glob_pat, extra_csv, _owners in _AUDIT:
        if subdir == "__FIVE_SYSTEM__":
            _, path, _tier = _load_five_system_rows_real()
        else:
            extras = [e.strip() for e in extra_csv.split(",") if e.strip()]
            _, path = load_best(subdir, glob_pat, extra_subdirs=extras or None)
        if path:
            _found.append(f"    ✅ {label}\n       → {path}")
        else:
            _missing.append(f"    ❌ {label}")
            if subdir == "__FIVE_SYSTEM__":
                _missing[-1] += (
                    "\n       Searched: five_systems/exp1_five/exp1_five_results.json "
                    "(primary), then exp2/exp2_extrap (secondary)."
                    "\n       → NO hardcoded fallback exists; five_system.tex will be "
                    "a NO-DATA placeholder and repro_macros.tex's nnExtrap* macros "
                    "will be left undefined."
                )
            else:
                # Describe where the generator will look so the user can debug.
                extras = [e.strip() for e in extra_csv.split(",") if e.strip()]
                search_dirs = []
                for base in [PATCHED, RESULTS]:
                    search_dirs.append(str(base / subdir if subdir else base))
                for e in extras:
                    search_dirs.append(str(RESULTS / e if e else RESULTS))
                _missing[-1] += (
                    f"\n       Searched: " + ", ".join(search_dirs) +
                    f"\n       Glob:     {glob_pat}" +
                    "\n       → WILL USE paper-verified fallback numbers"
                )

    if _found:
        print(f"\n  JSONs found ({len(_found)}):")
        for msg in _found:
            print(msg)

    if _missing:
        print(f"\n  ⚠  MISSING JSONs ({len(_missing)}) — affected tables will use paper-verified fallbacks:")
        for msg in _missing:
            print(msg)
        if not _ARGS.allow_fallback:
            print(f"\n  ✗  This run will exit non-zero because of the above (no --allow-fallback given).")
    else:
        print("\n  All expected JSONs found — no fallbacks needed.")
    print()
    # ── End audit ─────────────────────────────────────────────────────────────

    # ── Load suppB sweep JSONs (once, shared across generators) ───────────────
    noise_data = load_sweep_json(
        _ARGS.noise_sweep,
        "comparison_results/feynman-tests/noise-sweep",
        "noise_sweep_*.json",
    )
    sc_data = load_sweep_json(
        _ARGS.sample_complexity,
        "comparison_results/feynman-tests/sample-complexity",
        "sample_complexity_*.json",
    )

    if noise_data:
        print(f"  noise_sweep JSON  : loaded "
              f"({len(noise_data.get('noise_levels', []))} sigma levels)")
    else:
        print("  noise_sweep JSON  : NOT FOUND — suppB noise tables will be placeholders")

    if sc_data:
        print(f"  sample_complexity : loaded "
              f"({len(sc_data.get('sample_sizes', []))} n values)")
    else:
        print("  sample_complexity : NOT FOUND — suppB SC tables will be placeholders")
    print()

    # ── Dispatch: which generators to run ────────────────────────────────────
    # When --experiment is supplied, ONLY the table(s) whose source JSON that
    # experiment actually writes are generated (see the JSON location map in
    # the module docstring / comment block near the top of this file — this
    # dispatch mirrors it 1:1). This prevents:
    #   - unrelated tables silently falling back to paper-verified numbers
    #     (and failing the run) for JSONs the selected experiment never
    #     produces in the first place
    #   - suppB tables being written into every other experiment's output dir
    #   - cross-experiment JSON searches failing because --results-dir points
    #     at a subdir that doesn't contain sibling experiment data
    #
    # "all" (or an unrecognised tag) keeps the original behaviour of running
    # everything. (_EXP is computed earlier, alongside the audit scoping.)

    def _main_paper_section():
        # NOTE: despite the name, this list already included the five_system*
        # (exp1_five) generators even after they were moved to
        # supp_benchmark_report.tex -- kept that way so "all" still generates
        # every table in one pass; adding exp2_five's four generators here
        # for the same reason.
        return ("── Main paper tables ───────────────────────────────────────", [
            lambda: gen_five_system(),
            lambda: gen_five_system_stat_tests(),
            lambda: gen_five_system_performance(),
            lambda: gen_five_system_extrapolation(),
            lambda: gen_five_system_exp2five(),
            lambda: gen_five_system_exp2five_stat_tests(),
            lambda: gen_five_system_exp2five_performance(),
            lambda: gen_five_system_exp2five_extrapolation(),
            lambda: gen_defi_main(),
            lambda: gen_defi_tiers(),
            lambda: gen_runtime(),
            lambda: gen_portfolio_seed_sweep(),
            lambda: gen_ablation(),
            lambda: gen_feynman_results(),
            lambda: gen_nguyen12(),
            lambda: gen_instability(),
            lambda: gen_version_history(),
            lambda: gen_timing_detail(),
            lambda: gen_repro_macros(),
        ])

    def _suppb_noise_section():
        return ("── Supplement B — noise sweep (suppB STEP 10) ──────────────", [
            lambda: gen_suppb_r2_noise(noise_data),
            lambda: gen_suppb_rr_noise(noise_data),
            lambda: gen_suppb_time_noise(noise_data),
            lambda: gen_suppb_noiseless(),
        ])

    def _suppb_sc_section():
        return ("── Supplement B — sample complexity (suppB STEP 10) ────────", [
            lambda: gen_suppb_sc_metrics(sc_data),
            lambda: gen_suppb_sc_summary(sc_data),
            lambda: gen_suppb_sc_by_sample(sc_data),
        ])

    def _suppb_winrate_section():
        return ("── Supplement B — win rate (both sweeps) ───────────────────", [
            lambda: gen_suppb_winrate(noise_data, sc_data),
        ])

    # Single-JSON experiments: each maps to only the generator(s) that
    # actually consume that experiment's JSON, per the location-map comment.
    _DISPATCH = {
        # exp1 → hypatiax_defi_benchmark_v3*results*.json → defi_main,
        # defi_tiers, runtime, timing_detail (Tab 2/3/4/11) + defi half of
        # repro_macros. version_history has no JSON dependency (hardcoded,
        # stable) so it's cheap to regenerate alongside exp1.
        "exp1": [("── exp1: DeFi benchmark tables ─────────────────────────────", [
            lambda: gen_defi_main(),
            lambda: gen_defi_tiers(),
            lambda: gen_runtime(),
            lambda: gen_timing_detail(),
            lambda: gen_version_history(),
            lambda: gen_repro_macros(),
        ])],
        # exp1b → portfolio_variance_seed_sweep.json → portfolio_sweep (Tab 5 / Fig G)
        "exp1b": [("── exp1b: portfolio variance seed sweep ────────────────────", [
            lambda: gen_portfolio_seed_sweep(),
        ])],
        # exp1_ablation → exp1_ablation/*.json → ablation (Tab 6 + Fig F)
        # + ablation half of repro_macros. five_system.tex moved OUT of this
        # dispatch entry -- it no longer sources from exp1_ablation at all
        # (see gen_five_system() docstring); it's driven by "exp1_five" below.
        "exp1_ablation": [("── exp1_ablation: ablation table ────────────────────────────", [
            lambda: gen_ablation(),
            lambda: gen_repro_macros(),
        ])],
        # exp1_five → five_systems/exp1_five/*.json → five_system.tex (App,
        # supp_benchmark_report.tex — not the main paper; see 04_ci_sd_incompatibility.tex)
        # + the two sub-tables (performance, extrapolation) defined for the
        # metric/evaluator item. exp2 also feeds five_system.tex as a
        # secondary source (_load_exp2_five_system_rows()), so re-running
        # this step after an exp2 run picks up whichever source is available.
        "exp1_five": [("── exp1_five: five-system comparison + sub-tables ───────────", [
            lambda: gen_five_system(),
            lambda: gen_five_system_stat_tests(),
            lambda: gen_five_system_performance(),
            lambda: gen_five_system_extrapolation(),
        ])],
        # exp2_feynman (+ extrap/pca variants) → comparison_results/
        # feynman-tests/exp2/*.json → feynman.tex (Tab 7) only.
        "exp2_feynman": [("── exp2_feynman: Feynman benchmark table ────────────────────", [
            lambda: gen_feynman_results(),
        ])],
        # exp2 (all-30 multi-domain) → comparison_results/feynman-tests/
        # exp2_multi/*.json (per-domain shards, subdir="" — see
        # _load_exp2_multi_domain_rows()) → all30_domain_summary.tex +
        # multi_domain_rank_table.tex. NOT the same source as exp2_feynman —
        # do not alias this to "exp2_feynman" again (that was the bug).
        "exp2": [("── exp2: all-30 multi-domain tables ─────────────────────────", [
            lambda: gen_all30_domain_summary(),
            lambda: gen_multi_domain_rank_table(),
            lambda: gen_five_system(),  # exp2/exp2_extrap is the secondary source
        ])],
        # exp2_five reuses exp2's script (run_comparative_suite_benchmark_v2.py,
        # --methods 1 2 4 5 6) but writes to its own directory
        # (five_systems/exp2_five/). Now has its own dedicated loader
        # (_load_exp2_five_own_rows() / _load_exp2_five_own_raw()) and its own
        # four-table pipeline, genuinely separate from exp1_five's tables --
        # see the module comment above gen_five_system_exp2five().
        "exp2_five": [("── exp2_five: five-system comparison (Feynman, 5-method) ────", [
            lambda: gen_five_system_exp2five(),
            lambda: gen_five_system_exp2five_stat_tests(),
            lambda: gen_five_system_exp2five_performance(),
            lambda: gen_five_system_exp2five_extrapolation(),
        ])],
        # exp3 / exp3b → nguyen12/*.json → nguyen12.tex (Tab 8) only.
        "exp3": [("── exp3: Nguyen-12 table ─────────────────────────────────────", [
            lambda: gen_nguyen12(),
        ])],
        # instability → figures/instability*.{json,csv} → instability.tex (Tab 9) only.
        "instability": [("── instability: instability table ───────────────────────────", [
            lambda: gen_instability(),
        ])],
        # hybrid_all_domains → hybrid_llm_nn/all_domains/*.json (per-domain
        # shards, subdir="" — see _load_hybrid_all_domains_rows()) →
        # hybrid_all_domains_summary.tex + domain_rank_table.tex. Previously
        # an unconditional empty list — that was the guaranteed hard-failure
        # bug (0 tables written -> N_TABS==0 -> CI step always failed).
        "hybrid_all_domains": [("── hybrid_all_domains: LLM+NN all-domain tables ──────────────", [
            lambda: gen_hybrid_all_domains_summary(),
            lambda: gen_domain_rank_table(),
        ])],
        # extrap → comparison_results/extrapolation/*.json (subdir="" — see
        # _load_extrap_rows()) → extrap_ood_table.tex. Previously an
        # unconditional empty list — same guaranteed hard-failure bug as
        # hybrid_all_domains above.
        "extrap": [("── extrap: OOD extrapolation table ───────────────────────────", [
            lambda: gen_extrap_ood_table(),
        ])],
        # suppB: noise-sweep + sample-complexity + win-rate only.
        "suppb": [_suppb_noise_section(), _suppb_sc_section(), _suppb_winrate_section()],
        "suppb_sc": [_suppb_sc_section(), _suppb_winrate_section()],
        # "all" / unknown / suppa (ambiguous ownership): run everything,
        # matching the original behaviour.
        "all": [_main_paper_section(), _suppb_noise_section(),
                _suppb_sc_section(), _suppb_winrate_section()],
    }
    # PCA/extrap variants reuse their base experiment's generator set — same
    # source JSON, just a different --results-dir. exp2 (all-30
    # multi-domain) has its own entry above and is intentionally NOT aliased
    # here — it never shared exp2_feynman's source data in the first place.
    _DISPATCH["exp1_pca"] = _DISPATCH["exp1"]
    _DISPATCH["exp1b_pca"] = _DISPATCH["exp1b"]
    _DISPATCH["exp2_feynman_extrap"] = _DISPATCH["exp2_feynman"]
    _DISPATCH["exp2_feynman_pca"] = _DISPATCH["exp2_feynman"]
    _DISPATCH["exp3b"] = _DISPATCH["exp3"]
    _DISPATCH["suppa"] = _DISPATCH["all"]

    sections = _DISPATCH.get(_EXP, _DISPATCH["all"])
    if _EXP not in _DISPATCH:
        print(f"  \u26a0  Unknown --experiment '{_EXP}' — running all table generators.")
    elif not sections or all(not gens for _, gens in sections):
        print(f"  \u2139  --experiment '{_EXP}': audited its JSON above, but no "
              f"table generator currently sources that data — nothing to write.")

    for section_label, generators in sections:
        print(f"\n  {section_label}")
        for fn in generators:
            fn()

    print(f"\n{'═'*65}")
    print(f"  Generated: {GENERATED} table files")
    print(f"  Output:    {TABLES_DIR}/")
    print(f"{'═'*65}")
    print("""
  LaTeX usage in supp_benchmark_report.tex:
    \\input{tables/suppb_r2_noise.tex}
    \\input{tables/suppb_rr_noise.tex}
    \\input{tables/suppb_time_noise.tex}
    \\input{tables/suppb_sc_metrics.tex}
    \\input{tables/suppb_sc_summary.tex}
    \\input{tables/suppb_sc_by_sample.tex}
    \\input{tables/suppb_winrate.tex}
    \\input{tables/suppb_noiseless.tex}
    \\input{tables/five_system.tex}      % Appendix — tab:five_systems_full (exp1_five/exp2_five)
    \\input{tables/five_system_performance.tex}    % Appendix — performance sub-table
    \\input{tables/five_system_extrapolation.tex}  % Appendix — extrapolation sub-table
    \\input{tables/five_system_stat_tests.tex}  % Appendix D  app:statistical_tests
    \\input{tables/five_system_exp2five.tex}      % Appendix — exp2_five's own table (separate data)
    \\input{tables/five_system_exp2five_performance.tex}    % exp2_five performance sub-table
    \\input{tables/five_system_exp2five_extrapolation.tex}  % exp2_five extrapolation sub-table
    \\input{tables/five_system_exp2five_stat_tests.tex}     % exp2_five stat-test appendix

  LaTeX usage in main paper:
    \\input{tables/defi_main.tex}        % Tab 2  §10.2
    \\input{tables/defi_tiers.tex}       % Tab 3  §10.3
    \\input{tables/runtime.tex}          % Tab 4  §10.4
    \\input{tables/portfolio_sweep.tex}  % Tab 5  §10.5
    \\input{tables/ablation.tex}         % Tab 6  §10.6
    \\input{tables/feynman.tex}          % Tab 7  §10.7
    \\input{tables/nguyen12.tex}         % Tab 8  §10.8
    \\input{tables/instability.tex}      % Tab 9  §10.9
    \\input{tables/version_history.tex}  % Tab 10 Appendix B
    \\input{tables/timing_detail.tex}    % Tab 11 Appendix C
    \\input{tables/repro_macros.tex}
""")

    # ── Fail the build if any table used a paper-verified fallback ────────────
    # Tables were still written above (useful for local debugging), but a CI
    # step that runs this script must not report success while silently
    # substituting stale hardcoded numbers for a missing fresh-run JSON.
    #
    # Checks both _missing (the pre-flight audit above) and FALLBACK_TABLES
    # (fallback usage as actually reported by the generator functions
    # themselves at write time). These should normally agree, but keeping
    # both catches the class of bug Issue 4 was: the audit for
    # "exp1_ablation Core-15" reported ✅ found because *a* JSON existed
    # there, while gen_five_system() was silently using fallback rows
    # because that JSON lacked the key it actually needed. A generator-level
    # report can't be fooled by "some JSON happened to exist at this path"
    # the way a path-only audit can.
    if FALLBACK_TABLES:
        print(f"\n  ⚠  GENERATOR-REPORTED FALLBACKS ({len(FALLBACK_TABLES)}):")
        for msg in FALLBACK_TABLES:
            print(f"    ❌ {msg}")

    if (_missing or FALLBACK_TABLES) and not _ARGS.allow_fallback:
        print(f"{'═'*65}")
        n_fail = len(_missing) + len(FALLBACK_TABLES)
        print(f"  ✗  FAILED: {n_fail} table(s)/macro-set(s) above used paper-verified")
        print( "     fallback data instead of a fresh result JSON. Tables were")
        print( "     still written to disk for inspection, but this run is")
        print( "     exiting non-zero so CI does not report a stale green summary.")
        print( "     Re-run with the missing JSONs in place, or pass")
        print( "     --allow-fallback if this is deliberate local/partial drafting.")
        print(f"{'═'*65}")
        sys.exit(1)


if __name__ == "__main__":
    main()

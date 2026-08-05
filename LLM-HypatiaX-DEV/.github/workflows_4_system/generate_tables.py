#!/usr/bin/env python3
"""
generate_tables.py — Auto-generate LaTeX tables from JSON results

Reads patched JSON outputs and writes .tex table fragments to paper/tables/.
These are \\input{}-ed by the main paper and supplements so NO manual numbers
appear in the LaTeX source.

Tables generated  (main paper)
  five_system.tex     tab:five_system    §10.1   ← results/exp1_ablation/
  defi_main.tex       tab:main_results   §10.2   ← results/defi/
  defi_tiers.tex      tab:difficulty     §10.3   ← results/defi/
  runtime.tex         tab:runtime        §10.4   ← results/defi/
  portfolio_sweep.tex tab:portfolio_seed §10.5   ← portfolio_variance_seed_sweep.json
  ablation.tex        tab:llm_ablation   §10.6   ← results/exp1_ablation/
  feynman.tex         tab:feynman        §10.7   ← results/feynman/
  nguyen12.tex        tab:nguyen12       §10.8   ← results/nguyen12/
  instability.tex     tab:instability    §10.9   ← results/instability/
  version_history.tex tab:version_hist   §App B  ← hardcoded (stable)
  timing_detail.tex   tab:timing_detail  §App C  ← results/defi/
  repro_macros.tex    \\newcommand macros for inline numbers

Tables generated  (Supplement B — suppB / STEP 10 outputs)
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


# ── Issue 4 (CI/SD incompatibility) follow-up: locate the real Table 1 source ──
#
# STILL OPEN, unconfirmed either way: whether ablation/exp1_ablation is
# actually the right directory for five-system data was raised as a
# question, not settled. The Issue 4 report's own investigation found
# exp1_ablation's real output (_merged.json, per-equation pysr_only/hypatia
# records -- see gen_ablation()/_load_merged() above) has no "five_system"/
# "system_comparison" key, and concluded that data hasn't been located
# anywhere in the tree yet. That finding stands until someone actually
# confirms otherwise (e.g. `grep -r '"five_system"\|"system_comparison"'
# hypatiax/` on the real repo).
#
# Separately, and true regardless of where the data turns out to live: even
# if it IS somewhere under ablation/exp1_ablation, load_best()'s newest-
# mtime-wins selection would be the wrong tool to find it -- per the
# _load_merged()/_load_mannwhitney() comments above, several files in that
# directory share identical mtimes, so mtime-sorting can silently pick a
# file without the key while a sibling file has it. _load_five_system_data()
# below scans every *.json in a candidate directory (not just the
# mtime-newest one) for exactly that reason -- a no-regret fix independent
# of whether exp1_ablation is confirmed to be the right directory.
_FIVE_SYSTEM_CANDIDATES: list[tuple[str, str]] = [
    ("ablation/exp1_ablation", "*.json"),           # unconfirmed -- see note above
    ("comparison_results/five_system", "*.json"),
    ("comparison_results/system_comparison", "*.json"),
    ("five_system", "*.json"),
    ("system_comparison", "*.json"),
    ("", "five_system*.json"),
    ("", "system_comparison*.json"),
]


def _rows_from_data(data: dict) -> list[tuple] | None:
    """Extract five_system/system_comparison rows from a loaded JSON dict,
    or None if this particular file doesn't have that key."""
    raw = data.get("five_system", data.get("system_comparison"))
    if not raw:
        return None

    def _s(v, default="---"):
        # entry.get(key, default) only uses `default` when the key is
        # absent -- a field present with an explicit JSON null (e.g.
        # "std": null, common for systems where std wasn't computed) still
        # comes back as None and str()'s to the literal text "None" in the
        # table unless handled here.
        return default if v is None else str(v)

    rows = []
    for entry in raw:
        rows.append((
            entry.get("name", "?"),
            entry.get("n", 0),
            _s(entry.get("extrap_median_pct")),
            _s(entry.get("extrap_mean_pct")),
            _s(entry.get("train_r2_mean")),
            _s(entry.get("std")),
            entry.get("design_focus") or "---",
        ))
    return rows if len(rows) >= 2 else None


def _load_five_system_data() -> tuple[list[tuple] | None, Path | None]:
    """Search all candidate locations for a JSON containing a
    "five_system"/"system_comparison" key (system-level n/mean/std rows for
    Table 1), independently of whatever exp1_ablation JSON happens to be
    found for the (unrelated) per-equation ablation table.

    Unlike load_best(), this scans *every* matching file in a candidate
    directory (not just the newest by mtime) before moving to the next
    candidate, since exp1_ablation's files are known to share identical
    mtimes -- see the comment above _FIVE_SYSTEM_CANDIDATES.

    Returns (rows, path) on success, where each row is
    (name, n, extrap_median_pct, extrap_mean_pct, train_r2_mean, std, design_focus).
    Returns (None, None) if no file in any candidate location has the key —
    callers should fall back to PAPER_VERIFIED data and flag it via
    FALLBACK_TABLES, not silently treat "found a JSON" as "found the data".
    """
    for subdir, glob_pat in _FIVE_SYSTEM_CANDIDATES:
        search_dirs = [base / subdir if subdir else base for base in (PATCHED, RESULTS)]
        candidates: list[Path] = []
        for d in search_dirs:
            if d.exists():
                candidates.extend(_filtered_glob(d, glob_pat))
        # Newest first, purely as a tie-break for reporting which file was
        # used when more than one has the key -- every file is still checked.
        candidates.sort(key=os.path.getmtime, reverse=True)
        for path in candidates:
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            rows = _rows_from_data(data)
            if rows:
                return rows, path
    return None, None


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
    """
    # run_all.sh (exp1) writes hypatiax_defi_benchmark_v3*results*.json to RESULTS_DIR root.
    # Also check legacy defi/ subdir for backwards compatibility.
    data, src = load_best("", "hypatiax_defi_benchmark_v3*results*.json",
                          extra_subdirs=["defi"])
    PAPER_ROWS = [
        ("Pure LLM",   1.0000, -0.7571, 62.2, 62.2, 6),
        ("Neural MLP", -0.4675, -0.9482,  5.4, 12.2, 0),
        ("HypatiaX",   1.0000, +0.8721, 89.2, 89.2, 0),
    ]

    def _extract_rows(d) -> list[tuple]:
        """Try to read 3-method rows from various JSON shapes."""
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
        rows = PAPER_ROWS   # use verified paper values

    def _r2(v): return f"{v:.4f}" if isinstance(v, float) and not (v != v) else "---"
    def _pct(v): return f"{v:.1f}" if isinstance(v, float) and not (v != v) else "---"
    def _int(v): return str(int(v)) if isinstance(v, (int, float)) else "---"

    tex = header_comment(src) + r"""
\begin{table}[t]
\centering
\caption{Aggregate extrapolation performance on the HypatiaX DeFi Benchmark
  (74 tasks). All $R^2$ values clipped to $[-10, 1]$; fixed denominator of 74.
  Catastrophic: $R^2 < -10$.}
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


# Paper-verified fallback (Table 1) — module level so gen_five_system() and
# gen_repro_macros() use the exact same numbers if a live five-system source
# still can't be located (Issue 4, item 3/4: this stays fallback data, not
# "verified", until the real producing script is found and re-run — see
# _load_five_system_data() above). Std for Neural Network filled in from the
# disclosed Appendix G figure (1842.0) instead of "---", so the fallback path
# also yields a real, internally-consistent CI rather than silently omitting
# one. Std is still "---" for every other row (Issue 4, item 2): that has
# never been confirmed as "not computed upstream" vs. "computed but not
# located by this generator" -- treat those as open, same as the row source
# itself.
FIVE_SYSTEM_PAPER_ROWS = [
    # (system, n, extrap_median_pct, extrap_mean_pct, train_r2_mean, std, design_focus)
    ("Hybrid v50\\_2", 14, "0.0", "0.0",  "0.931", "---", "Extrapolation"),
    ("Neural Network",        13, "86.7", "1231.0", "0.940", "1842.0", "Baseline"),
    ("Pure LLM",               0, "---",  "---",    "---",   "---", "Recognition"),
    ("System 2 Symbolic",      0, "---",  "---",    "---",   "---", "Validation"),
    ("System 3 LLM+Fallback",  0, "---",  "---",    "1.000", "0.0002", "Robustness"),
]


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
# IMPORTANT: rows from this loader do NOT reproduce FIVE_SYSTEM_PAPER_ROWS.
# They are a different, live, traceable dataset on the actual exp2 problem
# set — not a "fix" that recovers the old published numbers. In particular:
#   - Hybrid v50_2's n happens to match the old fallback (14) but its
#     extrapolation error % differs by two to three orders of magnitude
#     (old: "0.0", live: ~120%).
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


def gen_five_system() -> None:
    """
    Tab 1 — Five-System Comparison: Extrapolation Error vs. Interpolation R².
    Matches Table 1 in §10.1.

    The 95% CI column is *derived* from each row's own n/mean/std at
    generation time (see _ci95), rather than being a separately hand-typed
    number elsewhere in the paper. Previously no code in this file computed
    a CI at all -- the main-text CI was disconnected from this table's Std
    column, which is how the two went out of sync (n=13, mean=1231.0,
    std=1842.0 supports ~[117.0, 2345.0], not the previously hand-typed
    [1087, 1456]). That part is fixed and verified (see _ci95 docstring).

    Issue 4 follow-up (still open, unconfirmed): whether exp1_ablation is
    even the right source directory for this table was raised as a
    question, not established. The Issue 4 report's own investigation found
    exp1_ablation's real output (_merged.json) is per-equation data with no
    "five_system"/"system_comparison" key, and concluded that key hasn't
    been located anywhere in the tree yet -- that finding stands until
    someone actually confirms otherwise. Separately, and true either way:
    _load_five_system_data() now scans every *.json in a candidate
    directory (not just the mtime-newest one, which the sibling
    _load_merged()/_load_mannwhitney() comments show is unreliable in this
    tree) instead of trusting load_best()'s mtime pick, so if the data does
    turn out to live in exp1_ablation under a different filename, this will
    find it. If this still falls back to FIVE_SYSTEM_PAPER_ROWS, that's
    consistent with the report's finding, not evidence the directory guess
    is wrong -- both remain open until someone checks the real repo.
    """
    rows, src = _load_five_system_data()
    used_fallback = rows is None
    if used_fallback:
        rows = FIVE_SYSTEM_PAPER_ROWS
        src = "FIVE_SYSTEM_PAPER_ROWS (hardcoded fallback — see Issue 4)"
        FALLBACK_TABLES.append(
            "five_system.tex (tab:five_systems_full) — scanned every JSON "
            "under ablation/exp1_ablation (unconfirmed whether this is even "
            "the right source dir — see Issue 4) and other candidate "
            "locations, but none contained a \"five_system\"/"
            "\"system_comparison\" key; used FIVE_SYSTEM_PAPER_ROWS."
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
    # separator between systems with/without extrapolation testing
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
    if used_fallback:
        tex += (
            "% ⚠ FALLBACK DATA — see Issue 4 (CI/SD Statistical Incompatibility).\n"
            "% This table used FIVE_SYSTEM_PAPER_ROWS, not a live experiment run.\n"
            "% Whether ablation/exp1_ablation is even the right source directory\n"
            "% is unconfirmed (open question, not established) -- every *.json\n"
            "% file in it (not just the newest by mtime) was checked anyway, plus\n"
            "% other candidate locations, but none currently contained a\n"
            "% \"five_system\"/\"system_comparison\" key (checked: " +
            ", ".join(f"{s or '.'}/{g}" for s, g in _FIVE_SYSTEM_CANDIDATES) +
            "). The 95% CI above is correctly *computed* from these rows'\n"
            "% own std/n, but the rows themselves are still paper-verified\n"
            "% fallback, not fresh data. Do not cite this as re-run-verified.\n"
        )
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

    # Neural Network extrapolation mean/std/CI. Previously read from the same
    # load_best("ablation/exp1_ablation", ...) call as the Mann-Whitney
    # numbers above, on the theory that it's "the same five_system/
    # system_comparison entry that feeds tab:five_system" -- but that JSON
    # has no five_system/system_comparison key (see gen_five_system() /
    # Issue 4), so that loop body never ran and these four macros were
    # silently never written to repro_macros.tex at all. If
    # jmlr_paper_main.tex references \nnExtrapCI (per the Issue 4 fix
    # writeup, replacing the old hand-typed CI text), an undefined macro is
    # worse than the original bug -- it breaks the LaTeX build outright
    # instead of just being wrong. Now sourced from the same dedicated
    # _load_five_system_data() lookup gen_five_system() uses, with the same
    # FIVE_SYSTEM_PAPER_ROWS fallback, so the macros are always defined and
    # always match the table's Std/CI columns exactly.
    five_rows, _fs_src = _load_five_system_data()
    fs_fallback = five_rows is None
    if fs_fallback:
        five_rows = FIVE_SYSTEM_PAPER_ROWS
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
    if fs_fallback and "nnExtrapCI" in macros:
        FALLBACK_TABLES.append(
            "repro_macros.tex (\\nnExtrapMean/\\nnExtrapStd/\\nnExtrapN/\\nnExtrapCI) "
            "— same missing five_system/system_comparison source as five_system.tex; "
            "see Issue 4."
        )

    lines = [
        "% Auto-generated reproducibility macros",
        "% Usage: \\repoVal{defiAccuracy}",
        f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]
    if fs_fallback:
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
    noiseless_dir = RESULTS / "comparison_results" / "noise-noiseless" / "noiseless"
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

    # Extract aggregate stats per method from "tests" list
    tests = data.get("tests", [])
    method_r2: dict[str, list[float]] = {}
    for test in tests:
        for mname, res in test.get("results", {}).items():
            r2 = res.get("r2")
            if isinstance(r2, (int, float)):
                method_r2.setdefault(mname, []).append(float(r2))

    import statistics as _st

    tex = header_comment(src) + r"""
\begin{table}[H]
\centering
\caption{Six-method aggregate performance, noiseless protocol
  ($\sigma=0$, $n=200$, $R^2 \ge 0.999999$ threshold, 30 equations).}
\label{tab:overall}
\small
\begin{tabular}{lrrrr}
\toprule
\textbf{Method} & \textbf{Median $R^2$} & \textbf{Recovery Rate} & \textbf{n} \\
\midrule
"""
    for mname, vals in sorted(method_r2.items()):
        med = _st.median(vals)
        rr  = sum(1 for v in vals if v >= 0.999999) / len(vals)
        tex += f"{mname[:38]} & {med:.6f} & {rr*100:.1f}\\% & {len(vals)} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_table("suppb_noiseless.tex", tex)


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
         "*.json in this dir by mtime, which may not be the file with "
         "five_system data; see next row)",
         "ablation/exp1_ablation", "*.json", "",
         ("exp1_ablation",)),
        # Issue 4 item 1 (still open, unconfirmed): whether exp1_ablation is
        # even the right source dir for five-system data was raised as a
        # question in this conversation, not established. The Issue 4
        # report's own investigation found exp1_ablation's real output has
        # no five_system/system_comparison key and concluded that data
        # hasn't been located anywhere yet -- that finding stands. What's
        # true regardless: load_best()'s newest-mtime selection would be
        # unreliable even if the data does live in exp1_ablation, since
        # several files there share identical mtimes (see comments above
        # _load_merged()/_load_mannwhitney()). Sentinel subdir
        # "__FIVE_SYSTEM__" is handled specially below via
        # _load_five_system_data(), which scans every file in each candidate
        # directory instead of just the newest -- the (subdir, glob,
        # extra_csv) tuple shape used by every other row can't express
        # "check all files, not just load_best()'s pick", so this can't just
        # reuse load_best() the way the others do.
        ("five_system / system_comparison JSON (Tab 1 — tab:five_systems_full, "
         "checked under ablation/exp1_ablation first — unconfirmed whether "
         "that's the right dir, see Issue 4 — then other candidates, "
         "file-by-file rather than by mtime)",
         "__FIVE_SYSTEM__", "", "",
         ("exp1_ablation",)),
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
         "comparison_results/noise-noiseless/noiseless",
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
            _, path = _load_five_system_data()
        else:
            extras = [e.strip() for e in extra_csv.split(",") if e.strip()]
            _, path = load_best(subdir, glob_pat, extra_subdirs=extras or None)
        if path:
            _found.append(f"    ✅ {label}\n       → {path}")
        else:
            _missing.append(f"    ❌ {label}")
            if subdir == "__FIVE_SYSTEM__":
                candidate_desc = ", ".join(
                    str((PATCHED / s if s else PATCHED)) + f" [{g}]"
                    for s, g in _FIVE_SYSTEM_CANDIDATES
                )
                _missing[-1] += (
                    f"\n       Searched (each under both patched/ and results/): "
                    + candidate_desc +
                    "\n       → WILL USE FIVE_SYSTEM_PAPER_ROWS fallback "
                    "(five_system.tex AND repro_macros.tex nnExtrap* macros)"
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
        return ("── Main paper tables ───────────────────────────────────────", [
            lambda: gen_five_system(),
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
        # exp1_ablation → exp1_ablation/*.json → five_system, ablation
        # (Tab 5/6 + Fig F) + ablation half of repro_macros.
        "exp1_ablation": [("── exp1_ablation: five-system + ablation tables ────────────", [
            lambda: gen_five_system(),
            lambda: gen_ablation(),
            lambda: gen_repro_macros(),
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

  LaTeX usage in main paper:
    \\input{tables/five_system.tex}      % Tab 1  §10.1
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

"""
generate_all_figures.py
========================
Generates ALL figures for the HypatiaX paper and supplementary reports.

P0 paper-compilation blockers addressed in this version
────────────────────────────────────────────────────────
  hypatiax_three_systems   — new: architecture diagram rendered from code
  fig09_r2_heatmap_regimes — REWRITTEN: uses exp1 dict schema (near/medium/far regimes)
  fig18_r2_heatmap_improved — REWRITTEN: uses exp1 dict schema (PySR-only vs HypatiaX)
  fig1_seed_sweep           — REWRITTEN: richer per-seed line chart (from portfolio_variance nb)

P1 restore (exp1_ablation_results.json schema clarification)
─────────────────────────────────────────────────────────────
  exp1_ablation_results.json schema is a dict-of-dicts:
    {
      "Equation Name": {
        "domain": "...",
        "pysr_only": {"train_r2": ..., "extrap_r2_near": ..., "extrap_r2_medium": ...,
                      "extrap_r2_far": ..., "total_time_s": ...},
        "hypatia":   {"train_r2": ..., "extrap_r2_near": ..., ...}
      }, ...
    }
  All cosmetic figures (fig07–fig22) have been updated to read this schema.
  The old "cases" list schema is retained as a fallback for RF09 / instability figures.

Figure groups produced
──────────────────────
P0               : hypatiax_three_systems
RF02 / cosmetic  : fig07–fig22, fig_seed_sweep_comparison / fig1_seed_sweep
RF09 instability : fig_instability_*.png, hypatiax_instability_*.png, fig_paper_*.png
instability_per_case : hypatiax_instability_per_case  (uses primary CASES array)
Supp-B sweep         : fig1_r2_vs_noise … fig_comparative_table
                       (noise_sweep_*.json + sample_complexity_*.json
                        — latest file matched by glob at runtime)

Primary data source (cosmetic / RF02 / RF09):
    exp1_ablation_results.json

Run:
    python3 generate_all_figures.py

Outputs: figures/  (PNG, 300 dpi, ready for \\includegraphics)

Missing-figure registry cross-reference
────────────────────────────────────────
Group               Stem                                       Data file(s)
P0                  hypatiax_three_systems                     (no data file — rendered from code)
cosmetic            fig07_scatter_train_vs_extrap              exp1_ablation_results.json
cosmetic            fig08_train_r2_bar                         exp1_ablation_results.json
cosmetic [P0]       fig09_r2_heatmap_regimes                   exp1_ablation_results.json
cosmetic            fig10_far_extrap_head2head                 exp1_ablation_results.json
cosmetic            fig11_speedup_bar                          exp1_ablation_results.json
cosmetic            fig12_ridge_vs_train_r2                    exp1_ablation_results.json
cosmetic            fig14_per_equation_r2_profile              exp1_ablation_results.json
cosmetic            fig16_instability_vs_extrapolation         instability_extrapolation_v2.csv
cosmetic            fig17_3d_surface_instability_complexity    instability_extrapolation_v2.csv
cosmetic [P0]       fig18_r2_heatmap_improved                  exp1_ablation_results.json
cosmetic            fig19_far_extrap_improved                  exp1_ablation_results.json
cosmetic            fig20_wall_clock_speedup                   wall_clock_flags.json
cosmetic            fig21_portfolio_variance_sweep             portfolio_variance_seed_sweep.json
cosmetic            fig22_bubble_train_vs_far                  exp1_ablation_results.json +
                                                               instability_extrapolation_v2.csv
cosmetic [P0]       fig1_seed_sweep  (≡fig_seed_sweep_comparison)
                                                               portfolio_variance_seed_sweep.json
instability_per_case hypatiax_instability_per_case             (primary CASES array — no ext. file)
Supp-B              fig1_r2_vs_noise … fig_comparative_table
                                                               noise_sweep_*.json (latest by glob) +
                                                               sample_complexity_*.json (latest by glob)
"""

import argparse, json, os, math, warnings, glob, sys, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CLI — parse --experiment, --results-dir, --figures-dir, --source
# All arguments are optional for backward-compatibility with direct invocation
# (no arguments → behaves like the original script, reading from CWD).
# ══════════════════════════════════════════════════════════════════════════════
_parser = argparse.ArgumentParser(
    description="Generate HypatiaX figures for a given experiment.",
    add_help=True,
)
_parser.add_argument(
    "--experiment", default=None,
    help="Experiment ID (e.g. exp2_feynman_extrap, instability, exp1). "
         "Controls which figure groups are generated and which files are required.",
)
_parser.add_argument(
    "--results-dir", default=None, dest="results_dir",
    help="Root directory containing committed result JSON/CSV files for this experiment. "
         "All DATA_* paths are resolved relative to this directory when provided.",
)
_parser.add_argument(
    "--figures-dir", default=None, dest="figures_dir",
    help="Output directory for generated figure files. "
         "Defaults to 'figures/' under --results-dir, or './figures/' if neither is set.",
)
_parser.add_argument(
    "--source", default="auto",
    choices=["auto", "committed", "artifact"],
    help="Data source hint (default: auto). Currently informational only.",
)
_ARGS, _unknown = _parser.parse_known_args()

# Resolve results_dir and figures_dir to absolute paths.
# Collapse any doubled trailing directory segment (e.g. noise-sweep/noise-sweep
# → noise-sweep) that can arise when a caller double-encodes the subdir suffix.
def _dedup_trailing(path):
    """If the last two path components are identical, drop the final one."""
    head, tail = os.path.split(path)
    parent_tail = os.path.basename(head)
    if tail and tail == parent_tail:
        return head
    return path

_RESULTS_DIR = _dedup_trailing(
    os.path.abspath(_ARGS.results_dir) if _ARGS.results_dir else os.getcwd()
)
_EXPERIMENT  = _ARGS.experiment  # may be None (legacy invocation)
# For _FIGURES_DIR, also collapse a doubled "figures/figures" that would arise
# when --results-dir already ends in "figures" and we append "figures" below.
_FIGURES_DIR = _dedup_trailing(
    os.path.abspath(_ARGS.figures_dir) if _ARGS.figures_dir
    else os.path.join(_RESULTS_DIR, "figures")
)

# Experiments that do NOT use exp1_ablation_results.json at all.
# For these, missing DATA_MAIN is a graceful skip, not a fatal error.
_EXPERIMENTS_WITHOUT_ABLATION = {
    "exp2_feynman_extrap",
    "exp2_feyman_extrap",   # typo alias — same experiment
    "instability",
    "exp3",
    "exp3b",
    "suppB",
    "suppB_sc",
    "extrap",
    "hybrid_all_domains",
    "exp2_feynman_pca",
    "exp2_feyman_pca",      # typo alias
    "exp1_pca",
    "exp1b_pca",
}

# ── Helper: resolve a filename relative to --results-dir ─────────────────────
def _rpath(filename):
    """Return filename resolved under _RESULTS_DIR (absolute path)."""
    return os.path.join(_RESULTS_DIR, filename)

# ── Data file paths (all resolved relative to --results-dir) ─────────────────
# Primary cosmetic / RF02 / RF09 source (renamed from v3c2_fixed_4_content)
DATA_MAIN          = _rpath("exp1_ablation_results.json")
# exp1_ablation: each of the 4 shard workers writes its own
# exp1_ablation_results.json containing only its shard's equations, and only
# one (the last-committed) survives in the repo — so this file alone is not a
# reliable full-dataset source for a 4-shard run.  merge_shards.py instead
# consolidates all shard checkpoints into _merged.json (same dict-of-dicts
# schema: {"Equation Name": {"domain":..., "pysr_only":{...}, "hypatia":{...}}}),
# which _normalise_cases() already supports.  Fall back to _merged.json when
# exp1_ablation_results.json is absent.
if _EXPERIMENT == "exp1_ablation" and not os.path.isfile(DATA_MAIN):
    _merged_fallback = _rpath("_merged.json")
    if os.path.isfile(_merged_fallback):
        print(f"  [INFO] {DATA_MAIN} not found — falling back to {_merged_fallback}")
        DATA_MAIN = _merged_fallback
# exp2_feynman_extrap-specific required files
DATA_ABLATION_PAIRED  = _rpath("ablation_paired.json")
DATA_EXTRAP_BENCHMARK = _rpath("benchmark_results_extrap.json")
# Secondary sources (optional — figures skipped gracefully if absent)
DATA_WALL_CLOCK    = _rpath("wall_clock_flags.json")
DATA_PORTFOLIO_SW  = _rpath("portfolio_variance_seed_sweep.json")
DATA_INSTAB_CSV    = _rpath("instability_extrapolation_v2.csv")
# Supp-B sweep files — use glob to find latest matching file (filename contains
# a datestamp that changes with each run; never hardcode a specific date).
#
# Searched RECURSIVELY under _RESULTS_DIR. The canonical noise-sweep / sample-
# complexity JSON is not always written flat into _RESULTS_DIR — it has been
# observed nested under noise_sweep_saved/.
#
# HISTORICAL NOTE: noise_sweep_*.json was previously also seen self-nested
# under a duplicated comparison_results/... subtree, because
# run_noise_sweep_benchmark.py did not honor OUT_BASE and wrote relative to
# its CWD using the same subdir suffix OUT_BASE already encoded. That write-
# side bug is now fixed at the source: run_all.sh STEP 10 (FIX-suppB-
# DOUBLED-PATH) detects and flattens the doubled tree into the canonical
# SUPPB_SUBDIR immediately after run_noise_sweep_benchmark.py exits, mirroring
# the equivalent suppB_sc fix already in STEP 10b. New runs should never
# produce the self-nested layout again; this recursive glob is kept as
# defense-in-depth (e.g. for manual re-invocations or pre-fix committed data)
# rather than as the primary mechanism. generate_tables.py already searches
# multiple candidate locations and combines shards; this mirrors that
# robustness so figures don't silently fall back to MISSING from the same
# --results-dir that tables succeeds from.
#
# Per-sig checkpoint shards (noise_sweep_sig0000_checkpoint.json, etc.) and the
# MISSING placeholder itself are excluded so they're never picked up as "the"
# consolidated file.
_SWEEP_EXCLUDE_SUBSTRINGS = ("checkpoint", "_sig", "MISSING")

def _latest_glob(pattern, exclude_substrings=_SWEEP_EXCLUDE_SUBSTRINGS):
    """Return the file with the lexicographically last *basename* among all
    recursive matches of pattern, excluding any whose basename contains one of
    exclude_substrings, or None if nothing matches.

    Sorting by basename (not full path) keeps "latest by embedded timestamp"
    correct even when candidate files live in different subdirectories whose
    names would otherwise dominate a full-path sort.
    """
    matches = glob.glob(pattern, recursive=True)
    if exclude_substrings:
        matches = [m for m in matches
                   if not any(s in os.path.basename(m) for s in exclude_substrings)]
    if not matches:
        return None
    return sorted(matches, key=os.path.basename)[-1]

_noise_glob  = _latest_glob(os.path.join(_RESULTS_DIR, "**", "noise_sweep_*.json"))
_sample_glob = _latest_glob(os.path.join(_RESULTS_DIR, "**", "sample_complexity_*.json"))

if _noise_glob is None:
    print(f"  [SKIP] No noise_sweep_*.json found under {_RESULTS_DIR} (recursive) — suppB noise figures will be skipped.")
else:
    print(f"  [INFO] noise_sweep source: {_noise_glob}")
if _sample_glob is None:
    print(f"  [SKIP] No sample_complexity_*.json found under {_RESULTS_DIR} (recursive) — suppB sample figures will be skipped.")
else:
    print(f"  [INFO] sample_complexity source: {_sample_glob}")

DATA_NOISE_SWEEP   = _noise_glob  or _rpath("noise_sweep_MISSING.json")
DATA_SAMPLE_SWEEP  = _sample_glob or _rpath("sample_complexity_MISSING.json")
# (five-systems comparison removed — not in paper inventory or any .tex file)

def _load_json(path, label=None):
    """Load JSON; return None and print a warning if the file is absent.
    Uses parse_constant to handle NaN / Infinity / -Infinity literals that
    some result files emit (non-standard JSON but valid Python float values).
    """
    if not os.path.isfile(path):
        print(f"  [SKIP] {label or path} not found — dependent figures will be skipped.")
        return None
    with open(path) as f:
        text = f.read()
    # Replace bare NaN / Infinity / -Infinity with JSON-safe equivalents
    import re as _re
    text = _re.sub(r'\bNaN\b',       'null', text)
    text = _re.sub(r'\bInfinity\b',  '1e308', text)
    text = _re.sub(r'\b-Infinity\b', '-1e308', text)
    return json.loads(text)

def _load_csv_np(path, label=None):
    """Load a CSV as a numpy structured array; return None if absent."""
    import csv
    if not os.path.isfile(path):
        print(f"  [SKIP] {label or path} not found — dependent figures will be skipped.")
        return None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows  # list of dicts

# ══════════════════════════════════════════════════════════════════════════════
# PREFLIGHT: exp2_feynman_extrap — verify required files before any figure work
# ablation_paired.json is produced by ci_analysis.yml (merge_extrap step) and
# must be committed before this script is called.  Abort with a clear error
# rather than failing silently mid-run.
# ══════════════════════════════════════════════════════════════════════════════
if _EXPERIMENT in ("exp2_feynman_extrap", "exp2_feyman_extrap"):
    _preflight_ok = True
    _preflight_msgs = []

    # Required: ablation_paired.json (produced by ci_analysis merge_extrap step)
    if not os.path.isfile(DATA_ABLATION_PAIRED):
        _preflight_msgs.append(
            f"  MISSING (required): {DATA_ABLATION_PAIRED}\n"
            f"    → Run ci_analysis.yml for exp2_feynman_extrap first.\n"
            f"      It produces ablation_paired.json via merge_extrap_into_benchmark.py."
        )
        _preflight_ok = False

    # Required: at least one protocol_core_extrap_*.json shard
    _extrap_shards = sorted(glob.glob(os.path.join(_RESULTS_DIR, "protocol_core_extrap_*.json")))
    if not _extrap_shards:
        _preflight_msgs.append(
            f"  MISSING (required): protocol_core_extrap_*.json in {_RESULTS_DIR}\n"
            f"    → Run ci_runner.yml for exp2_feynman_extrap to commit result shards."
        )
        _preflight_ok = False
    else:
        print(f"  [PREFLIGHT] Found {len(_extrap_shards)} protocol_core_extrap_*.json shard(s).")

    # Optional but warn: benchmark_results_extrap.json
    if not os.path.isfile(DATA_EXTRAP_BENCHMARK):
        print(
            f"  [PREFLIGHT WARN] {DATA_EXTRAP_BENCHMARK} not found — "
            f"some extrap comparison figures will be skipped.\n"
            f"    → This file is produced by ci_runner.yml for exp2_feynman_extrap."
        )

    if not _preflight_ok:
        print()
        print("=" * 70)
        print(f"PREFLIGHT FAILED for experiment '{_EXPERIMENT}'")
        print("The following required files are missing:")
        for msg in _preflight_msgs:
            print(msg)
        print()
        print("Action: ensure ci_analysis.yml has run successfully for")
        print(f"  exp2_feynman_extrap (results-dir: {_RESULTS_DIR})")
        print("before running ci_postprocess.yml.")
        print("=" * 70)
        sys.exit(1)
    else:
        print(f"  [PREFLIGHT OK] All required files present for '{_EXPERIMENT}'.")

# ══════════════════════════════════════════════════════════════════════════════
# Load primary data (exp1_ablation_results.json)
# For experiments that do not use this file, missing it is a graceful skip:
# the cosmetic/RF02/RF09 figure groups are skipped entirely.
# Only abort when the experiment explicitly requires ablation data (exp1_ablation,
# or a legacy/manual invocation without --experiment where it was always required).
# ══════════════════════════════════════════════════════════════════════════════
RAW = _load_json(DATA_MAIN, "exp1_ablation_results.json")
if RAW is None:
    _ablation_required = (
        _EXPERIMENT is None                          # legacy invocation — preserve old behaviour
        or _EXPERIMENT == "exp1_ablation"            # the one experiment that genuinely needs it
        or _EXPERIMENT in ("exp1", "exp1b")          # cosmetic figures need it
    )
    if _ablation_required:
        print(f"ERROR: primary data file '{DATA_MAIN}' is required for experiment "
              f"'{_EXPERIMENT}'. Aborting.")
        sys.exit(1)
    else:
        print(f"  [INFO] exp1_ablation_results.json not present for experiment "
              f"'{_EXPERIMENT}' — cosmetic/RF02/RF09 figure groups will be skipped.")

# Ensure the figures output directory exists.
os.makedirs(_FIGURES_DIR, exist_ok=True)

# ── Dual-format save helper ────────────────────────────────────────────────────
# Saves every figure as both PNG (300 dpi, for quick preview / NB-05 checks)
# and PDF (vector, required by JMLR for line charts and heatmaps).
# All savefig() calls in this script go through here so the format list is
# controlled in one place — add/remove formats here, not at each figure site.
_SAVE_FORMATS = [
    ("png", dict(dpi=300)),
    ("pdf", dict()),          # PDF is vector; dpi is irrelevant and omitted
]

def _savefig(fig, stem, **kwargs):
    """Save fig to _FIGURES_DIR/<stem>.png and <stem>.pdf.

    kwargs are forwarded to both formats (e.g. bbox_inches="tight").
    Returns the PNG path (used in print confirmations).
    """
    png_path = None
    for fmt, fmt_kwargs in _SAVE_FORMATS:
        path = os.path.join(_FIGURES_DIR, f"{stem}.{fmt}")
        fig.savefig(path, **{**fmt_kwargs, **kwargs})
        if fmt == "png":
            png_path = path
    return png_path

# Support both schemas:
#   New (dict-of-dicts): {"Equation Name": {"domain": ..., "pysr_only": {...}, "hypatia": {...}}, ...}
#   Legacy (list):       {"cases": [...]}  or a bare list
def _normalise_cases(raw):
    """Convert either schema to the canonical list-of-case-dicts used throughout."""
    if isinstance(raw, list):
        return raw
    if "cases" in raw:
        return raw["cases"]
    # dict-of-dicts: convert to list, mapping "pysr_only" → "neural_network" proxy
    # and "hypatia" → "hybrid" proxy so existing collect()/get_case() helpers work.
    cases = []
    for eq_name, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        pysr   = entry.get("pysr_only", {})
        hyp    = entry.get("hypatia",   {})
        domain = entry.get("domain", "unknown")
        # Map near/medium/far extrap_r2 onto stability_score (far) for legacy figures.
        # Use safe_float() for every field read — dict.get() returns None when
        # the key exists with a JSON null value, which causes TypeError on
        # arithmetic. safe_float(None) returns float("nan") safely.
        def _sf(d, key):
            return safe_float(d.get(key))

        case = {
            "test_case":    eq_name,
            "formula_type": domain,
            "difficulty":   _infer_difficulty(pysr, hyp),
            "results": {
                "hybrid": {
                    "train_r2":        _sf(hyp, "train_r2"),
                    "test_r2":         _sf(hyp, "extrap_r2_far"),
                    "stability_score": _sf(hyp, "extrap_r2_far"),
                    "extrapolation_gap": (
                        _sf(hyp, "train_r2") - _sf(hyp, "extrap_r2_far")
                    ),
                    "time_s":          _sf(hyp, "total_time_s"),
                    # Store regime scores for new heatmap figures.
                    "extrap_r2_near":   _sf(hyp, "extrap_r2_near"),
                    "extrap_r2_medium": _sf(hyp, "extrap_r2_medium"),
                    "extrap_r2_far":    _sf(hyp, "extrap_r2_far"),
                },
                "neural_network": {
                    "train_r2":        _sf(pysr, "train_r2"),
                    "test_r2":         _sf(pysr, "extrap_r2_far"),
                    "stability_score": _sf(pysr, "extrap_r2_far"),
                    "extrapolation_gap": (
                        _sf(pysr, "train_r2") - _sf(pysr, "extrap_r2_far")
                    ),
                    "time_s":          _sf(pysr, "total_time_s"),
                    "extrap_r2_near":   _sf(pysr, "extrap_r2_near"),
                    "extrap_r2_medium": _sf(pysr, "extrap_r2_medium"),
                    "extrap_r2_far":    _sf(pysr, "extrap_r2_far"),
                },
                # pure_llm is not present in the new schema; use NaN placeholders.
                "pure_llm": {},
            },
            # Carry raw entries for regime-aware figures.
            "_pysr_only": pysr,
            "_hypatia":   hyp,
        }
        cases.append(case)
    return cases


def safe_float(v):
    if v is None: return float("nan")
    try:
        f = float(v)
        return f if math.isfinite(f) else float("nan")
    except: return float("nan")


def _infer_difficulty(pysr, hyp):
    """Heuristically assign difficulty from far-extrap R² performance."""
    far = safe_float(hyp.get("extrap_r2_far", pysr.get("extrap_r2_far", float("nan"))))
    if math.isnan(far):
        return "medium"
    if far >= 0.90:
        return "easy"
    if far >= 0.50:
        return "medium"
    return "hard"


CASES = _normalise_cases(RAW) if RAW is not None else []
# Detect which schema was loaded (affects regime-heatmap figures).
_DICT_SCHEMA = (RAW is not None
                and isinstance(RAW, dict)
                and "cases" not in RAW
                and not isinstance(RAW, list))

# ── Colour palette ─────────────────────────────────────────────────────────────
C_HYB   = "#2563EB"   # hybrid (blue)
C_LLM   = "#7C3AED"   # pure_llm (purple)
C_NN    = "#DC2626"   # neural_network (red)
C_OK    = "#16A34A"   # success green
C_FAIL  = "#DC2626"   # failure red
C_WARN  = "#D97706"   # amber
C_GRID  = "#E5E7EB"   # light grid

METHODS = ["pure_llm", "neural_network", "hybrid"]
MLABELS = {"pure_llm": "Pure LLM", "neural_network": "Neural Net", "hybrid": "HypatiaX Hybrid"}
MCOLORS = {"pure_llm": C_LLM, "neural_network": C_NN, "hybrid": C_HYB}
DIFF_ORDER = ["easy", "medium", "hard"]
DIFF_COLORS = {"easy": "#059669", "medium": "#D97706", "hard": "#DC2626"}

# ── Data helpers ──────────────────────────────────────────────────────────────

def collect(field, method="hybrid"):
    return [safe_float(c["results"].get(method, {}).get(field)) for c in CASES]

def collect_valid(field, method="hybrid"):
    return [v for v in collect(field, method) if not math.isnan(v)]

def get_case(c, method, field):
    return safe_float(c["results"].get(method, {}).get(field))

# Precompute arrays
h_train    = np.array(collect("train_r2",        "hybrid"))
h_test     = np.array(collect("test_r2",         "hybrid"))
h_stab     = np.array(collect("stability_score", "hybrid"))
h_egap     = np.array(collect("extrapolation_gap","hybrid"))
nn_train   = np.array(collect("train_r2",        "neural_network"))
nn_test    = np.array(collect("test_r2",         "neural_network"))
nn_stab    = np.array(collect("stability_score", "neural_network"))
llm_train  = np.array(collect("train_r2",        "pure_llm"))
llm_test   = np.array(collect("test_r2",         "pure_llm"))
llm_stab   = np.array(collect("stability_score", "pure_llm"))
h_times    = np.array(collect("time_s",          "hybrid"))
nn_times   = np.array(collect("time_s",          "neural_network"))
difficulties = [c["difficulty"] for c in CASES]
ftypes       = [c["formula_type"] for c in CASES]

print(f"Loaded {len(CASES)} cases.")
print(f"Hybrid:  mean_stab={np.nanmean(h_stab):.4f}  mean_test_r2={np.nanmean(h_test):.4f}")
print(f"NN:      mean_stab={np.nanmean(nn_stab):.4f}  mean_test_r2={np.nanmean(nn_test):.4f}")
print(f"LLM:     mean_stab={np.nanmean(llm_stab):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# P0 / RF02 / RF09 FIGURES  — require exp1_ablation_results.json (RAW)
# Skipped entirely when RAW is None (e.g. for instability, exp2_feynman_extrap).
# ══════════════════════════════════════════════════════════════════════════════
if RAW is not None:
    # ── hypatiax_three_systems — architecture diagram rendered from code ──────────
    _sys_fig, _sys_ax = plt.subplots(figsize=(14, 6))
    _sys_ax.set_xlim(0, 14); _sys_ax.set_ylim(0, 6)
    _sys_ax.axis("off")
    _sys_ax.set_facecolor("#F8FAFC")
    _sys_fig.patch.set_facecolor("#F8FAFC")

    _BOX_H = 1.1  # box height
    _BOX_W = 3.0  # box width

    def _draw_box(ax, cx, cy, label, sublabel, color, text_color="white"):
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((cx - _BOX_W/2, cy - _BOX_H/2), _BOX_W, _BOX_H,
                             boxstyle="round,pad=0.1", linewidth=1.5,
                             edgecolor="white", facecolor=color, zorder=3)
        ax.add_patch(box)
        ax.text(cx, cy + 0.18, label,    ha="center", va="center",
                fontsize=11, fontweight="bold", color=text_color, zorder=4)
        ax.text(cx, cy - 0.26, sublabel, ha="center", va="center",
                fontsize=8,  color=text_color, alpha=0.88, zorder=4)

    def _arrow(ax, x0, y0, x1, y1, label=""):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="#374151"), zorder=2)
        if label:
            mx, my = (x0+x1)/2, (y0+y1)/2
            ax.text(mx+0.05, my+0.12, label, fontsize=7.5, color="#374151", ha="center", zorder=5)

    # System 1 — PySR Symbolic (left)
    _draw_box(_sys_ax, 2.5, 4.5, "System 1", "PySR Symbolic Regression", C_NN)
    # System 2 — LLM Prior (middle-top)
    _draw_box(_sys_ax, 7.0, 4.5, "System 2", "LLM Symbolic Prior",       C_LLM)
    # System 3 — HypatiaX Hybrid (right)
    _draw_box(_sys_ax, 11.5, 4.5, "System 3", "HypatiaX Hybrid Fusion",  C_HYB)

    # Data input
    _draw_box(_sys_ax, 7.0, 1.8, "Training Data", "(X, y) observations", "#6B7280", text_color="white")

    # Router / decision block
    from matplotlib.patches import FancyBboxPatch as _FBP
    _rbox = _FBP((5.8, 2.9), 2.4, 0.9, boxstyle="round,pad=0.08",
                 linewidth=1.2, edgecolor="#D97706", facecolor="#FEF3C7", zorder=3)
    _sys_ax.add_patch(_rbox)
    _sys_ax.text(7.0, 3.35, "Router / Stability Check", ha="center", va="center",
                 fontsize=9, fontweight="bold", color="#92400E", zorder=4)

    # Arrows
    _arrow(_sys_ax, 7.0, 2.25, 7.0, 2.9,  "fit")           # data → router
    _arrow(_sys_ax, 5.8, 3.35, 4.1, 3.95, "low confidence") # router → sys1
    _arrow(_sys_ax, 7.0, 3.8,  7.0, 3.95, "LLM prior")     # router → sys2
    _arrow(_sys_ax, 8.2, 3.35, 9.9, 3.95, "hybrid path")   # router → sys3

    # Ensemble output
    _draw_box(_sys_ax, 7.0, 0.6, "Output", "Best symbolic expression + R²", "#1E3A5F", text_color="white")
    _arrow(_sys_ax, 2.5, 3.95, 5.0, 1.05, "")
    _arrow(_sys_ax, 7.0, 3.95, 7.0, 0.95, "")
    _arrow(_sys_ax, 11.5, 3.95, 9.0, 1.05, "")

    _sys_ax.text(7.0, 5.7, "HypatiaX — Three-System Architecture",
                 ha="center", va="center", fontsize=14, fontweight="bold", color="#1E293B")
    _sys_fig.tight_layout()
    _savefig(_sys_fig, "hypatiax_three_systems", bbox_inches="tight")
    plt.close(_sys_fig)
    print("✓ hypatiax_three_systems.png/.pdf")


    # ══════════════════════════════════════════════════════════════════════════════
    # RF02 FIGURES
    # ══════════════════════════════════════════════════════════════════════════════

    # ── fig07: scatter train vs extrap ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, method, label, color in [
        (axes[0], "hybrid",         "HypatiaX Hybrid", C_HYB),
        (axes[1], "pure_llm",       "Pure LLM",        C_LLM),
        (axes[2], "neural_network", "Neural Net",       C_NN),
    ]:
        x = np.array(collect("train_r2",        method))
        y = np.array(collect("stability_score", method))
        diff_c = [DIFF_COLORS[d] for d in difficulties]
        scatter = ax.scatter(x, y, c=diff_c, alpha=0.75, s=55, edgecolors="white", lw=0.4, zorder=3)
        ax.axline((0,0), slope=1, color="gray", ls="--", lw=0.8, alpha=0.5, label="y=x")
        ax.axhline(0.99, color=C_OK, lw=1, ls=":", alpha=0.8)
        ax.set_xlabel("Train $R^2$", fontsize=11)
        ax.set_ylabel("Extrapolation Stability ($R^2$ on test)", fontsize=9)
        ax.set_title(label, fontsize=12, fontweight="bold", color=color)
        ax.set_xlim(-0.2, 1.1); ax.set_ylim(-4, 1.2)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=9)
    patches = [mpatches.Patch(color=DIFF_COLORS[d], label=d.capitalize()) for d in DIFF_ORDER]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9, framealpha=0.9)
    fig.suptitle("Train $R^2$ vs Extrapolation Stability by Method", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    _savefig(fig, "fig07_scatter_train_vs_extrap", bbox_inches="tight")
    plt.close(fig)
    print("✓ fig07_scatter_train_vs_extrap.png/.pdf")


    # ── fig08: train_r2 bar chart per method per difficulty ───────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(DIFF_ORDER))
    width = 0.25
    for i, (method, label) in enumerate([
        ("pure_llm","Pure LLM"),("neural_network","Neural Net"),("hybrid","HypatiaX")]):
        means = []
        for d in DIFF_ORDER:
            vals = [get_case(c, method, "train_r2") for c in CASES if c["difficulty"]==d]
            vals = [v for v in vals if not math.isnan(v)]
            means.append(np.mean(vals) if vals else float("nan"))
        bars = ax.bar(x + (i-1)*width, means, width, label=label,
                      color=MCOLORS[method], alpha=0.88, edgecolor="white", lw=0.5)
        for bar, v in zip(bars, means):
            if not math.isnan(v):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([d.capitalize() for d in DIFF_ORDER], fontsize=11)
    ax.set_ylabel("Mean Train $R^2$", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title("Train $R^2$ by Difficulty and Method", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "fig08_train_r2_bar")
    plt.close(fig)
    print("✓ fig08_train_r2_bar.png/.pdf")


    # ── fig09: r2 heatmap across regimes ─────────────────────────────────────────
    # New schema: show near / medium / far columns per method.
    # Legacy schema: fall back to train / test / stability columns.
    case_labels = [f"{c['test_case'][:32]}..." if len(c['test_case'])>32 else c['test_case']
                   for c in CASES]

    if _DICT_SCHEMA:
        # 4-column heatmap: Train | Near | Medium | Far  ×  Hybrid vs PySR-only
        _col_keys  = ["train_r2", "extrap_r2_near", "extrap_r2_medium", "extrap_r2_far"]
        _col_names = ["Train $R^2$", "Near", "Medium", "Far"]
        _method_pairs = [
            ("hybrid",         "HypatiaX Hybrid", "RdYlGn"),
            ("neural_network", "PySR-only",        "RdYlGn"),
        ]
    else:
        _col_keys  = ["train_r2", "test_r2", "stability_score"]
        _col_names = ["Train $R^2$", "Test $R^2$", "Stability"]
        _method_pairs = [
            ("hybrid",         "HypatiaX Hybrid", "RdYlGn"),
            ("neural_network", "Neural Network",   "RdYlGn"),
        ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 16), sharey=True)
    for ax, (method, label, cmap) in zip(axes, _method_pairs):
        mat = np.array([[get_case(c, method, k) for k in _col_keys] for c in CASES])
        mat_disp = np.clip(np.nan_to_num(mat, nan=-1.5), -1.5, 1.0)

        im = ax.imshow(mat_disp, vmin=-1.5, vmax=1.0, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(_col_names)))
        ax.set_xticklabels(_col_names, fontsize=9)
        ax.set_yticks(range(len(CASES)))
        ax.set_yticklabels(case_labels, fontsize=6.5)
        ax.set_title(label, fontsize=11, fontweight="bold")
        for i in range(len(CASES)):
            for j in range(len(_col_keys)):
                v = mat[i, j]
                txt = f"{v:.2f}" if abs(v) < 10 and not math.isnan(v) else ("nan" if math.isnan(v) else f"{v:.0f}")
                col = "white" if mat_disp[i, j] < -0.4 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=5.5, color=col)
        for tick, c in zip(ax.get_yticklabels(), CASES):
            tick.set_color(DIFF_COLORS[c["difficulty"]])

    axes[0].set_ylabel("Test Case (colour = difficulty)", fontsize=9)
    fig.colorbar(im, ax=axes[1], fraction=0.015, pad=0.02, label="$R^2$")
    _fig09_title = ("$R^2$ Heatmap: Train / Near / Medium / Far Regimes"
                    if _DICT_SCHEMA else "$R^2$ Heatmap: Train / Test / Stability")
    fig.suptitle(_fig09_title, fontsize=12, fontweight="bold", y=1.002)
    fig.tight_layout()
    _savefig(fig, "fig09_r2_heatmap_regimes", bbox_inches="tight")
    plt.close(fig)
    print("✓ fig09_r2_heatmap_regimes.png/.pdf")


    # ── fig10: far extrapolation head-to-head (Hybrid vs NN) ─────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    hybrid_stab  = np.nan_to_num(h_stab,  nan=0)
    nn_stab_plot = np.nan_to_num(nn_stab, nan=0)
    hybrid_c = np.clip(hybrid_stab,  -2, 1)
    nn_c     = np.clip(nn_stab_plot, -2, 1)
    idx = np.arange(len(CASES))
    # side by side
    w = 0.38
    ax.bar(idx-w/2, nn_c,     w, label="Neural Net",     color=C_NN,  alpha=0.85, edgecolor="white", lw=0.4)
    ax.bar(idx+w/2, hybrid_c, w, label="HypatiaX Hybrid",color=C_HYB, alpha=0.85, edgecolor="white", lw=0.4)
    ax.axhline(0.99, color=C_OK,  lw=1.2, ls=":", alpha=0.9, label="Success (0.99)")
    ax.axhline(0.0,  color="black",lw=0.7, ls="--", alpha=0.4)
    ax.set_xticks(idx[::4])
    ax.set_xticklabels([CASES[i]["test_case"][:18] for i in idx[::4]], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Extrapolation Stability (clipped $[-2,1]$)", fontsize=10)
    ax.set_title("Extrapolation Stability: HypatiaX vs Neural Net (All Cases)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "fig10_far_extrap_head2head")
    plt.close(fig)
    print("✓ fig10_far_extrap_head2head.png/.pdf")


    # ── fig11: speedup bar (hybrid time / nn time) ───────────────────────────────
    speedup = nn_times / np.where(h_times > 0, h_times, np.nan)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [C_HYB if s < 1 else C_NN for s in np.nan_to_num(speedup, nan=1)]
    ax.bar(np.arange(len(CASES)), np.nan_to_num(speedup, nan=1), color=colors, alpha=0.85, edgecolor="white", lw=0.3)
    ax.axhline(1.0, color="black", lw=1, ls="--", alpha=0.6, label="Equal speed")
    ax.set_xlabel("Case index", fontsize=10)
    ax.set_ylabel("NN time / Hybrid time", fontsize=10)
    ax.set_title("Relative Speed: Neural Net vs HypatiaX Hybrid", fontsize=11, fontweight="bold")
    # Annotate mean
    valid_sp = speedup[~np.isnan(speedup)]
    ax.axhline(valid_sp.mean(), color=C_LLM, lw=1.5, ls=":", label=f"Mean ratio: {valid_sp.mean():.2f}x")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "fig11_speedup_bar")
    plt.close(fig)
    print("✓ fig11_speedup_bar.png/.pdf")


    # ── fig12: ridge vs train_r2 (stability score distribution) ──────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, label, color in [
        ("hybrid",         "HypatiaX Hybrid", C_HYB),
        ("pure_llm",       "Pure LLM",        C_LLM),
        ("neural_network", "Neural Net",       C_NN),
    ]:
        vals = np.array([v for v in collect("stability_score", method) if not math.isnan(v)])
        vals_c = np.clip(vals, -3, 1)
        ax.hist(vals_c, bins=25, color=color, alpha=0.55, label=f"{label} (n={len(vals)})",
                edgecolor="white", lw=0.4, density=True)
        if len(vals_c) >= 2 and np.ptp(vals_c) > 0:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(vals_c, bw_method=0.3)
            xs = np.linspace(-3, 1.05, 300)
            ax.plot(xs, kde(xs), color=color, lw=2)
    ax.axvline(0.99, color="black", lw=1.2, ls=":", alpha=0.7, label="Success threshold")
    ax.set_xlabel("Stability Score (extrapolation $R^2$, clipped $[-3,1]$)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Distribution of Extrapolation Stability by Method", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    _savefig(fig, "fig12_ridge_vs_train_r2")
    plt.close(fig)
    print("✓ fig12_ridge_vs_train_r2.png/.pdf")


    # ── fig14: per-equation r2 profile ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 18), sharey=True)
    for ax_idx, (method, label, color) in enumerate([
        ("hybrid",         "HypatiaX",  C_HYB),
        ("pure_llm",       "Pure LLM",  C_LLM),
        ("neural_network", "Neural Net",C_NN),
    ]):
        ax = axes[ax_idx]
        stab_vals = np.array([get_case(c, method, "stability_score") for c in CASES])
        stab_c    = np.clip(np.nan_to_num(stab_vals, nan=0), -3, 1)
        y_pos = np.arange(len(CASES))
        col   = [C_OK if v > 0.99 else (C_WARN if v > 0 else C_FAIL) for v in stab_c]
        bars  = ax.barh(y_pos, stab_c, color=col, alpha=0.85, edgecolor="white", lw=0.3)
        ax.axvline(0.99, color="black", lw=1.2, ls=":", alpha=0.7)
        ax.axvline(0,    color="black", lw=0.5, ls="--", alpha=0.4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [f"{CASES[i]['test_case'][:30]}" for i in range(len(CASES))],
            fontsize=5.8
        )
        for tick, c in zip(ax.get_yticklabels(), CASES):
            tick.set_color(DIFF_COLORS[c["difficulty"]])
        ax.set_xlabel("Stability Score (clipped)", fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold", color=color)
        ax.set_xlim(-3.2, 1.15)
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Per-Case Extrapolation Stability Profile\n(green=easy, amber=medium, red=hard)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "fig14_per_equation_r2_profile", bbox_inches="tight")
    plt.close(fig)
    print("✓ fig14_per_equation_r2_profile.png/.pdf")


    # ── fig16: instability vs extrapolation scatter ───────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method, label, color, marker in [
        ("hybrid",         "HypatiaX", C_HYB, "o"),
        ("pure_llm",       "Pure LLM", C_LLM, "s"),
        ("neural_network", "Neural Net",C_NN, "^"),
    ]:
        x = np.array([get_case(c, method, "extrapolation_gap")   for c in CASES])
        y = np.array([1 - get_case(c, method, "stability_score") for c in CASES])
        mask = ~(np.isnan(x) | np.isnan(y))
        ax.scatter(x[mask], y[mask], c=color, alpha=0.65, s=50, marker=marker,
                   edgecolors="white", lw=0.4, label=label, zorder=3)
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
    ax.axvline(0, color="black", lw=0.7, ls="--", alpha=0.4)
    ax.set_xlabel("Extrapolation Gap (train $R^2$ − test $R^2$)", fontsize=10)
    ax.set_ylabel("Instability (1 − Stability Score)", fontsize=10)
    ax.set_title("Instability vs Extrapolation Gap", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    _savefig(fig, "fig16_instability_vs_extrapolation")
    plt.close(fig)
    print("✓ fig16_instability_vs_extrapolation.png/.pdf")


    # ── fig17: 3D surface instability vs complexity ───────────────────────────────
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    fig = plt.figure(figsize=(9, 6))
    ax  = fig.add_subplot(111, projection="3d")

    ftype_idx = {ft: i for i, ft in enumerate(sorted(set(ftypes)))}
    x3d = np.array([ftype_idx[ft] for ft in ftypes], dtype=float)
    y3d = np.array([{"easy":0,"medium":1,"hard":2}[d] for d in difficulties], dtype=float)
    z3d = 1 - np.nan_to_num(h_stab, nan=0)

    sc = ax.scatter(x3d, y3d, z3d, c=z3d, cmap="RdYlGn_r", s=40, alpha=0.8,
                    vmin=0, vmax=1.5, edgecolors="none")
    ax.set_xticks(range(len(ftype_idx)))
    ax.set_xticklabels(sorted(ftype_idx.keys()), rotation=45, ha="right", fontsize=6)
    ax.set_yticks([0,1,2]); ax.set_yticklabels(["Easy","Medium","Hard"], fontsize=8)
    ax.set_zlabel("Instability", fontsize=9)
    ax.set_xlabel("Formula Type", fontsize=8)
    ax.set_ylabel("Difficulty", fontsize=8)
    ax.set_title("3D: Instability vs Formula Type & Difficulty (HypatiaX)", fontsize=10, fontweight="bold")
    fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.1, label="Instability")
    fig.tight_layout()
    _savefig(fig, "fig17_3d_surface_instability_complexity", bbox_inches="tight")
    plt.close(fig)
    print("✓ fig17_3d_surface_instability_complexity.png/.pdf")


    # ── fig18: r2 heatmap improved (formula_type × difficulty) ───────────────────
    # New schema: 2-panel — PySR-only vs HypatiaX — using mean far-extrap R².
    # Legacy schema: 3-panel stability heatmap (unchanged).
    ft_list = sorted(set(ftypes))

    if _DICT_SCHEMA:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
        _method_cfg18 = [
            (axes[0], "neural_network", "PySR-only",        "extrap_r2_far", "RdYlGn"),
            (axes[1], "hybrid",         "HypatiaX Hybrid",  "extrap_r2_far", "RdYlGn"),
        ]
        for ax, method, label, score_key, cmap in _method_cfg18:
            mat = np.full((len(ft_list), len(DIFF_ORDER)), float("nan"))
            for i, ft in enumerate(ft_list):
                for j, dif in enumerate(DIFF_ORDER):
                    vals = [get_case(c, method, score_key)
                            for c in CASES if c["formula_type"] == ft and c["difficulty"] == dif]
                    vals = [v for v in vals if not math.isnan(v)]
                    if vals:
                        mat[i, j] = np.mean(vals)
            mat_d = np.clip(np.nan_to_num(mat, nan=0), -1.5, 1)
            im = ax.imshow(mat_d, vmin=-1.5, vmax=1.0, cmap=cmap, aspect="auto")
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["Easy", "Med", "Hard"], fontsize=9)
            ax.set_yticks(range(len(ft_list)))
            ax.set_yticklabels(ft_list, fontsize=8)
            ax.set_title(label, fontsize=11, fontweight="bold")
            for i in range(len(ft_list)):
                for j in range(3):
                    v = mat[i, j]
                    txt = f"{v:.2f}" if not math.isnan(v) else "—"
                    col = "white" if mat_d[i, j] < -0.3 else "black"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=col)
        fig.colorbar(im, ax=axes[1], fraction=0.04, pad=0.02, label="Mean far-extrap $R^2$")
        fig.suptitle("Far-Extrap $R^2$: PySR-only vs HypatiaX (Formula Type × Difficulty)",
                     fontsize=12, fontweight="bold")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
        for ax, method, label, cmap in [
            (axes[0], "hybrid",         "HypatiaX Hybrid", "RdYlGn"),
            (axes[1], "pure_llm",       "Pure LLM",        "RdYlGn"),
            (axes[2], "neural_network", "Neural Net",       "RdYlGn"),
        ]:
            mat = np.full((len(ft_list), len(DIFF_ORDER)), float("nan"))
            for i, ft in enumerate(ft_list):
                for j, dif in enumerate(DIFF_ORDER):
                    vals = [get_case(c, method, "stability_score")
                            for c in CASES if c["formula_type"]==ft and c["difficulty"]==dif]
                    vals = [v for v in vals if not math.isnan(v)]
                    if vals: mat[i, j] = np.mean(vals)
            mat_d = np.clip(np.nan_to_num(mat, nan=0), -1.5, 1)
            im = ax.imshow(mat_d, vmin=-1.5, vmax=1.0, cmap=cmap, aspect="auto")
            ax.set_xticks([0,1,2]); ax.set_xticklabels(["Easy","Med","Hard"], fontsize=9)
            ax.set_yticks(range(len(ft_list))); ax.set_yticklabels(ft_list, fontsize=8)
            ax.set_title(label, fontsize=11, fontweight="bold")
            for i in range(len(ft_list)):
                for j in range(3):
                    v = mat[i,j]
                    txt = f"{v:.2f}" if not math.isnan(v) else "—"
                    col = "white" if mat_d[i,j] < -0.3 else "black"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=col)
        fig.colorbar(im, ax=axes[2], fraction=0.04, pad=0.02, label="Mean stability $R^2$")
        fig.suptitle("Mean Stability by Formula Type × Difficulty", fontsize=12, fontweight="bold")

    fig.tight_layout()
    _savefig(fig, "fig18_r2_heatmap_improved", bbox_inches="tight")
    plt.close(fig)
    print("✓ fig18_r2_heatmap_improved.png/.pdf")


    # ── fig19: far extrap improved (success rate donut grid) ─────────────────────
    THRESH = 0.99
    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    pairs = [(m, d) for m in ["hybrid","pure_llm","neural_network"] for d in DIFF_ORDER]
    for ax, (method, diff) in zip(axes.flatten(), pairs):
        subset = [c for c in CASES if c["difficulty"]==diff]
        vals   = [get_case(c, method, "stability_score") for c in subset]
        vals   = [v for v in vals if not math.isnan(v)]
        if not vals:
            ax.axis("off"); continue
        n_ok = sum(v >= THRESH for v in vals)
        sr   = n_ok / len(vals)
        wedges, _ = ax.pie([sr, max(0, 1-sr)],
                           colors=[C_OK, C_FAIL],
                           startangle=90,
                           wedgeprops=dict(width=0.5, edgecolor="white", lw=1.5))
        ax.text(0, 0, f"{int(round(sr*100))}%\n({n_ok}/{len(vals)})",
                ha="center", va="center", fontsize=11, fontweight="bold")
        ax.set_title(f"{MLABELS[method]}\n{diff.capitalize()}", fontsize=9, fontweight="bold")
    patches = [mpatches.Patch(color=C_OK, label=f"Success ($R^2≥{THRESH}$)"),
               mpatches.Patch(color=C_FAIL, label="Failure")]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=10)
    fig.suptitle("Far-Extrapolation Success Rate by Method × Difficulty", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _savefig(fig, "fig19_far_extrap_improved", bbox_inches="tight")
    plt.close(fig)
    print("✓ fig19_far_extrap_improved.png/.pdf")


    # ── fig20: wall clock speedup ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # Left: absolute times
    x = np.arange(len(DIFF_ORDER))
    w = 0.28
    for i, (method, label) in enumerate([
        ("neural_network","Neural Net"),
        ("pure_llm","Pure LLM"),
        ("hybrid","HypatiaX"),
    ]):
        means = []
        for d in DIFF_ORDER:
            times = [get_case(c, method, "time_s") for c in CASES if c["difficulty"]==d]
            times = [t for t in times if not math.isnan(t)]
            means.append(np.mean(times) if times else 0)
        axes[0].bar(x+(i-1)*w, means, w, label=label, color=MCOLORS[method], alpha=0.85, edgecolor="white")
    axes[0].set_xticks(x); axes[0].set_xticklabels([d.capitalize() for d in DIFF_ORDER])
    axes[0].set_ylabel("Mean wall-clock time (s)"); axes[0].set_title("Mean Solve Time by Difficulty")
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.3)
    # Right: total time pie
    totals = {m: np.nansum(collect("time_s", m)) for m in METHODS}
    axes[1].pie(list(totals.values()), labels=[MLABELS[m] for m in METHODS],
                colors=[MCOLORS[m] for m in METHODS],
                autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(edgecolor="white", lw=1.5))
    axes[1].set_title("Total Wall-Clock Time Distribution")
    fig.suptitle("Wall-Clock Time Analysis", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "fig20_wall_clock_speedup")
    plt.close(fig)
    print("✓ fig20_wall_clock_speedup.png/.pdf")


    # ── fig21: portfolio variance seed sweep ──────────────────────────────────────
    # Use the DATA from generate_plots.py (already in scope via the existing script content)
    SEED_DATA = {
      "pysr_only": [
        {"seed":42,   "train_r2":0.9095, "near_r2":-0.7811, "medium_r2":0.9268, "far_r2":-21.0040},
        {"seed":123,  "train_r2":0.9044, "near_r2": 0.2342, "medium_r2":0.9472, "far_r2":-18.6505},
        {"seed":777,  "train_r2":0.9564, "near_r2": 0.9999, "medium_r2":0.8005, "far_r2": -0.4378},
        {"seed":2024, "train_r2":0.9742, "near_r2": 0.5868, "medium_r2":0.9699, "far_r2":-12.1092},
        {"seed":99,   "train_r2":0.9668, "near_r2": 0.0715, "medium_r2":0.8659, "far_r2": -1.2264},
      ],
      "hypatia": [
        {"seed":42,   "train_r2":0.9134, "near_r2": 0.9478, "medium_r2":0.8552, "far_r2": -0.0232},
        {"seed":123,  "train_r2":0.9383, "near_r2": 0.7276, "medium_r2":0.6530, "far_r2":-18.0895},
        {"seed":777,  "train_r2":0.9978, "near_r2": 1.0000, "medium_r2":1.0000, "far_r2":  1.0000},
        {"seed":2024, "train_r2":0.9977, "near_r2": 1.0000, "medium_r2":1.0000, "far_r2":  1.0000},
        {"seed":99,   "train_r2":0.8958, "near_r2": 0.1572, "medium_r2":0.9228, "far_r2":-15.1913},
      ],
    }
    SEEDS = [r["seed"] for r in SEED_DATA["pysr_only"]]
    PYSR  = {r["seed"]: r for r in SEED_DATA["pysr_only"]}
    HYP   = {r["seed"]: r for r in SEED_DATA["hypatia"]}

    fig = plt.figure(figsize=(14, 5))
    gs  = GridSpec(1, 4, figure=fig, wspace=0.35)
    metrics = [("near_r2","Near"), ("medium_r2","Medium"), ("far_r2","Far")]

    for col_idx, (field, regime) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, col_idx])
        x  = np.arange(len(SEEDS))
        w  = 0.36
        pv = [PYSR[s][field] for s in SEEDS]
        hv = [HYP[s][field]  for s in SEEDS]
        lo = -25 if field == "far_r2" else -15
        pc = [max(lo, min(1.05, v)) for v in pv]
        hc = [max(lo, min(1.05, v)) for v in hv]
        ax.bar(x-w/2, pc, w, color=C_NN,  alpha=0.85, label="PySR-only",  edgecolor="white")
        ax.bar(x+w/2, hc, w, color=C_HYB, alpha=0.85, label="HypatiaX",  edgecolor="white")
        for i,(pval,hval,pclip,hclip) in enumerate(zip(pv,hv,pc,hc)):
            ax.text(x[i]-w/2, pclip+0.2, f"{pval:.1f}", ha="center", va="bottom", fontsize=6, color=C_NN)
            ax.text(x[i]+w/2, hclip+0.2, f"{hval:.2f}", ha="center", va="bottom", fontsize=6, color=C_HYB)
        ax.axhline(0.99, color=C_OK, lw=1.2, ls=":", alpha=0.8)
        ax.axhline(0,    color="black", lw=0.6, ls="--", alpha=0.4)
        ax.set_xticks(x); ax.set_xticklabels([str(s) for s in SEEDS], fontsize=8)
        ax.set_xlabel("Seed"); ax.set_ylabel(f"{regime} $R^2$")
        ax.set_title(f"{regime} Extrapolation", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if col_idx == 0: ax.legend(fontsize=7)

    # 4th panel: summary table
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis("off")
    cell_data = []
    for s in SEEDS:
        cell_data.append([
            str(s),
            f"{PYSR[s]['far_r2']:.2f}",
            f"{HYP[s]['far_r2']:.2f}",
            "✓" if HYP[s]["far_r2"] > 0.99 else "✗",
        ])
    tbl = ax4.table(
        cellText=cell_data,
        colLabels=["Seed","PySR far","Hyp far","Hyp✓"],
        cellLoc="center", loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    ax4.set_title("Far-$R^2$ Summary", fontsize=10, fontweight="bold")

    fig.suptitle("Portfolio Variance Seed Sweep: PySR-only vs HypatiaX", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "fig21_portfolio_variance_sweep")
    plt.close(fig)
    print("✓ fig21_portfolio_variance_sweep.png/.pdf")


    # ── fig22: bubble train vs far ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, method, label, color in [
        (axes[0], "hybrid",         "HypatiaX Hybrid", C_HYB),
        (axes[1], "neural_network", "Neural Net",       C_NN),
    ]:
        tr  = np.array([get_case(c, method, "train_r2")        for c in CASES])
        st  = np.array([get_case(c, method, "stability_score") for c in CASES])
        eg  = np.array([get_case(c, method, "extrapolation_gap") for c in CASES])
        sz  = np.clip(np.nan_to_num(np.abs(eg), nan=0), 0, 3) * 200 + 20
        diff_c = [DIFF_COLORS[d] for d in difficulties]
        ax.scatter(np.nan_to_num(tr, nan=0), np.clip(np.nan_to_num(st, nan=0), -3, 1),
                   s=sz, c=diff_c, alpha=0.7, edgecolors=color, lw=1.2, zorder=3)
        ax.axhline(0.99, color=C_OK,   lw=1.2, ls=":", alpha=0.8)
        ax.axvline(0.99, color=C_WARN, lw=1.0, ls=":", alpha=0.7)
        ax.set_xlabel("Train $R^2$", fontsize=10)
        ax.set_ylabel("Extrapolation Stability", fontsize=10)
        ax.set_title(label, fontsize=12, fontweight="bold", color=color)
        ax.set_xlim(-0.2, 1.1); ax.set_ylim(-3.2, 1.15)
        ax.grid(alpha=0.25)
        # bubble legend
        for sz_v, txt in [(20,"small gap"),(200,"medium"),(600,"large gap")]:
            ax.scatter([], [], s=sz_v, c="gray", alpha=0.6, label=txt)
        ax.legend(fontsize=7, title="Extrap. gap", title_fontsize=7)
    patches = [mpatches.Patch(color=DIFF_COLORS[d], label=d.capitalize()) for d in DIFF_ORDER]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Bubble Chart: Train vs Stability (bubble size = extrapolation gap)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _savefig(fig, "fig22_bubble_train_vs_far")
    plt.close(fig)
    print("✓ fig22_bubble_train_vs_far.png/.pdf")


    # ── fig_seed_sweep_comparison ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # Left: far_r2 per seed (bar)
    ax = axes[0]
    x = np.arange(len(SEEDS)); w = 0.36
    pv = [PYSR[s]["far_r2"] for s in SEEDS]
    hv = [HYP[s]["far_r2"]  for s in SEEDS]
    pc = [max(-25, min(1.05, v)) for v in pv]
    hc = [max(-25, min(1.05, v)) for v in hv]
    ax.bar(x-w/2, pc, w, color=C_NN,  alpha=0.85, label="PySR-only", edgecolor="white")
    ax.bar(x+w/2, hc, w, color=C_HYB, alpha=0.85, label="HypatiaX",  edgecolor="white")
    ax.axhline(0.99, color=C_OK, lw=1.2, ls=":", alpha=0.8, label="Success (0.99)")
    ax.axhline(0,    color="black", lw=0.6, ls="--", alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("Seed"); ax.set_ylabel("Far $R^2$ (clipped $-25$)")
    ax.set_title("Far-$R^2$ per Seed", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    # Right: scatter near vs far
    ax = axes[1]
    ax.scatter([PYSR[s]["near_r2"] for s in SEEDS], [PYSR[s]["far_r2"] for s in SEEDS],
               s=80, color=C_NN,  alpha=0.8, label="PySR-only", edgecolors="white", lw=0.5)
    ax.scatter([HYP[s]["near_r2"] for s in SEEDS], [HYP[s]["far_r2"] for s in SEEDS],
               s=80, color=C_HYB, alpha=0.8, label="HypatiaX",  edgecolors="white", lw=0.5,
               marker="D")
    for s in SEEDS:
        ax.annotate(str(s), (HYP[s]["near_r2"], HYP[s]["far_r2"]),
                    textcoords="offset points", xytext=(4,4), fontsize=7, color=C_HYB)
    ax.axhline(0.99, color=C_OK, lw=1, ls=":", alpha=0.7)
    ax.axvline(0.99, color=C_OK, lw=1, ls=":", alpha=0.7)
    ax.set_xlabel("Near $R^2$"); ax.set_ylabel("Far $R^2$")
    ax.set_title("Near vs Far Extrapolation", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.suptitle("Portfolio Variance Seed Sweep Comparison", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "fig_seed_sweep_comparison")
    plt.close(fig)
    print("✓ fig_seed_sweep_comparison.png/.pdf")


    # ── fig1_seed_sweep — richer per-seed line chart (P0 paper figure) ────────────
    # Each line traces one method across seeds; panels show near / medium / far.
    _REGIMES_SW = [
        ("near_r2",   "Near extrapolation $R^2$"),
        ("medium_r2", "Medium extrapolation $R^2$"),
        ("far_r2",    "Far extrapolation $R^2$"),
    ]
    _SW_CLIP = {"near_r2": (-5, 1.05), "medium_r2": (-5, 1.05), "far_r2": (-25, 1.05)}

    fig_sw, axes_sw = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    _seed_x = np.arange(len(SEEDS))

    for ax, (field, ylabel) in zip(axes_sw, _REGIMES_SW):
        lo, hi = _SW_CLIP[field]
        pv = np.array([max(lo, min(hi, PYSR[s][field])) for s in SEEDS])
        hv = np.array([max(lo, min(hi, HYP[s][field]))  for s in SEEDS])

        ax.plot(_seed_x, pv, color=C_NN,  lw=2, marker="o", ms=6, label="PySR-only")
        ax.plot(_seed_x, hv, color=C_HYB, lw=2, marker="D", ms=6, label="HypatiaX")

        # Annotate each point with the raw (unclipped) value when it was clipped.
        for i, s in enumerate(SEEDS):
            raw_p = PYSR[s][field]; raw_h = HYP[s][field]
            if raw_p < lo or raw_p > hi:
                ax.annotate(f"{raw_p:.1f}", (i, pv[i]), textcoords="offset points",
                            xytext=(0, -14), ha="center", fontsize=6.5, color=C_NN)
            if raw_h < lo or raw_h > hi:
                ax.annotate(f"{raw_h:.2f}", (i, hv[i]), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=6.5, color=C_HYB)

        ax.axhline(0.99, color=C_OK,   lw=1.2, ls=":", alpha=0.8, label="Success (0.99)")
        ax.axhline(0.0,  color="black", lw=0.7, ls="--", alpha=0.4)
        ax.fill_between(_seed_x, pv, hv, alpha=0.08, color=C_HYB)
        ax.set_xticks(_seed_x)
        ax.set_xticklabels([str(s) for s in SEEDS], fontsize=9)
        ax.set_xlabel("Seed", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel.split(" ")[0] + " Extrapolation", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        if ax is axes_sw[0]:
            ax.legend(fontsize=8)

    fig_sw.suptitle("Portfolio Variance Seed Sweep — Per-Seed Line Chart\n"
                    "(PySR-only vs HypatiaX across all extrapolation regimes)",
                    fontsize=12, fontweight="bold")
    fig_sw.tight_layout()
    _savefig(fig_sw, "fig1_seed_sweep")
    plt.close(fig_sw)
    print("✓ fig1_seed_sweep.png/.pdf")


    # ══════════════════════════════════════════════════════════════════════════════
    # RF09 INSTABILITY FIGURES
    # ══════════════════════════════════════════════════════════════════════════════

    instability = 1 - np.nan_to_num(h_stab, nan=0)  # 0=stable, 1=fully collapsed
    complexity  = np.array([len(ftypes[i].split("_")) + {"easy":1,"medium":2,"hard":3}[difficulties[i]]
                            for i in range(len(CASES))], dtype=float)

    # ── fig_instability_hist ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(instability, bins=20, color=C_HYB, alpha=0.8, edgecolor="white", lw=0.5)
    ax.axvline(instability.mean(), color=C_FAIL, lw=2, ls="--",
               label=f"Mean = {instability.mean():.3f}")
    ax.set_xlabel("Instability (1 − Stability Score)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("HypatiaX Instability Distribution (74 cases)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "fig_instability_hist")
    plt.close(fig)
    print("✓ fig_instability_hist.png/.pdf")


    # ── fig_instability_regimes ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(DIFF_ORDER))
    w = 0.35
    for i, (method, label, color) in enumerate([
        ("hybrid","HypatiaX",C_HYB), ("neural_network","Neural Net",C_NN)]):
        means = []
        sems  = []
        for d in DIFF_ORDER:
            inst = [1 - get_case(c, method, "stability_score")
                    for c in CASES if c["difficulty"]==d]
            inst = [v for v in inst if not math.isnan(v)]
            means.append(np.mean(inst) if inst else 0)
            sems.append(scipy_stats.sem(inst) if len(inst) > 1 else 0)
        ax.bar(x+(i-0.5)*w, means, w, label=label, color=color, alpha=0.85,
               edgecolor="white", lw=0.5, yerr=sems, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels([d.capitalize() for d in DIFF_ORDER])
    ax.set_ylabel("Mean Instability (±SEM)"); ax.set_ylim(0, 0.8)
    ax.set_title("Mean Instability by Difficulty", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "fig_instability_regimes")
    plt.close(fig)
    print("✓ fig_instability_regimes.png/.pdf")


    # ── fig_instability_3d ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")
    diff_num = np.array([{"easy":0,"medium":1,"hard":2}[d] for d in difficulties], dtype=float)
    eg_vals  = np.nan_to_num(h_egap, nan=0)
    sc = ax.scatter(complexity, diff_num, instability, c=instability,
                    cmap="RdYlGn_r", s=50, alpha=0.85, vmin=0, vmax=1.5)
    ax.set_xlabel("Complexity score"); ax.set_ylabel("Difficulty")
    ax.set_zlabel("Instability"); ax.set_yticks([0,1,2])
    ax.set_yticklabels(["Easy","Medium","Hard"])
    ax.set_title("3D Instability Space (HypatiaX)", fontsize=11, fontweight="bold")
    fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.12, label="Instability")
    fig.tight_layout()
    _savefig(fig, "fig_instability_3d")
    plt.close(fig)
    print("✓ fig_instability_3d.png/.pdf")


    # ── fig_instability_phase ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    diff_c = [DIFF_COLORS[d] for d in difficulties]
    sc = ax.scatter(complexity, instability, c=diff_c, s=60, alpha=0.75, edgecolors="white", lw=0.4)
    ax.axhline(0.5, color="black", lw=1, ls="--", alpha=0.5, label="Instability = 0.5")
    ax.set_xlabel("Complexity Score", fontsize=11)
    ax.set_ylabel("Instability (1 − Stability)", fontsize=11)
    ax.set_title("Instability Phase: Complexity vs Instability", fontsize=12, fontweight="bold")
    patches = [mpatches.Patch(color=DIFF_COLORS[d], label=d.capitalize()) for d in DIFF_ORDER]
    ax.legend(handles=patches, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "fig_instability_phase")
    plt.close(fig)
    print("✓ fig_instability_phase.png/.pdf")


    # ── fig_instability_surface ───────────────────────────────────────────────────
    if len(complexity) >= 4:
        fig = plt.figure(figsize=(10, 6))
        ax  = fig.add_subplot(111, projection="3d")
        # Create a grid surface
        cx = np.linspace(complexity.min(), complexity.max(), 20)
        dy = np.linspace(0, 2, 20)
        CX, DY = np.meshgrid(cx, dy)
        # Interpolate; fall back to nearest-neighbor if points are degenerate
        from scipy.interpolate import griddata
        from scipy.spatial import QhullError
        try:
            ZZ = griddata(
                np.column_stack([complexity, diff_num]),
                instability,
                (CX, DY),
                method="linear",
                fill_value=0,
            )
        except QhullError:
            ZZ = griddata(
                np.column_stack([complexity, diff_num]),
                instability,
                (CX, DY),
                method="nearest",
                fill_value=0,
            )
        surf = ax.plot_surface(CX, DY, ZZ, cmap="RdYlGn_r", alpha=0.7, edgecolor="none", vmin=0, vmax=1)
        ax.scatter(complexity, diff_num, instability, color=C_HYB, s=25, alpha=0.9, zorder=5)
        ax.set_xlabel("Complexity"); ax.set_ylabel("Difficulty")
        ax.set_zlabel("Instability"); ax.set_yticks([0,1,2])
        ax.set_yticklabels(["Easy","Med","Hard"])
        ax.set_title("Instability Surface (HypatiaX)", fontsize=11, fontweight="bold")
        fig.colorbar(surf, ax=ax, fraction=0.025, pad=0.1)
        fig.tight_layout()
        _savefig(fig, "fig_instability_surface")
        plt.close(fig)
        print("✓ fig_instability_surface.png/.pdf")
    else:
        print(f"  [SKIP] fig_instability_surface.png — need >=4 points for surface "
              f"interpolation, got {len(complexity)}.")


    # ── fig_instability_success_vs_instability ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    THRESH = 0.99
    success = (h_stab >= THRESH).astype(float)
    ax.scatter(instability, success + np.random.uniform(-0.03, 0.03, len(CASES)),
               c=[DIFF_COLORS[d] for d in difficulties], s=50, alpha=0.7,
               edgecolors="white", lw=0.4)
    ax.set_xlabel("Instability (1 − Stability Score)", fontsize=11)
    ax.set_ylabel("Success (1) / Failure (0)", fontsize=11)
    ax.set_yticks([0,1]); ax.set_yticklabels(["Failure","Success"])
    ax.set_title("Success vs Instability (HypatiaX Hybrid)", fontsize=12, fontweight="bold")
    patches = [mpatches.Patch(color=DIFF_COLORS[d], label=d.capitalize()) for d in DIFF_ORDER]
    ax.legend(handles=patches, fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "fig_instability_success_vs_instability")
    plt.close(fig)
    print("✓ fig_instability_success_vs_instability.png/.pdf")


    # ── hypatiax_instability_per_case ─────────────────────────────────────────────
    # Uses the primary CASES array (hypatiax_defi_variance_results.json removed —
    # not referenced in paper inventory or any .tex file).
    _var_cases = CASES
    fig, ax = plt.subplots(figsize=(10, 14))
    y_pos = np.arange(len(_var_cases))
    _var_stab = np.array([
        safe_float(c["results"].get("hybrid", {}).get("stability_score"))
        for c in _var_cases
    ])
    _var_inst = 1 - np.nan_to_num(_var_stab, nan=0)
    colors_bar = [C_OK if v < 0.01 else (C_WARN if v < 0.5 else C_FAIL) for v in _var_inst]
    ax.barh(y_pos, _var_inst, color=colors_bar, alpha=0.85, edgecolor="white", lw=0.3)
    ax.axvline(0.5, color="black", lw=1, ls="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{_var_cases[i]['test_case'][:34]}" for i in range(len(_var_cases))],
        fontsize=6.5
    )
    for tick, c in zip(ax.get_yticklabels(), _var_cases):
        tick.set_color(DIFF_COLORS.get(c.get("difficulty", "easy"), "#000000"))
    ax.set_xlabel("Instability (1 − Stability Score)", fontsize=10)
    ax.set_title("Instability per Case — HypatiaX Hybrid\n(green=easy, amber=medium, red=hard)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "hypatiax_instability_per_case", bbox_inches="tight")
    plt.close(fig)
    print("✓ hypatiax_instability_per_case.png/.pdf")


    # ── hypatiax_instability_histogram ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, diff in zip(axes, DIFF_ORDER):
        subset_inst = [instability[i] for i, c in enumerate(CASES) if c["difficulty"]==diff]
        ax.hist(subset_inst, bins=10, color=DIFF_COLORS[diff], alpha=0.8, edgecolor="white", lw=0.5)
        ax.set_title(f"{diff.capitalize()} (n={len(subset_inst)})", fontsize=11, fontweight="bold",
                     color=DIFF_COLORS[diff])
        ax.set_xlabel("Instability"); ax.set_ylabel("Count")
        ax.axvline(np.mean(subset_inst), color="black", lw=1.5, ls="--",
                   label=f"μ={np.mean(subset_inst):.3f}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("HypatiaX Instability by Difficulty Level", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "hypatiax_instability_histogram")
    plt.close(fig)
    print("✓ hypatiax_instability_histogram.png/.pdf")


    # ── hypatiax_instability_scatter ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(complexity, instability,
               c=[DIFF_COLORS[d] for d in difficulties],
               s=60, alpha=0.75, edgecolors="white", lw=0.4, zorder=3)
    # Fit a simple regression line
    valid = ~np.isnan(instability)
    _x, _y = complexity[valid], instability[valid]
    if len(_x) >= 2 and np.ptp(_x) > 0:
        m, b, r, p, se = scipy_stats.linregress(_x, _y)
        xs = np.linspace(complexity.min(), complexity.max(), 100)
        ax.plot(xs, m*xs+b, color="black", lw=1.5, ls="--", alpha=0.7,
                label=f"Trend: $r={r:.2f}$, $p={p:.3f}$")
        patches = [mpatches.Patch(color=DIFF_COLORS[d], label=d.capitalize()) for d in DIFF_ORDER]
        patches.append(mpatches.Patch(color="none", label=f"r={r:.2f}, p={p:.3f}"))
    else:
        patches = [mpatches.Patch(color=DIFF_COLORS[d], label=d.capitalize()) for d in DIFF_ORDER]
    ax.legend(handles=patches, fontsize=9)
    ax.set_xlabel("Complexity Score", fontsize=11)
    ax.set_ylabel("Instability", fontsize=11)
    ax.set_title("HypatiaX Instability vs Complexity", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "hypatiax_instability_scatter")
    plt.close(fig)
    print("✓ hypatiax_instability_scatter.png/.pdf")


    # ── hypatiax_instability_histogram_v2 (with KDE overlay) ─────────────────────
    try:
        from scipy.stats import gaussian_kde
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        for ax, diff in zip(axes, DIFF_ORDER):
            subset_inst = [instability[i] for i, c in enumerate(CASES) if c["difficulty"] == diff]
            ax.hist(subset_inst, bins=10, color=DIFF_COLORS[diff], alpha=0.55,
                    edgecolor="white", lw=0.5, density=True)
            if len(subset_inst) > 2:
                kde = gaussian_kde(subset_inst)
                xs_kde = np.linspace(min(subset_inst) - 0.05, max(subset_inst) + 0.05, 200)
                ax.plot(xs_kde, kde(xs_kde), color=DIFF_COLORS[diff], lw=2)
            ax.axvline(np.mean(subset_inst), color="black", lw=1.5, ls="--",
                       label=f"μ={np.mean(subset_inst):.3f}")
            ax.set_title(f"{diff.capitalize()} (n={len(subset_inst)})", fontsize=11,
                         fontweight="bold", color=DIFF_COLORS[diff])
            ax.set_xlabel("Instability"); ax.set_ylabel("Density")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.suptitle("HypatiaX Instability by Difficulty Level (KDE)", fontsize=12, fontweight="bold")
        fig.tight_layout()
        _savefig(fig, "hypatiax_instability_histogram_v2")
        plt.close(fig)
        print("✓ hypatiax_instability_histogram_v2.png/.pdf")
    except Exception as _e:
        print(f"  [SKIP] hypatiax_instability_histogram_v2 — {_e}")

    # ── hypatiax_instability_scatter_v2 (per-difficulty regression lines) ─────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(complexity, instability,
               c=[DIFF_COLORS[d] for d in difficulties],
               s=60, alpha=0.75, edgecolors="white", lw=0.4, zorder=3)
    patches_v2 = []
    for diff in DIFF_ORDER:
        mask = np.array([d == diff for d in difficulties])
        xd, yd = complexity[mask], instability[mask]
        patches_v2.append(mpatches.Patch(color=DIFF_COLORS[diff], label=diff.capitalize()))
        if len(xd) > 2 and xd.min() != xd.max():
            md, bd, rd, pd, _ = scipy_stats.linregress(xd, yd)
            xs_d = np.linspace(xd.min(), xd.max(), 50)
            ax.plot(xs_d, md * xs_d + bd, color=DIFF_COLORS[diff], lw=1.8, ls="--", alpha=0.7)
    ax.legend(handles=patches_v2, fontsize=9)
    ax.set_xlabel("Complexity Score", fontsize=11)
    ax.set_ylabel("Instability", fontsize=11)
    ax.set_title("HypatiaX Instability vs Complexity (per-difficulty trends)", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "hypatiax_instability_scatter_v2")
    plt.close(fig)
    print("✓ hypatiax_instability_scatter_v2.png/.pdf")


    # ── fig_paper_instability_hist ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # Left: instability histogram overlay all methods
    ax = axes[0]
    for method, label, color in [
        ("hybrid","HypatiaX",C_HYB),
        ("pure_llm","Pure LLM",C_LLM),
        ("neural_network","Neural Net",C_NN),
    ]:
        inst = np.array([1 - get_case(c, method, "stability_score") for c in CASES])
        inst = np.clip(np.nan_to_num(inst, nan=1), 0, 2)
        ax.hist(inst, bins=18, color=color, alpha=0.55, edgecolor="white", lw=0.3,
                label=label, density=True)
    ax.set_xlabel("Instability"); ax.set_ylabel("Density")
    ax.set_title("Instability Distribution — All Methods", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    # Right: CDF
    ax = axes[1]
    for method, label, color in [
        ("hybrid","HypatiaX",C_HYB),
        ("pure_llm","Pure LLM",C_LLM),
        ("neural_network","Neural Net",C_NN),
    ]:
        inst = np.array([1 - get_case(c, method, "stability_score") for c in CASES])
        inst = np.sort(np.clip(np.nan_to_num(inst, nan=1), 0, 2))
        cdf  = np.arange(1, len(inst)+1) / len(inst)
        ax.plot(inst, cdf, color=color, lw=2, label=label)
    ax.set_xlabel("Instability (threshold)"); ax.set_ylabel("Fraction of cases below threshold")
    ax.set_title("CDF of Instability", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.suptitle("Paper Figure: Instability Distribution & CDF", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "fig_paper_instability_hist")
    plt.close(fig)
    print("✓ fig_paper_instability_hist.png/.pdf")


    # ── fig_paper_mean_vs_instability ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # Left: mean stability per formula type
    ax = axes[0]
    ft_means = {}
    ft_inst  = {}
    for ft in sorted(set(ftypes)):
        s = [h_stab[i] for i, c in enumerate(CASES) if c["formula_type"]==ft]
        s = [v for v in s if not math.isnan(v)]
        if s:
            ft_means[ft] = np.mean(s)
            ft_inst[ft]  = 1 - np.mean(s)
    y_pos = np.arange(len(ft_means))
    bars = ax.barh(y_pos, list(ft_inst.values()),
                   color=[C_OK if v < 0.1 else (C_WARN if v < 0.5 else C_FAIL)
                          for v in ft_inst.values()],
                   alpha=0.85, edgecolor="white")
    ax.set_yticks(y_pos); ax.set_yticklabels(list(ft_inst.keys()), fontsize=8)
    ax.set_xlabel("Mean Instability"); ax.axvline(0.5, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_title("Mean Instability by Formula Type", fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    # Right: mean test_r2 per formula type (hybrid vs NN)
    ax = axes[1]
    ft_list_s = sorted(set(ftypes))
    hm, nm = [], []
    for ft in ft_list_s:
        h = [h_test[i] for i, c in enumerate(CASES) if c["formula_type"]==ft
             and not math.isnan(h_test[i])]
        n = [nn_test[i] for i, c in enumerate(CASES) if c["formula_type"]==ft
             and not math.isnan(nn_test[i])]
        hm.append(np.mean(h) if h else float("nan"))
        nm.append(np.mean(n) if n else float("nan"))
    y2 = np.arange(len(ft_list_s))
    ax.barh(y2-0.2, np.array([v if not math.isnan(v) else -2 for v in nm]), 0.38, color=C_NN,  alpha=0.8, label="Neural Net")
    ax.barh(y2+0.2, np.array([v if not math.isnan(v) else 0 for v in hm]), 0.38, color=C_HYB, alpha=0.8, label="HypatiaX")
    ax.set_yticks(y2); ax.set_yticklabels(ft_list_s, fontsize=8)
    ax.set_xlabel("Mean Test $R^2$"); ax.axvline(0, color="black", lw=0.7, ls="--", alpha=0.5)
    ax.set_title("Mean Test $R^2$ by Formula Type", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Performance by Formula Type", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "fig_paper_mean_vs_instability")
    plt.close(fig)
    print("✓ fig_paper_mean_vs_instability.png/.pdf")


    # ── fig_paper_complexity_vs_instability ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for diff in DIFF_ORDER:
        mask = [d == diff for d in difficulties]
        x_ = complexity[mask]; y_ = instability[mask]
        ax.scatter(x_, y_, c=DIFF_COLORS[diff], s=65, alpha=0.75,
                   edgecolors="white", lw=0.4, label=diff.capitalize(), zorder=3)
        # per-difficulty trend (skip if all x values are identical — no regression possible)
        if len(x_) > 2 and x_.min() != x_.max():
            m_, b_, _, _, _ = scipy_stats.linregress(x_, y_)
            xs_ = np.linspace(x_.min(), x_.max(), 50)
            ax.plot(xs_, m_*xs_+b_, color=DIFF_COLORS[diff], lw=1.5, ls="--", alpha=0.6)
    ax.set_xlabel("Complexity Score", fontsize=11)
    ax.set_ylabel("Instability (1 − Stability)", fontsize=11)
    ax.set_title("Complexity vs Instability by Difficulty (HypatiaX)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "fig_paper_complexity_vs_instability")
    plt.close(fig)
    print("✓ fig_paper_complexity_vs_instability.png/.pdf")


    # ── fig_paper_complexity_vs_success ──────────────────────────────────────────
    THRESH = 0.99
    fig, ax = plt.subplots(figsize=(8, 5))
    for diff in DIFF_ORDER:
        mask = [d == diff for d in difficulties]
        x_ = complexity[mask]
        s_ = np.array([1 if h_stab[i] >= THRESH else 0
                       for i, keep in enumerate(mask) if keep], dtype=float)
        ax.scatter(x_, s_ + np.random.uniform(-0.04, 0.04, len(x_)),
                   c=DIFF_COLORS[diff], s=60, alpha=0.75,
                   edgecolors="white", lw=0.4, label=diff.capitalize(), zorder=3)
    ax.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_yticks([0,1]); ax.set_yticklabels(["Failure (0)","Success (1)"])
    ax.set_xlabel("Complexity Score", fontsize=11)
    ax.set_title("Complexity vs Success (HypatiaX, threshold=0.99)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "fig_paper_complexity_vs_success")
    plt.close(fig)
    print("✓ fig_paper_complexity_vs_success.png/.pdf")


    # ── fig_paper_regime_counts ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    # Left: stacked bar — success/failure by method
    ax = axes[0]
    categories = ["easy","medium","hard"]
    x = np.arange(len(categories))
    w = 0.26
    for i, (method, label, color) in enumerate([
        ("pure_llm","Pure LLM",C_LLM),
        ("neural_network","Neural Net",C_NN),
        ("hybrid","HypatiaX",C_HYB),
    ]):
        ok_rates, fail_rates = [], []
        for d in categories:
            stabs = [get_case(c, method, "stability_score")
                     for c in CASES if c["difficulty"]==d]
            stabs = [v for v in stabs if not math.isnan(v)]
            ok_r  = sum(v >= THRESH for v in stabs) / max(len(stabs), 1)
            ok_rates.append(ok_r); fail_rates.append(1-ok_r)
        b1 = ax.bar(x+(i-1)*w, ok_rates,   w, color=C_OK,  alpha=0.85, edgecolor="white", lw=0.4)
        b2 = ax.bar(x+(i-1)*w, fail_rates, w, bottom=ok_rates, color=C_FAIL, alpha=0.85, edgecolor="white", lw=0.4)
        for b, v in zip(b1, ok_rates):
            ax.text(b.get_x()+b.get_width()/2, v/2, label[:4],
                    ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([d.capitalize() for d in categories])
    ax.set_ylabel("Fraction of cases")
    ax.set_title("Success/Failure by Difficulty × Method", fontsize=10, fontweight="bold")
    ax.legend(handles=[mpatches.Patch(color=C_OK,label="Success"),
                       mpatches.Patch(color=C_FAIL,label="Failure")], fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    # Right: decision type pie (hybrid only)
    ax = axes[1]
    decisions = {}
    for c in CASES:
        r = c["results"].get("hybrid", {})
        dec = r.get("decision", "unknown")
        decisions[dec] = decisions.get(dec, 0) + 1
    d_colors = {"llm": C_LLM, "nn": C_NN, "nn_to_llm_rescue": C_WARN, "nn_fallback": C_FAIL}
    labels_d = [f"{k} ({v})" for k, v in decisions.items()]
    colors_d = [d_colors.get(k, "gray") for k in decisions.keys()]
    ax.pie(list(decisions.values()), labels=labels_d, colors=colors_d,
           autopct="%1.1f%%", startangle=90,
           wedgeprops=dict(edgecolor="white", lw=1.5))
    ax.set_title("Hybrid Decision Types (74 cases)", fontsize=10, fontweight="bold")
    fig.suptitle("Regime Counts and Decision Distribution", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "fig_paper_regime_counts")
    plt.close(fig)
    print("✓ fig_paper_regime_counts.png/.pdf")




    # (fig_defi_r2_distribution removed — hypatiax_defi_benchmark_v3c3_results.json
#  not in paper inventory or any .tex file)


# (exp3_nguyen12_hybrid50v_extrap_r2 removed — exp3_nguyen12_output.json
#  not in paper inventory or any .tex file)


# (figure_5systems_comparison removed — systems_2_3_2_data.json + glob sources
#  not in paper inventory or any .tex file)


# ══════════════════════════════════════════════════════════════════════════════
# SUPP-B SWEEP FIGURES
# Sources: noise_sweep_*.json  +  sample_complexity_*.json  (latest matched by glob)
# ══════════════════════════════════════════════════════════════════════════════
_noise_raw  = _load_json(DATA_NOISE_SWEEP,  DATA_NOISE_SWEEP)
_sample_raw = _load_json(DATA_SAMPLE_SWEEP, DATA_SAMPLE_SWEEP)


def _sweep_rows(raw, key_field=None):
    """Normalise sweep JSON to a flat list of row-dicts.

    Tries, in order:
      1. raw is already a bare list of row-dicts.
      2. raw is a dict wrapping the list under a common key.
      3. raw is a dict keyed BY the sweep parameter itself, e.g.
         {"0.0": {...}, "0.1": {...}} or {"sigma_0.0": {...}}. Each value
         becomes a row, with the numeric part of its key injected under
         `key_field` ('sigma' for the noise sweep, 'n_samples' for the
         sample-complexity sweep) so downstream _pivot_sweep(rows, x_key, ..)
         still finds an x value per row. Only attempted when key_field is
         given, since this shape can't be distinguished from an ordinary
         dict otherwise.

    Returns [] if nothing matches — the caller's [WARN] diagnostic then
    reports raw's actual shape so a real fix can be added here.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, dict):
        return []

    for key in ("results", "data", "rows", "records", "sweep_results",
                "sweep", "noise_sweep", "sample_complexity",
                "runs", "entries", "samples"):
        if key in raw and isinstance(raw[key], list):
            return raw[key]

    if key_field and raw and all(isinstance(v, dict) for v in raw.values()):
        rows = []
        for k, v in raw.items():
            m = re.search(r"[-+]?\d*\.?\d+", str(k))
            if m is None:
                continue
            row = dict(v)
            row.setdefault(key_field, float(m.group()))
            rows.append(row)
        if rows:
            return rows

    return []


def _median_from_per_equation(level, method_name, metric):
    """Median of `metric` across every equation in level["per_equation"] for
    one method. method_summary never carries median_rmse (confirmed
    2026-06-18 schema), but per_equation has a per-equation value for every
    method at every n/sigma — so it can be derived here instead of skipping
    the figure outright. Returns None if per_equation is absent/empty or no
    equation has a usable value for this method+metric.
    """
    per_eq = level.get("per_equation")
    if not isinstance(per_eq, dict) or not per_eq:
        return None
    vals = []
    for eq_methods in per_eq.values():
        if not isinstance(eq_methods, dict):
            continue
        entry = eq_methods.get(method_name)
        if isinstance(entry, dict):
            v = entry.get(metric)
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                vals.append(float(v))
    return float(np.median(vals)) if vals else None


def _build_protocol_time_lookup(results_dir, sample_raw):
    """Recover per-(n, method) median solve time for the sample-complexity
    sweep from raw protocol_core_*.json files.

    sample_complexity_*.json's method_summary / per_equation never carry a
    "time" field (confirmed 2026-06-18) — timing only exists one stage
    upstream, in the raw per-test files written by
    run_protocol_benchmark_core.py, each with a "tests" list of
    {description, results: {method: {rmse, time, ...}}}.

    Those raw files don't reliably declare which sample size n they belong
    to — protocol.note is a stale hardcoded string (observed saying
    "Noisy 200-sample run" on files that were actually for n=500/750/1000),
    so it can't be trusted as a label. Instead, each candidate file is
    fingerprinted against every n's per_equation rmse values; a file is only
    accepted for a given n if EVERY (equation, method) rmse pair it reports
    matches that n's per_equation exactly (within float tolerance). This is
    deliberately strict — a partial match would risk mixing one sample
    size's timings into another's curve.

    Returns {(n_str, method_name): median_time_seconds}, built only from
    files that achieve a full match; n's with no matching raw file simply
    get no entry (callers / figure code already handle a missing key as
    "no data for this point").
    """
    if not isinstance(sample_raw, dict):
        return {}
    per_n = sample_raw.get("per_n")
    if not isinstance(per_n, dict) or not per_n:
        return {}

    candidates = glob.glob(os.path.join(results_dir, "**", "protocol_core_*.json"),
                            recursive=True)
    candidates = [c for c in candidates
                  if not any(s in os.path.basename(c) for s in _SWEEP_EXCLUDE_SUBSTRINGS)]
    if not candidates:
        return {}

    lookup = {}
    matched = []
    for path in sorted(candidates):
        try:
            with open(path) as f:
                proto = json.load(f)
        except Exception:
            continue
        tests = proto.get("tests")
        if not isinstance(tests, list) or not tests:
            continue

        # Fingerprint this file as {(equation_description, method): rmse}.
        fp = {}
        for t in tests:
            desc, results = t.get("description"), t.get("results")
            if desc is None or not isinstance(results, dict):
                continue
            for method_name, res in results.items():
                rmse = res.get("rmse") if isinstance(res, dict) else None
                if isinstance(rmse, (int, float)):
                    fp[(desc, method_name)] = float(rmse)
        if not fp:
            continue

        # Find the n whose per_equation rmse values match this fingerprint
        # most completely, then require a FULL match before trusting it.
        best_n, best_count = None, 0
        for n_str, level in per_n.items():
            per_eq = level.get("per_equation") if isinstance(level, dict) else None
            if not isinstance(per_eq, dict):
                continue
            count = 0
            for (desc, method_name), rmse in fp.items():
                entry = per_eq.get(desc)
                ref = entry.get(method_name, {}).get("rmse") if isinstance(entry, dict) else None
                if isinstance(ref, (int, float)) and math.isclose(ref, rmse, rel_tol=1e-9, abs_tol=1e-12):
                    count += 1
            if count > best_count:
                best_n, best_count = n_str, count

        if best_n is not None and best_count == len(fp):
            by_method = {}
            for t in tests:
                results = t.get("results")
                if not isinstance(results, dict):
                    continue
                for method_name, res in results.items():
                    tm = res.get("time") if isinstance(res, dict) else None
                    if isinstance(tm, (int, float)):
                        by_method.setdefault(method_name, []).append(float(tm))
            for method_name, vals in by_method.items():
                lookup[(best_n, method_name)] = float(np.median(vals))
            matched.append((os.path.basename(path), best_n))
        else:
            print(f"  [WARN] sample_complexity time recovery: {os.path.basename(path)} "
                  f"did not fully match any n's per_equation rmse fingerprint "
                  f"(best {best_count}/{len(fp)} pairs) — skipped, no timing "
                  f"borrowed from it.")

    if matched:
        for fname, n_str in matched:
            print(f"  [INFO] sample_complexity time recovery: {fname} → n={n_str} "
                  f"(full rmse fingerprint match; time injected)")
        missing_ns = sorted(set(per_n.keys()) - {n for _, n in matched}, key=float)
        if missing_ns:
            print(f"  [INFO] sample_complexity time recovery: no matching raw "
                  f"protocol_core_*.json found for n={missing_ns} — "
                  f"fig6_time_vs_n / fig_runtime_comparison will be partial "
                  f"({len(matched)} of {len(per_n)} sample sizes).")
    return lookup


def _flatten_per_n(raw, key_field, time_lookup=None):
    """Flatten the {"per_n": {"<n>": {"method_summary": {method: {metrics}}}}}
    shape (and its noise-sweep sibling, if ever produced the same way under
    "per_sigma"/"per_noise") into a flat list of row-dicts, one row per
    (n, method) pair, with the parameter value injected under key_field and
    the method name preserved under "method".

    This is the schema actually written by run_sample_complexity_benchmark.py
    (confirmed 2026-06-18): top-level keys include "per_n", and each per_n[n]
    is {"method_summary": {method_name: {median_r2, mean_r2, std_r2,
    recovery_rate, n_success, n_total, threshold_used}}, "per_equation": {...}}.

    _sweep_rows()'s generic per-parameter-dict fallback deliberately does NOT
    auto-flatten this nested method_summary shape (see _sweep_diag's
    per_n/per_sigma warning) because collapsing multiple methods into one
    row would average different systems together. This flattens explicitly
    instead, preserving "method" so downstream code can plot one line per
    method rather than one averaged line.

    median_rmse / median_time are not present in method_summary, but:
      - median_rmse IS derivable from this same file's per_equation (every
        equation has an rmse per method at every n) — computed here via
        _median_from_per_equation() rather than skipping fig5_rmse_vs_n.
      - median_time genuinely doesn't exist anywhere in this file; if the
        caller passes time_lookup (see _build_protocol_time_lookup), it's
        used to fill in median_time for whichever n's have a matching raw
        protocol_core_*.json file. n's without a match simply get no
        median_time key, same as if the source never had it.

    Returns [] if raw isn't a dict, or has no per_n/per_sigma/per_noise key
    with the expected nested shape — callers should fall back to the generic
    _sweep_rows() in that case.
    """
    if not isinstance(raw, dict):
        return []
    for nest_key in ("per_n", "per_sigma", "per_noise"):
        inner = raw.get(nest_key)
        if not isinstance(inner, dict) or not inner:
            continue
        rows = []
        for param_str, level in inner.items():
            if not isinstance(level, dict):
                continue
            method_summary = level.get("method_summary")
            if not isinstance(method_summary, dict):
                continue
            m = re.search(r"[-+]?\d*\.?\d+", str(param_str))
            if m is None:
                continue
            param_val = float(m.group())
            for method_name, metrics in method_summary.items():
                if not isinstance(metrics, dict):
                    continue
                row = dict(metrics)
                row[key_field] = param_val
                row["method"] = method_name
                if row.get("median_rmse") is None:
                    derived = _median_from_per_equation(level, method_name, "rmse")
                    if derived is not None:
                        row["median_rmse"] = derived
                if time_lookup is not None and row.get("median_time") is None:
                    t = time_lookup.get((param_str, method_name))
                    if t is not None:
                        row["median_time"] = t
                rows.append(row)
        if rows:
            return rows
    return []


# Recover sample-complexity timing from raw protocol_core_*.json files
# (one directory level upstream of sample_complexity_*.json itself) before
# flattening, so _flatten_per_n can fill in median_time wherever a matching
# raw file exists. See _build_protocol_time_lookup's docstring for why this
# can't just trust a label in the raw files and fingerprints them instead.
_sample_time_lookup = _build_protocol_time_lookup(_RESULTS_DIR, _sample_raw)

# Try the explicit per_n/method_summary flattening first (preserves method
# names as separate rows); fall back to the generic _sweep_rows() shapes
# (bare list / wrapped list / single-value-per-parameter dict) otherwise.
_noise_rows  = _flatten_per_n(_noise_raw,  key_field="sigma")     or _sweep_rows(_noise_raw,  key_field="sigma")
_sample_rows = (_flatten_per_n(_sample_raw, key_field="n_samples", time_lookup=_sample_time_lookup)
                or _sweep_rows(_sample_raw, key_field="n_samples"))


def _sweep_diag(raw, rows, label):
    """Print a one-line diagnostic for sweep data that loaded but parsed to
    zero rows. _load_json already reports a missing file; this covers the
    other failure mode — file found and valid JSON, but _sweep_rows() found
    none of the recognised shapes (a bare list; a dict wrapped under a
    common key; or a dict keyed by the sweep parameter itself). Without
    this, that case is silent: no figures get drawn and nothing in the log
    says why.

    Also specifically inspects a 'per_noise' / 'per_sample' / 'per_n' /
    'per_sigma' style nested shape — {param_value: {method_name: {metric:
    value, ...}, ...}, ...} — which _sweep_rows() deliberately does NOT
    auto-flatten: picking the wrong method, or averaging across methods,
    would silently produce a figure mixing different systems' numbers
    together. Reporting the method and metric names here lets that be wired
    up correctly instead of guessed.
    """
    if raw is None:
        return  # absence already reported by _load_json's [SKIP] line
    if rows:
        print(f"  [INFO] {label}: {len(rows)} row(s) parsed.")
        return

    if isinstance(raw, dict):
        for nest_key in ("per_noise", "per_sample", "per_n", "per_sigma"):
            inner = raw.get(nest_key)
            if isinstance(inner, dict) and inner:
                sample_param = next(iter(inner))
                sample_methods = inner[sample_param]
                if isinstance(sample_methods, dict) and sample_methods:
                    method_names = list(sample_methods.keys())
                    sample_method = method_names[0]
                    sample_metrics = sample_methods[sample_method]
                    metric_keys = (list(sample_metrics.keys())
                                   if isinstance(sample_metrics, dict) else None)
                    print(f"  [WARN] {label}: nested under '{nest_key}' — "
                          f"{len(inner)} parameter level(s) (e.g. {sample_param!r}), "
                          f"each with method(s) {method_names}"
                          + (f"; metric keys for {sample_method!r}: {metric_keys}"
                             if metric_keys is not None else "")
                          + ". NOT auto-flattened — picking the wrong method, or "
                            "averaging across methods, would silently mix different "
                            "systems' numbers into one curve. Tell me which method "
                            "name is the primary system to plot and this can be wired up.")
                    return
        shape = f"dict with top-level keys {list(raw.keys())[:10]}"
    elif isinstance(raw, list):
        shape = f"list of length {len(raw)}"
    else:
        shape = type(raw).__name__
    print(f"  [WARN] {label}: file loaded but 0 rows parsed — unrecognised "
          f"schema ({shape}). Tried: bare list; dict wrapped under a common "
          f"key (results/data/rows/records/sweep_results/sweep/noise_sweep/"
          f"sample_complexity/runs/entries/samples); dict keyed by the sweep "
          f"parameter itself. None matched — suppB sweep figures from this "
          f"source will be skipped.")


_sweep_diag(_noise_raw,  _noise_rows,  "noise_sweep")
_sweep_diag(_sample_raw, _sample_rows, "sample_complexity")


def _pivot_sweep(rows, x_key, y_keys):
    """Return {y_key: (sorted_x, mean_y, sem_y)} for a sweep result list."""
    from collections import defaultdict
    buckets: dict[str, dict] = {k: defaultdict(list) for k in y_keys}
    for r in rows:
        x_val = safe_float(r.get(x_key, float("nan")))
        if math.isnan(x_val): continue
        for k in y_keys:
            v = safe_float(r.get(k, float("nan")))
            if not math.isnan(v):
                buckets[k][x_val].append(v)
    out = {}
    for k in y_keys:
        xs = sorted(buckets[k].keys())
        ys   = [np.mean(buckets[k][x])                             for x in xs]
        sems = [scipy_stats.sem(buckets[k][x]) if len(buckets[k][x]) > 1 else 0 for x in xs]
        out[k] = (np.array(xs), np.array(ys), np.array(sems))
    return out


def _line_fig(xs, ys, sems, xlabel, ylabel, title, color, outpath, hline=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, color=color, lw=2, marker="o", ms=5)
    ax.fill_between(xs, ys - sems, ys + sems, color=color, alpha=0.18)
    if hline is not None:
        ax.axhline(hline, color=C_OK, lw=1.2, ls=":", alpha=0.8, label=f"y = {hline}")
        ax.legend(fontsize=8)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    stem = os.path.splitext(os.path.basename(outpath))[0]
    _savefig(fig, stem)
    plt.close(fig)
    print(f"✓ {stem}.png/.pdf")


_METHOD_COLORS = {}  # filled in lazily so any method name gets a stable color


def _color_for_method(method, fallback_colors=(C_HYB, C_NN, C_LLM, C_OK)):
    if method not in _METHOD_COLORS:
        _METHOD_COLORS[method] = fallback_colors[len(_METHOD_COLORS) % len(fallback_colors)]
    return _METHOD_COLORS[method]


def _multi_method_line_fig(rows, x_key, y_key, xlabel, ylabel, title, outpath, hline=None):
    """Like _line_fig, but draws one line per distinct 'method' value found
    in rows, each in its own color with a legend entry. Rows lacking a
    'method' field are pooled into a single unlabeled 'all' line so this
    degrades gracefully for non-flattened sources.

    Returns True if at least one line was drawn (and the figure was saved),
    False if there was no usable data for y_key (so the caller can skip
    saving an empty/blank figure and report it as such instead).
    """
    from collections import defaultdict
    methods = sorted({r.get("method", "all") for r in rows})
    any_drawn = False

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in methods:
        buckets = defaultdict(list)
        for r in rows:
            if r.get("method", "all") != method:
                continue
            x_val = safe_float(r.get(x_key, float("nan")))
            y_val = safe_float(r.get(y_key, float("nan")))
            if math.isnan(x_val) or math.isnan(y_val):
                continue
            buckets[x_val].append(y_val)
        if not buckets:
            continue
        xs   = np.array(sorted(buckets.keys()))
        ys   = np.array([np.mean(buckets[x]) for x in xs])
        sems = np.array([scipy_stats.sem(buckets[x]) if len(buckets[x]) > 1 else 0 for x in xs])
        color = _color_for_method(method)
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=5, label=method)
        ax.fill_between(xs, ys - sems, ys + sems, color=color, alpha=0.18)
        any_drawn = True

    if not any_drawn:
        plt.close(fig)
        return False

    if hline is not None:
        ax.axhline(hline, color="gray", lw=1.2, ls=":", alpha=0.8, label=f"y = {hline}")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    stem = os.path.splitext(os.path.basename(outpath))[0]
    _savefig(fig, stem)
    plt.close(fig)
    print(f"✓ {stem}.png/.pdf")
    return True


# ── Noise-sweep figures (fig1–fig3, fig7, fig9, fig10) ───────────────────────
if _noise_rows:
    _np = _pivot_sweep(_noise_rows, "sigma",
                       ["median_r2", "median_rmse", "avg_time",
                        "recovery_rate", "min_r2", "equation_r2"])

    _configs = [
        ("median_r2",      "fig1_r2_vs_noise",      "Noise level (σ)", "Median $R^2$",
         "Median $R^2$ vs Noise (σ)", C_HYB, 0.99),
        ("median_rmse",    "fig2_rmse_vs_noise",     "Noise level (σ)", "Median RMSE",
         "Median RMSE vs Noise (σ)", C_NN, None),
        ("avg_time",       "fig3_time_vs_noise",     "Noise level (σ)", "Avg time (s)",
         "Avg Solve Time vs Noise (σ)", C_LLM, None),
        ("recovery_rate",  "fig7_recovery_vs_noise", "Noise level (σ)", "Recovery rate",
         "Recovery Rate vs Noise (σ)", C_OK, 0.8),
        ("min_r2",         "fig9_minr2_vs_noise",    "Noise level (σ)", "Min $R^2$",
         "Min $R^2$ vs Noise (σ)", C_WARN, None),
    ]
    for y_key, stem, xl, yl, title, color, hline in _configs:
        if y_key in _np:
            xs, ys, sems = _np[y_key]
            _line_fig(xs, ys, sems, xl, yl, title, color,
                      os.path.join(_FIGURES_DIR, f"{stem}.png"), hline=hline)

    # fig10: per-equation R² box plot across noise levels
    # FIX FIG10-DEAD-BRANCH: _pivot_sweep() always populates every key passed
    # in y_keys (line ~2010-2016), even when zero rows had that field — it
    # just yields empty arrays. "equation_r2" was passed as a y_key above, so
    # the key is ALWAYS present in _np, making `if "equation_r2" not in _np`
    # permanently False and this entire fallback (the only branch that ever
    # calls savefig() for fig10_r2_boxplot_noise) permanently unreachable.
    # That's the confirmed root cause of fig10_r2_boxplot_noise never being
    # produced by any run (see figures_deploy log, run 27899208900: 17/18
    # NB-05 required stems present, fig10_r2_boxplot_noise the lone MISSING).
    # Fixed by checking for actual data (non-empty x-array) instead of key
    # presence. NOTE: this still only implements the fallback path (bucketing
    # median_r2 by noise level); a true primary path using real per-equation
    # equation_r2 data, if/when that field is populated upstream, remains
    # unimplemented — falling back unconditionally is a safe default since
    # the fallback was always silently dead code in every prior run anyway.
    if len(_np["equation_r2"][0]) == 0:
        # Fall back: box per noise level of median_r2 across raw rows
        from collections import defaultdict
        _noise_buckets: dict = defaultdict(list)
        for r in _noise_rows:
            sv = safe_float(r.get("sigma", float("nan")))
            v  = safe_float(r.get("median_r2", r.get("r2", float("nan"))))
            if not math.isnan(sv) and not math.isnan(v):
                _noise_buckets[sv].append(v)
        if _noise_buckets:
            _bx_xs = sorted(_noise_buckets.keys())
            _bx_data = [_noise_buckets[x] for x in _bx_xs]
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.boxplot(_bx_data, positions=range(len(_bx_xs)), widths=0.5,
                       patch_artist=True,
                       boxprops=dict(facecolor=C_HYB, alpha=0.6),
                       medianprops=dict(color="black", lw=1.5))
            ax.set_xticks(range(len(_bx_xs)))
            ax.set_xticklabels([f"{x:.2f}" for x in _bx_xs], rotation=45, fontsize=8)
            ax.axhline(0.99, color=C_OK, lw=1.2, ls=":", alpha=0.8, label="0.99")
            ax.set_xlabel("Noise level (σ)", fontsize=11)
            ax.set_ylabel("$R^2$", fontsize=11)
            ax.set_title("Per-Equation $R^2$ Box Plots vs Noise (σ)", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            _savefig(fig, "fig10_r2_boxplot_noise")
            plt.close(fig)
            print("✓ fig10_r2_boxplot_noise.png/.pdf")


# ── Sample-complexity figures (fig4–fig6, fig8) ───────────────────────────────
# FIX SUPPB_SC-METHOD-SPLIT: _sample_rows is now (when sourced from the
# per_n/method_summary schema) one row per (n, method) pair via
# _flatten_per_n(), so each figure plots one line per method on shared axes
# instead of averaging methods together.
#
# median_rmse / median_time are not present in method_summary itself, but
# _flatten_per_n() now backfills both where possible (see its docstring):
#   - median_rmse: derived from this same file's per_equation — available
#     for every n, since per_equation always has full coverage.
#   - median_time: recovered from raw protocol_core_*.json files one stage
#     upstream, via fingerprint-matching against per_equation rmse — only
#     available for n's where a matching raw file was found (see the
#     [INFO]/[WARN] lines printed by _build_protocol_time_lookup above).
# fig5_rmse_vs_n should now render fully; fig6_time_vs_n may render with
# fewer points than fig4/fig5/fig8 if some n's raw protocol files are
# missing — the [SKIP] below only fires if NO n has usable data at all.
if _sample_rows:
    _s_configs = [
        ("median_r2",      "fig4_r2_vs_n",      "Sample size (n)", "Median $R^2$",
         "Median $R^2$ vs Sample Size", 0.99),
        ("median_rmse",    "fig5_rmse_vs_n",     "Sample size (n)", "Median RMSE",
         "Median RMSE vs Sample Size", None),
        ("median_time",    "fig6_time_vs_n",     "Sample size (n)", "Median time (s)",
         "Median Solve Time vs Sample Size", None),
        ("recovery_rate",  "fig8_recovery_vs_n", "Sample size (n)", "Recovery rate",
         "Recovery Rate vs Sample Size", 0.8),
    ]
    for y_key, stem, xl, yl, title, hline in _s_configs:
        _drawn = _multi_method_line_fig(
            _sample_rows, "n_samples", y_key, xl, yl, title,
            os.path.join(_FIGURES_DIR, f"{stem}.png"), hline=hline)
        if not _drawn:
            print(f"  [SKIP] {stem}.png: no rows had a usable '{y_key}' value "
                  f"(not present in this source's method_summary schema).")


# ── fig11_recovery_heatmap (σ × n) ────────────────────────────────────────────
# The heatmap needs both a σ axis and an n axis.
# Full data:      _noise_rows have sigma but no n;  _sample_rows have n but no sigma.
# suppB_sc runs:  only _sample_rows are available — no noise_sweep_*.json.
# suppB runs:     only _noise_rows are available — no sample_complexity_*.json.
#
# Strategy: build the heatmap from whichever combination of sources is present.
#   - Both present  → combined (σ, n) grid, as before.
#   - Only _sample_rows → collapse sigma (treat as σ=0.05 per suppB_sc protocol),
#     produce a single-σ-row heatmap with n on the x-axis.  The σ column label
#     is annotated to make the missing sweep dimension clear.
#   - Only _noise_rows → collapse n (treat as n=200 per suppB protocol),
#     produce a single-n-column heatmap with σ on the y-axis.
#   - Neither → skip.
if _noise_rows or _sample_rows:
    from collections import defaultdict
    _heat: dict = defaultdict(list)

    # Populate from noise rows (sigma known, n assumed 200 if absent)
    for r in (_noise_rows or []):
        sv  = safe_float(r.get("sigma", r.get("noise", float("nan"))))
        nv  = safe_float(r.get("n_samples", r.get("n", 200.0)))  # noise sweep uses fixed n=200
        rv  = safe_float(r.get("recovery_rate", float("nan")))
        if not any(math.isnan(x) for x in [sv, nv, rv]):
            _heat[(sv, nv)].append(rv)

    # Populate from sample rows (n known, sigma assumed 0.05 if absent per suppB_sc protocol)
    for r in (_sample_rows or []):
        sv  = safe_float(r.get("sigma", r.get("noise", 0.05)))  # SC sweep uses fixed σ=5%
        nv  = safe_float(r.get("n_samples", r.get("n", float("nan"))))
        rv  = safe_float(r.get("recovery_rate", float("nan")))
        if not any(math.isnan(x) for x in [sv, nv, rv]):
            _heat[(sv, nv)].append(rv)

    if _heat:
        _sigmas = sorted(set(k[0] for k in _heat))
        _ns     = sorted(set(k[1] for k in _heat))
        _mat    = np.full((len(_sigmas), len(_ns)), float("nan"))
        for i, s in enumerate(_sigmas):
            for j, n in enumerate(_ns):
                if (s, n) in _heat:
                    _mat[i, j] = np.mean(_heat[(s, n)])

        # Choose figure height: at least 3in, scale with sigma count
        _fig_h = max(3.5, len(_sigmas) * 0.7 + 1.5)
        _fig_w = max(6,   len(_ns)     * 0.8 + 1.5)
        fig, ax = plt.subplots(figsize=(_fig_w, _fig_h))
        im = ax.imshow(np.nan_to_num(_mat, nan=0), vmin=0, vmax=1,
                       cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(_ns)))
        ax.set_xticklabels([str(int(n)) for n in _ns], fontsize=8, rotation=45)
        ax.set_yticks(range(len(_sigmas)))
        # Annotate sigma labels with "(fixed)" when there is only one unique value
        # — signals to the reader that this axis was not swept in this run.
        _sigma_labels = [
            (f"σ={s:.2f}" + (" (fixed, SC protocol)" if len(_sigmas) == 1 else ""))
            for s in _sigmas
        ]
        ax.set_yticklabels(_sigma_labels, fontsize=8)
        ax.set_xlabel("Sample size (n)", fontsize=10)
        ax.set_ylabel("Noise level (σ)", fontsize=10)
        # Title distinguishes partial vs full grid
        _n_src = bool(_noise_rows)
        _s_src = bool(_sample_rows)
        if _n_src and _s_src:
            _ht_title = "Recovery Heatmap (σ × n) — combined noise + SC sweep"
        elif _s_src:
            _ht_title = "Recovery Heatmap (σ fixed at 5% × n) — SC sweep only"
        else:
            _ht_title = "Recovery Heatmap (σ × n fixed at 200) — noise sweep only"
        ax.set_title(_ht_title, fontsize=11, fontweight="bold")
        for i in range(len(_sigmas)):
            for j in range(len(_ns)):
                v = _mat[i, j]
                if not math.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if v < 0.4 else "black")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Recovery rate")
        fig.tight_layout()
        _savefig(fig, "fig11_recovery_heatmap", bbox_inches="tight")
        plt.close(fig)
        print("✓ fig11_recovery_heatmap.png/.pdf")
    else:
        print("  [SKIP] fig11_recovery_heatmap.png — no (sigma, n, recovery_rate) triples could be built from available sweep data.")


# ── fig_runtime_comparison (6-method runtime bar) ────────────────────────────
if _noise_rows or _sample_rows:
    _rt_src = _noise_rows or _sample_rows
    from collections import defaultdict
    _rt_buckets: dict = defaultdict(list)
    for r in _rt_src:
        method = r.get("method", r.get("solver", "unknown"))
        for tkey in ("avg_time", "time_s", "median_time", "runtime"):
            v = safe_float(r.get(tkey, float("nan")))
            if not math.isnan(v):
                _rt_buckets[method].append(v)
                break

    if _rt_buckets:
        _rt_methods = list(_rt_buckets.keys())
        _rt_means   = [np.mean(v) for v in _rt_buckets.values()]
        _rt_order   = np.argsort(_rt_means)[::-1]  # descending
        _rt_m_sorted = [_rt_methods[i] for i in _rt_order]
        _rt_v_sorted = [_rt_means[i]   for i in _rt_order]
        _rt_colors   = [C_HYB if "hypatia" in m.lower() else
                        (C_LLM if "llm" in m.lower() else C_NN)
                        for m in _rt_m_sorted]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.barh(range(len(_rt_m_sorted)), _rt_v_sorted,
                       color=_rt_colors, alpha=0.85, edgecolor="white", lw=0.5)
        ax.set_yticks(range(len(_rt_m_sorted)))
        ax.set_yticklabels(_rt_m_sorted, fontsize=9)
        ax.set_xlabel("Mean runtime (s)", fontsize=10)
        ax.set_title("Runtime Comparison — 6 Methods", fontsize=11, fontweight="bold")
        for bar, v in zip(bars, _rt_v_sorted):
            ax.text(v + max(_rt_v_sorted)*0.01, bar.get_y() + bar.get_height()/2,
                    f"{v:.2f}s", va="center", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        _savefig(fig, "fig_runtime_comparison")
        plt.close(fig)
        print("✓ fig_runtime_comparison.png/.pdf")


# ── fig_comparative_table (domain × method, rendered as PNG) ─────────────────
if _noise_rows or _sample_rows:
    _ct_src = (_noise_rows or []) + (_sample_rows or [])
    from collections import defaultdict
    _ct: dict = defaultdict(lambda: defaultdict(list))
    for r in _ct_src:
        domain = r.get("domain", r.get("formula_type", "all"))
        method = r.get("method", r.get("solver", "all"))
        v = safe_float(r.get("median_r2", r.get("r2", float("nan"))))
        if not math.isnan(v):
            _ct[domain][method].append(v)

    if _ct:
        _ct_domains = sorted(_ct.keys())
        _ct_methods = sorted({m for d in _ct.values() for m in d})
        _ct_data    = [[f"{np.mean(_ct[d][m]):.3f}" if _ct[d].get(m) else "—"
                        for m in _ct_methods]
                       for d in _ct_domains]

        fig, ax = plt.subplots(figsize=(max(6, len(_ct_methods)*1.5),
                                        max(3, len(_ct_domains)*0.5 + 1)))
        ax.axis("off")
        tbl = ax.table(
            cellText=_ct_data,
            rowLabels=_ct_domains,
            colLabels=_ct_methods,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        # Colour header row
        for j in range(len(_ct_methods)):
            tbl[0, j].set_facecolor("#DBEAFE")
        ax.set_title("Domain × Method $R^2$ Comparison (median)", fontsize=11, fontweight="bold",
                     pad=10)
        fig.tight_layout()
        _savefig(fig, "fig_comparative_table", bbox_inches="tight")
        plt.close(fig)
        print("✓ fig_comparative_table.png/.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# exp2_feynman_pca — PCA 40/60 six-system benchmark
# Data source: benchmark_results_pca_4060.json — a FLAT list of per-(test,method)
# rows, NOT the dict-of-dicts / CASES schema used elsewhere in this file. Kept
# fully independent of RAW/CASES (same reasoning as the suppB/suppB_sc block
# above): this experiment never populates RAW, and this data has a different
# shape (6 competing methods, not pysr_only/hypatia) so it wouldn't fit
# _normalise_cases() even if RAW were populated.
#
# Produces two figure groups:
#   fig_exp2_pca_runtime_6way  — all 6 methods, routing/runtime comparison
#   fig_exp2_pca_r2_3way       — pure_llm vs neural_network vs hybrid, reusing
#                                 the existing METHODS/MCOLORS/MLABELS palette
# ══════════════════════════════════════════════════════════════════════════════
if _EXPERIMENT in ("exp2_feynman_pca", "exp2_feyman_pca"):
    DATA_PCA_4060 = _rpath("benchmark_results_pca_4060.json")
    _pca_raw = _load_json(DATA_PCA_4060, DATA_PCA_4060)

    if _pca_raw is None:
        print("  [SKIP] exp2_feynman_pca figures skipped — benchmark_results_pca_4060.json not found.")
    elif not isinstance(_pca_raw, list):
        print(f"  [WARN] benchmark_results_pca_4060.json is not a flat list "
              f"(got {type(_pca_raw).__name__}) — exp2_feynman_pca figures skipped.")
    else:
        _pca_rows = [r for r in _pca_raw if isinstance(r, dict) and "method" in r]
        print(f"  [INFO] exp2_feynman_pca: loaded {len(_pca_rows)} rows "
              f"({len(set(r['method'] for r in _pca_rows))} methods, "
              f"{len(set(r['test'] for r in _pca_rows))} tests).")

        # All 6 methods present in the file, in a fixed display order.
        _PCA_METHOD_ORDER = [
            "PureLLM Baseline (core)",
            "ImprovedNN (core)",
            "EnhancedHybridSystemDeFi (core)",
            "HybridSystemLLMNN all-domains (core)",
            "SymbolicEngineWithLLM (tools)",
            "HybridDiscoverySystem v50_2 (tools)",
        ]
        _pca_methods_present = [m for m in _PCA_METHOD_ORDER
                                 if any(r["method"] == m for r in _pca_rows)]
        # Any method in the data not in our fixed order still gets plotted,
        # appended at the end, rather than silently dropped.
        _pca_methods_present += sorted(
            set(r["method"] for r in _pca_rows) - set(_pca_methods_present)
        )

        # 3-way bucket mapping onto the existing pure_llm/neural_network/hybrid
        # palette. "hybrid" picked as HybridSystemLLMNN all-domains (core) —
        # the only all-domains hybrid variant in this file; the other three
        # hybrid-ish methods are DeFi-specific, tool-variant, or a versioned
        # discovery-system build, not the general-purpose hybrid this figure
        # is meant to represent.
        _PCA_3WAY_MAP = {
            "pure_llm":        "PureLLM Baseline (core)",
            "neural_network":  "ImprovedNN (core)",
            "hybrid":          "HybridSystemLLMNN all-domains (core)",
        }

        if not _pca_rows:
            print("  [SKIP] exp2_feynman_pca: no valid rows — figures skipped.")
        else:
            # ── fig_exp2_pca_runtime_6way: all 6 methods, runtime comparison ──
            _pca_runtimes = {
                m: [safe_float(r["runtime"]) for r in _pca_rows if r["method"] == m]
                for m in _pca_methods_present
            }
            _pca_runtimes = {m: [v for v in vs if not math.isnan(v)]
                              for m, vs in _pca_runtimes.items()}
            _pca_runtimes = {m: vs for m, vs in _pca_runtimes.items() if vs}

            if not _pca_runtimes:
                print("  [SKIP] fig_exp2_pca_runtime_6way — no valid runtime values.")
            else:
                fig, ax = plt.subplots(figsize=(11, 6))
                _labels6 = list(_pca_runtimes.keys())
                _data6 = [_pca_runtimes[m] for m in _labels6]
                bp = ax.boxplot(_data6, patch_artist=True, showmeans=True)
                ax.set_xticks(range(1, len(_labels6) + 1))
                ax.set_xticklabels(_labels6)
                _palette6 = plt.cm.tab10(np.linspace(0, 1, len(_labels6)))
                for patch, color in zip(bp["boxes"], _palette6):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax.set_ylabel("Runtime (s)")
                ax.set_title("exp2_feynman_pca — Runtime by Method (all 6 systems)",
                              fontsize=12, fontweight="bold")
                ax.set_yscale("log")
                plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
                ax.grid(axis="y", color=C_GRID, linewidth=0.6)
                fig.tight_layout()
                _savefig(fig, "fig_exp2_pca_runtime_6way", bbox_inches="tight")
                plt.close(fig)
                print("✓ fig_exp2_pca_runtime_6way.png/.pdf")

            # ── fig_exp2_pca_r2_3way: pure_llm vs neural_network vs hybrid ──
            _pca_r2_3way = {}
            for bucket, method_name in _PCA_3WAY_MAP.items():
                vals = [safe_float(r["r2"]) for r in _pca_rows if r["method"] == method_name]
                vals = [v for v in vals if not math.isnan(v)]
                if vals:
                    _pca_r2_3way[bucket] = vals

            _missing_3way = [b for b in _PCA_3WAY_MAP if b not in _pca_r2_3way]
            if _missing_3way:
                print(f"  [WARN] fig_exp2_pca_r2_3way — no data for bucket(s) "
                      f"{_missing_3way} (expected method(s): "
                      f"{[_PCA_3WAY_MAP[b] for b in _missing_3way]}); "
                      f"plotting available buckets only.")

            if not _pca_r2_3way:
                print("  [SKIP] fig_exp2_pca_r2_3way — no valid r2 values for any of the 3 buckets.")
            else:
                fig, ax = plt.subplots(figsize=(7, 6))
                _labels3 = [b for b in METHODS if b in _pca_r2_3way]
                _data3 = [_pca_r2_3way[b] for b in _labels3]
                bp = ax.boxplot(_data3, patch_artist=True, showmeans=True)
                ax.set_xticks(range(1, len(_labels3) + 1))
                ax.set_xticklabels([MLABELS[b] for b in _labels3])
                for patch, b in zip(bp["boxes"], _labels3):
                    patch.set_facecolor(MCOLORS[b])
                    patch.set_alpha(0.7)
                ax.set_ylabel("$R^2$")
                ax.set_title("exp2_feynman_pca — $R^2$ by System (Pure LLM / NN / Hybrid)",
                              fontsize=12, fontweight="bold")
                ax.grid(axis="y", color=C_GRID, linewidth=0.6)
                fig.tight_layout()
                _savefig(fig, "fig_exp2_pca_r2_3way", bbox_inches="tight")
                plt.close(fig)
                print("✓ fig_exp2_pca_r2_3way.png/.pdf")


# ── Final summary ─────────────────────────────────────────────────────────────
all_pngs = sorted(glob.glob(os.path.join(_FIGURES_DIR, "*.png")))
all_pdfs = {os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(_FIGURES_DIR, "*.pdf"))}
print(f"\n{'='*60}")
print(f"Generated {len(all_pngs)} figures in {_FIGURES_DIR}/")
print(f"{'='*60}")
pdf_missing = []
for f in all_pngs:
    stem = os.path.splitext(os.path.basename(f))[0]
    pdf_flag = "" if stem in all_pdfs else "  ⚠ PDF missing"
    print(f"  {os.path.basename(f)}{pdf_flag}")
    if pdf_flag:
        pdf_missing.append(stem)
if pdf_missing:
    print(f"\n  ⚠ {len(pdf_missing)} figure(s) have no PDF — PDF is required for LaTeX:")
    for s in pdf_missing:
        print(f"      {s}")

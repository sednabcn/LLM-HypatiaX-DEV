"""
regen_fig3_fig4.py
===================
Regenerates Figures 3 & 4 of the HypatiaX paper DIRECTLY from the values that
already appear in Table 6 (Core-15 ablation), instead of re-running the
original plotting pipeline against the raw exp1_ablation_results.json used
in production (which contains NaN/Inf artifacts that were leaking, unclipped,
into cell-text annotations and producing the giant garbled numbers seen in
the submitted Fig. 3 / Fig. 4).

Data source: exp1_ablation/exp1_ablation_results.json, hand-transcribed from
Table 6 in the manuscript (15 Core equations x {pysr_only, hypatia} x
{train_r2, extrap_r2_near, extrap_r2_medium, extrap_r2_far, total_time_s}).

Key fix vs. the original generate_figures.py fig09/fig18 blocks:
  - All raw values are passed through `_safe(v)` BEFORE they are used for
    color-mapping (clip) AND before they are used for text annotation. The
    original code clipped the color-mapped copy but rendered the *raw*
    value as text, which is fine for ordinary finite floats but explodes if
    the source data ever contains inf/-inf/NaN (nan_to_num's default
    posinf/neginf substitution is ~1.8e308, printed with `:.0f`).
  - `_safe(v)` maps NaN -> None (rendered as "nan"/"-"), and any non-finite
    value (inf/-inf) -> a capped sentinel with an explicit "±INF" label
    rather than a 300-digit number, so a future corrupt source can never
    reproduce the Fig. 3/4 corruption again.
  - Every rendered cell is diffed against the Table 6 source dict at the end
    of the script and a PASS/FAIL report is printed.
"""
import argparse, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------
# CLI: lets ci_postprocess.yml call this the same way it calls
# scripts/generate_figures.py --results-dir ... --figures-dir ..., instead
# of relying on the script's own directory for both input and output. Every
# flag is optional and falls back to the original standalone-script layout
# (HERE/exp1_ablation/exp1_ablation_results.json -> HERE/figures) so local,
# no-args usage is unchanged.
# ------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(
        description="Regenerate Figures 3 & 4 (Core-15 ablation) from Table-6 "
                     "values, NaN/Inf-safe."
    )
    p.add_argument(
        "--results-dir", default=None,
        help="Canonical results dir for exp1_ablation (e.g. the CI "
             "${OUT_BASE}/ablation/exp1_ablation checkout). The script looks "
             "for exp1_ablation_results.json directly inside this dir, or "
             "inside an exp1_ablation/ subfolder of it.",
    )
    p.add_argument(
        "--figures-dir", default=None,
        help="Output directory for the regenerated .png/.pdf figures. "
             "Defaults to a 'figures' folder next to this script.",
    )
    p.add_argument(
        "--data-path", default=None,
        help="Explicit path to exp1_ablation_results.json. Overrides "
             "--results-dir when both are given.",
    )
    return p.parse_args()

ARGS = _parse_args()

def _resolve_data_path():
    if ARGS.data_path:
        return ARGS.data_path
    if ARGS.results_dir:
        direct = os.path.join(ARGS.results_dir, "exp1_ablation_results.json")
        nested = os.path.join(ARGS.results_dir, "exp1_ablation", "exp1_ablation_results.json")
        if os.path.isfile(direct):
            return direct
        if os.path.isfile(nested):
            return nested
        # Neither exists — return the more likely (direct) path so the
        # FileNotFoundError below points at a sensible location for debugging.
        return direct
    return os.path.join(HERE, "exp1_ablation", "exp1_ablation_results.json")

DATA_PATH = _resolve_data_path()
OUT_DIR = ARGS.figures_dir or os.path.join(HERE, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.isfile(DATA_PATH):
    print(f"::error::regen_fig3_fig4.py: exp1_ablation_results.json not found at {DATA_PATH}", file=sys.stderr)
    print("  This script regenerates Fig 3/4 from a hand-transcribed Table-6 JSON, "
          "not from the raw CI pipeline output — place that file at the path above "
          "(or pass --data-path) before re-running.", file=sys.stderr)
    sys.exit(1)

with open(DATA_PATH) as f:
    RAW = json.load(f)

# ------------------------------------------------------------------
# Explicit NaN/Inf-safe helpers (the actual bug fix)
# ------------------------------------------------------------------
CAP = 1.5e4  # values beyond this are almost certainly a data error, not real R^2

def _safe(v):
    """Return (finite_float_or_None, was_nonfinite: bool).

    FIX: the original check only caught literal inf/-inf via math.isinf().
    That misses the actual failure mode seen in the submitted Fig. 3/4: an
    upstream `np.nan_to_num()` call (somewhere in the production pipeline,
    before this JSON was written) replaces +/-inf with the largest
    *finite* representable float, +/-1.7976931348623157e+308 -- which is
    exactly the ~300-digit garbage seen rendered as text in the submitted
    figures. math.isinf() on that value is False, so the old check let it
    straight through to _fmt(), which printed it in full with `:.0f`.
    Any |value| beyond CAP is treated the same as literal inf: capped and
    flagged, regardless of whether Python considers it technically finite.
    """
    if v is None:
        return None, False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None, False
    if math.isnan(f):
        return None, False
    if math.isinf(f) or abs(f) > CAP:
        return (CAP if f > 0 else -CAP), True
    return f, False

def _clip_for_color(v, lo=-1.5, hi=1.0):
    if v is None:
        return 0.0  # neutral mid-color for missing data; text will say "—"
    return max(lo, min(hi, v))

def _fmt(v, was_inf):
    if v is None:
        return "—"
    if was_inf:
        sign = "+" if v > 0 else "-"
        return f"{sign}INF"
    if abs(v) < 10:
        return f"{v:.2f}"
    return f"{v:.0f}"

# ------------------------------------------------------------------
# Normalise Table-6 data into the flat per-equation record list
# ------------------------------------------------------------------
CASES = []
for eq_name, entry in RAW.items():
    # The raw JSON also contains domain-rollup entries (e.g. "Biology",
    # "Chemistry", "Physics", "DeFi AMM", "DeFi Risk") interleaved with the
    # real per-equation entries at the same top level. These rollups carry
    # an "equation" key pointing at the real case name and an empty
    # "hypatia": {} (not None, so the old `h is None` check let them
    # through) -- without this filter they show up as 5 bogus extra rows
    # in Fig. 4 with blank "—" cells, making it look like there are 20
    # Core equations instead of the documented 15.
    if "equation" in entry:
        continue

    h = entry["hypatia"]
    p = entry.get("pysr_only")  # Some experiment outputs are Hypatia-only.

    # Skip entries that don't contain the required Hypatia result.
    if h is None:
        continue

    CASES.append({
        "name": eq_name,
        "domain": entry.get("domain"),
        "pysr_only": p,
        "hypatia": h,
    })
 
CASES.sort(key=lambda c: c["name"])  # alphabetical, matches original Fig. 4 row order

C_HYB = "#2563EB"
DIFF_COLORS = {"easy": "#059669", "medium": "#D97706", "hard": "#DC2626"}

# ==================================================================
# FIGURE 4 — R^2 heatmap, RAW (unclipped) values: Train/Near/Medium/Far
# Two panels: HypatiaX Hybrid | PySR-only  (matches original panel order)
# ==================================================================
col_keys = ["train_r2", "extrap_r2_near", "extrap_r2_medium", "extrap_r2_far"]
col_names = ["Train $R^2$", "Near", "Medium", "Far"]
panels = [("hypatia", "HypatiaX Hybrid")]

# Only add the PySR panel if at least one case contains baseline results.
if any(c["pysr_only"] is not None for c in CASES):
    panels.append(("pysr_only", "PySR-only"))

fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 8), sharey=True)

# When only one panel exists, make axes iterable.
if len(panels) == 1:
    axes = [axes]

diff_report_fig4 = []

for ax, (method_key, label) in zip(axes, panels):
    mat_val = np.full((len(CASES), len(col_keys)), np.nan)
    mat_txt = [["" for _ in col_keys] for _ in CASES]
    mat_isinf = np.zeros((len(CASES), len(col_keys)), dtype=bool)

    for i, c in enumerate(CASES):
        for j, k in enumerate(col_keys):
            method = c.get(method_key)
            if not isinstance(method, dict):
                method = {}

            raw = method.get(k)
            v, was_inf = _safe(raw)
            
            mat_isinf[i, j] = was_inf
            mat_val[i, j] = v if v is not None else np.nan
            mat_txt[i][j] = _fmt(v, was_inf)
            diff_report_fig4.append((c["name"], method_key, k, raw, v))

    mat_color = np.array([[_clip_for_color(v) for v in row] for row in mat_val])
    im = ax.imshow(mat_color, vmin=-1.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, fontsize=10)
    ax.set_yticks(range(len(CASES)))
    ax.set_yticklabels([c["name"] for c in CASES], fontsize=8)
    ax.set_title(label, fontsize=12, fontweight="bold")
    for i in range(len(CASES)):
        for j in range(len(col_keys)):
            col = "white" if mat_color[i, j] < -0.4 else "black"
            if mat_isinf[i, j]:
                col = "#7C3AED"  # flag any (hypothetical) inf distinctly
            ax.text(j, i, mat_txt[i][j], ha="center", va="center", fontsize=7, color=col)

axes[0].set_ylabel("Test Case (Core-15)", fontsize=10)
fig.colorbar(im, ax=axes[1], fraction=0.03, pad=0.03, label="$R^2$ (color, clipped to [-1.5, 1])")
fig.suptitle("Figure 4 (regenerated): $R^2$ Heatmap — Raw (unclipped) values, Train/Near/Medium/Far",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_r2_heatmap_regimes_regen.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "fig4_r2_heatmap_regimes_regen.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved fig4_r2_heatmap_regimes_regen.png/.pdf")

# ==================================================================
# FIGURE 3 — Far-extrap R^2 heatmap: PySR-only vs HypatiaX, Formula Type x
# Difficulty. Difficulty per case is inferred from the hybrid far-R^2 the
# same way the original pipeline did it (>=0.90 easy, >=0.50 medium, else
# hard), since Table 6 itself carries no explicit per-equation difficulty
# tag. This mirrors upstream behaviour exactly; only the NaN/Inf handling
# and data source are being fixed here.
# ==================================================================
def infer_difficulty(far_r2):
    v, was_inf = _safe(far_r2)
    if v is None:
        return "medium"
    if was_inf and v < 0:
        return "hard"
    if v >= 0.90:
        return "easy"
    if v >= 0.50:
        return "medium"
    return "hard"

for c in CASES:
    c["difficulty"] = infer_difficulty(c["hypatia"].get("extrap_r2_far"))

domains = sorted(set(c["domain"] for c in CASES))
diffs = ["easy", "medium", "hard"]
diff_labels = ["Easy", "Med", "Hard"]

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
panel_cfg = [("pysr_only", "PySR-only"), ("hypatia", "HypatiaX Hybrid")]
diff_report_fig3 = []

# Figure 3 is the far-extrap heatmap, so it always reads this one key.
# (Previously this relied on `k` leaking out of Figure 4's
# `for j, k in enumerate(col_keys)` loop still holding "extrap_r2_far" —
# correct by accident, and silently wrong the moment col_keys is reordered
# or the Figure 4 block above is refactored. Set it explicitly instead.)
FAR_KEY = "extrap_r2_far"

for ax, (method_key, label) in zip(axes, panel_cfg):
    mat_val = np.full((len(domains), len(diffs)), np.nan)
    mat_n = np.zeros((len(domains), len(diffs)), dtype=int)
    mat_isinf = np.zeros((len(domains), len(diffs)), dtype=bool)
    for i, dom in enumerate(domains):
        for j, dif in enumerate(diffs):
            vals = []
            any_inf = False
            for c in CASES:
                if c["domain"] != dom or c["difficulty"] != dif:
                    continue
                method = c.get(method_key)
                if not isinstance(method, dict):
                    method = {}

                raw = method.get(FAR_KEY)
                v, was_inf = _safe(raw)
                if v is not None:
                    vals.append(v)
                    any_inf = any_inf or was_inf
            if vals:
                mat_val[i, j] = float(np.mean(vals))
                mat_n[i, j] = len(vals)
                # A cell is flagged the same way Fig. 4 flags a single cell:
                # if ANY contributing case was a capped sentinel/inf value,
                # the cell is not a trustworthy finite average, so it gets
                # the same ±INF/purple treatment rather than a misleading
                # plain number like "-15000".
                mat_isinf[i, j] = any_inf
            diff_report_fig3.append((dom, dif, method_key, vals))

    mat_color = np.array([[_clip_for_color(v) for v in row] for row in mat_val])
    im = ax.imshow(mat_color, vmin=-1.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(diffs)))
    ax.set_xticklabels(diff_labels, fontsize=10)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(domains, fontsize=9)
    ax.set_title(label, fontsize=12, fontweight="bold")
    for i in range(len(domains)):
        for j in range(len(diffs)):
            v = mat_val[i, j]
            if math.isnan(v):
                txt = "—"
            elif mat_isinf[i, j]:
                sign = "+" if v > 0 else "-"
                txt = f"{sign}INF"
            else:
                txt = f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"
            col = "white" if mat_color[i, j] < -0.4 else "black"
            if mat_isinf[i, j]:
                col = "#7C3AED"  # matches Fig. 4's capped/inf flag color
            n = mat_n[i, j]
            if n > 1:
                txt += f"\n(n={n})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=col)

fig.colorbar(im, ax=axes[1], fraction=0.04, pad=0.03, label="Mean far-extrap $R^2$ (clipped)")
fig.suptitle("Figure 3 (regenerated): Far-Extrap $R^2$ — PySR-only vs HypatiaX\n(Formula Type × Difficulty, from Table 6)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_r2_heatmap_improved_regen.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "fig3_r2_heatmap_improved_regen.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved fig3_r2_heatmap_improved_regen.png/.pdf")

# ==================================================================
# DIFF CHECK: every Fig.4 cell vs Table 6 source, for 5 spot-check cases
# ==================================================================
SPOT = ["Arrhenius", "Michaelis-Menten", "Gravitational Force", "Constant Product", "Value at Risk"]
print("\n=== Spot-check: Figure 4 cells vs Table 6 source values ===")
all_ok = True
for name in SPOT:
    entry = RAW[name]
    for method_key, mlabel in [("pysr_only", "P"), ("hypatia", "H")]:
        method_entry = entry.get(method_key)
        if not isinstance(method_entry, dict):
            method_entry = {}  # e.g. "pysr_only": null for Hypatia-only cases
        for k, klabel in zip(col_keys, ["train", "near", "med", "far"]):
            table6_v = method_entry.get(k)
            rendered_v, was_capped = _safe(table6_v)
            exact_match = (rendered_v == table6_v) or (
                rendered_v is not None and table6_v is not None
                and abs(rendered_v - table6_v) < 1e-9
            )
            # A cap is not a bug: it's this script correctly refusing to
            # print a sentinel/garbage value as a raw number. Only an
            # uncapped, unexplained numeric difference is a real mismatch.
            if exact_match:
                status = "OK"
            elif was_capped:
                status = "CAPPED (expected)"
            else:
                status = "MISMATCH"
                all_ok = False
            table6_disp = "null" if table6_v is None else f"{table6_v:>12}"
            rendered_disp = "null" if rendered_v is None else f"{rendered_v:>12}"
            print(f"  {name:22s} {mlabel} {klabel:5s}: table6={table6_disp:>12} rendered={rendered_disp:>12}  [{status}]")

print(f"\nAll spot-checked cells OK or expectedly capped (no unexplained mismatches): {all_ok}")

# Full diff over every cell used in Fig. 4 (not just the 5 spot-check cases).
# A capped cell (|raw| > CAP, or literal inf) is expected to differ from its
# raw source value by design -- that's the fix working, not a defect -- so
# it is reported separately from a genuine, unexplained numeric mismatch.
capped_by_design = []
mismatches = []
for d in diff_report_fig4:
    name, method_key, k, raw_v, safe_v = d
    if raw_v is None or safe_v is None:
        if (raw_v is None) != (safe_v is None):
            mismatches.append(d)
        continue
    if abs(raw_v - safe_v) <= 1e-9:
        continue
    if math.isinf(raw_v) or abs(raw_v) > CAP:
        capped_by_design.append(d)
    else:
        mismatches.append(d)

print(f"Full Fig.4 cell diff against Table 6: {len(diff_report_fig4)} cells checked, "
      f"{len(capped_by_design)} capped by design (expected), "
      f"{len(mismatches)} genuine mismatches.")
if capped_by_design:
    print("\nCells capped by design (raw value was a sentinel/garbage value, now safely flagged):")
    for m in capped_by_design[:20]:
        print("  CAPPED:", m)
if mismatches:
    print("\nGenuine, unexplained mismatches (investigate these):")
    for m in mismatches[:20]:
        print("  MISMATCH:", m)

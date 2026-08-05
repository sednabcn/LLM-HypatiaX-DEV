#!/usr/bin/env python3
"""
generate_exp2_pca_comparison_table.py
──────────────────────────────────────────────────────────────────────────────
Generates a LaTeX / CSV / Markdown comparison table for:
  • exp2           — original Feynman benchmark (random_state=42, 80/20 split)
  • exp2_feynman_pca_4060 — FIX-C3 corrected run  (PCA 40/60 split)

Output (written to analysis/ or postprocess/ alongside existing tables):
  • exp2_pca_comparison.tex   — LaTeX table for the paper
  • exp2_pca_comparison.csv   — CSV for downstream analysis
  • exp2_pca_comparison.md    — Markdown summary

Usage:
  python generate_exp2_pca_comparison_table.py
  python generate_exp2_pca_comparison_table.py \
      --results-dir hypatiax/data/results \
      --output-dir  hypatiax/data/results/comparison_results/feynman-tests/exp2_pca_4060
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    Path(__file__).resolve().parents[3] / "hypatiax" / "data" / "results",
)
DEFAULT_OUTPUT_DIR = None   # falls back to <results_dir>/comparison_results/feynman-tests/exp2_pca_4060

THRESHOLD = float(os.environ.get("FEYNMAN_NOISELESS_THRESHOLD", "0.999999"))

PREFERRED_METHODS = {
    "hypatiax", "hybridv50", "hybrid50", "hybridsymbolic", "hybriddefi",
    "hypatia", "hybrid", "hybridllm", "hybridnn", "hybridllmnn",
    "hybridsystem", "hybridmodel", "hypatiaxsystem",
    "ours", "proposed",
}

# ── R² extraction helpers (mirrors audit logic) ────────────────────────────────
def _r2(row):
    for key in ("r2", "r2_test", "r2_train", "best_r2", "R2", "R2_test",
                "test_r2", "train_r2", "extrap_r2_far", "extrap_r2"):
        v = row.get(key)
        if v is not None:
            try:
                f = float(v)
                if f <= 1.01:
                    return f
            except (TypeError, ValueError):
                pass
    return None


def _iter_rows(data):
    if isinstance(data, dict):
        for key in ("results", "equation_results", "data", "rows", "items",
                    "entries", "output", "benchmark_results"):
            v = data.get(key)
            if v is not None:
                yield from _iter_rows(v)
                return
        yield data
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item


# ── Per-equation result extraction ────────────────────────────────────────────
def extract_per_equation(result_dir: Path, label: str):
    """
    Scan all *.json in result_dir (excluding checkpoints / summaries / disclosures).
    Returns a dict: equation_id -> {"r2": float, "solved": bool, "method": str, "file": str}
    If an equation appears in multiple files, best R² wins.
    """
    equations = {}
    patterns = [
        str(result_dir / "*.json"),
        str(result_dir / "**" / "*.json"),
    ]
    seen = set()
    files = []
    for pat in patterns:
        for fp in sorted(glob.glob(pat, recursive=True)):
            if fp not in seen:
                seen.add(fp)
                files.append(Path(fp))

    for fp in files:
        name = fp.name
        if any(x in name for x in ("checkpoint", "disclosure", "summary",
                                    "baseline", "_analysis", "_report",
                                    "benchmark_results")):
            continue
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue

        for row in _iter_rows(data):
            raw_method = (row.get("method") or row.get("model") or
                          row.get("system") or "")
            method_norm = (str(raw_method).lower()
                           .replace("-", "").replace("_", "").replace(" ", ""))
            # Accept rows where method is empty (single-model files) or preferred
            if method_norm and not any(p in method_norm for p in PREFERRED_METHODS):
                continue

            eq_id = (row.get("equation") or row.get("eq_id") or
                     row.get("name") or row.get("equation_name") or "")
            if not eq_id:
                # Try to infer from domain in filename
                eq_id = fp.stem

            r2_val = _r2(row)
            if r2_val is None:
                continue

            if eq_id not in equations or r2_val > equations[eq_id]["r2"]:
                equations[eq_id] = {
                    "r2": r2_val,
                    "solved": r2_val >= THRESHOLD,
                    "method": str(raw_method),
                    "file": fp.name,
                    "label": label,
                }

    return equations


# ── Summary loading (prefer pre-computed summary JSON) ────────────────────────
def load_summary(results_dir: Path, exp_id: str):
    """
    Returns (n_pass, n_total, solve_rate, source_note, per_equation_dict).
    Tries summary JSON first, then per-equation extraction.
    """
    subdir_map = {
        "exp2":               results_dir / "comparison_results/feynman-tests/exp2",
        "exp2_feynman":       results_dir / "comparison_results/feynman-tests/exp2",
        "exp2_pca_4060":      results_dir / "comparison_results/feynman-tests/exp2_pca_4060",
        "exp2_feynman_pca_4060": results_dir / "comparison_results/feynman-tests/exp2_pca_4060",
    }
    rdir = subdir_map.get(exp_id)
    if rdir is None:
        raise ValueError(f"Unknown exp_id: {exp_id!r}. Expected one of {list(subdir_map)}")

    # Pre-computed summary (preferred)
    summary_file = rdir / "exp2_pca_4060_summary.json"
    if exp_id in ("exp2_pca_4060", "exp2_feynman_pca_4060") and summary_file.exists():
        try:
            s = json.loads(summary_file.read_text())
            n_pass  = s.get("n_pass",  s.get("n_solved", 0))
            n_total = s.get("n_total", s.get("n_cases",  0))
            rate    = s.get("solve_rate") or (n_pass / n_total if n_total else None)
            note    = f"exp2_pca_4060_summary.json  ({s.get('split_protocol','pca_40_60')})"
            per_eq  = {}   # summary file doesn't carry per-equation detail
            return int(n_pass), int(n_total), rate, note, per_eq
        except Exception:
            pass  # fall through to per-equation extraction

    # fixc3_baseline.json for the legacy exp2
    if exp_id in ("exp2", "exp2_feynman"):
        baseline = results_dir / "fixc3_baseline.json"
        if baseline.exists():
            try:
                b = json.loads(baseline.read_text())
                n_pass  = int(b.get("n_pass", 0))
                n_total = int(b.get("n_total", 0))
                rate    = b.get("solve_rate") or (n_pass / n_total if n_total else None)
                note    = f"fixc3_baseline.json  ({b.get('split_protocol','random_80_20')})"
                per_eq  = {}
                return n_pass, n_total, rate, note, per_eq
            except Exception:
                pass

    # Full per-equation extraction fallback
    label = "Legacy (random 80/20)" if exp_id in ("exp2", "exp2_feynman") else "Corrected (PCA 40/60)"
    per_eq = extract_per_equation(rdir, label)
    n_pass  = sum(1 for v in per_eq.values() if v["solved"])
    n_total = len(per_eq)
    rate    = (n_pass / n_total) if n_total else None
    note    = f"computed from {rdir.relative_to(results_dir) if rdir.is_relative_to(results_dir) else rdir}"
    return n_pass, n_total, rate, note, per_eq


# ── Table generation ──────────────────────────────────────────────────────────
def build_rows(legacy_eq: dict, pca_eq: dict):
    """
    Merge per-equation dicts.
    Returns list of row dicts sorted by equation id.
    """
    all_ids = sorted(set(legacy_eq) | set(pca_eq))
    rows = []
    for eq_id in all_ids:
        leg = legacy_eq.get(eq_id, {})
        pca = pca_eq.get(eq_id, {})
        r2_leg = leg.get("r2")
        r2_pca = pca.get("r2")
        delta  = (r2_pca - r2_leg) if (r2_leg is not None and r2_pca is not None) else None
        rows.append({
            "equation":       eq_id,
            "r2_random_8020": r2_leg,
            "solved_random":  leg.get("solved", False),
            "r2_pca_4060":    r2_pca,
            "solved_pca":     pca.get("solved", False),
            "delta_r2":       delta,
        })
    return rows


def write_csv(rows, path: Path, meta_legacy, meta_pca):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "equation",
            "r2_random_8020", "solved_random_8020",
            "r2_pca_4060",    "solved_pca_4060",
            "delta_r2 (pca - random)",
        ])
        for r in rows:
            writer.writerow([
                r["equation"],
                f"{r['r2_random_8020']:.6f}" if r["r2_random_8020"] is not None else "N/A",
                "Y" if r["solved_random"] else "N",
                f"{r['r2_pca_4060']:.6f}"    if r["r2_pca_4060"]    is not None else "N/A",
                "Y" if r["solved_pca"]    else "N",
                f"{r['delta_r2']:+.4f}"       if r["delta_r2"]       is not None else "N/A",
            ])
        # Summary footer
        n_leg_pass  = meta_legacy[0]
        n_leg_total = meta_legacy[1]
        n_pca_pass  = meta_pca[0]
        n_pca_total = meta_pca[1]
        writer.writerow([])
        writer.writerow(["TOTAL",
                         "", f"{n_leg_pass}/{n_leg_total}",
                         "", f"{n_pca_pass}/{n_pca_total}", ""])
    print(f"  [CSV]   {path}")


def write_markdown(rows, path: Path, meta_legacy, meta_pca):
    n_leg_pass, n_leg_total, r_leg, _, _ = meta_legacy
    n_pca_pass, n_pca_total, r_pca, _, _ = meta_pca
    with open(path, "w") as f:
        f.write("# exp2 Feynman Benchmark — Random 80/20 vs PCA 40/60 Comparison\n\n")
        f.write("> **FIX-C3**: The original Feynman benchmark (`exp2`) used a random "
                "80/20 train/test split (`train_test_split(test_size=0.2, random_state=42)`).  \n"
                "> All DeFi benchmarks use a PCA-directed 40/60 extrapolation split "
                "(`build_extrap_split`, `extrap_train_frac=0.6`).  \n"
                "> These protocols are **not directly comparable**. "
                "This table shows both results side-by-side.\n\n")

        f.write("## Summary\n\n")
        f.write("| Split | Threshold | Solved | Solve Rate |\n")
        f.write("|-------|-----------|--------|------------|\n")
        r_leg_str = f"{r_leg:.3f}" if r_leg is not None else "N/A"
        r_pca_str = f"{r_pca:.3f}" if r_pca is not None else "N/A"
        f.write(f"| Random 80/20 (`random_state=42`) | R²≥{THRESHOLD} "
                f"| {n_leg_pass}/{n_leg_total} | {r_leg_str} |\n")
        f.write(f"| PCA 40/60 (FIX-C3 corrected)     | R²≥{THRESHOLD} "
                f"| {n_pca_pass}/{n_pca_total} | {r_pca_str} |\n\n")

        if rows:
            f.write("## Per-Equation Results\n\n")
            f.write("| Equation | R² (Random 80/20) | Solved | R² (PCA 40/60) | Solved | ΔR² |\n")
            f.write("|----------|:-----------------:|:------:|:--------------:|:------:|:---:|\n")
            for r in rows:
                eq   = r["equation"]
                r2l  = f"{r['r2_random_8020']:.4f}" if r["r2_random_8020"] is not None else "—"
                sl   = "✓" if r["solved_random"] else "✗"
                r2p  = f"{r['r2_pca_4060']:.4f}"    if r["r2_pca_4060"]    is not None else "—"
                sp   = "✓" if r["solved_pca"]    else "✗"
                dr   = f"{r['delta_r2']:+.4f}"       if r["delta_r2"]       is not None else "—"
                f.write(f"| {eq} | {r2l} | {sl} | {r2p} | {sp} | {dr} |\n")
        else:
            f.write("*Per-equation breakdown not available — "
                    "run with full result JSON files to populate this section.*\n\n")

        f.write("\n---\n")
        f.write(f"Threshold: R² ≥ {THRESHOLD}  |  "
                f"Legacy source: `fixc3_baseline.json`  |  "
                f"PCA source: `exp2_pca_4060_summary.json`\n")
    print(f"  [MD]    {path}")


def write_latex(rows, path: Path, meta_legacy, meta_pca):
    n_leg_pass, n_leg_total, r_leg, _, _ = meta_legacy
    n_pca_pass, n_pca_total, r_pca, _, _ = meta_pca
    r_leg_str = f"{r_leg:.3f}" if r_leg is not None else r"\textemdash"
    r_pca_str = f"{r_pca:.3f}" if r_pca is not None else r"\textemdash"

    lines = [
        r"% Auto-generated by generate_exp2_pca_comparison_table.py",
        r"% FIX-C3: Feynman benchmark split protocol comparison (§10.7)",
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Feynman SR benchmark: random 80/20 split (legacy) vs.\ PCA-directed",
        r"           40/60 extrapolation split (FIX-C3 corrected). Threshold: $R^2 \geq "
        + f"{THRESHOLD}$." + r"}",
        r"  \label{tab:exp2_pca_comparison}",
    ]

    if rows:
        lines += [
            r"  \begin{tabular}{lccccc}",
            r"    \toprule",
            r"    \textbf{Equation} & \multicolumn{2}{c}{\textbf{Random 80/20}}"
            r" & \multicolumn{2}{c}{\textbf{PCA 40/60 (FIX-C3)}} & $\Delta R^2$ \\",
            r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"    & $R^2$ & Solved & $R^2$ & Solved & \\",
            r"    \midrule",
        ]
        for r in rows:
            eq   = str(r["equation"]).replace("_", r"\_")
            r2l  = f"{r['r2_random_8020']:.4f}" if r["r2_random_8020"] is not None else r"---"
            sl   = r"\checkmark" if r["solved_random"] else r"$\times$"
            r2p  = f"{r['r2_pca_4060']:.4f}"    if r["r2_pca_4060"]    is not None else r"---"
            sp   = r"\checkmark" if r["solved_pca"]    else r"$\times$"
            dr   = f"{r['delta_r2']:+.4f}"       if r["delta_r2"]       is not None else r"---"
            lines.append(f"    {eq} & {r2l} & {sl} & {r2p} & {sp} & {dr} \\\\")
        lines += [
            r"    \midrule",
            rf"    \textbf{{Total}} & \multicolumn{{2}}{{c}}{{{n_leg_pass}/{n_leg_total} = {r_leg_str}}}"
            rf" & \multicolumn{{2}}{{c}}{{{n_pca_pass}/{n_pca_total} = {r_pca_str}}} & \\",
            r"    \bottomrule",
            r"  \end{tabular}",
        ]
    else:
        # Compact summary-only table when no per-equation data
        lines += [
            r"  \begin{tabular}{llll}",
            r"    \toprule",
            r"    \textbf{Split} & \textbf{Protocol} & \textbf{Solved} & \textbf{Solve Rate} \\",
            r"    \midrule",
            rf"    Random 80/20 (legacy) & \texttt{{train\_test\_split(rs=42)}} & {n_leg_pass}/{n_leg_total} & {r_leg_str} \\",
            rf"    PCA 40/60 (FIX-C3) & \texttt{{pca\_directed\_split}} & {n_pca_pass}/{n_pca_total} & {r_pca_str} \\",
            r"    \bottomrule",
            r"  \end{tabular}",
        ]

    lines += [r"\end{table}", ""]
    path.write_text("\n".join(lines))
    print(f"  [LaTeX] {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate exp2 vs exp2_pca_4060 comparison table."
    )
    parser.add_argument(
        "--results-dir", default=str(DEFAULT_RESULTS_DIR),
        help="Path to RESULTS_DIR (hypatiax/data/results)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to write output files (default: <results_dir>/comparison_results/feynman-tests/exp2_pca_4060)"
    )
    parser.add_argument(
        "--formats", default="tex,csv,md",
        help="Comma-separated list of output formats: tex, csv, md"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir) if args.output_dir else (
        results_dir / "comparison_results" / "feynman-tests" / "exp2_pca_4060"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = {f.strip() for f in args.formats.split(",")}

    print(f"\n=== exp2 PCA comparison table generator ===")
    print(f"  results_dir : {results_dir}")
    print(f"  output_dir  : {output_dir}")
    print(f"  threshold   : R² ≥ {THRESHOLD}")
    print()

    # ── Load legacy exp2 ──────────────────────────────────────────────────────
    print("[1/2] Loading legacy exp2 (random 80/20, random_state=42) ...")
    try:
        meta_legacy = load_summary(results_dir, "exp2")
        n_leg, n_leg_tot, r_leg, note_leg, per_eq_leg = meta_legacy
        r_leg_disp = f"{r_leg:.3f}" if r_leg is not None else "N/A"
        print(f"      Solved: {n_leg}/{n_leg_tot}  rate={r_leg_disp}  src={note_leg}")
    except Exception as e:
        print(f"  ERROR loading exp2: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Load PCA exp2 ─────────────────────────────────────────────────────────
    print("[2/2] Loading corrected exp2_pca_4060 (PCA 40/60) ...")
    try:
        meta_pca = load_summary(results_dir, "exp2_pca_4060")
        n_pca, n_pca_tot, r_pca, note_pca, per_eq_pca = meta_pca
        r_pca_disp = f"{r_pca:.3f}" if r_pca is not None else "N/A"
        print(f"      Solved: {n_pca}/{n_pca_tot}  rate={r_pca_disp}  src={note_pca}")
    except Exception as e:
        print(f"  ERROR loading exp2_pca_4060: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Merge per-equation data (if available) ────────────────────────────────
    rows = build_rows(per_eq_leg, per_eq_pca) if (per_eq_leg or per_eq_pca) else []

    # ── Write outputs ─────────────────────────────────────────────────────────
    print()
    print(f"Writing to {output_dir} ...")

    if "csv" in formats:
        write_csv(rows, output_dir / "exp2_pca_comparison.csv", meta_legacy, meta_pca)
    if "md" in formats:
        write_markdown(rows, output_dir / "exp2_pca_comparison.md", meta_legacy, meta_pca)
    if "tex" in formats:
        write_latex(rows, output_dir / "exp2_pca_comparison.tex", meta_legacy, meta_pca)

    # ── Print summary ─────────────────────────────────────────────────────────
    r_leg_str = f"{r_leg:.3f}" if r_leg is not None else "N/A"
    r_pca_str = f"{r_pca:.3f}" if r_pca is not None else "N/A"
    print()
    print("=== Summary ===")
    print(f"  {'Split':<40}  {'Solved':>8}  {'Rate':>6}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*6}")
    print(f"  {'Random 80/20 (random_state=42)  [legacy]':<40}  {n_leg:>3}/{n_leg_tot:<4}   {r_leg_str:>6}")
    print(f"  {'PCA 40/60  (FIX-C3 corrected)':<40}  {n_pca:>3}/{n_pca_tot:<4}   {r_pca_str:>6}")
    print()
    if r_leg is not None and r_pca is not None:
        direction = "lower (harder split, as expected)" if r_pca < r_leg else \
                    "higher (verify split was applied correctly)"
        print(f"  ΔSolveRate = {r_pca - r_leg:+.3f}  → {direction}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()

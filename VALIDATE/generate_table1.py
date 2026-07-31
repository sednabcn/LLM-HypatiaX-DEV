#!/usr/bin/env python3
"""
generate_table1.py — Regenerate the HypatiaX DeFi runtime table ("Table 1")
directly from raw per-task benchmark result files, with no manual number-
patching involved.

WHY THIS EXISTS
----------------
The manuscript's runtime table (NN mean 3.0s/median 2.7s, Hybrid mean 6.8s/
median 1.7s, claimed 1.73x/1.64x speedup) does not reproduce from either of
the two seed-42 raw result files audited (hypatiax_defi_benchmark_v3_results
_seed42.json, hypatiax_defi_benchmark_pca_results.json — both give NN times
of 0.25-0.42s, and hybrid SLOWER than NN on every statistic checked). No
defect was found in the current benchmark scripts' NN training code itself;
the mismatch is most plausibly explained by version drift against a
heavier, superseded NN implementation (see accompanying audit report). The
fix is a RE-RUN of the current authoritative scripts, not a hand-edit of
the table. This script is the tool that turns that re-run's output into the
table, mechanically and reproducibly, every time.

EXPLICIT RE-RUN RECIPE (what to run before invoking this script)
-------------------------------------------------------------------
Both hypatiax_defi_benchmark_v3c.py (aggressive 40/60 split) and
hypatiax_defi_benchmark_pca.py (PCA-directed 40/60 split) already support a
multi-seed sweep; a single seed (as in the two files audited) is not enough
to report a table with any notion of run-to-run variance. Use the seed set
already given as the worked example in pca.py's own --help text:

    # Aggressive 40/60 split (v3c) — env-var seed sweep (no --seeds flag on this script)
    DEFI_SEEDS=42,99,123,777,2024 python hypatiax_defi_benchmark_v3c.py \\
        --output-dir ./results/v3c_multiseed

    # PCA-directed 40/60 split — has a native --seeds flag
    python hypatiax_defi_benchmark_pca.py --seeds 42 99 123 777 2024 \\
        --output-dir ./results/pca_multiseed

Each seed produces its own seed-suffixed file:
    ./results/v3c_multiseed/hypatiax_defi_benchmark_v3_results_seed{S}.json
    ./results/pca_multiseed/hypatiax_defi_benchmark_pca_results_seed{S}.json

Then generate the table from all of them at once:

    python generate_table1.py \\
        --group "Aggressive 40/60 (v3c)|./results/v3c_multiseed/*seed*.json" \\
        --group "PCA-directed 40/60|./results/pca_multiseed/*seed*.json" \\
        --out table1.tex --json-out table1_summary.json

Re-running against just the two files already on hand (single seed each,
for a sanity check that this script reproduces the numbers reported in the
audit) looks like:

    python generate_table1.py \\
        --group "Aggressive 40/60 (v3c), seed42 only|hypatiax_defi_benchmark_v3_results_seed42.json" \\
        --group "PCA-directed 40/60, seed42 only|hypatiax_defi_benchmark_pca_results.json" \\
        --out table1_seed42_only.tex

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
It does not call the Anthropic API and does not run the benchmark itself —
that requires ANTHROPIC_API_KEY and the actual DeFi task harness. This
script only consumes whatever raw per-task JSON result file(s) that harness
produces and computes the timing table from them, honestly (no back-
calculating one cell from a claimed speedup, no silent exclusion of
timed-out or failed tasks without reporting it).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def load_group_files(pattern_list: list[str]) -> list[Path]:
    files: list[Path] = []
    for pat in pattern_list:
        matches = sorted(glob.glob(pat))
        if not matches and Path(pat).exists():
            matches = [pat]
        if not matches:
            print(f"  [warn] no files matched pattern: {pat}", file=sys.stderr)
        files.extend(Path(m) for m in matches)
    # de-dupe, keep order
    seen = set()
    out = []
    for f in files:
        rf = f.resolve()
        if rf not in seen:
            seen.add(rf)
            out.append(f)
    return out


def load_tasks(files: list[Path]) -> tuple[list[dict], dict]:
    """Load and concatenate all tasks across files; return (tasks, diagnostics)."""
    tasks: list[dict] = []
    seeds_seen = set()
    per_file_counts = {}
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            print(f"  [warn] {f}: expected a list of tasks, got {type(data)} — skipping", file=sys.stderr)
            continue
        per_file_counts[str(f)] = len(data)
        for t in data:
            seeds_seen.add(t.get("seed"))
            tasks.append(t)
    diagnostics = {
        "n_files": len(files),
        "files": [str(f) for f in files],
        "per_file_task_counts": per_file_counts,
        "seeds_seen": sorted(s for s in seeds_seen if s is not None),
        "n_tasks_total": len(tasks),
    }
    return tasks, diagnostics


def arm_times(tasks: list[dict], arm: str, only_decision: str | None = None) -> list[float]:
    out = []
    for t in tasks:
        r = t.get("results", {}).get(arm, {})
        if only_decision is not None:
            hyb_decision = t.get("results", {}).get("hybrid", {}).get("decision")
            if hyb_decision != only_decision:
                continue
        v = r.get("time_s")
        if _is_num(v) and v > 0:
            out.append(float(v))
    return out


def timeout_flags(tasks: list[dict], arm: str) -> int:
    return sum(1 for t in tasks if t.get("results", {}).get(arm, {}).get("timed_out") is True)


def mean_median(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    return statistics_mean(vals), stats.median(vals)


def statistics_mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def speedup(nn_stat: float | None, hyb_stat: float | None) -> str:
    if not nn_stat or not hyb_stat:
        return "n/a"
    ratio = nn_stat / hyb_stat
    if ratio >= 1.0:
        return f"{ratio:.2f}x faster (hybrid)"
    return f"{1.0 / ratio:.2f}x SLOWER (hybrid)"


def compute_group_stats(label: str, tasks: list[dict], diagnostics: dict) -> dict:
    llm_all = arm_times(tasks, "pure_llm")
    nn_all = arm_times(tasks, "neural_network")
    hyb_all = arm_times(tasks, "hybrid")
    hyb_llm_routed = arm_times(tasks, "hybrid", only_decision="llm")

    llm_mean, llm_med = mean_median(llm_all)
    nn_mean, nn_med = mean_median(nn_all)
    hyb_mean, hyb_med = mean_median(hyb_all)
    hybllm_mean, hybllm_med = mean_median(hyb_llm_routed)

    n_llm_routed = len(hyb_llm_routed)
    n_total = len(tasks)

    return {
        "label": label,
        "diagnostics": diagnostics,
        "n_tasks": n_total,
        "n_llm_routed": n_llm_routed,
        "pure_llm": {"mean": llm_mean, "median": llm_med, "n": len(llm_all)},
        "neural_network": {
            "mean": nn_mean, "median": nn_med, "n": len(nn_all),
            "max": max(nn_all) if nn_all else None,
            "n_timed_out": timeout_flags(tasks, "neural_network"),
        },
        "hybrid_all": {
            "mean": hyb_mean, "median": hyb_med, "n": len(hyb_all),
            "n_timed_out": timeout_flags(tasks, "hybrid"),
        },
        "hybrid_llm_routed_only": {
            "mean": hybllm_mean, "median": hybllm_med, "n": len(hyb_llm_routed),
        },
        "speedup_all_mean": speedup(nn_mean, hyb_mean),
        "speedup_all_median": speedup(nn_med, hyb_med),
        "speedup_llm_routed_mean": speedup(nn_mean, hybllm_mean),
        "speedup_llm_routed_median": speedup(nn_med, hybllm_med),
    }


def print_console_table(group_stats: list[dict]) -> None:
    for g in group_stats:
        print(f"\n=== {g['label']} ===")
        d = g["diagnostics"]
        print(f"  files pooled: {d['n_files']}  seeds: {d['seeds_seen']}  "
              f"tasks total: {d['n_tasks_total']}")
        if len(set(d["per_file_task_counts"].values())) > 1:
            print(f"  [warn] file task counts are not uniform: {d['per_file_task_counts']}")
        nn = g["neural_network"]
        hy = g["hybrid_all"]
        hyl = g["hybrid_llm_routed_only"]
        llm = g["pure_llm"]
        print(f"  Pure LLM     mean={llm['mean']:.3f}s  median={llm['median']:.3f}s  (n={llm['n']})"
              if llm["mean"] is not None else "  Pure LLM     no data")
        print(f"  Neural MLP   mean={nn['mean']:.3f}s  median={nn['median']:.3f}s  "
              f"max={nn['max']:.3f}s  timed_out={nn['n_timed_out']}  (n={nn['n']})"
              if nn["mean"] is not None else "  Neural MLP   no data")
        print(f"  Hybrid (all) mean={hy['mean']:.3f}s  median={hy['median']:.3f}s  "
              f"timed_out={hy['n_timed_out']}  (n={hy['n']})"
              if hy["mean"] is not None else "  Hybrid (all) no data")
        print(f"  Hybrid (LLM-routed only, n={g['n_llm_routed']}/{g['n_tasks']}) "
              f"mean={hyl['mean']:.3f}s  median={hyl['median']:.3f}s"
              if hyl["mean"] is not None else "  Hybrid (LLM-routed only) no data")
        print(f"  Speedup, all tasks:        mean={g['speedup_all_mean']}   median={g['speedup_all_median']}")
        print(f"  Speedup, LLM-routed only:  mean={g['speedup_llm_routed_mean']}   median={g['speedup_llm_routed_median']}")


def to_latex_table(group_stats: list[dict]) -> str:
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Wall-clock time per task (seconds), regenerated directly from raw "
                 r"per-task result files by \texttt{generate\_table1.py}. Values are mean / median; "
                 r"``speedup'' compares the neural baseline to hybrid (>1$\times$ = hybrid faster).}")
    lines.append(r"\label{tab:timing}")
    lines.append(r"\begin{tabular}{lrrrl}")
    lines.append(r"\toprule")
    lines.append(r"Split & Pure LLM (s) & Neural MLP (s) & Hybrid, all (s) & Speedup (mean / median) \\")
    lines.append(r"\midrule")
    for g in group_stats:
        llm = g["pure_llm"]; nn = g["neural_network"]; hy = g["hybrid_all"]
        def fmt(d):
            if d["mean"] is None:
                return "---"
            return f"{d['mean']:.2f} / {d['median']:.2f}"
        label = g["label"].replace("&", r"\&")
        lines.append(
            f"{label} & {fmt(llm)} & {fmt(nn)} & {fmt(hy)} & "
            f"{g['speedup_all_mean']} / {g['speedup_all_median']} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append(r"% LLM-routed-only subset, for reference:")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{As above, restricted to LLM-routed tasks only "
                 r"(\texttt{hybrid.decision == ``llm''}).}")
    lines.append(r"\label{tab:timing_llm_routed}")
    lines.append(r"\begin{tabular}{lrrl}")
    lines.append(r"\toprule")
    lines.append(r"Split & Neural MLP (s) & Hybrid, LLM-routed (s) & Speedup (mean / median) \\")
    lines.append(r"\midrule")
    for g in group_stats:
        nn = g["neural_network"]; hyl = g["hybrid_llm_routed_only"]
        def fmt2(d):
            if d["mean"] is None:
                return "---"
            return f"{d['mean']:.2f} / {d['median']:.2f}"
        label = g["label"].replace("&", r"\&")
        lines.append(
            f"{label} ($n$={g['n_llm_routed']}/{g['n_tasks']}) & {fmt2(nn)} & {fmt2(hyl)} & "
            f"{g['speedup_llm_routed_mean']} / {g['speedup_llm_routed_median']} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate Table 1 (DeFi runtime table) from raw per-task result files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--group", action="append", required=True, metavar="LABEL|PATTERN[,PATTERN...]",
        help=(
            "Repeatable. One table row. Format: 'Label|glob_or_path[,glob_or_path...]'. "
            "Multiple files/globs in one group are pooled (e.g. a multi-seed sweep). "
            "Example: --group 'Aggressive 40/60 (v3c)|results/v3c_multiseed/*seed*.json'"
        ),
    )
    parser.add_argument("--out", default=None, help="Write LaTeX table(s) to this file.")
    parser.add_argument("--json-out", default=None, help="Write full machine-readable summary JSON here.")
    args = parser.parse_args()

    group_stats = []
    for spec in args.group:
        if "|" not in spec:
            print(f"[error] --group value must be 'Label|pattern', got: {spec!r}", file=sys.stderr)
            sys.exit(2)
        label, pattern_str = spec.split("|", 1)
        patterns = [p.strip() for p in pattern_str.split(",") if p.strip()]
        files = load_group_files(patterns)
        if not files:
            print(f"[error] group '{label}': no files found for patterns {patterns}", file=sys.stderr)
            sys.exit(2)
        tasks, diagnostics = load_tasks(files)
        if diagnostics["n_tasks_total"] == 0:
            print(f"[error] group '{label}': loaded 0 tasks from {len(files)} file(s)", file=sys.stderr)
            sys.exit(2)
        group_stats.append(compute_group_stats(label, tasks, diagnostics))

    print_console_table(group_stats)

    latex = to_latex_table(group_stats)
    if args.out:
        Path(args.out).write_text(latex + "\n")
        print(f"\n[ok] LaTeX table written to {args.out}")
    else:
        print("\n" + "=" * 60)
        print(latex)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(group_stats, indent=2))
        print(f"[ok] JSON summary written to {args.json_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Recover Issue 9 — corrected HypatiaX Mean R^2 (and success-rate columns)
after fixing the hybrid decision-attribution masking bug.

Bug: HypatiaX's own 'hybrid.success'/'hybrid.test_r2' fields report
success=True, R2~1.0 for a task even when the sub-method actually named by
'hybrid.decision' (pure_llm or neural_network) itself failed
(sub.success == False, often catastrophically, e.g. test_r2 ~ -141,000 or NaN).

Fix: for every task, look up the ACTUAL result of the sub-method that
'hybrid.decision' routed to, clip it to [-10, 1] per the paper's own
convention, and recompute Mean R2 / Median R2 / %>0.99 / %>0.9 / Catastrophic
count from that corrected per-task value instead of the hybrid's
self-reported (masked) value.

Usage:
    python3 recover_issue9.py hypatiax_defi_benchmark_v3_results_seed42.json
"""
import json
import math
import sys
import statistics as stats


def clip(x, lo=-10.0, hi=1.0):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return lo  # unrunnable / NaN sub-method result -> treat as catastrophic floor
    return max(lo, min(hi, x))


def attributed_submethod(task):
    """Return the (name, result-dict) of the sub-method hybrid.decision actually routed to."""
    dec = task["results"]["hybrid"].get("decision")
    if dec == "llm":
        return "pure_llm", task["results"]["pure_llm"]
    elif dec in ("nn", "nn_fallback"):
        return "neural_network", task["results"]["neural_network"]
    else:
        raise ValueError(f"Unknown decision value: {dec!r} for task {task.get('equation_id')}")


def corrected_row(task):
    """
    Corrected per-task value, matching the paper's own disclosed correction
    methodology (verified to reproduce the stated 60.8% / 45/74 figure exactly):
    - If this task is one of the 22 'masked' tasks (hybrid falsely reports
      success=True, R2~1.0 while the routed sub-method actually failed),
      replace the value with the ACTUAL routed sub-method's test_r2, clipped.
    - Otherwise, keep HypatiaX's own reported hybrid.test_r2 (clipped), since
      those are not affected by the attribution bug.
    """
    sub_name, sub = attributed_submethod(task)
    h = task["results"]["hybrid"]
    is_masked = (
        h.get("success") is True
        and sub.get("success") is False
        and h.get("test_r2") is not None
        and h.get("test_r2") > 0.99
    )
    raw_r2 = sub.get("test_r2") if is_masked else h.get("test_r2")
    clipped_r2 = clip(raw_r2)
    return {
        "equation_id": task["equation_id"],
        "decision": task["results"]["hybrid"].get("decision"),
        "sub_method": sub_name,
        "sub_success": sub.get("success"),
        "sub_test_r2_raw": raw_r2,
        "corrected_r2_clipped": clipped_r2,
        "hybrid_reported_success": h.get("success"),
        "hybrid_reported_r2": h.get("test_r2"),
        "corrected_source": "sub_method" if is_masked else "hybrid_reported",
        "masked": is_masked,
    }


def summarize(values, catastrophic_threshold=-10.0, n_fixed=74):
    """values: list of clipped-to-[-10,1] R2 floats, one per task (already includes all tasks)."""
    n = n_fixed
    mean_r2 = sum(values) / n
    median_r2 = stats.median(values)
    pct_gt_099 = 100.0 * sum(1 for v in values if v > 0.99) / n
    pct_gt_09 = 100.0 * sum(1 for v in values if v > 0.9) / n
    # Catastrophic here is defined pre-clip (raw < -10); since we already
    # clipped to the floor of -10, count values sitting exactly at the floor
    # that originated from a raw value < -10 (passed in separately below).
    return {
        "mean_r2": mean_r2,
        "median_r2": median_r2,
        "pct_gt_0.99": pct_gt_099,
        "pct_gt_0.9": pct_gt_09,
    }


# --- Original, uncorrected LaTeX table (Source B) --------------------------
# Verbatim from 09_mean_r2_masked_failure.tex, l.26-51 / jmlr_paper_main.tex
# tab:main_results, l.1261-1285. This is the buggy version being replaced.
ORIGINAL_TEX_TABLE = r"""\subsection{Overall Extrapolation Performance (DeFi v3.0)}\label{subsec:overall-extrapolation-performance-defi-v}
\label{sec:results_defi}

Table~\ref{tab:main_results} presents the primary results across all 74 benchmark tasks.

\begin{table}[h]
\centering
\caption{Aggregate extrapolation performance on the HypatiaX DeFi Benchmark (74 tasks).
All $\Rsq$ values clipped to $[-10,1]$; fixed denominator of 74.
Catastrophic: $\Rsq < -10$. \textbf{HypatiaX's $90.5\,\%$ figures are
uncorrected} for the hybrid decision-attribution bug
(\S\ref{sec:hybrid-attribution-bug}); the corrected overall near-perfect
rate is $60.8\,\%$ (below Pure LLM's $62.2\,\%$). The Catastrophic column
(0) is unaffected --- see the abstract's footnote for why.}
\label{tab:main_results}
\begin{tabular}{lrrrrr}
\toprule
Method & Median $\Rsq$ & Mean $\Rsq$ & $>0.99$ (\%) & $>0.9$ (\%) & Catastrophic \\
\midrule
Pure LLM      & 1.0000 & $-0.7571$ & 62.2 & 62.2 & 6 \\
Neural MLP    & $-0.4675$ & $-0.9482$ & 5.4  & 12.2 & 0 \\
HypatiaX      & 1.0000 & $+0.8721$ & \textbf{90.5} & \textbf{90.5} & \textbf{0} \\
\bottomrule
\end{tabular}
\end{table}
"""


def generate_corrected_tex(summary, catastrophic_count, n_masked):
    """
    Build the drop-in replacement for tab:main_results (jmlr_paper_main.tex,
    l.1261-1285). Only the HypatiaX row and the caption change — Pure LLM and
    Neural MLP rows are untouched because the decision-attribution bug only
    affects HypatiaX's own self-reported hybrid.success/test_r2 fields.
    """
    median_r2 = summary["median_r2"]
    mean_r2 = summary["mean_r2"]
    pct99 = summary["pct_gt_0.99"]
    pct9 = summary["pct_gt_0.9"]

    return r"""\subsection{Overall Extrapolation Performance (DeFi v3.0)}\label{subsec:overall-extrapolation-performance-defi-v}
\label{sec:results_defi}

Table~\ref{tab:main_results} presents the primary results across all 74 benchmark tasks.

\begin{table}[h]
\centering
\caption{Aggregate extrapolation performance on the HypatiaX DeFi Benchmark (74 tasks).
All $\Rsq$ values clipped to $[-10,1]$; fixed denominator of 74.
Catastrophic: $\Rsq < -10$. \textbf{All HypatiaX figures below are corrected}
for the hybrid decision-attribution bug (\S\ref{sec:hybrid-attribution-bug}):
for each of the """ + str(n_masked) + r""" tasks where the hybrid system's own
accounting falsely reported \texttt{success=True}, $\Rsq \approx 1.0$ while the
routed sub-method actually failed, the sub-method's true (clipped) $\Rsq$ is
used instead. The corrected near-perfect rate is $""" + f"{pct99:.1f}" + r"""\,\%$
(""" + str(int(round(pct99/100*74))) + r"""/74), below Pure LLM's $62.2\,\%$,
and the corrected Mean $\Rsq$ is $""" + f"{mean_r2:+.4f}" + r"""$.}
\label{tab:main_results}
\begin{tabular}{lrrrrr}
\toprule
Method & Median $\Rsq$ & Mean $\Rsq$ & $>0.99$ (\%) & $>0.9$ (\%) & Catastrophic \\
\midrule
Pure LLM      & 1.0000 & $-0.7571$ & 62.2 & 62.2 & 6 \\
Neural MLP    & $-0.4675$ & $-0.9482$ & 5.4  & 12.2 & 0 \\
HypatiaX      & """ + f"{median_r2:.4f}" + r""" & $""" + f"{mean_r2:+.4f}" + r"""$ & """ + f"{pct99:.1f}" + r""" & """ + f"{pct9:.1f}" + r""" & """ + str(catastrophic_count) + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""


def main(path):
    with open(path) as f:
        data = json.load(f)

    assert len(data) == 74, f"Expected 74 tasks, found {len(data)}"

    rows = [corrected_row(t) for t in data]

    masked = [r for r in rows if r["masked"]]
    print(f"Masked tasks found (hybrid success=True,R2>0.99 but routed sub-method "
          f"actually failed): {len(masked)}")
    print("Matches the abstract's disclosed count of 22:", len(masked) == 22)
    print()

    # --- Corrected HypatiaX row ---
    corrected_values = [r["corrected_r2_clipped"] for r in rows]

    # Catastrophic: raw pre-clip value < -10 (or NaN/unrunnable), counted only
    # for the corrected-source value actually used per task (matches the
    # paper's "Catastrophic: R2 < -10" definition applied post-correction).
    def is_catastrophic(r):
        raw = r["sub_test_r2_raw"] if r["corrected_source"] == "sub_method" else r["hybrid_reported_r2"]
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            return True
        return raw < -10

    catastrophic_raw = [r for r in rows if is_catastrophic(r)]
    summary = summarize(corrected_values)

    print("=== CORRECTED HypatiaX row (tab:main_results) ===")
    print(f"  Median R2      : {stats.median(corrected_values):.4f}")
    print(f"  Mean R2        : {summary['mean_r2']:+.4f}")
    print(f"  >0.99 (%)      : {summary['pct_gt_0.99']:.1f}")
    print(f"  >0.9  (%)      : {summary['pct_gt_0.9']:.1f}")
    print(f"  Catastrophic   : {len(catastrophic_raw)}  "
          f"(raw sub-method test_r2 < -10 or NaN/unrunnable, {len(catastrophic_raw)}/74 "
          f"= {100*len(catastrophic_raw)/74:.1f}%)")
    print()

    print("=== Cross-check against paper's disclosed 'corrected' figures ===")
    print("  Abstract footnote claims corrected near-perfect rate: 60.8% (45/74)")
    computed_45 = sum(1 for v in corrected_values if v > 0.99)
    print(f"  Computed >0.99 count from this script: {computed_45}/74 "
          f"= {100*computed_45/74:.1f}%  ->  MATCH: {computed_45 == 45}")
    print()

    print("=== The 22 masked tasks (hybrid claimed success=True, R2~1.0; "
          "actual routed sub-method failed) ===")
    for r in masked:
        print(f"  {r['equation_id']:<45s} decision={r['decision']:<11s} "
              f"sub={r['sub_method']:<15s} raw_test_r2={r['sub_test_r2_raw']}")
    print()

    print("=== Suggested corrected tab:main_results row for HypatiaX ===")
    print(f"HypatiaX & {stats.median(corrected_values):.4f} & "
          f"{summary['mean_r2']:+.4f} & {summary['pct_gt_0.99']:.1f} & "
          f"{summary['pct_gt_0.9']:.1f} & {len(catastrophic_raw)} \\\\")
    print()

    # --- Full drop-in LaTeX section replacement ---
    corrected_tex = generate_corrected_tex(summary, len(catastrophic_raw), len(masked))

    print("=" * 78)
    print("FULL REPLACEMENT for jmlr_paper_main.tex, subsec:overall-extrapolation-")
    print("performance-defi-v / tab:main_results (l. 1261-1285)")
    print("=" * 78)
    print()
    print("--- ORIGINAL (buggy, to be deleted) ---")
    print(ORIGINAL_TEX_TABLE)
    print("--- REPLACEMENT (corrected) ---")
    print(corrected_tex)

    out_path = "tab_main_results_corrected.tex"
    with open(out_path, "w") as f:
        f.write(corrected_tex)
    print(f"[Replacement section written to {out_path}]")

    return rows, summary, corrected_tex


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "hypatiax_defi_benchmark_v3_results_seed42.json"
    main(path)

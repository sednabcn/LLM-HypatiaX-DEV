# Issue 11 — Model/Temperature Mismatch Across Documents

**Category:** [ALREADY RESOLVED — doc propagation only] — root cause already found; only stale doc entries remain

**Location:** `supp_benchmark_report.tex`, reproducibility tables l. 1045–1046 and l. 1243/1261, and `supp_routing_improvements.tex`, "Hardware and Software" l. 756, vs. `jmlr_paper_main.tex`, Nguyen-12 reproducibility note, l. 2606–2637.

## What is wrong
`jmlr_paper_main.tex` already contains the corrected finding: the model that actually produced the Nguyen-12 results is `claude-sonnet-5` at `temperature=0.25`, via `hypatia.py` — confirmed as a fourth, distinct hardcoded string, with the `LLM_MODEL` environment variable and `repro.yaml`'s `llm_model` field both confirmed to be dead code that nothing reads. Three stale locations in the supplementary files still report the superseded value:

- `supp_benchmark_report.tex:1045` — "Anthropic SDK 0.73.0 (claude-sonnet-4-20250514, temperature 0.0)"
- `supp_benchmark_report.tex:1243, 1261` — "Model: claude-sonnet-4-20250514 — LLM temperature: 0.3"
- `supp_routing_improvements.tex:756` — "Anthropic SDK 0.73.0 — claude-sonnet-4-20250514, temp 0.0"

## Fix required
Overwrite all three stale entries with the confirmed value (`claude-sonnet-5`, `temperature=0.25`). No further investigation needed — this is doc propagation only.

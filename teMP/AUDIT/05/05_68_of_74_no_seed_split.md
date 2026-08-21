# Issue 5 — 68 of 74 Cited Without Seed/Split

**Category:** [TEXT — pure writing fix] — pure writing/citation fix, underlying data is consistent

**Location:** `jmlr_paper_main.tex`, Abstract l. 222–223 and `subsec:fivestage-routing-architecture-overview` l. 10–11, referencing Table `tab:timing_full` (`subsec:runtime-analysis`).

## What is wrong
Both the abstract and `subsec:fivestage-routing-architecture-overview` cite "68 of 74" LLM-routed tasks with no seed or split attached:

> A previously reported 1.73× median speedup of HypatiaX over neural-network inference on LLM-routed cases (68 of 74) does not reproduce...

> Stage 3 (optional): LLM proposes candidate expressions; on the v3.0 DeFi benchmark, 68 of 74 tasks are routed to this path.

But Table `tab:timing_full` shows two different rows with exactly 68/74 LLM-routed tasks — v3c seed123 (68/74) and PCA seed42 (68/74) — which are different seeds under different splits with different absolute runtimes. "68 of 74", as written, does not identify which of the two.

## Fix required
Attach the specific seed and split (e.g. "68 of 74 under v3c seed 123" or "... PCA seed 42") wherever "68 of 74" is cited. No table value needs to change.

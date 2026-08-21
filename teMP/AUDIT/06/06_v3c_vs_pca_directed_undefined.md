# Issue 6 — v3c vs. PCA-directed Never Defined as Two Variants

**Category:** [TEXT — pure writing fix] — pure writing fix, both variants' numbers are real

**Location:** `jmlr_paper_main.tex`, `subsec:extrapolation-protocol` (`sec:split`, l. 678–709) vs. Table `tab:timing_full` / Table `tab:timing_llm_routed_full` captions (l. 1630–1636).

## What is wrong
`subsec:extrapolation-protocol` ("Extrapolation Protocol") defines exactly one split methodology:

> To rigorously assess out-of-distribution generalisation, we employ a PCA-directed aggressive extrapolation split: ...

No "v3c" variant is introduced or distinguished anywhere in this section. Yet Table `tab:timing_full`'s caption (and every runtime table downstream) treats "v3c" and "PCA" as two distinct, sibling splits:

> Rows are grouped by split (v3c = aggressive 40/60; PCA = PCA-directed 40/60)...

A reader following `subsec:extrapolation-protocol` alone has no way to learn that "v3c" and "PCA-directed" name two different, non-identical protocols rather than being two names for the same one.

## Fix required
Add an explicit definition of both variants (what specifically distinguishes v3c's aggressive routing from the PCA-directed routing already described) to `subsec:extrapolation-protocol`, since the underlying runs for both are real.

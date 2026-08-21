# Issue 2 — Supp. Table 4 (30/30) vs. Abstract/Conclusion (27/30) on HyperSymLoop (M4)

**Category:** [STALE — reconciliation only] — classic same-run desync, no code investigation needed

**Location:** `supp_benchmark_report.tex`, `tab:overall` ("Six-Method Aggregate Performance", l. 394–423) vs. its own Abstract (l. 140–141).

## What is wrong
The Abstract states:

> HyperSymLoop achieves 100% recovery at all noisy conditions (σ > 0) and 90.0% (27/30 equations) under the strict noiseless protocol (R² > 0.9999).

But `tab:overall`, in the same document, prints:

> HyperSymLoop (M4) Core — 30/30 (v2)† — 0.999+ — 11.4 s

with a footnote reading "HyperSymLoop v1 showed 26/30 due to the Newton measurement bug; v2 achieves 30/30." The table's 30/30 and the abstract's 27/30 are both presented as the current (v2, corrected) figure for the same method on the same noiseless protocol, in the same document.

## Fix required
Regenerate `tab:overall`'s HyperSymLoop row from whichever run actually produced the 27/30 figure quoted in the abstract, and confirm both numbers come from the same run before republishing either.

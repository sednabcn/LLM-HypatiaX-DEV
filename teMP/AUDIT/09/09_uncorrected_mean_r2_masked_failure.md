# Issue 9 — Uncorrected Mean R² (+0.8721) Despite Disclosed −141,000 Masked Failure

**Category:** [OPEN — needs code/data investigation] — pending, not yet executed against a full live re-run

**Location:** `jmlr_paper_main.tex`, `subsec:overall-extrapolation-performance-defi-`, `tab:main_results`, l. 1261–1285, and the Abstract's masking-disclosure footnote, l. 209–215.

## What is wrong
The abstract discloses that HypatiaX's own success accounting silently masks catastrophic sub-method failures:

> The underlying computation is not free of catastrophic failures: 22 of the 74 tasks route to a sub-method (typically the LLM proposal) that itself failed catastrophically (e.g. pure_llm.test_r2 ≈ −141,000 on Loan-to-Value), but the hybrid system's own accounting reports success=True, R² ≈ 1.0 regardless. The catastrophic failure is masked, not eliminated.

Yet `tab:main_results`, whose own caption acknowledges the 90.5% pass-rate figure is uncorrected for this exact bug, still reports HypatiaX's Mean R² as +0.8721 — computed, like the pass rate, from the same masked success/R² fields that report ≈ 1.0 for the 22 tasks whose true sub-method result is catastrophic:

> HypatiaX — 1.0000 — +0.8721 — 90.5 — 90.5 — 0

The caption corrects the pass-rate columns ("the corrected overall near-perfect rate is 60.8%") but is silent on the Mean R² column, which is left standing as if unaffected by the same bug.

## Fix required
Recompute Mean R² from the corrected (decision-attribution-fixed) full live run described in `sec:hybrid-attribution-bug`; this is explicitly stated elsewhere as not yet executed, so the corrected number cannot be hand-estimated from the single disclosed −141,000 outlier alone.

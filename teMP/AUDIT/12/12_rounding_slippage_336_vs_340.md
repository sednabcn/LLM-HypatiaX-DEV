# Issue 12 — Rounding Slippage: 1.21/0.36 = 3.36×, Printed 3.40×

**Category:** [OPEN — needs code/data investigation] — narrow, mechanical script fix + regen

**Location:** `jmlr_paper_main.tex`, `subsec:runtime-analysis`, Table `tab:timing_full`, pooled PCA row, l. 1636.

## What is wrong
The pooled PCA row of Table `tab:timing_full` reads:

> PCA ALL SEEDS (pooled) — 370 — 2.41 / 0.18 — 0.36 / 0.35 — 1.21 / 0.89 — 94/370 — 3.40× slower

Dividing the printed Hybrid mean (1.21 s) by the printed Neural MLP mean (0.36 s) gives 1.21/0.36 ≈ 3.361×, not the printed 3.40×. (By contrast, every v3c-split row's printed speedup is internally consistent with its own printed mean columns.)

## Fix required
`generate_table1.py` (already named in the paper as the regeneration script) is almost certainly dividing pre-rounded display values instead of full-precision ones; needs a one-line fix plus a full table regeneration.

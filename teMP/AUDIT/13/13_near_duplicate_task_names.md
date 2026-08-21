# Issue 13 — Near-Duplicate Task Names Across Difficulty Tiers

**Category:** [TEXT — pure writing fix] — naming-clarity issue, not a data bug

**Location:** `jmlr_paper_main.tex`, `sec:hybrid-attribution-bug` fabricated-success task list, l. 1540–1550, vs. `subsec:lending-protocols` Lending Protocols task inventory, l. 2960–2977.

## What is wrong
The Lending Protocols task inventory lists two visually near-identical pairs of tasks as distinct benchmark items:

> Liquidation Price Long and Liquidation Price Short: prices at which a position is liquidated. ...Liquidation price for leveraged long and leveraged short: extended formulas accounting for funding rates.

Both pairs then appear, split across different difficulty tiers, in the fabricated-success task list:

> Medium (10): ...Liquidation Price Long, Liquidation Price Short, ...
> Hard (9): ...Liquidation price for leveraged long, Liquidation price for leveraged short, ...

A reader skimming either list alone could easily mistake "Liquidation Price Long" (Medium) for "Liquidation price for leveraged long" (Hard), or assume a duplicate/copy-paste error, when in fact these name four formulas that are genuinely distinct (the leveraged variants explicitly account for funding rates).

## Fix required
Rename one pair for clarity (e.g. "Liquidation Price, Basic Long/Short" vs. "Liquidation Price, Leveraged Long/Short (funding-adjusted)") so the two are visually distinguishable wherever they are listed. No numeric or code change needed.

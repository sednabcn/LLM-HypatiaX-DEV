# Issue 8 — Figure 3 Caption DeFi AMM (Hard) = −78.04 Untraceable, Plus Impermanent-Loss Tier Clash

**Category:** [OPEN — needs code/data investigation] — confirmed known-bad figure/caption, fix source already identified

**Location:** `jmlr_paper_main.tex`, Figure `fig:r2_heatmap_clipped` caption (`fig18_r2_heatmap_improved`, l. 1287–1300) vs. Table `tab:llm_ablation` (l. 1871–1897) vs. the Medium-tier definition (l. 632–634).

## What is wrong
The Figure 3 caption reads:

> HypatiaX is not uniformly better than PySR-only: it improves on Chemistry (Easy: 0.90 vs. −12.56) but produces −∞ — a genuine evaluation crash, not merely a poor fit — on DeFi AMM (Hard) and Physics (Hard), where PySR-only instead returns a finite (if poor, −78.04, or perfect, 1.00) value.

Cross-checking against Table `tab:llm_ablation` (explicitly stated to underlie this figure): no DeFi AMM equation shows PySR-only = −78.04 at any range (Constant Product P: 0.9982–0.9996; Impermanent Loss P: 0.7703 near, −2.7832 med, −157.0776 far; Price Impact P: all ≈ 1.0). The −∞ HypatiaX crash described for the DeFi AMM row also does not appear in the DeFi AMM rows of Table `tab:llm_ablation` (all DeFi AMM H values are finite, if very large-magnitude negative); the only −∞ H value in the whole table belongs to Gravitational Force, which is Physics, not DeFi AMM.

Separately, the caption implies a DeFi AMM task sits in the Hard tier, but the difficulty-tier definition explicitly lists "Impermanent Loss" — the DeFi AMM equation actually exhibiting catastrophic values in Table `tab:llm_ablation` — as a Medium-tier example:

> Medium (n = 29): nonlinear algebraic or exponential relationships, such as impermanent loss, Uniswap V3 liquidity range, constant-product price impact...

## Fix required
Regenerate Figure 3 from the same `_merged.json` data used to rebuild Table `tab:llm_ablation`, rather than whatever process produced the caption's −78.04/DeFi-AMM-Hard claims, and resolve whether the Impermanent-Loss-class task belongs to Medium or Hard consistently across the tier definition and the figure.

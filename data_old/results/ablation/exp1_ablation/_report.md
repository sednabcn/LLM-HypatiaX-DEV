
# HypatiaX Analysis Report — `exp1_ablation` (RF09 Feynman n=30)

Experiment mode: **ablation** | N equations: 20
Tier-1 (all-N) pairs: 10 | Tier-2 (excl-train-fail) pairs: 10 | Tier-3 (extrap R²≥0.99) pairs: 5 | Skipped: 10

## ✅ No Fatal Conditions


## ℹ️ Informational / Warnings

- INFO_MW_ALL_NOT_SIGNIFICANT: Tier-1 (all-N) Mann-Whitney one-sided p=0.8811 (two-sided p=0.2694, r=0.3, n=10) — directional but not significant. Expected: 21 discovery failures add noise. Report Tier-3 success-subset as primary claim. Workflow continues.
- WARN_MW_SUCCESS_NOT_SIGNIFICANT: Tier-3 (success-subset) Mann-Whitney one-sided p=0.4571 (n=5) — not significant at α=0.05. Primary paper claim (§10.7) may be weaker than expected. Investigate.

## A. Primary Result — Three-Tier MW Framing (§10.7)

**Tier 1 (all-N):** Expected non-significant — 21 discovery failures add variance. Report with explicit framing: 'not significant; expected given 21 failures.' 

**Tier 2 (excl-train-fail):** Excludes equations where HypatiaX train R²<0. Intermediate result; shows signal strengthens once degenerate outputs removed. 

**Tier 3 (success-subset, R²≥0.99):** The paper's primary claim (§10.7). Restricts to equations where HypatiaX achieved symbolic recovery. This is the publishable result — it answers whether symbolic recovery produces a qualitatively different extrapolation regime, not whether HypatiaX always wins.

  Tier 1 — All-N: U=35.0, p_one=0.8811, p_two=0.2694, n=10, r=0.3
  Tier 2 — Excl-train-fail (train R²≥0): U=35.0, p_one=0.8811, p_two=0.2694, n=10, r=0.3
  Tier 3 — Success-subset (extrap R²≥0.99) ★: U=13.5, p_one=0.4571, p_two=0.9142, n=5, r=-0.08
_** = p_one < 0.05  |  ★ = primary paper claim_

### Win / Loss by Tier

| Split | HypatiaX wins | PySR wins | Tied | N pairs |
|-------|---------------|-----------|------|---------|
| Tier 1 — All-N | 2 | 7 | 1 | 10 |
| Tier 2 — Excl-train-fail | 2 | 7 | 1 | 10 |
| Tier 3 — Success-subset ★ | 2 | 2 | 1 | 5 |

## B. Failure Analysis (0 equations — degenerate PySR, train R² < 0)

_None — all equations have hypatia train R² ≥ 0._

### Domain Stratification

| Domain | N | Hypatia Wins | Win Rate | Failures | Fail Rate |
|--------|---|-------------|----------|----------|-----------|
| Biology | 4 | 0 | 0.0 | 0 | 0.0 |
| Chemistry | 4 | 0 | 0.0 | 0 | 0.0 |
| DeFi AMM | 4 | 0 | 0.0 | 0 | 0.0 |
| DeFi Risk | 4 | 1 | 0.3333 | 0 | 0.0 |
| Physics | 4 | 1 | 0.5 | 0 | 0.0 |

### Fisher's Exact Test — Failure Cluster Non-Randomness

p=1.0000, OR=None, Not significant
Tests whether the failure cluster in physics-with-small-constants domains is larger than expected by chance.

## C. Scale / Magnitude Sensitivity

Spearman correlation between `scale_log` (log₁₀ of smallest constant magnitude) and HypatiaX performance. Positive ρ means larger-scale constants → better results.
  scale_log vs train R²: ρ=-0.433, p=0.1069, n=15
  scale_log vs far R²: ρ=nan, p=nan, n=10
scale_log available for 15 equations.
_** = p < 0.05. N/A if scale_log field absent from records._

## D. Expression Complexity — Success vs Failure

| Group | N | Min | Max | Mean | Median | IQR |
|-------|---|-----|-----|------|--------|-----|
| HypatiaX successes | 0 | N/A | N/A | N/A | N/A | N/A |
| HypatiaX failures | 0 | N/A | N/A | N/A | N/A | N/A |
| HypatiaX all | 15 | 13 | 193 | 121.6 | 143 | 126–152 |
| PySR-only all | 15 | 3 | 14 | 7.7 | 7 | 7–8 |
_** = p < 0.05_

## F. Train-R² Threshold Sweep — Robustness of Inclusion Cutoff

MW p_one at each train-R² inclusion threshold. A robust result stays significant across a range near 0.
| Threshold | N included | U | p_one | p_two | Significant? |
|-----------|------------|---|-------|-------|--------------|
| -0.50 | 10 | 35.0 | 0.8811 | 0.2694 | — |
| -0.25 | 10 | 35.0 | 0.8811 | 0.2694 | — |
| +0.00 | 10 | 35.0 | 0.8811 | 0.2694 | — |
| +0.10 | 10 | 35.0 | 0.8811 | 0.2694 | — |
| +0.25 | 10 | 35.0 | 0.8811 | 0.2694 | — |
| +0.50 | 10 | 35.0 | 0.8811 | 0.2694 | — |

## G. Leave-One-Out Sensitivity — Failure Equations

All-N MW re-run with each failure equation removed. Shows how much each discovery failure masks the signal.
_No LOO data (no failure equations or scipy unavailable)._

## Skipped from MW (10 equations)

| Equation | Domain | Reason |
|----------|--------|--------|
| Biology | Biology | hypatia.extrap_r2_far is None |
| Chemistry | Chemistry | hypatia.extrap_r2_far is None |
| DeFi AMM | DeFi AMM | hypatia.extrap_r2_far is None |
| DeFi Risk | DeFi Risk | hypatia.extrap_r2_far is None |
| Gravitational Force | Physics | hypatia.extrap_r2_far=-inf is non-finite |
| Henderson-Hasselbalch | Chemistry | hypatia.extrap_r2_far is None |
| Logistic Growth | Biology | hypatia.extrap_r2_far is None |
| Michaelis-Menten | Biology | hypatia.extrap_r2_far is None |
| Physics | Physics | hypatia.extrap_r2_far is None |
| Rate Law | Chemistry | hypatia.extrap_r2_far is None |

## Instability Index (1 − extrap_r2_far; None→0.0; unclamped)

| Equation | Domain | Near R² | Far R² | Instability | Skipped? |
|----------|--------|---------|--------|-------------|----------|
| Allometric Scaling | Biology | 0.9824 | -0.0644 | 1.0644 | no |
| Arrhenius | Chemistry | -1.9467 | -853975683890957312.0000 | 853975683890957312.0000 | no |
| Biology | Biology | N/A | 0.0000 | 0.0000 | yes |
| Chemistry | Chemistry | N/A | 0.0000 | 0.0000 | yes |
| Constant Product | DeFi AMM | -18071.4975 | -18519587853051914713806077427591725144592089088.0000 | 18519587853051914713806077427591725144592089088.0000 | no |
| DeFi AMM | DeFi AMM | N/A | 0.0000 | 0.0000 | yes |
| DeFi Risk | DeFi Risk | N/A | 0.0000 | 0.0000 | yes |
| Gravitational Force | Physics | N/A | 0.0000 | 0.0000 | yes |
| Henderson-Hasselbalch | Chemistry | N/A | 0.0000 | 0.0000 | yes |
| Ideal Gas Law | Physics | 0.9988 | 0.9993 | 0.0007 | no |
| Impermanent Loss | DeFi AMM | 0.9639 | -8259.3079 | 8260.3079 | no |
| Kinetic Energy | Physics | 1.0000 | 1.0000 | 0.0000 | no |
| Liquidation Price | DeFi Risk | 0.9999 | 0.9994 | 0.0006 | no |
| Logistic Growth | Biology | N/A | 0.0000 | 0.0000 | yes |
| Michaelis-Menten | Biology | N/A | 0.0000 | 0.0000 | yes |
| Physics | Physics | N/A | 0.0000 | 0.0000 | yes |
| Portfolio Std Dev | DeFi Risk | 0.9993 | 0.7303 | 0.2697 | no |
| Price Impact | DeFi AMM | 1.0000 | 1.0000 | 0.0000 | no |
| Rate Law | Chemistry | N/A | 0.0000 | 0.0000 | yes |
| Value at Risk | DeFi Risk | 1.0000 | 1.0000 | 0.0000 | no |

## Wall-clock Timing

| Method | Mean (s) | Median (s) | N |
|--------|----------|------------|---|
| HypatiaX | 299.6761 | 377.2088 | 15 |
| PySR-only | 1080.4040 | 1101.1529 | 15 |

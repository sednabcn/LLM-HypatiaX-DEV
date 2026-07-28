
# HypatiaX Analysis Report — `exp1_ablation` (RF09 Feynman n=30)

Experiment mode: **ablation** | N equations: 15
Tier-1 (all-N) pairs: 10 | Tier-2 (excl-train-fail) pairs: 10 | Tier-3 (extrap R²≥0.99) pairs: 6 | Skipped: 5

## ✅ No Fatal Conditions


## ℹ️ Informational / Warnings

- INFO_MW_ALL_NOT_SIGNIFICANT: Tier-1 (all-N) Mann-Whitney one-sided p=0.6907 (two-sided p=0.6735, r=0.12, n=10) — directional but not significant. Expected: 21 discovery failures add noise. Report Tier-3 success-subset as primary claim. Workflow continues.
- WARN_MW_SUCCESS_NOT_SIGNIFICANT: Tier-3 (success-subset) Mann-Whitney one-sided p=0.3988 (n=6) — not significant at α=0.05. Primary paper claim (§10.7) may be weaker than expected. Investigate.

## A. Primary Result — Three-Tier MW Framing (§10.7)

**Tier 1 (all-N):** Expected non-significant — 21 discovery failures add variance. Report with explicit framing: 'not significant; expected given 21 failures.' 

**Tier 2 (excl-train-fail):** Excludes equations where HypatiaX train R²<0. Intermediate result; shows signal strengthens once degenerate outputs removed. 

**Tier 3 (success-subset, R²≥0.99):** The paper's primary claim (§10.7). Restricts to equations where HypatiaX achieved symbolic recovery. This is the publishable result — it answers whether symbolic recovery produces a qualitatively different extrapolation regime, not whether HypatiaX always wins.

  Tier 1 — All-N: U=44.0, p_one=0.6907, p_two=0.6735, n=10, r=0.12
  Tier 2 — Excl-train-fail (train R²≥0): U=44.0, p_one=0.6907, p_two=0.6735, n=10, r=0.12
  Tier 3 — Success-subset (extrap R²≥0.99) ★: U=20.0, p_one=0.3988, p_two=0.7976, n=6, r=-0.1111
_** = p_one < 0.05  |  ★ = primary paper claim_

### Win / Loss by Tier

| Split | HypatiaX wins | PySR wins | Tied | N pairs |
|-------|---------------|-----------|------|---------|
| Tier 1 — All-N | 3 | 5 | 2 | 10 |
| Tier 2 — Excl-train-fail | 3 | 5 | 2 | 10 |
| Tier 3 — Success-subset ★ | 2 | 2 | 2 | 6 |

## B. Failure Analysis (0 equations — degenerate PySR, train R² < 0)

_None — all equations have hypatia train R² ≥ 0._

### Domain Stratification

| Domain | N | Hypatia Wins | Win Rate | Failures | Fail Rate |
|--------|---|-------------|----------|----------|-----------|
| Biology | 3 | 0 | 0.0 | 0 | 0.0 |
| Chemistry | 3 | 1 | 1.0 | 0 | 0.0 |
| DeFi AMM | 3 | 0 | 0.0 | 0 | 0.0 |
| DeFi Risk | 3 | 1 | 0.3333 | 0 | 0.0 |
| Physics | 3 | 1 | 0.5 | 0 | 0.0 |

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
| HypatiaX all | 15 | 13 | 173 | 116.0 | 143 | 80–155 |
| PySR-only all | 15 | 3 | 14 | 7.5 | 7 | 7–8 |
_** = p < 0.05_

## F. Train-R² Threshold Sweep — Robustness of Inclusion Cutoff

MW p_one at each train-R² inclusion threshold. A robust result stays significant across a range near 0.
| Threshold | N included | U | p_one | p_two | Significant? |
|-----------|------------|---|-------|-------|--------------|
| -0.50 | 10 | 44.0 | 0.6907 | 0.6735 | — |
| -0.25 | 10 | 44.0 | 0.6907 | 0.6735 | — |
| +0.00 | 10 | 44.0 | 0.6907 | 0.6735 | — |
| +0.10 | 10 | 44.0 | 0.6907 | 0.6735 | — |
| +0.25 | 10 | 44.0 | 0.6907 | 0.6735 | — |
| +0.50 | 10 | 44.0 | 0.6907 | 0.6735 | — |

## G. Leave-One-Out Sensitivity — Failure Equations

All-N MW re-run with each failure equation removed. Shows how much each discovery failure masks the signal.
_No LOO data (no failure equations or scipy unavailable)._

## Skipped from MW (5 equations)

| Equation | Domain | Reason |
|----------|--------|--------|
| ? | Physics | hypatia.extrap_r2_far=-inf is non-finite |
| ? | Chemistry | hypatia.extrap_r2_far is None |
| ? | Biology | hypatia.extrap_r2_far is None |
| ? | Biology | hypatia.extrap_r2_far is None |
| ? | Chemistry | hypatia.extrap_r2_far is None |

## Instability Index (1 − extrap_r2_far; None→0.0; unclamped)

| Equation | Domain | Near R² | Far R² | Instability | Skipped? |
|----------|--------|---------|--------|-------------|----------|
| ? | Biology | 0.9824 | -0.0644 | 1.0644 | no |
| ? | Chemistry | 0.9977 | 0.9028 | 0.0972 | no |
| ? | DeFi AMM | -18071.4975 | -18519587853051914713806077427591725144592089088.0000 | 18519587853051914713806077427591725144592089088.0000 | no |
| ? | Physics | 0.0000 | 0.0000 | 0.0000 | yes |
| ? | Chemistry | 0.0000 | 0.0000 | 0.0000 | yes |
| ? | Physics | 0.9988 | 0.9993 | 0.0007 | no |
| ? | DeFi AMM | 0.9639 | -8259.3079 | 8260.3079 | no |
| ? | Physics | 1.0000 | 1.0000 | 0.0000 | no |
| ? | DeFi Risk | 0.9998 | 0.9993 | 0.0007 | no |
| ? | Biology | 0.0000 | 0.0000 | 0.0000 | yes |
| ? | Biology | 0.0000 | 0.0000 | 0.0000 | yes |
| ? | DeFi Risk | 1.0000 | 1.0000 | 0.0000 | no |
| ? | DeFi AMM | 1.0000 | 1.0000 | 0.0000 | no |
| ? | Chemistry | 0.0000 | 0.0000 | 0.0000 | yes |
| ? | DeFi Risk | 1.0000 | 1.0000 | 0.0000 | no |

## Wall-clock Timing

| Method | Mean (s) | Median (s) | N |
|--------|----------|------------|---|
| HypatiaX | 319.4673 | 395.2167 | 15 |
| PySR-only | 1094.1037 | 1101.1313 | 15 |

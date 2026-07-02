
# HypatiaX Analysis Report — `exp1_ablation` (RF09 Feynman n=30)

Experiment mode: **ablation** | N equations: 3
Tier-1 (all-N) pairs: 2 | Tier-2 (excl-train-fail) pairs: 2 | Tier-3 (extrap R²≥0.99) pairs: 2 | Skipped: 1

## ✅ No Fatal Conditions


## ℹ️ Informational / Warnings

- WARN_TOO_FEW_MW_PAIRS: only 2 finite paired far-R² values (need ≥ 3) for Mann-Whitney test; test skipped. Likely cause: extrap_r2_far absent from records — confirm workers ran the extrapolation evaluation step and that merge_extrap_into_benchmark.py was called before this analysis. Workflow continues.

## A. Primary Result — Three-Tier MW Framing (§10.7)

**Tier 1 (all-N):** Expected non-significant — 21 discovery failures add variance. Report with explicit framing: 'not significant; expected given 21 failures.' 

**Tier 2 (excl-train-fail):** Excludes equations where HypatiaX train R²<0. Intermediate result; shows signal strengthens once degenerate outputs removed. 

**Tier 3 (success-subset, R²≥0.99):** The paper's primary claim (§10.7). Restricts to equations where HypatiaX achieved symbolic recovery. This is the publishable result — it answers whether symbolic recovery produces a qualitatively different extrapolation regime, not whether HypatiaX always wins.

  Tier 1 — All-N: N/A (insufficient pairs after filtering)
  Tier 2 — Excl-train-fail (train R²≥0): N/A (insufficient pairs after filtering)
  Tier 3 — Success-subset (extrap R²≥0.99) ★: N/A (insufficient pairs after filtering)
_** = p_one < 0.05  |  ★ = primary paper claim_

### Win / Loss by Tier

| Split | HypatiaX wins | PySR wins | Tied | N pairs |
|-------|---------------|-----------|------|---------|
| Tier 1 — All-N | 1 | 1 | 0 | 2 |
| Tier 2 — Excl-train-fail | 1 | 1 | 0 | 2 |
| Tier 3 — Success-subset ★ | 1 | 1 | 0 | 2 |

## B. Failure Analysis (0 equations — degenerate PySR, train R² < 0)

_None — all equations have hypatia train R² ≥ 0._

### Domain Stratification

| Domain | N | Hypatia Wins | Win Rate | Failures | Fail Rate |
|--------|---|-------------|----------|----------|-----------|
| Physics | 3 | 1 | 0.5 | 0 | 0.0 |

### Fisher's Exact Test — Failure Cluster Non-Randomness

p=1.0000, OR=None, Not significant
Tests whether the failure cluster in physics-with-small-constants domains is larger than expected by chance.

## C. Scale / Magnitude Sensitivity

Spearman correlation between `scale_log` (log₁₀ of smallest constant magnitude) and HypatiaX performance. Positive ρ means larger-scale constants → better results.
  scale_log vs train R²: ρ=-0.866, p=0.3333, n=3
  scale_log vs far R²: N/A (insufficient data or scipy missing)
scale_log available for 3 equations.
_** = p < 0.05. N/A if scale_log field absent from records._

## D. Expression Complexity — Success vs Failure

| Group | N | Min | Max | Mean | Median | IQR |
|-------|---|-----|-----|------|--------|-----|
| HypatiaX successes | 0 | N/A | N/A | N/A | N/A | N/A |
| HypatiaX failures | 0 | N/A | N/A | N/A | N/A | N/A |
| HypatiaX all | 3 | 14 | 163 | 106.7 | 143 | 78–153 |
| PySR-only all | 3 | 7 | 13 | 9.0 | 7 | 7–10 |
_** = p < 0.05_

## F. Train-R² Threshold Sweep — Robustness of Inclusion Cutoff

MW p_one at each train-R² inclusion threshold. A robust result stays significant across a range near 0.
| Threshold | N included | U | p_one | p_two | Significant? |
|-----------|------------|---|-------|-------|--------------|
| -0.50 | 2 | 2.0 | 0.6667 | 1.0000 | — |
| -0.25 | 2 | 2.0 | 0.6667 | 1.0000 | — |
| +0.00 | 2 | 2.0 | 0.6667 | 1.0000 | — |
| +0.10 | 2 | 2.0 | 0.6667 | 1.0000 | — |
| +0.25 | 2 | 2.0 | 0.6667 | 1.0000 | — |
| +0.50 | 2 | 2.0 | 0.6667 | 1.0000 | — |

## G. Leave-One-Out Sensitivity — Failure Equations

All-N MW re-run with each failure equation removed. Shows how much each discovery failure masks the signal.
_No LOO data (no failure equations or scipy unavailable)._

## Skipped from MW (1 equations)

| Equation | Domain | Reason |
|----------|--------|--------|
| ? | Physics | hypatia.extrap_r2_far=-inf is non-finite |

## Instability Index (1 − extrap_r2_far; None→0.0; unclamped)

| Equation | Domain | Near R² | Far R² | Instability | Skipped? |
|----------|--------|---------|--------|-------------|----------|
| ? | Physics | 0.0000 | 0.0000 | 0.0000 | yes |
| ? | Physics | 0.9988 | 0.9993 | 0.0007 | no |
| ? | Physics | 1.0000 | 1.0000 | 0.0000 | no |

## Wall-clock Timing

| Method | Mean (s) | Median (s) | N |
|--------|----------|------------|---|
| HypatiaX | 243.1146 | 207.8653 | 3 |
| PySR-only | 1106.0717 | 1101.1216 | 3 |

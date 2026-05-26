
# HypatiaX Analysis Report — `exp2_feynman` (RF09 Feynman n=30)

Experiment mode: **ablation** | N equations: 1166
Tier-1 (all-N) pairs: 0 | Tier-2 (excl-train-fail) pairs: 0 | Tier-3 (extrap R²≥0.99) pairs: 0 | Skipped: 0

## ⚠️ Fatal Conditions

- **WRONG_SCHEMA_FOR_ABLATION: exp2_feynman requires paired extrapolation records with hypatia.extrap_r2_far and pysr_only.extrap_r2_far, but the committed results are flat per-method benchmark records (keys: ['domain', 'formula', 'method', 'r2', 'rmse', 'runtime', 'success', 'test']). The workers must rerun with the extrapolation evaluation step enabled. See run_analysis.py EXPERIMENT_MODE['ablation'] docstring for the required record schema.**

## A. Primary Result — Three-Tier MW Framing (§10.7)

**Tier 1 (all-N):** Expected non-significant — 21 discovery failures add variance. Report with explicit framing: 'not significant; expected given 21 failures.' 

**Tier 2 (excl-train-fail):** Excludes equations where HypatiaX train R²<0. Intermediate result; shows signal strengthens once degenerate outputs removed. 

**Tier 3 (success-subset, R²≥0.99):** The paper's primary claim (§10.7). Restricts to equations where HypatiaX achieved symbolic recovery. This is the publishable result — it answers whether symbolic recovery produces a qualitatively different extrapolation regime, not whether HypatiaX always wins.

  Tier 1 — All-N: N/A (early exit)
  Tier 2 — Excl-train-fail (train R²≥0): N/A (early exit)
  Tier 3 — Success-subset (extrap R²≥0.99) ★: N/A (early exit)
_** = p_one < 0.05  |  ★ = primary paper claim_

### Win / Loss by Tier

| Split | HypatiaX wins | PySR wins | Tied | N pairs |
|-------|---------------|-----------|------|---------|
| Tier 1 — All-N | 0 | 0 | 0 | 0 |
| Tier 2 — Excl-train-fail | 0 | 0 | 0 | 0 |
| Tier 3 — Success-subset ★ | 0 | 0 | 0 | 0 |

## B. Failure Analysis (0 equations — degenerate PySR, train R² < 0)

_None — all equations have hypatia train R² ≥ 0._

### Domain Stratification


### Fisher's Exact Test — Failure Cluster Non-Randomness

N/A (early exit)

## C. Scale / Magnitude Sensitivity

Spearman correlation between `scale_log` (log₁₀ of smallest constant magnitude) and HypatiaX performance. Positive ρ means larger-scale constants → better results.
  scale_log vs train R²: N/A (early exit)
  scale_log vs far R²: N/A (early exit)
scale_log available for 0 equations.
_** = p < 0.05. N/A if scale_log field absent from records._

## D. Expression Complexity — Success vs Failure

| Group | N | Min | Max | Mean | Median | IQR |
|-------|---|-----|-----|------|--------|-----|
| HypatiaX successes | 0 | N/A | N/A | N/A | N/A | N/A |
| HypatiaX failures | 0 | N/A | N/A | N/A | N/A | N/A |
| HypatiaX all | 0 | N/A | N/A | N/A | N/A | N/A |
| PySR-only all | 0 | N/A | N/A | N/A | N/A | N/A |
_** = p < 0.05_

## F. Train-R² Threshold Sweep — Robustness of Inclusion Cutoff

MW p_one at each train-R² inclusion threshold. A robust result stays significant across a range near 0.
_No sweep data._

## G. Leave-One-Out Sensitivity — Failure Equations

All-N MW re-run with each failure equation removed. Shows how much each discovery failure masks the signal.
_No LOO data (no failure equations or scipy unavailable)._

## Skipped from MW (0 equations)

_None._

## Instability Index (1 − extrap_r2_far; None→0.0; unclamped)


## Wall-clock Timing

| Method | Mean (s) | Median (s) | N |
|--------|----------|------------|---|
| HypatiaX | N/A | N/A | 0 |
| PySR-only | N/A | N/A | 0 |

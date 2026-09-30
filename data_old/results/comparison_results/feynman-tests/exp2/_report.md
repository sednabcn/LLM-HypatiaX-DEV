
# HypatiaX Analysis Report — `exp2_feynman` (RF09 Feynman n=30)

Experiment mode: **ablation** | N equations: 30
Tier-1 (all-N) pairs: 0 | Tier-2 (excl-train-fail) pairs: 0 | Tier-3 (extrap R²≥0.99) pairs: 0 | Skipped: 30

## ✅ No Fatal Conditions


## ℹ️ Informational / Warnings

- WARN_TOO_FEW_MW_PAIRS: only 0 finite paired far-R² values (need ≥ 3) for Mann-Whitney test; test skipped. Likely cause: extrap_r2_far absent from records — confirm workers ran the extrapolation evaluation step and that merge_extrap_into_benchmark.py was called before this analysis. Workflow continues.

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
| Tier 1 — All-N | 0 | 0 | 0 | 0 |
| Tier 2 — Excl-train-fail | 0 | 0 | 0 | 0 |
| Tier 3 — Success-subset ★ | 0 | 0 | 0 | 0 |

## B. Failure Analysis (0 equations — degenerate PySR, train R² < 0)

_None — all equations have hypatia train R² ≥ 0._

### Domain Stratification

| Domain | N | Hypatia Wins | Win Rate | Failures | Fail Rate |
|--------|---|-------------|----------|----------|-----------|
| feynman_biology | 3 | 0 | N/A | 0 | 0.0 |
| feynman_chemistry | 2 | 0 | N/A | 0 | 0.0 |
| feynman_electrochemistry | 1 | 0 | N/A | 0 | 0.0 |
| feynman_electromagnetism | 5 | 0 | N/A | 0 | 0.0 |
| feynman_electrostatics | 2 | 0 | N/A | 0 | 0.0 |
| feynman_magnetism | 1 | 0 | N/A | 0 | 0.0 |
| feynman_mechanics | 4 | 0 | N/A | 0 | 0.0 |
| feynman_optics | 2 | 0 | N/A | 0 | 0.0 |
| feynman_probability | 1 | 0 | N/A | 0 | 0.0 |
| feynman_quantum | 5 | 0 | N/A | 0 | 0.0 |
| feynman_thermodynamics | 4 | 0 | N/A | 0 | 0.0 |

### Fisher's Exact Test — Failure Cluster Non-Randomness

p=1.0000, OR=None, Not significant
Tests whether the failure cluster in physics-with-small-constants domains is larger than expected by chance.

## C. Scale / Magnitude Sensitivity

Spearman correlation between `scale_log` (log₁₀ of smallest constant magnitude) and HypatiaX performance. Positive ρ means larger-scale constants → better results.
  scale_log vs train R²: N/A (insufficient data or scipy missing)
  scale_log vs far R²: N/A (insufficient data or scipy missing)
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
| Threshold | N included | U | p_one | p_two | Significant? |
|-----------|------------|---|-------|-------|--------------|
| -0.50 | 0 | N/A | N/A | N/A | — |
| -0.25 | 0 | N/A | N/A | N/A | — |
| +0.00 | 0 | N/A | N/A | N/A | — |
| +0.10 | 0 | N/A | N/A | N/A | — |
| +0.25 | 0 | N/A | N/A | N/A | — |
| +0.50 | 0 | N/A | N/A | N/A | — |

## G. Leave-One-Out Sensitivity — Failure Equations

All-N MW re-run with each failure equation removed. Shows how much each discovery failure masks the signal.
_No LOO data (no failure equations or scipy unavailable)._

## Skipped from MW (30 equations)

| Equation | Domain | Reason |
|----------|--------|--------|
| ? | feynman_biology | hypatia.extrap_r2_far is None |
| ? | feynman_biology | hypatia.extrap_r2_far is None |
| ? | feynman_biology | hypatia.extrap_r2_far is None |
| ? | feynman_chemistry | hypatia.extrap_r2_far is None |
| ? | feynman_chemistry | hypatia.extrap_r2_far is None |
| ? | feynman_electrochemistry | hypatia.extrap_r2_far is None |
| ? | feynman_electromagnetism | hypatia.extrap_r2_far is None |
| ? | feynman_electromagnetism | hypatia.extrap_r2_far is None |
| ? | feynman_electromagnetism | hypatia.extrap_r2_far is None |
| ? | feynman_electromagnetism | hypatia.extrap_r2_far is None |
| ? | feynman_electromagnetism | hypatia.extrap_r2_far is None |
| ? | feynman_electrostatics | hypatia.extrap_r2_far is None |
| ? | feynman_electrostatics | hypatia.extrap_r2_far is None |
| ? | feynman_magnetism | hypatia.extrap_r2_far is None |
| ? | feynman_mechanics | hypatia.extrap_r2_far is None |
| ? | feynman_mechanics | hypatia.extrap_r2_far is None |
| ? | feynman_mechanics | hypatia.extrap_r2_far is None |
| ? | feynman_mechanics | hypatia.extrap_r2_far is None |
| ? | feynman_optics | hypatia.extrap_r2_far is None |
| ? | feynman_optics | hypatia.extrap_r2_far is None |
| ? | feynman_probability | hypatia.extrap_r2_far is None |
| ? | feynman_quantum | hypatia.extrap_r2_far is None |
| ? | feynman_quantum | hypatia.extrap_r2_far is None |
| ? | feynman_quantum | hypatia.extrap_r2_far is None |
| ? | feynman_quantum | hypatia.extrap_r2_far is None |
| ? | feynman_quantum | hypatia.extrap_r2_far is None |
| ? | feynman_thermodynamics | hypatia.extrap_r2_far is None |
| ? | feynman_thermodynamics | hypatia.extrap_r2_far is None |
| ? | feynman_thermodynamics | hypatia.extrap_r2_far is None |
| ? | feynman_thermodynamics | hypatia.extrap_r2_far is None |

## Instability Index (1 − extrap_r2_far; None→0.0; unclamped)

| Equation | Domain | Near R² | Far R² | Instability | Skipped? |
|----------|--------|---------|--------|-------------|----------|
| ? | feynman_biology | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_biology | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_biology | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_chemistry | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_chemistry | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_electrochemistry | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_electromagnetism | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_electromagnetism | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_electromagnetism | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_electromagnetism | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_electromagnetism | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_electrostatics | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_electrostatics | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_magnetism | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_mechanics | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_mechanics | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_mechanics | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_mechanics | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_optics | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_optics | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_probability | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_quantum | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_quantum | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_quantum | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_quantum | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_quantum | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_thermodynamics | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_thermodynamics | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_thermodynamics | N/A | 0.0000 | 0.0000 | yes |
| ? | feynman_thermodynamics | N/A | 0.0000 | 0.0000 | yes |

## Wall-clock Timing

| Method | Mean (s) | Median (s) | N |
|--------|----------|------------|---|
| HypatiaX | N/A | N/A | 0 |
| PySR-only | N/A | N/A | 0 |

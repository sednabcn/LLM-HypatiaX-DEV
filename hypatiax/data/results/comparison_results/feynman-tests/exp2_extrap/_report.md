
# HypatiaX Analysis Report — `exp2_feynman_extrap` (RF09 Feynman n=30)

Experiment mode: **ablation** | N equations: 30
Tier-1 (all-N) pairs: 0 | Tier-2 (excl-train-fail) pairs: 0 | Tier-3 (extrap R²≥0.99) pairs: 0 | Skipped: 30

## ⚠️ Fatal Conditions

- **TOO_FEW_MW_PAIRS: only 0 finite paired far-R² values (need ≥ 3) for Mann-Whitney test.**

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
| Michaelis-Menten enzyme kinetics — cross-benchmark consistency check | feynman_biology | pysr_only.extrap_r2_far=nan is non-finite |
| Logistic growth rate — cross-benchmark consistency check | feynman_biology | pysr_only.extrap_r2_far=nan is non-finite |
| Allometric scaling law (metabolic rate vs mass) | feynman_biology | pysr_only.extrap_r2_far=nan is non-finite |
| Arrhenius rate constant (Feynman variant) — cross-benchmark consistency check | feynman_chemistry | hypatia.extrap_r2_far is None |
| Henderson-Hasselbalch equation for buffer pH | feynman_chemistry | hypatia.extrap_r2_far is None |
| Nernst equation for electrode potential — cross-benchmark consistency check | feynman_electrochemistry | hypatia.extrap_r2_far is None |
| Clausius-Mossotti: effective field in dielectric | feynman_electromagnetism | pysr_only.extrap_r2_far=nan is non-finite |
| Dielectric polarisation: P = n * alpha * E (dilute limit) | feynman_electromagnetism | hypatia.extrap_r2_far is None |
| Lorentz force on moving charge in magnetic field: F = qvB | feynman_electromagnetism | hypatia.extrap_r2_far is None |
| Ohm's law: voltage as product of current and resistance | feynman_electromagnetism | pysr_only.extrap_r2_far=nan is non-finite |
| Energy stored in a capacitor: E = 0.5 * C * V^2 | feynman_electromagnetism | pysr_only.extrap_r2_far=nan is non-finite |
| Coulomb force between two point charges (1D, simplified) | feynman_electrostatics | hypatia.extrap_r2_far is None |
| Coulomb's law: electric force between charges | feynman_electrostatics | hypatia.extrap_r2_far is None |
| Curie's law for magnetic susceptibility: chi = C/T | feynman_magnetism | hypatia.extrap_r2_far is None |
| Newton's gravitational force between two masses | feynman_mechanics | hypatia.extrap_r2_far is None |
| Kinetic energy (classical): KE = 0.5 * m * v² | feynman_mechanics | pysr_only.extrap_r2_far=nan is non-finite |
| Reduced mass of a two-body system | feynman_mechanics | pysr_only.extrap_r2_far=nan is non-finite |
| Total mechanical energy: spring potential + kinetic | feynman_mechanics | pysr_only.extrap_r2_far=nan is non-finite |
| Snell's law: refracted angle from incident angle and refractive indices | feynman_optics | hypatia.extrap_r2_far is None |
| Double-slit wave interference intensity | feynman_optics | hypatia.extrap_r2_far is None |
| Gaussian/normal distribution probability density | feynman_probability | hypatia.extrap_r2_far is None |
| Photon energy: E = h * f (Planck relation) | feynman_quantum | hypatia.extrap_r2_far is None |
| Zeeman energy: electron spin in magnetic field | feynman_quantum | pysr_only.extrap_r2_far=nan is non-finite |
| Bose-Einstein occupation number for bosons | feynman_quantum | hypatia.extrap_r2_far is None |
| Fermi-Dirac occupation number for fermions | feynman_quantum | hypatia.extrap_r2_far is None |
| Rabi frequency of two-level atom in magnetic field | feynman_quantum | hypatia.extrap_r2_far is None |
| Planck blackbody spectral radiance (dimensionless: x=hf/kT) | feynman_thermodynamics | pysr_only.extrap_r2_far=nan is non-finite |
| Fourier's law of heat conduction: heat flux across material | feynman_thermodynamics | hypatia.extrap_r2_far is None |
| Stefan-Boltzmann law: blackbody radiated power | feynman_thermodynamics | hypatia.extrap_r2_far is None |
| Ideal gas law: pressure from moles, temperature, volume | feynman_thermodynamics | hypatia.extrap_r2_far is None |

## Instability Index (1 − extrap_r2_far; None→0.0; unclamped)

| Equation | Domain | Near R² | Far R² | Instability | Skipped? |
|----------|--------|---------|--------|-------------|----------|
| Michaelis-Menten enzyme kinetics — cross-benchmark consistency check | feynman_biology | 0.0000 | 1.0000 | 0.0000 | no |
| Logistic growth rate — cross-benchmark consistency check | feynman_biology | 0.0000 | 1.0000 | 0.0000 | no |
| Allometric scaling law (metabolic rate vs mass) | feynman_biology | 0.0000 | 1.0000 | 0.0000 | no |
| Arrhenius rate constant (Feynman variant) — cross-benchmark consistency check | feynman_chemistry | 0.0000 | 0.0000 | 0.0000 | yes |
| Henderson-Hasselbalch equation for buffer pH | feynman_chemistry | 0.0000 | 0.0000 | 0.0000 | yes |
| Nernst equation for electrode potential — cross-benchmark consistency check | feynman_electrochemistry | 0.0000 | 0.0000 | 0.0000 | yes |
| Clausius-Mossotti: effective field in dielectric | feynman_electromagnetism | 0.0000 | 1.0000 | 0.0000 | no |
| Dielectric polarisation: P = n * alpha * E (dilute limit) | feynman_electromagnetism | 0.0000 | 0.0000 | 0.0000 | yes |
| Lorentz force on moving charge in magnetic field: F = qvB | feynman_electromagnetism | 0.0000 | 0.0000 | 0.0000 | yes |
| Ohm's law: voltage as product of current and resistance | feynman_electromagnetism | 0.0000 | 1.0000 | 0.0000 | no |
| Energy stored in a capacitor: E = 0.5 * C * V^2 | feynman_electromagnetism | 0.0000 | 1.0000 | 0.0000 | no |
| Coulomb force between two point charges (1D, simplified) | feynman_electrostatics | 0.0000 | 0.0000 | 0.0000 | yes |
| Coulomb's law: electric force between charges | feynman_electrostatics | 0.0000 | 0.0000 | 0.0000 | yes |
| Curie's law for magnetic susceptibility: chi = C/T | feynman_magnetism | 0.0000 | 0.0000 | 0.0000 | yes |
| Newton's gravitational force between two masses | feynman_mechanics | 0.0000 | 0.0000 | 0.0000 | yes |
| Kinetic energy (classical): KE = 0.5 * m * v² | feynman_mechanics | 0.0000 | 1.0000 | 0.0000 | no |
| Reduced mass of a two-body system | feynman_mechanics | 0.0000 | 1.0000 | 0.0000 | no |
| Total mechanical energy: spring potential + kinetic | feynman_mechanics | 0.0000 | 1.0000 | 0.0000 | no |
| Snell's law: refracted angle from incident angle and refractive indices | feynman_optics | 0.0000 | 0.0000 | 0.0000 | yes |
| Double-slit wave interference intensity | feynman_optics | 0.0000 | 0.0000 | 0.0000 | yes |
| Gaussian/normal distribution probability density | feynman_probability | 0.0000 | 0.0000 | 0.0000 | yes |
| Photon energy: E = h * f (Planck relation) | feynman_quantum | 0.0000 | 0.0000 | 0.0000 | yes |
| Zeeman energy: electron spin in magnetic field | feynman_quantum | 0.0000 | 1.0000 | 0.0000 | no |
| Bose-Einstein occupation number for bosons | feynman_quantum | 0.0000 | 0.0000 | 0.0000 | yes |
| Fermi-Dirac occupation number for fermions | feynman_quantum | 0.0000 | 0.0000 | 0.0000 | yes |
| Rabi frequency of two-level atom in magnetic field | feynman_quantum | 0.0000 | 0.0000 | 0.0000 | yes |
| Planck blackbody spectral radiance (dimensionless: x=hf/kT) | feynman_thermodynamics | 0.0000 | 1.0000 | 0.0000 | no |
| Fourier's law of heat conduction: heat flux across material | feynman_thermodynamics | 0.0000 | 0.0000 | 0.0000 | yes |
| Stefan-Boltzmann law: blackbody radiated power | feynman_thermodynamics | 0.0000 | 0.0000 | 0.0000 | yes |
| Ideal gas law: pressure from moles, temperature, volume | feynman_thermodynamics | 0.0000 | 0.0000 | 0.0000 | yes |

## Wall-clock Timing

| Method | Mean (s) | Median (s) | N |
|--------|----------|------------|---|
| HypatiaX | N/A | N/A | 0 |
| PySR-only | N/A | N/A | 0 |

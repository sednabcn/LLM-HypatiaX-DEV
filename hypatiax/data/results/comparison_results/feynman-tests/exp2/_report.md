
# HypatiaX Analysis Report — `exp2_feynman`

Experiment mode: **standard**
N total: 1166 | N standard: 1166 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 146 | 99.3% | 97.2% | 1.0000 | 0.9085 |
| Neural Net | 146 | 100.0% | 99.3% | 0.9999 | 0.9960 |
| Hybrid | 986 | 100.0% | 99.9% | 1.0000 | 0.9994 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=32342.0,  p=0.0000**,  direction=b_greater,  n=(986, 145)

### Hybrid vs Neural Net

  U=123514.0,  p=0.0000**,  direction=a_greater,  n=(986, 146)

### Neural Net vs Pure LLM

  U=4121.0,  p=0.0000**,  direction=b_greater,  n=(146, 145)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 146
Hybrid wins:  136  (93.2%)
NN wins:      10
Tied:         0

## Coverage Gaps (1022 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| Molecular viscosity from physicochemical props | None | None | N/A | N/A | 0.9986 | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 1.0000 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9999 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 1.0000 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 1.0000 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 1.0000 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9998 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 1.0000 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9999 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 1.0000 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 1.0000 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 1.0000 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 1.0000 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9988 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 1.0000 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 1.0000 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 1.0000 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 1.0000 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9996 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 1.0000 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9988 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 1.0000 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 1.0000 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 1.0000 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 1.0000 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9999 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 1.0000 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 1.0000 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 1.0000 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 1.0000 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 1.0000 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 1.0000 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 1.0000 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 1.0000 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9981 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9995 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9973 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9977 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9977 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9977 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9977 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9976 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9977 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9973 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9977 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9977 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9977 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9977 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9979 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9978 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9971 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9977 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9977 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9977 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9977 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9979 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9978 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9973 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9977 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9977 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9977 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9977 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9980 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9978 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9979 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9975 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9973 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9979 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9983 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9988 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9980 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9973 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9979 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9977 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9981 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9981 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9981 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9981 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9974 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9977 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9977 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9977 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9977 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9980 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9978 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9975 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9976 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9976 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9975 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9975 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9949 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9975 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9970 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9973 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9976 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9971 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9976 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9941 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9976 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9974 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9972 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9977 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9975 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9976 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9975 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9976 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Arrhenius rate constant (Feynman variant) | None | None | 0.7717 | 0.7385 | 0.7555 | 0.7717 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9973 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9977 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9977 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9977 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9977 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9980 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9978 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9973 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9977 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9977 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9977 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9978 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9979 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9978 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9973 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9976 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9977 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9977 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9977 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9980 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9978 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9972 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9977 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9977 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9977 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9977 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9979 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9978 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9973 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9977 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9977 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9977 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9977 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9979 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9978 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9972 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9977 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9977 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9977 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9977 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9980 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9978 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9973 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9977 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9977 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9977 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9977 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9976 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9977 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9972 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9976 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9977 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9977 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9977 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9978 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9978 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9973 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9977 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9977 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9977 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9977 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9980 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9978 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9973 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9977 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9977 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9977 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9977 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9955 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9977 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9972 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9976 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9977 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9977 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9977 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9977 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9978 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9973 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9977 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9977 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9974 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9977 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9980 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9977 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |
| Michaelis-Menten enzyme kinetics | None | None | N/A | N/A | N/A | 0.9973 |
| Logistic growth rate | None | None | N/A | N/A | N/A | 0.9977 |
| Allometric scaling law (metabolic rate vs mass) | None | None | N/A | N/A | N/A | 0.9977 |
| Arrhenius rate constant (Feynman variant) | None | None | N/A | N/A | N/A | 0.9977 |
| Henderson-Hasselbalch equation for buffer pH | None | None | N/A | N/A | N/A | 1.0000 |
| Nernst equation for electrode potential | None | None | N/A | N/A | N/A | 1.0000 |
| Clausius-Mossotti | None | None | N/A | N/A | N/A | 1.0000 |
| Dielectric polarisation | None | None | N/A | N/A | N/A | 1.0000 |
| Lorentz force on moving charge in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Ohm's law | None | None | N/A | N/A | N/A | 1.0000 |
| Energy stored in a capacitor | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb force between two point charges (1D, simplified) | None | None | N/A | N/A | N/A | 1.0000 |
| Coulomb's law | None | None | N/A | N/A | N/A | 1.0000 |
| Curie's law for magnetic susceptibility | None | None | N/A | N/A | N/A | 1.0000 |
| Newton's gravitational force between two masses | None | None | N/A | N/A | N/A | 1.0000 |
| Kinetic energy (classical) | None | None | N/A | N/A | N/A | 1.0000 |
| Reduced mass of a two-body system | None | None | N/A | N/A | N/A | 1.0000 |
| Total mechanical energy | None | None | N/A | N/A | N/A | 1.0000 |
| Snell's law | None | None | N/A | N/A | N/A | 1.0000 |
| Double-slit wave interference intensity | None | None | N/A | N/A | N/A | 1.0000 |
| Gaussian/normal distribution probability density | None | None | N/A | N/A | N/A | 1.0000 |
| Photon energy | None | None | N/A | N/A | N/A | 0.9977 |
| Zeeman energy | None | None | N/A | N/A | N/A | 1.0000 |
| Bose-Einstein occupation number for bosons | None | None | N/A | N/A | N/A | 0.9980 |
| Fermi-Dirac occupation number for fermions | None | None | N/A | N/A | N/A | 1.0000 |
| Rabi frequency of two-level atom in magnetic field | None | None | N/A | N/A | N/A | 1.0000 |
| Planck blackbody spectral radiance (dimensionless | None | None | N/A | N/A | N/A | 0.9978 |
| Fourier's law of heat conduction | None | None | N/A | N/A | N/A | 1.0000 |
| Stefan-Boltzmann law | None | None | N/A | N/A | N/A | 1.0000 |
| Ideal gas law | None | None | N/A | N/A | N/A | 1.0000 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| unknown | 146 | 97.2% | 99.3% | 99.9% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| unknown | 146 | 1.0000 | 0.9999 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | N/A | N/A | 0 |
| Neural Net | N/A | N/A | 0 |
| Hybrid | N/A | N/A | 0 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 6.6190 | 7.8310 | 966.37 | 146 |
| Neural Net | 1.8690 | 1.3346 | 272.88 | 146 |
| Hybrid | 5.5647 | 6.5394 | 5486.8 | 986 |

## Hybrid Routing Decisions

_No hybrid decision data available._

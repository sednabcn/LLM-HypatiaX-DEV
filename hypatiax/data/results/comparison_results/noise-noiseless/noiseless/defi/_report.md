
# HypatiaX Analysis Report — `exp1`

Experiment mode: **standard**
N total: 74 | N standard: 74 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 74 | 60.8% | 80.4% | 1.0000 | -0.3048 |
| Neural Net | 74 | 100.0% | 14.9% | -0.6908 | -1.4155 |
| Hybrid | 74 | 97.3% | 90.5% | 1.0000 | 0.8781 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=2443.0,  p=0.0113**,  direction=b_greater,  n=(74, 56)

### Hybrid vs Neural Net

  U=5284.0,  p=0.0000**,  direction=a_greater,  n=(74, 74)

### Neural Net vs Pure LLM

  U=660.0,  p=0.0000**,  direction=b_greater,  n=(74, 56)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 74
Hybrid wins:  68  (91.9%)
NN wins:      0
Tied:         6

## Coverage Gaps (18 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| Slashing penalty | easy | linear | N/A | N/A | -2.9037 | 1.0000 |
| Liquidation Price Long | medium | rational | N/A | N/A | 0.8624 | 1.0000 |
| Liquidation Price Short | medium | rational | N/A | N/A | -3.3805 | 1.0000 |
| Capital efficiency | medium | rational | N/A | N/A | -2.3826 | 1.0000 |
| Utilization rate of DeFi | medium | rational | N/A | N/A | 0.9061 | 1.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -9.9026 | 0.7709 |
| Collateral ratio | medium | rational | N/A | N/A | 0.9428 | 1.0000 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | N/A | N/A | -1.8315 | -1.8315 |
| Black-Scholes Put Price | hard | norm_cdf | N/A | N/A | -0.6802 | -0.6802 |
| Gamma of option | hard | norm_pdf | N/A | N/A | -0.1060 | -0.1060 |
| Vega of option | hard | norm_pdf | N/A | N/A | 0.6285 | 0.6285 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.2314 | 1.0000 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.4907 | 1.0000 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.6330 | 1.0000 |
| Required collateral | hard | rational | N/A | N/A | -1.2015 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.1273 | 1.0000 |
| Theta of option | hard | norm_pdf | N/A | N/A | -0.8034 | -0.8034 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 24 | 91.3% | 12.5% | 100.0% |
| hard | 21 | 63.6% | 0.0% | 76.2% |
| medium | 29 | 77.3% | 27.6% | 93.1% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 5 | 1.0000 | -2.1870 | 1.0000 |
| algebraic_with_sqrt | 4 | -0.5899 | -0.3526 | 1.0000 |
| exponential | 5 | 1.0000 | -2.3058 | 1.0000 |
| linear | 18 | 1.0000 | -2.3577 | 1.0000 |
| norm_cdf | 3 | N/A | -1.8315 | -0.6802 |
| norm_pdf | 3 | N/A | -0.1060 | -0.1060 |
| piecewise_linear | 1 | -3.5798 | 0.7545 | 1.0000 |
| quadratic_form | 3 | 0.3626 | -2.1273 | 1.0000 |
| rational | 24 | 1.0000 | -0.0396 | 1.0000 |
| rational_simple | 7 | 1.0000 | -0.1295 | 1.0000 |
| weighted_aggregate | 1 | 1.0000 | 0.6311 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | -748660.5169 | 0.0000 | 56 |
| Neural Net | 4.2864 | 1.6713 | 74 |
| Hybrid | -0.1393 | 0.0000 | 72 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 8.6251 | 7.5850 | 638.26 | 74 |
| Neural Net | 0.3559 | 0.3550 | 26.33 | 74 |
| Hybrid | 1.6971 | 1.3230 | 125.58 | 74 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| llm | 68 |
| nn | 5 |
| nn_fallback | 1 |

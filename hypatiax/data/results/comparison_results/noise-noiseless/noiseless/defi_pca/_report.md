
# HypatiaX Analysis Report — `exp1_pca`

Experiment mode: **standard**
N total: 74 | N standard: 74 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 74 | 62.2% | 76.7% | 1.0000 | -0.7521 |
| Neural Net | 74 | 100.0% | 12.2% | -0.6598 | -1.3643 |
| Hybrid | 74 | 93.2% | 89.2% | 1.0000 | 0.8489 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=2734.0,  p=0.0019**,  direction=b_greater,  n=(74, 60)

### Hybrid vs Neural Net

  U=5253.0,  p=0.0000**,  direction=a_greater,  n=(74, 74)

### Neural Net vs Pure LLM

  U=890.0,  p=0.0000**,  direction=b_greater,  n=(74, 60)
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
| Liquidation Price Long | medium | rational | N/A | N/A | -1.4658 | 1.0000 |
| Liquidation Price Short | medium | rational | N/A | N/A | -3.3805 | 1.0000 |
| Capital efficiency | medium | rational | N/A | N/A | -2.3826 | -2.3826 |
| Utilization rate of DeFi | medium | rational | N/A | N/A | 0.9873 | 1.0000 |
| Impermanent loss percentage | medium | algebraic_with_sqrt | N/A | N/A | -2.1889 | 1.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -9.9026 | 0.7709 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | -1.8315 | -193.6432 | -1.8315 | -1.8315 |
| Black-Scholes Put Price | hard | norm_cdf | 0.5415 | -1.3606 | -0.6802 | 0.5415 |
| Vega of option | hard | norm_pdf | 0.6285 | -18.5929 | 0.6285 | 0.6285 |
| Constant product formula (multivariate) | hard | rational | N/A | N/A | 0.2765 | 1.0000 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.2764 | 1.0000 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.4907 | 1.0000 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.6330 | 1.0000 |
| Required collateral | hard | rational | N/A | N/A | -1.2015 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.1273 | 1.0000 |
| Theta of option | hard | norm_pdf | -0.8034 | -16.0271 | -0.8034 | -0.8034 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 24 | 91.3% | 12.5% | 100.0% |
| hard | 21 | 46.7% | 0.0% | 76.2% |
| medium | 29 | 81.8% | 20.7% | 89.7% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 5 | 1.0000 | -2.1870 | 1.0000 |
| algebraic_with_sqrt | 4 | 1.0000 | -0.6669 | 1.0000 |
| exponential | 5 | 1.0000 | -2.3058 | 1.0000 |
| linear | 18 | 1.0000 | -2.3577 | 1.0000 |
| norm_cdf | 3 | -5.6803 | -1.8315 | 0.5415 |
| norm_pdf | 3 | -10.0000 | -0.1060 | -0.1060 |
| piecewise_linear | 1 | -1.3042 | 0.1272 | 1.0000 |
| quadratic_form | 3 | 0.3626 | -2.1273 | 1.0000 |
| rational | 24 | 0.9988 | -0.1071 | 1.0000 |
| rational_simple | 7 | 1.0000 | 0.7069 | 1.0000 |
| weighted_aggregate | 1 | 1.0000 | 0.7085 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | 82739.5045 | 0.0000 | 60 |
| Neural Net | 18.5805 | 1.6402 | 74 |
| Hybrid | -103.4925 | 0.0000 | 74 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 9.5994 | 9.0605 | 710.36 | 74 |
| Neural Net | 0.3341 | 0.3320 | 24.73 | 74 |
| Hybrid | 1.8385 | 1.6230 | 136.05 | 74 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| llm | 68 |
| nn | 5 |
| nn_fallback | 1 |

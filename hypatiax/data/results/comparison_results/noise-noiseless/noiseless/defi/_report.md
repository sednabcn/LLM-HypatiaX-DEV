
# HypatiaX Analysis Report — `exp1`

Experiment mode: **standard**
N total: 74 | N standard: 74 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 74 | 62.2% | 78.0% | 1.0000 | -0.7483 |
| Neural Net | 74 | 100.0% | 14.9% | -0.6908 | -1.4155 |
| Hybrid | 74 | 78.4% | 75.7% | 1.0000 | 0.2593 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=2198.0,  p=0.9379,  direction=b_greater,  n=(74, 59)

### Hybrid vs Neural Net

  U=4725.5,  p=0.0000**,  direction=a_greater,  n=(74, 74)

### Neural Net vs Pure LLM

  U=846.0,  p=0.0000**,  direction=b_greater,  n=(74, 59)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 74
Hybrid wins:  54  (73.0%)
NN wins:      0
Tied:         20

## Coverage Gaps (22 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| Validator commission adjusted | easy | linear | N/A | N/A | -2.5077 | 1.0000 |
| Slashing penalty | easy | linear | -2.9037 | -755.5310 | -2.9037 | -2.9037 |
| Liquidation Price Long | medium | rational | N/A | N/A | 0.8624 | 0.9927 |
| Liquidation Price Short | medium | rational | N/A | N/A | -3.3805 | 0.9832 |
| Capital efficiency | medium | rational | N/A | N/A | -2.3826 | -2.3826 |
| Utilization rate of DeFi | medium | rational | N/A | N/A | 0.9061 | 1.0000 |
| Impermanent loss percentage | medium | algebraic_with_sqrt | N/A | N/A | -2.1889 | 1.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -9.9026 | -9.9026 |
| Collateral ratio | medium | rational | N/A | N/A | 0.9428 | 1.0000 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | -1.8315 | -193.6432 | -1.8315 | -1.8315 |
| Black-Scholes Put Price | hard | norm_cdf | -0.6802 | -1.3606 | -0.6802 | -0.6802 |
| Component ES | hard | quadratic_form | -0.1921 | -0.2749 | -0.1921 | -0.1921 |
| Vega of option | hard | norm_pdf | 0.6285 | -18.5929 | 0.6285 | 0.6285 |
| Impermanent loss in constant product | hard | algebraic_with_sqrt | N/A | N/A | -1.2321 | 1.0000 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.2314 | 0.2314 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.4907 | -1.4907 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.6330 | -1.6330 |
| Required collateral | hard | rational | N/A | N/A | -1.2015 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.1273 | -0.1389 |
| Uniswap V3 virtual | hard | algebraic_with_sqrt | 0.5269 | -9408081.2323 | 0.5269 | 0.5269 |
| Theta of option | hard | norm_pdf | -0.8034 | -16.0271 | -0.8034 | -0.8034 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 24 | 87.0% | 12.5% | 91.7% |
| hard | 21 | 46.7% | 0.0% | 47.6% |
| medium | 29 | 90.5% | 27.6% | 82.8% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 5 | 1.0000 | -2.1870 | 1.0000 |
| algebraic_with_sqrt | 4 | -4.5000 | -0.3526 | 1.0000 |
| exponential | 5 | 1.0000 | -2.3058 | 1.0000 |
| linear | 18 | 1.0000 | -2.3577 | 1.0000 |
| norm_cdf | 3 | -5.6803 | -1.8315 | -1.8315 |
| norm_pdf | 3 | -10.0000 | -0.1060 | -0.1060 |
| piecewise_linear | 1 | -3.5798 | 0.7545 | 1.0000 |
| quadratic_form | 3 | 0.3626 | -2.1273 | -0.1389 |
| rational | 24 | 1.0000 | -0.0396 | 1.0000 |
| rational_simple | 7 | 1.0000 | -0.1295 | 1.0000 |
| weighted_aggregate | 1 | 1.0000 | 0.6311 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | 41985.1096 | 0.0000 | 59 |
| Neural Net | 4.2864 | 1.6713 | 74 |
| Hybrid | -98059.8404 | 0.0000 | 71 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 9.7807 | 9.0080 | 723.77 | 74 |
| Neural Net | 0.2510 | 0.2500 | 18.57 | 74 |
| Hybrid | 2.1062 | 1.4790 | 155.86 | 74 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| llm | 54 |
| nn | 20 |

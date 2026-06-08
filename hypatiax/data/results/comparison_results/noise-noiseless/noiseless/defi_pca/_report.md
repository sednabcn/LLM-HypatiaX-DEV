
# HypatiaX Analysis Report — `exp1_pca`

Experiment mode: **standard**
N total: 74 | N standard: 74 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 74 | 81.1% | 81.4% | 1.0000 | -0.3853 |
| Neural Net | 74 | 87.8% | 15.4% | -0.2493 | -1.2303 |
| Hybrid | 74 | 100.0% | 90.5% | 1.0000 | 0.9074 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=2471.5,  p=0.0478**,  direction=b_greater,  n=(74, 59)

### Hybrid vs Neural Net

  U=4649.0,  p=0.0000**,  direction=a_greater,  n=(74, 65)

### Neural Net vs Pure LLM

  U=606.0,  p=0.0000**,  direction=b_greater,  n=(65, 59)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 65
Hybrid wins:  59  (90.8%)
NN wins:      0
Tied:         6

## Coverage Gaps (17 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| Liquidation Price Long | medium | rational | N/A | N/A | -1.2187 | 1.0000 |
| Liquidation Price Short | medium | rational | N/A | N/A | -2.5432 | 1.0000 |
| Capital efficiency | medium | rational | N/A | N/A | -2.0963 | 1.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -10.3199 | 0.7709 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | 0.2464 | 0.2464 | -0.8981 | -0.8981 |
| Black-Scholes Put Price | hard | norm_cdf | N/A | N/A | 0.0459 | 0.0459 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.2542 | 1.0000 |
| Gamma of option | hard | norm_pdf | N/A | N/A | 0.3433 | 0.3433 |
| Vega of option | hard | norm_pdf | N/A | N/A | 0.6593 | 0.6593 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.4702 | 1.0000 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.4329 | 1.0000 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.3461 | 1.0000 |
| Required collateral | hard | rational | N/A | N/A | -0.7895 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | N/A | 1.0000 |
| Uniswap V3 virtual | hard | algebraic_with_sqrt | N/A | N/A | 0.0745 | 1.0000 |
| Theta of option | hard | norm_pdf | -0.7755 | -15.9373 | -0.7755 | -0.7755 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 24 | 91.7% | 14.3% | 100.0% |
| hard | 21 | 63.6% | 5.6% | 76.2% |
| medium | 29 | 79.2% | 23.1% | 93.1% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 5 | 1.0000 | -1.5028 | 1.0000 |
| algebraic_with_sqrt | 4 | 1.0000 | -0.8081 | 1.0000 |
| exponential | 5 | 1.0000 | -2.1634 | 1.0000 |
| linear | 18 | 1.0000 | -1.3664 | 1.0000 |
| norm_cdf | 3 | 0.2464 | -0.8981 | 0.0459 |
| norm_pdf | 3 | -10.0000 | 0.3433 | 0.3433 |
| piecewise_linear | 1 | -1.3042 | 0.2800 | 1.0000 |
| quadratic_form | 3 | 1.0000 | -0.2542 | 1.0000 |
| rational | 24 | 1.0000 | -0.0533 | 1.0000 |
| rational_simple | 7 | 1.0000 | 0.6471 | 1.0000 |
| weighted_aggregate | 1 | -10.0000 | 0.9774 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | 1150.8522 | 0.0000 | 59 |
| Neural Net | 10.9557 | 1.2492 | 65 |
| Hybrid | -0.1590 | 0.0000 | 72 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 11.5465 | 10.7250 | 854.44 | 74 |
| Neural Net | 0.3500 | 0.3790 | 25.9 | 74 |
| Hybrid | 2.2286 | 1.6445 | 164.91 | 74 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| llm | 68 |
| nn | 5 |
| nn_fallback | 1 |

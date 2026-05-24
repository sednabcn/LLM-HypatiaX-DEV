
# HypatiaX Analysis Report — `exp1`

Experiment mode: **standard**
N total: 74 | N standard: 74 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 74 | 77.0% | 82.5% | 1.0000 | -0.4098 |
| Neural Net | 74 | 87.8% | 20.0% | -0.2412 | -1.0878 |
| Hybrid | 74 | 100.0% | 90.5% | 1.0000 | 0.7618 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=2321.5,  p=0.1233,  direction=b_greater,  n=(74, 57)

### Hybrid vs Neural Net

  U=4584.0,  p=0.0000**,  direction=a_greater,  n=(74, 65)

### Neural Net vs Pure LLM

  U=579.0,  p=0.0000**,  direction=b_greater,  n=(65, 57)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 65
Hybrid wins:  58  (89.2%)
NN wins:      0
Tied:         7

## Coverage Gaps (19 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| Simple Staking APY | easy | linear | N/A | N/A | -0.2493 | 1.0000 |
| Liquidation Price Long | medium | rational | N/A | N/A | 0.8965 | 1.0000 |
| Liquidation Price Short | medium | rational | N/A | N/A | -2.5432 | 1.0000 |
| Capital efficiency | medium | rational | N/A | N/A | -2.0963 | 1.0000 |
| Options Delta | medium | norm_cdf | 0.7136 | 0.7136 | -10.3199 | -10.3199 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | N/A | N/A | -0.8981 | -0.8981 |
| Black-Scholes Put Price | hard | norm_cdf | N/A | N/A | 0.0459 | 0.0459 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.2542 | 1.0000 |
| Gamma of option | hard | norm_pdf | N/A | N/A | 0.3433 | 0.3433 |
| Vega of option | hard | norm_pdf | N/A | N/A | 0.6593 | 0.6593 |
| Convexity Adjustment | hard | algebraic | N/A | N/A | -2.4192 | 1.0000 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.6256 | 1.0000 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.4329 | 1.0000 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.3461 | 1.0000 |
| Required collateral | hard | rational | N/A | N/A | -0.7895 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | N/A | 1.0000 |
| Uniswap V3 virtual | hard | algebraic_with_sqrt | N/A | N/A | 0.7871 | 1.0000 |
| Theta of option | hard | norm_pdf | -0.7755 | -15.9304 | -0.7755 | -0.7755 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 24 | 91.3% | 14.3% | 100.0% |
| hard | 21 | 66.7% | 11.1% | 76.2% |
| medium | 29 | 80.0% | 30.8% | 93.1% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 5 | 1.0000 | -1.5028 | 1.0000 |
| algebraic_with_sqrt | 4 | 1.0000 | -0.4518 | 1.0000 |
| exponential | 5 | 1.0000 | -2.1634 | 1.0000 |
| linear | 18 | 1.0000 | -1.3664 | 1.0000 |
| norm_cdf | 3 | 0.7136 | -0.8981 | -0.8981 |
| norm_pdf | 3 | -10.0000 | 0.3433 | 0.3433 |
| piecewise_linear | 1 | -3.5798 | 0.8679 | 1.0000 |
| quadratic_form | 3 | 1.0000 | -0.2542 | 1.0000 |
| rational | 24 | 1.0000 | -0.0229 | 1.0000 |
| rational_simple | 7 | 1.0000 | 0.5461 | 1.0000 |
| weighted_aggregate | 1 | -10.0000 | 0.9725 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | 14.1570 | 0.0000 | 57 |
| Neural Net | 2.0881 | 1.2411 | 65 |
| Hybrid | -0.0059 | 0.0000 | 72 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 11.4353 | 10.1280 | 846.21 | 74 |
| Neural Net | 0.3194 | 0.3610 | 23.64 | 74 |
| Hybrid | 1.7369 | 1.3230 | 128.53 | 74 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| llm | 67 |
| nn | 6 |
| nn_fallback | 1 |

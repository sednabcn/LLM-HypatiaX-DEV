
# HypatiaX Analysis Report — `exp1b`

Experiment mode: **standard**
N total: 72 | N standard: 72 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 72 | 68.1% | 81.7% | 1.0000 | -0.3865 |
| Neural Net | 72 | 25.0% | 15.3% | -0.4684 | -1.2757 |
| Hybrid | 72 | 86.1% | 83.3% | 1.0000 | 0.2142 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=1986.0,  p=0.3335,  direction=b_greater,  n=(72, 60)

### Hybrid vs Neural Net

  U=4551.5,  p=0.0000**,  direction=a_greater,  n=(72, 72)

### Neural Net vs Pure LLM

  U=680.0,  p=0.0000**,  direction=b_greater,  n=(72, 60)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 72
Hybrid wins:  64  (88.9%)
NN wins:      7
Tied:         1

## Coverage Gaps (15 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| Black-Scholes Call Price | hard | norm_cdf | 0.2464 | 0.2464 | -1.1816 | -0.0696 |
| Black-Scholes Put Price | hard | norm_cdf | 0.5329 | -1.3606 | 0.1913 | 0.5329 |
| Capital efficiency | medium | rational | N/A | N/A | -2.1446 | 1.0000 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.2608 | 0.8854 |
| Liquidation Price Long | medium | rational | N/A | N/A | 0.6053 | 0.9996 |
| Liquidation Price Short | medium | rational | N/A | N/A | -3.1983 | 0.9998 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | -0.0248 | -6.5668 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.7713 | -2.3436 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.7419 | 0.9999 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -10.5222 | -4.2294 |
| Required collateral | hard | rational | N/A | N/A | -1.4092 | 1.0000 |
| Theta of option | hard | norm_pdf | N/A | N/A | -0.6818 | -1.4001 |
| Vega of option | hard | norm_pdf | 0.5988 | -18.5929 | 0.5988 | -16.1621 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.9789 | 0.6624 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 24 | 87.5% | 12.5% | 100.0% |
| hard | 20 | 61.5% | 5.0% | 60.0% |
| medium | 28 | 87.0% | 25.0% | 85.7% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 4 | 1.0000 | -1.4860 | 1.0000 |
| algebraic_with_sqrt | 4 | 1.0000 | -0.3989 | 0.9480 |
| exponential | 5 | 1.0000 | -2.4114 | 1.0000 |
| linear | 18 | 1.0000 | -2.7114 | 1.0000 |
| norm_cdf | 3 | -0.5571 | -1.1816 | -0.0696 |
| norm_pdf | 3 | -4.5017 | 0.4325 | -10.0000 |
| piecewise_linear | 1 | -3.5798 | 0.7504 | 1.0000 |
| quadratic_form | 2 | N/A | -1.6199 | 0.7739 |
| rational | 24 | 1.0000 | -0.0124 | 1.0000 |
| rational_simple | 7 | 1.0000 | -0.1223 | 1.0000 |
| weighted_aggregate | 1 | -10.0000 | 0.9744 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | -108.2933 | 0.0000 | 60 |
| Neural Net | 3.5009 | 1.4681 | 72 |
| Hybrid | 1.1374 | 0.0000 | 72 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 11.1607 | 9.8935 | 803.57 | 72 |
| Neural Net | 0.2494 | 0.2480 | 17.95 | 72 |
| Hybrid | 2.7126 | 2.1820 | 195.31 | 72 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| v4_llm | 45 |
| v4_linear_fallback | 18 |
| v4_residual_nn | 7 |
| v4_linear_fallback_local | 2 |

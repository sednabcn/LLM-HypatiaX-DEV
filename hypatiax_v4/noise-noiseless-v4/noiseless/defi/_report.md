
# HypatiaX Analysis Report — `exp1`

Experiment mode: **standard**
N total: 74 | N standard: 74 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 74 | 67.6% | 80.7% | 1.0000 | -0.5940 |
| Neural Net | 74 | 23.0% | 14.9% | -0.6932 | -1.4148 |
| Hybrid | 74 | 86.5% | 83.8% | 1.0000 | 0.2363 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=2157.5,  p=0.4604,  direction=b_greater,  n=(74, 62)

### Hybrid vs Neural Net

  U=4851.5,  p=0.0000**,  direction=a_greater,  n=(74, 74)

### Neural Net vs Pure LLM

  U=819.0,  p=0.0000**,  direction=b_greater,  n=(74, 62)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 74
Hybrid wins:  66  (89.2%)
NN wins:      7
Tied:         1

## Coverage Gaps (16 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| Liquidation Price Long | medium | rational | N/A | N/A | 0.8624 | 0.9993 |
| Liquidation Price Short | medium | rational | N/A | N/A | -3.3805 | 0.9999 |
| Capital efficiency | medium | rational | N/A | N/A | -2.3826 | 1.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -9.9118 | -4.2294 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | N/A | N/A | -1.8505 | -0.0696 |
| Black-Scholes Put Price | hard | norm_cdf | 0.6040 | -1.3606 | -0.6850 | 0.6040 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.1921 | 0.8854 |
| Gamma of option | hard | norm_pdf | -0.1060 | -4.3964 | -0.1060 | -32.3393 |
| Vega of option | hard | norm_pdf | 0.6285 | -18.5929 | 0.6285 | -16.1621 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.2314 | -6.5668 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.4908 | -2.3436 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.6330 | 0.9999 |
| Required collateral | hard | rational | N/A | N/A | -1.2015 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.1273 | 0.6608 |
| Theta of option | hard | norm_pdf | -0.8034 | -16.0271 | -0.8034 | -1.4001 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 24 | 87.5% | 12.5% | 100.0% |
| hard | 21 | 57.1% | 0.0% | 61.9% |
| medium | 29 | 87.5% | 27.6% | 86.2% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 5 | 1.0000 | -2.1870 | 1.0000 |
| algebraic_with_sqrt | 4 | 1.0000 | -0.3529 | 0.9480 |
| exponential | 5 | 1.0000 | -2.3058 | 1.0000 |
| linear | 18 | 1.0000 | -2.3577 | 1.0000 |
| norm_cdf | 3 | -1.3606 | -1.8505 | -0.0696 |
| norm_pdf | 3 | -10.0000 | -0.1060 | -10.0000 |
| piecewise_linear | 1 | -3.5798 | 0.7545 | 1.0000 |
| quadratic_form | 3 | 1.0000 | -2.1273 | 0.8854 |
| rational | 24 | 1.0000 | -0.0392 | 1.0000 |
| rational_simple | 7 | 1.0000 | -0.1295 | 1.0000 |
| weighted_aggregate | 1 | -10.0000 | 0.6298 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | -104.5690 | 0.0000 | 62 |
| Neural Net | 4.2862 | 1.6737 | 74 |
| Hybrid | 1.1056 | 0.0000 | 74 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 11.4186 | 10.5045 | 844.98 | 74 |
| Neural Net | 0.2014 | 0.1995 | 14.9 | 74 |
| Hybrid | 2.7553 | 2.0100 | 203.89 | 74 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| v4_llm | 48 |
| v4_linear_fallback | 17 |
| v4_residual_nn | 7 |
| v4_linear_fallback_local | 2 |

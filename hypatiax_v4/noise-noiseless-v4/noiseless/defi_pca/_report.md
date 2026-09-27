
# HypatiaX Analysis Report — `exp1_pca`

Experiment mode: **standard**
N total: 74 | N standard: 74 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 74 | 67.6% | 83.3% | 1.0000 | -0.3866 |
| Neural Net | 74 | 17.6% | 12.2% | -0.6622 | -1.3634 |
| Hybrid | 74 | 89.2% | 86.5% | 1.0000 | 0.5020 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=1979.5,  p=0.1844,  direction=b_greater,  n=(74, 60)

### Hybrid vs Neural Net

  U=5038.5,  p=0.0000**,  direction=a_greater,  n=(74, 74)

### Neural Net vs Pure LLM

  U=682.0,  p=0.0000**,  direction=b_greater,  n=(74, 60)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 74
Hybrid wins:  69  (93.2%)
NN wins:      4
Tied:         1

## Coverage Gaps (17 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| Liquidation Price Long | medium | rational | N/A | N/A | -1.4658 | 0.9682 |
| Liquidation Price Short | medium | rational | N/A | N/A | -3.3805 | 0.9750 |
| Capital efficiency | medium | rational | N/A | N/A | -2.3826 | 1.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -9.9118 | -4.2294 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | N/A | N/A | -1.8505 | -0.0696 |
| Black-Scholes Put Price | hard | norm_cdf | N/A | N/A | -0.6850 | 0.6047 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.1921 | 0.8854 |
| Gamma of option | hard | norm_pdf | -0.1060 | -4.3964 | -0.1060 | -32.3393 |
| Vega of option | hard | norm_pdf | 0.6285 | -18.5929 | 0.6285 | -16.1621 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.2764 | 0.9060 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.4908 | -2.3436 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.6330 | 0.9999 |
| Required collateral | hard | rational | N/A | N/A | -1.2015 | 0.9830 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.1273 | 0.6594 |
| Call option intrinsic | hard | piecewise_linear | 0.4532 | -1.3042 | 0.1272 | 0.4532 |
| Theta of option | hard | norm_pdf | N/A | N/A | -0.8034 | -1.4001 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 24 | 87.5% | 12.5% | 100.0% |
| hard | 21 | 66.7% | 0.0% | 61.9% |
| medium | 29 | 87.5% | 20.7% | 93.1% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 5 | 1.0000 | -2.1870 | 1.0000 |
| algebraic_with_sqrt | 4 | 1.0000 | -0.6669 | 1.0000 |
| exponential | 5 | 1.0000 | -2.3058 | 1.0000 |
| linear | 18 | 1.0000 | -2.3577 | 1.0000 |
| norm_cdf | 3 | N/A | -1.8505 | -0.0696 |
| norm_pdf | 3 | -7.1982 | -0.1060 | -10.0000 |
| piecewise_linear | 1 | -1.3042 | 0.1272 | 0.4532 |
| quadratic_form | 3 | 1.0000 | -2.1273 | 0.8854 |
| rational | 24 | 1.0000 | -0.1074 | 1.0000 |
| rational_simple | 7 | 1.0000 | 0.7069 | 1.0000 |
| weighted_aggregate | 1 | -10.0000 | 0.7101 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | -503.2406 | 0.0000 | 60 |
| Neural Net | 18.1147 | 1.6426 | 74 |
| Hybrid | 0.8375 | 0.0000 | 74 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 11.6010 | 9.5530 | 858.48 | 74 |
| Neural Net | 0.2177 | 0.2140 | 16.11 | 74 |
| Hybrid | 0.5990 | 0.6850 | 44.33 | 74 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| v4_llm | 45 |
| v4_linear_fallback | 25 |
| v4_residual_nn | 4 |

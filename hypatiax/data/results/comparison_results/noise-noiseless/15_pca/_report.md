
# HypatiaX Analysis Report — `exp1b_pca`

Experiment mode: **standard**
N total: 371 | N standard: 371 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 370 | 66.5% | 78.8% | 1.0000 | -0.6114 |
| Neural Net | 370 | 19.5% | 13.8% | -0.5008 | -1.2779 |
| Hybrid | 370 | 87.6% | 86.0% | 1.0000 | 0.4950 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=54328.5,  p=0.1333,  direction=b_greater,  n=(370, 311)

### Hybrid vs Neural Net

  U=124796.5,  p=0.0000**,  direction=a_greater,  n=(370, 370)

### Neural Net vs Pure LLM

  U=21228.0,  p=0.0000**,  direction=b_greater,  n=(370, 311)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 370
Hybrid wins:  341  (92.2%)
NN wins:      24
Tied:         5

## Coverage Gaps (87 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| ? | None | None | N/A | N/A | N/A | N/A |
| Liquidation Price Long | medium | rational | N/A | N/A | -1.2374 | 0.9682 |
| Liquidation Price Short | medium | rational | N/A | N/A | -3.1983 | 0.9750 |
| Capital efficiency | medium | rational | N/A | N/A | -2.1446 | 1.0000 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Spot price from AMM reserve | medium | rational | N/A | N/A | -0.1800 | -0.1360 |
| Black-Scholes Call Price | hard | norm_cdf | N/A | N/A | -1.1816 | -0.0696 |
| Black-Scholes Put Price | hard | norm_cdf | N/A | N/A | 0.1913 | 0.5100 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.2608 | 0.8854 |
| Vega of option | hard | norm_pdf | 0.5988 | -18.5929 | 0.5988 | -16.1621 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.4634 | 0.9060 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.7713 | -2.3436 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.7419 | 0.9999 |
| Required collateral | hard | rational | N/A | N/A | -1.4092 | 0.9830 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.9789 | 0.6594 |
| Call option intrinsic | hard | piecewise_linear | 0.4532 | -1.3042 | 0.2746 | 0.4532 |
| Theta of option | hard | norm_pdf | -0.6818 | -15.9373 | -0.6818 | -1.4001 |
| Liquidation Price Long | medium | rational | N/A | N/A | -1.1005 | 0.9682 |
| Liquidation Price Short | medium | rational | N/A | N/A | -2.5264 | 0.9750 |
| Capital efficiency | medium | rational | N/A | N/A | -2.0953 | 1.0000 |
| Impermanent loss percentage | medium | algebraic_with_sqrt | -1.7103 | -2.1798 | -1.7103 | -3.1510 |
| Options Delta | medium | norm_cdf | N/A | N/A | -10.3199 | -4.2294 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | -0.0696 | -193.6432 | -1.7767 | -0.0696 |
| Black-Scholes Put Price | hard | norm_cdf | 0.0337 | -1.3606 | 0.0337 | -1.1507 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.2316 | 0.8854 |
| Gamma of option | hard | norm_pdf | 0.4112 | -4.3964 | 0.4112 | -32.3393 |
| Vega of option | hard | norm_pdf | 0.7040 | 0.5695 | 0.7040 | -16.1621 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.4019 | 0.9060 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.4581 | -2.3436 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.3461 | 0.9999 |
| Required collateral | hard | rational | N/A | N/A | -0.7913 | 0.9830 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -3.1512 | 0.6594 |
| Call option intrinsic | hard | piecewise_linear | 0.4532 | -1.3042 | 0.3121 | 0.4532 |
| Theta of option | hard | norm_pdf | -0.8457 | -15.9196 | -0.8457 | -1.4001 |
| Liquidation Price Long | medium | rational | N/A | N/A | -1.4658 | 0.9682 |
| Liquidation Price Short | medium | rational | N/A | N/A | -3.3805 | 0.9750 |
| Capital efficiency | medium | rational | N/A | N/A | -2.3826 | 1.0000 |
| Options Delta | medium | norm_cdf | -4.2294 | -9.4800 | -9.9118 | -4.2294 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | 0.2464 | 0.2464 | -1.8505 | -0.0696 |
| Black-Scholes Put Price | hard | norm_cdf | N/A | N/A | -0.6850 | -1.1507 |
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
| Liquidation Price Long | medium | rational | N/A | N/A | -1.3762 | 0.9682 |
| Liquidation Price Short | medium | rational | N/A | N/A | -2.3697 | 0.9750 |
| Capital efficiency | medium | rational | N/A | N/A | -2.3308 | 1.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -10.2325 | -4.2294 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | 0.2464 | 0.2464 | -1.5010 | -0.0696 |
| Black-Scholes Put Price | hard | norm_cdf | -0.1265 | -1.3606 | -0.1265 | -1.1507 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.3230 | 0.8854 |
| Gamma of option | hard | norm_pdf | 0.4543 | -4.3964 | 0.4543 | -32.3393 |
| Vega of option | hard | norm_pdf | 0.7354 | -18.5929 | 0.7354 | -16.1621 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.5515 | 0.9060 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.5873 | -2.3436 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.1843 | 0.9999 |
| Required collateral | hard | rational | N/A | N/A | -1.3953 | 0.9830 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.5949 | 0.6594 |
| Call option intrinsic | hard | piecewise_linear | 0.4532 | -1.3042 | 0.2341 | 0.4532 |
| Theta of option | hard | norm_pdf | -0.9529 | -14.3044 | -0.9529 | -1.4001 |
| Liquidation Price Long | medium | rational | N/A | N/A | -1.3282 | 0.9682 |
| Liquidation Price Short | medium | rational | N/A | N/A | -2.6278 | 0.9750 |
| Capital efficiency | medium | rational | N/A | N/A | -2.1805 | 1.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -10.0434 | -4.2294 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | -0.0696 | -1.2400 | -1.2633 | -0.0696 |
| Black-Scholes Put Price | hard | norm_cdf | -1.0251 | -1.3606 | -1.0251 | -1.1507 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.3532 | 0.8854 |
| Gamma of option | hard | norm_pdf | 0.2974 | -4.3964 | 0.2974 | -32.3393 |
| Vega of option | hard | norm_pdf | 0.7164 | -18.5929 | 0.7164 | -16.1621 |
| Convexity Adjustment | hard | algebraic | N/A | N/A | -2.2925 | 1.0000 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.3263 | 0.9060 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.6733 | -2.3436 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.0140 | 0.9999 |
| Required collateral | hard | rational | N/A | N/A | -0.4883 | 0.9830 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.7078 | 0.6594 |
| Call option intrinsic | hard | piecewise_linear | 0.4532 | -1.3042 | 0.0076 | 0.4532 |
| Theta of option | hard | norm_pdf | -0.6335 | -15.9304 | -0.6335 | -1.4001 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 120 | 86.7% | 15.0% | 100.0% |
| hard | 105 | 54.3% | 2.9% | 62.9% |
| medium | 145 | 85.1% | 20.7% | 91.0% |
| unknown | 0 | 0.0% | 0.0% | 0.0% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 25 | 1.0000 | -2.3026 | 1.0000 |
| algebraic_with_sqrt | 20 | 1.0000 | -0.7580 | 1.0000 |
| exponential | 25 | 1.0000 | -2.3058 | 1.0000 |
| linear | 90 | 1.0000 | -2.4069 | 1.0000 |
| norm_cdf | 15 | -1.3606 | -1.5010 | -1.1507 |
| norm_pdf | 15 | -10.0000 | 0.4112 | -10.0000 |
| piecewise_linear | 5 | -1.3042 | 0.2341 | 0.4532 |
| quadratic_form | 15 | 1.0000 | -2.5949 | 0.8854 |
| rational | 120 | 1.0000 | -0.1696 | 1.0000 |
| rational_simple | 35 | 1.0000 | 0.8391 | 1.0000 |
| unknown | 0 | N/A | N/A | N/A |
| weighted_aggregate | 5 | -10.0000 | 0.9710 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | 33979.7384 | 0.0000 | 311 |
| Neural Net | 32.3911 | 1.5007 | 370 |
| Hybrid | 0.7841 | 0.0000 | 370 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 11.5191 | 10.1570 | 4262.08 | 370 |
| Neural Net | 0.3841 | 0.3840 | 142.12 | 370 |
| Hybrid | 1.0812 | 1.2450 | 400.05 | 370 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| v4_llm | 219 |
| v4_linear_fallback | 128 |
| v4_residual_nn | 22 |
| v4_linear_fallback_local | 1 |

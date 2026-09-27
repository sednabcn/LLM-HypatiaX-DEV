
# HypatiaX Analysis Report — `exp1b_pca`

Experiment mode: **standard**
N total: 371 | N standard: 371 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 370 | 64.6% | 78.8% | 1.0000 | -0.6651 |
| Neural Net | 370 | 19.5% | 13.8% | -0.5008 | -1.2781 |
| Hybrid | 370 | 86.5% | 84.8% | 1.0000 | 0.4695 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=52156.0,  p=0.0865,  direction=b_greater,  n=(369, 302)

### Hybrid vs Neural Net

  U=124165.5,  p=0.0000**,  direction=a_greater,  n=(369, 370)

### Neural Net vs Pure LLM

  U=20704.0,  p=0.0000**,  direction=b_greater,  n=(370, 302)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 369
Hybrid wins:  340  (92.1%)
NN wins:      24
Tied:         5

## Coverage Gaps (93 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| ? | None | None | N/A | N/A | N/A | N/A |
| Impermanent loss breakeven fee rate | easy | rational_simple | N/A | N/A | -1.0625 | 1.0000 |
| Liquidation Price Long | medium | rational | N/A | N/A | -1.2374 | 0.9682 |
| Liquidation Price Short | medium | rational | N/A | N/A | -3.1983 | 0.9750 |
| Capital efficiency | medium | rational | N/A | N/A | -2.1446 | 1.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -10.5222 | -4.2294 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Spot price from AMM reserve | medium | rational | N/A | N/A | -0.1800 | -0.1360 |
| Black-Scholes Call Price | hard | norm_cdf | 0.2464 | 0.2464 | -1.0637 | -0.0696 |
| Black-Scholes Put Price | hard | norm_cdf | 0.5919 | 0.5415 | 0.1922 | 0.5919 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.2608 | 0.8854 |
| Gamma of option | hard | norm_pdf | 0.4325 | -4.3964 | 0.4325 | -32.3393 |
| Vega of option | hard | norm_pdf | 0.5988 | -18.5929 | 0.5988 | -16.1621 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.4634 | 0.9060 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.7713 | -2.3436 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.7419 | 0.9999 |
| Required collateral | hard | rational | N/A | N/A | -1.4092 | 0.9830 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.9789 | 0.6594 |
| Call option intrinsic | hard | piecewise_linear | 0.4532 | -1.3042 | 0.2746 | 0.4532 |
| Theta of option | hard | norm_pdf | N/A | N/A | -0.6818 | -1.4001 |
| Liquidation Price Long | medium | rational | N/A | N/A | -1.1005 | 0.9682 |
| Liquidation Price Short | medium | rational | N/A | N/A | -2.5264 | 0.9750 |
| Capital efficiency | medium | rational | N/A | N/A | -2.0953 | 1.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -10.3199 | -4.2294 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | N/A | N/A | -1.8327 | -0.0696 |
| Black-Scholes Put Price | hard | norm_cdf | N/A | N/A | 0.0717 | -1.1507 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.2316 | 0.8854 |
| Gamma of option | hard | norm_pdf | 0.4112 | -4.3964 | 0.4112 | -32.3393 |
| Vega of option | hard | norm_pdf | 0.7040 | -18.5929 | 0.7040 | -16.1621 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.4019 | 0.9060 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.4581 | -2.3436 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.3461 | 0.9999 |
| Required collateral | hard | rational | N/A | N/A | -0.7913 | 0.9830 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -3.1512 | 0.6594 |
| Call option intrinsic | hard | piecewise_linear | 0.4532 | -1.3042 | 0.3121 | 0.4532 |
| Theta of option | hard | norm_pdf | -0.8457 | -16.0271 | -0.8457 | -1.4001 |
| Liquidation Price Long | medium | rational | N/A | N/A | -1.4658 | 0.9682 |
| Liquidation Price Short | medium | rational | N/A | N/A | -3.3805 | 0.9750 |
| Compounding Staking Returns | medium | exponential | 0.3332 | -0.2206 | -2.3058 | 0.3332 |
| Capital efficiency | medium | rational | N/A | N/A | -2.3826 | 1.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -9.9026 | -4.2294 |
| Collateral ratio | medium | rational | N/A | N/A | 0.9625 | 1.0000 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | -0.0696 | -20.2184 | -1.8315 | -0.0696 |
| Black-Scholes Put Price | hard | norm_cdf | N/A | N/A | -0.6802 | -1.1507 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.1921 | 0.8854 |
| Gamma of option | hard | norm_pdf | -0.1060 | -4.3964 | -0.1060 | -32.3393 |
| Vega of option | hard | norm_pdf | 0.6285 | -18.5929 | 0.6285 | -16.1621 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.2764 | 0.9060 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.4907 | -2.3436 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.6330 | 0.9999 |
| Required collateral | hard | rational | N/A | N/A | -1.2015 | 0.9830 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.1273 | 0.6594 |
| Call option intrinsic | hard | piecewise_linear | 0.4532 | -1.3042 | 0.1272 | 0.4532 |
| Theta of option | hard | norm_pdf | N/A | N/A | -0.8034 | -1.4001 |
| Liquidation Price Long | medium | rational | N/A | N/A | -1.3762 | 0.9682 |
| Liquidation Price Short | medium | rational | N/A | N/A | -2.3697 | 0.9750 |
| Compounding Staking Returns | medium | exponential | 0.3016 | -0.2206 | -2.4753 | 0.3016 |
| Capital efficiency | medium | rational | N/A | N/A | -2.3308 | 1.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -10.2335 | -4.2294 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Spot price from AMM reserve | medium | rational | N/A | N/A | -0.1796 | -0.1360 |
| Black-Scholes Call Price | hard | norm_cdf | N/A | N/A | -1.4957 | -0.0696 |
| Black-Scholes Put Price | hard | norm_cdf | N/A | N/A | -0.0988 | -1.1507 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.3230 | N/A |
| Gamma of option | hard | norm_pdf | 0.4543 | -4.3964 | 0.4543 | -32.3393 |
| Vega of option | hard | norm_pdf | 0.7354 | -18.5929 | 0.7354 | -16.1621 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.5515 | 0.9060 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.5873 | -2.3436 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.1843 | 0.9999 |
| Required collateral | hard | rational | N/A | N/A | -1.3953 | 0.9830 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.5949 | 0.6594 |
| Call option intrinsic | hard | piecewise_linear | 0.4532 | -1.3042 | 0.2341 | 0.4532 |
| Theta of option | hard | norm_pdf | -0.9529 | -15.9203 | -0.9529 | -1.4001 |
| Liquidation Price Long | medium | rational | N/A | N/A | -1.3282 | 0.9682 |
| Liquidation Price Short | medium | rational | N/A | N/A | -2.6278 | 0.9750 |
| Compounding Staking Returns | medium | exponential | 0.3003 | -0.2206 | -2.0727 | 0.3003 |
| Capital efficiency | medium | rational | N/A | N/A | -2.1805 | 1.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -10.1276 | -4.2294 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | 0.2464 | 0.2464 | -1.3531 | -0.0696 |
| Black-Scholes Put Price | hard | norm_cdf | N/A | N/A | -1.0543 | -1.1507 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.3532 | 0.8854 |
| Gamma of option | hard | norm_pdf | 0.2974 | -4.3964 | 0.2974 | -32.3393 |
| Vega of option | hard | norm_pdf | 0.7164 | -18.5929 | 0.7164 | -16.1621 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.3263 | 0.9060 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.6731 | -2.3436 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.0140 | 0.9999 |
| Required collateral | hard | rational | N/A | N/A | -0.4883 | 0.9830 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.7078 | 0.6594 |
| Call option intrinsic | hard | piecewise_linear | 0.4532 | -1.3042 | 0.0076 | 0.4532 |
| Theta of option | hard | norm_pdf | N/A | N/A | -0.6335 | -1.4001 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 120 | 85.7% | 15.0% | 100.0% |
| hard | 105 | 54.5% | 2.9% | 61.5% |
| medium | 145 | 85.5% | 20.7% | 89.0% |
| unknown | 0 | 0.0% | 0.0% | 0.0% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 25 | 1.0000 | -2.3026 | 1.0000 |
| algebraic_with_sqrt | 20 | 1.0000 | -0.7580 | 1.0000 |
| exponential | 25 | 1.0000 | -2.3058 | 1.0000 |
| linear | 90 | 1.0000 | -2.4069 | 1.0000 |
| norm_cdf | 15 | 0.2464 | -1.4957 | -1.1507 |
| norm_pdf | 15 | -10.0000 | 0.4112 | -10.0000 |
| piecewise_linear | 5 | -1.3042 | 0.2341 | 0.4532 |
| quadratic_form | 15 | 1.0000 | -2.5949 | 0.8854 |
| rational | 120 | 1.0000 | -0.1696 | 1.0000 |
| rational_simple | 35 | 1.0000 | 0.8391 | 1.0000 |
| unknown | 0 | N/A | N/A | N/A |
| weighted_aggregate | 5 | -10.0000 | 0.9655 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | 123697.0943 | 0.0000 | 302 |
| Neural Net | 32.4368 | 1.5007 | 370 |
| Hybrid | 0.8698 | 0.0000 | 369 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 11.4812 | 10.2435 | 4248.06 | 370 |
| Neural Net | 0.3284 | 0.3270 | 121.5 | 370 |
| Hybrid | 0.9485 | 1.0950 | 350.94 | 370 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| v4_llm | 217 |
| v4_linear_fallback | 127 |
| v4_residual_nn | 24 |
| v4_blend | 1 |


# HypatiaX Analysis Report — `exp1`

Experiment mode: **standard**
N total: 450 | N standard: 450 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 450 | 78.0% | 85.8% | 1.0000 | -0.1560 |
| Neural Net | 450 | 79.3% | 18.2% | -0.2412 | -1.1166 |
| Hybrid | 450 | 100.0% | 92.2% | 1.0000 | 0.8292 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=86400.0,  p=0.0005**,  direction=b_greater,  n=(450, 351)

### Hybrid vs Neural Net

  U=154994.5,  p=0.0000**,  direction=a_greater,  n=(450, 357)

### Neural Net vs Pure LLM

  U=16097.0,  p=0.0000**,  direction=b_greater,  n=(357, 351)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 357
Hybrid wins:  324  (90.8%)
NN wins:      0
Tied:         33

## Coverage Gaps (108 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| Liquidation Price Long | medium | rational | N/A | N/A | 0.8965 | 1.0000 |
| Liquidation Price Short | medium | rational | N/A | N/A | -2.5432 | 1.0000 |
| Capital efficiency | medium | rational | N/A | N/A | -2.0963 | 1.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -10.3199 | 0.7709 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | N/A | N/A | -0.8981 | -0.8981 |
| Black-Scholes Put Price | hard | norm_cdf | N/A | N/A | 0.0459 | 0.5415 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.2542 | 1.0000 |
| Gamma of option | hard | norm_pdf | N/A | N/A | 0.3433 | 0.3433 |
| Vega of option | hard | norm_pdf | N/A | N/A | 0.6593 | 0.6593 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.6256 | 1.0000 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.4329 | 1.0000 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.3461 | 1.0000 |
| Required collateral | hard | rational | N/A | N/A | -0.7895 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | N/A | 1.0000 |
| Theta of option | hard | norm_pdf | -0.7755 | -15.9304 | -0.7755 | -0.7755 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | N/A | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | N/A | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | N/A | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | N/A | 1.0000 |
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
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | N/A | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | N/A | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | N/A | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | N/A | 1.0000 |
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
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | N/A | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | N/A | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | N/A | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | N/A | 1.0000 |
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
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | N/A | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | N/A | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | N/A | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | N/A | 1.0000 |
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
| easy | 136 | 91.7% | 12.4% | 100.0% |
| hard | 137 | 76.2% | 11.1% | 81.8% |
| medium | 177 | 84.6% | 27.4% | 94.3% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 57 | 1.0000 | -2.7952 | 1.0000 |
| algebraic_with_sqrt | 20 | 1.0000 | -0.4518 | 1.0000 |
| exponential | 25 | 1.0000 | -2.1634 | 1.0000 |
| linear | 90 | 1.0000 | -1.3664 | 1.0000 |
| norm_cdf | 15 | 0.7136 | -0.8981 | -0.8981 |
| norm_pdf | 15 | -10.0000 | 0.3433 | 0.3433 |
| piecewise_linear | 5 | -3.5798 | 0.8679 | 1.0000 |
| quadratic_form | 47 | 1.0000 | -0.2542 | 1.0000 |
| rational | 136 | 1.0000 | -0.0229 | 1.0000 |
| rational_simple | 35 | 1.0000 | 0.5461 | 1.0000 |
| weighted_aggregate | 5 | -10.0000 | 0.9725 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | 11.5071 | 0.0000 | 351 |
| Neural Net | 2.1169 | 1.2411 | 357 |
| Hybrid | -0.0320 | 0.0000 | 442 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 11.9236 | 10.4680 | 5365.62 | 450 |
| Neural Net | 0.2848 | 0.3600 | 128.15 | 450 |
| Hybrid | 1.7090 | 1.3400 | 769.03 | 450 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| llm | 417 |
| nn | 28 |
| nn_fallback | 5 |

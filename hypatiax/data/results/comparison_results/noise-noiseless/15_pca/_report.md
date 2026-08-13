
# HypatiaX Analysis Report — `exp1b_pca`

Experiment mode: **standard**
N total: 370 | N standard: 370 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 370 | 59.7% | 76.5% | 1.0000 | -0.7954 |
| Neural Net | 370 | 100.0% | 13.8% | -0.5008 | -1.2781 |
| Hybrid | 370 | 93.8% | 90.5% | 1.0000 | 0.8989 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=66076.5,  p=0.0000**,  direction=b_greater,  n=(370, 289)

### Hybrid vs Neural Net

  U=132618.0,  p=0.0000**,  direction=a_greater,  n=(370, 370)

### Neural Net vs Pure LLM

  U=21648.0,  p=0.0000**,  direction=b_greater,  n=(370, 289)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 370
Hybrid wins:  342  (92.4%)
NN wins:      0
Tied:         28

## Coverage Gaps (102 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| Spot price from AMM | easy | rational_simple | N/A | N/A | -3333.1110 | 1.0000 |
| Long position unrealized PnL | easy | linear | N/A | N/A | 0.2585 | 1.0000 |
| Cross-margin available balance | easy | linear | N/A | N/A | -0.1390 | 1.0000 |
| Reserve ratio | easy | rational_simple | N/A | N/A | -0.1705 | 1.0000 |
| Liquidation Price Long | medium | rational | N/A | N/A | -1.2374 | 1.0000 |
| Liquidation Price Short | medium | rational | N/A | N/A | -3.1983 | 1.0000 |
| Effective Leverage | medium | rational | N/A | N/A | -4.6440 | 1.0000 |
| Capital efficiency | medium | rational | N/A | N/A | -2.1446 | 1.0000 |
| Utilization rate of DeFi | medium | rational | N/A | N/A | 0.9716 | 1.0000 |
| Impermanent loss percentage | medium | algebraic_with_sqrt | N/A | N/A | -2.0011 | 1.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -10.5222 | 0.7709 |
| Collateral ratio | medium | rational | N/A | N/A | 0.9770 | 1.0000 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | -1.0637 | -193.6432 | -1.0637 | -1.0637 |
| Black-Scholes Put Price | hard | norm_cdf | N/A | N/A | 0.1922 | 0.5415 |
| Vega of option | hard | norm_pdf | 0.5988 | -18.5929 | 0.5988 | 0.5988 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.4634 | 1.0000 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.7713 | 1.0000 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.7419 | 1.0000 |
| Required collateral | hard | rational | N/A | N/A | -1.4092 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.9789 | 1.0000 |
| Theta of option | hard | norm_pdf | -0.6818 | -15.9373 | -0.6818 | -0.6818 |
| Spot price from AMM | easy | rational_simple | N/A | N/A | -496.0379 | 1.0000 |
| Validator commission adjusted | easy | linear | N/A | N/A | -4.0519 | 1.0000 |
| Slashing penalty | easy | linear | N/A | N/A | -2.4424 | 1.0000 |
| Reserve ratio | easy | rational_simple | N/A | N/A | -0.1670 | 1.0000 |
| Liquidation Price Long | medium | rational | N/A | N/A | -1.1005 | 1.0000 |
| Liquidation Price Short | medium | rational | N/A | N/A | -2.5264 | 1.0000 |
| Capital efficiency | medium | rational | N/A | N/A | -2.0953 | 1.0000 |
| Utilization rate of DeFi | medium | rational | N/A | N/A | 0.9761 | 1.0000 |
| Impermanent loss percentage | medium | algebraic_with_sqrt | N/A | N/A | -1.7103 | 1.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -10.3199 | 0.7709 |
| Collateral ratio | medium | rational | N/A | N/A | 0.9939 | 1.0000 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | N/A | N/A | -1.8327 | -1.8327 |
| Black-Scholes Put Price | hard | norm_cdf | 0.0717 | -1.3606 | 0.0717 | 0.0717 |
| Vega of option | hard | norm_pdf | 0.7040 | -18.5929 | 0.7040 | 0.7040 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.4019 | 1.0000 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.4581 | 1.0000 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.3461 | 1.0000 |
| Required collateral | hard | rational | N/A | N/A | -0.7913 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -3.1512 | 1.0000 |
| Theta of option | hard | norm_pdf | -0.8457 | -16.0271 | -0.8457 | -0.8457 |
| Validator commission adjusted | easy | linear | N/A | N/A | -2.5077 | 1.0000 |
| Slashing penalty | easy | linear | N/A | N/A | -2.9037 | 1.0000 |
| Leveraged position notional | easy | linear | N/A | N/A | -2.9426 | 1.0000 |
| Liquidation Price Long | medium | rational | N/A | N/A | -1.4658 | 1.0000 |
| Liquidation Price Short | medium | rational | N/A | N/A | -3.3805 | 1.0000 |
| Capital efficiency | medium | rational | N/A | N/A | -2.3826 | 1.0000 |
| Utilization rate of DeFi | medium | rational | N/A | N/A | 0.9873 | 1.0000 |
| Impermanent loss percentage | medium | algebraic_with_sqrt | N/A | N/A | -2.1889 | 1.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -9.9026 | 0.7709 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | -1.8315 | -193.6432 | -1.8315 | -1.8315 |
| Black-Scholes Put Price | hard | norm_cdf | -0.6802 | -1.3606 | -0.6802 | -0.6802 |
| Gamma of option | hard | norm_pdf | -0.1060 | -4.3964 | -0.1060 | -0.1060 |
| Vega of option | hard | norm_pdf | 0.6285 | -18.5929 | 0.6285 | 0.6285 |
| Impermanent loss in constant product | hard | algebraic_with_sqrt | N/A | N/A | -1.2321 | 1.0000 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.2764 | 1.0000 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.4907 | 1.0000 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.6330 | 1.0000 |
| Required collateral | hard | rational | N/A | N/A | -1.2015 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.1273 | 1.0000 |
| Theta of option | hard | norm_pdf | -0.8034 | -16.0271 | -0.8034 | -0.8034 |
| Slashing penalty | easy | linear | N/A | N/A | -2.4080 | 1.0000 |
| Liquidation Price Long | medium | rational | N/A | N/A | -1.3762 | 1.0000 |
| Liquidation Price Short | medium | rational | N/A | N/A | -2.3697 | 1.0000 |
| Capital efficiency | medium | rational | N/A | N/A | -2.3308 | 1.0000 |
| Utilization rate of DeFi | medium | rational | N/A | N/A | 0.9953 | 1.0000 |
| Impermanent loss percentage | medium | algebraic_with_sqrt | N/A | N/A | -1.8948 | 1.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -10.2335 | 0.7709 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | -1.4957 | -193.6432 | -1.4957 | -1.4957 |
| Black-Scholes Put Price | hard | norm_cdf | -0.0988 | -1.3606 | -0.0988 | -0.0988 |
| Gamma of option | hard | norm_pdf | 0.4543 | -4.3964 | 0.4543 | 0.4543 |
| Vega of option | hard | norm_pdf | 0.7354 | -18.5929 | 0.7354 | 0.7354 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.5515 | 1.0000 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.5873 | 1.0000 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.1843 | 1.0000 |
| Required collateral | hard | rational | N/A | N/A | -1.3953 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.5949 | 1.0000 |
| Theta of option | hard | norm_pdf | -0.9529 | -16.0271 | -0.9529 | -0.9529 |
| Simple Staking APY | easy | linear | N/A | N/A | -0.2570 | 1.0000 |
| Slashing penalty | easy | linear | N/A | N/A | -2.6556 | 1.0000 |
| Reserve ratio | easy | rational_simple | N/A | N/A | -0.1609 | 1.0000 |
| Liquidation Price Long | medium | rational | N/A | N/A | -1.3282 | 1.0000 |
| Liquidation Price Short | medium | rational | N/A | N/A | -2.6278 | 1.0000 |
| Capital efficiency | medium | rational | N/A | N/A | -2.1805 | 1.0000 |
| Utilization rate of DeFi | medium | rational | N/A | N/A | 0.9954 | 1.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -10.1276 | 0.7709 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | -1.3531 | -193.6432 | -1.3531 | -1.3531 |
| Black-Scholes Put Price | hard | norm_cdf | 0.5415 | -1.3606 | -1.0543 | 0.5415 |
| Gamma of option | hard | norm_pdf | 0.2974 | -4.3964 | 0.2974 | 0.2974 |
| Vega of option | hard | norm_pdf | 0.7164 | -18.5929 | 0.7164 | 0.7164 |
| Impermanent loss in constant product | hard | algebraic_with_sqrt | N/A | N/A | -1.7783 | 1.0000 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.3263 | 1.0000 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.6731 | 1.0000 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.0140 | 1.0000 |
| Required collateral | hard | rational | N/A | N/A | -0.4883 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.7078 | 1.0000 |
| Theta of option | hard | norm_pdf | -0.6335 | -16.0271 | -0.6335 | -0.6335 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 120 | 92.4% | 15.0% | 100.0% |
| hard | 105 | 46.1% | 2.9% | 76.2% |
| medium | 145 | 82.4% | 20.7% | 93.1% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 25 | 1.0000 | -2.3026 | 1.0000 |
| algebraic_with_sqrt | 20 | 1.0000 | -0.7580 | 1.0000 |
| exponential | 25 | 1.0000 | -2.3058 | 1.0000 |
| linear | 90 | 1.0000 | -2.4069 | 1.0000 |
| norm_cdf | 15 | -5.6803 | -1.4957 | 0.0717 |
| norm_pdf | 15 | -10.0000 | 0.4112 | 0.4112 |
| piecewise_linear | 5 | -1.3042 | 0.2341 | 1.0000 |
| quadratic_form | 15 | 0.3626 | -2.5949 | 1.0000 |
| rational | 120 | 1.0000 | -0.1696 | 1.0000 |
| rational_simple | 35 | 1.0000 | 0.8391 | 1.0000 |
| weighted_aggregate | 5 | 1.0000 | 0.9655 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | 87832.2990 | 0.0000 | 289 |
| Neural Net | 32.4368 | 1.5007 | 370 |
| Hybrid | -0.1590 | 0.0000 | 366 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 9.5754 | 8.8100 | 3542.91 | 370 |
| Neural Net | 0.3334 | 0.3325 | 123.37 | 370 |
| Hybrid | 1.6779 | 1.3570 | 620.84 | 370 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| llm | 342 |
| nn | 23 |
| nn_fallback | 5 |

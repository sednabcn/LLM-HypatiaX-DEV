
# HypatiaX Analysis Report — `exp1b_pca`

Experiment mode: **standard**
N total: 371 | N standard: 371 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 370 | 64.6% | 78.0% | 1.0000 | -0.6869 |
| Neural Net | 370 | 19.5% | 13.8% | -0.5008 | -1.2781 |
| Hybrid | 370 | 77.0% | 73.2% | 1.0000 | 0.3428 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=53900.0,  p=0.3911,  direction=b_greater,  n=(365, 305)

### Hybrid vs Neural Net

  U=117030.5,  p=0.0000**,  direction=a_greater,  n=(365, 370)

### Neural Net vs Pure LLM

  U=21266.0,  p=0.0000**,  direction=b_greater,  n=(370, 305)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 365
Hybrid wins:  304  (83.3%)
NN wins:      56
Tied:         5

## Coverage Gaps (103 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| ? | None | None | N/A | N/A | N/A | N/A |
| Slashing penalty | easy | linear | -2.2875 | -4.7088 | -2.8448 | -2.2875 |
| Liquidation Price Long | medium | rational | N/A | N/A | -1.2374 | 0.9872 |
| Liquidation Price Short | medium | rational | N/A | N/A | -3.1983 | 0.9998 |
| Capital efficiency | medium | rational | N/A | N/A | -2.1446 | -2.4468 |
| Utilization rate of DeFi | medium | rational | N/A | N/A | 0.9716 | 1.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -10.5222 | -10.8407 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Spot price from AMM reserve | medium | rational | N/A | N/A | -0.1800 | 1.0000 |
| Black-Scholes Call Price | hard | norm_cdf | 0.2464 | 0.2464 | -1.0637 | -1.5196 |
| Black-Scholes Put Price | hard | norm_cdf | N/A | N/A | 0.1922 | 0.5160 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.2608 | -0.3325 |
| Gamma of option | hard | norm_pdf | 0.4325 | -4.3964 | 0.4325 | -0.0218 |
| Vega of option | hard | norm_pdf | 0.6026 | -18.5929 | 0.5988 | 0.6026 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.4634 | 0.2996 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.7713 | -1.6460 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.7419 | N/A |
| Required collateral | hard | rational | N/A | N/A | -1.4092 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.9789 | 0.6624 |
| Call option intrinsic | hard | piecewise_linear | 0.2909 | -1.3042 | 0.2746 | 0.2909 |
| Uniswap V3 virtual | hard | algebraic_with_sqrt | -0.1462 | -13444298.7199 | -0.2839 | -0.1462 |
| Theta of option | hard | norm_pdf | -0.6250 | -15.9292 | -0.6818 | -0.6250 |
| Slashing penalty | easy | linear | -2.4002 | -4.7088 | -2.4424 | -2.4002 |
| Impermanent loss breakeven fee rate | easy | rational_simple | N/A | N/A | 0.0653 | -10.2935 |
| Liquidation Price Long | medium | rational | N/A | N/A | -1.1005 | 0.9862 |
| Liquidation Price Short | medium | rational | N/A | N/A | -2.5264 | 0.9999 |
| Compounding Staking Returns | medium | exponential | -0.2206 | -0.2206 | -2.3734 | -2.4120 |
| Capital efficiency | medium | rational | N/A | N/A | -2.0953 | -2.3076 |
| Options Delta | medium | norm_cdf | N/A | N/A | -10.3199 | -11.8102 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | -1.8327 | -193.6432 | -1.8327 | -2.4351 |
| Black-Scholes Put Price | hard | norm_cdf | 0.6141 | -1.3606 | 0.0717 | 0.6141 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.2316 | -0.3883 |
| Gamma of option | hard | norm_pdf | 0.4112 | -4.3964 | 0.4112 | 0.3390 |
| Vega of option | hard | norm_pdf | 0.7040 | -18.5929 | 0.7040 | 0.6435 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.4019 | 0.4366 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.4581 | -1.5703 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.3461 | N/A |
| Required collateral | hard | rational | N/A | N/A | -0.7913 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -3.1512 | 0.6123 |
| Call option intrinsic | hard | piecewise_linear | 0.3121 | -1.3042 | 0.3121 | 0.0301 |
| Uniswap V3 virtual | hard | algebraic_with_sqrt | -0.0935 | -13444298.7199 | -0.0935 | -0.1478 |
| Theta of option | hard | norm_pdf | N/A | N/A | -0.8457 | -1.1634 |
| Slashing penalty | easy | linear | -2.4220 | -4.7088 | -2.9037 | -2.4220 |
| Liquidation Price Long | medium | rational | N/A | N/A | -1.4658 | 0.9825 |
| Liquidation Price Short | medium | rational | N/A | N/A | -3.3805 | 0.9999 |
| Capital efficiency | medium | rational | N/A | N/A | -2.3826 | -2.2776 |
| Options Delta | medium | norm_cdf | N/A | N/A | -9.9026 | -10.2587 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Spot price from AMM reserve | medium | rational | N/A | N/A | -0.1801 | 1.0000 |
| Black-Scholes Call Price | hard | norm_cdf | 0.2464 | 0.2464 | -1.8315 | -1.5971 |
| Black-Scholes Put Price | hard | norm_cdf | 0.6078 | -1.3606 | -0.6802 | 0.6078 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.1921 | -0.2104 |
| Vega of option | hard | norm_pdf | 0.6891 | -18.5929 | 0.6285 | 0.6891 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.2764 | 0.4698 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.4907 | -1.4817 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.6330 | N/A |
| Required collateral | hard | rational | N/A | N/A | -1.2015 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.1273 | 0.6608 |
| Call option intrinsic | hard | piecewise_linear | 0.1272 | -1.3042 | 0.1272 | 0.1058 |
| Uniswap V3 virtual | hard | algebraic_with_sqrt | -0.1017 | -13444298.7199 | -0.1017 | -0.1374 |
| Theta of option | hard | norm_pdf | -0.8034 | -15.9383 | -0.8034 | -1.1756 |
| Spot price from AMM | easy | rational_simple | -2768.8124 | -32651.6732 | -2768.8124 | -6006.4632 |
| Slashing penalty | easy | linear | -2.3358 | -4.7088 | -2.4080 | -2.3358 |
| Liquidation Price Long | medium | rational | N/A | N/A | -1.3762 | 0.9840 |
| Liquidation Price Short | medium | rational | N/A | N/A | -2.3697 | 0.9999 |
| Compounding Staking Returns | medium | exponential | -0.2206 | -0.2206 | -2.4753 | -2.0621 |
| Capital efficiency | medium | rational | N/A | N/A | -2.3308 | -2.0530 |
| Impermanent loss percentage | medium | algebraic_with_sqrt | -1.8948 | -2.1798 | -1.8948 | -2.0897 |
| Options Delta | medium | norm_cdf | N/A | N/A | -10.2335 | -10.6587 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Spot price from AMM reserve | medium | rational | N/A | N/A | -0.1796 | 1.0000 |
| Black-Scholes Call Price | hard | norm_cdf | -1.4957 | -193.6432 | -1.4957 | -3.0265 |
| Black-Scholes Put Price | hard | norm_cdf | 0.5727 | -1.3606 | -0.0988 | 0.5727 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.3230 | -0.2991 |
| Gamma of option | hard | norm_pdf | 0.4543 | -4.3964 | 0.4543 | 0.2163 |
| Vega of option | hard | norm_pdf | 0.7354 | 0.5695 | 0.7354 | 0.6594 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.5515 | 0.4370 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.5873 | -1.6677 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.1843 | N/A |
| Required collateral | hard | rational | N/A | N/A | -1.3953 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.5949 | 0.6231 |
| Call option intrinsic | hard | piecewise_linear | 0.2341 | -1.3042 | 0.2341 | 0.1023 |
| Theta of option | hard | norm_pdf | N/A | N/A | -0.9529 | -0.8663 |
| Slashing penalty | easy | linear | -2.3294 | -4.7088 | -2.6556 | -2.3294 |
| LP fee earnings | easy | rational_simple | N/A | N/A | 0.9597 | 1.0000 |
| Liquidation Price Long | medium | rational | N/A | N/A | -1.3282 | 0.9839 |
| Liquidation Price Short | medium | rational | N/A | N/A | -2.6278 | 0.9999 |
| Capital efficiency | medium | rational | N/A | N/A | -2.1805 | -2.2047 |
| Options Delta | medium | norm_cdf | N/A | N/A | -10.1276 | -10.8968 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | 0.2464 | 0.2464 | -1.3531 | -1.2801 |
| Black-Scholes Put Price | hard | norm_cdf | N/A | N/A | -1.0543 | 0.0653 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.3532 | -0.3897 |
| Gamma of option | hard | norm_pdf | 0.2974 | -4.3964 | 0.2974 | 0.1118 |
| Vega of option | hard | norm_pdf | 0.7164 | -18.5929 | 0.7164 | 0.6892 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.3263 | 0.5713 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.6731 | -1.6111 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.0140 | N/A |
| Required collateral | hard | rational | N/A | N/A | -0.4883 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.7078 | 0.6660 |
| Call option intrinsic | hard | piecewise_linear | 0.3388 | -1.3042 | 0.0076 | 0.3388 |
| Theta of option | hard | norm_pdf | -0.6335 | -15.9419 | -0.6335 | -0.9274 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 120 | 85.6% | 15.0% | 90.0% |
| hard | 105 | 53.5% | 2.9% | 45.0% |
| medium | 145 | 85.3% | 20.7% | 78.6% |
| unknown | 0 | 0.0% | 0.0% | 0.0% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 25 | 1.0000 | -2.3026 | 1.0000 |
| algebraic_with_sqrt | 20 | 1.0000 | -0.7580 | 0.5520 |
| exponential | 25 | 1.0000 | -2.3058 | 1.0000 |
| linear | 90 | 1.0000 | -2.4069 | 1.0000 |
| norm_cdf | 15 | -1.3606 | -1.4957 | -1.5971 |
| norm_pdf | 15 | -10.0000 | 0.4112 | 0.2163 |
| piecewise_linear | 5 | -1.3042 | 0.2341 | 0.1058 |
| quadratic_form | 15 | 1.0000 | -2.5949 | 0.6608 |
| rational | 120 | 1.0000 | -0.1696 | 1.0000 |
| rational_simple | 35 | 1.0000 | 0.8391 | 1.0000 |
| unknown | 0 | N/A | N/A | N/A |
| weighted_aggregate | 5 | -10.0000 | 0.9655 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | -29312708804846095353489900816048810028944901582156167468881910181329773900599056843985815901977043415173731128339562633078922756431022562642482074409485176674255493781619684801870355247314227469876039193410199939060385878775321172545372160.0000 | 0.0000 | 305 |
| Neural Net | 32.4368 | 1.5007 | 370 |
| Hybrid | 17.0933 | 0.0000 | 365 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 11.7459 | 10.5440 | 4345.97 | 370 |
| Neural Net | 0.3684 | 0.3680 | 136.31 | 370 |
| Hybrid | 3.7051 | 2.7555 | 1370.89 | 370 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| v4_llm | 233 |
| v4_nn | 98 |
| v4_residual_nn | 32 |
| v4_blend | 2 |

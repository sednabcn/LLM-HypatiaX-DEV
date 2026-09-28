
# HypatiaX Analysis Report — `exp1_pca`

Experiment mode: **standard**
N total: 74 | N standard: 74 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 74 | 14.9% | 91.7% | 1.0000 | 0.8080 |
| Neural Net | 74 | 100.0% | 12.2% | -0.6932 | -1.4961 |
| Hybrid | 74 | 98.7% | 90.5% | 1.0000 | 0.8943 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=430.5,  p=0.7710,  direction=b_greater,  n=(74, 12)

### Hybrid vs Neural Net

  U=5311.5,  p=0.0000**,  direction=a_greater,  n=(74, 74)

### Neural Net vs Pure LLM

  U=41.0,  p=0.0000**,  direction=b_greater,  n=(74, 12)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 74
Hybrid wins:  69  (93.2%)
NN wins:      0
Tied:         5

## Coverage Gaps (62 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| Value at Risk at 95% | easy | linear | N/A | N/A | -0.1908 | 1.0000 |
| Value at Risk at 99% | easy | linear | N/A | N/A | -3.0060 | 1.0000 |
| Partial liquidation amount | easy | linear | N/A | N/A | -3.1756 | 1.0000 |
| Simple Staking APY | easy | linear | N/A | N/A | -0.3073 | 1.0000 |
| Loan-to-Value | easy | rational_simple | N/A | N/A | 0.9758 | 1.0000 |
| Spot price from AMM | easy | rational_simple | N/A | N/A | -1175.3996 | 1.0000 |
| LP share percentage | easy | rational_simple | N/A | N/A | 0.7069 | 1.0000 |
| Funding rate cost | easy | linear | N/A | N/A | -0.4365 | 1.0000 |
| Slashing penalty | easy | linear | N/A | N/A | -2.9037 | 1.0000 |
| Protocol reserve accumulation | easy | linear | N/A | N/A | -2.4242 | 1.0000 |
| Cross-margin available balance | easy | linear | N/A | N/A | -0.1193 | 1.0000 |
| Realized PnL for long | easy | linear | N/A | N/A | 0.6322 | 1.0000 |
| LP fee earnings | easy | rational_simple | N/A | N/A | 0.9292 | 1.0000 |
| Multi-day Value at Risk | easy | algebraic | N/A | N/A | -0.4987 | 1.0000 |
| ES scaling for multi-day | easy | algebraic | N/A | N/A | -0.7015 | 1.0000 |
| Incremental VaR | easy | linear | N/A | N/A | -2.2912 | 1.0000 |
| Reserve ratio | easy | rational_simple | N/A | N/A | -0.1703 | 1.0000 |
| Impermanent loss breakeven fee rate | easy | rational_simple | N/A | N/A | -10.5287 | 1.0000 |
| Liquidation Price Long | medium | rational | N/A | N/A | -1.4658 | 1.0000 |
| Liquidation Price Short | medium | rational | N/A | N/A | -3.3805 | 1.0000 |
| Constant Product Price Impact | medium | rational | N/A | N/A | 0.9824 | 1.0000 |
| Effective Leverage | medium | rational | N/A | N/A | -3.6408 | 1.0000 |
| Compounding Staking Returns | medium | exponential | N/A | N/A | -2.3058 | 1.0000 |
| Portfolio Sharpe Ratio | medium | rational | N/A | N/A | -0.6394 | 1.0000 |
| Capital efficiency | medium | rational | N/A | N/A | -2.3826 | 1.0000 |
| AMM output amount | medium | rational | N/A | N/A | 0.0875 | 1.0000 |
| Price slippage percentage | medium | rational | N/A | N/A | 0.9798 | 1.0000 |
| Utilization rate of DeFi | medium | rational | N/A | N/A | 0.9873 | 1.0000 |
| Supply APY from borrow | medium | linear | N/A | N/A | -2.9355 | 1.0000 |
| Health factor | medium | rational | N/A | N/A | 0.9397 | 1.0000 |
| Impermanent loss percentage | medium | algebraic_with_sqrt | N/A | N/A | -2.1889 | 1.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -9.9118 | 0.7709 |
| Collateral ratio | medium | rational | N/A | N/A | 0.9625 | 1.0000 |
| Funding rate cost (extended) | medium | rational | N/A | N/A | -0.0582 | 1.0000 |
| Staking reward for fixed lock-up | medium | rational | N/A | N/A | -0.1565 | 1.0000 |
| Expected Shortfall at 95% | medium | linear | N/A | N/A | -0.1885 | 1.0000 |
| Expected Shortfall at 99% | medium | linear | N/A | N/A | -3.0060 | 1.0000 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Concentrated liquidity position width | medium | rational | N/A | N/A | 0.9309 | 1.0000 |
| Portfolio VaR for two correlated | medium | algebraic | N/A | N/A | -2.1870 | 1.0000 |
| Position margin ratio | medium | rational | N/A | N/A | -7.3303 | 1.0000 |
| Concentrated liquidity position width (v2) | medium | algebraic_with_sqrt | N/A | N/A | 0.0788 | 1.0000 |
| Spot price from AMM reserve | medium | rational | N/A | N/A | -1175.3996 | 1.0000 |
| Black-Scholes Call Price | hard | norm_cdf | N/A | N/A | -1.8505 | -1.8505 |
| Black-Scholes Put Price | hard | norm_cdf | N/A | N/A | -0.6850 | 0.5415 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.1921 | 1.0000 |
| Gamma of option | hard | norm_pdf | N/A | N/A | -0.1060 | -0.1060 |
| Vega of option | hard | norm_pdf | N/A | N/A | 0.6285 | 0.6285 |
| Multi-Collateral LTV | hard | weighted_aggregate | N/A | N/A | 0.7101 | 1.0000 |
| Impermanent loss in constant product | hard | algebraic_with_sqrt | N/A | N/A | -1.2321 | 1.0000 |
| Constant product formula (multivariate) | hard | rational | N/A | N/A | 0.2765 | 1.0000 |
| Convexity Adjustment | hard | algebraic | N/A | N/A | -2.6261 | 1.0000 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.2764 | 1.0000 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.4908 | 1.0000 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.6330 | 1.0000 |
| Required collateral | hard | rational | N/A | N/A | -1.2015 | 1.0000 |
| Forward price for derivative | hard | exponential | N/A | N/A | -2.1505 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.1273 | 1.0000 |
| Put-call parity | hard | exponential | N/A | N/A | -2.4539 | 1.0000 |
| Simple options moneyness | hard | rational | N/A | N/A | -2.2101 | 1.0000 |
| Uniswap V3 virtual | hard | algebraic_with_sqrt | N/A | N/A | -0.1017 | 1.0000 |
| Theta of option | hard | norm_pdf | N/A | N/A | -0.8034 | -0.8034 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 24 | 100.0% | 12.5% | 100.0% |
| hard | 21 | 50.0% | 0.0% | 76.2% |
| medium | 29 | 100.0% | 20.7% | 93.1% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 5 | 1.0000 | -2.1870 | 1.0000 |
| algebraic_with_sqrt | 4 | N/A | -0.6669 | 1.0000 |
| exponential | 5 | 1.0000 | -2.3058 | 1.0000 |
| linear | 18 | 1.0000 | -2.3577 | 1.0000 |
| norm_cdf | 3 | N/A | -1.8505 | 0.5415 |
| norm_pdf | 3 | N/A | -0.1060 | -0.1060 |
| piecewise_linear | 1 | -1.3042 | 0.1272 | 1.0000 |
| quadratic_form | 3 | 1.0000 | -2.1273 | 1.0000 |
| rational | 24 | 1.0000 | -0.1074 | 1.0000 |
| rational_simple | 7 | 1.0000 | 0.7069 | 1.0000 |
| weighted_aggregate | 1 | N/A | 0.7101 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | -1.0810 | 0.0000 | 12 |
| Neural Net | 33.9960 | 1.6737 | 74 |
| Hybrid | -0.1347 | 0.0000 | 73 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 8.3688 | 7.6295 | 619.29 | 74 |
| Neural Net | 0.4156 | 0.4140 | 30.75 | 74 |
| Hybrid | 1.7922 | 1.3720 | 132.62 | 74 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| llm | 69 |
| nn | 4 |
| nn_fallback | 1 |

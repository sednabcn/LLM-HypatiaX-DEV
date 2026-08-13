
# HypatiaX Analysis Report — `exp1b`

Experiment mode: **standard**
N total: 72 | N standard: 72 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 72 | 59.7% | 74.1% | 1.0000 | -0.8916 |
| Neural Net | 72 | 100.0% | 15.3% | -0.4684 | -1.2739 |
| Hybrid | 72 | 77.8% | 75.0% | 1.0000 | 0.2552 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=2128.0,  p=0.8296,  direction=b_greater,  n=(72, 58)

### Hybrid vs Neural Net

  U=4446.5,  p=0.0000**,  direction=a_greater,  n=(72, 72)

### Neural Net vs Pure LLM

  U=922.0,  p=0.0000**,  direction=b_greater,  n=(72, 58)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 72
Hybrid wins:  53  (73.6%)
NN wins:      0
Tied:         19

## Coverage Gaps (22 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| Black-Scholes Call Price | hard | norm_cdf | -1.0637 | -193.6432 | -1.0637 | -1.0637 |
| Black-Scholes Put Price | hard | norm_cdf | 0.5415 | -1.3606 | 0.1922 | 0.5415 |
| Capital efficiency | medium | rational | N/A | N/A | -2.1446 | -2.1446 |
| Component ES | hard | quadratic_form | -0.2608 | -0.2749 | -0.2608 | -0.2608 |
| Compounding Staking Returns | medium | exponential | -0.2406 | -0.2406 | -2.4114 | -2.4114 |
| Gamma of option | hard | norm_pdf | 0.4325 | -4.3964 | 0.4325 | 0.4325 |
| Impermanent loss percentage | medium | algebraic_with_sqrt | N/A | N/A | -2.0011 | 1.0000 |
| Liquidation Price Long | medium | rational | N/A | N/A | 0.6053 | 0.9927 |
| Liquidation Price Short | medium | rational | N/A | N/A | -3.1983 | 0.9832 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | -0.0248 | -0.0248 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.7713 | -1.7713 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.7419 | -1.7419 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -10.5222 | -10.5222 |
| Required collateral | hard | rational | N/A | N/A | -1.4092 | 1.0000 |
| Short position unrealized PnL | easy | linear | N/A | N/A | 0.5610 | 1.0000 |
| Slashing penalty | easy | linear | N/A | N/A | -2.8448 | -2.8448 |
| Theta of option | hard | norm_pdf | -0.6818 | -16.0271 | -0.6818 | -0.6818 |
| Uniswap V3 virtual | hard | algebraic_with_sqrt | 0.6678 | -9408081.2323 | 0.6678 | 0.6678 |
| Utilization rate of DeFi | medium | rational | N/A | N/A | 0.9838 | 1.0000 |
| Vega of option | hard | norm_pdf | 0.5988 | -18.5929 | 0.5988 | 0.5988 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.9789 | -0.1389 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 24 | 90.9% | 12.5% | 91.7% |
| hard | 20 | 40.0% | 5.0% | 45.0% |
| medium | 28 | 81.0% | 25.0% | 82.1% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 4 | 1.0000 | -1.4860 | 1.0000 |
| algebraic_with_sqrt | 4 | 1.0000 | -0.3989 | 1.0000 |
| exponential | 5 | 1.0000 | -2.4114 | 1.0000 |
| linear | 18 | 1.0000 | -2.7114 | 1.0000 |
| norm_cdf | 3 | -5.6803 | -1.0637 | -1.0637 |
| norm_pdf | 3 | -10.0000 | 0.4325 | 0.4325 |
| piecewise_linear | 1 | -3.5798 | 0.7504 | 1.0000 |
| quadratic_form | 2 | -0.2749 | -1.6199 | -0.1999 |
| rational | 24 | 1.0000 | -0.0124 | 1.0000 |
| rational_simple | 7 | 1.0000 | -0.1223 | 1.0000 |
| weighted_aggregate | 1 | 1.0000 | 0.9802 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | 42645.5050 | 0.0000 | 58 |
| Neural Net | 3.4811 | 1.4681 | 72 |
| Hybrid | -1046868.0563 | 0.0000 | 69 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 9.4272 | 8.3680 | 678.76 | 72 |
| Neural Net | 0.3593 | 0.3580 | 25.87 | 72 |
| Hybrid | 1.9414 | 1.4260 | 139.78 | 72 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| llm | 53 |
| nn | 19 |


# HypatiaX Analysis Report — `exp1`

Experiment mode: **standard**
N total: 74 | N standard: 74 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 74 | 62.2% | 76.7% | 1.0000 | -0.7123 |
| Neural Net | 74 | 23.0% | 14.9% | -0.6908 | -1.4155 |
| Hybrid | 74 | 78.4% | 74.0% | 1.0000 | 0.1376 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=2124.5,  p=0.7193,  direction=b_greater,  n=(73, 60)

### Hybrid vs Neural Net

  U=4625.5,  p=0.0000**,  direction=a_greater,  n=(73, 74)

### Neural Net vs Pure LLM

  U=899.0,  p=0.0000**,  direction=b_greater,  n=(74, 60)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 73
Hybrid wins:  63  (86.3%)
NN wins:      9
Tied:         1

## Coverage Gaps (20 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| Spot price from AMM | easy | rational_simple | -148.2832 | -519.5690 | -148.2832 | -247.8645 |
| Slashing penalty | easy | linear | -2.4220 | -4.7088 | -2.9037 | -2.4220 |
| Impermanent loss breakeven fee rate | easy | rational_simple | N/A | N/A | -10.5288 | -35.5347 |
| Liquidation Price Long | medium | rational | N/A | N/A | 0.8624 | 0.9993 |
| Liquidation Price Short | medium | rational | N/A | N/A | -3.3805 | 0.9999 |
| Compounding Staking Returns | medium | exponential | -0.2206 | -0.2206 | -2.3058 | -2.0945 |
| Capital efficiency | medium | rational | N/A | N/A | -2.3826 | -2.2776 |
| Options Delta | medium | norm_cdf | N/A | N/A | -9.9026 | -10.2587 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | 0.2464 | 0.2464 | -1.8315 | -1.5971 |
| Black-Scholes Put Price | hard | norm_cdf | N/A | N/A | -0.6802 | 0.6078 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.1921 | -0.2104 |
| Gamma of option | hard | norm_pdf | 0.2307 | -4.3964 | -0.1060 | 0.2307 |
| Convexity Adjustment | hard | algebraic | N/A | N/A | -2.6261 | 1.0000 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.2314 | -0.1113 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.4907 | -1.4817 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.6330 | N/A |
| Required collateral | hard | rational | N/A | N/A | -1.2015 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.1273 | 0.6608 |
| Theta of option | hard | norm_pdf | -0.8034 | -15.9406 | -0.8034 | -1.1756 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 24 | 78.3% | 12.5% | 87.5% |
| hard | 21 | 61.5% | 0.0% | 50.0% |
| medium | 29 | 83.3% | 27.6% | 79.3% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 5 | 1.0000 | -2.1870 | 1.0000 |
| algebraic_with_sqrt | 4 | 1.0000 | -0.3526 | 0.8283 |
| exponential | 5 | 1.0000 | -2.3058 | 1.0000 |
| linear | 18 | 1.0000 | -2.3577 | 1.0000 |
| norm_cdf | 3 | 0.2464 | -1.8315 | -1.5971 |
| norm_pdf | 3 | -4.3964 | -0.1060 | 0.2307 |
| piecewise_linear | 1 | -3.5798 | 0.7545 | 1.0000 |
| quadratic_form | 3 | 1.0000 | -2.1273 | 0.6608 |
| rational | 24 | 1.0000 | -0.0396 | 1.0000 |
| rational_simple | 7 | 1.0000 | -0.1295 | 1.0000 |
| weighted_aggregate | 1 | -10.0000 | 0.6311 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | 8598.7594 | 0.0000 | 60 |
| Neural Net | 4.2864 | 1.6713 | 74 |
| Hybrid | 4.4676 | 0.0000 | 73 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 11.5811 | 10.3820 | 857.0 | 74 |
| Neural Net | 0.3393 | 0.3380 | 25.11 | 74 |
| Hybrid | 3.3456 | 2.6310 | 247.57 | 74 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| v4_llm | 47 |
| v4_nn | 20 |
| v4_residual_nn | 6 |

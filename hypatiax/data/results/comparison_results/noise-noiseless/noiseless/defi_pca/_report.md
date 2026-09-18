
# HypatiaX Analysis Report — `exp1_pca`

Experiment mode: **standard**
N total: 74 | N standard: 74 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 74 | 63.5% | 75.8% | 1.0000 | -0.8232 |
| Neural Net | 74 | 17.6% | 12.2% | -0.6598 | -1.3643 |
| Hybrid | 74 | 77.0% | 74.0% | 1.0000 | 0.2707 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=2203.5,  p=0.7526,  direction=b_greater,  n=(73, 62)

### Hybrid vs Neural Net

  U=4667.5,  p=0.0000**,  direction=a_greater,  n=(73, 74)

### Neural Net vs Pure LLM

  U=971.0,  p=0.0000**,  direction=b_greater,  n=(74, 62)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 73
Hybrid wins:  65  (89.0%)
NN wins:      7
Tied:         1

## Coverage Gaps (21 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| Spot price from AMM | easy | rational_simple | -873.1997 | -32651.6732 | -1209.8414 | -873.1997 |
| Slashing penalty | easy | linear | -2.4220 | -4.7088 | -2.9037 | -2.4220 |
| Impermanent loss breakeven fee rate | easy | rational_simple | N/A | N/A | -10.5287 | 1.0000 |
| Liquidation Price Long | medium | rational | N/A | N/A | -1.4658 | 0.9825 |
| Liquidation Price Short | medium | rational | N/A | N/A | -3.3805 | 0.9999 |
| Compounding Staking Returns | medium | exponential | -0.2206 | -0.2206 | -2.3058 | -2.0945 |
| Capital efficiency | medium | rational | N/A | N/A | -2.3826 | -2.2776 |
| Options Delta | medium | norm_cdf | N/A | N/A | -9.9026 | -10.2587 |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Black-Scholes Call Price | hard | norm_cdf | -1.5971 | -193.6432 | -1.8315 | -1.5971 |
| Black-Scholes Put Price | hard | norm_cdf | 0.6078 | -1.3606 | -0.6802 | 0.6078 |
| Component ES | hard | quadratic_form | N/A | N/A | -0.1921 | -0.2104 |
| Gamma of option | hard | norm_pdf | 0.2307 | -4.3964 | -0.1060 | 0.2307 |
| Vega of option | hard | norm_pdf | 0.6891 | -18.5929 | 0.6285 | 0.6891 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | 0.2764 | 0.4698 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.4907 | -1.4817 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.6330 | N/A |
| Required collateral | hard | rational | N/A | N/A | -1.2015 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.1273 | 0.6608 |
| Call option intrinsic | hard | piecewise_linear | 0.1272 | -1.3042 | 0.1272 | 0.1058 |
| Theta of option | hard | norm_pdf | -0.8034 | -15.9373 | -0.8034 | -1.1756 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 24 | 82.6% | 12.5% | 91.7% |
| hard | 21 | 53.3% | 0.0% | 45.0% |
| medium | 29 | 83.3% | 20.7% | 79.3% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 5 | 1.0000 | -2.1870 | 1.0000 |
| algebraic_with_sqrt | 4 | 1.0000 | -0.6669 | 0.4313 |
| exponential | 5 | 1.0000 | -2.3058 | 1.0000 |
| linear | 18 | 1.0000 | -2.3577 | 1.0000 |
| norm_cdf | 3 | -5.6803 | -1.8315 | -1.5971 |
| norm_pdf | 3 | -10.0000 | -0.1060 | 0.2307 |
| piecewise_linear | 1 | -1.3042 | 0.1272 | 0.1058 |
| quadratic_form | 3 | 1.0000 | -2.1273 | 0.6608 |
| rational | 24 | 1.0000 | -0.1071 | 1.0000 |
| rational_simple | 7 | 1.0000 | 0.7069 | 1.0000 |
| weighted_aggregate | 1 | -10.0000 | 0.7085 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | -495.4712 | 0.0000 | 62 |
| Neural Net | 18.5805 | 1.6402 | 74 |
| Hybrid | 12.5511 | 0.0000 | 73 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 11.3896 | 10.2800 | 842.83 | 74 |
| Neural Net | 0.3669 | 0.3660 | 27.15 | 74 |
| Hybrid | 3.4426 | 2.7295 | 254.75 | 74 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| v4_llm | 46 |
| v4_nn | 20 |
| v4_residual_nn | 7 |

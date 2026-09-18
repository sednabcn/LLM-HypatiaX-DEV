
# HypatiaX Analysis Report — `exp1b`

Experiment mode: **standard**
N total: 72 | N standard: 72 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 72 | 63.9% | 80.7% | 1.0000 | -0.5218 |
| Neural Net | 72 | 25.0% | 15.3% | -0.4684 | -1.2739 |
| Hybrid | 72 | 79.2% | 74.7% | 1.0000 | 0.3833 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=1913.5,  p=0.5157,  direction=b_greater,  n=(71, 57)

### Hybrid vs Neural Net

  U=4417.0,  p=0.0000**,  direction=a_greater,  n=(71, 72)

### Neural Net vs Pure LLM

  U=716.0,  p=0.0000**,  direction=b_greater,  n=(72, 57)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 71
Hybrid wins:  62  (87.3%)
NN wins:      8
Tied:         1

## Coverage Gaps (19 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| Black-Scholes Call Price | hard | norm_cdf | N/A | N/A | -1.0637 | -1.5196 |
| Black-Scholes Put Price | hard | norm_cdf | N/A | N/A | 0.1922 | 0.5160 |
| Capital efficiency | medium | rational | N/A | N/A | -2.1446 | -2.4468 |
| Component ES | hard | quadratic_form | -0.2608 | -0.2749 | -0.2608 | -0.3325 |
| Gamma of option | hard | norm_pdf | 0.4325 | -4.3964 | 0.4325 | -0.0218 |
| Liquidation Price Long | medium | rational | N/A | N/A | 0.6053 | 0.9996 |
| Liquidation Price Short | medium | rational | N/A | N/A | -3.1983 | 0.9998 |
| Liquidation price for leveraged long | hard | rational | N/A | N/A | -0.0248 | -0.1143 |
| Liquidation price for leveraged short | hard | rational | N/A | N/A | -1.7713 | -1.6460 |
| Maximum safe leverage | hard | rational | N/A | N/A | -1.7419 | N/A |
| Optimal LP Position (Kelly) | medium | rational | N/A | N/A | 0.0000 | 0.0000 |
| Options Delta | medium | norm_cdf | N/A | N/A | -10.5222 | -10.8407 |
| Required collateral | hard | rational | N/A | N/A | -1.4092 | 1.0000 |
| Simple Staking APY | easy | linear | N/A | N/A | -0.3627 | 1.0000 |
| Slashing penalty | easy | linear | -2.2875 | -4.7088 | -2.8448 | -2.2875 |
| Spot price from AMM reserve | medium | rational | N/A | N/A | 0.6356 | 1.0000 |
| Theta of option | hard | norm_pdf | N/A | N/A | -0.6818 | -0.6250 |
| Vega of option | hard | norm_pdf | 0.6026 | -18.5929 | 0.5988 | 0.6026 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.9789 | 0.6624 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 24 | 87.0% | 12.5% | 91.7% |
| hard | 20 | 58.3% | 5.0% | 47.4% |
| medium | 28 | 86.4% | 25.0% | 78.6% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 4 | 1.0000 | -1.4860 | 1.0000 |
| algebraic_with_sqrt | 4 | 1.0000 | -0.3989 | 0.8408 |
| exponential | 5 | 1.0000 | -2.4114 | 1.0000 |
| linear | 18 | 1.0000 | -2.7114 | 1.0000 |
| norm_cdf | 3 | N/A | -1.0637 | -1.5196 |
| norm_pdf | 3 | -7.1982 | 0.4325 | -0.0218 |
| piecewise_linear | 1 | -3.5798 | 0.7504 | 1.0000 |
| quadratic_form | 2 | -0.2749 | -1.6199 | 0.1649 |
| rational | 24 | 1.0000 | -0.0124 | 1.0000 |
| rational_simple | 7 | 1.0000 | -0.1223 | 1.0000 |
| weighted_aggregate | 1 | -10.0000 | 0.9802 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | -113.8501 | 0.0000 | 57 |
| Neural Net | 3.4811 | 1.4681 | 72 |
| Hybrid | 0.6225 | 0.0000 | 71 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 11.4525 | 10.7750 | 824.58 | 72 |
| Neural Net | 0.3446 | 0.3440 | 24.81 | 72 |
| Hybrid | 3.3306 | 2.6525 | 239.81 | 72 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| v4_llm | 46 |
| v4_nn | 19 |
| v4_residual_nn | 6 |

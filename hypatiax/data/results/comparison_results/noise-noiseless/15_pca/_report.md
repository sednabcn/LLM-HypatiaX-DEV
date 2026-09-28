
# HypatiaX Analysis Report — `exp1b_pca`

Experiment mode: **standard**
N total: 25 | N standard: 25 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 25 | 44.0% | 100.0% | 1.0000 | 1.0000 |
| Neural Net | 25 | 100.0% | 0.0% | -2.6696 | -2.3210 |
| Hybrid | 25 | 100.0% | 100.0% | 1.0000 | 1.0000 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=122.5,  p=0.4406,  direction=b_greater,  n=(25, 11)

### Hybrid vs Neural Net

  U=625.0,  p=0.0000**,  direction=a_greater,  n=(25, 25)

### Neural Net vs Pure LLM

  U=0.0,  p=0.0000**,  direction=b_greater,  n=(25, 11)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 25
Hybrid wins:  25  (100.0%)
NN wins:      0
Tied:         0

## Coverage Gaps (14 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| Portfolio Sharpe Ratio | medium | rational | N/A | N/A | -0.5029 | 1.0000 |
| Portfolio VaR for two correlated | medium | algebraic | N/A | N/A | -3.0593 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.9789 | 1.0000 |
| Portfolio Sharpe Ratio | medium | rational | N/A | N/A | -0.1304 | 1.0000 |
| Portfolio VaR for two correlated | medium | algebraic | N/A | N/A | -3.2380 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -3.1512 | 1.0000 |
| Portfolio VaR for two correlated | medium | algebraic | N/A | N/A | -2.1870 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.1273 | 1.0000 |
| Portfolio Sharpe Ratio | medium | rational | N/A | N/A | -0.7173 | 1.0000 |
| Portfolio VaR for two correlated | medium | algebraic | N/A | N/A | -2.6696 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.5949 | 1.0000 |
| Portfolio Sharpe Ratio | medium | rational | N/A | N/A | -0.2796 | 1.0000 |
| Portfolio VaR for two correlated | medium | algebraic | N/A | N/A | -2.7807 | 1.0000 |
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.7078 | 1.0000 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 5 | 100.0% | 0.0% | 100.0% |
| hard | 10 | 100.0% | 0.0% | 100.0% |
| medium | 10 | 100.0% | 0.0% | 100.0% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 10 | 1.0000 | -2.7880 | 1.0000 |
| quadratic_form | 10 | 1.0000 | -2.7443 | 1.0000 |
| rational | 5 | 1.0000 | -0.5029 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | 0.0000 | 0.0000 | 11 |
| Neural Net | 3.3209 | 3.6695 | 25 |
| Hybrid | 0.0000 | 0.0000 | 25 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 8.2271 | 7.7170 | 205.68 | 25 |
| Neural Net | 0.3699 | 0.3690 | 9.25 | 25 |
| Hybrid | 1.5553 | 1.4990 | 38.88 | 25 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| llm | 25 |

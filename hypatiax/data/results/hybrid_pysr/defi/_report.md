
# HypatiaX Analysis Report — `suppA`

Experiment mode: **standard**
N total: 73 | N standard: 73 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 73 | 75.3% | 79.6% | 1.0000 | -0.5402 |
| Neural Net | 73 | 100.0% | 21.9% | 0.1364 | -0.3618 |
| Hybrid | 73 | 100.0% | 74.0% | 1.0000 | 0.1842 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=1862.5,  p=0.5349,  direction=b_greater,  n=(73, 54)

### Hybrid vs Neural Net

  U=4408.0,  p=0.0000**,  direction=a_greater,  n=(73, 73)

### Neural Net vs Pure LLM

  U=770.0,  p=0.0000**,  direction=b_greater,  n=(73, 54)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 73
Hybrid wins:  47  (64.4%)
NN wins:      0
Tied:         26

## Coverage Gaps (24 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| ? | easy | rational_simple | N/A | N/A | 0.9720 | 0.9720 |
| ? | easy | rational_simple | N/A | N/A | -0.0067 | -0.0067 |
| ? | easy | rational_simple | N/A | N/A | 0.9816 | 1.0000 |
| ? | easy | rational_simple | N/A | N/A | -854.7111 | -854.7111 |
| ? | easy | linear | N/A | N/A | -0.0773 | 1.0000 |
| ? | easy | linear | N/A | N/A | 0.1364 | 1.0000 |
| ? | easy | linear | N/A | N/A | -0.9074 | -0.9074 |
| ? | easy | linear | N/A | N/A | 0.6364 | 1.0000 |
| ? | medium | rational | N/A | N/A | 0.7984 | 0.7984 |
| ? | medium | rational | N/A | N/A | 0.0338 | 0.0338 |
| ? | medium | rational | N/A | N/A | -0.0280 | -0.0280 |
| ? | medium | rational | N/A | N/A | 0.9419 | 1.0000 |
| ? | medium | rational | -933.4692 | -6512458284.6293 | -933.4692 | -933.4692 |
| ? | medium | linear | 0.7675 | -0.0808 | 0.7675 | 0.7675 |
| ? | medium | linear | -0.1578 | -3.3277 | -0.1578 | -0.1578 |
| ? | hard | rational_with_min | N/A | N/A | 0.0000 | 0.0000 |
| ? | hard | transcendental | N/A | N/A | -0.0394 | -0.0394 |
| ? | hard | transcendental | -2.1172 | -193.6432 | -2.1172 | -2.1172 |
| ? | hard | polynomial | N/A | N/A | 0.3545 | 0.3545 |
| ? | hard | rational | N/A | N/A | 0.6840 | 0.6840 |
| ? | hard | rational | N/A | N/A | -0.3107 | -0.3107 |
| ? | hard | rational | N/A | N/A | 0.9627 | 0.9627 |
| ? | hard | rational | N/A | N/A | 0.9697 | 0.9697 |
| ? | hard | piecewise_linear | 0.6568 | -3.5798 | 0.6568 | 0.6568 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 24 | 100.0% | 16.7% | 83.3% |
| hard | 20 | 61.5% | 15.0% | 65.0% |
| medium | 29 | 76.0% | 31.0% | 72.4% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 4 | 1.0000 | 0.0851 | 1.0000 |
| algebraic_with_sqrt | 3 | -2.1798 | -0.1818 | 1.0000 |
| exponential | 6 | 1.0000 | -0.1336 | 1.0000 |
| linear | 23 | 1.0000 | -0.1157 | 1.0000 |
| piecewise_linear | 2 | -1.2899 | 0.7051 | 0.8284 |
| polynomial | 1 | N/A | 0.3545 | 0.3545 |
| quadratic_form | 1 | 1.0000 | -0.4882 | 1.0000 |
| rational | 23 | 1.0000 | 0.6840 | 0.9913 |
| rational_simple | 6 | 0.9814 | 0.9271 | 0.9860 |
| rational_with_min | 1 | N/A | 0.0000 | 0.0000 |
| transcendental | 2 | -10.0000 | -1.0783 | -1.0783 |
| weighted_aggregate | 1 | 1.0000 | 0.1941 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | 121252668.6009 | 0.0000 | 54 |
| Neural Net | 40.3517 | 0.8352 | 73 |
| Hybrid | 39.8116 | -0.0000 | 73 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | N/A | N/A | None | 0 |
| Neural Net | N/A | N/A | None | 0 |
| Hybrid | N/A | N/A | None | 0 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| ensemble | 52 |
| nn | 18 |
| fitted_llm | 3 |

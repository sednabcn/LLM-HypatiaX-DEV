
# HypatiaX Analysis Report — `suppA`

Experiment mode: **standard**
N total: 73 | N standard: 73 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 73 | 82.2% | 85.0% | 1.0000 | -0.2233 |
| Neural Net | 73 | 100.0% | 21.9% | 0.1364 | -0.3618 |
| Hybrid | 73 | 100.0% | 21.9% | 0.1364 | -0.3618 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=631.0,  p=0.0000**,  direction=b_greater,  n=(73, 60)

### Hybrid vs Neural Net

  U=2664.5,  p=1.0000,  direction=b_greater,  n=(73, 73)

### Neural Net vs Pure LLM

  U=631.0,  p=0.0000**,  direction=b_greater,  n=(73, 60)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 73
Hybrid wins:  0  (0.0%)
NN wins:      0
Tied:         73

## Coverage Gaps (18 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| ? | easy | rational_simple | -519.5690 | -519.5690 | -854.7111 | -854.7111 |
| ? | easy | linear | -0.9074 | -4.7088 | -0.9074 | -0.9074 |
| ? | medium | rational | N/A | N/A | 0.7984 | 0.7984 |
| ? | medium | rational | N/A | N/A | 0.0338 | 0.0338 |
| ? | medium | rational | N/A | N/A | -0.0280 | -0.0280 |
| ? | medium | linear | N/A | N/A | 0.8153 | 0.8153 |
| ? | medium | linear | N/A | N/A | 0.7675 | 0.7675 |
| ? | medium | linear | -0.1578 | -3.3277 | -0.1578 | -0.1578 |
| ? | hard | rational_with_min | N/A | N/A | 0.0000 | 0.0000 |
| ? | hard | transcendental | N/A | N/A | -0.0394 | -0.0394 |
| ? | hard | transcendental | N/A | N/A | -2.1172 | -2.1172 |
| ? | hard | weighted_aggregate | 0.1941 | -4032.9398 | 0.1941 | 0.1941 |
| ? | hard | rational | N/A | N/A | 0.6840 | 0.6840 |
| ? | hard | rational | N/A | N/A | -0.3107 | -0.3107 |
| ? | hard | rational | N/A | N/A | 0.9627 | 0.9627 |
| ? | hard | rational | N/A | N/A | 0.9697 | 0.9697 |
| ? | hard | piecewise_linear | 0.6568 | -3.5798 | 0.6568 | 0.6568 |
| ? | hard | algebraic_with_sqrt | N/A | N/A | 0.8320 | 0.8320 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 24 | 83.3% | 16.7% | 16.7% |
| hard | 20 | 83.3% | 15.0% | 15.0% |
| medium | 29 | 87.5% | 31.0% | 31.0% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 4 | 1.0000 | 0.0851 | 0.0851 |
| algebraic_with_sqrt | 3 | 1.0000 | -0.1818 | -0.1818 |
| exponential | 6 | 1.0000 | -0.1336 | -0.1336 |
| linear | 23 | 1.0000 | -0.1157 | -0.1157 |
| piecewise_linear | 2 | -1.2899 | 0.7051 | 0.7051 |
| polynomial | 1 | 1.0000 | 0.3545 | 0.3545 |
| quadratic_form | 1 | 1.0000 | -0.4882 | -0.4882 |
| rational | 23 | 1.0000 | 0.6840 | 0.6840 |
| rational_simple | 6 | -4.5000 | 0.9271 | 0.9271 |
| rational_with_min | 1 | N/A | 0.0000 | 0.0000 |
| transcendental | 2 | N/A | -1.0783 | -1.0783 |
| weighted_aggregate | 1 | -10.0000 | 0.1941 | 0.1941 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | -108.3375 | 0.0000 | 60 |
| Neural Net | 40.3517 | 0.8352 | 73 |
| Hybrid | 40.3553 | 0.8636 | 73 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | N/A | N/A | None | 0 |
| Neural Net | N/A | N/A | None | 0 |
| Hybrid | N/A | N/A | None | 0 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| ensemble | 62 |
| nn | 11 |

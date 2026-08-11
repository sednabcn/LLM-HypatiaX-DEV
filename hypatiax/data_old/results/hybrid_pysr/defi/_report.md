
# HypatiaX Analysis Report — `suppA`

Experiment mode: **standard**
N total: 73 | N standard: 73 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 73 | 79.5% | 79.3% | 1.0000 | -0.6176 |
| Neural Net | 73 | 100.0% | 21.9% | 0.1364 | -0.3628 |
| Hybrid | 73 | 100.0% | 79.5% | 1.0000 | 0.4968 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=2043.0,  p=0.6810,  direction=b_greater,  n=(73, 58)

### Hybrid vs Neural Net

  U=4584.0,  p=0.0000**,  direction=a_greater,  n=(73, 73)

### Neural Net vs Pure LLM

  U=834.0,  p=0.0000**,  direction=b_greater,  n=(73, 58)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 73
Hybrid wins:  53  (72.6%)
NN wins:      0
Tied:         20

## Coverage Gaps (19 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| ? | easy | rational_simple | N/A | N/A | 0.0280 | 1.0000 |
| ? | easy | linear | N/A | N/A | 0.1364 | 1.0000 |
| ? | easy | linear | N/A | N/A | -0.9074 | -0.9074 |
| ? | medium | rational | N/A | N/A | 0.7984 | 0.9927 |
| ? | medium | rational | N/A | N/A | 0.0338 | 0.9832 |
| ? | medium | rational | N/A | N/A | -0.0280 | -0.0280 |
| ? | medium | rational | N/A | N/A | 0.9742 | 1.0000 |
| ? | medium | rational | -750.2297 | -6512458284.6293 | -750.2297 | -750.2297 |
| ? | medium | linear | 0.7675 | -0.0808 | 0.7675 | 0.7675 |
| ? | medium | linear | -0.1578 | -3.3277 | -0.1578 | -0.1578 |
| ? | hard | algebraic_with_sqrt | N/A | N/A | -0.3011 | 1.0000 |
| ? | hard | rational_with_min | N/A | N/A | 0.0000 | 0.0000 |
| ? | hard | transcendental | N/A | N/A | -0.0394 | -0.0394 |
| ? | hard | transcendental | N/A | N/A | -2.2553 | -2.2553 |
| ? | hard | rational | N/A | N/A | 0.6840 | 0.6840 |
| ? | hard | rational | N/A | N/A | -0.3107 | -0.3107 |
| ? | hard | rational | N/A | N/A | 0.9627 | 0.9627 |
| ? | hard | rational | N/A | N/A | 0.9697 | 1.0000 |
| ? | hard | piecewise_linear | 0.6568 | -3.5798 | 0.6568 | 0.6568 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 24 | 85.7% | 16.7% | 87.5% |
| hard | 20 | 75.0% | 15.0% | 65.0% |
| medium | 29 | 76.0% | 31.0% | 82.8% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 4 | 1.0000 | 0.0851 | 1.0000 |
| algebraic_with_sqrt | 3 | -4.5000 | -0.1818 | 1.0000 |
| exponential | 6 | 1.0000 | -0.1336 | 1.0000 |
| linear | 23 | 1.0000 | -0.1157 | 1.0000 |
| piecewise_linear | 2 | -1.2899 | 0.7047 | 0.8284 |
| polynomial | 1 | 1.0000 | 0.3545 | 0.3545 |
| quadratic_form | 1 | 1.0000 | -0.4882 | 1.0000 |
| rational | 23 | 1.0000 | 0.6840 | 1.0000 |
| rational_simple | 6 | -1.8678 | 0.9275 | 1.0000 |
| rational_with_min | 1 | N/A | 0.0000 | 0.0000 |
| transcendental | 2 | N/A | -1.1473 | -1.1473 |
| weighted_aggregate | 1 | 1.0000 | 0.1941 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | 112890224.1116 | 0.0000 | 58 |
| Neural Net | 37.3667 | 0.8636 | 73 |
| Hybrid | 22.2106 | -0.0000 | 73 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | N/A | N/A | None | 0 |
| Neural Net | N/A | N/A | None | 0 |
| Hybrid | N/A | N/A | None | 0 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| ensemble | 61 |
| nn | 12 |

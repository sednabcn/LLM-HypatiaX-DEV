
# HypatiaX Analysis Report — `exp1`

Experiment mode: **standard**
N total: 15 | N standard: 15 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 3 | 66.7% | 100.0% | 1.0000 | 1.0000 |
| Neural Net | 3 | 100.0% | 0.0% | -2.4931 | -1.9898 |
| Hybrid | 3 | 100.0% | 66.7% | 1.0000 | 0.7920 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=2.0,  p=0.7609,  direction=b_greater,  n=(3, 2)

### Hybrid vs Neural Net

  U=9.0,  p=0.1000,  direction=a_greater,  n=(3, 3)

### Neural Net vs Pure LLM

  U=0.0,  p=0.2000,  direction=b_greater,  n=(3, 2)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 3
Hybrid wins:  3  (100.0%)
NN wins:      0
Tied:         0

## Coverage Gaps (13 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.4931 | 0.3760 |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |
| ? | None | None | N/A | N/A | N/A | N/A |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 1 | 100.0% | 0.0% | 100.0% |
| hard | 1 | 0.0% | 0.0% | 0.0% |
| medium | 1 | 100.0% | 0.0% | 100.0% |
| unknown | 0 | 0.0% | 0.0% | 0.0% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 1 | 1.0000 | -3.1838 | 1.0000 |
| quadratic_form | 1 | N/A | -2.4931 | 0.3760 |
| rational | 1 | 1.0000 | -0.2925 | 1.0000 |
| unknown | 0 | N/A | N/A | N/A |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | 0.0000 | 0.0000 | 2 |
| Neural Net | 2.9897 | 3.4930 | 3 |
| Hybrid | 0.2028 | 0.0000 | 3 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 13.9910 | 13.5680 | 41.97 | 3 |
| Neural Net | 10.4127 | 2.8090 | 31.24 | 3 |
| Hybrid | 1.6397 | 1.4520 | 4.92 | 3 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| llm | 3 |

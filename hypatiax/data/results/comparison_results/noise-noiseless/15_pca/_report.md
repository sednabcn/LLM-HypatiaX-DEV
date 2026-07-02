
# HypatiaX Analysis Report — `exp1b_pca`

Experiment mode: **standard**
N total: 5 | N standard: 5 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 5 | 80.0% | 100.0% | 1.0000 | 1.0000 |
| Neural Net | 5 | 40.0% | 0.0% | -1.4090 | -1.4090 |
| Hybrid | 5 | 100.0% | 100.0% | 1.0000 | 1.0000 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=10.5,  p=1.0000,  direction=b_greater,  n=(5, 4)

### Hybrid vs Neural Net

  U=10.0,  p=0.0545,  direction=a_greater,  n=(5, 2)

### Neural Net vs Pure LLM

  U=0.0,  p=0.0852,  direction=b_greater,  n=(2, 4)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 2
Hybrid wins:  2  (100.0%)
NN wins:      0
Tied:         0

## Coverage Gaps (1 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | N/A | 1.0000 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 1 | 100.0% | 0.0% | 100.0% |
| hard | 2 | 100.0% | 0.0% | 100.0% |
| medium | 2 | 100.0% | 0.0% | 100.0% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 2 | 1.0000 | -2.7952 | 1.0000 |
| quadratic_form | 2 | 1.0000 | N/A | 1.0000 |
| rational | 1 | 1.0000 | -0.0229 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | 0.0000 | 0.0000 | 4 |
| Neural Net | 2.4089 | 2.4089 | 2 |
| Hybrid | 0.0000 | 0.0000 | 5 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 14.3876 | 13.8950 | 71.94 | 5 |
| Neural Net | 0.3516 | 0.0000 | 1.76 | 5 |
| Hybrid | 1.6664 | 1.4960 | 8.33 | 5 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| llm | 5 |

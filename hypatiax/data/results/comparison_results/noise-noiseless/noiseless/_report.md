
# HypatiaX Analysis Report — `exp1`

Experiment mode: **standard**
N total: 5 | N standard: 5 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 5 | 80.0% | 100.0% | 1.0000 | 1.0000 |
| Neural Net | 5 | 100.0% | 0.0% | -2.5607 | -2.2182 |
| Hybrid | 5 | 100.0% | 80.0% | 1.0000 | 0.8752 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=8.0,  p=0.6605,  direction=b_greater,  n=(5, 4)

### Hybrid vs Neural Net

  U=25.0,  p=0.0109**,  direction=a_greater,  n=(5, 5)

### Neural Net vs Pure LLM

  U=0.0,  p=0.0175**,  direction=b_greater,  n=(5, 4)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 5
Hybrid wins:  5  (100.0%)
NN wins:      0
Tied:         0

## Coverage Gaps (1 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| Portfolio Expected Shortfall for correlated | hard | quadratic_form | N/A | N/A | -2.4931 | 0.3760 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| easy | 1 | 100.0% | 0.0% | 100.0% |
| hard | 2 | 100.0% | 0.0% | 50.0% |
| medium | 2 | 100.0% | 0.0% | 100.0% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| algebraic | 2 | 1.0000 | -2.8723 | 1.0000 |
| quadratic_form | 2 | 1.0000 | -2.5269 | 0.6880 |
| rational | 1 | 1.0000 | -0.2925 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | 0.0000 | 0.0000 | 4 |
| Neural Net | 3.2181 | 3.5606 | 5 |
| Hybrid | 0.1217 | 0.0000 | 5 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 13.3088 | 12.3360 | 66.54 | 5 |
| Neural Net | 7.3426 | 2.7850 | 36.71 | 5 |
| Hybrid | 1.6014 | 1.4570 | 8.01 | 5 |

## Hybrid Routing Decisions

| Decision | Count |
|----------|-------|
| llm | 5 |

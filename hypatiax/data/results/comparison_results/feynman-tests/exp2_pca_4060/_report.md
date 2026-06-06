
# HypatiaX Analysis Report — `exp2_feynman_pca`

Experiment mode: **standard**
N total: 420 | N standard: 420 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 60 | 100.0% | 100.0% | 1.0000 | 1.0000 |
| Neural Net | 60 | 100.0% | 100.0% | 1.0000 | 0.9974 |
| Hybrid | 60 | 100.0% | 100.0% | 1.0000 | 1.0000 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=92.0,  p=0.0000**,  direction=b_greater,  n=(60, 60)

### Hybrid vs Neural Net

  U=3412.0,  p=0.0000**,  direction=a_greater,  n=(60, 60)

### Neural Net vs Pure LLM

  U=0.0,  p=0.0000**,  direction=b_greater,  n=(60, 60)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 60
Hybrid wins:  54  (90.0%)
NN wins:      6
Tied:         0

## Coverage Gaps (360 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
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
| unknown | 60 | 100.0% | 100.0% | 100.0% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| unknown | 60 | 1.0000 | 1.0000 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | N/A | N/A | 0 |
| Neural Net | N/A | N/A | 0 |
| Hybrid | N/A | N/A | 0 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 5.7571 | 7.3961 | 345.43 | 60 |
| Neural Net | 2.4927 | 2.2334 | 149.56 | 60 |
| Hybrid | 5.5418 | 6.6602 | 332.51 | 60 |

## Hybrid Routing Decisions

_No hybrid decision data available._

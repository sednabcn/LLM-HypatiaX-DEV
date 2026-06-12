
# HypatiaX Analysis Report — `exp2_feyman_pca`

Experiment mode: **standard**
N total: 30 | N standard: 30 | N intractable: 0
R² success threshold: 0.8

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 30 | 100.0% | 100.0% | 1.0000 | 1.0000 |
| Neural Net | 30 | 100.0% | 100.0% | 1.0000 | 0.9974 |
| Hybrid | 30 | 100.0% | 100.0% | 1.0000 | 1.0000 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=23.0,  p=0.0000**,  direction=b_greater,  n=(30, 30)

### Hybrid vs Neural Net

  U=853.0,  p=0.0000**,  direction=a_greater,  n=(30, 30)

### Neural Net vs Pure LLM

  U=0.0,  p=0.0000**,  direction=b_greater,  n=(30, 30)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 30
Hybrid wins:  27  (90.0%)
NN wins:      3
Tied:         0

## Coverage Gaps (0 equations with best R² < 0.8)

_None — all standard equations have at least one method achieving R² ≥ threshold._

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| unknown | 30 | 100.0% | 100.0% | 100.0% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| unknown | 30 | 1.0000 | 1.0000 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | N/A | N/A | 0 |
| Neural Net | N/A | N/A | 0 |
| Hybrid | N/A | N/A | 0 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 5.7571 | 7.3961 | 172.71 | 30 |
| Neural Net | 2.4927 | 2.2334 | 74.78 | 30 |
| Hybrid | 5.5418 | 6.6602 | 166.25 | 30 |

## Hybrid Routing Decisions

_No hybrid decision data available._

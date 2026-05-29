
# HypatiaX Analysis Report — `extrap`

Experiment mode: **ood**
N total: 60 | N standard: 60 | N intractable: 0
R² success threshold: 0.8

> **OOD experiment**: hybrid losing to neural_network is the expected scientific result; `HYBRID_NEVER_BEATS_NN` is demoted to informational and does not block the workflow.

## ✅ No Fatal Conditions


## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 60 | 100.0% | 96.7% | 0.9971 | 0.9883 |
| Neural Net | 60 | 100.0% | 96.7% | 0.9972 | 0.9877 |
| Hybrid | 60 | 100.0% | 96.7% | 1.0000 | 0.9919 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=3236.0,  p=0.0000**,  direction=a_greater,  n=(60, 60)

### Hybrid vs Neural Net

  U=3216.0,  p=0.0000**,  direction=a_greater,  n=(60, 60)

### Neural Net vs Pure LLM

  U=1856.0,  p=0.7708,  direction=a_greater,  n=(60, 60)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 60
Hybrid wins:  58  (96.7%)
NN wins:      2
Tied:         0
_Note: hybrid losing NN is expected in OOD extrapolation._

## Coverage Gaps (2 equations with best R² < 0.8)

| Equation | Difficulty | Type | Best R² | LLM | NN | Hybrid |
|----------|------------|------|---------|-----|----|----|
| Arrhenius rate constant (Feynman variant) | None | None | 0.7717 | 0.7385 | 0.7555 | 0.7717 |
| Arrhenius rate constant (Feynman variant) | None | None | 0.7717 | 0.7385 | 0.7555 | 0.7717 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| unknown | 60 | 96.7% | 96.7% | 96.7% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| unknown | 60 | 0.9971 | 0.9972 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | N/A | N/A | 0 |
| Neural Net | N/A | N/A | 0 |
| Hybrid | N/A | N/A | 0 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 5.7944 | 7.8938 | 347.67 | 60 |
| Neural Net | 2.5957 | 2.4189 | 155.74 | 60 |
| Hybrid | 5.7854 | 6.9026 | 347.12 | 60 |

## Hybrid Routing Decisions

_No hybrid decision data available._

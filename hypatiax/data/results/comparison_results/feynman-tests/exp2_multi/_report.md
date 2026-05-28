
# HypatiaX Analysis Report — `exp2`

Experiment mode: **multi_method**
N total: 262 | N standard: 262 | N intractable: 0
R² success threshold: 0.8

> **Multi-method experiment**: a 4th method key (`HybridSystemLLMNN all-domains`) is present in the raw output but is not in `METHODS` and is excluded from comparisons. Verify `merge_shards.py` translates method names correctly.

## ✅ No Fatal Conditions


## ℹ️ Informational / Warnings

- WARN_MULTI_METHOD: this experiment produces a 4th method key (HybridSystemLLMNN all-domains) not in METHODS. It is excluded from all method-comparison statistics. Confirm merge_shards.py translates method names before analysis.

## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 82 | 97.6% | 92.5% | 1.0000 | 0.6770 |
| Neural Net | 82 | 100.0% | 100.0% | 0.9999 | 0.9956 |
| Hybrid | 82 | 100.0% | 100.0% | 1.0000 | 1.0000 |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  U=693.0,  p=0.0000**,  direction=b_greater,  n=(82, 80)

### Hybrid vs Neural Net

  U=6469.0,  p=0.0000**,  direction=a_greater,  n=(82, 82)

### Neural Net vs Pure LLM

  U=492.0,  p=0.0000**,  direction=b_greater,  n=(82, 80)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 82
Hybrid wins:  78  (95.1%)
NN wins:      4
Tied:         0

## Coverage Gaps (182 equations with best R² < 0.8)

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
| Molecular viscosity from physicochemical props | None | None | N/A | N/A | 0.9986 | 1.0000 |
| Molecular viscosity from physicochemical props | None | None | N/A | N/A | 0.9986 | 1.0000 |

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| unknown | 82 | 92.5% | 100.0% | 100.0% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| unknown | 82 | 1.0000 | 0.9999 | 1.0000 |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | N/A | N/A | 0 |
| Neural Net | N/A | N/A | 0 |
| Hybrid | N/A | N/A | 0 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | 8.7991 | 8.2371 | 721.53 | 82 |
| Neural Net | 2.7946 | 1.1074 | 229.15 | 82 |
| Hybrid | 6.8343 | 7.4028 | 560.41 | 82 |

## Hybrid Routing Decisions

_No hybrid decision data available._

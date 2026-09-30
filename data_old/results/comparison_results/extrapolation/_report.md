
# HypatiaX Analysis Report — `extrap`

Experiment mode: **ood**
N total: 30 | N standard: 30 | N intractable: 0
R² success threshold: 0.8

> **OOD experiment**: hybrid losing to neural_network is the expected scientific result; `HYBRID_NEVER_BEATS_NN` is demoted to informational and does not block the workflow.

## ✅ No Fatal Conditions


## ℹ️ Informational / Warnings

- WARN_NO_METHOD_RECORDS: records were loaded but none contained results under canonical keys (pure_llm, neural_network, hybrid). This usually means the JSON uses non-standard method names that were not translated by merge_shards.py, or the experiment should be mapped to 'multi_method' mode in EXPERIMENT_MODE. Verify _normalise_protocol_record() output or add an EXPERIMENT_MODE entry. Workflow continues.

## Method Summary (standard equations only)

| Method | N | Success% (flag) | R²≥0.80% | Median test R² | Mean test R² |
|--------|---|-----------------|----------|----------------|--------------|
| Pure LLM | 0 | 0.0% | 0.0% | N/A | N/A |
| Neural Net | 0 | 0.0% | 0.0% | N/A | N/A |
| Hybrid | 0 | 0.0% | 0.0% | N/A | N/A |

## Mann-Whitney U Tests (two-sided, clipped R², standard equations)


### Hybrid vs Pure LLM

  N/A (insufficient samples)

### Hybrid vs Neural Net

  N/A (insufficient samples)

### Neural Net vs Pure LLM

  N/A (insufficient samples)
_** = p < 0.05_

## Hybrid vs Neural Net (head-to-head, equation level)

Equations with both finite R²: 0
Hybrid wins:  0  (N/A)
NN wins:      0
Tied:         0
_Note: hybrid losing NN is expected in OOD extrapolation._

## Coverage Gaps (30 equations with best R² < 0.8)

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

## R²≥0.80 Rate by Difficulty

| Difficulty | N | LLM R²≥0.80 | NN R²≥0.80 | Hybrid R²≥0.80 |
|------------|---|-------------|------------|----------------|
| unknown | 0 | 0.0% | 0.0% | 0.0% |

## Median Test R² by Formula Type

| Formula Type | N | LLM median R² | NN median R² | Hybrid median R² |
|--------------|---|---------------|--------------|------------------|
| unknown | 0 | N/A | N/A | N/A |

## Extrapolation Gap (train R² − test R²)

| Method | Mean gap | Median gap | N |
|--------|----------|------------|---|
| Pure LLM | N/A | N/A | 0 |
| Neural Net | N/A | N/A | 0 |
| Hybrid | N/A | N/A | 0 |

## Wall-clock Timing (standard equations)

| Method | Mean (s) | Median (s) | Total (s) | N |
|--------|----------|------------|-----------|---|
| Pure LLM | N/A | N/A | None | 0 |
| Neural Net | N/A | N/A | None | 0 |
| Hybrid | N/A | N/A | None | 0 |

## Hybrid Routing Decisions

_No hybrid decision data available._

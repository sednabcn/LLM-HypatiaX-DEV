# Variant comparison: 5 hybrid variants vs. NN / LLM / PySR

Each panel is one benchmark on which the compared methods share the same equations; panels are not comparable with one another. Evidence: `variant_comparison_evidence.json`.

## A. Feynman-30, in-distribution R² (n=30; pass = R²≥0.99)

| Role | Recorded method label | ok/n | Median R² | Pass | Median time (s) |
|---|---|---|---|---|---|
| LLM | `PureLLM Baseline (core)` | 29/30 | 1.0000 | 29/30 | 6.0 |
| NN | `ImprovedNN (core)` | 30/30 | 0.9999 | 30/30 | 0.6 |
| V1 | `EnhancedHybridSystemDeFi (core)` | 30/30 | 1.0000 | 30/30 | 9.9 |
| V2 | `HybridSystemLLMNN all-domains (core)` | 30/30 | 1.0000 | 30/30 | 8.9 |
| V4 (llm_mode=none, LLM off) | `SymbolicEngineWithLLM (tools)` | 30/30 | 1.0000 | 30/30 | 37.3 |
| PySR engine (v50_2, no LLM) | `HybridDiscoverySystem v50_2 (tools)` | 30/30 | 1.0000 | 30/30 | 61.4 |

## B. Core-15, extrapolation R² (median over successful runs; (k) = equations with a finite value)

| Role | Source row | ok/n | Train | Near (k) | Med (k) | Far (k) | Time (s) |
|---|---|---|---|---|---|---|---|
| NN | Neural Network | 15/15 | 0.9976 | -0.1929 (14) | -0.4931 (14) | -14.7408 (12) | 0.5 |
| LLM | Pure LLM | 15/15 | 0.9930 | -1.4868 (1) | --- (0) | --- (0) | 7.1 |
| PySR-only | exp1_ablation pysr_only (raw PySRRegressor) | 15/15 | 0.9975 | 1.0000 (15) | 1.0000 (14) | 1.0000 (15) | 1101 |
| V2 | System 3 LLM+Fallback | 15/15 | 1.0000 | --- (0) | --- (0) | --- (0) | 9.3 |
| V4 (llm_mode=none, LLM off) | System 2 Symbolic | 15/15 | 0.9978 | 0.9987 (12) | 0.9969 (12) | 0.9461 (12) | 33.0 |
| V4 (LLM hybrid mode, indirect via HybridDiscoverySystem) | exp1_ablation hypatia arm | 15/15 | 0.9977 | 0.9993 (10) | 0.9999 (10) | 0.9993 (10) | 357 |
| PySR engine (v50_2, no LLM) | Hybrid v50_2 | 15/15 | 0.9977 | 0.9788 (12) | 0.9923 (11) | 0.9707 (9) | 105 |

## C. DeFi-73, extrapolation test R² (n=73; pass = R²≥0.99)

| Role | Subset | ok/n | Median R² | Pass |
|---|---|---|---|---|
| V1 | hybrid arm, all cases (includes the max(LLM,NN) ensemble route) | 73/73 | 1.0000 | 48/73 |
| V1, no test-set selection | hybrid arm, non-ensemble routes (nn / llm / fitted_llm) | 21/21 | 0.2264 | 1/21 |
| V1, ensemble route | hybrid arm, 'ensemble' route (recorded via max(LLM,NN) fallback) | 52/52 | 1.0000 | 47/52 |
| NN | neural_network arm, all cases | 73/73 | 0.1364 | 1/73 |
| NN (same cases as V1 row 2) | same non-ensemble cases | 21/21 | 0.2264 | 1/21 |
| LLM | pure_llm arm, all cases | 54/73 | 1.0000 | 41/73 |
| LLM (same cases as V1 row 2) | same non-ensemble cases | 9/21 | -3.5798 | 3/21 |

## D. Variants without a usable result source

| Variant | Status | Evidence |
|---|---|---|
| V3 `ensemble_llm_nn()` | NO CONFIRMED RESULT SOURCE | the only result-writing caller (test_enhanced_defi_extrapolation.py) imports it from hypatiax.experiments.tests.hybrid_ensemble_system_defi_domain, which does not exist; a bare except then falls back to max(LLM, NN) on TEST R2; 41/45 'ensemble'-route rows equal max(LLM,NN) to 1e-6 (0 below it); the DeFi v4 benchmark uses a private _ensemble_llm_nn (not this function); its blend route fired 2/370 cases |
| V5 `PhysicsAwareRegressor` | NOT EXERCISED | PhysicsAwareRegressor is reachable only as the [FALLBACK] inside HybridDiscoverySystem, gated by enable_physics_fallback, whose default is False (PIN-4); explicit settings in experiments/: False; 0/600 recorded strategy tags are physics_aware (211 result files scanned); SmartStructureDetector is not imported by any other module |

**Notes**

1. V1 = EnhancedHybridSystemDeFi; V2 = HybridSystemAllDomains; V3 = ensemble_llm_nn(); V4 = SymbolicEngineWithLLM; V5 = PhysicsAwareRegressor / SmartStructureDetector. V2's identity is established by name-list elimination, not an explicit import (see evidence file).
2. V4 in panels A/B is run with llm_mode="none" (the benchmark pins it), i.e. V4's PySR path only; its LLM modes are reached only indirectly, through HybridDiscoverySystem in the ablation's hypatia arm. That arm's llm_expression field is empty in 15/15 records, so the LLM's own contribution cannot be verified from the file.
3. PySR engine (v50_2, no LLM) is HybridDiscoverySystem with use_llm off (a retry / adaptive-iteration wrapper around PySR), not a raw PySRRegressor; panel B's PySR-only row is the raw baseline.
4. In panel A the V1 and V2 wrappers force the LLM path on Feynman domains (force_llm) and almost all methods saturate near R2 = 1; the panel shows what was run, not that the methods are equivalent.
5. In panel B, NN / LLM / V2 have few or no finite extrapolation values because formula-based methods record none for those columns; compare (k) before reading a median.
6. In panel C the V1 rows are split by routing path. The ensemble route is scored by max(LLM, NN) on the test set (see D), which flatters V1. On the remaining cases (routes: fitted_llm=3, nn=18) V1's score equals the NN arm's in 18/18 nn-routed cases, so those rows compare V1 with NN on essentially the same predictions.

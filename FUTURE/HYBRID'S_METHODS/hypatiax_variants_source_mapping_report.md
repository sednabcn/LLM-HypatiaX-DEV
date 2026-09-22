# HypatiaX Variants — Source Mapping & Gate Mechanism Report

**Repo verified:** `github.com/sednabcn/LLM-HypatiaX-REPRO` (public, cloned and
grepped directly — every mapping below is confirmed against real source, not
inferred from documentation or a prior transcript).

**Date:** 2026-09-22

---

## Mapping table

| # | Standalone variant script | Real `hypatiax` source | Class / function ported | Gate mechanism |
|---|---|---|---|---|
| 1 | `variant1_enhanced_hybrid_defi.py` | `hypatiax/core/generation/hybrid_defi_system/hybrid_system_nn_defi_domain.py` | `EnhancedHybridSystemDeFi` | R² threshold gate: `fitted_r2 ≥ 0.85 AND margin > 0.05` → LLM formula wins; `both R² ≥ 0.5 AND \|margin\| ≤ 0.15` → weighted ensemble (weighted by R²-share, not inverse-std); otherwise → NN |
| 2 | `variant2_hybrid_all_domains.py` | `hypatiax/core/generation/hybrid_all_domains_llm_nn/hybrid_system_llm_nn_all_domains.py` | `HybridSystemAllDomains` | `llm_ok` gate fires first (LLM must succeed **and** have R² > 0); ties/wins favor LLM; ensemble triggers only when NN beats LLM **and** NN R² > 0.90 (weighted by R²-share); `force_llm` does **not** rescue a broken formula |
| 3 | `variant3_ensemble_example.py` | `hypatiax/core/generation/hybrid_defi_llm_nn/hybrid_ensemble_system_defi_domain.py` | `ensemble_llm_nn()` *(standalone function, not a class)* | Weighted blend: inverse-uncertainty × R²-strength (multiplicative weighting); falls back to an equal 0.5 / 0.5 weight when that product is undefined |
| 4 | `variant4_llm_prior_pysr_family.py` | `hypatiax/tools/symbolic/symbolic_engine.py` | `SymbolicEngineWithLLM` | Mode-dependent dispatch. `hybrid` mode: LLM R² > 0.95 → LLM only; 0.5 ≤ R² ≤ 0.95 → search seeded by LLM prior, keep whichever is better; R² < 0.5 → prior discarded, pure search. `fallback` mode: search R² > 0.90 → done; else ask LLM and keep the better of the two |
| 5 | `variant5_retry_physics_fallback.py` | `hypatiax/tools/symbolic/hybrid_system_v50_2.py` + `physics_aware_regressor.py` + `smart_structure_detector.py` | `HybridDiscoverySystem._discover_with_retry`, `PhysicsAwareRegressor`, `SmartStructureDetector` *(orphaned — see note)* | Retry loop, `max_retries = 5`, seed = `42 + attempt`. Early stop at R² ≥ 0.95 (or ≥ 0.9999 if `use_transcendental_compositions`); overall **success** requires R² ≥ 0.97 after retries — a separate, higher bar than early-stop. Physics fallback fires only if `enable_physics_fallback = True` (**off by default**) **and** best R² < `physics_fallback_threshold` (default 0.85) |

---

## Notes

- **Variant 3** is the only entry whose real source is a standalone function
  (`def ensemble_llm_nn(...)`), not a class.
- **Variant 5** is the only entry mapping to three real source files. One of
  them, `smart_structure_detector.py`, contributes a class
  (`SmartStructureDetector`) that is ported in the standalone script but
  confirmed **not wired into the actual discovery pipeline anywhere in the
  real repo** — grep of the full source tree shows it is never imported or
  called outside its own module, and it isn't even exported from
  `hypatiax/tools/symbolic/__init__.py`. It's orphaned code, not a live
  fallback path, despite what its name suggests.
- **Variant 4**'s mode thresholds (0.95 / 0.5 / 0.90) matched the real
  source exactly on first build — no correction was needed there.
- All five standalone scripts substitute the real PySR / genetic-programming
  search engines with honest, documented, weaker replacements (a
  numpy/scipy feature-library greedy search). Gate *logic* and *thresholds*
  above are faithful to real source; the underlying search quality is not —
  do not compare these scripts' R² numbers to PySR-family results from the
  paper.

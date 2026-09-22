# HypatiaX Hybrid Variants — Standalone Scripts

Five self-contained Python files, one per hybrid variant described in
`Variant_s-Hybrid-Method.txt`. Each file has **no dependency on the
`hypatiax` package** — only `numpy`, `scipy`, and `scikit-learn` (all
standard). Each runs on its own with a built-in synthetic benchmark, an
offline mock-LLM provider, and a `--selftest` suite.

| # | File | Class/function ported | Combination strategy | Selftest |
|---|------|------------------------|------------------------|----------|
| 1 | `variant1_enhanced_hybrid_defi.py` | `EnhancedHybridSystemDeFi` | R² threshold gate + scipy constant refit | 14/14 ✅ |
| 2 | `variant2_hybrid_all_domains.py` | `HybridSystemAllDomains` | Tie-favours-symbolic gate + `force_llm` override | 14/14 ✅ |
| 3 | `variant3_ensemble_example.py` | `ensemble_llm_nn()` | Inverse-residual-std weighted blend | 15/15 ✅ |
| 4 | `variant4_llm_prior_pysr_family.py` | `SymbolicEngineWithLLM` (4 modes) | LLM ↔ search take turns proposing/refining | 15/15 ✅ |
| 5 | `variant5_template_physics_fallback.py` | `PhysicsAwareRegressor` / `SmartStructureDetector` | Template-library fit, template-free fallback | 16/16 ✅ |

## Honesty note on provenance

These are **standalone re-implementations of the documented decision logic**
for each variant, not a byte-for-byte port of a private repo. The gate
thresholds, mode names, and fallback behaviours match what's described in
`Variant_s-Hybrid-Method.txt`. Two deliberate substitutions, called out in
each file's docstring:

- **Variant 4's search backend** is a compact numpy/scipy feature-library
  greedy search, not PySR (needs Julia) or a full genetic-programming tree
  search. It's a real structure search, just weaker — don't compare its R²
  to PySR-family numbers.
- **All LLM calls default to an offline mock provider** with deliberately
  imperfect fixture formulas per case, so the gate logic is actually
  exercised (some fixtures are exact, some are wrong on purpose). Pass
  `--llm anthropic` to use a real model instead (needs `pip install
  anthropic` and `ANTHROPIC_API_KEY` set — this path is documented but was
  **not** executed while building these files).

## Quick start

```bash
pip install numpy scipy scikit-learn      # torch/anthropic optional

python variant1_enhanced_hybrid_defi.py --selftest
python variant1_enhanced_hybrid_defi.py              # run the built-in DeFi suite

python variant2_hybrid_all_domains.py --selftest
python variant2_hybrid_all_domains.py --force-llm

python variant3_ensemble_example.py --selftest
python variant3_ensemble_example.py --force-fallback  # exercise the max(LLM,NN) fallback

python variant4_llm_prior_pysr_family.py --selftest
python variant4_llm_prior_pysr_family.py --modes hybrid --suite nguyen5

python variant5_template_physics_fallback.py --selftest
python variant5_template_physics_fallback.py --threshold 0.995
```

Every script supports `--list-cases`, `--suite <ids>`, `--samples N`,
`--noise N`, `--holdout N`, `--seed N`, `--out-dir DIR`, `--verbose`, and
saves a timestamped JSON of results under `hypatiax_variant_results/`.

Each file is also importable as a library, e.g.:

```python
from variant1_enhanced_hybrid_defi import EnhancedHybridSystemDeFi
from variant3_ensemble_example import ensemble_llm_nn
from variant5_template_physics_fallback import template_physics_fallback
```

# HypatiaX Hybrid Variants — Standalone Scripts

Five self-contained Python files, one per hybrid variant in the HypatiaX
symbolic-regression system. Each file has **no dependency on the
`hypatiax` package** — only `numpy`, `scipy`, and `scikit-learn` (all
standard). Each runs on its own with a built-in synthetic benchmark, an
offline mock-LLM provider, and a `--selftest` suite.

## Provenance — verified against real public source

Earlier drafts of this package assumed the source repo was private and
built these files from a transcript description alone. That assumption was
wrong: **github.com/sednabcn/LLM-HypatiaX-REPRO is public.** It was cloned
directly and every ported class/function below was grep-verified and
read line-by-line against the real source, not reconstructed from memory
or a description. Two corrections that came out of that verification pass
are called out explicitly per-variant below because they change the
*documented* behavior, not just an implementation detail:

- **Variant 4**'s mode-dispatch thresholds (0.95 / 0.5 / 0.90) already
  matched the real source exactly on first build — no correction needed.
- **Variant 5** was substantially wrong in its first draft (described as
  "template library → template-free fallback"; the real architecture is a
  retry-with-early-stop loop around one search engine, with a physics
  regressor as an optional, *off-by-default* last resort) and has been
  rewritten. See its own docstring for the full corrected algorithm and an
  important finding: `SmartStructureDetector` is defined in the real repo
  but never actually called anywhere outside its own module — it's
  orphaned code, not a live fallback path, despite what its name and
  docstring suggest.

| # | File | Class/function ported | Combination strategy | Selftest |
|---|------|------------------------|------------------------|----------|
| 1 | `variant1_enhanced_hybrid_defi.py` | `EnhancedHybridSystemDeFi` | R² threshold gate + scipy constant refit | 14/14 ✅ |
| 2 | `variant2_hybrid_all_domains.py` | `HybridSystemAllDomains` | `llm_ok`-gated dispatch + `force_llm` override | 14/14 ✅ |
| 3 | `variant3_ensemble_example.py` | `ensemble_llm_nn()` | Inverse-uncertainty × R²-strength weighted blend | 15/15 ✅ |
| 4 | `variant4_llm_prior_pysr_family.py` | `SymbolicEngineWithLLM` (4 modes) | LLM ↔ search take turns proposing/refining | 15/15 ✅ |
| 5 | `variant5_retry_physics_fallback.py` | `HybridDiscoverySystem._discover_with_retry`, `PhysicsAwareRegressor`, `SmartStructureDetector` | Retry-with-early-stop, then optional (off by default) physics fallback | 29/29 ✅ |

## Honesty note on substitutions

These are **faithful ports of the real, verified control flow and
thresholds**, but two categories of code are deliberately substituted
rather than reproduced line-for-line, called out in every affected file's
docstring:

- **The symbolic search backend** (Variants 4 and 5) is a compact
  numpy/scipy feature-library greedy search, not PySR (needs Julia) or a
  full genetic-programming tree search. It's a real structure search, just
  weaker — don't compare its R² numbers to PySR-family results from the
  paper.
- **Variant 5's `PhysicsAwareRegressor`** stands in for the real 2117-line
  domain-seeded genetic-programming engine. The noise-adaptive preset
  selection (population_size / generations / parsimony_coefficient / min_r2
  chosen from `noise_level`) *is* a faithful port of the real values; the
  evolutionary search loop itself is substituted with a much smaller
  template-seeded fit. Off by default in the real system either way
  (`enable_physics_fallback=False`), so this path is rarely exercised.
- **All LLM calls default to an offline mock provider** with deliberately
  imperfect fixture formulas per case, so the gate logic is actually
  exercised (some fixtures are exact, some are wrong on purpose). Pass
  `--llm anthropic` to use a real model instead (needs `pip install
  anthropic` and `ANTHROPIC_API_KEY` set — this path is documented but was
  **not** executed while building these files).

Every other piece of decision logic in these five files — every threshold,
every gate condition, every default value, the quality/overfit check, and
the rational-pattern detector in Variant 5 — is either an exact numeric
match confirmed against the real source, or (for the two pieces named
above) a plain numpy/sklearn function ported verbatim with no PySR
dependency to substitute.

## Quick start

```bash
pip install numpy scipy scikit-learn      # torch/anthropic optional

python variant1_enhanced_hybrid_defi.py --selftest
python variant1_enhanced_hybrid_defi.py              # run the built-in DeFi suite

python variant2_hybrid_all_domains.py --selftest
python variant2_hybrid_all_domains.py --force-llm

python variant3_ensemble_example.py --selftest
python variant3_ensemble_example.py --force-fallback  # exercise the ensemble path

python variant4_llm_prior_pysr_family.py --selftest
python variant4_llm_prior_pysr_family.py --modes hybrid --suite nguyen5

python variant5_retry_physics_fallback.py --selftest
python variant5_retry_physics_fallback.py --physics-fallback   # turn on the off-by-default fallback
```

Every script supports `--list-cases`, `--suite <ids>`, `--samples N`,
`--noise N`, `--holdout N`, `--seed N`, `--out-dir DIR`, `--verbose`, and
saves a timestamped JSON of results under `hypatiax_variant_results/`.

Each file is also importable as a library, e.g.:

```python
from variant1_enhanced_hybrid_defi import EnhancedHybridSystemDeFi
from variant3_ensemble_example import ensemble_llm_nn
from variant5_retry_physics_fallback import HybridDiscoverySystem
```

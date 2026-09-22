# Plan: `run_comparative_hybrid_methods.py` + `run_hybrid_all.sh` + `ci_runner_hybrid_repo.yml`

Grounded against a fresh clone of `github.com/sednabcn/LLM-HypatiaX-REPRO`
(2026-09-22). Every file/function path below was confirmed to exist by
direct inspection, not assumed.

---

## 0. What already exists (reuse, don't rebuild)

The repo already has almost the exact skeleton you're asking for, under
different names. The job is mostly **extension**, not new architecture.

| You asked for | Closest existing thing | Location |
|---|---|---|
| Multi-method comparative harness | `run_comparative_suite_benchmark_v2.py` — already runs a `METHOD_REGISTRY` of wrapper classes against a protocol and prints a comparison table | `hypatiax/experiments/benchmarks/` |
| `run_hybrid_all.sh` | `run_all.sh` — already has a named-step runner (`run <step> <desc> <cmd>`), a `_STEP_ORDER` list, `--step`/`--from`/`--dry-run` flags, and steps literally named `exp1 exp1b exp1_ablation exp1_five exp1_pca exp1b_pca ... exp2 exp2_five exp3 exp3b ...` | repo root |
| `ci_runner_hybrid_repo.yml` | `ci_runner_repro.yml` (full sharded paper-repro pipeline, 3346 lines) and `ci_smoke_test_v4.yml` (353-line fast wiring check — the better structural template for a first version) | `.github/workflows/` |
| 4 protocols | `experiment_protocol_defi.py`, `experiment_protocol_benchmark_v2.py`, `experiment_protocol_nguyen12.py`, `experiment_protocol_all_30.py` | `hypatiax/protocols/` |

**The 7 experiments already map to protocols exactly as you listed them:**

| Experiment | Real script(s) invoked (from `run_all.sh`) | Protocol |
|---|---|---|
| `exp1` | `hypatiax_defi_benchmark_v4.py` (all 74 cases) | `experiment_protocol_defi.py` |
| `exp1_pca` | `hypatiax_defi_benchmark_v4_pca.py` (PCA 40/60 split) | same, PCA split |
| `exp1b` | `hypatiax_defi_benchmark_v4.py --resume` (seed sweep) + `portfolio_variance_v4c2.py` | same |
| `exp1b_pca` | PCA counterpart of exp1b | same, PCA split |
| `exp2` | `run_comparative_suite_benchmark_v2.py --protocol all_domains` per Feynman domain | `experiment_protocol_benchmark_v2.py` |
| `exp3` | `exp3_nguyen12_consolidated.py --seed 42` | `experiment_protocol_nguyen12.py` |
| `exp3b` | same script, seeds 99/123/777/2024 | same |

`experiment_protocol_all_30.py` is the 4th protocol file but backs
`exp1_five`/`exp2_five` (five-system comparison, M.01–M.30), which are
siblings to your 7, not members of it — worth deciding in step 1 below
whether it's in scope.

**The method-registry pattern you need already exists — just incomplete.**
`run_comparative_suite_benchmark_v2.py` has exactly the shape you want:

```python
METHOD_REGISTRY = [
    (1, PureLLMBaselineMethod,   "core/base_pure_llm/baseline_pure_llm_defi_discovery.py"),
    (2, ImprovedNNMethod,        "core/training/baseline_neural_network_defi_improved.py"),
    (3, HybridDeFiMethod,        "core/generation/hybrid_defi_system/hybrid_system_nn_defi_domain.py"),   # Variant 1
    (4, HybridAllDomainsMethod,  "core/generation/hybrid_all_domains_llm_nn/hybrid_system_llm_nn_all_domains.py"),  # Variant 2
    (5, SymbolicEngineMethod,    "tools/symbolic/symbolic_engine.py"),          # Variant 4
    (6, HybridSystemV50_2Method, "tools/symbolic/hybrid_system_v50_2.py"),      # Variant 5
]
```
Every entry is a `BaseMethod` subclass implementing one method:
`run(description, X, y, var_names, metadata, verbose) -> MethodResult`.
**Variant 3 (`ensemble_llm_nn`), Variant 6/6-PCA (`_select_v4_candidate`),
and Variant 7 (`get_llm_prior` seeding) are not in this registry at all.**
That's the actual gap `run_comparative_hybrid_methods.py` needs to close.

---

## 1. Decisions I need from you before writing code

1. **Scope of "all variants".** All 8 catalogued (1–5, 6, 6-PCA, 7), or a
   subset? Variant 7 doesn't naturally fit the DeFi/Feynman `BaseMethod`
   interface (it's Nguyen-12/PySR-specific, symbolic-regression only, no
   NN/LLM-blend concept) — it may only ever run under `exp3`/`exp3b`, not
   `exp1`/`exp2`. Confirm that's expected, not a gap.
2. **New script vs. extended registry.** Two shapes are possible:
   - (a) **Extend** `run_comparative_suite_benchmark_v2.py`'s
     `METHOD_REGISTRY` in place with entries 7 (Variant 3), 8 (Variant 6),
     9 (Variant 7), and let `run_comparative_hybrid_methods.py` just be a
     thin CLI wrapper that calls it with `--methods 3 4 5 6 7 8 9` plus a
     `--protocol` switch across all 4 protocol files.
   - (b) **New, protocol-agnostic script** that owns its own registry and
     calls each protocol's `load_test_data()`/`get_all_domains()` itself.
   (a) is far less code and reuses the already-hardened `BaseMethod`
   plumbing (timeouts, `_safe_r2` sign-flip correction, subprocess
   isolation for PySR-backed methods). I'd default to (a) unless you have
   a reason for a clean-room script.
3. **`_pca.py` sibling — same relationship as `hypatiax_defi_benchmark_v4.py`
   vs. `hypatiax_defi_benchmark_v4_pca.py`?** i.e. same registry/logic,
   different feature preprocessing only, kept as two files because CI job
   matrices key off filename? Confirm before I duplicate ~2000 lines.
4. **exp1b/exp1b_pca's `portfolio_variance_v4c2.py` step** — does the
   hybrid comparison need to reproduce that downstream analysis step too,
   or only the benchmark step it consumes?
5. **API budget.** Variants involving `get_llm_prior`/LLM formula
   generation call the Anthropic API per case. Running 8 variants ×
   7 experiments × full case counts is a large multiplier on API spend and
   wall-clock vs. today's single-variant-per-experiment runs. Do you want
   a `--sample-cases N` / smoke-scale knob from day one (mirrors
   `ci_smoke_test_v4.yml`'s pattern), or full-scale only?

---

## 2. Step-by-step build plan

### Phase 1 — `run_comparative_hybrid_methods.py` (DeFi + Feynman side)
1. Copy `run_comparative_suite_benchmark_v2.py`'s `BaseMethod`,
   `MethodResult`, and the six existing wrapper classes verbatim (don't
   refactor working, hardened code).
2. Add three new wrapper classes, each delegating to real source
   (no reimplementation, matching the pattern of `HybridDeFiMethod` etc.
   already delegating to `EnhancedHybridSystemDeFi`):
   - `EnsembleLLMNNMethod` → `hypatiax/core/generation/hybrid_defi_llm_nn/hybrid_ensemble_system_defi_domain.py::ensemble_llm_nn` (Variant 3)
   - `V4SelectorMethod` → `hypatiax/experiments/benchmarks/hypatiax_defi_benchmark_v4.py::_select_v4_candidate` / `_v4_hybrid_predict_and_eval` (Variant 6)
   - `LLMPriorSeedingMethod` → `hypatiax/experiments/benchmarks/hypatia.py::get_llm_prior` (Variant 7) — flag as Nguyen-12-only per decision #1
3. Extend `METHOD_REGISTRY` with indices 7, 8, 9.
4. Add a `--protocol {defi,benchmark_v2,nguyen12,all_30}` flag that swaps
   which protocol module's `load_test_data()`/`get_all_domains()` feeds the
   run loop — the existing script is already DeFi/Feynman-agnostic
   internally (it takes `description, X, y, var_names, metadata`, which
   every protocol module produces), so this is mostly CLI plumbing, not new
   logic.
5. Reuse `_safe_r2`, the outer-timeout/`_proc_box` subprocess isolation for
   PySR-backed methods, and the existing comparison-table printer
   unchanged.

### Phase 2 — `run_comparative_hybrid_methods_pca.py`
6. Once Phase 1 is confirmed working, fork it the same way
   `hypatiax_defi_benchmark_v4_pca.py` forks `hypatiax_defi_benchmark_v4.py`
   — diff the two existing files first to isolate exactly what changes
   (expected: the train/test split function and a
   `split_protocol_disclosure.json` write), and apply only that diff.

### Phase 3 — `run_hybrid_all.sh`
7. Copy `run_all.sh`'s harness functions verbatim: `run()`, `log()`,
   `warn()`, `die()`, the `_STEP_ORDER` pattern, `--step`/`--from`/
   `--dry-run` flags, and the `SHARD_IDS`/`TASK_IDS` sharding
   convention already used by `exp1b` and `exp3b`.
8. Define step order as the 7 experiments × the variant sets each supports:
   `hybrid_exp1 hybrid_exp1_pca hybrid_exp1b hybrid_exp1b_pca hybrid_exp2 hybrid_exp3 hybrid_exp3b`.
9. Each step calls `run_comparative_hybrid_methods.py` (or `_pca.py`) with
   `--protocol <matching protocol>` and `--methods 1 2 3 4 5 6 7 8 9`
   (or the confirmed subset from decision #1), mirroring exactly how
   `run_all.sh`'s `exp2` step already loops per-domain and tees to a
   per-step log file.
10. Reuse the existing env var contract (`PYSR_*`, `METHOD_TIMEOUT`,
    `JULIA_NUM_THREADS`, `RESULTS_DIR`, `EXPERIMENTS_DIR`) rather than
    inventing new ones — `run_comparative_hybrid_methods.py` should read
    the same env vars the wrapper classes it borrowed already expect.

### Phase 4 — `ci_runner_hybrid_repo.yml`
11. Start from `ci_smoke_test_v4.yml`'s structure, not
    `ci_runner_repro.yml`'s (that one's sharded plan/worker architecture is
    3346 lines and built for the full paper-repro matrix — over-scoped for
    a first version of a new pipeline). Carry over its hardening
    conventions exactly, since they were each added to fix a real failure
    mode in this repo:
    - `shell: bash` + `set -o pipefail` on every `run:` block (a piped
      script death must not look like a clean run)
    - probe `--help` before running each script; missing an expected flag
      is a hard failure, not a silent fallback to an unbounded run
    - sentinel-file timestamps so only JSON produced *by this run* counts
    - inputs passed via `env:`, never interpolated directly into `${{ }}`
      in shell (injection risk)
    - `concurrency.cancel-in-progress: false` (a cancelled job's `if:
      always()` upload step still needs to run)
12. Triggers: `workflow_dispatch` (with case/seed/domain inputs mirroring
    `ci_smoke_test_v4.yml`'s) plus a `pull_request.paths` filter on
    `run_comparative_hybrid_methods*.py` and `run_hybrid_all.sh`.
13. Steps: checkout → setup Python 3.12 → setup Julia 1.11 + depot cache →
    install deps (same pinned list as `ci_smoke_test_v4.yml`: `pysr==2.0.0a1`,
    torch CPU wheel, etc.) → validate `ANTHROPIC_API_KEY` → one job step per
    experiment, each running `run_hybrid_all.sh <step-name>` and asserting
    result-JSON count > 0 via the sentinel pattern.
14. Once the smoke-scale version is green, decide whether a full-scale
    sharded version belongs in `ci_runner_repro.yml`'s existing matrix
    (adding hybrid-method shards there) or stays a separate pipeline —
    this is really decision #5 revisited at CI scale.

---

## 3. Suggested build order (small, verifiable increments)

1. Land the 3 new `BaseMethod` wrapper classes + registry entries in a
   branch of `run_comparative_suite_benchmark_v2.py`; run `--methods 7 8 9`
   locally against a 2–3-case slice of the DeFi protocol only. Confirms the
   delegation works before anything else is built on top of it.
2. Rename/extract into `run_comparative_hybrid_methods.py`; add the
   `--protocol` switch; verify against all 4 protocols on a tiny slice each.
3. Fork the `_pca.py` sibling.
4. Write `run_hybrid_all.sh` with all 7 steps in `--dry-run` first (proves
   the step wiring and CLI flags without spending API budget or compute).
5. Write `ci_runner_hybrid_repo.yml` targeting only `hybrid_exp3` (cheapest,
   smallest case count) first, get it green, then add the remaining 6 steps
   one at a time.

I can start on step 1 as soon as you confirm decisions #1–#5 above —
particularly whether Variant 7 is Nguyen-12-only (almost certainly yes,
given its PySR-guesses mechanism has no analog in the DeFi/Feynman
protocols) and whether you want (a) extend-in-place or (b) clean-room for
the registry.

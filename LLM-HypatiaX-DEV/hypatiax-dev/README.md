# HypatiaX: A Hybrid Symbolic-Neural Framework for Extrapolation-Reliable Analytical Discovery

[![JMLR](https://img.shields.io/badge/JMLR-2026-blue)](https://jmlr.org)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Official code repository for the JMLR paper:

> **HypatiaX: A Hybrid Symbolic-Neural Framework for Extrapolation-Reliable Analytical Discovery**
> Ruperto Pedro Bonet Chaple
> *Journal of Machine Learning Research*, 2026

---

## What is HypatiaX?

HypatiaX is a hybrid symbolic regression system that combines LLM-guided search
with a symbolic engine (PySR) and multi-layer validation cascade. It discovers
closed-form equations from data, with a particular focus on **extrapolation
reliability** — the ability to recover equations that remain accurate well beyond
the training distribution.

The key finding: neural networks trained in-distribution (R² ≈ 1) collapse
catastrophically outside the training range, while HypatiaX's symbolic output
achieves near floating-point precision (error < 10⁻¹²) on the same benchmarks.

---

## Repository Structure

All paths below are relative to `papers/2025-JMLR/hypatiax/`.

```
hypatiax/
├── core/
│   ├── base_pure_llm/          # Campaign 1 — Pure LLM baseline
│   ├── generation/             # Hybrid generation systems
│   │   ├── hybrid_llm_guide_validation/   # LLM-guided discovery (canonical)
│   │   ├── hybrid_defi_system/            # DeFi hybrid system (canonical)
│   │   └── hybrid_defi_llm_guided/        # DeFi LLM-guided (canonical)
│   └── training/               # Neural network baselines
├── protocols/                  # Experiment entry points ← run these
├── experiments/
│   ├── benchmarks/             # Feynman SR benchmark runners
│   │   └── prod/               # Production benchmark runner v3
│   └── comparison/             # Comparative analysis
├── analysis/                   # Statistical analysis
├── tools/
│   ├── symbolic/               # hybrid_system_v40.py (canonical core)
│   ├── validation/             # Four-layer validation pipeline
│   ├── visualizations/         # Figure generation
│   └── utils/                  # Shared utilities
├── data/results/               # All experiment outputs
├── notebooks/                  # Jupyter walkthroughs (01–06)
└── legacy/                     # Archived non-canonical versions (do not run)
```

Audit and traceability scripts are in `../code/scripts/`
(i.e. `papers/2025-JMLR/code/scripts/`).

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/sednabcn/LLM-HypatiaX-PAPERS-Public
cd LLM-HypatiaX-PAPERS-Public/papers/2025-JMLR
pip install -r requirements.txt
cd hypatiax
```

Requires Python 3.12. Julia ≥ 1.9 + PySR required for Campaigns 2, 3, 4, 7.
Campaign 1 (Pure LLM) only needs Python dependencies.

All canonical scripts use `SEED = 42` for full reproducibility.

### 2. Set your API key

```bash
export OPENAI_API_KEY="sk-..."   # or ANTHROPIC_API_KEY
```

### 3. Run the Core 15 benchmark

```bash
python protocols/experiment_protocol_all_30.py --no-cache
# → data/results/llm_guided/all_domains/llm_20260114_183940/
```

### 4. Reproduce all paper figures

```bash
python tools/visualizations/create_visualizations.py
python tools/utils/regenerate_figures.py
python tools/visualizations/create_visualizations.py --defi73   # Figs 11–13
# → data/results/figures/
```

---

## Full Reproduction Run Order

> All paths relative to `papers/2025-JMLR/hypatiax/`. Run in order — each campaign's outputs feed the next.

```bash
# ── Campaign 1: Pure LLM Baseline (~10 min, no Julia needed) ─────────────
python core/base_pure_llm/baseline_pure_llm.py
python core/base_pure_llm/baseline_pure_llm_defi.py
# → data/results/standalone_llm_nn/

# ── Campaign 2: Pure Symbolic PySR (~2 h, Julia required) ────────────────
python protocols/experiment_protocol_all_18_a.py --seed 42 --no-cache --v2
# → data/results/hybrid_pysr/all_domains/llm_20260113_112048/

# ── Campaign 3: LLM-Guided Hybrid v40 (~35 min) ───────────────────────────
python protocols/experiment_protocol_all_30.py --seed 42 --no-cache --v2
# → data/results/llm_guided/all_domains/llm_20260114_183940/

# ── Campaign 4: DeFi Suite (~45 min) ─────────────────────────────────────
python protocols/experiment_protocol_defi_20.py --seed 42 --no-cache --v2
# → data/results/hybrid_pysr/defi/

# ── Campaign 5: Comparative Analysis & Statistics ─────────────────────────
# Step 5a — cross-system comparative run (canonical script, copied from legacy/):
#   cp legacy/experiments/ultimate_comparative_suite_complete_.py \
#      experiments/comparison/
python experiments/comparison/ultimate_comparative_suite_complete_.py
# → data/results/comparison_results/extrapolation/comparison_FIXED_<ts>.json

# Step 5b — merge all campaign outputs (reads Campaign 1 + Step 5a outputs)
python experiments/comparison/merge_all_systems.py
# → data/results/to_generate_figures/all_systems_merged.json

# Step 5c — independent cross-system comparison (does not depend on 5b)
python experiments/comparison/test_suite_comparative_v3.py
# → data/results/comparison_results/strategic_tests_<domain>_<ts>.json

python analysis/statistical_analysis_full.py --v2
# → data/results/comparison_results/

# ── Campaign 6: DeFi Extrapolation 73-Case Benchmark (v2) (~44 min) ──────
python experiments/tests/test_enhanced_defi_extrapolation.py \
    --fixed-denominator 66 --nan-penalty --v2
# → data/results/extrapolation/extrapolation_73cases_enhanced.json

# ── Campaign 7: Feynman SR Benchmark (v2, ~73 min, Julia required) ────────
python experiments/benchmarks/run_comparative_suite_benchmark_v2.py \
    --noiseless --threshold 0.9999 --nn-seeds 3 --samples 200 \
    --method-timeout 900 --pysr-timeout 900 --v2
python experiments/benchmarks/run_noise_sweep_benchmark.py
python experiments/benchmarks/run_sample_complexity_benchmark.py
# → data/results/comparison_results/noise-noiseless/noiseless/
# → data/results/comparison_results/feynman-tests/noise-sweep/
```

---

## Benchmarks

| Benchmark | Equations | Domains | Canonical script |
|-----------|-----------|---------|-----------------|
| Core 15 | 15 | Physics, Biology/Chem, DeFi AMM, DeFi Risk | `protocols/experiment_protocol_all_30.py` |
| DeFi Suite | 23 | DeFi AMM, Risk, Liquidity, ES, Liquidation, Staking | `protocols/experiment_protocol_defi_20.py` |
| DeFi Extrapolation | 73 (66 standard) | DeFi | `experiments/tests/test_enhanced_defi_extrapolation.py` |
| Feynman SR | 30 | Physics | `experiments/benchmarks/run_comparative_suite_benchmark_v2.py` |
| Comparative suite | All | All | `experiments/comparison/test_suite_comparative_v3.py` |

---

## v2 Note (March 2026)

A measurement bug in `evaluate_llm_formula` was corrected before final paper
submission. The fix replaces a hardcoded absolute sum-of-squares threshold with
a relative one that scales correctly for small-magnitude outputs (gravitational
forces ~10⁻¹¹ N, Zeeman splittings ~10⁻²³ J).

**If `data/results/` predates March 2026, regenerate:**

```bash
python protocols/experiment_protocol_all_30.py --no-cache
```

**Cohen's d correction:** The paper reports d = 0.95 for Core 15 (NOT 3.21).
The hybrid distribution is degenerate (all errors ≈ 0); pooled d understates
the true separation. Mann-Whitney U = 0 is the primary statistical claim.
See Appendix `app:statistical_tests`.

**DeFi 73-case denominator:** Always use fixed n = 66 for cross-method
comparison (7 intractable cases excluded). Raw per-method rates
(83.6% / 77.4%) are not comparable across methods.

---

## Key Results

| System | Core 15 success | Extrapolation error (median) | Feynman R²>0.9999 |
|--------|----------------|------------------------------|-------------------|
| **HypatiaX Hybrid DeFi** | **95.8%** | **< 10⁻¹²** | **96.7%** |
| HypatiaX v40 | 95.8% | < 10⁻¹² | 90.0% |
| AI Feynman (prior SotA) | — | — | 79.3% |
| Neural Network | — | > 1,200% | 56.7% |
| Pure LLM | 60.0% | N/A | — |

Mann-Whitney U = 0, p = 1.11×10⁻⁶ (complete rank separation,
hybrid vs neural network, Core 15 extrapolation errors).

---

## Claim Verification

The full traceability map is in `../code/tracer_output/`
(`papers/2025-JMLR/code/tracer_output/`):

| File | Purpose |
|------|---------|
| `reviewer_workflow.md` | Section-by-section verification guide |
| `reviewer_guide.html` | Colour-coded HTML — open in browser |
| `traceability_map.json` | Machine-readable claim → artefact map |

To regenerate:

```bash
python ../code/scripts/paper_repo_tracer.py \
    --tex  data/results/TEXFILES/LAST/jmlr_paper_final.tex \
    --repo . \
    --out  ../code/tracer_output/
```

### Quick-reference

| Claim | Script | Result file | JSON key |
|-------|--------|-------------|----------|
| Mann-Whitney U=0, p=1.11×10⁻⁶ | `analysis/statistical_analysis_full.py` | `comparison_results/methods-all/15/comparison_FIXED_20260124_150744.json` | `mann_whitney.U / .p` |
| Hybrid 96.7% (29/30) | `protocols/experiment_protocol_all_30.py` | `llm_guided/.../checkpoint.json` | `passed / total_tests` |
| NN error 1,231% | `experiments/comparison/test_suite_comparative_v3.py` | `extrap/all_domains_extrap_v4_20260124_131545.json` | `neural_network.extrap_error_pct_mean` |
| DeFi hybrid 72.7% (n=66) | `experiments/tests/test_enhanced_defi_extrapolation.py` | `extrapolation/extrapolation_73cases_enhanced.json` | `honest_comparison.hybrid.r2_99_rate` |
| Feynman 96.7% recovery | `experiments/benchmarks/run_comparative_suite_benchmark_v2.py` | `comparison_results/noise-noiseless/noiseless/protocol_core_noiseless_20260304_154510.json` | `results.Hybrid_DeFi.recovery_rate` |

---

## Code Quality

```bash
cd papers/2025-JMLR
python code/scripts/code_quality_auditor.py \
    --repo hypatiax/ \
    --out  code/quality_output/
```

Current status (March 2026):

| Check | Status |
|-------|--------|
| Canonical scripts with seed | ✅ 16/16 |
| Critical issues | ✅ 0 |
| Non-canonical in live tree | ✅ 0 (18 archived to `legacy/`) |
| Requirements.txt | ✅ `papers/2025-JMLR/requirements.txt` |

---

## Tutorials

Step-by-step walkthroughs for reproducing the paper's analyses:

1. [Tutorial 1: Installation and Setup](supplementaries/tutorials/tutorial-1-setup/)
2. [Tutorial 2: Running Experiments](supplementaries/tutorials/tutorial-2-experiments/)
3. [Tutorial 3: Statistical Analysis and Publication Figures](supplementaries/tutorials/tutorial-3-analysis/)
4. [Tutorial 4: Custom Applications](supplementaries/tutorials/tutorial-4-extensions/)

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/01_data_generation.ipynb` | Generate synthetic test data |
| `notebooks/02_pure_llm_experiments.ipynb` | Campaign 1 walkthrough |
| `notebooks/03_hybrid_experiments.ipynb` | Campaigns 2–4 walkthrough |
| `notebooks/04_extrapolation_analysis.ipynb` | Extrapolation testing + DeFi 73 |
| `notebooks/05_figure_generation.ipynb` | Reproduce all paper figures |
| `notebooks/06_statistical_tests.ipynb` | Statistical validation |

---

## Legacy

`legacy/` contains archived non-canonical scripts kept for development
transparency. **Do not run these** — they are superseded by the canonical
scripts listed above.

---

## Citation

```bibtex
@article{bonetchaple2026hypatiax,
  title     = {HypatiaX: A Hybrid Symbolic-Neural Framework for Extrapolation-Reliable Analytical Discovery},
  author    = {Bonet Chaple, Ruperto Pedro},
  journal   = {Journal of Machine Learning Research},
  year      = {2026}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
The benchmark datasets in `data/` and LaTeX source in `paper/` are released
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

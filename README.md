# LLM-HypatiaX-DEV

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-yellow.svg)](requirements.txt)
[![Status](https://img.shields.io/badge/Status-Internal%20Dev-orange.svg)](#)

**Internal development repository** for HypatiaX, a hybrid symbolic-neural
system for scientific equation discovery combining symbolic regression
(PySR), LLM interpretation, and multi-layer validation.

> ⚠️ **This is the working/dev repo, not the public release.** It contains
> experimental branches, multiple orchestration backends, legacy code, and
> in-progress CI infrastructure. For the clean, reproducible JMLR paper
> artifact, see **`LLM-HypatiaX-PAPERS-Public`**.

---

## 📁 Repository Layout

This repo is a monorepo covering the core library, several parallel
execution/orchestration variants (local, AWS, cloud, checkpointed, parallel,
Windows), CI/CD workflows, audit tooling, and paper source files.

```
LLM-HypatiaX-DEV/
├── hypatiax/                    # Core package (source of truth)
│   ├── analysis/                 # Statistical analysis scripts
│   ├── audit/                    # Provenance & pipeline-trace auditing
│   ├── config/                   # repro.yaml, key-status checks
│   ├── core/
│   │   ├── base_pure_llm/         # Pure-LLM baselines
│   │   ├── generation/            # Hybrid system architectures
│   │   ├── runners/                # Shared run harness
│   │   └── training/               # NN training & adaptive config
│   ├── data/results/              # Experimental outputs (large — see note)
│   ├── experiments/
│   │   ├── benchmarks/             # Noise-sweep, sample-complexity, hybrid
│   │   ├── comparison/             # Cross-system comparative suite
│   │   └── tests/                  # Extrapolation & DeFi test protocols
│   ├── figures/                    # Generated plots (png/pdf)
│   ├── protocols/                  # Experiment protocol scripts
│   ├── reproducibility/            # Hash-locking, repro guides
│   ├── shared/                     # Shared utilities
│   └── tools/
│       ├── symbolic/                # HypatiaX engine (v40/v50/v52) & detectors
│       ├── utils/                    # JSON, figure, comparison helpers
│       ├── validation/               # Dimensional, domain, ensemble validators
│       └── visualizations/           # Plot & figure generation
│
├── hypatiax-all/                 # Standalone "run everything" variant
├── hypatiax-all-aws/             # AWS-orchestrated experiment runner
├── hypatiax-all-checkpoint/      # Checkpointed/resumable experiment runner
├── hypatiax-all-cloud/           # Generic cloud CI dispatch variant
├── hypatiax-orchestrator/        # Sequential orchestration + reviewer docs
├── hypatiax_parallel_arch/       # Parallel dispatch architecture (DAG, sharding)
├── hypatiax-windows/             # Windows PowerShell entry point
├── no-used-hypatiax/             # Retired/legacy experiment code (reference only)
│
├── .github/
│   ├── dependabot.yml
│   ├── scripts/                  # CI helper scripts (sweep coverage, symbolic equiv, etc.)
│   └── workflows*/               # Active + archived workflow sets
│       ├── workflows/              # Current CI pipelines
│       ├── workflows_may24/        # Snapshot archive
│       ├── workflows_old/          # Deprecated pipelines
│       ├── workflows_parallel/     # Parallel-execution pipelines
│       ├── workflows_parallel_simplify/
│       └── workflows_simplify/     # Simplified checkpoint pipelines
│
├── notebooks/                    # Paper-integrity audit notebooks (NB-01…NB-06)
├── paper/                        # LaTeX paper source (final + supplements)
├── scripts/                      # Repo-level build/patch/audit scripts
│   ├── figures/                    # Figure cleanup & comparison tools
│   ├── patches/                    # Automated patch generation/application
│   └── paper/tables/               # Generated LaTeX tables
├── tables/                       # Top-level generated LaTeX tables
│
├── activate_hypatiax.sh
├── setup_environment.sh
├── requirements.txt
├── pyproject.toml
├── setup.py
├── jmlr_paper_main.tex
├── references.bib
├── VERSION
├── LICENSE
└── README.md                     # ← you are here
```

---

## 🧭 Which entry point do I use?

| Variant | Use when |
|---|---|
| `hypatiax/` | Developing or debugging core logic, running a single experiment locally |
| `hypatiax-all/` | Running the full experiment suite locally, single machine |
| `hypatiax-all-checkpoint/` | Long-running suites that need resume/checkpoint support |
| `hypatiax-all-aws/` | Dispatching experiments to AWS (see `hypatiax-all-aws/AWS/`) |
| `hypatiax-all-cloud/` | Generic CI-triggered cloud dispatch |
| `hypatiax_parallel_arch/` | Sharded/parallel dispatch across many jobs |
| `hypatiax-orchestrator/` | Sequential, reviewer-facing orchestration with guide docs |
| `hypatiax-windows/` | Windows-only local runs (`run_all.ps1`) |

Each variant has its own `run_all.sh` (or `.ps1`) and, in most cases, its own
copy of `protocols/` and `experiments/benchmarks/` — these are intentionally
duplicated per-variant rather than shared, so check you're editing the copy
that matches the entry point you're running.

---

## 🚀 Quick Start (local, core package)

```bash
git clone <this-repo-url> LLM-HypatiaX-DEV
cd LLM-HypatiaX-DEV
bash setup_environment.sh
source activate_hypatiax.sh
pip install -r requirements.txt        # Julia ≥ 1.9 required for PySR campaigns
```

Run the full local suite:

```bash
bash run_all.sh
# or, for a resumable run:
python run_all_checkpoint.py
```

Reproduction and validation details: **[`hypatiax/README.md`](hypatiax/README.md)**.

---

## 🧪 CI / Workflows

Active pipelines live in `.github/workflows/` and cover analysis, PDF/report
generation, model validation, paper audits, notebook checks, PCA split
testing, pipeline tracing, and cache/PR cleanup. Older or experimental
workflow sets are kept side-by-side under `workflows_old/`,
`workflows_parallel/`, `workflows_parallel_simplify/`, and
`workflows_simplify/` for reference — these are not run automatically and
may be pruned over time.

Repo-level audit and patch tooling lives in `scripts/patches/` (e.g.
`gate_c3_verify.py`, `hypatia_inspector.py`, `validate_code.py`) and is
wired into the `ci_paper_audit.yml` / `ci_report*.yml` pipelines.

---

## 📓 Paper Integrity Notebooks

`notebooks/` contains audit notebooks used to check the paper draft against
the code and data before submission:

| Notebook | Checks |
|---|---|
| `NB-01_Citation_Bibliography_Audit.ipynb` | Citations vs. `references.bib` |
| `NB-02_CrossReference_Label_Audit.ipynb` | LaTeX label/ref consistency |
| `NB-03_Section_Structure_Numbering.ipynb` | Section/heading numbering |
| `NB-04_Numerical_Consistency_Checker.ipynb` | Reported numbers vs. results JSON |
| `NB-05_Figure_Image_Dependency_Checker.ipynb` | Figure references vs. files on disk |
| `NB-06_Code_Quality_Pipeline_Integrity.ipynb` | Pipeline/lint integrity checks |

---

## 📦 Notes on `data/results/`

Experimental outputs (benchmarks, ablations, extrapolation runs, noise
sweeps, PCA splits) are organized under `hypatiax/data/results/`, mirrored
per-variant in some of the orchestration folders above. This tree is large
and includes many duplicated/renamed figure copies (`figures__figures__...`)
left over from merge/patch operations — see `scripts/figures/` and
`scripts/patches/` for the cleanup and dedup tooling used to manage this.

---

## 🔬 Reproducibility

All canonical scripts use `SEED = 42`. See `hypatiax/reproducibility/` for
hash-locking (`hash_lock.py`) and step-by-step reproduction guides. The
`evaluate_llm_formula` v2 correction (March 2026) is the baseline for all
current results — see `hypatiax/README.md` for details.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

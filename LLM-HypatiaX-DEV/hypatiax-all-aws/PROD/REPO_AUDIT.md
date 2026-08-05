# HypatiaX — repository audit & consolidation guide

> **Scope:** Cross-analysis of `LLM-HypatiaX-PAPERS-Public` (455 files, 106 dirs),
> `LLM-HypatiaX-PAPERS` (reprod), and `LLM-HypatiaX-Colab`.
> Goal: one clean public repo with local / GitHub CI / AWS all supported from the
> same codebase.

---

## Table of contents

1. [Summary scorecard](#1-summary-scorecard)
2. [Security — act first](#2-security--act-first)
3. [Duplicate entry points](#3-duplicate-entry-points)
4. [Duplicate experiment directories](#4-duplicate-experiment-directories)
5. [CI workflows — wrong location](#5-ci-workflows--wrong-location)
6. [Protocols — shadow copies](#6-protocols--shadow-copies)
7. [Output generators](#7-output-generators)
8. [Audit / trace scripts](#8-audit--trace-scripts)
9. [Committed generated outputs](#9-committed-generated-outputs)
10. [Deprecated / unused code](#10-deprecated--unused-code)
11. [Stale editor artefacts](#11-stale-editor-artefacts)
12. [Files to add to reprod](#12-files-to-add-to-reprod)
13. [Target repo structure](#13-target-repo-structure)
14. [.gitignore additions](#14-gitignore-additions)
15. [Action checklist](#15-action-checklist)

---

## 1. Summary scorecard

| Category | Count | Action |
|---|---|---|
| Security risks | 4 | Purge from git history immediately |
| Duplicate entry points | 8 | Keep 1, delete 7 |
| Duplicate experiment dirs | 2 extra copies | Delete, keep `hypatiax/experiments/benchmarks/` |
| CI workflows in wrong location | 4 files | Move to `.github/workflows/` |
| Protocol shadow copies | 2+ copies each | Consolidate to `hypatiax/protocols/` |
| Committed generated outputs | ~50 files | Remove + gitignore |
| Deprecated / unused code | 1 full dir + 10 files | Delete |
| Stale editor artefacts | 2 | Delete + gitignore |
| Files missing from reprod | 6 | Add |

---

## 2. Security — act first

**Do these before any other step.** Deleting files is not enough — they must be
removed from git history.

### Files to purge

| File | Location | Risk |
|---|---|---|
| `secrets.py` | reprod root | Likely contains API keys |
| `secrets-old.py` | reprod root | Previous key version |
| `secrets-copy.txt` | reprod root | Plaintext copy of keys |
| `config_secrets.py` | `hypatiax/` | May import or reference keys |

### How to purge

```bash
# Install git-filter-repo (preferred over git filter-branch)
pip install git-filter-repo

# Run from repo root — repeat for each file
git filter-repo --path secrets.py --invert-paths
git filter-repo --path secrets-old.py --invert-paths
git filter-repo --path secrets-copy.txt --invert-paths

# For config_secrets.py: audit first
grep -n "key\|token\|secret\|password\|api" hypatiax/config_secrets.py
# If it contains hardcoded values → purge; if it only reads from env → keep but audit imports
```

### After purging

```bash
# Force-push all branches (coordinate with collaborators)
git push origin --force --all
git push origin --force --tags

# Rotate any exposed keys immediately via their provider dashboard
```

### Prevention

Add to `.gitignore`:
```
secrets*.py
secrets*.txt
*.env
config_secrets.py   # if purged
```

---

## 3. Duplicate entry points

**Problem:** The repo has 8+ versions of `run_all` — the canonical entrypoint is
ambiguous.

| File | Location | Action |
|---|---|---|
| `run_all.sh` | `./` | **Keep — canonical** |
| `run_all.sh` | `hypatiax-all-cloud/` | Delete |
| `run_all.sh` | `hypatiax/RUN_ALL_OLD/` | Delete (entire `RUN_ALL_OLD/` dir) |
| `run_all.sh` | `hypatiax-all-checkpoint/RUN_ALL/` | Delete |
| `run_all_checkpoint.py` | `./` | Delete |
| `run_all_injected.sh` | `hypatiax/RUN_ALL_OLD/` | Delete |
| `run_all_orchestrator.sh` | `hypatiax-all-checkpoint/RUN_ALL_INJECTED/` | Delete |
| `repro_master.py` | `./` | Delete — superseded by CI |
| `recover_failed_experiments.py` | `./` | Delete — one-time script |

### Clean up

```bash
rm -rf hypatiax/RUN_ALL_OLD/
rm -rf hypatiax-all-checkpoint/RUN_ALL/
rm -rf hypatiax-all-checkpoint/RUN_ALL_INJECTED/
rm    hypatiax-all-cloud/run_all.sh
rm    run_all_checkpoint.py
rm    repro_master.py
rm    recover_failed_experiments.py
```

---

## 4. Duplicate experiment directories

**Problem:** The same 10–11 experiment scripts exist in three locations.

| Directory | Files | Action |
|---|---|---|
| `hypatiax/experiments/benchmarks/` | 11 scripts | **Keep — canonical** |
| `hypatiax-all-checkpoint/experiments/benchmarks/` | 11 scripts (identical) | Delete entire dir |
| `hypatiax-all-cloud/experiments/` | 10 scripts (identical) | Delete entire dir |

```bash
# Verify they are truly identical before deleting
diff -rq hypatiax/experiments/benchmarks/ hypatiax-all-checkpoint/experiments/benchmarks/
diff -rq hypatiax/experiments/benchmarks/ hypatiax-all-cloud/experiments/

# Then delete
rm -rf hypatiax-all-checkpoint/experiments/
rm -rf hypatiax-all-cloud/experiments/
```

### One script not yet in canonical location

`exp1_ablation.py` exists in `hypatiax-all-checkpoint/experiments/benchmarks/`
and `hypatiax-all-cloud/experiments/` but **not** in
`hypatiax/experiments/benchmarks/`. Promote it:

```bash
cp hypatiax-all-checkpoint/experiments/benchmarks/exp1_ablation.py \
   hypatiax/experiments/benchmarks/
```

---

## 5. CI workflows — wrong location

**Problem:** All four CI workflows live inside `hypatiax-all-cloud/` instead of
`.github/workflows/`. GitHub Actions does not pick them up from there.

| File | Current location | Target location |
|---|---|---|
| `ci_experiment.yml` | `hypatiax-all-cloud/` | `.github/workflows/` |
| `ci_consolidate_experiment.yml` | `hypatiax-all-cloud/` | `.github/workflows/` |
| `ci_schedule_all.yml` | `hypatiax-all-cloud/` | `.github/workflows/` |
| `dispatch_experiment.sh` | `hypatiax-all-cloud/` | `.github/scripts/` |
| `ci_trace_pipeline.yml` | *(new — not yet created)* | `.github/workflows/` |

```bash
mkdir -p .github/workflows .github/scripts

mv hypatiax-all-cloud/ci_experiment.yml            .github/workflows/
mv hypatiax-all-cloud/ci_consolidate_experiment.yml .github/workflows/
mv hypatiax-all-cloud/ci_schedule_all.yml           .github/workflows/
mv hypatiax-all-cloud/dispatch_experiment.sh        .github/scripts/
# Add ci_trace_pipeline.yml (already written separately)
```

### AWS CI files — retire

These were AWS/GCP experiments and are no longer needed:

```bash
# Inside hypatiax-all-aws/
rm ci.yml ci-aws.yml ci-gcp.yml ci-change-arch2.yml buildspec.yml
```

---

## 6. Protocols — shadow copies

**Problem:** Protocol files exist in three locations simultaneously.

| Location | Files | Action |
|---|---|---|
| `hypatiax/protocols/` | 3 files (all_30, benchmark_v2, defi) | **Keep — canonical** |
| `hypatiax-all-checkpoint/protocols/` | 14 files (full set) | Merge missing ones up, then delete dir |
| `hypatiax-all-cloud/protocols/` | 3 files (same as hypatiax/protocols/) | Delete dir |
| `no-used-hypatiax/protocols/` | 7 deprecated versions | Delete dir |

### Merge and consolidate

```bash
# Copy the 11 files missing from hypatiax/protocols/ (from checkpoint)
MISSING=(
  _base.py
  experiment_protocol_ablation_exp1.py
  experiment_protocol_defi_v3.py
  experiment_protocol_extrapolation_comparative.py
  experiment_protocol_feynman_exp2.py
  experiment_protocol_hybrid_routing.py
  experiment_protocol_instability_rf02_04.py
  experiment_protocol_nguyen12_exp3.py
  experiment_protocol_noise_sweep.py
  experiment_protocol_provenance_audit.py
  universal_protocol.py
)
for f in "${MISSING[@]}"; do
  cp hypatiax-all-checkpoint/protocols/$f hypatiax/protocols/
done

# Delete shadow copies
rm -rf hypatiax-all-checkpoint/protocols/
rm -rf hypatiax-all-cloud/protocols/
rm -rf no-used-hypatiax/protocols/
```

---

## 7. Output generators

| File | Location | Action |
|---|---|---|
| `generate_figures.py` | `figures/` | **Keep — canonical** |
| `generate_figures.py` | `scripts/patches/` | Delete — duplicate |
| `generate_tables.py` | `tables/` | **Keep — canonical** |
| `generate_tables.py` | `scripts/patches/` | Delete — duplicate |

```bash
rm scripts/patches/generate_figures.py
rm scripts/patches/generate_tables.py
```

### scripts/patches/ — what to keep

| File | Keep? | Reason |
|---|---|---|
| `apply_patches.py` | Yes | reprod utility |
| `generate_patches.py` | Yes | reprod utility |
| `verify_results.py` | Yes | reprod utility |
| `validate_code.py` | Yes | reprod utility |
| `clean_results.py` | Yes | reprod utility |
| `check_hypatiax_protocols.py` | Yes | pub utility |
| `patch_log.jsonl` | Audit | May contain sensitive paths — check before keeping |
| `pipeline--patches.txt` | Audit | Same |
| `generated/*.patch.json` | Gitignore | Generated — do not commit |

---

## 8. Audit / trace scripts

**Problem:** Trace/provenance scripts are split between two locations and not
unified.

| File | Location | Action |
|---|---|---|
| `pipeline_trace.py` | `audit/` (public) | **Base of tracer — keep** |
| `discover_provenance.py` | `audit/` (public) | Keep |
| `provenance_audit.py` | `audit/` (public) | Keep |
| `scan_internal_imports.py` | `audit/` (public) | Keep |
| `trace_pipeline_1_.py` | reprod root | Move to `audit/`, rename `trace_pipeline.py` |
| `ci_trace_pipeline.yml` | *(new)* | Add to `.github/workflows/` |

```bash
mv trace_pipeline_1_.py audit/trace_pipeline.py
# Update ci_trace_pipeline.yml: TRACE_SCRIPT="audit/trace_pipeline.py"
```

---

## 9. Committed generated outputs

These files are the **product** of running experiments — they should never be in
git. They bloat the repo, cause merge conflicts, and are reproducible from code.

### Root-level results (delete + gitignore)

```bash
rm exp1_ablation_checkpoint.json
rm exp1_ablation_results.json
rm exp1_ablation_table.tex
rm exp1_instability_stats.json
rm exp1_rf01_mannwhitney.json
rm instability_extrapolation_v2.csv
rm provenance_map_exp1.json
rm clean_exp2_sym_checkpoint.py   # one-time cleanup script
```

### hypatiax/data/results/ (gitignore entire dir)

```bash
# Keep the directory structure, remove committed files
git rm -r --cached hypatiax/data/results/
# Contents are reproduced by running the pipeline
```

### logs/ (gitignore)

```bash
git rm -r --cached logs/
# Exception: logs/README.md — keep if it has instructions
```

### scripts/paper/tables/*.tex (gitignore)

```bash
git rm --cached scripts/paper/tables/*.tex
```

### scripts/patches/generated/ (gitignore)

```bash
git rm -r --cached scripts/patches/generated/
```

---

## 10. Deprecated / unused code

### `no-used-hypatiax/` — delete entire directory

The directory name is self-documenting. Contains old experiments and deprecated
protocol versions that have been superseded.

```bash
rm -rf no-used-hypatiax/
```

### `hypatiax-all-checkpoint/` — delete after migration

Once protocols and experiment scripts are merged into canonical locations
(sections 4 and 6), the entire dir is redundant:

```bash
rm -rf hypatiax-all-checkpoint/
```

### `hypatiax-all-aws/` — archive or move

This contains legitimate AWS infrastructure docs but should not live at repo
root. Options:

- Move useful files to `aws/` (docs, `merge_exp2_shards.py`, `PROD/`)
- Delete CI files already covered by `.github/workflows/`

```bash
mkdir -p aws/docs
mv hypatiax-all-aws/AWS-GUIDE.txt         aws/docs/
mv hypatiax-all-aws/AWS-GUIDE-REVIEWER.txt aws/docs/
mv hypatiax-all-aws/budget-to-results.md  aws/docs/
mv hypatiax-all-aws/merge_exp2_shards.py  aws/
mv hypatiax-all-aws/PROD/                 aws/prod/
rm -rf hypatiax-all-aws/
```

### `HypatiaX_progressive_working_colab.ipynb` (root)

Working notebook committed at root — move to `notebooks/` or delete:

```bash
mv HypatiaX_progressive_working_colab.ipynb notebooks/
```

### Duplicate paper source files at root

```bash
# paper/ dir is canonical
rm jmlr-hypatiax-paper-final.tex   # duplicate of paper/jmlr-hypatiax-paper-final.tex
rm supp_benchmark_report.tex       # duplicate of paper/supp_benchmark_report.tex
rm supp_routing_improvements.tex   # duplicate of paper/supp_routing_improvements.tex
```

---

## 11. Stale editor artefacts

Emacs autosave files were accidentally committed:

```bash
git rm '#run_all.sh#'
git rm '#run_all.sh#~'
```

Add to `.gitignore`:
```
\#*\#
*~
.#*
```

Also remove all `__pycache__/` dirs and `.pyc` files:

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
find . -name "*.pyc" -delete
```

---

## 12. Files to add to reprod

Files that exist in the public repo but are missing from the reprod repo and
should be added:

| File | Public location | Add to reprod at |
|---|---|---|
| `LICENSE` | `./` | `./` |
| `validation_paper_config.py` | `./` | `./` |
| `audit/pipeline_trace.py` | `audit/` | `audit/` |
| `audit/discover_provenance.py` | `audit/` | `audit/` |
| `audit/provenance_audit.py` | `audit/` | `audit/` |
| `audit/scan_internal_imports.py` | `audit/` | `audit/` |
| `exp1_ablation.py` | `hypatiax-all-checkpoint/` | `hypatiax/experiments/benchmarks/` |
| `purge_actions_cache.sh` | `./` | `.github/scripts/` |

---

## 13. Target repo structure

```
LLM-HypatiaX-PAPERS-Public/
├── .github/
│   ├── workflows/
│   │   ├── ci_experiment.yml
│   │   ├── ci_consolidate_experiment.yml
│   │   ├── ci_schedule_all.yml
│   │   └── ci_trace_pipeline.yml
│   └── scripts/
│       ├── dispatch_experiment.sh
│       ├── print_repro.py
│       ├── purge_actions_cache.sh
│       └── test_key_status.py
│
├── hypatiax/                          ← package — identical in public + reprod
│   ├── analysis/
│   ├── core/
│   │   ├── base_pure_llm/
│   │   ├── generation/
│   │   ├── runners/
│   │   └── training/
│   ├── experiments/
│   │   └── benchmarks/               ← all 12 experiment scripts here only
│   ├── protocols/                    ← all 14 protocol files here only
│   ├── reproducibility/
│   │   └── hash_lock.py
│   ├── shared/
│   │   └── utilities.py
│   └── tools/
│
├── audit/                             ← trace + provenance tools
│   ├── trace_pipeline.py
│   ├── discover_provenance.py
│   ├── provenance_audit.py
│   └── scan_internal_imports.py
│
├── config/
│   ├── repro.yaml                     ← hyperparams — single source of truth
│   ├── envs/
│   │   ├── local.yaml
│   │   ├── ci.yaml
│   │   └── aws.yaml
│   └── test_key_status.py
│
├── figures/
│   └── generate_figures.py
│
├── tables/
│   └── generate_tables.py
│
├── scripts/
│   └── patches/
│       ├── apply_patches.py
│       ├── generate_patches.py
│       ├── verify_results.py
│       ├── validate_code.py
│       ├── clean_results.py
│       └── check_hypatiax_protocols.py
│
├── notebooks/
│   ├── HypatiaX_Experiments_v7_PAPER_QUALITY.ipynb
│   ├── HypatiaX_Experiments_v7_PUBLIC_fast.ipynb
│   ├── HypatiaX_pipeline.ipynb
│   └── NB-01..NB-06_*.ipynb
│
├── paper/
│   ├── jmlr-hypatiax-paper-final.tex
│   ├── supp_benchmark_report.tex
│   └── supp_routing_improvements.tex
│
├── aws/
│   ├── run_aws.sh
│   ├── merge_exp2_shards.py
│   ├── docs/
│   └── prod/
│
├── docs/
│   ├── REPRODUCTION_GUIDE.md
│   ├── COMPATIBILITY-LIST.txt
│   └── protocols.txt
│
├── hypatiax/data/results/             ← gitignored, populated by CI
├── logs/                              ← gitignored
│
├── activate_hypatiax.sh
├── install.sh
├── run_all.sh                         ← single canonical entrypoint
├── Makefile
├── requirements.txt
├── requirements-exact.txt
├── pyproject.toml
├── setup.py
├── VERSION
├── LICENSE
├── README.md
└── REPRODUCIBILITY.md
```

---

## 14. .gitignore additions

Add these to `.gitignore` if not already present:

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# Secrets
secrets*.py
secrets*.txt
*.env
.env.*
config_secrets.py

# Editor artefacts
\#*\#
*~
.#*
*.orig
*.bak

# Generated outputs — reproduced by pipeline
hypatiax/data/results/
logs/*.json
logs/*.zip
logs/*.jsonl
exp1_ablation_checkpoint.json
exp1_ablation_results.json
exp1_ablation_table.tex
exp1_instability_stats.json
exp1_rf01_mannwhitney.json
instability_extrapolation_v2.csv
provenance_map_exp1.json
scripts/paper/tables/*.tex
scripts/patches/generated/
patches/generated/

# Checkpoints
*.checkpoint.json
*_checkpoint.json
*_shard*.json

# Build
*.egg-info/
dist/
build/
```

---

## 15. Action checklist

Work through these in order — security first, then structure, then cleanup.

### Immediate (today)

- [ ] Audit `config_secrets.py` for hardcoded keys
- [ ] Purge `secrets.py`, `secrets-old.py`, `secrets-copy.txt` from git history
- [ ] Rotate any exposed API keys
- [ ] Add secrets patterns to `.gitignore`
- [ ] Delete Emacs autosave files (`#run_all.sh#`, `#run_all.sh#~`)

### Structure (this week)

- [ ] Move CI workflows from `hypatiax-all-cloud/` to `.github/workflows/`
- [ ] Move `dispatch_experiment.sh` to `.github/scripts/`
- [ ] Add `ci_trace_pipeline.yml` to `.github/workflows/`
- [ ] Promote `exp1_ablation.py` to `hypatiax/experiments/benchmarks/`
- [ ] Consolidate all protocols into `hypatiax/protocols/`
- [ ] Move `trace_pipeline_1_.py` to `audit/trace_pipeline.py`
- [ ] Create `config/envs/local.yaml`, `ci.yaml`, `aws.yaml`

### Cleanup (this week)

- [ ] Delete `hypatiax-all-checkpoint/` (after migration)
- [ ] Delete `hypatiax-all-cloud/experiments/` and `protocols/` (after migration)
- [ ] Delete `no-used-hypatiax/`
- [ ] Delete duplicate `run_all*` entry points (keep only root `run_all.sh`)
- [ ] Delete duplicate paper `.tex` files at root (keep `paper/` copies)
- [ ] Remove and gitignore `scripts/patches/generate_figures.py` and `generate_tables.py`

### Generated outputs (this week)

- [ ] Run `git rm -r --cached hypatiax/data/results/ logs/`
- [ ] Remove root-level result JSONs and `.tex` files
- [ ] Remove `scripts/patches/generated/` from git
- [ ] Update `.gitignore` with full list from section 14
- [ ] Commit and push `.gitignore` changes

### AWS infra (next sprint)

- [ ] Consolidate `hypatiax-all-aws/` useful files into `aws/`
- [ ] Write `aws/run_aws.sh` wrapping `run_all.sh` with S3 upload + teardown
- [ ] Delete `hypatiax-all-aws/` after migration

### Verify

- [ ] Run `trace_pipeline.py --shell run_all.sh --repo-root .` — expect 0 errors
- [ ] Run `ci_trace_pipeline.yml` via `workflow_dispatch` — expect green
- [ ] Run `python -c 'from hypatiax.protocols.universal_protocol import *; print("OK")'`
- [ ] Run one experiment end-to-end locally: `HYPATIAX_ENV=local bash run_all.sh exp1`

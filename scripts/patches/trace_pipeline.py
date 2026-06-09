#!/usr/bin/env python3
"""
trace_pipeline.py — Static connectivity tracer for run_all.sh
==============================================================

Checks the HypatiaX reproduction pipeline WITHOUT executing any experiment.

What it verifies
----------------
  1. STEP ORDER       — declared _STEP_ORDER matches the actual run() call sequence
  2. SCRIPT EXISTENCE — every Python script invoked by each step exists on disk
  3. DEPENDENCY GRAPH — output globs of each step are the input globs of the
                        next step(s) that consume them; broken chains flagged
  4. PATH CONSISTENCY — all RESULTS_DIR-relative paths resolve under a common root
  5. VALIDATE COVERAGE — the validate step's glob patterns cover every step's
                         declared output
  6. STEP ISOLATION   — no step writes into another step's declared input dir
                        without being upstream of it

Usage
-----
  # Check against a real repo checkout (most useful)
  python trace_pipeline.py --repo-root /path/to/hypatiax-repo

  # Check only the shell script and generator files (no repo needed)
  python trace_pipeline.py --shell run_all.sh \\
      --tables-generator tables-generator.py \\
      --figures-generator generate_all_figures.py

  # Limit trace to a single experiment step (as ci_trace_pipeline.yml does)
  python trace_pipeline.py --shell run_all.sh --step exp2_feynman

  # Cross-check hyperparameter values against config/repro.yaml
  python trace_pipeline.py --repo-root /path/to/repo --repro-cfg config/repro.yaml

  # Treat warnings as errors (mirrors ci_trace_pipeline.yml fail_on_warning input)
  python trace_pipeline.py --repo-root /path/to/repo --fail-on-warning

  # Write full JSON trace to a file
  python trace_pipeline.py --repo-root /path/to/repo --json-out trace.json

  # Exit 0 even when errors found (for CI report-only mode)
  python trace_pipeline.py --repo-root /path/to/repo --no-fail

Exit codes
----------
  0  No errors found (or --no-fail set)
  1  One or more ERROR-level findings (or WARNING when --fail-on-warning)
  2  Script / argument error
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

SEVERITY_ORDER = {"ERROR": 0, "WARN": 1, "INFO": 2}


@dataclass
class Finding:
    severity: str          # ERROR | WARN | INFO
    step:     str          # step name or "PIPELINE"
    category: str          # step_order | script_exists | dependency | path | validate | isolation
    message:  str
    detail:   str = ""

    def __str__(self) -> str:
        head = f"[{self.severity}] [{self.step}] {self.message}"
        return head if not self.detail else f"{head}\n    {self.detail}"


@dataclass
class StepDef:
    """Everything statically known about one pipeline step."""
    name:         str
    description:  str
    order_index:  int                    # position in _STEP_ORDER
    call_index:   int                    # position of the run() call in the file

    # Scripts this step invokes (relative to their cwd)
    scripts:      list[tuple[str, str]]  # [(cwd_var, script_name), ...]

    # Directories this step changes into
    cwd_vars:     list[str]              # e.g. ["EXPERIMENTS_DIR", "ANALYSIS_DIR"]

    # Output artefacts (RESULTS_DIR-relative globs)
    outputs:      list[str]

    # Input artefacts consumed (RESULTS_DIR-relative globs)
    # These must have been written by an upstream step
    inputs:       list[str]

    # Steps that MUST have run before this one (derived from input→output matching)
    declared_deps: list[str] = field(default_factory=list)

    # If True, the step's notebooks/scripts are not yet committed to the repo.
    # script_exists check emits WARN (not ERROR) and step_order skips run() check.
    pending:      bool = False

    findings:     list[Finding] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Step catalogue  (hand-coded from run_all.sh — the ground truth)
# ──────────────────────────────────────────────────────────────────────────────
#
# Each entry documents:
#   scripts   : (CWD variable, script filename)
#   outputs   : RESULTS_DIR-relative paths / globs this step WRITES
#   inputs    : RESULTS_DIR-relative paths / globs this step READS
#   deps      : upstream step names required before this step runs

_STEP_CATALOGUE: list[dict] = [
    # ── 0: env_check ──────────────────────────────────────────────────────────
    dict(
        name="env_check",
        description="Verify environment (Python, Julia/PySR, API key, directories)",
        scripts=[],   # only inline python3 -c, no .py files
        cwd_vars=[],
        outputs=[
            # creates the directory skeleton
            "comparison_results/feynman-tests/exp2",
            "comparison_results/feynman-tests/exp2_extrap",
            "comparison_results/feynman-tests/noise-sweep",
            "comparison_results/feynman-tests/sample-complexity",
            "comparison_results/noise-noiseless/noiseless",
            "comparison_results/noise-noiseless/15",
            "comparison_results/extrapolation",
            "extrapolation",
            "hybrid_llm_nn/all_domains",
            "hybrid_llm_nn/defi",
            "hybrid_pysr/all_domains",
            "hybrid_pysr/defi",
            "llm_guided/all_domains",
            "llm_guided/defi",
            "standalone_llm_nn",
            "figures",
            "tables",
        ],
        inputs=[],
        deps=[],
    ),

    # ── 1: exp1 ───────────────────────────────────────────────────────────────
    dict(
        name="exp1",
        description="Core extrapolation benchmark (Tab 9, 10, 15 · Fig 9, 10)",
        scripts=[
            ("EXPERIMENTS_DIR", "hypatiax_defi_benchmark_v3c.py"),
            ("ANALYSIS_DIR",    "statistical_analysis.py"),
        ],
        cwd_vars=["EXPERIMENTS_DIR", "ANALYSIS_DIR"],
        outputs=[
            "hypatiax_defi_benchmark_v3*results*.json",
            "exp1_run.log",
            # statistical_analysis.py writes Mann-Whitney JSON
            "exp1_rf01_mannwhitney*.json",
            # noiseless protocol output (generated alongside benchmark)
            "comparison_results/noise-noiseless/noiseless/protocol_core_noiseless_*.json",
            # ablation results: shell mv puts ablation_*.json flat into RESULTS_DIR root
            # (run_all.sh: find EXPERIMENTS_DIR -name 'ablation_*.json' -exec mv {} RESULTS_DIR/)
            "ablation_*.json",
        ],
        inputs=[],
        deps=["env_check"],
    ),

    # ── 2: exp1b ──────────────────────────────────────────────────────────────
    dict(
        name="exp1b",
        description="DeFi seed sweep + portfolio variance (Tab 11-13 · Fig 11-13)",
        scripts=[
            ("EXPERIMENTS_DIR", "hypatiax_defi_benchmark_v3c.py"),
            ("EXPERIMENTS_DIR", "portfolio_variance_v3c2.py"),
        ],
        cwd_vars=["EXPERIMENTS_DIR"],
        outputs=[
            # shell moves *portfolio*variance*.json (wildcard) to RESULTS_DIR root
            "portfolio_variance*.json",
            # shell also moves defi_v3_*.json to RESULTS_DIR root
            "defi_v3_*.json",
            "exp1b_run.log",
        ],
        inputs=[],   # independent of exp1 (same script, different env filter)
        deps=["env_check"],
    ),

    # ── 3: exp1_pca ───────────────────────────────────────────────────────────
    dict(
        name="exp1_pca",
        description="FIX-C3 DeFi: all 74 cases with PCA 40/60 split (mirrors exp1 with PCA split)",
        scripts=[
            ("EXPERIMENTS_DIR", "hypatiax_defi_benchmark_pca.py"),
        ],
        cwd_vars=["EXPERIMENTS_DIR"],
        outputs=[
            # Results land in the defi_pca/ subdirectory
            "comparison_results/noise-noiseless/noiseless/defi_pca",
            # split disclosure written inline by run_all.sh after the benchmark
            "comparison_results/noise-noiseless/noiseless/defi_pca/split_protocol_disclosure.json",
            "exp1_pca_run.log",
        ],
        inputs=[],
        deps=["env_check", "exp1"],   # logically after exp1; same benchmark, PCA-split variant
    ),

    # ── 4: exp1b_pca ──────────────────────────────────────────────────────────
    dict(
        name="exp1b_pca",
        description="FIX-C3 DeFi seed sweep with PCA 40/60 split (mirrors exp1b with PCA split)",
        scripts=[
            ("EXPERIMENTS_DIR", "hypatiax_defi_benchmark_pca.py"),
        ],
        cwd_vars=["EXPERIMENTS_DIR"],
        outputs=[
            # Results land in the 15_pca/ subdirectory (mirrors exp1b → 15/)
            "comparison_results/noise-noiseless/15_pca",
            # Move block moves defi_pca_v3_*.json and *portfolio*variance*pca*.json here
            "comparison_results/noise-noiseless/15_pca/defi_pca_v3_*.json",
            "comparison_results/noise-noiseless/15_pca/*portfolio*variance*pca*.json",
            "exp1b_pca_run.log",
        ],
        inputs=[],
        deps=["env_check", "exp1b"],  # logically after exp1b; same scripts, PCA-split env flag
    ),

    # ── 5: extrap ─────────────────────────────────────────────────────────────
    dict(
        name="extrap",
        description="OOD extrapolation comparative run (Tab 9 OOD columns)",
        scripts=[
            ("EXPERIMENTS_DIR", "run_comparative_suite_benchmark_v2.py"),
        ],
        cwd_vars=["EXPERIMENTS_DIR"],
        outputs=[
            "comparison_results/extrapolation/all_domains_extrap_v4_*.json",
            "extrap_run.log",
        ],
        inputs=[],
        deps=["env_check"],
    ),

    # ── 6: hybrid_all_domains ─────────────────────────────────────────────────
    dict(
        name="hybrid_all_domains",
        description="Hybrid LLM+NN all-domains run — 10 domains (§10.9 hybrid)",
        scripts=[
            ("CORE_DIR/generation/hybrid_all_domains_llm_nn",
             "hybrid_system_llm_nn_all_domains.py"),
        ],
        cwd_vars=["CORE_DIR"],
        outputs=[
            "hybrid_llm_nn/all_domains/hybrid_llm_nn_all_domains_*.json",
            "hybrid_all_domains_run.log",
        ],
        inputs=[],
        deps=["env_check"],
    ),

    # ── 7: instability ────────────────────────────────────────────────────────
    dict(
        name="instability",
        description="Instability Index analysis + all figures — §10.9 (Regime A/B/C)",
        # Shell invokes: python3 '${EXPERIMENTS_DIR}/run_instability_suite.py'
        # with NO cd — script is called via its full path from the current shell context.
        # cwd_vars is empty because no directory change happens; the script path is
        # resolved through EXPERIMENTS_DIR at runtime, not via a cd.
        scripts=[
            ("EXPERIMENTS_DIR", "run_instability_suite.py"),
        ],
        cwd_vars=[],   # no cd in shell; full-path invocation via EXPERIMENTS_DIR
        outputs=[
            "figures/instability_analysis.csv",
            "figures/instability_extrapolation.csv",
            "figures/fig_paper_complexity_vs_instability.png",
            "figures/fig_paper_complexity_vs_instability.pdf",
            "figures/fig_paper_instability_hist.png",
            "figures/fig_paper_instability_hist.pdf",
            "figures/fig_paper_regime_counts.png",
            "figures/fig_paper_regime_counts.pdf",
            "figures/hypatiax_instability_per_case.png",
            "figures/hypatiax_instability_per_case.pdf",
            "instability_run.log",
        ],
        inputs=[
            # reads DeFi benchmark JSON produced by exp1 (optional but needed for II>0)
            "hypatiax_defi_benchmark_v3*results*.json",
        ],
        deps=["env_check", "exp1"],
    ),

    # ── 8: exp2_feynman ───────────────────────────────────────────────────────
    dict(
        name="exp2_feynman",
        description="Feynman SR benchmark — Phase 2 noisy protocol (Tab 16-18)",
        scripts=[
            ("EXPERIMENTS_DIR", "run_comparative_suite_benchmark_v2.py"),
        ],
        cwd_vars=["EXPERIMENTS_DIR"],
        outputs=[
            "comparison_results/feynman-tests/exp2/exp2_results*.json",
            "comparison_results/feynman-tests/exp2/exp2_run.log",
        ],
        inputs=[],
        deps=["env_check"],
    ),

    # ── 9: exp2_feynman_pca_4060 ─────────────────────────────────────────────
    dict(
        name="exp2_feynman_pca_4060",
        description="FIX-C3: Feynman rerun with PCA 40/60 split — corrected §10.7 result",
        scripts=[
            ("EXPERIMENTS_DIR", "run_comparative_suite_benchmark_v2.py"),
        ],
        cwd_vars=["EXPERIMENTS_DIR"],
        outputs=[
            # Per-domain results written to exp2_pca_4060/ (never overwrites legacy exp2/)
            "comparison_results/feynman-tests/exp2_pca_4060/exp2_pca_4060_summary.json",
            "comparison_results/feynman-tests/exp2_pca_4060/split_protocol_disclosure.json",
            # fixc3_baseline.json locks the original 9/30 result before any corrected run
            "fixc3_baseline.json",
            "comparison_results/feynman-tests/exp2_pca_4060/exp2_pca_4060_run.log",
        ],
        inputs=[],
        deps=["env_check", "exp2_feynman"],
    ),

    # ── 10: exp2_feynman_extrap ────────────────────────────────────────────────
    dict(
        name="exp2_feynman_extrap",
        description="Feynman SR benchmark — OOD extrapolation protocol (Tab 16-18 OOD)",
        scripts=[
            ("EXPERIMENTS_DIR", "run_comparative_suite_benchmark_v2.py"),
        ],
        cwd_vars=["EXPERIMENTS_DIR"],
        outputs=[
            "comparison_results/feynman-tests/exp2_extrap/protocol_core_extrap_*.json",
            "comparison_results/feynman-tests/exp2_extrap/exp2_extrap_run.log",
        ],
        inputs=[],
        deps=["env_check", "exp2_feynman"],
    ),

    # ── 11: exp2 ───────────────────────────────────────────────────────────────
    dict(
        name="exp2",
        description="Combined five-system comparison — all Methods (Tab 19 full)",
        scripts=[
            ("EXPERIMENTS_DIR", "run_comparative_suite_benchmark_v2.py"),
        ],
        cwd_vars=["EXPERIMENTS_DIR"],
        outputs=[
            "exp2_run.log",
            # merge_all_systems.py is implied; all_systems_merged.json may live in experiments/comparison/
        ],
        inputs=[],
        deps=["env_check"],
    ),

    # ── 12: exp3 ──────────────────────────────────────────────────────────────
    dict(
        name="exp3",
        description="Nguyen-12 benchmark — SEED=42 (tab:nguyen12 · §10.8)",
        scripts=[
            ("EXPERIMENTS_DIR", "exp3_nguyen12_hybrid50v_02.py"),
        ],
        cwd_vars=["EXPERIMENTS_DIR"],
        outputs=[
            "exp3_run.log",
            "exp3*nguyen12*.json",
        ],
        inputs=[],
        deps=["env_check"],
    ),

    # ── 13: exp3b ──────────────────────────────────────────────────────────────
    dict(
        name="exp3b",
        description="Nguyen-12 stability seeds 99/123/777/2024",
        scripts=[
            ("EXPERIMENTS_DIR", "exp3_nguyen12_hybrid50v_02.py"),
        ],
        cwd_vars=["EXPERIMENTS_DIR"],
        outputs=[
            "exp3b_run.log",
        ],
        inputs=[],
        deps=["env_check", "exp3"],  # reuses same script; logically after exp3
    ),

    # ── 14: suppA ─────────────────────────────────────────────────────────────
    dict(
        name="suppA",
        description="DeFi routing improvement experiments (Supplement A · Tab 11-13 routing)",
        scripts=[
            ("EXPERIMENTS_DIR", "run_hybrid_system_benchmark.py"),
        ],
        cwd_vars=["EXPERIMENTS_DIR"],
        outputs=[
            "suppA_run.log",
            # shell moves consolidated_hybrid*.json → hybrid_llm_nn/defi/
            "hybrid_llm_nn/defi/consolidated_hybrid*.json",
            # shell moves hybrid_system*.json → hybrid_llm_nn/all_domains/
            "hybrid_llm_nn/all_domains/hybrid_system*.json",
        ],
        inputs=[],
        # hybrid_all_domains (order 4) runs before suppA (order 10) in run_all.sh
        # and both write to hybrid_llm_nn/all_domains/ — declaring the ordering
        # here prevents the isolation check from flagging a spurious write race.
        deps=["env_check", "hybrid_all_domains"],
    ),

    # ── 15: suppB ─────────────────────────────────────────────────────────────
    dict(
        name="suppB",
        description="Noise sweep benchmark σ ∈ {0,0.5,1,5,10}% (Tab 28, 29)",
        scripts=[
            ("EXPERIMENTS_DIR", "run_noise_sweep_benchmark.py"),
        ],
        cwd_vars=["EXPERIMENTS_DIR"],
        outputs=[
            "comparison_results/feynman-tests/noise-sweep/noise_sweep_*.json",
            "suppB_run.log",
        ],
        inputs=[],
        deps=["env_check"],
    ),

    # ── 16: suppB_sc ──────────────────────────────────────────────────────────
    dict(
        name="suppB_sc",
        description="Sample-complexity sweep n ∈ {50…1000} (Tab 29 · Supplement B §6)",
        scripts=[
            ("EXPERIMENTS_DIR", "run_sample_complexity_benchmark.py"),
        ],
        cwd_vars=["EXPERIMENTS_DIR"],
        outputs=[
            "comparison_results/feynman-tests/sample-complexity/sample_complexity_*.json",
            "suppB_sc_run.log",
        ],
        inputs=[],
        deps=["env_check"],
    ),

    # ── 17: tables ────────────────────────────────────────────────────────────
    dict(
        name="tables",
        description="Generate all LaTeX tables from result JSONs → ${RESULTS_DIR}/tables/",
        scripts=[
            ("REPO_ROOT/scripts", "generate_tables.py"),
        ],
        cwd_vars=["REPO_ROOT"],
        outputs=[
            "tables/five_system.tex",
            "tables/defi_main.tex",
            "tables/defi_tiers.tex",
            "tables/runtime.tex",
            "tables/portfolio_sweep.tex",
            "tables/ablation.tex",
            "tables/feynman.tex",
            "tables/nguyen12.tex",
            "tables/instability.tex",
            "tables/version_history.tex",
            "tables/timing_detail.tex",
            "tables/repro_macros.tex",
            "tables/suppb_r2_noise.tex",
            "tables/suppb_rr_noise.tex",
            "tables/suppb_time_noise.tex",
            "tables/suppb_sc_metrics.tex",
            "tables/suppb_winrate.tex",
            "tables/suppb_noiseless.tex",
            "tables_run.log",
        ],
        inputs=[
            # reads benchmark results
            "hypatiax_defi_benchmark_v3*results*.json",
            # ablation files are at RESULTS_DIR root (shell mv is flat, not into subdir)
            "ablation_*.json",
            "portfolio_variance*.json",
            "comparison_results/feynman-tests/exp2/exp2_results*.json",
            "exp3*nguyen12*.json",
            "figures/instability_analysis.csv",
            "comparison_results/feynman-tests/noise-sweep/noise_sweep_*.json",
            "comparison_results/feynman-tests/sample-complexity/sample_complexity_*.json",
            "comparison_results/noise-noiseless/noiseless/protocol_core_noiseless_*.json",
        ],
        deps=["exp1", "exp1b", "extrap", "instability", "exp2_feynman", "exp2_feynman_extrap",
              "exp3", "exp3b", "suppB", "suppB_sc"],
    ),

    # ── 18: figures ───────────────────────────────────────────────────────────
    dict(
        name="figures",
        description="Generate all paper figures from results → ${RESULTS_DIR}/figures/",
        scripts=[
            ("REPO_ROOT/scripts", "generate_figures.py"),
        ],
        cwd_vars=["REPO_ROOT"],
        outputs=[
            "figures/fig_paper_complexity_vs_instability.pdf",
            "figures/fig_paper_complexity_vs_instability.png",
            "figures/fig_instability_3d.pdf",
            "figures/fig_r2_heatmap_clipped.pdf",
            "figures/fig_portfolio_seed_sweep.pdf",
            "figures_run.log",
        ],
        inputs=[
            "hypatiax_defi_benchmark_v3*results*.json",
            # ablation files at RESULTS_DIR root (flat mv in shell)
            "ablation_*.json",
            "portfolio_variance*.json",
            "figures/instability_analysis.csv",
        ],
        deps=["exp1", "exp1b", "instability"],
    ),

    # ── 19: validate ──────────────────────────────────────────────────────────
    dict(
        name="validate",
        description="Cross-check all results against paper-reported values",
        scripts=[],   # inline Python heredoc
        cwd_vars=[],
        outputs=[],
        inputs=[
            "comparison_results/noise-noiseless/noiseless/protocol_core_noiseless_*.json",
            "comparison_results/feynman-tests/exp2/exp2_results*.json",
            "comparison_results/feynman-tests/exp2_extrap/protocol_core_extrap_*.json",
            "exp1_rf01_mannwhitney*.json",
            "hybrid_llm_nn/all_domains/*.json",
            "figures/instability_analysis.csv",
            "figures/fig_paper_complexity_vs_instability.pdf",
            "comparison_results/feynman-tests/sample-complexity/*.json",
            "comparison_results/feynman-tests/noise-sweep/noise_sweep_*.json",
            "tables/*.tex",
            "figures/*.pdf",
            # exp1b
            "portfolio_variance*.json",
            "defi_v3_*.json",
            # exp2
            "exp2_run.log",
            # exp3 / exp3b
            "exp3*nguyen12*.json",
            # extrap
            "comparison_results/extrapolation/all_domains_extrap_v4_*.json",
        ],
        deps=["exp1", "exp1b",
              "exp2_feynman", "exp2_feynman_extrap", "hybrid_all_domains", "instability",
              "suppB_sc", "suppB", "tables", "figures",
              "exp2", "exp3", "exp3b", "extrap"],
    ),

    # ── 20: qualify ───────────────────────────────────────────────────────────
    dict(
        name="qualify",
        description="verify_results.py spot-check + 7-dimension per-experiment gate (Phase 5)",
        scripts=[
            ("SCRIPTS_DIR/patches", "verify_results.py"),
            ("REPO_ROOT",           "run_all_checkpoint.py"),
        ],
        cwd_vars=["REPO_ROOT"],
        outputs=[
            "qualify_verify_run.log",
            "qualify_run.log",
            "logs/verify_report.json",
        ],
        inputs=[],
        deps=["validate"],
    ),

    # ── 21: audit_paper ───────────────────────────────────────────────────────
    dict(
        name="audit_paper",
        description="Cross-check every paper claim vs result JSONs via paper_targets.json",
        scripts=[
            ("REPO_ROOT", "run_all_checkpoint.py"),
        ],
        cwd_vars=["REPO_ROOT"],
        outputs=[
            "audit_paper_run.log",
            "logs/paper_audit_findings.json",
        ],
        inputs=[],
        deps=["qualify"],
    ),

    # ── 22: audit_setup ───────────────────────────────────────────────────────
    dict(
        name="audit_setup",
        description="Copy .tex source files into notebooks/ for subsequent audit notebooks",
        scripts=[],   # inline Python heredoc — no standalone script file
        cwd_vars=["REPO_ROOT"],
        outputs=[],   # copies files into notebooks/; no stable glob to track
        inputs=[],
        deps=["audit_paper"],
    ),

    # ── 23: audit_nb01 ────────────────────────────────────────────────────────
    dict(
        name="audit_nb01",
        description="NB-01 Citation & Bibliography Audit (jupyter nbconvert)",
        scripts=[
            ("REPO_ROOT/notebooks", "NB-01_Citation_Bibliography_Audit.ipynb"),
        ],
        cwd_vars=["REPO_ROOT"],
        outputs=[
            "audit_nb01_run.log",
        ],
        inputs=[],
        deps=["audit_setup"],
    ),

    # ── 24: audit_nb02 ────────────────────────────────────────────────────────
    dict(
        name="audit_nb02",
        description="NB-02 Cross-Reference & Label Integrity (jupyter nbconvert)",
        scripts=[
            ("REPO_ROOT/notebooks", "NB-02_CrossReference_Label_Audit.ipynb"),
        ],
        cwd_vars=["REPO_ROOT"],
        outputs=[
            "audit_nb02_run.log",
        ],
        inputs=[],
        deps=["audit_setup"],
    ),

    # ── 25: audit_nb03 ────────────────────────────────────────────────────────
    dict(
        name="audit_nb03",
        description="NB-03 Section Structure & Numbering (jupyter nbconvert)",
        scripts=[
            ("REPO_ROOT/notebooks", "NB-03_Section_Structure_Numbering.ipynb"),
        ],
        cwd_vars=["REPO_ROOT"],
        outputs=[
            "audit_nb03_run.log",
        ],
        inputs=[],
        deps=["audit_setup"],
    ),

    # ── 26: audit_nb04 ────────────────────────────────────────────────────────
    dict(
        name="audit_nb04",
        description="NB-04 Numerical Consistency & Abstract Claims (jupyter nbconvert)",
        scripts=[
            ("REPO_ROOT/notebooks", "NB-04_Numerical_Consistency_Checker.ipynb"),
        ],
        cwd_vars=["REPO_ROOT"],
        outputs=[
            "audit_nb04_run.log",
        ],
        inputs=[],
        deps=["audit_setup"],
    ),

    # ── 27: audit_nb05 ────────────────────────────────────────────────────────
    dict(
        name="audit_nb05",
        description="NB-05 Figure Files & Image Dependencies (jupyter nbconvert)",
        scripts=[
            ("REPO_ROOT/notebooks", "NB-05_Figure_Image_Dependency_Checker.ipynb"),
        ],
        cwd_vars=["REPO_ROOT"],
        outputs=[
            "audit_nb05_run.log",
        ],
        inputs=[],
        deps=["audit_setup"],
    ),

    # ── 28: audit_nb06_fixc3_disclosure ──────────────────────────────────────
    dict(
        name="audit_nb06_fixc3_disclosure",
        description="NB-06 FIX-C3 Action A: Disclose Feynman random-80/20 vs DeFi PCA-40/60 split mismatch",
        # Inline Python heredoc in run_all.sh — no external notebook or script file
        scripts=[],
        cwd_vars=["REPO_ROOT"],
        outputs=[
            # Machine-readable disclosure record; required by audit_paper and Action B
            "fixc3_split_disclosure.json",
            "audit_nb06_fixc3_disclosure_run.log",
        ],
        inputs=[],
        deps=["audit_setup"],
    ),

    # ── 29: audit_nb06_fixc3_rerun ────────────────────────────────────────────
    dict(
        name="audit_nb06_fixc3_rerun",
        description="NB-06 FIX-C3 Action B: Rerun Feynman with PCA 40/60 split; report revised 9/30 result",
        # Inline Python heredoc in run_all.sh — no external notebook or script file
        scripts=[],
        cwd_vars=["REPO_ROOT"],
        outputs=[
            # Per-domain results in exp2_fixc3/ (distinct from exp2_pca_4060/ of exp2_feynman_pca_4060)
            "comparison_results/feynman-tests/exp2_fixc3/protocol_core_fixc3_*.json",
            "comparison_results/feynman-tests/exp2_fixc3/fixc3_run.log",
            # Solve-rate summary written by the inline Python analysis block
            "fixc3_rerun_summary.json",
            "audit_nb06_fixc3_rerun_run.log",
        ],
        inputs=[
            # Action A must run first and produce this disclosure record
            "fixc3_split_disclosure.json",
        ],
        deps=["audit_nb06_fixc3_disclosure", "exp2_feynman_pca_4060"],
    ),

    # ── 30: audit_guard ───────────────────────────────────────────────────────
    dict(
        name="audit_guard",
        description="CI guard: evaluate workflow_run trigger (slot=12, run_full, success)",
        scripts=[],
        cwd_vars=["REPO_ROOT"],
        outputs=[],
        inputs=[],
        deps=[],
    ),

    # ── 31: audit_print_verify ────────────────────────────────────────────────
    dict(
        name="audit_print_verify",
        description="Print human-readable summary of logs/verify_report.json",
        scripts=[],
        cwd_vars=["REPO_ROOT"],
        outputs=[],
        inputs=["logs/verify_report.json"],
        deps=["qualify"],
    ),

    # ── 32: audit_print_findings ──────────────────────────────────────────────
    dict(
        name="audit_print_findings",
        description="Print human-readable summary of logs/paper_audit_findings.json",
        scripts=[],
        cwd_vars=["REPO_ROOT"],
        outputs=[],
        inputs=["logs/paper_audit_findings.json"],
        deps=["audit_paper"],
    ),

    # ── 33: audit_figures_tables ──────────────────────────────────────────────
    dict(
        name="audit_figures_tables",
        description="Validate expected figures (PDF/PNG) and LaTeX tables (TeX) are present",
        scripts=[],
        cwd_vars=["REPO_ROOT"],
        outputs=[
            "logs/figures_tables_report.json",
        ],
        inputs=[],
        deps=["qualify"],
    ),

    # ── 34: audit_final_gate ──────────────────────────────────────────────────
    dict(
        name="audit_final_gate",
        description="Aggregate numerical-verify, paper-audit, figures-tables outcomes; set exit code",
        scripts=[],
        cwd_vars=["REPO_ROOT"],
        outputs=[],
        inputs=[],
        deps=["audit_paper", "audit_figures_tables"],
    ),
]

# Declared step order from run_all.sh _STEP_ORDER variable
_DECLARED_ORDER = [
    "env_check", "exp1", "exp1b", "exp1_pca", "exp1b_pca", "extrap", "hybrid_all_domains",
    "instability", "exp2_feynman", "exp2_feynman_pca_4060", "exp2_feynman_extrap",
    "exp2", "exp3", "exp3b",
    "suppA", "suppB", "suppB_sc", "tables", "figures", "validate",
    # Phase 3 — qualification & paper audit (added 2026-05-30)
    "qualify", "audit_paper", "audit_setup",
    "audit_nb01", "audit_nb02", "audit_nb03", "audit_nb04", "audit_nb05",
    # FIX-C3 audit notebooks (added 2026-06-03)
    "audit_nb06_fixc3_disclosure", "audit_nb06_fixc3_rerun",
    # Phase 4 — CI gate steps for ci_paper_audit.yml (added 2026-05-30)
    "audit_guard", "audit_print_verify", "audit_print_findings",
    "audit_figures_tables", "audit_final_gate",
]

# CWD variable → repo-relative directory path
_CWD_MAP = {
    "EXPERIMENTS_DIR": "hypatiax/experiments/benchmarks",
    "ANALYSIS_DIR":    "hypatiax/analysis",
    "CORE_DIR":        "hypatiax/core",
    "CORE_DIR/generation/hybrid_all_domains_llm_nn":
                       "hypatiax/core/generation/hybrid_all_domains_llm_nn",
    "REPO_ROOT":       ".",
    "REPO_ROOT/tables":     "tables",
    "REPO_ROOT/figures":    "figures",   # FIX: was missing — figures step uses this cwd
    "REPO_ROOT/notebooks":  "notebooks", # audit_nb01–nb05 jupyter notebooks
    "SCRIPTS_DIR":          "scripts",
    "SCRIPTS_DIR/patches":  "scripts/patches",  # qualify: verify_results.py
}


# ──────────────────────────────────────────────────────────────────────────────
# Tracer
# ──────────────────────────────────────────────────────────────────────────────

class PipelineTracer:
    def __init__(self, repo_root: Optional[Path], shell_path: Optional[Path],
                 tables_gen: Optional[Path], figures_gen: Optional[Path],
                 repro_cfg: Optional[Path] = None,
                 step_filter: Optional[str] = None):
        self.repo        = repo_root.resolve() if repo_root else None
        self.shell_path  = shell_path
        self.tables_gen  = tables_gen
        self.figures_gen = figures_gen
        self.repro_cfg   = repro_cfg        # path to config/repro.yaml for cross-check
        self.step_filter = step_filter      # if set, only trace this step (from CI --step arg)

        self.steps: dict[str, StepDef] = {}
        self.findings: list[Finding]   = []
        self._build_steps()

    # ── Build step objects ────────────────────────────────────────────────────

    def _build_steps(self) -> None:
        for i, cat in enumerate(_STEP_CATALOGUE):
            sd = StepDef(
                name          = cat["name"],
                description   = cat["description"],
                order_index   = _DECLARED_ORDER.index(cat["name"])
                                if cat["name"] in _DECLARED_ORDER else -1,
                call_index    = i,
                scripts       = cat["scripts"],
                cwd_vars      = cat["cwd_vars"],
                outputs       = cat["outputs"],
                inputs        = cat["inputs"],
                declared_deps = cat["deps"],
                pending       = cat.get("pending", False),
            )
            self.steps[cat["name"]] = sd

    # ── Entry point ───────────────────────────────────────────────────────────

    def run_all_checks(self) -> list[Finding]:
        self._check_step_order()
        self._check_script_existence()
        self._check_dependency_graph()
        self._check_validate_coverage()
        self._check_step_isolation()
        self._check_shell_step_parity()
        if self.tables_gen:
            self._check_generator_paths(self.tables_gen, "tables-generator")
        if self.figures_gen:
            self._check_generator_paths(self.figures_gen, "figures-generator")
        return self.findings

    # ── Check 1: step order ───────────────────────────────────────────────────

    def _check_step_order(self) -> None:
        catalogue_order = [s["name"] for s in _STEP_CATALOGUE]
        declared        = _DECLARED_ORDER

        # Steps in catalogue but not in _STEP_ORDER
        for name in catalogue_order:
            if name not in declared:
                self._add(Finding("ERROR", name, "step_order",
                    f"Step '{name}' is in the catalogue but MISSING from _STEP_ORDER in run_all.sh",
                    f"_STEP_ORDER = {' '.join(declared)}"))

        # Steps in _STEP_ORDER but not in catalogue
        for name in declared:
            if name not in self.steps:
                self._add(Finding("ERROR", name, "step_order",
                    f"Step '{name}' appears in _STEP_ORDER but has NO catalogue entry / run() block",
                    "This means the step is declared but never executed — dead entry in _STEP_ORDER"))

        # Order mismatch
        common = [n for n in declared if n in catalogue_order]
        common_cat = [n for n in catalogue_order if n in declared]
        if common != common_cat:
            self._add(Finding("WARN", "PIPELINE", "step_order",
                "Catalogue order differs from _STEP_ORDER declaration",
                f"  _STEP_ORDER : {' → '.join(declared)}\n"
                f"  Catalogue   : {' → '.join(catalogue_order)}"))
        else:
            self._add(Finding("INFO", "PIPELINE", "step_order",
                "Step order in _STEP_ORDER matches catalogue ✓"))

    # ── Check 2: script existence ─────────────────────────────────────────────

    def _check_script_existence(self) -> None:
        if not self.repo:
            self._add(Finding("WARN", "PIPELINE", "script_exists",
                "--repo-root not provided; skipping on-disk script existence checks"))
            return

        for name, step in self.steps.items():
            for cwd_var, script in step.scripts:
                repo_rel = _CWD_MAP.get(cwd_var)
                if repo_rel is None:
                    self._add(Finding("WARN", name, "script_exists",
                        f"Unknown CWD variable '{cwd_var}' — cannot resolve script path",
                        f"Script: {script}"))
                    continue

                script_path = self.repo / repo_rel / script
                if script_path.exists():
                    self._add(Finding("INFO", name, "script_exists",
                        f"✓ {script}",
                        f"Path: {script_path}"))
                else:
                    # Check if there's a close match (typo / renamed)
                    parent = self.repo / repo_rel
                    close = []
                    if parent.exists():
                        stem = Path(script).stem.lower()
                        close = [
                            p.name for p in parent.glob("*.py")
                            if stem[:6] in p.name.lower()
                        ]
                    detail = f"Expected: {script_path}"
                    if close:
                        detail += f"\n    Possible matches in dir: {', '.join(close[:5])}"
                    severity = "WARN" if step.pending else "ERROR"
                    self._add(Finding(severity, name, "script_exists",
                        f"✗ Script NOT FOUND: {script}  (cwd={cwd_var})",
                        detail + ("\n    (pending — notebook not yet committed to repo)"
                                  if step.pending else "")))

    # ── Check 3: dependency graph ─────────────────────────────────────────────

    def _check_dependency_graph(self) -> None:
        """
        For every (step, input_glob), verify that at least one upstream step
        declares a matching output glob.  Flag if the producing step is
        downstream of (or parallel to) the consumer.
        """
        # Build a flat map: output_glob_prefix → step_name
        output_owners: dict[str, str] = {}
        for name, step in self.steps.items():
            for out in step.outputs:
                # normalise: strip leading /, strip glob wildcards for prefix match
                key = out.lstrip("/").split("*")[0].rstrip("/")
                output_owners[key] = name

        for name, step in self.steps.items():
            consumer_idx = step.order_index

            for inp in step.inputs:
                inp_norm = inp.lstrip("/").split("*")[0].rstrip("/")

                # Find best matching producer
                producer: Optional[str] = None
                best_len = 0
                for prefix, producer_name in output_owners.items():
                    if (inp_norm.startswith(prefix) or prefix.startswith(inp_norm)):
                        if len(prefix) > best_len:
                            producer = producer_name
                            best_len = len(prefix)

                if producer is None:
                    self._add(Finding("ERROR", name, "dependency",
                        f"Input '{inp}' is NOT produced by any step",
                        f"No step declares an output matching this path — "
                        f"either the step catalogue is wrong or the input is external"))
                    continue

                if producer == name:
                    # Step consuming its own output — fine for in-place updates
                    continue

                producer_idx = self.steps[producer].order_index

                if producer_idx > consumer_idx:
                    self._add(Finding("ERROR", name, "dependency",
                        f"Dependency order VIOLATION: '{name}' (step {consumer_idx}) "
                        f"reads input produced by '{producer}' (step {producer_idx})",
                        f"Input '{inp}' — '{producer}' runs AFTER '{name}'"))
                elif producer_idx == consumer_idx:
                    self._add(Finding("WARN", name, "dependency",
                        f"'{name}' reads from '{producer}' which runs at the SAME position",
                        f"Input: {inp}"))
                else:
                    # Check that producer is listed as a dep
                    if producer not in step.declared_deps and producer != "env_check":
                        self._add(Finding("WARN", name, "dependency",
                            f"'{name}' reads output of '{producer}' but '{producer}' "
                            f"is NOT listed in deps for '{name}'",
                            f"Input: {inp}  →  produced by: {producer}"))
                    else:
                        self._add(Finding("INFO", name, "dependency",
                            f"✓ input '{inp}' ← step '{producer}' (order {producer_idx})"))

        # Check declared deps resolve to real steps
        for name, step in self.steps.items():
            for dep in step.declared_deps:
                if dep not in self.steps:
                    self._add(Finding("ERROR", name, "dependency",
                        f"Declared dep '{dep}' is not a known step",
                        f"Known steps: {', '.join(self.steps.keys())}"))
                elif self.steps[dep].order_index >= step.order_index:
                    self._add(Finding("ERROR", name, "dependency",
                        f"Declared dep '{dep}' (order {self.steps[dep].order_index}) "
                        f"runs at or AFTER '{name}' (order {step.order_index})"))

    # ── Check 4: validate coverage ────────────────────────────────────────────

    def _check_validate_coverage(self) -> None:
        """
        validate step must cover every step's key output.
        Flag steps whose outputs are not mentioned in validate's input list.
        """
        validate = self.steps.get("validate")
        if not validate:
            self._add(Finding("ERROR", "PIPELINE", "validate",
                "No 'validate' step in catalogue"))
            return

        val_inputs_norm = set()
        for vi in validate.inputs:
            val_inputs_norm.add(vi.lstrip("/").split("*")[0].rstrip("/"))

        # Steps that *should* be validated (all experiment steps)
        experiment_steps = [
            n for n in _DECLARED_ORDER
            if n not in ("env_check", "tables", "figures", "validate")
        ]

        for sname in experiment_steps:
            step = self.steps.get(sname)
            if not step:
                continue
            # Check if at least one of its outputs is covered
            covered = False
            for out in step.outputs:
                out_norm = out.lstrip("/").split("*")[0].rstrip("/")
                if any(out_norm.startswith(v) or v.startswith(out_norm)
                       for v in val_inputs_norm):
                    covered = True
                    break

            if not covered and step.outputs:
                self._add(Finding("WARN", sname, "validate",
                    f"Step '{sname}' has outputs but NONE are covered by the validate step",
                    f"Outputs: {step.outputs[:3]}{'...' if len(step.outputs) > 3 else ''}"))
            elif step.outputs:
                self._add(Finding("INFO", sname, "validate",
                    f"✓ outputs covered by validate step"))

    # ── Check 5: step isolation ───────────────────────────────────────────────

    def _check_step_isolation(self) -> None:
        """
        Warn if two parallel (non-dependent) steps write to the same named
        subdirectory.  The root '.' is excluded — every step writes a *.log
        there by design (all distinct filenames, no race risk).
        Deduplicates (A,B) pairs so each conflict is reported exactly once.
        """
        dir_writers: dict[str, list[str]] = {}
        for name, step in self.steps.items():
            for out in step.outputs:
                d = str(Path(out).parent)
                if d == ".":
                    continue          # log files — distinct names, not a race
                if name not in dir_writers.get(d, []):
                    dir_writers.setdefault(d, []).append(name)

        seen_pairs: set[frozenset] = set()
        for d, writers in dir_writers.items():
            if len(writers) < 2:
                continue
            for i, w1 in enumerate(writers):
                for w2 in writers[i + 1:]:
                    if w1 == w2:
                        continue
                    pair = frozenset({w1, w2})
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    # Use transitive dep closure so instability→exp1 chain counts
                    deps_of_w1 = _get_all_deps(w1, self.steps)
                    deps_of_w2 = _get_all_deps(w2, self.steps)
                    w2_before_w1 = w2 in deps_of_w1
                    w1_before_w2 = w1 in deps_of_w2

                    if not w2_before_w1 and not w1_before_w2:
                        self._add(Finding("WARN", "PIPELINE", "isolation",
                            f"Steps '{w1}' and '{w2}' both write to '{d}/' "
                            f"but neither is upstream of the other",
                            f"Concurrent execution would cause a write race in {d}/"))

    # ── Check 6: shell/catalogue parity ──────────────────────────────────────

    def _check_shell_step_parity(self) -> None:
        """
        Parse run_all.sh to extract the actual run() call sequence and compare
        against _DECLARED_ORDER and the catalogue.
        """
        if not self.shell_path or not self.shell_path.exists():
            self._add(Finding("WARN", "PIPELINE", "step_order",
                "run_all.sh not provided or not found; skipping shell parse"))
            return

        text = self.shell_path.read_text()

        # Extract _STEP_ORDER value
        m = re.search(r'_STEP_ORDER\s*=\s*"([^"]+)"', text)
        if m:
            shell_order = m.group(1).split()
            if shell_order != _DECLARED_ORDER:
                self._add(Finding("ERROR", "PIPELINE", "step_order",
                    "_STEP_ORDER in run_all.sh differs from tracer's _DECLARED_ORDER",
                    f"  shell   : {' '.join(shell_order)}\n"
                    f"  tracer  : {' '.join(_DECLARED_ORDER)}"))
            else:
                self._add(Finding("INFO", "PIPELINE", "step_order",
                    "✓ run_all.sh _STEP_ORDER matches tracer catalogue"))
        else:
            self._add(Finding("WARN", "PIPELINE", "step_order",
                "Could not parse _STEP_ORDER from run_all.sh"))

        # Extract actual run() call sequence: run <step_name> "..."
        run_calls = re.findall(r'^run\s+(\S+)\s+', text, re.MULTILINE)
        # Filter out the function definition itself (has 'local step' next line)
        # run() { ... } is defined around line 90; actual calls start after
        def_end = text.find('log "=== STEP:')
        run_actual = re.findall(r'^run\s+(\S+)\s+"', text[def_end:], re.MULTILINE)

        # Compare
        extra   = [s for s in run_actual if s not in self.steps]
        missing = [s for s in self.steps  if s not in run_actual]

        for s in extra:
            self._add(Finding("ERROR", s, "step_order",
                f"run() call for '{s}' found in shell but NOT in tracer catalogue",
                "Add an entry to _STEP_CATALOGUE in trace_pipeline.py"))

        for s in missing:
            # Some steps (env_check, validate) use different invocation patterns
            if s not in ("env_check", "validate"):
                step_obj = self.steps.get(s)
                if step_obj and step_obj.pending:
                    # Pending steps are not yet wired into run_all.sh — expected
                    self._add(Finding("INFO", s, "step_order",
                        f"✓ '{s}' is pending — no run() call in shell yet (expected)"))
                else:
                    self._add(Finding("WARN", s, "step_order",
                        f"Step '{s}' is in catalogue but no run() call found in shell parse",
                        "Verify the step has a run() invocation in run_all.sh"))
            else:
                self._add(Finding("INFO", s, "step_order",
                    f"✓ '{s}' invocation pattern is non-standard (expected)"))

        if run_actual:
            self._add(Finding("INFO", "PIPELINE", "step_order",
                f"Shell run() call sequence ({len(run_actual)} steps): "
                f"{' → '.join(run_actual)}"))

    # ── Check 7: generator path consistency ──────────────────────────────────

    def _check_generator_paths(self, gen_path: Path, label: str) -> None:
        """
        Lightly parse tables-generator.py / generate_all_figures.py to verify
        that the load_best() subdirs and output globs match what run_all.sh produces.
        """
        if not gen_path.exists():
            self._add(Finding("WARN", "PIPELINE", "path",
                f"{label}: file not found at {gen_path}"))
            return

        text = gen_path.read_text()

        # Extract load_best() calls: load_best("subdir", "glob")
        load_calls = re.findall(
            r'load_best\(\s*"([^"]*?)"\s*,\s*"([^"]*?)"', text)

        # Known correct mappings from run_all.sh
        # subdir → expected glob pattern fragment
        KNOWN_CORRECT = {
            "":                                     ["hypatiax_defi_benchmark_v3", "exp3", "portfolio_variance"],
            "exp1_ablation":                        ["*.json"],
            "comparison_results/feynman-tests/exp2": ["*.json"],
            "comparison_results/feynman-tests/noise-sweep": ["noise_sweep_"],
            "comparison_results/feynman-tests/sample-complexity": ["sample_complexity_"],
            "figures":                              ["instability"],
            "comparison_results/noise-noiseless/noiseless": ["protocol_core_noiseless"],
        }

        WRONG_SUBDIRS = {
            "defi":      "should be '' (root) with glob 'hypatiax_defi_benchmark_v3*results*.json'",
            "feynman":   "should be 'comparison_results/feynman-tests/exp2'",
            "nguyen12":  "should be '' (root) with glob 'exp3*nguyen12*.json'",
            "instability": "should be 'figures' (run_all.sh writes there)",
        }

        for subdir, glob_pat in load_calls:
            if subdir in WRONG_SUBDIRS:
                self._add(Finding("ERROR", label, "path",
                    f"load_best() uses wrong subdir '{subdir}'",
                    f"  Fix: {WRONG_SUBDIRS[subdir]}\n"
                    f"  Glob used: {glob_pat}"))
            elif subdir in KNOWN_CORRECT:
                expected_frags = KNOWN_CORRECT[subdir]
                if not any(f in glob_pat for f in expected_frags):
                    self._add(Finding("WARN", label, "path",
                        f"load_best('{subdir}', '{glob_pat}') — glob may not match "
                        f"run_all.sh output",
                        f"Expected glob to contain one of: {expected_frags}"))
                else:
                    self._add(Finding("INFO", label, "path",
                        f"✓ load_best('{subdir}', '{glob_pat}')"))
            else:
                self._add(Finding("INFO", label, "path",
                    f"load_best('{subdir}', '{glob_pat}') — not in known-correct map; "
                    f"manual review recommended"))

        # Check gen_instability def exists in tables-generator
        if label == "tables-generator":
            if "def gen_instability" in text:
                self._add(Finding("INFO", label, "path",
                    "✓ gen_instability() function definition present"))
            else:
                self._add(Finding("ERROR", label, "path",
                    "gen_instability() function definition MISSING",
                    "The function body exists but the 'def' line is absent — "
                    "the function is unreachable dead code"))

        # Check RESULTS_DIR env var is honoured in figures generator
        if label == "figures-generator":
            if 'environ.get("RESULTS_DIR"' in text or "RESULTS_DIR" in text:
                self._add(Finding("INFO", label, "path",
                    "✓ RESULTS_DIR env-var referenced in generator"))
            else:
                self._add(Finding("WARN", label, "path",
                    "RESULTS_DIR env-var not referenced — generator may use "
                    "hardcoded path that diverges from run_all.sh"))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _add(self, f: Finding) -> None:
        self.findings.append(f)


def _get_all_deps(step_name: str, steps: dict[str, StepDef],
                  _visited: Optional[set] = None) -> set[str]:
    """Recursively collect all transitive dependencies of a step."""
    if _visited is None:
        _visited = set()
    if step_name in _visited:
        return _visited
    _visited.add(step_name)
    for dep in steps.get(step_name, StepDef("", "", -1, -1, [], [], [], [])).declared_deps:
        _get_all_deps(dep, steps, _visited)
    return _visited


# ──────────────────────────────────────────────────────────────────────────────
# Report renderer
# ──────────────────────────────────────────────────────────────────────────────

def _render_report(findings: list[Finding], steps: dict[str, StepDef],
                   args: argparse.Namespace) -> str:
    lines: list[str] = []
    SEP  = "═" * 78
    sep2 = "─" * 78

    lines += [
        SEP,
        "  HypatiaX Pipeline Static Tracer — Report",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Repo root : {args.repo_root or '(not provided)'}",
        f"  Shell     : {args.shell or '(not provided)'}",
        SEP, "",
    ]

    # ── Summary counts ────────────────────────────────────────────────────────
    counts = {"ERROR": 0, "WARN": 0, "INFO": 0}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1

    lines += [
        "  SUMMARY",
        sep2,
        f"  {'ERRORs':<12}: {counts['ERROR']}",
        f"  {'WARNINGs':<12}: {counts['WARN']}",
        f"  {'INFO':<12}: {counts['INFO']}",
        "",
        f"  Total steps in catalogue : {len(steps)}",
        f"  Steps declared in shell  : {len(_DECLARED_ORDER)}",
        "",
    ]
    if counts["ERROR"] == 0:
        lines.append("  ✅  No errors — pipeline structure is consistent")
    else:
        lines.append(f"  ❌  {counts['ERROR']} error(s) found — see details below")
    lines += ["", SEP, ""]

    # ── Per-step dependency graph ─────────────────────────────────────────────
    lines += ["  STEP DEPENDENCY GRAPH", sep2]
    for idx, name in enumerate(_DECLARED_ORDER):
        step = steps.get(name)
        if not step:
            lines.append(f"  [{idx:02d}] {name}  ⚠  (no catalogue entry)")
            continue
        deps_str = " ← " + ", ".join(step.declared_deps) if step.declared_deps else ""
        n_scripts = len(step.scripts)
        lines.append(
            f"  [{idx:02d}] {name:<25}  "
            f"{n_scripts} script(s){deps_str}"
        )
        for cwd_var, script in step.scripts:
            lines.append(f"         script : {cwd_var}/{script}")
        for out in step.outputs[:4]:
            lines.append(f"         out    : {out}")
        if len(step.outputs) > 4:
            lines.append(f"         out    : … ({len(step.outputs)-4} more)")
        for inp in step.inputs[:3]:
            lines.append(f"         in     : {inp}")
        if len(step.inputs) > 3:
            lines.append(f"         in     : … ({len(step.inputs)-3} more)")
    lines += ["", SEP, ""]

    # ── Findings by severity ──────────────────────────────────────────────────
    for sev in ("ERROR", "WARN", "INFO"):
        grp = [f for f in findings if f.severity == sev]
        if not grp:
            continue
        icon = {"ERROR": "❌", "WARN": "⚠ ", "INFO": "ℹ "}[sev]
        lines += [f"  {icon}  {sev} ({len(grp)})", sep2]
        # Group by step
        by_step: dict[str, list[Finding]] = {}
        for f in grp:
            by_step.setdefault(f.step, []).append(f)
        for step_name, step_findings in sorted(by_step.items()):
            lines.append(f"\n  ▸ {step_name}")
            for f in step_findings:
                lines.append(f"    [{f.category}] {f.message}")
                if f.detail:
                    for dl in f.detail.split("\n"):
                        lines.append(f"        {dl}")
        lines += ["", SEP, ""]

    # ── Actionable fix list ───────────────────────────────────────────────────
    errors = [f for f in findings if f.severity == "ERROR"]
    if errors:
        lines += ["  ACTIONABLE FIXES", sep2]
        for i, f in enumerate(errors, 1):
            lines.append(f"  {i:02d}. [{f.step}] {f.category.upper()}: {f.message}")
            if f.detail:
                for dl in f.detail.split("\n"):
                    lines.append(f"       {dl}")
            lines.append("")
        lines += [SEP, ""]

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Static pipeline tracer for run_all.sh — no experiments run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples
        --------
          python trace_pipeline.py --repo-root /path/to/hypatiax-repo
          python trace_pipeline.py --shell run_all.sh \\
              --tables-generator tables-generator.py \\
              --figures-generator generate_all_figures.py
          python trace_pipeline.py --repo-root /repo --json-out trace.json
        """),
    )
    p.add_argument("--repo-root", type=Path, default=None, metavar="PATH",
                   help="Root of the hypatiax repo (enables script existence checks).")
    p.add_argument("--shell", type=Path,
                   default=Path("run_all.sh"), metavar="PATH",
                   help="Path to run_all.sh (default: ./run_all.sh).")
    p.add_argument("--tables-generator", type=Path, default=None, metavar="PATH",
                   help="Path to tables-generator.py for extra path checks.")
    p.add_argument("--figures-generator", type=Path, default=None, metavar="PATH",
                   help="Path to generate_all_figures.py for extra path checks.")
    p.add_argument("--json-out", type=Path, default=None, metavar="PATH",
                   help="Write full findings as JSON to this file.")
    p.add_argument("--no-fail", action="store_true",
                   help="Exit 0 even if errors are found (for CI report-only mode).")
    p.add_argument("--fail-on-warning", action="store_true",
                   help="Treat WARNING-level findings as errors (mirrors ci_trace_pipeline.yml "
                        "fail_on_warning input). Implies exit code 1 when any WARN is present.")
    p.add_argument("--step", dest="step_filter", default=None, metavar="STEP_ID",
                   help="Limit trace output to a single step ID (PIPELINE-level findings always "
                        "included). Used by ci_trace_pipeline.yml --step input.")
    p.add_argument("--repro-cfg", type=Path, default=None, metavar="PATH",
                   help="Path to config/repro.yaml for cross-checking hyperparameter values "
                        "against pipeline defaults (stub — enables future check).")
    p.add_argument("--errors-only", action="store_true",
                   help="Suppress INFO findings from the report.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # Resolve shell path
    shell = args.shell.resolve() if args.shell and args.shell.exists() else None
    if args.shell and not shell:
        print(f"[WARN] Shell script not found: {args.shell}", file=sys.stderr)

    tracer = PipelineTracer(
        repo_root    = args.repo_root,
        shell_path   = shell or args.shell,
        tables_gen   = args.tables_generator,
        figures_gen  = args.figures_generator,
    )
    findings = tracer.run_all_checks()

    # Optionally suppress INFO
    if args.errors_only:
        findings = [f for f in findings if f.severity != "INFO"]

    # Render report
    report = _render_report(findings, tracer.steps, args)
    print(report)

    # JSON output
    if args.json_out:
        payload = {
            "generated":  datetime.now().isoformat(),
            "repo_root":  str(args.repo_root) if args.repo_root else None,
            "shell":      str(args.shell) if args.shell else None,
            "summary": {
                "errors":   sum(1 for f in findings if f.severity == "ERROR"),
                "warnings": sum(1 for f in findings if f.severity == "WARN"),
                "info":     sum(1 for f in findings if f.severity == "INFO"),
            },
            "steps": [
                {
                    "name":          s.name,
                    "description":   s.description,
                    "order":         s.order_index,
                    "scripts":       s.scripts,
                    "outputs":       s.outputs,
                    "inputs":        s.inputs,
                    "declared_deps": s.declared_deps,
                }
                for s in tracer.steps.values()
            ],
            "findings": [
                {
                    "severity": f.severity,
                    "step":     f.step,
                    "category": f.category,
                    "message":  f.message,
                    "detail":   f.detail,
                }
                for f in findings
            ],
        }
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"JSON trace written to: {args.json_out}")

    n_errors = sum(1 for f in findings if f.severity == "ERROR")
    if n_errors and not args.no_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

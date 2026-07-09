#!/usr/bin/env python3
"""
scaffold_repo.py — create stub .py files at the paths we've mapped out from
scanning .github/workflows/*.yml and run_all.sh, so that find_py_deps.py can
resolve real entry points against an actual (if empty) file tree instead of
reporting everything as unresolved.

This does NOT invent file contents — every stub is a near-empty placeholder
(docstring only). It only fixes *placement* ambiguity for bare filenames
where the containing directory was inferred from context (e.g. "pca.py" is
placed under hypatiax/experiments/benchmarks/ because ci_runner.yml invokes
it alongside other hypatiax/experiments/benchmarks/ scripts).

Run from inside the repo skeleton (the dir containing .github/ and run_all.sh):
    python3 scaffold_repo.py
"""

from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()

# path (relative to repo root) -> short docstring describing where it came from
STUBS = {
    # --- .github/scripts (CI-only helpers, referenced by workflows, not run_all.sh)
    ".github/scripts/check_sweep_coverage.py": "referenced in ci_postprocess.yml",
    ".github/scripts/clean_figures_dir.py": "referenced in ci_postprocess.yml",
    ".github/scripts/clean_stale_checkpoints.py": "referenced in ci_postprocess.yml",
    ".github/scripts/merge_shards.py": "referenced in ci_analysis.yml, ci_pipeline_analysis.yml, ci_pipeline_check.yml, ci_runner.yml",
    ".github/scripts/purge_figures_dest.py": "referenced in ci_postprocess.yml",
    ".github/scripts/run_analysis.py": "referenced in ci_analysis.yml, ci_pipeline_analysis.yml",
    ".github/scripts/validate_analysis_input.py": "referenced in ci_analysis.yml, ci_pipeline_analysis.yml",
    ".github/scripts/check_symbolic_equivalence.py": "referenced in ci_analysis.yml, ci_pipeline_analysis.yml, run_all.sh",
    ".github/scripts/print_repro.py": "referenced in ci_runner.yml, run_all.sh",
    ".github/scripts/flatten_suppb_doubled_path.py": "referenced in ci_postprocess.yml",
    ".github/scripts/merge_extrap_into_benchmark.py": "referenced in ci_analysis.yml",
    ".github/scripts/generate_exp2_pca_comparison_table.py": "referenced in ci_analysis.yml",

    # --- config/
    "config/test_key_status.py": "referenced in ci_runner.yml",

    # --- hypatiax/analysis/
    "hypatiax/analysis/analyze_hybrid_performance.py": "referenced in run_all.sh",

    # --- hypatiax/core/generation/hybrid_all_domains_llm_nn/
    "hypatiax/core/generation/hybrid_all_domains_llm_nn/hybrid_system_llm_nn_all_domains.py": "referenced in ci_runner.yml, ci_postprocess.yml, run_all.sh",

    # --- hypatiax/experiments/tests/
    "hypatiax/experiments/tests/test_enhanced_defi_extrapolation.py": "referenced in run_all.sh",

    # --- hypatiax/experiments/benchmarks/
    "hypatiax/experiments/benchmarks/exp1_ablation.py": "referenced in ci_runner.yml, run_all.sh",
    "hypatiax/experiments/benchmarks/exp3_nguyen12_hybrid50v_02.py": "referenced in ci_runner.yml, run_all.sh",
    "hypatiax/experiments/benchmarks/hypatiax_defi_benchmark_v3c.py": "referenced in ci_runner.yml, run_all.sh",
    "hypatiax/experiments/benchmarks/hypatiax_defi_benchmark_pca.py": "referenced in ci_runner.yml, run_all.sh",
    "hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_pca.py": "referenced in ci_runner.yml, run_all.sh",
    "hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_v2.py": "referenced in ci_runner.yml, run_all.sh",
    "hypatiax/experiments/benchmarks/run_hybrid_system_benchmark.py": "referenced in ci_runner.yml, ci_postprocess.yml, run_all.sh",
    "hypatiax/experiments/benchmarks/run_instability_suite.py": "referenced in ci_runner.yml, ci_postprocess.yml, run_all.sh",
    "hypatiax/experiments/benchmarks/run_noise_sweep_benchmark.py": "referenced in ci_runner.yml, run_all.sh",
    "hypatiax/experiments/benchmarks/run_sample_complexity_benchmark.py": "referenced in ci_runner.yml, run_all.sh",
    "hypatiax/experiments/benchmarks/extrap_r2_far.py": "referenced in run_all.sh",
    "hypatiax/experiments/benchmarks/portfolio_variance_v3c2.py": "referenced in run_all.sh",
    "hypatiax/experiments/benchmarks/pca.py": "bare ref in ci_runner.yml, placed by context (alongside other benchmarks/*)",
    "hypatiax/experiments/benchmarks/symbolic_engine.py": "bare ref in ci_runner.yml, placed by context",
    "hypatiax/experiments/benchmarks/experiment_protocol_all_30.py": "referenced in ci_runner.yml, run_all.sh",
    "hypatiax/experiments/benchmarks/experiment_protocol_benchmark_v2.py": "referenced in ci_runner.yml, run_all.sh",
    "hypatiax/experiments/benchmarks/run_all_checkpoint.py": "referenced in ci_pipeline_check.yml, run_all.sh",
    "hypatiax/experiments/benchmarks/run_analysis.py": "referenced in ci_analysis.yml, ci_pipeline_analysis.yml, run_all.sh",
    "hypatiax/experiments/benchmarks/run_dual_sweep_benchmarks.py": "referenced in run_all.sh",
    "hypatiax/experiments/benchmarks/statistical_analysis.py": "referenced in run_all.sh",
    "hypatiax/experiments/benchmarks/tables-generator.py": "bare ref in ci_runner.yml, placed by context",
    "hypatiax/experiments/benchmarks/hybrid_system_v50_2.py": "bare ref in ci_runner.yml, placed by context",
    "hypatiax/experiments/benchmarks/plot_results.py": "bare ref in ci_postprocess.yml, placed by context",

    # --- scripts/ (repo-root scripts, not .github/scripts)
    "scripts/check_symbolic_equivalence.py": "referenced in run_all.sh",
    "scripts/generate_figures.py": "referenced in ci_pipeline_analysis.yml, ci_postprocess.yml, run_all.sh",
    "scripts/generate_tables.py": "referenced in ci_postprocess.yml, run_all.sh",
    "scripts/merge_extrap_into_benchmark.py": "referenced in run_all.sh",
    "scripts/patches/generate_exp2_pca_comparison_table.py": "referenced in ci_postprocess.yml, run_all.sh",
    "scripts/patches/generate_nguyen12_symequiv_table.py": "referenced in ci_analysis.yml, ci_pipeline_analysis.yml, ci_postprocess.yml, run_all.sh",

    # --- supplementaries/
    "supplementaries/generate_figures/generate_figures.py": "referenced in ci_postprocess.yml",
}


def main():
    created = []
    skipped = []
    for relpath, note in sorted(STUBS.items()):
        target = REPO_ROOT / relpath
        if target.exists():
            skipped.append(relpath)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f'"""Stub. {note}"""\n')
        created.append(relpath)

    print(f"Created {len(created)} stub files.")
    if skipped:
        print(f"Skipped {len(skipped)} already-existing files.")
    for f in created:
        print(f"  + {f}")


if __name__ == "__main__":
    main()

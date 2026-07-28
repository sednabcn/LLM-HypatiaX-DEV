"""
result_subdir_map.py - single source of truth for experiment -> canonical
result subdirectory mapping.

WHY THIS EXISTS:
  Prior to this module, the same {experiment: subdir} mapping was hand-copied
  into six separate places:
    - ci_runner.yml       ("Resolve script/paths" step, per-experiment case)
    - ci_analysis.yml      (MAPPING dict, "Resolve result directory" step)
    - ci_pipeline_public.yml (SUBDIR dict, "diagnose" job)
    - ci_pipeline_check.yml  (SUBDIR dict, x3 separate jobs)
  When ci_runner.yml's suppB path was fixed from the doubled
  ".../noise-sweep/noise-sweep" to the correct single-level
  ".../noise-sweep", the other copies were not updated, and the stale
  doubled path kept being forwarded as an explicit result_subdir override
  from ci_pipeline_public.yml / ci_pipeline_check.yml into ci_analysis.yml
  (which trusts a non-empty forwarded override over its own MAPPING),
  causing merge_shards.py to search an empty directory and fail with
  "FATAL: sweep merge produced zero sweep points".

  This module is the ONLY place this mapping should be defined from now on.
  Every workflow YAML step that needs it should import from here instead of
  hardcoding a dict, exactly as MERGE_REQUIRED_EXPERIMENTS is imported from
  merge_shards.py.

USAGE FROM A WORKFLOW YAML STEP:
  python3 -c "
  import sys
  sys.path.insert(0, '.github/scripts')
  from result_subdir_map import RESULT_SUBDIR_MAP
  print(RESULT_SUBDIR_MAP.get('${EXPERIMENT}', ''))
  "
"""

from __future__ import annotations

RESULT_SUBDIR_MAP: dict[str, str] = {
    "exp1":                "comparison_results/noise-noiseless/noiseless/defi",
    "exp1b":               "comparison_results/noise-noiseless/15",
    "exp1_pca":            "comparison_results/noise-noiseless/noiseless/defi_pca",
    "exp1b_pca":           "comparison_results/noise-noiseless/15_pca",
    "exp1_ablation":       "ablation/exp1_ablation",
    "exp2_feynman":        "comparison_results/feynman-tests/exp2",
    "exp2_feynman_extrap": "comparison_results/feynman-tests/exp2_extrap",
    "exp2_feynman_pca":    "comparison_results/feynman-tests/exp2_pca_4060",
    "exp2_feyman_pca":     "comparison_results/feynman-tests/exp2_pca_4060",  # typo alias (missing 'n')
    "exp2_feyman_extrap":  "comparison_results/feynman-tests/exp2_extrap",   # typo alias (missing 'n')
    "exp2":                "comparison_results/feynman-tests/exp2_multi",
    "exp3":                "extrapolation",
    "exp3b":               "extrapolation/multi_seed",
    "suppA":               "hybrid_pysr/defi",
    "suppB":               "comparison_results/feynman-tests/noise-sweep",
    "suppB_sc":            "comparison_results/feynman-tests/sample-complexity",
    "hybrid_all_domains":  "hybrid_llm_nn/all_domains",
    "instability":         "figures",
    "extrap":              "comparison_results/extrapolation",
}


def resolve_result_subdir(experiment: str, default: str = "") -> str:
    """Look up the canonical result subdir for an experiment.

    Returns `default` (empty string by default) when unmapped, so callers
    that forward this value downstream as an *override* can choose to
    forward nothing rather than a guess -- letting the receiving workflow's
    own copy of this same mapping take over. See the "Only forward a
    result_subdir when we have a confirmed mapping" comment previously in
    ci_pipeline_public.yml's diagnose step for the rationale.
    """
    return RESULT_SUBDIR_MAP.get(experiment, default)


# Experiments whose worker shards must be merged via merge_shards.py before
# analysis. Duplicated here only as a cross-check; merge_shards.py's
# MERGE_REQUIRED_EXPERIMENTS remains the authoritative source for that set.
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: result_subdir_map.py <experiment>", file=sys.stderr)
        sys.exit(1)
    print(resolve_result_subdir(sys.argv[1]))

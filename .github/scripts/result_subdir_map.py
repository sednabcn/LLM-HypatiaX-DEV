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
    "exp1_five":           "five_systems/exp1_five",
    "exp2_five":           "five_systems/exp2_five",
    "exp2_feynman":        "comparison_results/feynman-tests/exp2",
    "exp2_feyman":         "comparison_results/feynman-tests/exp2",  # legacy typo alias
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
    """Look up the canonical result subdir for an experiment."""
    return RESULT_SUBDIR_MAP.get(experiment, default)


def validate_result_subdir(value: str, results_root: str = "hypatiax/data/results") -> str:
    """Validate an explicit/fallback result subdir as a safe relative path.

    The returned value is POSIX-normalised and can safely be joined beneath
    hypatiax/data/results. Reject absolute paths, traversal, control chars,
    and backslashes (which can be path separators on other platforms).
    """
    from pathlib import PurePosixPath

    if not value or value != value.strip():
        raise ValueError("result_subdir must be a non-empty, trimmed relative path")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value) or "=" in value:
        raise ValueError("result_subdir contains a control character or '='")
    if "\\" in value:
        raise ValueError("result_subdir must use POSIX '/' separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("result_subdir must not be absolute or contain traversal components")
    from pathlib import Path
    root = Path(results_root).resolve()
    candidate = (root / Path(*path.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("result_subdir resolves outside hypatiax/data/results") from exc
    return path.as_posix()


if __name__ == "__main__":
    import argparse
    import re
    import sys

    parser = argparse.ArgumentParser(description="Resolve a safe experiment result subdirectory.")
    parser.add_argument("experiment")
    parser.add_argument("--override", default="")
    parser.add_argument("--results-root", default="hypatiax/data/results")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.experiment):
        print("::error::experiment must be a simple identifier (letters, digits, '_' or '-').", file=sys.stderr)
        sys.exit(2)

    if args.override:
        candidate = args.override
    else:
        candidate = resolve_result_subdir(args.experiment, args.experiment)
        if args.experiment not in RESULT_SUBDIR_MAP:
            print(
                f"::warning::No canonical subdir mapping for {args.experiment!r}; "
                "using the experiment identifier as the fallback subdirectory.",
                file=sys.stderr,
            )
    try:
        print(validate_result_subdir(candidate, args.results_root))
    except ValueError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        sys.exit(2)

"""hypatiax/core/case_registry.py

Shared utilities for the case registry pattern used across all benchmark
scripts.  Every benchmark script must expose a ``build_cases()`` function that
returns a deterministic, serialisable list of case dicts.  At runtime the list
is sliced by ``slice_cases()`` according to the CASE_RANGE_START / CASE_RANGE_END
environment variables set by the CI job.

Case dict contract
──────────────────
{
    "id":   str,   # unique, stable across runs, safe for filenames
    "args": list,  # CLI args passed to the underlying worker (all strings)
}

Indexing convention
───────────────────
CI uses 1-based ranges (--case-range 1-4).
``slice_cases`` converts to 0-based Python slices internally so the caller
never has to think about it.
"""

import os
from typing import Dict, List


def slice_cases(cases: List[Dict]) -> List[Dict]:
    """Return the subset of *cases* selected by CASE_RANGE_START / CASE_RANGE_END.

    Environment variables use 1-based, inclusive indexing to match CI syntax:
        --case-range 1-4  →  CASE_RANGE_START=1, CASE_RANGE_END=4

    When neither variable is set the full list is returned unchanged, so scripts
    work normally when run outside CI.
    """
    if not cases:
        return cases

    start_env = os.getenv("CASE_RANGE_START")
    end_env   = os.getenv("CASE_RANGE_END")

    # Default: return everything
    start = int(start_env) - 1 if start_env is not None else 0
    end   = int(end_env)        if end_env   is not None else len(cases)

    # Clamp to valid range
    start = max(0, start)
    end   = min(len(cases), end)

    return cases[start:end]


def log_case_info(cases: List[Dict]) -> None:
    """Print a standardised header so CI logs are debuggable.

    Example output::

        [case-registry] total cases: 15
        [case-registry] slicing: 5-8  (running 4 of 15)
    """
    total = len(cases)
    print(f"[case-registry] total cases: {total}")

    start_env = os.getenv("CASE_RANGE_START")
    end_env   = os.getenv("CASE_RANGE_END")

    if start_env or end_env:
        start_label = start_env or "1"
        end_label   = end_env   or str(total)
        sliced = slice_cases(cases)
        print(
            f"[case-registry] slicing: {start_label}-{end_label}"
            f"  (running {len(sliced)} of {total})"
        )
    else:
        print("[case-registry] no slicing — running all cases")


def validate_case_list(cases: List[Dict], *, source: str = "<unknown>") -> None:
    """Raise ``ValueError`` if *cases* does not satisfy the registry contract.

    Intended for use inside ``build_cases()`` as a lightweight self-check during
    development.  The full structural validator lives in
    ``scripts/validate_case_registry.py`` and is run by CI.
    """
    if not isinstance(cases, list):
        raise ValueError(f"{source}: build_cases() must return a list, got {type(cases)}")
    if len(cases) == 0:
        raise ValueError(f"{source}: build_cases() returned an empty list")

    seen_ids: set = set()
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"{source}: case {i} is not a dict")
        for field in ("id", "args"):
            if field not in case:
                raise ValueError(f"{source}: case {i} missing required field '{field}'")
        if not isinstance(case["id"], str):
            raise ValueError(f"{source}: case {i} 'id' must be a str")
        if not isinstance(case["args"], list):
            raise ValueError(f"{source}: case {i} 'args' must be a list")
        if not all(isinstance(a, str) for a in case["args"]):
            raise ValueError(f"{source}: case {i} 'args' must contain only strings")
        if case["id"] in seen_ids:
            raise ValueError(f"{source}: duplicate case id '{case['id']}'")
        seen_ids.add(case["id"])

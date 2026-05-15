#!/usr/bin/env python3
""".github/scripts/validate_case_registry.py

CI gate that verifies every benchmark script in
``hypatiax/experiments/benchmarks/`` correctly implements the case registry
contract before any compute is spent.

Checks performed per script
────────────────────────────
Structural
  • exposes a ``build_cases()`` callable
  • ``build_cases()`` returns a non-empty ``list``
  • every element is a ``dict`` with ``"id"`` (str) and ``"args"`` (list[str])
  • all case IDs are unique

Behavioural
  • ``build_cases()`` is deterministic (called twice, results compared)

Safety
  • case count warning when > 5 000
  • ID format must match ``[A-Za-z0-9_-]+`` (filesystem-safe)

Import-time side-effects
  • scripts are imported in a subprocess so that heavy imports or accidental
    top-level execution cannot hang or crash the validator process itself.

Exit codes
  0 — all scripts passed
  1 — one or more scripts failed, or no scripts found
"""

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Path is resolved relative to the repo root regardless of where this script
# lives (.github/scripts/).  __file__ gives us an anchor to navigate up.
BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "hypatiax" / "experiments" / "benchmarks"
_ID_RE    = re.compile(r"^[A-Za-z0-9_\-]+$")

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    pass


# ---------------------------------------------------------------------------
# Subprocess-isolated case extraction
# ---------------------------------------------------------------------------
# Benchmarks are imported inside a child process so that heavy top-level
# imports (torch, tensorflow, …) and accidental sys.exit() calls cannot
# crash or hang the validator itself.  The child serialises build_cases()
# output as JSON on stdout; the parent deserialises it.

_EXTRACT_SCRIPT = textwrap.dedent("""
import importlib.util, json, sys

path = sys.argv[1]
spec   = importlib.util.spec_from_file_location("_bench", path)
module = importlib.util.module_from_spec(spec)
module.__name__ = "_bench"          # suppress __main__ guards
spec.loader.exec_module(module)

if not hasattr(module, "build_cases"):
    print(json.dumps({"error": "missing build_cases() function"}))
    sys.exit(0)

fn = module.build_cases
if not callable(fn):
    print(json.dumps({"error": "build_cases is not callable"}))
    sys.exit(0)

try:
    c1 = fn()
    c2 = fn()
    print(json.dumps({"cases1": c1, "cases2": c2}))
except Exception as exc:
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
""")


def _extract_cases_subprocess(path: Path) -> Dict[str, Any]:
    """Run the benchmark in a child process; return its JSON output."""
    result = subprocess.run(
        [sys.executable, "-c", _EXTRACT_SCRIPT, str(path)],
        capture_output=True, text=True, timeout=60,
    )
    raw = result.stdout.strip()
    if not raw:
        stderr = result.stderr.strip()
        raise ValidationError(
            f"subprocess produced no output (exit {result.returncode})"
            + (f": {stderr[:200]}" if stderr else "")
        )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"subprocess output is not valid JSON: {exc}") from exc


def _validate_module(path: Path) -> int:
    """Run all checks against a benchmark script via subprocess.  Returns case count."""
    payload = _extract_cases_subprocess(path)

    if "error" in payload:
        raise ValidationError(payload["error"])

    cases1 = payload["cases1"]
    cases2 = payload["cases2"]

    # Determinism check
    try:
        j1 = json.dumps(cases1, sort_keys=True)
        j2 = json.dumps(cases2, sort_keys=True)
    except TypeError as exc:
        raise ValidationError(
            f"build_cases() returned non-serialisable data: {exc}"
        ) from exc

    if j1 != j2:
        raise ValidationError(
            "build_cases() is not deterministic — two calls returned different results"
        )

    count = _validate_cases(cases1, path)
    return count


def _validate_cases(cases: Any, path: Path) -> int:
    """Validate *cases* against the registry contract.  Returns case count."""
    if not isinstance(cases, list):
        raise ValidationError(
            f"build_cases() must return a list, got {type(cases).__name__}"
        )
    if len(cases) == 0:
        raise ValidationError("build_cases() returned an empty list — no cases defined")

    seen_ids: set = set()
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValidationError(f"case[{i}] is not a dict")
        for field in ("id", "args"):
            if field not in case:
                raise ValidationError(f"case[{i}] missing required field '{field}'")
        if not isinstance(case["id"], str):
            raise ValidationError(
                f"case[{i}]['id'] must be str, got {type(case['id']).__name__}"
            )
        if not isinstance(case["args"], list):
            raise ValidationError(
                f"case[{i}]['args'] must be list, got {type(case['args']).__name__}"
            )
        if not all(isinstance(a, str) for a in case["args"]):
            raise ValidationError(f"case[{i}]['args'] must contain only strings")
        if not _ID_RE.match(case["id"]):
            raise ValidationError(
                f"case[{i}] id '{case['id']}' contains invalid characters "
                f"(must match [A-Za-z0-9_-]+)"
            )
        if case["id"] in seen_ids:
            raise ValidationError(f"duplicate case id: '{case['id']}'")
        seen_ids.add(case["id"])

    return len(cases)




# ---------------------------------------------------------------------------
# Per-file entry point
# ---------------------------------------------------------------------------

def validate_file(path: Path) -> Tuple[bool, Optional[str]]:
    """Validate a single benchmark script.

    Returns ``(ok, warning_or_none)``.
    Prints a ✅ / ❌ line immediately so CI logs stream in real time.
    """
    warning: Optional[str] = None
    try:
        count  = _validate_module(path)

        msg = f"✅ {path.name}: {count} cases"
        if count > 5000:
            warning = f"⚠  {path.name}: very large case count ({count}) — consider splitting"
        print(msg)
        if warning:
            print(warning)
        return True, warning

    except ValidationError as exc:
        print(f"❌ {path.name}: {exc}")
        return False, None
    except subprocess.TimeoutExpired:
        print(f"❌ {path.name}: timed out after 60 s — possible infinite loop at import time")
        return False, None
    except Exception as exc:  # noqa: BLE001
        print(f"❌ {path.name}: unexpected error — {type(exc).__name__}: {exc}")
        return False, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not BENCH_DIR.exists():
        print(
            f"❌ Benchmark directory not found: {BENCH_DIR}\n"
            "   Run this script from the repository root."
        )
        sys.exit(1)

    files = sorted(BENCH_DIR.glob("*.py"))
    # Skip __init__.py and any private helpers
    files = [f for f in files if not f.name.startswith("_")]

    if not files:
        print(f"❌ No benchmark scripts found in {BENCH_DIR}")
        sys.exit(1)

    print(f"Validating {len(files)} benchmark script(s) in {BENCH_DIR}/\n")
    print("-" * 60)

    all_ok   = True
    warnings: List[str] = []

    for path in files:
        ok, warning = validate_file(path)
        if not ok:
            all_ok = False
        if warning:
            warnings.append(warning)

    print("-" * 60)

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  {w}")

    print()
    if all_ok:
        print(f"✅ All {len(files)} script(s) passed validation")
    else:
        print("❌ Validation FAILED — fix the errors above before merging")
        sys.exit(1)


if __name__ == "__main__":
    main()

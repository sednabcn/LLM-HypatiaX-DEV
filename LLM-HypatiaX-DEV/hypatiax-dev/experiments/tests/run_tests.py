#!/usr/bin/env python3
"""
run_tests.py
============
Standalone test runner for test_sweep_benchmarks.py.
No pytest command required — just:

    python run_tests.py
    python run_tests.py -v
    python run_tests.py -k noise
    python run_tests.py -k catastrophic
    python run_tests.py -k "TopTwo"
    python run_tests.py -k "collision or mtime"
    python run_tests.py --list

Drop this file next to test_sweep_benchmarks.py (i.e. inside tests/).
It auto-detects the test file and the parent benchmark directory.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate the test file relative to this runner.
# Expected layout (from QUICKSTART):
#
#   experiments/benchmarks/
#   ├── run_noise_sweep_benchmark.py
#   ├── run_sample_complexity_benchmark.py
#   └── tests/
#       ├── run_tests.py           ← this file
#       └── test_sweep_benchmarks.py
#
# The test file imports its siblings from _HERE.parent, so we make sure
# that directory is on sys.path before handing off to pytest.
# ---------------------------------------------------------------------------

_HERE      = Path(__file__).resolve().parent
_TEST_FILE = _HERE / "test_sweep_benchmarks.py"
_BENCH_DIR = _HERE.parent   # experiments/benchmarks/

if not _TEST_FILE.exists():
    print(
        f"ERROR: test file not found at {_TEST_FILE}\n"
        f"Make sure run_tests.py lives in the same directory as "
        f"test_sweep_benchmarks.py.",
        file=sys.stderr,
    )
    sys.exit(1)

# Ensure the benchmark scripts are importable (mirrors what the test file does)
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))


# ---------------------------------------------------------------------------
# Check pytest is available — give a helpful message if not.
# ---------------------------------------------------------------------------
try:
    import pytest
except ImportError:
    print(
        "ERROR: pytest is not installed.\n"
        "Install it with:  pip install pytest\n"
        "Then re-run:      python run_tests.py",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Custom pytest plugin — captures counts and formats the summary line.
# ---------------------------------------------------------------------------
class _SummaryPlugin:
    """Collects pass/fail/skip/error counts and wall-clock time."""

    def __init__(self) -> None:
        self.passed  = 0
        self.failed  = 0
        self.errors  = 0
        self.skipped = 0
        self._t0: float = 0.0

    # hooks
    def pytest_sessionstart(self, session) -> None:          # noqa: ARG002
        self._t0 = time.perf_counter()

    def pytest_runtest_logreport(self, report) -> None:
        if report.when != "call":
            # Count errors that happen in setup/teardown too
            if report.when in ("setup", "teardown") and report.failed:
                self.errors += 1
            return
        if report.passed:
            self.passed  += 1
        elif report.failed:
            self.failed  += 1
        elif report.skipped:
            self.skipped += 1

    def summary_line(self) -> str:
        elapsed = time.perf_counter() - self._t0
        total   = self.passed + self.failed + self.errors + self.skipped
        return (
            f"{total} run   "
            f"{self.passed} passed   "
            f"{self.failed} failed   "
            f"{self.errors} errors   "
            f"{self.skipped} skipped   "
            f"({elapsed:.1f}s)"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python run_tests.py",
        description="Standalone runner for test_sweep_benchmarks.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run_tests.py                        # run all 109 tests
  python run_tests.py -v                     # verbose (one line per test)
  python run_tests.py -k noise               # filter by name
  python run_tests.py -k catastrophic
  python run_tests.py -k "TopTwo"
  python run_tests.py -k "collision or mtime"
  python run_tests.py --list                 # list without running
  python run_tests.py -v --tb short          # show short tracebacks
""",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show one line per test (pytest -v)",
    )
    p.add_argument(
        "-k",
        metavar="EXPR",
        dest="keyword",
        default=None,
        help="Only run tests whose names match EXPR (same as pytest -k)",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List all collected test names without running them",
    )
    p.add_argument(
        "--tb",
        metavar="STYLE",
        default="short",
        choices=["short", "long", "line", "no", "auto"],
        help="Traceback style on failure (default: short)",
    )
    p.add_argument(
        "-x", "--exitfirst",
        action="store_true",
        help="Stop after first failure",
    )
    p.add_argument(
        "--no-header",
        action="store_true",
        help="Suppress the pytest header",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    # Build pytest argv
    pytest_args: list[str] = [str(_TEST_FILE)]

    if args.verbose:
        pytest_args.append("-v")

    if args.keyword:
        pytest_args += ["-k", args.keyword]

    if args.list:
        pytest_args += ["--collect-only", "-q"]

    if args.exitfirst:
        pytest_args.append("-x")

    if args.no_header:
        pytest_args.append("-p")
        pytest_args.append("no:header")

    # Always suppress pytest's own summary in favour of our compact line,
    # unless the user asked for verbose (where pytest's output is useful).
    pytest_args += ["--tb", args.tb]

    if not args.list:
        # -p no:terminal keeps pytest quiet about its own summary section;
        # we print our own summary line instead.
        pytest_args += ["-p", "no:cacheprovider"]

    plugin  = _SummaryPlugin()

    print(f"Running tests in: {_TEST_FILE.relative_to(Path.cwd()) if _TEST_FILE.is_relative_to(Path.cwd()) else _TEST_FILE}")
    print()

    ret = pytest.main(pytest_args, plugins=[plugin])

    if not args.list:
        print()
        print("─" * 60)
        print(plugin.summary_line())

    return ret


if __name__ == "__main__":
    sys.exit(main())

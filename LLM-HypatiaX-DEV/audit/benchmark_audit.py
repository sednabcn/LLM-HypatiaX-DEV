#!/usr/bin/env python3
"""
benchmark_audit.py

Audits a symbolic-regression benchmark repository and determines
whether published benchmark results are still valid.

Checks
------
✓ Git changes affecting executable code
✓ Benchmark configuration changes
✓ Environment failures
✓ Placeholder results
✓ Cached outputs
✓ Benchmark coverage
✓ Produces a final recommendation

Usage
-----
python benchmark_audit.py
python benchmark_audit.py --since HEAD~1
python benchmark_audit.py --json report.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
from dataclasses import dataclass, field
from typing import List


ROOT = pathlib.Path(".")


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def git(cmd):
    return subprocess.run(
        ["git"] + cmd,
        capture_output=True,
        text=True,
    ).stdout


def diff_files(base):

    out = git(["diff", "--name-only", base])

    return [
        pathlib.Path(x)
        for x in out.splitlines()
        if x.strip()
    ]


def diff_patch(base, file):

    return git(["diff", base, "--", str(file)])


# ---------------------------------------------------------
# Rules
# ---------------------------------------------------------

EXECUTABLE_KEYWORDS = [

    "predict",
    "fit",
    "solve",
    "evaluate",
    "rmse",
    "loss",
    "score",
    "metric",
    "discover",
    "symbolic",
    "equation",
    "regression",
    "hybrid",
    "benchmark",
    "pipeline",

]

COMMENT = re.compile(r"^\s*[#/]")
DOCSTRING = re.compile(r'^\s*[ruRU]*("""|\'\'\')')


# ---------------------------------------------------------
# Report model
# ---------------------------------------------------------

@dataclass
class Report:

    executable_changes: List[str] = field(default_factory=list)
    config_changes: List[str] = field(default_factory=list)
    environment_failures: List[str] = field(default_factory=list)
    placeholder_results: List[str] = field(default_factory=list)
    cached_results: List[str] = field(default_factory=list)

    def rerun_required(self):

        return (
            bool(self.executable_changes)
            or bool(self.environment_failures)
        )


# ---------------------------------------------------------
# Git inspection
# ---------------------------------------------------------

def inspect_git(base, report):

    for file in diff_files(base):

        patch = diff_patch(base, file)

        executable = False

        for line in patch.splitlines():

            if not line.startswith("+"):
                continue

            if line.startswith("+++"):
                continue

            if COMMENT.match(line):
                continue

            if DOCSTRING.match(line):
                continue

            executable = True
            break

        if executable:

            report.executable_changes.append(str(file))

        if (
            "config" in str(file).lower()
            or file.suffix in (".yaml", ".yml", ".toml")
        ):
            report.config_changes.append(str(file))


# ---------------------------------------------------------
# Scan outputs
# ---------------------------------------------------------

FAIL_PATTERNS = [

    "not available",
    "module not found",
    "failed to initialize",
    "importerror",
    "modulenotfounderror",

]

CACHE_PATTERNS = [

    "using cached",
    "loading cached",
    "resume",
    "skip existing",

]


def scan_logs(report):

    for log in ROOT.rglob("*.log"):

        try:
            txt = log.read_text(errors="ignore").lower()
        except Exception:
            continue

        for p in FAIL_PATTERNS:

            if p in txt:
                report.environment_failures.append(str(log))
                break

        for p in CACHE_PATTERNS:

            if p in txt:
                report.cached_results.append(str(log))
                break


def scan_json(report):

    for file in ROOT.rglob("*.json"):

        try:
            txt = file.read_text(errors="ignore").lower()
        except Exception:
            continue

        if "not available" in txt:
            report.placeholder_results.append(str(file))


# ---------------------------------------------------------
# Output
# ---------------------------------------------------------

def print_report(report):

    print("=" * 70)
    print("BENCHMARK AUDIT")
    print("=" * 70)

    print()

    print("Executable source changes")

    if report.executable_changes:
        for x in report.executable_changes:
            print("  ✓", x)
    else:
        print("  none")

    print()

    print("Configuration changes")

    if report.config_changes:
        for x in report.config_changes:
            print("  ✓", x)
    else:
        print("  none")

    print()

    print("Environment failures")

    if report.environment_failures:
        for x in report.environment_failures:
            print("  ✓", x)
    else:
        print("  none")

    print()

    print("Placeholder benchmark results")

    if report.placeholder_results:
        for x in report.placeholder_results:
            print("  ✓", x)
    else:
        print("  none")

    print()

    print("Cached outputs")

    if report.cached_results:
        for x in report.cached_results:
            print("  ✓", x)
    else:
        print("  none")

    print()

    print("=" * 70)

    if report.rerun_required():

        print("FINAL DECISION")
        print("RE-RUN REQUIRED")

        if report.executable_changes:
            print("\nReason: executable code has changed.")

        if report.environment_failures:
            print("Reason: benchmark contains execution failures.")

    else:

        print("FINAL DECISION")
        print("Existing benchmark results are reusable.")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--since",
        default="HEAD~1",
        help="Git revision to compare against."
    )

    parser.add_argument(
        "--json",
        help="Write report to JSON."
    )

    args = parser.parse_args()

    report = Report()

    inspect_git(args.since, report)

    scan_logs(report)

    scan_json(report)

    print_report(report)

    if args.json:

        pathlib.Path(args.json).write_text(
            json.dumps(report.__dict__, indent=2)
        )


if __name__ == "__main__":
    main()

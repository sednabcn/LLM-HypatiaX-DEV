#!/usr/bin/env python3
"""
detect_prefix_duplication_code.py

Scans SOURCE CODE (not output filenames) for constructs that are likely to
PRODUCE double-prefixed / double-suffixed filenames such as:
    fig_fig_...            figures_figures_...
    figures_back__figures__...   PROD__PROD__...
    <name>_<name>.png       <dir>__<dir>__<file>

This is the complement of detect_contaminated_files.py:
  - detect_contaminated_files.py -> finds ALREADY-CONTAMINATED files on disk.
  - detect_prefix_duplication_code.py -> finds the CODE that would create them.

It's a heuristic static scanner (regex-based), not a full parser, so treat
hits as "worth a manual look" rather than definitive proof of a bug.

Detection strategies
---------------------
1. self-glued-token
   A literal token immediately glued to itself with _ / __ / - in the code
   text itself, e.g.  "figures_figures_"  or  "fig_fig_".

2. shell-var-self-concat
   A shell variable referenced twice in the same statement with only
   underscores/literal glue between the two references, where the
   surrounding line looks path-like, e.g.:
       "${DIR}_${DIR}"   "${d}_back__${d}__$f"   "$name/$name"

3. python-fstring-self-concat
   An f-string (or .format/%-style) that interpolates the same variable
   name twice in a row, e.g.  f"{prefix}_{prefix}_{stem}.png"

4. cp-mv-rsync-self-prefix
   A cp/mv/rsync/shutil.copy call whose destination is built by prefixing
   the SAME directory/variable name onto a path that is itself derived
   from that directory (classic "backup into a name that already contains
   the source dir name" bug) e.g.:
       cp "$f" "${DEST}/${DEST}_$(basename "$f")"
       cp "$SRC_DIR/$f" "backup_${SRC_DIR}/${SRC_DIR}_$f"

5. loop-over-own-output
   A glob/find/for-loop whose source directory is the SAME as its
   destination directory while also renaming with a prefix -- a common
   root cause of repeated re-prefixing on every re-run (each run doubles
   the prefix again).

Usage:
    python3 detect_prefix_duplication_code.py
    python3 detect_prefix_duplication_code.py path/to/repo
    python3 detect_prefix_duplication_code.py run_all.sh other_script.py
    python3 detect_prefix_duplication_code.py --json report.json .

Exit code:
    0 - no suspicious constructs found
    1 - suspicious constructs found (useful as a CI gate)
    2 - usage / path error
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

SOURCE_EXTS = {".sh", ".bash", ".py", ".yml", ".yaml"}

# generic "token glued to itself" -- catches it appearing literally in code
# (string literals, heredocs, echo statements, comments describing the bug, etc.)
RE_SELF_GLUED_TOKEN = re.compile(r"\b([A-Za-z][A-Za-z0-9]{2,})[_-]{1,2}\1\b", re.IGNORECASE)

# shell variable referenced twice with only underscore/literal glue between,
# e.g. ${DIR}_${DIR}, ${d}_back__${d}__, "$name/$name"
RE_SHELL_VAR = re.compile(
    r"""\$\{?([A-Za-z_][A-Za-z0-9_]*)\}? [-_./]{1,10} (?:back__|Back__)? \$\{?\1\}?""",
    re.VERBOSE,
)

# python f-string / .format / % interpolating the same identifier twice in a row
RE_PY_FSTRING_VAR = re.compile(
    r"""\{([A-Za-z_][A-Za-z0-9_]*)\}[-_]{0,2}\{?\1\}?""",
)

# cp / mv / rsync / shutil.copy* lines
RE_COPY_CMD = re.compile(r"\b(cp|mv|rsync|shutil\.copy\w*|shutil\.move)\b", re.IGNORECASE)

# a variable appearing twice anywhere within a cp/mv destination argument
RE_VAR_TOKEN = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")


@dataclass
class Finding:
    path: str
    line_no: int
    rule: str
    detail: str
    snippet: str


def iter_source_files(roots: list[str]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        p = Path(root)
        if not p.exists():
            continue
        if p.is_file():
            files.append(p)
            continue
        for dirpath, dirnames, filenames in os.walk(p):
            # skip vcs/venv/node_modules noise
            dirnames[:] = [d for d in dirnames if d not in (".git", "node_modules", "__pycache__", ".venv", "venv")]
            for fn in filenames:
                fp = Path(dirpath) / fn
                if fp.suffix.lower() in SOURCE_EXTS:
                    files.append(fp)
    return files


def scan_file(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return findings
    lines = text.splitlines()

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            # still worth checking comments that describe a bug, but skip
            # pure separator comments to cut noise
            if not stripped.startswith("#"):
                continue

        # Rule 1: literal self-glued token anywhere in the line
        for m in RE_SELF_GLUED_TOKEN.finditer(line):
            token = m.group(1)
            if len(token) >= 3:
                findings.append(Finding(
                    path=str(path), line_no=i, rule="self-glued-token",
                    detail=f"literal token '{token}' glued to itself: '{m.group(0)}'",
                    snippet=stripped[:160],
                ))

        # Rule 2: shell variable self-concat (only meaningful in shell files)
        if path.suffix in (".sh", ".bash", ".yml", ".yaml") or "$" in line:
            for m in RE_SHELL_VAR.finditer(line):
                findings.append(Finding(
                    path=str(path), line_no=i, rule="shell-var-self-concat",
                    detail=f"variable '{m.group(1)}' referenced twice with glue: '{m.group(0)}'",
                    snippet=stripped[:160],
                ))

        # Rule 3: python f-string self concat
        if path.suffix == ".py" and ("f'" in line or 'f"' in line or ".format(" in line):
            for m in RE_PY_FSTRING_VAR.finditer(line):
                findings.append(Finding(
                    path=str(path), line_no=i, rule="python-fstring-self-concat",
                    detail=f"variable '{m.group(1)}' interpolated twice in a row: '{m.group(0)}'",
                    snippet=stripped[:160],
                ))

        # Rule 4: cp/mv/rsync/shutil.copy destination reusing a var already
        # present earlier in the same statement (source) -> classic doubling
        if RE_COPY_CMD.search(line):
            vars_seen = RE_VAR_TOKEN.findall(line)
            if len(vars_seen) != len(set(vars_seen)):
                dupes = sorted({v for v in vars_seen if vars_seen.count(v) > 1})
                if dupes:
                    findings.append(Finding(
                        path=str(path), line_no=i, rule="cp-mv-rsync-self-prefix",
                        detail=f"copy/move command reuses variable(s) {dupes} in both source and destination "
                               f"-- check whether the destination re-prefixes a name that already contains it",
                        snippet=stripped[:160],
                    ))

    return findings


def print_report(findings: list[Finding], files_scanned: int) -> None:
    print(f"Scanned {files_scanned} source file(s).\n")
    if not findings:
        print("✔ No suspicious prefix/suffix-doubling constructs found in source code.")
        return

    by_rule: dict[str, list[Finding]] = {}
    for f in findings:
        by_rule.setdefault(f.rule, []).append(f)

    print(f"✘ Found {len(findings)} suspicious construct(s) across {len(by_rule)} rule(s):\n")
    for rule, group in by_rule.items():
        print(f"[{rule}]  ({len(group)} hit(s))")
        for f in group:
            print(f"    {f.path}:{f.line_no}")
            print(f"        {f.detail}")
            print(f"        > {f.snippet}")
        print()


def write_json(findings: list[Finding], out_path: str) -> None:
    with open(out_path, "w") as fh:
        json.dump([asdict(f) for f in findings], fh, indent=2)
    print(f"JSON report written to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", default=["."], help="Files or directories to scan. Default: .")
    parser.add_argument("--json", metavar="PATH", help="Write findings to a JSON file.")
    args = parser.parse_args()

    files = iter_source_files(args.paths)
    if not files:
        print("No .sh/.bash/.py/.yml/.yaml files found under the given path(s).")
        return 2

    all_findings: list[Finding] = []
    for f in files:
        all_findings.extend(scan_file(f))

    print_report(all_findings, len(files))

    if args.json:
        write_json(all_findings, args.json)

    return 1 if all_findings else 0


if __name__ == "__main__":
    sys.exit(main())

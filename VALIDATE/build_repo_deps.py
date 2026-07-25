#!/usr/bin/env python3
"""
build_repo_deps.py — take the JSON report produced by find_py_deps.py and
copy every touched file (entry points + their local dependencies) into a
separate mirror directory, preserving relative paths.

Usage:
    python3 build_repo_deps.py [REPO_ROOT] [REPORT_JSON] [DEST_DIR]

Defaults:
    REPO_ROOT   = .
    REPORT_JSON = ./py_deps_report.json
    DEST_DIR    = $HOME/Downloads/repo-deps

Example:
    # after running:
    #   python3 find_py_deps.py . --workflow ci_pipeline_public.yml
    # then:
    python3 build_repo_deps.py . py_deps_report.json ~/Downloads/repo-deps
"""
import json
import shutil
import sys
from pathlib import Path


def main():
    repo_root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    report_path = Path(sys.argv[2] if len(sys.argv) > 2 else "py_deps_report.json").resolve()
    dest_root = Path(
        sys.argv[3] if len(sys.argv) > 3 else str(Path.home() / "Downloads" / "repo-deps")
    ).resolve()

    if not report_path.is_file():
        print(f"ERROR: report not found at {report_path}", file=sys.stderr)
        print("Run find_py_deps.py first to generate it.", file=sys.stderr)
        sys.exit(1)

    report = json.loads(report_path.read_text())
    touched = report.get("all_files_touched", [])

    if not touched:
        print("No files listed under 'all_files_touched' in the report — nothing to copy.")
        sys.exit(0)

    dest_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = []
    for relpath in touched:
        src = repo_root / relpath
        dst = dest_root / relpath
        if not src.is_file():
            missing.append(relpath)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    # Also drop the report itself alongside the mirror for reference
    shutil.copy2(report_path, dest_root / "py_deps_report.json")

    print(f"Repo root:   {repo_root}")
    print(f"Report:      {report_path}")
    print(f"Destination: {dest_root}")
    print(f"Copied:      {copied} file(s)")
    if missing:
        print(f"Missing (listed in report but not found on disk): {len(missing)}")
        for m in missing:
            print(f"  - {m}")
    print("Done.")


if __name__ == "__main__":
    main()

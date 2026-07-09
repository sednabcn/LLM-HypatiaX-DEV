#!/usr/bin/env python3
"""
extract_repo_subset.py — build a separate, self-contained repo-root
directory containing only the .py files that ONE workflow (e.g.
ci_pipeline_public.yml) actually depends on, plus their transitively
imported local dependencies.

This is deliberately different from scaffold_repo_separate.py:
  - scaffold_repo_separate.py writes empty STUB files for a hardcoded,
    manually-curated list (used when the real files don't exist yet).
  - extract_repo_subset.py COPIES REAL FILE CONTENTS out of an existing
    repo, scoped to whatever a single workflow file references, using
    find_py_deps.py's own scanning/resolution/import-tracing so the
    subset always matches what that workflow would actually execute.

The source repo is only ever READ. Nothing is written back into it.

Usage:
    python3 extract_repo_subset.py REPO_ROOT --workflow ci_pipeline_public.yml --out /path/to/subset
    python3 extract_repo_subset.py REPO_ROOT --workflow ci_pipeline_public.yml --out /path/to/subset --include-run-all

Requires find_py_deps.py to be in the same directory (imported, not re-run
as a subprocess, so it reuses the same regex/resolution/tracing code).
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_py_deps import collect_entry_points, resolve_reference, trace_dependencies  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("repo_root", help="Path to the real repo root (read-only)")
    parser.add_argument(
        "-w", "--workflow", required=True,
        help="Workflow file to scope the subset to, e.g. ci_pipeline_public.yml "
             "(matched by exact filename or filename without extension)",
    )
    parser.add_argument(
        "-o", "--out", required=True, type=Path,
        help="Output directory for the extracted subset repo (must not exist or must be empty)",
    )
    parser.add_argument(
        "--include-run-all", action="store_true",
        help="Also pull in files referenced from run_all.sh (default: workflow only)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_root = args.out.expanduser().resolve()

    if not repo_root.is_dir():
        print(f"ERROR: {repo_root} is not a directory", file=sys.stderr)
        sys.exit(1)

    if out_root.exists() and any(out_root.iterdir()):
        print(f"ERROR: --out {out_root} already exists and is not empty; "
              f"refusing to write into it", file=sys.stderr)
        sys.exit(1)

    # Guard: never allow the subset to land inside the real repo tree.
    try:
        out_root.relative_to(repo_root)
        print(f"ERROR: --out must not be inside the repo root ({repo_root})", file=sys.stderr)
        sys.exit(1)
    except ValueError:
        pass  # good: out_root is not inside repo_root

    workflows_dir = repo_root / ".github" / "workflows"
    if not workflows_dir.is_dir():
        print(f"ERROR: {workflows_dir} does not exist.", file=sys.stderr)
        print(f"  You passed repo_root={args.repo_root!r} (resolved to {repo_root}). "
              f"Make sure that's the actual repo root (the directory containing "
              f".github/ and run_all.sh), not a subdirectory or its parent.",
              file=sys.stderr)
        sys.exit(1)

    all_workflow_files = sorted(p.name for p in workflows_dir.iterdir() if p.is_file())
    matched = [
        name for name in all_workflow_files
        if name == args.workflow or Path(name).stem == args.workflow
    ]
    if not matched:
        print(f"ERROR: no file in {workflows_dir} matches --workflow {args.workflow!r}.",
              file=sys.stderr)
        if all_workflow_files:
            print(f"  Files actually present there:", file=sys.stderr)
            for name in all_workflow_files:
                print(f"    - {name}", file=sys.stderr)
        else:
            print(f"  {workflows_dir} exists but is empty.", file=sys.stderr)
        print(f"  Note: only files directly inside .github/workflows/ are scanned "
              f"(not subdirectories).", file=sys.stderr)
        sys.exit(1)

    sources = collect_entry_points(
        repo_root, workflow_filter=args.workflow, include_run_all=args.include_run_all
    )
    if not sources:
        scope = f"workflow {args.workflow!r}" + (" or run_all.sh" if args.include_run_all else "")
        print(f"Matched workflow file(s) {matched}, but found zero strings ending in "
              f".py inside {'them' if len(matched) > 1 else 'it'} for {scope}.", file=sys.stderr)
        print(f"  This usually means the workflow doesn't call Python scripts directly "
              f"— e.g. it delegates to a reusable/composite workflow ('uses:'), calls "
              f"a shell script instead, or invokes scripts by a name that doesn't end "
              f"in literal '.py' in the text. Open the file and check what it actually "
              f"runs.", file=sys.stderr)
        sys.exit(1)

    touched: set[Path] = set()
    unresolved: dict[str, list[str]] = {}
    for source_file, refs in sources.items():
        for ref in sorted(refs):
            resolved = resolve_reference(ref, repo_root)
            if resolved:
                touched.add(resolved)
                touched.update(trace_dependencies(resolved, repo_root))
            else:
                unresolved.setdefault(ref, []).append(source_file)

    out_root.mkdir(parents=True, exist_ok=True)
    copied = []
    for f in sorted(touched):
        rel = f.relative_to(repo_root)
        dest = out_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        copied.append(str(rel))

    print(f"Repo root (read-only): {repo_root}")
    print(f"Subset output dir:     {out_root}")
    print(f"Scoped to:             {args.workflow}"
          + (" + run_all.sh" if args.include_run_all else ""))
    print(f"Copied {len(copied)} files:")
    for c in copied:
        print(f"  + {c}")

    if unresolved:
        print(f"\nUnresolved references ({len(unresolved)}) — not copied, "
              f"no matching file found in repo:")
        for ref, found_in in sorted(unresolved.items()):
            print(f"  - {ref}  (in: {', '.join(sorted(set(found_in)))})")
        print(
            "\nThese are likely dynamically-constructed filenames (shell/CI "
            "variables) that can't be resolved statically. If they're real "
            "files, add them manually or extend scaffold_repo_separate.py's "
            "STUBS dict."
        )


if __name__ == "__main__":
    main()

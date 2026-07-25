#!/usr/bin/env python3
"""
scaffold_repo_full.py — dynamically scan a real repo's .github/workflows/*.yml
and run_all.sh for every referenced .py path, then write a FULL stub mirror
of all of them into a SEPARATE output directory.

Unlike scaffold_repo_separate.py (which used a hardcoded, one-off list of 46
paths derived from a single earlier analysis), this version re-derives the
list every run by actually reading your workflow files and run_all.sh. That
means:
  - it picks up ALL ci_*.yml files under .github/workflows/, not just the
    ones seen in a previous session
  - it picks up everything run_all.sh invokes
  - the output tree includes every referenced path, not just the ones
    missing from the real repo (so you get the complete picture, including
    files that already exist)

The real repo is only READ, never written to.

USAGE (run from inside your real repo root, the dir containing .github/ and
run_all.sh):

    python3 scaffold_repo_full.py --out ~/Downloads/hypatiax_stubs

Options:
    --out PATH       output directory for the full stub mirror
                      (default: ./scaffolded_stubs_full)
    --repo PATH      repo root to scan (default: current directory)
    --report-only    just print what was found, don't write any files
"""

import argparse
import re
import sys
from pathlib import Path

# Matches things like:  python3 path/to/thing.py, python path/to/thing.py,
# ./path/to/thing.py, bash path/to/thing.py (rare), or a bare "thing.py"
# token anywhere in a shell/yaml line.
PY_REF_RE = re.compile(
    r"""
    (?:
        (?:python3?|py)\s+       # python invocation
        (?P<path_a>[./\w\-]+\.py)
    )
    |
    (?:
        (?P<path_b>(?:\./|[./\w\-]*/)[\w\-]+\.py)   # explicit relative/absolute path
    )
    |
    (?P<path_c>\b[\w\-]+\.py\b)   # bare filename fallback
    """,
    re.VERBOSE,
)


def find_py_refs(text: str) -> set:
    refs = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for m in PY_REF_RE.finditer(line):
            path = m.group("path_a") or m.group("path_b") or m.group("path_c")
            if path:
                while path.startswith("./"):
                    path = path[2:]
                refs.add(path)
    return refs


def resolve_bare_names(refs: set, repo_root: Path) -> dict:
    """
    For bare filenames (no directory component), try to find an existing
    file of that name anywhere in the repo to infer placement. If found,
    map to that path. If not found anywhere, leave it at repo root and
    flag it for manual review.
    """
    resolved = {}
    ambiguous = []
    unresolved = []

    existing_by_name = {}
    if repo_root.exists():
        for p in repo_root.rglob("*.py"):
            existing_by_name.setdefault(p.name, []).append(p.relative_to(repo_root))

    for ref in sorted(refs):
        if "/" in ref:
            resolved[ref] = ref
            continue
        matches = existing_by_name.get(ref, [])
        if len(matches) == 1:
            resolved[ref] = str(matches[0])
        elif len(matches) > 1:
            resolved[ref] = str(matches[0])  # best-effort, first match
            ambiguous.append((ref, [str(m) for m in matches]))
        else:
            resolved[ref] = ref  # falls at repo root, needs manual check
            unresolved.append(ref)

    return resolved, ambiguous, unresolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."), help="Repo root to scan")
    parser.add_argument("--out", type=Path, default=None, help="Output dir for stub mirror")
    parser.add_argument("--report-only", action="store_true", help="Only print findings")
    args = parser.parse_args()

    repo_root = args.repo.expanduser().resolve()
    out_root = (args.out or (repo_root / "scaffolded_stubs_full")).expanduser().resolve()

    workflows_dir = repo_root / ".github" / "workflows"
    run_all = repo_root / "run_all.sh"

    if not workflows_dir.exists():
        print(f"WARNING: no {workflows_dir} found.", file=sys.stderr)
    if not run_all.exists():
        print(f"WARNING: no {run_all} found.", file=sys.stderr)

    sources = []
    all_refs = {}  # ref -> set of source filenames

    if workflows_dir.exists():
        yml_files = sorted(list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml")))
        for yml in yml_files:
            sources.append(yml)
            text = yml.read_text(errors="ignore")
            for ref in find_py_refs(text):
                all_refs.setdefault(ref, set()).add(yml.name)

    if run_all.exists():
        sources.append(run_all)
        text = run_all.read_text(errors="ignore")
        for ref in find_py_refs(text):
            all_refs.setdefault(ref, set()).add(run_all.name)

    print(f"Scanned {len(sources)} source file(s):")
    for s in sources:
        print(f"  - {s.relative_to(repo_root) if s.is_relative_to(repo_root) else s}")

    refs = set(all_refs.keys())
    resolved, ambiguous, unresolved = resolve_bare_names(refs, repo_root)

    print(f"\nFound {len(refs)} unique .py references.")
    if ambiguous:
        print(f"\n{len(ambiguous)} bare filenames matched multiple existing files "
              f"(used first match, review manually):")
        for ref, matches in ambiguous:
            print(f"  ? {ref} -> could be: {', '.join(matches)}")
    if unresolved:
        print(f"\n{len(unresolved)} bare filenames had no existing match anywhere in the repo "
              f"(placed at repo root in stub tree, needs manual placement):")
        for ref in unresolved:
            print(f"  ! {ref}")

    if args.report_only:
        return

    out_root.mkdir(parents=True, exist_ok=True)
    created, already_in_output, existed_in_repo = [], [], []

    for ref, relpath in sorted(resolved.items()):
        real_target = repo_root / relpath
        out_target = out_root / relpath
        note_sources = ", ".join(sorted(all_refs[ref]))

        if out_target.exists():
            already_in_output.append(relpath)
            continue

        out_target.parent.mkdir(parents=True, exist_ok=True)
        exists_in_repo = real_target.exists()
        if exists_in_repo:
            existed_in_repo.append(relpath)
            note = f"already exists in real repo; referenced in {note_sources}"
        else:
            note = f"missing from real repo; referenced in {note_sources}"
        out_target.write_text(f'"""Stub. {note}"""\n')
        created.append(relpath)

    print(f"\nRepo root (read-only): {repo_root}")
    print(f"Full stub mirror dir:  {out_root}")
    print(f"Wrote {len(created)} stub files (full mirror of every reference found).")
    print(f"  of which {len(existed_in_repo)} already exist for real in the repo "
          f"(stub copy still written to the mirror for a complete tree)")
    print(f"  and {len(created) - len(existed_in_repo)} are genuinely missing from the repo")
    if already_in_output:
        print(f"Skipped {len(already_in_output)} (already present in output dir from a prior run).")


if __name__ == "__main__":
    main()

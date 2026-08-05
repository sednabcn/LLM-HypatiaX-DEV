#!/usr/bin/env python3
"""
find_py_deps.py — find every .py/.sh/.yml file referenced from .github/workflows
and run_all.sh, then recursively trace each one's local (in-repo) Python
import dependencies.

Usage:
    python3 find_py_deps.py [REPO_ROOT]
    python3 find_py_deps.py [REPO_ROOT] --workflow ci_pipeline_public.yml
    python3 find_py_deps.py [REPO_ROOT] --workflow ci_pipeline_public.yml --no-run-all
    python3 find_py_deps.py [REPO_ROOT] --copy-deps-to ~/Downloads/repo-deps

REPO_ROOT defaults to the current directory. The repo must contain:
    REPO_ROOT/.github/workflows/*   (yml/yaml/or any text files)
    REPO_ROOT/run_all.sh            (optional — skipped if absent)

Options:
    -w, --workflow NAME   Only scan this one workflow file (matched by
                           exact filename, e.g. ci_pipeline_public.yml,
                           or by filename without extension). Without
                           this flag, every file under .github/workflows/
                           is scanned.
    --no-run-all           Skip scanning run_all.sh entirely. By default,
                           run_all.sh is scanned ONLY if the selected
                           workflow(s) actually reference it (e.g.
                           `bash run_all.sh`) — it is not an unconditional
                           independent entry point. Any child workflow
                           .yml files it dispatches (and grandchildren,
                           etc.) are followed recursively.
    -o, --output PATH      Where to write the JSON report
                           (default: ./py_deps_report.json)
    -x, --exclude SUBSTR   Extra path substring to exclude (repeatable),
                           on top of the built-in DEFAULT_EXCLUDES list
                           defined near the top of this file (currently:
                           workflows_simplify, run_all_checkpoint.sh,
                           paper, audit). Edit DEFAULT_EXCLUDES directly
                           for exclusions you always want.
    --copy-deps-to [DEST]  After building the report, copy every file in
                           'all_files_touched' into DEST (preserving
                           relative paths) plus the JSON report itself.
                           DEST defaults to $HOME/Downloads/repo-deps
                           if omitted. Replaces the need for a separate
                           companion script.
    --diff-dir SUBDIR       Split every real .py/.sh/.yml/.yaml file under
                           SUBDIR (relative to REPO_ROOT, e.g. 'core') into
                           IN (reachable from a scanned entry point) vs OUT
                           (present on disk but never reached). Files that
                           match an active exclude pattern are reported
                           separately as EXCLUDED rather than counted as
                           OUT. Repeatable for multiple subdirectories.
                           Printed to stdout and written into the JSON
                           report under "diffs".

Resolution notes:
    - A bare "X.yml" referenced via `gh workflow run X.yml`, `uses: X.yml`,
      or `workflow_call` from *within* a file already under
      .github/workflows/ is resolved directly against
      .github/workflows/X.yml when present there, even if a same-named
      file happened to exist elsewhere in the repo. GitHub Actions can
      only ever dispatch workflows that live in .github/workflows/, so
      this is not a guess — any other location is structurally
      unreachable via `gh workflow run` / `uses:` regardless of name
      collisions. All other basename/
      suffix matches still require uniqueness and are flagged as
      ambiguous otherwise.

Output:
    - Prints a human-readable report to stdout, including the list of
      actual subdirectories (relative to REPO_ROOT) that contain at
      least one touched file.
    - Writes a JSON report (entry points -> sorted list of every local
      file they transitively import/reference, the combined union of all
      files touched, and the combined union of all directories touched).

Notes / limitations:
    - "Local" imports are those that resolve to a .py file or package
      inside REPO_ROOT. Anything resolving to an installed / stdlib
      package is ignored.
    - Only static `import x`, `import x.y as z`, `from x import y`,
      and relative `from . import y` / `from .x import y` forms are
      detected (via the ast module). Dynamic imports (importlib,
      __import__, exec) are not detected.
    - Path references in workflow/shell files are found via a regex
      that matches typical file-path tokens ending in .py/.sh/.yml/.yaml.
    - Truncated fragments from f-strings/concatenations (e.g. _v2.py,
      v3c.py) are matched by suffix against real repo filenames at
      path-component boundaries. Ambiguous matches are flagged, not guessed.
"""

import argparse
import ast
import functools
import json
import os
import re
import shlex
import shutil
import sys
from pathlib import Path

try:
    import yaml
    _HAVE_YAML = True
except ImportError:
    _HAVE_YAML = False

# Match path-like tokens ending in .py, .sh, .yml, or .yaml.
# Requires at least one word character before the dot so bare ".py" is skipped.
PATH_RE = re.compile(r'[A-Za-z0-9_./\-]+\.(?:py|sh|yml|yaml)\b')

# Always-excluded path substrings (case-insensitive), regardless of CLI flags.
# These are known-decoy / off-tree files & dirs for the ci_pipeline_public.yml
# tree: unrelated variant directories and unrelated helper scripts that get
# referenced somewhere in the repo but are not actually part of this pipeline.
# Add more here any time a new decoy shows up — no need to pass -x every run.
# Use -x/--exclude on the CLI to add MORE exclusions on top of this list for
# a specific run; it is not a replacement for it.
DEFAULT_EXCLUDES: list[str] = [
    "workflows_simplify",
    "run_all_checkpoint.sh",
    "paper",
    "audit",
]


# ---------------------------------------------------------------------------
# Index of all real files in the repo (built once, used by resolve_reference)
# ---------------------------------------------------------------------------
_repo_file_index: dict[str, list[Path]] = {}   # stem_and_suffix -> [Path, ...]
_repo_root_cached: Path | None = None


def _build_index(repo_root: Path) -> None:
    global _repo_file_index, _repo_root_cached
    if _repo_root_cached == repo_root:
        return
    _repo_file_index = {}
    _repo_root_cached = repo_root
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in (".py", ".sh", ".yml", ".yaml"):
            continue
        key = p.name  # e.g. "run_all.sh"
        _repo_file_index.setdefault(key, []).append(p.resolve())


def find_py_references(text_file: Path) -> set[str]:
    """Return raw path-like strings ending in .py/.sh/.yml/.yaml found in a text file.

    This is the OLD whole-file, comment-blind method. It is kept only as a
    last-resort fallback (e.g. a workflow file that fails to parse as YAML).
    Prefer scan_source_file() everywhere — it uses structured YAML parsing
    for workflows and comment-stripped, invocation-aware parsing for shell
    scripts, so it doesn't pick up stray filenames sitting in comments,
    unrelated jobs, echo/log strings, or other non-invocation text.
    """
    try:
        content = text_file.read_text(errors="ignore")
    except OSError:
        return set()
    return set(PATH_RE.findall(content))


# ---------------------------------------------------------------------------
# Structured, invocation-aware reference extraction
# ---------------------------------------------------------------------------

_SCRIPT_TOKEN_RE = re.compile(r'[A-Za-z0-9_./\-]+\.(?:py|sh)\b')
_YAML_TOKEN_RE = re.compile(r'[A-Za-z0-9_./\-]+\.ya?ml\b')


def strip_shell_comments(text: str) -> str:
    """Remove full-line and inline '#' comments from shell script text.

    Heuristic: a '#' starts a comment unless it's inside single or double
    quotes. This is not a full shell lexer, but it correctly handles the
    common cases (quoted strings, simple inline comments) that matter for
    not mistaking a commented-out invocation for a real one.
    """
    out_lines = []
    for line in text.splitlines():
        result = []
        in_single = in_double = False
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "'" and not in_double:
                in_single = not in_single
                result.append(ch)
            elif ch == '"' and not in_single:
                in_double = not in_double
                result.append(ch)
            elif ch == "#" and not in_single and not in_double:
                break  # rest of line is a comment
            else:
                result.append(ch)
            i += 1
        out_lines.append("".join(result))
    return "\n".join(out_lines)


# Keywords that, as the first token of a shell statement, mean "the next
# token is a script being executed" — i.e. this is a genuine invocation,
# not just a filename appearing in unrelated text (echo, a variable value,
# a log message, an argument to some other command, etc).
_INVOKE_FIRST_WORDS = {"python", "python3", "bash", "sh", "source", "."}


def extract_shell_invocations(text: str) -> set[str]:
    """Extract .py/.sh references from shell script text, but ONLY from
    statements that are actual command invocations — not from comments,
    echo/log strings, variable assignments, or any other incidental text.

    Recognizes:
      - `python3 scripts/foo.py`, `bash scripts/foo.sh`, `source foo.sh`,
        `. foo.sh` (POSIX dot-source)
      - a bare relative/path invocation: `./scripts/foo.sh args...`
      - `gh workflow run child.yml`, `gh workflow run child.yml --ref main`
    Statements are split on `;`, `&&`, `||`, `|`, and newlines, so multiple
    commands per line are each considered independently.
    """
    refs: set[str] = set()
    clean = strip_shell_comments(text)
    for raw_line in clean.splitlines():
        for stmt in re.split(r'&&|\|\||;|\|', raw_line):
            stmt = stmt.strip()
            if not stmt:
                continue
            try:
                tokens = shlex.split(stmt, posix=True)
            except ValueError:
                tokens = stmt.split()
            if not tokens:
                continue
            first = tokens[0]

            # gh workflow run X.yml [--ref ...]
            if first == "gh" and "workflow" in tokens and "run" in tokens:
                idx = tokens.index("run")
                if idx + 1 < len(tokens):
                    m = _YAML_TOKEN_RE.search(tokens[idx + 1])
                    if m:
                        refs.add(m.group(0))
                continue

            # python3 foo.py / bash foo.sh / source foo.sh / . foo.sh
            if first in _INVOKE_FIRST_WORDS and len(tokens) > 1:
                m = _SCRIPT_TOKEN_RE.search(tokens[1])
                if m:
                    refs.add(m.group(0))
                continue

            # bare invocation: ./scripts/foo.sh, scripts/foo.py (as argv[0])
            if first.startswith("./") or first.startswith("/") or "/" in first:
                m = _SCRIPT_TOKEN_RE.search(first)
                if m:
                    refs.add(m.group(0))

    return refs


def _walk_yaml_strings_for_run_and_uses(node, refs: set[str]) -> None:
    """Walk a parsed YAML structure, extracting refs ONLY from 'run:' step
    bodies (via extract_shell_invocations) and 'uses:' fields (job-level
    reusable-workflow calls, or step-level local action/workflow refs) —
    never from 'name:', 'env:', comments (already gone after yaml.safe_load),
    or any other field. This is what keeps unrelated jobs/steps/metadata in
    the SAME workflow file from leaking in as false positives.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "run" and isinstance(value, str):
                refs |= extract_shell_invocations(value)
            elif key == "uses" and isinstance(value, str):
                if value.startswith("./") or value.startswith(".github/"):
                    m = _YAML_TOKEN_RE.search(value)
                    if m:
                        refs.add(m.group(0))
            else:
                _walk_yaml_strings_for_run_and_uses(value, refs)
    elif isinstance(node, list):
        for item in node:
            _walk_yaml_strings_for_run_and_uses(item, refs)


def extract_workflow_refs_structured(wf_path: Path) -> set[str] | None:
    """Parse a workflow YAML file structurally and extract .py/.sh/.yml/.yaml
    references ONLY from real 'run:' command bodies and 'uses:' fields —
    i.e. only from what GitHub Actions would actually execute or dispatch.

    Returns None if the file can't be parsed as YAML (caller should fall
    back to find_py_references() with a warning) or if PyYAML isn't
    installed.
    """
    if not _HAVE_YAML:
        return None
    try:
        text = wf_path.read_text(errors="ignore")
        data = yaml.safe_load(text)
    except Exception:
        return None
    if data is None:
        return set()
    refs: set[str] = set()
    _walk_yaml_strings_for_run_and_uses(data, refs)
    return refs


def scan_source_file(path: Path) -> set[str]:
    """Extract .py/.sh/.yml/.yaml references from a source file using the
    most precise method available for its type:
      - .yml/.yaml -> structured YAML parse (run:/uses: fields only),
        falling back to whole-file regex ONLY if YAML parsing fails.
      - .sh        -> comment-stripped, invocation-aware shell parsing.
      - anything else -> whole-file regex (legacy fallback).
    """
    suffix = path.suffix.lower()
    if suffix in (".yml", ".yaml"):
        refs = extract_workflow_refs_structured(path)
        if refs is None:
            print(
                f"WARNING: {path} could not be parsed as YAML "
                f"(or PyYAML is unavailable) — falling back to whole-file "
                f"text scan for this file, which may pick up decoy/unrelated "
                f"references.",
                file=sys.stderr,
            )
            refs = find_py_references(path)
        return refs
    elif suffix == ".sh":
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            return set()
        return extract_shell_invocations(text)
    else:
        return find_py_references(path)


WORKFLOW_DISPATCH_RE = re.compile(
    r'(?:gh\s+workflow\s+run|uses:|workflow_call)\s*:?\s*["\']?'
    r'([A-Za-z0-9_.\-/]+\.ya?ml)\b'
)


def _github_workflows_dir(repo_root: Path) -> Path:
    return repo_root / ".github" / "workflows"


def _is_workflow_dispatch_ref(ref: str, source_file: Path | None, source_text: str | None) -> bool:
    """
    True if `ref` looks like a workflow-to-workflow dispatch (`gh workflow run
    X.yml`, `uses: X.yml`, `workflow_call`) made *from within* a file that
    itself lives under .github/workflows/. GitHub Actions can only resolve
    such references against .github/workflows/ — never against any other
    same-named directory in the repo.
    """
    if source_file is None or source_text is None:
        return False
    if ".github/workflows" not in str(source_file).replace(os.sep, "/"):
        return False
    if os.path.basename(ref) != ref:
        return False  # already has a path component, not a bare dispatch name
    return any(
        os.path.basename(m) == ref for m in WORKFLOW_DISPATCH_RE.findall(source_text)
    )


def _is_yaml_ref(ref: str) -> bool:
    return ref.lower().endswith((".yml", ".yaml"))


def _is_py_sh_ref(ref: str) -> bool:
    return ref.lower().endswith((".py", ".sh"))


def _restrict_to_live_workflows(paths: list[Path], repo_root: Path) -> list[Path]:
    """
    For .yml/.yaml candidates, .github/workflows/ is the ONLY valid location
    a live GitHub Actions workflow can come from. If at least one candidate
    already lives there, every candidate outside it is a decoy and is
    dropped outright — not merely deprioritized.

    If NONE of the candidates live under .github/workflows/, this ref was
    never a workflow name to begin with (e.g. a plain config file like
    config/repro.yaml or environment.yml) — the full candidate list is
    returned unchanged so ordinary basename/suffix resolution still applies.
    """
    live_dir = _github_workflows_dir(repo_root)
    live = [p for p in paths if p.is_relative_to(live_dir)]
    return live if live else paths


def _is_preferred_py_sh_dir(path: Path, repo_root: Path) -> bool:
    """
    True if `path` lives under one of the known "real" locations for
    .py/.sh scripts in this repo layout: any directory with "hypatiax" in
    its name (case-insensitive, anywhere in the path — e.g. hypatiax/,
    hypatiax/core/), .github/scripts/, or a top-level scripts/ dir.
    """
    try:
        rel_parts = [p.lower() for p in path.relative_to(repo_root).parts]
    except ValueError:
        return False
    if any("hypatiax" in part for part in rel_parts):
        return True
    if rel_parts[:2] == [".github", "scripts"]:
        return True
    if rel_parts[:1] == ["scripts"]:
        return True
    return False


def _restrict_to_preferred_py_sh_dirs(paths: list[Path], repo_root: Path) -> list[Path]:
    """
    For .py/.sh candidates, prefer matches living under hypatiax*/,
    .github/scripts/, or scripts/ over matches elsewhere in the repo (e.g.
    stray same-named scripts under an unrelated directory). If at least one
    candidate lives in a preferred dir, every candidate outside those dirs
    is dropped — mirroring how .yml/.yaml resolution restricts to
    .github/workflows/ only (see _restrict_to_live_workflows()).

    If NONE of the candidates live in a preferred dir, the full candidate
    list is returned unchanged so ordinary ambiguity handling still applies.
    """
    preferred = [p for p in paths if _is_preferred_py_sh_dir(p, repo_root)]
    return preferred if preferred else paths


def resolve_reference(
    ref: str, repo_root: Path, source_file: Path | None = None, source_text: str | None = None
) -> Path | None:
    """
    Try to resolve a raw referenced path string to an actual file in the repo.

    Resolution order:
      0. GitHub Actions workflow-dispatch shortcut: if `ref` is a bare
         "X.yml" dispatched via `gh workflow run` / `uses:` / `workflow_call`
         from a file already inside .github/workflows/, resolve directly
         against .github/workflows/X.yml when it exists there.
      1. Direct path: repo_root / ref  (or with leading ./ stripped) — an
         explicit path is honoured as-written even outside .github/workflows/,
         since it's a literal reference rather than a basename guess.
      2. Exact basename glob anywhere in the repo (unique match only). For
         .yml/.yaml refs, candidates are first restricted to
         .github/workflows/ only — see _restrict_to_live_workflows().
      3. Suffix/boundary match for truncated fragments (unique match only).
         Same .github/workflows/-only restriction applies for .yml/.yaml.

    Returns the resolved absolute Path, or None if unresolvable / ambiguous.
    """
    _build_index(repo_root)

    # ── 0. GitHub Actions workflow-dispatch shortcut ──────────────────────────
    if _is_workflow_dispatch_ref(ref, source_file, source_text):
        candidate = _github_workflows_dir(repo_root) / ref
        if candidate.is_file():
            return candidate.resolve()

    # ── 1. Direct path ────────────────────────────────────────────────────────
    for candidate in (repo_root / ref, repo_root / ref.lstrip("./")):
        if candidate.is_file():
            return candidate.resolve()

    # ── 2. Exact basename ─────────────────────────────────────────────────────
    basename = os.path.basename(ref)
    exact = _repo_file_index.get(basename, [])
    exact = [p for p in exact if p.is_file()]
    if _is_yaml_ref(ref):
        exact = _restrict_to_live_workflows(exact, repo_root)
    elif _is_py_sh_ref(ref):
        exact = _restrict_to_preferred_py_sh_dirs(exact, repo_root)
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        # Ambiguous exact basename — don't guess
        return None

    # ── 3. Suffix/boundary match for truncated fragments ─────────────────────
    # ref looks like "_benchmark_v2.py" or "v3c.py" — a literal tail of a
    # longer filename. Match against every known file where the ref appears at
    # a real boundary (start of name, or preceded by / _ - digit→letter).
    suffix_matches: list[Path] = []
    ref_norm = ref.lstrip("_-/")   # strip leading separators that may vary
    for name, paths in _repo_file_index.items():
        if not name.endswith(ref_norm):
            continue
        # Check that the match starts at a boundary in the real filename
        prefix = name[: len(name) - len(ref_norm)]
        if prefix == "" or prefix[-1] in ("_", "-", "/"):
            suffix_matches.extend(paths)

    suffix_matches = list({p for p in suffix_matches if p.is_file()})
    if _is_yaml_ref(ref):
        suffix_matches = _restrict_to_live_workflows(suffix_matches, repo_root)
    elif _is_py_sh_ref(ref):
        suffix_matches = _restrict_to_preferred_py_sh_dirs(suffix_matches, repo_root)
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    # >1 or 0 — ambiguous or genuinely missing; caller will report it
    return None


def resolve_reference_with_reason(
    ref: str, repo_root: Path, source_file: Path | None = None, source_text: str | None = None
) -> tuple[Path | None, str]:
    """
    Like resolve_reference, but also returns a human-readable reason string
    for use in the 'unresolved' section of the report.
    """
    _build_index(repo_root)

    if _is_workflow_dispatch_ref(ref, source_file, source_text):
        candidate = _github_workflows_dir(repo_root) / ref
        if candidate.is_file():
            return candidate.resolve(), "gh-workflow-dispatch"

    for candidate in (repo_root / ref, repo_root / ref.lstrip("./")):
        if candidate.is_file():
            return candidate.resolve(), "direct"

    basename = os.path.basename(ref)
    exact = [p for p in _repo_file_index.get(basename, []) if p.is_file()]
    if _is_yaml_ref(ref):
        exact = _restrict_to_live_workflows(exact, repo_root)
    elif _is_py_sh_ref(ref):
        exact = _restrict_to_preferred_py_sh_dirs(exact, repo_root)
    if len(exact) == 1:
        return exact[0], "basename"
    if len(exact) > 1:
        rel = sorted(str(p.relative_to(repo_root)) for p in exact)
        return None, f"ambiguous basename — candidates: {rel}"

    ref_norm = ref.lstrip("_-/")
    suffix_matches: list[Path] = []
    for name, paths in _repo_file_index.items():
        if not name.endswith(ref_norm):
            continue
        prefix = name[: len(name) - len(ref_norm)]
        if prefix == "" or prefix[-1] in ("_", "-", "/"):
            suffix_matches.extend(paths)
    suffix_matches = list({p for p in suffix_matches if p.is_file()})
    if _is_yaml_ref(ref):
        suffix_matches = _restrict_to_live_workflows(suffix_matches, repo_root)
    elif _is_py_sh_ref(ref):
        suffix_matches = _restrict_to_preferred_py_sh_dirs(suffix_matches, repo_root)
    if len(suffix_matches) == 1:
        return suffix_matches[0], "suffix-match"
    if len(suffix_matches) > 1:
        rel = sorted(str(p.relative_to(repo_root)) for p in suffix_matches)
        return None, f"ambiguous suffix-match — candidates: {rel}"

    return None, "not found in repo"


def collect_entry_points(
    repo_root: Path,
    workflow_filter: str | None = None,
    include_run_all: bool = True,
) -> dict[str, set[str]]:
    """Scan .github/workflows/* and run_all.sh for .py/.sh/.yml references.

    If workflow_filter is given, only that workflow file (matched by exact
    filename or filename with no extension) is scanned instead of every
    file under .github/workflows/.

    run_all.sh is NOT treated as an unconditional, independent entry point.
    It's only pulled in as a *resource* of an already-scanned workflow file:
    run_all.sh is included only if at least one scanned workflow's text
    mentions "run_all.sh" (e.g. `bash run_all.sh`, `./run_all.sh`). All of
    its .py/.sh/.yml/.yaml references are extracted (including any child
    workflow .yml files it dispatches — those are picked up and, in turn,
    recursively scanned by the caller so the full ci_pipeline_public.yml ->
    run_all.sh -> child workflow tree is captured). Pass include_run_all=False
    (--no-run-all) to skip run_all.sh entirely, regardless of whether it's
    referenced.
    """
    sources: dict[str, set[str]] = {}
    scanned_workflow_texts: list[str] = []

    workflows_dir = repo_root / ".github" / "workflows"
    if workflows_dir.is_dir():
        for wf in sorted(workflows_dir.iterdir()):
            if not wf.is_file():
                continue
            if workflow_filter and wf.name != workflow_filter and wf.stem != workflow_filter:
                continue
            refs = scan_source_file(wf)
            if refs:
                sources[str(wf.relative_to(repo_root))] = refs
            try:
                scanned_workflow_texts.append(wf.read_text(errors="ignore"))
            except OSError:
                pass

    if include_run_all:
        run_all = repo_root / "run_all.sh"
        if run_all.is_file():
            referenced = any("run_all.sh" in t for t in scanned_workflow_texts)
            if referenced:
                refs = scan_source_file(run_all)
                if refs:
                    sources[str(run_all.relative_to(repo_root))] = refs
            # else: run_all.sh not mentioned by the scanned workflow(s) at
            # all — skip it entirely, it's not a resource of this tree.

    return sources


def _all_repo_py_files(repo_root: Path) -> list[Path]:
    """All .py files in the repo, drawn from the index built by _build_index."""
    _build_index(repo_root)
    return [p for paths in _repo_file_index.values() for p in paths if p.suffix == ".py"]


def _match_package_dir_fallback(parts: tuple, repo_root: Path) -> Path | None:
    """Same idea as _match_module_fallback, but resolves a dotted path to a
    *package directory* (one containing __init__.py) rather than a file.
    Used to locate the directory 'from x.y import z' should look inside for
    'z' as a submodule, when x.y itself didn't resolve on a literal path."""
    all_py = _all_repo_py_files(repo_root)
    for k in range(len(parts), 0, -1):
        tail = parts[len(parts) - k:]
        candidates: list[Path] = []
        for p in all_py:
            if p.name != "__init__.py":
                continue
            dir_parts = p.relative_to(repo_root).parts[:-1]
            if len(dir_parts) >= len(tail) and dir_parts[-len(tail):] == tuple(tail):
                candidates.append(p.parent)
        if not candidates:
            continue
        candidates = list({c.resolve() for c in candidates})
        pool = _restrict_to_preferred_py_sh_dirs(candidates, repo_root)
        if not pool:
            pool = candidates
        if len(pool) == 1:
            return pool[0]
        return None
    return None


@functools.lru_cache(maxsize=8)
def _local_top_level_names(repo_root: Path) -> frozenset:
    """Names (lowercased) of direct children of repo_root that are either a
    directory or a .py file's stem — i.e. plausible top-level local package
    names. Used to decide whether a failed import is worth reporting as a
    likely-local miss vs. silently assumed to be a genuine external/stdlib
    package (numpy, os, requests, etc.)."""
    names: set[str] = set()
    try:
        for child in repo_root.iterdir():
            if child.is_dir():
                names.add(child.name.lower())
            elif child.is_file() and child.suffix == ".py":
                names.add(child.stem.lower())
    except OSError:
        pass
    return frozenset(names)


def module_to_file_with_reason(module: str, repo_root: Path) -> tuple[Path | None, str]:
    """Like module_to_file, but also returns a human-readable reason —
    'resolved', 'ambiguous: ...', or 'not found' — for diagnostics."""
    parts = module.split(".")
    base = repo_root.joinpath(*parts)
    candidate_file = base.with_suffix(".py")
    if candidate_file.is_file():
        return candidate_file.resolve(), "resolved"
    candidate_pkg = base / "__init__.py"
    if candidate_pkg.is_file():
        return candidate_pkg.resolve(), "resolved"
    return _match_module_fallback_with_reason(tuple(parts), repo_root)


def _match_module_fallback_with_reason(parts: tuple, repo_root: Path) -> tuple[Path | None, str]:
    all_py = _all_repo_py_files(repo_root)
    for k in range(len(parts), 0, -1):
        tail = parts[len(parts) - k:]
        candidates: list[Path] = []
        for p in all_py:
            rel_parts = p.relative_to(repo_root).parts
            if p.name == "__init__.py":
                dir_parts = rel_parts[:-1]
                if len(dir_parts) >= len(tail) and dir_parts[-len(tail):] == tuple(tail):
                    candidates.append(p)
            else:
                if p.stem != tail[-1]:
                    continue
                head = tail[:-1]
                dir_parts = rel_parts[:-1]
                if not head or (len(dir_parts) >= len(head) and dir_parts[-len(head):] == tuple(head)):
                    candidates.append(p)
        if not candidates:
            continue
        candidates = list({p.resolve() for p in candidates})
        pool = _restrict_to_preferred_py_sh_dirs(candidates, repo_root)
        if not pool:
            pool = candidates
        if len(pool) == 1:
            return pool[0], "resolved"
        rel = sorted(str(p.relative_to(repo_root)) for p in pool)
        return None, f"ambiguous match for '{'.'.join(tail)}' — candidates: {rel}"
    return None, "not found"


def module_to_file(module: str, repo_root: Path) -> Path | None:
    """Resolve a dotted module name (e.g. 'pkg.sub.mod') to a repo-local file."""
    return module_to_file_with_reason(module, repo_root)[0]


def module_to_file_in_dir(pkg_dir: Path, name: str) -> Path | None:
    """Resolve `name` as a submodule/subpackage living inside pkg_dir."""
    candidate_file = pkg_dir / f"{name}.py"
    if candidate_file.is_file():
        return candidate_file.resolve()
    candidate_pkg = pkg_dir / name / "__init__.py"
    if candidate_pkg.is_file():
        return candidate_pkg.resolve()
    return None


def relative_import_to_file(
    py_file: Path, level: int, module: str | None, repo_root: Path
) -> Path | None:
    """Resolve a relative import ('from . import x' / 'from .mod import y')."""
    pkg_dir = py_file.parent
    for _ in range(level - 1):
        pkg_dir = pkg_dir.parent
    if module:
        target = pkg_dir.joinpath(*module.split("."))
    else:
        target = pkg_dir
    candidate_file = target.with_suffix(".py")
    if candidate_file.is_file():
        return candidate_file.resolve()
    candidate_pkg = target / "__init__.py"
    if candidate_pkg.is_file():
        return candidate_pkg.resolve()
    return None


def extract_imports(
    py_file: Path, repo_root: Path, unresolved: list | None = None
) -> set[Path]:
    """Parse a .py file's AST imports and return the set of local files they
    resolve to. If `unresolved` is passed, failed resolutions whose
    top-level name looks like it's meant to be local (matches a real
    top-level dir/file in the repo — see _local_top_level_names) are
    appended to it as {"in_file", "statement", "reason"} dicts, so silent
    misses are no longer invisible. Imports whose top-level name doesn't
    match anything local are assumed to be genuine external/stdlib
    packages and are not reported — this is a heuristic, not a real
    import-resolution check (see module docstring's Notes/limitations)."""
    deps: set[Path] = set()
    try:
        tree = ast.parse(py_file.read_text(errors="ignore"), filename=str(py_file))
    except (SyntaxError, OSError):
        return deps

    local_names = _local_top_level_names(repo_root) if unresolved is not None else frozenset()

    def _report(statement: str, reason: str) -> None:
        if unresolved is not None:
            unresolved.append({"in_file": str(py_file), "statement": statement, "reason": reason})

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolved, reason = module_to_file_with_reason(alias.name, repo_root)
                if resolved and resolved != py_file:
                    deps.add(resolved)
                elif unresolved is not None and alias.name.split(".")[0].lower() in local_names:
                    _report(f"import {alias.name}", reason)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                base = relative_import_to_file(py_file, node.level, node.module, repo_root)
                if base and base != py_file:
                    deps.add(base)
                elif unresolved is not None and base is None:
                    dots = "." * node.level
                    _report(f"from {dots}{node.module or ''} import ...", "not found (relative import)")
                pkg_dir = py_file.parent
                for _ in range(node.level - 1):
                    pkg_dir = pkg_dir.parent
                if node.module:
                    pkg_dir = pkg_dir.joinpath(*node.module.split("."))
                for alias in node.names:
                    sub = module_to_file_in_dir(pkg_dir, alias.name)
                    if sub and sub != py_file:
                        deps.add(sub)
            elif node.module:
                mod_parts = node.module.split(".")
                base, reason = module_to_file_with_reason(node.module, repo_root)
                if base and base != py_file:
                    deps.add(base)
                elif unresolved is not None and mod_parts[0].lower() in local_names:
                    names_str = ", ".join(a.name for a in node.names)
                    _report(f"from {node.module} import {names_str}", reason)
                # Figure out which real directory to look in for
                # `alias.name` as a submodule. Prefer the parent of the
                # resolved __init__.py (handles src/ layouts, sys.path
                # shims, hyphenated real dirs, etc. via the fallback
                # matching in module_to_file). Only fall back to a literal
                # repo_root/<dotted path> join if that path actually
                # exists — and if even that literal join fails, try the
                # same suffix-matching fallback used for module_to_file,
                # but for directories.
                if base and base.name == "__init__.py":
                    pkg_dir = base.parent
                else:
                    literal_dir = repo_root.joinpath(*mod_parts)
                    if (literal_dir / "__init__.py").is_file():
                        pkg_dir = literal_dir
                    else:
                        pkg_dir = _match_package_dir_fallback(tuple(mod_parts), repo_root)
                if pkg_dir:
                    for alias in node.names:
                        sub = module_to_file_in_dir(pkg_dir, alias.name)
                        if sub and sub != py_file:
                            deps.add(sub)
    return deps


def get_direct_children(
    path: Path,
    repo_root: Path,
    active_excludes: list[str],
    unresolved_refs: dict,
    unresolved_imports: list,
) -> list[Path]:
    """Return ONLY the direct (one-hop) children of `path` — never a
    transitive closure. This is the single dispatch point that makes every
    node in the dependency forest, regardless of type, behave the same way:
    a .yml/.yaml/.sh file's children are the launched/dispatched files found
    by scan_source_file(); a .py file's children are its direct AST-level
    imports (via extract_imports(), which already only walks one file's own
    import statements — no recursion happens inside it).

    Each returned child is itself later re-queued as an independent new
    root by the caller's worklist, so multi-level chains are built up by
    repeated one-hop calls rather than by any single call doing a deep walk.
    Ambiguous/unresolved references are recorded into unresolved_refs /
    unresolved_imports (both mutated in place) rather than silently dropped.
    """
    suffix = path.suffix.lower()
    children: list[Path] = []

    if suffix in (".yml", ".yaml", ".sh"):
        try:
            source_text = path.read_text(errors="ignore")
        except OSError:
            source_text = None
        refs = scan_source_file(path)
        for ref in sorted(refs):
            resolved, reason = resolve_reference_with_reason(
                ref, repo_root, source_file=path, source_text=source_text
            )
            if resolved:
                relpath = str(resolved.relative_to(repo_root))
                if _is_excluded(relpath, active_excludes):
                    continue  # excluded — treat as not-found, silently
                children.append(resolved)
            else:
                rec = unresolved_refs.setdefault(ref, {"sources": [], "reason": reason})
                rec["sources"].append(str(path.relative_to(repo_root)))
                if "ambiguous" in reason:
                    rec["reason"] = reason

    elif suffix == ".py":
        entry_unresolved: list = []
        deps = extract_imports(path, repo_root, entry_unresolved)
        for rec in entry_unresolved:
            try:
                rec["in_file"] = str(Path(rec["in_file"]).relative_to(repo_root))
            except ValueError:
                pass
            if not _is_excluded(rec["in_file"], active_excludes):
                unresolved_imports.append(rec)
        for dep in sorted(deps):
            relpath = str(dep.relative_to(repo_root))
            if _is_excluded(relpath, active_excludes):
                continue
            children.append(dep)

    return children


def _is_excluded(relpath: str, patterns: list[str]) -> bool:
    """True if relpath contains any of the exclude substrings (case-insensitive)."""
    if not patterns:
        return False
    low = relpath.lower()
    return any(p.lower() in low for p in patterns)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "repo_root", nargs="?", default=".", help="Path to repo root (default: current dir)"
    )
    parser.add_argument(
        "-o", "--output", default="py_deps_report.json", help="JSON report output path"
    )
    parser.add_argument(
        "-w", "--workflow", default=None,
        help="Only scan this workflow file (e.g. ci_pipeline_public.yml) "
             "instead of every file in .github/workflows/",
    )
    parser.add_argument(
        "--no-run-all", action="store_true",
        help="Skip scanning run_all.sh entirely. By default run_all.sh is "
             "scanned only if the selected workflow(s) actually reference it "
             "(e.g. `bash run_all.sh`) — this flag turns that off completely.",
    )
    parser.add_argument(
        "-x", "--exclude", action="append", default=[], metavar="SUBSTRING",
        help="Drop any resolved file whose repo-relative path contains this "
             "substring (case-insensitive). Repeatable. Adds ON TOP OF the "
             f"built-in DEFAULT_EXCLUDES ({', '.join(DEFAULT_EXCLUDES)}) "
             "defined near the top of this script — edit that list directly "
             "for exclusions you always want, use -x for one-off extras. "
             "Applied to entry points AND their traced dependencies, so it "
             "also removes them from all_files_touched and the copied "
             "repo-deps mirror.",
    )
    parser.add_argument(
        "--copy-deps-to", nargs="?", const="__HOME_DOWNLOADS__", default=None, metavar="DEST",
        help="After building the report, copy every file in 'all_files_touched' "
             "into DEST, preserving relative paths (a lightweight 'repo-deps' "
             "mirror of just what's actually reachable from the scanned "
             "entry points). If DEST is omitted, defaults to "
             "$HOME/Downloads/repo-deps.",
    )
    parser.add_argument(
        "--diff-dir", action="append", default=[], metavar="SUBDIR",
        help="After building the report, split every real .py/.sh/.yml/.yaml "
             "file under SUBDIR (a path relative to REPO_ROOT, e.g. 'core') "
             "into IN (present in all_files_touched) and OUT (never reached "
             "from any scanned entry point) sets, and print both. Repeatable "
             "to check multiple subdirectories in one run. Files matching "
             "an active exclude pattern are reported separately under OUT "
             "with reason 'excluded' rather than silently dropped. The "
             "split is also written into the JSON report under 'diffs'.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if not repo_root.is_dir():
        print(f"ERROR: {repo_root} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Pre-build the file index so all resolution calls share it
    _build_index(repo_root)

    sources = collect_entry_points(
        repo_root,
        workflow_filter=args.workflow,
        include_run_all=not args.no_run_all,
    )
    if not sources:
        print("No references found in .github/workflows or run_all.sh")

    # Built-in exclusions always apply; -x on the CLI adds more for this run.
    active_excludes = DEFAULT_EXCLUDES + args.exclude

    if not sources:
        entry_points_order: list[str] = []
    else:
        entry_points_order = sorted(sources.keys())

    # ------------------------------------------------------------------
    # Build the dependency FOREST: a worklist/BFS where every node —
    # workflow, shell script, or python module alike — is scanned for its
    # DIRECT children only (get_direct_children never recurses on its
    # own). Each newly-discovered child is pushed onto the worklist and
    # will, on its own turn, become an independent new root: its children
    # are whatever IT directly references/imports, nothing more.
    #
    # This deliberately does NOT build a nested "parent owns everything
    # beneath it" tree. adjacency[node] holds only node's immediate
    # children; a node that is reached from three different parents still
    # gets exactly one children-list, computed once, the same as any
    # other node — there is no ownership, only discovery.
    # ------------------------------------------------------------------
    unresolved: dict[str, dict] = {}          # raw ref string -> {sources, reason}
    unresolved_imports: list[dict] = []        # python import diagnostics
    adjacency: dict[str, list[str]] = {}       # relpath -> sorted direct children (relpaths)
    discovery_order: list[str] = []            # nodes, in the order first discovered
    visited: set[str] = set()

    worklist = list(entry_points_order)
    for root in worklist:
        visited.add(root)
        discovery_order.append(root)

    while worklist:
        node_rel = worklist.pop(0)
        node_path = repo_root / node_rel
        children_paths = get_direct_children(
            node_path, repo_root, active_excludes, unresolved, unresolved_imports
        )
        child_rels: list[str] = []
        for child_path in children_paths:
            child_rel = str(child_path.relative_to(repo_root))
            child_rels.append(child_rel)
            if child_rel not in visited:
                visited.add(child_rel)
                discovery_order.append(child_rel)
                worklist.append(child_rel)
        adjacency[node_rel] = sorted(set(child_rels))

    # Dedupe unresolved-import diagnostics (a shared module reached from
    # multiple parents only needs to be reported once per statement).
    seen_import_diags = set()
    deduped_unresolved_imports = []
    for rec in sorted(unresolved_imports, key=lambda r: (r["in_file"], r["statement"])):
        key = (rec["in_file"], rec["statement"], rec["reason"])
        if key in seen_import_diags:
            continue
        seen_import_diags.add(key)
        deduped_unresolved_imports.append(rec)

    # ------------------------------------------------------------------
    # Print the forest: one block per discovered node that has children,
    # in discovery order. Each block shows ONLY that node's direct
    # children — never a flattened/transitive list — matching the actual
    # adjacency structure above.
    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"Repo root: {repo_root}")
    print(f"Entry points scanned: {', '.join(entry_points_order) or '(none found)'}")
    print("=" * 70)

    for node_rel in discovery_order:
        children = adjacency.get(node_rel, [])
        if not children:
            continue
        print(f"\n{node_rel}")
        for i, child in enumerate(children):
            branch = "└──" if i == len(children) - 1 else "├──"
            print(f"    {branch} {child}")

    if unresolved:
        print(f"\nUnresolved references ({len(unresolved)}) — mentioned but no matching file found in repo:")
        for ref, info in sorted(unresolved.items()):
            sources_str = ", ".join(sorted(set(info["sources"])))
            print(f"  - {ref}")
            print(f"      in:     {sources_str}")
            print(f"      reason: {info['reason']}")

    if deduped_unresolved_imports:
        print(
            f"\nUnresolved local-looking Python imports ({len(deduped_unresolved_imports)}) "
            f"— top-level name matches something in the repo, but the exact import "
            f"couldn't be resolved to a file:"
        )
        for rec in deduped_unresolved_imports:
            print(f"  - {rec['statement']}")
            print(f"      in:     {rec['in_file']}")
            print(f"      reason: {rec['reason']}")

    all_files = sorted(discovery_order)
    all_dirs = sorted(
        {
            str(Path(f).parent) if Path(f).parent != Path(".") else "."
            for f in all_files
        }
    )

    print("\n" + "=" * 70)
    print(f"Total distinct files touched: {len(all_files)}")
    print(f"Total distinct subdirectories touched: {len(all_dirs)}")
    for d in all_dirs:
        print(f"    - {d}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # --diff-dir: for each requested subdirectory, split every real
    # .py/.sh/.yml/.yaml file on disk under it into IN (reached from some
    # entry point) vs OUT (never reached). Files that were excluded via
    # DEFAULT_EXCLUDES/-x are reported separately — they're "out" for a
    # different reason (deliberately dropped, not simply unreferenced)
    # and lumping them in with genuinely-orphaned files would be misleading.
    # ------------------------------------------------------------------
    diffs: dict = {}
    touched_set = set(all_files)
    for subdir in args.diff_dir:
        sub_path = (repo_root / subdir).resolve()
        if not sub_path.is_dir():
            print(f"\nWARNING: --diff-dir '{subdir}' is not a directory under {repo_root}; skipping", file=sys.stderr)
            continue
        real_files = sorted(
            str(p.relative_to(repo_root))
            for p in sub_path.rglob("*")
            if p.is_file()
            and p.suffix in (".py", ".sh", ".yml", ".yaml")
            and "__pycache__" not in p.parts
        )
        in_set = [f for f in real_files if f in touched_set]
        excluded_set = [
            f for f in real_files
            if f not in touched_set and _is_excluded(f, active_excludes)
        ]
        out_set = [
            f for f in real_files
            if f not in touched_set and f not in excluded_set
        ]
        diffs[subdir] = {"in": in_set, "out": out_set, "excluded": excluded_set}

        print("\n" + "=" * 70)
        print(f"--diff-dir {subdir}")
        print("=" * 70)
        print(f"IN  ({len(in_set)}) — reachable from a scanned entry point:")
        for f in in_set:
            print(f"    {f}")
        print(f"\nOUT ({len(out_set)}) — present on disk, never reached:")
        for f in out_set:
            print(f"    {f}")
        if excluded_set:
            print(f"\nEXCLUDED ({len(excluded_set)}) — matched an active exclude pattern:")
            for f in excluded_set:
                print(f"    {f}")

    report: dict = {
        "repo_root": str(repo_root),
        "entry_points": entry_points_order,
        "dependency_forest": {node: adjacency[node] for node in discovery_order},
        "unresolved_references": {
            ref: {"sources": sorted(set(info["sources"])), "reason": info["reason"]}
            for ref, info in sorted(unresolved.items())
        },
        "unresolved_local_imports": deduped_unresolved_imports,
        "all_files_touched": all_files,
        "all_dirs_touched": all_dirs,
        "diffs": diffs,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nJSON report written to: {out_path.resolve()}")

    if args.copy_deps_to is not None:
        dest = args.copy_deps_to
        if dest == "__HOME_DOWNLOADS__":
            dest_root = Path.home() / "Downloads" / "repo-deps"
        else:
            dest_root = Path(dest).expanduser()
        dest_root.mkdir(parents=True, exist_ok=True)

        copied = 0
        missing = []
        for relpath in report["all_files_touched"]:
            src = repo_root / relpath
            dst = dest_root / relpath
            if not src.is_file():
                missing.append(relpath)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        shutil.copy2(out_path, dest_root / out_path.name)

        print(f"\nrepo-deps mirror written to: {dest_root}")
        print(f"  Copied: {copied} file(s)")
        if missing:
            print(f"  Missing (in report but not found on disk): {len(missing)}")
            for m in missing:
                print(f"    - {m}")

    print("Done")


if __name__ == "__main__":
    main()

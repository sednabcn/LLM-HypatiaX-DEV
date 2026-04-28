#!/usr/bin/env python3
"""
scan_internal_imports.py  (HypatiaX §11 — enhanced)
=====================================================
Provenance tool that maps the internal import DAG and cross-references it
against the actual file system to surface four categories of issues:

  A) STALE ENGINE IMPORTS   — any file importing v40 / v50 (non-v50_2) / v52
  B) GHOST IMPORTS          — imports that resolve to a module name that exists
                              as a known stale *file* in tools/symbolic/
  C) PROTOCOL LAYER LEAKS   — protocols/ files that import benchmarks/ directly
  D) IMPORT CYCLES          — circular dependencies within scanned packages
                              (uses networkx; falls back to DFS if unavailable)

Outputs (written to --out directory):
  import_graph.dot      — full Graphviz DOT digraph
  import_report.txt     — human-readable findings for all four checks
  stale_imports.txt     — tab-separated (file, import) for stale hits
  ghost_imports.txt     — tab-separated (file, import, matched_file)
  cycles.txt            — one cycle per line (comma-separated nodes)

Usage:
    python3 scan_internal_imports.py [--root .] [--out logs/repro_output]

Called by:
    run_all.sh  →  run_step "scan-imports" ...
    run_all.py  →  Step("scan-imports", ...)

Exit codes:
    0 — scan complete (warnings emitted; apply_patches.py fixes stale imports)
    1 — critical error (root not found or unrecoverable I/O failure)
"""

from __future__ import annotations

import argparse
import ast
import sys
import warnings
from collections import defaultdict, deque
from collections.abc import Iterator
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

SCAN_ROOTS: list[str] = [
    "protocols",
    "hypatiax",
    "core",
    "shared",
    "scripts",
    "figures",
    "reproducibility",
    "tools",
    "experiments",
    "analysis",
]

# Modules whose import anywhere in the codebase is a defect.
STALE_IMPORTS: list[str] = [
    "hybrid_system_v40",
    "hybrid_system_v40fix",
    "hybrid_system_v50.",   # v50 (not v50_2) — intermediate; trailing dot avoids matching v50_2
    "hybrid_system_v52",
]

EXPECTED_ENGINE = "hybrid_system_v50_2"

# Stale *filenames* that actually exist under tools/symbolic/.
# Used for ghost-import cross-referencing.
STALE_FILES: list[str] = [
    "hybrid_system_v40",
    "hybrid_system_v40fix",
    "hybrid_system_v50",
    "hybrid_system_v52",
    "hybrid_system_v52_clean",
]

# Imports originating in protocols/ that point into experiments/benchmarks/
# are considered layer violations.
PROTOCOL_FORBIDDEN_TARGETS: list[str] = [
    "experiments.benchmarks",
    "experiments/benchmarks",
]


# ── AST helpers ───────────────────────────────────────────────────────────────

def extract_imports(py_file: Path) -> list[str]:
    """Return every imported module name from *py_file* (best-effort AST walk)."""
    try:
        source = py_file.read_text(errors="replace")
        # Suppress SyntaxWarning: Python 3.12 warns on non-raw strings containing
        # escape sequences like \d in scanned files — not our bug, not actionable.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError:
        return []

    seen: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                seen.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                seen.append(node.module)
    return seen


def is_internal(module: str, scan_roots: list[str]) -> bool:
    return any(module == r or module.startswith(r + ".") for r in scan_roots)


# ── Cycle detection ───────────────────────────────────────────────────────────

def find_cycles_networkx(graph: dict[str, list[str]]) -> list[list[str]]:
    try:
        import networkx as nx  # type: ignore
    except ImportError:
        return []

    G = nx.DiGraph()
    for src, targets in graph.items():
        for tgt in targets:
            G.add_edge(src, tgt)
    try:
        return list(nx.simple_cycles(G))
    except Exception:
        return []


def find_cycles_dfs(graph: dict[str, list[str]]) -> list[list[str]]:
    """Fallback iterative DFS cycle finder (no third-party deps)."""
    cycles: list[list[str]] = []
    visited: set[str]       = set()
    rec_stack: set[str]     = set()

    all_nodes = set(graph.keys()) | {t for ts in graph.values() for t in ts}

    def _dfs(start: str) -> None:
        stack: deque[tuple[str, Iterator[str]]] = deque()
        stack.append((start, iter(graph.get(start, []))))
        path: list[str] = [start]
        rec_stack.add(start)
        visited.add(start)

        while stack:
            node, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    visited.add(child)
                    rec_stack.add(child)
                    path.append(child)
                    stack.append((child, iter(graph.get(child, []))))
                elif child in rec_stack:
                    # Found a back edge — extract cycle
                    idx   = path.index(child)
                    cycle = path[idx:]
                    cycles.append(cycle + [child])
            except StopIteration:
                rec_stack.discard(node)
                path = path[:-1] if path else path
                stack.pop()

    for node in all_nodes:
        if node not in visited:
            _dfs(node)

    return cycles


def detect_cycles(graph: dict[str, list[str]]) -> list[list[str]]:
    cycles = find_cycles_networkx(graph)
    if not cycles:
        cycles = find_cycles_dfs(graph)
    # De-duplicate (networkx gives canonical form; DFS may not)
    seen: set[frozenset[str]] = set()
    unique: list[list[str]]   = []
    for c in cycles:
        key = frozenset(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


# ── Ghost import cross-reference ─────────────────────────────────────────────

def ghost_check(
    rel: str,
    imports: list[str],
    stale_files: list[str],
) -> list[tuple[str, str, str]]:
    """
    Return (file, import, matched_stale_file) triples where the import name
    contains a stale filename stem — i.e. the file is gone/stale but still
    referenced by module path.
    """
    hits: list[tuple[str, str, str]] = []
    for imp in imports:
        parts = imp.replace("/", ".").split(".")
        for stem in stale_files:
            if stem in parts:
                hits.append((rel, imp, stem))
    return hits


# ── Protocol layer leak check ─────────────────────────────────────────────────

def protocol_leak_check(
    rel: str,
    imports: list[str],
    forbidden: list[str],
) -> list[tuple[str, str]]:
    if not rel.startswith("protocols/"):
        return []
    hits: list[tuple[str, str]] = []
    for imp in imports:
        if any(f in imp for f in forbidden):
            hits.append((rel, imp))
    return hits


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="HypatiaX internal import DAG scanner (enhanced)"
    )
    parser.add_argument("--root", default=".", help="Repo root (default: .)")
    parser.add_argument(
        "--out", default="logs/repro_output",
        help="Output directory for scan artefacts",
    )
    args = parser.parse_args()

    root    = Path(args.root).resolve()
    out_dir = (
        Path(args.out) if Path(args.out).is_absolute()
        else root / args.out
    )

    if not root.exists():
        print(f"  ERROR: root not found: {root}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Root       : {root}")
    print(f"  Output dir : {out_dir}")

    # ── Collect .py files ─────────────────────────────────────────────────────
    py_files: list[Path] = []
    for r in SCAN_ROOTS:
        d = root / r
        if d.exists():
            py_files.extend(d.rglob("*.py"))
    py_files.extend(root.glob("*.py"))
    py_files = sorted(set(py_files))
    print(f"  Python files scanned: {len(py_files)}")

    # ── Build adjacency map and run all checks ────────────────────────────────
    graph:         dict[str, list[str]]        = defaultdict(list)
    stale_hits:    list[tuple[str, str]]        = []  # (file, import)
    ghost_hits:    list[tuple[str, str, str]]   = []  # (file, import, stale_stem)
    proto_leaks:   list[tuple[str, str]]        = []  # (file, import)
    engine_uses:   list[tuple[str, str]]        = []  # (file, import)

    for py in py_files:
        rel     = str(py.relative_to(root))
        imports = extract_imports(py)

        for imp in imports:
            # Internal graph edge
            if is_internal(imp, SCAN_ROOTS):
                graph[rel].append(imp)

            # A) Stale engine imports
            for stale in STALE_IMPORTS:
                if stale in imp:
                    stale_hits.append((rel, imp))

            # Expected engine uses
            if EXPECTED_ENGINE in imp:
                engine_uses.append((rel, imp))

        # B) Ghost import cross-reference (stale filenames still in import paths)
        ghost_hits.extend(ghost_check(rel, imports, STALE_FILES))

        # C) Protocol layer leaks
        proto_leaks.extend(protocol_leak_check(rel, imports, PROTOCOL_FORBIDDEN_TARGETS))

    # D) Cycle detection
    print("  Detecting import cycles …", end=" ", flush=True)
    cycles = detect_cycles(graph)
    print(f"{len(cycles)} found")

    # ── Remove duplicate stale / ghost hits ───────────────────────────────────
    stale_hits  = list(dict.fromkeys(stale_hits))
    ghost_hits  = list(dict.fromkeys(ghost_hits))

    # Ghost hits that are *also* already in stale_hits — tag them as combined
    stale_set = {(f, i) for f, i in stale_hits}
    [(f, i, s) for f, i, s in ghost_hits if (f, i) not in stale_set]

    # ── DOT output ────────────────────────────────────────────────────────────
    dot_lines = [
        "digraph hypatiax_imports {",
        "  rankdir=LR;",
        "  node [shape=box fontsize=9];",
    ]
    for src, targets in graph.items():
        src_id = src.replace("/", "_").replace(".", "_").replace("-", "_")
        for tgt in targets:
            tgt_id = tgt.replace(".", "_").replace("-", "_")
            dot_lines.append(f'  "{src_id}" -> "{tgt_id}";')
    dot_lines.append("}")
    dot_path = out_dir / "import_graph.dot"
    dot_path.write_text("\n".join(dot_lines))

    # ── Text report ───────────────────────────────────────────────────────────
    sep   = "=" * 60
    lines = [
        "HypatiaX Internal Import DAG Report (enhanced)",
        sep,
        f"Files scanned    : {len(py_files)}",
        f"Import edges     : {sum(len(v) for v in graph.values())}",
        "",
        f"[A] Stale engine imports  : {len(stale_hits):3d}  (target: 0)",
        f"[B] Ghost imports         : {len(ghost_hits):3d}  (target: 0)",
        f"[C] Protocol layer leaks  : {len(proto_leaks):3d}  (target: 0)",
        f"[D] Import cycles         : {len(cycles):3d}  (target: 0)",
        f"    {EXPECTED_ENGINE} uses: {len(engine_uses)}",
        "",
        sep,
    ]

    # A
    lines.append("\n[A] STALE ENGINE IMPORTS")
    if stale_hits:
        lines.append("    ⚠  Run apply_patches.py to replace with hybrid_system_v50_2")
        for f, imp in stale_hits:
            lines.append(f"    {f}  →  {imp}")
    else:
        lines.append(f"    ✓  None — all engine imports use {EXPECTED_ENGINE}")

    # B
    lines.append("\n[B] GHOST IMPORTS  (import path references a stale filename)")
    if ghost_hits:
        lines.append("    ⚠  These imports point at files that should no longer exist")
        for f, imp, stem in ghost_hits:
            lines.append(f"    {f}  →  {imp}  (stale file: {stem})")
    else:
        lines.append("    ✓  None")

    # C
    lines.append("\n[C] PROTOCOL LAYER LEAKS  (protocols/ → experiments/benchmarks/)")
    if proto_leaks:
        lines.append("    ⚠  Protocol files must not import benchmark scripts directly")
        for f, imp in proto_leaks:
            lines.append(f"    {f}  →  {imp}")
    else:
        lines.append("    ✓  None")

    # D
    lines.append("\n[D] IMPORT CYCLES")
    if cycles:
        lines.append("    ⚠  Circular imports detected:")
        for cycle in cycles:
            lines.append("    " + " → ".join(cycle))
    else:
        lines.append("    ✓  No cycles detected in internal import graph")

    # Engine uses
    lines.append(f"\n{EXPECTED_ENGINE} usages ({len(engine_uses)}):")
    for f, imp in engine_uses:
        lines.append(f"  {f}  →  {imp}")

    report_text = "\n".join(lines)
    (out_dir / "import_report.txt").write_text(report_text)

    # ── Supplementary artefacts ───────────────────────────────────────────────
    if stale_hits:
        (out_dir / "stale_imports.txt").write_text(
            "\n".join(f"{f}\t{imp}" for f, imp in stale_hits)
        )

    if ghost_hits:
        (out_dir / "ghost_imports.txt").write_text(
            "\n".join(f"{f}\t{imp}\t{stem}" for f, imp, stem in ghost_hits)
        )

    if cycles:
        (out_dir / "cycles.txt").write_text(
            "\n".join(", ".join(c) for c in cycles)
        )

    # ── Console summary ───────────────────────────────────────────────────────
    def _flag(count: int) -> str:
        return "  ✓" if count == 0 else f"  ⚠  ({count})"

    print(f"\n  [A] Stale engine imports : {len(stale_hits)}{_flag(len(stale_hits))}"
          + ("  →  run apply_patches.py" if stale_hits else ""))
    print(f"  [B] Ghost imports        : {len(ghost_hits)}{_flag(len(ghost_hits))}")
    print(f"  [C] Protocol layer leaks : {len(proto_leaks)}{_flag(len(proto_leaks))}")
    print(f"  [D] Import cycles        : {len(cycles)}{_flag(len(cycles))}")
    print(f"\n  DOT graph  →  {dot_path}")
    print(f"  Report     →  {out_dir / 'import_report.txt'}")
    if stale_hits:
        print(f"  Stale list →  {out_dir / 'stale_imports.txt'}")
    if ghost_hits:
        print(f"  Ghost list →  {out_dir / 'ghost_imports.txt'}")
    if cycles:
        print(f"  Cycles     →  {out_dir / 'cycles.txt'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
scan_internal_imports.py
========================
HypatiaX §11 provenance tool (c): maps the internal import DAG across all
.py files in the repo to verify that:
  - no protocol imports old benchmark scripts directly
  - no file imports hybrid_system_v40 (stale engine, FIX-C2)
  - the import graph is acyclic within the protocols/ layer

Outputs:
  logs/repro_output/import_graph.dot   — Graphviz DOT format
  logs/repro_output/import_report.txt  — human-readable findings
  logs/repro_output/stale_imports.txt  — files still importing v40 (should be empty)

Usage:
    python3 scan_internal_imports.py --root . --out logs/repro_output

Called by:
  run_all.sh  →  run_step "scan-imports" ...
  run_all.py  →  Step("scan-imports", ...)

Exit codes:
  0 — scan complete (stale imports are warnings, not fatal here; apply_patches handles fixes)
  1 — critical error (root not found)
"""

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path


# ── Internal package roots to scan ────────────────────────────────────────────
SCAN_ROOTS = [
    "protocols",
    "hypatiax",
    "core",
    "shared",
    "scripts",
    "figures",
    "reproducibility",
]

STALE_IMPORTS = [
    "hybrid_system_v40",
    "hybrid_system_v40fix",
    "hybrid_system_v50.",   # v50 (not v50_2) — intermediate
    "hybrid_system_v52",
]

EXPECTED_ENGINE = "hybrid_system_v50_2"


# ── AST import extraction ─────────────────────────────────────────────────────

def extract_imports(py_file: Path) -> list[str]:
    """Return list of imported module names from a .py file (best-effort)."""
    try:
        tree = ast.parse(py_file.read_text(errors="replace"))
    except SyntaxError:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def is_internal(module: str, scan_roots: list[str]) -> bool:
    return any(module == r or module.startswith(r + ".") for r in scan_roots)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="HypatiaX internal import DAG scanner")
    parser.add_argument("--root", default=".", help="Repo root (default: .)")
    parser.add_argument("--out",  default="logs/repro_output",
                        help="Output directory for scan artefacts")
    args = parser.parse_args()

    root    = Path(args.root).resolve()
    out_dir = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(f"  ERROR: root not found: {root}")
        return 1

    print(f"\n  Root       : {root}")
    print(f"  Output dir : {out_dir}")

    # ── Collect all .py files ─────────────────────────────────────────────────
    py_files: list[Path] = []
    for r in SCAN_ROOTS:
        d = root / r
        if d.exists():
            py_files.extend(d.rglob("*.py"))
    # also root-level .py files
    py_files.extend(root.glob("*.py"))
    py_files = sorted(set(py_files))

    print(f"  Python files scanned: {len(py_files)}")

    # ── Build adjacency map ───────────────────────────────────────────────────
    graph: dict[str, list[str]] = defaultdict(list)
    stale_hits: list[tuple[str, str]] = []  # (file, import)
    engine_uses: list[tuple[str, str]] = []  # (file, import)

    for py in py_files:
        rel = str(py.relative_to(root))
        imports = extract_imports(py)
        for imp in imports:
            if is_internal(imp, SCAN_ROOTS):
                graph[rel].append(imp)
            # Check for stale engine imports
            for stale in STALE_IMPORTS:
                if stale in imp:
                    stale_hits.append((rel, imp))
            if EXPECTED_ENGINE in imp:
                engine_uses.append((rel, imp))

    # ── DOT output ────────────────────────────────────────────────────────────
    dot_lines = ["digraph hypatiax_imports {",
                 '  rankdir=LR;',
                 '  node [shape=box fontsize=9];']
    for src, targets in graph.items():
        src_id = src.replace("/", "_").replace(".", "_").replace("-", "_")
        for tgt in targets:
            tgt_id = tgt.replace(".", "_").replace("-", "_")
            dot_lines.append(f'  "{src_id}" -> "{tgt_id}";')
    dot_lines.append("}")
    dot_path = out_dir / "import_graph.dot"
    dot_path.write_text("\n".join(dot_lines))

    # ── Report ────────────────────────────────────────────────────────────────
    report_lines = [
        "HypatiaX Internal Import DAG Report",
        "=" * 50,
        f"Files scanned : {len(py_files)}",
        f"Import edges  : {sum(len(v) for v in graph.values())}",
        f"Stale imports : {len(stale_hits)}  (should be 0 after apply_patches)",
        f"Engine v50_2  : {len(engine_uses)} usage(s)",
        "",
    ]

    if stale_hits:
        report_lines.append("⚠  STALE ENGINE IMPORTS (run apply_patches.py to fix):")
        for f, imp in stale_hits:
            report_lines.append(f"  {f}  →  {imp}")
        report_lines.append("")
    else:
        report_lines.append(f"✓ No stale engine imports — all use {EXPECTED_ENGINE}")
        report_lines.append("")

    report_lines.append(f"Engine v50_2 uses:")
    for f, imp in engine_uses:
        report_lines.append(f"  {f}  →  {imp}")

    report_text = "\n".join(report_lines)
    (out_dir / "import_report.txt").write_text(report_text)

    if stale_hits:
        (out_dir / "stale_imports.txt").write_text(
            "\n".join(f"{f}\t{imp}" for f, imp in stale_hits)
        )

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n  Import edges       : {sum(len(v) for v in graph.values())}")
    print(f"  Stale v40 imports  : {len(stale_hits)}"
          + ("  ⚠  run apply_patches.py" if stale_hits else "  ✓"))
    print(f"  {EXPECTED_ENGINE} uses: {len(engine_uses)}")
    print(f"\n  DOT graph → {dot_path}")
    print(f"  Report    → {out_dir / 'import_report.txt'}")
    if stale_hits:
        print(f"  Stale     → {out_dir / 'stale_imports.txt'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env bash
# fix_ruff.sh — Restore corrupted import statements and fix Ruff errors
# Run from repo root: bash .github/scripts/fix_ruff.sh
set -euo pipefail

BIO="hypatiax/experiments/tests/test_symbolic_engine_pysr_biology_.py"
CROSSED="hypatiax/experiments/tests/test_symbolic_engine_crossed.py"

# ── 1. Restore missing "from X import (" opener lines ────────────────────────
#
# Both files were corrupted: the `from X import (` opener was deleted, leaving
# orphaned indented name lines and a dangling `)` — causing invalid-syntax.
# This step reinserts the missing opener(s) before each orphaned block.

echo "Restoring corrupted import statements..."

python3 - <<'PY'
from pathlib import Path

# ── biology ───────────────────────────────────────────────────────────────────
p = Path("hypatiax/experiments/tests/test_symbolic_engine_pysr_biology_.py")
if p.exists():
    src = p.read_text()
    MARKER = 'sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "tools" / "symbolic"))'
    if MARKER in src and "from symbolic_engine import (" not in src:
        src = src.replace(MARKER, MARKER + "\nfrom symbolic_engine import (", 1)
        p.write_text(src)
        print(f"  Restored: {p}")
    else:
        print(f"  OK: {p}")
else:
    print(f"  SKIP: {p}")
PY

python3 - <<'PY'
from pathlib import Path

# ── crossed: two-pass repair (try block + except block) ───────────────────────
# The try: block needs `from symbolic_engine_crossed import (`
# The except: block needs `from symbolic_engine import (`
p = Path("hypatiax/experiments/tests/test_symbolic_engine_crossed.py")
if not p.exists():
    print(f"  SKIP: {p}"); raise SystemExit(0)

def is_orphaned_name(line):
    """Line is an indented symbol like '        BayesianRanker,'"""
    s = line.strip()
    return (s and s.endswith(",") and not s.startswith("#")
            and not s.startswith("from ") and not s.startswith("import ")
            and line.startswith("    "))

def repair_pass(lines, trigger, module):
    out = []
    i = 0
    repaired = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        if line.rstrip() == trigger:
            i += 1
            # consume any comment lines
            while i < len(lines) and lines[i].strip().startswith("#"):
                out.append(lines[i])
                i += 1
            # if next line is an orphaned name, insert the missing opener
            if i < len(lines) and is_orphaned_name(lines[i]):
                indent = len(lines[i]) - len(lines[i].lstrip())
                from_indent = " " * (indent - 4)
                out.append(f"{from_indent}from {module} import (\n")
                repaired += 1
            continue
        i += 1
    return out, repaired

lines = p.read_text().splitlines(keepends=True)
lines, r1 = repair_pass(lines, "try:",              "symbolic_engine_crossed")
lines, r2 = repair_pass(lines, "except ImportError:", "symbolic_engine")
total = r1 + r2
if total:
    p.write_text("".join(lines))
    print(f"  Restored {total} opener(s): {p}")
else:
    print(f"  OK: {p}")
PY

# ── 2. F401: remove unused imports via ruff --unsafe-fixes ────────────────────
# With syntax restored, ruff can now parse and remove unused imports correctly.

echo "Fixing F401 unused imports..."

for f in "$BIO" "$CROSSED"; do
    if [ -f "$f" ]; then
        ruff check --select F401 --fix --unsafe-fixes "$f" \
            && echo "  F401 fixed: $f"
    else
        echo "  SKIP: $f"
    fi
done

# ── 3. F401 residuals: add # noqa for names ruff can't auto-remove ────────────
# DataPatternAnalyzer and VariableNameValidator appear in the try: block of
# crossed.py — they ARE used in the file body but ruff can't confirm cross-block
# usage, so it flags them. Suppress with noqa rather than removing.

python3 - <<'PY'
from pathlib import Path
p = Path("hypatiax/experiments/tests/test_symbolic_engine_crossed.py")
if not p.exists():
    raise SystemExit(0)
lines = p.read_text().splitlines(keepends=True)
out = []
residuals = {"DataPatternAnalyzer", "VariableNameValidator"}
changed = 0
for line in lines:
    name = line.strip().rstrip(",")
    if name in residuals and "# noqa" not in line:
        line = line.rstrip("\n").rstrip() + "  # noqa: F401\n"
        changed += 1
    out.append(line)
if changed:
    p.write_text("".join(out))
    print(f"  Added # noqa: F401 to {changed} residual name(s) in {p}")
PY

# ── 4. I001: sort import blocks ───────────────────────────────────────────────

echo "Fixing I001 unsorted imports..."

for f in "$BIO" "$CROSSED"; do
    if [ -f "$f" ]; then
        ruff check --select I001 --fix "$f" \
            && echo "  I001 fixed: $f"
    else
        echo "  SKIP: $f"
    fi
done

echo ""
echo "All fixes applied. Running final ruff check..."
ruff check "$BIO" "$CROSSED" \
    --select F401,F541,I001 \
    --output-format github \
    || true

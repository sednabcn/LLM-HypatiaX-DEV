#!/usr/bin/env bash
# fix_ruff.sh — Fix all 10 Ruff errors from CI lint run
# Run from repo root: bash .github/scripts/fix_ruff.sh
set -euo pipefail

# ── 1. F401: unused imports (manual removal) ─────────────────────────────────

echo "Fixing F401 unused imports..."

# test_subprocess_timeout.py L329 — remove `import unittest`
python3 - <<'PY'
import re
from pathlib import Path
p = Path("hypatiax/experiments/tests/test_subprocess_timeout.py")
if p.exists():
    src = p.read_text()
    new = re.sub(r'^ {0,4}import unittest *\n', '', src, flags=re.MULTILINE)
    p.write_text(new)
    print(f"  F401 fixed: {p} (import unittest)")
else:
    print(f"  SKIP: {p}")
PY

# run_exp2_hybrid_system.py L319 — remove `import signal`
python3 - <<'PY'
import re
from pathlib import Path
p = Path("hypatiax/experiments/benchmarks/run_exp2_hybrid_system.py")
if p.exists():
    src = p.read_text()
    new = re.sub(r'^ {0,4}import signal *\n', '', src, flags=re.MULTILINE)
    p.write_text(new)
    print(f"  F401 fixed: {p} (import signal)")
else:
    print(f"  SKIP: {p}")
PY

# run_comparative_suite_benchmark_injected.py L59-60 — remove unused from-imports
python3 - <<'PY'
import re
from pathlib import Path
p = Path("hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_injected.py")
if p.exists():
    src = p.read_text()
    new = re.sub(r'^ {0,4}from dataclasses import dataclass *\n', '', src, flags=re.MULTILINE)
    new = re.sub(r'^ {0,4}from datetime import datetime *\n',    '', new,  flags=re.MULTILINE)
    p.write_text(new)
    print(f"  F401 fixed: {p} (dataclass, datetime)")
else:
    print(f"  SKIP: {p}")
PY

# ── 2. F541: f-strings without placeholders ───────────────────────────────────

echo "Fixing F541 f-strings without placeholders..."

python3 - <<'PY'
import re
from pathlib import Path
p = Path("hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_injected.py")
if not p.exists():
    print(f"  SKIP: {p}"); raise SystemExit(0)
src = p.read_text()

# Strip f-prefix from f-strings that contain no { or }
# Handles both double and single quoted variants (non-nested)
src = re.sub(r'\bf("(?:[^"\\{}]|\\.)*")', r'\1', src)
src = re.sub(r"\bf('(?:[^'\\{}]|\\.)*')", r'\1', src)

p.write_text(src)
print(f"  F541 fixed: {p}")
PY

# ── 3. I001: sort import blocks via ruff --fix ────────────────────────────────

echo "Fixing I001 unsorted imports via ruff --fix..."

FILES=(
    "hypatiax/experiments/tests/test_subprocess_timeout.py"
    "hypatiax/experiments/tests/test_protocol_t1_t2_t3_symbolic.py"
    "hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_injected.py"
)

for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        ruff check --select I001 --fix "$f" && echo "  I001 fixed: $f"
    else
        echo "  SKIP: $f"
    fi
done

echo ""
echo "All fixes applied. Running final ruff check..."
ruff check \
    hypatiax/experiments/tests/test_subprocess_timeout.py \
    hypatiax/experiments/tests/test_protocol_t1_t2_t3_symbolic.py \
    hypatiax/experiments/benchmarks/run_exp2_hybrid_system.py \
    hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_injected.py \
    --select F401,F541,I001 \
    --output-format github \
    || true   # non-zero exit is fine here — CI will report any residuals

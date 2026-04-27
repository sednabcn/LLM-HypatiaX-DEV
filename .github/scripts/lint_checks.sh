#!/usr/bin/env bash
# .github/scripts/lint_checks.sh
# Runs all lint & code-quality checks used by the lint job.
# Designed to be called as a single step after ruff runs separately.
#
# Checks:
#   1. No stale hybrid_system_v40 references (FIX-C2)
#   2. No exposed API keys
#   3. hypatiax/protocols/ input-data module validity
#   4. Patched source syntax (fixup-tex + validate_code)

set -euo pipefail

echo "── 1 · Stale v40 engine check (FIX-C2) ─────────────────────────────"
STALE=$(grep -r "hybrid_system_v40[^_]" hypatiax/ protocols/ scripts/ \
          --include="*.py" \
          --exclude-dir=BACKUP \
          -l 2>/dev/null || true)
if [[ -n "${STALE}" ]]; then
  echo "::error ::Stale hybrid_system_v40 import — must be v50_2:"
  echo "${STALE}"
  exit 1
fi
echo "  ✓ No stale v40 engine references"

echo "── 2 · Exposed API-key scan ─────────────────────────────────────────"
EXPOSED=$(grep -r "sk-ant-api" . \
            --include="*.py" --include="*.ipynb" \
            --include="*.yaml" --include="*.yml" \
            -l 2>/dev/null | grep -v ".git" || true)
if [[ -n "${EXPOSED}" ]]; then
  echo "::error ::Exposed API key — revoke at console.anthropic.com"
  echo "${EXPOSED}"
  exit 1
fi
echo "  ✓ No exposed API keys"

echo "── 3 · hypatiax/protocols/ input-data modules ───────────────────────"
python3 scripts/patches/check_hypatiax_protocols.py

echo "── 4 · Patched source syntax (fixup-tex + validate_code) ───────────"
python3 run_all_checkpoint.py --only fixup-tex
python3 scripts/patches/validate_code.py

echo "✓ All lint checks passed"

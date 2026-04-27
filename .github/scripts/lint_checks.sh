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

FAILED=0

# ── Helper: emit a GitHub Actions error annotation ───────────────────────────
fail() {
  echo "::error file=.github/scripts/lint_checks.sh::$*"
  FAILED=1
}

# ── 1 · Stale v40 engine check (FIX-C2) ──────────────────────────────────────
echo "── 1 · Stale v40 engine check (FIX-C2) ─────────────────────────────"
STALE=$(find hypatiax/ protocols/ scripts/ -name "*.py" \
          ! -path "*/BACKUP/*" -print0 2>/dev/null \
        | xargs -0 grep -ln "hybrid_system_v40[^_]" 2>/dev/null || true)
if [[ -n "${STALE}" ]]; then
  echo "::error ::Stale hybrid_system_v40 import — must be v50_2:"
  # Print file + line numbers for easy navigation
  find hypatiax/ protocols/ scripts/ -name "*.py" \
    ! -path "*/BACKUP/*" -print0 2>/dev/null \
  | xargs -0 grep -n "hybrid_system_v40[^_]" 2>/dev/null || true
  fail "Check 1 FAILED: stale v40 engine references"
else
  echo "  ✓ No stale v40 engine references"
fi

# ── 2 · Exposed API-key scan ──────────────────────────────────────────────────
echo "── 2 · Exposed API-key scan ─────────────────────────────────────────"
EXPOSED=$(grep -r "sk-ant-api" . \
            --include="*.py" --include="*.ipynb" \
            --include="*.yaml" --include="*.yml" \
            -l 2>/dev/null \
          | grep -v "\.git" || true)
if [[ -n "${EXPOSED}" ]]; then
  echo "::error ::Exposed API key — revoke immediately at console.anthropic.com"
  echo "${EXPOSED}"
  fail "Check 2 FAILED: exposed API key"
else
  echo "  ✓ No exposed API keys"
fi

# ── 3 · hypatiax/protocols/ input-data modules ────────────────────────────────
echo "── 3 · hypatiax/protocols/ input-data modules ───────────────────────"
PROTO_SCRIPT="scripts/patches/check_hypatiax_protocols.py"
if [[ -f "${PROTO_SCRIPT}" ]]; then
  if ! python3 "${PROTO_SCRIPT}"; then
    fail "Check 3 FAILED: ${PROTO_SCRIPT} exited non-zero"
  else
    echo "  ✓ protocols check passed"
  fi
else
  echo "  ⚠ ${PROTO_SCRIPT} not found — skipped"
fi

# ── 4a · fixup-tex ────────────────────────────────────────────────────────────
echo "── 4 · Patched source syntax (fixup-tex + validate_code) ───────────"
if [[ -f "run_all_checkpoint.py" ]]; then
  if ! python3 run_all_checkpoint.py --only fixup-tex; then
    fail "Check 4a FAILED: run_all_checkpoint.py --only fixup-tex"
  else
    echo "  ✓ fixup-tex passed"
  fi
else
  echo "  ⚠ run_all_checkpoint.py not found — fixup-tex skipped"
fi

# ── 4b · validate_code ────────────────────────────────────────────────────────
VALIDATE_SCRIPT="scripts/patches/validate_code.py"
if [[ -f "${VALIDATE_SCRIPT}" ]]; then
  if ! python3 "${VALIDATE_SCRIPT}"; then
    fail "Check 4b FAILED: ${VALIDATE_SCRIPT} exited non-zero"
  else
    echo "  ✓ validate_code passed"
  fi
else
  echo "  ⚠ ${VALIDATE_SCRIPT} not found — skipped"
fi

# ── Final result ──────────────────────────────────────────────────────────────
echo ""
if [[ "${FAILED}" -ne 0 ]]; then
  echo "::error ::lint_checks.sh — one or more checks FAILED (see annotations above)"
  exit 1
fi
echo "✓ All lint checks passed"

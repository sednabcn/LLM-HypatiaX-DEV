#!/usr/bin/env bash
# run_patch_ci.sh — CI wrapper around patch_runner.py
#
# Responsibilities that live HERE rather than in patch_runner.py itself,
# so the runner stays manuscript-agnostic and testable outside of CI:
#   1. Preflight: fail fast with a clear message if the manifest, the
#      .tex targets, or investigation scripts referenced by the manifest
#      are missing, before wasting a compile pass on a doomed run.
#   2. Run patch_runner.py with whatever mode CI was asked for (dry-run
#      report on PRs, --apply [--allow-value-change] on manual dispatch).
#   3. Render a GitHub Step Summary table from the runner's own output
#      so reviewers see PATCHED/SKIPPED/ERROR/etc. per item without
#      opening the raw log.
#   4. Propagate patch_runner.py's exit code unchanged — it already
#      exits 1 if any item ended in ERROR/WARN, which is what should
#      fail the job.
#
# Usage (env-var driven, set by the workflow):
#   APPLY=true|false
#   ALLOW_VALUE_CHANGE=true|false
#   IDS=comma,separated,ids            (optional, empty = all items)
#   MANIFEST=path/to/patch_manifest.yaml   (optional, has a default)
set -uo pipefail

MANIFEST="${MANIFEST:-patch_manifest.yaml}"
APPLY="${APPLY:-false}"
ALLOW_VALUE_CHANGE="${ALLOW_VALUE_CHANGE:-false}"
IDS="${IDS:-}"

log() { printf '%s\n' "$*" >&2; }

# --- 1. Preflight -----------------------------------------------------
log "== preflight =="

if [ ! -f "$MANIFEST" ]; then
    log "FATAL: manifest not found at '$MANIFEST'"
    exit 1
fi

if ! python3 -c "import yaml" 2>/dev/null; then
    log "FATAL: PyYAML not installed (pip install pyyaml --break-system-packages)"
    exit 1
fi

missing=0
while IFS=$'\t' read -r file script; do
    if [ -n "$file" ] && [ ! -f "$file" ]; then
        log "  MISSING target file: $file"
        missing=1
    fi
    if [ -n "$script" ] && [ "$script" != "None" ] && [ ! -f "$script" ]; then
        log "  MISSING investigation script: $script"
        missing=1
    fi
done < <(python3 - "$MANIFEST" <<'PYEOF'
import sys, yaml
with open(sys.argv[1]) as f:
    manifest = yaml.safe_load(f)
for item in manifest:
    print(f"{item.get('file','')}\t{item.get('investigation_script','')}")
PYEOF
)

if [ "$missing" -eq 1 ]; then
    log "FATAL: preflight found missing files referenced by the manifest — aborting before any patch attempt."
    exit 1
fi
log "preflight OK"

# --- 2. Run the runner --------------------------------------------------
ARGS=(--manifest "$MANIFEST")
[ "$APPLY" = "true" ] && ARGS+=(--apply)
[ "$ALLOW_VALUE_CHANGE" = "true" ] && ARGS+=(--allow-value-change)
[ -n "$IDS" ] && ARGS+=(--ids "$IDS")

log "== running: python3 patch_runner.py ${ARGS[*]} =="
RUN_LOG="$(mktemp)"
python3 patch_runner.py "${ARGS[@]}" 2>&1 | tee "$RUN_LOG"
RUNNER_EXIT="${PIPESTATUS[0]}"

# --- 3. Job summary -------------------------------------------------------
{
    echo "## Patch runner — $([ "$APPLY" = "true" ] && echo APPLY || echo "DRY RUN")"
    echo
    echo "Manifest: \`$MANIFEST\`"
    [ -n "$IDS" ] && echo "Restricted to ids: \`$IDS\`"
    [ "$ALLOW_VALUE_CHANGE" = "true" ] && echo "\`--allow-value-change\` was set."
    echo
    echo "| id | category | outcome | detail |"
    echo "|----|----------|---------|--------|"
    grep -E '^\s*#[0-9]+ \[' "$RUN_LOG" | sed -E \
        -e 's/^\s*#([0-9]+) \[([^]]*)\]\s*([A-Za-z-]+)\s*(.*)$/| \1 | \2 | \3 | \4 |/' \
        -e 's/<-- needs attention//'
    echo
    if [ "$RUNNER_EXIT" -ne 0 ]; then
        echo "**One or more items need manual attention (ERROR/WARN) — job failed.**"
    else
        echo "All items completed cleanly."
    fi
} >> "${GITHUB_STEP_SUMMARY:-/dev/stdout}"

rm -f "$RUN_LOG"
exit "$RUNNER_EXIT"

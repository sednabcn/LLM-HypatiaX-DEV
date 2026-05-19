#!/usr/bin/env bash
# .github/scripts/dispatch_experiment.sh
#
# Dispatch a single HypatiaX experiment via the GitHub CLI.
#
# Usage:
#   dispatch_experiment.sh <experiment_id> <n_shards> [task_ids_override] [workflow_file] [--dry-run]
#
#   <experiment_id>      e.g. exp1, exp2_feynman, suppB
#   <n_shards>           number of parallel worker shards (the only value you type)
#                        Default: 1 (all experiments run single-shard).
#   task_ids_override    space-separated task IDs to run (blank = full experiment set)
#   workflow_file        workflow filename to dispatch (default: ci_experiment_simplify.yml)
#                        e.g. ci_experiment_simplify.yml or ci_experiment.yml
#   --dry-run            print the gh command without executing it (any position after $2)
#
# All other workflow inputs (pysr_generations, n_samples, noise_levels, …)
# are read automatically from config/repro.yaml.
#
# Prerequisites:
#   - gh CLI authenticated (GITHUB_TOKEN / GH_TOKEN in env)
#   - python3 + PyYAML installed  (pip install pyyaml)
#   - Run from the repository root (or set REPO_ROOT below)
#
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────
EXP="${1:?Usage: dispatch_experiment.sh <experiment_id> <n_shards> [task_ids_override] [workflow_file] [--dry-run]}"
# Default to 1 shard for all experiments.
N_SHARDS="${2:-1}"
TASK_IDS_OVERRIDE="${3:-}"
# $4: workflow file — defaults to ci_experiment_simplify.yml.
# ci_schedule_simplify.yml passes EXP_WORKFLOW_FILE here so the scheduler
# always targets the correct (simplify) runner, not the legacy ci_experiment.yml.
WORKFLOW_FILE="${4:-ci_experiment_simplify.yml}"

# --dry-run can appear at any position after $2 for manual use.
DRY_RUN=""
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN="--dry-run"
done

# Path to the repo's canonical repro config (relative to repo root).
REPRO="${REPRO_CFG:-config/repro.yaml}"

# ── Helpers ───────────────────────────────────────────────────────────────────

# Read a top-level key from repro.yaml; fall back to $default if absent.
get() {
  local key="$1" default="$2"
  python3 - <<PYEOF 2>/dev/null || echo "$default"
import yaml, sys
try:
    cfg = yaml.safe_load(open("$REPRO"))
    val = cfg.get("$key")
    print(val if val is not None else "$default")
except Exception:
    print("$default")
PYEOF
}

# ── Validate n_shards (1–4) ───────────────────────────────────────────────────
if ! [[ "$N_SHARDS" =~ ^[1-4]$ ]]; then
  echo "WARNING: n_shards=$N_SHARDS is outside the accepted range (1–4)."
  echo "         Accepted values: 1 (default, all experiments), 2, 3, 4."
  echo "         Proceeding with n_shards=$N_SHARDS as requested."
fi

# ── Read parameters from config/repro.yaml ────────────────────────────────────
PYSR_GEN=$(get  pysr_generations   "10000")
PYSR_POP=$(get  pysr_populations   "4")
N_SAMPLES=$(get feynman_samples    "200")
NOISE=$(get     noise_levels       "0.0,0.5,1.0,5.0,10.0")
SEEDS=$(get     default_seeds      "")
TIMEOUT=$(get   feynman_timeout    "1100")

# ── Summary ───────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════"
echo "  Experiment    : $EXP"
echo "  Workflow file : $WORKFLOW_FILE"
echo "  n_shards      : $N_SHARDS   (default=1; all experiments single-shard)"
echo "  config source : $REPRO      (auto)"
echo "───────────────────────────────────────────────────────"
echo "  pysr_generations  : $PYSR_GEN"
echo "  pysr_populations  : $PYSR_POP"
echo "  feynman_timeout   : $TIMEOUT  (repro.yaml → workflow env FEYNMAN_TIMEOUT, not a dispatch field)"
echo "  n_samples         : $N_SAMPLES"
echo "  noise_levels      : $NOISE"
echo "  seeds             : ${SEEDS:-<default>}"
echo "  task_ids_override : ${TASK_IDS_OVERRIDE:-<none — full experiment set>}"
echo "═══════════════════════════════════════════════════════"

# ── Dry-run guard ─────────────────────────────────────────────────────────────
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "[DRY RUN] would execute:"
  echo "  # 1. Capture DISPATCH_ISO 5 s before gh call (clock-skew fix)"
  echo "  gh workflow run $WORKFLOW_FILE \\"
  echo "    --field experiment=\"$EXP\" \\"
  echo "    --field n_shards=\"$N_SHARDS\" \\"
  echo "    --field pysr_generations=\"$PYSR_GEN\" \\"
  echo "    --field pysr_populations=\"$PYSR_POP\" \\"
  echo "    --field n_samples=\"$N_SAMPLES\" \\"
  echo "    --field noise_levels=\"$NOISE\" \\"
  echo "    --field seeds=\"$SEEDS\" \\"
  echo "    --field task_ids_override=\"$TASK_IDS_OVERRIDE\" \\"
  echo "    --field resume=\"true\" \\"
  echo "    --field dry_run=\"false\""
  echo "  # 2. Poll via gh run list (strategy-1: 20×20s) then API filter (strategy-2: 30×20s)"
  echo "  # 3. Print EXP_RUN_ID=<id> for callers to capture"
  exit 0
fi

# ── Dispatch ──────────────────────────────────────────────────────────────────
# Capture timestamp BEFORE dispatch so created_at >= DISPATCH_ISO is always true.
# Subtract 5 s as a clock-skew safety margin (portable: pure bash arithmetic).
_NOW_EPOCH=$(date -u +%s)
_BEFORE_EPOCH=$(( _NOW_EPOCH - 5 ))
DISPATCH_ISO=$(date -u -d "@${_BEFORE_EPOCH}" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null \
  || date -u -r "${_BEFORE_EPOCH}" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null \
  || date -u +%Y-%m-%dT%H:%M:%SZ)   # fallback: current time (no subtraction)

gh workflow run "$WORKFLOW_FILE" \
  --field experiment="$EXP" \
  --field n_shards="$N_SHARDS" \
  --field pysr_generations="$PYSR_GEN" \
  --field pysr_populations="$PYSR_POP" \
  --field n_samples="$N_SAMPLES" \
  --field noise_levels="$NOISE" \
  --field seeds="$SEEDS" \
  --field task_ids_override="$TASK_IDS_OVERRIDE" \
  --field resume="true" \
  --field dry_run="false"

echo "✓ Dispatched $EXP → $WORKFLOW_FILE (n_shards=$N_SHARDS)"
echo "  dispatch_iso (pre-dispatch, -5s): $DISPATCH_ISO"

# ── Locate the dispatched run ID ──────────────────────────────────────────────
# Strategy (in order of preference):
#   1. gh run list --workflow immediately after dispatch (most reliable — no
#      timestamp filter needed, just grab the newest run whose display_title
#      matches this experiment).
#   2. Fallback: gh API timestamp filter (created_at >= DISPATCH_ISO) with
#      30 attempts × 20 s = 10 min total (up from the old 20 × 15 s = 5 min).
#
# The run ID is printed as EXP_RUN_ID= so callers (ci_schedule_simplify.yml
# Job B) can capture it with:
#   EXP_RUN_ID=$(... dispatch_experiment.sh ... | grep '^EXP_RUN_ID=' | cut -d= -f2)

echo "  Waiting 30 s for GitHub to register the run ..."
sleep 30

EXP_RUN_ID=""

# -- Strategy 1: gh run list (title match, no timestamp filter) ---------------
for attempt in $(seq 1 20); do
  EXP_RUN_ID=$(gh run list \
    --workflow="$WORKFLOW_FILE" \
    --limit 10 \
    --json databaseId,displayTitle,createdAt \
    --jq --arg exp "$EXP" --arg ts "$DISPATCH_ISO" \
      '.[] | select(.displayTitle | startswith("HypatiaX - " + $exp))
           | select(.createdAt >= $ts)
           | .databaseId' \
    2>/dev/null | head -1 || true)

  if [[ -n "$EXP_RUN_ID" ]]; then
    echo "  [strategy-1] Found run ID: $EXP_RUN_ID (attempt $attempt/20)"
    break
  fi

  # Debug output every 5 attempts
  if (( attempt % 5 == 1 )); then
    echo "  [debug attempt $attempt/20] Recent runs for $WORKFLOW_FILE:"
    gh run list --workflow="$WORKFLOW_FILE" --limit 5 \
      --json databaseId,displayTitle,createdAt \
      --jq '.[] | "    id=\(.databaseId) title=\(.displayTitle) created=\(.createdAt)"' \
      2>/dev/null | head -6 || echo "    (gh run list failed)"
  fi

  echo "  Not found yet (attempt $attempt/20) - retrying in 20 s ..."
  sleep 20
done

# -- Strategy 2: API timestamp filter fallback --------------------------------
if [[ -z "$EXP_RUN_ID" ]]; then
  echo "  [strategy-1 exhausted] Falling back to API timestamp filter ..."
  for attempt in $(seq 1 30); do
    RAW=$(gh api \
      "/repos/$(gh repo view --json nameWithOwner --jq .nameWithOwner)/actions/workflows/${WORKFLOW_FILE}/runs?per_page=50" \
      2>&1) || true
    EXP_RUN_ID=$(echo "$RAW" | jq -r --arg exp "$EXP" --arg ts "$DISPATCH_ISO" '
      .workflow_runs[]?
      | select(.display_title | startswith("HypatiaX - " + $exp))
      | select(.created_at >= $ts)
      | .id' \
      | head -1)
    if [[ -n "$EXP_RUN_ID" ]]; then
      echo "  [strategy-2] Found run ID: $EXP_RUN_ID (attempt $attempt/30)"
      break
    fi
    if (( attempt % 5 == 1 )); then
      echo "  [debug attempt $attempt/30] API response titles:"
      echo "$RAW" | jq -r '.workflow_runs[]? | "    id=\(.id) title=\(.display_title) created=\(.created_at)"' \
        2>/dev/null | head -10 || echo "    (jq parse failed: ${RAW:0:300})"
    fi
    echo "  Not found yet (strategy-2 attempt $attempt/30) - retrying in 20 s ..."
    sleep 20
  done
fi

if [[ -z "$EXP_RUN_ID" ]]; then
  echo "ERROR: could not locate the dispatched run for $EXP" >&2
  echo "  Workflow queried : $WORKFLOW_FILE"                  >&2
  echo "  dispatch_iso     : $DISPATCH_ISO"                   >&2
  echo "  display_title    : must startswith 'HypatiaX - $EXP'" >&2
  echo "  Hint: verify the run-name field in $WORKFLOW_FILE." >&2
  exit 1
fi

# Emit for callers to capture
echo "EXP_RUN_ID=$EXP_RUN_ID"

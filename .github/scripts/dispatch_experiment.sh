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
  exit 0
fi

# ── Dispatch ──────────────────────────────────────────────────────────────────
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

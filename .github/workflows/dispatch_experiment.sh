#!/usr/bin/env bash
# .github/scripts/dispatch_experiment.sh
#
# Dispatch a single HypatiaX experiment via the GitHub CLI.
#
# Usage:
#   dispatch_experiment.sh <experiment_id> <n_shards> [--dry-run] [task_ids_override]
#
#   <experiment_id>      e.g. exp1, exp2_feynman, suppB
#   <n_shards>           number of parallel worker shards (the only value you type)
#                        Recommended: 4 (default). Use 3 only when GitHub quota is tight.
#   --dry-run            print the gh command without executing it
#   task_ids_override    space-separated task IDs to run (blank = full experiment set)
#
# All other workflow inputs (pysr_generations, n_samples, noise_levels, ...)
# are read automatically from config/repro.yaml.
#
# Prerequisites:
#   - gh CLI authenticated (GITHUB_TOKEN / GH_TOKEN in env)
#   - python3 + PyYAML installed  (pip install pyyaml)
#   - Run from the repository root (or set REPO_ROOT below)
#
# -----------------------------------------------------------------------------

set -euo pipefail

# -- Args ----------------------------------------------------------------------
#
# Usage (updated):
#   dispatch_experiment.sh <experiment_id> <n_shards> [task_ids_override] [--dry-run]
#
# --dry-run is now a named flag accepted anywhere after $2, not a positional $3.
# Previously, passing a task_ids_override as $3 silently landed in the dry-run
# slot and was ignored; the actual override slot ($4) stayed empty.  The
# scheduler's call-site ("" "$TASK_IDS_OVERRIDE") is still compatible: the
# empty string at $3 is harmless and --dry-run is simply absent.
EXP="${1:?Usage: dispatch_experiment.sh <experiment_id> <n_shards> [task_ids_override] [--dry-run]}"
N_SHARDS="${2:-4}"
TASK_IDS_OVERRIDE="${3:-}"  # was $4; $3 was the --dry-run positional slot (removed)

# Parse --dry-run from anywhere in args $3+; strip it from TASK_IDS_OVERRIDE if needed.
DRY_RUN=""
for _arg in "${@:3}"; do
  if [[ "$_arg" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    [[ "$TASK_IDS_OVERRIDE" == "--dry-run" ]] && TASK_IDS_OVERRIDE=""
    break
  fi
done

# Path to the repo's canonical repro config (relative to repo root).
REPRO="${REPRO_CFG:-config/repro.yaml}"

# -- Helpers -------------------------------------------------------------------

# Read a top-level key from repro.yaml; fall back to $default if absent.
# If the value is a YAML list, it is coerced to a comma-separated string so
# that --field seeds="42,99,123" is correct rather than --field seeds="[42, 99, 123]".
get() {
  local key="$1" default="$2"
  python3 - <<PYEOF 2>/dev/null || echo "$default"
import yaml, sys
try:
    cfg = yaml.safe_load(open("$REPRO"))
    val = cfg.get("$key")
    if val is None:
        print("$default")
    elif isinstance(val, list):
        print(",".join(str(v) for v in val))
    else:
        print(val)
except Exception:
    print("$default")
PYEOF
}

# -- Validate n_shards --------------------------------------------------------
# Fail fast: 0 or a non-integer here causes cryptic shard-startup failures in
# ci_experiment.yml. Values outside 3-4 are unusual but allowed with a warning.
if ! [[ "$N_SHARDS" =~ ^[0-9]+$ ]] || [[ "$N_SHARDS" -lt 1 ]]; then
  echo "ERROR: n_shards='$N_SHARDS' is not a positive integer - aborting."
  exit 1
fi
if [[ "$N_SHARDS" != "3" && "$N_SHARDS" != "4" ]]; then
  echo "WARNING: n_shards=$N_SHARDS is outside the recommended range (3-4)."
  echo "         Accepted values are 3 (quota-limited) or 4 (preferred)."
  echo "         Proceeding with n_shards=$N_SHARDS as requested."
fi

# -- Read parameters from config/repro.yaml ------------------------------------
PYSR_GEN=$(get  pysr_generations   "10000")
PYSR_POP=$(get  pysr_populations   "4")
N_SAMPLES=$(get feynman_samples    "200")
NOISE=$(get     noise_levels       "0.0,0.5,1.0,5.0,10.0")
SEEDS=$(get     default_seeds      "")
# feynman_timeout is intentionally NOT dispatched via --field: ci_experiment.yml
# injects it as a workflow-level env var (FEYNMAN_TIMEOUT) from repro.yaml
# directly, so passing it here would be ignored and create a misleading field.
# TIMEOUT=$(get feynman_timeout "1100")  <-- removed to avoid dead variable

# -- Summary -------------------------------------------------------------------
echo "======================================================="
echo "  Experiment    : $EXP"
echo "  n_shards      : $N_SHARDS   (manual; recommended=4, minimum=3)"
echo "  config source : $REPRO      (auto)"
echo "-------------------------------------------------------"
echo "  pysr_generations  : $PYSR_GEN"
echo "  pysr_populations  : $PYSR_POP"
echo "  feynman_timeout   : (set via FEYNMAN_TIMEOUT workflow env from repro.yaml - not a dispatch field)"
echo "  n_samples         : $N_SAMPLES"
echo "  noise_levels      : $NOISE"
echo "  seeds             : ${SEEDS:-<default>}"
echo "  task_ids_override : ${TASK_IDS_OVERRIDE:-<none - full experiment set>}"
echo "======================================================="

# -- Dry-run guard -------------------------------------------------------------
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "[DRY RUN] would execute:"
  echo "  gh workflow run ci_experiment.yml \\"
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

# -- Dispatch ------------------------------------------------------------------
gh workflow run ci_experiment.yml \
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

echo "OK Dispatched $EXP (n_shards=$N_SHARDS)"

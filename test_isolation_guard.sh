#!/usr/bin/env bash
# ==============================================================================
#  test_isolation_guard.sh
#
#  Regression test for ci_runner.yml's per-shard isolation-overwrite guard
#  (the "Commit and push shard results to repo" step's OVERWRITTEN check).
#
#  WHY THIS EXISTS:
#    exp1b / exp3b / suppB / suppB_sc each rely on a per-shard filename suffix
#    (HYPATIAX_NSHARDS_SUFFIX or _shard${SHARD_INDEX} tag, set in run_all.sh)
#    to guarantee that parallel CI matrix shards never overwrite each other's
#    committed result files. The actual safety net for that guarantee lives
#    in ci_runner.yml's commit-and-push step: if a shard's `git add` would
#    modify (not just add) an already-committed file, that is the literal
#    signature of a collision. For those four experiments this now hard-fails
#    the job instead of just printing a warning.
#
#  WHAT THIS SCRIPT DOES:
#    1. Parses ci_runner.yml with PyYAML and extracts the ACTUAL, LIVE guard
#       block from the "Commit and push shard results to repo" step — not a
#       hand-copied duplicate that could silently drift out of sync with the
#       real file.
#    2. Wraps that block in a bash function and exercises it against three
#       scenarios per experiment, in a real throwaway git repo:
#         - COLLISION   : a second shard writes a file with the SAME name as
#                         an already-committed one (suffix logic broken/bypassed)
#         - CLEAN       : a second shard writes a file with a DIFFERENT,
#                         correctly-suffixed name (the actual fix working)
#    3. Asserts the guard's exit code matches what each experiment's
#       isolation guarantee promises, and prints a PASS/FAIL summary.
#
#  USAGE:
#    ./test_isolation_guard.sh [path/to/ci_runner.yml]
#    Exit code 0 = all assertions passed. Exit code 1 = at least one failed.
#
#  CI USAGE (optional):
#    Add as a job/step that runs on any PR touching ci_runner.yml or
#    run_all.sh, so the isolation guarantee is regression-tested automatically
#    instead of only being caught the next time a real shard collides.
# ==============================================================================

set -euo pipefail

YML="${1:-ci_runner.yml}"
if [[ ! -f "$YML" ]]; then
  echo "ERROR: cannot find $YML" >&2
  exit 2
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

# ── Step 1: extract the LIVE guard block from the real YAML ──────────────────
python3 - "$YML" "$WORKDIR/guard_block.sh" << 'PYEOF'
import sys, yaml

yml_path, out_path = sys.argv[1], sys.argv[2]
with open(yml_path) as f:
    data = yaml.safe_load(f)

steps = data["jobs"]["worker"]["steps"]
run_text = None
for s in steps:
    if s.get("name") == "Commit and push shard results to repo":
        run_text = s["run"]
        break
if run_text is None:
    sys.exit("ERROR: could not find 'Commit and push shard results to repo' "
              "step in jobs.worker.steps — has it been renamed?")

start_marker = "# Detect any MODIFIED"
end_marker   = "STAGED=$(git diff --cached --name-only"
try:
    start = run_text.index(start_marker)
    end   = run_text.index(end_marker)
except ValueError:
    sys.exit("ERROR: guard block markers not found — has the step been "
              "restructured? Update start_marker/end_marker in this test.")

block = run_text[start:end].rstrip() + "\n"

with open(out_path, "w") as f:
    f.write("# --- extracted live block from ci_runner.yml, do not edit ---\n")
    f.write("overwrite_guard() {\n")
    f.write(block)
    f.write("}\n")
PYEOF

echo "=== Extracted live guard block (from: $YML) ==="
cat "$WORKDIR/guard_block.sh"
echo "================================================"
echo

# ── Step 2: build a throwaway git repo + test runner ─────────────────────────
REPO="$WORKDIR/repo"
FAILED=0

setup_repo() {
  rm -rf "$REPO"
  mkdir -p "$REPO/target"
  cd "$REPO"
  git init -q
  git config user.name  test
  git config user.email test@test.com
  # seed a "previously committed" shard result
  echo '{"shard":"prev"}' > target/result_nshards01.json
  git add -A
  git commit -qm "seed: previous shard result" -q
}

# Runs the extracted guard against a prepared working tree and returns its
# exit code (0 = clean/warn-only, 1 = hard-failed) via $RC.
run_guard() {
  local exp="$1" shard="$2"
  set +e
  RC_OUTPUT=$(EXP="$exp" SHARD="$shard" TARGET="target" bash -c '
    source "'"$WORKDIR"'/guard_block.sh"
    git add -f "$TARGET"
    if git diff --cached --quiet; then
      exit 0
    fi
    overwrite_guard
    exit 0
  ' 2>&1)
  RC=$?
  set -e
}

assert_case() {
  local label="$1" exp="$2" expect_rc="$3"
  if [[ "$RC" -eq "$expect_rc" ]]; then
    echo "PASS  [$label] exp=$exp rc=$RC (expected $expect_rc)"
  else
    echo "FAIL  [$label] exp=$exp rc=$RC (expected $expect_rc)"
    echo "      --- guard output ---"
    echo "$RC_OUTPUT" | sed 's/^/      /'
    FAILED=1
  fi
}

echo "=== Running scenarios ==="
echo

for exp in exp1b exp3b suppB suppB_sc; do
  # COLLISION: second shard writes to the SAME filename as the seeded commit.
  setup_repo
  echo '{"shard":"new","CORRUPTED":true}' > target/result_nshards01.json
  run_guard "$exp" 1
  assert_case "COLLISION (isolation experiment)" "$exp" 1

  # CLEAN: second shard writes a DIFFERENTLY-suffixed file (the real fix).
  setup_repo
  echo '{"shard":"new"}' > target/result_nshards02.json
  run_guard "$exp" 1
  assert_case "CLEAN (isolation experiment)      " "$exp" 0
done

for exp in exp1 exp2_feynman suppA; do
  # COLLISION on a NON-isolation experiment: should warn, not fail.
  setup_repo
  echo '{"shard":"new","CORRUPTED":true}' > target/result_nshards01.json
  run_guard "$exp" 1
  assert_case "COLLISION (non-isolation experiment)" "$exp" 0
done

echo
if [[ "$FAILED" -eq 1 ]]; then
  echo "=== RESULT: FAIL — one or more isolation-guard assertions did not hold ==="
  exit 1
else
  echo "=== RESULT: PASS — isolation guard behaves correctly for all scenarios ==="
  exit 0
fi

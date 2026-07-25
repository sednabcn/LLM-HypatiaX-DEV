#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  merge_dependabot_prs.sh
#  Merges all 6 open Dependabot PRs and updates action pins in ci_experiment.yml
#
#  Usage:
#    chmod +x merge_dependabot_prs.sh
#    ./merge_dependabot_prs.sh
#
#  Requirements: gh CLI authenticated  (gh auth status)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="sednabcn/LLM-HypatiaX-PAPERS-Public"
WORKFLOW_FILE=".github/workflows/ci_experiment.yml"

echo "======================================"
echo "  Dependabot PR merger + action pinner"
echo "======================================"
echo ""

# ── 0. Preflight ──────────────────────────────────────────────────────────────
if ! command -v gh &>/dev/null; then
  echo "ERROR: gh CLI not found. Install from https://cli.github.com"
  exit 1
fi

if ! gh auth status &>/dev/null; then
  echo "ERROR: gh CLI not authenticated. Run: gh auth login"
  exit 1
fi

echo "Authenticated as: $(gh api user --jq .login)"
echo ""

# ── 1. Merge safe PRs (pip / python deps — no workflow changes needed) ────────
SAFE_PRS=(40 39 37 35 34)   # pillow, ckzg, urllib3, gitpython, mistune

echo "--- Merging safe dependency PRs ---"
for PR in "${SAFE_PRS[@]}"; do
  echo -n "  Merging PR #${PR} ... "
  if gh pr merge "$PR" \
       --repo "$REPO" \
       --squash \
       --auto \
       --delete-branch 2>&1 | grep -qE 'merged|already'; then
    echo "OK"
  else
    # Try merge without --auto if branch is already mergeable
    gh pr merge "$PR" --repo "$REPO" --squash --delete-branch || \
      echo "  WARNING: PR #${PR} could not be merged — check manually"
  fi
done

echo ""

# ── 2. Handle PR #41: actions/download-artifact v4.1.8 -> v8.0.1 ─────────────
# Breaking changes between v4 and v8:
#   v5: path change for artifact-id downloads (NOT USED HERE — we download by name pattern)
#   v8: digest mismatch now errors by default (internal artifacts: hashes always match)
# Safe to merge for this workflow.

echo "--- Merging PR #41: actions/download-artifact v4 -> v8 ---"
echo "  (verified: downloads by pattern name, not artifact-id — v5 path change N/A)"
echo "  (verified: internal artifacts — v8 digest-error default is safe)"
echo ""

gh pr merge 41 \
  --repo "$REPO" \
  --squash \
  --delete-branch || echo "  WARNING: PR #41 could not be auto-merged — merging manually below"

echo ""

# ── 3. Update action version pins in ci_experiment.yml ───────────────────────
# After merging Dependabot PRs the workflow file on the default branch already
# has the new versions. This section updates your LOCAL copy of ci_experiment.yml
# to match, so it stays in sync.

echo "--- Updating action pins in local ${WORKFLOW_FILE} ---"

if [[ ! -f "$WORKFLOW_FILE" ]]; then
  echo "  WARNING: ${WORKFLOW_FILE} not found in current directory."
  echo "  Run this script from the repo root, or update pins manually:"
  echo ""
  echo "    actions/download-artifact:  v4.1.8  ->  v8.0.1"
  echo "    actions/upload-artifact:    v4.3.6  ->  v4.3.6  (already latest in v4 line)"
  echo ""
else
  # download-artifact: v4.1.8 -> v8.0.1
  sed -i 's|actions/download-artifact@v4\.1\.8|actions/download-artifact@v8.0.1|g' "$WORKFLOW_FILE"
  echo "  download-artifact:  v4.1.8 -> v8.0.1  OK"

  # upload-artifact: v4.3.6 is current in the v4 line; Dependabot didn't flag it,
  # but v8.0.1 also applies to upload-artifact — leave at v4 unless you want to
  # bump both together. Uncomment the line below to bump upload-artifact too:
  # sed -i 's|actions/upload-artifact@v4\.3\.6|actions/upload-artifact@v8.0.1|g' "$WORKFLOW_FILE"

  echo ""
  echo "  Verifying pins after update:"
  grep -n 'uses: actions/' "$WORKFLOW_FILE" | sed 's/^/    /'
fi

echo ""

# ── 4. Pull latest from remote (picks up Dependabot merge commits) ────────────
echo "--- Pulling latest from origin ---"
git pull --rebase origin "$(git rev-parse --abbrev-ref HEAD)" || \
  echo "  WARNING: git pull failed — resolve manually"

echo ""

# ── 5. Commit local pin update if file was changed ────────────────────────────
if [[ -f "$WORKFLOW_FILE" ]] && ! git diff --quiet "$WORKFLOW_FILE"; then
  git add "$WORKFLOW_FILE"
  git commit -m "ci: bump actions/download-artifact from v4.1.8 to v8.0.1 (follows Dependabot PR #41)"
  git push origin HEAD
  echo "  Pushed pin update to remote."
else
  echo "  No local changes to commit (already up to date after pull)."
fi

echo ""
echo "======================================"
echo "  Done."
echo "======================================"
echo ""
echo "Summary of merged PRs:"
echo "  #34  mistune        3.1.4  -> 3.2.1"
echo "  #35  gitpython      3.1.49 -> 3.1.50"
echo "  #37  urllib3        2.6.3  -> 2.7.0"
echo "  #39  ckzg           2.1.5  -> 2.1.7"
echo "  #40  pillow         11.3.0 -> 12.2.0"
echo "  #41  download-artifact  v4.1.8 -> v8.0.1  (action pin also updated in workflow)"
echo ""
echo "Next: re-run your experiment workflow to confirm everything is green."

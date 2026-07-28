#!/usr/bin/env bash
# .github/scripts/setup_env.sh
# Shared environment bootstrap used by every pipeline job.
# Usage: bash .github/scripts/setup_env.sh [--julia]
#
# Sets:
#   PYTHONPATH  → repo workspace root
#   RESULTS_DIR → hypatiax/data/results/
#
# Flags:
#   --julia   also installs Julia via julia-actions (must be in calling job's steps)

set -euo pipefail

echo "PYTHONPATH=${GITHUB_WORKSPACE}" >> "${GITHUB_ENV}"
echo "RESULTS_DIR=${GITHUB_WORKSPACE}/hypatiax/data/results" >> "${GITHUB_ENV}"

pip install --quiet -r requirements.txt

if [[ "${1:-}" == "--jupyter" ]]; then
  pip install --quiet notebook nbconvert
fi

echo "✓ Environment ready (PYTHONPATH + RESULTS_DIR set)"

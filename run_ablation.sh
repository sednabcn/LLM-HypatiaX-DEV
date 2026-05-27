RESULTS_DIR="hypatiax/data/results"

BENCH_DIR="${RESULTS_DIR}/comparison_results/feynman-tests/exp2"
EXTRAP_DIR="${RESULTS_DIR}/comparison_results/feynman-tests/exp2_extrap"

python3 .github/scripts/merge_extrap_into_benchmark.py \
  --benchmark-dir "${BENCH_DIR}" \
  --extrap-dir "${EXTRAP_DIR}" \
  --output "${BENCH_DIR}/ablation_paired.json"

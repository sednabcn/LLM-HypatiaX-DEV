# validation_paper_config.py  — values sourced from repro.yaml v3.0
#
# Source mapping (all values must match repro.yaml exactly):
#   PYSR_TIMEOUT         = repro.yaml: timeouts.pysr_attempt_seconds  (1100)
#   METHOD_TIMEOUT       = repro.yaml: timeouts.method_seconds         (900)
#   EQUATION_WALL_CLOCK  = repro.yaml: timeouts.equation_wall_clock    (1200)
#   POPULATIONS          = repro.yaml: pysr.populations                (30)
#   N_ITERATIONS         = repro.yaml: pysr.niterations                (1000)
#   PYSR_POPULATION_SIZE = repro.yaml: pysr.population_size            (33)
#   PYSR_PARSIMONY       = repro.yaml: pysr.parsimony                  (0.01)
#   PYSR_MAXSIZE         = repro.yaml: pysr.maxsize                    (30)
#   PYSR_PARALLELISM     = repro.yaml: pysr.parallelism                (multithreading)
#   NN_SEED              = repro.yaml: seeds.default                   (42)
#   PYSR_SEED            = repro.yaml: seeds.pysr_seed                 (42)
#   LLM_MODEL            = repro.yaml: llm_model
#   ENGINE_NAME          = repro.yaml: engine.name
#   PCA_TRAIN_FRAC       = repro.yaml: pca_train_frac                  (0.4)
#   NN_TIME_LIMIT        = repro.yaml: nn_time_limit                   (120)
#
# Called as a pipeline step ("validate-paper-config") in run_all_checkpoint.py
# Phase 0 · Setup, and as a dedicated CI step in .github/workflows/ci.yml
# for both full-pipeline and slow-pipeline jobs — runs before any experiment.
#
# Exit codes: 0 = all correct; 1 = one or more vars wrong or unset.
import os

# ---------------------------------------------------------------------------
# Ground-truth values from repro.yaml v3.0
# ---------------------------------------------------------------------------
PAPER_CONFIG: dict[str, str] = {
    # ── Timeouts ──────────────────────────────────────────────────────────
    "PYSR_TIMEOUT":        "1100",   # timeouts.pysr_attempt_seconds
    "METHOD_TIMEOUT":      "900",    # timeouts.method_seconds
    "EQUATION_WALL_CLOCK": "1200",   # timeouts.equation_wall_clock
    # ── PySR search parameters ────────────────────────────────────────────
    "POPULATIONS":         "30",     # pysr.populations
    "N_ITERATIONS":        "1000",   # pysr.niterations
    "PYSR_POPULATION_SIZE": "33",    # pysr.population_size  ← previously unvalidated
    "PYSR_PARSIMONY":      "0.01",   # pysr.parsimony        ← previously unvalidated
    "PYSR_MAXSIZE":        "30",     # pysr.maxsize          ← previously unvalidated
    "PYSR_PARALLELISM":    "multithreading",  # pysr.parallelism  ← previously unvalidated
    # ── Seeds ─────────────────────────────────────────────────────────────
    "NN_SEED":             "42",     # seeds.default
    "PYSR_SEED":           "42",     # seeds.pysr_seed
    # ── LLM ───────────────────────────────────────────────────────────────
    "LLM_MODEL":           "claude-sonnet-4-5",  # llm_model
    # ── Engine / misc ─────────────────────────────────────────────────────
    "ENGINE_NAME":         "hybrid_system_v50_2",  # engine.name  ← previously unvalidated
    "PCA_TRAIN_FRAC":      "0.4",   # pca_train_frac        ← previously unvalidated
    "NN_TIME_LIMIT":       "120",    # nn_time_limit         ← previously unvalidated
}

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
SEP = "=" * 68

print(SEP)
print("PAPER CONFIGURATION VALIDATION  (repro.yaml v3.0)")
print(SEP)

failures: list[str] = []
for var, expected in PAPER_CONFIG.items():
    actual = os.environ.get(var)
    if actual == expected:
        print(f"  ✓ {var}={actual}")
    elif actual is None:
        print(f"  ✗ {var}  NOT SET  (expected {expected})")
        failures.append(var)
    else:
        print(f"  ⚠ {var}={actual!r}  (expected {expected!r}  ← FIX THIS)")
        failures.append(var)

print(SEP)
if not failures:
    print("✓ Paper configuration is CORRECT")
    print("  All env vars match repro.yaml v3.0 — results will match paper targets.")
else:
    print(f"✗ Paper configuration is INCORRECT  ({len(failures)} var(s) wrong or unset)")
    print("  Set the following before running experiments:")
    for var in failures:
        print(f"    export {var}={PAPER_CONFIG[var]}")
    print()
    print("  In CI these are set by the global env: block in ci.yml.")
    print("  Locally, re-run the Runtime Config cell (cell 2 / 0-C) or source setup_env.sh.")
    raise SystemExit("Abort: fix paper configuration before running experiments.")
print(SEP)

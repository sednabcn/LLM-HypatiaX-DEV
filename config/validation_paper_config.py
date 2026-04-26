# validate_paper_config.py  — values sourced from repro.yaml
# PYSR_TIMEOUT=1100   (repro.yaml: timeouts.pysr_attempt_seconds)
# METHOD_TIMEOUT=900  (repro.yaml: timeouts.method_seconds)
# POPULATIONS=30      (repro.yaml: pysr.populations)
# N_ITERATIONS=1000   (repro.yaml: pysr.niterations)
# NN_SEED=42          (repro.yaml: seeds.default)
# PYSR_SEED=42        (repro.yaml: seeds.pysr_seed)
# LLM_MODEL           (repro.yaml: llm_model)
import os

PAPER_CONFIG = {
    "PYSR_TIMEOUT":   "1100",
    "METHOD_TIMEOUT": "900",
    "POPULATIONS":    "30",
    "N_ITERATIONS":   "1000",
    "NN_SEED":        "42",
    "PYSR_SEED":      "42",
    "LLM_MODEL":      "claude-sonnet-4-20250514",
}

print("=" * 68)
print("PAPER CONFIGURATION VALIDATION  (repro.yaml v3.0)")
print("=" * 68)

all_ok = True
for var, expected in PAPER_CONFIG.items():
    actual = os.environ.get(var)
    if actual == expected:
        print(f"  \u2713 {var}={actual}")
    elif actual is None:
        print(f"  \u2717 {var} not set  (expected {expected})")
        all_ok = False
    else:
        print(f"  \u26a0 {var}={actual}  (expected {expected}  \u2190 FIX THIS)")
        all_ok = False

print("=" * 68)
if all_ok:
    print("\u2713 Paper configuration is CORRECT")
    print("  Results will match the paper targets.")
else:
    print("\u2717 Paper configuration is INCORRECT")
    print("  Run the Runtime Config cell (cell 2 / 0-C) first, or set:")
    for var, val in PAPER_CONFIG.items():
        if os.environ.get(var) != val:
            print(f"    export {var}={val}")
    raise SystemExit("Abort: fix paper configuration before running experiments.")
print("=" * 68)

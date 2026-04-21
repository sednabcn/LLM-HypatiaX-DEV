import sys
import os
from protocols.universal_protocol import run_protocol
from core.runners.common import run_task


def run():
    # Read seed from env — set by run_all.py and passed to every child process.
    # Falls back to 42 so a direct `python3 experiment_protocol_ablation_exp1.py`
    # still works without any env setup.
    seed = int(os.environ.get("PYSR_SEED", os.environ.get("NN_SEED", 42)))

    config = {
        "name": "ablation_exp1",
        "seed": seed,          # propagated into run_task → SCRIPT_MAP target
    }
    result = run_protocol(config, run_task)

    # The orchestrator (this script) decides whether to propagate failure to
    # the shell.  run_protocol itself never calls sys.exit, so partial results
    # from other configs in the same process are never lost.
    if not result.get("success", False):
        reason = result.get("reason") or result.get("error") or "unknown"
        print(f"[FATAL] ablation_exp1 failed: {reason}", file=sys.stderr)
        sys.exit(1)

    return result


if __name__ == "__main__":
    run()

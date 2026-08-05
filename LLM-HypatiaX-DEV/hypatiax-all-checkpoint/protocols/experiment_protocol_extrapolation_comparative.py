"""
experiment_protocol_extrapolation_comparative.py
=================================================

Protocol entry-point for the HypatiaX extrapolation + comparative benchmark.

What this file does
-------------------
Wraps ``run_comparative_suite_benchmark_v2.py`` (the six-method comparative
runner) and ``experiment_protocol_benchmark_v2.py`` (the Feynman / SRBench
protocol) into a single importable module that follows the same naming
convention as the other ``experiment_protocol_*.py`` files in this package.

It can be:
  * **imported** by orchestration scripts that need the ``BenchmarkProtocol``
    class or the ``ProtocolBenchmarkSuite`` runner directly.
  * **run as a script** (``python experiment_protocol_extrapolation_comparative.py``),
    in which case it delegates to ``main()`` in the comparative runner,
    forwarding all CLI arguments unchanged.

Previous version imported ``hypatiax.protocols.universal_protocol`` and
``hypatiax.core.runners.common`` — neither of which exists in the codebase.
Those imports have been replaced with direct references to the actual modules.

CI / run_all.sh usage
---------------------
This file is not referenced directly in ``ci_experiment.yml``.
The CI dispatches to ``run_comparative_suite_benchmark_v2.py`` for the
Feynman comparative experiments (exp2_feynman, exp2, exp2b, …).
This module exists for orchestrators and notebooks that need a single
importable protocol object covering both the Feynman benchmark and the
comparative method suite.

Environment variables honoured (passed through to the runner)
-------------------------------------------------------------
  TASK_IDS          Space/comma-separated Feynman IDs  (shard filter)
  SHARD_IDS         Space/comma-separated domain keys  (domain filter)
  PYSR_SEED         Random seed for PySR / experiments
  EXPERIMENT_SEED   Fallback seed
  METHOD_TIMEOUT    Per-method hard timeout in seconds (default 900)
  PYSR_TIMEOUT      PySR subprocess timeout in seconds (default 1100)
  FEYNMAN_SAMPLES   Data points per equation           (default 200)
  REPRO_CFG         Path to repro.yaml
"""

from __future__ import annotations

import os as _os
import pathlib as _pathlib
import sys as _sys

# ── sys.path bootstrap ────────────────────────────────────────────────────
# Ensures hypatiax.* imports resolve whether this file is run directly
# or imported by run_all_checkpoint.py / an orchestrator notebook.
_PROTO_DIR = _pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _pathlib.Path(_os.environ.get("REPRO_ROOT", str(_PROTO_DIR.parent)))
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "hypatiax")]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _os, _pathlib, _sys, _PROTO_DIR, _REPO_ROOT, _p


# ── Public re-exports ─────────────────────────────────────────────────────
# Import the two objects most likely to be needed by orchestrators so they
# can do:
#   from experiment_protocol_extrapolation_comparative import BenchmarkProtocol
#   from experiment_protocol_extrapolation_comparative import ProtocolBenchmarkSuite

try:
    # Post-restructure package layout (preferred)
    from hypatiax.protocols.experiment_protocol_benchmark_v2 import (
        BenchmarkProtocol,
    )
except ImportError:
    # Pre-restructure / flat layout
    from protocols.experiment_protocol_benchmark_v2 import (  # type: ignore[no-redef]
        BenchmarkProtocol,
    )

try:
    from hypatiax.experiments.benchmarks.run_comparative_suite_benchmark_v2 import (
        ProtocolBenchmarkSuite,
        main as _runner_main,
    )
except ImportError:
    # Same directory — common when run directly inside experiments/benchmarks/
    from run_comparative_suite_benchmark_v2 import (  # type: ignore[no-redef]
        ProtocolBenchmarkSuite,
        main as _runner_main,
    )


# ── Entry point ───────────────────────────────────────────────────────────

def run() -> None:
    """
    Run the extrapolation comparative benchmark.

    Delegates to ``main()`` in ``run_comparative_suite_benchmark_v2``,
    which parses ``sys.argv`` and runs the full six-method suite against
    the Feynman (or SRBench) protocol.

    All CLI flags supported by the runner are forwarded automatically,
    e.g.::

        python experiment_protocol_extrapolation_comparative.py \\
            --benchmark feynman --noiseless --methods 1 2 3 --verbose
    """
    _runner_main()


if __name__ == "__main__":
    run()

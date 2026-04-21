import subprocess, sys, os, json
from pathlib import Path

# Map protocol config names → actual benchmark scripts
SCRIPT_MAP = {
    "nguyen12_exp3":        "hypatiax/experiments/benchmarks/exp3_nguyen12_hybrid50v_02.py",
    "nguyen12_exp3b":       "hypatiax/experiments/benchmarks/exp3_nguyen12_hybrid50v_02_seed_123.py",
    "ablation_exp1":        "hypatiax/experiments/benchmarks/exp1_ablation_populations_30_updated.py",
    "defi_v3":              "hypatiax/experiments/benchmarks/hypatiax_defi_benchmark_v3c.py",
    "feynman_exp2":         "hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_v2.py",
    "noise_sweep":          "hypatiax/experiments/benchmarks/run_noise_sweep_benchmark.py",
    "hybrid_routing":       "hypatiax/experiments/benchmarks/run_hybrid_system_benchmark.py",
    "instability_rf02_04":  "hypatiax/experiments/benchmarks/run_dual_sweep_benchmarks.py",
    "extrap_comparative":   "hypatiax/experiments/benchmarks/run_dual_condition_benchmark.py",
    "provenance_audit":     "scripts/patches/provenance_audit.py",
}

def run_task(config):
    name   = config.get("name", "")
    script = SCRIPT_MAP.get(name)

    if not script:
        print(f"  ⚠  run_task: no script mapped for config name '{name}' — skipping")
        return {"status": "no_script", "name": name}

    script_path = Path(script)
    if not script_path.exists():
        print(f"  ⚠  run_task: script not found: {script}")
        return {"status": "missing_script", "script": script}

    # Pass through any extra args from config
    extra_args = config.get("args", [])

    # Inject FAST-mode speed args for scripts that support argparse flags
    import os
    fast_args = []
    script_name = str(script_path)
    _supports_timeout = any(s in script_name for s in [
        "run_dual_sweep", "run_noise_sweep",
        "run_dual_condition", "run_hybrid_system",
        "hypatiax_defi_benchmark_v3c",
    ])
    if _supports_timeout:
        pysr_timeout   = os.environ.get("PYSR_TIMEOUT",   "1100")
        method_timeout = os.environ.get("METHOD_TIMEOUT",  "900")
        fast_args = ["--pysr-timeout", pysr_timeout,
                     "--method-timeout", method_timeout]

    cmd = [sys.executable, str(script_path)] + fast_args + extra_args

    print(f"  → running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    return {
        "status":      "ok" if result.returncode == 0 else "failed",
        "returncode":  result.returncode,
        "script":      script,
        "name":        name,
    }

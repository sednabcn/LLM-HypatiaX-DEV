"""Print key fields from config/repro.yaml for CI environment-check step.

Usage (from repo root):
    python3 .github/scripts/print_repro.py

Reads GITHUB_WORKSPACE from the environment (set automatically by Actions).
Falls back to the current working directory when run locally.
"""

import os
import pathlib
import sys

try:
    import yaml
except ImportError:
    sys.exit("ERROR: PyYAML not installed — run: pip install pyyaml")

workspace = pathlib.Path(os.environ.get("GITHUB_WORKSPACE", "."))
config_path = workspace / "config" / "repro.yaml"

try:
    raw = config_path.read_text()
except FileNotFoundError:
    sys.exit(f"ERROR: config file not found at {config_path}")

r = yaml.safe_load(raw)
t = r.get("timeouts", {})
s = r.get("pysr", {})

print("  run_id                       :", r.get("run_id"))
print("  timeouts.feynman_pysr_seconds:", t.get("feynman_pysr_seconds"))
print("  timeouts.fit_wall_timeout    :", t.get("fit_wall_timeout"))
print("  timeouts.fit_grace_secs      :", t.get("fit_grace_secs"))
print("  timeouts.method_seconds      :", t.get("method_seconds"))
print("  pysr.populations             :", s.get("populations"))
print("  pysr.population_size         :", s.get("population_size"))
print("  pysr.niterations             :", s.get("niterations"))
print("  pysr.maxsize                 :", s.get("maxsize"))
print("  pysr.parsimony               :", s.get("parsimony"))
print("  seeds.pysr_seed              :", r.get("seeds", {}).get("pysr_seed"))

#!/usr/bin/env python3
"""
diagnose_hds_v50_2.py
======================

Root-causes the HybridDiscoverySystem v50_2 (tools) ENV-FAIL / "not available"
bug seen in check_run_health.py (Bug 1 / env-probe) and confirmed in
investigate_section_10_7.py (30/30 ENV-FAIL, r2=0.0 across every shard).

Unlike the health-check scripts, this does NOT swallow the failure into a
generic "not available" string. It captures full tracebacks, isolates
whether the failure is import-time vs. instantiate-time vs. run-time, and
checks the usual suspects for a "(tools)" method that fails uniformly across
every test case (a systematic env/config issue rather than per-equation
model failure):

  1. Can the module be found and imported at all?
  2. Are its declared dependencies importable?
  3. Are expected env vars / API keys / config files present?
  4. Does instantiating the class succeed?
  5. Does a minimal smoke-test call succeed?

Usage:
    cd ~/Downloads/GITHUB/LLM-HypatiaX-DEV
    python diagnose_hds_v50_2.py

    # If you know the exact module path already:
    python diagnose_hds_v50_2.py --module-path path/to/hybrid_system_v50_2.py

    # If you know the class/callable name already (skip guessing):
    python diagnose_hds_v50_2.py --class-name HybridDiscoverySystem

Output:
    Human-readable report to stdout, plus a machine-readable JSON report
    written to diagnose_hds_v50_2_report.json in the current directory.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

REPORT = {
    "search": {},
    "import": {},
    "dependencies": {},
    "env": {},
    "instantiate": {},
    "smoke_test": {},
    "verdict": None,
}


def find_candidate_files(repo_root: Path, explicit_path: str | None):
    """Locate hybrid_system_v50_2.py (or the closest match) under repo_root."""
    if explicit_path:
        p = Path(explicit_path)
        found = [p] if p.exists() else []
        REPORT["search"]["mode"] = "explicit"
        REPORT["search"]["explicit_path"] = str(p)
        REPORT["search"]["found"] = [str(f) for f in found]
        return found

    patterns = ["*hybrid_system_v50_2*.py", "*hybrid*discovery*.py", "*v50_2*.py"]
    found = []
    for pattern in patterns:
        found.extend(sorted(repo_root.rglob(pattern)))
    # de-dup while preserving order
    seen = set()
    unique = []
    for f in found:
        if f not in seen and "test" not in f.parts and ".git" not in f.parts:
            seen.add(f)
            unique.append(f)

    REPORT["search"]["mode"] = "glob"
    REPORT["search"]["patterns"] = patterns
    REPORT["search"]["found"] = [str(f) for f in unique]
    return unique


def try_import(module_file: Path):
    """Import the module directly from its file path, capturing full traceback."""
    module_name = module_file.stem
    entry = {"file": str(module_file), "success": False, "traceback": None, "error_type": None}
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # this is where ImportErrors/etc. actually surface
        entry["success"] = True
        return module, entry
    except Exception as e:
        entry["error_type"] = type(e).__name__
        entry["error_message"] = str(e)
        entry["traceback"] = traceback.format_exc()
        return None, entry


def check_pip_deps(module_file: Path):
    """
    Best-effort scan of import statements in the file to check which
    third-party packages are actually installed in THIS interpreter.
    Flags likely culprits for a uniform ENV-FAIL across all 30 tests.
    """
    results = {}
    try:
        text = module_file.read_text(errors="ignore")
    except Exception as e:
        return {"error": f"could not read file: {e}"}

    import re
    imports = set()
    for line in text.splitlines():
        m = re.match(r"^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)", line)
        if m:
            top_level = m.group(1).split(".")[0]
            imports.add(top_level)

    stdlib_ish = {
        "os", "sys", "re", "json", "math", "time", "typing", "itertools",
        "collections", "functools", "pathlib", "logging", "subprocess",
        "dataclasses", "abc", "warnings", "copy", "random", "traceback",
    }

    for pkg in sorted(imports - stdlib_ish):
        try:
            __import__(pkg)
            results[pkg] = "OK"
        except Exception as e:
            results[pkg] = f"MISSING/FAILED: {type(e).__name__}: {e}"

    return results


def check_env_and_config(repo_root: Path, module_file: Path):
    """
    Looks for env-var references and nearby config/credentials files that a
    'tools'-tagged method commonly depends on (API keys, tool endpoints,
    solver binaries, etc).
    """
    results = {"env_vars_referenced": {}, "config_files_nearby": []}
    try:
        text = module_file.read_text(errors="ignore")
    except Exception:
        text = ""

    import re
    env_var_names = set(re.findall(r"os\.environ(?:\.get)?\(\s*['\"]([A-Z0-9_]+)['\"]", text))
    env_var_names |= set(re.findall(r"os\.getenv\(\s*['\"]([A-Z0-9_]+)['\"]", text))
    for name in sorted(env_var_names):
        results["env_vars_referenced"][name] = "SET" if os.environ.get(name) else "NOT SET"

    # look for likely config/credential files near the module or repo root
    candidate_names = [
        ".env", "config.yaml", "config.yml", "config.json",
        "tools_config.json", "credentials.json",
    ]
    for name in candidate_names:
        for base in {module_file.parent, repo_root}:
            candidate = base / name
            if candidate.exists():
                results["config_files_nearby"].append(str(candidate))

    return results


def try_instantiate(module, class_name_hint: str | None):
    """Find a plausible class in the module and try to instantiate it."""
    entry = {"class_tried": None, "success": False, "traceback": None}

    candidates = []
    if class_name_hint and hasattr(module, class_name_hint):
        candidates.append(class_name_hint)
    else:
        for name in dir(module):
            if "Hybrid" in name or "Discovery" in name:
                obj = getattr(module, name)
                if isinstance(obj, type):
                    candidates.append(name)

    if not candidates:
        entry["error"] = "No plausible class found (looked for names containing 'Hybrid' or 'Discovery'). Pass --class-name explicitly."
        return None, entry

    for name in candidates:
        entry["class_tried"] = name
        cls = getattr(module, name)
        try:
            instance = cls()  # no-arg first
            entry["success"] = True
            entry["instantiation"] = "no-arg constructor succeeded"
            return instance, entry
        except TypeError as te:
            entry["no_arg_error"] = str(te)
            # try common kwarg patterns used by these harnesses
            for kwargs in ({"config": {}}, {"mode": "core"}, {"use_tools": True}):
                try:
                    instance = cls(**kwargs)
                    entry["success"] = True
                    entry["instantiation"] = f"succeeded with kwargs={kwargs}"
                    return instance, entry
                except Exception:
                    continue
            entry["traceback"] = traceback.format_exc()
        except Exception:
            entry["traceback"] = traceback.format_exc()
    return None, entry


def try_smoke_test(instance):
    """
    Try calling the most likely 'run'/'discover'/'fit'/'predict' method with
    trivial dummy data, just to see whether the failure happens at call-time
    (e.g. inside a try/except that swallows it upstream in the harness).
    """
    entry = {"method_tried": None, "success": False, "traceback": None}
    if instance is None:
        entry["skipped"] = "no instance to test"
        return entry

    method_names = ["run", "discover", "fit_predict", "solve", "fit", "predict"]
    for name in method_names:
        if hasattr(instance, name):
            entry["method_tried"] = name
            try:
                import numpy as np
                X = np.linspace(0.1, 1.0, 20).reshape(-1, 1)
                y = X.flatten() * 2.0
                getattr(instance, name)(X, y)
                entry["success"] = True
                return entry
            except Exception:
                entry["traceback"] = traceback.format_exc()
                return entry
    entry["error"] = "No method named run/discover/fit_predict/solve/fit/predict found."
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", default=".", help="Repo root to search from (default: cwd)")
    parser.add_argument("--module-path", default=None, help="Exact path to hybrid_system_v50_2.py, if known")
    parser.add_argument("--class-name", default=None, help="Exact class name to instantiate, if known")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    print(f"=== HybridDiscoverySystem v50_2 ENV-FAIL Diagnostic ===")
    print(f"Repo root: {repo_root}\n")

    # 1. Locate the module
    candidates = find_candidate_files(repo_root, args.module_path)
    if not candidates:
        print("[FATAL] Could not find hybrid_system_v50_2.py under repo root.")
        print("        Pass --module-path explicitly, or --repo-root pointing at the right checkout.")
        REPORT["verdict"] = "MODULE_NOT_FOUND"
        write_report()
        sys.exit(1)

    print(f"Found {len(candidates)} candidate file(s):")
    for c in candidates:
        print(f"  - {c}")
    module_file = candidates[0]
    print(f"\nUsing: {module_file}\n")

    # 2. Import
    print("--- Step 1: Import ---")
    module, import_entry = try_import(module_file)
    REPORT["import"] = import_entry
    if not import_entry["success"]:
        print(f"[FAIL] Import failed: {import_entry['error_type']}: {import_entry.get('error_message')}")
        print(import_entry["traceback"])
        REPORT["verdict"] = "IMPORT_FAILURE"
    else:
        print("[OK] Module imported successfully in THIS environment.")
    print()

    # 3. Dependency check (runs regardless, useful even if import succeeded here
    #    but fails in the harness's actual runtime environment/venv)
    print("--- Step 2: Third-party dependency check ---")
    dep_results = check_pip_deps(module_file)
    REPORT["dependencies"] = dep_results
    if not dep_results:
        print("  (no third-party imports detected)")
    for pkg, status in dep_results.items():
        flag = "OK" if status == "OK" else "MISSING"
        print(f"  [{flag:7}] {pkg}: {status}")
    print()

    # 4. Env / config check
    print("--- Step 3: Environment variables & config files ---")
    env_results = check_env_and_config(repo_root, module_file)
    REPORT["env"] = env_results
    if env_results["env_vars_referenced"]:
        for name, status in env_results["env_vars_referenced"].items():
            print(f"  [{status:8}] {name}")
    else:
        print("  (no os.environ / os.getenv references found in this file)")
    if env_results["config_files_nearby"]:
        print("  Nearby config files found:")
        for f in env_results["config_files_nearby"]:
            print(f"    - {f}")
    else:
        print("  No nearby .env/config.* files found.")
    print()

    # 5. Instantiate (only if import succeeded)
    instance = None
    if module is not None:
        print("--- Step 4: Instantiation ---")
        instance, inst_entry = try_instantiate(module, args.class_name)
        REPORT["instantiate"] = inst_entry
        if inst_entry.get("success"):
            print(f"[OK] Instantiated class '{inst_entry['class_tried']}' ({inst_entry['instantiation']})")
        else:
            print(f"[FAIL] Could not instantiate (tried: {inst_entry.get('class_tried')})")
            if inst_entry.get("traceback"):
                print(inst_entry["traceback"])
            elif inst_entry.get("error"):
                print(f"  {inst_entry['error']}")
        print()

        # 6. Smoke test
        print("--- Step 5: Smoke test call ---")
        smoke_entry = try_smoke_test(instance)
        REPORT["smoke_test"] = smoke_entry
        if smoke_entry.get("success"):
            print(f"[OK] Method '{smoke_entry['method_tried']}' ran without raising.")
        elif smoke_entry.get("skipped"):
            print(f"[SKIPPED] {smoke_entry['skipped']}")
        else:
            print(f"[FAIL] Method '{smoke_entry.get('method_tried')}' raised:")
            if smoke_entry.get("traceback"):
                print(smoke_entry["traceback"])
            elif smoke_entry.get("error"):
                print(f"  {smoke_entry['error']}")
        print()

    # 7. Verdict
    print("=== Verdict ===")
    if not import_entry["success"]:
        verdict = "Import fails in THIS environment -> matches 'not available'/ENV-FAIL. Fix the traceback above (likely a missing/broken dependency or bad relative import path)."
    elif any(v != "OK" for v in dep_results.values()):
        verdict = "Module imports here, but a dependency is missing/broken -> if the harness's venv differs from this one, that mismatch is the likely cause of ENV-FAIL. Check the harness's actual interpreter/venv against this report."
    elif env_results["env_vars_referenced"] and any(s == "NOT SET" for s in env_results["env_vars_referenced"].values()):
        verdict = "Module imports and deps are OK, but a referenced env var is NOT SET here -> if the harness's process doesn't have this var either, that's the cause. Check how/where the harness sets its environment."
    elif REPORT.get("instantiate", {}).get("success") is False:
        verdict = "Import and deps are fine, but instantiation fails -> likely a constructor signature or missing constructor argument the harness isn't providing correctly."
    elif REPORT.get("smoke_test", {}).get("success") is False and REPORT.get("smoke_test", {}).get("traceback"):
        verdict = "Instantiation succeeds but the run/discover/fit call fails -> this is very likely the actual bug; see traceback under Step 5 above."
    else:
        verdict = "No failure reproduced in this environment. The bug may be specific to the harness's subprocess/venv/working-directory, or triggered by particular input data not exercised by this smoke test. Recommend: instrument the harness itself to log the raw exception instead of coercing it to 'not available'."
    print(verdict)
    REPORT["verdict"] = verdict

    write_report()


def write_report():
    out_path = Path("diagnose_hds_v50_2_report.json")
    with open(out_path, "w") as f:
        json.dump(REPORT, f, indent=2, default=str)
    print(f"\nMachine-readable report written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()

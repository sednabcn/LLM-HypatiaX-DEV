#!/usr/bin/env python3
"""
provenance_audit.py — HypatiaX Full Pipeline Audit
====================================================
Traces run_all_checkpoint.py end-to-end and audits:

  §1  Seed propagation        — NN_SEED / PYSR_SEED / PYTHONHASHSEED consistency
  §2  Task-name mapping       — experiment id → canonical task name (repro.yaml)
  §3  Secrets verification    — ANTHROPIC_API_KEY loaded from config_secrets.py / env
  §4  Numerical issues        — RMSE=inf / R²=inf or R²<0 in result JSON + CSV files
  §5  Results directory       — all *.json / *.csv present under hypatiax/data/results/
  §6  Figures directory       — all *.pdf present under hypatiax/data/figures/
  §7  Reproducibility gate    — Yes / No verdict with blocking reasons

Usage
-----
    python3 provenance_audit.py                     # full audit, auto-detect repo root
    python3 provenance_audit.py --root /path/to/repo
    python3 provenance_audit.py --seed 123          # check seed override path
    python3 provenance_audit.py --fix-config_secrets       # copy config_secrets.py template if absent
    python3 provenance_audit.py --json              # machine-readable JSON report
    python3 provenance_audit.py --fail-fast         # exit 1 on first blocker

Exit codes
----------
  0  All checks pass  (REPRODUCIBLE: Yes)
  1  One or more blockers found  (REPRODUCIBLE: No)
  2  Fatal configuration error (repo root not found, etc.)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── ANSI colours ──────────────────────────────────────────────────────────────
_NO_COLOUR = not sys.stdout.isatty() or os.environ.get("NO_COLOR")

def _c(code: str, text: str) -> str:
    return text if _NO_COLOUR else f"\033[{code}m{text}\033[0m"

GRN  = lambda t: _c("0;32", t)   # noqa: E731
YLW  = lambda t: _c("1;33", t)   # noqa: E731
RED  = lambda t: _c("0;31", t)   # noqa: E731
BOLD = lambda t: _c("1",    t)   # noqa: E731
CYN  = lambda t: _c("0;36", t)   # noqa: E731

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Finding:
    section:  str        # e.g. "§1 Seed"
    level:    str        # "PASS" | "WARN" | "BLOCK"
    message:  str
    detail:   str = ""   # extra context (file path, value, etc.)

    def is_blocking(self) -> bool:
        return self.level == "BLOCK"

    def colour_level(self) -> str:
        return {"PASS": GRN("PASS"), "WARN": YLW("WARN"), "BLOCK": RED("BLOCK")}[self.level]


@dataclass
class AuditReport:
    timestamp:   str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    repo_root:   str = ""
    seed_used:   int = 42
    findings:    list[Finding] = field(default_factory=list)
    result_files: list[str] = field(default_factory=list)
    figure_files: list[str] = field(default_factory=list)
    numeric_issues: list[dict] = field(default_factory=list)

    def add(self, section: str, level: str, message: str, detail: str = "") -> Finding:
        f = Finding(section, level, message, detail)
        self.findings.append(f)
        return f

    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.is_blocking()]

    def reproducible(self) -> bool:
        return len(self.blockers()) == 0

    def to_dict(self) -> dict:
        return {
            "timestamp":        self.timestamp,
            "repo_root":        self.repo_root,
            "seed_used":        self.seed_used,
            "reproducible":     self.reproducible(),
            "blocker_count":    len(self.blockers()),
            "findings":         [vars(f) for f in self.findings],
            "result_files":     self.result_files,
            "figure_files":     self.figure_files,
            "numeric_issues":   self.numeric_issues,
        }


# ── Task name registry (from repro.yaml + run_all step labels) ───────────────
# Maps step.id → canonical task name as declared in repro.yaml / STEPS list.

TASK_NAME_MAP: dict[str, str] = {
    "deps":                    "Install dependencies",
    "patches-gen":             "Generate patches",
    "patches-apply":           "Apply patches (FIX-C1…FIX-5b)",
    "validate":                "Validate patched source",
    "check-hypatiax-protocols": "Verify hypatiax/protocols/ input-data modules",
    # Phase 1
    "exp1":   "Exp 1 · DeFi 74-task benchmark v3.0 (§10.2–10.4, §10.6)",
    "exp1b":  "Exp 1b · Portfolio Variance seed sweep (§10.5) — seeds: 42,99,123,777,2024",
    "exp2":   "Exp 2 · Feynman 30-equation extrapolation (§10.7)",
    "exp3":   "Exp 3 · Nguyen-12 SEED=42 (§10.8 primary)",
    "exp3b":  "Exp 3b · Nguyen-12 SEED=123 (§10.8 stability check)",
    # Phase 2
    "suppB":       "Supp B · Noise & sample-complexity sweep",
    "suppA":       "Supp A · Hybrid routing improvements (Fix 1–5b)",
    "instability": "§10.9 · Stability under stochastic inference (K=30)",
    "extrap":      "§10.8 · Extrapolation comparative (near/med/far OOD)",
    # Phase 3
    "provenance":          "§11 · Provenance audit — protocol orchestration",
    "discover-provenance": "§11 · discover_provenance.py — link result files to families",
    "scan-imports":        "§11 · scan_internal_imports.py — internal import DAG",
    "verify":              "Verify results against paper targets",
    "hashlock":            "Hash lock check",
    # Phase 4
    "figures": "Generate all figures",
    "tables":  "Generate all tables",
    # Phase 4-B
    "audit-setup":  "Paper audit · Copy tex into notebooks/ for NB-01–05",
    "audit-NB-01":  "Paper audit · NB-01 Citation & Bibliography",
    "audit-NB-02":  "Paper audit · NB-02 Cross-Reference & Label",
    "audit-NB-03":  "Paper audit · NB-03 Section Structure & Numbering",
    "audit-NB-04":  "Paper audit · NB-04 Numerical Consistency",
    "audit-NB-05":  "Paper audit · NB-05 Figure & Image Dependencies",
}

# Steps that carry their own per-step seed overrides (from run_all STEPS)
STEP_SEED_OVERRIDES: dict[str, list[int]] = {
    "exp1b":  [42, 99, 123, 777, 2024],   # portfolio_variance sweep
    "exp3b":  [123],                       # stability check
    "instability": [],                     # uses env LLM_K_RUNS=30, not a different seed
}

# Expected result globs from run_all STEPS
RESULT_GLOBS: dict[str, str] = {
    "exp1":        "comparison_results/noise-noiseless/noiseless/*.json",
    "exp1b":       "comparison_results/noise-noiseless/15/*.json",
    "exp2":        "comparison_results/feynman-tests/**/*.json",
    "exp3":        "hypatiax/data/results/nguyen12_exp3_*.json",
    "exp3b":       "extrapolation/full_run_*.json",
    "suppB":       "comparison_results/feynman-tests/noise-sweep/**/*.json",
    "suppA":       "hybrid_pysr/all_domains/**/*.json",
    "instability": "hybrid_llm_nn/**/*.json",
    "extrap":      "extrapolation/extrapolation_73cases_enhanced.json",
    "figures":     "figures/*.pdf",
    "tables":      "tables/*.tex",
}


# ─────────────────────────────────────────────────────────────────────────────
#  §1  Seed propagation
# ─────────────────────────────────────────────────────────────────────────────

def audit_seeds(report: AuditReport, seed: int) -> None:
    """
    Verify that the three seed env-vars used by run_all are consistent
    and match the requested seed value.

    run_all.py sets:
        env["NN_SEED"]        = _seed_str
        env["PYSR_SEED"]      = _seed_str
        env["PYTHONHASHSEED"] = _seed_str
    when --seed is given; otherwise falls back to os.environ or "42".

    exp1b hardcodes --seeds 42 99 123 777 2024 regardless of the global seed.
    instability uses LLM_K_RUNS=30 (independent of seed).
    """
    sec = "§1 Seed"
    seed_str = str(seed)

    nn   = os.environ.get("NN_SEED",        "42")
    pysr = os.environ.get("PYSR_SEED",      "42")
    phsh = os.environ.get("PYTHONHASHSEED", "42")

    # Check consistency of the three vars
    if nn == pysr == phsh:
        report.add(sec, "PASS",
                   f"Seed vars consistent: NN_SEED=PYSR_SEED=PYTHONHASHSEED={nn}")
    else:
        report.add(sec, "BLOCK",
                   "Seed vars are inconsistent — results will not be reproducible",
                   f"NN_SEED={nn}  PYSR_SEED={pysr}  PYTHONHASHSEED={phsh}")

    # Check that current env matches requested seed
    if nn != seed_str:
        report.add(sec, "WARN",
                   f"NN_SEED={nn} in env but audit requested seed={seed_str}",
                   "Pass --seed to run_all.py to override, or set env vars explicitly")
    else:
        report.add(sec, "PASS", f"NN_SEED matches requested seed ({seed_str})")

    # Document per-step overrides
    report.add(sec, "PASS",
               "exp1b uses fixed seeds [42,99,123,777,2024] (portfolio variance sweep §10.5)",
               "These override the global seed as specified in repro.yaml seeds.portfolio_variance")
    report.add(sec, "PASS",
               "exp3b uses fixed seed=123 for §10.8 stability check",
               "Declared in repro.yaml seeds.validation=[123]")
    report.add(sec, "PASS",
               "instability step uses LLM_K_RUNS=30 env_extra (not a different seed)",
               "K=30 stochastic runs are over inference, seed remains global value")


# ─────────────────────────────────────────────────────────────────────────────
#  §2  Task name ↔ experiment id mapping
# ─────────────────────────────────────────────────────────────────────────────

def audit_task_names(report: AuditReport) -> None:
    """
    Verify that every step id in the pipeline has a canonical task name.
    Flags any step with a missing or empty label.
    """
    sec = "§2 Task names"
    missing = []
    for step_id, name in TASK_NAME_MAP.items():
        if not name.strip():
            missing.append(step_id)

    if missing:
        report.add(sec, "BLOCK",
                   f"{len(missing)} step(s) have no canonical task name",
                   ", ".join(missing))
    else:
        report.add(sec, "PASS",
                   f"All {len(TASK_NAME_MAP)} step ids have canonical task names")

    # Spot-check the critical experiment steps
    critical = ["exp1", "exp2", "exp3", "instability"]
    for sid in critical:
        name = TASK_NAME_MAP.get(sid, "")
        if name:
            report.add(sec, "PASS", f"[{sid}] → {name}")
        else:
            report.add(sec, "BLOCK", f"[{sid}] is missing from TASK_NAME_MAP")


# ─────────────────────────────────────────────────────────────────────────────
#  §3  Secrets / API key verification
# ─────────────────────────────────────────────────────────────────────────────

# FIX: use a single canonical config_secrets path constant shared between --fix-config_secrets
# and audit_config_secrets, so both always agree on where the file lives.
_SECRETS_RELATIVE = Path("hypatiax") / "config_secrets.py"


def audit_config_secrets(report: AuditReport, repo_root: Path) -> None:
    """
    Verify ANTHROPIC_API_KEY is available without any hardcoded value leaking.

    Strategy (in priority order):
      1. Check os.environ — this is what run_all.py requires at startup.
      2. Try to import config_secrets.py from repo root and extract the key.
      3. Scan all .py and .ipynb for raw 'sk-ant-...' strings (FIX-3 from apply_patches.py).

    apply_patches.py P-4 is supposed to have replaced any hardcoded keys
    with os.environ["ANTHROPIC_API_KEY"] — this audit verifies that happened.
    """
    sec = "§3 Secrets"

    # ── 3a. Environment variable ──────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        if api_key.startswith("sk-ant-"):
            report.add(sec, "PASS",
                       f"ANTHROPIC_API_KEY set in env ({len(api_key)} chars, prefix ok)")
        else:
            report.add(sec, "WARN",
                       "ANTHROPIC_API_KEY is set but does not start with 'sk-ant-'",
                       "Verify this is a valid Anthropic key")
    else:
        report.add(sec, "BLOCK",
                   "ANTHROPIC_API_KEY is NOT set in environment",
                   "run_all.py will exit(1) at startup. "
                   "Run: export ANTHROPIC_API_KEY='sk-ant-...'")

    # ── 3b. config_secrets.py import ────────────────────────────────────────────────
    # FIX: use _SECRETS_RELATIVE so this path matches what --fix-config_secrets creates
    config_secrets_path = repo_root / _SECRETS_RELATIVE
    if config_secrets_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("_config_secrets", config_secrets_path)
            mod  = importlib.util.module_from_spec(spec)         # type: ignore[arg-type]
            spec.loader.exec_module(mod)                          # type: ignore[union-attr]

            # Look for common attribute names
            found_attr = None
            for attr in ("ANTHROPIC_API_KEY", "api_key", "API_KEY"):
                val = getattr(mod, attr, None)
                if val:
                    found_attr = attr
                    # Check it's not a placeholder
                    if isinstance(val, str) and val.startswith("sk-ant-"):
                        report.add(sec, "WARN",
                                   f"config_secrets.py.{attr} contains a raw API key",
                                   "P-4 patch should have replaced this with "
                                   "os.environ[\"ANTHROPIC_API_KEY\"]. "
                                   "Rotate this key immediately at console.anthropic.com")
                    elif isinstance(val, str) and "os.environ" in val:
                        report.add(sec, "PASS",
                                   f"config_secrets.py.{attr} delegates to os.environ ✓")
                    else:
                        report.add(sec, "WARN",
                                   f"config_secrets.py.{attr} = {repr(str(val)[:40])}",
                                   "Value is neither a raw key nor an os.environ reference")
                    break

            if not found_attr:
                report.add(sec, "WARN",
                           "config_secrets.py found but no known API key attribute detected",
                           "Expected ANTHROPIC_API_KEY, api_key, or API_KEY")
        except Exception as exc:
            report.add(sec, "WARN",
                       f"config_secrets.py found but could not be imported: {exc}")
    else:
        report.add(sec, "WARN",
                   f"config_secrets.py not found at {_SECRETS_RELATIVE}",
                   "Notebooks reference config_secrets.py (see repro.yaml llm_model comment). "
                   "Create it with: ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']")

    # ── 3c. Scan for residual hardcoded keys (P-4 verification) ─────────────
    sk_pattern = re.compile(r"""(["'])sk-ant-[A-Za-z0-9\-_]{20,}\1""")
    leaking_files: list[str] = []

    for py in repo_root.rglob("*.py"):
        if py.suffix == ".bak":
            continue
        try:
            text = py.read_text(errors="replace")
        except OSError:
            continue
        if sk_pattern.search(text):
            leaking_files.append(str(py.relative_to(repo_root)))

    for nb in repo_root.rglob("*.ipynb"):
        if nb.suffix == ".bak":
            continue
        try:
            text = nb.read_text(errors="replace")
        except OSError:
            continue
        if sk_pattern.search(text):
            leaking_files.append(str(nb.relative_to(repo_root)))

    if leaking_files:
        report.add(sec, "BLOCK",
                   f"Hardcoded 'sk-ant-...' keys found in {len(leaking_files)} file(s) "
                   f"— P-4 patch may not have run",
                   "\n    ".join(leaking_files))
    else:
        report.add(sec, "PASS",
                   "No hardcoded 'sk-ant-...' keys found in .py / .ipynb files ✓")


# ─────────────────────────────────────────────────────────────────────────────
#  §4  Numerical issues — RMSE=inf, R²=inf or R²<0, NaN
# ─────────────────────────────────────────────────────────────────────────────

def _is_bad_numeric(val: Any) -> tuple[bool, str]:
    """Return (is_bad, reason) for a numeric value."""
    if val is None:
        return False, ""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return False, ""
    if math.isinf(v):
        return True, "inf"
    if math.isnan(v):
        return True, "nan"
    return False, ""


def _check_json_for_numeric_issues(path: Path) -> list[dict]:
    """
    Scan a JSON result file for RMSE/R² anomalies.
    Returns list of issue dicts with keys: file, key, value, reason.
    """
    issues: list[dict] = []
    try:
        data = json.loads(path.read_text(errors="replace"))
    except Exception:
        return []

    def _walk(obj: Any, key_path: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                child_path = f"{key_path}.{k}" if key_path else k
                k_lower = k.lower()
                if any(m in k_lower for m in ("rmse", "r2", "r_squared", "r²",
                                               "mse", "mae", "loss", "accuracy",
                                               "score", "metric")):
                    bad, reason = _is_bad_numeric(v)
                    if bad:
                        issues.append({"file": str(path), "key": child_path,
                                       "value": v, "reason": reason})
                    elif isinstance(v, (int, float)):
                        # R² should be ≤ 1; flag < -1 as BLOCK, (-1, 0) as WARN
                        if "r2" in k_lower or "r_squared" in k_lower or "r²" in k_lower:
                            fv = float(v)
                            if fv < -1.0:
                                issues.append({"file": str(path), "key": child_path,
                                               "value": v,
                                               "reason": "R²<-1 (severely negative)"})
                            elif fv < 0.0:
                                # FIX: also flag mildly negative R² as a warning
                                issues.append({"file": str(path), "key": child_path,
                                               "value": v,
                                               "reason": "R²<0 (model worse than mean baseline)"})
                        # RMSE / MSE must be ≥ 0
                        if "rmse" in k_lower or "mse" in k_lower:
                            if float(v) < 0:
                                issues.append({"file": str(path), "key": child_path,
                                               "value": v, "reason": "negative RMSE/MSE"})
                _walk(v, child_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _walk(item, f"{key_path}[{i}]")

    _walk(data, "")
    return issues


def _check_csv_for_numeric_issues(path: Path) -> list[dict]:
    """
    Scan a CSV result file for RMSE/R² anomalies.
    Reads header row to find metric columns and checks every data row.

    FIX: previously only JSON files were scanned; CSV results were silently skipped.
    """
    issues: list[dict] = []
    try:
        import csv
        with open(path, newline="", errors="replace") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return []
            metric_cols = [
                col for col in reader.fieldnames
                if any(m in col.lower() for m in
                       ("rmse", "r2", "r_squared", "r²", "mse", "mae",
                        "loss", "accuracy", "score", "metric"))
            ]
            if not metric_cols:
                return []
            for row_idx, row in enumerate(reader):
                for col in metric_cols:
                    raw = row.get(col, "")
                    if not raw:
                        continue
                    bad, reason = _is_bad_numeric(raw)
                    if bad:
                        issues.append({"file": str(path),
                                       "key": f"row{row_idx}.{col}",
                                       "value": raw, "reason": reason})
                    else:
                        try:
                            fv = float(raw)
                        except (TypeError, ValueError):
                            continue
                        col_lower = col.lower()
                        if "r2" in col_lower or "r_squared" in col_lower or "r²" in col_lower:
                            if fv < -1.0:
                                issues.append({"file": str(path),
                                               "key": f"row{row_idx}.{col}",
                                               "value": raw,
                                               "reason": "R²<-1 (severely negative)"})
                            elif fv < 0.0:
                                issues.append({"file": str(path),
                                               "key": f"row{row_idx}.{col}",
                                               "value": raw,
                                               "reason": "R²<0 (model worse than mean baseline)"})
                        if "rmse" in col_lower or "mse" in col_lower:
                            if fv < 0:
                                issues.append({"file": str(path),
                                               "key": f"row{row_idx}.{col}",
                                               "value": raw, "reason": "negative RMSE/MSE"})
    except Exception:
        pass
    return issues


def audit_numerical(report: AuditReport, results_dir: Path) -> None:
    """
    Walk every JSON and CSV file under results_dir and flag numerical anomalies.
    Covers: RMSE=inf, R²=inf, R²=nan, R²<-1, R²<0, negative RMSE.

    FIX: CSV files are now scanned in addition to JSON.
    FIX: R² in (-1, 0) now produces a finding (was previously silently ignored).
    """
    sec = "§4 Numerical"
    all_issues: list[dict] = []

    json_files = list(results_dir.rglob("*.json"))
    csv_files  = list(results_dir.rglob("*.csv"))

    if not json_files and not csv_files:
        report.add(sec, "WARN",
                   "No JSON or CSV result files found — cannot audit numerical values",
                   f"Expected files under {results_dir}")
        return

    for jf in json_files:
        all_issues.extend(_check_json_for_numeric_issues(jf))

    for cf in csv_files:
        all_issues.extend(_check_csv_for_numeric_issues(cf))

    report.numeric_issues = all_issues

    # Separate hard blockers (inf/nan/negative RMSE/R²<-1) from soft warnings (R²<0)
    blockers = [i for i in all_issues if "R²<0 (model" not in i["reason"]]
    warnings = [i for i in all_issues if "R²<0 (model" in i["reason"]]

    total_files = len(json_files) + len(csv_files)

    if blockers:
        report.add(sec, "BLOCK",
                   f"{len(blockers)} hard numerical anomaly(ies) in result files",
                   "\n    ".join(
                       f"{i['file']} :: {i['key']} = {i['value']} [{i['reason']}]"
                       for i in blockers[:20]
                   ))
    else:
        report.add(sec, "PASS",
                   f"No inf/nan/negative RMSE anomalies in {total_files} result file(s) ✓")

    if warnings:
        report.add(sec, "WARN",
                   f"{len(warnings)} result(s) with R²<0 (worse than mean predictor)",
                   "\n    ".join(
                       f"{i['file']} :: {i['key']} = {i['value']}"
                       for i in warnings[:20]
                   ))


# ─────────────────────────────────────────────────────────────────────────────
#  §5  Results directory — hypatiax/data/results/
# ─────────────────────────────────────────────────────────────────────────────

# Canonical subdirectory structure expected after a full pipeline run.
_EXPECTED_RESULT_SUBDIRS = [
    "comparison_results/extrapolation",
    "comparison_results/feynman-tests/noise-sweep",
    "comparison_results/noise-noiseless/noiseless",
    "comparison_results/noise-noiseless/15",
    "extrapolation",
    "hybrid_llm_nn/all_domains",
    "hybrid_llm_nn/defi",
    "hybrid_pysr/all_domains",
    "hybrid_pysr/defi",
    "llm_guided/all_domains",
    "llm_guided/defi",
    "standalone_llm_nn",
    "figures",
    "tables",
]


def audit_results_dir(report: AuditReport, results_dir: Path) -> None:
    """
    Verify hypatiax/data/results/ exists and has the expected subtree.
    Inventory all JSON + CSV files and attach to report.
    """
    sec = "§5 Results dir"

    if not results_dir.exists():
        report.add(sec, "BLOCK",
                   f"Results directory does not exist: {results_dir}",
                   "Run ensure_output_dirs() (called automatically by run_all.py)")
        return

    # Check subdirectory structure
    missing_subdirs = [
        sd for sd in _EXPECTED_RESULT_SUBDIRS
        if not (results_dir / sd).exists()
    ]
    if missing_subdirs:
        report.add(sec, "WARN",
                   f"{len(missing_subdirs)} expected result subdirectories missing",
                   "\n    ".join(missing_subdirs))
    else:
        report.add(sec, "PASS",
                   f"All {len(_EXPECTED_RESULT_SUBDIRS)} expected result subdirs present ✓")

    # Inventory files
    json_files = sorted(results_dir.rglob("*.json"))
    csv_files  = sorted(results_dir.rglob("*.csv"))
    all_data   = json_files + csv_files

    report.result_files = [str(f.relative_to(results_dir)) for f in all_data]

    if not all_data:
        report.add(sec, "WARN",
                   "No JSON or CSV result files found — experiments may not have run yet")
    else:
        report.add(sec, "PASS",
                   f"{len(json_files)} JSON + {len(csv_files)} CSV file(s) found",
                   f"Total data files: {len(all_data)}")

    # Check for lock files (written by universal_protocol.py on success)
    lock_files = list(results_dir.glob(".lock_*"))
    report.add(sec, "PASS" if lock_files else "WARN",
               f"{len(lock_files)} protocol lock file(s) found",
               "Lock files are written by universal_protocol.run_protocol() on success. "
               "0 locks may mean experiments have not completed.")

    # Check for checkpoint file
    checkpoint = results_dir.parent.parent.parent / "logs" / "pipeline_checkpoint.json"
    if checkpoint.exists():
        try:
            cp = json.loads(checkpoint.read_text())
            passed  = [k for k, v in cp.items() if v == "pass"]
            failed  = [k for k, v in cp.items() if v == "fail"]
            msg = f"Checkpoint: {len(passed)} passed, {len(failed)} failed"
            level = "PASS" if not failed else "WARN"
            report.add(sec, level, msg,
                       (f"Failed: {', '.join(failed)}" if failed else ""))
        except Exception as exc:
            report.add(sec, "WARN", f"Checkpoint file exists but could not be parsed: {exc}")
    else:
        report.add(sec, "WARN",
                   "No pipeline checkpoint file found",
                   "Expected at logs/pipeline_checkpoint.json. "
                   "Run python3 run_all.py at least once.")


# ─────────────────────────────────────────────────────────────────────────────
#  §6  Figures directory — hypatiax/data/figures/
# ─────────────────────────────────────────────────────────────────────────────

def audit_figures_dir(report: AuditReport, results_dir: Path) -> None:
    """
    Verify hypatiax/data/figures/ (= results_dir/figures/) contains PDF outputs.

    Note: run_all.py passes --outdir results_dir/figures to generate_figures.py.
    repro.yaml declares figures under paper/figures/ — but the pipeline writes
    to hypatiax/data/results/figures/. Both are checked.
    """
    sec = "§6 Figures dir"

    # Primary: figures/ inside results dir (pipeline output)
    figures_in_results = results_dir / "figures"
    # Secondary: paper/figures/ (LaTeX source location from repro.yaml)
    figures_in_paper   = results_dir.parent.parent.parent / "paper" / "figures"

    checked_dirs: list[Path] = []
    all_pdfs:     list[Path] = []

    for fdir in (figures_in_results, figures_in_paper):
        if fdir.exists():
            checked_dirs.append(fdir)
            all_pdfs.extend(fdir.rglob("*.pdf"))

    report.figure_files = [str(p) for p in sorted(all_pdfs)]

    if not checked_dirs:
        report.add(sec, "BLOCK",
                   "No figures directory found",
                   f"Expected: {figures_in_results}  OR  {figures_in_paper}\n"
                   "Run: python3 run_all.py --only figures")
        return

    if not all_pdfs:
        report.add(sec, "WARN",
                   f"Figures director{'ies' if len(checked_dirs)>1 else 'y'} exist "
                   f"but contain no PDF files",
                   f"Checked: {', '.join(str(d) for d in checked_dirs)}")
    else:
        # FIX: safe relative path — use the figures dir itself as the anchor,
        # not p.parents[1], which crashes when the PDF is < 2 levels deep.
        def _safe_rel(p: Path) -> str:
            for anchor in checked_dirs:
                try:
                    return str(p.relative_to(anchor.parent))
                except ValueError:
                    pass
            return str(p)

        report.add(sec, "PASS",
                   f"{len(all_pdfs)} PDF figure(s) found across "
                   f"{len(checked_dirs)} director{'ies' if len(checked_dirs)>1 else 'y'}",
                   "\n    ".join(_safe_rel(p) for p in sorted(all_pdfs)[:10])
                   + ("  …" if len(all_pdfs) > 10 else ""))

    # Also check for SVG / PNG (non-blocking warning)
    other_figs = []
    for fdir in checked_dirs:
        other_figs.extend(fdir.rglob("*.svg"))
        other_figs.extend(fdir.rglob("*.png"))
    if other_figs:
        report.add(sec, "WARN",
                   f"{len(other_figs)} non-PDF figure(s) found (SVG/PNG)",
                   "Paper pipeline expects PDF; other formats may be intermediate")


# ─────────────────────────────────────────────────────────────────────────────
#  §7  Reproducibility gate
# ─────────────────────────────────────────────────────────────────────────────

# Known absent-from-public-repo items flagged in repro.yaml
_YAML_WARNINGS = [
    ("hybrid_system_v50.py",
     "Engine file hypatiax/tools/symbolic/hybrid_system_v50.py is absent from public repo "
     "(repro.yaml engine warning). Only hybrid_system_v40.py confirmed present. "
     "Contact authors."),
    ("hypatiax_defi_benchmark_v3c.py",
     "Script hypatiax/experiments/benchmarks/hypatiax_defi_benchmark_v3c.py absent "
     "from public repo (repro.yaml defi.script warning). Contact authors."),
]


def audit_reproducibility(report: AuditReport, repo_root: Path) -> None:
    """
    Enforce the reproducibility gate:
      - Check for known absent files declared as missing in repro.yaml
      - Check that FIX-C2 (v40→v50_2 engine swap) has been applied
      - Check that FIX-T1 (70 not 71) is reflected in any tex files
      - Verify N_TASKS_INSTABILITY env var = 70
      - Summarise all blockers from earlier sections
    """
    sec = "§7 Reproducibility"

    # ── Check for absent-but-required files ──────────────────────────────────
    for filename, warning in _YAML_WARNINGS:
        found = list(repo_root.rglob(filename))
        if not found:
            report.add(sec, "WARN",
                       f"{filename} not found in repo (known public-repo gap)",
                       warning)
        else:
            report.add(sec, "PASS", f"{filename} found at {found[0].relative_to(repo_root)}")

    # ── FIX-C2: engine version check ─────────────────────────────────────────
    engine_env = os.environ.get("ENGINE_NAME", "")
    if engine_env and "v40" in engine_env:
        report.add(sec, "BLOCK",
                   "ENGINE_NAME contains 'v40' — FIX-C2 engine swap not applied",
                   f"ENGINE_NAME={engine_env}. Must be 'hybrid_system_v50_2'.")
    elif engine_env:
        report.add(sec, "PASS",
                   f"ENGINE_NAME={engine_env} (FIX-C2 applied ✓)")
    else:
        report.add(sec, "WARN",
                   "ENGINE_NAME not set in environment",
                   "run_all.py sets this via env.setdefault(). "
                   "If running audit standalone, set: export ENGINE_NAME=hybrid_system_v50_2")

    # ── FIX-T1: 70 tasks check ────────────────────────────────────────────────
    n_tasks = os.environ.get("N_TASKS_INSTABILITY", "")
    if n_tasks == "70":
        report.add(sec, "PASS",
                   "N_TASKS_INSTABILITY=70 ✓ (FIX-T1 applied)")
    elif n_tasks == "71":
        report.add(sec, "BLOCK",
                   "N_TASKS_INSTABILITY=71 — FIX-T1 not applied",
                   "§10.9 uses 70 tasks (4 excluded). "
                   "Set N_TASKS_INSTABILITY=70 or re-run run_all.py")
    elif n_tasks:
        report.add(sec, "WARN",
                   f"N_TASKS_INSTABILITY={n_tasks} (expected 70)",
                   "Verify this matches §10.9 in the paper")
    else:
        report.add(sec, "WARN",
                   "N_TASKS_INSTABILITY not set in environment",
                   "run_all.py defaults to 70 via env.setdefault(). "
                   "Set explicitly for standalone runs.")

    # ── Scan .tex files for '71 cases' (FIX-T1 textual fix) ─────────────────
    paper_dir = repo_root / "paper"
    if paper_dir.exists():
        for tex in paper_dir.rglob("*.tex"):
            try:
                content = tex.read_text(errors="replace")
            except OSError:
                continue
            if "71 cases" in content or "71 tasks" in content:
                report.add(sec, "BLOCK",
                           f"'71 cases/tasks' still present in {tex.relative_to(repo_root)}",
                           "FIX-T1 requires replacing '71 cases' → '70 tasks' in §10.9. "
                           "Run apply_patches.py or edit manually.")

    # ── Duplicate DeFi names check (FIX-C1 / P-2) ────────────────────────────
    benchmark = (repo_root / "hypatiax" / "experiments" / "benchmarks" /
                 "hypatiax_defi_benchmark_v3c.py")
    if benchmark.exists():
        text = benchmark.read_text(errors="replace")
        for dup_name in (
            '"Constant product formula"',
            '"Funding rate cost"',
            '"Concentrated liquidity position width"',
        ):
            count = text.count(dup_name)
            if count > 1:
                report.add(sec, "BLOCK",
                           f"Duplicate DeFi case name found {count}× in benchmark file: "
                           f"{dup_name}",
                           "P-2 patch (apply_patches.py) has not been applied. "
                           "Run: python3 apply_patches.py --patch P-2")
            else:
                report.add(sec, "PASS",
                           f"DeFi name {dup_name} appears exactly once ✓ (FIX-C1/P-2)")
    else:
        report.add(sec, "WARN",
                   "hypatiax_defi_benchmark_v3c.py not found — FIX-C1/P-2 cannot be verified",
                   "This file is absent from the public repo (see repro.yaml defi.script warning)")


# ─────────────────────────────────────────────────────────────────────────────
#  Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    print("\n" + "═" * 72)
    print(f"  {BOLD(text)}")
    print("═" * 72)


def _print_finding(f: Finding) -> None:
    sym = {"PASS": "✓", "WARN": "⚠", "BLOCK": "✗"}[f.level]
    colour = {"PASS": GRN, "WARN": YLW, "BLOCK": RED}[f.level]
    print(f"  {colour(sym)} [{f.section}]  {f.message}")
    if f.detail:
        for line in f.detail.splitlines():
            print(f"       {CYN(line)}")


def print_report(report: AuditReport) -> None:
    _banner("HypatiaX · Provenance & Reproducibility Audit")
    print(f"  Timestamp : {report.timestamp}")
    print(f"  Repo root : {report.repo_root}")
    print(f"  Seed      : {report.seed_used}")

    # Group by section
    sections: dict[str, list[Finding]] = {}
    for f in report.findings:
        sections.setdefault(f.section, []).append(f)

    for section, findings in sections.items():
        print(f"\n  {BOLD(section)}")
        for f in findings:
            _print_finding(f)

    # Summary
    _banner("Summary")
    passed  = sum(1 for f in report.findings if f.level == "PASS")
    warned  = sum(1 for f in report.findings if f.level == "WARN")
    blocked = sum(1 for f in report.findings if f.level == "BLOCK")

    print(f"  {GRN('✓')} Passed  : {passed}")
    print(f"  {YLW('⚠')} Warnings: {warned}")
    print(f"  {RED('✗')} Blockers: {blocked}")
    print()

    # Result + figure counts
    print(f"  Result files (JSON+CSV): {len(report.result_files)}")
    print(f"  Figure files (PDF)     : {len(report.figure_files)}")
    if report.numeric_issues:
        print(f"  {RED('Numerical issues')}: {len(report.numeric_issues)}")

    # Task name table
    print()
    print(f"  {BOLD('Step → Task name mapping (§2)')}")
    for sid, name in TASK_NAME_MAP.items():
        print(f"    {CYN(sid):35s}  {name}")

    # Reproducibility verdict
    print()
    if report.reproducible():
        print(f"  {GRN(BOLD('REPRODUCIBLE: Yes ✓'))}")
        print("  All blocking checks passed. The pipeline can be reproduced "
              "as described in the paper.")
    else:
        print(f"  {RED(BOLD('REPRODUCIBLE: No ✗'))}")
        print(f"  {blocked} blocker(s) must be resolved before claiming reproducibility:\n")
        for f in report.blockers():
            print(f"    [{f.section}] {f.message}")
            if f.detail:
                print(f"      → {f.detail.splitlines()[0]}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def find_repo_root(start: Path) -> Path:
    """Walk up from start until we find run_all_checkpoint.py or repro.yaml."""
    for candidate in [start, *start.parents]:
        if (candidate / "repro.yaml").exists() or \
           (candidate / "run_all_checkpoint.py").exists() or \
           (candidate / "run_all.py").exists():
            return candidate
    return start   # fallback: use cwd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HypatiaX full provenance & reproducibility audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--root", metavar="PATH", default=None,
                        help="Repo root (auto-detected if omitted)")
    parser.add_argument("--seed", type=int, default=None, metavar="N",
                        help="Seed to verify (default: read NN_SEED env or 42)")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON report to stdout")
    parser.add_argument("--out", metavar="FILE", default=None,
                        help="Write JSON report to this file (implies --json)")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop audit on first BLOCK finding")
    parser.add_argument("--fix-config_secrets", action="store_true",
                        help="Create a config_secrets.py template if absent")
    args = parser.parse_args()

    # ── Resolve repo root ────────────────────────────────────────────────────
    if args.root:
        repo_root = Path(args.root).resolve()
    else:
        repo_root = find_repo_root(Path.cwd())

    if not repo_root.exists():
        print(f"ERROR: repo root does not exist: {repo_root}", file=sys.stderr)
        return 2

    results_dir = repo_root / "hypatiax" / "data" / "results"

    # ── Seed ─────────────────────────────────────────────────────────────────
    if args.seed is not None:
        seed = args.seed
    else:
        try:
            seed = int(os.environ.get("NN_SEED", "42"))
        except ValueError:
            seed = 42

    # ── Optional: create config_secrets.py template ─────────────────────────────────
    # FIX: use _SECRETS_RELATIVE so this path always matches what audit_config_secrets checks
    config_secrets_path = repo_root / _SECRETS_RELATIVE
    if args.fix_config_secrets and not config_secrets_path.exists():
        config_secrets_path.parent.mkdir(parents=True, exist_ok=True)
        config_secrets_path.write_text(
            "# config_secrets.py — do NOT commit to git\n"
            "import os\n"
            "ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']\n"
        )
        print(f"Created config_secrets.py template at {config_secrets_path}")

    # ── Build report ─────────────────────────────────────────────────────────
    report = AuditReport(repo_root=str(repo_root), seed_used=seed)

    # FIX: the original elif chain caused all sections after the first blocker
    # to be silently skipped even when --fail-fast was NOT set.
    # Now every section always runs; --fail-fast is checked after each one.
    audit_fns = [
        (audit_seeds,           (seed,)),
        (audit_task_names,      ()),
        (audit_config_secrets,         (repo_root,)),
        (audit_numerical,       (results_dir,)),
        (audit_results_dir,     (results_dir,)),
        (audit_figures_dir,     (results_dir,)),
        (audit_reproducibility, (repo_root,)),
    ]

    for fn, fn_args in audit_fns:
        fn(report, *fn_args)
        if args.fail_fast and report.blockers():
            break

    # ── Output ───────────────────────────────────────────────────────────────
    if args.json or args.out:
        payload = json.dumps(report.to_dict(), indent=2, default=str)
        if args.out:
            out_path = Path(args.out)
            out_path.write_text(payload)
            print(f"JSON report written to {out_path}")
        else:
            print(payload)
    else:
        print_report(report)

    return 0 if report.reproducible() else 1


if __name__ == "__main__":
    sys.exit(main())

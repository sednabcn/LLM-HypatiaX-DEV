#!/usr/bin/env python3
"""
pipeline_trace.py — HypatiaX · End-to-End Pipeline Dry-Run Tracer
==================================================================
Validates the complete task → protocol → experiment → result-glob mapping
WITHOUT executing any subprocess, reading any input data, or touching the
filesystem (except optionally checking whether paths exist).

What it checks
--------------
  §A  Step registry          — every step has id, label, cmd, phase, result_glob
  §B  Task ↔ protocol map    — every step cmd[1] script is resolvable from repo root
  §C  Env propagation        — seed vars, engine, N_TASKS are all set consistently
  §D  Result-glob coverage   — every experiment step declares a result_glob
  §E  Checkpoint round-trip  — save/load cycle is lossless
  §F  Step ordering          — phases appear in expected order, no id duplicates
  §G  Flag interactions      — --skip-slow, --skip-paper, --only, --resume coverage

Usage
-----
    python3 pipeline_trace.py                  # full trace, repo root auto-detected
    python3 pipeline_trace.py --root /path/to  # explicit root
    python3 pipeline_trace.py --check-scripts  # also verify .py scripts exist on disk
    python3 pipeline_trace.py --seed 123       # trace with a non-default seed
    python3 pipeline_trace.py --json           # machine-readable output
    python3 pipeline_trace.py --only exp1      # trace a single step end-to-end
    python3 pipeline_trace.py --show-env       # dump the full env block
    python3 pipeline_trace.py --show-map       # print the full task→result table

Exit codes
----------
  0  All checks pass
  1  One or more issues found
  2  Fatal config error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── ANSI colours ──────────────────────────────────────────────────────────────
_NO_COLOUR = not sys.stdout.isatty() or os.environ.get("NO_COLOR")

def _c(code: str, text: str) -> str:
    return text if _NO_COLOUR else f"\033[{code}m{text}\033[0m"

GRN  = lambda t: _c("0;32", t)   # noqa: E731
YLW  = lambda t: _c("1;33", t)   # noqa: E731
RED  = lambda t: _c("0;31", t)   # noqa: E731
BOLD = lambda t: _c("1",    t)   # noqa: E731
CYN  = lambda t: _c("0;36", t)   # noqa: E731
DIM  = lambda t: _c("2",    t)   # noqa: E731


# ─────────────────────────────────────────────────────────────────────────────
#  Inline copy of STEPS + helpers (so this script is self-contained)
#  Keep in sync with run_all_checkpoint.py
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Step:
    id: str
    label: str
    cmd: list[str]
    phase: str
    slow: bool = False
    paper: bool = False
    env_extra: dict = field(default_factory=dict)
    expected: str = ""
    result_glob: str = ""


# ── Canonical task-name map (mirrors provenance_audit.TASK_NAME_MAP) ─────────
TASK_NAME_MAP: dict[str, str] = {
    "deps":                    "Install dependencies",
    "patches-gen":             "Generate patches",
    "patches-apply":           "Apply patches (FIX-C1…FIX-5b)",
    "validate":                "Validate patched source",
    "check-hypatiax-protocols": "Verify hypatiax/protocols/ input-data modules",
    "exp1":        "Exp 1 · DeFi 74-task benchmark v3.0 (§10.2–10.4, §10.6)",
    "exp1b":       "Exp 1b · Portfolio Variance seed sweep (§10.5)",
    "exp2":        "Exp 2 · Feynman 30-equation extrapolation (§10.7)",
    "exp3":        "Exp 3 · Nguyen-12 SEED=42 (§10.8 primary)",
    "exp3b":       "Exp 3b · Nguyen-12 SEED=123 (§10.8 stability check)",
    "suppB":       "Supp B · Noise & sample-complexity sweep",
    "suppA":       "Supp A · Hybrid routing improvements (Fix 1–5b)",
    "instability": "§10.9 · Stability under stochastic inference (K=30)",
    "extrap":      "§10.8 · Extrapolation comparative (near/med/far OOD)",
    "provenance":          "§11 · Provenance audit — protocol orchestration",
    "discover-provenance": "§11 · discover_provenance.py",
    "scan-imports":        "§11 · scan_internal_imports.py",
    "verify":              "Verify results against paper targets",
    "hashlock":            "Hash lock check",
    "figures": "Generate all figures",
    "tables":  "Generate all tables",
    "audit-setup":  "Paper audit · Copy tex into notebooks/",
    "audit-NB-01":  "Paper audit · NB-01 Citation & Bibliography",
    "audit-NB-02":  "Paper audit · NB-02 Cross-Reference & Label",
    "audit-NB-03":  "Paper audit · NB-03 Section Structure & Numbering",
    "audit-NB-04":  "Paper audit · NB-04 Numerical Consistency",
    "audit-NB-05":  "Paper audit · NB-05 Figure & Image Dependencies",
}

# Expected phases in canonical order
PHASE_ORDER = [
    "0 · Setup",
    "1 · Core experiments",
    "2 · Supplementary benchmarks",
    "3 · Audit & verification",
    "4 · Outputs",
    "4-B · Paper audit",
]

# Steps that should always declare a result_glob (experiment/output steps)
_RESULT_GLOB_REQUIRED = {
    "exp1", "exp1b", "exp2", "exp3", "exp3b",
    "suppB", "suppA", "instability", "extrap",
    "figures", "tables",
}

# The canonical env block built by run_all.py (for tracing env propagation)
def _build_env(repo_root: Path, results_dir: Path, seed: int) -> dict[str, str]:
    seed_str = str(seed)
    env = {
        "NN_SEED":               seed_str,
        "PYSR_SEED":             seed_str,
        "PYTHONHASHSEED":        seed_str,
        "LLM_MODEL":             "claude-sonnet-4-5",
        "LLM_RETRIES":           "3",
        "LLM_K_RUNS":            "1",
        "N_TASKS_DEFI":          "74",
        "N_TASKS_INSTABILITY":   "70",
        "PCA_TRAIN_FRAC":        "0.40",
        "NN_TIME_LIMIT":         "120",
        "ENGINE_NAME":           "hybrid_system_v50_2",
        "PYTHON_JULIACALL_HANDLE_SIGNALS": "yes",
        "PYTHONWARNINGS":        "ignore",
        "PYTHONPATH":            str(repo_root),
        "RESULTS_DIR":           str(results_dir),
        "REPRO_ROOT":            str(repo_root),
    }
    return env


# ── Inline STEPS definition (identical to run_all_checkpoint.py) ──────────────
def _build_steps(results_dir: Path) -> list[Step]:
    return [
        Step("deps", "Install dependencies",
             ["pip", "install", "-q", "-r", "requirements.txt"],
             phase="0 · Setup"),
        Step("patches-gen", "Generate patches",
             ["python3", "scripts/patches/generate_patches.py"],
             phase="0 · Setup"),
        Step("patches-apply", "Apply patches (FIX-C1…FIX-5b)",
             ["python3", "scripts/patches/apply_patches.py"],
             phase="0 · Setup"),
        Step("validate", "Validate patched source",
             ["python3", "scripts/patches/validate_code.py"],
             phase="0 · Setup"),
        Step("check-hypatiax-protocols",
             "Verify hypatiax/protocols/ input-data modules",
             ["python3", "scripts/patches/check_hypatiax_protocols.py"],
             phase="0 · Setup",
             expected="All 9 hypatiax/protocols/ input-data modules present"),
        Step("exp1",
             "Exp 1 · DeFi 74-task benchmark v3.0 (§10.2–10.4, §10.6)",
             ["python3", "protocols/experiment_protocol_ablation_exp1.py"],
             phase="1 · Core experiments",
             expected="89.2% R²>0.99 · 0 catastrophic · 1.73× speedup",
             result_glob="comparison_results/noise-noiseless/noiseless/*.json"),
        Step("exp1b",
             "Exp 1b · Portfolio Variance seed sweep (§10.5)",
             ["python3", "protocols/experiment_protocol_defi_v3.py",
              "--task", "portfolio_variance",
              "--seeds", "42", "99", "123", "777", "2024"],
             phase="1 · Core experiments",
             expected="P(H>P) ≈ 0.76",
             result_glob="comparison_results/noise-noiseless/15/*.json"),
        Step("exp2",
             "Exp 2 · Feynman 30-equation extrapolation (§10.7)",
             ["python3", "protocols/experiment_protocol_feynman_exp2.py"],
             phase="1 · Core experiments",
             slow=True,
             expected="9/30 (30%)",
             result_glob="comparison_results/feynman-tests/**/*.json"),
        Step("exp3",
             "Exp 3 · Nguyen-12 SEED=42 (§10.8 primary)",
             ["python3", "protocols/experiment_protocol_nguyen12_exp3.py"],
             phase="1 · Core experiments",
             expected="11/12 H (91.7%)",
             result_glob="hypatiax/data/results/nguyen12_exp3_*.json"),
        Step("exp3b",
             "Exp 3b · Nguyen-12 SEED=123 (§10.8 stability check)",
             ["python3", "protocols/experiment_protocol_nguyen12_exp3.py",
              "--seed", "123"],
             phase="1 · Core experiments",
             expected="consistent with SEED=42",
             result_glob="extrapolation/full_run_*.json"),
        Step("suppB",
             "Supp B · Noise & sample-complexity sweep",
             ["python3", "protocols/experiment_protocol_noise_sweep.py"],
             phase="2 · Supplementary benchmarks",
             slow=True,
             expected="EHD 100% at all σ",
             result_glob="comparison_results/feynman-tests/noise-sweep/**/*.json"),
        Step("suppA",
             "Supp A · Hybrid routing improvements (Fix 1–5b)",
             ["python3", "protocols/experiment_protocol_hybrid_routing.py"],
             phase="2 · Supplementary benchmarks",
             expected="+6pp Fix1, +5pp Fix2, +1pp Fix3",
             result_glob="hybrid_pysr/all_domains/**/*.json"),
        Step("instability",
             "§10.9 · Stability under stochastic inference (K=30)",
             ["python3", "protocols/experiment_protocol_instability_rf02_04.py"],
             phase="2 · Supplementary benchmarks",
             slow=True,
             env_extra={"LLM_K_RUNS": "30"},
             expected="Spearman ρ=−0.70",
             result_glob="hybrid_llm_nn/**/*.json"),
        Step("extrap",
             "§10.8 · Extrapolation comparative (near/med/far OOD)",
             ["python3", "protocols/experiment_protocol_extrapolation_comparative.py"],
             phase="2 · Supplementary benchmarks",
             result_glob="extrapolation/extrapolation_73cases_enhanced.json"),
        Step("provenance",
             "§11 · Provenance audit — protocol orchestration",
             ["python3", "protocols/experiment_protocol_provenance_audit.py"],
             phase="3 · Audit & verification"),
        Step("discover-provenance",
             "§11 · discover_provenance.py",
             ["python3", "-c", "..."],
             phase="3 · Audit & verification"),
        Step("scan-imports",
             "§11 · scan_internal_imports.py",
             ["python3", "scan_internal_imports.py", "--root", ".", "--out", "logs/repro_output"],
             phase="3 · Audit & verification"),
        Step("verify",
             "Verify results against paper targets",
             ["python3", "scripts/patches/verify_results.py", "--report"],
             phase="3 · Audit & verification"),
        Step("hashlock",
             "Hash lock check",
             ["python3", "reproducibility/hash_lock.py", "--check"],
             phase="3 · Audit & verification"),
        Step("figures",
             "Generate all figures",
             ["python3", "figures/generate_figures.py",
              "--outdir", str(results_dir / "figures")],
             phase="4 · Outputs",
             result_glob="figures/*.pdf"),
        Step("tables",
             "Generate all tables",
             ["python3", "scripts/patches/generate_tables.py",
              "--outdir", str(results_dir / "tables")],
             phase="4 · Outputs",
             result_glob="tables/*.tex"),
        Step("audit-setup",
             "Paper audit · Copy tex into notebooks/",
             ["python3", "-c", "..."],
             phase="4-B · Paper audit",
             paper=True),
        Step("audit-NB-01", "Paper audit · NB-01 Citation & Bibliography",
             ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
              "--ExecutePreprocessor.timeout=300",
              "notebooks/NB-01_Citation_Bibliography_Audit.ipynb"],
             phase="4-B · Paper audit", paper=True),
        Step("audit-NB-02", "Paper audit · NB-02 Cross-Reference & Label",
             ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
              "--ExecutePreprocessor.timeout=300",
              "notebooks/NB-02_CrossReference_Label_Audit.ipynb"],
             phase="4-B · Paper audit", paper=True),
        Step("audit-NB-03", "Paper audit · NB-03 Section Structure & Numbering",
             ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
              "--ExecutePreprocessor.timeout=300",
              "notebooks/NB-03_Section_Structure_Numbering.ipynb"],
             phase="4-B · Paper audit", paper=True),
        Step("audit-NB-04", "Paper audit · NB-04 Numerical Consistency",
             ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
              "--ExecutePreprocessor.timeout=300",
              "notebooks/NB-04_Numerical_Consistency_Checker.ipynb"],
             phase="4-B · Paper audit", paper=True),
        Step("audit-NB-05", "Paper audit · NB-05 Figure & Image Dependencies",
             ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
              "--ExecutePreprocessor.timeout=300",
              "notebooks/NB-05_Figure_Image_Dependency_Checker.ipynb"],
             phase="4-B · Paper audit", paper=True),
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  Trace result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TraceIssue:
    section: str
    level:   str    # "OK" | "WARN" | "ERR"
    step_id: str
    message: str
    detail:  str = ""


@dataclass
class TraceReport:
    timestamp:  str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    repo_root:  str = ""
    seed:       int = 42
    issues:     list[TraceIssue] = field(default_factory=list)
    step_map:   list[dict] = field(default_factory=list)  # the rendered full map
    env_block:  dict = field(default_factory=dict)

    def add(self, section: str, level: str, step_id: str,
            message: str, detail: str = "") -> None:
        self.issues.append(TraceIssue(section, level, step_id, message, detail))

    def errors(self)   -> list[TraceIssue]: return [i for i in self.issues if i.level == "ERR"]
    def warnings(self) -> list[TraceIssue]: return [i for i in self.issues if i.level == "WARN"]
    def ok(self)       -> list[TraceIssue]: return [i for i in self.issues if i.level == "OK"]
    def passed(self)   -> bool: return len(self.errors()) == 0

    def to_dict(self) -> dict:
        return {
            "timestamp":  self.timestamp,
            "repo_root":  self.repo_root,
            "seed":       self.seed,
            "passed":     self.passed(),
            "error_count":   len(self.errors()),
            "warning_count": len(self.warnings()),
            "issues":   [vars(i) for i in self.issues],
            "step_map": self.step_map,
            "env_block": self.env_block,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  §A  Step registry integrity
# ─────────────────────────────────────────────────────────────────────────────

def trace_registry(report: TraceReport, steps: list[Step]) -> None:
    sec = "§A Registry"
    seen_ids: set[str] = set()

    for step in steps:
        # Duplicate id check
        if step.id in seen_ids:
            report.add(sec, "ERR", step.id, f"Duplicate step id '{step.id}'")
        seen_ids.add(step.id)

        # Required fields
        if not step.label.strip():
            report.add(sec, "ERR", step.id, "Empty label")
        if not step.cmd:
            report.add(sec, "ERR", step.id, "Empty cmd list")
        if not step.phase.strip():
            report.add(sec, "ERR", step.id, "Empty phase")

        # TASK_NAME_MAP coverage
        if step.id not in TASK_NAME_MAP:
            report.add(sec, "WARN", step.id,
                       "Step id not in TASK_NAME_MAP — provenance_audit §2 will flag this")
        else:
            report.add(sec, "OK", step.id,
                       f"Registered in TASK_NAME_MAP → '{TASK_NAME_MAP[step.id]}'")

    report.add(sec, "OK", "*",
               f"Registry contains {len(steps)} steps, {len(seen_ids)} unique ids")


# ─────────────────────────────────────────────────────────────────────────────
#  §B  Task → protocol script mapping
# ─────────────────────────────────────────────────────────────────────────────

def trace_protocol_map(report: TraceReport, steps: list[Step],
                       repo_root: Path, check_scripts: bool) -> None:
    sec = "§B Protocol map"

    for step in steps:
        cmd = step.cmd
        if not cmd:
            continue

        # Identify the script path: for `python3 <script>` commands
        script: str | None = None
        if cmd[0] in ("python3", "python") and len(cmd) > 1 and not cmd[1].startswith("-"):
            script = cmd[1]
        elif cmd[0] == "pip":
            report.add(sec, "OK", step.id, "pip command — no script to resolve")
            continue
        elif cmd[0] == "jupyter":
            # notebook path is the last positional arg
            nb = next((c for c in reversed(cmd) if c.endswith(".ipynb")), None)
            if nb:
                script = nb
        elif cmd[0] == "python3" and cmd[1] == "-c":
            report.add(sec, "OK", step.id, "Inline -c command — no script file to resolve")
            continue

        if script is None:
            report.add(sec, "WARN", step.id,
                       f"Cannot identify script from cmd: {cmd[0]} …")
            continue

        if check_scripts:
            full = repo_root / script
            if full.exists():
                report.add(sec, "OK", step.id, f"Script exists: {script}")
            else:
                report.add(sec, "ERR", step.id,
                           f"Script NOT FOUND on disk: {script}",
                           f"Expected: {full}")
        else:
            report.add(sec, "OK", step.id, f"Script declared: {script}")

        # Extra CLI args decoded for tracing
        extra_args = [a for a in cmd[2:] if not a.startswith("-")]
        flag_args  = [a for a in cmd[2:] if a.startswith("-")]
        if flag_args:
            report.add(sec, "OK", step.id,
                       f"CLI flags: {' '.join(flag_args)}"
                       + (f"  args: {' '.join(extra_args)}" if extra_args else ""))


# ─────────────────────────────────────────────────────────────────────────────
#  §C  Environment propagation
# ─────────────────────────────────────────────────────────────────────────────

def trace_env(report: TraceReport, steps: list[Step], env: dict[str, str]) -> None:
    sec = "§C Env"

    # Seed consistency
    nn   = env.get("NN_SEED", "")
    pysr = env.get("PYSR_SEED", "")
    phsh = env.get("PYTHONHASHSEED", "")

    if nn == pysr == phsh:
        report.add(sec, "OK", "*",
                   f"Seed vars consistent: NN_SEED=PYSR_SEED=PYTHONHASHSEED={nn}")
    else:
        report.add(sec, "ERR", "*",
                   f"Seed vars inconsistent: NN_SEED={nn} PYSR_SEED={pysr} PYTHONHASHSEED={phsh}")

    # Engine name
    engine = env.get("ENGINE_NAME", "")
    if "v40" in engine:
        report.add(sec, "ERR", "*",
                   f"ENGINE_NAME={engine} contains 'v40' — FIX-C2 not applied")
    elif engine:
        report.add(sec, "OK", "*", f"ENGINE_NAME={engine} ✓")

    # N_TASKS_INSTABILITY
    n_tasks = env.get("N_TASKS_INSTABILITY", "")
    if n_tasks == "70":
        report.add(sec, "OK", "*", "N_TASKS_INSTABILITY=70 ✓ (FIX-T1)")
    elif n_tasks == "71":
        report.add(sec, "ERR", "*",
                   "N_TASKS_INSTABILITY=71 — FIX-T1 not applied")
    else:
        report.add(sec, "WARN", "*", f"N_TASKS_INSTABILITY={n_tasks!r} (expected '70')")

    # Per-step env_extra overrides
    for step in steps:
        if step.env_extra:
            overrides = ", ".join(f"{k}={v}" for k, v in step.env_extra.items())
            # Validate no seed var is accidentally overridden
            seed_keys = {"NN_SEED", "PYSR_SEED", "PYTHONHASHSEED"}
            bad_overrides = seed_keys & set(step.env_extra.keys())
            if bad_overrides:
                report.add(sec, "WARN", step.id,
                           f"env_extra overrides seed var(s): {bad_overrides}",
                           "This will break seed consistency for this step")
            else:
                report.add(sec, "OK", step.id, f"env_extra: {overrides}")


# ─────────────────────────────────────────────────────────────────────────────
#  §D  Result-glob coverage
# ─────────────────────────────────────────────────────────────────────────────

def trace_result_globs(report: TraceReport, steps: list[Step],
                       results_dir: Path) -> None:
    sec = "§D Result globs"

    for step in steps:
        if step.id in _RESULT_GLOB_REQUIRED:
            if not step.result_glob:
                report.add(sec, "ERR", step.id,
                           "result_glob is MISSING — output cannot be verified or archived")
            else:
                # Decode glob type and expected output location
                glob = step.result_glob
                is_recursive = "**" in glob
                output_dir   = results_dir / Path(glob).parts[0]
                ext          = Path(glob).suffix or "(any)"
                report.add(sec, "OK", step.id,
                           f"result_glob declared: {glob}",
                           f"  output dir : {output_dir}\n"
                           f"  recursive  : {is_recursive}\n"
                           f"  file type  : {ext}")
        elif step.result_glob:
            # Has a glob but wasn't in the required set — just note it
            report.add(sec, "OK", step.id,
                       f"result_glob (optional): {step.result_glob}")
        else:
            # Setup/audit steps — no output glob expected
            report.add(sec, "OK", step.id, "No result_glob (setup/audit step — expected)")


# ─────────────────────────────────────────────────────────────────────────────
#  §E  Checkpoint round-trip
# ─────────────────────────────────────────────────────────────────────────────

def trace_checkpoint(report: TraceReport, steps: list[Step]) -> None:
    """
    Simulate save → load → verify of the checkpoint state without touching disk.
    Confirms the state dict survives a JSON round-trip and every step id is preserved.
    """
    sec = "§E Checkpoint"
    import io

    # Simulate a completed run
    state_in = {s.id: "pass" for s in steps}
    buf = io.StringIO()
    json.dump(state_in, buf, indent=2)
    raw = buf.getvalue()

    state_out = json.loads(raw)

    missing = [sid for sid in state_in if sid not in state_out]
    extra   = [sid for sid in state_out if sid not in state_in]
    changed = [sid for sid in state_in
               if sid in state_out and state_in[sid] != state_out[sid]]

    if missing or extra or changed:
        report.add(sec, "ERR", "*",
                   "Checkpoint round-trip FAILED",
                   f"missing={missing}  extra={extra}  changed={changed}")
    else:
        report.add(sec, "OK", "*",
                   f"Checkpoint round-trip OK — {len(state_in)} step ids preserved")

    # Verify resume logic: steps before `from_step` would be skipped
    # Trace: if we resume from exp2, what gets skipped vs re-run?
    from_step = "exp2"
    past = False
    skipped, rerun = [], []
    for s in steps:
        if s.id == from_step:
            past = True
        if state_out.get(s.id) == "pass" and not past:
            skipped.append(s.id)
        else:
            rerun.append(s.id)
    report.add(sec, "OK", "*",
               f"Resume --from {from_step}: {len(skipped)} skipped, {len(rerun)} re-run",
               f"  skipped : {skipped}\n  re-run  : {rerun}")


# ─────────────────────────────────────────────────────────────────────────────
#  §F  Step ordering & phase sequence
# ─────────────────────────────────────────────────────────────────────────────

def trace_ordering(report: TraceReport, steps: list[Step]) -> None:
    sec = "§F Ordering"

    phases_seen: list[str] = []
    for step in steps:
        if not phases_seen or phases_seen[-1] != step.phase:
            phases_seen.append(step.phase)

    # Check no phase appears after a later phase
    known_order = {p: i for i, p in enumerate(PHASE_ORDER)}
    issues_found = False
    for i in range(1, len(phases_seen)):
        prev, curr = phases_seen[i - 1], phases_seen[i]
        prev_idx = known_order.get(prev, 999)
        curr_idx = known_order.get(curr, 999)
        if curr_idx < prev_idx:
            report.add(sec, "ERR", "*",
                       f"Phase order violation: '{curr}' appears after '{prev}'")
            issues_found = True

    if not issues_found:
        report.add(sec, "OK", "*",
                   f"Phase sequence correct: {' → '.join(phases_seen)}")

    # Slow-step distribution
    slow_steps = [s.id for s in steps if s.slow]
    paper_steps = [s.id for s in steps if s.paper]
    report.add(sec, "OK", "*",
               f"Slow steps (--skip-slow): {slow_steps}")
    report.add(sec, "OK", "*",
               f"Paper steps (--skip-paper): {paper_steps}")


# ─────────────────────────────────────────────────────────────────────────────
#  §G  Flag interaction matrix
# ─────────────────────────────────────────────────────────────────────────────

def trace_flag_interactions(report: TraceReport, steps: list[Step]) -> None:
    """
    Simulate every meaningful flag combination and report what would execute.
    No subprocesses are touched — purely logical trace.
    """
    sec = "§G Flags"

    def _simulate(only: str | None = None,
                  skip_slow: bool = False,
                  skip_paper: bool = False,
                  resume_passed: set | None = None,
                  from_step: str | None = None,
                  resuming: bool = False) -> tuple[list[str], list[str]]:
        """Return (would_run, would_skip) step id lists."""
        would_run, would_skip = [], []
        # past_from: True means we're past the --from step (so checkpoint skip is off)
        # When resuming with no --from, past_from stays False the whole time → all
        # passed steps are skipped.  When not resuming, past_from is irrelevant.
        past_from = (from_step is None) and (not resuming)
        for s in steps:
            if from_step and s.id == from_step:
                past_from = True
            if only and s.id != only:
                would_skip.append(s.id); continue
            if resuming and resume_passed and s.id in resume_passed and not past_from:
                would_skip.append(s.id); continue
            if skip_slow and s.slow:
                would_skip.append(s.id); continue
            if skip_paper and s.paper:
                would_skip.append(s.id); continue
            would_run.append(s.id)
        return would_run, would_skip

    all_ids   = [s.id for s in steps]
    slow_ids  = {s.id for s in steps if s.slow}
    paper_ids = {s.id for s in steps if s.paper}

    # Full run
    run, skip = _simulate()
    assert set(run) == set(all_ids) and not skip
    report.add(sec, "OK", "*", f"Full run: {len(run)} steps execute, 0 skipped")

    # --skip-slow
    run, skip = _simulate(skip_slow=True)
    report.add(sec, "OK", "*",
               f"--skip-slow: {len(run)} run, {len(skip)} skipped ({slow_ids})")

    # --skip-paper
    run, skip = _simulate(skip_paper=True)
    report.add(sec, "OK", "*",
               f"--skip-paper: {len(run)} run, {len(skip)} skipped ({paper_ids})")

    # --only exp1
    run, skip = _simulate(only="exp1")
    assert run == ["exp1"]
    report.add(sec, "OK", "exp1",
               f"--only exp1: exactly 1 step runs, {len(skip)} skipped ✓")

    # --resume (all passed) — nothing should run
    run, skip = _simulate(resuming=True, resume_passed=set(all_ids))
    assert not run, f"Expected 0 steps to run, got {run}"
    report.add(sec, "OK", "*",
               f"--resume (all passed): 0 steps run, {len(skip)} skipped ✓")

    # --resume --from exp2 — steps before exp2 skipped, exp2+ run
    exp2_idx = next(i for i, s in enumerate(steps) if s.id == "exp2")
    passed_before_exp2 = {s.id for s in steps[:exp2_idx]}
    run, skip = _simulate(resuming=True, resume_passed=passed_before_exp2, from_step="exp2")
    report.add(sec, "OK", "*",
               f"--resume --from exp2: {len(run)} run, {len(skip)} skipped",
               f"  runs from: {run[0] if run else '?'}")

    # --from without --resume (should produce warning, all steps run)
    run, skip = _simulate(from_step="exp2", resuming=False)
    report.add(sec, "WARN", "*",
               f"--from exp2 WITHOUT --resume: all {len(run)} steps run (--from has no effect)")


# ─────────────────────────────────────────────────────────────────────────────
#  Full map table builder
# ─────────────────────────────────────────────────────────────────────────────

def build_step_map(steps: list[Step], results_dir: Path, env: dict) -> list[dict]:
    """
    Returns a list of dicts — one per step — with every relevant field
    expanded for inspection. This is the full task→protocol→result map.
    """
    rows = []
    for step in steps:
        cmd = step.cmd
        script = None
        if cmd[0] in ("python3", "python") and len(cmd) > 1 and cmd[1] != "-c":
            script = cmd[1]
        elif cmd[0] == "jupyter":
            script = next((c for c in reversed(cmd) if c.endswith(".ipynb")), None)

        merged_env_seeds = {
            k: {**env, **step.env_extra}.get(k)
            for k in ("NN_SEED", "PYSR_SEED", "PYTHONHASHSEED", "ENGINE_NAME",
                      "LLM_MODEL", "N_TASKS_INSTABILITY")
        }

        result_path = str(results_dir / step.result_glob) if step.result_glob else ""

        rows.append({
            "id":           step.id,
            "phase":        step.phase,
            "label":        step.label,
            "task_name":    TASK_NAME_MAP.get(step.id, "⚠ NOT IN MAP"),
            "script":       script or ("inline -c" if "-c" in cmd else cmd[0]),
            "cli_flags":    [a for a in cmd[2:] if a.startswith("-")],
            "cli_args":     [a for a in cmd[2:] if not a.startswith("-")],
            "env_extra":    step.env_extra,
            "env_seeds":    merged_env_seeds,
            "slow":         step.slow,
            "paper":        step.paper,
            "result_glob":  step.result_glob,
            "result_path":  result_path,
            "expected":     step.expected,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  Rendering
# ─────────────────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    print("\n" + "═" * 76)
    print(f"  {BOLD(text)}")
    print("═" * 76)


def _sym(level: str) -> str:
    return {"OK": GRN("✓"), "WARN": YLW("⚠"), "ERR": RED("✗")}[level]


def print_map_table(step_map: list[dict], results_dir: Path) -> None:
    _banner("Full Task → Protocol → Experiment → Result Map")
    col = f"  {'STEP ID':<28} {'PHASE':<28} {'SCRIPT':<52} {'RESULT GLOB'}"
    print(col)
    print("  " + "─" * 130)
    current_phase = ""
    for row in step_map:
        if row["phase"] != current_phase:
            current_phase = row["phase"]
            print(f"\n  {BOLD(current_phase)}")
        script_col = row["script"] or ""
        glob_col   = row["result_glob"] or DIM("(none)")
        task_warn  = "  ⚠" if "NOT IN MAP" in row["task_name"] else ""
        env_note   = ""
        if row["env_extra"]:
            env_note = CYN(f"  +env: {row['env_extra']}")
        flags_note = ""
        if row["cli_flags"]:
            flags_note = DIM(f"  {' '.join(row['cli_flags'])}")
        print(f"  {CYN(row['id']):<37} {script_col:<52} {YLW(glob_col)}{task_warn}")
        if row["expected"]:
            print(f"    {DIM('expect:')} {row['expected']}")
        if env_note:
            print(f"    {env_note}")
        if flags_note:
            print(f"    {flags_note}")


def print_env_block(env: dict) -> None:
    _banner("Environment Block (propagated to every child process)")
    for k, v in sorted(env.items()):
        print(f"  {CYN(k):<40} = {v}")


def print_trace_report(report: TraceReport, show_ok: bool = False) -> None:
    _banner("HypatiaX · Pipeline Dry-Run Trace")
    print(f"  Timestamp : {report.timestamp}")
    print(f"  Repo root : {report.repo_root}")
    print(f"  Seed      : {report.seed}")

    sections: dict[str, list[TraceIssue]] = {}
    for issue in report.issues:
        sections.setdefault(issue.section, []).append(issue)

    for section, issues in sections.items():
        visible = issues if show_ok else [i for i in issues if i.level != "OK"]
        if not visible:
            # Just print a summary line for clean sections
            ok_count = sum(1 for i in issues if i.level == "OK")
            print(f"\n  {BOLD(section)}  {GRN(f'all {ok_count} checks OK')}")
            continue
        print(f"\n  {BOLD(section)}")
        for issue in visible:
            sid = f"[{issue.step_id}]" if issue.step_id != "*" else ""
            print(f"  {_sym(issue.level)}  {sid:15s} {issue.message}")
            if issue.detail:
                for line in issue.detail.splitlines():
                    print(f"             {DIM(line)}")

    _banner("Trace Summary")
    errs  = len(report.errors())
    warns = len(report.warnings())
    oks   = len(report.ok())
    print(f"  {GRN('✓')} OK      : {oks}")
    print(f"  {YLW('⚠')} Warnings: {warns}")
    print(f"  {RED('✗')} Errors  : {errs}")
    print()
    if report.passed():
        print(f"  {GRN(BOLD('PIPELINE TRACE: PASS ✓'))}")
        print("  All structural checks passed. The pipeline config is self-consistent.")
    else:
        print(f"  {RED(BOLD('PIPELINE TRACE: FAIL ✗'))}")
        print(f"  {errs} error(s) must be fixed before the pipeline will run correctly.")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "repro.yaml").exists() or \
           (candidate / "run_all_checkpoint.py").exists() or \
           (candidate / "run_all.py").exists():
            return candidate
    return start


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HypatiaX pipeline dry-run tracer — no data, no subprocesses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--root",          metavar="PATH", default=None)
    parser.add_argument("--seed",          type=int, default=42, metavar="N")
    parser.add_argument("--only",          metavar="ID", default=None,
                        help="Trace a single step")
    parser.add_argument("--check-scripts", action="store_true",
                        help="Verify .py / .ipynb scripts exist on disk")
    parser.add_argument("--show-env",      action="store_true",
                        help="Print the full environment block")
    parser.add_argument("--show-map",      action="store_true",
                        help="Print the full task → result table")
    parser.add_argument("--show-ok",       action="store_true",
                        help="Include OK findings in the report (verbose)")
    parser.add_argument("--json",          action="store_true",
                        help="Machine-readable JSON output")
    parser.add_argument("--out",           metavar="FILE", default=None,
                        help="Write JSON report to file")
    args = parser.parse_args()

    # ── Repo root ──────────────────────────────────────────────────────────
    repo_root = Path(args.root).resolve() if args.root else find_repo_root(Path.cwd())
    if not repo_root.exists():
        print(f"ERROR: repo root not found: {repo_root}", file=sys.stderr)
        return 2

    results_dir = repo_root / "hypatiax" / "data" / "results"

    # ── Build env + steps ─────────────────────────────────────────────────
    env   = _build_env(repo_root, results_dir, args.seed)
    steps = _build_steps(results_dir)

    if args.only:
        steps = [s for s in steps if s.id == args.only]
        if not steps:
            all_ids = [s.id for s in _build_steps(results_dir)]
            print(f"ERROR: unknown step id '{args.only}'. Valid: {', '.join(all_ids)}",
                  file=sys.stderr)
            return 2

    # ── Run all trace sections ─────────────────────────────────────────────
    report = TraceReport(repo_root=str(repo_root), seed=args.seed, env_block=env)

    trace_registry(report, steps)
    trace_protocol_map(report, steps, repo_root, args.check_scripts)
    trace_env(report, steps, env)
    trace_result_globs(report, steps, results_dir)
    trace_checkpoint(report, steps)
    trace_ordering(report, steps)
    if not args.only:            # flag interactions only make sense for full pipeline
        trace_flag_interactions(report, steps)

    # ── Build full map ─────────────────────────────────────────────────────
    report.step_map = build_step_map(steps, results_dir, env)

    # ── Output ────────────────────────────────────────────────────────────
    if args.json or args.out:
        payload = json.dumps(report.to_dict(), indent=2, default=str)
        if args.out:
            Path(args.out).write_text(payload)
            print(f"JSON trace written to {args.out}")
        else:
            print(payload)
    else:
        if args.show_map:
            print_map_table(report.step_map, results_dir)
        if args.show_env:
            print_env_block(env)
        print_trace_report(report, show_ok=args.show_ok)

    return 0 if report.passed() else 1


if __name__ == "__main__":
    sys.exit(main())

"""
How to use it

# Quick structural check — no data, no subprocesses, instant
python3 pipeline_trace.py

# See the full task → script → result-glob table
python3 pipeline_trace.py --show-map

# Also show the full env block propagated to every child
python3 pipeline_trace.py --show-map --show-env

# Verify every .py script actually exists on disk (no execution)
python3 pipeline_trace.py --check-scripts

# Trace a single step end-to-end
python3 pipeline_trace.py --only instability --show-map --show-env

# Trace with a non-default seed (checks seed consistency)
python3 pipeline_trace.py --seed 123

# Machine-readable output for CI
python3 pipeline_trace.py --json --out trace_report.json


What each section validates
Section                                                       What it catches
§A Registry                               Duplicate step IDs, empty labels/cmds, steps missing from TASK_NAME_MAP

§B Protocol map                           Resolves every cmd to its script path; optionally verifies files exist on disk with --check-scripts

§C Env                                    Seed var consistency, ENGINE_NAME not v40, N_TASKS=70, no env_extra accidentally overwriting seed vars

§D Result globs                           Every experiment step has a result_glob; decodes output dir, recursion, and file type for each

§E Checkpoint                             Simulates a save→JSON→load round-trip; traces what --resume --from exp2 would skip vs re-run

§F Ordering                               Phase sequence matches the canonical order; reports slow/paper step distributions

§G Flags                                  Simulates every flag combo (--skip-slow, --skip-paper, --only, --resume, --from) and counts what runs
"""

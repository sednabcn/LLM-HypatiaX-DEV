#!/usr/bin/env python3
"""
run_all.py  —  HypatiaX · Full reproducibility pipeline (Python version)
Paper: "HypatiaX: A Hybrid Symbolic-Neural Framework for
        Extrapolation-Reliable Analytical Discovery"  (JMLR v3.0, Apr 2026)

Usage:
    python3 run_all.py                    # full pipeline
    python3 run_all.py --skip-slow        # skip Feynman (exp2), noise sweep (suppB), instability
    python3 run_all.py --only exp3        # run one step by id
    python3 run_all.py --continue-on-fail # log failures but keep going

Step IDs (use with --only):
    Setup   : deps  patches-gen  patches-apply  patches-verify  validate
    Phase 1 : exp1  exp1b  exp2  exp3  exp3b
    Phase 2 : suppB  suppA  instability  extrap
    Phase 3 : provenance  verify  hashlock
    Phase 4 : figures  tables

Prerequisites:
    export ANTHROPIC_API_KEY="sk-ant-..."
    pip install -r requirements.txt
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Experiment registry ────────────────────────────────────────────────────────
@dataclass
class Step:
    id: str
    label: str
    cmd: list[str]
    phase: str
    slow: bool = False                   # skipped by --skip-slow
    env_extra: dict = field(default_factory=dict)
    expected: str = ""                   # human note shown in summary

STEPS: list[Step] = [
    # ── Phase 0: Setup ──────────────────────────────────────────────────────
    Step("deps",          "Install dependencies",
         ["pip", "install", "-q", "-r", "requirements.txt"],
         phase="0 · Setup"),

    Step("patches-gen",   "Generate patches",
         ["python3", "scripts/patches/generate_patches.py"],
         phase="0 · Setup"),

    Step("patches-apply", "Apply patches (FIX-C1…FIX-5b)",
         ["python3", "scripts/patches/apply_patches.py"],
         phase="0 · Setup"),

    Step("patches-verify", "Verify patches (import scan + 0-cycle check)",
         ["python3", "scripts/patches/apply_patches.py", "--verify"],
         phase="0 · Setup",
         expected="All 5 patches applied · 0 stale imports · 0 cycles"),

    Step("validate",      "Validate patched source",
         ["python3", "scripts/patches/validate_code.py"],
         phase="0 · Setup"),


    Step("check-hypatiax-protocols",
         "Verify hypatiax/protocols/ input-data modules",
         ["python3", "scripts/patches/check_hypatiax_protocols.py"],
         phase="0 · Setup",
         expected="All 9 hypatiax/protocols/ input-data modules present"),

    # ── Phase 1: Core experiments ────────────────────────────────────────────

    # Exp 1 covers §10.2–10.4 (DeFi 74-task) and §10.6 (Core-15 ablation)
    # in a single protocol — do NOT split or substitute run_dual_condition_benchmark.py
    Step("exp1",
         "Exp 1 · DeFi 74-task benchmark v3.0 (§10.2–10.4, §10.6)",
         ["python3", "protocols/experiment_protocol_ablation_exp1.py"],
         phase="1 · Core experiments",
         expected="89.2% R²>0.99 · 0 catastrophic · 1.73× speedup"),

    # §10.5: five-seed robustness sweep for Portfolio Variance only
    Step("exp1b",
         "Exp 1b · Portfolio Variance seed sweep (§10.5)",
         ["python3", "protocols/experiment_protocol_defi_v3.py",
          "--task", "portfolio_variance",
          "--seeds", "42", "99", "123", "777", "2024"],
         phase="1 · Core experiments",
         expected="P(H>P) ≈ 0.76"),

    # §10.7: primary run is Kaggle 4-vCPU; this protocol reproduces that environment
    Step("exp2",
         "Exp 2 · Feynman 30-equation extrapolation (§10.7)",
         ["python3", "protocols/experiment_protocol_feynman_exp2.py"],
         phase="1 · Core experiments",
         slow=True,
         expected="9/30 (30%)  [Kaggle 4-vCPU primary · wall time 4–8 h]"),

    # §10.8 primary: SEED=42, source exp3_nguyen12_hybrid50v_02.py logic
    Step("exp3",
         "Exp 3 · Nguyen-12 SEED=42 (§10.8 primary)",
         ["python3", "protocols/experiment_protocol_nguyen12_exp3.py"],
         phase="1 · Core experiments",
         expected="11/12 H (91.7%) · 10/12 P · MW U=113, p=0.0097"),

    # §10.8 stability check: SEED=123, source exp3_nguyen12_hybrid50v_02_seed_123.py logic
    Step("exp3b",
         "Exp 3b · Nguyen-12 SEED=123 (§10.8 stability check)",
         ["python3", "protocols/experiment_protocol_nguyen12_exp3.py",
          "--seed", "123"],
         phase="1 · Core experiments",
         expected="consistent with SEED=42"),

    # ── Phase 2: Supplementary benchmarks ───────────────────────────────────

    # Supp B: noise σ ∈ {0,0.5,1,5,10}% AND sample n ∈ {50…1000} in one protocol
    # Do NOT substitute run_noise_sweep_benchmark.py + run_sample_complexity_benchmark.py
    Step("suppB",
         "Supp B · Noise & sample-complexity sweep",
         ["python3", "protocols/experiment_protocol_noise_sweep.py"],
         phase="2 · Supplementary benchmarks",
         slow=True,
         expected="EHD 100% at all σ · plateau ≈ N=500"),

    # Supp A: routing improvements Fix 1–5b
    # Do NOT substitute run_hybrid_system_benchmark.py
    Step("suppA",
         "Supp A · Hybrid routing improvements (Fix 1–5b)",
         ["python3", "protocols/experiment_protocol_hybrid_routing.py"],
         phase="2 · Supplementary benchmarks",
         expected="+6pp Fix1, +5pp Fix2, +1pp Fix3"),

    # §10.9: 70 tasks × K=30 stochastic runs — LLM_K_RUNS injected via env_extra
    # Do NOT substitute run_dual_sweep_benchmarks.py (that is the noise/sample sweep)
    Step("instability",
         "§10.9 · Stability under stochastic inference (K=30)",
         ["python3", "protocols/experiment_protocol_instability_rf02_04.py"],
         phase="2 · Supplementary benchmarks",
         slow=True,
         env_extra={"LLM_K_RUNS": "30"},
         expected="Spearman ρ=−0.70, p<0.001 · 70 tasks · C-Collapse anomaly (RF-06)"),

    Step("extrap",
         "§10.8 · Extrapolation comparative (near/med/far OOD)",
         ["python3", "protocols/experiment_protocol_extrapolation_comparative.py"],
         phase="2 · Supplementary benchmarks"),

    # ── Phase 3: Audit & verification ───────────────────────────────────────
    # §11 provenance: three complementary tools
    # (a) experiment_protocol_provenance_audit.py — orchestration layer (existing)
    # (b) discover_provenance.py — links every result file to family/patch/paper section
    # (c) scan_internal_imports.py — maps the internal import DAG for repro verification
    Step("provenance",
         "§11 · Provenance audit — protocol orchestration",
         ["python3", "protocols/experiment_protocol_provenance_audit.py"],
         phase="3 · Audit & verification"),

    Step("discover-provenance",
         "§11 · discover_provenance.py — link result files to families",
         ["python3", "-c",
          "import subprocess, sys, pathlib; "
          "m = pathlib.Path('provenance_map.json'); "
          "pathlib.Path('logs/provenance_audit').mkdir(parents=True, exist_ok=True); "
          "(print('INFO: provenance_map.json absent — skipping discover_provenance (public repo)') or sys.exit(0)) "
          "if not m.exists() else "
          "sys.exit(subprocess.run([sys.executable, 'discover_provenance.py', "
          "'--root', '.', '--map', str(m), '--out', 'logs/provenance_audit']).returncode)"],
         phase="3 · Audit & verification"),

    Step("scan-imports",
         "§11 · scan_internal_imports.py — internal import DAG",
         ["python3", "scan_internal_imports.py",
          "--root", ".", "--out", "logs/repro_output"],
         phase="3 · Audit & verification"),

    Step("verify",
         "Verify results against paper targets",
         ["python3", "scripts/patches/verify_results.py", "--report", "--json"],
         phase="3 · Audit & verification"),

    Step("hashlock",
         "Hash lock check",
         ["python3", "reproducibility/hash_lock.py", "--check"],
         phase="3 · Audit & verification"),

    # ── Phase 4: Outputs ─────────────────────────────────────────────────────
    Step("figures",
         "Generate all figures",
         ["python3", "figures/generate_figures.py"],
         phase="4 · Outputs"),

    Step("tables",
         "Generate all tables",
         ["python3", "scripts/patches/generate_tables.py"],
         phase="4 · Outputs"),

    # ── Phase 4-B: Paper audit notebooks ─────────────────────────────────────
    # NB-01: FIX-B1/B2/B3  — bibliography (missing koza1994genetic, dup entries)
    # NB-02: FIX-XR1–XR4   — undefined refs, duplicate labels, Supp A cross-refs
    # NB-03: (diagnostic)   — section structure & equation inventory
    # NB-04: FIX-N1/N2/N3  — numerical consistency (70 vs 71, five-stage/layer)
    # NB-05: FIX-F1–F4      — missing figure files and fbox placeholders
    # All notebooks scan paper/jmlr-hypatiax-paper-final.tex (copied to notebooks/).
    # They are diagnostic: they exit 0 regardless, and surface issues in their output.

    Step("audit-setup",
         "Paper audit · Copy tex into notebooks/ for NB-01–05",
         ["python3", "-c",
          "import shutil, pathlib; "
          "nb = pathlib.Path('notebooks'); nb.mkdir(exist_ok=True); "
          "p = pathlib.Path('paper'); "
          "tex = next(p.glob('jmlr-hypatiax*.tex'), None) if p.exists() else None; "
          "shutil.copy(tex, nb / tex.name) if tex else print('TEX not found — paper/ absent or empty')"],
         phase="4-B · Paper audit"),

    Step("audit-NB-01",
         "Paper audit · NB-01 Citation & Bibliography",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-01_Citation_Bibliography_Audit.ipynb"],
         phase="4-B · Paper audit"),

    Step("audit-NB-02",
         "Paper audit · NB-02 Cross-Reference & Label",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-02_CrossReference_Label_Audit.ipynb"],
         phase="4-B · Paper audit"),

    Step("audit-NB-03",
         "Paper audit · NB-03 Section Structure & Numbering",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-03_Section_Structure_Numbering.ipynb"],
         phase="4-B · Paper audit"),

    Step("audit-NB-04",
         "Paper audit · NB-04 Numerical Consistency",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-04_Numerical_Consistency_Checker.ipynb"],
         phase="4-B · Paper audit"),

    Step("audit-NB-05",
         "Paper audit · NB-05 Figure & Image Dependencies",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-05_Figure_Image_Dependency_Checker.ipynb"],
         phase="4-B · Paper audit"),
]

# ── Result tracking ────────────────────────────────────────────────────────────
@dataclass
class Result:
    id: str
    label: str
    status: str          # "pass" | "fail" | "skip"
    elapsed: float = 0.0
    log_path: Optional[Path] = None
    returncode: int = 0

# ── Runner ────────────────────────────────────────────────────────────────────
def run_step(step: Step, log_dir: Path, env: dict) -> Result:
    log_path = log_dir / f"{step.id}.log"
    merged_env = {**env, **step.env_extra}

    print(f"\n┌─── [{step.id}] {step.label}")
    print(f"│    {time.strftime('%H:%M:%S')}")
    if step.expected:
        print(f"│    Expected: {step.expected}")
    if step.env_extra:
        for k, v in step.env_extra.items():
            print(f"│    env override: {k}={v}")

    t0 = time.time()
    try:
        with open(log_path, "w") as log_fh:
            proc = subprocess.run(
                step.cmd,
                env=merged_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log_fh.write(proc.stdout)
            lines = proc.stdout.splitlines()
            for line in lines[-20:]:
                print(f"│  {line}")

        elapsed = time.time() - t0
        ok = proc.returncode == 0
        sym = "✓" if ok else "✗"
        print(f"└─── {sym} {'done' if ok else 'FAILED'}  ({elapsed:.0f}s)"
              + (f"  — see {log_path}" if not ok else ""))
        return Result(step.id, step.label,
                      "pass" if ok else "fail",
                      elapsed, log_path, proc.returncode)

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"└─── ✗ ERROR: {exc}")
        return Result(step.id, step.label, "fail", elapsed, log_path)


def banner(msg: str) -> None:
    print("\n" + "═" * 68)
    print(f"  {msg}")
    print("═" * 68)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="HypatiaX reproducibility pipeline")
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip Feynman (exp2), noise sweep (suppB), instability")
    parser.add_argument("--only", metavar="ID",
                        help="Run only this step id (e.g. exp3)")
    parser.add_argument("--continue-on-fail", action="store_true",
                        help="Log failures but continue remaining steps")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)
    log_dir = repo_root / "logs"
    log_dir.mkdir(exist_ok=True)

    banner("HypatiaX · Reproducibility Pipeline v5.0")
    print(f"  Repo  : {repo_root}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Date  : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("\n  ERROR: ANTHROPIC_API_KEY is not set.")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)
    print(f"  API key: set ({len(api_key)} chars)")

    # Verify hypatiax/protocols/ input-data modules
    hypatiax_proto = repo_root / "hypatiax" / "protocols"
    required_hp = [
        "experiment_protocol_defi.py",
        "experiment_protocol_defi_20.py",
        "experiment_protocol_nguyen12.py",
        "experiment_protocol_all_18_a.py",
        "experiment_protocol_all_20.py",
        "experiment_protocol_all_30.py",
        "experiment_protocol_benchmark.py",
        "experiment_protocol_benchmark_v2.py",
        "experiment_protocol_comparative.py",
    ]
    missing_hp = [f for f in required_hp if not (hypatiax_proto / f).exists()]
    if missing_hp:
        print(f"\n  ERROR: {len(missing_hp)} input-data module(s) missing from hypatiax/protocols/:")
        for f in missing_hp:
            print(f"    ✗  {f}")
        print("  Copy the files into hypatiax/protocols/ and re-run.")
        sys.exit(1)
    print(f"  hypatiax/protocols/ — all {len(required_hp)} input-data modules present ✓")

    # NB-06 code-quality pre-audit (diagnostic, non-blocking)
    nb06 = repo_root / "notebooks" / "NB-06_Code_Quality_Pipeline_Integrity.ipynb"
    if nb06.exists():
        try:
            import subprocess as _sp
            r = _sp.run(
                ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                 "--inplace", "--ExecutePreprocessor.timeout=120", str(nb06)],
                capture_output=True, text=True, timeout=150,
            )
            if r.returncode == 0:
                print("  NB-06 code quality pre-audit passed ✓")
            else:
                print(f"  NB-06 pre-audit warnings (non-blocking) — see {nb06}")
        except FileNotFoundError:
            print("  jupyter not found — skipping NB-06 pre-audit (pip install notebook)")
        except Exception as exc:
            print(f"  NB-06 pre-audit skipped: {exc}")
    else:
        print(f"  NB-06 not found at {nb06} — skipping pre-audit")


    # Build env — mirrors run_all.sh and notebook cell 2
    env = {**os.environ}
    env.setdefault("NN_SEED",               "42")
    env.setdefault("PYSR_SEED",             "42")
    env.setdefault("LLM_MODEL",             "claude-sonnet-4-6")
    env.setdefault("LLM_RETRIES",           "3")
    env.setdefault("LLM_K_RUNS",            "1")   # overridden to 30 for instability
    env.setdefault("N_TASKS_DEFI",          "74")
    env.setdefault("N_TASKS_INSTABILITY",   "70")  # FIX-T1
    env.setdefault("PCA_TRAIN_FRAC",        "0.40")
    env.setdefault("NN_TIME_LIMIT",         "120")
    env.setdefault("ENGINE_NAME",           "hybrid_system_v50_2")  # FIX-C2
    env.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    print(f"\n  NN_SEED={env['NN_SEED']}  PYSR_SEED={env['PYSR_SEED']}")
    print(f"  LLM_MODEL={env['LLM_MODEL']}")
    print(f"  ENGINE={env['ENGINE_NAME']}  N_TASKS_INSTABILITY={env['N_TASKS_INSTABILITY']}")

    results: list[Result] = []
    current_phase = ""
    t_total = time.time()

    for step in STEPS:
        if step.phase != current_phase:
            banner(f"Phase {step.phase}")
            current_phase = step.phase

        if args.only and step.id != args.only:
            results.append(Result(step.id, step.label, "skip"))
            print(f"  ── skip [{step.id}]  (--only {args.only})")
            continue

        if args.skip_slow and step.slow:
            results.append(Result(step.id, step.label, "skip"))
            print(f"  ── skip [{step.id}]  (--skip-slow)")
            continue

        result = run_step(step, log_dir, env)
        results.append(result)

        if result.status == "fail" and not args.continue_on_fail:
            print(f"\n  Pipeline aborted at [{step.id}].")
            print(f"  Re-run with --continue-on-fail to keep going.")
            print(f"  Re-run this step:  python3 run_all.py --only {step.id}")
            _print_summary(results, time.time() - t_total)
            sys.exit(1)

    _print_summary(results, time.time() - t_total)
    failed = [r for r in results if r.status == "fail"]
    sys.exit(1 if failed else 0)


def _print_summary(results: list[Result], elapsed: float) -> None:
    passed  = [r for r in results if r.status == "pass"]
    failed  = [r for r in results if r.status == "fail"]
    skipped = [r for r in results if r.status == "skip"]

    hh, rem = divmod(int(elapsed), 3600)
    mm, ss  = divmod(rem, 60)

    banner("Pipeline summary")
    col = {"pass": "✓", "fail": "✗", "skip": "─"}
    for r in results:
        t = f"  {r.elapsed:6.0f}s" if r.status != "skip" else "       "
        print(f"  {col[r.status]} [{r.id:15s}] {r.label[:55]:55s}{t}")

    print()
    print(f"  ✓ passed : {len(passed)}")
    print(f"  ✗ failed : {len(failed)}")
    print(f"  ─ skipped: {len(skipped)}")
    print(f"  Wall time: {hh:02d}:{mm:02d}:{ss:02d}")
    print()

    # Show provenance coverage if available
    prov_summary = Path(__file__).resolve().parent / "logs" / "provenance_audit" / "provenance_audit_summary.txt"
    if prov_summary.exists():
        print()
        print("  Provenance map coverage:")
        for line in prov_summary.read_text().splitlines():
            if any(k in line for k in ("AUTHORITATIVE", "ORPHAN", "Total")):
                print(f"    {line.strip()}")

    if failed:
        print("  Failed steps:")
        for r in failed:
            print(f"    [{r.id}] → {r.log_path}")
        print()
        print("  Some steps FAILED. Re-run a single step:")
        print("    python3 run_all.py --only <id>")
    else:
        print("  All steps passed. Results ready for paper verification.")
        print(f"  Results    : hypatiax/data/results/")
        print(f"  Provenance : logs/provenance_audit/")
        print(f"  Import DAG : logs/repro_output/import_graph.dot")
        print(f"  Logs       : logs/")


if __name__ == "__main__":
    main()

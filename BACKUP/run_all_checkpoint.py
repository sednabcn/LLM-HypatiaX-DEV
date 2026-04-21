#!/usr/bin/env python3
"""
run_all.py  —  HypatiaX · Full reproducibility pipeline (Python version)
Paper: "HypatiaX: A Hybrid Symbolic-Neural Framework for
        Extrapolation-Reliable Analytical Discovery"  (JMLR v3.0, Apr 2026)

Usage:
    python3 run_all.py                      # full pipeline
    python3 run_all.py --skip-slow          # skip slow steps (Feynman, noise sweep, instability)
    python3 run_all.py --only exp3          # run one step by id
    python3 run_all.py --resume             # resume from last checkpoint
    python3 run_all.py --resume --from exp2 # resume, but force-rerun from this step
    python3 run_all.py --clear-checkpoint   # delete checkpoint file and exit
    python3 run_all.py --continue-on-fail   # log failures but keep going
    python3 run_all.py --verify-only        # re-check existing results without re-running
    python3 run_all.py --seed 123           # override seed for all steps (default: 42)
    python3 run_all.py --only exp3 --seed 777  # single step with custom seed
    python3 run_all.py --skip-paper         # skip pdflatex compile steps

Step IDs (use with --only / --from):
    Setup   : deps  patches-gen  patches-apply  validate  check-hypatiax-protocols
    Phase 1 : exp1  exp1b  exp2  exp3  exp3b
    Phase 2 : suppB  suppA  instability  extrap
    Phase 3 : provenance  discover-provenance  scan-imports  verify  hashlock
    Phase 4 : figures  tables
    Phase 4B: audit-setup  audit-NB-01 ... audit-NB-05

Prerequisites:
    export ANTHROPIC_API_KEY="sk-ant-..."
    pip install -r requirements.txt
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Canonical paths ────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "hypatiax" / "data" / "results"
LOG_DIR     = REPO_ROOT / "logs"
CHECKPOINT  = LOG_DIR / "pipeline_checkpoint.json"

# Steps that are part of a paper compile (skipped by --skip-paper)
_PAPER_STEP_IDS = {"audit-NB-01", "audit-NB-02", "audit-NB-03",
                   "audit-NB-04", "audit-NB-05", "audit-setup"}


# ── Experiment registry ────────────────────────────────────────────────────────
@dataclass
class Step:
    id: str
    label: str
    cmd: list[str]
    phase: str
    slow: bool = False                 # skipped by --skip-slow
    paper: bool = False                # skipped by --skip-paper
    env_extra: dict = field(default_factory=dict)
    expected: str = ""                 # human note shown in summary
    result_glob: str = ""              # glob relative to RESULTS_DIR to verify output


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

    Step("validate",      "Validate patched source",
         ["python3", "scripts/patches/validate_code.py"],
         phase="0 · Setup"),

    Step("check-hypatiax-protocols",
         "Verify hypatiax/protocols/ input-data modules",
         ["python3", "scripts/patches/check_hypatiax_protocols.py"],
         phase="0 · Setup",
         expected="All 9 hypatiax/protocols/ input-data modules present"),

    # ── Phase 1: Core experiments ────────────────────────────────────────────
    # Exp 1 covers §10.2–10.4 (DeFi 74-task) and §10.6 (Core-15 ablation) in a
    # single protocol. Do NOT substitute run_dual_condition_benchmark.py.
    Step("exp1",
         "Exp 1 · DeFi 74-task benchmark v3.0 (§10.2–10.4, §10.6)",
         ["python3", "protocols/experiment_protocol_ablation_exp1.py"],
         phase="1 · Core experiments",
         expected="89.2% R²>0.99 · 0 catastrophic · 1.73× speedup",
         result_glob="comparison_results/noise-noiseless/noiseless/*.json"),

    # §10.5: five-seed robustness sweep for Portfolio Variance only
    Step("exp1b",
         "Exp 1b · Portfolio Variance seed sweep (§10.5)",
         ["python3", "protocols/experiment_protocol_defi_v3.py",
          "--task", "portfolio_variance",
          "--seeds", "42", "99", "123", "777", "2024"],
         phase="1 · Core experiments",
         expected="P(H>P) ≈ 0.76",
         result_glob="comparison_results/noise-noiseless/15/*.json"),

    # §10.7: primary run is Kaggle 4-vCPU; this protocol reproduces that environment
    Step("exp2",
         "Exp 2 · Feynman 30-equation extrapolation (§10.7)",
         ["python3", "protocols/experiment_protocol_feynman_exp2.py"],
         phase="1 · Core experiments",
         slow=True,
         expected="9/30 (30%)  [Kaggle 4-vCPU primary · wall time 4–8 h]",
         result_glob="comparison_results/feynman-tests/**/*.json"),

    # §10.8 primary: SEED=42, source exp3_nguyen12_hybrid50v_02.py logic
    Step("exp3",
         "Exp 3 · Nguyen-12 SEED=42 (§10.8 primary)",
         ["python3", "protocols/experiment_protocol_nguyen12_exp3.py"],
         phase="1 · Core experiments",
         expected="11/12 H (91.7%) · 10/12 P · MW U=113, p=0.0097",
         result_glob="hypatiax/data/results/nguyen12_exp3_*.json"),

    # §10.8 stability: SEED=123
    Step("exp3b",
         "Exp 3b · Nguyen-12 SEED=123 (§10.8 stability check)",
         ["python3", "protocols/experiment_protocol_nguyen12_exp3.py",
          "--seed", "123"],
         phase="1 · Core experiments",
         expected="consistent with SEED=42",
         result_glob="extrapolation/full_run_*.json"),

    # ── Phase 2: Supplementary benchmarks ───────────────────────────────────
    # Supp B: noise σ ∈ {0,0.5,1,5,10}% AND sample n ∈ {50…1000} in one protocol
    Step("suppB",
         "Supp B · Noise & sample-complexity sweep",
         ["python3", "protocols/experiment_protocol_noise_sweep.py"],
         phase="2 · Supplementary benchmarks",
         slow=True,
         expected="EHD 100% at all σ · plateau ≈ N=500",
         result_glob="comparison_results/feynman-tests/noise-sweep/**/*.json"),

    # Supp A: routing improvements Fix 1–5b
    Step("suppA",
         "Supp A · Hybrid routing improvements (Fix 1–5b)",
         ["python3", "protocols/experiment_protocol_hybrid_routing.py"],
         phase="2 · Supplementary benchmarks",
         expected="+6pp Fix1, +5pp Fix2, +1pp Fix3",
         result_glob="hybrid_pysr/all_domains/**/*.json"),

    # §10.9: 70 tasks × K=30 stochastic runs — LLM_K_RUNS injected via env_extra
    Step("instability",
         "§10.9 · Stability under stochastic inference (K=30)",
         ["python3", "protocols/experiment_protocol_instability_rf02_04.py"],
         phase="2 · Supplementary benchmarks",
         slow=True,
         env_extra={"LLM_K_RUNS": "30"},
         expected="Spearman ρ=−0.70, p<0.001 · 70 tasks · C-Collapse anomaly (RF-06)",
         result_glob="hybrid_llm_nn/**/*.json"),

    Step("extrap",
         "§10.8 · Extrapolation comparative (near/med/far OOD)",
         ["python3", "protocols/experiment_protocol_extrapolation_comparative.py"],
         phase="2 · Supplementary benchmarks",
         result_glob="extrapolation/extrapolation_73cases_enhanced.json"),

    # ── Phase 3: Audit & verification ───────────────────────────────────────
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
         ["python3", "scripts/patches/verify_results.py", "--report"],
         phase="3 · Audit & verification"),

    Step("hashlock",
         "Hash lock check",
         ["python3", "reproducibility/hash_lock.py", "--check"],
         phase="3 · Audit & verification"),

    # ── Phase 4: Outputs — figures & tables written to hypatiax/data/results/ ─
    Step("figures",
         "Generate all figures",
         ["python3", "figures/generate_figures.py",
          "--outdir", str(RESULTS_DIR / "figures")],
         phase="4 · Outputs",
         result_glob="figures/*.pdf"),

    Step("tables",
         "Generate all tables",
         ["python3", "scripts/patches/generate_tables.py",
          "--outdir", str(RESULTS_DIR / "tables")],
         phase="4 · Outputs",
         result_glob="tables/*.tex"),

    # ── Phase 4-B: Paper audit notebooks ─────────────────────────────────────
    Step("audit-setup",
         "Paper audit · Copy tex into notebooks/ for NB-01–05",
         ["python3", "-c",
          "import shutil, pathlib; "
          "nb = pathlib.Path('notebooks'); nb.mkdir(exist_ok=True); "
          "p = pathlib.Path('paper'); "
          "tex = next(p.glob('jmlr-hypatiax*.tex'), None) if p.exists() else None; "
          "shutil.copy(tex, nb / tex.name) if tex else print('TEX not found — paper/ absent or empty')"],
         phase="4-B · Paper audit",
         paper=True),

    Step("audit-NB-01",
         "Paper audit · NB-01 Citation & Bibliography",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-01_Citation_Bibliography_Audit.ipynb"],
         phase="4-B · Paper audit",
         paper=True),

    Step("audit-NB-02",
         "Paper audit · NB-02 Cross-Reference & Label",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-02_CrossReference_Label_Audit.ipynb"],
         phase="4-B · Paper audit",
         paper=True),

    Step("audit-NB-03",
         "Paper audit · NB-03 Section Structure & Numbering",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-03_Section_Structure_Numbering.ipynb"],
         phase="4-B · Paper audit",
         paper=True),

    Step("audit-NB-04",
         "Paper audit · NB-04 Numerical Consistency",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-04_Numerical_Consistency_Checker.ipynb"],
         phase="4-B · Paper audit",
         paper=True),

    Step("audit-NB-05",
         "Paper audit · NB-05 Figure & Image Dependencies",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-05_Figure_Image_Dependency_Checker.ipynb"],
         phase="4-B · Paper audit",
         paper=True),
]

STEP_IDS = [s.id for s in STEPS]


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    """Return {step_id: status} from the checkpoint file, or {} if none exists."""
    if CHECKPOINT.exists():
        try:
            return json.loads(CHECKPOINT.read_text())
        except Exception:
            pass
    return {}


def save_checkpoint(state: dict) -> None:
    """Persist {step_id: status} to disk atomically."""
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(CHECKPOINT)


def clear_checkpoint() -> None:
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print(f"  Checkpoint cleared: {CHECKPOINT}")
    else:
        print("  No checkpoint file found.")


# ── Result-file helpers ────────────────────────────────────────────────────────

def ensure_output_dirs() -> None:
    """Create the canonical output subdirectories under hypatiax/data/results/."""
    for sub in [
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
    ]:
        (RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)


def archive_step_results(step: Step) -> None:
    """
    After a step completes, snapshot any newly produced output files into
    logs/<step_id>_results/ for provenance tracing.

    FIX: use rglob() for patterns containing '**'; plain glob() does not expand
    recursive wildcards, so suppB/instability/exp2 previously archived 0 files.
    """
    if not step.result_glob:
        return

    # Split into a base anchor and the glob pattern so rglob works correctly
    pattern = step.result_glob
    if "**" in pattern:
        # e.g. "comparison_results/feynman-tests/**/*.json"
        # Split at the first '**' component
        parts = Path(pattern).parts
        star_idx = next(i for i, p in enumerate(parts) if "**" in p)
        base_dir = RESULTS_DIR / Path(*parts[:star_idx])
        sub_pattern = str(Path(*parts[star_idx:]))
        matches = list(base_dir.rglob(sub_pattern.lstrip("**/").lstrip("/"))) \
                  if base_dir.exists() else []
    else:
        matches = list(RESULTS_DIR.glob(pattern))

    if not matches:
        return

    dest = LOG_DIR / f"{step.id}_results"
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in matches:
        dst = dest / src.name
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            shutil.copy2(src, dst)
            count += 1
    if count:
        print(f"  📁  {count} result file(s) archived → logs/{step.id}_results/")


def inventory_results() -> tuple[int, int, int]:
    """Return (data_file_count, pdf_count, tex_count) under RESULTS_DIR."""
    jsons = sum(1 for _ in RESULTS_DIR.rglob("*.json"))
    csvs  = sum(1 for _ in RESULTS_DIR.rglob("*.csv"))
    pdfs  = sum(1 for _ in (RESULTS_DIR / "figures").glob("*.pdf")) \
            if (RESULTS_DIR / "figures").exists() else 0
    texs  = sum(1 for _ in (RESULTS_DIR / "tables").glob("*.tex")) \
            if (RESULTS_DIR / "tables").exists() else 0
    return jsons + csvs, pdfs, texs


# ── Result tracking ────────────────────────────────────────────────────────────
@dataclass
class StepResult:
    id: str
    label: str
    status: str          # "pass" | "fail" | "skip" | "resume-skip"
    elapsed: float = 0.0
    log_path: Optional[Path] = None
    returncode: int = 0


# ── Step runner ────────────────────────────────────────────────────────────────
def run_step(step: Step, env: dict) -> StepResult:
    """
    FIX: stream subprocess output line-by-line to both the log file and stdout,
    rather than buffering all output into memory before printing the last 20 lines.
    This prevents OOM on long-running steps (exp2, suppB, instability).
    """
    log_path = LOG_DIR / f"{step.id}.log"
    merged_env = {**env, **step.env_extra}

    print(f"\n┌─── [{step.id}] {step.label}")
    print(f"│    {time.strftime('%H:%M:%S')}")
    if step.expected:
        print(f"│    Expected : {step.expected}")
    if step.env_extra:
        for k, v in step.env_extra.items():
            print(f"│    env+  {k}={v}")
    print(f"│    cmd: {' '.join(str(x) for x in step.cmd)}")

    t0 = time.time()
    try:
        with open(log_path, "w") as log_fh:
            proc = subprocess.Popen(
                step.cmd,
                env=merged_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            # Stream line-by-line: write to log and echo to terminal
            assert proc.stdout is not None
            for line in proc.stdout:
                log_fh.write(line)
                print(f"│  {line}", end="")
            proc.wait()

        elapsed = time.time() - t0
        ok = proc.returncode == 0
        sym = "✓" if ok else "✗"
        print(f"\n└─── {sym} {'done' if ok else 'FAILED'}  ({elapsed:.0f}s)"
              + (f"  — see {log_path}" if not ok else ""))

        if ok:
            archive_step_results(step)

        return StepResult(step.id, step.label,
                          "pass" if ok else "fail",
                          elapsed, log_path, proc.returncode)

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"└─── ✗ ERROR: {exc}")
        return StepResult(step.id, step.label, "fail", elapsed, log_path)


def banner(msg: str) -> None:
    print("\n" + "═" * 68)
    print(f"  {msg}")
    print("═" * 68)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="HypatiaX reproducibility pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip Feynman (exp2), noise sweep (suppB), instability")
    parser.add_argument("--only", metavar="ID",
                        help="Run only this step id")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint — skip steps already marked pass")
    parser.add_argument("--from", dest="from_step", metavar="ID",
                        help="With --resume: force-rerun from this step id onwards")
    parser.add_argument("--clear-checkpoint", action="store_true",
                        help="Delete the checkpoint file and exit")
    parser.add_argument("--continue-on-fail", action="store_true",
                        help="Log failures but continue remaining steps")
    parser.add_argument("--verify-only", action="store_true",
                        help="Re-check existing results without re-running experiments")
    parser.add_argument("--skip-paper", action="store_true",
                        help="Skip Phase 4-B paper audit notebook steps")
    parser.add_argument("--seed", type=int, default=None, metavar="N",
                        help="Override PYSR_SEED / NN_SEED / PYTHONHASHSEED for all steps "
                             "(e.g. --seed 123). Defaults to 42 when omitted.")
    args = parser.parse_args()

    # FIX: validate --from is only used alongside --resume; warn otherwise
    if args.from_step and not args.resume:
        print("  WARNING: --from has no effect without --resume. "
              "Did you mean: python3 run_all.py --resume --from <id>?",
              file=sys.stderr)

    os.chdir(REPO_ROOT)
    LOG_DIR.mkdir(exist_ok=True)
    ensure_output_dirs()

    # ── --clear-checkpoint ─────────────────────────────────────────────────
    if args.clear_checkpoint:
        clear_checkpoint()
        sys.exit(0)

    banner("HypatiaX · Reproducibility Pipeline v4.1 (checkpoint/resume)")
    print(f"  Repo      : {REPO_ROOT}")
    print(f"  Python    : {sys.version.split()[0]}")
    print(f"  Date      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Results   : {RESULTS_DIR}")
    print(f"  Logs      : {LOG_DIR}")
    print(f"  Checkpoint: {CHECKPOINT}")

    # ── API key ────────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("\n  ERROR: ANTHROPIC_API_KEY is not set.")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)
    print(f"\n  API key   : set ({len(api_key)} chars)")

    # ── hypatiax/protocols/ check ──────────────────────────────────────────
    hypatiax_proto = REPO_ROOT / "hypatiax" / "protocols"
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
        print(f"\n  ERROR: {len(missing_hp)} input-data module(s) missing "
              "from hypatiax/protocols/:")
        for f in missing_hp:
            print(f"    ✗  {f}")
        sys.exit(1)
    print(f"  Protocols : all {len(required_hp)} hypatiax/protocols/ modules ✓")

    # ── NB-06 code-quality pre-audit (non-blocking) ────────────────────────
    nb06 = REPO_ROOT / "notebooks" / "NB-06_Code_Quality_Pipeline_Integrity.ipynb"
    if nb06.exists():
        try:
            r = subprocess.run(
                ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                 "--inplace", "--ExecutePreprocessor.timeout=120", str(nb06)],
                capture_output=True, text=True, timeout=150,
            )
            status = "✓" if r.returncode == 0 else "⚠ warnings (non-blocking)"
            print(f"  NB-06     : code quality pre-audit {status}")
        except FileNotFoundError:
            print("  NB-06     : jupyter not found — skipped (pip install notebook)")
        except Exception as exc:
            print(f"  NB-06     : skipped ({exc})")

    # ── verify-only shortcut ───────────────────────────────────────────────
    if args.verify_only:
        banner("Verify-only mode")
        subprocess.run(["python3", "scripts/patches/verify_results.py", "--report"],
                       check=False)
        subprocess.run(["python3", "reproducibility/hash_lock.py", "--check"],
                       check=False)
        sys.exit(0)

    # ── Build environment (mirrors run_all.sh and notebook cell 2) ─────────
    _seed_str = str(args.seed) if args.seed is not None else "42"

    env = {**os.environ}
    env["PYTHONWARNINGS"] = "ignore"
    env["NN_SEED"]               = os.environ.get("NN_SEED",   _seed_str)
    env["PYSR_SEED"]             = os.environ.get("PYSR_SEED", _seed_str)
    env["PYTHONHASHSEED"]        = os.environ.get("PYTHONHASHSEED", _seed_str)
    # If --seed was given explicitly it overrides any pre-existing env value.
    if args.seed is not None:
        env["NN_SEED"]        = _seed_str
        env["PYSR_SEED"]      = _seed_str
        env["PYTHONHASHSEED"] = _seed_str
    env.setdefault("LLM_MODEL",             "claude-sonnet-4-20250514")
    env.setdefault("LLM_RETRIES",           "3")
    env.setdefault("LLM_K_RUNS",            "1")   # overridden to 30 for instability
    env.setdefault("N_TASKS_DEFI",          "74")
    env.setdefault("N_TASKS_INSTABILITY",   "70")  # FIX-T1: must be 70 not 71
    env.setdefault("PCA_TRAIN_FRAC",        "0.40")
    env.setdefault("NN_TIME_LIMIT",         "120")
    env.setdefault("ENGINE_NAME",           "hybrid_system_v50_2")  # FIX-C2: never v40
    env.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    env["PYTHONPATH"]  = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["RESULTS_DIR"] = str(RESULTS_DIR)
    env["REPRO_ROOT"]  = str(REPO_ROOT)

    _seed_source = f"--seed flag" if args.seed is not None else "default (env or 42)"
    print(f"\n  NN_SEED={env['NN_SEED']}  PYSR_SEED={env['PYSR_SEED']}  "
          f"PYTHONHASHSEED={env['PYTHONHASHSEED']}  (source: {_seed_source})")
    print(f"  LLM_MODEL={env['LLM_MODEL']}")
    print(f"  ENGINE={env['ENGINE_NAME']}  N_TASKS_INSTABILITY={env['N_TASKS_INSTABILITY']}")
    if args.skip_paper:
        print(f"  --skip-paper: Phase 4-B notebook steps will be skipped")

    # ── Validate --only / --from ───────────────────────────────────────────
    if args.only and args.only not in STEP_IDS:
        print(f"\n  ERROR: unknown step id '{args.only}'.")
        print(f"  Valid ids: {', '.join(STEP_IDS)}")
        sys.exit(1)
    if args.from_step and args.from_step not in STEP_IDS:
        print(f"\n  ERROR: unknown step id '{args.from_step}'.")
        print(f"  Valid ids: {', '.join(STEP_IDS)}")
        sys.exit(1)

    # ── Load checkpoint state ──────────────────────────────────────────────
    checkpoint_state: dict[str, str] = {}
    if args.resume:
        checkpoint_state = load_checkpoint()
        if checkpoint_state:
            n_prev = sum(1 for v in checkpoint_state.values() if v == "pass")
            print(f"\n  ── Resuming: {n_prev} step(s) already passed in checkpoint")
            if args.from_step:
                print(f"  ── --from {args.from_step}: force-rerun from here onwards")
        else:
            print("\n  ── No checkpoint found — running full pipeline")

    # ── Run pipeline ───────────────────────────────────────────────────────
    results: list[StepResult] = []
    current_phase = ""
    t_total = time.time()
    # once we reach --from step, all subsequent steps run regardless of checkpoint
    past_from = args.from_step is None

    for step in STEPS:
        if step.phase != current_phase:
            banner(f"Phase {step.phase}")
            current_phase = step.phase

        if args.from_step and step.id == args.from_step:
            past_from = True

        # --only filter
        if args.only and step.id != args.only:
            results.append(StepResult(step.id, step.label, "skip"))
            print(f"  ── skip [{step.id}]  (--only {args.only})")
            continue

        # --resume: skip steps that already passed, unless we're past --from
        if (args.resume
                and checkpoint_state.get(step.id) == "pass"
                and not past_from):
            results.append(StepResult(step.id, step.label, "resume-skip"))
            print(f"  ↩  skip [{step.id}]  (checkpoint: already passed)")
            continue

        # --skip-slow filter
        if args.skip_slow and step.slow:
            results.append(StepResult(step.id, step.label, "skip"))
            print(f"  ── skip [{step.id}]  (--skip-slow)")
            continue

        # FIX: --skip-paper filter — previously parsed but never applied
        if args.skip_paper and step.paper:
            results.append(StepResult(step.id, step.label, "skip"))
            print(f"  ── skip [{step.id}]  (--skip-paper)")
            continue

        result = run_step(step, env)
        results.append(result)

        # save checkpoint after every step
        checkpoint_state[step.id] = result.status
        save_checkpoint(checkpoint_state)

        if result.status == "fail" and not args.continue_on_fail:
            print(f"\n  Pipeline aborted at [{step.id}].")
            print(f"  Checkpoint saved → {CHECKPOINT}")
            print(f"  To resume:         python3 run_all.py --resume")
            print(f"  To rerun this step: python3 run_all.py --only {step.id}")
            _print_summary(results, time.time() - t_total)
            sys.exit(1)

    _print_summary(results, time.time() - t_total)

    # Clear checkpoint only after a complete, fully-passing run
    failed = [r for r in results if r.status == "fail"]
    if not failed and not args.only:
        clear_checkpoint()

    sys.exit(1 if failed else 0)


def _print_summary(results: list[StepResult], elapsed: float) -> None:
    passed       = [r for r in results if r.status == "pass"]
    failed       = [r for r in results if r.status == "fail"]
    skipped      = [r for r in results if r.status == "skip"]
    resume_skips = [r for r in results if r.status == "resume-skip"]

    hh, rem = divmod(int(elapsed), 3600)
    mm, ss  = divmod(rem, 60)

    banner("Pipeline summary")
    col = {"pass": "✓", "fail": "✗", "skip": "─", "resume-skip": "↩"}
    for r in results:
        t = f"  {r.elapsed:6.0f}s" if r.status in ("pass", "fail") else "        "
        print(f"  {col[r.status]} [{r.id:28s}] {r.label[:48]:48s}{t}")

    print()
    print(f"  ✓ passed      : {len(passed)}")
    print(f"  ✗ failed      : {len(failed)}")
    print(f"  ─ skipped     : {len(skipped)}")
    print(f"  ↩ resume-skip : {len(resume_skips)}")
    print(f"  Wall time     : {hh:02d}:{mm:02d}:{ss:02d}")

    # Result file inventory
    data_files, fig_files, tbl_files = inventory_results()
    print(f"\n  Results → {RESULTS_DIR}")
    print(f"    Data files (JSON+CSV) : {data_files}")
    print(f"    Figures (PDF)         : {fig_files}")
    print(f"    Tables  (TeX)         : {tbl_files}")

    # Provenance coverage
    prov = LOG_DIR / "provenance_audit" / "provenance_audit_summary.txt"
    if prov.exists():
        print(f"\n  Provenance ({prov}):")
        for line in prov.read_text().splitlines():
            if any(k in line for k in ("AUTHORITATIVE", "ORPHAN", "Total")):
                print(f"    {line.strip()}")

    if failed:
        print("\n  Failed steps:")
        for r in failed:
            print(f"    [{r.id}] → {r.log_path}")
        print()
        print(f"  Checkpoint : {CHECKPOINT}")
        print("  Resume     : python3 run_all.py --resume")
        print("  Single step: python3 run_all.py --only <id>")
    else:
        print("\n  ✓ All steps passed.")
        print(f"  Results    : {RESULTS_DIR}/")
        print(f"  Figures    : {RESULTS_DIR}/figures/")
        print(f"  Tables     : {RESULTS_DIR}/tables/")
        print(f"  Provenance : {LOG_DIR}/provenance_audit/")
        print(f"  Import DAG : {LOG_DIR}/repro_output/import_graph.dot")
        print("  Checkpoint : cleared (all steps passed)")


if __name__ == "__main__":
    main()

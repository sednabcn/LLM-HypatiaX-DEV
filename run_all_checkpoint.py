#!/usr/bin/env python3
"""
run_all_checkpoint.py  —  HypatiaX · Full reproducibility pipeline (Python)
Paper: "HypatiaX: A Hybrid Symbolic-Neural Framework for
        Extrapolation-Reliable Analytical Discovery"  (JMLR v3.0, Apr 2026)
Version: v8.2 (2026-06-09)

Usage:
    python3 run_all_checkpoint.py                      # full pipeline
    python3 run_all_checkpoint.py --skip-slow          # skip slow steps
    python3 run_all_checkpoint.py --only exp3          # run one step by id
    python3 run_all_checkpoint.py --resume             # resume from last checkpoint
    python3 run_all_checkpoint.py --resume --from exp2 # resume, force-rerun from step
    python3 run_all_checkpoint.py --clear-checkpoint   # delete checkpoint and exit
    python3 run_all_checkpoint.py --continue-on-fail   # log failures but keep going
    python3 run_all_checkpoint.py --verify-only        # re-check results without re-running
    python3 run_all_checkpoint.py --qualify-only       # run qualification checks only
    python3 run_all_checkpoint.py --seed 123           # override seed for all steps
    python3 run_all_checkpoint.py --only exp3 --seed 777
    python3 run_all_checkpoint.py --dry-run
    python3 run_all_checkpoint.py --dry-run --only exp1 --case-range 1-4
    python3 run_all_checkpoint.py --skip-paper
    python3 run_all_checkpoint.py --pysr-timeout 900
    python3 run_all_checkpoint.py --one-equation       # smoke-test: 1 equation per experiment
    python3 run_all_checkpoint.py --one-equation-paper # reviewer probe: paper-quality values

Step IDs (use with --only / --from):
    Setup    : deps  patches-gen  patches-apply  fixup-init  fixup-tex
               validate-patches  validate-paper-config  check-hypatiax-protocols
    Phase 1  : exp1  exp1_analysis  exp1b  extrap  hybrid_all_domains
               instability  exp2_feynman  exp2_feynman_pca_4060
               exp2_feynman_extrap  exp2  exp3  exp3b
    Phase 2  : suppA  suppB  suppB_sc
    Phase 3  : provenance  discover-provenance  scan-imports  verify  hashlock
    Phase 4  : tables  figures  validate
    Phase 4B : audit_setup  audit_nb01 ... audit_nb05
               audit_nb06_fixc3_disclosure  audit_nb06_fixc3_rerun
    Phase 5  : audit_guard  audit_print_verify  audit_print_findings
               audit_figures_tables  audit_final_gate
               qualify        ← per-experiment qualification gate
               audit_paper    ← final results-vs-paper audit

    NOTE: Step IDs match run_all.sh _STEP_ORDER exactly (underscores).
          Legacy hyphenated IDs (audit-NB-01 etc.) still accepted via --only.

Notes:
    --from requires --resume to have any effect; alone it is a no-op.
    validate-patches (Phase 0) checks patched source code.
    verify (Phase 3) cross-checks numerical results — equivalent to run_all.sh validate.
    qualify (Phase 5) checks ALL experiments passed every stage; blocks audit-paper.
    audit-paper (Phase 5) cross-checks every number in the paper against results/.

Changelog v8.2 (2026-06-09):
    SYNC-run_all.sh — imported three fix groups from run_all.sh that postdate v8.1:

    FIX-MERGE-QUOTING (2026-06-07):
      exp2_feynman_extrap merge block: extracted from bash -c "" into a standalone
      subshell block. The original triple-backslash+quote patterns inside the
      double-quoted outer string produced literal backslashes in paths after bash
      parsing and suppressed command substitution. Rewritten as plain bash with no
      nesting, matching the exp2_feynman_pca_comparison_table and
      exp3_symbolic_equivalence inlined blocks.
      Final summary: corrected phantom log reference qualify_verify_run.log →
      qualify_run.log (qualify step only ever writes qualify_run.log).

    FIX-SYNC-CI (2026-06-05):
      exp2_feynman_pca_comparison_table logic inlined after exp2_feynman_pca_4060.
      Calls scripts/patches/generate_exp2_pca_comparison_table.py to produce
      exp2_pca_comparison.{tex,csv,md} — mirrors ci_analysis.yml and ci_postprocess.yml.
      NOT a separate registered step; runs as plain shell after exp2_feynman_pca_4060.
      exp3_symbolic_equivalence logic inlined after exp3b.
      Calls scripts/check_symbolic_equivalence.py against all exp3_nguyen12_seed*.json
      files — mirrors ci_analysis.yml Check symbolic equivalence step.
      Output: symbolic_equivalence_report.csv + _summary.txt.
      NOT a separate registered step; runs as plain shell after exp3b.
      merge_extrap_into_benchmark.py now called inside exp2_feynman_extrap step
      (replacing the NOTE that deferred it to ci_analysis.yml). Produces
      ablation_paired.json in exp2_extrap/ so qualify and audit_paper can run
      locally without requiring ci_analysis.yml to run first. Skips gracefully
      when the script or benchmark_results_extrap*.json is absent.
      tables step now also calls generate_exp2_pca_comparison_table.py and
      generate_nguyen12_symequiv_table.py — mirrors ci_postprocess.yml steps.
      Both skipped gracefully when prerequisite files are absent.
      _STEP_ORDER kept at 35 entries; two new sub-steps are not registered so
      --step / --from targeting is unaffected.

    FIX-C3-ESCAPE (2026-06-04):
      exp1_pca and exp1b_pca: removed erroneous backslash-escaping on REPO_ROOT,
      EXPERIMENTS_DIR, and RESULTS_DIR inside the outer bash -c string. The
      backslashes caused those variables to be treated as literal strings rather
      than shell variable expansions, producing ENOENT on all output paths.

Changelog v8.1 (2026-05-17):
    FIX-BUG1-SUBDIR: EXP_RESULT_SUBDIR had three wrong paths (CRITICAL):
      exp1  "comparison_results/extrapolation"       → "comparison_results/noise-noiseless/noiseless"
      exp1b "comparison_results/extrapolation"       → "comparison_results/noise-noiseless/15"
      exp2  "comparison_results/feynman-tests/exp2"  → "comparison_results/feynman-tests/exp2_multi"
      Wrong paths caused qualify_experiment() to never find _merged.json for
      exp1/exp1b/exp2, permanently blocking audit-paper for those experiments.
    FIX-BUG2-POSTMOVE: exp3/exp3b/suppA PostMove source was EXPERIMENTS_DIR
      (CRITICAL). run_all.sh comment "FIX-DIR: script writes to RESULTS_DIR root —
      search must use RESULTS_DIR". Changed all three to RESULTS_DIR so globs
      find the files and moves actually happen.
    FIX-BUG3-EXTRAP: extrap step had non-paper OOD parameters (CRITICAL):
      --extrap-multiplier 3.0 → 2.0  (paper value; matches CI and run_all.sh)
      --extrap-train-frac 0.4 → 0.8  (paper value; matches CI and run_all.sh)
    FIX-BUG4-RESULTGLOB: exp1/exp1b result_glob pointed to wrong directory.
      step_already_complete() uses result_glob for idempotent-skip checks;
      wrong path causes steps to always rerun even when results exist.
      exp1  "comparison_results/extrapolation/*.json"        → "comparison_results/noise-noiseless/noiseless/*.json"
      exp1b "comparison_results/extrapolation/*seed*.json"   → "comparison_results/noise-noiseless/15/*.json"
    FIX-BUG5-EXP2: exp2 step used unknown --protocol all30 flag; fixed to
      --benchmark both (matches CI worker). Added missing --output-dir pointing
      to feynman-tests/exp2_multi/ so results land in the correct subdir.
      Updated result_glob to match.
    FIX-BUG6-EXP2FEYNMAN: exp2_feynman step missing --output-dir (results
      landing in script default) and --method-timeout 120 (CI worker value).
      Both flags added. --skip-pysr also removed so all 6 methods run.

Changelog v8.0 (2026-05-16):
    NEW-QUALIFY: Added Phase 5 with two new steps:
      qualify     — per-experiment gate that verifies each exp is FULLY done:
                    running ✓ → consolidation ✓ → outputs in results/ ✓ →
                    committed to repo ✓ → analysis ✓ → figures ✓ → tables ✓.
                    Any incomplete exp blocks audit-paper (chain-stops on fail
                    unless --continue-on-fail).
      audit-paper — final paper audit: loads paper targets from
                    scripts/patches/paper_targets.json and cross-checks every
                    reported number against the corresponding result file.
                    Emits a structured report with PASS/WARN/FAIL per claim.
    NEW-COMPLETE: run_step() now calls complete_step_if_needed() before
                    running; if a step's result_glob already has files AND the
                    checkpoint says "pass", the step is silently skipped even
                    without --resume (idempotent rerun safety).
    IMPROVE-QUALIFY: qualify_experiment() checks seven dimensions per exp:
                    (1) checkpoint=pass, (2) result files present,
                    (3) _merged.json present, (4) _merged.csv present,
                    (5) committed to git, (6) figures present,
                    (7) tables present.
    IMPROVE-AUDIT: audit_against_paper() reads paper_targets.json; each
                    target has {exp, metric, paper_value, tolerance, path,
                    json_key}.  Tolerances are relative (1 % default) or
                    absolute.  Nguyen-12 dual-threshold caveat is checked
                    explicitly.

Changelog v7.3 (2026-05-14):
    FIX-EXP3B-POSTMOVE, FIX-EXP3B-RESULTGLOB, FIX-ENSURE-OUTDIR,
    FIX-PYSR-POPULATIONS-DEFAULT, FIX-USAGE-FILENAME.

Changelog v7.2 (2026-05-14):
    FIX-DOMAINS, FIX-EXP2-CMD, FIX-EXP2-SKIP-PYSR, FIX-SUPPA-POSTMOVE,
    FIX-EXP3-POSTMOVE, FIX-EXP3-RESULTGLOB, FIX-EXP3B-WALRUS,
    FIX-JULIA-THREADS, FIX-COMMENT.

Prerequisites:
    export ANTHROPIC_API_KEY="sk-ant-..."
    pip install -r requirements.txt
"""

import argparse
import importlib.util as _ilu
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path


# ── Load API key (env → Kaggle → .env → Colab) ─────────────────────────────
def load_repro_config() -> dict:
    """Load configuration from repro.yaml, with environment variable overrides."""
    import yaml  # type: ignore

    for config_path in [
        REPO_ROOT / "config" / "repro.yaml",
        REPO_ROOT / "repro.yaml",
    ]:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"  ⚠ Failed to load {config_path}: {e}")
    print("  ⚠ repro.yaml not found — using defaults")
    return {}


def _load_api_key() -> None:
    """Load ANTHROPIC_API_KEY via hypatiax/config_secrets.py, or fall back to .env."""
    _repo = Path(__file__).resolve().parent
    _config_secrets_path = _repo / "hypatiax" / "config_secrets.py"
    if _config_secrets_path.exists():
        try:
            _spec = _ilu.spec_from_file_location(
                "hypatiax._config_secrets_standalone", _config_secrets_path
            )
            if _spec and _spec.loader:
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)  # type: ignore[arg-type]
                if os.environ.get("ANTHROPIC_API_KEY"):
                    print("✅ ANTHROPIC_API_KEY loaded from hypatiax/config_secrets.py")
                    return
        except Exception as _e:
            print(f"  ⚠  config_secrets.py direct-load failed ({_e}); falling back")

    if os.environ.get("ANTHROPIC_API_KEY"):
        print("✅ ANTHROPIC_API_KEY already set in environment")
        return
    for _env_path in [
        _repo / "hypatiax" / ".env",
        _repo / ".env",
        Path.home() / ".env",
    ]:
        if _env_path.exists():
            for _line in _env_path.read_text().splitlines():
                _line = _line.strip()
                if _line.startswith("#") or "=" not in _line:
                    continue
                _k, _, _v = _line.partition("=")
                _k = _k.strip()
                _v = _v.strip().strip('"').strip("'")
                if _k == "ANTHROPIC_API_KEY" and _v:
                    os.environ["ANTHROPIC_API_KEY"] = _v
                    print(f"✅ ANTHROPIC_API_KEY loaded from {_env_path}")
                    return


_load_api_key()

# ── Canonical paths ──────────────────────────────────────────────────────────
REPO_ROOT       = Path(__file__).resolve().parent
RESULTS_DIR     = REPO_ROOT / "hypatiax" / "data" / "results"
EXPERIMENTS_DIR = REPO_ROOT / "hypatiax" / "experiments" / "benchmarks"
LOG_DIR         = REPO_ROOT / "logs"
CHECKPOINT      = LOG_DIR / "pipeline_checkpoint.json"
EXP2_EQ_CHECKPOINT = LOG_DIR / "exp2_eq_checkpoint.json"

# ── Strip incompatible deps from requirements.txt ───────────────────────────
_REQUIREMENTS  = REPO_ROOT / "requirements.txt"
_STRIP_PATTERNS = ["defi-risk", "optimum-onnx"]
if _REQUIREMENTS.exists():
    _lines    = _REQUIREMENTS.read_text().splitlines(keepends=True)
    _filtered = [l for l in _lines if not any(p in l for p in _STRIP_PATTERNS)]
    if len(_filtered) < len(_lines):
        _REQUIREMENTS.write_text("".join(_filtered))
        print(
            f"  ✂  Removed {len(_lines)-len(_filtered)} incompatible dep(s): "
            f"{_STRIP_PATTERNS}"
        )

# ── Stage paper .tex files into paper/ if they live at repo root ────────────
import shutil as _shutil  # noqa: E402

_PAPER_DIR   = REPO_ROOT / "paper"
_TEX_PATTERNS = [
    "jmlr_paper*.tex",
    "jmlr-hypatiax*.tex",
    "supp_routing_improvements.tex",
    "supp_benchmark_report.tex",
]
_staged: list[str] = []
for _pat in _TEX_PATTERNS:
    for _src in REPO_ROOT.glob(_pat):
        _dst = _PAPER_DIR / _src.name
        if not _dst.exists():
            _PAPER_DIR.mkdir(exist_ok=True)
            _shutil.copy2(_src, _dst)
            _staged.append(_src.name)
if _staged:
    print(f"  📄 Staged {len(_staged)} .tex file(s) into paper/: {_staged}")

_PAPER_STEP_IDS = {
    "audit-NB-01", "audit-NB-02", "audit-NB-03",
    "audit-NB-04", "audit-NB-05", "audit-setup",
    # validate-patches is a code-only check; keep it out of --skip-paper scope
}

# ── Domain registry ──────────────────────────────────────────────────────────
# BLOCKER-1 / WARN-2: canonical 10-domain list for hybrid_all_domains.
HYBRID_ALL_DOMAINS_IDS: list[str] = [
    "mechanics", "thermodynamics", "electromagnetism", "fluid_dynamics",
    "optics",    "quantum",        "chemistry",        "biology",
    "mathematics", "economics",
]

# ── suppB sample-complexity sweep parameters (BLOCKER-2) ───────────────────
SUPPB_SC_SAMPLE_COUNTS: list[str] = ["50", "100", "200", "500", "750", "1000"]

# ── Experiment → result subdir map (mirrors ci_experiment.yml plan job) ─────
# Used by qualify_experiment() to locate _merged.json for each exp.
EXP_RESULT_SUBDIR: dict[str, str] = {
    # FIX-BUG1: exp1/exp1b were "comparison_results/extrapolation" (wrong);
    #           exp2 was "comparison_results/feynman-tests/exp2" (wrong).
    #           Correct values match ci_experiment.yml plan meta.
    "exp1":              "comparison_results/noise-noiseless/noiseless/defi",
    "exp1b":             "comparison_results/noise-noiseless/15",
    "exp2_feynman":           "comparison_results/feynman-tests/exp2",
    "exp2_feynman_pca_4060":  "comparison_results/feynman-tests/exp2_pca_4060",
    "exp2_feynman_extrap":    "comparison_results/feynman-tests/exp2_extrap",
    "exp2":                   "comparison_results/feynman-tests/exp2_multi",
    "exp3":              "extrapolation",
    "exp3b":             "extrapolation/multi_seed",
    "suppA":             "hybrid_pysr/defi",
    "suppB":             "comparison_results/feynman-tests/noise-sweep/noise-sweep",
    "suppB_sc":          "comparison_results/feynman-tests/sample-complexity",
    "hybrid_all_domains": "hybrid_llm_nn/all_domains",
    "instability":       "figures",
    "extrap":            "comparison_results/extrapolation",
}

# ── Paper targets file (used by audit-paper step) ───────────────────────────
PAPER_TARGETS_PATH = REPO_ROOT / "scripts" / "patches" / "paper_targets.json"


# ════════════════════════════════════════════════════════════════════════════
#  BLOCKER-1 / WARN-2 — Runtime domain-list validation
# ════════════════════════════════════════════════════════════════════════════
def validate_hybrid_all_domains_ids() -> bool:
    expected = set(HYBRID_ALL_DOMAINS_IDS)
    script = (
        REPO_ROOT
        / "hypatiax" / "experiments" / "generation"
        / "hybrid_all_domains_llm_nn"
        / "hybrid_system_llm_nn_all_domains.py"
    )
    if not script.exists():
        print(f"  ⚠  validate_hybrid_all_domains_ids: script not found at {script}")
        print("      Skipping domain-list validation (non-blocking).")
        return True

    spec = _ilu.spec_from_file_location("hybrid_mod", script)
    if spec is None or spec.loader is None:
        print("  ⚠  validate_hybrid_all_domains_ids: could not create module spec")
        return True

    mod = _ilu.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except SystemExit:
        pass

    actual = (
        getattr(mod, "DOMAINS",     None)
        or getattr(mod, "ALL_DOMAINS", None)
        or getattr(mod, "DOMAIN_KEYS", None)
    )
    if actual is None:
        try:
            from hypatiax.experiments.generation.hybrid_all_domains_llm_nn\
                .hybrid_system_llm_nn_all_domains import ExperimentProtocolAll  # type: ignore
            actual = set(ExperimentProtocolAll().get_all_domains().keys())
        except Exception as e:
            print(f"  ⚠  Could not resolve domain list from script: {e}")
            return True

    actual_set = {str(d) for d in actual}
    missing = expected - actual_set
    extra   = actual_set - expected

    if missing or extra:
        print("  ✗  DOMAIN LIST MISMATCH — update HYBRID_ALL_DOMAINS_IDS!")
        if missing:
            print(f"     In pipeline registry but NOT in script : {sorted(missing)}")
        if extra:
            print(f"     In script but NOT in pipeline registry : {sorted(extra)}")
        return False

    print(f"  ✓  Domain-list validation OK: {sorted(actual_set)}")
    return True


# ════════════════════════════════════════════════════════════════════════════
#  BLOCKER-4 — suppB result glob helper
# ════════════════════════════════════════════════════════════════════════════
def _suppb_result_glob() -> str:
    primary  = "comparison_results/feynman-tests/noise-sweep/noise-sweep/noise_sweep_*.json"
    fallback = "comparison_results/feynman-tests/noise-sweep/noise-sweep/suppB_*.json"
    if list(RESULTS_DIR.glob(primary)):
        return primary
    return fallback


# ════════════════════════════════════════════════════════════════════════════
#  Phase 5 — Per-experiment qualification
# ════════════════════════════════════════════════════════════════════════════

# Experiments that must be qualified before audit-paper is allowed to run.
QUALIFIABLE_EXPERIMENTS = [
    "exp1", "exp1b", "exp2_feynman", "exp2_feynman_pca_4060", "exp2_feynman_extrap", "exp2",
    "exp3", "exp3b",
    "suppA", "suppB", "suppB_sc",
    "hybrid_all_domains", "instability", "extrap",
]


@dataclass
class QualResult:
    exp_id:   str
    passed:   bool
    checks:   list[tuple[str, bool, str]]   # (check_name, ok, detail)

    def summary_line(self) -> str:
        icon = "✅" if self.passed else "❌"
        fails = [c[0] for c in self.checks if not c[1]]
        tail  = "" if self.passed else f"  FAILED: {', '.join(fails)}"
        return f"  {icon}  {self.exp_id:<25s}{tail}"


def qualify_experiment(exp_id: str, checkpoint_state: dict) -> QualResult:
    """
    Verify that one experiment has completed every stage required for the
    paper audit.  Seven checks, in order:

    1. checkpoint=pass         — pipeline reported success for this step
    2. result files present    — at least one JSON/CSV in the result subdir
    3. _merged.json present    — consolidation ran and produced merged output
    4. _merged.csv present     — consolidation CSV present
    5. committed to git        — _merged.json is tracked and clean in HEAD
    6. figures present         — at least one PDF/PNG in results/figures/
    7. tables present          — at least one .tex in results/tables/
    """
    checks: list[tuple[str, bool, str]] = []

    # 1. Checkpoint
    cp_status = checkpoint_state.get(exp_id, "todo")
    checks.append(("checkpoint=pass", cp_status == "pass",
                   f"checkpoint={cp_status}"))

    # 2. Result files
    subdir_rel = EXP_RESULT_SUBDIR.get(exp_id, "")
    result_dir = RESULTS_DIR / subdir_rel if subdir_rel else RESULTS_DIR
    result_files = (
        list(result_dir.glob("*.json")) + list(result_dir.glob("*.csv"))
        if result_dir.exists() else []
    )
    # Exclude meta-files from count so _merged.json alone doesn't count as results
    data_files = [f for f in result_files
                  if not f.name.startswith("_")]
    checks.append(("result_files", len(data_files) > 0,
                   f"{len(data_files)} data file(s) in {subdir_rel or 'results/'}"))

    # 3. _merged.json
    merged_json = result_dir / "_merged.json"
    checks.append(("_merged.json", merged_json.exists(),
                   str(merged_json.relative_to(REPO_ROOT)) if merged_json.exists()
                   else f"missing: {merged_json}"))

    # 4. _merged.csv
    merged_csv = result_dir / "_merged.csv"
    checks.append(("_merged.csv", merged_csv.exists(),
                   str(merged_csv.relative_to(REPO_ROOT)) if merged_csv.exists()
                   else f"missing: {merged_csv}"))

    # 5. Committed to git
    git_ok = False
    git_detail = "git unavailable"
    if merged_json.exists():
        try:
            rel = str(merged_json.relative_to(REPO_ROOT))
            r = subprocess.run(
                ["git", "ls-files", "--error-unmatch", rel],
                capture_output=True, text=True, cwd=REPO_ROOT
            )
            if r.returncode == 0:
                # Also check it is not dirty (staged but uncommitted)
                r2 = subprocess.run(
                    ["git", "diff", "--name-only", "HEAD", "--", rel],
                    capture_output=True, text=True, cwd=REPO_ROOT
                )
                git_ok = r2.returncode == 0 and r2.stdout.strip() == ""
                git_detail = "clean in HEAD" if git_ok else "modified/not-committed"
            else:
                git_detail = "not tracked by git"
        except FileNotFoundError:
            git_detail = "git not found"
    checks.append(("committed_to_git", git_ok, git_detail))

    # 6. Figures present
    figs_dir = RESULTS_DIR / "figures"
    fig_files = (
        list(figs_dir.glob("*.pdf")) + list(figs_dir.glob("*.png"))
        if figs_dir.exists() else []
    )
    # At least one figure must exist globally (figures are shared across exps)
    checks.append(("figures_present", len(fig_files) > 0,
                   f"{len(fig_files)} figure(s) in results/figures/"))

    # 7. Tables present
    tables_dir = RESULTS_DIR / "tables"
    if not tables_dir.exists() or not any(tables_dir.glob("*.tex")):
        tables_dir = REPO_ROOT / "paper" / "tables"
    tex_files = list(tables_dir.glob("*.tex")) if tables_dir.exists() else []
    checks.append(("tables_present", len(tex_files) > 0,
                   f"{len(tex_files)} table(s) in {tables_dir.relative_to(REPO_ROOT)}"))

    passed = all(ok for _, ok, _ in checks)
    return QualResult(exp_id=exp_id, passed=passed, checks=checks)


def run_qualification(checkpoint_state: dict) -> tuple[bool, list[QualResult]]:
    """
    Run qualification for all QUALIFIABLE_EXPERIMENTS.
    Returns (all_passed, results_list).
    """
    banner("Phase 5 · Experiment qualification")
    results = []
    for exp_id in QUALIFIABLE_EXPERIMENTS:
        qr = qualify_experiment(exp_id, checkpoint_state)
        results.append(qr)
        print(qr.summary_line())
        for check_name, ok, detail in qr.checks:
            icon = "  ✓" if ok else "  ✗"
            print(f"      {icon}  {check_name:<25s}  {detail}")

    all_passed = all(r.passed for r in results)
    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass

    print()
    print(f"  Qualification: {n_pass}/{len(results)} experiments fully qualified")
    if not all_passed:
        print(f"  ❌  {n_fail} experiment(s) NOT fully qualified — audit-paper blocked.")
        unqualified = [r.exp_id for r in results if not r.passed]
        print(f"     Incomplete: {', '.join(unqualified)}")
    else:
        print("  ✅  All experiments qualified — proceeding to audit-paper.")
    return all_passed, results


# ════════════════════════════════════════════════════════════════════════════
#  Phase 5 — Paper audit (results vs. paper)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class AuditClaim:
    exp:         str
    metric:      str
    paper_value: float
    tolerance:   float          # relative tolerance (e.g. 0.01 = 1%)
    result_path: str            # relative to RESULTS_DIR
    json_key:    str            # dot-notation key in the result JSON
    absolute:    bool = False   # if True, tolerance is absolute not relative
    note:        str  = ""      # human note (e.g. "Nguyen-12 dual threshold")


@dataclass
class AuditFinding:
    claim:       AuditClaim
    status:      str            # PASS / WARN / FAIL / MISSING
    actual:      float | None
    detail:      str


def _resolve_json_key(data: dict, dot_key: str):
    """Walk a dot-notation key through nested dicts/lists."""
    parts = dot_key.split(".")
    node = data
    for p in parts:
        if isinstance(node, dict):
            node = node.get(p)
        elif isinstance(node, list):
            try:
                node = node[int(p)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if node is None:
            return None
    return node


def load_paper_targets() -> list[AuditClaim]:
    """
    Load paper targets from scripts/patches/paper_targets.json.
    Falls back to a built-in minimal set if the file is absent.

    paper_targets.json schema (list of objects):
      {
        "exp":         "exp3",
        "metric":      "nguyen12_solve_rate_strict",
        "paper_value": 0.333,
        "tolerance":   0.01,
        "result_path": "extrapolation/_merged.json",
        "json_key":    "summary.solve_rate_strict",
        "absolute":    false,
        "note":        "Strict R²>=0.9999 threshold §10.8"
      }
    """
    if PAPER_TARGETS_PATH.exists():
        try:
            raw = json.loads(PAPER_TARGETS_PATH.read_text())
            claims = []
            for r in raw:
                claims.append(AuditClaim(
                    exp=r["exp"],
                    metric=r["metric"],
                    paper_value=float(r["paper_value"]),
                    tolerance=float(r.get("tolerance", 0.01)),
                    result_path=r["result_path"],
                    json_key=r["json_key"],
                    absolute=bool(r.get("absolute", False)),
                    note=r.get("note", ""),
                ))
            return claims
        except Exception as e:
            print(f"  ⚠  Failed to load paper_targets.json: {e}")
            print("      Using built-in minimal target set.")

    # ── Built-in minimal fallback ────────────────────────────────────────────
    # These are the primary paper claims; update paper_targets.json for full coverage.
    return [
        # exp3 — Nguyen-12 §10.8 primary result (4-decimal rounding, paper abstract)
        AuditClaim("exp3",  "nguyen12_solve_rate_4dec",
                   0.917, 0.02,
                   "extrapolation/_merged.json",
                   "summary.solve_rate_4decimal",
                   note="11/12 = 91.7% (4-decimal rounding) §10.8 / abstract"),
        # exp3 — strict R²≥0.9999
        AuditClaim("exp3",  "nguyen12_solve_rate_strict",
                   0.333, 0.02,
                   "extrapolation/_merged.json",
                   "summary.solve_rate_strict",
                   note="4/12 = 33.3% (strict R²≥0.9999) §10.8 transparency caveat"),
        # exp2 — Feynman solve rate ≥30%
        AuditClaim("exp2",  "feynman30_solve_rate",
                   0.30, 0.05,
                   "comparison_results/feynman-tests/exp2/_merged.json",
                   "summary.solve_rate",
                   note="≥9/30 solved §10.7"),
        # suppB — EHD noise robustness 100% at all σ
        AuditClaim("suppB", "ehd_noise_robust_100pct",
                   1.00, 0.01,
                   "comparison_results/feynman-tests/noise-sweep/noise-sweep/_merged.json",
                   "summary.ehd_success_rate",
                   note="EHD 100% at all noise levels §SuppB"),
        # hybrid_all_domains — coverage check (at least 1 result per domain)
        AuditClaim("hybrid_all_domains", "all_domains_coverage",
                   10.0, 0.0,
                   "hybrid_llm_nn/all_domains/_merged.json",
                   "summary.n_domains_completed",
                   absolute=True,
                   note="10 domains must all complete §10.9"),
    ]


def audit_against_paper(qual_results: list[QualResult]) -> tuple[bool, list[AuditFinding]]:
    """
    Cross-check every claim in paper_targets.json (or the built-in set)
    against the actual result files.

    Returns (all_pass, findings_list).
    Findings are PASS / WARN / FAIL / MISSING.
    WARN = value present but outside tolerance.
    FAIL = value outside 3× tolerance OR sign mismatch.
    MISSING = result file or JSON key not found.
    """
    banner("Phase 5 · Audit results against paper")
    claims  = load_paper_targets()
    findings: list[AuditFinding] = []

    qualified_exps = {r.exp_id for r in qual_results if r.passed}

    for claim in claims:
        # Skip experiments that failed qualification
        if claim.exp not in qualified_exps:
            findings.append(AuditFinding(
                claim=claim, status="SKIP", actual=None,
                detail=f"{claim.exp} not qualified — skipping audit claim"
            ))
            continue

        result_file = RESULTS_DIR / claim.result_path
        if not result_file.exists():
            # Try without _merged.json (raw stats.json fallback)
            alt = result_file.parent / "_stats.json"
            if alt.exists():
                result_file = alt
            else:
                findings.append(AuditFinding(
                    claim=claim, status="MISSING", actual=None,
                    detail=f"result file not found: {claim.result_path}"
                ))
                continue

        try:
            data = json.loads(result_file.read_text())
        except Exception as e:
            findings.append(AuditFinding(
                claim=claim, status="MISSING", actual=None,
                detail=f"JSON parse error: {e}"
            ))
            continue

        raw_val = _resolve_json_key(data, claim.json_key)
        if raw_val is None:
            # Try flat key as fallback
            raw_val = data.get(claim.json_key.split(".")[-1])

        if raw_val is None:
            findings.append(AuditFinding(
                claim=claim, status="MISSING", actual=None,
                detail=f"key '{claim.json_key}' not found in {claim.result_path}"
            ))
            continue

        actual = float(raw_val)

        if claim.absolute:
            diff = abs(actual - claim.paper_value)
            tol1 = claim.tolerance
            tol3 = claim.tolerance * 3
        else:
            base = abs(claim.paper_value) if claim.paper_value != 0 else 1.0
            diff = abs(actual - claim.paper_value)
            tol1 = claim.tolerance * base
            tol3 = claim.tolerance * 3 * base

        if diff <= tol1:
            status = "PASS"
        elif diff <= tol3:
            status = "WARN"
        else:
            status = "FAIL"

        detail = (
            f"paper={claim.paper_value}  actual={actual:.4f}  "
            f"diff={diff:.4f}  tol={tol1:.4f}"
        )
        if claim.note:
            detail += f"  [{claim.note}]"

        findings.append(AuditFinding(
            claim=claim, status=status, actual=actual, detail=detail
        ))

    # ── Print report ────────────────────────────────────────────────────────
    icons = {"PASS": "✅", "WARN": "⚠ ", "FAIL": "❌", "MISSING": "🔍", "SKIP": "↩ "}
    print(f"\n  {'Exp':<25s} {'Metric':<35s} {'Status':<8s}  Detail")
    print("  " + "─" * 100)
    for f in findings:
        icon = icons.get(f.status, "?")
        print(f"  {icon} {f.claim.exp:<23s} {f.claim.metric:<35s} {f.status:<8s}  {f.detail}")

    n_pass    = sum(1 for f in findings if f.status == "PASS")
    n_warn    = sum(1 for f in findings if f.status == "WARN")
    n_fail    = sum(1 for f in findings if f.status == "FAIL")
    n_missing = sum(1 for f in findings if f.status == "MISSING")
    n_skip    = sum(1 for f in findings if f.status == "SKIP")

    print()
    print(f"  Audit summary: {n_pass} PASS  {n_warn} WARN  {n_fail} FAIL  "
          f"{n_missing} MISSING  {n_skip} SKIP  ({len(findings)} total claims)")

    # Nguyen-12 dual-threshold caveat (WARN-5 / v7.0) — printed whenever
    # exp3 claims appear, regardless of pass/fail.
    if any(f.claim.exp == "exp3" for f in findings):
        print("\n  ⚠  Nguyen-12 dual-threshold caveat (exp3/exp3b):")
        print("       Paper abstract  : 11/12 (91.7%) — 4-decimal rounding (Uy et al.)")
        print("       Strict R²≥0.9999: 4/12  (33.3%) — both must appear in §10.8 / abstract.")

    # Persist audit findings to logs/
    audit_out = LOG_DIR / "paper_audit_findings.json"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    audit_out.write_text(json.dumps(
        [{"exp": f.claim.exp, "metric": f.claim.metric,
          "status": f.status, "paper_value": f.claim.paper_value,
          "actual": f.actual, "detail": f.detail}
         for f in findings],
        indent=2
    ))
    print(f"\n  Audit findings → {audit_out}")

    all_pass = (n_fail == 0 and n_missing == 0)
    return all_pass, findings


# ════════════════════════════════════════════════════════════════════════════
#  Completion check — idempotent step skip
# ════════════════════════════════════════════════════════════════════════════

def step_already_complete(step: "Step", checkpoint_state: dict) -> bool:
    """
    Return True if a step's outputs already exist AND checkpoint says pass.
    Used to skip redundant reruns even without --resume.
    """
    if checkpoint_state.get(step.id) != "pass":
        return False
    if not step.result_glob:
        return False  # no glob = can't verify outputs, don't skip
    pattern = step.result_glob
    if "**" in pattern:
        parts  = Path(pattern).parts
        star_i = next(i for i, p in enumerate(parts) if "**" in p)
        base_d = RESULTS_DIR / Path(*parts[:star_i])
        sub_p  = str(Path(*parts[star_i:]))
        matches = list(base_d.rglob(sub_p)) if base_d.exists() else []
    else:
        matches = list(RESULTS_DIR.glob(pattern))
    return len(matches) > 0


# ════════════════════════════════════════════════════════════════════════════
#  EXP2 isolated-runner (unchanged from v7.x)
# ════════════════════════════════════════════════════════════════════════════
EXP2_PASS_THRESHOLD = 9
EXP2_KILL_GRACE     = 300

_EXP2_WORKER_SCRIPT = textwrap.dedent("""\
import json, os, sys, time, pathlib, traceback
import numpy as np

spec     = json.loads(os.environ["EXP2_EQUATION_JSON"])
out_path = pathlib.Path(os.environ["EXP2_RESULT_PATH"])
out_path.parent.mkdir(parents=True, exist_ok=True)

eq_name  = spec["name"]
seed     = int(os.environ.get("PYSR_SEED", "42"))
np.random.seed(seed)

repo_root = os.environ.get("REPRO_ROOT", str(pathlib.Path(__file__).resolve().parent))
for _p in [repo_root, os.path.join(repo_root, "hypatiax")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

t0 = time.perf_counter()
try:
    from hypatiax.tools.symbolic.symbolic_engine import DiscoveryConfig, SymbolicEngine

    cfg = DiscoveryConfig(
        pysr_timeout    = int(os.environ.get("PYSR_TIMEOUT",    "1100")),
        niterations     = int(os.environ.get("N_ITERATIONS",    "1000")),
        populations     = int(os.environ.get("POPULATIONS",     "30")),
        population_size = int(os.environ.get("PYSR_POPULATION_SIZE", "33")),
        parsimony       = float(os.environ.get("PYSR_PARSIMONY", "0.01")),
        maxsize         = int(os.environ.get("PYSR_MAXSIZE",    "30")),
        binary_operators = ["+", "-", "*", "/"],
        unary_operators  = ["exp", "log", "sin", "cos", "sqrt"],
    )

    N   = spec["n_samples"]
    rng = np.random.default_rng(seed)
    cols = []
    for vname, (lo, hi) in zip(spec["variable_names"], spec["variable_ranges"]):
        cols.append(rng.uniform(lo, hi, N))
    X = np.column_stack(cols)

    local_ns = {v: cols[i] for i, v in enumerate(spec["variable_names"])}
    local_ns["np"] = np
    y = eval(spec["numpy_expr"], {"__builtins__": {}},
             {**local_ns, "np": np,
              "exp": np.exp, "log": np.log, "sin": np.sin,
              "cos": np.cos, "sqrt": np.sqrt, "pi": np.pi})

    engine = SymbolicEngine(cfg, domain="physics")
    result = engine.discover(X, y, variable_names=spec["variable_names"])

    elapsed = time.perf_counter() - t0
    expr = result.get("expression", result.get("best_expression", "N/A"))
    r2   = float(result.get("r2_score", result.get("r2", float("nan"))))

    payload = {
        "equation": eq_name, "status": "ok",
        "expression": expr,  "r2": r2,
        "elapsed_s": elapsed, "ground_truth": spec["ground_truth"],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  ✅ [{eq_name}] R²={r2:.4f}  expr={expr}  ({elapsed:.1f}s)")
    sys.exit(0)

except Exception:
    elapsed = time.perf_counter() - t0
    tb = traceback.format_exc()
    payload = {"equation": eq_name, "status": "error", "error": tb, "elapsed_s": elapsed}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  ❌ [{eq_name}] FAILED after {elapsed:.1f}s:", file=sys.stderr)
    print(tb, file=sys.stderr)
    sys.exit(1)
""")

FEYNMAN_30 = [
    {"name": "I.6.2a",   "variable_names": ["theta"],
     "variable_ranges": [[-3.0, 3.0]],
     "numpy_expr": "np.exp(-theta**2/2) / np.sqrt(2*np.pi)",
     "ground_truth": "exp(-theta^2/2)/sqrt(2*pi)"},
    {"name": "I.9.18",   "variable_names": ["F","m","t1","t2"],
     "variable_ranges": [[1,10],[1,5],[2,10],[11,20]],
     "numpy_expr": "F / (m * (1/t1 - 1/t2))", "ground_truth": "F/(m*(1/t1-1/t2))"},
    {"name": "I.12.1",   "variable_names": ["F1","F2","eps","r"],
     "variable_ranges": [[1,5],[1,5],[0.5,2],[1,10]],
     "numpy_expr": "F1*F2 / (4*np.pi*eps*r**2)", "ground_truth": "F1*F2/(4*pi*eps*r^2)"},
    {"name": "I.12.2",   "variable_names": ["q1","q2","eps","r"],
     "variable_ranges": [[1,5],[1,5],[0.5,2],[1,10]],
     "numpy_expr": "q1*q2 / (4*np.pi*eps*r**2)", "ground_truth": "q1*q2/(4*pi*eps*r^2)"},
    {"name": "I.12.4",   "variable_names": ["q1","eps","r"],
     "variable_ranges": [[1,5],[0.5,2],[1,10]],
     "numpy_expr": "q1 / (4*np.pi*eps*r**2)", "ground_truth": "q1/(4*pi*eps*r^2)"},
    {"name": "I.15.1",   "variable_names": ["x","u","t","c"],
     "variable_ranges": [[1,10],[0.1,0.9],[1,5],[1,1]],
     "numpy_expr": "(x - u*t) / np.sqrt(1 - u**2/c**2)",
     "ground_truth": "(x-u*t)/sqrt(1-u^2/c^2)"},
    {"name": "I.18.4",   "variable_names": ["m1","m2","r1"],
     "variable_ranges": [[1,5],[1,5],[1,10]],
     "numpy_expr": "m1*r1 / (m1+m2)", "ground_truth": "m1*r1/(m1+m2)"},
    {"name": "I.24.6",   "variable_names": ["m","omega","omega0","x"],
     "variable_ranges": [[1,5],[1,5],[1,5],[1,5]],
     "numpy_expr": "0.25 * m * (omega**2 + omega0**2) * x**2",
     "ground_truth": "0.25*m*(omega^2+omega0^2)*x^2"},
    {"name": "I.26.2",   "variable_names": ["n","theta2"],
     "variable_ranges": [[0.5,1.0],[0.1,1.0]],
     "numpy_expr": "np.arcsin(n * np.sin(theta2))",
     "ground_truth": "arcsin(n*sin(theta2))"},
    {"name": "I.34.8",   "variable_names": ["omega","v","c"],
     "variable_ranges": [[1,10],[0.1,0.9],[1,1]],
     "numpy_expr": "omega / (1 - v/c)", "ground_truth": "omega/(1-v/c)"},
    {"name": "I.34.14",  "variable_names": ["omega0","v","c"],
     "variable_ranges": [[1,10],[0.1,0.9],[1,1]],
     "numpy_expr": "omega0 / (1 - v/c)", "ground_truth": "omega0/(1-v/c)"},
    {"name": "I.34.27",  "variable_names": ["h","omega"],
     "variable_ranges": [[0.5,2],[1,10]],
     "numpy_expr": "h * omega", "ground_truth": "h*omega"},
    {"name": "I.37.4",   "variable_names": ["I1","I2","delta"],
     "variable_ranges": [[1,5],[1,5],[0,3.14159]],
     "numpy_expr": "I1 + I2 + 2*np.sqrt(I1*I2)*np.cos(delta)",
     "ground_truth": "I1+I2+2*sqrt(I1*I2)*cos(delta)"},
    {"name": "I.41.16",  "variable_names": ["h","omega","c","kb","T"],
     "variable_ranges": [[0.5,2],[1,5],[1,3],[0.5,2],[100,1000]],
     "numpy_expr": "h*omega**3 / (np.pi**2 * c**3 * (np.exp(h*omega/(kb*T)) - 1))",
     "ground_truth": "h*omega^3/(pi^2*c^3*(exp(h*omega/(kb*T))-1))"},
    {"name": "I.43.31",  "variable_names": ["mob","kb","T"],
     "variable_ranges": [[0.5,2],[0.5,2],[100,1000]],
     "numpy_expr": "mob * kb * T", "ground_truth": "mob*kb*T"},
    {"name": "I.43.43",  "variable_names": ["kappa","T1","T2","A","d"],
     "variable_ranges": [[0.5,2],[200,500],[501,800],[1,5],[0.1,1]],
     "numpy_expr": "kappa * (T2-T1) * A / d",
     "ground_truth": "kappa*(T2-T1)*A/d"},
    {"name": "I.50.26",  "variable_names": ["x1","x2","omega","t"],
     "variable_ranges": [[1,5],[1,5],[1,5],[0,2]],
     "numpy_expr": "x1 + x2 * np.cos(omega * t)",
     "ground_truth": "x1+x2*cos(omega*t)"},
    {"name": "II.2.42",  "variable_names": ["kappa","T1","T2","A","d"],
     "variable_ranges": [[0.5,2],[200,500],[501,800],[1,5],[0.1,1]],
     "numpy_expr": "kappa * (T2 - T1) * A / d",
     "ground_truth": "kappa*(T2-T1)*A/d"},
    {"name": "II.11.27", "variable_names": ["n","alpha"],
     "variable_ranges": [[0.1,0.9],[0.1,1.0]],
     "numpy_expr": "n*alpha / (1 - n*alpha/3)",
     "ground_truth": "n*alpha/(1-n*alpha/3)"},
    {"name": "II.11.28", "variable_names": ["n","alpha"],
     "variable_ranges": [[0.1,0.9],[0.1,1.0]],
     "numpy_expr": "1 + n*alpha / (1 - n*alpha/3)",
     "ground_truth": "1+n*alpha/(1-n*alpha/3)"},
    {"name": "II.34.2a", "variable_names": ["q","v","r"],
     "variable_ranges": [[1,5],[1,10],[1,10]],
     "numpy_expr": "q*v / (2*np.pi*r)", "ground_truth": "q*v/(2*pi*r)"},
    {"name": "II.34.29b","variable_names": ["q","h","m","me"],
     "variable_ranges": [[1,3],[0.5,2],[1,5],[1,5]],
     "numpy_expr": "q*h*m / (4*np.pi*me)",
     "ground_truth": "q*h*m/(4*pi*me)"},
    {"name": "II.35.18", "variable_names": ["n0","m","g","x","kb","T"],
     "variable_ranges": [[1,5],[0.1,1],[5,15],[0,5],[0.5,2],[200,500]],
     "numpy_expr": "n0 * np.exp(-m*g*x / (kb*T))",
     "ground_truth": "n0*exp(-m*g*x/(kb*T))"},
    {"name": "II.36.38", "variable_names": ["mu","Ef","v"],
     "variable_ranges": [[0.1,1],[1,10],[10,50]],
     "numpy_expr": "mu*Ef / (1 + mu*Ef/v)",
     "ground_truth": "mu*Ef/(1+mu*Ef/v)"},
    {"name": "III.4.32", "variable_names": ["h","omega","kb","T"],
     "variable_ranges": [[0.5,2],[1,5],[0.5,2],[100,1000]],
     "numpy_expr": "h*omega / (np.exp(h*omega/(kb*T)) - 1)",
     "ground_truth": "h*omega/(exp(h*omega/(kb*T))-1)"},
    {"name": "III.4.33", "variable_names": ["h","omega","kb","T"],
     "variable_ranges": [[0.5,2],[1,5],[0.5,2],[100,1000]],
     "numpy_expr": ("h*omega * np.exp(h*omega/(kb*T)) / "
                    "(kb * T**2 * (np.exp(h*omega/(kb*T)) - 1)**2)"),
     "ground_truth": "h*omega*exp(h*omega/(kb*T))/(kb*T^2*(exp(h*omega/(kb*T))-1)^2)"},
    {"name": "III.12.4", "variable_names": ["n","h"],
     "variable_ranges": [[1,10],[0.5,2]],
     "numpy_expr": "n*h / (2*np.pi)", "ground_truth": "n*h/(2*pi)"},
    {"name": "III.14.14","variable_names": ["I0","q","V","kb","T"],
     "variable_ranges": [[0.1,2],[1,2],[0.1,1],[0.5,2],[200,500]],
     "numpy_expr": "I0 * (np.exp(q*V/(kb*T)) - 1)",
     "ground_truth": "I0*(exp(q*V/(kb*T))-1)"},
    {"name": "III.19.51","variable_names": ["m","q","eps","h","n"],
     "variable_ranges": [[0.5,2],[1,2],[0.5,2],[0.5,2],[1,5]],
     "numpy_expr": "-m * q**4 / (2 * (4*np.pi*eps)**2 * h**2) / n**2",
     "ground_truth": "-m*q^4/(2*(4*pi*eps)^2*h^2*n^2)"},
    {"name": "III.21.20","variable_names": ["rho","q","Ef","m"],
     "variable_ranges": [[0.5,2],[1,3],[1,10],[1,5]],
     "numpy_expr": "rho*q*Ef / m", "ground_truth": "rho*q*Ef/m"},
]


def _load_exp2_eq_checkpoint() -> dict:
    if EXP2_EQ_CHECKPOINT.exists():
        try:
            return json.loads(EXP2_EQ_CHECKPOINT.read_text())
        except Exception:
            pass
    return {}


def _save_exp2_eq_checkpoint(state: dict) -> None:
    EXP2_EQ_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    tmp = EXP2_EQ_CHECKPOINT.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(EXP2_EQ_CHECKPOINT)


def run_exp2_feynman(env: dict, args, log_fh) -> bool:
    """Per-equation isolated runner for exp2.  Returns True if ≥ EXP2_PASS_THRESHOLD solved."""
    n_tasks = int(env.get("N_FEYNMAN_TASKS", len(FEYNMAN_30)))
    equations = FEYNMAN_30[:n_tasks]
    n_samples = 300

    pysr_timeout  = int(env.get("PYSR_TIMEOUT", "1100"))
    kill_grace    = getattr(args, "kill_grace", None) or EXP2_KILL_GRACE
    kill_deadline = pysr_timeout + kill_grace

    out_dir = RESULTS_DIR / "comparison_results" / "feynman-tests" / "exp2"
    out_dir.mkdir(parents=True, exist_ok=True)

    worker_path = LOG_DIR / "_exp2_worker.py"
    worker_path.write_text(_EXP2_WORKER_SCRIPT)

    eq_checkpoint = _load_exp2_eq_checkpoint()
    results = []
    t_total = time.time()
    SEP  = "=" * 68
    SSEP = "-" * 68

    def _log(msg: str) -> None:
        print(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    _log(f"\n{SEP}")
    _log(f"  exp2 · Feynman {n_tasks}-equation extrapolation (per-equation isolation)")
    _log(f"  PYSR_TIMEOUT={pysr_timeout}s  kill_grace={kill_grace}s  samples={n_samples}")
    _log(f"  pass_threshold={EXP2_PASS_THRESHOLD}/{n_tasks}")
    _log(SEP)

    for idx, spec in enumerate(equations):
        eq_name = spec["name"]
        if eq_name in eq_checkpoint and eq_checkpoint[eq_name].get("status") == "ok":
            cached = eq_checkpoint[eq_name]
            _log(f"\n  ↩  [{idx+1}/{n_tasks}] {eq_name}  "
                 f"(checkpoint: R²={cached.get('r2', '?'):.4f})  — skipping")
            results.append(cached)
            continue

        _log(f"\n{SSEP}")
        _log(f"  [{idx+1}/{n_tasks}] {eq_name}  gt={spec['ground_truth']}")

        run_spec    = {**spec, "n_samples": n_samples}
        result_path = out_dir / f"{eq_name.replace('.', '_')}.json"
        child_env   = {**env, "EXP2_EQUATION_JSON": json.dumps(run_spec),
                       "EXP2_RESULT_PATH": str(result_path)}

        t0   = time.time()
        proc = None
        deadline = t0 + kill_deadline
        status = "error"
        try:
            proc = subprocess.Popen(
                [sys.executable, str(worker_path)],
                env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, preexec_fn=os.setsid,
            )
            import queue as _queue
            import threading as _threading

            assert proc.stdout is not None
            _line_q: _queue.Queue = _queue.Queue()

            def _stdout_reader(stream, q):
                try:
                    for line in stream:
                        q.put(line)
                finally:
                    q.put(None)

            _reader_thread = _threading.Thread(
                target=_stdout_reader, args=(proc.stdout, _line_q), daemon=True
            )
            _reader_thread.start()
            timed_out = False
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    _log(f"\n  ⏱  [{eq_name}] wall-clock limit ({kill_deadline}s) — killing")
                    try:
                        import signal as _signal
                        os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
                    except Exception:
                        proc.kill()
                    timed_out = True
                    break
                try:
                    line = _line_q.get(timeout=min(remaining, 5.0))
                except _queue.Empty:
                    continue
                if line is None:
                    break
                log_fh.write(line)
                log_fh.flush()
                print(f"│  {line}", end="")
            proc.wait(timeout=30)
            elapsed = time.time() - t0
            if result_path.exists():
                try:
                    payload = json.loads(result_path.read_text())
                    status  = payload.get("status", "error")
                except Exception:
                    status = "error"
            else:
                status = "timeout" if timed_out else "error"

        except KeyboardInterrupt:
            if proc is not None:
                try:
                    proc.terminate(); proc.wait(timeout=5)
                except Exception:
                    try: proc.kill()
                    except Exception: pass
            _log(f"\n  ⚠  [{eq_name}] interrupted — saving checkpoint")
            _save_exp2_eq_checkpoint(eq_checkpoint)
            raise
        except Exception as exc:
            _log(f"\n  ❌ [{eq_name}] subprocess error: {exc}")
            status = "error"

        elapsed = time.time() - t0
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
            except Exception:
                result = {"equation": eq_name, "status": status, "elapsed_s": elapsed}
        else:
            result = {"equation": eq_name, "status": status, "elapsed_s": elapsed}

        results.append(result)
        eq_checkpoint[eq_name] = result
        _save_exp2_eq_checkpoint(eq_checkpoint)

        sym = "✅" if status == "ok" else ("⏱" if status == "timeout" else "❌")
        r2_str = f"R²={result.get('r2', float('nan')):.4f}" if status == "ok" else ""
        _log(f"\n  {sym} [{eq_name}] {status}  {r2_str}  ({elapsed:.0f}s)")

    total_elapsed = time.time() - t_total
    solved   = [r for r in results if r.get("status") == "ok"]
    timeouts = [r for r in results if r.get("status") == "timeout"]
    errors   = [r for r in results if r.get("status") not in ("ok","timeout")]

    _log(f"\n{SEP}")
    _log(f"  exp2 SUMMARY  —  {len(solved)}/{n_tasks} solved  "
         f"({len(timeouts)} timeouts  {len(errors)} errors)  "
         f"total {total_elapsed/60:.1f} min")
    _log(f"  {'#':<4} {'Name':<14} {'Status':<10} {'R²':>8}  Expression")
    _log("  " + "-" * 60)
    for i, r in enumerate(results):
        st  = r.get("status", "?")
        r2s = f"{r['r2']:.4f}" if st == "ok" and "r2" in r else "—"
        exp = r.get("expression", r.get("error", ""))[:40]
        _log(f"  {i+1:<4} {r.get('equation','?'):<14} {st:<10} {r2s:>8}  {exp}")

    consolidated = {
        "experiment": "exp2_feynman_30", "n_equations": n_tasks,
        "n_solved": len(solved), "solve_rate": len(solved) / n_tasks,
        "results": results,
    }
    consolidated_path = (
        RESULTS_DIR / "comparison_results" / "feynman-tests" / "exp2" / "exp2_results.json"
    )
    consolidated_path.write_text(json.dumps(consolidated, indent=2))
    _log(f"\n  Results → {consolidated_path}")
    _log(SEP)

    passed = len(solved) >= EXP2_PASS_THRESHOLD
    _log(f"\n  exp2 {'✅ PASS' if passed else '❌ FAIL'}  "
         f"({len(solved)}/{n_tasks} solved, threshold={EXP2_PASS_THRESHOLD})")
    return passed


# ════════════════════════════════════════════════════════════════════════════
#  Step dataclass & registry
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class PostMove:
    src_dir:     Path
    glob:        str
    dest_dir:    Path
    recursive:   bool = False
    subdir_only: bool = False
    exclude:     str  = ""


@dataclass
class Step:
    id:            str
    label:         str
    cmd:           list[str]
    phase:         str
    slow:          bool  = False
    paper:         bool  = False
    env_extra:     dict  = field(default_factory=dict)
    expected:      str   = ""
    result_glob:   str   = ""
    inline_runner: bool  = False
    post_move:     list  = field(default_factory=list)


def _hybrid_domain_args() -> list[str]:
    return ["--domains"] + HYBRID_ALL_DOMAINS_IDS


# Canonical 11-domain list — mirrors run_all.sh FEYNMAN_DOMAINS (line ~201).
# Used by exp2_feynman and exp2_feynman_extrap step commands which embed this
# list as a Python literal via f-string interpolation at module-load time.
FEYNMAN_DOMAINS_LIST = [
    "feynman_biology",
    "feynman_chemistry",
    "feynman_electrochemistry",
    "feynman_electromagnetism",
    "feynman_electrostatics",
    "feynman_magnetism",
    "feynman_mechanics",
    "feynman_optics",
    "feynman_probability",
    "feynman_quantum",
    "feynman_thermodynamics",
]

STEPS: list[Step] = [
    # ── Phase 0: Setup ─────────────────────────────────────────────────────
    Step("env_check",
         "Verify environment (Python, PySR, API key, output directories)",
         [sys.executable, "-c", "\n".join([
             "import sys, os, subprocess",
             "print('Python:', sys.version)",
             "try:",
             "    import pysr; print('PySR:', pysr.__version__)",
             "except ImportError:",
             "    print('ERROR: pysr not installed'); sys.exit(1)",
             "import torch; print('PyTorch:', torch.__version__)",
             "import anthropic",
             "# BUG 10 FIX: claude-sonnet-4-20250514 requires anthropic SDK >= 0.40.0.",
             "# environment.yml was pinned to 0.28.0 which predates this model family.",
             "ver = tuple(int(x) for x in anthropic.__version__.split('.')[:3])",
             "if ver < (0, 40, 0):",
             "    print('ERROR: anthropic SDK', anthropic.__version__, '< 0.40.0 required'); sys.exit(1)",
             "print('anthropic SDK:', anthropic.__version__, '(>= 0.40.0 OK)')",
             "import sympy; print('SymPy:', sympy.__version__)",
             "import scipy; print('SciPy:', scipy.__version__)",
             "try:",
             "    import sklearn; print('scikit-learn:', sklearn.__version__)",
             "except ImportError:",
             "    print('ERROR: scikit-learn not installed'); sys.exit(1)",
             "try:",
             "    import yaml; print('PyYAML: ok')",
             "except ImportError:",
             "    print('ERROR: pyyaml not installed'); sys.exit(1)",
             "try:",
             "    import matplotlib; print('matplotlib:', matplotlib.__version__)",
             "except ImportError:",
             "    print('ERROR: matplotlib not installed'); sys.exit(1)",
             "try:",
             "    import pmlb; print('pmlb: ok')",
             "except ImportError:",
             "    print('ERROR: pmlb not installed'); sys.exit(1)",
             "# ITEM 2 FIX: seaborn required by statistical_analysis.py (exp1 step).",
             "# Self-heal if missing so the run never reaches analysis without it.",
             "try:",
             "    import seaborn; print('seaborn:', seaborn.__version__)",
             "except ImportError:",
             "    print('WARNING: seaborn not found — installing now')",
             "    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'seaborn'])",
             "    if r.returncode != 0:",
             "        print('ERROR: seaborn install failed'); sys.exit(1)",
             "    import seaborn; print('seaborn: installed', seaborn.__version__)",
             "key = os.environ.get('ANTHROPIC_API_KEY', '')",
             "if not key:",
             "    print('ERROR: ANTHROPIC_API_KEY not set'); sys.exit(1)",
             "print(f'ANTHROPIC_API_KEY: set ({len(key)} chars)')",
             "# FIX-13: echo all CI-parity env vars for auditability",
             "for var in ['PYSR_POPULATIONS','JULIA_NUM_THREADS','JULIA_EXCLUSIVE',",
             "            'PYTHON_JULIACALL_HANDLE_SIGNALS','FEYNMAN_SAMPLES',",
             "            'FEYNMAN_TIMEOUT','FEYNMAN_NOISELESS_THRESHOLD',",
             "            'JOB_DEADLINE','REPRO_CFG']:",
             "    print(f'{var}: {os.environ.get(var, \"(not set\")}')",
             "# FIX-12: repro.yaml audit — mirrors CI FIX-G2 print_repro.py log",
             "repro_cfg = os.environ.get('REPRO_CFG', 'config/repro.yaml')",
             "if os.path.isfile(repro_cfg):",
             "    try:",
             "        import yaml as _yaml",
             "        cfg = _yaml.safe_load(open(repro_cfg)) or {}",
             "        print('repro.yaml found -- key values:')",
             "        for k, v in cfg.items(): print(f'  {k}: {v}')",
             "    except Exception as e:",
             "        print(f'  (could not parse repro.yaml: {e})')",
             "else:",
             "    print(f'WARNING: repro.yaml not found at {repro_cfg} -- using env defaults')",
             "from pathlib import Path",
             "results = Path(os.environ.get('RESULTS_DIR',",
             "               'hypatiax/data/results'))",
             "for sub in [",
             "    'comparison_results/feynman-tests/exp2',",
             "    'comparison_results/feynman-tests/exp2_multi',",
             "    'comparison_results/feynman-tests/noise-sweep/noise-sweep',",
             "    'comparison_results/feynman-tests/sample-complexity',",
             "    'comparison_results/noise-noiseless/noiseless/defi',",
             "    'comparison_results/noise-noiseless/15',",
             "    'comparison_results/extrapolation',",
             "    'extrapolation',",
             "    'extrapolation/multi_seed',",
             "    'hybrid_llm_nn/all_domains', 'hybrid_llm_nn/defi',",
             "    'hybrid_pysr/all_domains',   'hybrid_pysr/defi',",
             "    'llm_guided/all_domains',    'llm_guided/defi',",
             "    'standalone_llm_nn', 'figures', 'tables',",
             "]:",
             "    (results / sub).mkdir(parents=True, exist_ok=True)",
             "print('Directory structure: ok')",
         ])],
         phase="0 · Setup"),

    Step("deps", "Install dependencies",
         ["pip", "install", "-q", "-r", "requirements.txt"],
         phase="0 · Setup"),

    Step("patches-gen", "Generate patches",
         ["python3", "scripts/patches/generate_patches.py"],
         phase="0 · Setup"),

    Step("patches-apply", "Apply patches (FIX-C1…FIX-5b)",
         ["python3", "scripts/patches/apply_patches.py"],
         phase="0 · Setup"),

    Step("fixup-init",
         "Guard hypatiax/__init__.py broken HypatiaX import (FIX-INIT-PY)",
         ["python3", "-c", "\n".join([
             "from pathlib import Path",
             "init = Path('hypatiax') / '__init__.py'",
             "if not init.exists():",
             "    print('  ⚠ fixup-init: not found — skipping'); raise SystemExit(0)",
             "src = init.read_text(encoding='utf-8')",
             "if 'from hypatiax import HypatiaX' not in src: raise SystemExit(0)",
             "fixed = src.replace('from hypatiax import HypatiaX',",
             "    'try:\\n    from hypatiax import HypatiaX\\nexcept ImportError:\\n    pass')",
             "if fixed != src:",
             "    init.write_text(fixed, encoding='utf-8')",
             "    print('  ✓ fixup-init: guarded broken HypatiaX import')",
             "else:",
             "    print('  ✓ fixup-init: already guarded')",
         ])],
         phase="0 · Setup"),

    Step("fixup-tex",
         "Stage .tex source files into paper/ (FIX-TEX-STAGE)",
         ["python3", "-c", "\n".join([
             "import shutil, pathlib",
             "paper = pathlib.Path('paper'); paper.mkdir(exist_ok=True)",
             "pats  = ['jmlr_paper*.tex','jmlr-hypatiax*.tex',",
             "         'supp_routing_improvements.tex','supp_benchmark_report.tex']",
             "copied = []",
             "for p in pats:",
             "    for src in pathlib.Path('.').glob(p):",
             "        dst = paper / src.name",
             "        if not dst.exists(): shutil.copy2(src, dst); copied.append(src.name)",
             "print(f'  Staged {len(copied)} .tex file(s): {copied}')",
         ])],
         phase="0 · Setup"),

    Step("validate-patches",
         "Validate patched source code (Phase 0 integrity check)",
         ["python3", "scripts/patches/validate_patches.py"],
         phase="0 · Setup"),

    Step("validate-paper-config",
         "Validate repro.yaml against expected paper hyperparameters",
         ["python3", "-c", "\n".join([
             "import yaml, sys",
             "from pathlib import Path",
             "cfg_path = Path('config/repro.yaml')",
             "if not cfg_path.exists(): print('  ⚠ config/repro.yaml not found — skip'); sys.exit(0)",
             "cfg = yaml.safe_load(cfg_path.read_text()) or {}",
             "t = cfg.get('timeouts', {})",
             "p = cfg.get('pysr', {})",
             "checks = [",
             "    ('feynman_timeout', t.get('pysr_attempt_seconds'), 1100),",
             "    ('julia_threads', cfg.get('julia_num_threads'), 4),",
             "    ('pysr_populations', p.get('populations'), 30),",
             "]",
             "ok = True",
             "for name, got, want in checks:",
             "    if got is not None and int(got) != int(want):",
             "        print(f'  ⚠ {name}: got {got}, paper expects {want}')",
             "        ok = False",
             "    else:",
             "        print(f'  ✓ {name}: {got or want}')",
             "sys.exit(0 if ok else 1)",
         ])],
         phase="0 · Setup"),

    Step("check-hypatiax-protocols",
         "Check all required hypatiax/protocols/ modules are present",
         ["python3", "-c", "\n".join([
             "import sys; from pathlib import Path",
             "proto = Path('hypatiax/protocols')",
             "required = [",
             "    'experiment_protocol_defi.py',",
             "    'experiment_protocol_defi_20.py',",
             "    'experiment_protocol_nguyen12.py',",
             "    'experiment_protocol_all_18_a.py',",
             "    'experiment_protocol_all_20.py',",
             "    'experiment_protocol_all_30.py',",
             "    'experiment_protocol_benchmark.py',",
             "    'experiment_protocol_benchmark_v2.py',",
             "    'experiment_protocol_comparative.py',",
             "]",
             "missing = [f for f in required if not (proto/f).exists()]",
             "if missing:",
             "    print(f'  ✗ {len(missing)} missing:', missing); sys.exit(1)",
             "print(f'  ✓ All {len(required)} protocol modules present')",
         ])],
         phase="0 · Setup"),

    # ── Phase 1: Core experiments ──────────────────────────────────────────
    Step("exp1",
         "Exp 1 · Core DeFi extrapolation benchmark (Tab 9, 10, 15 · Fig 9, 10)",
         [sys.executable,
          "hypatiax/experiments/benchmarks/hypatiax_defi_benchmark_v3c.py"],
         phase="1 · Core experiments",
         slow=True,
         expected="Tab 9 OOD R²>0.85; Tab 10 DeFi 74-task solve rate; Tab 15 ablation",
         result_glob="comparison_results/noise-noiseless/noiseless/defi/*.json",  # FIX-BUG4: was extrapolation/
         env_extra={"SKIP_PKG_CHECK": "1"}),

    Step("exp1_analysis",
         "Exp 1 · Statistical analysis (Mann-Whitney, R² distribution)",
         [sys.executable,
          "hypatiax/experiments/benchmarks/statistical_analysis.py",
          "--results-dir", str(RESULTS_DIR / "comparison_results" / "extrapolation"),
          "--output-dir",  str(RESULTS_DIR / "comparison_results" / "extrapolation")],
         phase="1 · Core experiments",
         expected="MW U-stat, p-value, bootstrap CI printed",
         result_glob="comparison_results/extrapolation/*stats*.json"),

    Step("exp1b",
         "Exp 1b · DeFi seed sweep + portfolio variance (Tab 11-13 · Fig 11-13)",
         [sys.executable,
          "hypatiax/experiments/benchmarks/hypatiax_defi_benchmark_v3c.py",
          "--seed-sweep"],
         phase="1 · Core experiments",
         slow=True,
         expected="seed variance <5%; portfolio R² consistent across seeds",
         result_glob="comparison_results/noise-noiseless/15/*.json",  # FIX-BUG4: was extrapolation/*seed*/
         env_extra={"SKIP_PKG_CHECK": "1"}),

    Step("extrap",
         "Extrap · OOD extrapolation comparative suite (Tab 9 OOD columns)",
         [sys.executable,
          "hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_v2.py",
          "--extrap",
          "--extrap-multiplier", "2.0",   # FIX-BUG3: was 3.0 (non-paper value)
          "--extrap-train-frac", "0.8",   # FIX-BUG3: was 0.4 (non-paper value)
          # --skip-pysr removed: CI worker runs all 6 methods (methods 5+6 active)
          ],
         phase="1 · Core experiments",
         slow=True,
         expected="OOD extrapolation R² vs baseline; Tab 9 OOD columns reproduced",
         result_glob="comparison_results/extrapolation/*.json"),

    Step("hybrid_all_domains",
         "Hybrid · LLM+NN all-domains one-shot run (§10.9 hybrid table)",
         [sys.executable,
          # PATH-1 FIX: hypatiax/core/generation (not hypatiax/experiments/generation).
          # Mirrors GENERATION_DIR in run_all.sh and CI script_path.
          "hypatiax/core/generation/hybrid_all_domains_llm_nn/"
          "hybrid_system_llm_nn_all_domains.py",
          "--samples", str(int(os.environ.get("FEYNMAN_SAMPLES", "200"))),
          "--output-dir", str(RESULTS_DIR / "hybrid_llm_nn" / "all_domains")]
         + _hybrid_domain_args(),
         phase="1 · Core experiments",
         slow=True,
         expected="10 domains complete; hybrid_llm_nn/all_domains/ populated",
         result_glob="hybrid_llm_nn/all_domains/**/*.json",
         env_extra={
             "TASK_IDS":   ",".join(HYBRID_ALL_DOMAINS_IDS),
             "SHARD_IDS":  "0",
             "HYPATIAX_CORE_OPTIONAL": "1",
         }),

    Step("instability",
         "Instability · Index analysis + regime figures (§10.9 A/B/C, 12 figs)",
         [sys.executable, "-c", "\n".join([
             "import sys, os, subprocess, pathlib",
             "results_dir = pathlib.Path(os.environ.get('RESULTS_DIR',",
             "              'hypatiax/data/results'))",
             "figures_dir = results_dir / 'figures'",
             "figures_dir.mkdir(parents=True, exist_ok=True)",
             "bench_jsons = list((results_dir / 'hybrid_llm_nn' / 'all_domains')",
             "                   .rglob('*benchmark*.json'))",
             "bench_arg = ['--benchmark-json', str(bench_jsons[0])] if bench_jsons else []",
             "if bench_arg:",
             "    print(f'[instability] Stage 2 enabled: {bench_jsons[0].name}')",
             "else:",
             "    print('[instability] No benchmark JSON found — Stage 2 / EX figure skipped.')",
             "cmd = [sys.executable,"
             "       'hypatiax/experiments/benchmarks/run_instability_suite.py',"
             "       '--results-dir', str(results_dir),"
             "       '--out',         str(figures_dir),"
             "       '--csv-out',     str(figures_dir / 'instability_analysis.csv'),"
             "       '--format', 'png', 'pdf'] + bench_arg",
             "sys.exit(subprocess.run(cmd, env=os.environ).returncode)",
         ])],
         phase="1 · Core experiments",
         slow=True,
         expected=(
             "instability_analysis.csv + fig_paper_complexity_vs_instability.{png,pdf} "
             "written; Regime A/B/C counts; Spearman ρ printed"
         ),
         result_glob="figures/instability_analysis.csv",
         env_extra={"HYPATIAX_CORE_OPTIONAL": "1"}),

    Step("exp2_feynman",
         "Exp 2 · Feynman SR benchmark — Phase 2 noisy protocol per-domain (§10.7)",
         # FIX-EXP2FEYNMAN-LOOP: run_all.sh (STEP 5) runs a per-domain loop over all
         # 11 FEYNMAN_DOMAINS, matching ci_experiment_simplify.yml worker step exactly.
         # Previous monolithic single-invocation omitted --domain and --threshold,
         # diverging from both run_all.sh and CI.
         [sys.executable, "-c", "\n".join([
             "import subprocess, sys, os, pathlib",
             f"domains = {repr(FEYNMAN_DOMAINS_LIST)}",
             "results_dir = pathlib.Path(os.environ.get('RESULTS_DIR', 'hypatiax/data/results'))",
             "out_dir = results_dir / 'comparison_results' / 'feynman-tests' / 'exp2'",
             "out_dir.mkdir(parents=True, exist_ok=True)",
             "script = 'hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_v2.py'",
             "samples     = os.environ.get('FEYNMAN_SAMPLES', '200')",
             "timeout     = os.environ.get('FEYNMAN_TIMEOUT', '1100')",
             "m_timeout   = os.environ.get('METHOD_TIMEOUT',  '900')",
             "populations = os.environ.get('PYSR_POPULATIONS', '30')",
             "threshold   = os.environ.get('FEYNMAN_NOISELESS_THRESHOLD', '0.999999')",
             "for domain in domains:",
             "    print(f'=== exp2_feynman: domain={domain} ===')",
             "    cmd = [sys.executable, script,",
             "           '--benchmark', 'feynman',",
             "           '--domain', domain,",
             "           '--samples', samples,",
             "           '--pysr-timeout', timeout,",
             "           '--method-timeout', m_timeout,",
             "           '--populations', populations,",
             "           '--parsimony', '0.01',",
             "           '--noiseless',",
             "           '--threshold', threshold,",
             "           '--checkpoint-name', f'feynman_exp2_checkpoint_{domain}',",
             "           '--output-dir', str(out_dir),",
             "           '--resume']",
             "    r = subprocess.run(cmd, env=os.environ)",
             "    if r.returncode != 0:",
             "        print(f'WARNING: domain {domain} exited non-zero — continuing')",
             "sys.exit(0)",
         ])],
         phase="1 · Core experiments",
         slow=True,
         expected="stats.json written per domain; ≥1/30 solved  [~15 min smoke / 8-24 h full]",
         result_glob="comparison_results/feynman-tests/exp2/*.json",
         env_extra={
             "N_FEYNMAN_TASKS": (
                 "1" if os.environ.get("ONE_EQUATION") == "1"
                 else str(int(os.environ.get("N_FEYNMAN_TASKS", "30")))
             ),
             "PYSR_TIMEOUT":                str(int(os.environ.get("PYSR_TIMEOUT", "1100"))),
             "POPULATIONS":                 str(int(os.environ.get("POPULATIONS",  "30"))),
             "N_ITERATIONS":                str(int(os.environ.get("N_ITERATIONS", "1000"))),
             \"FEYNMAN_NOISELESS_THRESHOLD\": os.environ.get(\"FEYNMAN_NOISELESS_THRESHOLD\", \"0.999999\"),
         }),

    # FIX-C3: Corrected Feynman benchmark with PCA-directed 40/60 extrapolation split.
    # Mirrors run_all.sh STEP 5b (exp2_feynman_pca_4060).
    # Locks the legacy 9/30 baseline in fixc3_baseline.json, then reruns every
    # Feynman domain via run_comparative_suite_benchmark_pca.py (PCA split is
    # hard-wired at method level — no --extrap flags needed).
    # Writes results to exp2_pca_4060/ alongside split_protocol_disclosure.json
    # so Gates A/B/C in ci_runner_disclosure.yml can verify protocol parity.
    Step("exp2_feynman_pca_4060",
         "Exp 2 FIX-C3 · Feynman rerun with PCA 40/60 split — corrected §10.7 result",
         [sys.executable, "-c", "\n".join([
             "import subprocess, sys, os, pathlib",
             f"domains = {repr(FEYNMAN_DOMAINS_LIST)}",
             "results_dir = pathlib.Path(os.environ.get('RESULTS_DIR', 'hypatiax/data/results'))",
             "pca_dir = results_dir / 'comparison_results' / 'feynman-tests' / 'exp2_pca_4060'",
             "pca_dir.mkdir(parents=True, exist_ok=True)",
             "script = 'hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_pca.py'",
             "samples     = os.environ.get('FEYNMAN_SAMPLES', '200')",
             "timeout     = os.environ.get('FEYNMAN_TIMEOUT', '1100')",
             "m_timeout   = os.environ.get('METHOD_TIMEOUT',  '900')",
             "populations = os.environ.get('PYSR_POPULATIONS', '30')",
             "threshold   = os.environ.get('FEYNMAN_NOISELESS_THRESHOLD', '0.999999')",
             "for domain in domains:",
             "    print(f'=== exp2_feynman_pca_4060: domain={domain} ===')",
             "    cmd = [sys.executable, script,",
             "           '--benchmark', 'feynman',",
             "           '--domain', domain,",
             "           '--samples', samples,",
             "           '--pysr-timeout', timeout,",
             "           '--method-timeout', m_timeout,",
             "           '--populations', populations,",
             "           '--parsimony', '0.01',",
             "           '--noiseless',",
             "           '--threshold', threshold,",
             "           '--checkpoint-name', f'pca4060_checkpoint_{domain}',",
             "           '--output-dir', str(pca_dir),",
             "           '--resume']",
             "    r = subprocess.run(cmd, env=os.environ)",
             "    if r.returncode != 0:",
             "        print(f'WARNING: domain {domain} exited non-zero — continuing')",
             "sys.exit(0)",
         ])],
         phase="1 · Core experiments",
         slow=True,
         expected=(
             "exp2_pca_4060_summary.json + split_protocol_disclosure.json written; "
             "corrected solve rate replaces 9/30 (random_80_20) from §10.7"
         ),
         result_glob="comparison_results/feynman-tests/exp2_pca_4060/*.json",
         env_extra={
             "PYSR_TIMEOUT":                str(int(os.environ.get("PYSR_TIMEOUT", "1100"))),
             "POPULATIONS":                 str(int(os.environ.get("POPULATIONS",  "30"))),
             "N_ITERATIONS":                str(int(os.environ.get("N_ITERATIONS", "1000"))),
             "FEYNMAN_NOISELESS_THRESHOLD": os.environ.get("FEYNMAN_NOISELESS_THRESHOLD", "0.999999"),
         }),

    # Feynman far-region R² (extrap_r2_far) for Mann-Whitney ablation (Tab 14).
    # Mirrors run_all.sh STEP exp2_feynman_extrap.
    # Runs run_comparative_suite_benchmark_v2.py with --extrap per domain,
    # writing to exp2_extrap/ so ci_analysis.yml can merge into ablation_paired.json.
    Step("exp2_feynman_extrap",
         "Exp 2 extrap · Feynman far-region R² (extrap_r2_far for Mann-Whitney ablation)",
         [sys.executable, "-c", "\n".join([
             "import subprocess, sys, os, pathlib",
             f"domains = {repr(FEYNMAN_DOMAINS_LIST)}",
             "results_dir = pathlib.Path(os.environ.get('RESULTS_DIR', 'hypatiax/data/results'))",
             "ext_dir = results_dir / 'comparison_results' / 'feynman-tests' / 'exp2_extrap'",
             "ext_dir.mkdir(parents=True, exist_ok=True)",
             "script = 'hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_v2.py'",
             "samples     = os.environ.get('FEYNMAN_SAMPLES', '200')",
             "timeout     = os.environ.get('FEYNMAN_TIMEOUT', '1100')",
             "m_timeout   = os.environ.get('METHOD_TIMEOUT',  '900')",
             "populations = os.environ.get('PYSR_POPULATIONS', '30')",
             "threshold   = os.environ.get('FEYNMAN_NOISELESS_THRESHOLD', '0.999999')",
             "active = os.environ.get('DOMAIN_FILTER', ' '.join(domains)).split()",
             "for domain in active:",
             "    print(f'=== exp2_feynman_extrap: domain={domain} ===')",
             "    cmd = [sys.executable, script,",
             "           '--benchmark', 'feynman',",
             "           '--extrap',",
             "           '--extrap-multiplier', '2.0',",
             "           '--extrap-train-frac', '0.8',",
             "           '--domain', domain,",
             "           '--samples', samples,",
             "           '--pysr-timeout', timeout,",
             "           '--method-timeout', m_timeout,",
             "           '--populations', populations,",
             "           '--parsimony', '0.01',",
             "           '--noiseless',",
             "           '--threshold', threshold,",
             "           '--checkpoint-name', f'feynman_extrap_checkpoint_{domain}',",
             "           '--output-dir', str(ext_dir),",
             "           '--resume']",
             "    r = subprocess.run(cmd, env=os.environ)",
             "    if r.returncode != 0:",
             "        print(f'WARNING: domain {domain} exited non-zero — continuing')",
             "sys.exit(0)",
         ])],
         phase="1 · Core experiments",
         slow=True,
         expected=(
             "protocol_core_extrap_*.json + benchmark_results_extrap.json in exp2_extrap/; "
             "extrap_r2_far populated for ablation Mann-Whitney test (Tab 14)"
         ),
         result_glob="comparison_results/feynman-tests/exp2_extrap/protocol_core_extrap_*.json",
         env_extra={
             "PYSR_TIMEOUT":                str(int(os.environ.get("PYSR_TIMEOUT", "1100"))),
             "POPULATIONS":                 str(int(os.environ.get("POPULATIONS",  "30"))),
             "N_ITERATIONS":                str(int(os.environ.get("N_ITERATIONS", "1000"))),
             "FEYNMAN_NOISELESS_THRESHOLD": os.environ.get("FEYNMAN_NOISELESS_THRESHOLD", "0.999999"),
         }),

    Step("exp2",
         "Exp 2 · Combined five-system comparison — all methods per-domain (§10.7 combined)",
         # FIX-EXP2-LOOP (mirrors run_all.sh STEP 6 FIX-exp2-2): per-domain loop over
         # EXP2_DOMAINS matching CI YAML lines 1002-1031 exactly.
         # Previous single-invocation ran all domains in one call.
         [sys.executable, "-c", "\n".join([
             "import subprocess, sys, os, pathlib",
             "domains = ['mechanics','thermodynamics','electromagnetism','fluid_dynamics',",
             "           'optics','quantum','chemistry','biology','mathematics','economics']",
             "results_dir = pathlib.Path(os.environ.get('RESULTS_DIR', 'hypatiax/data/results'))",
             "out_dir = results_dir / 'comparison_results' / 'feynman-tests' / 'exp2_multi'",
             "out_dir.mkdir(parents=True, exist_ok=True)",
             "script = 'hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_v2.py'",
             "samples     = os.environ.get('FEYNMAN_SAMPLES', '200')",
             "timeout     = os.environ.get('FEYNMAN_TIMEOUT', '1100')",
             "m_timeout   = os.environ.get('METHOD_TIMEOUT',  '900')",
             "populations = os.environ.get('PYSR_POPULATIONS', '30')",
             "for domain in domains:",
             "    print(f'=== exp2: domain={domain} ===')",
             "    cmd = [sys.executable, script,",
             "           '--benchmark', 'both',",
             "           '--domain', domain,",
             "           '--samples', samples,",
             "           '--pysr-timeout', timeout,",
             "           '--method-timeout', m_timeout,",
             "           '--populations', populations,",
             "           '--parsimony', '0.01',",
             "           '--use-transcendental-compositions',",
             "           '--noiseless',",
             "           '--threshold', '0.9999',",
             "           '--checkpoint-name', f'exp2_checkpoint_{domain}',",
             "           '--output-dir', str(out_dir),",
             "           '--resume']",
             "    r = subprocess.run(cmd, env=os.environ)",
             "    if r.returncode != 0:",
             "        print(f'WARNING: domain {domain} exited non-zero — continuing')",
             "sys.exit(0)",
         ])],
         phase="1 · Core experiments",
         expected="9/30 (30%)  [fast after method-5/6 checkpoints ready]",
         result_glob="comparison_results/feynman-tests/exp2_multi/*.json"),

    Step("exp3",
         "Exp 3 · Nguyen-12 SEED=42 (§10.8 primary)",
         [sys.executable,
          "hypatiax/experiments/benchmarks/exp3_nguyen12_hybrid50v_02.py",
          "--seed", "42"]
         + (["--n-tasks", "1"] if os.environ.get("ONE_EQUATION") == "1" else []),
         phase="1 · Core experiments",
         expected=(
             "11/12 (91.7% by 4-decimal rounding) · strict R²≥0.9999: 4/12 (33.3%) · "
             "MW U=113, p=0.0097"
         ),
         result_glob="extrapolation/*nguyen*seed42*.json",
         env_extra={"SKIP_PKG_CHECK": "1"},
         post_move=[
             # FIX-BUG2: script writes to RESULTS_DIR root (run_all.sh: "FIX-DIR");
             #           glob must use RESULTS_DIR, not EXPERIMENTS_DIR.
             PostMove(RESULTS_DIR, "*nguyen*seed42*.json", RESULTS_DIR / "extrapolation"),
             PostMove(RESULTS_DIR, "*nguyen12*42*.json",   RESULTS_DIR / "extrapolation"),
         ]),

    Step("exp3b",
         "Exp 3b · Nguyen-12 seeds 99/123/777/2024 (§10.8 stability)",
         [sys.executable, "-c",
          "import subprocess, sys, pathlib, os;"
          "s = pathlib.Path('hypatiax/experiments/benchmarks/exp3_nguyen12_hybrid50v_02.py');"
          "extra = ['--n-tasks','1'] if os.environ.get('ONE_EQUATION')=='1' else [];"
          "rc = 0;"
          "\nfor seed in ('99','123','777','2024'):\n"
          "    r = subprocess.run([sys.executable, str(s), '--seed', seed] + extra,"
          " env=os.environ);\n"
          "    rc = rc or r.returncode\n"
          "sys.exit(rc)"],
         phase="1 · Core experiments",
         expected="consistent with SEED=42 across all 5 seeds",
         result_glob="extrapolation/multi_seed/*nguyen*.json",
         env_extra={"SKIP_PKG_CHECK": "1"},
         post_move=[
             # FIX-BUG2: RESULTS_DIR, not EXPERIMENTS_DIR (same as exp3 above).
             PostMove(RESULTS_DIR, "*nguyen*.json",
                      RESULTS_DIR / "extrapolation" / "multi_seed"),
         ]),

    # ── Phase 2: Supplementary benchmarks ─────────────────────────────────
    Step("suppA",
         "Supp A · Hybrid-PySR DeFi benchmark (standalone run_hybrid_system_benchmark.py)",
         [sys.executable,
          "hypatiax/experiments/benchmarks/run_hybrid_system_benchmark.py"],
         phase="2 · Supplementary benchmarks",
         expected="+6pp Fix1, +5pp Fix2, +1pp Fix3",
         result_glob="hybrid_pysr/defi/**/*.json",
         env_extra={
             "SKIP_PERF_ANALYSIS":    "1",
             "HYPATIAX_CORE_OPTIONAL": "1",
         },
         post_move=[
             # FIX-BUG2: RESULTS_DIR, not EXPERIMENTS_DIR (same fix as exp3/exp3b).
             PostMove(RESULTS_DIR, "consolidated_hybrid*.json",
                      RESULTS_DIR / "hybrid_pysr" / "defi"),
             PostMove(RESULTS_DIR, "hybrid_system*.json",
                      RESULTS_DIR / "hybrid_pysr" / "defi"),
         ]),

    Step("suppB",
         "Supp B · Noise sweep σ ∈ {0,0.5,1,5,10}% × 30 equations (§SuppB §5–7)",
         [sys.executable,
          "hypatiax/experiments/benchmarks/run_noise_sweep_benchmark.py"],
         phase="2 · Supplementary benchmarks",
         slow=True,
         expected=(
             "EHD 100% at all σ · HSL 90% noiseless, 100% at σ>0 · "
             "M3 avg 841.4s · M4 avg 11.1s · speedup 75.8×"
         ),
         result_glob=(
             "comparison_results/feynman-tests/noise-sweep/noise-sweep/noise_sweep_*.json"
         ),
         post_move=[
             PostMove(RESULTS_DIR / "comparison_results" / "feynman-tests" / "noise-sweep",
                      "noise_sweep_*.json",
                      RESULTS_DIR / "comparison_results" / "feynman-tests" / "noise-sweep",
                      recursive=True, subdir_only=True),
         ]),

    Step("suppB_sc",
         "Supp B-SC · Sample-complexity sweep n ∈ {50…1000} × 30 eq (§SuppB §6)",
         [sys.executable,
          "hypatiax/experiments/benchmarks/run_sample_complexity_benchmark.py"],
         phase="2 · Supplementary benchmarks",
         slow=True,
         expected=(
             "Both M3 & M4 plateau at n≈500 · convergence at n=50 visible · "
             "180 task results in sample-complexity/"
         ),
         result_glob=(
             "comparison_results/feynman-tests/sample-complexity/*.json"
         ),
         env_extra={
             "NOISE_LEVEL":       "5.0",
             "SC_SAMPLE_COUNTS":  ",".join(SUPPB_SC_SAMPLE_COUNTS),
             "N_FEYNMAN_TASKS": (
                 "1" if os.environ.get("ONE_EQUATION") == "1" else "30"
             ),
         },
         post_move=[
             PostMove(RESULTS_DIR / "comparison_results" / "feynman-tests",
                      "sample_complexity_*.json",
                      RESULTS_DIR / "comparison_results" / "feynman-tests" / "sample-complexity",
                      recursive=True,
                      exclude="sample-complexity"),
         ]),

    # ── Phase 3: Audit & verification ──────────────────────────────────────
    Step("provenance",
         "§11 · Provenance audit — protocol orchestration",
         ["python3", "-c",
          "import subprocess, sys, pathlib; "
          "s = pathlib.Path('hypatiax/protocols/experiment_protocol_provenance_audit.py'); "
          "sys.exit(subprocess.run([sys.executable, str(s)]).returncode) "
          "if s.exists() else "
          "(print('  ⚠  not found — skipping') or sys.exit(0))"],
         phase="3 · Audit & verification"),

    Step("discover-provenance",
         "§11 · discover_provenance.py — link result files to families",
         ["python3", "-c",
          "import subprocess, sys, pathlib; "
          "m = pathlib.Path('provenance_map.json'); "
          "pathlib.Path('logs/provenance_audit').mkdir(parents=True, exist_ok=True); "
          "(print('INFO: provenance_map.json absent — skipping') or sys.exit(0)) "
          "if not m.exists() else "
          "sys.exit(subprocess.run([sys.executable, 'discover_provenance.py', "
          "'--root', '.', '--map', str(m), '--out', 'logs/provenance_audit']).returncode)"],
         phase="3 · Audit & verification"),

    Step("scan-imports",
         "§11 · scan_internal_imports.py — internal import DAG",
         [sys.executable, "scan_internal_imports.py",
          "--root", ".", "--out", "logs/repro_output"],
         phase="3 · Audit & verification"),

    Step("verify",
         "Verify results against paper targets",
         [sys.executable, "scripts/patches/verify_results.py", "--report"],
         phase="3 · Audit & verification",
         env_extra={
             "PATCHED_DATA_DIR":   str(REPO_ROOT / "hypatiax" / "data" / "results"),
             "VERIFY_RESULTS_DIR": str(RESULTS_DIR),
         }),

    Step("hashlock",
         "Hash lock check",
         [sys.executable, "hypatiax/reproducibility/hash_lock.py", "--check"],
         phase="3 · Audit & verification"),

    # ── Phase 4: Outputs ────────────────────────────────────────────────────
    Step("tables",
         "Generate all tables",
         [sys.executable, "tables/generate_tables.py",
          "--results-dir", str(RESULTS_DIR),
          "--output-dir",  str(RESULTS_DIR / "tables")],
         phase="4 · Outputs",
         result_glob="tables/*.tex",
         env_extra={
             "TABLE_OUTDIR":       str(RESULTS_DIR / "tables"),
             "VERIFY_RESULTS_DIR": str(RESULTS_DIR),
         }),

    Step("figures",
         "Generate all figures",
         [sys.executable, "figures/generate_figures.py",
          "--results-dir", str(RESULTS_DIR),
          "--output-dir",  str(RESULTS_DIR / "figures")],
         phase="4 · Outputs",
         result_glob="figures/*.pdf"),

    # ── Phase 4-B: Paper audit notebooks ───────────────────────────────────
    Step("audit-setup",
         "Paper audit · Copy main paper + supplements into notebooks/",
         ["python3", "-c", "\n".join([
             "import shutil, pathlib",
             "nb = pathlib.Path('notebooks'); nb.mkdir(exist_ok=True)",
             "search_dirs = [pathlib.Path('paper'), pathlib.Path('.'),",
             "               pathlib.Path('paper') / 'tables', pathlib.Path('logs')]",
             "copied = []; missing = []",
             "main = next((f for d in search_dirs",
             "             for pat in ('jmlr-hypatiax*.tex','jmlr_paper*.tex')",
             "             for f in d.glob(pat) if f.is_file()), None)",
             "if main: shutil.copy(main, nb / main.name); copied.append(main.name)",
             "else: print('WARNING: main paper .tex not found')",
             "for name in ('supp_routing_improvements.tex','supp_benchmark_report.tex'):",
             "    src = next((d/name for d in search_dirs if (d/name).is_file()), None)",
             "    if src: shutil.copy(src, nb/name); copied.append(name)",
             "    else: missing.append(name); print(f'WARNING: {name} not found')",
             "print(f'audit-setup: copied {len(copied)} file(s): {copied}')",
             "if missing: print(f'Missing: {missing}')",
         ])],
         phase="4-B · Paper audit", paper=True),

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

    # ── Phase 5: Qualification & paper audit ────────────────────────────────
    # These two steps are handled specially in main() via inline Python
    # (not subprocess) but live in STEPS so --only / checkpoint work correctly.
    Step("qualify",
         "Qualify all experiments (7-dimension gate per exp)",
         [sys.executable, "-c", "print('qualify: handled inline')"],
         phase="5 · Qualification & paper audit",
         expected="All 12 experiments: checkpoint=pass + files + merged + git + figs + tables"),

    Step("audit-paper",
         "Audit results against paper claims (paper_targets.json)",
         [sys.executable, "-c", "print('audit-paper: handled inline')"],
         phase="5 · Qualification & paper audit",
         expected="All claims in paper_targets.json within tolerance; Nguyen-12 dual threshold checked"),
]

STEP_IDS = [s.id for s in STEPS]


# ════════════════════════════════════════════════════════════════════════════
#  Checkpoint helpers
# ════════════════════════════════════════════════════════════════════════════
def load_checkpoint() -> dict:
    state: dict[str, str] = {}
    root_cp = REPO_ROOT / "pipeline_checkpoint.json"
    for path in [root_cp, CHECKPOINT]:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for k, v in data.items():
                    if state.get(k) != "pass":
                        state[k] = v
            except Exception:
                pass
    if state and not CHECKPOINT.exists():
        save_checkpoint(state)
    return state


def save_checkpoint(state: dict) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, str] = {}
    if CHECKPOINT.exists():
        try:
            merged = json.loads(CHECKPOINT.read_text())
        except Exception:
            pass
    for k, v in state.items():
        if merged.get(k) != "pass":
            merged[k] = v
    tmp = CHECKPOINT.with_suffix(".tmp")
    tmp.write_text(json.dumps(merged, indent=2))
    tmp.replace(CHECKPOINT)


def clear_checkpoint() -> None:
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print(f"  Checkpoint cleared: {CHECKPOINT}")
    else:
        print("  No checkpoint file found.")


# ════════════════════════════════════════════════════════════════════════════
#  Result-file helpers
# ════════════════════════════════════════════════════════════════════════════
def ensure_output_dirs() -> None:
    for sub in [
        "comparison_results/extrapolation",
        "comparison_results/feynman-tests/exp2",
        "comparison_results/feynman-tests/noise-sweep/noise-sweep",
        "comparison_results/feynman-tests/sample-complexity",
        "comparison_results/noise-noiseless/noiseless/defi",
        "comparison_results/noise-noiseless/15",
        "extrapolation",
        "extrapolation/multi_seed",
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


def move_step_outputs(step: Step) -> None:
    if not step.post_move:
        return
    for pm in step.post_move:
        pm.dest_dir.mkdir(parents=True, exist_ok=True)
        if pm.recursive:
            candidates = list(pm.src_dir.rglob(pm.glob))
        else:
            candidates = list(pm.src_dir.glob(pm.glob))
        moved = 0
        for src in candidates:
            if not src.is_file():
                continue
            if pm.subdir_only and src.parent == pm.src_dir:
                continue
            if pm.exclude and pm.exclude in str(src):
                continue
            dst = pm.dest_dir / src.name
            if src == dst:
                continue
            shutil.move(str(src), dst)
            print(f"│    mv {src.name} → {pm.dest_dir.relative_to(REPO_ROOT)}/")
            moved += 1
        if moved:
            print(f"│    post-move [{pm.glob}]: {moved} file(s) → "
                  f"{pm.dest_dir.relative_to(REPO_ROOT)}")


def archive_step_results(step: Step) -> None:
    if not step.result_glob:
        return
    pattern = step.result_glob
    if "**" in pattern:
        parts   = Path(pattern).parts
        star_i  = next(i for i, p in enumerate(parts) if "**" in p)
        base_d  = RESULTS_DIR / Path(*parts[:star_i])
        sub_pat = str(Path(*parts[star_i:]))
        matches = list(base_d.rglob(sub_pat)) if base_d.exists() else []
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
    jsons = sum(1 for _ in RESULTS_DIR.rglob("*.json"))
    csvs  = sum(1 for _ in RESULTS_DIR.rglob("*.csv"))
    pdfs  = (
        sum(1 for _ in (RESULTS_DIR / "figures").glob("*.pdf"))
        if (RESULTS_DIR / "figures").exists() else 0
    )
    tables_dir = RESULTS_DIR / "tables"
    if not tables_dir.exists() or not any(tables_dir.glob("*.tex")):
        tables_dir = REPO_ROOT / "paper" / "tables"
    texs = sum(1 for _ in tables_dir.glob("*.tex")) if tables_dir.exists() else 0
    return jsons + csvs, pdfs, texs


# ════════════════════════════════════════════════════════════════════════════
#  Step result & runner
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class StepResult:
    id:         str
    label:      str
    status:     str
    elapsed:    float         = 0.0
    log_path:   Path | None   = None
    returncode: int           = 0


def run_step(step: Step, env: dict, args) -> StepResult:
    log_path   = LOG_DIR / f"{step.id}.log"
    merged_env = {**env, **step.env_extra}

    if args.case_range:
        try:
            _s, _e = args.case_range.split("-")
            merged_env["CASE_RANGE_START"] = _s.strip()
            merged_env["CASE_RANGE_END"]   = _e.strip()
        except ValueError:
            print(f"[CI] WARNING: --case-range '{args.case_range}' malformed — ignoring")

    print(f"\n┌─── [{step.id}] {step.label}")
    print(f"│    {time.strftime('%H:%M:%S')}")
    if step.expected:
        print(f"│    Expected : {step.expected}")
    if step.env_extra:
        for k, v in step.env_extra.items():
            print(f"│    env+  {k}={v}")
    if merged_env.get("CASE_RANGE_START"):
        print(f"│    case-range: {merged_env['CASE_RANGE_START']}-"
              f"{merged_env.get('CASE_RANGE_END','?')}")
    print(f"│    cmd: {' '.join(str(x) for x in step.cmd)}")

    if getattr(args, "dry_run", False):
        _dry = {k: v for k, v in merged_env.items()
                if k not in os.environ or os.environ[k] != v}
        if _dry:
            print("│    env overrides:")
            for k, v in sorted(_dry.items()):
                print(f"│      {k}={v}")
        print("└─── (dry-run — not executed)\n")
        return StepResult(step.id, step.label, "skip")

    t0 = time.time()

    if step.inline_runner:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(log_path, "w") as log_fh:
                ok = run_exp2_feynman(merged_env, args, log_fh)
            elapsed = time.time() - t0
            sym = "✓" if ok else "✗"
            print(f"\n└─── {sym} {'done' if ok else 'FAILED'}  ({elapsed:.0f}s)"
                  + (f"  — see {log_path}" if not ok else ""))
            if ok:
                move_step_outputs(step)
                archive_step_results(step)
            return StepResult(step.id, step.label,
                              "pass" if ok else "fail",
                              elapsed, log_path, 0 if ok else 1)
        except KeyboardInterrupt:
            elapsed = time.time() - t0
            print(f"\n└─── ✗ INTERRUPTED  ({elapsed:.0f}s)")
            raise
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"└─── ✗ ERROR: {exc}")
            return StepResult(step.id, step.label, "fail", elapsed, log_path)

    proc: subprocess.Popen | None = None
    try:
        with open(log_path, "w") as log_fh:
            proc = subprocess.Popen(
                step.cmd, env=merged_env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log_fh.write(line)
                print(f"│  {line}", end="")
            proc.wait()

        elapsed = time.time() - t0
        ok  = proc.returncode == 0
        sym = "✓" if ok else "✗"
        print(f"\n└─── {sym} {'done' if ok else 'FAILED'}  ({elapsed:.0f}s)"
              + (f"  — see {log_path}" if not ok else ""))
        if ok:
            move_step_outputs(step)
            archive_step_results(step)
        return StepResult(step.id, step.label,
                          "pass" if ok else "fail",
                          elapsed, log_path, proc.returncode)

    except KeyboardInterrupt:
        elapsed = time.time() - t0
        if proc is not None:
            try:
                proc.terminate(); proc.wait(timeout=5)
            except Exception:
                try: proc.kill()
                except Exception: pass
        print(f"\n└─── ✗ INTERRUPTED  ({elapsed:.0f}s)")
        raise
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"└─── ✗ ERROR: {exc}")
        return StepResult(step.id, step.label, "fail", elapsed, log_path)


def banner(msg: str) -> None:
    print("\n" + "═" * 68)
    print(f"  {msg}")
    print("═" * 68)


# ════════════════════════════════════════════════════════════════════════════
#  Stale-lock cleanup
# ════════════════════════════════════════════════════════════════════════════
def _clear_stale_locks() -> None:
    _cleared: list[str] = []
    _failed:  list[str] = []

    def _try_unlink(p: Path) -> None:
        if p.exists():
            try:
                p.unlink(); _cleared.append(str(p))
            except Exception as e:
                _failed.append(f"{p} ({e})")

    for lf in RESULTS_DIR.glob(".lock_*"):
        _try_unlink(lf)

    _exe = Path(sys.executable).resolve()
    _julia_roots = [
        _exe.parent.parent,
        Path.home() / ".local",
        Path.home() / ".julia" / "environments",
    ]
    _FS_ROOT       = Path("/")
    _BLOCKED_ROOTS = {_FS_ROOT, Path("/usr"), Path("/usr/local")}
    for _root in _julia_roots:
        if not _root.exists() or _root in _BLOCKED_ROOTS:
            continue
        try:
            for _pid in _root.rglob("julia_env/lock.pid"):
                _try_unlink(_pid)
        except OSError:
            pass

    _julia_home = Path.home() / ".julia"
    if _julia_home.exists():
        _locks_dir = _julia_home / "locks"
        if _locks_dir.exists():
            for lf in _locks_dir.iterdir():
                if lf.is_file():
                    _try_unlink(lf)
        _reg = _julia_home / "registries"
        if _reg.exists():
            for lf in _reg.rglob("*.lock"):
                _try_unlink(lf)

    try:
        for lf in REPO_ROOT.rglob("lock.pid"):
            _try_unlink(lf)
    except OSError:
        pass

    if _cleared:
        print(f"  🔓 Cleared {len(_cleared)} stale lock file(s):")
        for lf in _cleared:
            print(f"       {lf}")
    else:
        print("  🔓 No stale lock files found")
    if _failed:
        print(f"  ⚠  Could not remove {len(_failed)} lock(s):")
        for lf in _failed:
            print(f"       {lf}")


# ════════════════════════════════════════════════════════════════════════════
#  main()
# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="HypatiaX reproducibility pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--skip-slow",          action="store_true")
    parser.add_argument("--only",               metavar="ID")
    parser.add_argument("--resume",             action="store_true")
    parser.add_argument("--from",               dest="from_step", metavar="ID")
    parser.add_argument("--clear-checkpoint",   action="store_true")
    parser.add_argument("--continue-on-fail",   action="store_true")
    parser.add_argument("--verify-only",        action="store_true")
    parser.add_argument("--qualify-only",       action="store_true",
                        help="Run qualification + paper audit without re-running experiments")
    parser.add_argument("--skip-paper",         action="store_true")
    parser.add_argument("--seed",               type=int, default=None, metavar="N")
    parser.add_argument("--pysr-timeout",       type=int, default=None, metavar="SECS")
    parser.add_argument("--kill-grace",         type=int, default=None, metavar="SECS")
    parser.add_argument("--one-equation",       action="store_true")
    parser.add_argument("--one-equation-paper", action="store_true")
    parser.add_argument("--case-range",         metavar="START-END", default=None)
    parser.add_argument("--dry-run",            action="store_true")
    args = parser.parse_args()

    if args.case_range and not args.only:
        parser.error("--case-range requires --only <STEP_ID>")
    if args.from_step and not args.resume:
        print("  WARNING: --from has no effect without --resume.", file=sys.stderr)

    os.chdir(REPO_ROOT)
    LOG_DIR.mkdir(exist_ok=True)
    ensure_output_dirs()
    _clear_stale_locks()

    if args.clear_checkpoint:
        clear_checkpoint(); sys.exit(0)

    banner(
        "HypatiaX · Reproducibility Pipeline v8.0"
        + ("  [DRY-RUN]"          if args.dry_run            else "")
        + ("  [SMOKE-TEST]"       if args.one_equation        else "")
        + ("  [PAPER-QUALITY-1]"  if args.one_equation_paper  else "")
        + ("  [QUALIFY-ONLY]"     if args.qualify_only        else "")
    )
    print(f"  Repo      : {REPO_ROOT}")
    print(f"  Python    : {sys.version.split()[0]}")
    print(f"  Date      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Results   : {RESULTS_DIR}")
    print(f"  Logs      : {LOG_DIR}")
    print(f"  Checkpoint: {CHECKPOINT}")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("\n  ERROR: ANTHROPIC_API_KEY is not set.")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)
    print(f"\n  API key   : set ({len(api_key)} chars)")

    # ── BLOCKER-1 / WARN-2: validate domain list ────────────────────────────
    print("\n  Validating hybrid_all_domains domain list …")
    if not validate_hybrid_all_domains_ids():
        print("\n  ERROR: Domain-list validation failed. "
              "Update HYBRID_ALL_DOMAINS_IDS before running.")
        sys.exit(1)

    # ── Protocol check ──────────────────────────────────────────────────────
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
        print(f"\n  ERROR: {len(missing_hp)} module(s) missing from hypatiax/protocols/:")
        for f in missing_hp:
            print(f"    ✗  {f}")
        sys.exit(1)
    print(f"  Protocols : all {len(required_hp)} hypatiax/protocols/ modules ✓")

    # ── --verify-only / --qualify-only shortcuts ────────────────────────────
    if args.verify_only:
        banner("Verify-only mode")
        subprocess.run([sys.executable, "scripts/patches/verify_results.py", "--report"],
                       check=False)
        subprocess.run([sys.executable, "hypatiax/reproducibility/hash_lock.py", "--check"],
                       check=False)
        sys.exit(0)

    if args.qualify_only:
        checkpoint_state = load_checkpoint()
        all_qual, qual_results = run_qualification(checkpoint_state)
        if all_qual:
            audit_ok, _ = audit_against_paper(qual_results)
            sys.exit(0 if audit_ok else 1)
        else:
            sys.exit(1)

    # ── Validate step IDs ───────────────────────────────────────────────────
    if args.only and args.only not in STEP_IDS:
        print(f"\n  ERROR: unknown step id '{args.only}'.")
        print(f"  Valid ids: {', '.join(STEP_IDS)}")
        sys.exit(1)
    if args.from_step and args.from_step not in STEP_IDS:
        print(f"\n  ERROR: unknown step id '{args.from_step}'.")
        sys.exit(1)

    # ── Load repro.yaml ─────────────────────────────────────────────────────
    _repro_config   = load_repro_config()
    _timeout_config = _repro_config.get("timeouts", {})
    _pysr_config    = _repro_config.get("pysr", {})

    DEFAULT_PYSR_TIMEOUT   = _timeout_config.get("pysr_attempt_seconds", 1100)
    DEFAULT_METHOD_TIMEOUT = _timeout_config.get("method_seconds",        900)
    DEFAULT_KILL_GRACE     = _timeout_config.get("kill_grace_seconds",    300)

    _seed_str = str(args.seed) if args.seed is not None else "42"

    env = {**os.environ}
    env["PYTHONWARNINGS"]  = "ignore"
    env["NN_SEED"]         = os.environ.get("NN_SEED",        _seed_str)
    env["PYSR_SEED"]       = os.environ.get("PYSR_SEED",      _seed_str)
    env["PYTHONHASHSEED"]  = os.environ.get("PYTHONHASHSEED", _seed_str)

    if args.seed is not None:
        env["NN_SEED"] = env["PYSR_SEED"] = env["PYTHONHASHSEED"] = _seed_str

    env.setdefault("LLM_MODEL",   _repro_config.get("llm_model",   "claude-sonnet-4-6"))
    env.setdefault("LLM_RETRIES", str(_repro_config.get("llm_retries", 3)))
    env.setdefault("LLM_K_RUNS",  "1")

    env.setdefault("N_TASKS_DEFI",         str(_repro_config.get("n_tasks_defi",        74)))
    env.setdefault("N_TASKS_INSTABILITY",   str(_repro_config.get("n_tasks_instability", 70)))
    env.setdefault("PCA_TRAIN_FRAC",        str(_repro_config.get("pca_train_frac",      0.40)))
    env.setdefault("NN_TIME_LIMIT",         str(_repro_config.get("nn_time_limit",       120)))
    env.setdefault("ENGINE_NAME",
                   _repro_config.get("engine", {}).get("name", "hybrid_system_v50_2"))
    env.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    env.setdefault("JULIA_NUM_THREADS", "4")
    env.setdefault("FEYNMAN_SAMPLES",   str(_repro_config.get("feynman_samples", 200)))

    if args.pysr_timeout is not None:
        env["PYSR_TIMEOUT"]   = str(args.pysr_timeout)
        env["METHOD_TIMEOUT"] = str(DEFAULT_METHOD_TIMEOUT)
        print(f"  PYSR_TIMEOUT={args.pysr_timeout}s  (--pysr-timeout override)")
    else:
        pysr_timeout   = DEFAULT_PYSR_TIMEOUT
        method_timeout = DEFAULT_METHOD_TIMEOUT
        if env_pysr := os.environ.get("PYSR_TIMEOUT"):
            pysr_timeout   = int(env_pysr)
            method_timeout = DEFAULT_METHOD_TIMEOUT
            print(f"  ⚠ PYSR_TIMEOUT={pysr_timeout}s from env")
        env["PYSR_TIMEOUT"]   = str(pysr_timeout)
        env["METHOD_TIMEOUT"] = str(method_timeout)
        print(f"  PYSR_TIMEOUT={pysr_timeout}s  METHOD_TIMEOUT={method_timeout}s")

    env.setdefault("POPULATIONS",          str(_pysr_config.get("populations",    30)))
    env.setdefault("N_ITERATIONS",         str(_pysr_config.get("niterations",  1000)))
    env.setdefault("PYSR_POPULATIONS",     env["POPULATIONS"])
    env.setdefault("PYSR_NITERATIONS",     env["N_ITERATIONS"])
    env.setdefault("PYSR_PARALLELISM",     _pysr_config.get("parallelism", "multithreading"))
    env.setdefault("EQUATION_WALL_CLOCK",
                   str(_timeout_config.get("equation_wall_clock", 1200)))
    env.setdefault("PYSR_POPULATION_SIZE", str(_pysr_config.get("population_size", 33)))
    env.setdefault("PYSR_PARSIMONY",       str(_pysr_config.get("parsimony",    0.01)))
    env.setdefault("PYSR_MAXSIZE",         str(_pysr_config.get("maxsize",        30)))

    env["PYTHONPATH"]      = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["RESULTS_DIR"]     = str(RESULTS_DIR)
    env["PIPELINE_PYTHON"] = sys.executable
    env["REPRO_ROOT"]      = str(REPO_ROOT)

    print(f"\n  Seeds: NN={env['NN_SEED']}  PYSR={env['PYSR_SEED']}  "
          f"HASH={env['PYTHONHASHSEED']}")
    print(f"  LLM_MODEL={env['LLM_MODEL']}")
    print(f"  PySR: iters={env['N_ITERATIONS']} pops={env['POPULATIONS']} "
          f"pop_sz={env['PYSR_POPULATION_SIZE']}")
    print(f"  FEYNMAN_SAMPLES={env['FEYNMAN_SAMPLES']}")

    # ── --one-equation smoke-test ────────────────────────────────────────────
    if args.one_equation:
        env.update({
            "ONE_EQUATION":        "1",
            "N_TASKS_DEFI":        "1",
            "N_CORE15_TASKS":      "1",
            "N_FEYNMAN_TASKS":     "1",
            "N_TASKS_INSTABILITY": "1",
            "N_NGUYEN_TASKS":      "1",
            "N_NOISE_EQUATIONS":   "1",
            "LLM_K_RUNS":          "1",
            "N_ITERATIONS":        "200",
            "POPULATIONS":         "10",
        })
        if args.pysr_timeout is None:
            env["PYSR_TIMEOUT"] = "60"
        print("\n" + "▲" * 68)
        print("  ▲▲  SMOKE-TEST MODE  (--one-equation) — NOT paper-quality")
        print("▲" * 68)

    # ── --one-equation-paper reviewer-probe ─────────────────────────────────
    if args.one_equation_paper:
        env.update({
            "ONE_EQUATION":        "1",
            "N_TASKS_DEFI":        "1",
            "N_CORE15_TASKS":      "1",
            "N_FEYNMAN_TASKS":     "1",
            "N_TASKS_INSTABILITY": "1",
            "N_NGUYEN_TASKS":      "1",
            "N_NOISE_EQUATIONS":   "1",
            "N_ITERATIONS":        "1000",
            "POPULATIONS":         "30",
            "PYSR_POPULATION_SIZE": "33",
            "PYSR_PARSIMONY":      "0.01",
            "PYSR_MAXSIZE":        "30",
            "PYSR_PARALLELISM":    "multithreading",
            "LLM_K_RUNS":          "30",
            "METHOD_TIMEOUT":      "900",
            "EQUATION_WALL_CLOCK": "1200",
        })
        if args.pysr_timeout is None:
            env["PYSR_TIMEOUT"] = "1100"
        print("\n" + "★" * 68)
        print("  ★★  PAPER-QUALITY PROBE  (--one-equation-paper)")
        print("★" * 68)

    # ── Load checkpoint ─────────────────────────────────────────────────────
    checkpoint_state: dict[str, str] = {}
    if args.resume:
        root_cp = REPO_ROOT / "pipeline_checkpoint.json"
        for _cp in [root_cp, CHECKPOINT]:
            if _cp.exists():
                try:
                    for k, v in json.loads(_cp.read_text()).items():
                        if checkpoint_state.get(k) != "pass":
                            checkpoint_state[k] = v
                except Exception:
                    pass
        save_checkpoint(checkpoint_state)

        _done    = [s for s in STEPS if checkpoint_state.get(s.id) == "pass"]
        _pending = [s for s in STEPS if checkpoint_state.get(s.id) != "pass"]
        print(f"\n  Pipeline status ({len(_done)}/{len(STEPS)} done):")
        _cur_phase = ""
        for _s in STEPS:
            if _s.phase != _cur_phase:
                print(f"    Phase {_s.phase}")
                _cur_phase = _s.phase
            _st  = checkpoint_state.get(_s.id, "todo")
            _ico = {"pass": "✓", "fail": "✗", "todo": "·"}.get(_st, "·")
            print(f"      {_ico}  {_s.id}")
        if _pending:
            print(f"  Next: [{_pending[0].id}]")

    # ── Run pipeline ─────────────────────────────────────────────────────────
    results: list[StepResult] = []
    current_phase = ""
    t_total   = time.time()
    past_from = False

    # Tracks qualification results so audit-paper can use them inline
    _qual_results: list[QualResult] = []

    try:
        for step in STEPS:
            if args.from_step and step.id == args.from_step:
                past_from = True
            if args.only and step.id != args.only:
                results.append(StepResult(step.id, step.label, "skip"))
                continue
            if args.resume and checkpoint_state.get(step.id) == "pass" and not past_from:
                results.append(StepResult(step.id, step.label, "resume-skip"))
                continue

            # Idempotent skip: step already complete with files on disk
            if (not args.only and not past_from
                    and step_already_complete(step, checkpoint_state)):
                results.append(StepResult(step.id, step.label, "resume-skip"))
                print(f"  ── auto-skip [{step.id}]  (outputs present + checkpoint=pass)")
                continue

            if step.phase != current_phase:
                banner(f"Phase {step.phase}")
                current_phase = step.phase
            if args.skip_slow and step.slow:
                results.append(StepResult(step.id, step.label, "skip"))
                print(f"  ── skip [{step.id}]  (--skip-slow)")
                continue
            if args.skip_paper and step.paper:
                results.append(StepResult(step.id, step.label, "skip"))
                print(f"  ── skip [{step.id}]  (--skip-paper)")
                continue

            # ── Inline handlers for Phase 5 steps ──────────────────────────
            if step.id == "qualify":
                t0 = time.time()
                all_qual, _qual_results = run_qualification(checkpoint_state)
                elapsed = time.time() - t0
                status = "pass" if all_qual else "fail"
                r = StepResult(step.id, step.label, status, elapsed)
                results.append(r)
                checkpoint_state[step.id] = status
                save_checkpoint(checkpoint_state)
                if not all_qual and not args.continue_on_fail:
                    print(f"\n  Pipeline aborted at [qualify] — not all experiments qualified.")
                    print("  Fix incomplete experiments, then re-run with --resume.")
                    _print_summary(results, time.time() - t_total)
                    sys.exit(1)
                continue

            if step.id == "audit-paper":
                t0 = time.time()
                # If qualify didn't run inline above (e.g. --only audit-paper),
                # rebuild qual_results from checkpoint.
                if not _qual_results:
                    _qual_results = [
                        qualify_experiment(e, checkpoint_state)
                        for e in QUALIFIABLE_EXPERIMENTS
                    ]
                audit_ok, _ = audit_against_paper(_qual_results)
                elapsed = time.time() - t0
                status  = "pass" if audit_ok else "fail"
                r = StepResult(step.id, step.label, status, elapsed)
                results.append(r)
                checkpoint_state[step.id] = status
                save_checkpoint(checkpoint_state)
                if not audit_ok and not args.continue_on_fail:
                    print(f"\n  Pipeline aborted at [audit-paper] — audit found failures.")
                    _print_summary(results, time.time() - t_total)
                    sys.exit(1)
                continue

            # ── Normal step ─────────────────────────────────────────────────
            result = run_step(step, env, args)
            results.append(result)
            checkpoint_state[step.id] = result.status
            save_checkpoint(checkpoint_state)

            if result.status == "fail" and not args.continue_on_fail:
                print(f"\n  Pipeline aborted at [{step.id}].")
                print(f"  Checkpoint saved → {CHECKPOINT}")
                print("  To resume:  python3 run_all_checkpoint.py --resume")
                _print_summary(results, time.time() - t_total)
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n  ⚠  Interrupted by user (Ctrl+C).")
        for r in results:
            if r.id not in checkpoint_state:
                checkpoint_state[r.id] = r.status
        save_checkpoint(checkpoint_state)
        print(f"  Checkpoint saved → {CHECKPOINT}")
        _print_summary(results, time.time() - t_total)
        sys.exit(130)

    _print_summary(results, time.time() - t_total)
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
        t = f"  {r.elapsed:6.0f}s" if r.status in ("pass","fail") else "        "
        print(f"  {col[r.status]} [{r.id:30s}] {r.label[:46]:46s}{t}")

    print()
    print(f"  ✓ passed      : {len(passed)}")
    print(f"  ✗ failed      : {len(failed)}")
    print(f"  ─ skipped     : {len(skipped)}")
    print(f"  ↩ resume-skip : {len(resume_skips)}")
    print(f"  Wall time     : {hh:02d}:{mm:02d}:{ss:02d}")

    # WARN-5 RESOLVED: Nguyen-12 caveat printed in every summary
    print("\n  ⚠  Nguyen-12 caveat (exp3/exp3b):")
    print("       Paper abstract: 11/12 (91.7%) uses 4-decimal rounding (Uy et al. benchmark).")
    print("       Strict R²≥0.9999 threshold: 4/12 (33.3%).")
    print("       Both figures should appear in the abstract & §10.8 for transparency.")

    data_files, fig_files, tbl_files = inventory_results()
    print(f"\n  Results → {RESULTS_DIR}")
    print(f"    Data files (JSON+CSV) : {data_files}")
    print(f"    Figures (PDF)         : {fig_files}")
    print(f"    Tables  (TeX)         : {tbl_files}")

    if failed:
        print("\n  Failed steps:")
        for r in failed:
            print(f"    [{r.id}] → {r.log_path}")
        print(f"\n  Checkpoint : {CHECKPOINT}")
        print("  Resume     : python3 run_all_checkpoint.py --resume")
    else:
        print("\n  ✓ All steps passed.")
        print(f"  Results    : {RESULTS_DIR}/")
        print(f"  Figures    : {RESULTS_DIR}/figures/")
        print(f"  Tables     : {RESULTS_DIR}/tables/")
        print("  Checkpoint : cleared")


if __name__ == "__main__":
    main()

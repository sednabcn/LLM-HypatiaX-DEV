"""
scripts/tests/test_generate_tables_day2.py

Day 2 of the 8-day fix plan: "Run and sanity-check the 18 newly-added
JSON-backed generators... two subsets need extra scrutiny": arch.tex
(unconfirmed decision/strategy schema) and validation_stats.tex/
fix5_cases.tex/timing_breakdown.tex/scalability.tex (guessed file paths,
no named source). Plus: "Start the nguyen12.tex generator... needs a real
run before it can be trusted."

WHAT THIS FILE CAN AND CANNOT CONFIRM
--------------------------------------
No real result JSON for any of these six tables was available in this
conversation (File A is the only real data file uploaded so far, and it
only feeds exp1's three tables, already covered by test_generate_tables.py).
So this suite cannot confirm what Day 2 actually needs confirmed: whether
the GUESSED filenames (validation_log*.json, fix5_cases*.json,
*timing_breakdown*.json, scalability_*.json) and the GUESSED schema
(decision-vs-strategy field name in arch.tex's source) match what the real
pipeline produces. That confirmation requires either the real repo's
result directories or someone who's seen the scripts that write those
files -- neither is available here.

What this DOES confirm, with synthetic data built to exactly match each
generator's documented/coded expectations:
  - Each generator's parsing logic is internally correct (no crashes, no
    off-by-one, correct column extraction) when given well-formed input.
  - Each generator's skip_table() fallback fires cleanly on missing/
    malformed input -- i.e. if the guessed filename never matches
    anything real, the failure mode is "table silently absent" (visible
    in SKIPPED_TABLES / the missing-JSON audit), never "wrong numbers
    silently substituted." This is the property Day 2's caution is
    actually protecting against, and it holds for all six.
  - gen_suppb_arch() specifically: verified BOTH the "decision" and
    "strategy" field-name branches parse correctly (it's coded as
    `res.get("decision") or res.get("strategy")`), so whichever name the
    real JSON uses, the parsing itself won't be the failure point -- the
    remaining unconfirmed question is only "does a real noiseless JSON
    even have either field," not "is the OR-fallback logic sound."
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "generate_tables.py"


def _run(tmp_path: Path, experiment: str,
          files: dict[str, object] | None = None) -> tuple[subprocess.CompletedProcess, Path]:
    """files: {relative_path_under_results_dir: json-serializable content}"""
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    for rel, content in (files or {}).items():
        p = results_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(content))
    results_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--experiment", experiment,
         "--results-dir", str(results_dir),
         "--output-dir", str(output_dir)],
        capture_output=True, text=True, timeout=120,
    )
    return proc, output_dir


def _read(output_dir: Path, name: str) -> str | None:
    p = output_dir / name
    return p.read_text() if p.exists() else None


# ── arch.tex — decision/strategy schema (flagged: unconfirmed) ─────────────

@pytest.mark.parametrize("field_name", ["decision", "strategy"])
def test_arch_parses_either_decision_or_strategy_field(tmp_path, field_name):
    """
    gen_suppb_arch() reads res.get("decision") or res.get("strategy") --
    confirms BOTH branches of that OR actually work. Does NOT confirm which
    one (if either) the real pipeline writes; that's still open per Day 2.
    """
    noiseless = {
        "tests": [
            {"results": {
                "MethodA": {field_name: "route_x"},
                "MethodB": {field_name: "route_y"},
            }},
            {"results": {
                "MethodA": {field_name: "route_x"},
                "MethodB": {field_name: "route_x"},
            }},
        ]
    }
    files = {
        "comparison_results/noise-noiseless/noiseless/defi/protocol_core_noiseless_1.json": noiseless
    }
    proc, output_dir = _run(tmp_path, "suppb_extra", files)
    assert proc.returncode == 0, proc.stderr
    tex = _read(output_dir, "arch.tex")
    assert tex is not None, f"arch.tex not written for field {field_name!r}.\nstderr:\n{proc.stderr}"
    assert "MethodA & route_x & 2" in tex
    assert "MethodB & route_x & 1" in tex
    assert "MethodB & route_y & 1" in tex


def test_arch_skips_cleanly_when_neither_field_present(tmp_path):
    """Neither 'decision' nor 'strategy' present -> skip_table(), not a
    crash and not an empty-but-written table."""
    noiseless = {"tests": [{"results": {"MethodA": {"some_other_field": 1}}}]}
    files = {
        "comparison_results/noise-noiseless/noiseless/defi/protocol_core_noiseless_1.json": noiseless
    }
    proc, output_dir = _run(tmp_path, "suppb_extra", files)
    assert proc.returncode == 0, proc.stderr
    assert _read(output_dir, "arch.tex") is None
    assert "arch.tex" in (proc.stdout + proc.stderr)


# ── validation_stats.tex — guessed path: validation_log*.json ──────────────

def test_validation_stats_parses_numeric_fields(tmp_path):
    data = {"schema_violations": 3, "range_checks_failed": 1, "duplicate_ids": 0,
            "non_numeric_string_field": "should be ignored, not a table row"}
    files = {"validation/validation_log_20260913.json": data}
    proc, output_dir = _run(tmp_path, "suppb_extra", files)
    assert proc.returncode == 0, proc.stderr
    tex = _read(output_dir, "validation_stats.tex")
    assert tex is not None, f"stderr:\n{proc.stderr}"
    assert "schema_violations & 3" in tex
    assert "range_checks_failed & 1" in tex
    assert "non_numeric_string_field" not in tex  # non-numeric fields silently dropped, not stringified


def test_validation_stats_skips_when_guessed_filename_absent(tmp_path):
    """This is the actual open question from Day 2: if the real pipeline
    names this file something other than validation_log*.json, this is
    the failure mode -- confirmed silent-skip, not a wrong table."""
    proc, output_dir = _run(tmp_path, "suppb_extra", files=None)
    assert proc.returncode == 0, proc.stderr
    assert _read(output_dir, "validation_stats.tex") is None


# ── fix5_cases.tex — guessed path: fix5_cases*.json ─────────────────────────

def test_fix5_cases_parses_list_and_dict_shapes(tmp_path):
    # Shape 1: bare list
    list_shape = [
        {"case": "case_17", "before": "fabricated pass", "after": "corrected fail"},
        {"case": "case_42", "before": "0.87", "after": "0.87"},
    ]
    proc, output_dir = _run(tmp_path, "routing", {"routing/fix5_cases_run1.json": list_shape})
    assert proc.returncode == 0, proc.stderr
    tex = _read(output_dir, "fix5_cases.tex")
    assert tex is not None, f"stderr:\n{proc.stderr}"
    assert "case_17 & fabricated pass & corrected fail" in tex
    assert "case_42 & 0.87 & 0.87" in tex


def test_fix5_cases_parses_dict_with_cases_key(tmp_path):
    # Shape 2: {"cases": [...]}
    dict_shape = {"cases": [{"case": "case_5", "before": "X", "after": "Y"}]}
    proc, output_dir = _run(tmp_path, "routing", {"routing/fix5_cases_run2.json": dict_shape})
    assert proc.returncode == 0, proc.stderr
    tex = _read(output_dir, "fix5_cases.tex")
    assert tex is not None, f"stderr:\n{proc.stderr}"
    assert "case_5 & X & Y" in tex


def test_fix5_cases_skips_when_guessed_filename_absent(tmp_path):
    proc, output_dir = _run(tmp_path, "routing", files=None)
    assert proc.returncode == 0, proc.stderr
    assert _read(output_dir, "fix5_cases.tex") is None


# ── timing_breakdown.tex — guessed path: *timing_breakdown*.json ───────────

def test_timing_breakdown_parses_numeric_fields(tmp_path):
    data = {"routing_decision_s": 0.02, "sub_method_eval_s": 1.35, "total_s": 1.37}
    proc, output_dir = _run(tmp_path, "routing", {"routing/routing_timing_breakdown.json": data})
    assert proc.returncode == 0, proc.stderr
    tex = _read(output_dir, "timing_breakdown.tex")
    assert tex is not None, f"stderr:\n{proc.stderr}"
    assert "routing_decision_s & 0.02" in tex
    assert "total_s & 1.37" in tex


def test_timing_breakdown_skips_when_guessed_filename_absent(tmp_path):
    proc, output_dir = _run(tmp_path, "routing", files=None)
    assert proc.returncode == 0, proc.stderr
    assert _read(output_dir, "timing_breakdown.tex") is None


# ── scalability.tex — guessed path: scalability_*.json ──────────────────────

def test_scalability_parses_sizes_and_per_size(tmp_path):
    data = {
        "dataset_sizes": [100, 1000, 10000],
        "per_size": {
            "100": {"mean_time_s": 0.5, "mean_r2": 0.91},
            "1000": {"mean_time_s": 4.2, "mean_r2": 0.95},
            "10000": {"mean_time_s": 41.0, "mean_r2": 0.97},
        },
    }
    proc, output_dir = _run(tmp_path, "routing", {"routing/scalability_20260913.json": data})
    assert proc.returncode == 0, proc.stderr
    tex = _read(output_dir, "scalability.tex")
    assert tex is not None, f"stderr:\n{proc.stderr}"
    assert "100 & 0.50 & 0.9100" in tex
    assert "10000 & 41.00 & 0.9700" in tex


def test_scalability_skips_when_dataset_sizes_missing(tmp_path):
    """File found, but missing the one required key -- must still skip
    cleanly, not write a table with only '---' rows."""
    data = {"per_size": {"100": {"mean_time_s": 0.5, "mean_r2": 0.91}}}
    proc, output_dir = _run(tmp_path, "routing", {"routing/scalability_bad.json": data})
    assert proc.returncode == 0, proc.stderr
    assert _read(output_dir, "scalability.tex") is None


def test_scalability_skips_when_guessed_filename_absent(tmp_path):
    proc, output_dir = _run(tmp_path, "routing", files=None)
    assert proc.returncode == 0, proc.stderr
    assert _read(output_dir, "scalability.tex") is None


# ── nguyen12.tex — "never run against real result JSON" (Day 2 flag) ───────

def _shape_h_record(eq_id: str, r2: float) -> dict:
    return {"metadata": {"nguyen_id": eq_id, "name": eq_id}, "evaluation": {"r2": r2}}


def test_nguyen12_parses_raw_exp3_seed42_shape_h(tmp_path):
    """12 equations, raw exp3 seed-42 file only (no merged file)."""
    eq_ids = [f"nguyen{i}" for i in range(1, 13)]
    raw = {
        "config": {"seed": 42},
        "results": {
            "pysr": [_shape_h_record(e, 0.95) for e in eq_ids],
            "hypatiax": [_shape_h_record(e, 0.99) for e in eq_ids],
        },
    }
    proc, output_dir = _run(tmp_path, "exp3", {"extrapolation/exp3_nguyen12_seed42_run1.json": raw})
    assert proc.returncode == 0, proc.stderr
    tex = _read(output_dir, "nguyen12.tex")
    assert tex is not None, f"nguyen12.tex not written with 12 equations.\nstderr:\n{proc.stderr}"
    assert "nguyen1 " in tex or "nguyen1 &" in tex
    assert "n seeds" in tex
    # every equation should show exactly 1 seed contributing
    assert tex.count(" & 1 \\\\") == 12


def test_nguyen12_parses_merged_exp3b_shape(tmp_path):
    """12 equations via the merged exp3b shape only (no raw exp3 file).
    NOTE: load_best()'s glob_pat for this source is the LITERAL string
    "_merged.json" (no wildcard) -- confirmed by reading load_best()'s
    Path.glob() call directly. The file must be named exactly that."""
    eq_ids = [f"nguyen{i}" for i in range(1, 13)]
    merged = []
    for e in eq_ids:
        for seed in (99, 123):
            merged.append({
                "nguyen_id": e, "name": e, "seed": seed,
                "systems": {"pysr": {"r2_raw": 0.93}, "hypatiax": {"r2_raw": 0.999}},
            })
    proc, output_dir = _run(tmp_path, "exp3", {"extrapolation/multi_seed/_merged.json": merged})
    assert proc.returncode == 0, proc.stderr
    tex = _read(output_dir, "nguyen12.tex")
    assert tex is not None, f"stderr:\n{proc.stderr}"
    # every equation should show exactly 2 seeds contributing (99, 123)
    assert tex.count(" & 2 \\\\") == 12


def test_nguyen12_skips_below_12_equations(tmp_path):
    """Regression guard for the documented `if len(eq_ids) < 12: skip_table()`
    threshold -- 11 equations must not produce a partial table."""
    eq_ids = [f"nguyen{i}" for i in range(1, 12)]  # 11, not 12
    raw = {
        "config": {"seed": 42},
        "results": {"pysr": [_shape_h_record(e, 0.9) for e in eq_ids],
                    "hypatiax": [_shape_h_record(e, 0.9) for e in eq_ids]},
    }
    proc, output_dir = _run(tmp_path, "exp3", {"extrapolation/exp3_nguyen12_seed42_run1.json": raw})
    assert proc.returncode == 0, proc.stderr
    assert _read(output_dir, "nguyen12.tex") is None
    assert "11/12" in (proc.stdout + proc.stderr)


def test_nguyen12_skips_when_no_source_found(tmp_path):
    proc, output_dir = _run(tmp_path, "exp3", files=None)
    assert proc.returncode == 0, proc.stderr
    assert _read(output_dir, "nguyen12.tex") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

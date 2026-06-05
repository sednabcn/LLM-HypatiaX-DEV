"""
check_symbolic_equivalence.py
─────────────────────────────
Checks whether expressions recovered by HypatiaX / PySR are symbolically
equivalent to the Nguyen-12 ground truths.

Three-tier classification per (seed, system, task):
  SYMBOLIC  – SymPy simplify(found − gt) == 0  (with domain assumptions)
  NUMERICAL – not symbolic, but max |found − gt| < NUM_TOL on held-out grid
  FAIL      – r² < R2_THRESHOLD  OR  numerical error >= NUM_TOL

Usage:
  python check_symbolic_equivalence.py [seed_file1.json ...]

  If no files are given, the script globs for exp3_nguyen12_seed*.json
  in the current directory.

Output:
  • Console table (per seed)
  • symbolic_equivalence_report.csv   (machine-readable)
  • symbolic_equivalence_summary.txt  (paper-ready summary)
"""

import json
import sys
import glob
import csv
import time
import warnings
import traceback
from pathlib import Path

import numpy as np
import sympy as sp
from sympy import symbols, sympify, simplify, lambdify

# ── Thresholds ────────────────────────────────────────────────────────────────
R2_THRESHOLD  = 0.9999   # below this → FAIL immediately
NUM_TOL       = 1e-3     # max pointwise error for NUMERICAL match
SYMPY_TIMEOUT = 8        # seconds before we skip symbolic check (approx)
N_SAMPLE      = 1000     # points for numerical check

# ── SymPy symbols with domain hints ──────────────────────────────────────────
# Most Nguyen tasks use x ∈ (0,1] or [1,6]; using positive avoids log/sqrt issues.
x_pos = sp.Symbol('x', positive=True)
y_pos = sp.Symbol('y', positive=True)
x_sym = sp.Symbol('x', real=True)
y_sym = sp.Symbol('y', real=True)

# ── Nguyen-12 sampling domains (training range from benchmark) ────────────────
DOMAINS = {
    'N1':  {'x': (0, 1)},
    'N2':  {'x': (0, 1)},
    'N3':  {'x': (0, 1)},
    'N4':  {'x': (0, 1)},
    'N5':  {'x': (-1, 1)},
    'N6':  {'x': (0, 1)},
    'N7':  {'x': (0, 2)},
    'N8':  {'x': (0, 4)},
    'N9':  {'x': (-1, 1), 'y': (-1, 1)},
    'N10': {'x': (-1, 1), 'y': (-1, 1)},
    'N11': {'x': (1, 2),  'y': (1, 2)},
    'N12': {'x': (-1, 1), 'y': (-1, 1)},
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_expr(expr_str: str, assume_positive: bool = True):
    """Parse a string expression under positive or real assumptions."""
    xs = x_pos if assume_positive else x_sym
    ys = y_pos if assume_positive else y_sym
    local = {'x': xs, 'y': ys,
             'sqrt': sp.sqrt, 'log': sp.log, 'exp': sp.exp,
             'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
             'abs': sp.Abs, 'pi': sp.pi, 'E': sp.E}
    return sympify(expr_str, locals=local)


def try_symbolic(found_str: str, gt_str: str, task_id: str):
    """
    Attempt SymPy simplification.  Returns (is_symbolic, detail_str).
    Tries positive-domain assumptions first, then real.
    """
    for assume_pos in (True, False):
        try:
            t0 = time.time()
            found = parse_expr(found_str, assume_pos)
            gt    = parse_expr(gt_str,    assume_pos)
            diff  = simplify(found - gt)
            elapsed = time.time() - t0
            if diff == 0:
                domain = "positive" if assume_pos else "real"
                return True, f"simplify→0 [{domain} domain, {elapsed:.1f}s]"
            if time.time() - t0 > SYMPY_TIMEOUT:
                return False, "sympy timeout"
        except Exception as e:
            pass  # try next assumption
    return False, "simplify≠0"


def sample_domain(task_id: str, n: int = N_SAMPLE, seed: int = 42):
    """Return a dict {var_name: np.array} of random samples in the task domain."""
    rng = np.random.default_rng(seed)
    domain = DOMAINS.get(task_id, {'x': (0, 1)})
    return {
        var: rng.uniform(lo, hi, n)
        for var, (lo, hi) in domain.items()
    }


def try_numerical(found_str: str, gt_str: str, task_id: str):
    """
    Evaluate both expressions on a random grid and compute max |err|.
    Returns (is_numerical, max_err, detail_str).
    """
    samples = sample_domain(task_id)
    sym_vars = [x_sym] + ([y_sym] if 'y' in samples else [])
    np_vars  = [samples['x']] + ([samples['y']] if 'y' in samples else [])

    try:
        found_expr = parse_expr(found_str, assume_positive=False)
        gt_expr    = parse_expr(gt_str,    assume_positive=False)

        f_func = lambdify(sym_vars, found_expr, modules='numpy')
        g_func = lambdify(sym_vars, gt_expr,    modules='numpy')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_vals = np.asarray(f_func(*np_vars), dtype=complex)
            g_vals = np.asarray(g_func(*np_vars), dtype=complex)

        # Discard complex / nan / inf results
        valid = np.isfinite(f_vals.real) & np.isfinite(g_vals.real) & \
                (np.abs(f_vals.imag) < 1e-6) & (np.abs(g_vals.imag) < 1e-6)

        if valid.sum() < 10:
            return False, float('nan'), "too few valid points"

        err = np.abs(f_vals.real[valid] - g_vals.real[valid])
        max_err = float(np.max(err))
        matched = max_err < NUM_TOL
        return matched, max_err, f"max_err={max_err:.2e} on {valid.sum()} pts"

    except Exception as e:
        return False, float('nan'), f"eval error: {e}"


def classify(r2: float, found_str: str, gt_str: str, task_id: str):
    """
    Returns dict with keys: tier, r2, symbolic, numerical, max_err, detail
    """
    result = dict(r2=r2, tier='FAIL', symbolic=False,
                  numerical=False, max_err=float('nan'), detail='')

    if r2 < R2_THRESHOLD:
        result['detail'] = f"r²={r2:.6f} < threshold {R2_THRESHOLD}"
        return result

    # ── Tier 1: symbolic ─────────────────────────────────────────────────────
    is_sym, sym_detail = try_symbolic(found_str, gt_str, task_id)
    if is_sym:
        result.update(tier='SYMBOLIC', symbolic=True, detail=sym_detail)
        return result

    # ── Tier 2: numerical ────────────────────────────────────────────────────
    is_num, max_err, num_detail = try_numerical(found_str, gt_str, task_id)
    result['max_err'] = max_err
    if is_num:
        result.update(tier='NUMERICAL', numerical=True,
                      detail=f"{sym_detail} | {num_detail}")
    else:
        result.update(tier='FAIL',
                      detail=f"{sym_detail} | {num_detail}")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

TIER_SYMBOL = {'SYMBOLIC': '✓✓', 'NUMERICAL': '✓~', 'FAIL': '✗ '}
TIER_COLOR  = {'SYMBOLIC': '\033[92m', 'NUMERICAL': '\033[93m', 'FAIL': '\033[91m'}
RESET = '\033[0m'

def color(tier, text):
    return f"{TIER_COLOR.get(tier,'')}{text}{RESET}"


def process_file(path: str):
    with open(path) as f:
        data = json.load(f)

    seed = data['config']['seed']
    rows = []

    print(f"\n{'═'*72}")
    print(f"  SEED {seed}  —  {Path(path).name}")
    print(f"{'═'*72}")
    print(f"  {'Task':<5} {'System':<10} {'r²':>8}  {'Tier':<10}  Detail")
    print(f"  {'─'*5} {'─'*10} {'─'*8}  {'─'*10}  {'─'*40}")

    for system_key in ('hypatiax', 'pysr'):
        entries = data['results'][system_key]
        for entry in entries:
            tid      = entry['metadata']['nguyen_id']
            gt_str   = entry['metadata']['ground_truth']
            found_str = entry['expression']
            r2       = entry['evaluation']['r2']

            res = classify(r2, found_str, gt_str, tid)

            tier = res['tier']
            sym  = TIER_SYMBOL[tier]
            print(f"  {tid:<5} {system_key:<10} {r2:>8.6f}  "
                  f"{color(tier, f'{sym} {tier:<8}')}  {res['detail'][:55]}")

            rows.append(dict(
                seed=seed, system=system_key, task=tid,
                ground_truth=gt_str,
                found_expression=found_str[:120],
                r2=r2, tier=tier,
                symbolic=res['symbolic'],
                numerical=res['numerical'],
                max_err=res['max_err'],
                detail=res['detail'],
            ))

    return rows


def print_summary(all_rows):
    from collections import defaultdict

    print(f"\n{'═'*72}")
    print("  CROSS-SEED SUMMARY")
    print(f"{'═'*72}")

    seeds   = sorted(set(r['seed'] for r in all_rows))
    systems = sorted(set(r['system'] for r in all_rows))
    tasks   = [f'N{i}' for i in range(1, 13)]

    for system in systems:
        print(f"\n  System: {system.upper()}")
        header = f"  {'Task':<5}" + "".join(f"  Seed {s:<5}" for s in seeds)
        print(header)
        print("  " + "─" * (len(header) - 2))

        sym_counts  = defaultdict(int)
        num_counts  = defaultdict(int)
        fail_counts = defaultdict(int)

        for task in tasks:
            line = f"  {task:<5}"
            for s in seeds:
                match = [r for r in all_rows
                         if r['seed']==s and r['system']==system and r['task']==task]
                if match:
                    tier = match[0]['tier']
                    sym  = TIER_SYMBOL[tier]
                    line += f"  {color(tier, f'{sym} {tier[:3]}')}  "
                    sym_counts[s]  += (tier == 'SYMBOLIC')
                    num_counts[s]  += (tier == 'NUMERICAL')
                    fail_counts[s] += (tier == 'FAIL')
                else:
                    line += "  N/A      "
            print(line)

        print(f"\n  {'Totals':<5}", end="")
        for s in seeds:
            total = sym_counts[s] + num_counts[s] + fail_counts[s]
            line = f"S:{sym_counts[s]} N:{num_counts[s]} F:{fail_counts[s]}"
            print(f"  {line:<11}", end="")
        print()

    # Paper-ready sentence
    print(f"\n{'═'*72}")
    print("  PAPER-READY CLAIM")
    print(f"{'═'*72}")
    for system in systems:
        sym_by_seed  = {}
        num_by_seed  = {}
        fail_by_seed = {}
        for s in seeds:
            rows_s = [r for r in all_rows if r['seed']==s and r['system']==system]
            sym_by_seed[s]  = sum(r['tier']=='SYMBOLIC'  for r in rows_s)
            num_by_seed[s]  = sum(r['tier']=='NUMERICAL' for r in rows_s)
            fail_by_seed[s] = sum(r['tier']=='FAIL'      for r in rows_s)

        sym_range  = f"{min(sym_by_seed.values())}–{max(sym_by_seed.values())}"
        num_range  = f"{min(num_by_seed.values())}–{max(num_by_seed.values())}"
        fail_tasks = sorted(set(
            r['task'] for r in all_rows
            if r['system']==system and r['tier']=='FAIL'
        ))
        print(f"\n  [{system.upper()}]")
        print(f"  Symbolic matches : {sym_range}/12 across seeds")
        print(f"  Numerical matches: {num_range}/12 across seeds")
        if fail_tasks:
            print(f"  Failures         : {', '.join(fail_tasks)} (seed-dependent)")
        else:
            print(f"  Failures         : none")


def write_csv(all_rows, out_path="symbolic_equivalence_report.csv"):
    fields = ['seed','system','task','r2','tier','symbolic','numerical',
              'max_err','ground_truth','found_expression','detail']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n  CSV saved → {out_path}")


def write_summary_txt(all_rows, out_path="symbolic_equivalence_summary.txt"):
    seeds   = sorted(set(r['seed'] for r in all_rows))
    systems = sorted(set(r['system'] for r in all_rows))
    tasks   = [f'N{i}' for i in range(1, 13)]

    lines = ["Symbolic Equivalence Summary — Nguyen-12 Benchmark", "=" * 55, ""]
    lines += [
        "Legend:",
        "  SYMBOLIC  = SymPy simplify(found − gt) == 0",
        "  NUMERICAL = max pointwise error < 1e-3 on held-out grid",
        "  FAIL      = r² < 0.9999 OR numerical error ≥ 1e-3",
        "",
    ]

    for system in systems:
        lines.append(f"System: {system.upper()}")
        header = f"  {'Task':<5}" + "".join(f"  Seed {s:<7}" for s in seeds)
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for task in tasks:
            row_line = f"  {task:<5}"
            for s in seeds:
                match = [r for r in all_rows
                         if r['seed']==s and r['system']==system and r['task']==task]
                if match:
                    tier = match[0]['tier']
                    row_line += f"  {tier:<10}"
                else:
                    row_line += f"  {'N/A':<10}"
            lines.append(row_line)
        lines.append("")

    with open(out_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"  TXT saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(
        glob.glob("exp3_nguyen12_seed*.json")
    )

    if not files:
        print("No JSON files found. Pass file paths as arguments or run from "
              "the directory containing exp3_nguyen12_seed*.json files.")
        sys.exit(1)

    print(f"\nChecking {len(files)} file(s): {[Path(f).name for f in files]}")
    print(f"Thresholds: r²≥{R2_THRESHOLD}, num_tol={NUM_TOL}, n_sample={N_SAMPLE}")

    all_rows = []
    for path in files:
        try:
            all_rows.extend(process_file(path))
        except Exception as e:
            print(f"\nERROR reading {path}: {e}")
            traceback.print_exc()

    print_summary(all_rows)
    write_csv(all_rows)
    write_summary_txt(all_rows)
    print("\nDone.\n")

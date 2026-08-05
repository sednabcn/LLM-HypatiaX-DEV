"""
generate_nguyen12_symequiv_table.py
────────────────────────────────────
Reads  symbolic_equivalence_report.csv  (produced by check_symbolic_equivalence.py)
and writes  tables/nguyen12_symequiv.tex  — a publication-ready LaTeX longtable
showing the three-tier classification (SYMBOLIC / NUMERICAL / FAIL) for every
(task × system) pair, with one column per seed.

Interface matches generate_tables.py so ci_postprocess.yml can call it uniformly:

    python scripts/generate_nguyen12_symequiv_table.py \
        --results-dir  <RESULT_DIR>   \
        --output-dir   <RESULT_DIR>/tables

Exit codes:
  0  — table written successfully
  1  — symbolic_equivalence_report.csv not found (hard error)
  2  — CSV is empty or has no recognised rows (hard error)
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# ── Tier display ──────────────────────────────────────────────────────────────
# LaTeX commands are defined inline in the table preamble so the .tex file is
# self-contained (no external sty dependency required).
TIER_CMD = {
    "SYMBOLIC":  r"\symT",   # green checkmark ✓✓
    "NUMERICAL": r"\numT",   # amber  ✓~
    "FAIL":      r"\failT",  # red    ✗
}

TASK_ORDER  = [f"N{i}" for i in range(1, 13)]
SYS_ORDER   = ["hypatiax", "pysr"]
SYS_DISPLAY = {"hypatiax": r"\textsc{HypatiaX}", "pysr": r"\textsc{PySR}"}

# ─────────────────────────────────────────────────────────────────────────────

def load_csv(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def build_index(rows: list[dict]):
    """
    Returns:
      index  : {(system, task, seed): tier}
      seeds  : sorted list of seed values (ints)
      systems: list of systems present in file
    """
    index   = {}
    seeds   = set()
    systems = set()
    for r in rows:
        seed   = int(r["seed"])
        system = r["system"].strip().lower()
        task   = r["task"].strip()
        tier   = r["tier"].strip().upper()
        index[(system, task, seed)] = tier
        seeds.add(seed)
        systems.add(system)
    seeds_sorted = sorted(seeds)
    # Preserve canonical ordering; fall back to sorted for unknown systems.
    sys_sorted = [s for s in SYS_ORDER if s in systems] + \
                 sorted(systems - set(SYS_ORDER))
    return index, seeds_sorted, sys_sorted


def tier_summary(index, system, task, seeds):
    """Return (n_sym, n_num, n_fail) counts across seeds for one (system, task)."""
    tiers = [index.get((system, task, s), "FAIL") for s in seeds]
    return (
        sum(t == "SYMBOLIC"  for t in tiers),
        sum(t == "NUMERICAL" for t in tiers),
        sum(t == "FAIL"      for t in tiers),
    )


def render_table(index, seeds, systems, single_seed: bool) -> str:
    n_seeds = len(seeds)

    # ── column spec ───────────────────────────────────────────────────────────
    # l  (task) | l (system) | c…c (one per seed) | ccc (summary S/N/F)
    seed_cols  = "c" * n_seeds
    col_spec   = f"ll{seed_cols}ccc"

    # ── seed column headers ───────────────────────────────────────────────────
    seed_headers = " & ".join(
        r"\multicolumn{1}{c}{\scriptsize seed\," + str(s) + "}" for s in seeds
    )

    # ── caption suffix ────────────────────────────────────────────────────────
    if single_seed:
        caption_detail = f"single seed ({seeds[0]})"
    else:
        seed_str = ", ".join(str(s) for s in seeds)
        caption_detail = f"{n_seeds} seeds ({seed_str})"

    lines: list[str] = []
    a = lines.append   # shorthand

    # ── preamble macros (self-contained) ─────────────────────────────────────
    a(r"% ── Tier display macros (defined locally; no extra sty required) ────────")
    a(r"\newcommand{\symT}{\textcolor{OliveGreen}{\bfseries S}}")
    a(r"\newcommand{\numT}{\textcolor{Goldenrod}{\bfseries N}}")
    a(r"\newcommand{\failT}{\textcolor{BrickRed}{\bfseries F}}")
    a(r"% Requires: \usepackage[dvipsnames]{xcolor}")
    a(r"%           \usepackage{longtable,booktabs}")
    a("")

    # ── table environment ────────────────────────────────────────────────────
    a(r"\begin{longtable}{" + col_spec + r"}")
    a(r"\caption{Symbolic equivalence classification on the Nguyen-12 benchmark")
    a(r"  (" + caption_detail + r").")
    a(r"  \textbf{S}\,=\,SYMBOLIC (SymPy simplify\,=\,0);")
    a(r"  \textbf{N}\,=\,NUMERICAL (max pointwise error $<10^{-3}$);")
    a(r"  \textbf{F}\,=\,FAIL ($r^2 < 0.9999$ or numerical error $\ge 10^{-3}$).}")
    a(r"\label{tab:nguyen12_symequiv}\\")
    a(r"\toprule")

    # header row
    summary_header = (
        r"\multicolumn{3}{c}{\scriptsize across seeds}"
        if not single_seed
        else r"\multicolumn{3}{c}{\scriptsize —}"
    )
    a(
        r"\multicolumn{1}{l}{Task} & "
        r"\multicolumn{1}{l}{System} & "
        + seed_headers
        + r" & "
        + summary_header
        + r" \\"
    )
    # sub-header for summary columns
    a(r"\cmidrule(lr){1-2}"
      + (r"\cmidrule(lr){3-" + str(2 + n_seeds) + "}" if n_seeds > 0 else "")
      + r"\cmidrule(lr){"  + str(3 + n_seeds) + r"-" + str(5 + n_seeds) + r"}")
    a(
        r" & "
        + " & " * n_seeds
        + r"{\scriptsize\#S} & {\scriptsize\#N} & {\scriptsize\#F} \\"
    )
    a(r"\midrule")
    a(r"\endfirsthead")
    a(r"\multicolumn{" + str(5 + n_seeds) + r"}{l}{\small\itshape (continued)} \\")
    a(r"\toprule")
    a(
        r"\multicolumn{1}{l}{Task} & "
        r"\multicolumn{1}{l}{System} & "
        + seed_headers
        + r" & "
        + summary_header
        + r" \\"
    )
    a(r"\midrule")
    a(r"\endhead")
    a(r"\midrule")
    a(r"\multicolumn{" + str(5 + n_seeds) + r"}{r}{\small\itshape (continued on next page)} \\")
    a(r"\endfoot")
    a(r"\bottomrule")
    a(r"\endlastfoot")

    # ── body rows ─────────────────────────────────────────────────────────────
    for t_idx, task in enumerate(TASK_ORDER):
        for s_idx, system in enumerate(systems):
            # task cell: spans both system rows with cmidrule above each task block
            if s_idx == 0 and t_idx > 0:
                a(r"\cmidrule(lr){1-" + str(5 + n_seeds) + r"}")

            task_cell   = task   if s_idx == 0 else ""
            system_cell = SYS_DISPLAY.get(system, system)

            tier_cells = " & ".join(
                TIER_CMD.get(index.get((system, task, s), "FAIL"), r"\failT")
                for s in seeds
            )

            n_sym, n_num, n_fail = tier_summary(index, system, task, seeds)
            # Only show summary for first system row (spans both visually via
            # manual inspection); for clarity print per-system.
            summary_cells = f"{n_sym} & {n_num} & {n_fail}"

            a(
                f"{task_cell} & {system_cell} & {tier_cells} & {summary_cells} \\\\"
            )

    a(r"\end{longtable}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description="Generate nguyen12_symequiv.tex from symbolic_equivalence_report.csv"
    )
    ap.add_argument("--results-dir", required=True,
                    help="Directory containing symbolic_equivalence_report.csv")
    ap.add_argument("--output-dir",  required=True,
                    help="Directory to write nguyen12_symequiv.tex into")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    csv_path    = results_dir / "symbolic_equivalence_report.csv"

    # ── Guards ────────────────────────────────────────────────────────────────
    if not csv_path.exists():
        print(f"::error::symbolic_equivalence_report.csv not found: {csv_path}")
        sys.exit(1)

    rows = load_csv(csv_path)
    if not rows:
        print(f"::error::symbolic_equivalence_report.csv is empty: {csv_path}")
        sys.exit(2)

    # ── Build ─────────────────────────────────────────────────────────────────
    index, seeds, systems = build_index(rows)

    if not seeds:
        print(f"::error::No valid rows parsed from {csv_path}")
        sys.exit(2)

    single_seed = len(seeds) == 1
    table_tex   = render_table(index, seeds, systems, single_seed)

    # ── Write ─────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "nguyen12_symequiv.tex"
    out_path.write_text(table_tex, encoding="utf-8")

    n_rows = len(rows)
    n_tasks_covered = len({r["task"] for r in rows})
    print(
        f"  nguyen12_symequiv.tex written → {out_path}\n"
        f"  {n_rows} CSV rows  |  {n_tasks_covered}/12 tasks  "
        f"|  {len(seeds)} seed(s)  |  {len(systems)} system(s)"
    )


if __name__ == "__main__":
    main()

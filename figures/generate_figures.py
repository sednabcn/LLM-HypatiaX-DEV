#!/usr/bin/env python3
"""
generate_all_figures.py
=======================
Master orchestrator for ALL HypatiaX paper figures.

Dispatches to the five specialist tools already in the repo, routing each one
to the correct data source and output directory.  A single call produces every
figure referenced in the JMLR paper and its supplements.

Figures produced
----------------
GROUP A — Instability analysis (5 seaborn figures)
  Produced by: hypatiax/tools/visualizations/hypatiax_instability_analysis_pipeline.py
  Output stems (png + pdf each):
    fig_paper_complexity_vs_instability  ← KEY / Fig §4 in paper
    fig_paper_complexity_vs_success
    fig_paper_mean_vs_instability
    fig_paper_instability_hist
    fig_paper_regime_counts
  Also writes: hypatiax/data/figures/instability_analysis.csv

GROUP B — 3D instability plots (6 figures including 3D surface)
  Produced by: hypatiax/tools/visualizations/hypatiax_3D_plot_instability.py
  Output stems:
    fig_instability_3d
    fig_instability_phase
    fig_instability_hist
    fig_instability_success_vs_instability
    fig_instability_regimes
    fig_instability_surface

GROUP C — Extrapolation pipeline (scatter II vs extrap R²)
  Produced by: hypatiax/tools/visualizations/build_extrapolation_pipeline_final.py
  Output files:
    fig_instability_vs_extrapolation.png   (+ instability_extrapolation.csv)
  Requires: --input benchmark JSON  +  --instability-csv from GROUP A CSV

GROUP D — Comparative system visualisations (boxplot, barplot, scatter)
  Produced by: hypatiax/tools/visualizations/create_visualizations.py
  Output files:
    figure_boxplot_comparison.{pdf,png}
    figure_barplot_comparison.{pdf,png}
    figure_scatter_r2_vs_extrap.{pdf,png}
  Requires: all_systems_merged.json  (produced by experiments/comparison/merge_all_systems.py)

GROUP E — Legacy / paper results plot
  Produced by: hypatiax/tools/visualizations/plot_results.py
  Output files:
    figures/results.pdf

GROUP F — R² heatmaps (Fig 3 & Fig 4 in paper §10.2 / §10.1)
  Produced by: hypatiax/tools/visualizations/plot_r2_heatmaps.py
  Output files:
    fig_r2_heatmap_clipped.{png,pdf}   ← Fig 3  (values clipped to [−1, 1])
    fig_r2_heatmap_raw.{png,pdf}       ← Fig 4  (raw / unclipped values)
  Requires: Core-15 ablation JSON (same source as GROUP D / ablation table)

GROUP G — Portfolio Variance seed-sweep bar chart (Fig 5 in paper §10.5)
  Produced by: hypatiax/tools/visualizations/plot_portfolio_seed_sweep.py
  Output files:
    fig_portfolio_seed_sweep.{png,pdf}
  Requires: portfolio_variance_seed_sweep.json

PATH RESOLUTION
---------------
All paths are resolved relative to the repo root (the directory that contains
the hypatiax/ package).  Pass --repo-root to override.

USAGE
-----
  # Run all groups (auto-detect repo root from script location)
  python generate_all_figures.py

  # Specify repo root explicitly
  python generate_all_figures.py --repo-root /path/to/hypatiax-repo

  # Run only specific groups
  python generate_all_figures.py --groups A B C

  # Custom data inputs
  python generate_all_figures.py \\
      --benchmark-json hypatiax/data/results/hypatiax_defi_benchmark_v3c2_results.json \\
      --instability-csv hypatiax/data/figures/instability_analysis.csv \\
      --merged-json hypatiax/experiments/comparison/all_systems_merged.json

  # Choose output formats (default: png pdf)
  python generate_all_figures.py --format png pdf svg

  # Dry run — print commands without executing
  python generate_all_figures.py --dry-run

DEPENDENCIES
------------
  pip install matplotlib seaborn numpy pandas scipy scikit-learn sympy

EXIT CODES
----------
  0  All groups succeeded (or were skipped due to missing data)
  1  One or more groups failed hard
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_repo_root(start: Path) -> Path:
    """
    Walk up from `start` until we find a directory that contains a hypatiax/
    sub-package (identified by hypatiax/__init__.py).  Falls back to `start`.
    """
    for candidate in [start, *start.parents]:
        if (candidate / "hypatiax" / "__init__.py").exists():
            return candidate
    return start


def _resolve(root: Path, *parts: str) -> Path:
    return root.joinpath(*parts)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

class _Runner:
    """
    Thin wrapper around subprocess that records pass/fail per group and
    supports dry-run mode.
    """

    def __init__(self, dry_run: bool = False, verbose: bool = False):
        self.dry_run  = dry_run
        self.verbose  = verbose
        self.results:  dict[str, str] = {}   # group → "ok" | "skip" | "fail"
        self._errors:  list[str]      = []

    # ── public ────────────────────────────────────────────────────────────────

    def run(
        self,
        group:  str,
        label:  str,
        cmd:    list[str],
        *,
        cwd:    Path | None = None,
        skip_reason: str | None = None,
    ) -> bool:
        """
        Execute `cmd` (a list of strings) in `cwd`.
        Returns True on success / skip, False on failure.
        """
        _sep = "─" * 70
        print(f"\n{_sep}")
        print(f"  GROUP {group}  {label}")
        print(f"{_sep}")

        if skip_reason:
            print(f"  ⚠  SKIP — {skip_reason}")
            self.results[group] = "skip"
            return True

        pretty_cmd = " \\\n    ".join(cmd)
        if cwd:
            print(f"  cwd : {cwd}")
        print(f"  cmd : {pretty_cmd}\n")

        if self.dry_run:
            print("  [DRY RUN — not executed]")
            self.results[group] = "ok"
            return True

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                check=False,          # we handle non-zero ourselves
            )
        except FileNotFoundError as exc:
            msg = f"  ✗ Command not found: {exc}"
            print(msg)
            self._errors.append(f"Group {group}: {msg}")
            self.results[group] = "fail"
            return False

        if proc.returncode == 0:
            print(f"\n  ✓ Group {group} finished OK")
            self.results[group] = "ok"
            return True
        else:
            msg = f"  ✗ Group {group} exited with code {proc.returncode}"
            print(msg)
            self._errors.append(msg)
            self.results[group] = "fail"
            return False

    # ── summary ───────────────────────────────────────────────────────────────

    def print_summary(self) -> int:
        """Print final report.  Returns 0 if no hard failures, else 1."""
        _sep = "═" * 70
        print(f"\n{_sep}")
        print("  FIGURE GENERATION SUMMARY")
        print(_sep)

        icon = {"ok": "✓", "skip": "–", "fail": "✗"}
        n_fail = 0
        for grp, status in sorted(self.results.items()):
            print(f"  {icon.get(status,'?')}  Group {grp}  [{status.upper()}]")
            if status == "fail":
                n_fail += 1

        if self._errors:
            print("\n  Errors:")
            for e in self._errors:
                print(f"    {e}")

        print(f"\n  Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(_sep)
        return 1 if n_fail else 0


# ─────────────────────────────────────────────────────────────────────────────
# Per-group helpers
# ─────────────────────────────────────────────────────────────────────────────

def _first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_benchmark_json(root: Path) -> Path | None:
    """Auto-detect the most-recent DeFi benchmark JSON.

    run_all.sh (exp1) writes timestamped files directly to RESULTS_DIR:
      hypatiax/data/results/hypatiax_defi_benchmark_v3*results*.json
    The extrap step writes to comparison_results/extrapolation/:
      all_domains_extrap_v4_*.json
    Legacy / explicit paths are also checked for backwards compatibility.
    """
    res = root / "hypatiax/data/results"
    candidates: list[Path] = [
        # explicit v3c2 name
        res / "hypatiax_defi_benchmark_v3c2_results.json",
        # timestamped multi-run files written by exp1 to RESULTS_DIR root
        *sorted(res.glob("hypatiax_defi_benchmark_v3*results*.json"), reverse=True),
        # extrap step
        *sorted((res / "comparison_results/extrapolation").glob(
            "all_domains_extrap_v4_*.json"), reverse=True),
        res / "extrapolation/extrapolation_73cases_enhanced.json",
        res / "extrapolation/full_run_20260227_231742.json",
        res / "extrapolation/hybrid_defi_20260227_222427.json",
        # hybrid_defi under results root (matches analyze_hybrid_performance loader)
        *sorted(res.glob("hybrid_defi_*.json"), reverse=True),
    ]
    return _first_existing(candidates)


def _find_merged_json(root: Path) -> Path | None:
    """Auto-detect all_systems_merged.json."""
    candidates = [
        root / "hypatiax/experiments/comparison/all_systems_merged.json",
        root / "all_systems_merged.json",
        root / "hypatiax/data/results/all_systems_merged.json",
    ]
    return _first_existing(candidates)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate ALL HypatiaX paper figures (master orchestrator).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Groups
        ------
          A  Instability analysis (5 seaborn figures)        hypatiax_instability_analysis_pipeline.py
          B  3D instability plots (6 figures incl. surface)  hypatiax_3D_plot_instability.py
          C  Extrapolation scatter (II vs extrap R²)         build_extrapolation_pipeline_final.py
          D  Comparative visualisations (box/bar/scatter)    create_visualizations.py
          E  Legacy results plot                             plot_results.py
          F  R² heatmaps clipped+raw (Fig 3 & 4)            plot_r2_heatmaps.py
          G  Portfolio Variance seed-sweep bar chart (Fig 5) plot_portfolio_seed_sweep.py
        """),
    )

    parser.add_argument(
        "--repo-root", type=Path, default=None,
        metavar="PATH",
        help="Root of the hypatiax repository (auto-detected if omitted).",
    )
    parser.add_argument(
        "--groups", nargs="+", choices=list("ABCDEFG"), default=list("ABCDEFG"),
        metavar="GROUP",
        help="Which groups to run (default: A B C D E F G).",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=None,
        metavar="PATH",
        help="Override output directory for figures (default: hypatiax/data/figures).",
    )
    parser.add_argument(
        "--format", nargs="+", choices=["png", "pdf", "svg"],
        default=["png", "pdf"], metavar="FMT",
        help="Output format(s) for figures (default: png pdf).",
    )
    parser.add_argument(
        "--source", choices=["auto", "variance", "multi", "single"],
        default="auto",
        help="JSON source for instability groups A & B (default: auto).",
    )
    parser.add_argument(
        "--benchmark-json", type=Path, default=None,
        metavar="PATH",
        help="Path to DeFi benchmark JSON for group C (auto-detected if omitted).",
    )
    parser.add_argument(
        "--instability-csv", type=Path, default=None,
        metavar="PATH",
        help="Path to instability_analysis.csv for group C "
             "(default: <figures-dir>/instability_analysis.csv — produced by group A).",
    )
    parser.add_argument(
        "--merged-json", type=Path, default=None,
        metavar="PATH",
        help="Path to all_systems_merged.json for group D (auto-detected if omitted).",
    )
    parser.add_argument(
        "--elev", type=float, default=28.0, metavar="DEG",
        help="3D surface elevation angle for group B (default: 28).",
    )
    parser.add_argument(
        "--azim", type=float, default=225.0, metavar="DEG",
        help="3D surface azimuth angle for group B (default: 225).",
    )
    parser.add_argument(
        "--ablation-json", type=Path, default=None,
        metavar="PATH",
        help="Path to Core-15 ablation JSON for group F heatmaps (auto-detected if omitted).",
    )
    parser.add_argument(
        "--seed-sweep-json", type=Path, default=None,
        metavar="PATH",
        help="Path to portfolio_variance_seed_sweep.json for group G (auto-detected if omitted).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Pass --verbose to sub-scripts where supported.",
    )
    parser.add_argument(
        "--no-regline", action="store_true",
        help="Omit regression line on group A complexity-vs-instability figure.",
    )

    args = parser.parse_args()

    # ── Resolve repo root ─────────────────────────────────────────────────────
    if args.repo_root:
        repo = args.repo_root.resolve()
    else:
        repo = _find_repo_root(Path(__file__).resolve().parent)
    print(f"\n  Repo root : {repo}")

    # Honour RESULTS_DIR env-var override (mirrors run_all.sh behaviour)
    import os as _os
    results_dir: Path = Path(
        _os.environ.get("RESULTS_DIR", str(repo / "hypatiax" / "data" / "results"))
    ).resolve()

    # ── Script paths (all live in hypatiax/tools/visualizations/) ─────────────
    VIS = repo / "hypatiax" / "tools" / "visualizations"
    SCRIPT = {
        "A": VIS / "hypatiax_instability_analysis_pipeline.py",
        "B": VIS / "hypatiax_3D_plot_instability.py",
        "C": VIS / "build_extrapolation_pipeline_final.py",
        "D": VIS / "create_visualizations.py",
        "E": VIS / "plot_results.py",
        "F": VIS / "plot_r2_heatmaps.py",
        "G": VIS / "plot_portfolio_seed_sweep.py",
    }

    # ── Output directory ──────────────────────────────────────────────────────
    figures_dir: Path = args.figures_dir or (results_dir / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Results   : {results_dir}")
    print(f"  Figures   : {figures_dir}")

    # ── Auto-detect optional inputs ───────────────────────────────────────────
    benchmark_json: Path | None = args.benchmark_json or _find_benchmark_json(repo)
    merged_json:    Path | None = args.merged_json    or _find_merged_json(repo)
    instability_csv: Path = (
        args.instability_csv
        or (figures_dir / "instability_analysis.csv")
    )

    # ── Auto-detect ablation and seed-sweep JSONs for groups F & G ─────────────
    def _find_ablation_json(root: Path) -> Path | None:
        abl_dir = results_dir / "exp1_ablation"
        extra = sorted(abl_dir.glob("*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True) \
                if abl_dir.exists() else []
        candidates = [
            results_dir / "exp1_ablation/core15_ablation_results.json",
            *extra,
        ]
        return _first_existing([c for c in candidates if isinstance(c, Path)])

    def _find_seed_sweep_json(root: Path) -> Path | None:
        extra = sorted(results_dir.glob("portfolio_variance*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True) \
                if results_dir.exists() else []
        candidates = [
            results_dir / "portfolio_variance_seed_sweep.json",
            root / "portfolio_variance_seed_sweep.json",
            *extra,
        ]
        return _first_existing([c for c in candidates if isinstance(c, Path)])

    ablation_json:   Path | None = args.ablation_json   or _find_ablation_json(repo)
    seed_sweep_json: Path | None = args.seed_sweep_json or _find_seed_sweep_json(repo)

    # ── Missing data / script audit ────────────────────────────────────────────
    print("\n  ── Missing data/script audit ────────────────────────────────")
    _AUDIT_ITEMS = [
        ("A+B: instability JSON source",
         None,   # scripts auto-detect in RESULTS_DIR — just note the dir
         f"Scripts will search {results_dir} for variance/multi/single JSON"),
        ("C: benchmark JSON (DeFi / extrap)",  benchmark_json,
         f"Searched {results_dir} — groups C will be SKIPPED"),
        ("C: instability_analysis.csv",        instability_csv if instability_csv.exists() else None,
         f"Expected {instability_csv} — group C runs extrapolation-only mode (no II axis)"),
        ("D: all_systems_merged.json",         merged_json,
         f"Searched repo — group D will be SKIPPED"),
        ("E: plot_results.py script",          SCRIPT["E"] if SCRIPT["E"].exists() else None,
         f"Not found at {SCRIPT['E']} — group E will be SKIPPED"),
        ("F: Core-15 ablation JSON",           ablation_json,
         f"Searched {results_dir}/exp1_ablation/ — group F will be SKIPPED"),
        ("G: portfolio_variance_seed_sweep.json", seed_sweep_json,
         f"Searched {results_dir} — group G will be SKIPPED"),
    ]
    _audit_missing: list[str] = []
    _audit_ok:      list[str] = []
    for label, path_or_none, missing_msg in _AUDIT_ITEMS:
        if path_or_none:
            _audit_ok.append(f"    ✅ {label}: {path_or_none}")
        else:
            _audit_missing.append(f"    ❌ {label}\n       {missing_msg}")

    # also check each vis script
    for grp in args.groups:
        s = SCRIPT[grp]
        if not s.exists():
            _audit_missing.append(f"    ❌ Group {grp} script missing: {s}")

    if _audit_ok:
        print(f"\n  Found ({len(_audit_ok)}):")
        for m in _audit_ok:
            print(m)
    if _audit_missing:
        print(f"\n  ⚠  Missing ({len(_audit_missing)}):")
        for m in _audit_missing:
            print(m)
    else:
        print("\n  All inputs present.")
    print()
    # ── End audit ─────────────────────────────────────────────────────────────

    # ── Runner ────────────────────────────────────────────────────────────────
    runner = _Runner(dry_run=args.dry_run, verbose=args.verbose)
    py     = sys.executable          # same interpreter as the caller
    fmt    = args.format             # list of format strings
    groups = args.groups

    # ═════════════════════════════════════════════════════════════════════════
    # GROUP A — hypatiax_instability_analysis_pipeline.py
    #   Reads:  variance / multi-run / single JSON (auto-detected by the script)
    #   Writes: 5 × (png + pdf) + instability_analysis.csv
    # ═════════════════════════════════════════════════════════════════════════
    if "A" in groups:
        script_a = SCRIPT["A"]
        skip_a   = None if script_a.exists() else f"script not found: {script_a}"
        cmd_a    = [
            py, str(script_a),
            "--source",      args.source,
            "--results-dir", str(results_dir),
            "--out",         str(figures_dir),
            "--csv-out",     str(instability_csv),
            "--format",      *fmt,
        ]
        if args.no_regline:
            cmd_a.append("--no-regline")
        runner.run("A", "Instability analysis pipeline (5 figs + CSV)", cmd_a,
                   cwd=repo, skip_reason=skip_a)

    # ═════════════════════════════════════════════════════════════════════════
    # GROUP B — hypatiax_3D_plot_instability.py
    #   Reads:  same JSON sources as group A
    #   Writes: 6 × (png + pdf) including 3D surface
    # ═════════════════════════════════════════════════════════════════════════
    if "B" in groups:
        script_b = SCRIPT["B"]
        skip_b   = None if script_b.exists() else f"script not found: {script_b}"
        cmd_b    = [
            py, str(script_b),
            "--source",      args.source,
            "--results-dir", str(results_dir),
            "--out",         str(figures_dir),
            "--format",      *fmt,
            "--elev",        str(args.elev),
            "--azim",        str(args.azim),
        ]
        runner.run("B", "3D instability plots (6 figs incl. surface)", cmd_b,
                   cwd=repo, skip_reason=skip_b)

    # ═════════════════════════════════════════════════════════════════════════
    # GROUP C — build_extrapolation_pipeline_final.py
    #   Reads:  benchmark JSON  +  instability_analysis.csv (from group A)
    #   Writes: instability_extrapolation.csv
    #           fig_instability_vs_extrapolation.png
    #
    #   NOTE: group A should run before C so instability_analysis.csv exists.
    #         If C runs standalone and the CSV is absent, the script degrades
    #         gracefully to extrapolation-only mode (no II axis).
    # ═════════════════════════════════════════════════════════════════════════
    if "C" in groups:
        script_c = SCRIPT["C"]
        skip_c: str | None = None
        if not script_c.exists():
            skip_c = f"script not found: {script_c}"
        elif benchmark_json is None:
            skip_c = (
                "No DeFi benchmark JSON found.  "
                "Run the benchmark first or pass --benchmark-json."
            )
        cmd_c = [
            py, str(script_c),
            "--input",           str(benchmark_json) if benchmark_json else "",
            "--output",          str(figures_dir / "instability_extrapolation.csv"),
            "--plot",            str(figures_dir / "fig_instability_vs_extrapolation.png"),
        ]
        # Attach instability CSV only when it exists (group A may have just
        # produced it; this handles both sequential and parallel invocations).
        if instability_csv.exists():
            cmd_c += ["--instability-csv", str(instability_csv)]
        else:
            print(
                f"  [C] NOTE: instability_analysis.csv not found at {instability_csv}.\n"
                "       Group C will run in extrapolation-only mode (no II axis).\n"
                "       Run group A first for the full scatter."
            )

        runner.run("C", "Extrapolation pipeline (II vs extrap R² scatter)", cmd_c,
                   cwd=repo, skip_reason=skip_c)

    # ═════════════════════════════════════════════════════════════════════════
    # GROUP D — create_visualizations.py
    #   Reads:  all_systems_merged.json
    #   Writes: figure_boxplot_comparison.{pdf,png}
    #           figure_barplot_comparison.{pdf,png}
    #           figure_scatter_r2_vs_extrap.{pdf,png}
    #
    #   create_visualizations.py hard-codes Path.cwd() / "all_systems_merged.json"
    #   and Path.cwd() / "figures".  We therefore run it from the directory
    #   that contains all_systems_merged.json (or from the repo root if not found,
    #   so the script prints its own "file not found" diagnostic).
    # ═════════════════════════════════════════════════════════════════════════
    if "D" in groups:
        script_d = SCRIPT["D"]
        skip_d: str | None = None
        if not script_d.exists():
            skip_d = f"script not found: {script_d}"
        elif merged_json is None:
            skip_d = (
                "all_systems_merged.json not found.  "
                "Run experiments/comparison/merge_all_systems.py first, "
                "or pass --merged-json."
            )
        # create_visualizations.py uses Path.cwd() for both its input and output,
        # so cwd must be the directory that holds all_systems_merged.json.
        # We symlink / copy nothing — instead we patch the cwd so the script
        # finds its file naturally.
        cwd_d = merged_json.parent if merged_json is not None else repo
        cmd_d = [py, str(script_d)]
        runner.run(
            "D",
            "Comparative visualisations (boxplot + barplot + scatter)",
            cmd_d,
            cwd=cwd_d,
            skip_reason=skip_d,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # GROUP E — plot_results.py
    #   Reads:  data/all_systems_merged.json  (relative to script's parent^2)
    #   Writes: figures/results.pdf
    # ═════════════════════════════════════════════════════════════════════════
    if "E" in groups:
        script_e = SCRIPT["E"]
        skip_e   = None if script_e.exists() else f"script not found: {script_e}"
        cmd_e    = [py, str(script_e)]
        # plot_results.py uses Path(__file__).parent.parent / "data" / ...
        # so running from repo root or the script's own location is equivalent.
        runner.run("E", "Legacy results plot (figures/results.pdf)", cmd_e,
                   cwd=repo, skip_reason=skip_e)

    # ═════════════════════════════════════════════════════════════════════════
    # GROUP F — plot_r2_heatmaps.py   (Fig 3 & Fig 4 in paper)
    #   Reads:  Core-15 ablation JSON (per-equation Train/Near/Med/Far R² for
    #           PySR-only and HypatiaX)
    #   Writes: fig_r2_heatmap_clipped.{png,pdf}   ← Fig 3  [−1,1] clip
    #           fig_r2_heatmap_raw.{png,pdf}        ← Fig 4  raw / unclipped
    # ═════════════════════════════════════════════════════════════════════════
    if "F" in groups:
        script_f = SCRIPT["F"]
        skip_f: str | None = None
        if not script_f.exists():
            skip_f = f"script not found: {script_f}"
        elif ablation_json is None:
            skip_f = (
                "No Core-15 ablation JSON found.  "
                "Run the ablation experiment first or pass --ablation-json."
            )
        cmd_f = [
            py, str(script_f),
            "--input",      str(ablation_json) if ablation_json else "",
            "--out",        str(figures_dir),
            "--format",     *fmt,
        ]
        runner.run(
            "F",
            "R² heatmaps clipped+raw — Fig 3 & Fig 4 (§10.1/§10.2)",
            cmd_f,
            cwd=repo,
            skip_reason=skip_f,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # GROUP G — plot_portfolio_seed_sweep.py   (Fig 5 in paper)
    #   Reads:  portfolio_variance_seed_sweep.json
    #           (5 seeds × Near/Med/Far R² for PySR-only and HypatiaX)
    #   Writes: fig_portfolio_seed_sweep.{png,pdf}
    # ═════════════════════════════════════════════════════════════════════════
    if "G" in groups:
        script_g = SCRIPT["G"]
        skip_g: str | None = None
        if not script_g.exists():
            skip_g = f"script not found: {script_g}"
        elif seed_sweep_json is None:
            skip_g = (
                "portfolio_variance_seed_sweep.json not found.  "
                "Run the seed-sweep experiment first or pass --seed-sweep-json."
            )
        cmd_g = [
            py, str(script_g),
            "--input",  str(seed_sweep_json) if seed_sweep_json else "",
            "--out",    str(figures_dir),
            "--format", *fmt,
        ]
        runner.run(
            "G",
            "Portfolio Variance seed-sweep bar chart — Fig 5 (§10.5)",
            cmd_g,
            cwd=repo,
            skip_reason=skip_g,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    rc = runner.print_summary()

    # Print a flat inventory of expected output files for quick verification
    print("\n  Expected output files in", figures_dir, ":")
    expected = [
        # Group A
        "fig_paper_complexity_vs_instability.{png,pdf}",
        "fig_paper_complexity_vs_success.{png,pdf}",
        "fig_paper_mean_vs_instability.{png,pdf}",
        "fig_paper_instability_hist.{png,pdf}",
        "fig_paper_regime_counts.{png,pdf}",
        "instability_analysis.csv",
        # Group B
        "fig_instability_3d.{png,pdf}",
        "fig_instability_phase.{png,pdf}",
        "fig_instability_hist.{png,pdf}",
        "fig_instability_success_vs_instability.{png,pdf}",
        "fig_instability_regimes.{png,pdf}",
        "fig_instability_surface.{png,pdf}",
        # Group C
        "fig_instability_vs_extrapolation.png",
        "instability_extrapolation.csv",
        # Group D  (written to cwd of create_visualizations.py / figures/)
        "figure_boxplot_comparison.{pdf,png}",
        "figure_barplot_comparison.{pdf,png}",
        "figure_scatter_r2_vs_extrap.{pdf,png}",
        # Group E  (written to <repo>/figures/)
        "results.pdf",
        # Group F  (Fig 3 & Fig 4 in paper)
        "fig_r2_heatmap_clipped.{png,pdf}",
        "fig_r2_heatmap_raw.{png,pdf}",
        # Group G  (Fig 5 in paper)
        "fig_portfolio_seed_sweep.{png,pdf}",
    ]
    for stem in expected:
        print(f"    {stem}")

    return rc


if __name__ == "__main__":
    sys.exit(main())

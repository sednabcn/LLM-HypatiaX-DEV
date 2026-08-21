#!/usr/bin/env python3
"""
plot_pysr_trajectories.py

Plot the PySR outer-iteration trajectories saved by
exp3_nguyen12_hybrid50v_04.py.

The experiment JSON has the following relevant structure:

{
  "config": {"seed": 42, ...},
  "results": {
    "hypatiax": [
      {
        "metadata": {"nguyen_id": "N1", ...},
        "trajectory": [
          {
            "iteration": 1,
            "elapsed_seconds": ...,
            "best_loss": ...,
            "best_expression": ...,
            "best_complexity": ...,
            "best_score": ...
          },
          ...
        ],
        "trajectory_summary": {...}
      },
      ...
    ],
    "pysr": [
      {
        "metadata": {"nguyen_id": "N1", ...},
        "trajectory": [...],
        "trajectory_summary": {...}
      },
      ...
    ]
  }
}

This script deliberately calls the x-axis "observed outer iteration", not
"generation": the experiment script polls PySR's Hall-of-Fame checkpoint and
does not expose every individual mutation/generation.

Examples
--------
Single seed:

    python plot_pysr_trajectories.py exp3_nguyen12_seed42.json

Multiple seeds:

    python plot_pysr_trajectories.py results/exp3_nguyen12_seed*.json

Output:

    trajectories/
      seed42/
        N1_trajectory.png
        N2_trajectory.png
        ...
        seed42_overview.png
        seed42_summary.csv
      seed123/
        ...

Useful options:

    --metric loss
    --metric both
    --include-expressions
    --output-dir trajectories
    --dpi 180
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def seed_from_payload(payload: Dict[str, Any], path: Path) -> Any:
    seed = payload.get("config", {}).get("seed")
    if seed is not None:
        return seed

    # Fallback for files named exp3_nguyen12_seed42.json.
    stem = path.stem
    marker = "seed"
    if marker in stem:
        tail = stem.split(marker, 1)[1]
        digits = ""
        for ch in tail:
            if ch.isdigit() or (ch == "-" and not digits):
                digits += ch
            else:
                break
        if digits:
            try:
                return int(digits)
            except ValueError:
                pass
    return "unknown"


def finite_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def clean_trajectory(rows: Any) -> List[Dict[str, Any]]:
    """Keep usable trajectory observations and sort by iteration."""
    if not isinstance(rows, list):
        return []

    cleaned = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        iteration = finite_float(row.get("iteration"))
        elapsed = finite_float(row.get("elapsed_seconds"))
        loss = finite_float(row.get("best_loss"))

        if iteration is None or loss is None:
            continue

        item = dict(row)
        item["_iteration"] = iteration
        item["_elapsed"] = elapsed
        item["_loss"] = loss
        cleaned.append(item)

    cleaned.sort(key=lambda r: r["_iteration"])
    return cleaned


def equation_id(h_record: Dict[str, Any], p_record: Optional[Dict[str, Any]] = None) -> str:
    metadata = h_record.get("metadata", {})
    nguyen_id = metadata.get("nguyen_id")
    if nguyen_id:
        return str(nguyen_id)

    if p_record:
        metadata = p_record.get("metadata", {})
        nguyen_id = metadata.get("nguyen_id")
        if nguyen_id:
            return str(nguyen_id)

    return "unknown"


def index_records(
    records: Any,
) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    if not isinstance(records, list):
        return indexed

    for record in records:
        if not isinstance(record, dict):
            continue
        metadata = record.get("metadata", {})
        key = metadata.get("nguyen_id")
        if key is None:
            key = metadata.get("id") or metadata.get("name")
        if key is None:
            # Preserve records with missing IDs instead of silently dropping them.
            key = f"unknown_{len(indexed) + 1}"
        indexed[str(key)] = record
    return indexed


def safe_filename(value: str) -> str:
    chars = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars)


def plot_equation(
    seed: Any,
    nguyen_id: str,
    h_record: Optional[Dict[str, Any]],
    p_record: Optional[Dict[str, Any]],
    output_path: Path,
    metric: str,
    include_expressions: bool,
    dpi: int,
) -> None:
    h_traj = clean_trajectory((h_record or {}).get("trajectory", []))
    p_traj = clean_trajectory((p_record or {}).get("trajectory", []))

    if not h_traj and not p_traj:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))

    if h_traj:
        ax.plot(
            [r["_iteration"] for r in h_traj],
            [r["_loss"] for r in h_traj],
            marker="o",
            markersize=2.5,
            linewidth=1.4,
            label="H / LLM warm-start",
        )

    if p_traj:
        ax.plot(
            [r["_iteration"] for r in p_traj],
            [r["_loss"] for r in p_traj],
            marker="o",
            markersize=2.5,
            linewidth=1.4,
            label="P / cold PySR",
        )

    ax.set_xlabel("Observed outer iteration")
    ax.set_ylabel("Best loss (MSE/error)")
    ax.set_title(f"Nguyen {nguyen_id} — seed {seed}")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    # Mark the first observed threshold crossing using the summary already
    # written by the experiment. We do not reconstruct R² here because the
    # JSON does not store Var(y), which is required for exact R² = 1 - MSE/Var(y).
    for label, record, linestyle in (
        ("H", h_record, "--"),
        ("P", p_record, ":"),
    ):
        if not record:
            continue
        summary = record.get("trajectory_summary", {})
        threshold_iteration = summary.get("first_threshold_iteration")
        if threshold_iteration is not None:
            ax.axvline(
                float(threshold_iteration),
                linestyle=linestyle,
                linewidth=1.0,
                alpha=0.7,
                label=f"{label}: first R²≥0.9999 observed",
            )

    if include_expressions:
        # Annotate only the last expression from each trajectory to keep the
        # figure readable. The full history remains available in JSON.
        annotations = []
        for label, traj in (("H", h_traj), ("P", p_traj)):
            if traj:
                expr = str(traj[-1].get("best_expression") or "").strip()
                if expr:
                    annotations.append(f"{label}: {expr}")

        if annotations:
            ax.text(
                0.01,
                0.01,
                "\n".join(annotations),
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=8,
                wrap=True,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_overview(
    seed: Any,
    h_records: Dict[str, Dict[str, Any]],
    p_records: Dict[str, Dict[str, Any]],
    output_path: Path,
    dpi: int,
) -> None:
    """One compact figure containing all equations for one seed."""
    ids = sorted(set(h_records) | set(p_records))
    if not ids:
        return

    fig, ax = plt.subplots(figsize=(11, 7))

    for nguyen_id in ids:
        h_traj = clean_trajectory(h_records.get(nguyen_id, {}).get("trajectory", []))
        p_traj = clean_trajectory(p_records.get(nguyen_id, {}).get("trajectory", []))

        # Use thin lines in the overview; individual equation figures are
        # provided for detailed inspection.
        if h_traj:
            ax.plot(
                [r["_iteration"] for r in h_traj],
                [r["_loss"] for r in h_traj],
                linewidth=0.9,
                alpha=0.75,
                label=f"{nguyen_id} H",
            )
        if p_traj:
            ax.plot(
                [r["_iteration"] for r in p_traj],
                [r["_loss"] for r in p_traj],
                linewidth=0.9,
                alpha=0.45,
                linestyle="--",
                label=f"{nguyen_id} P",
            )

    ax.set_xlabel("Observed outer iteration")
    ax.set_ylabel("Best loss (MSE/error)")
    ax.set_title(f"PySR trajectory overview — seed {seed}")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.2)

    # Put a legend outside because up to 24 curves may be present.
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=7,
        ncol=1,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_summary_rows(
    seed: Any,
    h_records: Dict[str, Dict[str, Any]],
    p_records: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []

    for nguyen_id in sorted(set(h_records) | set(p_records)):
        h = h_records.get(nguyen_id, {})
        p = p_records.get(nguyen_id, {})

        hs = h.get("trajectory_summary", {}) or {}
        ps = p.get("trajectory_summary", {}) or {}

        h_traj = clean_trajectory(h.get("trajectory", []))
        p_traj = clean_trajectory(p.get("trajectory", []))

        rows.append(
            {
                "seed": seed,
                "nguyen_id": nguyen_id,
                "h_observed_iterations": len(h_traj),
                "p_observed_iterations": len(p_traj),
                "h_first_threshold_iteration": hs.get("first_threshold_iteration"),
                "p_first_threshold_iteration": ps.get("first_threshold_iteration"),
                "h_first_threshold_time_seconds": hs.get("first_threshold_time_seconds"),
                "p_first_threshold_time_seconds": ps.get("first_threshold_time_seconds"),
                "h_final_best_loss": hs.get("final_best_loss"),
                "p_final_best_loss": ps.get("final_best_loss"),
                "h_final_expression": hs.get("final_best_expression"),
                "p_final_expression": ps.get("final_best_expression"),
                "same_final_expression": h.get("same_final_expression_as_p"),
                "same_final_r2": h.get("same_final_r2_as_p"),
                "warm_start_status": h.get("warm_start_status"),
                "effective_method": h.get("effective_method"),
                "h_independent_fit": h.get("independent_fit"),
            }
        )

    return rows


def write_summary_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def process_json(
    json_path: Path,
    output_root: Path,
    metric: str,
    include_expressions: bool,
    dpi: int,
) -> None:
    payload = load_json(json_path)
    seed = seed_from_payload(payload, json_path)

    results = payload.get("results", {})
    h_records = index_records(results.get("hypatiax", []))
    p_records = index_records(results.get("pysr", []))

    seed_dir = output_root / f"seed{safe_filename(str(seed))}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    equation_ids = sorted(set(h_records) | set(p_records))

    n_plotted = 0
    for nguyen_id in equation_ids:
        h = h_records.get(nguyen_id)
        p = p_records.get(nguyen_id)

        if not clean_trajectory((h or {}).get("trajectory", [])) and not clean_trajectory(
            (p or {}).get("trajectory", [])
        ):
            continue

        out = seed_dir / f"{safe_filename(nguyen_id)}_trajectory.png"
        plot_equation(
            seed,
            nguyen_id,
            h,
            p,
            out,
            metric=metric,
            include_expressions=include_expressions,
            dpi=dpi,
        )
        n_plotted += 1

    overview = seed_dir / f"seed{safe_filename(str(seed))}_overview.png"
    plot_overview(seed, h_records, p_records, overview, dpi)

    rows = make_summary_rows(seed, h_records, p_records)
    write_summary_csv(rows, seed_dir / f"seed{safe_filename(str(seed))}_summary.csv")

    print(
        f"seed={seed}: {n_plotted} equation trajectories written to {seed_dir}"
    )


def expand_inputs(inputs: Sequence[str]) -> List[Path]:
    """Expand files and shell-style glob patterns without requiring a shell."""
    paths: List[Path] = []

    for item in inputs:
        p = Path(item)
        if p.exists() and p.is_file():
            paths.append(p)
            continue

        # Path.glob handles patterns such as results/*.json.
        parent = p.parent if str(p.parent) else Path(".")
        matches = sorted(parent.glob(p.name))
        paths.extend(q for q in matches if q.is_file())

    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot PySR outer-iteration trajectories from experiment JSON files."
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        help="One or more result JSON files, or glob patterns.",
    )
    parser.add_argument(
        "--output-dir",
        default="trajectories",
        help="Directory for generated plots and summaries (default: trajectories).",
    )
    parser.add_argument(
        "--metric",
        choices=("loss", "both"),
        default="loss",
        help="Currently loss is the primary trajectory metric. 'both' is accepted for compatibility.",
    )
    parser.add_argument(
        "--include-expressions",
        action="store_true",
        help="Annotate each figure with the final H/P expressions.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution (default: 180).",
    )
    args = parser.parse_args()

    files = expand_inputs(args.json_files)
    if not files:
        raise SystemExit("No JSON files found.")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for path in files:
        try:
            process_json(
                path,
                output_root,
                metric=args.metric,
                include_expressions=args.include_expressions,
                dpi=args.dpi,
            )
        except Exception as exc:
            print(f"ERROR processing {path}: {exc}")

    print(f"Done. Output directory: {output_root.resolve()}")


if __name__ == "__main__":
    main()

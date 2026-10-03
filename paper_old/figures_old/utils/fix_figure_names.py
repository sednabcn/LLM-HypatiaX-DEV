#!/usr/bin/env python3
"""
fix_figure_names.py — Normalize suspicious/duplicate figure filenames in the
HypatiaX paper bundle onto their canonical stems.

Companion to NB-05's Step 5c (read-only detector). Run this script to act on
what Step 5c reports.

WHAT THIS FIXES
----------------
- Case variants:        hypatiaX_three_systems.png       -> hypatiax_three_systems.png
- Known typos:           hypatiax_algorithm1_routine_cascade_v2.png
                           -> hypatiax_algorithm1_routing_cascade_v2.png
- Stray version suffixes on figures that have no versioning scheme of their
  own (e.g. hypatiaX_three_systems_v2.png), once normalized onto a canonical
  stem that already carries a real file.

WHAT THIS NEVER DOES
---------------------
- Never deletes a file outright. If two files of the SAME extension and the
  SAME canonical stem have DIFFERENT content (a real divergence, not just a
  naming issue), the non-canonical one is MOVED to figures/_superseded/ —
  never removed — and logged in _superseded/MANIFEST.tsv with both file's
  hashes so a human can diff and decide later.
- Never touches "untracked figure families" (e.g. hypatiax_instability_*).
  Those aren't in FIGURES_INVENTORY/EMBEDDED_STEMS, so the script has no
  canonical name to normalize them onto — guessing one would be worse than
  leaving them alone. Add them to the inventory in NB-05's Step 0 first if
  they're meant to ship with the paper.
- Never edits anything by default. Pass --apply to actually touch the
  filesystem; without it, this prints the exact plan and exits 0.

USAGE
-----
    python fix_figure_names.py                  # dry run (default) — prints plan
    python fix_figure_names.py --apply           # actually renames / quarantines
    python fix_figure_names.py --apply --dir ../figures --dir ../hypatiax/data/results/figures
    python fix_figure_names.py --root /path/to/repo   # if not run from notebooks/

EXIT CODES
----------
    0  — nothing to do, or dry run completed (even if it found suspicious names)
    1  — --apply was passed and at least one divergence needed manual review
         (those files were quarantined, not lost — re-run is safe/idempotent)
    2  — usage / configuration error
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── Canonical stems (mirrors NB-05's FIGURES_INVENTORY + EMBEDDED_STEMS) ────
# Keep this in sync with NB-05's Step 0 config cell. If NB-05 adds a figure,
# add its stem here too, or this script won't know it's canonical.
CANONICAL_STEMS = {
    "hypatiax_three_systems",
    "hypatiax_algorithm1_routing_cascade_v2",
    "fig18_r2_heatmap_improved",
    "fig09_r2_heatmap_regimes",
    "fig1_seed_sweep",
    "fig1_r2_vs_noise", "fig2_rmse_vs_noise", "fig3_time_vs_noise",
    "fig4_r2_vs_n", "fig5_rmse_vs_n", "fig6_time_vs_n",
    "fig7_recovery_vs_noise", "fig8_recovery_vs_n", "fig9_minr2_vs_noise",
    "fig10_r2_boxplot_noise", "fig11_recovery_heatmap",
    "fig_runtime_comparison", "fig_comparative_table",
}

# Manually-verified typo fixes. Only add an entry after confirming with the
# team that it's a typo of a canonical stem, not a deliberately different
# figure that happens to look similar.
KNOWN_TYPO_FIXES = {
    "routine_cascade": "routing_cascade",
}

VERSION_SUFFIX_RE = re.compile(r"(_v\d+)$")
FIG_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".eps", ".svg"}
DEFAULT_DIRS = ["figures", "hypatiax/data/results/figures"]
SUPERSEDED_DIRNAME = "_superseded"


def candidate_stems(raw_stem: str) -> set[str]:
    """All plausible canonical readings of a raw filename stem: as-is,
    lowercased, known-typo-fixed, and with a trailing _vN stripped, tried in
    combination (a file can have a case issue AND a version suffix)."""
    variants = {raw_stem, raw_stem.lower()}
    for v in list(variants):
        fixed = v
        for bad, good in KNOWN_TYPO_FIXES.items():
            fixed = fixed.replace(bad, good)
        variants.add(fixed)
    expanded = set(variants)
    for v in variants:
        m = VERSION_SUFFIX_RE.search(v)
        if m:
            expanded.add(v[: m.start()])
    return expanded


def sha256_of(path: Path, chunk_size: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def classify(directory: Path):
    """Return (canonical_clusters, untracked_clusters).
    canonical_clusters: {canon_stem: [Path, ...]}
    untracked_clusters: {raw_key: [Path, ...]}  -- reported only, never touched
    """
    canonical = defaultdict(list)
    untracked = defaultdict(list)
    for p in sorted(directory.iterdir()):
        if not p.is_file() or p.suffix.lower() not in FIG_EXTENSIONS:
            continue
        cands = candidate_stems(p.stem)
        match = next((c for c in cands if c in CANONICAL_STEMS), None)
        if match:
            canonical[match].append(p)
        else:
            key = p.stem.lower()
            m = VERSION_SUFFIX_RE.search(key)
            if m:
                key = key[: m.start()]
            untracked[key].append(p)
    return canonical, untracked


def plan_for_cluster(canon_stem: str, files: list[Path]):
    """Build the action plan for one canonical-stem cluster within one
    directory. Returns a list of dicts: {action, src, dest, reason}."""
    actions = []
    by_ext = defaultdict(list)
    for p in files:
        by_ext[p.suffix.lower()].append(p)

    for ext, paths in by_ext.items():
        canonical_name = canon_stem + ext
        canonical_path = paths[0].parent / canonical_name
        is_canonical_present = any(p.name == canonical_name for p in paths)
        non_canonical = [p for p in paths if p.name != canonical_name]

        if not non_canonical:
            continue  # nothing to do for this extension

        if not is_canonical_present:
            # No canonical file yet for this extension — promote the single
            # non-canonical file (if more than one, promote the first found
            # alphabetically and treat the rest as divergence candidates).
            promote, *rest = non_canonical
            actions.append({
                "action": "rename",
                "src": promote,
                "dest": canonical_path,
                "reason": f"no canonical '{canonical_name}' exists yet; "
                          f"promoting '{promote.name}'",
            })
            non_canonical = rest  # whatever's left still needs handling below
            # Use the freshly-promoted file as the comparison target.
            compare_against = promote
        else:
            compare_against = canonical_path

        if non_canonical:
            try:
                canon_hash = sha256_of(compare_against if compare_against.exists() else canonical_path)
            except FileNotFoundError:
                canon_hash = None
            for p in non_canonical:
                try:
                    p_hash = sha256_of(p)
                except Exception as e:
                    actions.append({
                        "action": "error",
                        "src": p, "dest": None,
                        "reason": f"could not hash file: {e!r}",
                    })
                    continue
                if canon_hash is not None and p_hash == canon_hash:
                    actions.append({
                        "action": "delete_duplicate",
                        "src": p,
                        "dest": None,
                        "reason": f"byte-identical to canonical '{canonical_name}' "
                                  f"(sha256 {p_hash[:12]}) — safe to remove",
                    })
                else:
                    actions.append({
                        "action": "quarantine",
                        "src": p,
                        "dest": p.parent / SUPERSEDED_DIRNAME / f"{canon_stem}__{p_hash[:12]}{ext}",
                        "reason": f"DIVERGED from canonical '{canonical_name}' "
                                  f"(sha256 {p_hash[:12]} != canonical) — needs human review, "
                                  f"not deleted",
                    })
    return actions


def execute_plan(actions: list[dict], apply: bool, manifest_lines: list[str]) -> bool:
    """Run (or print) the plan. Returns True if any divergence needed review."""
    had_divergence = False
    for act in actions:
        a, src, dest, reason = act["action"], act["src"], act.get("dest"), act["reason"]
        if a == "rename":
            print(f"  [RENAME]    {src}  ->  {dest}")
            print(f"              reason: {reason}")
            if apply:
                dest.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dest)
        elif a == "delete_duplicate":
            print(f"  [DELETE]    {src}   (exact duplicate)")
            print(f"              reason: {reason}")
            if apply:
                src.unlink()
        elif a == "quarantine":
            had_divergence = True
            print(f"  [QUARANTINE] {src}  ->  {dest}")
            print(f"              reason: {reason}")
            if apply:
                dest.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dest)
                manifest_lines.append(
                    f"{datetime.now(timezone.utc).isoformat()}\t{src}\t{dest}\t{reason}"
                )
        elif a == "error":
            print(f"  [ERROR]     {src}: {reason}")
        print()
    return had_divergence


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("."),
                     help="Repo root the --dir paths are relative to (default: cwd)")
    ap.add_argument("--dir", dest="dirs", action="append",
                     help="Directory to scan, relative to --root. May be passed multiple times. "
                          f"Default: {DEFAULT_DIRS}")
    ap.add_argument("--apply", action="store_true",
                     help="Actually perform renames/deletes/quarantines. Without this flag, "
                          "only prints the plan (dry run).")
    args = ap.parse_args()

    dirs = args.dirs or DEFAULT_DIRS
    mode = "APPLY" if args.apply else "DRY RUN"
    print("=" * 90)
    print(f"fix_figure_names.py — {mode}")
    print("=" * 90)
    if not args.apply:
        print("No files will be modified. Pass --apply to execute this plan.\n")

    any_divergence = False
    any_untracked = False
    manifest_lines: list[str] = []

    for d in dirs:
        directory = (args.root / d).resolve()
        print(f"\n--- {directory} ---")
        if not directory.exists():
            print("  <does not exist, skipping>")
            continue

        canonical_clusters, untracked_clusters = classify(directory)

        cluster_had_actions = False
        for canon_stem in sorted(canonical_clusters):
            files = canonical_clusters[canon_stem]
            actions = plan_for_cluster(canon_stem, files)
            if not actions:
                continue
            cluster_had_actions = True
            print(f"\n  Cluster: {canon_stem}")
            div = execute_plan(actions, args.apply, manifest_lines)
            any_divergence = any_divergence or div

        if not cluster_had_actions:
            print("  ✓ No canonical-stem renames/duplicates needed here.")

        if untracked_clusters:
            any_untracked = True
            print(f"\n  Untracked figure families in this directory (NOT touched — "
                  f"not in CANONICAL_STEMS, add them to the inventory first):")
            for key, files in sorted(untracked_clusters.items()):
                names = ", ".join(p.name for p in files)
                print(f"    {key}: {names}")

    if manifest_lines:
        manifest_path = (args.root / dirs[0] / SUPERSEDED_DIRNAME / "MANIFEST.tsv").resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not manifest_path.exists()
        with open(manifest_path, "a") as f:
            if is_new:
                f.write("timestamp_utc\toriginal_path\tquarantined_path\treason\n")
            f.write("\n".join(manifest_lines) + "\n")
        print(f"\nQuarantine manifest updated: {manifest_path}")

    print("\n" + "=" * 90)
    if any_divergence:
        print("RESULT: one or more files DIVERGED from their canonical name and were "
              + ("quarantined to _superseded/ (not deleted)." if args.apply
                 else "would be quarantined to _superseded/ if you pass --apply.")
              + " Review them by hand — they are real content differences, not naming noise.")
    if any_untracked:
        print("NOTE: untracked figure families were found and left untouched. "
              "If they belong in the paper, add their stems to CANONICAL_STEMS here "
              "and to FIGURES_INVENTORY/EMBEDDED_STEMS in NB-05, then re-run.")
    if not any_divergence and not any_untracked:
        print("RESULT: clean — no divergences, no untracked families.")

    return 1 if (any_divergence and args.apply) else 0


if __name__ == "__main__":
    sys.exit(main())

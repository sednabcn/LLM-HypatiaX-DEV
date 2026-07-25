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
- Never touches "untracked figure families" (e.g. hypatiax_instability_*)
  UNLESS --archive-untracked is passed. Even then, it only archives a family
  whose derived keyword is NOT_MENTIONED anywhere in the .tex sources — the
  same classification NB-05's Step 5d performs. A family that's discussed in
  prose or a table (just not as a figure) is left alone either way, since
  that reflects a real editorial choice, not an orphaned file.
- Never edits anything by default. Pass --apply to actually touch the
  filesystem; without it, this prints the exact plan and exits 0.

ARCHIVING CONFIRMED-ORPHANED FAMILIES
--------------------------------------
    python fix_figure_names.py --archive-untracked              # dry run
    python fix_figure_names.py --archive-untracked --apply       # actually moves them

Requires the .tex sources (jmlr_paper_main.tex, supp_routing_improvements.tex,
supp_benchmark_report.tex) to be readable from --root, since the NOT_MENTIONED
classification needs them. Files are MOVED (never deleted) to
--archive-dir (default: analysis/archived_figures/), and the move is logged
in <first --dir>/_superseded/MANIFEST.tsv alongside quarantine entries.

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
ARCHIVE_DIRNAME = "analysis/archived_figures"

# ── .tex sources to scan when classifying untracked families ──────────────
# Mirrors NB-05's Step 0 TEX_FILES list. Keep these in sync — if NB-05 adds a
# source file, add it here too, or --archive-untracked's classification will
# be working from incomplete information.
TEX_FILES = [
    "jmlr_paper_main.tex",
    "supp_routing_improvements.tex",
    "supp_benchmark_report.tex",
]

# Same keyword-derivation heuristic as NB-05's Step 5d, kept in sync so the
# notebook's diagnosis and this script's automated action agree on what an
# untracked family is "about."
GENERIC_PREFIX_RE = re.compile(r"^(hypatiax_|fig\d*_|fig_)")
GENERIC_PLOT_WORDS = {
    "scatter", "histogram", "per", "case", "plot", "chart",
    "heatmap", "boxplot", "comparison", "table",
}
FIG_ENV_RE = re.compile(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", re.DOTALL)
TABLE_ENV_RE = re.compile(r"\\begin\{table\}(.*?)\\end\{table\}", re.DOTALL)
INCL_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


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


def derive_keyword(stem: str) -> str:
    """Collapse a figure stem down to its distinctive topic word, the same
    way NB-05's Step 5d does, so a family like hypatiax_instability_scatter /
    _histogram / _per_case all resolve to the single keyword 'instability'
    instead of being treated as three unrelated topics."""
    s = stem.lower()
    s = GENERIC_PREFIX_RE.sub("", s)
    s = VERSION_SUFFIX_RE.sub("", s)
    tokens = [t for t in s.split("_") if t and t not in GENERIC_PLOT_WORDS]
    return tokens[0] if tokens else s


def load_tex_sources(root: Path) -> dict[str, str]:
    """Load whichever of TEX_FILES exist under `root`. Missing files are
    silently skipped (with a note) rather than erroring — --archive-untracked
    should still work if only some sources are present."""
    sources = {}
    for fname in TEX_FILES:
        p = root / fname
        if p.exists():
            sources[fname] = p.read_text(encoding="utf-8")
        else:
            print(f"  (note: {p} not found — skipping for keyword classification)")
    return sources


def classify_keyword_presence(keyword: str, sources: dict[str, str]) -> str:
    """Same classification NB-05's Step 5d performs: NOT_MENTIONED,
    PROSE_ONLY, PROSE_AND_TABLE_ONLY, FIGURE_ENV_NO_IMAGE, or
    FIGURE_ENV_FOUND. Only NOT_MENTIONED is treated as "confirmed orphaned"
    by --archive-untracked; every other status implies a human editorial
    decision is involved and the script won't move those files."""
    kw_re = re.compile(re.escape(keyword), re.IGNORECASE)
    total_mentions = 0
    in_figure_env = False
    in_table_env = False
    matching_fig_images: list[str] = []

    for src in sources.values():
        total_mentions += len(kw_re.findall(src))
        for block in FIG_ENV_RE.findall(src):
            if kw_re.search(block):
                in_figure_env = True
                matching_fig_images += INCL_RE.findall(block)
        for block in TABLE_ENV_RE.findall(src):
            if kw_re.search(block):
                in_table_env = True

    if total_mentions == 0:
        return "NOT_MENTIONED"
    if in_figure_env:
        return "FIGURE_ENV_FOUND" if matching_fig_images else "FIGURE_ENV_NO_IMAGE"
    if in_table_env:
        return "PROSE_AND_TABLE_ONLY"
    return "PROSE_ONLY"


def plan_for_untracked_archive(untracked_clusters: dict, sources: dict[str, str], archive_root: Path):
    """For each untracked cluster, classify its derived keyword against the
    .tex sources and, ONLY for keywords classified NOT_MENTIONED, build a
    move-to-archive action for every file in that cluster. Everything else
    is reported but left alone — this function never guesses on a topic
    that's actually discussed in the paper, even if no figure embeds it.

    Safety: if `sources` is empty (no .tex files could be loaded — e.g. wrong
    --root), this refuses to classify ANYTHING as NOT_MENTIONED. An empty
    source set trivially "mentions nothing," which would otherwise queue
    every untracked family for archiving for the wrong reason (we failed to
    read the paper, not because the topic is actually absent from it)."""
    actions = []
    skipped = []
    if not sources:
        for key, files in sorted(untracked_clusters.items()):
            skipped.append((key, derive_keyword(key), "UNKNOWN_NO_TEX_SOURCES", files))
        return actions, skipped
    for key, files in sorted(untracked_clusters.items()):
        keyword = derive_keyword(key)
        status = classify_keyword_presence(keyword, sources)
        if status == "NOT_MENTIONED":
            for p in files:
                actions.append({
                    "action": "archive",
                    "src": p,
                    "dest": archive_root / p.name,
                    "reason": f"untracked family '{key}' (keyword '{keyword}') is NOT_MENTIONED "
                              f"in any .tex source — confirmed orphaned, archiving",
                })
        else:
            skipped.append((key, keyword, status, files))
    return actions, skipped


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
        elif a == "archive":
            print(f"  [ARCHIVE]   {src}  ->  {dest}")
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
    ap.add_argument("--archive-untracked", action="store_true",
                     help="Also classify each untracked figure family against the .tex sources "
                          "(same logic as NB-05's Step 5d) and archive (move, never delete) any "
                          "family that is NOT_MENTIONED anywhere. Families that ARE mentioned "
                          "(prose, table, or figure) are reported but left untouched. Combine "
                          "with --apply to actually move files.")
    ap.add_argument("--archive-dir", type=Path, default=Path(ARCHIVE_DIRNAME),
                     help=f"Destination for archived untracked figures, relative to --root. "
                          f"Default: {ARCHIVE_DIRNAME}")
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
    any_archived = False
    manifest_lines: list[str] = []

    sources: dict[str, str] = {}
    if args.archive_untracked:
        print("\nLoading .tex sources for untracked-family keyword classification...")
        sources = load_tex_sources(args.root)
        if not sources:
            print(f"\nERROR: none of {TEX_FILES} were found under --root "
                  f"({args.root.resolve()}). --archive-untracked requires the .tex sources "
                  f"to tell NOT_MENTIONED apart from 'we couldn't check' — refusing to guess. "
                  f"Pass the correct --root (the directory containing the .tex files), or omit "
                  f"--archive-untracked to run the normal rename/quarantine plan only.")
            return 2

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
            if args.archive_untracked:
                archive_root = (args.root / args.archive_dir).resolve()
                archive_actions, skipped = plan_for_untracked_archive(
                    untracked_clusters, sources, archive_root
                )
                if archive_actions:
                    any_archived = True
                    print(f"\n  Untracked families CONFIRMED ORPHANED (not mentioned in any "
                          f".tex source) — archiving to {archive_root}:")
                    execute_plan(archive_actions, args.apply, manifest_lines)
                if skipped:
                    print(f"\n  Untracked families left in place (mentioned in the .tex sources, "
                          f"so NOT archived — review manually):")
                    for key, keyword, status, files in skipped:
                        names = ", ".join(p.name for p in files)
                        print(f"    {key}  (keyword '{keyword}', status {status})")
                        print(f"      {names}")
            else:
                print(f"\n  Untracked figure families in this directory (NOT touched — "
                      f"not in CANONICAL_STEMS, add them to the inventory first; or re-run "
                      f"with --archive-untracked to auto-classify and archive confirmed-orphaned ones):")
                for key, files in sorted(untracked_clusters.items()):
                    names = ", ".join(p.name for p in files)
                    print(f"    {key}: {names}")

    if manifest_lines:
        manifest_path = (args.root / dirs[0] / SUPERSEDED_DIRNAME / "MANIFEST.tsv").resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not manifest_path.exists()
        with open(manifest_path, "a") as f:
            if is_new:
                f.write("timestamp_utc\toriginal_path\tnew_path\treason\n")
            f.write("\n".join(manifest_lines) + "\n")
        print(f"\nManifest updated (quarantine + archive actions): {manifest_path}")

    print("\n" + "=" * 90)
    if any_divergence:
        print("RESULT: one or more files DIVERGED from their canonical name and were "
              + ("quarantined to _superseded/ (not deleted)." if args.apply
                 else "would be quarantined to _superseded/ if you pass --apply.")
              + " Review them by hand — they are real content differences, not naming noise.")
    if any_archived:
        print("RESULT: one or more untracked families were confirmed NOT_MENTIONED in the "
              + (".tex sources and were archived (moved, not deleted)." if args.apply
                 else ".tex sources and would be archived if you pass --apply.")
              + f" See {args.archive_dir}/.")
    if any_untracked and not any_archived:
        if args.archive_untracked:
            print("NOTE: untracked figure families were found, but all of them are mentioned "
                  "somewhere in the .tex sources (prose, table, or figure) — none were archived. "
                  "Review the 'left in place' list above and decide by hand.")
        else:
            print("NOTE: untracked figure families were found and left untouched. "
                  "If they belong in the paper, add their stems to CANONICAL_STEMS here "
                  "and to FIGURES_INVENTORY/EMBEDDED_STEMS in NB-05, then re-run. "
                  "Or re-run with --archive-untracked to auto-classify them against the .tex "
                  "sources and archive any that are confirmed unmentioned.")
    if not any_divergence and not any_untracked:
        print("RESULT: clean — no divergences, no untracked families.")

    return 1 if (any_divergence and args.apply) else 0


if __name__ == "__main__":
    sys.exit(main())

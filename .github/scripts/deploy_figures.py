#!/usr/bin/env python3
"""Assemble every experiment's canonical figures into one flat directory.

This is the standalone version of the "Deploy assembled figures" step in
ci_postprocess.yml (the `figures_deploy` aggregate). It is config-driven:
every experiment's source directory and which ones generate figures comes
from config/experiments.yml — nothing here hardcodes an experiment name.

Filename collision handling
----------------------------
Two different experiments can legitimately produce a file with the same
name (e.g. suppB and suppB_sc both write fig11_recovery_heatmap.pdf with
different content). There's no automatically "correct" winner in that case,
so collisions are resolved in this order:

  1. scripts/figures_manifest.yml has an entry for the filename that names
     every current producer -> deploy each producer under its
     manifest-specified name. This is the primary, expected path.
  2. config/experiments.yml's optional `figure_owners:` map
     ({ filename: experiment_id }) is a separate, OPT-IN override for the
     case where the colliding outputs really are duplicates/superseded and
     only one should survive under the original bare name. The others are
     dropped (and logged).
  3. Neither covers it -> deploy every producer under an auto-suffixed name
     (`stem__eid.ext`) so no canonical result is ever silently dropped, and
     the build never hard-fails on an unmapped collision.

A manifest entry that no longer matches an actual collision (typo'd/renamed
filename, or the collision no longer exists) is flagged as stale rather
than silently ignored.

Non-colliding files are copied under their bare filename. A pre-existing
deployed file that differs in content from its (single, unambiguous)
source experiment is treated as a stale deploy and overwritten.

Usage
-----
    python3 deploy_figures.py \\
        [--config config/experiments.yml] \\
        [--manifest scripts/figures_manifest.yml] \\
        [--results-root PATH]   # overrides results_root from config

Exit status is 0 on success (including "nothing to do"); non-zero if the
config or manifest can't be parsed as expected.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required: python -m pip install pyyaml"
    ) from exc


def warn(msg: str) -> None:
    """Emit a warning. Uses the GitHub Actions annotation format when
    running in CI (harmless, ignored elsewhere) and always goes to stderr."""
    print(f"::warning::{msg}", file=sys.stderr)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"config not found: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    if "experiments" not in cfg:
        raise SystemExit(f"{path}: missing required top-level key 'experiments'")
    if "results_root" not in cfg:
        raise SystemExit(f"{path}: missing required top-level key 'results_root'")
    return cfg


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Returns { source_filename: { experiment_id: deployed_filename } }."""
    if not path.is_file():
        warn(
            f"{path} not found — collisions will fall back to figure_owners "
            f"/ auto-suffixed filenames."
        )
        return {}
    with path.open() as f:
        manifest_cfg = yaml.safe_load(f) or {}
    figures = manifest_cfg.get("figures", {}) or {}
    if not isinstance(figures, dict):
        raise SystemExit(f"{path}: 'figures' must be a mapping")
    return figures


def collect_sources(
    cfg: dict, results_root: str
) -> dict[str, list[tuple[str, Path, str]]]:
    """Returns { filename: [(experiment_id, path, sha256), ...] }, sorted
    by experiment ID / filename so results are identical on every run,
    independent of dict or filesystem iteration order."""
    sources: dict[str, list[tuple[str, Path, str]]] = {}
    for eid, entry in sorted(cfg["experiments"].items()):
        if not entry.get("generates_figures"):
            continue
        src_dir = Path(results_root) / entry["source_dir"] / "figures"
        if not src_dir.is_dir():
            continue
        for fn in sorted(os.listdir(src_dir)):
            src = src_dir / fn
            if not src.is_file():
                continue
            sources.setdefault(fn, []).append((eid, src, sha256_of(src)))
    return sources


def resolve_collisions(
    collisions: dict[str, list[tuple[str, Path, str]]],
    manifest: dict[str, dict[str, str]],
    figure_owners: dict[str, str],
    sources: dict[str, list[tuple[str, Path, str]]],
) -> tuple[dict[str, list[tuple[str, Path, str, str]]], set[str]]:
    """Mutates `sources` in place (removing/collapsing resolved collisions)
    and returns (split_deploy, manifest_seen_for)."""
    split_deploy: dict[str, list[tuple[str, Path, str, str]]] = {}
    manifest_seen_for: set[str] = set()
    resolved_manifest_log: list[tuple[str, list[str]]] = []
    resolved_owner_log: list[tuple[str, str, list[str]]] = []

    for fn, entries in sorted(collisions.items()):
        manifest_map = manifest.get(fn)
        producer_eids = {eid for eid, _, _ in entries}

        if manifest_map and producer_eids <= set(manifest_map):
            # Manifest covers every current producer: deploy each one under
            # its manifest-specified name. Primary, expected resolution path.
            del sources[fn]
            manifest_seen_for.add(fn)
            split_deploy[fn] = [
                (eid, src, src_sha, manifest_map[eid]) for eid, src, src_sha in entries
            ]
            resolved_manifest_log.append((fn, sorted(producer_eids)))
            continue

        if manifest_map:
            # Entry exists but doesn't cover every current producer
            # (stale/incomplete) — don't trust a partial mapping.
            warn(
                f"manifest has an entry for {fn} but it doesn't cover all "
                f"current producers ({', '.join(sorted(producer_eids))}) — "
                f"ignoring it for this run and falling back to figure_owners "
                f"/ auto-suffixing. Update the manifest."
            )

        owner = figure_owners.get(fn)
        owning_entries = [e for e in entries if e[0] == owner] if owner else []
        if owning_entries:
            # Explicit opt-in: collapse to the owner's single entry under
            # the original filename; the others are dropped.
            sources[fn] = owning_entries
            skipped_eids = sorted({eid for eid, _, _ in entries if eid != owner})
            resolved_owner_log.append((fn, owner, skipped_eids))
        else:
            # No manifest entry, no override (or a misconfigured one naming
            # a non-producer): deploy every producer under its own
            # auto-suffixed name rather than silently dropping any
            # canonical source.
            del sources[fn]
            stem, ext = os.path.splitext(fn)
            split_deploy[fn] = [
                (eid, src, src_sha, f"{stem}__{eid}{ext}") for eid, src, src_sha in entries
            ]
            if owner:
                warn(
                    f"figure_owners names '{owner}' for {fn}, but that "
                    f"experiment did not produce it — ignoring the override "
                    f"and deploying all producers under auto-suffixed "
                    f"filenames instead."
                )

    if resolved_manifest_log:
        print("figures_deploy: resolved filename collision(s) via manifest:")
        for fn, eids in resolved_manifest_log:
            print(f"  {fn} — deployed all of: {', '.join(eids)}, each under its manifest name")

    if resolved_owner_log:
        print("figures_deploy: resolved filename collision(s) via figure_owners:")
        for fn, owner, skipped_eids in resolved_owner_log:
            print(f"  {fn} — deployed from '{owner}', skipped: {', '.join(skipped_eids)}")

    return split_deploy, manifest_seen_for


def deploy(
    cfg: dict,
    manifest: dict[str, dict[str, str]],
    results_root: str,
    dest: Path,
) -> int:
    figure_owners: dict[str, str] = cfg.get("figure_owners", {}) or {}

    dest.mkdir(parents=True, exist_ok=True)

    sources = collect_sources(cfg, results_root)

    # Genuine cross-experiment collision: same filename, different content,
    # from two different experiments.
    collisions = {
        fn: entries for fn, entries in sources.items() if len({s for _, _, s in entries}) > 1
    }

    split_deploy, manifest_seen_for = resolve_collisions(
        collisions, manifest, figure_owners, sources
    )

    stale_manifest_entries = sorted(set(manifest) - manifest_seen_for)
    if stale_manifest_entries:
        warn(
            "manifest has entries that don't match any current filename "
            f"collision (stale?): {', '.join(stale_manifest_entries)}"
        )

    copied, skipped, overwritten = 0, 0, []

    if split_deploy:
        warn(
            "Filename collision(s) resolved by deploying EACH producer's "
            "copy under its own name so no canonical result is dropped. "
            "Update the LaTeX source to reference these filenames (the old "
            "bare name is no longer deployed for these stems):"
        )
        for fn, entries in sorted(split_deploy.items()):
            for eid, src, src_sha, suffixed_fn in sorted(entries):
                dst = dest / suffixed_fn
                print(f"  {fn} ({eid}) -> {suffixed_fn}")
                if dst.exists() and sha256_of(dst) == src_sha:
                    skipped += 1
                    continue
                is_overwrite = dst.exists()
                shutil.copy2(src, dst)
                if is_overwrite:
                    overwritten.append((suffixed_fn, eid))
                else:
                    copied += 1

    for fn, entries in sorted(sources.items()):
        eid, src, src_sha = entries[0]  # all entries agree on content
        dst = dest / fn
        if dst.exists():
            if sha256_of(dst) == src_sha:
                skipped += 1
                continue
            shutil.copy2(src, dst)
            overwritten.append((fn, eid))
            continue
        shutil.copy2(src, dst)
        copied += 1

    print(f"figures_deploy: copied {copied}, already up to date {skipped}, "
          f"{len(overwritten)} overwritten.")
    if overwritten:
        warn("Deployed copy differed from its source experiment (stale deploy) and was overwritten:")
        for fn, eid in overwritten:
            print(f"  {fn} <- {eid}")

    return copied + len(overwritten)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("CONFIG_FILE", "config/experiments.yml")),
        help="Path to config/experiments.yml (default: %(default)s, or $CONFIG_FILE)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(os.environ.get("MANIFEST_FILE", "scripts/figures_manifest.yml")),
        help="Path to figures_manifest.yml (default: %(default)s, or $MANIFEST_FILE)",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Override results_root from config (default: read from config)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Deployed-figures directory. Defaults to <results_root>/<figures_deploy figures_dir>.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = load_manifest(args.manifest)

    results_root = args.results_root or cfg["results_root"]

    if args.dest is not None:
        dest = args.dest
    else:
        aggregates = cfg.get("aggregates", {}) or {}
        entry = aggregates.get("figures_deploy")
        if not entry or "figures_dir" not in entry:
            raise SystemExit(
                f"{args.config}: no aggregates.figures_deploy.figures_dir found; "
                f"pass --dest explicitly."
            )
        dest = Path(results_root) / entry["figures_dir"]

    deploy(cfg, manifest, results_root, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

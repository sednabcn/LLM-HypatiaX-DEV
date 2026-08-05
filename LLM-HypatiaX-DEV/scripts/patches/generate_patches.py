#!/usr/bin/env python3
"""
generate_patches.py — Auto-generate diff-based patches

Compares raw results (data/results/) against curated JMLR-SOURCE-LAST outputs.
Produces minimal field-level patches in patches/generated/.

Usage:
    python scripts/generate_patches.py [--dry-run]
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT       = Path(__file__).parent.parent
RAW_DIR    = ROOT / "hypatiax" / "data" / "results"
JMLR_DIR   = ROOT / "JMLR-SOURCE-LAST"
PATCH_DIR  = ROOT / "patches" / "generated"
PATCH_LOG  = ROOT / "patches" / "patch_log.jsonl"

DRY_RUN    = "--dry-run" in sys.argv

PATCH_DIR.mkdir(parents=True, exist_ok=True)

# ── Known manual fixes (FIX-C1: duplicate DeFi case names) ───────────────────
KNOWN_RENAMES = {
    "Constant product formula": [
        "Constant product formula (basic)",
        "Constant product formula (multivariate)",
    ],
    "Funding rate cost": [
        "Funding rate cost (simple)",
        "Funding rate cost (extended)",
    ],
    "Concentrated liquidity position width": [
        "Concentrated liquidity position width",
        "Concentrated liquidity position width (v2)",
    ],
}

# ── Field-level diff ──────────────────────────────────────────────────────────
def dict_diff(raw: dict, curated: dict, path="") -> dict:
    """Return minimal dict of fields that differ between raw and curated."""
    diff = {}
    for key, curated_val in curated.items():
        full_key = f"{path}.{key}" if path else key
        raw_val = raw.get(key)
        if isinstance(curated_val, dict) and isinstance(raw_val, dict):
            sub = dict_diff(raw_val, curated_val, full_key)
            if sub:
                diff[key] = sub
        elif isinstance(curated_val, list) and isinstance(raw_val, list):
            if curated_val != raw_val:
                diff[key] = curated_val   # lists: full replacement
        elif curated_val != raw_val:
            diff[key] = curated_val
    return diff

# ── Find matching file pairs ──────────────────────────────────────────────────
def find_pairs():
    """Yield (raw_path, curated_path) pairs."""
    # Look in known result subdirectories
    result_subdirs = [
        ("defi",        "input_cosmetic"),
        ("feynman",     "input_cosmetic"),
        ("exp1_ablation", "results-ablation"),
        ("noise",       "input_cosmetic"),
        ("routing",     "input_cosmetic"),
        ("instability", "input_cosmetic"),
    ]

    pairs = []
    for raw_sub, jmlr_sub in result_subdirs:
        raw_d    = RAW_DIR / raw_sub
        jmlr_d   = JMLR_DIR / jmlr_sub
        if not raw_d.exists() or not jmlr_d.exists():
            continue
        for jmlr_file in sorted(jmlr_d.glob("*.json")):
            # Find best-matching raw file (same stem, most recent)
            candidates = sorted(
                raw_d.glob(f"*{jmlr_file.stem}*.json"),
                key=os.path.getmtime,
                reverse=True,
            )
            if candidates:
                pairs.append((candidates[0], jmlr_file))

    # Also diff full merged results if they exist
    merged_raw    = RAW_DIR / "merged" / "all_systems_merged.json"
    merged_jmlr   = JMLR_DIR / "input_cosmetic" / "all_systems_merged.json"
    if merged_raw.exists() and merged_jmlr.exists():
        pairs.append((merged_raw, merged_jmlr))

    return pairs

# ── Generate one patch ────────────────────────────────────────────────────────
def generate_patch(raw_path: Path, curated_path: Path) -> dict | None:
    try:
        raw     = json.loads(raw_path.read_text())
        curated = json.loads(curated_path.read_text())
    except Exception as e:
        print(f"  ⚠  Could not read {raw_path.name} or {curated_path.name}: {e}")
        return None

    diff = dict_diff(raw, curated)
    if not diff:
        return None   # identical — no patch needed

    patch = {
        "meta": {
            "generated":    datetime.now().isoformat(),
            "raw_file":     str(raw_path.relative_to(ROOT)),
            "curated_file": str(curated_path.relative_to(ROOT)),
            "patch_type":   "field_level_diff",
        },
        "diff": diff,
    }
    return patch

# ── Hardcoded patches (FIX-C1, FIX-C2 etc.) ──────────────────────────────────
def generate_hardcoded_patches():
    patches = []

    # FIX-C1: Duplicate DeFi case names
    defi_src = next(ROOT.rglob("hypatiax_defi_benchmark_v3c.py"), None)
    if defi_src:
        patches.append({
            "meta": {
                "id":          "FIX-C1",
                "description": "Rename duplicate DeFi case names",
                "target_file": str(defi_src.relative_to(ROOT)),
            },
            "renames": KNOWN_RENAMES,
        })

    # FIX-C2: Stale v40 imports
    patches.append({
        "meta": {
            "id":          "FIX-C2",
            "description": "Replace hybrid_system_v40 with hybrid_system_v50_2",
            "auto_applied_by": "run_all.sh Phase 0 sed command",
        },
        "sed": "s/hybrid_system_v40[^_]/hybrid_system_v50_2/g",
    })

    # FIX-T1: 71 cases → 70 tasks
    patches.append({
        "meta": {
            "id":          "FIX-T1",
            "description": "Fix '71 cases' → '70 tasks' in §10.9 of main paper",
            "target_file": "paper/jmlr-hypatiax-paper-final.tex",
        },
        "sed": "s/across all 71 cases/across all 70 tasks/g",
    })

    # FIX-T2: five-layer → five-stage
    patches.append({
        "meta": {
            "id":          "FIX-T2",
            "description": "Standardise 'five-stage' terminology in §8.3",
            "target_file": "paper/jmlr-hypatiax-paper-final.tex",
        },
        "sed": "s/Five-Layer Architecture Overview/Five-Stage Architecture Overview/g",
    })

    # FIX-XR3: Supp A section number
    patches.append({
        "meta": {
            "id":          "FIX-XR3",
            "description": "Fix 'Section 7.3' → 'Section 7.4' in supp_routing_improvements.tex",
            "target_file": "paper/supp_routing_improvements.tex",
        },
        "sed": "s/Section 7\\.3 (Component 3)/Section 7.4 (Component 3)/g",
    })

    return patches

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("═" * 60)
    print("  Patch Generator — HypatiaX JMLR")
    print("═" * 60)

    all_patches = []

    # Auto-diff pairs
    pairs = find_pairs()
    print(f"\n  Found {len(pairs)} raw/curated file pairs to diff")

    for raw_path, curated_path in pairs:
        patch = generate_patch(raw_path, curated_path)
        if patch:
            name = f"auto_{curated_path.stem}.patch.json"
            all_patches.append((name, patch))
            print(f"  📝 {name}  ({len(patch['diff'])} changed fields)")
        else:
            print(f"  ✅ {curated_path.stem}  (identical — no patch needed)")

    # Hardcoded FIX-* patches
    hardcoded = generate_hardcoded_patches()
    for hp in hardcoded:
        name = f"fix_{hp['meta']['id'].lower()}.patch.json"
        all_patches.append((name, hp))
        print(f"  📝 {name}  ({hp['meta']['description']})")

    if DRY_RUN:
        print(f"\n  [dry-run] Would write {len(all_patches)} patch files to patches/generated/")
        return

    # Write patches
    log_entries = []
    for name, patch in all_patches:
        out = PATCH_DIR / name
        out.write_text(json.dumps(patch, indent=2))
        log_entries.append({"file": name, "meta": patch["meta"]})

    # Append to patch log
    with open(PATCH_LOG, "a") as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\n✅ {len(all_patches)} patches written to patches/generated/")
    print("   Log appended to patches/patch_log.jsonl")

if __name__ == "__main__":
    main()

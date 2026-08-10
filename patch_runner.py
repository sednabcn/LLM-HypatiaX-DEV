#!/usr/bin/env python3
"""
patch_runner.py — apply patch_manifest.yaml to the manuscript .tex files.

Design goals:
  - Every text change is a unique-match find/replace (never line numbers,
    which drift as earlier patches shift line counts).
  - "open" items with status=needs_investigation are never hand-filled;
    the runner shells out to the item's investigation_script, reads
    investigation_results.json, and renders new_str_template with the
    result. If the investigation hasn't been run yet, the item is
    reported as SKIPPED, not silently guessed.
  - "blocked" items only ever get a caveat/flag inserted (action:
    insert_caveat), never a fabricated corrected value.
  - Dry-run by default. Nothing is written to disk unless --apply is passed.
  - After applying, recompiles every touched .tex file (pdflatex, 2 passes)
    and reports new errors/undefined refs so a bad patch is caught
    immediately rather than at the next full report cycle.

Usage:
    python3 patch_runner.py --manifest patch_manifest.yaml            # dry run, report only
    python3 patch_runner.py --manifest patch_manifest.yaml --apply    # write changes
    python3 patch_runner.py --manifest patch_manifest.yaml --apply --ids 2,3,11
    python3 patch_runner.py --manifest patch_manifest.yaml --compile-only
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("Requires PyYAML: pip install pyyaml --break-system-packages")

INVESTIGATION_RESULTS = Path("investigation_results.json")
APPLIED_LOG = Path("applied_log.json")   # ledger of what's CURRENTLY in the files, with provenance


def load_manifest(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_applied_log():
    if APPLIED_LOG.exists():
        return json.loads(APPLIED_LOG.read_text())
    return {}


def load_investigation_results():
    if INVESTIGATION_RESULTS.exists():
        return json.loads(INVESTIGATION_RESULTS.read_text())
    return {}


def save_applied_log(log):
    APPLIED_LOG.write_text(json.dumps(log, indent=2))


def record_application(log, item_id, new_str, value, source, apply_changes):
    """Append-only history per item: every value this item has ever held in
    the manuscript, with where it came from and when. `current` always
    points at the latest entry so the next run knows what's actually in
    the file right now, without re-deriving it from the pristine manifest."""
    key = str(item_id)
    entry = log.setdefault(key, {"history": []})
    record = {
        "new_str": new_str,
        "value": value,
        "source": source,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "applied": apply_changes,
    }
    entry["history"].append(record)
    entry["current"] = record
    if apply_changes:
        save_applied_log(log)
    return entry
    if INVESTIGATION_RESULTS.exists():
        return json.loads(INVESTIGATION_RESULTS.read_text())
    return {}


def run_investigation(item, results):
    """Run an item's investigation script if its result isn't already cached."""
    key = str(item["id"])
    if key in results:
        return results[key]
    script = item.get("investigation_script")
    if not script or not Path(script).exists():
        return None
    print(f"  [investigate] item {item['id']}: running {script}")
    proc = subprocess.run([sys.executable, script], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"    FAILED (exit {proc.returncode}): {proc.stderr.strip()[:400]}")
        return None
    try:
        value = json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        print(f"    investigation script must print a single JSON value; got: {proc.stdout[:200]}")
        return None
    results[key] = value
    INVESTIGATION_RESULTS.write_text(json.dumps(results, indent=2))
    return value


def apply_item(item, apply_changes, investigation_results, applied_log, allow_value_change):
    file_path = Path(item["file"])
    if not file_path.exists():
        return item["id"], "ERROR", f"file not found: {file_path}"

    text = file_path.read_text()
    status = item.get("status", "ready")

    if status in ("SUPERSEDED", "TODO"):
        return item["id"], status, item.get("note", "").strip().split("\n")[0][:120]

    if status == "blocked":
        if item.get("action") != "insert_caveat":
            return item["id"], "SKIPPED", "blocked item, no caveat action defined"
        new_str = item["new_str"]
        anchor = item.get("anchor_str") or item.get("old_str")
        if not anchor or anchor not in text:
            return item["id"], "ERROR", "anchor string for caveat insertion not found"
        if new_str in text:
            return item["id"], "NOOP", "caveat already present"
        if apply_changes:
            file_path.write_text(text.replace(anchor, anchor + new_str, 1))
        return item["id"], "CAVEAT-INSERTED" if apply_changes else "DRY-RUN-CAVEAT", ""

    # --- Determine the value and its provenance for this run ---
    if status == "needs_investigation":
        value = run_investigation(item, investigation_results)
        if value is None:
            return item["id"], "SKIPPED", "investigation not yet run or failed"
        new_str = item["new_str_template"].format(value=value)
        source = item.get("investigation_script")
    else:
        new_str = item["new_str"]
        value = item.get("value", new_str)
        source = item.get("source", "manual/text-fix")

    # --- Determine what to search for: the CURRENT state, not always the
    #     pristine manifest old_str. If this item was already applied in an
    #     earlier run, search for what we last put in the file. ---
    ledger_entry = applied_log.get(str(item["id"]))
    if ledger_entry and ledger_entry.get("current"):
        search_str = ledger_entry["current"]["new_str"]
        prior_value = ledger_entry["current"]["value"]
    else:
        search_str = item["old_str"]
        prior_value = None

    n = text.count(search_str)
    if n == 0:
        # Neither the pristine old_str nor the last-applied value is present.
        # Something changed the file out-of-band (manual edit, different
        # patch, etc.) — refuse rather than guess where to write.
        return item["id"], "ERROR", f"expected text not found (searched for last-known state: {search_str[:60]!r})"
    if n > 1:
        return item["id"], "ERROR", f"match is not unique ({n} occurrences) — refusing to patch"

    if search_str == new_str:
        return item["id"], "NOOP", "already at this value"

    value_changed = prior_value is not None and prior_value != value
    if value_changed and not allow_value_change:
        return (item["id"], "VALUE-CHANGED",
                f"already patched with {prior_value!r} (from {ledger_entry['current']['source']}); "
                f"new investigation says {value!r} (from {source}). "
                f"Re-run with --allow-value-change to accept the update.")

    if apply_changes:
        text = text.replace(search_str, new_str, 1)
        footnote = item.get("also_remove_footnote")
        if footnote and footnote in text:
            text = text.replace(footnote, "")
        file_path.write_text(text)
        record_application(applied_log, item["id"], new_str, value, source, apply_changes)

    for must_be_absent in item.get("verify_absent", []):
        check_text = file_path.read_text() if apply_changes else text.replace(search_str, new_str, 1)
        if must_be_absent in check_text:
            return item["id"], "WARN", f"expected '{must_be_absent}' to be gone, still present"

    label = "VALUE-UPDATED" if value_changed else ("PATCHED" if apply_changes else "DRY-RUN-OK")
    detail = f"{prior_value!r} -> {value!r}" if value_changed else ""
    return item["id"], label, detail


def compile_check(tex_files):
    if not shutil.which("pdflatex"):
        print("pdflatex not found, skipping compile check")
        return
    for tex in tex_files:
        print(f"\n[compile] {tex}")
        for _ in range(2):
            proc = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex],
                capture_output=True, text=True, cwd=Path(tex).parent or ".",
            )
        log = Path(tex).with_suffix(".log")
        if log.exists():
            log_text = log.read_text(errors="ignore")
            errors = re.findall(r"^! .*$", log_text, re.MULTILINE)
            undefined = re.findall(r"Reference `[^']*' .*undefined", log_text)
            if errors:
                print(f"  {len(errors)} LaTeX error(s):")
                for e in errors[:10]:
                    print(f"    {e}")
            if undefined:
                print(f"  {len(undefined)} undefined reference(s) (rerun pdflatex once more if just added)")
            if not errors:
                print("  OK — no LaTeX errors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="patch_manifest.yaml")
    ap.add_argument("--apply", action="store_true", help="write changes (default is dry-run)")
    ap.add_argument("--ids", help="comma-separated list of item ids to restrict to")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--allow-value-change", action="store_true",
                     help="permit overwriting an already-applied value with a different one")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    if args.ids:
        wanted = {int(x) for x in args.ids.split(",")}
        manifest = [m for m in manifest if m["id"] in wanted]

    tex_files = sorted({m["file"] for m in manifest if "file" in m})

    if args.compile_only:
        compile_check(tex_files)
        return

    investigation_results = load_investigation_results()
    applied_log = load_applied_log()
    print(f"{'APPLY' if args.apply else 'DRY RUN'} — {len(manifest)} item(s)\n")
    rows = []
    for item in manifest:
        item_id, outcome, detail = apply_item(item, args.apply, investigation_results,
                                               applied_log, args.allow_value_change)
        rows.append((item_id, item.get("category", "?"), outcome, detail))
        ok_states = ("PATCHED", "DRY-RUN-OK", "NOOP", "CAVEAT-INSERTED", "DRY-RUN-CAVEAT", "VALUE-UPDATED")
        flag = "" if outcome in ok_states else "  <-- needs attention"
        print(f"  #{item_id:>2} [{item.get('category','?'):6}] {outcome:16} {detail}{flag}")

    if args.apply:
        compile_check(tex_files)

    failed = [r for r in rows if r[2] in ("ERROR", "WARN")]
    if failed:
        print(f"\n{len(failed)} item(s) need manual attention before this counts as done.")
        sys.exit(1)


if __name__ == "__main__":
    main()

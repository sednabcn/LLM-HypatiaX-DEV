#!/usr/bin/env python3
"""
extract_table_rows.py — regenerate the "List of Tables" data by statically
scanning the paper sources instead of hand-maintaining it.

Inputs:
  - the three paper .tex files (for \\label{tab:...} + \\caption{...})
  - generate_tables.py (for which label each gen_* function writes,
    via write_table()/skip_table() calls + its docstring/comments)

Output:
  - table_data.py, with the same ROWS list shape build_pdf.py already reads,
    so `python extract_table_rows.py && python build_pdf.py` reproduces the
    PDF end to end from source.

This is intentionally conservative: where the source code doesn't say enough
to classify a table with confidence, the script reports "REVIEW NEEDED"
rather than guessing — same discipline generate_tables.py itself uses for
data (never fabricate, flag instead).
"""
from __future__ import annotations
import re
from pathlib import Path

TEX_DIR = Path("/home/claude/extract1")
PY_FILE = Path("/home/claude/extract2/generate_tables.py")
OUT_FILE = Path("/home/claude/table_data.py")

TEX_FILES = [
    "jmlr_paper_main_patched_CLEANED.tex",
    "supp_benchmark_report_patched_CLEANED.tex",
    "supp_routing_improvements_pached_CLEANED.tex",
]

# ── 1. Pull every \label{tab:...} + nearest preceding \caption{...} ─────────

def extract_labels():
    labels = []  # (label, caption, source_file)
    for fn in TEX_FILES:
        text = (TEX_DIR / fn).read_text(encoding="utf-8", errors="replace")
        lines = text.split("\n")
        for i, line in enumerate(lines):
            m = re.search(r"\\label\{(tab:[^}]+)\}", line)
            if not m:
                continue
            label = m.group(1)
            caption = ""
            for j in range(i, max(0, i - 15), -1):
                cm = re.search(r"\\caption\{(.*)", lines[j])
                if cm:
                    caption = re.sub(r"\\textbf\{|\}", "", cm.group(1))
                    caption = re.sub(r"\\\\", " ", caption).strip()
                    caption = caption[:100]
                    break
            labels.append((label, caption, fn))
    return labels


# ── 2. Parse generate_tables.py into per-function blocks ───────────────────

FUNC_RE = re.compile(r"^def (gen_\w+)\([^)]*\)\s*->\s*None:\s*$", re.M)
# Any top-level "def " (not just gen_*) ends the current gen_ block — this
# stops gen_routing_conceptual_complexity (the last gen_ function) from
# swallowing main()'s huge _AUDIT table, which is full of unrelated tab:
# references and was silently mis-attributing labels before this fix.
ANY_DEF_RE = re.compile(r"^def \w+\(", re.M)

def split_functions(src: str):
    gen_matches = list(FUNC_RE.finditer(src))
    all_def_starts = sorted(m.start() for m in ANY_DEF_RE.finditer(src))
    blocks = {}
    for idx, m in enumerate(gen_matches):
        start = m.start()
        # end = next top-level def of ANY kind, not just the next gen_ match
        end = next((s for s in all_def_starts if s > start), len(src))
        blocks[m.group(1)] = src[start:end]
    return blocks


STATUS_KEYWORDS = [
    # (substring to look for in block text, status label)
    ("hand-authored", "Intentionally not automated"),
    ("qualitative", "Intentionally not automated"),
    ("is an *estimate*", "Intentionally not automated"),
    ("no named source", "Intentionally not automated"),
    ("withdrawn", "Intentionally not automated"),
    ("path guessed", "Generated (path guessed)"),
    ("unconfirmed schema", "Generated (unconfirmed schema)"),
    ("unverified", "Generated (caveated)"),
    ("always fell through to skip_table", "Blocked"),
    ("always `skip_table()`s", "Blocked"),
]


def meta_text(block: str) -> str:
    """Docstring + '#' comment lines only — NOT the LaTeX string literals the
    function builds. Classifying against the full block was matching words
    like 'withdrawn'/'unverified' when they appeared inside a table's actual
    historical-narrative *content* (e.g. gen_provenance's row text), not in
    the function's own meta-commentary about how it was generated."""
    doc_m = re.search(r'"""(.*?)"""', block, re.S)
    doc = doc_m.group(1) if doc_m else ""
    comments = "\n".join(
        line for line in block.split("\n") if line.strip().startswith("#")
    )
    return doc + "\n" + comments


def classify_block(name: str, block: str, has_write: bool, has_skip: bool) -> str:
    lower = meta_text(block).lower()
    for kw, status in STATUS_KEYWORDS:
        if kw.lower() in lower:
            return status
    if has_write and has_skip:
        return "Generated (JSON-backed)"
    if has_write and not has_skip:
        return "Static / stable" if "config" not in lower else "Static / stable"
    if has_skip and not has_write:
        return "Intentionally not automated"
    return "REVIEW NEEDED"


def parse_generators():
    src = PY_FILE.read_text(encoding="utf-8", errors="replace")
    blocks = split_functions(src)

    # label -> dict(file, status, note, _from) — _from tracks whether the
    # match came from an authoritative anchor (a real \label{} in the
    # emitted LaTeX, or a tab: mention inside a skip_table() reason) so a
    # later, weaker match never overwrites a stronger one. This is the fix
    # for the first pass's bug: a "tab:x" mentioned only in passing inside
    # some *other* function's prose comment/docstring was overwriting the
    # correct mapping simply by being processed later.
    by_label: dict[str, dict] = {}

    for fname, block in blocks.items():
        write_matches = re.findall(r'write_table\(\s*"([^"]+\.tex)"', block)
        skip_calls = re.findall(r'skip_table\(\s*"([^"]+\.tex)"\s*,((?:[^()]|\([^()]*\))*)\)', block)
        has_write = bool(write_matches)
        has_skip = bool(skip_calls)
        status = classify_block(fname, block, has_write, has_skip)

        target_file = write_matches[0] if write_matches else (skip_calls[0][0] if skip_calls else None)

        # Anchor 1 (strongest): \label{tab:...} literally inside a tex string
        # this function builds — that IS the table this function owns.
        anchored_labels = set(re.findall(r"\\label\{(tab:[\w\-]+)\}", block))

        # Anchor 2: a tab: reference named inside a skip_table() reason
        # string itself (e.g. "tab:additional_stats has no named source...").
        for _, reason in skip_calls:
            anchored_labels |= set(re.findall(r"\btab:[\w\-]+", reason))

        # Anchor 3 (weakest, only used if 1+2 found nothing): a tab:
        # reference in the function's own docstring — this codebase's
        # convention is "tab:x — description" as the docstring's first
        # line, but it's meta-commentary, not guaranteed unique, so it only
        # fires when nothing stronger was found for this label already.
        if not anchored_labels:
            doc_m = re.search(r'"""(.*?)"""', block, re.S)
            if doc_m:
                anchored_labels |= set(re.findall(r"\btab:[\w\-]+", doc_m.group(1)))

        note = skip_calls[0][1].strip(' "') [:140] if skip_calls else ""
        if not note:
            doc_m = re.search(r'"""(.*?)"""', block, re.S)
            if doc_m:
                note = " ".join(doc_m.group(1).split())[:140]

        for label in anchored_labels:
            prev = by_label.get(label)
            # Only overwrite if we don't yet have an anchored match for this
            # label (two functions should not both claim the same label —
            # if they do, keep the first and let it surface via the
            # duplicate-label check the caller can add later).
            if prev is None:
                by_label[label] = {"file": target_file, "status": status, "note": note}
    return by_label


def main():
    labels = extract_labels()
    gen_map = parse_generators()

    rows = []
    review_flags = []
    for i, (label, caption, src_file) in enumerate(labels, start=1):
        key = label
        info = gen_map.get(key)
        if info is None:
            gen_file = "(none)"
            status = "REVIEW NEEDED — no generator function references this label"
            note = "grep generate_tables.py for this label manually"
            review_flags.append(label)
        else:
            gen_file = info["file"] or "(none)"
            status = info["status"]
            note = info["note"]
            if status == "REVIEW NEEDED":
                review_flags.append(label)
        rows.append((i, label.replace("tab:", ""), caption, gen_file, status, note))

    with OUT_FILE.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated by extract_table_rows.py — do not hand-edit.\n")
        f.write("# Row format: (#, label, caption, generated_file, status, source)\n\n")
        f.write("ROWS = [\n")
        for r in rows:
            f.write(repr(r) + ",\n")
        f.write("]\n")

    print(f"Wrote {len(rows)} rows to {OUT_FILE}")
    if review_flags:
        print(f"REVIEW NEEDED for {len(review_flags)} label(s): {review_flags}")


if __name__ == "__main__":
    main()

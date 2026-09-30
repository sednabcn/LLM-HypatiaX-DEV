#!/usr/bin/env python3
"""
check_paper_tables.py -- fail CI when a table hard-coded in the paper differs from the one
generate_tables.py produces from the result files.

Usage
-----
    python3 scripts/generate_tables.py --results-dir hypatiax/data/results --output-dir /tmp/tables
    python3 scripts/check_paper_tables.py --tables /tmp/tables --docs paper/*.tex \
        --exceptions paper/paper_table_exceptions.json --strict-nosource \
        [--report report.md] [--emit-template todo_exceptions.json]

Both scripts live in scripts/; the exceptions file lives in paper/ (next to the .tex it
excuses, so every exception is reviewed in the same PR as the paper change).

For every \\label{tab:X} found in the documents it locates a generated file containing the same
label and compares the NUMBERS in the two tables (formatting, bold, column order and TeX macros
are ignored). Statuses:

    MATCH      every number in the generated table appears in the paper table
    PARTIAL    >= 80 % of generated numbers appear in the paper table  (reported, fails CI)
    DIFF       < 80 % appear                                           (fails CI)
    NOSOURCE   label is in the paper but the generator did not write a file for it
               (generator skipped it, or has no generator)               (reported, fails only
               with --strict-nosource)
    EXCEPTION  listed in the exceptions file with a written reason      (never fails)
    UNVERIFIED an exception whose reason starts with "UNVERIFIED"        (never fails, but
               is counted and listed separately: the paper number has no data behind it)
    WIRED      the paper pulls the table in with \\input{...} from the generated
               directory, so it cannot drift                            (never fails)

Exit status is 1 if any DIFF/PARTIAL is not covered by an exception, else 0.

The exceptions file is a JSON object {"<label>": "<reason>"}. Every entry must carry a reason;
an empty reason is rejected, so an exception cannot be added silently.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
LABEL = re.compile(r"\\label\{(tab:[^}]+)\}")


def _strip_tex(t: str) -> str:
    t = re.sub(r"(?<=\d)\{,\}(?=\d)", "", t)  # TeX thousands separator: 2{,}740.0 -> 2740.0
    t = t.replace("\u2212", "-").replace("\u2013", "--").replace("\u2014", "---")  # unicode minus / dashes
    t = re.sub(r"(?<!\\)%.*", "", t)                                 # comments
    # 1.852\times10^{46}  ->  1.852e46   (else mantissa, 10 and 46 count as 3 numbers)
    t = re.sub(r"(\d(?:\.\d+)?)\s*\\times\s*10\^\{?\s*(-?\d+)\s*\}?", r"\1e\2", t)
    t = re.sub(r"(?<=\d)\s*--+\s*(?=\d)", " ", t)                     # ranges 1--3 are two numbers, not 1 and -3
    t = re.sub(r"\\(label|ref|Cref|cref|cite|citep|citet|texttt|url)\{[^}]*\}", " ", t)
    t = re.sub(r"\\caption\{.*?\n\}", " ", t, flags=re.S)             # captions carry prose numbers
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    return t


def _table_env(text: str, label: str) -> str:
    i = text.find("\\label{%s}" % label)
    if i < 0:
        return ""
    starts = [text.rfind(k, 0, i) for k in ("\\begin{table", "\\begin{longtable")]
    a = max(starts)
    ends = [e for e in (text.find("\\end{table", i), text.find("\\end{longtable", i)) if e >= 0]
    return text[a:min(ends)] if a >= 0 and ends else ""


def _body(env: str) -> str:
    """Rows of the tabular only (drops the caption/notes so prose numbers do not count)."""
    m = re.search(r"\\begin\{(?:tabular|longtable)\}.*?\\end\{(?:tabular|longtable)\}", env, re.S)
    return m.group(0) if m else env


def numbers(env: str) -> list[float]:
    out = []
    for tok in NUM.findall(_strip_tex(_body(env))):
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def close(a: float, b: float) -> bool:
    if a == b:
        return True
    tol = max(5e-4, 5e-4 * abs(b))          # tolerate rounding to 3-4 decimals / 4 sig. figs.
    return abs(a - b) <= tol or abs(a - round(b, 3)) <= 5e-4


def compare(gen: list[float], paper: list[float]) -> tuple[float, list[float]]:
    if not gen:
        return 1.0, []
    pool = list(paper)
    missing = []
    for g in gen:
        hit = next((k for k, p in enumerate(pool) if close(g, p)), None)
        if hit is None:
            hit = next((k for k, p in enumerate(paper) if close(g, p)), None)   # allow re-use
            if hit is None:
                missing.append(g)
        else:
            pool.pop(hit)
    return 1 - len(missing) / len(gen), missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True, type=Path)
    ap.add_argument("--docs", required=True, nargs="+", type=Path)
    ap.add_argument("--exceptions", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--strict-nosource", action="store_true")
    ap.add_argument("--emit-template", type=Path,
                    help="write a JSON skeleton of every failing label (reason 'UNVERIFIED: TODO')")
    a = ap.parse_args()

    exc: dict[str, str] = {}
    if a.exceptions and a.exceptions.exists():
        exc = json.loads(a.exceptions.read_text())
        bad = [k for k, v in exc.items() if not str(v).strip()]
        if bad:
            print(f"ERROR: exceptions without a reason: {bad}", file=sys.stderr)
            return 2

    docs = {p.name: p.read_text() for p in a.docs}
    # \\input{...}/\\include{...} targets that resolve to a generated file -> labels are WIRED
    wired: dict[str, str] = {}
    gen_names = {p.name for p in a.tables.glob("*.tex")}
    for dn, dt in docs.items():
        for tgt in re.findall(r"\\(?:input|include)\{([^}]+)\}", re.sub(r"(?<!\\)%.*", "", dt)):
            base = Path(tgt).name
            base = base if base.endswith(".tex") else base + ".tex"
            if base in gen_names:
                for lb in LABEL.findall((a.tables / base).read_text()):
                    wired.setdefault(lb, dn)
    gen_files = {p.name: p.read_text() for p in a.tables.glob("*.tex")}
    gen_by_label: dict[str, tuple[str, str]] = {}
    for fn, t in gen_files.items():
        for lb in LABEL.findall(t):
            gen_by_label.setdefault(lb, (fn, t))

    rows, fail = [], 0
    labels = sorted({lb for t in docs.values() for lb in LABEL.findall(t)} | set(wired))
    stale = [k for k in exc if k not in labels]
    if stale:
        print(f"WARNING: exceptions for labels not found in any doc: {stale}", file=sys.stderr)
    for lb in labels:
        if lb in wired and not any("\\label{%s}" % lb in t for t in docs.values()):
            rows.append((lb, wired[lb], "WIRED", None, [], "")); continue
        dname, env = next(((n, _table_env(t, lb)) for n, t in docs.items() if "\\label{%s}" % lb in t), (None, ""))
        if lb not in gen_by_label:
            st, frac, miss = "NOSOURCE", None, []
        else:
            fn, gt = gen_by_label[lb]
            frac, miss = compare(numbers(_table_env(gt, lb)), numbers(env))
            st = "MATCH" if frac == 1.0 else ("PARTIAL" if frac >= 0.8 else "DIFF")
        if lb in exc and st in ("PARTIAL", "DIFF", "NOSOURCE"):
            st = "UNVERIFIED" if str(exc[lb]).strip().upper().startswith("UNVERIFIED") else "EXCEPTION"
        if st in ("PARTIAL", "DIFF") or (st == "NOSOURCE" and a.strict_nosource):
            fail += 1
        rows.append((lb, dname, st, frac, miss[:4], exc.get(lb, "")))

    counts = {}
    for r in rows:
        counts[r[2]] = counts.get(r[2], 0) + 1
    lines = ["| label | doc | status | match | first numbers not in paper | exception reason |",
             "|---|---|---|---|---|---|"]
    for lb, dn, st, fr, ms, why in rows:
        lines.append(f"| {lb} | {dn} | {st} | {'' if fr is None else f'{fr:.0%}'} | "
                     f"{', '.join(f'{m:g}' for m in ms)} | {why} |")
    summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    out = "\n".join(lines) + f"\n\n{summary}\n"
    print(out)
    if a.report:
        a.report.write_text(out)
    if a.emit_template:
        todo = {r[0]: "UNVERIFIED: TODO explain or fix" for r in rows
                if r[2] in ("PARTIAL", "DIFF") or (r[2] == "NOSOURCE" and a.strict_nosource)}
        a.emit_template.write_text(json.dumps(todo, indent=2) + "\n")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

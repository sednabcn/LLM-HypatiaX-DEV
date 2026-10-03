#!/usr/bin/env python3
"""
audit_and_fix.py - investigate (and where safe, fix) the six open items in todonext.txt.

Default mode is READ-ONLY: it prints a report and writes audit_report.md.
Add --apply to make edits. Every edited file gets a .bak copy first.

Items
  1  benchmark_results.json mismatch (methods/rows/domains, claims that cite it, merge of per-domain runs)
  2  Item 11 + remaining unverified blocks (Kruskal-Wallis, power analysis, timing, scalability, repro paths)
  3  noise-level "off by 10x" correction vs shard files (noise_level / threshold_used)
  4  bracketed working notes ([Unverified...], [Removed...], [Withdrawn...])
  5  main paper correction notes + unsupported "near-zero extrapolation error"
  6  threshold entries for 2% / 20% noise and the stale print line in the sweep script

Example
  python audit_and_fix.py --supp supp.tex --paper main.tex --bench benchmark_results.json \
      --exp2-dir feynman-tests/exp2 --shards 'noise_shards/*.json' \
      --sweep-script run_noise_sweep_benchmark.py --run-all run_all.sh
  # then, to apply item 4 / 5 / 6 / merge edits:
  python audit_and_fix.py ... --apply --brackets comment --thr-new 0.995
"""
import argparse, glob, json, math, os, re, shutil, sys

EXPECTED_METHODS = 6
EXPECTED_DOMAINS = 11
EXPECTED_NOISE_PCT = [0.0, 0.05, 0.1, 0.5, 1.0]      # corrected levels, percent
OLD_NOISE_PCT = [0.0, 0.5, 1.0, 5.0, 10.0]           # original (suspect) labels
REPORT = []


def out(s=""):
    print(s)
    REPORT.append(s)


def head(n, title):
    out(f"\n## Item {n}: {title}")


def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def backup_write(path, text):
    shutil.copy2(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    out(f"  wrote {path} (backup: {path}.bak)")


def missing(label, path):
    out(f"  SKIPPED - {label} not found: {path!r}. Pass the right path to enable this check.")


def line_of(text, idx):
    return text.count("\n", 0, idx) + 1


# --------------------------------------------------------------------------- Item 1
def collect_tests(d):
    """Return list of tests from a result JSON of shape {'tests':[{'description','results':{method:{...}}}]}."""
    return d.get("tests", []) if isinstance(d, dict) else []


def flat_records(d):
    """Normalise either file shape to flat rows like run_comparative_suite_benchmark_v2.py exports
    (test, domain, method, formula, r2, rmse, runtime, success).
    Accepts the flat benchmark_results.json list, or protocol_core_*.json ({'tests': [...]})."""
    if isinstance(d, list):
        return [r for r in d if isinstance(r, dict)]
    rows = []
    for t in collect_tests(d):
        for m, res in t.get("results", {}).items():
            rows.append({"test": t.get("description", ""), "domain": t.get("domain", ""), "method": m,
                         "formula": res.get("formula", ""), "r2": res.get("r2"), "rmse": res.get("rmse"),
                         "runtime": res.get("time"), "success": res.get("success", False)})
    return rows


def summarize_bench(path):
    rows = flat_records(json.load(open(path)))
    methods = {r.get("method") for r in rows if r.get("method")}
    domains = {r.get("domain") for r in rows if r.get("domain")}
    tests = {r.get("test") for r in rows}
    return tests, methods, domains, len(rows)


MACRO = {"PureLLM Baseline (core)": r"\\PureLLM", "SymbolicEngineWithLLM (tools)": r"\\SEL",
         "HybridDiscoverySystem v50_2 (tools)": r"\\HDS", "EnhancedHybridSystemDeFi (core)": r"\\EHD",
         "ImprovedNN (core)": r"\\INN", "HybridSystemLLMNN all-domains (core)": r"\\HSL"}


def item1(a):
    head(1, "benchmark_results.json mismatch")
    per = {}
    if not a.bench or not os.path.exists(a.bench):
        missing("benchmark_results.json", a.bench)
    else:
        tests, methods, domains, rows = summarize_bench(a.bench)
        out(f"  {a.bench}: {len(tests)} tests, {rows} method-rows, {len(methods)} methods, {len(domains)} domains")
        for m in sorted(methods):
            out(f"    method: {m}")
        rows_all = flat_records(json.load(open(a.bench)))
        per = {}
        for r in rows_all:
            x = r.get("r2")
            ok = isinstance(x, (int, float)) and x == x and x >= a.pass_thr
            e = per.setdefault(r["method"], {"n": 0, "pass": 0, "succ": 0, "bad": 0, "sum": 0.0, "k": 0})
            e["n"] += 1
            e["pass"] += ok
            e["succ"] += bool(r.get("success"))
            e["bad"] += not (isinstance(x, (int, float)) and x == x)
            if isinstance(x, (int, float)) and math.isfinite(x):
                e["sum"] += x
                e["k"] += 1
        out(f"  recomputed from this file (pass = r2 >= {a.pass_thr}):")
        for m, e in sorted(per.items()):
            out(f"    {m}: pass {e['pass']}/{e['n']}, success-flag {e['succ']}/{e['n']}, missing/NaN r2 {e['bad']}, "
                f"mean r2 {e['sum'] / max(e['k'], 1):.6f}")
        out("  compare these with the 29/30 and 27/30 claims listed below; any method whose count differs needs its claim fixed.")
        if len(methods) < EXPECTED_METHODS:
            out(f"  MISMATCH: file has {len(methods)} methods, paper/supplement describe {EXPECTED_METHODS}.")
    if a.exp2_dir and os.path.isdir(a.exp2_dir):
        files = sorted(f for f in glob.glob(os.path.join(a.exp2_dir, a.exp2_glob))
                       if "benchmark_results" not in os.path.basename(f))
        out(f"  per-domain runs in {a.exp2_dir}: {len(files)} files")
        merged, allm = {}, set()   # keyed by (test, method) like the exporter's dedupe; later file wins
        for f in files:
            try:
                t = flat_records(json.load(open(f)))
            except Exception as e:
                out(f"    could not parse {f}: {e}")
                continue
            ms = {r["method"] for r in t}
            allm |= ms
            for r in t:
                merged[(r["test"], r["method"])] = r
            out(f"    {os.path.basename(f)}: {len({r['test'] for r in t})} tests, {len(ms)} methods")
        merged = list(merged.values())
        out(f"  union of methods across per-domain runs: {len(allm)}")
        if len(allm) >= EXPECTED_METHODS:
            out("  -> six-method data exists in the per-domain runs; a merge can regenerate benchmark_results.json.")
        else:
            out("  -> per-domain runs also have <6 methods. The script's --methods default is 'all available (1-6)', "
                "so check exp2_run.log for methods that failed to import, or a run that used --methods 3 4.")
        if a.apply and merged:
            dest = a.merge_out
            json.dump(merged, open(dest, "w"), indent=2, default=str)
            out(f"  wrote merged flat file {dest} ({len(merged)} rows, same schema as the exporter) - "
                "verify before replacing benchmark_results.json.")
    else:
        missing("exp2 per-domain directory (--exp2-dir)", a.exp2_dir)
    log = os.path.join(a.exp2_dir, "exp2_run.log") if a.exp2_dir else None
    if log and os.path.exists(log):
        lt = read(log).splitlines()
        bad = [(i + 1, l.strip()[:140]) for i, l in enumerate(lt)
               if "[SE-TRACE]" not in l and re.search(r"ImportError|ModuleNotFound|not available|unavailable|skipped|Traceback", l, re.I)]
        out(f"  exp2_run.log: {len(bad)} line(s) suggesting a method was skipped/unavailable")
        for ln, s in bad[:15]:
            out(f"    log line {ln}: {s}")
    if a.supp and os.path.exists(a.supp):
        t = read(a.supp)
        pat = re.compile(r"29\s*/\s*30|27\s*/\s*30|benchmark_results\.json|six-method|PureLLM recount", re.I)
        hits = [(line_of(t, m.start()), m.group(0)) for m in pat.finditer(t)]
        out(f"  claims in supplement that depend on the six-method file: {len(hits)}")
        for ln, s in hits[:40]:
            out(f"    line {ln}: {s}")
        out("  If the six-method file can't be produced, rewrite each of these to cite only the M3/M4 file.")
        if per:
            out("  attribution check: each 'N/30' is compared with the last method macro (\\PureLLM, \\SEL, \\HDS, \\EHD) before it:")
            flagged = 0
            for m in re.finditer(r"(\d+)\s*/\s*30", t):
                before = t[max(0, m.start() - 150):m.start()]
                last = None
                for meth, mac in MACRO.items():
                    for mm in re.finditer(mac + r"(?![A-Za-z])", before):
                        if last is None or mm.start() > last[0]:
                            last = (mm.start(), meth, mac)
                if last and last[1] in per and int(m.group(1)) != per[last[1]]["pass"]:
                    flagged += 1
                    out(f"    line {line_of(t, m.start())}: {last[2].replace(chr(92)*2, chr(92))} then {m.group(0)}, "
                        f"but this file gives {per[last[1]]['pass']}/30 - check (may be a different table/threshold)")
            out(f"  {flagged} claim(s) to check")
    if a.paper and os.path.exists(a.paper):
        t = read(a.paper)
        for m in re.finditer(r"benchmark_results\.json|random[- ]split", t, re.I):
            out(f"    main paper line {line_of(t, m.start())}: {m.group(0)}")


# --------------------------------------------------------------------------- Item 2
UNVERIFIED_PATTERNS = {
    "Kruskal-Wallis": r"Kruskal",
    "power analysis": r"power analysis|statistical power|\bpower\s*=",
    "timing": r"\btiming\b|wall[- ]clock|runtime",
    "scalability": r"scalab",
    "manually entered": r"manually entered|hand[- ]entered|entered manually",
    "reproduction commands": r"python3?\s+\S+\.py|bash\s+\S+\.sh|\./run_all\.sh",
}


def item2(a):
    head(2, "Item 11 and remaining unverified content")
    if not (a.supp and os.path.exists(a.supp)):
        missing("supplement .tex", a.supp)
        return
    t = read(a.supp)
    m = re.search(r"[Ii]tem[~\s\\]*11\b|item11\b", t)
    if not m:
        for lab, pth in (("main paper", a.paper), ("routing report", a.routing), ("post-submission report", a.post)):
            if pth and os.path.exists(pth):
                t2 = read(pth)
                m2 = re.search(r"[Ii]tem[~\s\\]*11\b|item11\b", t2)
                out(f"  Item 11 in {lab} ({pth}): " + (f"line {line_of(t2, m2.start())}" if m2 else "not found"))
                if m2:
                    out("    " + t2[m2.start():m2.start() + 400].replace("\n", "\n    "))
                    m = True
                    break
            else:
                out(f"  ({lab} not available to search)")
    if m is True:
        pass
    elif m:
        out(f"  'Item 11' first appears at line {line_of(t, m.start())}:")
        out("    " + t[m.start():m.start() + 300].replace("\n", "\n    "))
        out("  (paste this block back to Claude to have the list updated)")
    else:
        out("  no 'Item 11' string found in the supplement; send the text of the list.")
    for label, pat in UNVERIFIED_PATTERNS.items():
        hits = [line_of(t, x.start()) for x in re.finditer(pat, t, re.I)]
        out(f"  {label}: {len(hits)} hit(s) at lines {hits[:15]}{' ...' if len(hits) > 15 else ''}")
    # reproduction-command paths: do they exist relative to --root?
    root = a.root or "."
    seen = set()
    t = t.replace("\\_", "_")   # LaTeX-escaped underscores, else paths split at every underscore
    for x in re.finditer(r"[\w./-]+\.(?:py|sh|json)\b", t):
        p = x.group(0)
        if p in seen or p.startswith("http") or "TIMESTAMP" in p or p.startswith("_") or p.startswith(".."):
            continue
        seen.add(p)
        if not (os.path.exists(os.path.join(root, p)) or glob.glob(os.path.join(root, "**", os.path.basename(p)), recursive=True)):
            out(f"  path not found under {root!r}: {p} (line {line_of(t, x.start())})")


# --------------------------------------------------------------------------- Item 3
def find_key(obj, key):
    """Yield every value stored under `key` anywhere in a nested JSON structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            yield from find_key(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from find_key(v, key)


def item3(a):
    head(3, "noise-level 'off by 10x' correction vs shard files")
    files = sorted(glob.glob(a.shards)) if a.shards else []
    if not files:
        missing("noise shard files (--shards glob)", a.shards)
        return
    obs = {}
    for f in files:
        try:
            d = json.load(open(f))
        except Exception as e:
            out(f"  could not parse {f}: {e}")
            continue
        nl = sorted({round(float(x), 6) for x in find_key(d, "noise_level") if isinstance(x, (int, float))})
        if not nl and isinstance(d, dict) and d.get("noise_levels") is not None:
            raw = d["noise_levels"]
            out(f"    stored noise_levels = {raw}; per_noise keys = {list(d.get('per_noise', {}))[:10]}")
            nl = sorted({round(float(x), 6) for x in (raw if isinstance(raw, list) else [raw])
                         if isinstance(x, (int, float))})
        if not nl:
            alt = {}
            def walk(o, depth=0):
                if isinstance(o, dict):
                    for k, v in o.items():
                        if re.search(r"noise|sigma", str(k), re.I) and isinstance(v, (int, float)):
                            alt.setdefault(k, set()).add(round(float(v), 6))
                        walk(v, depth + 1)
                elif isinstance(o, list):
                    for v in o:
                        walk(v, depth + 1)
            walk(d)
            if alt:
                out(f"    (no noise_level key in {os.path.basename(f)}; using {sorted(alt)})")
                nl = sorted(set().union(*alt.values()))
            else:
                top = list(d)[:12] if isinstance(d, dict) else f"list[{len(d)}]"
                out(f"    top-level structure of {os.path.basename(f)}: {top}")
        th = sorted({float(x) for x in find_key(d, "threshold_used") if isinstance(x, (int, float))})
        out(f"  {os.path.basename(f)}: noise_level={nl} threshold_used={th}")
        for n in nl:
            obs.setdefault(n, set()).update(th)
    levels = sorted(obs)
    if not levels:
        out("  no noise_level fields found.")
        return
    # shard stores a fraction (script divides percent by 100); compare as percent
    pct = [round(x * 100, 6) for x in levels]
    pct_if_pct = [round(x, 6) for x in levels]
    out(f"  observed levels as fractions->percent: {pct}")
    for name, ref in (("corrected", EXPECTED_NOISE_PCT), ("original", OLD_NOISE_PCT)):
        if all(any(abs(p - r) < 1e-6 for r in ref) for p in pct):
            out(f"  -> consistent with the {name} labels {ref} if noise_level is a FRACTION.")
        if all(any(abs(p - r) < 1e-6 for r in ref) for p in pct_if_pct):
            out(f"  -> consistent with the {name} labels {ref} if noise_level is already in PERCENT.")
    out("  threshold per level:")
    for n in levels:
        out(f"    {n}: {sorted(obs[n])}")
    ths = {t for v in obs.values() for t in v}
    if len(ths) > 1:
        out(f"  Multiple thresholds in use {sorted(ths)} - add a sentence in Appendix H explaining why the two noise tables differ.")


# --------------------------------------------------------------------------- Item 4
def find_bracket_blocks(text, starts=("Unverified", "Removed", "Withdrawn")):
    """Return (start, end, kind) for each balanced [Kind ...] block."""
    blocks = []
    for m in re.finditer(r"\[(" + "|".join(starts) + r")\b", text):
        depth, i = 0, m.start()
        while i < len(text):
            c = text[i]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    blocks.append((m.start(), i + 1, m.group(1)))
                    break
            i += 1
    return blocks


def host_span(t, s, e):
    r"""Smallest construct that carries an unverified/removed note: the enclosing table, \item, or \paragraph."""
    b = max(t.rfind("\\begin{table}", 0, s), t.rfind("\\begin{table*}", 0, s))
    if b != -1 and not re.search(r"\\end\{table\*?\}", t[b:s]):
        m = re.compile(r"\\end\{table\*?\}").search(t, e)
        if m:
            return b, m.end(), "table"
    i = t.rfind("\\item", 0, s)
    if i != -1 and "\\end{itemize}" not in t[i:s] and "\\end{enumerate}" not in t[i:s]:
        m = re.compile(r"\\item\b|\\end\{(?:itemize|enumerate)\}").search(t, e)
        if m:
            return i, m.start(), "item"
    pg = t.rfind("\\paragraph{", 0, s)
    if pg != -1 and not re.search(r"\\(?:paragraph|subsection|section)\{", t[pg + 11:s]):
        m = re.compile(r"\\(?:paragraph|subsection|section|begin\{table)").search(t, e)
        if m:
            return pg, m.start(), "paragraph"
    return None


def drop_hosts(t, blocks):
    spans = {}
    for s_, e_, k in blocks:
        hs = host_span(t, s_, e_)
        spans[hs[:2] if hs else (s_, e_)] = hs[2] if hs else "note only"
    keys = sorted(spans)
    kept = [k for k in keys if not any(o != k and o[0] <= k[0] and k[1] <= o[1] for o in keys)]
    new = t
    for b, e in reversed(kept):
        new = new[:b] + "\\iffalse % audit: removed (" + spans[(b, e)] + ") - unverified content\n" + new[b:e] + "\n\\fi " + new[e:]
    removed_text = "".join(t[b:e] for b, e in kept)
    labels = re.findall(r"\\label\{([^}]+)\}", removed_text)
    rest = t
    for b, e in reversed(kept):   # blank out (keep newlines) so reported line numbers stay original
        rest = rest[:b] + re.sub(r"[^\n]", " ", rest[b:e]) + rest[e:]
    dangling = []
    for lab in labels:
        for m in re.finditer(r"\\(?:[Cc]ref|ref|autoref|eqref)\{[^}]*" + re.escape(lab) + r"[^}]*\}", rest):
            dangling.append((line_of(rest, m.start()), lab))
    return new, [(line_of(t, b), spans[(b, e)], e - b) for b, e in kept], labels, dangling


def item4(a):
    head(4, "bracketed working notes")
    if not (a.supp and os.path.exists(a.supp)):
        missing("supplement .tex", a.supp)
        return
    t = read(a.supp)
    blocks = find_bracket_blocks(t)
    out(f"  {len(blocks)} block(s) found")
    for s, e, k in blocks:
        out(f"    line {line_of(t, s)} [{k}] {len(t[s:e])} chars")
        out("      before: " + re.sub(r"\s+", " ", t[max(0, s - 200):s]).strip()[-200:])
        out("      block : " + re.sub(r"\s+", " ", t[s:e])[:600])
        out("      after : " + re.sub(r"\s+", " ", t[e:e + 200]).strip())
    if a.apply and blocks:
        if a.brackets == "delete":
            new = t
            for s, e, _ in reversed(blocks):
                new = new[:s] + new[e:]
        elif a.brackets == "comment":
            new = t
            for s, e, _ in reversed(blocks):
                body = "\n".join("% " + ln for ln in t[s:e].splitlines())
                new = new[:s] + "% --- working note moved out of the text ---\n" + body + "\n" + new[e:]
        elif a.brackets == "drop-host":
            new, kept, labels, dangling = drop_hosts(t, blocks)
            out("  wrapped in \\iffalse...\\fi (undo by deleting the two marker lines):")
            for ln, kind, n in kept:
                out(f"    line {ln}: {kind}, {n} chars")
            out(f"  labels removed: {labels}")
            for ln, lab in dangling:
                out(f"  DANGLING reference to removed label {lab!r} at (original) line {ln} - fix this sentence")
            out("  recompile and compare error and page counts before/after, as Item 11 does.")
        else:
            out("  --brackets must be 'comment', 'delete' or 'drop-host' to apply.")
            return
        backup_write(a.supp, new)
    elif blocks:
        out("  (--apply --brackets comment hides only the note; 'drop-host' also hides the table/item/paragraph it sits in; both reversible)")


# --------------------------------------------------------------------------- Item 5
PHRASE = re.compile(r"near[- ]zero extrapolation error", re.I)
NOTE = re.compile(r"\[(?:Correct(?:ed|ion)|Note|Fixed|Updated|Revised)[^\]]*\]|\\todo\{|%\s*(?:CORRECTION|FIXME|TODO)", re.I)


def item5(a):
    head(5, "main paper correction notes and unsupported phrase")
    for label, p in (("supplement", a.supp), ("main paper", a.paper)):
        if not (p and os.path.exists(p)):
            missing(label, p)
            continue
        t = read(p)
        out(f"  {label}: {len(NOTE.findall(t))} correction-style note(s) (review by hand; stripping them automatically is unsafe)")
        for m in NOTE.finditer(t):
            out(f"    line {line_of(t, m.start())}: {t[m.start():m.start()+80].replace(chr(10), ' ')}")
        for m in PHRASE.finditer(t):
            out(f"    UNSUPPORTED PHRASE line {line_of(t, m.start())}: ...{t[max(0,m.start()-60):m.end()+40]!r}")
        if a.apply and PHRASE.search(t):
            def fix(m):
                ctx = t[max(0, m.start() - 60):m.start()].lower()
                if re.search(r"earlier draft|previously|claimed|withdrawn|\[correct|removed", ctx):
                    out(f"  left alone (looks like a retraction note quoting the claim): line {line_of(t, m.start())}")
                    return m.group(0)
                return "% UNSUPPORTED (flagged by audit): " + m.group(0) + "\n" + "extrapolation error as reported in the tables"
            new = PHRASE.sub(fix, t)
            out("  replacing phrase with neutral wording; review the sentence by hand.")
            backup_write(p, new)


# --------------------------------------------------------------------------- Item 6
def item6(a):
    head(6, "threshold entries and stale print line in the sweep script")
    for label, p in (("sweep script", a.sweep_script), ("run_all.sh", a.run_all)):
        if not (p and os.path.exists(p)):
            missing(label, p)
    if a.sweep_script and os.path.exists(a.sweep_script):
        t = read(a.sweep_script)
        stale = re.search(r"threshold\s*0\.9999\s*/\s*0\.995", t)
        if stale:
            out(f"  stale print line at line {line_of(t, stale.start())}: {t[stale.start()-40:stale.end()+40]!r}")
        # find threshold maps keyed by noise fractions, e.g. {0.0005: 0.99, 0.001: 0.99, 0.01: 0.995}
        dicts = list(re.finditer(r"(\w*[Tt][Hh][Rr][Ee][Ss][Hh]\w*)\s*=\s*\{([^{}]*)\}", t, re.S))
        for m in dicts:
            out(f"  threshold map '{m.group(1)}' at line {line_of(t, m.start())}: "
                + re.sub(r"\s+", " ", m.group(2)).strip())
        for lvl in ("0.02", "0.2", "0.20"):
            out(f"  entry for {lvl} present: {'yes' if re.search(r'(?<![0-9.])' + re.escape(lvl) + r'\s*:', t) else 'NO'}")
        if a.apply:
            if a.thr_new is None:
                out("  --thr-new is required to apply: choose ONE threshold for 2% and 20% so runner and aggregator agree "
                    "(runner uses 0.95, aggregator uses 0.995).")
            elif dicts:
                m = dicts[0]
                body = m.group(2)
                add = ""
                for lvl in ("0.02", "0.2"):
                    if not re.search(r"(?<![0-9.])" + re.escape(lvl) + r"\s*:", body):
                        add += f"    {lvl}: {a.thr_new},\n"
                new = t[:m.end(2)] + ("\n" if not body.rstrip().endswith(",") and body.strip() else "") + add + t[m.end(2):]
                new = new.replace("threshold 0.9999 / 0.995", f"thresholds per noise level (see threshold map; 2%/20% = {a.thr_new})")
                backup_write(a.sweep_script, new)
                out("  NOTE: also add the same entries to the aggregation script's map - the two must match.")
            else:
                out("  no threshold dict matched; patch by hand using the line numbers above.")
    if a.run_all and os.path.exists(a.run_all):
        t = read(a.run_all)
        m = re.search(r"NOISE_LEVELS=\S+", t)
        if m:
            out(f"  run_all.sh line {line_of(t, m.start())}: {m.group(0)}")
            out("  the sweep script reads singular NOISE_LEVEL (percent) or --noise-levels (fractions); "
                "plural NOISE_LEVELS is ignored. Launch one task per level or pass --noise-levels.")


# --------------------------------------------------------------------------- Item 7
def item7(a):
    head(7, "are the source files tracked in git (what CI sees)?")
    import subprocess
    root = a.repo or "."

    def git(*args):
        return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)
    try:
        if git("rev-parse", "--is-inside-work-tree").returncode != 0:
            out(f"  SKIPPED - {root!r} is not a git work tree.")
            return
    except FileNotFoundError:
        out("  SKIPPED - git not installed.")
        return
    targets = []
    if a.bench:
        targets.append(a.bench)
    if a.exp2_dir:
        targets += sorted(glob.glob(os.path.join(a.exp2_dir, "protocol_core_noiseless_*.json")))
    if a.shards:
        targets += sorted(glob.glob(a.shards))
        targets += sorted(glob.glob(os.path.join(os.path.dirname(a.shards), "protocol_core_noiseless_*.json")))
    bad = 0
    for tp in dict.fromkeys(targets):
        if not os.path.exists(tp):
            continue
        rel = os.path.relpath(tp, root)
        if git("ls-files", "--error-unmatch", rel).returncode == 0:
            continue
        ig = git("check-ignore", "-v", rel)
        why = "IGNORED by " + ig.stdout.strip() if ig.returncode == 0 else "UNTRACKED (not ignored)"
        out(f"  not in git: {rel} - {why}")
        bad += 1
    out(f"  checked {len(set(targets))} file(s); {bad} not tracked. CI regenerates tables from tracked files only, "
        "so untracked sources explain 'no result file in the repository' and 'incomplete sweep' rows.")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--supp", help="supplement .tex")
    ap.add_argument("--paper", help="main paper .tex")
    ap.add_argument("--bench", help="benchmark_results.json")
    ap.add_argument("--exp2-dir", help="feynman-tests/exp2 directory of per-domain JSONs")
    ap.add_argument("--exp2-glob", default="protocol_core_noiseless_*.json",
                    help="which files in --exp2-dir are per-domain runs (excludes benchmark_results*.json)")
    ap.add_argument("--pass-thr", type=float, default=0.999999, help="R2 pass threshold for noiseless claims")
    ap.add_argument("--post", help="post_submission_report.tex (searched for Item 11 too)")
    ap.add_argument("--routing", help="supp_routing_improvements.tex (searched for Item 11 too)")
    ap.add_argument("--repo", help="repo root; fills in every path below from the HypatiaX layout")
    ap.add_argument("--merge-out", default="benchmark_results_merged.json")
    ap.add_argument("--shards", help="glob of noise shard JSONs")
    ap.add_argument("--sweep-script", help="run_noise_sweep_benchmark.py")
    ap.add_argument("--run-all", help="run_all.sh")
    ap.add_argument("--root", help="project root for checking reproduction-command paths")
    ap.add_argument("--apply", action="store_true", help="make edits (backups are written)")
    ap.add_argument("--brackets", choices=["comment", "delete", "drop-host"], default="comment")
    ap.add_argument("--thr-new", type=float, help="threshold to use for 2%% and 20%% noise")
    ap.add_argument("--only", type=int, nargs="*", help="run only these item numbers")
    a = ap.parse_args()
    if a.repo:
        r = a.repo
        fs = os.path.join(r, "hypatiax/data/results/comparison_results/feynman-tests")
        a.supp = a.supp or os.path.join(r, "paper/supp_benchmark_report.tex")
        a.paper = a.paper or os.path.join(r, "paper/jmlr_paper_main.tex")
        a.post = a.post or os.path.join(r, "paper/post_submission_report.tex")
        a.routing = a.routing or os.path.join(r, "paper/supp_routing_improvements.tex")
        a.exp2_dir = a.exp2_dir or os.path.join(fs, "exp2")
        a.bench = a.bench or os.path.join(a.exp2_dir, "benchmark_results.json")
        a.shards = a.shards or os.path.join(fs, "noise-sweep", "noise_sweep_*_nshards*.json")
        a.sweep_script = a.sweep_script or os.path.join(r, "hypatiax/experiments/benchmarks/run_noise_sweep_benchmark.py")
        a.run_all = a.run_all or os.path.join(r, "run_all.sh")
        a.root = a.root or r
        if a.merge_out == "benchmark_results_merged.json":
            a.merge_out = os.path.join(a.exp2_dir, "benchmark_results_merged.json")
    items = {1: item1, 2: item2, 3: item3, 4: item4, 5: item5, 6: item6, 7: item7}
    out("# Audit report" + (" (APPLY mode)" if a.apply else " (read-only)"))
    for n, fn in items.items():
        if not a.only or n in a.only:
            fn(a)
    with open("audit_report.md", "w") as f:
        f.write("\n".join(REPORT) + "\n")
    print("\nreport saved to audit_report.md")


if __name__ == "__main__":
    main()

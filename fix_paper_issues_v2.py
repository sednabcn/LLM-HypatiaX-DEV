#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_paper_issues.py -- resolve three open issues in the HypatiaX paper sources.

  Issue 1  Abstract / contributions / conclusion / supplement say the result files
           "do not record whether the single-axis or PCA-directed variant was used".
           -> Recompute the Table-tab:main_results metrics from the seed-42 file AND the
              five 74-task PCA files, validate the loader against the numbers already
              published in the paper, generate a PCA accuracy table (tab:pca_accuracy),
              gather evidence on the seed-42 file's split variant, and rewrite the caveat.
              Also fixes two statements that contradict the restored PCA files.

  Issue 2  Mann-Whitney appendix says "medium (2x)"; the code uses the far regime.
           -> Text patch (appendix + the cross-reference in the main paper), plus an
              optional recomputation from exp1_five_results.json that shows which regime
              reproduces U=74.0 / medians 2415.2 and 13283.9.

  Issue 3  tab:arch, tab:noise_sensitivity, tab:portfolio_seed_sweep are tagged Unverified.
           -> Regenerate each from its source files when they can be found and the numbers
              match (then drop the tag).  Otherwise the tag is NARROWED to what has
              actually been checked; it is never silently removed.

Safety: the original .tex files are never modified.  Output goes to --out:
    fix_report.md        what was found, applied, and what is still open
    changes.diff         unified diff of every text change
    generated/*.tex      regenerated tables
    patched/*.tex        patched copies (only with --apply)

Typical use:
    python fix_paper_issues.py --tex-dir paper/ --repo ~/hypatiax            # dry run
    python fix_paper_issues.py --tex-dir paper/ --repo ~/hypatiax --apply    # write patched/

Requires: python>=3.9, numpy, scipy.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

MAIN = "jmlr_paper_main_cleaned.tex"
SUPP = "supp_benchmark_report_cleaned.tex"
PAPER_V3_MD5 = "cbf503d0b3985b4c1bccf7ed18e0316b"   # seed-42 file cited in the paper
NOISE_LEVELS = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]

REPORT: list[str] = []


def say(msg: str = "") -> None:
    REPORT.append(msg)
    print(msg)


# ----------------------------------------------------------------------------- utils
def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def skeleton(o, depth=3, _d=0):
    """Short structural summary, used in error messages."""
    if _d >= depth:
        return type(o).__name__
    if isinstance(o, dict):
        return {k: skeleton(v, depth, _d + 1) for k, v in list(o.items())[:12]}
    if isinstance(o, list):
        return [skeleton(o[0], depth, _d + 1), f"... {len(o)} items"] if o else []
    return type(o).__name__


def tt(s: str) -> str:
    """Escape for \\texttt{}."""
    return s.replace("\\", "/").replace("_", r"\_").replace("%", r"\%").replace("#", r"\#").replace("&", r"\&")


def num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def dig(rec, path, default=None):
    cur = rec
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def sysdig(rec, path, default=None):
    """dig() for per-system fields: schema is record["results"][<system>][<field>]; fall back to top level."""
    v = dig(rec, "results." + path)
    return v if v is not None else dig(rec, path, default)


def walk(o, path=()):
    """Yield (path_tuple, dict) for every dict in a JSON tree."""
    if isinstance(o, dict):
        yield path, o
        for k, v in o.items():
            yield from walk(v, path + (str(k),))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, path + (str(i),))


def task_records(obj) -> list[dict]:
    """Locate the per-task record list of a DeFi result file."""
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        return obj
    if isinstance(obj, dict):
        best = None
        for k, v in obj.items():
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                cand = v
            elif isinstance(v, dict) and len(v) >= 20 and all(isinstance(x, dict) for x in v.values()):
                cand = [dict(x, _task=k2) for k2, x in v.items()]
            else:
                continue
            if best is None or len(cand) > len(best):
                best = cand
        if best:
            return best
        if len(obj) >= 20 and all(isinstance(x, dict) for x in obj.values()):
            return [dict(x, _task=k) for k, x in obj.items()]
    raise ValueError("cannot locate per-task records; structure is " + json.dumps(skeleton(obj))[:600])


# ------------------------------------------------------------------ patch engine
class Doc:
    def __init__(self, tex_dir: Path):
        self.orig = {n: (tex_dir / n).read_text(encoding="utf-8") for n in (MAIN, SUPP)}
        self.text = dict(self.orig)
        self.edits: list[tuple[str, str]] = []

    def sub(self, fname, pattern, repl, tag, *, regex=False, flags=0, marker=None, all_=False):
        t = self.text[fname]
        if marker and marker in t:
            self.edits.append((tag, "already applied"))
            return False
        if regex:
            new, n = re.subn(pattern, lambda m: repl, t, count=0 if all_ else 1, flags=flags)
        else:
            n = t.count(pattern)
            new = t.replace(pattern, repl, -1 if all_ else 1) if n else t
        if n == 0:
            self.edits.append((tag, "ANCHOR NOT FOUND - not changed"))
            return False
        self.text[fname] = new
        self.edits.append((tag, "applied" + (f" x{n}" if n > 1 and all_ else "")))
        return True


# ============================================================ ISSUE 2: regime label
REGIME_OLD = r"For medium extrapolation regime (2$\times$ training range):"
REGIME_NEW = (r"For the far extrapolation regime ($5\times$ training range; extrapolation error "
              r"$=100\cdot\mathrm{extrap\_rmse\_far}/\mathrm{train\_rmse}$ per equation, "
              r"finite values only):")


def regime_from_data(path: Path) -> None:
    """Recompute the Mann-Whitney test for near/medium/far and show which one the appendix reports."""
    say(f"\n  Data check: {path}")
    data = load_json(path)
    rows = []
    for p, rec in walk(data):
        if "train_rmse" in rec and any(k.startswith("extrap_rmse_") for k in rec):
            hay = " ".join(p + tuple(str(rec.get(k, "")) for k in ("system", "method", "name", "model"))).lower()
            sysname = "hybrid" if "hybrid" in hay else "nn" if ("neural" in hay or re.search(r"\bnn\b", hay)) else None
            if sysname:
                rows.append((sysname, rec))
    if not rows:
        say("  Could not find per-equation records with train_rmse / extrap_rmse_*; structure: "
            + json.dumps(skeleton(data))[:500])
        return
    say("  regime | n_hyb n_nn | median_hyb  median_nn |     U   p(one-tailed, hyb<nn)")
    hits = []
    for regime in ("near", "medium", "far"):
        vals = {"hybrid": [], "nn": []}
        for sysname, rec in rows:
            r, t = num(rec.get(f"extrap_rmse_{regime}")), num(rec.get("train_rmse"))
            if math.isfinite(r) and math.isfinite(t) and t > 0:
                vals[sysname].append(100 * r / t)
        h, n = np.array(vals["hybrid"]), np.array(vals["nn"])
        if len(h) < 2 or len(n) < 2:
            say(f"  {regime:6s} | too few finite values (hyb {len(h)}, nn {len(n)})")
            continue
        res = stats.mannwhitneyu(h, n, alternative="less", method="asymptotic")
        say(f"  {regime:6s} | {len(h):5d} {len(n):4d} | {np.median(h):10.1f} {np.median(n):10.1f} | "
            f"{res.statistic:6.1f}   {res.pvalue:.4f}")
        if abs(res.statistic - 74.0) < 0.6 and (len(h), len(n)) == (12, 15):
            hits.append(regime)
    say(f"  Appendix reports: n=12/15, medians 2415.2 / 13283.9, U=74.0, p=0.225.  "
        f"Regime reproducing U and n: {hits or 'NONE - check the file/regime mapping'}")


def fix_regime(doc: Doc, args) -> None:
    say("\n## Issue 2 - Mann-Whitney regime label")
    say("Paper-internal evidence for the far regime:\n"
        "  * the appendix medians (2415.2 % Hybrid, 13283.9 % NN) are the same numbers as the main paper's\n"
        "    tab:five_systems_full, whose caption defines the metric as 100*extrap_rmse_far/train_rmse;\n"
        "  * n=12 / n=15 equal the finite-value counts that table reports before its Tukey-fence\n"
        "    trimming ('Hybrid keeps 10 of 12, Neural Network 12 of 15');\n"
        "  * U=74.0 with n=12/15 gives one-tailed p=0.2247 (normal approximation, continuity\n"
        "    correction), matching the appendix's 2.25e-01, so those three numbers are mutually consistent.\n"
        "  * the 2x multiplier is the medium regime (supplement S2 lists 1.2x, 2x, 5x); the far regime is 5x.")
    doc.sub(SUPP, REGIME_OLD, REGIME_NEW, "supp: Mann-Whitney regime label (medium 2x -> far 5x)",
            marker="For the far extrapolation regime")
    doc.sub(MAIN, "medium-regime Mann--Whitney test", "far-regime Mann--Whitney test",
            "main: cross-reference to the Mann-Whitney test (medium -> far)",
            marker="far-regime Mann--Whitney test")
    if args.exp1_json:
        regime_from_data(Path(args.exp1_json))
    elif args.repo:
        cands = sorted(Path(args.repo).rglob("exp1_five_results.json"))
        if cands:
            regime_from_data(cands[0])
        else:
            say("  exp1_five_results.json not found under --repo; data check skipped (pass --exp1-json).")


# ================================================== ISSUE 1: split variant / PCA accuracy
def parse_main_results(tex: str) -> dict:
    block = re.search(r"\\label\{tab:main_results\}.*?\\end\{tabular\}", tex, re.S)
    if not block:
        return {}
    rows = {}
    for line in block.group(0).splitlines():
        m = re.match(r"\s*(Pure LLM|Neural MLP|HypatiaX \(uncorrected\)|HypatiaX \(corrected\))\s*&(.*)\\\\", line)
        if m:
            cells = [c.strip().strip("$") for c in m.group(2).split("&")]
            try:
                rows[m.group(1)] = [float(c) for c in cells[:5]]
            except ValueError:
                pass
    return rows


def arm_r2(recs: list[dict]) -> dict:
    llm = np.array([num(sysdig(r, "pure_llm.test_r2")) for r in recs])
    nn = np.array([num(sysdig(r, "neural_network.test_r2")) for r in recs])
    hyb = np.array([num(sysdig(r, "hybrid.test_r2")) for r in recs])
    corr = []
    for i, r in enumerate(recs):
        d = sysdig(r, "hybrid.decision")
        corr.append(llm[i] if d == "llm" else nn[i] if d in ("nn", "nn_fallback") else float("nan"))
    return {"Pure LLM": llm, "Neural MLP": nn, "HypatiaX (uncorrected)": hyb, "HypatiaX (corrected)": np.array(corr)}


def metrics(r2: np.ndarray) -> dict:
    """Paper rules: clip to [-10,1] over non-NaN for median/mean; fixed denominator n for the rates."""
    n, ok = len(r2), ~np.isnan(r2)
    v = np.clip(r2[ok], -10, 1)
    return dict(n=n, median=float(np.median(v)) if v.size else float("nan"),
                mean=float(v.mean()) if v.size else float("nan"),
                p099=100.0 * float(np.sum(r2[ok] > 0.99)) / n, p09=100.0 * float(np.sum(r2[ok] > 0.9)) / n,
                cata=int(np.sum(r2[ok] < -10)))


def validate_loader(m_by_arm: dict, ref: dict) -> list[str]:
    bad = []
    for name, vals in ref.items():
        m = m_by_arm.get(name)
        if m is None:
            bad.append(f"{name}: missing")
            continue
        got = [m["median"], m["mean"], m["p099"], m["p09"], m["cata"]]
        tol = [1e-3, 1e-3, 0.06, 0.06, 0.5]
        for lab, g, r, t in zip(["median", "mean", ">0.99", ">0.9", "catastrophic"], got, vals, tol):
            if not (abs(g - r) <= t):
                bad.append(f"{name} {lab}: file gives {g:.4f}, paper has {r}")
    return bad


def explicit_variant(obj) -> str | None:
    """A variant name stored INSIDE the records (key containing split/variant/protocol)."""
    vals = set()
    for _, d in walk(obj):
        for k, v in d.items():
            if isinstance(v, str) and re.search(r"split|variant|protocol", str(k), re.I):
                vals.add(v.lower())
    if not vals:
        return None
    if all("pca" in v for v in vals):
        return "pca"
    if all(re.search(r"v3c|axis|single", v) for v in vals):
        return "v3c"
    return None


def variant_fields(obj, top=12) -> list[tuple[str, str, int]]:
    cnt = Counter()
    pat = re.compile(r"pca|axis|split|primary[_ ]?var|variant|v3c|protocol", re.I)
    for p, d in walk(obj):
        for k, v in d.items():
            if pat.search(str(k)) and not isinstance(v, (dict, list)):
                cnt[(re.sub(r"\.\d+", "[]", ".".join(p[:1]) + "." + str(k)), str(v)[:50])] += 1
    return [(k, v, c) for (k, v), c in cnt.most_common(top)]


def task_key(r):
    for k in ("task_id", "task", "name", "id", "_task", "equation"):
        if k in r:
            return str(r[k])
    return None


def agreement(a: list[dict], b: list[dict]) -> dict:
    ka, kb = [task_key(r) for r in a], [task_key(r) for r in b]
    if None not in ka and None not in kb and len(set(ka)) == len(ka) and len(set(kb)) == len(kb):
        ia, ib = {k: i for i, k in enumerate(ka)}, {k: i for i, k in enumerate(kb)}
        common = sorted(set(ia) & set(ib))
        pairs = [(a[ia[k]], b[ib[k]]) for k in common]
    else:
        pairs = list(zip(a, b))
    out = {}
    for arm in ("pure_llm", "neural_network", "hybrid"):
        x = np.array([num(sysdig(p, f"{arm}.test_r2")) for p, _ in pairs])
        y = np.array([num(sysdig(q, f"{arm}.test_r2")) for _, q in pairs])
        same = (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=1e-9, atol=1e-12)
        out[arm] = float(same.mean()) if len(same) else float("nan")
    return out


def find_pca_files(repo: Path, explicit: list[str] | None) -> list[tuple[int, Path, list[dict]]]:
    cands = [Path(p) for p in explicit] if explicit else \
        [p for p in repo.rglob("*.json") if any("defi_pca" in part for part in p.parts)]
    found, skipped = [], []
    for p in sorted(cands):
        try:
            recs = task_records(load_json(p))
        except Exception as e:                       # noqa: BLE001
            skipped.append(f"{p.name}: unreadable ({str(e)[:80]})")
            continue
        if len(recs) != 74:
            skipped.append(f"{p.name}: {len(recs)} records (not a 74-task file)")
            continue
        s = recs[0].get("seed")
        if s is None:
            m = re.search(r"seed[_-]?(\d+)", p.name)
            s = int(m.group(1)) if m else None
        if s is None:
            skipped.append(f"{p.name}: no seed in records or filename")
            continue
        found.append((int(s), p, recs))
    for s in skipped:
        say(f"  skipped: {s}")
    return found


def pca_table_tex(v3row, seed_rows, pooled) -> str:
    def cells(label, n, m):
        cata = "/".join(str(m[a]["cata"]) for a in ("Pure LLM", "Neural MLP", "HypatiaX (uncorrected)", "HypatiaX (corrected)"))
        pct = " & ".join("%.1f" % m[a]["p099"] for a in ("Pure LLM", "Neural MLP", "HypatiaX (uncorrected)", "HypatiaX (corrected)"))
        return "%s & %d & %s & %s \\\\" % (label, n, pct, cata)
    L = [r"\begin{table}[h]", r"\centering", r"\small",
         r"\caption{Near-perfect success rate ($\Rsq>0.99$, \%, fixed denominator $n$, \texttt{NaN} counts as a miss) "
         r"and catastrophic-failure counts ($\Rsq<-10$) on the five 74-task PCA-directed seed files "
         r"(\texttt{defi\_pca/}, \texttt{defi\_pca/multi-seeds/}), regenerated by \texttt{fix\_paper\_issues.py} with the "
         r"metric rules of \S\ref{sec:setup_metrics}. ``Corr.'' replaces each task's hybrid $\Rsq$ by that of the "
         r"sub-method named by \texttt{hybrid.decision}. The first row is the seed-42 file behind "
         r"Table~\ref{tab:main_results}, whose records do not name their split variant; it is shown for comparison.}",
         r"\label{tab:pca_accuracy}",
         r"\begin{tabular}{lrrrrrl}", r"\toprule",
         r"Run & $n$ & Pure LLM & Neural MLP & HypatiaX (unc.) & HypatiaX (corr.) & Catastrophic (LLM/NN/unc./corr.) \\",
         r"\midrule", cells("seed42, file named v3", 74, v3row), r"\midrule"]
    for s, m in seed_rows:
        L.append(cells("PCA seed%d" % s, 74, m))
    L += [r"\midrule", r"\rowcolor{black!8}", cells(r"\textbf{PCA pooled}", pooled["n"], pooled["m"]),
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def fix_split(doc: Doc, args, outdir: Path) -> None:
    say("\n## Issue 1 - split variant behind the accuracy results")
    variant, have_table = None, False
    if args.split_variant in ("pca", "v3c"):
        variant = args.split_variant
        say(f"  --split-variant {variant} given: treated as a statement by the author, not as evidence.")
    if args.repo:
        repo = Path(args.repo)
        v3 = Path(args.v3_file) if args.v3_file else None
        if v3 is None:
            copies = sorted(repo.rglob("hypatiax_defi_benchmark_v3_results_seed42.json"))
            for c in copies:
                say(f"  seed-42 copy: {c}  md5={md5(c)}")
            by_md5 = [c for c in copies if md5(c) == PAPER_V3_MD5]
            v3 = by_md5[0] if by_md5 else (copies[0] if copies else None)
        if v3 is None:
            say("  seed-42 v3 file not found under --repo (pass --v3-file); Issue 1 data steps skipped.")
        else:
            variant, have_table = _split_with_data(doc, args, outdir, repo, v3, variant)
    else:
        say("  --repo not given; only the text rewrite can be done (needs --split-variant).")
    _rewrite_caveat(doc, variant, have_table)


def _split_with_data(doc, args, outdir, repo, v3: Path, variant_in):
    variant, have_table = variant_in, False
    raw_v3 = load_json(v3)
    recs_v3 = task_records(raw_v3)
    m_v3 = {k: metrics(v) for k, v in arm_r2(recs_v3).items()}
    ref = parse_main_results(doc.orig[MAIN])
    same_md5 = md5(v3) == PAPER_V3_MD5
    bad = validate_loader(m_v3, ref) if ref else ["could not parse tab:main_results from the tex"]
    say(f"\n  Loader validation on {v3.name} (md5 {md5(v3)[:8]}..., paper cites {PAPER_V3_MD5[:8]}...):")
    if same_md5 and not bad:
        say("  PASS: all four rows of tab:main_results are reproduced from the file.")
        validated = True
    elif same_md5 and bad:
        say("  FAIL: the md5 matches but the numbers do not -> the field mapping is wrong. Not continuing.")
        for b in bad:
            say("    - " + b)
        say("  Record structure: " + json.dumps(skeleton(recs_v3[0]))[:500])
        return variant, False
    else:
        validated = args.trust_loader
        say("  NOT the file the paper cites (md5 differs)" + ("; mismatches vs paper: " + "; ".join(bad[:3]) if bad else "")
            + ("\n  --trust-loader given: continuing." if validated else "\n  Re-run with --trust-loader to proceed anyway."))
    if not validated:
        return variant, False

    found = find_pca_files(repo, args.pca_files)
    by_seed = {}
    for s, p, r in found:
        by_seed.setdefault(s, []).append((p, r))
    dup = {s: [str(p) for p, _ in v] for s, v in by_seed.items() if len(v) > 1}
    if dup:
        say("  Multiple 74-task PCA files for the same seed; pass --pca-files explicitly: " + json.dumps(dup))
        return variant, False
    if len(by_seed) < 2:
        say(f"  Found {len(by_seed)} 74-task PCA file(s); need the five seed files. Table not generated.")
        return variant, False
    seeds = sorted(by_seed)
    seed_rows, pool = [], {a: [] for a in m_v3}
    for s in seeds:
        arms = arm_r2(by_seed[s][0][1])
        seed_rows.append((s, {k: metrics(v) for k, v in arms.items()}))
        for k, v in arms.items():
            pool[k].append(v)
        say(f"  PCA seed {s}: {by_seed[s][0][0]}")
    pooled = {"n": 74 * len(seeds), "m": {k: metrics(np.concatenate(v)) for k, v in pool.items()}}

    (outdir / "generated").mkdir(parents=True, exist_ok=True)
    tex = pca_table_tex(m_v3, seed_rows, pooled)
    (outdir / "generated" / "pca_accuracy_table.tex").write_text(tex, encoding="utf-8")
    say("\n  Accuracy, near-perfect rate (R2>0.99, %):  v3-s42 | " + " ".join(f"s{s}" for s in seeds) + " | pooled")
    for arm in m_v3:
        say("    %-24s %5.1f | %s | %5.1f" % (arm, m_v3[arm]["p099"],
            " ".join("%5.1f" % m[arm]["p099"] for _, m in seed_rows), pooled["m"][arm]["p099"]))

    # ---- evidence on the seed-42 file's variant
    say("\n  Evidence on the split variant of the seed-42 file:")
    explicit = explicit_variant(raw_v3)
    flds = variant_fields(raw_v3)
    say("   * fields naming a split/variant inside the records: " + ("none" if not flds else
        "; ".join(f"{k}={v} (x{c})" for k, v, c in flds[:6])))
    if 42 in by_seed:
        agr = agreement(recs_v3, by_seed[42][0][1])
        say("   * per-task test_r2 identical, v3-seed42 vs PCA-seed42 (pure_llm / NN / hybrid): "
            + " / ".join("%.0f%%" % (100 * agr[a]) for a in ("pure_llm", "neural_network", "hybrid")))
        if len(seeds) > 1:
            other = [s for s in seeds if s != 42][0]
            ctl = agreement(by_seed[42][0][1], by_seed[other][0][1])
            say(f"   * control, PCA-seed42 vs PCA-seed{other} (different seeds)  : "
                + " / ".join("%.0f%%" % (100 * ctl[a]) for a in ("pure_llm", "neural_network", "hybrid")))
        say("     Reading: near-100% agreement on the NN arm (deterministic given a seed and a partition) "
            "means the same train/test partition; low agreement means a different one. This is evidence only.")
    if variant is None and explicit:
        variant = explicit
        say(f"   * records name the variant explicitly: {explicit}")
    if variant is None:
        say("   => Not determinable from the files. Caveat is rewritten to say exactly that and to point at the new PCA table.\n"
            "      If you know from the code which variant produced the seed-42 file, re-run with --split-variant pca|v3c.")
    have_table = True
    if "\\label{tab:pca_accuracy}" not in doc.text[MAIN]:
        doc.sub(MAIN, "Several key observations emerge from Table~\\ref{tab:main_results}.",
                tex + "\n" + "Several key observations emerge from Table~\\ref{tab:main_results}.",
                "main: insert tab:pca_accuracy before the observations paragraph")
    # contradictions with the restored 74-task PCA files
    doc.sub(MAIN, r"and how the\s+PCA-directed split compares, for which no 74-task run is available\.",
            r"and whether the five-seed PCA-directed results (Tables~\ref{tab:pca_accuracy} and~\ref{tab:timing_full}) "
            r"hold at larger seed counts.", "main: conclusion said no 74-task PCA run is available",
            regex=True, flags=re.S)
    doc.sub(MAIN, r"that is withdrawn, since the only PCA-directed seed files in the repository\s+hold 5 portfolio-task records each\.",
            r"that description is superseded: the five 74-task PCA-directed seed files located later are reported in "
            r"\S\ref{sec:timing} and Table~\ref{tab:pca_accuracy}.",
            "main: appendix said the only PCA seed files hold 5 records", regex=True, flags=re.S)
    doc.sub(MAIN, r"is representative of those five files", r"is representative of the five 74-task v3 files",
            "main: disambiguate 'those five files' after the previous edit", regex=True)
    return variant, have_table


def _rewrite_caveat(doc: Doc, variant, have_table: bool) -> None:
    old_main = "(the released result files do not record whether the single-axis or PCA-directed variant was used)"
    old_supp = "(the result file does not record whether the single-axis or PCA-directed variant was used)"
    ref = r"; five-seed PCA-directed accuracy is in Table~\ref{tab:pca_accuracy}" if have_table else ""
    if variant == "pca":
        new_main, new_supp = r"(PCA-directed variant, \S\ref{sec:split})", "(PCA-directed variant)"
    elif variant == "v3c":
        new_main, new_supp = r"(single-axis \texttt{v3c} variant, \S\ref{sec:split}" + ref + ")", "(single-axis variant)"
    elif have_table:
        new_main = (r"(the seed-42 file behind these figures does not name its split variant; PCA-directed accuracy "
                    r"on five 74-task seed files is in Table~\ref{tab:pca_accuracy})")
        new_supp = ("(the seed-42 result file does not name its split variant; the main paper reports PCA-directed "
                    "accuracy on five 74-task seed files)")
    else:
        say("  No variant and no PCA table available: caveat left unchanged.")
        return
    doc.sub(MAIN, old_main, new_main, "main: abstract / contributions / conclusion split-variant caveat", all_=True)
    doc.sub(SUPP, old_supp, new_supp, "supp: split-variant caveat")


# ===================================================================== ISSUE 3
def clopper(k, n):
    r = stats.binomtest(k, n).proportion_ci(method="exact")
    return 100 * r.low, 100 * r.high


# ---- 3a portfolio seed sweep
class ExtractError(Exception):
    pass


def find_seed_map(o):
    if isinstance(o, dict):
        if len(o) >= 2 and all(re.fullmatch(r"(?:seed[_-]?)?\d+", str(k)) for k in o):
            return {int(re.search(r"\d+", str(k)).group()): v for k, v in o.items()}
        for v in o.values():
            r = find_seed_map(v)
            if r:
                return r
    elif isinstance(o, list):
        if len(o) >= 2 and all(isinstance(x, dict) and "seed" in x for x in o):
            return {int(x["seed"]): x for x in o}
        for v in o:
            r = find_seed_map(v)
            if r:
                return r
    return None


def leaves(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path.lower(), o


def extract_sweep(obj) -> dict:
    """ADAPT HERE if your sweep JSON is structured differently. Returns {seed: {p, h, exact}}."""
    smap = find_seed_map(obj)
    if not smap:
        raise ExtractError("no per-seed container found; structure: " + json.dumps(skeleton(obj))[:500])
    out = {}
    for seed, entry in smap.items():
        lv = list(leaves(entry))
        pick = lambda who, extra: [(p, v) for p, v in lv if re.search(who, p) and "far" in p and re.search(extra, p)
                                   and isinstance(v, (int, float)) and not isinstance(v, bool)]
        pc, hc = pick(r"pysr", r"r2"), pick(r"hypatia|hybrid", r"r2")
        if len(pc) != 1 or len(hc) != 1:
            raise ExtractError(f"seed {seed}: ambiguous far-R2 leaves; PySR={[p for p, _ in pc]} H={[p for p, _ in hc]}")
        ex = [v for p, v in lv if re.search(r"hypatia|hybrid", p) and re.search(r"recover|exact", p) and isinstance(v, bool)]
        out[seed] = {"p": float(pc[0][1]), "h": float(hc[0][1]), "exact": ex[0] if len(ex) == 1 else None}
    return out


def parse_sweep_table(tex: str) -> dict:
    blk = re.search(r"\\label\{tab:portfolio_seed_sweep\}.*?\\end\{tabular\}", tex, re.S)
    rows = {}
    if blk:
        for m in re.finditer(r"^(\d+)(?: \(default\))?\s*&\s*\$([^$]+)\$\s*&\s*\$([^$]+)\$\s*&\s*([^&]+?)\s*&\s*(Yes|No)\s*&\s*(Yes|No)\s*\\\\",
                             blk.group(0), re.M):
            rows[int(m.group(1))] = {"p": float(m.group(2)), "h": float(m.group(3)), "exact": m.group(5) == "Yes",
                                     "win": m.group(6) == "Yes"}
    return rows


def fix_portfolio(doc: Doc, args, outdir: Path) -> None:
    say("\n### tab:portfolio_seed_sweep")
    tag = r"\\textbf\{\[Unverified: the five seed result files.*?reproduced\.\]\}"
    paper = parse_sweep_table(doc.orig[MAIN])
    if not paper:
        say("  could not parse the table from the tex; skipped.")
        return
    pm = np.mean([r["p"] for r in paper.values()])
    hm = np.mean([r["h"] for r in paper.values()])
    wins = sum(r["h"] > r["p"] for r in paper.values())
    arith = (all(r["win"] == (r["h"] > r["p"]) for r in paper.values()))
    say(f"  arithmetic in the table: mean P {pm:.3f}, mean H {hm:.3f}, strict wins {wins}/{len(paper)}, "
        f"win column consistent: {arith}")
    path = Path(args.sweep_json) if args.sweep_json else None
    if path is None and args.repo:
        c = [p for p in Path(args.repo).rglob("*.json") if re.search(r"portfolio.*seed|seed.*sweep", p.name, re.I)]
        c = sorted(c)
        path = c[0] if c else None
        if len(c) > 1:
            say("  several candidates, using the first (override with --sweep-json): " + ", ".join(map(str, c)))
    data = None
    if path:
        try:
            data = extract_sweep(load_json(path))
        except Exception as e:                       # noqa: BLE001
            say(f"  {path}: could not extract per-seed values ({e}). Edit extract_sweep() or pass a different file.")
    if data:
        diffs = []
        for s, r in paper.items():
            d = data.get(s)
            if d is None:
                diffs.append(f"seed {s} missing from file")
                continue
            for key in ("p", "h"):
                if abs(d[key] - r[key]) > 6e-4:
                    diffs.append(f"seed {s} {key}: file {d[key]:.3f} vs paper {r[key]:.3f}")
            if d["exact"] is not None and d["exact"] != r["exact"]:
                diffs.append(f"seed {s} exact: file {d['exact']} vs paper {r['exact']}")
        if not diffs:
            say(f"  VERIFIED against {path} (md5 {md5(path)[:8]}...): all five seeds match.")
            doc.sub(MAIN, tag, r"Regenerated from \texttt{%s} (md5 \texttt{%s}); means and win count recomputed."
                    % (tt(path.name), md5(path)), "main: tab:portfolio_seed_sweep tag removed (verified)",
                    regex=True, flags=re.S)
            return
        say("  MISMATCH with the source file - tag kept, nothing patched. Differences:")
        for d in diffs:
            say("    - " + d)
        say("  Prose that depends on this table must be re-checked too: '4 of 5 seeds', '2/5 exact', the figure "
            "caption, and the Bayesian P~0.76 (which this script does not regenerate).")
        return
    say("  Source files not located/readable: tag narrowed to what was checked.")
    note = (r"\textbf{[Unverified: the five seed result files were not located, so these figures are not regenerated "
            r"or compared with source data. Arithmetic checked only: the means ($%.3f$, $%.3f$) and the %d/%d win count "
            r"follow from the rows.]}" % (pm, hm, wins, len(paper)))
    if arith:
        doc.sub(MAIN, tag, note, "main: tab:portfolio_seed_sweep tag narrowed", regex=True, flags=re.S)


# ---- 3b tab:arch
def fix_arch(doc: Doc, args) -> None:
    say("\n### tab:arch")
    tag = r"\\textbf\{\[Unverified: the routing percentages.*?Post-Submission Report\)\.\]\}"
    m3 = re.search(r"(\d+\.\d)\\%\s+ensemble-first", doc.orig[SUPP])
    m4 = re.search(r"(\d+\.\d)\\%\s+LLM-first", doc.orig[SUPP])
    if not (m3 and m4):
        say("  could not parse the percentages; skipped.")
        return
    p3, p4 = float(m3.group(1)), float(m4.group(1))
    k3 = [k for k in range(151) if round(100 * k / 150, 1) == p3]
    k4 = [k for k in range(151) if round(100 * k / 150, 1) == p4]
    say(f"  {p3}% = {k3}/150 and {p4}% = {k4}/150 -> both are shares of the 150 noise-sweep comparisons "
        f"(neither is a whole count out of the 180 sample-complexity comparisons).")
    shards = [Path(p) for p in args.noise_shards] if args.noise_shards else \
        ([p for p in Path(args.repo).rglob("*.json") if re.search(r"noise(?!less)", str(p), re.I) and not re.search(r"sample", str(p), re.I)]
         if args.repo else [])
    # dedupe by content: identical shard copies would otherwise be counted twice
    seen, uniq = {}, []
    for sh in shards:
        try:
            h = md5(sh)
        except Exception:                            # noqa: BLE001
            continue
        if h in seen:
            say(f"  skipped duplicate: {sh.name} (identical to {seen[h].name})")
            continue
        seen[h] = sh
        uniq.append(sh)
    xtab = {"M3": Counter(), "M4": Counter()}
    for sh in uniq:
        try:
            data = load_json(sh)
        except Exception:                            # noqa: BLE001
            continue
        for p, d in walk(data):
            md = d.get("metadata") if isinstance(d.get("metadata"), dict) else None
            if md is None or not isinstance(md.get("decision"), str):
                continue
            hay = " ".join(p).replace("_", "").lower()
            which = "M3" if ("enhancedhybridsystemdefi" in hay or re.search(r"\bm3\b", hay)) else \
                    "M4" if ("hybridsystemllmnn" in hay or re.search(r"\bm4\b", hay)) else None
            if which:
                xtab[which][(md["decision"], md.get("nn_applied"), d.get("success"))] += 1
    n3, n4 = sum(xtab["M3"].values()), sum(xtab["M4"].values())
    if uniq and n3 and n4:
        say(f"  {len(uniq)} unique noise shard file(s) after dedupe; M3 records: {n3}, M4 records: {n4} (paper: 150 each)")
        for w in ("M3", "M4"):
            say(f"  {w} cross-tab (decision, nn_applied, success): "
                + "; ".join(f"{k}={v}" for k, v in sorted(xtab[w].items(), key=str)))
        def cnt(w, f):
            return sum(c for (dec, nn, ok), c in xtab[w].items() if f(dec, nn, ok))
        cands = {"M3": {"decision == ensemble": lambda d, n, o: d == "ensemble",
                        "ensemble and nn_applied is False": lambda d, n, o: d == "ensemble" and n is False,
                        "ensemble and success": lambda d, n, o: d == "ensemble" and o is True},
                 "M4": {"decision == llm": lambda d, n, o: d == "llm",
                        "llm and nn_applied is False": lambda d, n, o: d == "llm" and n is False,
                        "llm and success": lambda d, n, o: d == "llm" and o is True}}
        target = {"M3": k3, "M4": k4}
        hit = {}
        say(f"  Candidate definitions (paper: M3 {k3}/150, M4 {k4}/150):")
        for w in ("M3", "M4"):
            for name, f in cands[w].items():
                c = cnt(w, f)
                ok = c in target[w]
                if ok:
                    hit.setdefault(w, []).append(name)
                say(f"    {w}  {name:<36} {c}/{n3 if w == 'M3' else n4}  {100 * c / (n3 if w == 'M3' else n4):.1f}%" + ("   <- reproduces" if ok else ""))
        say("  NOTE: a definition that reproduces a number is a hypothesis about what was counted, not a verification.")
        say("        The tag is NOT removed automatically: confirm the definition in the code that generated the table.")
        if len(hit) == 2 and n3 == n4 == 150:
            say(f"  Both counts reproduce under: {hit}. Tag narrowed to say which definition reproduces them.")
            note2 = (r"\textbf{[Unverified: not confirmed against the generating code. Recounted from the noise-sweep result files, "
                     r"$%s\%%$ and $%s\%%$ are reproduced when routing is read from \texttt{metadata.decision} together with "
                     r"\texttt{metadata.nn\_applied}; the definition of the routing labels is inferred.]}" % (p3, p4))
            doc.sub(SUPP, tag, note2, "supp: tab:arch tag narrowed (counts reproduced, definition inferred)", regex=True, flags=re.S)
            return
        say("  No single definition reproduces both counts (or n != 150) - tag kept.")
    else:
        say("  No routing records (metadata.decision) found in noise shards (pass --noise-shards).")
    note = (r"\textbf{[Unverified: not regenerated from result files. Arithmetic only: $%s\%%=%d/150$ and $%s\%%=%d/150$, "
            r"i.e.\ both are shares of the 150 noise-sweep comparisons, not of the 180 sample-complexity ones.]}"
            % (p3, k3[0] if k3 else 0, p4, k4[0] if k4 else 0))
    if k3 and k4:
        doc.sub(SUPP, tag, note, "supp: tab:arch tag narrowed", regex=True, flags=re.S)


# ---- 3c tab:noise_sensitivity
def parse_noise_table(tex: str) -> dict:
    blk = re.search(r"\\label\{tab:noise_sensitivity\}.*?\\end\{tabular\}", tex, re.S)
    rows = {}
    if blk:
        for m in re.finditer(r"^([0-9.]+)\s*&\s*[0-9.]+\\%\s*\((\d+)/(\d+)\)\s*&\s*\[([0-9.]+)\\%,\s*([0-9.]+)\\%\]", blk.group(0), re.M):
            rows[float(m.group(1))] = (int(m.group(2)), int(m.group(3)), float(m.group(4)), float(m.group(5)))
    return rows


def find_noise_source(repo: Path, limit=1500):
    """Look for a JSON file whose records carry all seven noise levels plus a success flag."""
    best, seen = [], 0
    for p in repo.rglob("*.json"):
        if p.stat().st_size > 150e6:
            continue
        seen += 1
        if seen > limit:
            break
        try:
            data = load_json(p)
        except Exception:                            # noqa: BLE001
            continue
        cnt = {lv: [0, 0] for lv in NOISE_LEVELS}
        for _, d in walk(data):
            nk = [k for k in d if re.search(r"noise", str(k), re.I) and isinstance(d[k], (int, float)) and not isinstance(d[k], bool)]
            sk = [k for k in d if re.fullmatch(r"success|passed|ok|recovered|exact", str(k), re.I) and isinstance(d[k], (bool, int))]
            if nk and sk:
                lv = next((l for l in NOISE_LEVELS if math.isclose(float(d[nk[0]]), l, abs_tol=1e-9)), None)
                if lv is not None:
                    cnt[lv][1] += 1
                    cnt[lv][0] += int(bool(d[sk[0]]))
        if all(cnt[l][1] > 0 for l in NOISE_LEVELS):
            best.append((p, cnt))
    return best


def fix_noise(doc: Doc, args) -> None:
    say("\n### tab:noise_sensitivity")
    tag = r"\\textbf\{\[Unverified: the source of this seven-level.*?Post-Submission Report\)\.\]\}"
    rows = parse_noise_table(doc.orig[SUPP])
    if len(rows) != 7:
        say(f"  parsed {len(rows)} rows, expected 7; skipped.")
        return
    ok = True
    for lv, (k, n, lo, hi) in sorted(rows.items()):
        clo, chi = clopper(k, n)
        match = abs(clo - lo) < 0.06 and abs(chi - hi) < 0.06
        ok &= match
    ka = sum(k for lv, (k, n, _, _) in rows.items() if lv <= 0.10)
    na = sum(n for lv, (k, n, _, _) in rows.items() if lv <= 0.10)
    kb = sum(k for lv, (k, n, _, _) in rows.items() if lv > 0.10)
    nb = sum(n for lv, (k, n, _, _) in rows.items() if lv > 0.10)
    fp = stats.fisher_exact([[ka, na - ka], [kb, nb - kb]])[1]
    say(f"  arithmetic: all 7 CIs are exact Clopper-Pearson intervals of the stated counts: {ok}; "
        f"Fisher exact (sigma<=0.10: {ka}/{na} vs >0.10: {kb}/{nb}) p={fp:.2e}")
    cand = []
    if args.repo:
        say("  scanning --repo for a result file with all seven noise levels (this can take a while)...")
        cand = find_noise_source(Path(args.repo))
    if cand:
        for p, cnt in cand[:3]:
            say(f"  candidate {p}: " + ", ".join(f"{l}:{c[0]}/{c[1]}" for l, c in cnt.items()))
        p, cnt = cand[0]
        if all(cnt[l][0] == rows[l][0] and cnt[l][1] == rows[l][1] for l in NOISE_LEVELS):
            say("  VERIFIED: counts match the table exactly.")
            doc.sub(SUPP, tag, r"Regenerated from \texttt{%s} (md5 \texttt{%s})." % (tt(p.name), md5(p)),
                    "supp: tab:noise_sensitivity tag removed (verified)", regex=True, flags=re.S)
            return
        say("  Candidate counts differ from the table - tag kept. (Note the table defines success as R2>=0.99 AND the "
            "correct functional form; a plain success flag may not encode the latter.)")
    else:
        say("  No file with all seven noise levels found.")
    if ok and fp < 1e-3:
        note = (r"\textbf{[Unverified: the source of the success counts is still unidentified, so the counts are not "
                r"regenerated from result files. Checked arithmetic only: the percentages, the exact (Clopper--Pearson) "
                r"95\,\% intervals and the Fisher test below follow from the stated counts.]}")
        doc.sub(SUPP, tag, note, "supp: tab:noise_sensitivity tag narrowed", regex=True, flags=re.S)


def fix_unverified(doc: Doc, args, outdir: Path) -> None:
    say("\n## Issue 3 - tables labelled Unverified")
    fix_portfolio(doc, args, outdir)
    fix_arch(doc, args)
    fix_noise(doc, args)


# ======================================================================== main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex-dir", default=".", help="directory holding the three .tex files")
    ap.add_argument("--repo", help="repository root holding the result files")
    ap.add_argument("--out", default="fix_out")
    ap.add_argument("--apply", action="store_true", help="write patched copies to <out>/patched/")
    ap.add_argument("--split-variant", choices=["auto", "pca", "v3c"], default="auto",
                    help="state which variant produced the seed-42 accuracy file, if you know it from the code")
    ap.add_argument("--v3-file"); ap.add_argument("--pca-files", nargs="*")
    ap.add_argument("--sweep-json"); ap.add_argument("--noise-shards", nargs="*"); ap.add_argument("--exp1-json")
    ap.add_argument("--trust-loader", action="store_true",
                    help="proceed even if the seed-42 file is not the one (md5) the paper cites")
    args = ap.parse_args()

    tex_dir, outdir = Path(args.tex_dir), Path(args.out)
    for n in (MAIN, SUPP):
        if not (tex_dir / n).exists():
            sys.exit(f"missing {tex_dir / n}")
    outdir.mkdir(parents=True, exist_ok=True)
    doc = Doc(tex_dir)

    say("# fix_paper_issues report")
    fix_split(doc, args, outdir)
    fix_regime(doc, args)
    fix_unverified(doc, args, outdir)

    say("\n## Edits")
    for tag, st in doc.edits:
        say(f"  [{st}] {tag}")
    diff = []
    for n in (MAIN, SUPP):
        diff += list(difflib.unified_diff(doc.orig[n].splitlines(True), doc.text[n].splitlines(True),
                                          f"a/{n}", f"b/{n}", n=1))
    (outdir / "changes.diff").write_text("".join(diff), encoding="utf-8")
    if args.apply:
        (outdir / "patched").mkdir(exist_ok=True)
        for n in (MAIN, SUPP):
            (outdir / "patched" / n).write_text(doc.text[n], encoding="utf-8")
        say(f"\nPatched copies written to {outdir / 'patched'} (originals untouched).")
    else:
        say(f"\nDry run: nothing written except the report, changes.diff and generated/ ({outdir}). Add --apply for patched copies.")
    (outdir / "fix_report.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

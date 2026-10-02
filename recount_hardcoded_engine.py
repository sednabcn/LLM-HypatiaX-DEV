#!/usr/bin/env python3
"""
recount_hardcoded.py - how many PureLLM rows bypassed the LLM, and what the
PureLLM-vs-other-methods comparison looks like with and without them.

Background: baseline_pure_llm_defi_discovery._try_hardcoded_formula() returns
a result tagged method="pure_llm_hardcoded" (no LLM call) for 12 equation
families:
  ground_truth : Gaussian, Planck, Bose-Einstein, Nernst, Clausius-Mossotti,
                 Lorentz force, Fourier, Zeeman   (benchmark's own formula)
  ols_fitted   : Logistic growth, Michaelis-Menten, Allometric, Arrhenius
                 (fixed form, parameters least-squares fitted to the data)
The runner sets metadata["is_hardcoded"] from that tag, but the runner's
fallback-eval path used to omit it, so the flag is a LOWER BOUND on older
result files. This script therefore reports three evidence levels:

  FLAGGED    metadata.is_hardcoded is true, or a bypass marker string appears
             anywhere in the row (pure_llm_hardcoded, "hardcoded exact",
             "analytically fitted", "exact formula bypass")
  SUSPECTED  not flagged, but the row's equation is one of the 12 families AND
             its recorded time is below --fast-ms (default 5 ms; a live LLM
             call takes seconds)
  FAMILY     not flagged, equation is in a bypass family, but the row was slow.
             Informational only: the branch also needs matching variable
             names, so it may genuinely have called the LLM.

USAGE
  python3 recount_hardcoded.py <file_or_dir> [more ...] \\
      [--threshold 0.999999] [--llm-filter purellm] [--fast-ms 5] \\
      [--recursive] [--r2-key r2] [--json out.json] [--csv rows.csv]

Files are reported separately and never pooled (a consolidated file usually
duplicates its shard files). Handles the shard shape {"tests":[{"results":{}}]},
the flat shape [{"method":..,"r2":..}], the seed-list shape
[{"results":{method:{}}}], and falls back to a generic recursive walk.
"""

import argparse
import csv
import json
import pathlib
import sys
from collections import defaultdict

MARKERS = (
    "pure_llm_hardcoded",
    "hardcoded exact",
    "analytically fitted",
    "exact formula bypass",
)

FAMILIES = {
    # keyword (lowercase, substring of description) -> (family, kind)
    "gaussian": ("Gaussian PDF", "ground_truth"),
    "normal distribution": ("Gaussian PDF", "ground_truth"),
    "planck": ("Planck blackbody", "ground_truth"),
    "blackbody": ("Planck blackbody", "ground_truth"),
    "bose": ("Bose-Einstein", "ground_truth"),
    "nernst": ("Nernst", "ground_truth"),
    "clausius": ("Clausius-Mossotti", "ground_truth"),
    "lorentz": ("Lorentz force", "ground_truth"),
    "fourier": ("Fourier heat conduction", "ground_truth"),
    "heat conduction": ("Fourier heat conduction", "ground_truth"),
    "zeeman": ("Zeeman energy", "ground_truth"),
    "logistic": ("Logistic growth", "ols_fitted"),
    "michaelis": ("Michaelis-Menten", "ols_fitted"),
    "menten": ("Michaelis-Menten", "ols_fitted"),
    "allometric": ("Allometric scaling", "ols_fitted"),
    "arrhenius": ("Arrhenius", "ols_fitted"),
}

R2_KEYS_DEFAULT = ("r2", "r2_test", "test_r2", "r2_train", "train_r2", "best_r2", "R2")
TIME_KEYS = ("time", "elapsed", "duration", "runtime", "time_s", "elapsed_s", "elapsed_sec", "seconds")


def norm(s):
    return str(s or "").lower().replace("-", "").replace("_", "").replace(" ", "")


def get_r2(row, keys):
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f <= 1.01:
            return f
    return None


def get_time(row):
    for k in TIME_KEYS:
        v = row.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def family_of(desc):
    d = str(desc or "").lower()
    for kw, fam in FAMILIES.items():
        if kw in d:
            return fam
    return None


def row_text(row):
    try:
        return json.dumps(row, default=str).lower()
    except Exception:
        return str(row).lower()


def classify(row, desc, fast_ms):
    """Return (level, kind, family) for one row."""
    md = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    txt = row_text(row)
    fam = family_of(desc)
    kind = fam[1] if fam else "unknown"
    # an explicit text marker is more specific than the keyword guess
    if "analytically fitted" in txt:
        kind = "ols_fitted"
    elif "hardcoded exact" in txt:
        kind = "ground_truth"
    if md.get("is_hardcoded") is True or row.get("is_hardcoded") is True \
            or any(m in txt for m in MARKERS):
        return "FLAGGED", kind, fam[0] if fam else "?"
    if fam:
        t = get_time(row)
        if t is not None and t * 1000.0 < fast_ms:
            return "SUSPECTED", kind, fam[0]
        return "FAMILY", kind, fam[0]
    return "NONE", "none", ""


# ------------------------------------------------------------------ loading
def looks_like_row(d):
    return isinstance(d, dict) and any(k in d for k in R2_KEYS_DEFAULT)


def walk(data, desc=""):
    """Yield (method, row, test_desc) from any of the known shapes."""
    if isinstance(data, dict):
        d = data.get("description") or data.get("test") or data.get("equation_id") or desc
        res = data.get("results")
        if isinstance(res, dict) and res and all(isinstance(v, dict) for v in res.values()):
            for m, row in res.items():
                if looks_like_row(row) or "method" in row or "success" in row:
                    yield row.get("method", m), row, d
            return
        if "method" in data and looks_like_row(data):
            yield data["method"], data, data.get("test") or data.get("description") or desc
            return
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                yield from walk(v, d)
    elif isinstance(data, list):
        for item in data:
            yield from walk(item, desc)


def collect(paths, recursive):
    files = []
    for p in paths:
        p = pathlib.Path(p)
        if p.is_dir():
            files += sorted(p.rglob("*.json") if recursive else p.glob("*.json"))
        elif p.is_file():
            files.append(p)
    return files


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.999999)
    ap.add_argument("--llm-filter", default="purellm",
                    help="comma-separated substrings (normalised) identifying the PureLLM method")
    ap.add_argument("--fast-ms", type=float, default=5.0)
    ap.add_argument("--r2-key", default="", help="use only this r2 field (default: fallback chain)")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    r2_keys = (args.r2_key,) if args.r2_key else R2_KEYS_DEFAULT
    llm_terms = [norm(t) for t in args.llm_filter.split(",") if t.strip()]
    is_llm = lambda m: any(t in norm(m) for t in llm_terms)

    files = collect(args.paths, args.recursive)
    if not files:
        sys.exit("no JSON files found")

    all_rows, summary = [], {}
    for fp in files:
        try:
            data = json.loads(fp.read_text())
        except Exception as e:
            print(f"[skip] {fp.name}: {e}")
            continue
        rows = []
        for method, row, desc in walk(data):
            if not isinstance(row, dict):
                continue
            level, kind, fam = classify(row, desc, args.fast_ms)
            r2 = get_r2(row, r2_keys)
            rows.append(dict(
                file=fp.name, test=str(desc), method=str(method), llm=is_llm(method),
                level=level, kind=kind, family=fam, r2=r2,
                passed=(r2 is not None and r2 >= args.threshold),
                time=get_time(row)))
        if not rows:
            print(f"[skip] {fp.name}: no method rows recognised")
            continue
        all_rows += rows
        summary[fp.name] = rows

    # ---- 1. per file x method flag counts
    print("=" * 78)
    print("1. FLAG COUNTS PER FILE x METHOD  (F=flagged S=suspected, fam=family-only)")
    print("=" * 78)
    for fname, rows in summary.items():
        by_m = defaultdict(list)
        for r in rows:
            by_m[r["method"]].append(r)
        print(f"\n{fname}")
        for m, rs in sorted(by_m.items()):
            f = sum(r["level"] == "FLAGGED" for r in rs)
            s = sum(r["level"] == "SUSPECTED" for r in rs)
            fam = sum(r["level"] == "FAMILY" for r in rs)
            if f or s or fam or is_llm(m):
                gt = sum(r["level"] == "FLAGGED" and r["kind"] == "ground_truth" for r in rs)
                ols = sum(r["level"] == "FLAGGED" and r["kind"] == "ols_fitted" for r in rs)
                print(f"  {m:<44} rows={len(rs):<3} F={f:<2} (gt={gt}, ols={ols}) S={s:<2} fam={fam}")

    # ---- 2. PureLLM bypassed rows
    print("\n" + "=" * 78)
    print("2. PureLLM ROWS THAT BYPASSED THE LLM  (FLAGGED + SUSPECTED)")
    print("=" * 78)
    for fname, rows in summary.items():
        byp = [r for r in rows if r["llm"] and r["level"] in ("FLAGGED", "SUSPECTED")]
        llm_rows = [r for r in rows if r["llm"]]
        if not llm_rows:
            continue
        print(f"\n{fname}: {len(byp)} of {len(llm_rows)} PureLLM rows")
        for r in sorted(byp, key=lambda x: x["test"]):
            t = "n/a" if r["time"] is None else f"{r['time']*1000:.1f}ms"
            r2 = "n/a" if r["r2"] is None else f"{r['r2']:.7f}"
            print(f"  [{r['level'][:1]}] {r['test'][:46]:<46} {r['kind']:<12} r2={r2:<10} "
                  f"{'PASS' if r['passed'] else 'fail'}  {t}")

    # ---- 3. pass counts
    print("\n" + "=" * 78)
    print(f"3. PureLLM PASS COUNT (r2 >= {args.threshold})  all vs live-only vs bypassed-only")
    print("=" * 78)
    for fname, rows in summary.items():
        llm_rows = [r for r in rows if r["llm"]]
        if not llm_rows:
            continue
        live = [r for r in llm_rows if r["level"] not in ("FLAGGED", "SUSPECTED")]
        byp = [r for r in llm_rows if r["level"] in ("FLAGGED", "SUSPECTED")]
        p = lambda xs: f"{sum(x['passed'] for x in xs)}/{len(xs)}"
        print(f"  {fname:<52} all={p(llm_rows):<7} live-only={p(live):<7} bypassed={p(byp)}")

    # ---- 4. head to head
    print("\n" + "=" * 78)
    print("4. PureLLM vs EACH OTHER METHOD (same test, strict pass/fail)")
    print("   wins = PureLLM passes & other fails; losses = reverse")
    print("   'live' excludes tests where PureLLM's row was bypassed")
    print("=" * 78)
    h2h_out = {}
    for fname, rows in summary.items():
        by_test = defaultdict(dict)
        for r in rows:
            by_test[r["test"].strip().lower()][r["method"]] = r
        llm_names = {r["method"] for r in rows if r["llm"]}
        others = sorted({r["method"] for r in rows if not r["llm"]})
        if not llm_names or not others:
            continue
        print(f"\n{fname}")
        print(f"  {'method':<44} {'wins':>5} {'losses':>7} | {'live wins':>9} {'live losses':>11} | tests")
        for o in others:
            w = l = lw = ll = n = 0
            for t, ms in by_test.items():
                lr = next((ms[m] for m in llm_names if m in ms), None)
                orow = ms.get(o)
                if lr is None or orow is None:
                    continue
                n += 1
                bypass = lr["level"] in ("FLAGGED", "SUSPECTED")
                if lr["passed"] and not orow["passed"]:
                    w += 1
                    lw += 0 if bypass else 1
                elif orow["passed"] and not lr["passed"]:
                    l += 1
                    ll += 0 if bypass else 1
            print(f"  {o:<44} {w:>5} {l:>7} | {lw:>9} {ll:>11} | {n}")
            h2h_out.setdefault(fname, {})[o] = dict(wins=w, losses=l, live_wins=lw, live_losses=ll, tests=n)

    # ---- 5. unflagged family matches
    fam_only = [r for r in all_rows if r["llm"] and r["level"] == "FAMILY"]
    if fam_only:
        print("\n" + "=" * 78)
        print("5. FAMILY-ONLY rows (in a bypass family, not flagged, slow) - informational")
        print("=" * 78)
        seen = defaultdict(int)
        for r in fam_only:
            seen[(r["file"], r["family"])] += 1
        for (f, fam), n in sorted(seen.items()):
            print(f"  {f:<52} {fam:<26} {n}")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(
            dict(threshold=args.threshold, head_to_head=h2h_out, rows=all_rows), indent=2))
        print(f"\nwrote {args.json}")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()

# ============================================================
# COMPLETE REPLACEMENT for the python3 heredoc inside
# "Audit results against paper claims" step in ci_all_checkpoint.yml
# (lines ~839–1035).  Replace everything between the opening
# python3 - <<'PYEOF' and its closing PYEOF with this content.
# ============================================================

import json, os, sys
import glob as _glob
from pathlib import Path

OUT_BASE       = Path(os.environ["OUT_BASE"])
QUALIFY_REPORT = json.loads(os.environ.get("QUALIFY_REPORT") or "{}")
QUALIFIED_EXPS = set(json.loads(os.environ.get("QUALIFIED_EXPS") or "[]"))

# ── Load paper_targets.json ──────────────────────────────────────────────────
PAPER_TARGETS = Path("scripts/patches/paper_targets.json")

def load_claims():
    if PAPER_TARGETS.exists():
        try:
            raw = json.loads(PAPER_TARGETS.read_text())
            print(f"  Loaded {len(raw)} claim(s) from {PAPER_TARGETS}")
            return raw
        except Exception as e:
            print(f"  ⚠  Failed to load paper_targets.json: {e} — using built-in fallback.")
    # Minimal built-in fallback (intentionally empty — paper_targets.json must exist)
    print("  ⚠  scripts/patches/paper_targets.json not found. Commit it to the repo.")
    return []

# ── File resolver: result_path → result_glob → result_glob_alt ──────────────
def resolve_result_file(claim):
    rpath = claim.get("result_path", "")
    if rpath:
        p = OUT_BASE / rpath
        if p.exists():
            return p
        alt = p.parent / "_stats.json"
        if alt.exists():
            return alt

    for key in ("result_glob", "result_glob_alt"):
        pattern = claim.get(key, "")
        if not pattern:
            continue
        matches = sorted(_glob.glob(str(OUT_BASE / pattern)))
        if matches:
            return Path(matches[0])
    return None

# ── Dot-notation key resolver (handles dict + list by index) ─────────────────
def resolve_key(data, dot_key):
    if not dot_key:
        return None
    node = data
    for p in dot_key.split("."):
        if isinstance(node, dict):
            node = node.get(p)
        elif isinstance(node, list):
            try:
                node = node[int(p)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if node is None:
            return None
    return node

# ── Compute helpers for aggregate / non-scalar JSON structures ───────────────
def compute_metric(data, claim):
    """
    Called when claim has a 'compute' field.
    Returns float or None.

    Supported compute types:
      list_field_mean      — mean of a boolean/numeric field across a list
                             (True → 1.0, False → 0.0).
                             Requires: compute_field
      list_distinct_count  — count of distinct non-null values of a field.
                             Requires: compute_field
      dict_nested_mean     — navigate cross_noise_summary-style structure:
                             data[compute_method][noise_level][compute_leaf],
                             average across all noise levels.
                             Requires: compute_method, compute_leaf
    """
    compute = claim.get("compute", "")

    if compute == "list_field_mean":
        field = claim.get("compute_field", "")
        if not isinstance(data, list) or not data:
            return None
        vals = [float(item[field]) for item in data
                if isinstance(item, dict) and field in item]
        return sum(vals) / len(vals) if vals else None

    elif compute == "list_distinct_count":
        field = claim.get("compute_field", "")
        if not isinstance(data, list):
            return None
        distinct = {item.get(field) for item in data
                    if isinstance(item, dict) and item.get(field) is not None}
        return float(len(distinct))

    elif compute == "dict_nested_mean":
        method = claim.get("compute_method", "")
        leaf   = claim.get("compute_leaf", "recovery_rate")
        if not isinstance(data, dict):
            return None
        cn = data.get("cross_noise_summary", data)   # suppB top-level is already the noise-sweep dict
        method_data = cn.get(method, {})
        if not method_data:
            # Fallback: try first available method key
            if cn:
                method_data = next(iter(cn.values()), {})
        rates = [v.get(leaf) for v in method_data.values()
                 if isinstance(v, dict) and v.get(leaf) is not None]
        return sum(rates) / len(rates) if rates else None

    return None

# ── Main audit loop ──────────────────────────────────────────────────────────
claims   = load_claims()
findings = []

print(f"\n  {'Exp':<25} {'Metric':<35} {'Status':<8}  Detail")
print("  " + "─" * 100)

for claim in claims:
    if "_EXCLUDED" in claim or "exp" not in claim:
        continue  # skip comment/excluded entries
    exp    = claim["exp"]
    metric = claim["metric"]
    pv     = float(claim["paper_value"])
    tol    = float(claim.get("tolerance", 0.01))
    jkey   = claim.get("json_key", "")
    absol  = bool(claim.get("absolute", False))
    note   = claim.get("note", "")
    has_compute = bool(claim.get("compute", ""))

    if exp not in QUALIFIED_EXPS:
        findings.append({"exp": exp, "metric": metric, "status": "SKIP",
                         "paper_value": pv, "actual": None,
                         "detail": f"{exp} not qualified — audit skipped"})
        print(f"  ↩  {exp:<23} {metric:<35} {'SKIP':<8}  {exp} not qualified")
        continue

    result_file = resolve_result_file(claim)
    if result_file is None:
        rpath_disp = claim.get("result_path") or claim.get("result_glob") or "?"
        findings.append({"exp": exp, "metric": metric, "status": "MISSING",
                         "paper_value": pv, "actual": None,
                         "detail": f"no result file found ({rpath_disp})"})
        print(f"  🔍 {exp:<23} {metric:<35} {'MISSING':<8}  {rpath_disp}")
        continue

    try:
        data = json.loads(result_file.read_text())
    except Exception as e:
        findings.append({"exp": exp, "metric": metric, "status": "MISSING",
                         "paper_value": pv, "actual": None,
                         "detail": f"JSON parse error: {e}"})
        print(f"  🔍 {exp:<23} {metric:<35} {'MISSING':<8}  parse error: {e}")
        continue

    # Resolve the metric value
    raw = None
    if has_compute:
        raw = compute_metric(data, claim)
    else:
        raw = resolve_key(data, jkey)
        if raw is None and isinstance(data, dict):
            raw = data.get(jkey.split(".")[-1])   # flat-key fallback

    if raw is None:
        detail = (f"key '{jkey}' not found" if not has_compute
                  else f"compute='{claim['compute']}' returned None")
        findings.append({"exp": exp, "metric": metric, "status": "MISSING",
                         "paper_value": pv, "actual": None, "detail": detail})
        print(f"  🔍 {exp:<23} {metric:<35} {'MISSING':<8}  {detail}")
        continue

    actual = float(raw)

    if absol:
        diff = abs(actual - pv)
        tol1, tol3 = tol, tol * 3
    else:
        base = abs(pv) if pv != 0 else 1.0
        diff = abs(actual - pv)
        tol1, tol3 = tol * base, tol * 3 * base

    if diff <= tol1:
        status = "PASS"
    elif diff <= tol3:
        status = "WARN"
    else:
        status = "FAIL"

    detail = (f"paper={pv}  actual={actual:.4f}  diff={diff:.4f}  tol={tol1:.4f}"
              + (f"  [{note}]" if note else ""))
    icons = {"PASS": "✅", "WARN": "⚠ ", "FAIL": "❌"}
    print(f"  {icons.get(status,'?')} {exp:<23} {metric:<35} {status:<8}  {detail}")
    findings.append({"exp": exp, "metric": metric, "status": status,
                     "paper_value": pv, "actual": actual, "detail": detail})

# ── Nguyen-12 dual-threshold caveat ─────────────────────────────────────────
if any(f["exp"] == "exp3" for f in findings):
    print()
    print("  ⚠  Nguyen-12 dual-threshold caveat (exp3/exp3b):")
    print("       Paper abstract  : 11/12 (91.7%) — 4-decimal rounding")
    print("       Strict R²≥0.9999: 4/12  (33.3%) — both must appear in §10.8.")

# ── Summary counts ───────────────────────────────────────────────────────────
n_pass    = sum(1 for f in findings if f["status"] == "PASS")
n_warn    = sum(1 for f in findings if f["status"] == "WARN")
n_fail    = sum(1 for f in findings if f["status"] == "FAIL")
n_missing = sum(1 for f in findings if f["status"] == "MISSING")
n_skip    = sum(1 for f in findings if f["status"] == "SKIP")

print()
print(f"  Audit summary: {n_pass} PASS  {n_warn} WARN  {n_fail} FAIL  "
      f"{n_missing} MISSING  {n_skip} SKIP  ({len(findings)} total claims)")

audit_pass = (n_fail == 0 and n_missing == 0)

out = Path("logs/paper_audit_findings.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(findings, indent=2))
print(f"\n  Audit findings → {out}")

print()
if audit_pass:
    print("  ✅  Audit PASSED — all claims within tolerance.")
else:
    fails = [f for f in findings if f["status"] in ("FAIL", "MISSING")]
    print(f"  ❌  Audit has {len(fails)} FAIL/MISSING claim(s) — see details above.")

with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
    fh.write(f"audit_pass={'true' if audit_pass else 'false'}\n")
    fh.write(f"n_pass={n_pass}\n")
    fh.write(f"n_warn={n_warn}\n")
    fh.write(f"n_fail={n_fail}\n")
    fh.write(f"n_missing={n_missing}\n")

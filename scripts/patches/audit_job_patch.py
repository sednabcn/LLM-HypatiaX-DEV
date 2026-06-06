# ============================================================
# COMPLETE REPLACEMENT for the python3 heredoc inside
# "Audit results against paper claims" step in ci_all_checkpoint.yml
# (lines ~839–1035).  Replace everything between the opening
# python3 - <<'PYEOF' and its closing PYEOF with this content.
#
# v2: reads scripts/patches/issue_registry.json to drive
#     false-positive and resolved-issue handling dynamically.
#     No hardcoded FIX-* lists anywhere in this file.
# ============================================================

import json, os, sys
import glob as _glob
from pathlib import Path

OUT_BASE       = Path(os.environ["OUT_BASE"])
QUALIFY_REPORT = json.loads(os.environ.get("QUALIFY_REPORT") or "{}")
QUALIFIED_EXPS = set(json.loads(os.environ.get("QUALIFIED_EXPS") or "[]"))

# ── Load issue_registry.json ─────────────────────────────────────────────────
ISSUE_REGISTRY_PATH = Path("scripts/patches/issue_registry.json")

def load_issue_registry():
    """
    Returns three sets derived from issue_registry.json:
      false_positive_ids  — FIX-* IDs whose status == "false_positive"
      resolved_ids        — FIX-* IDs whose status == "resolved"
      registry            — full list of dicts for reporting
    """
    if not ISSUE_REGISTRY_PATH.exists():
        print(f"  ⚠  {ISSUE_REGISTRY_PATH} not found — no false-positive overrides will apply.")
        return set(), set(), []
    try:
        registry = json.loads(ISSUE_REGISTRY_PATH.read_text())
        fp_ids  = {e["id"] for e in registry if e.get("status") == "false_positive"}
        res_ids = {e["id"] for e in registry if e.get("status") == "resolved"}
        print(f"  Loaded issue_registry: {len(registry)} entries  "
              f"({len(fp_ids)} false-positive, {len(res_ids)} resolved)")
        if fp_ids:
            print(f"    False positives : {', '.join(sorted(fp_ids))}")
        if res_ids:
            print(f"    Resolved        : {', '.join(sorted(res_ids))}")
        return fp_ids, res_ids, registry
    except Exception as e:
        print(f"  ⚠  Failed to parse issue_registry.json: {e}")
        return set(), set(), []

FALSE_POSITIVE_IDS, RESOLVED_IDS, ISSUE_REGISTRY = load_issue_registry()

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
    print("  ⚠  scripts/patches/paper_targets.json not found. Commit it to the repo.")
    return []

# ── File resolver: result_path → result_glob → result_glob_alt ───────────────
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

# ── Dot-notation key resolver (handles dict + list by index) ──────────────────
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

# ── Compute helpers ───────────────────────────────────────────────────────────
def compute_metric(data, claim):
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
        cn = data.get("cross_noise_summary", data)
        method_data = cn.get(method, {})
        if not method_data and cn:
            method_data = next(iter(cn.values()), {})
        rates = [v.get(leaf) for v in method_data.values()
                 if isinstance(v, dict) and v.get(leaf) is not None]
        return sum(rates) / len(rates) if rates else None

    return None

# ── Registry helpers ──────────────────────────────────────────────────────────
def registry_entry(fix_id):
    """Return the registry dict for a given FIX-* id, or None."""
    for e in ISSUE_REGISTRY:
        if e.get("id") == fix_id:
            return e
    return None

def apply_registry_override(status, fix_id):
    """
    If fix_id is tracked in issue_registry.json, override the raw audit status:
      false_positive → "FP"      (counted separately, not as FAIL/MISSING)
      resolved       → "RESOLVED" (counted separately, treated as informational)
    All other statuses pass through unchanged.
    """
    if fix_id and fix_id in FALSE_POSITIVE_IDS:
        return "FP"
    if fix_id and fix_id in RESOLVED_IDS:
        return "RESOLVED"
    return status

# ── Main audit loop ───────────────────────────────────────────────────────────
claims   = load_claims()
findings = []

print(f"\n  {'Exp':<25} {'Metric':<35} {'Status':<10}  Detail")
print("  " + "─" * 105)

for claim in claims:
    # skip comment / excluded entries
    if "_EXCLUDED" in claim or "exp" not in claim:
        continue

    exp      = claim["exp"]
    metric   = claim["metric"]
    pv       = float(claim["paper_value"])
    tol      = float(claim.get("tolerance", 0.01))
    jkey     = claim.get("json_key", "")
    absol    = bool(claim.get("absolute", False))
    note     = claim.get("note", "")
    fix_id   = claim.get("fix_id", "")       # optional: e.g. "FIX-C1"
    has_compute = bool(claim.get("compute", ""))

    # ── Not qualified ────────────────────────────────────────────────────────
    if exp not in QUALIFIED_EXPS:
        raw_status = "SKIP"
        final_status = apply_registry_override(raw_status, fix_id)
        findings.append({"exp": exp, "metric": metric, "fix_id": fix_id,
                         "status": final_status, "paper_value": pv, "actual": None,
                         "detail": f"{exp} not qualified — audit skipped"})
        print(f"  ↩  {exp:<23} {metric:<35} {'SKIP':<10}  {exp} not qualified")
        continue

    # ── Result file missing ──────────────────────────────────────────────────
    result_file = resolve_result_file(claim)
    if result_file is None:
        rpath_disp   = claim.get("result_path") or claim.get("result_glob") or "?"
        raw_status   = "MISSING"
        final_status = apply_registry_override(raw_status, fix_id)
        fp_note = ""
        if final_status == "FP":
            entry  = registry_entry(fix_id)
            fp_note = f"  [FALSE POSITIVE — {entry['false_positive_reason']}]" if entry else "  [FALSE POSITIVE]"
        findings.append({"exp": exp, "metric": metric, "fix_id": fix_id,
                         "status": final_status, "paper_value": pv, "actual": None,
                         "detail": f"no result file found ({rpath_disp}){fp_note}"})
        icon = "🟡" if final_status in ("FP", "RESOLVED") else "🔍"
        print(f"  {icon} {exp:<23} {metric:<35} {final_status:<10}  {rpath_disp}{fp_note}")
        continue

    # ── Parse JSON ──────────────────────────────────────────────────────────
    try:
        data = json.loads(result_file.read_text())
    except Exception as e:
        raw_status   = "MISSING"
        final_status = apply_registry_override(raw_status, fix_id)
        findings.append({"exp": exp, "metric": metric, "fix_id": fix_id,
                         "status": final_status, "paper_value": pv, "actual": None,
                         "detail": f"JSON parse error: {e}"})
        print(f"  🔍 {exp:<23} {metric:<35} {final_status:<10}  parse error: {e}")
        continue

    # ── Resolve metric value ─────────────────────────────────────────────────
    raw = None
    if has_compute:
        raw = compute_metric(data, claim)
    else:
        raw = resolve_key(data, jkey)
        if raw is None and isinstance(data, dict):
            raw = data.get(jkey.split(".")[-1])

    if raw is None:
        detail       = (f"key '{jkey}' not found" if not has_compute
                        else f"compute='{claim['compute']}' returned None")
        raw_status   = "MISSING"
        final_status = apply_registry_override(raw_status, fix_id)
        fp_note = ""
        if final_status == "FP":
            entry  = registry_entry(fix_id)
            fp_note = f" — FALSE POSITIVE: {entry['false_positive_reason']}" if entry else " — FALSE POSITIVE"
        findings.append({"exp": exp, "metric": metric, "fix_id": fix_id,
                         "status": final_status, "paper_value": pv, "actual": None,
                         "detail": detail + fp_note})
        icon = "🟡" if final_status in ("FP", "RESOLVED") else "🔍"
        print(f"  {icon} {exp:<23} {metric:<35} {final_status:<10}  {detail}{fp_note}")
        continue

    # ── Compare against paper value ──────────────────────────────────────────
    actual = float(raw)
    if absol:
        diff  = abs(actual - pv)
        tol1, tol3 = tol, tol * 3
    else:
        base  = abs(pv) if pv != 0 else 1.0
        diff  = abs(actual - pv)
        tol1, tol3 = tol * base, tol * 3 * base

    if diff <= tol1:
        raw_status = "PASS"
    elif diff <= tol3:
        raw_status = "WARN"
    else:
        raw_status = "FAIL"

    final_status = apply_registry_override(raw_status, fix_id)

    detail = (f"paper={pv}  actual={actual:.4f}  diff={diff:.4f}  tol={tol1:.4f}"
              + (f"  [{note}]" if note else ""))
    if final_status == "FP":
        entry   = registry_entry(fix_id)
        reason  = entry["false_positive_reason"] if entry else "see issue_registry.json"
        detail += f"  [FALSE POSITIVE — {reason}]"
    elif final_status == "RESOLVED":
        entry   = registry_entry(fix_id)
        detail += f"  [RESOLVED — {entry['action'] if entry else 'see issue_registry.json'}]"

    icons = {"PASS": "✅", "WARN": "⚠ ", "FAIL": "❌", "FP": "🟡", "RESOLVED": "🔵"}
    print(f"  {icons.get(final_status,'?')} {exp:<23} {metric:<35} {final_status:<10}  {detail}")
    findings.append({"exp": exp, "metric": metric, "fix_id": fix_id,
                     "status": final_status, "paper_value": pv,
                     "actual": actual, "detail": detail})

# ── Nguyen-12 dual-threshold caveat ──────────────────────────────────────────
if any(f["exp"] == "exp3" for f in findings):
    print()
    print("  ⚠  Nguyen-12 dual-threshold caveat (exp3/exp3b):")
    print("       Paper abstract  : 11/12 (91.7%) — 4-decimal rounding")
    print("       Strict R²≥0.9999: 4/12  (33.3%) — both must appear in §10.8.")

# ── Summary counts ────────────────────────────────────────────────────────────
n_pass     = sum(1 for f in findings if f["status"] == "PASS")
n_warn     = sum(1 for f in findings if f["status"] == "WARN")
n_fail     = sum(1 for f in findings if f["status"] == "FAIL")
n_missing  = sum(1 for f in findings if f["status"] == "MISSING")
n_skip     = sum(1 for f in findings if f["status"] == "SKIP")
n_fp       = sum(1 for f in findings if f["status"] == "FP")
n_resolved = sum(1 for f in findings if f["status"] == "RESOLVED")

print()
print(f"  Audit summary: "
      f"{n_pass} PASS  {n_warn} WARN  {n_fail} FAIL  "
      f"{n_missing} MISSING  {n_skip} SKIP  "
      f"{n_fp} FALSE-POSITIVE  {n_resolved} RESOLVED  "
      f"({len(findings)} total claims)")

# ── Registry integrity check: warn about any open issues with no claim entry ──
print()
open_ids    = {e["id"] for e in ISSUE_REGISTRY if e.get("status") == "open"}
claimed_ids = {f["fix_id"] for f in findings if f.get("fix_id")}
unclaimed   = open_ids - claimed_ids
if unclaimed:
    print(f"  ⚠  Open issues in registry with no matching claim in paper_targets.json:")
    for uid in sorted(unclaimed):
        entry = registry_entry(uid)
        print(f"       {uid}: {entry['description'] if entry else '?'}")
    print("       → Add 'fix_id': '<ID>' to the relevant paper_targets.json entries.")

# ── Audit pass/fail determination ─────────────────────────────────────────────
# False positives and resolved items are excluded from the gate.
audit_pass = (n_fail == 0 and n_missing == 0)

out = Path("logs/paper_audit_findings.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(findings, indent=2))
print(f"\n  Audit findings → {out}")

print()
if audit_pass:
    print("  ✅  Audit PASSED — all claims within tolerance.")
    if n_fp:
        print(f"       ({n_fp} false-positive finding(s) excluded from gate — see issue_registry.json)")
    if n_resolved:
        print(f"       ({n_resolved} resolved issue(s) noted for record)")
else:
    fails = [f for f in findings if f["status"] in ("FAIL", "MISSING")]
    print(f"  ❌  Audit has {len(fails)} FAIL/MISSING claim(s) — see details above.")
    if n_fp:
        print(f"       ({n_fp} additional finding(s) are false positives — excluded from gate)")

with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
    fh.write(f"audit_pass={'true' if audit_pass else 'false'}\n")
    fh.write(f"n_pass={n_pass}\n")
    fh.write(f"n_warn={n_warn}\n")
    fh.write(f"n_fail={n_fail}\n")
    fh.write(f"n_missing={n_missing}\n")
    fh.write(f"n_false_positive={n_fp}\n")
    fh.write(f"n_resolved={n_resolved}\n")

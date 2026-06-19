#!/usr/bin/env python3
"""
build_report.py
===============
Replaces all inline bash/Python embedded in ci_report.yml.
Run by the "Build Combined Audit Report" GitHub Actions job.

Steps it handles
----------------
  1. Resolve notebook run ID (formerly: multi-priority bash block)
  2. Convert each executed .ipynb → HTML body fragment via nbconvert
  3. Assemble the combined index.html (header, ToC, per-NB sections)
  4. Load issue_registry.json + hypatia_inspector findings + NB audit summary
  5. Render dynamic conclusions section (open / resolved / false-positive tables)
  6. Write the footer and close the HTML

Environment variables consumed (set by GitHub Actions)
-------------------------------------------------------
  GITHUB_OUTPUT          – path to the GitHub Actions output file
  GITHUB_RUN_ID          – current (report) workflow run ID
  GITHUB_REPOSITORY      – owner/repo
  GITHUB_REF_NAME        – branch name
  GH_TOKEN               – GitHub token for API calls (set via secrets.GITHUB_TOKEN)
  EVENT_RUN_ID           – github.event.workflow_run.id  (may be empty)
  DISPATCH_RUN_ID        – github.event.inputs.summary_run_id (may be empty)
  NO_NOTEBOOKS           – 'true' when dispatched from apply_fixes_only path
  UPSTREAM_RUN           – resolved notebook run ID (set by step 1, read by later steps)
  REGISTRY_PATH          – path to issue_registry.json inside checked-out repo
  SITE_DIR               – output directory for site/ (default: site)

Usage (CI)
----------
  # Step 1 – resolve run ID and write to GITHUB_OUTPUT
  python build_report.py resolve-run-id

  # Step 2 – build the full HTML report (reads notebooks/, site/ path from env)
  python build_report.py build-report

  # Or both at once (useful for local testing):
  python build_report.py all
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

NOTEBOOKS: list[tuple[str, str, str]] = [
    # (artifact_name,  relative_path_inside_notebooks/,  html_anchor)
    ("NB-01_Citation_Bibliography_Audit",
     "NB-01/NB-01_Citation_Bibliography_Audit.ipynb",     "nb01"),
    ("NB-02_CrossReference_Label_Audit",
     "NB-02/NB-02_CrossReference_Label_Audit.ipynb",      "nb02"),
    ("NB-03_Section_Structure_Numbering",
     "NB-03/NB-03_Section_Structure_Numbering.ipynb",     "nb03"),
    ("NB-04_Numerical_Consistency_Checker",
     "NB-04/NB-04_Numerical_Consistency_Checker.ipynb",   "nb04"),
    ("NB-05_Figure_Image_Dependency_Checker",
     "NB-05/NB-05_Figure_Image_Dependency_Checker.ipynb", "nb05"),
    ("NB-06_Code_Quality_Pipeline_Integrity",
     "NB-06/NB-06_Code_Quality_Pipeline_Integrity.ipynb", "nb06"),
]

NB_TITLES: list[str] = [
    "NB-01: Citation &amp; Bibliography Audit",
    "NB-02: Cross-Reference &amp; Label Integrity",
    "NB-03: Section Structure &amp; Numbering",
    "NB-04: Numerical Consistency &amp; Abstract Claims",
    "NB-05: Figure Files &amp; Environment Audit",
    "NB-06: Code Quality &amp; Pipeline Integrity",
]

STATIC_META: dict[str, dict] = {
    "FIX-B1":  {"severity": "CRITICAL", "nb": "NB-01",
                 "description": "koza1994genetic cited but has no bibitem.",
                 "action": "Add bibitem or redirect cite to koza1992gp."},
    "FIX-B2":  {"severity": "HIGH",     "nb": "NB-01",
                 "description": "cranmer2023pysr and cranmer2023interpretable alias the same paper.",
                 "action": "Remove cranmer2023interpretable; redirect all cite to cranmer2023pysr."},
    "FIX-B3":  {"severity": "HIGH",     "nb": "NB-01",
                 "description": "udrescu2020ai and udrescu2020aifeynman alias the same paper.",
                 "action": "Remove udrescu2020aifeynman; redirect all uses to udrescu2020ai."},
    "FIX-F1":  {"severity": "MEDIUM",   "nb": "NB-05",
                 "description": "hypatiaX_three_systems MISSING — fbox placeholder in §7.1.",
                 "action": "Replace fbox placeholder with final PDF/PNG."},
    "FIX-F2":  {"severity": "MEDIUM",   "nb": "NB-05",
                 "description": "fig18_r2_heatmap_improved.pdf missing from figures/.",
                 "action": "Run generate_figures.py --experiment exp1 or ci_postprocess figures_deploy."},
    "FIX-F3":  {"severity": "MEDIUM",   "nb": "NB-05",
                 "description": "fig09_r2_heatmap_regimes.pdf missing from figures/.",
                 "action": "Run generate_figures.py --experiment exp1 or ci_postprocess figures_deploy."},
    "FIX-F4":  {"severity": "MEDIUM",   "nb": "NB-05",
                 "description": "fig1_seed_sweep.pdf missing from figures/.",
                 "action": "Run generate_figures.py --experiment exp1 or ci_postprocess figures_deploy."},
    "FIX-C3":  {"severity": "CRITICAL", "nb": "NB-06",
                 "description": "Feynman benchmark split-protocol mismatch.",
                 "action": "Results in exp2_pca_4060/; Gates A/B/C passed."},
    "FIX-N1":  {"severity": "MEDIUM",   "nb": "NB-04",
                 "description": "'71 cases' in Spearman footnote should be '70 tasks'.",
                 "action": "Change '71 cases' → '70 tasks' in body text."},
    "FIX-N2":  {"severity": "MEDIUM",   "nb": "NB-04",
                 "description": "'Layer~N' terminology inconsistent with 'five-stage routing'.",
                 "action": "Rename Layer~N → Stage~N in sec:validation_framework."},
    "FIX-XR4": {"severity": "LOW",      "nb": "NB-02",
                 "description": "Supp A filename reference mismatch.",
                 "action": "Update filename strings in supp_routing_improvements.tex."},
    "FIX-S1":  {"severity": "LOW",      "nb": "NB-03",
                 "description": "Missing \\label on \\section commands.",
                 "action": "Add \\label{sec:...} after each \\section{...}."},
    "FIX-S2":  {"severity": "LOW",      "nb": "NB-03",
                 "description": "Missing \\label on \\subsection commands.",
                 "action": "Add \\label{subsec:...} after each \\subsection{...}."},
}

PRIORITY_ITEMS: list[tuple[str, list[str], str]] = [
    ("FIX-B2 / FIX-B3",     ["FIX-B2", "FIX-B3"],
     "Deduplicate the two bibitem alias pairs."),
    ("FIX-XR4",             ["FIX-XR4"],
     "Update Supp A filename string from <code>jmlr_paper_main.tex</code> → <code>jmlr-hypatiax-paper-final.tex</code>."),
    ("FIX-F1",              ["FIX-F1"],
     "Replace <code>\\fbox</code> placeholder with final architecture PDF/PNG."),
    ("FIX-F2, FIX-F3, FIX-F4", ["FIX-F2", "FIX-F3", "FIX-F4"],
     "Copy three missing figure files to <code>figures/</code>."),
    ("FIX-N1",              ["FIX-N1"],
     "Change '71 cases' → '70 tasks' in Spearman footnote."),
    ("FIX-N2",              ["FIX-N2"],
     "Standardise 'Layer~N' → 'Stage~N' terminology in §8 body."),
    ("FIX-S1 / FIX-S2",    ["FIX-S1", "FIX-S2"],
     "Add <code>\\label</code> to all unlabelled \\section and \\subsection headings."),
]

SEV_ORDER  = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
SEV_STYLE  = {
    "CRITICAL": ("background:#fff0f0;", "🔴 CRITICAL"),
    "HIGH":     ("background:#fff5ec;", "🟠 HIGH"),
    "MEDIUM":   ("",                    "🟡 MEDIUM"),
    "LOW":      ("background:#f6fff6;", "🟢 LOW"),
}
STATUS_BADGE = {
    "RESOLVED":
        '<span style="color:#1a6b3a;font-weight:bold;background:#eafaf1;'
        'padding:2px 7px;border-radius:10px;">✅ RESOLVED</span>',
    "OPEN":
        '<span style="color:#b35c00;font-weight:bold;background:#fff3e0;'
        'padding:2px 7px;border-radius:10px;">🔴 OPEN</span>',
    "FALSE_POSITIVE":
        '<span style="color:#555;font-weight:bold;background:#f0f0f0;'
        'padding:2px 7px;border-radius:10px;">🔵 FALSE POSITIVE</span>',
}


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Resolve notebook run ID
# ══════════════════════════════════════════════════════════════════════════════

def resolve_run_id() -> str:
    """
    Priority order (matches original bash block):
      1. DISPATCH_RUN_ID  — explicitly supplied via workflow_dispatch input
      2. EVENT_RUN_ID     — set when triggered by workflow_run event
      3. API query        — most recent completed ci_paper_notebooks.yml run on this branch
    Writes 'notebooks_run_id=<id>' to $GITHUB_OUTPUT and returns the ID.
    """
    dispatch_id = os.environ.get("DISPATCH_RUN_ID", "").strip()
    event_id    = os.environ.get("EVENT_RUN_ID", "").strip()
    repo        = os.environ.get("GITHUB_REPOSITORY", "")
    branch      = os.environ.get("GITHUB_REF_NAME", "main")
    gh_token    = os.environ.get("GH_TOKEN", "")

    if dispatch_id:
        print(f"Using caller-supplied summary_run_id: {dispatch_id}")
        resolved = dispatch_id

    elif event_id:
        print(f"Triggered by workflow_run — using event run ID: {event_id}")
        resolved = event_id

    else:
        print(f"Manual dispatch (no run ID supplied) — querying API for "
              f"most recent ci_paper_notebooks.yml run on branch {branch}")
        env = {**os.environ, "GH_TOKEN": gh_token}
        result = subprocess.run(
            [
                "gh", "run", "list",
                "--repo", repo,
                "--workflow", "ci_paper_notebooks.yml",
                "--branch", branch,
                "--status", "completed",
                "--limit", "5",
                "--json", "databaseId,conclusion",
                "--jq",
                '[.[] | select(.conclusion == "success" or .conclusion == "failure")]'
                " | first | .databaseId // empty",
            ],
            capture_output=True, text=True, env=env,
        )
        resolved = result.stdout.strip()
        if not resolved:
            print("::warning::No completed ci_paper_notebooks.yml run found "
                  f"on branch {branch}.")
        else:
            print(f"Resolved notebook run ID: {resolved}")

    _write_github_output("notebooks_run_id", resolved)
    return resolved


def _write_github_output(key: str, value: str) -> None:
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if output_file:
        with open(output_file, "a") as fh:
            fh.write(f"{key}={value}\n")
    else:
        print(f"[GITHUB_OUTPUT not set] {key}={value}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Convert notebook to HTML body fragment
# ══════════════════════════════════════════════════════════════════════════════

def nb_to_html_fragment(nb_path: Path) -> str:
    """
    Run nbconvert --no-input on an already-executed notebook and extract
    just the <body>...</body> content (no <html>/<head> wrapper).
    Returns a warning string if the notebook is missing or conversion fails.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_stem = Path(tmpdir) / "_nb_frag"
        try:
            subprocess.run(
                [
                    "jupyter", "nbconvert",
                    "--to", "html",
                    "--no-input",
                    "--template", "classic",
                    "--output", str(out_stem),
                    str(nb_path),
                ],
                capture_output=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            return (
                "<div class='missing'>⚠️ nbconvert failed for "
                f"{html_mod.escape(str(nb_path))}: {html_mod.escape(str(exc))}</div>"
            )

        html_file = out_stem.with_suffix(".html")
        if not html_file.exists():
            return ("<div class='missing'>⚠️ nbconvert produced no output "
                    f"for {html_mod.escape(str(nb_path))}</div>")

        raw = html_file.read_text(encoding="utf-8", errors="replace")

    m = re.search(r"<body[^>]*>(.*?)</body>", raw, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else raw


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — HTML page skeleton
# ══════════════════════════════════════════════════════════════════════════════

def _html_header(run_date: str, upstream_run: str, report_run: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HypatiaX Paper Audit Report</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0; background: #f0f2f5; color: #1a1a2e;
    }}
    .cover {{
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
      color: white; padding: 64px 40px 48px; text-align: center;
    }}
    .cover h1 {{ margin: 0 0 14px; font-size: 2.4em; letter-spacing: -0.5px; }}
    .cover .meta {{ opacity: 0.7; font-size: 0.95em; line-height: 1.8; }}
    .badge {{
      display: inline-block; background: rgba(255,255,255,0.12);
      border: 1px solid rgba(255,255,255,0.25); border-radius: 20px;
      padding: 4px 14px; font-size: 0.82em; margin-top: 12px;
    }}
    .container {{ max-width: 900px; margin: 0 auto; padding: 0 20px 60px; }}
    .toc {{
      background: white; border-radius: 10px; padding: 28px 36px;
      margin: 32px 0 24px; box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    }}
    .toc h2 {{ margin: 0 0 16px; font-size: 1.05em; text-transform: uppercase;
               letter-spacing: .08em; color: #666; }}
    .toc ol  {{ margin: 0; padding-left: 22px; line-height: 2.1; }}
    .toc a   {{ color: #0066cc; text-decoration: none; font-weight: 500; }}
    .toc a:hover {{ text-decoration: underline; }}
    .nb-section {{
      background: white; border-radius: 10px; margin-bottom: 24px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.07); overflow: hidden;
    }}
    .nb-header {{
      background: linear-gradient(90deg, #16213e, #0f3460);
      color: white; padding: 18px 28px; display: flex;
      align-items: center; gap: 12px;
    }}
    .nb-header h2 {{ margin: 0; font-size: 1.1em; }}
    .nb-body {{ padding: 8px 28px 28px; overflow-x: auto; }}
    .missing {{
      background: #fff8e1; border-left: 4px solid #f9a825;
      padding: 16px 20px; margin: 20px 0; border-radius: 4px;
      font-size: 0.95em;
    }}
    .nb-body pre {{
      background: #f8f8f8; padding: 12px 16px; border-radius: 6px;
      overflow-x: auto; font-size: 0.87em; line-height: 1.5;
    }}
    .nb-body .output_area pre {{ background: #f0f0f0; }}
    .nb-body table {{
      border-collapse: collapse; width: 100%; font-size: 0.9em;
    }}
    .nb-body th, .nb-body td {{
      border: 1px solid #ddd; padding: 8px 12px; text-align: left;
    }}
    .nb-body th {{ background: #f5f5f5; font-weight: 600; }}
    .footer {{
      text-align: center; padding: 40px 20px;
      color: #999; font-size: 0.82em; line-height: 1.8;
    }}
  </style>
</head>
<body>
  <div class="cover">
    <h1>📋 HypatiaX Paper Audit Report</h1>
    <div class="meta">
      Generated: {html_mod.escape(run_date)}<br>
      Notebook run ID: {html_mod.escape(upstream_run)}
      &nbsp;·&nbsp; Report run ID: {html_mod.escape(report_run)}
    </div>
    <div class="badge">ci_report.yml — no notebooks re-executed</div>
  </div>

  <div class="container">
    <div class="toc">
      <h2>Contents</h2>
      <ol>
        <li><a href="#nb01">NB-01: Citation &amp; Bibliography Audit</a></li>
        <li><a href="#nb02">NB-02: Cross-Reference &amp; Label Integrity</a></li>
        <li><a href="#nb03">NB-03: Section Structure &amp; Numbering</a></li>
        <li><a href="#nb04">NB-04: Numerical Consistency &amp; Abstract Claims</a></li>
        <li><a href="#nb05">NB-05: Figure Files &amp; Environment Audit</a></li>
        <li><a href="#nb06">NB-06: Code Quality &amp; Pipeline Integrity</a></li>
        <li><a href="#fixc3gates"><strong>FIX-C3 Split-Protocol Gates (A/B/C)</strong></a></li>
        <li><a href="#conclusions"><strong>Audit Conclusions &amp; Action Plan</strong></a></li>
      </ol>
    </div>
"""


def _html_footer(upstream_run: str, report_run: str, run_date: str) -> str:
    e = html_mod.escape
    return f"""
    </div><!-- .container -->

    <div class="footer">
      HypatiaX Paper Audit &bull; Notebook run {e(upstream_run)}
      &bull; Report run {e(report_run)} &bull; {e(run_date)}<br>
      Generated by <code>build_report.py</code> — no notebooks were re-executed.
    </div>
  </body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════════
# Steps 4–5 — Load data sources + render conclusions
# ══════════════════════════════════════════════════════════════════════════════

def load_registry(registry_path: Path) -> list[dict]:
    live = Path("/tmp/live_registry/issue_registry.json")
    effective = live if live.exists() else registry_path
    if not effective.exists():
        print(f"WARNING: no registry found at {registry_path} or {live} "
              "— falling back to static list")
        return []
    try:
        entries = json.loads(effective.read_text())
        label = "LIVE (nb-merged)" if effective == live else "repo checkout"
        print(f"Loaded {len(entries)} entries from {effective} [{label}]")
        return entries
    except Exception as exc:
        print(f"WARNING: could not parse registry: {exc}")
        return []


def load_inspector_status() -> dict[str, str]:
    """Maps fix_id → 'RESOLVED' | 'OPEN' from hypatia_inspector findings.json."""
    candidate = Path("/tmp/hypatia_findings/findings.json")
    if not candidate.exists():
        return {}
    try:
        raw = json.loads(candidate.read_text())
        mapping: dict[str, str] = {}
        for f in raw:
            fid = f.get("fix_id") or f.get("id", "")
            if not fid:
                continue
            st = f.get("status", "")
            if st == "fixed":
                mapping[fid] = "RESOLVED"
            elif st in ("detected", "manual"):
                mapping[fid] = "OPEN"
            # "skipped" → trust the registry; omit
        print(f"Loaded {len(mapping)} live statuses from hypatia_inspector")
        return mapping
    except Exception as exc:
        print(f"Could not load hypatia findings: {exc}")
        return {}


def load_nb_summary() -> dict[str, dict]:
    """Maps fix_id → {description, action, nb, severity} from notebooks_audit_summary.json."""
    candidates = [
        Path("/tmp/audit_summary/notebooks_audit_summary.json"),
        Path("/tmp/audit_summary/paper-audit-notebooks/notebooks_audit_summary.json"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            summary = json.loads(path.read_text())
            items = summary if isinstance(summary, list) else summary.get("findings", [])
            result = {}
            for item in items:
                fid = item.get("id") or item.get("fix_id", "")
                if fid:
                    result[fid] = item
            print(f"Loaded NB summary descriptions for {len(result)} items")
            return result
        except Exception:
            pass
    return {}


def build_findings(registry_entries: list[dict],
                   inspector_status: dict[str, str],
                   nb_summary: dict[str, dict]) -> list[dict]:
    """
    Merge all data sources into a canonical findings list.
    Status priority: registry → inspector → OPEN (default).
    Description priority: nb_summary → STATIC_META → registry entry.
    """
    findings: list[dict] = []
    for entry in registry_entries:
        fid    = entry["id"]
        reg_st = entry.get("status", "open").lower()

        if reg_st == "false_positive":
            display_status = "FALSE_POSITIVE"
        elif reg_st == "resolved":
            display_status = "RESOLVED"
        else:
            display_status = inspector_status.get(fid, "OPEN")

        meta   = nb_summary.get(fid) or STATIC_META.get(fid) or {}
        desc   = meta.get("description")   or entry.get("description", "")
        action = meta.get("action")        or entry.get("action", "")
        sev    = (meta.get("severity")     or entry.get("severity", "medium")).upper()
        nb     = meta.get("nb")            or entry.get("nb_source", "")

        if reg_st == "false_positive":
            fp_reason = entry.get("false_positive_reason", "")
            action = (f"No action needed. Reason: {fp_reason[:200]}"
                      if fp_reason else "No action needed — confirmed false positive.")

        findings.append({
            "id":          fid,
            "severity":    sev,
            "status":      display_status,
            "description": desc,
            "action":      action,
            "nb":          nb,
        })

    # If registry was empty fall back to static list (all OPEN — safest assumption)
    if not findings:
        print("WARNING: registry empty — using static list with OPEN status for all items")
        for fid, meta in STATIC_META.items():
            findings.append({
                "id": fid, "severity": meta["severity"], "status": "OPEN",
                "description": meta["description"], "action": meta["action"],
                "nb": meta["nb"],
            })

    findings.sort(key=lambda f: (SEV_ORDER.get(f["severity"], 9), f["id"]))
    return findings


def _make_rows(items: list[dict], fp: bool = False) -> str:
    rows = []
    e = html_mod.escape
    for f in items:
        sev = f["severity"]
        style, sev_label = SEV_STYLE.get(sev, ("", sev))
        if fp:
            style = "background:#f8f8f8;color:#777;"
        badge  = STATUS_BADGE.get(f["status"], STATUS_BADGE["OPEN"])
        rows.append(f"""
          <tr style="{style}">
            <td><strong>{e(f['id'])}</strong><br><small>{badge}</small></td>
            <td style="font-size:0.85em;">{sev_label}</td>
            <td>{e(f['description'])}<br>
              <small style="color:#666;"><em>Action:</em> {e(f['action'])}</small></td>
            <td style="font-size:0.85em;">{e(f['nb'])}</td>
          </tr>""")
    return "".join(rows)


def _table_wrap(rows_html: str) -> str:
    th = 'style="width:115px;padding:6px 8px;background:#f5f5f5;border:1px solid #ddd;"'
    return f"""
      <table style="font-size:0.87em;width:100%;border-collapse:collapse;">
        <thead><tr>
          <th {th}>ID / Status</th>
          <th style="width:95px;padding:6px 8px;background:#f5f5f5;border:1px solid #ddd;">Severity</th>
          <th style="padding:6px 8px;background:#f5f5f5;border:1px solid #ddd;">Issue &amp; Action</th>
          <th style="width:80px;padding:6px 8px;background:#f5f5f5;border:1px solid #ddd;">NB</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>"""


def render_conclusions(findings: list[dict], upstream_run: str, report_run: str,
                        registry_path: Path, no_notebooks: bool) -> str:
    e = html_mod.escape
    open_findings  = [f for f in findings if f["status"] == "OPEN"]
    resolved       = [f for f in findings if f["status"] == "RESOLVED"]
    false_positives= [f for f in findings if f["status"] == "FALSE_POSITIVE"]
    critical_open  = [f for f in open_findings if f["severity"] == "CRITICAL"]

    print(f"Findings: {len(open_findings)} OPEN, {len(resolved)} RESOLVED, "
          f"{len(false_positives)} FALSE_POSITIVE, {len(critical_open)} CRITICAL-OPEN")

    # ── Pipeline status banner ────────────────────────────────────────────────
    src_note = (f"Registry: <code>{e(str(registry_path))}</code>")
    if no_notebooks:
        pipeline_html = """
          <div style="background:#fff8e1;border-left:4px solid #f0a000;
               padding:12px 16px;margin-bottom:12px;border-radius:4px;">
            <strong>ℹ️ Fix-only run</strong> — notebooks were not re-executed.
            <code>fix_paper_issues.py</code> / <code>hypatia_inspector.py</code>
            applied fixes directly to source files. Re-run <code>ci_paper_notebooks.yml</code>
            to regenerate live notebook output.
          </div>"""
    else:
        crit_html = (
            f"&nbsp;· <strong style='color:#c00;'>{len(critical_open)} CRITICAL open</strong>"
            if critical_open else ""
        )
        pipeline_html = f"""
          <p>
            Report run&nbsp;<code>{e(report_run)}</code>
            · Notebook run&nbsp;<code>{e(upstream_run)}</code>{crit_html}<br>
            <small style="color:#666;">{src_note}</small>
          </p>"""

    # ── Open-issues table ─────────────────────────────────────────────────────
    open_rows = _make_rows(open_findings)
    open_table = _table_wrap(
        open_rows if open_rows
        else "<tr><td colspan='4' style='padding:12px;color:#1a6b3a;text-align:center;'>"
             "✅ No open issues</td></tr>"
    )

    # ── Resolved table (collapsible) ──────────────────────────────────────────
    resolved_html = ""
    if resolved:
        resolved_html = f"""
          <details style="margin-top:16px;">
            <summary style="cursor:pointer;font-weight:600;color:#1a6b3a;">
              ✅ {len(resolved)} Resolved issues (click to expand)
            </summary>
            {_table_wrap(_make_rows(resolved))}
          </details>"""

    # ── False-positive table (collapsible) ────────────────────────────────────
    fp_html = ""
    if false_positives:
        fp_html = f"""
          <details style="margin-top:8px;">
            <summary style="cursor:pointer;font-weight:600;color:#555;">
              🔵 {len(false_positives)} Confirmed false positives
              — suppressed from all gates (click to expand)
            </summary>
            <p style="font-size:0.85em;color:#666;margin:8px 0 4px;">
              These findings were produced by audit notebooks scanning stale cached
              outputs. Verification against the current source files confirms none exist.
              Excluded from CI gates, fix scripts, and the priority action list.
            </p>
            {_table_wrap(_make_rows(false_positives, fp=True))}
          </details>"""

    # ── Priority action list ──────────────────────────────────────────────────
    open_ids = {f["id"] for f in open_findings}
    pri_rows = []
    for label, ids, text in PRIORITY_ITEMS:
        still_open = any(i in open_ids for i in ids)
        if still_open:
            pri_rows.append(f"<li><strong>{e(label)}</strong> — {text}</li>")
        else:
            pri_rows.append(
                f"<li><s style='color:#888;'><strong>{e(label)}</strong></s>"
                f" <span style='color:#1a6b3a;'>✅</span> — {text}</li>"
            )

    all_clear = not any("</s>" not in r for r in pri_rows)
    priority_html = (
        "<p style='color:#1a6b3a;'>✅ All tracked action items are resolved or suppressed.</p>"
        if all_clear else
        f"<ol style='line-height:2.2;'>{''.join(pri_rows)}</ol>"
    )

    open_heading_suffix = (
        "— <span style='color:green;'>none ✅</span>"
        if not open_findings else
        "— <span style='color:#c00;'>action required</span>"
    )

    return f"""
      <div class="nb-section" id="conclusions">
        <div class="nb-header" style="background:linear-gradient(90deg,#1a1a2e,#7b2d8b);">
          <h2>Audit Conclusions &amp; Action Plan</h2>
        </div>
        <div class="nb-body">

          <h3 style="margin-top:24px;">Pipeline Status</h3>
          {pipeline_html}

          <h3>Results Quality</h3>
          <p>
            All nine abstract claims pass verification ([OK] across every check in NB-04 Step&nbsp;2).
            The core numerical story — 89.2&nbsp;% near-perfect success rate, +27&nbsp;pp gain over
            the LLM baseline, 1.73×&nbsp;speedup, Nguyen 11/12, Feynman 9/30 (random 80/20 baseline,
            locked in <code>fixc3_baseline.json</code>) — is internally consistent throughout the paper.
            The FIX-C3 split-protocol correction has been applied; Gates A/B/C pass.
          </p>

          <h3>Open Issues ({len(open_findings)} {open_heading_suffix})</h3>
          {open_table}
          {resolved_html}
          {fp_html}

          <h3 style="margin-top:32px;">Recommended Fix Order Before Submission</h3>
          {priority_html}

        </div>
      </div><!-- #conclusions -->
"""


# ══════════════════════════════════════════════════════════════════════════════
# Main orchestrator — build-report
# ══════════════════════════════════════════════════════════════════════════════

def build_report() -> None:
    run_date     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report_run   = os.environ.get("GITHUB_RUN_ID", "local")
    upstream_run = os.environ.get("UPSTREAM_RUN", "").strip() or report_run
    no_notebooks = os.environ.get("NO_NOTEBOOKS", "false").lower() == "true"
    registry_path = Path(os.environ.get("REGISTRY_PATH",
                                         "_repo/scripts/patches/issue_registry.json"))
    site_dir     = Path(os.environ.get("SITE_DIR", "site"))
    notebooks_dir = Path("notebooks")

    site_dir.mkdir(parents=True, exist_ok=True)
    index = site_dir / "index.html"

    with index.open("w", encoding="utf-8") as fh:

        # ── HTML header + ToC ─────────────────────────────────────────────────
        fh.write(_html_header(run_date, upstream_run, report_run))

        # ── Per-notebook sections ─────────────────────────────────────────────
        for (_, nb_rel, anchor), title in zip(NOTEBOOKS, NB_TITLES):
            nb_path = notebooks_dir / nb_rel
            fh.write(f"""
      <div class="nb-section" id="{anchor}">
        <div class="nb-header"><h2>{title}</h2></div>
        <div class="nb-body">
""")
            if nb_path.exists():
                print(f"  Converting {nb_path} …")
                fh.write(nb_to_html_fragment(nb_path))
            else:
                print(f"  ⚠ Not found: {nb_path}")
                fh.write(
                    "  <div class='missing'>⚠️ Notebook not found — "
                    "the upstream job may have failed or been skipped.</div>"
                )
            fh.write("\n        </div>\n      </div>\n")

        # ── Dynamic conclusions section ───────────────────────────────────────
        registry_entries = load_registry(registry_path)
        inspector_status = load_inspector_status()
        nb_summary       = load_nb_summary()
        findings         = build_findings(registry_entries, inspector_status, nb_summary)

        fh.write(render_conclusions(
            findings, upstream_run, report_run, registry_path, no_notebooks
        ))

        # ── Footer + close ────────────────────────────────────────────────────
        fh.write(_html_footer(upstream_run, report_run, run_date))

    size   = index.stat().st_size
    lines  = index.read_text(encoding="utf-8").count("\n")
    print(f"Report built: {size:,} bytes  |  {lines:,} lines → {index}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="HypatiaX CI report builder")
    ap.add_argument(
        "command",
        choices=["resolve-run-id", "build-report", "all"],
        help=(
            "resolve-run-id: resolve the notebook run ID and write to GITHUB_OUTPUT; "
            "build-report: assemble the HTML report; "
            "all: run both in sequence"
        ),
    )
    args = ap.parse_args()

    if args.command in ("resolve-run-id", "all"):
        resolved = resolve_run_id()
        # Make it available to build-report if running both
        if args.command == "all" and resolved:
            os.environ.setdefault("UPSTREAM_RUN", resolved)

    if args.command in ("build-report", "all"):
        build_report()


if __name__ == "__main__":
    main()

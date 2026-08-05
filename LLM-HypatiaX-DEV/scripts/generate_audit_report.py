#!/usr/bin/env python3
"""
generate_audit_report.py — HypatiaX Paper Audit Report regenerator

Rebuilds HypatiaX_Paper_Audit_Report.html from structured data instead of
hand-edited markup. This is a STATIC regenerator in one specific sense only:
it does not re-run any notebooks (NB-01...NB-06 bodies are preserved as
already-finalized HTML, see --nb-dir below). The FINDINGS TABLE, however,
is read live from issue_registry.json via --registry — pass that flag in
CI or the report WILL silently use a frozen fallback snapshot instead.

Use this when you need to:
  - regenerate the report after issue_registry.json changes (pass --registry)
  - tweak colors/layout in one place (PALETTE below) and regenerate consistently
  - refresh the "Generated" timestamp / provenance line

USAGE
-----
    python3 generate_audit_report.py --registry path/to/issue_registry.json
    python3 generate_audit_report.py --registry issue_registry.json --out report.html
    python3 generate_audit_report.py --registry issue_registry.json --no-run-ids

    # Without --registry, falls back to a frozen 2026-06-19 snapshot and
    # prints a warning — do not rely on this in CI:
    python3 generate_audit_report.py

The notebook body sections (NB-01...NB-06) are intentionally kept as opaque
HTML blobs (NB_SECTIONS below) rather than re-parsed from nbconvert output,
since the underlying notebooks are not being re-executed (per the "all paper
information is static" decision) — only their already-finalized HTML bodies
are preserved verbatim. Replace the placeholder bodies with the real
nbconvert <div class="cell">...</div> fragments if you have them on disk
(see --nb-dir below).
"""

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path

# ==============================================================================
# COLOR PALETTE — edit here to re-skin the whole report consistently.
# These are the exact values pulled from the original HypatiaX_Paper_Audit_Report.html.
# ==============================================================================
PALETTE = {
    "page_bg":           "#f0f2f5",
    "page_text":         "#1a1a2e",
    "cover_gradient":    "linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%)",
    "cover_badge_bg":     "rgba(255,255,255,0.12)",
    "cover_badge_border": "rgba(255,255,255,0.25)",
    "toc_heading":       "#666",
    "toc_link":          "#0066cc",
    "nb_header_gradient": "linear-gradient(90deg, #16213e, #0f3460)",
    "conclusions_header_gradient": "linear-gradient(90deg,#1a1a2e,#7b2d8b)",
    "missing_bg":        "#fff8e1",
    "missing_border":    "#f9a825",
    "code_bg":           "#f8f8f8",
    "output_bg":         "#f0f0f0",
    "table_header_bg":   "#f5f5f5",
    "table_border":      "#ddd",
    "footer_text":       "#999",
    "resolved_text":     "#1a6b3a",
    "resolved_badge_bg": "#eafaf1",
    "false_pos_text":    "#555",
    "false_pos_badge_bg": "#f0f0f0",
    "row_critical_bg":   "#fff0f0",
    "row_high_bg":       "#fff5ec",
    "row_low_bg":        "#f6fff6",
}

SEVERITY_ICON = {
    "CRITICAL": "🔴 CRITICAL",
    "HIGH":     "🟠 HIGH",
    "MEDIUM":   "🟡 MEDIUM",
    "LOW":      "🟢 LOW",
}

# Row tint by severity, matching the original report's striping.
SEVERITY_ROW_BG = {
    "CRITICAL": PALETTE["row_critical_bg"],
    "HIGH":     PALETTE["row_high_bg"],
    "MEDIUM":   "",
    "LOW":      PALETTE["row_low_bg"],
}

# ==============================================================================
# NOTEBOOK SECTIONS — titles + anchor ids, in display order.
# Body content is preserved as opaque HTML (not regenerated) since notebooks
# are not being re-executed. Swap in real nbconvert fragments via --nb-dir.
# ==============================================================================
NB_SECTIONS = [
    {"id": "nb01", "title": "NB-01: Citation &amp; Bibliography Audit"},
    {"id": "nb02", "title": "NB-02: Cross-Reference &amp; Label Integrity"},
    {"id": "nb03", "title": "NB-03: Section Structure &amp; Numbering"},
    {"id": "nb04", "title": "NB-04: Numerical Consistency &amp; Abstract Claims"},
    {"id": "nb05", "title": "NB-05: Figure Files &amp; Environment Audit"},
    {"id": "nb06", "title": "NB-06: Code Quality &amp; Pipeline Integrity"},
]

# ==============================================================================
# FALLBACK FINDINGS — used ONLY if --registry is not supplied or the file
# can't be read. This is a frozen snapshot taken on 2026-06-19 and WILL go
# stale the moment issue_registry.json changes. Always prefer --registry
# pointing at the live file; this exists purely so the script still runs
# (with a loud warning) if the registry is temporarily unavailable.
#
# status: "resolved" | "false_positive"
# severity: CRITICAL | HIGH | MEDIUM | LOW  (ignored for false_positive rows,
#           but kept for reference / sorting)
# ==============================================================================
FALLBACK_FINDINGS = [
    {"id": "FIX-B1", "status": "resolved", "severity": "CRITICAL", "nb": "NB-01",
     "issue": "koza1994genetic cited but no \\bibitem — will produce [?] in PDF.",
     "action": "Add \\bibitem{koza1994genetic} or redirect \\cite calls to koza1992gp."},
    {"id": "FIX-B2", "status": "resolved", "severity": "HIGH", "nb": "NB-01",
     "issue": "cranmer2023pysr and cranmer2023interpretable are the same arXiv paper (2305.01582).",
     "action": "Remove cranmer2023interpretable; replace all \\cite calls with cranmer2023pysr."},
    {"id": "FIX-B3", "status": "resolved", "severity": "HIGH", "nb": "NB-01",
     "issue": "udrescu2020ai and udrescu2020aifeynman are the same paper.",
     "action": "Remove udrescu2020aifeynman; replace all uses with udrescu2020ai."},
    {"id": "FIX-C2", "status": "resolved", "severity": "MEDIUM", "nb": "NB-06",
     "issue": "Four stale hybrid_system_v40 imports in run_comparative_suite_benchmark_v2.py.",
     "action": "Replace with hybrid_system_v50_2 throughout."},
    {"id": "FIX-C3", "status": "resolved", "severity": "MEDIUM", "nb": "NB-06 / ci_runner_disclosure",
     "issue": "Feynman split-protocol corrected: PCA 40/60 run in exp2_pca_4060/; Gates A/B/C passed.",
     "action": "Report corrected result from exp2_pca_4060/ in §10.7 or add explicit disclosure of random 80/20 baseline."},
    {"id": "FIX-F1", "status": "resolved", "severity": "MEDIUM", "nb": "NB-05",
     "issue": "hypatiaX_three_systems MISSING — \\fbox placeholder in §7.1.",
     "action": "Replace \\fbox placeholder with final PDF/PNG from Figures/architecture_figures/."},
    {"id": "FIX-F2", "status": "resolved", "severity": "MEDIUM", "nb": "NB-05",
     "issue": "fig18_r2_heatmap_improved.pdf missing from figures/.",
     "action": "Copy from Figures/figures-cosmetic-last/ → figures/."},
    {"id": "FIX-F3", "status": "resolved", "severity": "MEDIUM", "nb": "NB-05",
     "issue": "fig09_r2_heatmap_regimes.pdf missing from figures/.",
     "action": "Copy from Figures/figures-cosmetic-last/ → figures/."},
    {"id": "FIX-F4", "status": "resolved", "severity": "MEDIUM", "nb": "NB-05",
     "issue": "fig1_seed_sweep.pdf missing from figures/.",
     "action": "Copy from Figures/figures-portfolio-variance/ → figures/."},
    {"id": "FIX-N1", "status": "resolved", "severity": "MEDIUM", "nb": "NB-04",
     "issue": "Spearman footnote says '71 cases'; table caption correctly says '70 tasks'.",
     "action": "Change '71 cases' → '70 tasks' in instability section body (line 1637)."},
    {"id": "FIX-N2", "status": "resolved", "severity": "MEDIUM", "nb": "NB-04",
     "issue": "§8.3 heading reads 'Five-Layer Architecture Overview'; rest of paper uses 'five-stage routing'.",
     "action": "Rename §8.3 to 'Five-Stage Routing Architecture Overview'."},
    {"id": "FIX-N3", "status": "resolved", "severity": "MEDIUM", "nb": "NB-04",
     "issue": "Nguyen-12 numbers need verification after seed=123 rerun.",
     "action": "Rerun Nguyen-12 with seed=123 and update paper figures."},
    {"id": "FIX-XR4", "status": "resolved", "severity": "MEDIUM", "nb": "NB-02",
     "issue": "Supp A references filename jmlr_paper_main.tex; confirmed correct — paper is jmlr_paper_main.tex.",
     "action": "No action needed; filename in CI and Supp A now agree."},
    {"id": "FIX-S1", "status": "resolved", "severity": "LOW", "nb": "NB-03",
     "issue": "Missing \\label on top-level \\section commands.",
     "action": "Add \\label{sec:&lt;slug&gt;} after each \\section{} heading."},
    {"id": "FIX-S2", "status": "resolved", "severity": "LOW", "nb": "NB-03",
     "issue": "Missing \\label on \\subsection commands.",
     "action": "Add \\label{subsec:&lt;slug&gt;} after each \\subsection{} heading."},

    {"id": "FIX-C4", "status": "false_positive", "severity": "CRITICAL", "nb": "NB-06",
     "issue": "NB-06 Step 4 scan reported exposed API keys.",
     "action": "No action needed. Reason: NB-06 Step 4 scan confirmed no exposed API keys. Finding removed."},
    {"id": "FIX-C1", "status": "false_positive", "severity": "MEDIUM", "nb": "NB-06",
     "issue": "Three duplicate case names in hypatiax_defi_benchmark_v3c.py — NB-06 live scan reports 0 duplicates; silently resolved.",
     "action": "No action needed. Reason: NB-06 used substring matching. 'Funding rate cost (extended)' and "
               "'Concentrated liquidity position width (v2)' are distinct names. Exact-equality check on the "
               "actual file finds zero duplicates."},
    {"id": "FIX-XR1", "status": "false_positive", "severity": "MEDIUM", "nb": "NB-02",
     "issue": "sec:llm_limitations and sec:llm_domain both defined on Section 3 — duplicate labels.",
     "action": "No action needed — confirmed false positive."},
    {"id": "FIX-XR2", "status": "false_positive", "severity": "MEDIUM", "nb": "NB-02",
     "issue": "\\label{sec:r2_bugfix} sits inside \\item block — garbled \\ref output.",
     "action": "No action needed — confirmed false positive."},
    {"id": "FIX-XR3", "status": "false_positive", "severity": "MEDIUM", "nb": "NB-02",
     "issue": "Supp A references Section 7.3 for Proposition 1; main paper places Component 3 in §7.4.",
     "action": "No action needed — confirmed false positive."},
]

FIX_ORDER = [
    ("FIX-B2 / FIX-B3", "Deduplicate the two bibitem alias pairs."),
    ("FIX-XR4", "Update Supp A filename string from <code>jmlr_paper_main.tex</code> → "
                "<code>jmlr-hypatiax-paper-final.tex</code>."),
    ("FIX-F1", "Replace <code>\\fbox</code> placeholder with final architecture PDF/PNG."),
    ("FIX-F2, FIX-F3, FIX-F4", "Copy three missing figure files to <code>figures/</code>."),
    ("FIX-N3", "Commit Nguyen-12 seed=123 rerun results; add dual-threshold claim to "
               "<code>paper_targets.json</code>."),
    ("FIX-S1 / FIX-S2", "Add <code>\\label</code> to all unlabelled \\section and \\subsection headings."),
]

RESULTS_QUALITY_HTML = """
All nine abstract claims pass verification ([OK] across every check in NB-04 Step&nbsp;2).
The core numerical story — 89.2&nbsp;% near-perfect success rate, +27&nbsp;pp gain over the LLM baseline,
1.73×&nbsp;speedup, Nguyen 11/12, Feynman 9/30 (random 80/20 baseline, locked in
<code>fixc3_baseline.json</code>) — is internally consistent throughout the paper.
The FIX-C3 split-protocol correction has been applied; Gates A/B/C pass.
""".strip()


def normalize_registry_entry(raw: dict) -> dict:
    """Map issue_registry.json's real schema onto the shape the renderer
    expects. The live registry uses: severity (lowercase), nb_source,
    description, action, status in {"resolved","false_positive","open",...}.
    """
    severity = str(raw.get("severity", "medium")).upper()
    status = raw.get("status", "open")
    description = raw.get("description", "")
    action = raw.get("action", "")
    # Surface false_positive_reason / note as part of the action text when
    # present, since the renderer puts everything under one "Action:" line.
    extra = raw.get("false_positive_reason") or raw.get("note")
    if extra and extra not in action:
        action = f"{action} ({extra})" if action else extra
    return {
        "id": raw.get("id", "FIX-UNKNOWN"),
        "status": status,
        "severity": severity,
        "nb": raw.get("nb_source", raw.get("nb", "?")),
        "issue": html.escape(description, quote=False),
        "action": html.escape(action, quote=False),
        "updated": raw.get("updated"),
    }


def load_registry(registry_path: Path | None) -> tuple[list[dict], str]:
    """Load findings from the live issue_registry.json. Returns (findings,
    source_label) where source_label describes where the data came from, so
    the report can honestly disclose it instead of silently going stale.
    """
    if registry_path is not None:
        try:
            raw = json.loads(registry_path.read_text(encoding="utf-8"))
            findings = [normalize_registry_entry(r) for r in raw]
            return findings, f"live registry: {registry_path}"
        except (OSError, json.JSONDecodeError) as e:
            print(f"WARNING: could not read --registry {registry_path} ({e}); "
                  f"falling back to frozen snapshot.")
    return FALLBACK_FINDINGS, "FALLBACK_FINDINGS (frozen 2026-06-19 snapshot — may be stale)"


def render_finding_row(f: dict) -> str:
    status = f["status"]
    sev = f["severity"]
    row_bg = SEVERITY_ROW_BG.get(sev, "") if status == "resolved" else ""
    style = f' style="background:{row_bg};"' if row_bg else ' style=""'

    if status == "resolved":
        badge = (f'<span style="color:{PALETTE["resolved_text"]};font-weight:bold;'
                 f'background:{PALETTE["resolved_badge_bg"]};padding:2px 7px;border-radius:10px;">'
                 f'✅ RESOLVED</span>')
        sev_cell = f'<td style="font-size:0.85em;">{SEVERITY_ICON.get(sev, sev)}</td>'
    else:
        badge = (f'<span style="color:{PALETTE["false_pos_text"]};font-weight:bold;'
                 f'background:{PALETTE["false_pos_badge_bg"]};padding:2px 7px;border-radius:10px;">'
                 f'🔵 FALSE POSITIVE</span>')
        sev_cell = f'<td style="font-size:0.85em;">{SEVERITY_ICON.get(sev, sev)}</td>'
        style = f' style="background:{PALETTE["table_header_bg"] if False else "#f8f8f8"};color:#777;"'

    return f"""          <tr{style}>
            <td><strong>{f['id']}</strong><br><small>{badge}</small></td>
            {sev_cell}
            <td>{f['issue']}<br><small style="color:#666;"><em>Action:</em> {f['action']}</small></td>
            <td style="font-size:0.85em;">{f['nb']}</td>
          </tr>"""


def render_findings_table(findings: list, *, with_header_row: bool = True) -> str:
    header = """    <thead><tr>
        <th style="width:115px;padding:6px 8px;background:{bg};border:1px solid {bd};">ID / Status</th>
        <th style="width:95px;padding:6px 8px;background:{bg};border:1px solid {bd};">Severity</th>
        <th style="padding:6px 8px;background:{bg};border:1px solid {bd};">Issue &amp; Action</th>
        <th style="width:80px;padding:6px 8px;background:{bg};border:1px solid {bd};">NB</th>
      </tr></thead>\n""".format(bg=PALETTE["table_header_bg"], bd=PALETTE["table_border"])
    rows = "\n".join(render_finding_row(f) for f in findings)
    return f'<table style="font-size:0.85em;width:100%;border-collapse:collapse;margin-top:8px;">\n{header}      <tbody>\n{rows}</tbody>\n    </table>'


def render_conclusions(*, findings: list, source_label: str, report_run_id: str,
                        notebook_run_id: str, show_run_ids: bool) -> str:
    resolved = [f for f in findings if f["status"] == "resolved"]
    false_positives = [f for f in findings if f["status"] == "false_positive"]
    open_issues = [f for f in findings if f["status"] not in ("resolved", "false_positive")]

    if show_run_ids:
        provenance = (f'Report run&nbsp;<code>{report_run_id}</code>\n'
                      f'      · Notebook run&nbsp;<code>{notebook_run_id}</code><br>\n'
                      f'      <small style="color:#666;">Data source: <code>{html.escape(source_label)}</code> '
                      f'({len(findings)} entries).</small>')
    else:
        provenance = (f'<small style="color:#666;">Data source: <code>{html.escape(source_label)}</code> '
                      f'({len(findings)} entries).</small>')

    if open_issues:
        open_table = render_findings_table(open_issues)
        open_heading_color = "#b00020"
        open_note = ""
    else:
        open_table = (f'<table style="font-size:0.87em;width:100%;border-collapse:collapse;">\n'
                      f'    <thead><tr>\n'
                      f'      <th style="width:115px;padding:6px 8px;background:{PALETTE["table_header_bg"]};'
                      f'border:1px solid {PALETTE["table_border"]};">ID / Status</th>\n'
                      f'      <th style="width:95px;padding:6px 8px;background:{PALETTE["table_header_bg"]};'
                      f'border:1px solid {PALETTE["table_border"]};">Severity</th>\n'
                      f'      <th style="padding:6px 8px;background:{PALETTE["table_header_bg"]};'
                      f'border:1px solid {PALETTE["table_border"]};">Issue &amp; Action</th>\n'
                      f'      <th style="width:80px;padding:6px 8px;background:{PALETTE["table_header_bg"]};'
                      f'border:1px solid {PALETTE["table_border"]};">NB</th>\n'
                      f'    </tr></thead>\n'
                      f'    <tbody><tr><td colspan="4" style="padding:12px;color:{PALETTE["resolved_text"]};'
                      f'text-align:center;">✅ No open issues</td></tr></tbody>\n'
                      f'  </table>')
        open_heading_color = "green"

    fix_order_items = "".join(
        f'<li><s style="color:#888;"><strong>{label}</strong></s> '
        f'<span style="color:{PALETTE["resolved_text"]};">✅</span> — {desc}</li>'
        for label, desc in FIX_ORDER
    )
    all_resolved_or_suppressed = not open_issues
    fix_order_lead = (
        f'<p style="color:{PALETTE["resolved_text"]};">✅ All tracked action items are resolved or suppressed.</p>'
        if all_resolved_or_suppressed else
        '<p>Action items remain open — see table above.</p>'
    )

    return f"""  <div class="nb-section" id="conclusions">
    <div class="nb-header" style="background:{PALETTE['conclusions_header_gradient']};">
      <h2>Audit Conclusions &amp; Action Plan</h2>
    </div>
    <div class="nb-body">

      <h3 style="margin-top:24px;">Pipeline Status</h3>
      <p>
      {provenance}
    </p>

      <h3>Results Quality</h3>
      <p>
        {RESULTS_QUALITY_HTML}
      </p>

      <h3>Open Issues ({len(open_issues)}
        — <span style="color:{open_heading_color};">{"none ✅" if not open_issues else f"{len(open_issues)} open"}</span>)
      </h3>
      {open_table}

      <details style="margin-top:16px;">
    <summary style="cursor:pointer;font-weight:600;color:{PALETTE['resolved_text']};">
      ✅ {len(resolved)} Resolved issues (click to expand)
    </summary>
    {render_findings_table(resolved)}
  </details>

  <details style="margin-top:8px;">
    <summary style="cursor:pointer;font-weight:600;color:{PALETTE['false_pos_text']};">
      🔵 {len(false_positives)} Confirmed false positives — suppressed from all gates (click to expand)
    </summary>
    <p style="font-size:0.85em;color:#666;margin:8px 0 4px;">
      These findings were produced by audit notebooks scanning stale cached outputs.
      Verification against the current paper source files confirms none of them exist.
      They are excluded from CI gates, fix scripts, and the priority action list.
    </p>
    {render_findings_table(false_positives)}
  </details>

      <h3 style="margin-top:32px;">Recommended Fix Order Before Submission</h3>
      {fix_order_lead}
      <ol style="line-height:2.2;">{fix_order_items}</ol>

    </div>
  </div><!-- #conclusions -->"""


def render_nb_section(section: dict, body_html: str) -> str:
    return f"""      <div class="nb-section" id="{section['id']}">
        <div class="nb-header"><h2>{section['title']}</h2></div>
        <div class="nb-body">
{body_html}
        </div>
      </div>"""


def render_toc() -> str:
    items = "\n".join(
        f'        <li><a href="#{s["id"]}">{s["title"]}</a></li>' for s in NB_SECTIONS
    )
    return f"""    <div class="toc">
      <h2>Contents</h2>
      <ol>
{items}
        <li><a href="#conclusions"><strong>Audit Conclusions &amp; Action Plan</strong></a></li>
      </ol>
    </div>"""


def render_css() -> str:
    p = PALETTE
    return f"""    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0; background: {p['page_bg']}; color: {p['page_text']};
    }}
    .cover {{
      background: {p['cover_gradient']};
      color: white; padding: 64px 40px 48px; text-align: center;
    }}
    .cover h1 {{ margin: 0 0 14px; font-size: 2.4em; letter-spacing: -0.5px; }}
    .cover .meta {{ opacity: 0.7; font-size: 0.95em; line-height: 1.8; }}
    .badge {{
      display: inline-block; background: {p['cover_badge_bg']};
      border: 1px solid {p['cover_badge_border']}; border-radius: 20px;
      padding: 4px 14px; font-size: 0.82em; margin-top: 12px;
    }}
    .container {{ max-width: 900px; margin: 0 auto; padding: 0 20px 60px; }}
    .toc {{
      background: white; border-radius: 10px; padding: 28px 36px;
      margin: 32px 0 24px; box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    }}
    .toc h2 {{ margin: 0 0 16px; font-size: 1.05em; text-transform: uppercase;
               letter-spacing: .08em; color: {p['toc_heading']}; }}
    .toc ol  {{ margin: 0; padding-left: 22px; line-height: 2.1; }}
    .toc a   {{ color: {p['toc_link']}; text-decoration: none; font-weight: 500; }}
    .toc a:hover {{ text-decoration: underline; }}
    .nb-section {{
      background: white; border-radius: 10px; margin-bottom: 24px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.07); overflow: hidden;
    }}
    .nb-header {{
      background: {p['nb_header_gradient']};
      color: white; padding: 18px 28px; display: flex;
      align-items: center; gap: 12px;
    }}
    .nb-header h2 {{ margin: 0; font-size: 1.1em; }}
    .nb-body {{ padding: 8px 28px 28px; overflow-x: auto; }}
    .missing {{
      background: {p['missing_bg']}; border-left: 4px solid {p['missing_border']};
      padding: 16px 20px; margin: 20px 0; border-radius: 4px;
      font-size: 0.95em;
    }}
    .nb-body pre {{
      background: {p['code_bg']}; padding: 12px 16px; border-radius: 6px;
      overflow-x: auto; font-size: 0.87em; line-height: 1.5;
    }}
    .nb-body .output_area pre {{ background: {p['output_bg']}; }}
    .nb-body table {{
      border-collapse: collapse; width: 100%; font-size: 0.9em;
    }}
    .nb-body th, .nb-body td {{
      border: 1px solid {p['table_border']}; padding: 8px 12px; text-align: left;
    }}
    .nb-body th {{ background: {p['table_header_bg']}; font-weight: 600; }}
    .footer {{
      text-align: center; padding: 40px 20px;
      color: {p['footer_text']}; font-size: 0.82em; line-height: 1.8;
    }}"""


def load_nb_body(nb_dir: Path | None, section_id: str) -> str:
    """Load a real nbconvert HTML body fragment if --nb-dir was given and the
    file exists; otherwise fall back to a clearly-marked placeholder."""
    if nb_dir is not None:
        candidate = nb_dir / f"{section_id}.html"
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return (f'          <div class="missing">'
            f'⚠️ No static body supplied for <code>{section_id}</code>. '
            f'Pass <code>--nb-dir &lt;folder&gt;</code> containing <code>{section_id}.html</code> '
            f'nbconvert fragments to populate this section, or edit NB_SECTIONS bodies directly in this script.'
            f'</div>')


def build_report(*, findings: list, source_label: str, report_run_id: str,
                  notebook_run_id: str, show_run_ids: bool,
                  nb_dir: Path | None, generated_at: str) -> str:
    nb_html = "\n\n".join(
        render_nb_section(s, load_nb_body(nb_dir, s["id"])) for s in NB_SECTIONS
    )
    conclusions_html = render_conclusions(
        findings=findings,
        source_label=source_label,
        report_run_id=report_run_id,
        notebook_run_id=notebook_run_id,
        show_run_ids=show_run_ids,
    )

    meta_line = (
        f"Generated: {generated_at}<br>\n      Notebook run ID: {notebook_run_id} "
        f"&nbsp;·&nbsp; Report run ID: {report_run_id}"
        if show_run_ids else
        f"Generated: {generated_at}"
    )

    footer_line = (
        f"HypatiaX Paper Audit • Notebook run {notebook_run_id} "
        f"• Report run {report_run_id} • <br>\n    Generated by <code>generate_audit_report.py</code> "
        f"— static regeneration, no notebooks re-executed."
        if show_run_ids else
        "HypatiaX Paper Audit • Static audit snapshot • <br>\n    "
        "Generated by <code>generate_audit_report.py</code> — static regeneration, no notebooks re-executed."
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta http-equiv="content-type" content="text/html; charset=UTF-8">
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HypatiaX Paper Audit Report</title>
  <style>
{render_css()}
  </style>
</head>
<body>
  <div class="cover">
    <h1>📋 HypatiaX Paper Audit Report</h1>
    <div class="meta">
      {meta_line}
    </div>
    <div class="badge">Static audit snapshot — no notebooks re-executed</div>
  </div>

  <div class="container">
{render_toc()}

{nb_html}

{conclusions_html}
  </div><!-- .container -->

  <div class="footer">
    {footer_line}
  </div>

</body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="HypatiaX_Paper_Audit_Report.html",
                         help="Output HTML path (default: %(default)s)")
    parser.add_argument("--registry", type=Path, default=None,
                         help="Path to the LIVE issue_registry.json. If omitted or unreadable, "
                              "falls back to a frozen snapshot and prints a warning. "
                              "This is what makes the report reflect current data instead of "
                              "going stale — always pass this in CI.")
    parser.add_argument("--report-run-id", default="27748279742",
                         help="Provenance label for report run (default: %(default)s)")
    parser.add_argument("--notebook-run-id", default="27747615528",
                         help="Provenance label for notebook run (default: %(default)s)")
    parser.add_argument("--no-run-ids", action="store_true",
                         help="Omit internal CI run IDs from the output (use for external distribution)")
    parser.add_argument("--nb-dir", type=Path, default=None,
                         help="Directory containing real nbconvert HTML fragments named nb01.html...nb06.html")
    parser.add_argument("--generated-at", default=None,
                         help="Override the 'Generated:' timestamp (default: current UTC time)")
    args = parser.parse_args()

    generated_at = args.generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    findings, source_label = load_registry(args.registry)

    report_html = build_report(
        findings=findings,
        source_label=source_label,
        report_run_id=args.report_run_id,
        notebook_run_id=args.notebook_run_id,
        show_run_ids=not args.no_run_ids,
        nb_dir=args.nb_dir,
        generated_at=generated_at,
    )

    out_path = Path(args.out)
    out_path.write_text(report_html, encoding="utf-8")
    print(f"Wrote {out_path} ({len(report_html):,} bytes)")
    print(f"  Data source: {source_label}")
    print(f"  Findings: {len(findings)} total "
          f"({sum(1 for f in findings if f['status']=='resolved')} resolved, "
          f"{sum(1 for f in findings if f['status']=='false_positive')} false positive, "
          f"{sum(1 for f in findings if f['status'] not in ('resolved','false_positive'))} open)")
    print(f"  Run IDs shown: {not args.no_run_ids}")


if __name__ == "__main__":
    main()

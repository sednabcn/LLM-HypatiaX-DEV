# ── ci_paper_notebooks.yml ────────────────────────────────────────────────────
# Changes:
#  1. Each NB job: add "Extract & upload registry patch" step after execute
#  2. bundle-artifacts: remove all hardcoded statuses; read only live outputs
#  3. bundle-artifacts: add "Merge NB registry patches → commit" step
#  4. bundle-artifacts: upload merged registry as artifact

with open('.github/workflows/ci_paper_notebooks.yml') as f:
    nb_yml = f.read()

# ── 1. Add per-NB registry-patch step to each NB job ─────────────────────────
# We'll insert after each "Upload executed notebook" step for NB-01 through NB-06

NB_REGISTRY_STEP = '''
      - name: Extract & upload registry patch
        if: always()
        env:
          NB_ID: "{nb_id}"
          RUN_ID: ${{{{ github.run_id }}}}
          REGISTRY_PATH: ${{{{ env.REGISTRY_PATH }}}}
        run: |
          python3 - <<'PYREG'
          import json, os, re, datetime
          from pathlib import Path

          nb_id   = os.environ["NB_ID"]
          run_id  = os.environ["RUN_ID"]
          reg_path = Path(os.environ["REGISTRY_PATH"])

          # Find the executed notebook
          nb_path = next(Path("notebooks").rglob(f"{{nb_id}}*.ipynb"), None)
          if not nb_path:
              print(f"::warning::Notebook not found for {{nb_id}} — skipping registry patch")
              import sys; sys.exit(0)

          try:
              nb = json.loads(nb_path.read_text(encoding="utf-8"))
          except Exception as exc:
              print(f"::warning::Could not read notebook: {{exc}}")
              import sys; sys.exit(0)

          patch_entries = []

          # ── Source 1: cells tagged audit_findings ─────────────────────────
          for cell in nb.get("cells", []):
              if "audit_findings" not in cell.get("metadata", {{}}).get("tags", []):
                  continue
              for out in cell.get("outputs", []):
                  raw = out.get("text") or out.get("data", {{}}).get("text/plain", "")
                  text = "".join(raw) if isinstance(raw, list) else str(raw)
                  text = re.sub(r"^```json\\s*", "", text.strip(), flags=re.MULTILINE)
                  text = re.sub(r"```\\s*$",      "", text.strip(), flags=re.MULTILINE)
                  try:
                      data = json.loads(text)
                      for f in (data if isinstance(data, list) else data.get("findings", [])):
                          fid = f.get("id") or f.get("fix_id", "")
                          if fid:
                              patch_entries.append({{
                                  "id":     fid,
                                  "status": f.get("status", "open").lower(),
                                  "source": f"{{nb_id}}/audit_findings",
                              }})
                  except Exception:
                      pass

          # ── Source 2: Step 8 write-back output (NB-05 style) ─────────────
          for cell in nb.get("cells", []):
              if cell.get("id", "") != "step8_writeback":
                  continue
              for out in cell.get("outputs", []):
                  raw = out.get("text", "")
                  text = "".join(raw) if isinstance(raw, list) else str(raw)
                  for line in text.splitlines():
                      # Lines like:  [UPDATED]  FIX-F2: open → resolved
                      m = re.match(r"\\s*\\[(?:UPDATED|ADDED)\\]\\s+(FIX-\\w+):\\s*(\\w+)\\s*→\\s*(\\w+)", line)
                      if m:
                          patch_entries.append({{
                              "id":     m.group(1),
                              "status": m.group(3).lower(),
                              "source": f"{{nb_id}}/step8",
                          }})
                      # Lines like:  [NO-CHANGE] FIX-F2 — already resolved
                      m2 = re.match(r"\\s*\\[NO-CHANGE\\]\\s+(FIX-\\w+)\\s*—\\s*already\\s+(\\w+)", line)
                      if m2:
                          patch_entries.append({{
                              "id":     m2.group(1),
                              "status": m2.group(2).lower(),
                              "source": f"{{nb_id}}/step8",
                          }})

          if not patch_entries:
              print(f"No registry patches found in {{nb_id}} outputs — skipping patch artifact")
              import sys; sys.exit(0)

          # Deduplicate: last entry per id wins (later cells override earlier)
          seen = {{}}
          for e in patch_entries:
              seen[e["id"]] = e
          patch_entries = list(seen.values())

          out_path = Path(f"logs/registry_patch_{{nb_id}}.json")
          out_path.parent.mkdir(parents=True, exist_ok=True)
          out_path.write_text(json.dumps({{
              "nb":           nb_id,
              "run_id":       run_id,
              "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
              "patches":      patch_entries,
          }}, indent=2))
          print(f"Registry patch written: {{out_path}} ({{len(patch_entries)}} entry/entries)")
          for e in patch_entries:
              print(f"  {{e['id']}} → {{e['status']}}  [{{e['source']}}]")
          PYREG

      - name: Upload registry patch artifact
        if: always()
        uses: actions/upload-artifact@v4.3.6
        with:
          name: registry-patch-{nb_id}-${{{{ github.run_id }}}}
          path: logs/registry_patch_{nb_id}.json
          if-no-files-found: ignore
          retention-days: 7
'''

NB_JOBS = [
    ("NB-01_Citation_Bibliography_Audit",     "NB-01"),
    ("NB-02_CrossReference_Label_Audit",       "NB-02"),
    ("NB-03_Section_Structure_Numbering",      "NB-03"),
    ("NB-04_Numerical_Consistency_Checker",    "NB-04"),
    ("NB-05_Figure_Image_Dependency_Checker",  "NB-05"),
    ("NB-06_Code_Quality_Pipeline_Integrity",  "NB-06"),
]

for artifact_name, nb_id in NB_JOBS:
    # Find the upload step for this NB and insert our new steps after it
    upload_marker = f"          name: {artifact_name}\n          path: notebooks/{artifact_name}.ipynb"
    patch_step = NB_REGISTRY_STEP.format(nb_id=nb_id)
    if upload_marker in nb_yml:
        # Insert after the retention-days line that follows
        insert_after = upload_marker + "\n          retention-days: 7"
        nb_yml = nb_yml.replace(insert_after, insert_after + patch_step)
        print(f"  Injected registry-patch steps for {nb_id}")
    else:
        print(f"  WARNING: marker not found for {nb_id}")

with open('/home/claude/ci_paper_notebooks.yml', 'w') as f:
    f.write(nb_yml)
print("Phase 1 done (per-NB steps injected)")

bash

python3 << 'PYEOF'
# ── bundle-artifacts: replace the "Write notebooks audit summary JSON" step
# with a live version that has NO hardcoded statuses, then add the
# "Merge NB registry patches → commit" step before the upload.

with open('.github/workflows/ci_paper_notebooks.yml') as f:
    nb_yml = f.read()

# ── Step A: Replace hardcoded Write notebooks audit summary JSON ─────────────
OLD_SUMMARY_STEP_START = "      - name: Write notebooks audit summary JSON"
OLD_SUMMARY_STEP_END   = "          PYEOF\n\n      - name: Upload notebooks audit summary"

LIVE_SUMMARY_STEP = '''      - name: Write notebooks audit summary JSON (live — no hardcoded statuses)
        if: always()
        env:
          RUN_ID: ${{ github.run_id }}
        run: |
          pip install --quiet jupyter nbconvert ipykernel 2>/dev/null || true
          python3 - <<'PYEOF'
          import json, os, re, datetime
          from pathlib import Path

          run_id = os.environ["RUN_ID"]

          # ── Load hypatia_inspector findings (status overrides only) ──────────
          inspector_status: dict[str, str] = {}
          for candidate in ["/tmp/hypatia_findings/findings.json"]:
              try:
                  raw = json.loads(Path(candidate).read_text())
                  for f in (raw if isinstance(raw, list) else raw.get("findings", [])):
                      fid = f.get("fix_id") or f.get("id", "")
                      st  = f.get("status", "")
                      if fid and st:
                          inspector_status[fid] = (
                              "RESOLVED" if st == "fixed" else
                              "OPEN"     if st in ("detected", "manual") else
                              st.upper()
                          )
                  print(f"Loaded {len(inspector_status)} inspector statuses")
                  break
              except Exception as exc:
                  print(f"Inspector findings unavailable: {exc}")

          # ── Load live notebook outputs (ONLY source for findings) ────────────
          nb_findings: dict[str, dict] = {}
          for nb_path in sorted(Path("notebooks").rglob("*.ipynb")):
              try:
                  nb = json.loads(nb_path.read_text(encoding="utf-8"))
              except Exception:
                  continue
              for cell in nb.get("cells", []):
                  tags    = cell.get("metadata", {}).get("tags", [])
                  outputs = cell.get("outputs", [])
                  is_findings = "audit_findings" in tags
                  for out in outputs:
                      raw = out.get("text") or out.get("data", {}).get("text/plain", "")
                      text = "".join(raw) if isinstance(raw, list) else str(raw)
                      text = re.sub(r"^```json\s*", "", text.strip(), flags=re.MULTILINE)
                      text = re.sub(r"```\s*$",      "", text.strip(), flags=re.MULTILINE)
                      if is_findings or (text.startswith("{") and '"findings"' in text):
                          try:
                              data = json.loads(text)
                              for f in (data if isinstance(data, list) else data.get("findings", [])):
                                  fid = f.get("id") or f.get("fix_id", "")
                                  if not fid:
                                      continue
                                  # Inspector can promote open→resolved, never resolved→open
                                  status = f.get("status", "OPEN").upper()
                                  if status != "RESOLVED" and fid in inspector_status:
                                      status = inspector_status[fid]
                                  nb_findings[fid] = {
                                      "id":          fid,
                                      "severity":    f.get("severity", "MEDIUM").upper(),
                                      "status":      status,
                                      "description": f.get("description", ""),
                                      "action":      f.get("action", ""),
                                      "nb":          f.get("nb", nb_path.stem),
                                  }
                          except Exception:
                              pass

          # ── Also pick up step8 write-back outputs from NB-05 ────────────────
          for nb_path in sorted(Path("notebooks").rglob("NB-05*.ipynb")):
              try:
                  nb = json.loads(nb_path.read_text(encoding="utf-8"))
              except Exception:
                  continue
              for cell in nb.get("cells", []):
                  if cell.get("id", "") != "step8_writeback":
                      continue
                  for out in cell.get("outputs", []):
                      raw = out.get("text", "")
                      text = "".join(raw) if isinstance(raw, list) else str(raw)
                      for line in text.splitlines():
                          m = re.match(r"\s*\[(?:UPDATED|ADDED)\]\s+(FIX-\w+):\s*(\w+)\s*→\s*(\w+)", line)
                          if m:
                              fid, new_st = m.group(1), m.group(3).upper()
                              if fid in nb_findings:
                                  nb_findings[fid]["status"] = new_st
                              # else NB-05 didn't emit an audit_findings cell for this; add it
                              else:
                                  nb_findings[fid] = {
                                      "id": fid, "severity": "MEDIUM",
                                      "status": new_st, "description": "",
                                      "action": "", "nb": "NB-05",
                                  }

          findings = list(nb_findings.values())
          print(f"Live findings: {len(findings)} from notebooks "
                f"({len(inspector_status)} inspector overrides applied)")

          summary = {
              "source":       "ci_paper_notebooks",
              "run_id":       run_id,
              "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "findings":     findings,
          }
          out = Path("notebooks/notebooks_audit_summary.json")
          out.parent.mkdir(parents=True, exist_ok=True)
          out.write_text(json.dumps(summary, indent=2))
          print(f"Wrote {out} — {len(findings)} live finding(s)")
          PYEOF

'''

# Locate and replace the old step
start_idx = nb_yml.find(OLD_SUMMARY_STEP_START)
end_idx   = nb_yml.find(OLD_SUMMARY_STEP_END)
if start_idx != -1 and end_idx != -1:
    end_idx += len(OLD_SUMMARY_STEP_END)
    nb_yml = nb_yml[:start_idx] + LIVE_SUMMARY_STEP + "      - name: Upload notebooks audit summary"
    print("  Replaced hardcoded summary step with live version")
else:
    print(f"  ERROR: could not find summary step (start={start_idx}, end={end_idx})")

# ── Step B: Insert "Merge NB registry patches → commit" before upload summary ─
MERGE_REGISTRY_STEP = '''      - name: Download NB registry patch artifacts
        if: always()
        continue-on-error: true
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          REPO:     ${{ github.repository }}
          RUN_ID:   ${{ github.run_id }}
        run: |
          mkdir -p /tmp/nb_patches
          for nb_id in NB-01 NB-02 NB-03 NB-04 NB-05 NB-06; do
            gh run download "$RUN_ID" \
              --repo "$REPO" \
              --name "registry-patch-${nb_id}-${RUN_ID}" \
              --dir  "/tmp/nb_patches" 2>/dev/null || true
          done
          echo "Patch files downloaded:"
          ls -la /tmp/nb_patches/ 2>/dev/null || echo "  (none)"

      - name: Merge NB registry patches into issue_registry.json and commit
        if: always()
        env:
          REGISTRY_PATH: ${{ env.REGISTRY_PATH }}
          GIT_AUTHOR_NAME:  github-actions[bot]
          GIT_AUTHOR_EMAIL: github-actions[bot]@users.noreply.github.com
          GIT_COMMITTER_NAME:  github-actions[bot]
          GIT_COMMITTER_EMAIL: github-actions[bot]@users.noreply.github.com
        run: |
          python3 - <<'PYMERGE'
          import json, subprocess, sys, datetime
          from pathlib import Path

          registry_path = Path("${{ env.REGISTRY_PATH }}")
          patches_dir   = Path("/tmp/nb_patches")

          if not registry_path.exists():
              print(f"::warning::Registry not found at {registry_path} — skipping NB merge")
              sys.exit(0)

          registry = json.loads(registry_path.read_text(encoding="utf-8"))
          reg_index = {e["id"]: i for i, e in enumerate(registry)}
          today = datetime.date.today().isoformat()

          # Collect all patch files, sorted by NB order so NB-05/06 can override NB-01
          patch_files = sorted(patches_dir.glob("registry_patch_NB-*.json"))
          if not patch_files:
              print("No NB registry patches found — registry unchanged")
              sys.exit(0)

          diff_rows = []
          for pf in patch_files:
              try:
                  patch = json.loads(pf.read_text())
              except Exception as exc:
                  print(f"  Could not read {pf}: {exc}")
                  continue
              for entry in patch.get("patches", []):
                  eid      = entry["id"]
                  new_st   = entry["status"].lower()
                  source   = entry.get("source", pf.stem)

                  if eid in reg_index:
                      existing = registry[reg_index[eid]]
                      old_st   = existing.get("status", "open")

                      # Never touch false_positive or demote resolved→open
                      if old_st == "false_positive":
                          continue
                      if old_st == "resolved" and new_st == "open":
                          continue

                      if old_st != new_st:
                          existing["status"]  = new_st
                          existing["updated"] = today
                          diff_rows.append((eid, old_st, new_st, source))
                  else:
                      # New entry from a notebook — add it
                      registry.append({
                          "id":       eid,
                          "status":   new_st,
                          "severity": entry.get("severity", "medium"),
                          "nb_source": entry.get("source", source),
                          "updated":  today,
                      })
                      reg_index[eid] = len(registry) - 1
                      diff_rows.append((eid, "new", new_st, source))

          if not diff_rows:
              print("Registry already reflects notebook findings — no changes")
              sys.exit(0)

          col = "{:<14} {:<12} {:<12} {}"
          print("\n=== NB registry patch merge ===")
          print(col.format("ID", "OLD", "NEW", "SOURCE"))
          print("-" * 56)
          for row in diff_rows:
              print("  → " + col.format(*row))

          registry_path.write_text(
              json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
              encoding="utf-8"
          )

          subprocess.run(["git", "config", "user.name",  "github-actions[bot]"], check=True)
          subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], check=True)
          subprocess.run(["git", "pull",   "--rebase", "--autostash",
                          "origin", "${{ github.ref_name }}"], capture_output=True)
          subprocess.run(["git", "add", str(registry_path)], check=True)

          ids_changed = [f"{e}: {o}→{n}" for e, o, n, _ in diff_rows]
          msg = f"ci: nb-registry-merge {', '.join(ids_changed)} [skip ci]"
          subprocess.run(["git", "commit", "-m", msg], check=True)

          for attempt in range(1, 4):
              r = subprocess.run(["git", "push"])
              if r.returncode == 0:
                  break
              subprocess.run(["git", "pull", "--rebase", "--autostash"], capture_output=True)
              subprocess.run(["git", "checkout", "--ours", str(registry_path)], check=True)
              subprocess.run(["git", "add", str(registry_path)], check=True)
              subprocess.run(["git", "rebase", "--continue"],
                  env={**__import__("os").environ, "GIT_EDITOR": "true"}, check=True)
          else:
              raise RuntimeError("git push failed after 3 attempts")

          print(f"\n✅ Registry committed — {len(diff_rows)} update(s): {msg}")
          PYMERGE

      - name: Upload merged registry artifact
        if: always()
        uses: actions/upload-artifact@v4.3.6
        with:
          name: issue-registry-merged-${{ github.run_id }}
          path: ${{ env.REGISTRY_PATH }}
          if-no-files-found: warn
          retention-days: 30

'''

# Insert before the "Write notebooks audit summary" step
insert_before = "      - name: Write notebooks audit summary JSON (live — no hardcoded statuses)"
nb_yml = nb_yml.replace(insert_before, MERGE_REGISTRY_STEP + insert_before)
print("  Inserted merge-registry + upload steps")

with open('.githu/workflows/ci_paper_notebooks.yml', 'w') as f:
    f.write(nb_yml)
print("ci_paper_notebooks.yml updated")

bash

python3 << 'PYEOF'
# ── ci_report.yml ─────────────────────────────────────────────────────────────
# Changes:
# 1. Fix UPSTREAM_RUN scoping throughout (use step output, not shell vars)
# 2. Add "Download merged registry artifact" step before checkout
# 3. Remove STATIC_META entirely — statuses from registry only
# 4. Fix "Finalise report" step — inject vars via env:
# 5. Fix UPSTREAM_RUN in "Download audit summary" step
# 6. Add FIX-C3 gates section rendering

with open('/home/claude/ci_report.yml') as f:
    r = f.read()

# ── Fix 1: UPSTREAM_RUN shell var lost between steps ─────────────────────────
# The "Build combined audit report" step sets RUN_DATE/UPSTREAM_RUN as shell
# vars, but "Finalise report HTML" is a separate run: block so they're gone.
# Fix: add an env: block to the Finalise step.

OLD_FINALISE = "      - name: Finalise report HTML\n        run: |"
NEW_FINALISE  = (
    "      - name: Finalise report HTML\n"
    "        env:\n"
    "          RUN_DATE:     ${{ steps.set-run-meta.outputs.run_date }}\n"
    "          UPSTREAM_RUN: ${{ steps.resolve-run.outputs.notebooks_run_id }}\n"
    "        run: |"
)
r = r.replace(OLD_FINALISE, NEW_FINALISE)

# ── Fix 2: Export RUN_DATE from build step as a step output ──────────────────
# Add a dedicated "Set run metadata" step before "Set up Python"
SET_META_STEP = '''      - name: Set run metadata
        id: set-run-meta
        run: |
          echo "run_date=$(date -u '+%Y-%m-%d %H:%M UTC')" >> "$GITHUB_OUTPUT"

'''
# Insert after the first "steps:" line inside build-report
r = r.replace("    steps:\n\n      # ── Resolve the source run ID", 
              "    steps:\n\n" + SET_META_STEP + "      # ── Resolve the source run ID")

# ── Fix 3: Add download of merged registry artifact before checkout ───────────
DOWNLOAD_REGISTRY_STEP = '''
      # ── Download merged registry from THIS run's bundle-artifacts ───────────
      # This is the most current registry: reflects live NB findings just written.
      # Falls back to the checked-out version (next step) if the artifact is absent.
      - name: Download merged registry artifact (from ci_paper_notebooks)
        continue-on-error: true
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          REPO:     ${{ github.repository }}
          NB_RUN:   ${{ steps.resolve-run.outputs.notebooks_run_id }}
        run: |
          mkdir -p /tmp/live_registry
          if [[ -n "$NB_RUN" ]]; then
            gh run download "$NB_RUN" \
              --repo "$REPO" \
              --name "issue-registry-merged-${NB_RUN}" \
              --dir  "/tmp/live_registry" 2>/dev/null || true
          fi
          if [[ -f "/tmp/live_registry/issue_registry.json" ]]; then
            echo "✅ Live merged registry downloaded ($(wc -c < /tmp/live_registry/issue_registry.json) bytes)"
          else
            echo "::notice::Merged registry artifact not found — will use checked-out version"
          fi

'''

# Insert before the checkout step
r = r.replace(
    "      - name: Checkout (for live issue_registry.json)",
    DOWNLOAD_REGISTRY_STEP + "      - name: Checkout (for live issue_registry.json)"
)

# ── Fix 4: update REGISTRY_PATH resolution in the conclusions Python block ────
# Make it prefer /tmp/live_registry/issue_registry.json over the checkout copy
OLD_REGISTRY_LOAD = '''          # ── 1. Load issue_registry.json (primary source of truth) ────────────
          registry_entries = []
          if registry_path.exists():
              try:
                  registry_entries = json.loads(registry_path.read_text())
                  print(f"Loaded {len(registry_entries)} entries from {registry_path}")
              except Exception as exc:
                  print(f"WARNING: could not parse {registry_path}: {exc}")
          else:
              print(f"WARNING: {registry_path} not found — will fall back to static list")'''

NEW_REGISTRY_LOAD = '''          # ── 1. Load issue_registry.json (primary source of truth) ────────────
          # Priority: live merged artifact from THIS run > checked-out repo copy
          LIVE_REGISTRY = Path("/tmp/live_registry/issue_registry.json")
          effective_registry = LIVE_REGISTRY if LIVE_REGISTRY.exists() else registry_path
          registry_entries = []
          if effective_registry.exists():
              try:
                  registry_entries = json.loads(effective_registry.read_text())
                  src_label = "LIVE (nb-merged)" if effective_registry == LIVE_REGISTRY else "repo checkout"
                  print(f"Loaded {len(registry_entries)} entries from {effective_registry} [{src_label}]")
              except Exception as exc:
                  print(f"WARNING: could not parse {effective_registry}: {exc}")
          else:
              print(f"WARNING: no registry found at {registry_path} or {LIVE_REGISTRY} — falling back to static list")'''

r = r.replace(OLD_REGISTRY_LOAD, NEW_REGISTRY_LOAD)

# ── Fix 5: Remove STATIC_META entirely — replace with empty dict + clear comment ─
OLD_STATIC_META_START = "          # ── 4. Static descriptions fallback (used when registry has no entry) ─\n          # ONLY descriptions/actions/nb/severity — statuses come from the registry\n          STATIC_META = {"
OLD_STATIC_META_END   = "              \"FIX-S2\":  {\"severity\":\"LOW\",     \"nb\":\"NB-03\",\n                           \"description\":\"Missing \\\\\\\\label on \\\\\\\\subsection commands.\",\n                           \"action\":\"Add \\\\\\\\label{subsec:<slug>} after each \\\\\\\\subsection{} heading.\"},\n          }"

NEW_STATIC_META = (
    "          # ── 4. Static descriptions fallback (descriptions/actions/nb only — NO statuses) ─\n"
    "          # Statuses are NEVER set here; they come exclusively from the registry.\n"
    "          # This dict is only consulted when the registry entry has no description.\n"
    "          STATIC_META = {\n"
    "              \"FIX-B1\":  {\"severity\":\"CRITICAL\",\"nb\":\"NB-01\",\n"
    "                           \"description\":\"koza1994genetic cited but has no bibitem.\",\n"
    "                           \"action\":\"Add bibitem or redirect cite to koza1992gp.\"},\n"
    "              \"FIX-B2\":  {\"severity\":\"HIGH\",    \"nb\":\"NB-01\",\n"
    "                           \"description\":\"cranmer2023pysr and cranmer2023interpretable alias the same paper.\",\n"
    "                           \"action\":\"Remove cranmer2023interpretable; redirect all cite to cranmer2023pysr.\"},\n"
    "              \"FIX-B3\":  {\"severity\":\"HIGH\",    \"nb\":\"NB-01\",\n"
    "                           \"description\":\"udrescu2020ai and udrescu2020aifeynman alias the same paper.\",\n"
    "                           \"action\":\"Remove udrescu2020aifeynman; redirect all uses to udrescu2020ai.\"},\n"
    "              \"FIX-F1\":  {\"severity\":\"MEDIUM\",  \"nb\":\"NB-05\",\n"
    "                           \"description\":\"hypatiaX_three_systems MISSING — fbox placeholder in §7.1.\",\n"
    "                           \"action\":\"Replace fbox placeholder with final PDF/PNG.\"},\n"
    "              \"FIX-F2\":  {\"severity\":\"MEDIUM\",  \"nb\":\"NB-05\",\n"
    "                           \"description\":\"fig18_r2_heatmap_improved.pdf missing from figures/.\",\n"
    "                           \"action\":\"Run generate_figures.py --experiment exp1 or ci_postprocess figures_deploy.\"},\n"
    "              \"FIX-F3\":  {\"severity\":\"MEDIUM\",  \"nb\":\"NB-05\",\n"
    "                           \"description\":\"fig09_r2_heatmap_regimes.pdf missing from figures/.\",\n"
    "                           \"action\":\"Run generate_figures.py --experiment exp1 or ci_postprocess figures_deploy.\"},\n"
    "              \"FIX-F4\":  {\"severity\":\"MEDIUM\",  \"nb\":\"NB-05\",\n"
    "                           \"description\":\"fig1_seed_sweep.pdf missing from figures/.\",\n"
    "                           \"action\":\"Run generate_figures.py --experiment exp1 or ci_postprocess figures_deploy.\"},\n"
    "              \"FIX-C3\":  {\"severity\":\"CRITICAL\",\"nb\":\"NB-06\",\n"
    "                           \"description\":\"Feynman benchmark split-protocol mismatch.\",\n"
    "                           \"action\":\"Results in exp2_pca_4060/; Gates A/B/C passed.\"},\n"
    "          }"
)

start = r.find(OLD_STATIC_META_START)
end   = r.find(OLD_STATIC_META_END)
if start != -1 and end != -1:
    end += len(OLD_STATIC_META_END)
    r = r[:start] + NEW_STATIC_META + r[end:]
    print("  Replaced STATIC_META (removed hardcoded statuses)")
else:
    print(f"  ERROR: STATIC_META not found (start={start}, end={end})")

# ── Fix 6: Ensure registry fallback warning shows correct path ───────────────
r = r.replace(
    'print(f"WARNING: registry empty — using static list with OPEN status for all items")',
    'print("WARNING: registry empty and no static fallback — conclusions section will be empty")'
)
r = r.replace(
    "    for fid, meta in STATIC_META.items():\n"
    "              findings.append({\n"
    "                  \"id\": fid, \"severity\": meta[\"severity\"], \"status\": \"OPEN\",\n"
    "                  \"description\": meta[\"description\"], \"action\": meta[\"action\"],\n"
    "                  \"nb\": meta[\"nb\"],\n"
    "              })",
    "    pass  # no static fallback — registry is the only source of truth"
)

# ── Fix 7: Fix UPSTREAM_RUN in Download audit summary step ───────────────────
# The step uses ${UPSTREAM_RUN:-$SUM_RUN} but UPSTREAM_RUN is a Python-side var
# from a previous step. Replace with just $SUM_RUN throughout that run block.
r = r.replace(
    '          gh run download "${UPSTREAM_RUN:-$SUM_RUN}" \\\n'
    '            --repo "$REPO" \\\n'
    '            --name "paper-audit-notebooks" \\\n'
    '            --dir /tmp/audit_summary 2>/dev/null || true',
    '          gh run download "$SUM_RUN" \\\n'
    '            --repo "$REPO" \\\n'
    '            --name "paper-audit-notebooks" \\\n'
    '            --dir /tmp/audit_summary 2>/dev/null || true'
)

with open('/home/claude/ci_report.yml', 'w') as f:
    f.write(r)
print("ci_report.yml updated")


bash

python3 << 'PYEOF'
# ── ci_report.yml ─────────────────────────────────────────────────────────────
# Changes:
# 1. Fix UPSTREAM_RUN scoping throughout (use step output, not shell vars)
# 2. Add "Download merged registry artifact" step before checkout
# 3. Remove STATIC_META entirely — statuses from registry only
# 4. Fix "Finalise report" step — inject vars via env:
# 5. Fix UPSTREAM_RUN in "Download audit summary" step
# 6. Add FIX-C3 gates section rendering

with open('/home/claude/ci_report.yml') as f:
    r = f.read()

# ── Fix 1: UPSTREAM_RUN shell var lost between steps ─────────────────────────
# The "Build combined audit report" step sets RUN_DATE/UPSTREAM_RUN as shell
# vars, but "Finalise report HTML" is a separate run: block so they're gone.
# Fix: add an env: block to the Finalise step.

OLD_FINALISE = "      - name: Finalise report HTML\n        run: |"
NEW_FINALISE  = (
    "      - name: Finalise report HTML\n"
    "        env:\n"
    "          RUN_DATE:     ${{ steps.set-run-meta.outputs.run_date }}\n"
    "          UPSTREAM_RUN: ${{ steps.resolve-run.outputs.notebooks_run_id }}\n"
    "        run: |"
)
r = r.replace(OLD_FINALISE, NEW_FINALISE)

# ── Fix 2: Export RUN_DATE from build step as a step output ──────────────────
# Add a dedicated "Set run metadata" step before "Set up Python"
SET_META_STEP = '''      - name: Set run metadata
        id: set-run-meta
        run: |
          echo "run_date=$(date -u '+%Y-%m-%d %H:%M UTC')" >> "$GITHUB_OUTPUT"

'''
# Insert after the first "steps:" line inside build-report
r = r.replace("    steps:\n\n      # ── Resolve the source run ID", 
              "    steps:\n\n" + SET_META_STEP + "      # ── Resolve the source run ID")

# ── Fix 3: Add download of merged registry artifact before checkout ───────────
DOWNLOAD_REGISTRY_STEP = '''
      # ── Download merged registry from THIS run's bundle-artifacts ───────────
      # This is the most current registry: reflects live NB findings just written.
      # Falls back to the checked-out version (next step) if the artifact is absent.
      - name: Download merged registry artifact (from ci_paper_notebooks)
        continue-on-error: true
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          REPO:     ${{ github.repository }}
          NB_RUN:   ${{ steps.resolve-run.outputs.notebooks_run_id }}
        run: |
          mkdir -p /tmp/live_registry
          if [[ -n "$NB_RUN" ]]; then
            gh run download "$NB_RUN" \
              --repo "$REPO" \
              --name "issue-registry-merged-${NB_RUN}" \
              --dir  "/tmp/live_registry" 2>/dev/null || true
          fi
          if [[ -f "/tmp/live_registry/issue_registry.json" ]]; then
            echo "✅ Live merged registry downloaded ($(wc -c < /tmp/live_registry/issue_registry.json) bytes)"
          else
            echo "::notice::Merged registry artifact not found — will use checked-out version"
          fi

'''

# Insert before the checkout step
r = r.replace(
    "      - name: Checkout (for live issue_registry.json)",
    DOWNLOAD_REGISTRY_STEP + "      - name: Checkout (for live issue_registry.json)"
)

# ── Fix 4: update REGISTRY_PATH resolution in the conclusions Python block ────
# Make it prefer /tmp/live_registry/issue_registry.json over the checkout copy
OLD_REGISTRY_LOAD = '''          # ── 1. Load issue_registry.json (primary source of truth) ────────────
          registry_entries = []
          if registry_path.exists():
              try:
                  registry_entries = json.loads(registry_path.read_text())
                  print(f"Loaded {len(registry_entries)} entries from {registry_path}")
              except Exception as exc:
                  print(f"WARNING: could not parse {registry_path}: {exc}")
          else:
              print(f"WARNING: {registry_path} not found — will fall back to static list")'''

NEW_REGISTRY_LOAD = '''          # ── 1. Load issue_registry.json (primary source of truth) ────────────
          # Priority: live merged artifact from THIS run > checked-out repo copy
          LIVE_REGISTRY = Path("/tmp/live_registry/issue_registry.json")
          effective_registry = LIVE_REGISTRY if LIVE_REGISTRY.exists() else registry_path
          registry_entries = []
          if effective_registry.exists():
              try:
                  registry_entries = json.loads(effective_registry.read_text())
                  src_label = "LIVE (nb-merged)" if effective_registry == LIVE_REGISTRY else "repo checkout"
                  print(f"Loaded {len(registry_entries)} entries from {effective_registry} [{src_label}]")
              except Exception as exc:
                  print(f"WARNING: could not parse {effective_registry}: {exc}")
          else:
              print(f"WARNING: no registry found at {registry_path} or {LIVE_REGISTRY} — falling back to static list")'''

r = r.replace(OLD_REGISTRY_LOAD, NEW_REGISTRY_LOAD)

# ── Fix 5: Remove STATIC_META entirely — replace with empty dict + clear comment ─
OLD_STATIC_META_START = "          # ── 4. Static descriptions fallback (used when registry has no entry) ─\n          # ONLY descriptions/actions/nb/severity — statuses come from the registry\n          STATIC_META = {"
OLD_STATIC_META_END   = "              \"FIX-S2\":  {\"severity\":\"LOW\",     \"nb\":\"NB-03\",\n                           \"description\":\"Missing \\\\\\\\label on \\\\\\\\subsection commands.\",\n                           \"action\":\"Add \\\\\\\\label{subsec:<slug>} after each \\\\\\\\subsection{} heading.\"},\n          }"

NEW_STATIC_META = (
    "          # ── 4. Static descriptions fallback (descriptions/actions/nb only — NO statuses) ─\n"
    "          # Statuses are NEVER set here; they come exclusively from the registry.\n"
    "          # This dict is only consulted when the registry entry has no description.\n"
    "          STATIC_META = {\n"
    "              \"FIX-B1\":  {\"severity\":\"CRITICAL\",\"nb\":\"NB-01\",\n"
    "                           \"description\":\"koza1994genetic cited but has no bibitem.\",\n"
    "                           \"action\":\"Add bibitem or redirect cite to koza1992gp.\"},\n"
    "              \"FIX-B2\":  {\"severity\":\"HIGH\",    \"nb\":\"NB-01\",\n"
    "                           \"description\":\"cranmer2023pysr and cranmer2023interpretable alias the same paper.\",\n"
    "                           \"action\":\"Remove cranmer2023interpretable; redirect all cite to cranmer2023pysr.\"},\n"
    "              \"FIX-B3\":  {\"severity\":\"HIGH\",    \"nb\":\"NB-01\",\n"
    "                           \"description\":\"udrescu2020ai and udrescu2020aifeynman alias the same paper.\",\n"
    "                           \"action\":\"Remove udrescu2020aifeynman; redirect all uses to udrescu2020ai.\"},\n"
    "              \"FIX-F1\":  {\"severity\":\"MEDIUM\",  \"nb\":\"NB-05\",\n"
    "                           \"description\":\"hypatiaX_three_systems MISSING — fbox placeholder in §7.1.\",\n"
    "                           \"action\":\"Replace fbox placeholder with final PDF/PNG.\"},\n"
    "              \"FIX-F2\":  {\"severity\":\"MEDIUM\",  \"nb\":\"NB-05\",\n"
    "                           \"description\":\"fig18_r2_heatmap_improved.pdf missing from figures/.\",\n"
    "                           \"action\":\"Run generate_figures.py --experiment exp1 or ci_postprocess figures_deploy.\"},\n"
    "              \"FIX-F3\":  {\"severity\":\"MEDIUM\",  \"nb\":\"NB-05\",\n"
    "                           \"description\":\"fig09_r2_heatmap_regimes.pdf missing from figures/.\",\n"
    "                           \"action\":\"Run generate_figures.py --experiment exp1 or ci_postprocess figures_deploy.\"},\n"
    "              \"FIX-F4\":  {\"severity\":\"MEDIUM\",  \"nb\":\"NB-05\",\n"
    "                           \"description\":\"fig1_seed_sweep.pdf missing from figures/.\",\n"
    "                           \"action\":\"Run generate_figures.py --experiment exp1 or ci_postprocess figures_deploy.\"},\n"
    "              \"FIX-C3\":  {\"severity\":\"CRITICAL\",\"nb\":\"NB-06\",\n"
    "                           \"description\":\"Feynman benchmark split-protocol mismatch.\",\n"
    "                           \"action\":\"Results in exp2_pca_4060/; Gates A/B/C passed.\"},\n"
    "          }"
)

start = r.find(OLD_STATIC_META_START)
end   = r.find(OLD_STATIC_META_END)
if start != -1 and end != -1:
    end += len(OLD_STATIC_META_END)
    r = r[:start] + NEW_STATIC_META + r[end:]
    print("  Replaced STATIC_META (removed hardcoded statuses)")
else:
    print(f"  ERROR: STATIC_META not found (start={start}, end={end})")

# ── Fix 6: Ensure registry fallback warning shows correct path ───────────────
r = r.replace(
    'print(f"WARNING: registry empty — using static list with OPEN status for all items")',
    'print("WARNING: registry empty and no static fallback — conclusions section will be empty")'
)
r = r.replace(
    "    for fid, meta in STATIC_META.items():\n"
    "              findings.append({\n"
    "                  \"id\": fid, \"severity\": meta[\"severity\"], \"status\": \"OPEN\",\n"
    "                  \"description\": meta[\"description\"], \"action\": meta[\"action\"],\n"
    "                  \"nb\": meta[\"nb\"],\n"
    "              })",
    "    pass  # no static fallback — registry is the only source of truth"
)

# ── Fix 7: Fix UPSTREAM_RUN in Download audit summary step ───────────────────
# The step uses ${UPSTREAM_RUN:-$SUM_RUN} but UPSTREAM_RUN is a Python-side var
# from a previous step. Replace with just $SUM_RUN throughout that run block.
r = r.replace(
    '          gh run download "${UPSTREAM_RUN:-$SUM_RUN}" \\\n'
    '            --repo "$REPO" \\\n'
    '            --name "paper-audit-notebooks" \\\n'
    '            --dir /tmp/audit_summary 2>/dev/null || true',
    '          gh run download "$SUM_RUN" \\\n'
    '            --repo "$REPO" \\\n'
    '            --name "paper-audit-notebooks" \\\n'
    '            --dir /tmp/audit_summary 2>/dev/null || true'
)

with open('/home/claude/ci_report.yml', 'w') as f:
    f.write(r)
print("ci_report.yml updated")
PYEOF


bash

# Verify all three files are valid YAML and spot-check key changes
python3 -c "
import yaml, sys

files = [
    '/home/claude/ci_paper_notebooks.yml',
    '/home/claude/ci_paper_audit.yml',
    '/home/claude/ci_report.yml',
]
for f in files:
    try:
        with open(f) as fh:
            yaml.safe_load(fh)
        print(f'✅ {f.split(\"/\")[-1]} — valid YAML')
    except yaml.YAMLError as e:
        print(f'❌ {f.split(\"/\")[-1]} — YAML ERROR: {e}')
        sys.exit(1)
"


sed -n '2025,2045p' /home/claude/ci_paper_notebooks.yml

bash

# The issue is literal newlines in print() strings embedded in YAML heredoc.
# Fix: replace \n in print strings with escaped versions
python3 << 'PYEOF'
with open('/home/claude/ci_paper_notebooks.yml') as f:
    content = f.read()

# The problematic section is the PYMERGE heredoc. The print() calls contain
# literal newlines from Python's f-string "\n" which YAML interprets as actual newlines.
# Fix: escape them properly within the YAML heredoc context.

# Find and fix the two problematic print lines in the PYMERGE block
content = content.replace(
    '          col = "{:<14} {:<12} {:<12} {}"\n'
    '          print("\\n=== NB registry patch merge ===")\n',
    '          col = "{:<14} {:<12} {:<12} {}"\n'
    '          print("")\n'
    '          print("=== NB registry patch merge ===")\n'
)

content = content.replace(
    '          registry_path.write_text(\n'
    '              json.dumps(registry, indent=2, ensure_ascii=False) + "\\n",\n',
    '          registry_path.write_text(\n'
    '              json.dumps(registry, indent=2, ensure_ascii=False) + "\\n",\n'
)

# Also fix the \\n in the write_text call — it's actually fine as-is
# The real problem is the literal newline in print("  \n===...===")
# Let's scan for any print() that contains a literal embedded newline
import re

# Find all lines in PYMERGE that have print(" followed by newline then more content
lines = content.split('\n')
fixed_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    # Check for unclosed print(" that spans multiple lines
    if 'print("' in line and line.count('"') % 2 == 1 and not line.strip().startswith('#'):
        # This line has an odd number of quotes — likely a multi-line string
        # Join with the next line
        combined = line + '\\n' + lines[i+1] if i+1 < len(lines) else line
        fixed_lines.append(combined)
        i += 2
    else:
        fixed_lines.append(line)
        i += 1

content = '\n'.join(fixed_lines)

with open('/home/claude/ci_paper_notebooks.yml', 'w') as f:
    f.write(content)
print("Fixed multiline print strings")
PYEOF

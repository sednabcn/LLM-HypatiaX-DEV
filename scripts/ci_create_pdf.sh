#!/usr/bin/env bash
# =============================================================================
#  ci_create_pdf.sh
#  CI Pipeline: Cross-reference check & PDF compilation for HypatiaX JMLR paper
#
#  Files handled:
#    jmlr_paper_main.tex          — Main paper
#    supp_routing_improvements.tex — Supplementary A
#    supp_benchmark_report.tex    — Supplementary B
#    references.bib               — Shared bibliography
#
#  Usage:
#    chmod +x ci_create_pdf.sh
#    ./ci_create_pdf.sh [--clean] [--main-only] [--supp-only] [--no-merge]
#
#  Exit codes:
#    0  — All steps passed; PDF(s) produced
#    1  — LaTeX or BibTeX error
#    2  — Undefined references / citations detected
#    3  — Missing required files
#    4  — Missing required tools

# chmod +x ci_create_pdf.sh
# ./ci_create_pdf.sh                # full pipeline
# ./ci_create_pdf.sh --clean        # wipe build dir first
# ./ci_create_pdf.sh --main-only    # main paper only
# ./ci_create_pdf.sh --no-merge     # skip combined PDF
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log_step()  { echo -e "\n${CYAN}${BOLD}▶ STEP $1: $2${NC}"; }
log_ok()    { echo -e "  ${GREEN}✔  $1${NC}"; }
log_warn()  { echo -e "  ${YELLOW}⚠  $1${NC}"; }
log_err()   { echo -e "  ${RED}✘  $1${NC}"; }
log_info()  { echo -e "     $1"; }
separator() { echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# ── CLI flags ─────────────────────────────────────────────────────────────────
CLEAN=false
MAIN_ONLY=false
SUPP_ONLY=false
NO_MERGE=false

for arg in "$@"; do
  case "$arg" in
    --clean)     CLEAN=true ;;
    --main-only) MAIN_ONLY=true ;;
    --supp-only) SUPP_ONLY=true ;;
    --no-merge)  NO_MERGE=true ;;
    --help|-h)
      echo "Usage: $0 [--clean] [--main-only] [--supp-only] [--no-merge]"
      echo "  --clean      Remove build artefacts before compiling"
      echo "  --main-only  Compile main paper only"
      echo "  --supp-only  Compile supplementaries only"
      echo "  --no-merge   Skip merging PDFs into combined output"
      exit 0 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

# ── Configuration ─────────────────────────────────────────────────────────────
MAIN_TEX="jmlr_paper_main.tex"
SUPP_A_TEX="supp_routing_improvements.tex"
SUPP_B_TEX="supp_benchmark_report.tex"
BIB_FILE="references.bib"

MAIN_BASE="jmlr_paper_main"
SUPP_A_BASE="supp_routing_improvements"
SUPP_B_BASE="supp_benchmark_report"

BUILD_DIR="build"
OUTPUT_DIR="output"
COMBINED_PDF="output/HypatiaX_JMLR_complete.pdf"
LOG_FILE="build/ci_compile.log"

LATEX_CMD="pdflatex"
LATEX_OPTS="-interaction=nonstopmode -halt-on-error -file-line-error -output-directory=${BUILD_DIR}"
BIBTEX_CMD="bibtex"

# Track warnings/errors across the run
WARNINGS=()
ERRORS=()
EXIT_CODE=0

# ── Record start time ─────────────────────────────────────────────────────────
START_TIME=$(date +%s)

separator
echo -e "${BOLD}  HypatiaX JMLR — CI PDF Compilation Pipeline${NC}"
echo -e "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
separator

# =============================================================================
# STEP 1 — Environment check
# =============================================================================
log_step 1 "Environment & Tool Check"

check_tool() {
  if command -v "$1" &>/dev/null; then
    log_ok "$1 found: $(command -v "$1")"
  else
    log_err "$1 not found — please install it"
    ERRORS+=("Missing tool: $1")
    EXIT_CODE=4
  fi
}

check_tool pdflatex
check_tool bibtex
check_tool kpsewhich

# Optional tools (for merging and checks)
HAS_QPDF=false
HAS_PDFTK=false
if command -v qpdf &>/dev/null;  then HAS_QPDF=true;  log_ok "qpdf found (merge available)"; fi
if command -v pdftk &>/dev/null; then HAS_PDFTK=true; log_ok "pdftk found (merge available)"; fi

if $HAS_QPDF || $HAS_PDFTK; then
  :
else
  log_warn "Neither qpdf nor pdftk found — PDF merge step will be skipped"
  NO_MERGE=true
fi

if [[ $EXIT_CODE -eq 4 ]]; then
  log_err "Required tools missing. Aborting."
  exit 4
fi

# Check LaTeX packages
log_info "Checking required LaTeX packages..."
REQUIRED_PKGS=(jmlr2e amsmath amssymb bm microtype graphicx booktabs
               caption subcaption algorithm algpseudocode multirow
               threeparttable longtable tabularx colortbl hyperref
               xcolor enumitem natbib geometry)

PKG_MISSING=()
for pkg in "${REQUIRED_PKGS[@]}"; do
  if kpsewhich "${pkg}.sty" &>/dev/null || kpsewhich "${pkg}.cls" &>/dev/null; then
    : # found
  else
    PKG_MISSING+=("$pkg")
    log_warn "Package not found via kpsewhich: $pkg"
  fi
done

if [[ ${#PKG_MISSING[@]} -gt 0 ]]; then
  log_warn "Some packages could not be verified: ${PKG_MISSING[*]}"
  log_warn "Compilation will attempt anyway — pdflatex will report errors if missing"
  WARNINGS+=("Unverified packages: ${PKG_MISSING[*]}")
else
  log_ok "All required LaTeX packages verified"
fi

# =============================================================================
# STEP 2 — Source file check
# =============================================================================
log_step 2 "Source File Validation"

check_file() {
  local f="$1" required="${2:-true}"
  if [[ -f "$f" ]]; then
    local size; size=$(wc -c < "$f")
    log_ok "$f  (${size} bytes)"
    return 0
  else
    if [[ "$required" == "true" ]]; then
      log_err "Required file missing: $f"
      ERRORS+=("Missing file: $f")
      EXIT_CODE=3
    else
      log_warn "Optional file not found: $f"
    fi
    return 1
  fi
}

check_file "$MAIN_TEX"
check_file "$SUPP_A_TEX"
check_file "$SUPP_B_TEX"
check_file "$BIB_FILE"

if [[ $EXIT_CODE -eq 3 ]]; then
  log_err "Required source files missing. Aborting."
  exit 3
fi

# Quick syntax pre-check: balanced \begin{document} / \end{document}
log_info "Pre-checking document structure..."
for tex in "$MAIN_TEX" "$SUPP_A_TEX" "$SUPP_B_TEX"; do
  begin_count=$(grep -c '\\begin{document}' "$tex" 2>/dev/null || true)
  end_count=$(grep -c   '\\end{document}'   "$tex" 2>/dev/null || true)
  if [[ "$begin_count" -eq 1 && "$end_count" -eq 1 ]]; then
    log_ok "$tex — document environment OK"
  else
    log_warn "$tex — unexpected \\begin/\\end{document} count (begin=${begin_count}, end=${end_count})"
    WARNINGS+=("$tex: document environment mismatch")
  fi
done

# =============================================================================
# STEP 3 — Cross-reference pre-analysis
# =============================================================================
log_step 3 "Cross-Reference Pre-Analysis"

log_info "Extracting \\label definitions..."

# Collect all labels from all source files
ALL_LABELS=()
declare -A LABEL_SOURCES

for tex in "$MAIN_TEX" "$SUPP_A_TEX" "$SUPP_B_TEX"; do
  while IFS= read -r label; do
    ALL_LABELS+=("$label")
    LABEL_SOURCES["$label"]="$tex"
  done < <(grep -oP '(?<=\\label\{)[^}]+' "$tex" 2>/dev/null || true)
done

log_info "  Total \\label definitions found: ${#ALL_LABELS[@]}"

# Collect all \ref and \eqref calls from main paper
ALL_REFS=()
while IFS= read -r ref; do
  ALL_REFS+=("$ref")
done < <(grep -oP '(?<=\\ref\{)[^}]+|(?<=\\eqref\{)[^}]+|(?<=\\autoref\{)[^}]+' "$MAIN_TEX" 2>/dev/null || true)

log_info "  Total \\ref / \\eqref calls in main paper: ${#ALL_REFS[@]}"

# Collect all \cite calls
ALL_CITES=()
while IFS= read -r cite; do
  # split comma-separated cite keys
  IFS=',' read -ra keys <<< "$cite"
  for key in "${keys[@]}"; do
    key=$(echo "$key" | tr -d ' ')
    ALL_CITES+=("$key")
  done
done < <(grep -oP '(?<=\\cite\{)[^}]+|(?<=\\citep\{)[^}]+|(?<=\\citet\{)[^}]+' "$MAIN_TEX" 2>/dev/null || true)

log_info "  Total \\cite keys in main paper: ${#ALL_CITES[@]}"

# Check for duplicate labels
log_info "Checking for duplicate labels..."
DUPE_LABELS=()
declare -A SEEN_LABELS
for label in "${ALL_LABELS[@]}"; do
  if [[ -v SEEN_LABELS["$label"] ]]; then
    DUPE_LABELS+=("$label")
    log_warn "Duplicate label: $label  (in ${LABEL_SOURCES[$label]})"
    WARNINGS+=("Duplicate label: $label")
  else
    SEEN_LABELS["$label"]=1
  fi
done
if [[ ${#DUPE_LABELS[@]} -eq 0 ]]; then
  log_ok "No duplicate labels found"
fi

# Check for undefined \refs (labels used but not defined anywhere)
log_info "Checking for potentially undefined \\ref targets..."
UNDEF_REFS=()
for ref in "${ALL_REFS[@]}"; do
  if [[ ! -v SEEN_LABELS["$ref"] ]]; then
    UNDEF_REFS+=("$ref")
    log_warn "\\ref{$ref} — label not found in any source file"
  fi
done
if [[ ${#UNDEF_REFS[@]} -eq 0 ]]; then
  log_ok "All \\ref targets have corresponding \\label definitions"
else
  log_warn "${#UNDEF_REFS[@]} potentially undefined reference(s) found — LaTeX will report these"
  WARNINGS+=("Potentially undefined refs: ${UNDEF_REFS[*]}")
fi

# Check cite keys exist in .bib
log_info "Checking citation keys against ${BIB_FILE}..."
BIB_KEYS=()
while IFS= read -r key; do
  BIB_KEYS+=("$key")
done < <(grep -oP '(?<=@\w{2,20}\{)[^,]+' "$BIB_FILE" 2>/dev/null | tr -d ' ' || true)

declare -A BIB_KEY_SET
for key in "${BIB_KEYS[@]}"; do BIB_KEY_SET["$key"]=1; done
log_info "  .bib entries found: ${#BIB_KEYS[@]}"

UNDEF_CITES=()
for cite in "${ALL_CITES[@]}"; do
  if [[ ! -v BIB_KEY_SET["$cite"] ]]; then
    UNDEF_CITES+=("$cite")
    log_warn "\\cite{$cite} — key not found in ${BIB_FILE}"
  fi
done
if [[ ${#UNDEF_CITES[@]} -eq 0 ]]; then
  log_ok "All citation keys found in bibliography"
else
  log_warn "${#UNDEF_CITES[@]} citation key(s) not found in .bib"
  WARNINGS+=("Missing bib keys: ${UNDEF_CITES[*]}")
fi

# =============================================================================
# STEP 4 — Prepare build directories
# =============================================================================
log_step 4 "Prepare Build Directories"

if $CLEAN; then
  log_info "Cleaning previous build artefacts..."
  rm -rf "$BUILD_DIR"
  log_ok "Build directory cleaned"
fi

mkdir -p "$BUILD_DIR" "$OUTPUT_DIR"
log_ok "Directories ready: ${BUILD_DIR}/, ${OUTPUT_DIR}/"

# Initialise log
echo "=== HypatiaX CI Compile Log — $(date) ===" > "$LOG_FILE"

# =============================================================================
# Helper: compile a single .tex file with BibTeX
# =============================================================================
compile_tex() {
  local tex_file="$1"    # e.g. jmlr_paper_main.tex
  local base="$2"        # e.g. jmlr_paper_main
  local label="$3"       # human label e.g. "Main Paper"
  local run_bibtex="${4:-true}"

  log_info "Compiling: ${tex_file}  →  ${BUILD_DIR}/${base}.pdf"

  # ── Pass 1: initial compile ─────────────────────────────────────────────
  log_info "  [Pass 1] pdflatex (initial) ..."
  if ! $LATEX_CMD $LATEX_OPTS "$tex_file" >> "$LOG_FILE" 2>&1; then
    log_err "${label}: pdflatex Pass 1 failed — see ${LOG_FILE}"
    ERRORS+=("${label}: pdflatex Pass 1 failed")
    EXIT_CODE=1
    return 1
  fi
  log_ok "${label}: Pass 1 complete"

  # ── BibTeX ─────────────────────────────────────────────────────────────
  if [[ "$run_bibtex" == "true" ]]; then
    log_info "  [BibTeX] Running bibtex on ${BUILD_DIR}/${base}.aux ..."
    # bibtex needs to be run from build dir, or with the full aux path
    if ! $BIBTEX_CMD "${BUILD_DIR}/${base}" >> "$LOG_FILE" 2>&1; then
      # bibtex exit code 1 = warnings only; 2 = errors
      local bt_exit=$?
      if [[ $bt_exit -eq 2 ]]; then
        log_err "${label}: bibtex reported errors — see ${LOG_FILE}"
        ERRORS+=("${label}: bibtex errors")
        EXIT_CODE=1
        return 1
      else
        log_warn "${label}: bibtex reported warnings (exit ${bt_exit})"
        WARNINGS+=("${label}: bibtex warnings")
      fi
    fi
    log_ok "${label}: bibtex complete"
  fi

  # ── Pass 2: resolve citations ───────────────────────────────────────────
  log_info "  [Pass 2] pdflatex (resolve citations) ..."
  if ! $LATEX_CMD $LATEX_OPTS "$tex_file" >> "$LOG_FILE" 2>&1; then
    log_err "${label}: pdflatex Pass 2 failed"
    ERRORS+=("${label}: pdflatex Pass 2 failed")
    EXIT_CODE=1
    return 1
  fi
  log_ok "${label}: Pass 2 complete"

  # ── Pass 3: resolve cross-references ────────────────────────────────────
  log_info "  [Pass 3] pdflatex (stabilise cross-references) ..."
  if ! $LATEX_CMD $LATEX_OPTS "$tex_file" >> "$LOG_FILE" 2>&1; then
    log_err "${label}: pdflatex Pass 3 failed"
    ERRORS+=("${label}: pdflatex Pass 3 failed")
    EXIT_CODE=1
    return 1
  fi
  log_ok "${label}: Pass 3 complete"

  # ── Pass 4 (conditional): check if another pass is needed ───────────────
  local aux_file="${BUILD_DIR}/${base}.aux"
  if grep -q 'Rerun' "${BUILD_DIR}/${base}.log" 2>/dev/null; then
    log_info "  [Pass 4] pdflatex ('Rerun' detected in log — stabilising) ..."
    if ! $LATEX_CMD $LATEX_OPTS "$tex_file" >> "$LOG_FILE" 2>&1; then
      log_err "${label}: pdflatex Pass 4 failed"
      ERRORS+=("${label}: pdflatex Pass 4 failed")
      EXIT_CODE=1
      return 1
    fi
    log_ok "${label}: Pass 4 complete"
  else
    log_info "  [Pass 4] Skipped — no 'Rerun' in log"
  fi

  # ── Post-compile: check for undefined refs / citations in log ───────────
  local pdflatex_log="${BUILD_DIR}/${base}.log"
  log_info "  Scanning log for warnings..."

  local undef_ref_count; undef_ref_count=$(grep -c 'undefined' "$pdflatex_log" 2>/dev/null || true)
  local citation_warn;   citation_warn=$(grep -c 'Citation.*undefined' "$pdflatex_log" 2>/dev/null || true)
  local label_warn;      label_warn=$(grep -c 'multiply defined' "$pdflatex_log" 2>/dev/null || true)
  local overfull_warn;   overfull_warn=$(grep -c 'Overfull' "$pdflatex_log" 2>/dev/null || true)
  local missing_file;    missing_file=$(grep -c "File.*not found" "$pdflatex_log" 2>/dev/null || true)

  if [[ $undef_ref_count -gt 0 ]]; then
    log_warn "${label}: ${undef_ref_count} undefined reference(s) in log"
    grep 'undefined' "$pdflatex_log" | head -10 >> "$LOG_FILE" || true
    WARNINGS+=("${label}: ${undef_ref_count} undefined refs")
    if [[ $EXIT_CODE -eq 0 ]]; then EXIT_CODE=2; fi
  else
    log_ok "${label}: No undefined references"
  fi

  if [[ $citation_warn -gt 0 ]]; then
    log_warn "${label}: ${citation_warn} undefined citation(s)"
    WARNINGS+=("${label}: ${citation_warn} undefined citations")
    if [[ $EXIT_CODE -eq 0 ]]; then EXIT_CODE=2; fi
  else
    log_ok "${label}: No undefined citations"
  fi

  if [[ $label_warn -gt 0 ]]; then
    log_warn "${label}: ${label_warn} multiply-defined label(s)"
    WARNINGS+=("${label}: multiply-defined labels")
  fi

  if [[ $overfull_warn -gt 0 ]]; then
    log_info "  Note: ${overfull_warn} Overfull \\hbox warning(s) (cosmetic)"
  fi

  if [[ $missing_file -gt 0 ]]; then
    log_warn "${label}: Missing file(s) referenced (figures?)"
    WARNINGS+=("${label}: missing included files")
  fi

  # ── Move PDF to output ──────────────────────────────────────────────────
  if [[ -f "${BUILD_DIR}/${base}.pdf" ]]; then
    cp "${BUILD_DIR}/${base}.pdf" "${OUTPUT_DIR}/${base}.pdf"
    local pdf_size; pdf_size=$(du -h "${OUTPUT_DIR}/${base}.pdf" | cut -f1)
    log_ok "${label}: PDF written → ${OUTPUT_DIR}/${base}.pdf  (${pdf_size})"
  else
    log_err "${label}: PDF not produced despite no fatal errors reported"
    ERRORS+=("${label}: PDF missing after compilation")
    EXIT_CODE=1
    return 1
  fi

  return 0
}

# =============================================================================
# STEP 5 — Compile Main Paper
# =============================================================================
if ! $SUPP_ONLY; then
  log_step 5 "Compile Main Paper (${MAIN_TEX})"
  compile_tex "$MAIN_TEX" "$MAIN_BASE" "Main Paper" true
else
  log_step 5 "Compile Main Paper — SKIPPED (--supp-only)"
fi

# =============================================================================
# STEP 6 — Compile Supplementary A
# =============================================================================
if ! $MAIN_ONLY; then
  log_step 6 "Compile Supplementary A (${SUPP_A_TEX})"
  compile_tex "$SUPP_A_TEX" "$SUPP_A_BASE" "Supp-A" true
else
  log_step 6 "Compile Supplementary A — SKIPPED (--main-only)"
fi

# =============================================================================
# STEP 7 — Compile Supplementary B
# =============================================================================
if ! $MAIN_ONLY; then
  log_step 7 "Compile Supplementary B (${SUPP_B_TEX})"
  compile_tex "$SUPP_B_TEX" "$SUPP_B_BASE" "Supp-B" true
else
  log_step 7 "Compile Supplementary B — SKIPPED (--main-only)"
fi

# =============================================================================
# STEP 8 — Merge PDFs into single combined output
# =============================================================================
log_step 8 "Merge PDFs into Combined Output"

PDFS_TO_MERGE=()
for pdf in "${OUTPUT_DIR}/${MAIN_BASE}.pdf" \
           "${OUTPUT_DIR}/${SUPP_A_BASE}.pdf" \
           "${OUTPUT_DIR}/${SUPP_B_BASE}.pdf"; do
  [[ -f "$pdf" ]] && PDFS_TO_MERGE+=("$pdf")
done

if $NO_MERGE; then
  log_info "Merge skipped (--no-merge)"
elif [[ ${#PDFS_TO_MERGE[@]} -lt 2 ]]; then
  log_warn "Fewer than 2 PDFs available — merge skipped"
else
  log_info "Merging: ${PDFS_TO_MERGE[*]}"
  if $HAS_QPDF; then
    qpdf --empty --pages "${PDFS_TO_MERGE[@]}" -- "$COMBINED_PDF" 2>> "$LOG_FILE" \
      && log_ok "Combined PDF → ${COMBINED_PDF}  ($(du -h "$COMBINED_PDF" | cut -f1))" \
      || { log_err "qpdf merge failed"; WARNINGS+=("PDF merge failed"); }
  elif $HAS_PDFTK; then
    pdftk "${PDFS_TO_MERGE[@]}" cat output "$COMBINED_PDF" 2>> "$LOG_FILE" \
      && log_ok "Combined PDF → ${COMBINED_PDF}  ($(du -h "$COMBINED_PDF" | cut -f1))" \
      || { log_err "pdftk merge failed"; WARNINGS+=("PDF merge failed"); }
  fi
fi

# =============================================================================
# STEP 9 — Final cross-reference report from compiled .aux files
# =============================================================================
log_step 9 "Post-Compile Cross-Reference Report"

CROSS_REF_REPORT="${OUTPUT_DIR}/cross_ref_report.txt"
{
  echo "=== HypatiaX Cross-Reference Report ==="
  echo "Generated: $(date)"
  echo ""
  echo "--- Labels defined (all source files) ---"
  for tex in "$MAIN_TEX" "$SUPP_A_TEX" "$SUPP_B_TEX"; do
    echo ""
    echo "  ${tex}:"
    grep -oP '(?<=\\label\{)[^}]+' "$tex" 2>/dev/null | sort | sed 's/^/    /' || echo "    (none)"
  done
  echo ""
  echo "--- Citation keys in references.bib ---"
  grep -oP '(?<=@\w{2,20}\{)[^,]+' "$BIB_FILE" 2>/dev/null | sort | sed 's/^/  /' || echo "  (none)"
  echo ""
  echo "--- Undefined references detected (pre-compile static check) ---"
  if [[ ${#UNDEF_REFS[@]} -gt 0 ]]; then
    for r in "${UNDEF_REFS[@]}"; do echo "  UNDEF REF: $r"; done
  else
    echo "  None"
  fi
  echo ""
  echo "--- Undefined citations detected (pre-compile static check) ---"
  if [[ ${#UNDEF_CITES[@]} -gt 0 ]]; then
    for c in "${UNDEF_CITES[@]}"; do echo "  UNDEF CITE: $c"; done
  else
    echo "  None"
  fi
} > "$CROSS_REF_REPORT"

log_ok "Cross-reference report → ${CROSS_REF_REPORT}"

# Also scan compiled .log files for any LaTeX-detected issues
log_info "Scanning compiled logs for LaTeX-detected ref/cite warnings..."
for base in "$MAIN_BASE" "$SUPP_A_BASE" "$SUPP_B_BASE"; do
  logfile="${BUILD_DIR}/${base}.log"
  [[ -f "$logfile" ]] || continue
  issues=$(grep -E 'undefined|multiply defined|Citation.*undefined' "$logfile" 2>/dev/null | \
           grep -v '^$' | sort -u | head -20 || true)
  if [[ -n "$issues" ]]; then
    echo "" >> "$CROSS_REF_REPORT"
    echo "--- LaTeX log issues in ${base}.log ---" >> "$CROSS_REF_REPORT"
    echo "$issues" >> "$CROSS_REF_REPORT"
    log_warn "Ref/cite issues found in ${base}.log (see report)"
  fi
done

# =============================================================================
# STEP 10 — Summary
# =============================================================================
log_step 10 "Summary"
separator

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo -e "\n${BOLD}  Output files:${NC}"
for f in "${OUTPUT_DIR}"/*.pdf "${OUTPUT_DIR}"/*.txt; do
  [[ -f "$f" ]] && echo -e "    ${GREEN}✔${NC}  $f  ($(du -h "$f" | cut -f1))"
done

echo -e "\n${BOLD}  Warnings (${#WARNINGS[@]}):${NC}"
if [[ ${#WARNINGS[@]} -eq 0 ]]; then
  echo -e "    ${GREEN}None${NC}"
else
  for w in "${WARNINGS[@]}"; do echo -e "    ${YELLOW}⚠  ${w}${NC}"; done
fi

echo -e "\n${BOLD}  Errors (${#ERRORS[@]}):${NC}"
if [[ ${#ERRORS[@]} -eq 0 ]]; then
  echo -e "    ${GREEN}None${NC}"
else
  for e in "${ERRORS[@]}"; do echo -e "    ${RED}✘  ${e}${NC}"; done
fi

echo -e "\n  Full log: ${LOG_FILE}"
echo -e "  Cross-ref report: ${CROSS_REF_REPORT}"
echo -e "  Elapsed: ${ELAPSED}s"
separator

if [[ $EXIT_CODE -eq 0 ]]; then
  echo -e "\n${GREEN}${BOLD}  ✔ PIPELINE PASSED${NC}\n"
elif [[ $EXIT_CODE -eq 2 ]]; then
  echo -e "\n${YELLOW}${BOLD}  ⚠ PIPELINE PASSED WITH CROSS-REFERENCE WARNINGS${NC}\n"
else
  echo -e "\n${RED}${BOLD}  ✘ PIPELINE FAILED (exit ${EXIT_CODE})${NC}\n"
fi

exit $EXIT_CODE

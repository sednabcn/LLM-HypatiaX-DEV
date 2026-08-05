# =============================================================================
#  Makefile — HypatiaX JMLR paper build
#
#  Single source of truth for compiling the main paper and both supplements.
#  .github/workflows/ci_create_pdf.yml calls these same targets (`make main`,
#  `make suppa`, `make suppb`, `make verify-main`, ...) so local builds and CI
#  builds can never drift apart.
#
#  Usage:
#    make validate          # source/structure/xref checks — no LaTeX needed
#    make main               # build $(OUTPUT_DIR)/jmlr_paper_main.pdf
#    make suppa                # build $(OUTPUT_DIR)/supp_routing_improvements.pdf
#    make suppb                  # build $(OUTPUT_DIR)/supp_benchmark_report.pdf
#    make all                      # main + suppa + suppb
#    make merge                      # combine the three into $(MERGED_PDF)
#    make verify-main|verify-suppa|verify-suppb|verify-merged
#                                       # qpdf structure + page count + size sanity
#    make clean                          # remove build/ output/
#    make distclean                        # clean + remove *.pdf from repo root
#
#  Requires on PATH: pdflatex, bibtex, qpdf, pdfinfo (poppler-utils), python3.
#  Override any path/dir via env or `make VAR=value`.
# =============================================================================

MAIN_TEX      ?= jmlr_paper_main.tex
SUPP_A_TEX    ?= supp_routing_improvements.tex
SUPP_B_TEX    ?= supp_benchmark_report.tex
BIB_FILE      ?= references.bib
BUILD_DIR     ?= build
OUTPUT_DIR    ?= output
MERGED_PDF    ?= HypatiaX_JMLR_complete.pdf
MIN_PDF_BYTES ?= 10240

MAIN_BASE  := $(basename $(MAIN_TEX))
SUPPA_BASE := $(basename $(SUPP_A_TEX))
SUPPB_BASE := $(basename $(SUPP_B_TEX))

PDFLATEX := pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=$(BUILD_DIR)

.PHONY: help all main suppa suppb merge validate clean distclean \
        verify-main verify-suppa verify-suppb verify-merged _verify

help:
	@echo "Targets: validate main suppa suppb all merge verify-main verify-suppa verify-suppb verify-merged clean distclean"

$(BUILD_DIR):
	mkdir -p "$(BUILD_DIR)"

$(OUTPUT_DIR):
	mkdir -p "$(OUTPUT_DIR)"

# ── Validation (no LaTeX toolchain required) ────────────────────────────────
validate:
	@echo "== Required files =="
	@missing=0; \
	for f in "$(MAIN_TEX)" "$(SUPP_A_TEX)" "$(SUPP_B_TEX)" "$(BIB_FILE)"; do \
		if [ -f "$$f" ]; then \
			echo "OK       $$f ($$(wc -c < "$$f") bytes)"; \
		else \
			echo "MISSING  $$f" >&2; \
			missing=$$((missing + 1)); \
		fi; \
	done; \
	[ $$missing -eq 0 ] || exit 3
	@echo "== Document structure =="
	@for tex in "$(MAIN_TEX)" "$(SUPP_A_TEX)" "$(SUPP_B_TEX)"; do \
		begins=$$(grep -c '\\begin{document}' "$$tex" 2>/dev/null || true); \
		ends=$$(grep -c '\\end{document}' "$$tex" 2>/dev/null || true); \
		if [ "$$begins" = "1" ] && [ "$$ends" = "1" ]; then \
			echo "OK       $$tex — begin/end{document} balanced"; \
		else \
			echo "WARN     $$tex — unbalanced document environment (begin=$$begins end=$$ends)" >&2; \
		fi; \
	done
	@echo "== Cross-reference check (main paper) =="
	@python3 scripts/check_xrefs.py "$(MAIN_TEX)" "$(SUPP_A_TEX)" "$(SUPP_B_TEX)" "$(BIB_FILE)"

# ── Compile: main paper ──────────────────────────────────────────────────────
main: $(BUILD_DIR) $(OUTPUT_DIR)
	$(PDFLATEX) "$(MAIN_TEX)"
	-bibtex "$(BUILD_DIR)/$(MAIN_BASE)"
	$(PDFLATEX) "$(MAIN_TEX)"
	$(PDFLATEX) "$(MAIN_TEX)"
	@if grep -q 'Rerun' "$(BUILD_DIR)/$(MAIN_BASE).log" 2>/dev/null; then \
		echo "Rerun detected — running pass 4 for $(MAIN_BASE)"; \
		$(PDFLATEX) "$(MAIN_TEX)"; \
	fi
	cp "$(BUILD_DIR)/$(MAIN_BASE).pdf" "$(OUTPUT_DIR)/$(MAIN_BASE).pdf"
	@echo "Built $(OUTPUT_DIR)/$(MAIN_BASE).pdf"

# ── Compile: Supplementary A ─────────────────────────────────────────────────
suppa: $(BUILD_DIR) $(OUTPUT_DIR)
	$(PDFLATEX) "$(SUPP_A_TEX)"
	-bibtex "$(BUILD_DIR)/$(SUPPA_BASE)"
	$(PDFLATEX) "$(SUPP_A_TEX)"
	$(PDFLATEX) "$(SUPP_A_TEX)"
	@if grep -q 'Rerun' "$(BUILD_DIR)/$(SUPPA_BASE).log" 2>/dev/null; then \
		echo "Rerun detected — running pass 4 for $(SUPPA_BASE)"; \
		$(PDFLATEX) "$(SUPP_A_TEX)"; \
	fi
	cp "$(BUILD_DIR)/$(SUPPA_BASE).pdf" "$(OUTPUT_DIR)/$(SUPPA_BASE).pdf"
	@echo "Built $(OUTPUT_DIR)/$(SUPPA_BASE).pdf"

# ── Compile: Supplementary B ─────────────────────────────────────────────────
suppb: $(BUILD_DIR) $(OUTPUT_DIR)
	$(PDFLATEX) "$(SUPP_B_TEX)"
	-bibtex "$(BUILD_DIR)/$(SUPPB_BASE)"
	$(PDFLATEX) "$(SUPP_B_TEX)"
	$(PDFLATEX) "$(SUPP_B_TEX)"
	@if grep -q 'Rerun' "$(BUILD_DIR)/$(SUPPB_BASE).log" 2>/dev/null; then \
		echo "Rerun detected — running pass 4 for $(SUPPB_BASE)"; \
		$(PDFLATEX) "$(SUPP_B_TEX)"; \
	fi
	cp "$(BUILD_DIR)/$(SUPPB_BASE).pdf" "$(OUTPUT_DIR)/$(SUPPB_BASE).pdf"
	@echo "Built $(OUTPUT_DIR)/$(SUPPB_BASE).pdf"

all: main suppa suppb

# ── Merge all three into one combined PDF ───────────────────────────────────
merge: $(OUTPUT_DIR)
	qpdf --empty --pages \
		"$(OUTPUT_DIR)/$(MAIN_BASE).pdf" \
		"$(OUTPUT_DIR)/$(SUPPA_BASE).pdf" \
		"$(OUTPUT_DIR)/$(SUPPB_BASE).pdf" \
		-- "$(OUTPUT_DIR)/$(MERGED_PDF)"
	@echo "Combined PDF: $$(du -h "$(OUTPUT_DIR)/$(MERGED_PDF)" | cut -f1)"

# ── Verification: qpdf structure + page count + size sanity ────────────────
verify-main:
	@$(MAKE) --no-print-directory _verify PDF="$(OUTPUT_DIR)/$(MAIN_BASE).pdf"

verify-suppa:
	@$(MAKE) --no-print-directory _verify PDF="$(OUTPUT_DIR)/$(SUPPA_BASE).pdf"

verify-suppb:
	@$(MAKE) --no-print-directory _verify PDF="$(OUTPUT_DIR)/$(SUPPB_BASE).pdf"

verify-merged:
	@$(MAKE) --no-print-directory _verify PDF="$(OUTPUT_DIR)/$(MERGED_PDF)"

_verify:
	@test -n "$(PDF)" || { echo "PDF= not set" >&2; exit 2; }
	@echo "== $(PDF) =="
	@qpdf --check "$(PDF)" >/dev/null && echo "OK       qpdf structure valid" \
		|| { echo "ERROR    qpdf structural errors in $(PDF)" >&2; exit 1; }
	@pages=$$(pdfinfo "$(PDF)" 2>/dev/null | awk '/^Pages:/ {print $$2}'); \
	 echo "Pages: $${pages:-0}"; \
	 [ "$${pages:-0}" -gt 0 ] || { echo "ERROR    0 pages in $(PDF)" >&2; exit 1; }
	@size=$$(stat -c%s "$(PDF)" 2>/dev/null || stat -f%z "$(PDF)"); \
	 echo "Size: $$size bytes"; \
	 [ "$$size" -gt $(MIN_PDF_BYTES) ] || { echo "ERROR    $(PDF) suspiciously small ($$size bytes)" >&2; exit 1; }

# ── Cleaning ─────────────────────────────────────────────────────────────────
clean:
	rm -rf "$(BUILD_DIR)" "$(OUTPUT_DIR)"

distclean: clean
	rm -f "$(MAIN_BASE).pdf" "$(SUPPA_BASE).pdf" "$(SUPPB_BASE).pdf" "$(MERGED_PDF)"

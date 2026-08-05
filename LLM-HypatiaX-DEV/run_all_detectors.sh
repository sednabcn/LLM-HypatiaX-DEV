#!/usr/bin/env bash
# run_all_detectors.sh
#
# Runs both contamination detectors back-to-back and prints one combined
# report:
#   1. detect_contaminated_files.py     -> already-contaminated FILENAMES on disk
#   2. detect_prefix_duplication_code.py -> SOURCE CODE likely to generate them
#
# Assumes detect_contaminated_files.py and detect_prefix_duplication_code.py
# live in the same directory as this script (override with --tools-dir if not).
#
# Usage:
#   bash run_all_detectors.sh [REPO_ROOT] [options]
#
# Options:
#   --root PATH          Repo root to scan (default: current directory)
#   --tools-dir PATH      Directory containing the two detect_*.py scripts
#                         (default: same directory as this script)
#   --paths "p1 p2 ..."   Space-separated list of subpaths to scan, relative
#                         to --root (default: ".github scripts hypatiax")
#   --json-dir PATH       If set, write report1.json / report2.json here
#   -h, --help            Show this help
#
# Examples:
#   bash run_all_detectors.sh .
#   bash run_all_detectors.sh --root /path/to/repo
#   bash run_all_detectors.sh --root . --paths ".github/workflows .github/scripts scripts hypatiax"
#   bash run_all_detectors.sh --root . --json-dir ./detector_reports
#
# Exit code:
#   0 - clean on both detectors
#   1 - either detector found something
#   2 - usage / setup error (e.g. tool scripts not found)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$SCRIPT_DIR"
REPO_ROOT="."
SCAN_PATHS=".github scripts hypatiax"
JSON_DIR=""

print_help() {
    sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# ── Arg parsing ──────────────────────────────────────────────────────────
POSITIONAL_ROOT_SET=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            REPO_ROOT="$2"; shift 2 ;;
        --tools-dir)
            TOOLS_DIR="$2"; shift 2 ;;
        --paths)
            SCAN_PATHS="$2"; shift 2 ;;
        --json-dir)
            JSON_DIR="$2"; shift 2 ;;
        -h|--help)
            print_help; exit 0 ;;
        --*)
            echo "Unknown option: $1" >&2; exit 2 ;;
        *)
            if ! $POSITIONAL_ROOT_SET; then
                REPO_ROOT="$1"
                POSITIONAL_ROOT_SET=true
            else
                echo "Unexpected extra argument: $1" >&2; exit 2
            fi
            shift ;;
    esac
done

DETECT_CONTAM="${TOOLS_DIR}/detect_contaminated_files.py"
DETECT_CODE="${TOOLS_DIR}/detect_prefix_duplication_code.py"

for f in "$DETECT_CONTAM" "$DETECT_CODE"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: expected tool not found: $f" >&2
        echo "  (pass --tools-dir /path/to/scripts if they live elsewhere)" >&2
        exit 2
    fi
done

if [[ ! -d "$REPO_ROOT" ]]; then
    echo "ERROR: repo root not found: $REPO_ROOT" >&2
    exit 2
fi

if [[ -n "$JSON_DIR" ]]; then
    mkdir -p "$JSON_DIR"
fi

SEP="════════════════════════════════════════════════════════════════════"

echo "$SEP"
echo "  Combined figure-contamination detector"
echo "  Repo root  : $(cd "$REPO_ROOT" && pwd)"
echo "  Scan paths : $SCAN_PATHS"
echo "$SEP"
echo ""

# ── Part 1: contaminated filenames already on disk ──────────────────────
echo "── [1/2] detect_contaminated_files.py — files already contaminated ──"
echo ""

CONTAM_JSON_ARGS=()
if [[ -n "$JSON_DIR" ]]; then
    CONTAM_JSON_ARGS=(--json "${JSON_DIR}/contaminated_files_report.json")
fi

(
    cd "$REPO_ROOT" && \
    # shellcheck disable=SC2086
    python3 "$DETECT_CONTAM" $SCAN_PATHS "${CONTAM_JSON_ARGS[@]}"
)
RC1=$?

echo ""
echo "$SEP"
echo ""

# ── Part 2: source code likely to generate the contamination ────────────
echo "── [2/2] detect_prefix_duplication_code.py — code that could produce it ──"
echo ""

CODE_JSON_ARGS=()
if [[ -n "$JSON_DIR" ]]; then
    CODE_JSON_ARGS=(--json "${JSON_DIR}/prefix_duplication_code_report.json")
fi

(
    cd "$REPO_ROOT" && \
    # shellcheck disable=SC2086
    python3 "$DETECT_CODE" $SCAN_PATHS "${CODE_JSON_ARGS[@]}"
)
RC2=$?

echo ""
echo "$SEP"
echo "  Summary"
echo "$SEP"

report_status() {
    local label="$1" rc="$2"
    case "$rc" in
        0) echo "  [CLEAN]   $label" ;;
        1) echo "  [FLAGGED] $label" ;;
        *) echo "  [ERROR]   $label (exit $rc)" ;;
    esac
}

report_status "detect_contaminated_files.py      (files on disk)" "$RC1"
report_status "detect_prefix_duplication_code.py  (source code)"  "$RC2"

if [[ -n "$JSON_DIR" ]]; then
    echo ""
    echo "  JSON reports written to: $JSON_DIR"
fi

echo "$SEP"

# Overall exit code: worst of the two (treat any non-zero as "found something")
OVERALL=0
[[ "$RC1" -ne 0 || "$RC2" -ne 0 ]] && OVERALL=1
[[ "$RC1" -ge 2 || "$RC2" -ge 2 ]] && OVERALL=2

exit "$OVERALL"

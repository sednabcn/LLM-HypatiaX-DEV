#!/bin/bash
# Activate HypatiaX JMLR environment
# Usage: source activate_hypatiax.sh
#
# This works on any system - paths are auto-detected!

# Detect where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -f "$SCRIPT_DIR/.env" ]; then
    # Source the environment configuration
    source "$SCRIPT_DIR/.env"
    
    echo ""
    echo "======================================"
    echo "✅ HypatiaX JMLR Environment Active"
    echo "======================================"
    echo "  Project Root: $HYPATIAX_ROOT"
    echo "  Python Path:  ${PYTHONPATH%%:*}"  # Show first entry
    echo ""
    echo "Quick commands:"
    echo "  • Test import:       python -c 'from hypatiax.protocols.experiment_protocol import ExperimentProtocol; print(\"✓ OK\")'"
    echo "  • Run experiments:   python hypatiax/core/base_pure_llm/baseline_pure_llm.py"
    echo "  • Deactivate:        unset HYPATIAX_ROOT PYTHONPATH HYPATIAX_OUTPUT_DIR HYPATIAX_DATA_DIR"
    echo ""
else
    echo ""
    echo "======================================"
    echo "❌ Error: .env file not found"
    echo "======================================"
    echo "Please run setup first:"
    echo "  cd $SCRIPT_DIR"
    echo "  ./setup_environment.sh"
    echo ""
fi

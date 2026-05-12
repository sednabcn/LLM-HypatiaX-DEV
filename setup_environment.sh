#!/bin/bash
# Setup script for HypatiaX JMLR environment
# This creates a .env file with proper PYTHONPATH configuration
# Works regardless of where the project is installed

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

echo "======================================"
echo "HypatiaX Environment Setup"
echo "======================================"
echo ""
echo "Detected project root: $PROJECT_ROOT"
echo ""

# Verify we're in the right directory
if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found!"
    echo "   Please run this script from the project root directory."
    echo "   Expected: ~/path/to/LLM-HypatiaX-PAPERS/papers/2025-JMLR/"
    exit 1
fi

if [ ! -d "$PROJECT_ROOT/hypatiax" ]; then
    echo "❌ Error: hypatiax/ directory not found!"
    echo "   This doesn't appear to be the HypatiaX project directory."
    exit 1
fi

# Create .env file with DYNAMIC paths (uses shell variable expansion)
cat > "$PROJECT_ROOT/.env" << 'EOF'
# HypatiaX JMLR Environment Configuration
# Auto-generated - DO NOT EDIT MANUALLY
# 
# This file uses relative path detection - it will work wherever you install the project.
# If you move the project, re-run: ./setup_environment.sh

# Detect project root dynamically (where this .env file lives)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export HYPATIAX_ROOT="$SCRIPT_DIR"

# Set PYTHONPATH to prioritize this project
# This ensures imports work correctly even with other hypatiax versions installed
export PYTHONPATH="${HYPATIAX_ROOT}:${PYTHONPATH}"

# Optional: Configure output directories
export HYPATIAX_OUTPUT_DIR="${HYPATIAX_ROOT}/outputs"
export HYPATIAX_DATA_DIR="${HYPATIAX_ROOT}/data"

# Optional: Add project scripts to PATH if needed
# export PATH="${HYPATIAX_ROOT}/scripts:${PATH}"
EOF

echo "✅ Created .env file at: $PROJECT_ROOT/.env"
echo ""

# Create or update .gitignore
if [ -f "$PROJECT_ROOT/.gitignore" ]; then
    if ! grep -q "^\.env$" "$PROJECT_ROOT/.gitignore"; then
        echo ".env" >> "$PROJECT_ROOT/.gitignore"
        echo "✅ Added .env to .gitignore"
    else
        echo "✅ .env already in .gitignore"
    fi
else
    echo ".env" > "$PROJECT_ROOT/.gitignore"
    echo "✅ Created .gitignore with .env entry"
fi

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Your environment is configured for:"
echo "  Location: $PROJECT_ROOT"
echo ""
echo "Next steps:"
echo "  1. Activate the environment:"
echo "     source activate_hypatiax.sh"
echo ""
echo "  2. Verify installation:"
echo "     python -c 'from hypatiax.protocols.experiment_protocol import ExperimentProtocol; print(\"✓ Import successful\")'"
echo ""
echo "  3. Run your experiments:"
echo "     python hypatiax/core/base_pure_llm/baseline_pure_llm.py"
echo ""
echo "NOTE: The .env file will auto-detect its location."
echo "      It works on any system - don't commit it to git."
echo ""

#!/usr/bin/env bash
# Run the unit test suite for diffusion_planner.
# Usage:
#   ./run_tests.sh               # run all tests under tests/
#   ./run_tests.sh -k augment    # filter by keyword (pytest only)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure the package is on the Python path
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

if python -m pytest --version &>/dev/null 2>&1; then
    echo "Using pytest"
    python -m pytest tests/ -v "$@"
else
    echo "pytest not found — running tests directly"
    if [[ $# -gt 0 ]]; then
        echo "Note: argument filtering is only supported with pytest. Running all tests."
    fi
    for f in tests/test_*.py; do
        echo ""
        echo "=== $f ==="
        python "$f"
    done
fi

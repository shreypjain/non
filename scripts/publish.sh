#!/bin/bash
# Script to build and publish the nons package to PyPI
# Usage: ./scripts/publish.sh [testpypi|pypi]

set -e

TARGET="${1:-pypi}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYPIRC="$PROJECT_ROOT/.pypirc"

echo "Building nons package..."
cd "$PROJECT_ROOT"
uv build

if [ "$TARGET" = "testpypi" ]; then
    echo "Publishing to TestPyPI..."
    if [ -f "$PYPIRC" ]; then
        TOKEN=$(grep -A 2 "\[testpypi\]" "$PYPIRC" | grep password | cut -d'=' -f2 | tr -d ' ')
        uv publish --token "$TOKEN" --publish-url https://test.pypi.org/legacy/
    else
        echo "Error: .pypirc not found. Please set up your PyPI credentials."
        exit 1
    fi
elif [ "$TARGET" = "pypi" ]; then
    echo "Publishing to PyPI..."
    if [ -f "$PYPIRC" ]; then
        TOKEN=$(grep -A 2 "\[pypi\]" "$PYPIRC" | grep password | cut -d'=' -f2 | tr -d ' ')
        uv publish --token "$TOKEN"
    else
        echo "Error: .pypirc not found. Please set up your PyPI credentials."
        exit 1
    fi
else
    echo "Error: Invalid target. Use 'testpypi' or 'pypi'"
    exit 1
fi

echo "Package published successfully to $TARGET!"
echo "View at: https://$([ "$TARGET" = "testpypi" ] && echo "test.")pypi.org/project/nons/"

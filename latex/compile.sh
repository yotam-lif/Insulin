#!/usr/bin/env bash
# Compile the LaTeX project for "Microscale Chemoreception in a Cylinder-Cylinder Model"
# Runs pdflatex twice to resolve all cross-references and labels.

set -e

# Move to the directory containing this script so the script works from anywhere.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

JOB="main"
OUTDIR="${SCRIPT_DIR}/compile_output"
mkdir -p "$OUTDIR"

# First pass: generates .aux with labels and references.
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$OUTDIR" "${SCRIPT_DIR}/${JOB}.tex"

# Second pass: resolves all \ref / \eqref / \cite cross-references.
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$OUTDIR" "${SCRIPT_DIR}/${JOB}.tex"

# Optional third pass to stabilize references (uncomment if needed).
# pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$OUTDIR" "${SCRIPT_DIR}/${JOB}.tex"

echo ""
echo "Build complete: ${OUTDIR}/${JOB}.pdf"

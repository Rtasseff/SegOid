#!/bin/bash
# Run LOO CV to validate multi-scale training branch produces
# equivalent results to the original model.
#
# Original baseline: Dice 0.917 ± 0.023 (6-fold LOOCV)
#
# Steps:
#   1. Generate dataset manifest (data/splits/all.csv)
#   2. Run 6-fold leave-one-out cross-validation
#   3. Print summary for comparison

set -euo pipefail

cd "$(dirname "$0")/.."

echo "========================================"
echo "  LOO CV Validation - Multi-Scale Branch"
echo "========================================"
echo ""
echo "Baseline (original): Dice 0.917 ± 0.023"
echo ""

# Step 1: Generate manifest
echo "Step 1: Generating dataset manifest..."
validate_dataset --input-dir data/working/ --output-dir data/splits/
echo ""

# Step 2: Run cross-validation
echo "Step 2: Running 6-fold LOO cross-validation..."
echo "  Config: configs/cv_config.yaml"
echo ""
run_cv --config configs/cv_config.yaml

# Step 3: Report results
echo ""
echo "========================================"
echo "  Results Summary"
echo "========================================"
echo ""
echo "Baseline (original): Dice 0.917 ± 0.023"
echo ""

# Find the most recent CV results directory
CV_DIR=$(ls -dt runs/cv_* 2>/dev/null | head -1)
if [ -n "$CV_DIR" ]; then
    echo "CV results directory: $CV_DIR"
    echo ""
    echo "--- Fold Metrics ---"
    cat "$CV_DIR/results/fold_metrics.csv" 2>/dev/null || echo "  (fold_metrics.csv not found)"
    echo ""
    echo "--- Summary ---"
    cat "$CV_DIR/results/summary.yaml" 2>/dev/null || echo "  (summary.yaml not found)"
else
    echo "ERROR: No CV results directory found in runs/"
fi

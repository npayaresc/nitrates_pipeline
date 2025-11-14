#!/bin/bash
# Analyze concentration range performance for all optimization prediction files

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_DIR="reports/range_analysis"

echo "=================================================="
echo "Analyzing Range Performance for All Models"
echo "=================================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Find all prediction CSV files from optimizations
PRED_FILES=$(find reports/ -name "predictions_*optimized*.csv" -type f | sort -r | head -10)

if [ -z "$PRED_FILES" ]; then
    echo "No optimization prediction files found!"
    exit 1
fi

echo "Found $(echo "$PRED_FILES" | wc -l) prediction files to analyze"
echo ""

# Analyze each file
for pred_file in $PRED_FILES; do
    echo "---------------------------------------------------"
    echo "Analyzing: $(basename $pred_file)"
    echo "---------------------------------------------------"

    uv run python analyze_range_performance.py \
        --predictions "$pred_file" \
        --output-dir "$OUTPUT_DIR" \
        --ranges "auto"

    echo ""
done

echo "=================================================="
echo "Analysis Complete!"
echo "=================================================="
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Generated files for each model:"
echo "  - *_range_performance.csv - Detailed metrics by range"
echo "  - *_range_performance.png - Performance visualizations"
echo "  - *_predictions_by_range.png - Scatter plots by range"
echo "  - *_range_analysis_report.txt - Text summary report"
echo ""

#!/bin/bash
# Script to run full dataset inspection on server using Poetry

echo "=================================================="
echo "CT Dataset Full Inspection Script"
echo "=================================================="

# Configuration
DATA_DIR="/mnt2/ct/RadGenome-ChestCT/dataset/train_preprocessed"
OUTPUT_DIR="inspection_results_full"
FORMAT="nifti"

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found! Please install Poetry first."
    echo "   curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

echo "Using Poetry for package management..."
echo ""

# Run inspection on all data
echo "Starting inspection of all volumes..."
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

poetry run python inspect_ct_volumes.py \
    --data_dir "$DATA_DIR" \
    --format "$FORMAT" \
    --output_dir "$OUTPUT_DIR"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Inspection completed successfully!"
    echo "Results saved to: $OUTPUT_DIR/"
    echo ""
    echo "Summary files:"
    ls -lh "$OUTPUT_DIR"/*.txt "$OUTPUT_DIR"/*.json "$OUTPUT_DIR"/*.csv 2>/dev/null
else
    echo ""
    echo "❌ Inspection failed! Check error messages above."
    exit 1
fi

echo ""
echo "Done! You can review results with:"
echo "  cat $OUTPUT_DIR/dataset_summary.txt"
echo "  less $OUTPUT_DIR/volume_statistics.csv"

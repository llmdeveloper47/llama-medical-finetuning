#!/bin/bash

# Script to evaluate fine-tuned LLaMA 3.1 8B model on medical QA dataset

set -e  # Exit on error

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path> [--compare-baseline]"
    echo "  <model_path>       - Path to the fine-tuned model"
    echo "  --compare-baseline - Compare with baseline model (optional)"
    exit 1
fi

MODEL_PATH=$1
COMPARE_BASELINE=0

if [[ "$2" == "--compare-baseline" ]]; then
    COMPARE_BASELINE=1
fi

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create directories
mkdir -p "$PROJECT_ROOT/outputs/evaluation"

# Prepare for evaluation
echo "=== Preparing for evaluation ==="

# Set output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$PROJECT_ROOT/outputs/evaluation/$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

# Logging setup
LOG_FILE="$PROJECT_ROOT/outputs/evaluation/evaluate_$TIMESTAMP.log"
echo "Logs will be saved to: $LOG_FILE"

# Check if model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found: $MODEL_PATH"
    exit 1
fi

# Check if test data exists
TEST_DATA="$PROJECT_ROOT/data/processed/test.jsonl"
if [ ! -f "$TEST_DATA" ]; then
    echo "Error: Test data not found: $TEST_DATA"
    exit 1
fi

# Build evaluation command
eval_cmd="python \"$PROJECT_ROOT/src/evaluation/evaluate.py\" \
    --model_path \"$MODEL_PATH\" \
    --data_path \"$TEST_DATA\" \
    --output_dir \"$OUTPUT_DIR\" \
    --config \"$PROJECT_ROOT/configs/train_config.json\" \
    --num_samples 100 \
    --bits 4"

if [ $COMPARE_BASELINE -eq 1 ]; then
    eval_cmd+=" --compare_baseline"
fi

# Run evaluation
echo "=== Starting evaluation ==="
echo "Model: $MODEL_PATH"
echo "Test data: $TEST_DATA"
echo "Output directory: $OUTPUT_DIR"

eval $eval_cmd 2>&1 | tee "$LOG_FILE"

# Check results
if [ -f "$OUTPUT_DIR/metrics.json" ]; then
    echo "=== Evaluation complete ==="
    echo "Results:"
    cat "$OUTPUT_DIR/metrics.json"
    echo ""
    echo "Detailed report available at: $OUTPUT_DIR/evaluation_report.md"
    
    if [ $COMPARE_BASELINE -eq 1 ] && [ -f "$OUTPUT_DIR/model_comparison.md" ]; then
        echo "Model comparison available at: $OUTPUT_DIR/model_comparison.md"
    fi
else
    echo "=== Evaluation failed ==="
    echo "Check the log file for errors: $LOG_FILE"
    exit 1
fi

# Create a symbolic link to the latest evaluation
LATEST_LINK="$PROJECT_ROOT/outputs/evaluation/latest"
rm -f "$LATEST_LINK"
ln -s "$OUTPUT_DIR" "$LATEST_LINK"
echo "Latest evaluation results linked at: $LATEST_LINK"

#!/bin/bash

# Script to download and process the HealthCareMagic-100K dataset

set -e  # Exit on error

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is required but not found. Please install Python 3.6+."
    exit 1
fi

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create directories
mkdir -p "$PROJECT_ROOT/data/raw"
mkdir -p "$PROJECT_ROOT/data/processed"

echo "=== Preparing HealthCareMagic-100K Dataset ==="

# Download dataset if not exists
echo "Step 1: Downloading dataset..."
python "$PROJECT_ROOT/src/data/processor.py" \
    --input_dir "$PROJECT_ROOT/data/raw" \
    --output_dir "$PROJECT_ROOT/data/processed" \
    --download

# Process dataset
echo "Step 2: Processing dataset into instruction format..."
python "$PROJECT_ROOT/src/data/processor.py" \
    --input_dir "$PROJECT_ROOT/data/raw" \
    --output_dir "$PROJECT_ROOT/data/processed"

# Create train/validation/test splits
echo "Step 3: Creating dataset splits..."
if [ -f "$PROJECT_ROOT/data/processed/merged_medical_instructions.jsonl" ]; then
    python -c "
from src.data.dataset import create_dataset_splits
create_dataset_splits(
    input_file='$PROJECT_ROOT/data/processed/merged_medical_instructions.jsonl',
    output_dir='$PROJECT_ROOT/data/processed',
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
)
"
    echo "Dataset splits created."
else
    echo "Warning: Merged dataset file not found. Skipping split creation."
fi

# Verify dataset
echo "Step 4: Verifying dataset..."
if [ -f "$PROJECT_ROOT/data/processed/train.jsonl" ]; then
    TRAIN_COUNT=$(wc -l < "$PROJECT_ROOT/data/processed/train.jsonl")
    echo "Train set: $TRAIN_COUNT examples"
else
    echo "Warning: Train set not found."
fi

if [ -f "$PROJECT_ROOT/data/processed/validation.jsonl" ]; then
    VAL_COUNT=$(wc -l < "$PROJECT_ROOT/data/processed/validation.jsonl")
    echo "Validation set: $VAL_COUNT examples"
else
    echo "Warning: Validation set not found."
fi

if [ -f "$PROJECT_ROOT/data/processed/test.jsonl" ]; then
    TEST_COUNT=$(wc -l < "$PROJECT_ROOT/data/processed/test.jsonl")
    echo "Test set: $TEST_COUNT examples"
else
    echo "Warning: Test set not found."
fi

echo "=== Dataset preparation complete ==="
echo "The processed data is available in: $PROJECT_ROOT/data/processed/"

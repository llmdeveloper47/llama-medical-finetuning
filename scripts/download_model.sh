#!/bin/bash

# Script to download LLaMA 3.1 model from Hugging Face

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables if .env file exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Check for Hugging Face token
if [ -z "$HF_TOKEN" ]; then
    if [ -f "$HOME/.huggingface/token" ]; then
        # Read from Hugging Face CLI token file
        HF_TOKEN=$(cat "$HOME/.huggingface/token")
    else
        echo "HF_TOKEN environment variable not set."
        echo "Please set it to access gated models like LLaMA 3.1"
        echo "You can set it in a .env file, as an environment variable, or log in with the Hugging Face CLI."
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Create models directory
mkdir -p "$PROJECT_ROOT/models"

# Set model names
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B"
OUTPUT_DIR="$PROJECT_ROOT/models"

# Build download command
DOWNLOAD_CMD="python -m huggingface_hub download $MODEL_NAME --local-dir \"$OUTPUT_DIR/$(basename "$MODEL_NAME")\" --local-dir-use-symlinks False"

if [ ! -z "$HF_TOKEN" ]; then
    DOWNLOAD_CMD+=" --token $HF_TOKEN"
fi

# Run download command
echo "=== Downloading $MODEL_NAME ==="
echo "Output directory: $OUTPUT_DIR/$(basename "$MODEL_NAME")"
echo "This may take a while depending on your internet connection..."

eval $DOWNLOAD_CMD

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "=== Download complete ==="
    echo "Model files saved to: $OUTPUT_DIR/$(basename "$MODEL_NAME")"
    
    # Update config
    CONFIG_FILE="$PROJECT_ROOT/configs/train_config.json"
    if [ -f "$CONFIG_FILE" ]; then
        echo "Updating config file with local model path..."
        TMP_CONFIG="${CONFIG_FILE}.tmp"
        
        # Use Python to update the config
        python -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)

config['model']['model_name'] = '$OUTPUT_DIR/$(basename "$MODEL_NAME")'
config['model']['use_local_model'] = True

with open('$TMP_CONFIG', 'w') as f:
    json.dump(config, f, indent=2)
"
        mv "$TMP_CONFIG" "$CONFIG_FILE"
        echo "Config updated."
    fi
else
    echo "=== Download failed ==="
    echo "Check error messages above."
    exit 1
fi

echo "You can now proceed with training using the downloaded model."

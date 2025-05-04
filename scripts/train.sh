#!/bin/bash

# Script to fine-tune LLaMA 3.1 8B on medical QA dataset

set -e  # Exit on error

# Check arguments
MODE=${1:-"local"}  # Default to local training
if [[ "$MODE" != "local" && "$MODE" != "sagemaker" ]]; then
    echo "Usage: $0 [local|sagemaker]"
    echo "  local     - Run training locally (default)"
    echo "  sagemaker - Run training on SageMaker"
    exit 1
fi

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables if .env file exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Create directories
mkdir -p "$PROJECT_ROOT/outputs"
mkdir -p "$PROJECT_ROOT/configs"
mkdir -p "$PROJECT_ROOT/logs"

# Check if DeepSpeed config exists, create if not
DEEPSPEED_CONFIG="$PROJECT_ROOT/configs/deepspeed_config.json"
if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "Creating DeepSpeed config..."
    python -c "
from src.training.config import save_deepspeed_config
save_deepspeed_config('$DEEPSPEED_CONFIG')
"
fi

# Check if training config exists, create if not
TRAIN_CONFIG="$PROJECT_ROOT/configs/train_config.json"
if [ ! -f "$TRAIN_CONFIG" ]; then
    echo "Creating training config..."
    python -c "
from src.training.config import Config
config = Config()
config.save('$TRAIN_CONFIG')
"
fi

# Prepare for training
echo "=== Preparing for $MODE training ==="

# Set output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$PROJECT_ROOT/outputs/medical-llama-$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

# Logging setup
LOG_FILE="$PROJECT_ROOT/logs/train_$TIMESTAMP.log"
echo "Logs will be saved to: $LOG_FILE"

# Set default training parameters
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
BATCH_SIZE=2
GRAD_ACCUM=4
EPOCHS=3
LR=2e-4

# Adjust based on available GPUs
if [ $NUM_GPUS -gt 1 ]; then
    echo "Using $NUM_GPUS GPUs"
    # No need to change batch size as each GPU gets the same batch
    # Instead, adjust gradient accumulation for better efficiency
    GRAD_ACCUM=$((16 / NUM_GPUS))
    if [ $GRAD_ACCUM -lt 1 ]; then
        GRAD_ACCUM=1
    fi
else
    echo "Using 1 GPU"
fi

echo "Batch size: $BATCH_SIZE, Gradient accumulation: $GRAD_ACCUM"

# Build training command
COMMON_ARGS="--config $TRAIN_CONFIG --output_dir $OUTPUT_DIR --bits 4"

if [ "$MODE" = "local" ]; then
    echo "=== Starting local training ==="
    
    # If multiple GPUs, use DeepSpeed
    if [ $NUM_GPUS -gt 1 ]; then
        echo "Using DeepSpeed for distributed training"
        
        # Run with DeepSpeed
        deepspeed --num_gpus=$NUM_GPUS \
            "$PROJECT_ROOT/src/training/train.py" \
            $COMMON_ARGS \
            --deepspeed "$DEEPSPEED_CONFIG" \
            2>&1 | tee "$LOG_FILE"
    else
        # Single GPU training
        python "$PROJECT_ROOT/src/training/train.py" \
            $COMMON_ARGS \
            2>&1 | tee "$LOG_FILE"
    fi
elif [ "$MODE" = "sagemaker" ]; then
    echo "=== Preparing for SageMaker training ==="
    
    # Check for AWS credentials
    if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
        echo "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
        exit 1
    fi
    
    # Check for SageMaker bucket
    if [ -z "$SAGEMAKER_BUCKET" ]; then
        echo "SAGEMAKER_BUCKET not set. Please set it in .env file or as an environment variable."
        exit 1
    fi
    
    # Upload data to S3
    echo "Uploading data to S3..."
    aws s3 cp "$PROJECT_ROOT/data/processed" "s3://$SAGEMAKER_BUCKET/medical-llm-finetuning/data" --recursive
    
    # Upload config to S3
    echo "Uploading configs to S3..."
    aws s3 cp "$TRAIN_CONFIG" "s3://$SAGEMAKER_BUCKET/medical-llm-finetuning/configs/train_config.json"
    aws s3 cp "$DEEPSPEED_CONFIG" "s3://$SAGEMAKER_BUCKET/medical-llm-finetuning/configs/deepspeed_config.json"
    
    # Create SageMaker training job
    echo "Starting SageMaker training job..."
    
    # Create temporary script for SageMaker
    SAGEMAKER_SCRIPT="$PROJECT_ROOT/sagemaker_train.py"
    cat > "$SAGEMAKER_SCRIPT" << 'EOF'
import os
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace

session = sagemaker.Session()
role = os.environ.get('SAGEMAKER_ROLE', sagemaker.get_execution_role())
bucket = os.environ.get('SAGEMAKER_BUCKET')

# Configure hyperparameters
hyperparameters = {
    'model_name': 'meta-llama/Meta-Llama-3.1-8B',
    'batch_size': 2,
    'gradient_accumulation_steps': 4,
    'learning_rate': 2e-4,
    'num_train_epochs': 3,
    'max_seq_length': 4096,
    'use_lora': 'True',
    'lora_r': 16,
    'lora_alpha': 32,
    'bits': 4,
}

# Configure distribution for multi-GPU training
distribution = {
    'torch_distributed': {
        'enabled': True
    }
}

# Configure metrics
metric_definitions = [
    {'Name': 'train:loss', 'Regex': 'train_loss: ([0-9\\.]+)'},
    {'Name': 'eval:loss', 'Regex': 'eval_loss: ([0-9\\.]+)'},
]

# Create estimator
huggingface_estimator = HuggingFace(
    entry_point='src/training/train.py',
    source_dir='.',
    role=role,
    instance_type='ml.p4d.24xlarge',  # 8x A100 GPUs
    instance_count=1,
    volume_size=100,  # 100 GB storage
    py_version='py38',
    pytorch_version='1.12',
    transformers_version='4.26.0',
    hyperparameters=hyperparameters,
    metric_definitions=metric_definitions,
    distribution=distribution,
    max_run=86400,  # 24 hours in seconds
)

# Start training
huggingface_estimator.fit({
    'train': f's3://{bucket}/medical-llm-finetuning/data/train.jsonl',
    'validation': f's3://{bucket}/medical-llm-finetuning/data/validation.jsonl',
    'config': f's3://{bucket}/medical-llm-finetuning/configs/train_config.json',
    'deepspeed': f's3://{bucket}/medical-llm-finetuning/configs/deepspeed_config.json',
})

# Get model artifacts and save
model_data = huggingface_estimator.model_data
print(f"Model artifacts saved to: {model_data}")
EOF
    
    # Run SageMaker script
    export SAGEMAKER_BUCKET=$SAGEMAKER_BUCKET
    python "$SAGEMAKER_SCRIPT" 2>&1 | tee "$LOG_FILE"
    
    # Clean up
    rm "$SAGEMAKER_SCRIPT"
    
    echo "SageMaker training job submitted. Check the AWS SageMaker console for progress."

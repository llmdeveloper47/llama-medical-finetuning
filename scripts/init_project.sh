#!/bin/bash

# Project initialization script
# This script sets up the entire project structure and installs dependencies

set -e  # Exit on error

echo "===== Medical LLM Fine-Tuning Project Initialization ====="
echo ""

# Get project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Create project structure
echo "Creating project directory structure..."
mkdir -p data/raw data/processed
mkdir -p models
mkdir -p configs
mkdir -p logs
mkdir -p outputs
mkdir -p outputs/evaluation
mkdir -p notebooks
mkdir -p src/{data,training,evaluation,analysis,utils}
mkdir -p scripts

echo "Creating Python package files..."
touch src/__init__.py
touch src/data/__init__.py
touch src/training/__init__.py
touch src/evaluation/__init__.py
touch src/analysis/__init__.py
touch src/utils/__init__.py

# Create empty utility files if they don't exist
for file in src/utils/logging_utils.py src/utils/aws_utils.py src/data/utils.py src/evaluation/metrics.py src/analysis/error_analysis.py; do
    if [ ! -f "$file" ]; then
        echo "Creating empty $file"
        mkdir -p "$(dirname "$file")"
        touch "$file"
    fi
done

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is required but not found. Please install Python 3.8+."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if (( $(echo "$PYTHON_VERSION < 3.8" | bc -l) )); then
    echo "Error: Python 3.8 or higher is required. Found: Python $PYTHON_VERSION"
    exit 1
fi

# Install dependencies
echo ""
echo "Setting up Python environment..."
if [ -f "pyproject.toml" ]; then
    # Try installing using pip
    echo "Installing dependencies using pip..."
    if python -m pip install -e .; then
        echo "Dependencies installed successfully."
    else
        echo "Package installation failed. Setting up Python path manually..."
        # Create a .env file with the PYTHONPATH
        echo "export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT" > .env
        echo "Please run: source .env"
        
        # Install dependencies manually
        echo "Installing required packages..."
        python -m pip install torch transformers peft accelerate datasets evaluate scipy scikit-learn matplotlib pandas tqdm wandb deepspeed sentencepiece protobuf safetensors jsonlines nltk bitsandbytes huggingface_hub boto3 sagemaker
    fi
else
    echo "Error: pyproject.toml not found. Creating it..."
    # Create pyproject.toml
    cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "medical-llm-finetuning"
version = "0.1.0"
description = "Fine-tuning LLaMA 3.1 on medical QA datasets"
readme = "README.md"
authors = [
    {name = "LLM Engineer", email = "example@email.com"}
]
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "torch>=2.1.0",
    "transformers>=4.36.0",
    "peft>=0.7.0",
    "accelerate>=0.25.0",
    "datasets>=2.14.0",
    "evaluate>=0.4.0",
    "scipy>=1.10.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "pandas>=2.0.0",
    "tqdm>=4.66.0",
    "wandb>=0.15.0",
    "deepspeed>=0.11.0",
    "sentencepiece>=0.1.99",
    "protobuf>=3.20.0",
    "safetensors>=0.4.0",
    "jsonlines>=3.1.0",
    "nltk>=3.8.0",
    "bitsandbytes>=0.41.0",
    "huggingface_hub>=0.19.0",
    "boto3>=1.28.0",
    "sagemaker>=2.173.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=6.0",
    "black>=22.0",
    "isort>=5.0",
    "flake8>=4.0",
]

[tool.black]
line-length = 100
target-version = ["py38", "py39", "py310"]

[tool.isort]
profile = "black"
line_length = 100
EOF
    echo "pyproject.toml created."
    
    # Try installing again
    echo "Installing dependencies using pip..."
    if python -m pip install -e .; then
        echo "Dependencies installed successfully."
    else
        echo "Package installation failed. Setting up Python path manually..."
        # Create a .env file with the PYTHONPATH
        echo "export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT" > .env
        echo "Please run: source .env"
        
        # Install dependencies manually
        echo "Installing required packages..."
        python -m pip install torch transformers peft accelerate datasets evaluate scipy scikit-learn matplotlib pandas tqdm wandb deepspeed sentencepiece protobuf safetensors jsonlines nltk bitsandbytes huggingface_hub boto3 sagemaker
    fi
fi

# Create default config files if they don't exist
if [ ! -f "configs/train_config.json" ]; then
    echo ""
    echo "Creating default training configuration..."
    mkdir -p configs
    cat > configs/train_config.json << 'EOF'
{
  "model": {
    "model_name": "meta-llama/Meta-Llama-3.1-8B",
    "tokenizer_name": null,
    "cache_dir": null,
    "trust_remote_code": true,
    "use_local_model": false,
    "precision": "bf16"
  },
  "lora": {
    "use_lora": true,
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "bias": "none",
    "target_modules": [
      "q_proj",
      "k_proj",
      "v_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj"
    ],
    "task_type": "CAUSAL_LM",
    "lora_path": null
  },
  "data": {
    "data_dir": "data/processed",
    "train_file": "train.jsonl",
    "validation_file": "validation.jsonl",
    "test_file": "test.jsonl",
    "max_seq_length": 4096,
    "pad_to_max_length": false,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "dataloader_num_workers": 4,
    "prefetch_factor": 2
  },
  "train": {
    "output_dir": "outputs/medical-llama-3.1-8B",
    "logging_dir": "logs",
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "num_train_epochs": 3,
    "max_steps": -1,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "gradient_accumulation_steps": 4,
    "logging_steps": 10,
    "eval_steps": 200,
    "save_steps": 500,
    "save_total_limit": 3,
    "local_rank": -1,
    "deepspeed": "configs/deepspeed_config.json",
    "resume_from_checkpoint": null,
    "save_only_model": true,
    "seed": 42,
    "fp16": false,
    "bf16": true,
    "aws_region": "us-west-2",
    "use_sagemaker": false,
    "sagemaker_bucket": null,
    "sagemaker_role": null,
    "compute_additional_metrics": true
  }
}
EOF
fi

if [ ! -f "configs/deepspeed_config.json" ]; then
    echo "Creating DeepSpeed configuration..."
    mkdir -p configs
    cat > configs/deepspeed_config.json << 'EOF'
{
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "allgather_bucket_size": 5e8,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  },
  "fp16": {
    "enabled": false
  },
  "bf16": {
    "enabled": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 1e-6,
      "warmup_max_lr": 2e-4,
      "warmup_num_steps": 100,
      "total_num_steps": 5000
    }
  },
  "steps_per_print": 10,
  "wall_clock_breakdown": false
}
EOF
fi

# Create a simple Makefile if it doesn't exist
if [ ! -f "Makefile" ]; then
    echo "Creating Makefile..."
    cat > Makefile << 'EOF'
.PHONY: setup prepare-data download-model train-local train-sagemaker evaluate clean help create-dirs

# Default target
help:
	@echo "Available commands:"
	@echo "  make create-dirs      - Create the project directory structure"
	@echo "  make setup            - Install dependencies"
	@echo "  make prepare-data     - Download and process the HealthCareMagic-100K dataset"
	@echo "  make download-model   - Download LLaMA 3.1 8B model from Hugging Face"
	@echo "  make train-local      - Run training on local machine"
	@echo "  make train-sagemaker  - Run training on SageMaker"
	@echo "  make evaluate MODEL=path/to/model [BASELINE=1] - Evaluate model (with baseline comparison if BASELINE=1)"
	@echo "  make clean            - Remove temporary files"

# Create project directory structure
create-dirs:
	mkdir -p data/raw data/processed
	mkdir -p models
	mkdir -p configs
	mkdir -p logs
	mkdir -p outputs
	mkdir -p outputs/evaluation
	mkdir -p notebooks
	mkdir -p src/{data,training,evaluation,analysis,utils}
	touch src/__init__.py
	touch src/{data,training,evaluation,analysis,utils}/__init__.py
	mkdir -p scripts
	@echo "Project directory structure created successfully"

# Setup environment
setup: create-dirs
	pip install -e .
	@echo "Python environment setup complete"

# Setup environment (alternative if pip install -e . doesn't work)
setup-alt: create-dirs
	@echo "Setting up Python environment without installation..."
	@echo "export PYTHONPATH=$${PYTHONPATH}:$(shell pwd)" > .env
	@echo "Run 'source .env' to add the project to your Python path"

# Data preparation
prepare-data:
	python -m src.data.processor --input_dir data/raw --output_dir data/processed --download

# Download model
download-model:
	python -m src.utils.download_model --model_name meta-llama/Meta-Llama-3.1-8B --output_dir models

# Local training
train-local:
	python -m src.training.train --config configs/train_config.json

# SageMaker training
train-sagemaker:
	python -m src.training.train --config configs/train_config.json --sagemaker

# Evaluation
evaluate:
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL parameter is required. Usage: make evaluate MODEL=path/to/model [BASELINE=1]"; \
		exit 1; \
	fi
	@if [ "$(BASELINE)" = "1" ]; then \
		python -m src.evaluation.evaluate --model_path "$(MODEL)" --compare_baseline; \
	else \
		python -m src.evaluation.evaluate --model_path "$(MODEL)"; \
	fi

# Clean temporary files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
EOF
fi

# Verify the installation
echo ""
echo "Verifying the setup..."
if python -c "import sys; sys.path.append('$PROJECT_ROOT'); import src.utils; print('Setup successful!')"; then
    echo ""
    echo "===== Project initialization completed successfully! ====="
    echo ""
    echo "Next steps:"
    echo "1. Download and process the dataset: make prepare-data"
    echo "2. Download the model (optional): make download-model"
    echo "3. Run training: make train-local"
    echo "4. Evaluate the model: make evaluate MODEL=path/to/model"
    echo ""
    echo "For more details, see the README.md file."
else
    echo ""
    echo "===== Project initialization partially completed! ====="
    echo ""
    echo "There might be issues with the Python path configuration."
    echo "Please run: source .env"
    echo "Then verify the setup by running: python -c \"import src.data; print('Setup successful!')\""
fi

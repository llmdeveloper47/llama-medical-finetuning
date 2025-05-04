# Medical LLM Fine-Tuning Project

This repository contains a comprehensive implementation for fine-tuning LLaMA 3.1 8B on the HealthCareMagic-100K medical question answering dataset. The project uses Hugging Face's transformers library with QLoRA (Quantized Low-Rank Adaptation) to efficiently fine-tune the model on either local hardware or Amazon SageMaker.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Complete Setup Guide](#complete-setup-guide)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Initialize the Project](#2-initialize-the-project)
  - [3. Environment Setup](#3-environment-setup)
- [Workflow Guide](#workflow-guide)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Model Download](#2-model-download)
  - [3. Training Configuration](#3-training-configuration)
  - [4. Model Training](#4-model-training)
  - [5. Model Evaluation](#5-model-evaluation)
  - [6. Results Analysis](#6-results-analysis)
- [Advanced Options](#advanced-options)
  - [Using SageMaker](#using-sagemaker)
  - [DeepSpeed Configuration](#deepspeed-configuration)
  - [Custom LoRA Settings](#custom-lora-settings)
- [Training Details](#training-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Troubleshooting](#troubleshooting)
- [Extending the Project](#extending-the-project)

## Project Overview

This project aims to fine-tune the LLaMA 3.1 8B model to become an effective medical question answering system. By fine-tuning on the HealthCareMagic-100K dataset, which contains real patient questions and doctor responses, the model learns to generate accurate, helpful, and structured medical advice.

Key features:

- **QLoRA Fine-tuning**: Efficient parameter-efficient fine-tuning with 4-bit quantization
- **Medical Domain Adaptation**: Specialized training on medical consultations
- **Structured Response Format**: Training to generate diagnosis and treatment plans
- **Comprehensive Evaluation**: Domain-specific metrics to assess medical response quality
- **SageMaker Integration**: Scalable training on AWS infrastructure
- **DeepSpeed Support**: Optimized distributed training on multiple GPUs

## Requirements

- Python 3.8+
- PyTorch 2.1.0+
- CUDA-compatible GPU(s) with at least 16GB VRAM for local training
- AWS account (for SageMaker training)
- Hugging Face account with LLaMA 3.1 8B model access

## Project Structure

```
medical-llm-finetuning/
├── README.md                   # Project documentation
├── requirements.txt            # Dependencies
├── pyproject.toml              # Package information
├── Makefile                    # Command shortcuts
├── data/                       # Dataset storage
│   ├── raw/                    # Raw dataset files
│   └── processed/              # Processed dataset files
├── configs/                    # Configuration files
├── scripts/                    # Utility scripts
├── src/                        # Source code
│   ├── data/                   # Dataset handling
│   │   ├── dataset.py          # Dataset classes
│   │   ├── processor.py        # Data processing
│   │   └── utils.py            # Data utilities
│   ├── training/               # Training code
│   │   ├── config.py           # Configuration
│   │   ├── trainer.py          # Custom trainer
│   │   └── train.py            # Training script
│   ├── evaluation/             # Evaluation
│   │   ├── evaluate.py         # Evaluation script
│   │   └── metrics.py          # Medical metrics
│   ├── analysis/               # Analysis
│   │   └── error_analysis.py   # Error analysis
│   └── utils/                  # Utilities
│       ├── aws_utils.py        # AWS utilities
│       └── logging_utils.py    # Logging utilities
└── notebooks/                  # Analysis notebooks
    ├── data_exploration.ipynb  # Dataset exploration
    └── results_analysis.ipynb  # Results analysis
```

## Complete Setup Guide

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/your-username/medical-llm-finetuning.git
cd medical-llm-finetuning
```

### 2. Initialize the Project

The easiest way to set up the complete project structure is to use the provided initialization script:

```bash
# Make the script executable
chmod +x scripts/init_project.sh

# Run the initialization script
./scripts/init_project.sh
```

This script will:
1. Create all necessary directories and files
2. Generate default configuration files
3. Create a Makefile for easy command execution
4. Install dependencies
5. Set up Python paths

Alternatively, you can create the directory structure manually:

```bash
make create-dirs
```

### 3. Environment Setup

If the `init_project.sh` script did not complete the dependency installation, you can set up your environment manually:

```bash
# Create and activate a conda environment
conda create -n medical-llm python=3.10
conda activate medical-llm

# Option A: Install as a package (recommended)
pip install -e .

# Option B: Manual dependency installation
pip install torch>=2.1.0 transformers>=4.36.0 peft>=0.7.0 accelerate>=0.25.0 \
  datasets>=2.14.0 evaluate>=0.4.0 scipy>=1.10.0 scikit-learn>=1.3.0 \
  matplotlib>=3.7.0 pandas>=2.0.0 tqdm>=4.66.0 wandb>=0.15.0 \
  deepspeed>=0.11.0 sentencepiece>=0.1.99 protobuf>=3.20.0 \
  safetensors>=0.4.0 jsonlines>=3.1.0 nltk>=3.8.0 bitsandbytes>=0.41.0 \
  huggingface_hub>=0.19.0 boto3>=1.28.0 sagemaker>=2.173.0

# Set up Python path if needed
echo "export PYTHONPATH=\$PYTHONPATH:$(pwd)" > .env
source .env
```

Verify the installation:

```bash
python -c "import src.data; import src.training; import src.evaluation; print('Setup successful!')"
```

## Workflow Guide

### 1. Data Preparation

To download and process the HealthCareMagic-100K dataset:

```bash
# Using the Makefile
make prepare-data

# Or using the direct command
python -m src.data.processor --input_dir data/raw --output_dir data/processed --download
```

This command performs the following steps:

1. Downloads the HealthCareMagic-100K dataset from HuggingFace
2. Processes raw data into instruction format with the following structure:
   ```json
   {
     "instruction": "Based on the patient's description, provide a diagnosis and treatment plan.",
     "input": "I have been experiencing chest pain and shortness of breath...",
     "output": "Diagnosis: Anxiety attack. Treatment: Deep breathing exercises..."
   }
   ```
3. Formats doctor responses with diagnosis and treatment sections
4. Splits the data into train (80%), validation (10%), and test (10%) sets
5. Saves processed data to `data/processed/` in JSONL format

You can check the processed data files:

```bash
# View the number of examples in each split
wc -l data/processed/*.jsonl

# Peek at the first example
head -n 1 data/processed/train.jsonl | python -m json.tool
```

### 2. Model Download

To download the LLaMA 3.1 8B model:

```bash
# Using the Makefile
make download-model

# Or using the direct command
python -m src.utils.download_model --model_name meta-llama/Meta-Llama-3.1-8B --output_dir models
```

Important notes for model download:

1. You need to have access to the LLaMA 3.1 model on HuggingFace
2. Set your HuggingFace access token:
   ```bash
   export HF_TOKEN=your_huggingface_token
   # Or save it to your environment
   echo "export HF_TOKEN=your_huggingface_token" >> ~/.bashrc
   ```
3. This step is optional if:
   - Using SageMaker (which can access the model directly)
   - Already have the model downloaded somewhere else (update the config file in that case)

Verify the model download:

```bash
# Check if model files were downloaded
ls -la models/Meta-Llama-3.1-8B/
```

### 3. Training Configuration

Customize the training configuration by editing `configs/train_config.json`:

```bash
# Edit the configuration file
nano configs/train_config.json
```

Key configuration sections:

#### Model Configuration

```json
"model": {
  "model_name": "meta-llama/Meta-Llama-3.1-8B",   // Use "models/Meta-Llama-3.1-8B" for local model
  "tokenizer_name": null,                         // null means use model_name
  "cache_dir": null,                              // Optional cache directory
  "trust_remote_code": true,
  "use_local_model": true,                        // Set to true for local model
  "precision": "bf16"                             // Options: fp32, fp16, bf16
}
```

#### LoRA Configuration

```json
"lora": {
  "use_lora": true,                               // Enable LoRA fine-tuning
  "r": 16,                                        // LoRA rank (higher = more parameters)
  "alpha": 32,                                    // LoRA alpha
  "dropout": 0.05,                                // LoRA dropout rate
  "bias": "none",                                 // Bias training mode
  "target_modules": [                             // Modules to apply LoRA to
    "q_proj", "k_proj", "v_proj", "o_proj", 
    "gate_proj", "up_proj", "down_proj"
  ]
}
```

#### Data Configuration

```json
"data": {
  "data_dir": "data/processed",
  "max_seq_length": 4096,                         // Max token length
  "per_device_train_batch_size": 2,               // Batch size per GPU
  "per_device_eval_batch_size": 2
}
```

#### Training Configuration

```json
"train": {
  "output_dir": "outputs/medical-llama-3.1-8B",
  "learning_rate": 2e-4,                          // Learning rate
  "num_train_epochs": 3,                          // Number of epochs
  "warmup_ratio": 0.1,                            // Portion of steps for warmup
  "gradient_accumulation_steps": 4,               // Gradient accumulation
  "logging_steps": 10,
  "eval_steps": 200,                              // Evaluation frequency
  "save_steps": 500,                              // Checkpoint frequency
  "deepspeed": "configs/deepspeed_config.json",   // DeepSpeed config for multi-GPU
  "fp16": false,
  "bf16": true                                    // Enable bf16 precision
}
```

### 4. Model Training

#### Local Training

For training on local machine:

```bash
# Using the Makefile
make train-local

# Or using the direct command
python -m src.training.train --config configs/train_config.json
```

Additional training options:

```bash
# Resume from a checkpoint
python -m src.training.train --config configs/train_config.json --resume_from_checkpoint outputs/medical-llama-3.1-8B/checkpoint-1000

# Use specific bits for quantization (4 or 8)
python -m src.training.train --config configs/train_config.json --bits 4

# Disable LoRA for full fine-tuning (requires more memory)
python -m src.training.train --config configs/train_config.json --disable_lora

# Use specific output directory
python -m src.training.train --config configs/train_config.json --output_dir outputs/custom-name
```

Training with multiple GPUs using DeepSpeed:

```bash
# Using DeepSpeed
deepspeed --num_gpus=8 src/training/train.py --config configs/train_config.json --deepspeed configs/deepspeed_config.json

# Or via torchrun
torchrun --nproc_per_node=8 src/training/train.py --config configs/train_config.json
```

Monitor training progress:

```bash
# Check training logs
tail -f logs/training_*.log

# Start TensorBoard (if enabled)
tensorboard --logdir outputs/medical-llama-3.1-8B/runs
```

#### SageMaker Training

For training on AWS SageMaker:

```bash
# Set up AWS credentials
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2

# Update SageMaker configuration in train_config.json
# "train": {
#   ...
#   "use_sagemaker": true,
#   "sagemaker_bucket": "your-s3-bucket",
#   "sagemaker_role": "your-sagemaker-role-arn"
# }

# Using the Makefile
make train-sagemaker

# Or using the direct command
python -m src.training.train --config configs/train_config.json --sagemaker
```

### 5. Model Evaluation

To evaluate the fine-tuned model:

```bash
# Using the Makefile (compare with baseline)
make evaluate MODEL=outputs/medical-llama-3.1-8B/final BASELINE=1

# Using the Makefile (evaluate fine-tuned model only)
make evaluate MODEL=outputs/medical-llama-3.1-8B/final

# Or using the direct command
python -m src.evaluation.evaluate --model_path outputs/medical-llama-3.1-8B/final --compare_baseline
```

Additional evaluation options:

```bash
# Specify test data path
python -m src.evaluation.evaluate --model_path outputs/medical-llama-3.1-8B/final --data_path data/processed/test.jsonl

# Evaluate with specific sampling parameters
python -m src.evaluation.evaluate --model_path outputs/medical-llama-3.1-8B/final --temperature 0.2 --top_p 0.95

# Limit the number of samples for faster evaluation
python -m src.evaluation.evaluate --model_path outputs/medical-llama-3.1-8B/final --num_samples 50
```

View evaluation results:

```bash
# View evaluation report
cat outputs/evaluation/latest/evaluation_report.md

# View model comparison (if baseline comparison was run)
cat outputs/evaluation/latest/model_comparison.md
```

### 6. Results Analysis

For detailed error analysis:

```bash
# Run error analysis
python -m src.analysis.error_analysis --predictions_file outputs/evaluation/latest/predictions.jsonl --output_dir outputs/error_analysis
```

Using Jupyter notebooks for interactive analysis:

```bash
# Start Jupyter notebook
jupyter notebook notebooks/results_analysis.ipynb
```

The notebooks provide:
1. Detailed performance metrics visualization
2. Comparison between fine-tuned and baseline models
3. Analysis by medical specialty
4. Error type distribution
5. Sample prediction analysis

## Advanced Options

### Using SageMaker

For SageMaker training, update your configuration:

1. Setup AWS credentials
2. Update train_config.json with SageMaker settings:
   ```json
   "train": {
     "use_sagemaker": true,
     "sagemaker_bucket": "your-s3-bucket",
     "sagemaker_role": "your-sagemaker-role-arn",
     "aws_region": "us-west-2"
   }
   ```
3. Run training with SageMaker:
   ```bash
   python -m src.training.train --config configs/train_config.json --sagemaker
   ```

### DeepSpeed Configuration

Customize DeepSpeed settings in `configs/deepspeed_config.json`:

```json
{
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 4,
  "zero_optimization": {
    "stage": 3,                 // ZeRO stage (higher = more memory efficient)
    "contiguous_gradients": true,
    "overlap_comm": true        // Overlap communication and computation
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
  }
}
```

### Custom LoRA Settings

Fine-tune LoRA parameters for better performance:

```json
"lora": {
  "use_lora": true,
  "r": 16,                      // Try 8, 16, 32 (higher = more parameters)
  "alpha": 32,                  // Typically 2x of r
  "dropout": 0.05,              // Try 0.05, 0.1
  "target_modules": [           // Add/remove modules based on performance
    "q_proj", "k_proj", "v_proj", "o_proj", 
    "gate_proj", "up_proj", "down_proj"
  ]
}
```

## Training Details

### QLoRA Implementation

The project uses QLoRA for efficient fine-tuning:

1. **4-bit Quantization**: The base model is loaded in 4-bit precision to reduce memory usage.
2. **Low-Rank Adapters**: Small, trainable matrices are added to attention layers.
3. **Target Modules**: Adapters are applied to key attention and MLP components.

This approach allows fine-tuning on consumer GPUs and reduces training costs while maintaining most of the performance of full fine-tuning.

### Hugging Face Integration

The project uses Hugging Face's libraries for training:

1. **Transformers**: For model loading and tokenization
2. **PEFT**: For QLoRA implementation
3. **Accelerate**: For multi-GPU training
4. **BitsAndBytes**: For 4-bit quantization
5. **Datasets**: For data loading and processing

### DeepSpeed Integration

For multi-GPU training, DeepSpeed is used with the following optimizations:

1. **ZeRO Stage 3**: Model states partitioned across GPUs
2. **BF16 Precision**: Mixed precision training with bfloat16
3. **Gradient Accumulation**: Simulates larger batch sizes
4. **Gradient Checkpointing**: Trades compute for memory efficiency

## Evaluation Metrics

The evaluation includes both general and medical-specific metrics:

### General Metrics
- **ROUGE**: Measures n-gram overlap
- **BLEU**: Measures translation quality
- **BERTScore**: Measures semantic similarity using contextualized embeddings

### Medical Metrics
- **Diagnosis Presence Rate**: Percentage of responses with clear diagnoses
- **Treatment Presence Rate**: Percentage of responses with treatment plans
- **Structure Preservation Rate**: Consistency of response structure
- **Medical Terminology Ratio**: Usage of medical terminology compared to reference

## Troubleshooting

### Out of Memory Errors
- Reduce batch size in `configs/train_config.json`
- Increase gradient accumulation steps
- Use 4-bit precision instead of 8-bit
- Try a smaller model or reduce context length

```bash
# Try with smaller batch size and more gradient accumulation
python -m src.training.train --config configs/train_config.json \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8
```

### SageMaker Issues
- Check IAM permissions for S3 and SageMaker
- Ensure your HF_TOKEN is set correctly for model access
- Check CloudWatch logs for detailed error information

```bash
# Check SageMaker logs
aws logs get-log-events --log-group-name /aws/sagemaker/TrainingJobs --log-stream-name your-training-job-name
```

### Module Import Errors
- Make sure you've run `make create-dirs` to create all necessary files
- If using the alternative setup, ensure you ran `source .env`
- Check that the directory structure matches what the imports expect

```bash
# Fix Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### CUDA Out of Memory
- If you get CUDA OOM errors, try:
  ```bash
  # Use gradient checkpointing and more aggressive quantization
  python -m src.training.train --config configs/train_config.json \
    --gradient_checkpointing \
    --bits 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16
  ```

### Common Error: Can't find dataset
- Make sure the processed dataset exists:
  ```bash
  # Run data processing
  python -m src.data.processor --input_dir data/raw --output_dir data/processed --download
  ```

### Common Error: Permission denied for script execution
- Make scripts executable:
  ```bash
  chmod +x scripts/*.sh
  ```

## Extending the Project

### Supporting Other Models
- Update `config.py` with new model architecture details
- Modify tokenizer handling in `dataset.py` if needed
- Adjust target modules for LoRA in configuration

### Adding New Datasets
- Create a new processor in `src/data/processor.py`
- Implement dataset-specific formatting logic
- Update the dataset splits creation

### Custom Evaluation Metrics
- Add new metrics in `src/evaluation/metrics.py`
- Implement domain-specific logic as needed
- Update reporting templates in `evaluate.py`

### Using Larger Models
For larger models like LLaMA 3.1 70B:
1. Update model name in config
2. Increase hardware specs or use distributed training
3. Modify LoRA settings for more efficient training

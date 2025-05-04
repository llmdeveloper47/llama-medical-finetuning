# Medical LLM Fine-Tuning Project

This repository contains a comprehensive implementation for fine-tuning LLaMA 3.1 8B on the HealthCareMagic-100K medical question answering dataset. The project uses Hugging Face's transformers library with QLoRA (Quantized Low-Rank Adaptation) to efficiently fine-tune the model on either local hardware or Amazon SageMaker.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Step-by-Step Usage Guide](#step-by-step-usage-guide)
  - [0. Create Project Structure](#0-create-project-structure)
  - [1. Setup Environment](#1-setup-environment)
  - [2. Prepare Dataset](#2-prepare-dataset)
  - [3. Download Model (Optional)](#3-download-model-optional)
  - [4. Configure Training](#4-configure-training)
  - [5. Run Training](#5-run-training)
  - [6. Evaluate Model](#6-evaluate-model)
  - [7. Analyze Results](#7-analyze-results)
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
- CUDA-compatible GPU(s) with at least 16GB VRAM (for local training)
- AWS account (for SageMaker training)
- Hugging Face account with access to LLaMA 3.1 8B model (or locally downloaded model)

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

## Installation

### Clone the repository:

```bash
git clone https://github.com/your-username/medical-llm-finetuning.git
cd medical-llm-finetuning
```

### Create project structure:

```bash
# Create all necessary directories and files
make create-dirs
```

### Install dependencies:

```bash
# Create and activate a conda environment
conda create -n medical-llm python=3.10
conda activate medical-llm

# Install dependencies
pip install -e .
```

If you encounter issues with the pip installation, you can set up the Python path manually:

```bash
# Create and activate a conda environment
conda create -n medical-llm python=3.10
conda activate medical-llm

# Install dependencies without package installation
pip install torch transformers peft accelerate datasets evaluate scipy scikit-learn matplotlib pandas tqdm wandb deepspeed sentencepiece protobuf safetensors jsonlines nltk bitsandbytes huggingface_hub boto3 sagemaker

# Set up Python path
make setup-alt
source .env
```

## Step-by-Step Usage Guide

### 0. Create Project Structure

Before beginning, you need to create the project directory structure:

```bash
# Create all required directories and files
make create-dirs
```

**What happens**:
- The script creates all necessary directories for the project:
  - `data/raw/`: For storing the raw dataset
  - `data/processed/`: For storing the processed dataset
  - `models/`: For downloaded model files
  - `configs/`: For configuration files
  - `logs/`: For training and evaluation logs
  - `outputs/`: For trained models
  - `src/`: For source code with subdirectories
  - `notebooks/`: For Jupyter notebooks
  - `scripts/`: For shell scripts
- It also creates necessary `__init__.py` files to make the Python package structure work properly
- Creates empty skeleton files for utilities and modules
- This ensures all paths referenced in the code exist before running any other commands

### 1. Setup Environment

```bash
# Install dependencies
make setup
```

**What happens**: 
- The dependencies are installed using pip's editable mode
- This includes PyTorch, transformers, PEFT, Accelerate, and other libraries
- The project is added to your Python path

### 2. Prepare Dataset

```bash
# Download and process the HealthCareMagic-100K dataset
make prepare-data
```

**What happens**:
- The script downloads the HealthCareMagic-100K dataset from Hugging Face
- The dataset is processed into an instruction format suitable for fine-tuning
- Patient questions and doctor answers are formatted with consistent structure
- The data is split into train/validation/test sets (80%/10%/10%)
- Processed data is saved in JSONL format in `data/processed/`
- Metadata about the dataset (patient age, gender) is extracted where available
- Doctor responses are structured to include diagnosis and treatment sections where possible

### 3. Download Model (Optional)

```bash
# Download LLaMA 3.1 8B model from Hugging Face
make download-model
```

**What happens**:
- The script uses Hugging Face Hub API to download the LLaMA 3.1 8B model
- Model files are saved to `models/Meta-Llama-3.1-8B/`
- Your Hugging Face authentication token is used to access the model (requires prior access)
- The config file is updated to use the local model path
- This step is optional if you're using SageMaker or if the model is already accessible through Hugging Face

### 4. Configure Training

Edit the configuration file at `configs/train_config.json` to customize training parameters:

```bash
# View and edit the configuration
nano configs/train_config.json
```

Key parameters to consider:
- `model.model_name`: Path or name of the base model
- `lora.r`: LoRA rank (higher = more parameters, better quality)
- `data.max_seq_length`: Maximum sequence length for tokenization
- `data.per_device_train_batch_size`: Batch size per GPU
- `train.learning_rate`: Learning rate for optimization
- `train.num_train_epochs`: Number of training epochs

**What happens**:
- The configuration file contains all parameters for model, data, and training
- Changes to this file control the behavior of training and evaluation
- For SageMaker training, you can also set `train.use_sagemaker` to `true` and provide a bucket name

### 5. Run Training

#### Local Training
```bash
# Train on local machine
make train-local
```

**What happens**:
- The script loads the LLaMA 3.1 8B model with 4-bit quantization to save memory
- QLoRA adapters are initialized with the specified rank and target modules
- Training dataset is loaded and processed with the model's tokenizer
- If multiple GPUs are available, DeepSpeed is used for distributed training
- The model is trained for the specified number of epochs with gradient accumulation
- Training metrics (loss, learning rate) are logged to the console and tensorboard
- Model checkpoints are saved periodically to the output directory
- At the end, a final model is saved with merged LoRA weights if desired

#### SageMaker Training
```bash
# Train on SageMaker
make train-sagemaker
```

**What happens**:
- The script uploads datasets to S3
- It configures a SageMaker training job using an ml.p4d.24xlarge instance (8 A100 GPUs)
- The training job runs the same training code, but in a distributed AWS environment
- Training progress can be monitored in the AWS SageMaker console
- Model artifacts are saved to S3 upon completion
- This is ideal for training large models or when local GPU resources are limited

### 6. Evaluate Model

```bash
# Evaluate the fine-tuned model
make evaluate MODEL=outputs/medical-llama-3.1-8B/final BASELINE=1
```

**What happens**:
- The script loads both the fine-tuned model and (if BASELINE=1) the base model
- Both models generate responses to the test set questions
- Standard NLP metrics are computed (ROUGE, BLEU, BERTScore)
- Medical-specific metrics are computed:
  - Diagnosis accuracy
  - Treatment plan inclusion
  - Structure preservation
  - Medical terminology usage
- Results are saved to `outputs/evaluation/`
- If BASELINE=1, a comparison report is generated to show improvements

### 7. Analyze Results

```bash
# View evaluation results
cat outputs/evaluation/latest/evaluation_report.md

# If baseline comparison was run
cat outputs/evaluation/latest/model_comparison.md

# Run Jupyter notebook for detailed analysis
jupyter notebook notebooks/results_analysis.ipynb
```

**What happens**:
- The evaluation reports show detailed metrics and example predictions
- You can see how well the model performs on different types of medical questions
- The comparison report shows the improvement over the base model
- The notebook provides interactive visualizations and deeper analysis of the results
- Error analysis identifies areas where the model can be improved

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

### SageMaker Issues
- Check IAM permissions for S3 and SageMaker
- Ensure your HF_TOKEN is set correctly for model access
- Check CloudWatch logs for detailed error information

### Module Import Errors
- Make sure you've run `make create-dirs` to create all necessary files
- If using the alternative setup, ensure you ran `source .env`
- Check that the directory structure matches what the imports expect

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

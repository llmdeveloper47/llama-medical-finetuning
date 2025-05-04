#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for fine-tuning LLaMA 3.1 8B on medical QA dataset using Hugging Face Transformers.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from peft import (
    LoraConfig,
    TaskType,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import WandbCallback
from transformers.trainer_utils import get_last_checkpoint

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.dataset import MedicalInstructionDataset
from src.training.config import Config
from src.training.trainer import MedicalQATrainer
from src.utils.logging_utils import get_logger, setup_file_logging, log_system_info
from src.utils.aws_utils import upload_to_s3


logger = get_logger("training")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA on medical QA dataset")
    parser.add_argument(
        "--config", type=str, default="configs/train_config.json", help="Path to training configuration"
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
    )
    parser.add_argument(
        "--deepspeed", type=str, default=None, help="Path to deepspeed config file"
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Override output directory from config"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--no_wandb", action="store_true", help="Disable wandb logging"
    )
    parser.add_argument(
        "--bits", type=int, default=4, help="Number of bits for quantization (4 or 8)"
    )
    parser.add_argument(
        "--disable_lora", action="store_true", help="Disable LoRA for full fine-tuning"
    )
    parser.add_argument(
        "--compute_metrics", action="store_true", help="Compute additional metrics during evaluation"
    )
    
    args = parser.parse_args()
    return args


def load_config(config_path: str) -> Config:
    """Load configuration from file."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}, using default config")
        config = Config()
        
        # Save default config
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        config.save(config_path)
    else:
        logger.info(f"Loading config from {config_path}")
        config = Config.load(config_path)
    
    return config


def setup_logging(config: Config, args: argparse.Namespace) -> None:
    """Setup logging for training."""
    log_dir = os.path.join(config.train.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup file logging
    setup_file_logging(logger, log_dir, "training")
    
    # Log system information
    log_system_info(logger)
    
    # Log configuration
    logger.info(f"Configuration: {json.dumps(config.__dict__, indent=2, default=str)}")
    logger.info(f"Arguments: {args}")


def setup_wandb(config: Config, args: argparse.Namespace) -> bool:
    """Setup Weights & Biases logging."""
    if args.no_wandb or not os.environ.get("WANDB_API_KEY"):
        logger.info("Weights & Biases logging disabled")
        return False
    
    try:
        import wandb
        
        run_name = f"medical-llama-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "medical-llm-finetuning"),
            name=run_name,
            config={
                "model_name": config.model.model_name,
                "lora": {
                    "enabled": config.lora.use_lora and not args.disable_lora,
                    "r": config.lora.r,
                    "alpha": config.lora.alpha,
                    "dropout": config.lora.dropout,
                    "target_modules": config.lora.target_modules,
                },
                "training": {
                    "learning_rate": config.train.learning_rate,
                    "batch_size": config.data.per_device_train_batch_size,
                    "gradient_accumulation_steps": config.train.gradient_accumulation_steps,
                    "epochs": config.train.num_train_epochs,
                    "warmup_ratio": config.train.warmup_ratio,
                    "max_seq_length": config.data.max_seq_length,
                    "bits": args.bits,
                },
                "dataset": "HealthCareMagic-100K",
            },
            tags=["llama-3.1", "medical", "qlora", "healthcaremagic"],
        )
        
        logger.info(f"Weights & Biases logging enabled: {run_name}")
        return True
    
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        return False


def prepare_model_and_tokenizer(config: Config, args: argparse.Namespace) -> Tuple:
    """
    Load and prepare the model and tokenizer for training.
    
    Args:
        config: Training configuration
        args: Command line arguments
        
    Returns:
        Tuple of (model, tokenizer, peft_config)
    """
    logger.info(f"Loading tokenizer: {config.model.tokenizer_name or config.model.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name or config.model.model_name,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    # Configure model loading
    logger.info(f"Loading model: {config.model.model_name}")
    
    # Quantization configuration
    if args.bits in (4, 8):
        logger.info(f"Using {args.bits}-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        logger.info("Using full precision (no quantization)")
        quantization_config = None
    
    # Model loading
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    
    # For QLoRA training, prepare the model
    peft_config = None
    if config.lora.use_lora and not args.disable_lora:
        logger.info("Preparing model for LoRA fine-tuning")
        
        # Prepare model for k-bit training if using quantization
        if args.bits in (4, 8):
            model = prepare_model_for_kbit_training(
                model, 
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        
        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        
        # Configure LoRA
        peft_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias=config.lora.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=config.lora.target_modules,
        )
        
        # Apply LoRA
        model = get_peft_model(model, peft_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
    else:
        logger.info("Using full fine-tuning (no LoRA)")
    
    return model, tokenizer, peft_config


def prepare_datasets(config: Config, tokenizer) -> Tuple:
    """
    Prepare train and validation datasets.
    
    Args:
        config: Training configuration
        tokenizer: Tokenizer for processing data
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    logger.info("Preparing datasets")
    
    # Create train dataset
    train_dataset = MedicalInstructionDataset(
        data_path=config.data.data_dir,
        tokenizer_name_or_path=tokenizer,
        max_length=config.data.max_seq_length,
        split="train",
    )
    
    # Create validation dataset
    eval_dataset = MedicalInstructionDataset(
        data_path=config.data.data_dir,
        tokenizer_name_or_path=tokenizer,
        max_length=config.data.max_seq_length,
        split="validation",
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def create_training_args(config: Config, args: argparse.Namespace) -> TrainingArguments:
    """
    Create HuggingFace training arguments.
    
    Args:
        config: Training configuration
        args: Command line arguments
        
    Returns:
        TrainingArguments object
    """
    # Override output directory if specified
    output_dir = args.output_dir or config.train.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Get world size for distributed training
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Calculate effective batch size
    per_device_batch_size = config.data.per_device_train_batch_size
    gradient_accumulation_steps = config.train.gradient_accumulation_steps
    effective_batch_size = per_device_batch_size * gradient_accumulation_steps * world_size
    
    logger.info(f"Effective batch size: {effective_batch_size}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=config.data.per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        max_grad_norm=config.train.max_grad_norm,
        num_train_epochs=config.train.num_train_epochs,
        max_steps=config.train.max_steps if config.train.max_steps > 0 else -1,
        warmup_ratio=config.train.warmup_ratio,
        lr_scheduler_type=config.train.lr_scheduler_type,
        logging_steps=config.train.logging_steps,
        eval_steps=config.train.eval_steps,
        save_steps=config.train.save_steps,
        save_total_limit=config.train.save_total_limit,
        report_to="wandb" if not args.no_wandb else "none",
        ddp_find_unused_parameters=False,
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
        remove_unused_columns=False,
        seed=args.seed or config.train.seed,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )
    
    return training_args


def train(config: Config, args: argparse.Namespace) -> str:
    """
    Train the model.
    
    Args:
        config: Training configuration
        args: Command line arguments
        
    Returns:
        Path to the trained model
    """
    # Set seed for reproducibility
    set_seed(args.seed or config.train.seed)
    
    # Setup logging
    setup_logging(config, args)
    
    # Setup wandb
    wandb_enabled = setup_wandb(config, args)
    
    # Load model and tokenizer
    model, tokenizer, peft_config = prepare_model_and_tokenizer(config, args)
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer)
    
    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
    )
    
    # Create training arguments
    training_args = create_training_args(config, args)
    
    # Initialize Trainer
    trainer = MedicalQATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,  # We'll compute metrics manually in evaluation
    )
    
    # Resume from checkpoint if specified
    checkpoint = args.resume_from_checkpoint
    if checkpoint is None and training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    
    # Check if there's a checkpoint to resume from
    if checkpoint is None:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
            logger.info(f"Found checkpoint: {checkpoint}")
    
    # Train the model
    logger.info("Starting training")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save model
    logger.info("Saving model")
    trainer.save_model()
    trainer.save_state()
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Upload to S3 if using SageMaker
    if config.train.use_sagemaker and config.train.sagemaker_bucket:
        logger.info("Uploading model to S3")
        s3_path = f"s3://{config.train.sagemaker_bucket}/medical-llm-finetuning/models/{Path(training_args.output_dir).name}"
        upload_to_s3(training_args.output_dir, s3_path)
        logger.info(f"Model uploaded to {s3_path}")
    
    return training_args.output_dir


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update config with arguments
    if args.deepspeed:
        config.train.deepspeed = args.deepspeed
    if args.resume_from_checkpoint:
        config.train.resume_from_checkpoint = args.resume_from_checkpoint
    if args.output_dir:
        config.train.output_dir = args.output_dir
    if args.disable_lora:
        config.lora.use_lora = False
    if args.compute_metrics:
        config.train.compute_additional_metrics = True
    
    # Start training
    output_dir = train(config, args)
    
    logger.info(f"Training completed. Model saved to {output_dir}")
    
    # Return success
    return 0


if __name__ == "__main__":
    sys.exit(main())

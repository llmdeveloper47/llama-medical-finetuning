#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration settings for fine-tuning LLaMA 3.1 8B model on medical QA data.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
from transformers import TrainingArguments


@dataclass
class ModelConfig:
    """Model configuration."""
    
    # Base model settings
    model_name: str = "meta-llama/Meta-Llama-3.1-8B"
    tokenizer_name: Optional[str] = None  # Uses model_name if None
    cache_dir: Optional[str] = None
    trust_remote_code: bool = True
    use_local_model: bool = False  # Flag for using local model
    
    # Precision settings
    precision: str = "bf16"  # Options: "fp32", "fp16", "bf16", "int8", "int4"
    
    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name


@dataclass
class LoraConfig:
    """Configuration for LoRA fine-tuning."""
    
    use_lora: bool = True
    r: int = 16  # LoRA rank
    alpha: int = 32  # LoRA alpha
    dropout: float = 0.05
    bias: str = "none"  # Options: "none", "all", "lora_only"
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    task_type: str = "CAUSAL_LM"
    lora_path: Optional[str] = None  # Path to existing LoRA weights for continued training


@dataclass
class DataConfig:
    """Data configuration."""
    
    # Data paths
    data_dir: str = "data/processed"
    train_file: str = "train.jsonl"
    validation_file: str = "validation.jsonl"
    test_file: str = "test.jsonl"
    
    # Data processing
    max_seq_length: int = 4096
    pad_to_max_length: bool = False
    
    # DataLoader settings
    per_device_train_batch_size: int = 2  # Smaller batch size for LoRA
    per_device_eval_batch_size: int = 2
    dataloader_num_workers: int = 4
    prefetch_factor: int = 2
    
    def get_train_path(self) -> str:
        """Get path to training data."""
        return os.path.join(self.data_dir, self.train_file)
    
    def get_validation_path(self) -> str:
        """Get path to validation data."""
        return os.path.join(self.data_dir, self.validation_file)
    
    def get_test_path(self) -> str:
        """Get path to test data."""
        return os.path.join(self.data_dir, self.test_file)


@dataclass
class TrainConfig:
    """Training configuration."""
    
    # Output directories
    output_dir: str = "outputs/medical-llama-3.1-8B"
    logging_dir: str = "logs"
    
    # Training hyperparameters
    learning_rate: float = 2e-4  # Higher learning rate for LoRA
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Training schedule
    num_train_epochs: int = 3
    max_steps: int = -1  # Overrides num_train_epochs if > 0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"  # Options: "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    
    # Regularization
    gradient_accumulation_steps: int = 4
    
    # Logging & Evaluation
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Distributed training
    local_rank: int = -1
    deepspeed: Optional[str] = "configs/deepspeed_config.json"
    
    # Checkpointing
    resume_from_checkpoint: Optional[str] = None
    save_only_model: bool = True  # If True, only save model weights, not optimizer state
    
    # Miscellaneous
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    
    # SageMaker specific
    aws_region: str = "us-west-2"
    use_sagemaker: bool = False
    sagemaker_bucket: Optional[str] = None
    sagemaker_role: Optional[str] = None
    
    # Additional options
    compute_additional_metrics: bool = False  # For medical metrics
    
    def to_transformers_training_args(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        return TrainingArguments(
            output_dir=self.output_dir,
            logging_dir=self.logging_dir,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            adam_epsilon=self.adam_epsilon,
            max_grad_norm=self.max_grad_norm,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            per_device_train_batch_size=2,  # Default to small batch size
            per_device_eval_batch_size=2,   # Default to small batch size
            logging_steps=self.logging_steps,
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            fp16=self.fp16,
            bf16=self.bf16,
            local_rank=self.local_rank,
            deepspeed=self.deepspeed,
            save_strategy="steps",
            evaluation_strategy="steps",
            report_to=["tensorboard", "wandb"],
            remove_unused_columns=False,
            label_names=["labels"],
        )


@dataclass
class Config:
    """Main configuration that combines all other configs."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    def save(self, output_path: str) -> None:
        """Save configuration to a JSON file."""
        import json
        from dataclasses import asdict
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to dict with a custom handler for non-serializable objects
        def custom_json_serializer(obj):
            if isinstance(obj, (set, map, filter)):
                return list(obj)
            raise TypeError(f"Type {type(obj)} not serializable")
        
        # Get dict representation and save
        config_dict = asdict(self)
        
        with open(output_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=custom_json_serializer)
    
    @classmethod
    def load(cls, input_path: str) -> "Config":
        """Load configuration from a JSON file."""
        import json
        
        with open(input_path, "r") as f:
            config_dict = json.load(f)
        
        # Extract specific configurations
        model_dict = config_dict.pop("model", {})
        lora_dict = config_dict.pop("lora", {})
        data_dict = config_dict.pop("data", {})
        train_dict = config_dict.pop("train", {})
        
        # Create config objects
        model_config = ModelConfig(**model_dict)
        lora_config = LoraConfig(**lora_dict)
        data_config = DataConfig(**data_dict)
        train_config = TrainConfig(**train_dict)
        
        return cls(
            model=model_config,
            lora=lora_config,
            data=data_config,
            train=train_config,
        )


# Export default configuration
config = Config()


# DeepSpeed configuration for ml.p4d.24xlarge (8x A100 GPUs)
def generate_deepspeed_config(
    bits: int = 4,
    stage: int = 2,
    offload: bool = False,
) -> Dict:
    """
    Generate DeepSpeed configuration for SageMaker instances.
    
    Args:
        bits: Quantization bits (4 or 8)
        stage: ZeRO stage (0-3)
        offload: Whether to offload to CPU/NVMe
        
    Returns:
        DeepSpeed configuration dictionary
    """
    ds_config = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 4,
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": stage,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "allgather_bucket_size": 5e8,
            "reduce_bucket_size": 5e8,
        },
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": True
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
        "wall_clock_breakdown": False
    }
    
    # Add offloading if requested
    if offload:
        ds_config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True
        }
    
    # Configure for 4-bit precision if requested
    if bits == 4:
        # 4-bit requires specific settings
        ds_config["zero_optimization"]["stage"] = 3  # Stage 3 is best for 4-bit
        ds_config["bf16"]["enabled"] = True
        ds_config["fp16"]["enabled"] = False
        # Add ZeRO-3 specific parameters
        ds_config["zero_optimization"]["stage3_prefetch_bucket_size"] = 5e8
        ds_config["zero_optimization"]["stage3_param_persistence_threshold"] = 1e6
    
    return ds_config


def save_deepspeed_config(output_path: str = "configs/deepspeed_config.json", **kwargs) -> None:
    """
    Save DeepSpeed configuration to a JSON file.
    
    Args:
        output_path: Path to save the configuration
        **kwargs: Additional arguments for generate_deepspeed_config
    """
    import json
    import os
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(generate_deepspeed_config(**kwargs), f, indent=2)
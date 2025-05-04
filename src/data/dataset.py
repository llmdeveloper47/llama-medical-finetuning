#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset handling for fine-tuning LLaMA 3.1 8B model on medical QA data.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Union, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer, AutoTokenizer

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MedicalInstructionDataset(Dataset):
    """
    Dataset class for medical instruction-tuning data.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name_or_path: Union[str, PreTrainedTokenizer],
        max_length: int = 4096,
        split: str = "train",
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the processed dataset
            tokenizer_name_or_path: HF tokenizer name, path, or instance
            max_length: Maximum sequence length
            split: Dataset split ("train", "validation", or "test")
        """
        self.data_path = data_path
        self.max_length = max_length
        self.split = split
        
        # Load tokenizer if string is provided
        if isinstance(tokenizer_name_or_path, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")
        else:
            # Use provided tokenizer
            self.tokenizer = tokenizer_name_or_path
        
        # Load dataset
        self.raw_data = self._load_dataset()
        
        # Process the dataset
        logger.info(f"Processing {len(self.raw_data)} examples for {split} set")
        self.processed_data = self._process_dataset()
    
    def _load_dataset(self) -> List[Dict]:
        """
        Load the dataset from file.
        
        Returns:
            List of data examples
        """
        # Check if we have split files
        split_file = os.path.join(self.data_path, f"{self.split}.jsonl")
        
        if os.path.exists(split_file):
            # Load specific split
            logger.info(f"Loading dataset from {split_file}")
            with open(split_file, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
        else:
            # Try to load from merged file
            merged_file = os.path.join(self.data_path, "merged_medical_instructions.jsonl")
            if os.path.exists(merged_file):
                logger.info(f"Loading merged dataset from {merged_file}")
                with open(merged_file, "r", encoding="utf-8") as f:
                    data = [json.loads(line) for line in f]
                    
                # For merged file, implement basic split (80/10/10)
                np.random.seed(42)  # For reproducibility
                indices = np.random.permutation(len(data))
                
                if self.split == "train":
                    split_indices = indices[:int(0.8 * len(indices))]
                elif self.split == "validation":
                    split_indices = indices[int(0.8 * len(indices)):int(0.9 * len(indices))]
                else:  # test
                    split_indices = indices[int(0.9 * len(indices)):]
                
                data = [data[i] for i in split_indices]
                logger.info(f"Split '{self.split}' has {len(data)} examples")
            else:
                # Try using HuggingFace datasets
                try:
                    logger.info(f"Trying to load dataset using HuggingFace datasets")
                    dataset = load_dataset("json", data_files={"train": os.path.join(self.data_path, "*.jsonl")})
                    data = list(dataset["train"])
                    
                    # Same split logic
                    np.random.seed(42)
                    indices = np.random.permutation(len(data))
                    
                    if self.split == "train":
                        split_indices = indices[:int(0.8 * len(indices))]
                    elif self.split == "validation":
                        split_indices = indices[int(0.8 * len(indices)):int(0.9 * len(indices))]
                    else:  # test
                        split_indices = indices[int(0.9 * len(indices)):]
                    
                    data = [data[i] for i in split_indices]
                    logger.info(f"Split '{self.split}' has {len(data)} examples")
                except Exception as e:
                    logger.error(f"Error loading dataset: {e}")
                    raise FileNotFoundError(f"Could not find dataset files in {self.data_path}")
        
        return data
    
    def _process_dataset(self) -> List[Dict]:
        """
        Process the dataset for training.
        
        Returns:
            List of processed examples
        """
        processed_data = []
        
        for example in self.raw_data:
            # Extract fields
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output = example.get("output", "")
            
            # Skip examples with missing fields
            if not instruction or not output:
                continue
            
            # Format prompt based on LLaMA chat template
            if hasattr(self.tokenizer, "apply_chat_template"):
                # Use the model's chat template if available
                messages = [
                    {"role": "user", "content": f"{instruction}\n\n{input_text}" if input_text else instruction},
                    {"role": "assistant", "content": output}
                ]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
            else:
                # Fallback to a basic template
                if input_text:
                    formatted_prompt = f"<|user|>\n{instruction}\n\n{input_text}<|endofuser|>\n<|assistant|>\n{output}<|endofassistant|>"
                else:
                    formatted_prompt = f"<|user|>\n{instruction}<|endofuser|>\n<|assistant|>\n{output}<|endofassistant|>"
            
            # Tokenize
            tokenized = self.tokenizer(
                formatted_prompt,
                max_length=self.max_length,
                truncation=True,
                padding="max_length" if self.split != "train" else False,
                return_tensors="pt",
            )
            
            # Create labels (same as input_ids, but -100 for prompt tokens to ignore in loss)
            input_ids = tokenized["input_ids"][0]
            attention_mask = tokenized["attention_mask"][0]
            labels = input_ids.clone()
            
            # Find assistant's response position
            assistant_token_ids = self.tokenizer.encode("<|assistant|>", add_special_tokens=False)
            assistant_positions = []
            
            for i in range(len(input_ids) - len(assistant_token_ids) + 1):
                if input_ids[i:i+len(assistant_token_ids)].tolist() == assistant_token_ids:
                    assistant_positions.append(i)
            
            if assistant_positions:
                # Set tokens before assistant's response to -100 (ignore in loss)
                assistant_pos = assistant_positions[0]
                labels[:assistant_pos] = -100
            
            # Add to processed data
            processed_data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                # Keep original data for reference
                "instruction": instruction,
                "input": input_text,
                "output": output,
            })
        
        return processed_data
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item by index."""
        return {
            "input_ids": self.processed_data[idx]["input_ids"],
            "attention_mask": self.processed_data[idx]["attention_mask"],
            "labels": self.processed_data[idx]["labels"],
        }
    
    def get_dataloader(self, batch_size: int = 1, shuffle: bool = True) -> DataLoader:
        """
        Get DataLoader for the dataset.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            PyTorch DataLoader
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle and self.split == "train",
            pin_memory=True,
            drop_last=self.split == "train",
        )


def create_dataset_splits(
    input_file: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Create train/validation/test splits from the processed dataset.
    
    Args:
        input_file: Path to the merged dataset file
        output_dir: Path to save the split datasets
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Check ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0: {train_ratio} + {val_ratio} + {test_ratio} = {train_ratio + val_ratio + test_ratio}")
    
    # Load data
    logger.info(f"Loading data from {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    # Shuffle data
    np.random.seed(seed)
    indices = np.random.permutation(len(data))
    
    # Create splits
    train_size = int(train_ratio * len(data))
    val_size = int(val_ratio * len(data))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    train_file = os.path.join(output_dir, "train.jsonl")
    val_file = os.path.join(output_dir, "validation.jsonl")
    test_file = os.path.join(output_dir, "test.jsonl")
    
    logger.info(f"Saving {len(train_data)} examples to {train_file}")
    with open(train_file, "w", encoding="utf-8") as f:
        for example in train_data:
            f.write(json.dumps(example) + "\n")
    
    logger.info(f"Saving {len(val_data)} examples to {val_file}")
    with open(val_file, "w", encoding="utf-8") as f:
        for example in val_data:
            f.write(json.dumps(example) + "\n")
    
    logger.info(f"Saving {len(test_data)} examples to {test_file}")
    with open(test_file, "w", encoding="utf-8") as f:
        for example in test_data:
            f.write(json.dumps(example) + "\n")
    
    logger.info(f"Dataset splits created: train={len(train_data)}, validation={len(val_data)}, test={len(test_data)}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Create dataset splits")
    parser.add_argument("--input_file", type=str, required=True, help="Path to merged dataset file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for splits")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training data ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation data ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test data ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    create_dataset_splits(
        args.input_file,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

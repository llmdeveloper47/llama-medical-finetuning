#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom Trainer for medical QA fine-tuning.
"""

import os
import re
import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MedicalQATrainer(Trainer):
    """
    Custom Trainer class for medical QA fine-tuning.
    
    This trainer adds:
    1. Medical-specific metrics (if enabled)
    2. Example logging during training
    3. GPU memory optimization
    4. Custom loss computation
    """
    
    def __init__(self, **kwargs):
        """Initialize trainer with additional medical QA settings."""
        super().__init__(**kwargs)
        
        # Flag to check if we're doing medical metrics
        self.compute_medical_metrics = getattr(self.args, "compute_medical_metrics", False)
        
        # Save the tokenizer for generation
        self.tokenizer = kwargs.get("tokenizer")
        
        # Example logging frequency (in evaluations)
        self.example_log_freq = 1
        self.eval_count = 0
        
        # Initialize best metrics for checkpoint selection
        self.best_metrics = {"loss": float("inf")}
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation that ignores padding and handles medical-specific losses.
        
        Args:
            model: The model to compute loss for
            inputs: The inputs to the model
            return_outputs: Whether to return model outputs along with loss
            
        Returns:
            Loss value or tuple (loss, outputs) if return_outputs=True
        """
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Add auxiliary metrics if needed
        if hasattr(self, "compute_medical_metrics") and self.compute_medical_metrics:
            # Access logits and labels
            logits = outputs.logits
            labels = inputs["labels"]
            
            # You could implement custom medical QA metrics here
            # For example, a diagnostic accuracy loss term
            
            # For now, just standard loss
            pass
        
        return (loss, outputs) if return_outputs else loss
    
    def log_metrics(self, split, metrics):
        """
        Log metrics with additional information.
        
        Args:
            split: Dataset split ("train", "eval", etc.)
            metrics: Metrics dictionary
        """
        # Log metrics using the parent method
        super().log_metrics(split, metrics)
        
        # Log to local logger for better console output
        metric_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info(f"{split} metrics: {metric_str}")
        
        # Store best metrics for checkpoint selection
        if split == "eval" and "loss" in metrics:
            if metrics["loss"] < self.best_metrics["loss"]:
                self.best_metrics["loss"] = metrics["loss"]
                logger.info(f"New best eval loss: {metrics['loss']:.4f}")
                
                # Save the best model
                best_model_path = os.path.join(self.args.output_dir, "best_model")
                self.save_model(best_model_path)
                logger.info(f"Saved best model to {best_model_path}")
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalPrediction:
        """
        Custom evaluation loop that adds example logging.
        
        Args:
            dataloader: Evaluation dataloader
            description: Description for progress bar
            prediction_loss_only: Whether to only compute loss
            ignore_keys: Keys to ignore in model output
            metric_key_prefix: Prefix for metric keys
            
        Returns:
            EvalPrediction with predictions and labels
        """
        # Increment eval count
        self.eval_count += 1
        
        # Regular evaluation
        eval_output = super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        
        # Log examples if it's time
        if self.eval_count % self.example_log_freq == 0 and self.tokenizer is not None:
            self._log_examples(dataloader)
        
        return eval_output
    
    def _log_examples(self, dataloader: DataLoader, num_examples: int = 2):
        """
        Generate and log example predictions.
        
        Args:
            dataloader: Evaluation dataloader
            num_examples: Number of examples to log
        """
        logger.info("Generating example predictions...")
        
        # Get samples from dataloader
        examples = []
        for batch in dataloader:
            if len(examples) >= num_examples:
                break
            examples.extend([{k: v[i] for k, v in batch.items()} for i in range(min(len(batch["input_ids"]), num_examples - len(examples)))])
        
        # Generate predictions
        for i, example in enumerate(examples):
            # Prepare input
            input_ids = example["input_ids"].unsqueeze(0).to(self.model.device)
            
            # Find the assistant start token
            if hasattr(self.tokenizer, "assistant_id"):
                assistant_token_id = self.tokenizer.assistant_id
            else:
                # Try to encode "<|assistant|>" or similar
                assistant_tokens = ["<|assistant|>", "<assistant>", "<bot>", "Assistant:"]
                for token in assistant_tokens:
                    try:
                        assistant_token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                        if len(assistant_token_ids) > 0:
                            assistant_token_id = assistant_token_ids[0]
                            break
                    except:
                        pass
                else:
                    # Default to EOS token if nothing else works
                    assistant_token_id = self.tokenizer.eos_token_id
            
            # Find position of assistant token in input
            assistant_pos = (input_ids == assistant_token_id).nonzero()
            if len(assistant_pos) > 0:
                prompt_ids = input_ids[:, :assistant_pos[0][1]+1]
            else:
                # If assistant token not found, use half of input
                prompt_ids = input_ids[:, :input_ids.size(1)//2]
            
            # Generate response
            with torch.no_grad():
                try:
                    generation = self.model.generate(
                        prompt_ids,
                        max_new_tokens=256,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                except Exception as e:
                    logger.warning(f"Generation failed: {e}")
                    continue
            
            # Decode input and output
            input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            generated_text = self.tokenizer.decode(generation[0], skip_special_tokens=True)
            label_text = self.tokenizer.decode(
                [t for t in example["labels"] if t != -100], 
                skip_special_tokens=True
            )
            
            # Extract reference answer - just the label part
            reference = label_text
            
            # Extract generated answer - just the part after the prompt
            if generated_text.startswith(input_text):
                prediction = generated_text[len(input_text):].strip()
            else:
                # Fallback if direct substring match fails
                prediction = generated_text
            
            # Log the example
            logger.info(f"\n--- Example {i+1} ---")
            logger.info(f"Input: {input_text[:100]}...")
            logger.info(f"Generated: {prediction[:100]}...")
            logger.info(f"Reference: {reference[:100]}...")
            
            # Try to extract diagnosis and treatment
            self._extract_medical_components(prediction)
    
    def _extract_medical_components(self, text: str) -> Dict[str, str]:
        """
        Extract medical components (diagnosis, treatment) from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            Dictionary with extracted components
        """
        components = {}
        
        # Extract diagnosis
        diagnosis_pattern = r"(?i)diagnosis:?\s*(.*?)(?=treatment|plan|\n\n|$)"
        diagnosis_match = re.search(diagnosis_pattern, text)
        if diagnosis_match:
            diagnosis = diagnosis_match.group(1).strip()
            components["diagnosis"] = diagnosis
            logger.info(f"Extracted Diagnosis: {diagnosis[:100]}..." if len(diagnosis) > 100 else f"Extracted Diagnosis: {diagnosis}")
        
        # Extract treatment
        treatment_pattern = r"(?i)treatment:?\s*(.*?)(?=\n\n|$)"
        treatment_match = re.search(treatment_pattern, text)
        if treatment_match:
            treatment = treatment_match.group(1).strip()
            components["treatment"] = treatment
            logger.info(f"Extracted Treatment: {treatment[:100]}..." if len(treatment) > 100 else f"Extracted Treatment: {treatment}")
        
        return components
    
    def save_model(self, output_dir=None, _internal_call=False):
        """
        Save the model with added metadata for medical QA.
        
        Args:
            output_dir: Directory to save the model
            _internal_call: Whether this is an internal call
        """
        # Call parent method
        super().save_model(output_dir, _internal_call)
        
        # Add medical QA specific metadata
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # Save metadata about medical QA task
        metadata = {
            "task": "medical_qa",
            "dataset": "HealthCareMagic-100K",
            "model_type": "llama",
            "best_metrics": self.best_metrics,
            "instruction_format": "Chat template with user/assistant roles",
        }
        
        # Save metadata
        metadata_file = os.path.join(output_dir, "medical_metadata.json")
        import json
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved medical QA metadata to {metadata_file}")

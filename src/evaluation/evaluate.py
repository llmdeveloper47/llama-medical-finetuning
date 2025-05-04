#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for fine-tuned LLaMA 3.1 8B model on medical QA dataset.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import evaluate
import numpy as np
import pandas as pd
import torch
import tqdm
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Add project root to path to allow imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.training.config import Config
from src.utils.logging_utils import get_logger, setup_file_logging

logger = get_logger("evaluation")


class MedicalQAEvaluator:
    """
    Evaluator for medical QA models.
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        data_path: str = "data/processed/test.jsonl",
        output_dir: str = "outputs/evaluation",
        device: str = "cuda",
        bits: int = 4,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the model or model name
            tokenizer_path: Path to the tokenizer (uses model_path if None)
            data_path: Path to the test dataset
            output_dir: Directory to save evaluation results
            device: Device to use for evaluation
            bits: Quantization bits (4, 8, or None for full precision)
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.device = device
        self.bits = bits
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup file logging
        setup_file_logging(logger, output_dir, "evaluation")
        
        # Check if device is available
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
        # Load metrics
        try:
            self.rouge = evaluate.load("rouge")
            self.bleu = evaluate.load("sacrebleu")
            self.bertscore = evaluate.load("bertscore")
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            logger.warning("Falling back to basic metrics only")
            self.rouge = None
            self.bleu = None
            self.bertscore = None
        
        # Load dataset
        self.test_dataset = self._load_dataset()
    
    def _load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading tokenizer from {self.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            padding_side="right",
            trust_remote_code=True,
        )
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
        
        # Configure quantization
        if self.bits in (4, 8):
            logger.info(f"Loading model with {self.bits}-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.bits == 4,
                load_in_8bit=self.bits == 8,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            logger.info("Loading model with full precision")
            quantization_config = None
        
        # Load model
        logger.info(f"Loading model from {self.model_path}")
        try:
            # Check if model is a LoRA model
            if os.path.exists(os.path.join(self.model_path, "adapter_config.json")):
                logger.info("Detected LoRA model, loading base model first")
                
                # Load adapter config to find base model
                with open(os.path.join(self.model_path, "adapter_config.json"), "r") as f:
                    adapter_config = json.load(f)
                
                base_model_name = adapter_config.get("base_model_name_or_path", "meta-llama/Meta-Llama-3.1-8B")
                logger.info(f"Using base model: {base_model_name}")
                
                # Load base model with quantization
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    token=os.environ.get("HF_TOKEN"),
                )
                
                # Load LoRA adapter
                model = PeftModel.from_pretrained(
                    base_model,
                    self.model_path,
                    device_map="auto",
                )
            else:
                # Load regular model
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    token=os.environ.get("HF_TOKEN"),
                )
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Attempting to load with safetensors")
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                token=os.environ.get("HF_TOKEN"),
                use_safetensors=True,
            )
        
        model.eval()
        return model, tokenizer
    
    def _load_dataset(self) -> List[Dict]:
        """
        Load and prepare the test dataset.
        
        Returns:
            Test dataset
        """
        logger.info(f"Loading test dataset from {self.data_path}")
        
        # Handle different file formats
        if self.data_path.endswith(".jsonl"):
            # Load from JSONL file
            with open(self.data_path, "r", encoding="utf-8") as f:
                test_dataset = [json.loads(line) for line in f]
        
        elif self.data_path.endswith(".json"):
            # Load from JSON file
            with open(self.data_path, "r", encoding="utf-8") as f:
                test_dataset = json.load(f)
        
        else:
            # Try loading as a Hugging Face dataset
            try:
                test_dataset = load_dataset("json", data_files=self.data_path)["train"]
                test_dataset = list(test_dataset)
            except:
                logger.error(f"Could not load dataset from {self.data_path}")
                test_dataset = []
        
        logger.info(f"Loaded {len(test_dataset)} test examples")
        return test_dataset
    
    def generate_predictions(
        self,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        num_samples: Optional[int] = None,
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Generate predictions for the test dataset.
        
        Args:
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_new_tokens: Maximum number of new tokens to generate
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            Tuple of (instructions, inputs, references, predictions)
        """
        instructions = []
        inputs = []
        references = []
        predictions = []
        
        # Limit number of samples if specified
        if num_samples is not None and num_samples < len(self.test_dataset):
            logger.info(f"Using {num_samples} samples from the test dataset")
            indices = np.random.choice(
                len(self.test_dataset), num_samples, replace=False
            )
            test_dataset = [self.test_dataset[i] for i in indices]
        else:
            test_dataset = self.test_dataset
        
        logger.info(f"Generating predictions for {len(test_dataset)} examples")
        
        for example in tqdm.tqdm(test_dataset, desc="Generating predictions"):
            instruction = example["instruction"]
            input_text = example["input"]
            reference = example["output"]
            
            # Format prompt based on model's chat template
            if hasattr(self.tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "user", "content": f"{instruction}\n\n{input_text}" if input_text else instruction}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                # Fallback template
                if input_text:
                    prompt = f"<|user|>\n{instruction}\n\n{input_text}<|endofuser|>\n<|assistant|>"
                else:
                    prompt = f"<|user|>\n{instruction}<|endofuser|>\n<|assistant|>"
            
            # Tokenize prompt
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            
            # Generate response
            with torch.no_grad():
                try:
                    output_ids = self.model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=(temperature > 0),
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                except Exception as e:
                    logger.error(f"Error generating prediction: {e}")
                    # Use empty prediction if generation fails
                    predictions.append("")
                    instructions.append(instruction)
                    inputs.append(input_text)
                    references.append(reference)
                    continue
            
            # Decode output
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
            
            # Extract only the assistant's reply
            if "<|assistant|>" in generated_text:
                # Template-style extraction
                assistant_part = generated_text.split("<|assistant|>")[1]
                if "<|endofassistant|>" in assistant_part:
                    assistant_part = assistant_part.split("<|endofassistant|>")[0]
            elif "assistant" in generated_text.lower():
                # Try regex extraction for other formats
                match = re.search(r"(?:assistant[:\s])(.*?)(?:$|(?:user[:\s]))", 
                                  generated_text, re.DOTALL | re.IGNORECASE)
                if match:
                    assistant_part = match.group(1)
                else:
                    # Just take everything after the prompt
                    assistant_part = generated_text[len(prompt):]
            else:
                # Just take everything after the prompt
                assistant_part = generated_text[len(prompt):]
            
            # Store results
            instructions.append(instruction)
            inputs.append(input_text)
            references.append(reference)
            predictions.append(assistant_part.strip())
        
        return instructions, inputs, references, predictions
    
    def compute_metrics(
        self,
        references: List[str],
        predictions: List[str],
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            references: Reference outputs
            predictions: Model predictions
            
        Returns:
            Dictionary of metrics
        """
        # Initialize metrics dictionary
        metrics = {}
        
        # ROUGE scores (if available)
        if self.rouge:
            logger.info("Computing ROUGE scores")
            try:
                rouge_output = self.rouge.compute(predictions=predictions, references=references)
                metrics.update({f"rouge_{k}": v for k, v in rouge_output.items()})
            except Exception as e:
                logger.error(f"Error computing ROUGE scores: {e}")
        
        # BLEU score (if available)
        if self.bleu:
            logger.info("Computing BLEU score")
            try:
                bleu_output = self.bleu.compute(predictions=predictions, references=[[r] for r in references])
                metrics["bleu"] = bleu_output["score"]
            except Exception as e:
                logger.error(f"Error computing BLEU score: {e}")
        
        # BERTScore (if available, on a subset to avoid OOM)
        if self.bertscore:
            try:
                logger.info("Computing BERTScore on a subset")
                max_bertscore_samples = min(50, len(predictions))
                bertscore_output = self.bertscore.compute(
                    predictions=predictions[:max_bertscore_samples],
                    references=references[:max_bertscore_samples],
                    lang="en",
                )
                metrics["bertscore_precision"] = np.mean(bertscore_output["precision"])
                metrics["bertscore_recall"] = np.mean(bertscore_output["recall"])
                metrics["bertscore_f1"] = np.mean(bertscore_output["f1"])
            except Exception as e:
                logger.error(f"Error computing BERTScore: {e}")
        
        # Medical domain-specific metrics
        logger.info("Computing medical domain-specific metrics")
        medical_metrics = self._compute_medical_metrics(references, predictions)
        metrics.update(medical_metrics)
        
        return metrics
    
    def _compute_medical_metrics(
        self,
        references: List[str],
        predictions: List[str],
    ) -> Dict[str, float]:
        """
        Compute medical domain-specific metrics.
        
        Args:
            references: Reference outputs
            predictions: Model predictions
            
        Returns:
            Dictionary of medical metrics
        """
        medical_metrics = {}
        
        # Diagnosis presence
        diagnosis_pattern = r"(?i)diagnosis:?\s*(.*?)(?=treatment|plan|\n\n|$)"
        
        diagnosis_in_ref = [bool(re.search(diagnosis_pattern, ref)) for ref in references]
        diagnosis_in_pred = [bool(re.search(diagnosis_pattern, pred)) for pred in predictions]
        
        # Diagnosis presence rate
        diagnosis_presence_rate = np.mean([a and b for a, b in zip(diagnosis_in_ref, diagnosis_in_pred)])
        medical_metrics["diagnosis_presence_rate"] = diagnosis_presence_rate
        
        # Treatment presence
        treatment_pattern = r"(?i)treatment|plan|therapy|management"
        
        treatment_in_ref = [bool(re.search(treatment_pattern, ref)) for ref in references]
        treatment_in_pred = [bool(re.search(treatment_pattern, pred)) for pred in predictions]
        
        # Treatment presence rate
        treatment_presence_rate = np.mean([a and b for a, b in zip(treatment_in_ref, treatment_in_pred)])
        medical_metrics["treatment_presence_rate"] = treatment_presence_rate
        
        # Diagnosis-treatment structure
        structure_in_ref = [a and b for a, b in zip(diagnosis_in_ref, treatment_in_ref)]
        structure_in_pred = [a and b for a, b in zip(diagnosis_in_pred, treatment_in_pred)]
        
        # Structure preservation rate
        structure_preservation_rate = np.mean([a and b for a, b in zip(structure_in_ref, structure_in_pred)])
        medical_metrics["structure_preservation_rate"] = structure_preservation_rate
        
        # Medical terminology
        med_terms = [
            "symptom", "condition", "diagnosis", "treatment", "medication", "therapy",
            "prognosis", "specialist", "prescription", "dosage", "examination", "test"
        ]
        
        med_terms_regex = r"(?i)\b(" + "|".join(med_terms) + r")\b"
        
        med_terms_in_ref = [len(re.findall(med_terms_regex, ref)) for ref in references]
        med_terms_in_pred = [len(re.findall(med_terms_regex, pred)) for pred in predictions]
        
        # Medical terminology ratio (pred / ref)
        med_terms_ratio = []
        for ref_count, pred_count in zip(med_terms_in_ref, med_terms_in_pred):
            if ref_count > 0:
                med_terms_ratio.append(min(pred_count / ref_count, 2.0))  # Cap at 2.0
            else:
                med_terms_ratio.append(1.0 if pred_count == 0 else 0.0)
        
        medical_metrics["medical_terminology_ratio"] = np.mean(med_terms_ratio)
        
        # Response length comparison
        ref_lengths = [len(ref.split()) for ref in references]
        pred_lengths = [len(pred.split()) for pred in predictions]
        
        # Length ratio (pred / ref)
        length_ratio = []
        for ref_len, pred_len in zip(ref_lengths, pred_lengths):
            if ref_len > 0:
                length_ratio.append(min(pred_len / ref_len, 2.0))  # Cap at 2.0
            else:
                length_ratio.append(1.0 if pred_len == 0 else 0.0)
        
        medical_metrics["length_ratio"] = np.mean(length_ratio)
        
        return medical_metrics
    
    def evaluate(
        self,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        num_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the model on the test dataset.
        
        Args:
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_new_tokens: Maximum number of new tokens to generate
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Starting evaluation")
        start_time = time.time()
        
        # Generate predictions
        instructions, inputs, references, predictions = self.generate_predictions(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_samples=num_samples,
        )
        
        # Compute metrics
        metrics = self.compute_metrics(references, predictions)
        
        # Save predictions and metrics
        self._save_results(instructions, inputs, references, predictions, metrics)
        
        # Log metrics
        logger.info(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
        
        return metrics
    
    def _save_results(
        self,
        instructions: List[str],
        inputs: List[str],
        references: List[str],
        predictions: List[str],
        metrics: Dict[str, float],
    ) -> None:
        """
        Save evaluation results.
        
        Args:
            instructions: Instructions
            inputs: Input texts
            references: Reference outputs
            predictions: Model predictions
            metrics: Evaluation metrics
        """
        # Save predictions
        results = []
        for instruction, input_text, reference, prediction in zip(
            instructions, inputs, references, predictions
        ):
            results.append({
                "instruction": instruction,
                "input": input_text,
                "reference": reference,
                "prediction": prediction,
            })
        
        predictions_file = os.path.join(self.output_dir, "predictions.jsonl")
        with open(predictions_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        
        # Save metrics
        metrics_file = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved predictions to {predictions_file}")
        logger.info(f"Saved metrics to {metrics_file}")
        
        # Generate a human-readable report
        report_file = os.path.join(self.output_dir, "evaluation_report.md")
        self._generate_report(results, metrics, report_file)
    
    def _generate_report(
        self,
        results: List[Dict[str, str]],
        metrics: Dict[str, float],
        output_file: str,
    ) -> None:
        """
        Generate a human-readable evaluation report.
        
        Args:
            results: Evaluation results
            metrics: Evaluation metrics
            output_file: Output file path
        """
        with open(output_file, "w") as f:
            f.write("# Medical QA Model Evaluation Report\n\n")
            
            # Model information
            f.write("## Model Information\n\n")
            f.write(f"- **Model Path:** {self.model_path}\n")
            f.write(f"- **Test Dataset:** {self.data_path}\n")
            f.write(f"- **Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Add metrics
            f.write("## Evaluation Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            
            # Group metrics by category
            metric_groups = {
                "Text Generation": ["rouge_1", "rouge_2", "rouge_l", "bleu"],
                "Semantic Similarity": ["bertscore_precision", "bertscore_recall", "bertscore_f1"],
                "Medical Domain": ["diagnosis_presence_rate", "treatment_presence_rate", 
                                   "structure_preservation_rate", "medical_terminology_ratio"],
                "General": ["length_ratio"]
            }
            
            # Write metrics by group
            for group, metric_keys in metric_groups.items():
                f.write(f"\n### {group}\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                
                for key in metric_keys:
                    if key in metrics:
                        f.write(f"| {key.replace('_', ' ').title()} | {metrics[key]:.4f} |\n")
            
            # Add sample predictions
            f.write("\n## Sample Predictions\n\n")
            
            # Select a few examples (max 10)
            num_examples = min(10, len(results))
            sample_indices = np.random.choice(len(results), num_examples, replace=False)
            
            for i, idx in enumerate(sample_indices):
                result = results[idx]
                
                f.write(f"### Example {i+1}\n\n")
                f.write(f"**Instruction:** {result['instruction']}\n\n")
                f.write(f"**Input:** {result['input']}\n\n")
                f.write(f"**Reference:** {result['reference']}\n\n")
                f.write(f"**Prediction:** {result['prediction']}\n\n")
                
                # Extract diagnosis and treatment from reference and prediction
                ref_diagnosis = re.search(r"(?i)diagnosis:?\s*(.*?)(?=treatment|plan|\n\n|$)", result['reference'])
                ref_treatment = re.search(r"(?i)treatment:?\s*(.*?)(?=\n\n|$)", result['reference'])
                
                pred_diagnosis = re.search(r"(?i)diagnosis:?\s*(.*?)(?=treatment|plan|\n\n|$)", result['prediction'])
                pred_treatment = re.search(r"(?i)treatment:?\s*(.*?)(?=\n\n|$)", result['prediction'])
                
                f.write("**Comparison:**\n\n")
                
                if ref_diagnosis and pred_diagnosis:
                    f.write("- **Diagnosis:** Present in both reference and prediction\n")
                elif ref_diagnosis:
                    f.write("- **Diagnosis:** Present in reference but missing in prediction\n")
                elif pred_diagnosis:
                    f.write("- **Diagnosis:** Missing in reference but present in prediction\n")
                else:
                    f.write("- **Diagnosis:** Missing in both reference and prediction\n")
                
                if ref_treatment and pred_treatment:
                    f.write("- **Treatment:** Present in both reference and prediction\n")
                elif ref_treatment:
                    f.write("- **Treatment:** Present in reference but missing in prediction\n")
                elif pred_treatment:
                    f.write("- **Treatment:** Missing in reference but present in prediction\n")
                else:
                    f.write("- **Treatment:** Missing in both reference and prediction\n")
                
                f.write("\n---\n\n")
        
        logger.info(f"Generated evaluation report at {output_file}")


def compare_models(
    finetuned_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    output_file: str,
) -> None:
    """
    Compare fine-tuned model with baseline model.
    
    Args:
        finetuned_metrics: Fine-tuned model metrics
        baseline_metrics: Baseline model metrics
        output_file: Output file path
    """
    with open(output_file, "w") as f:
        f.write("# Model Comparison: Fine-tuned vs. Baseline\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report compares the performance of the fine-tuned LLaMA 3.1 8B model against the baseline (pre-trained) model on the HealthCareMagic-100K medical QA dataset.\n\n")
        
        f.write("## Metrics Comparison\n\n")
        f.write("| Metric | Baseline | Fine-tuned | Improvement |\n")
        f.write("|--------|----------|------------|-------------|\n")
        
        # Group metrics by category
        metric_groups = {
            "Text Generation": ["rouge_1", "rouge_2", "rouge_l", "bleu"],
            "Semantic Similarity": ["bertscore_precision", "bertscore_recall", "bertscore_f1"],
            "Medical Domain": ["diagnosis_presence_rate", "treatment_presence_rate", 
                               "structure_preservation_rate", "medical_terminology_ratio"],
            "General": ["length_ratio"]
        }
        
        # Write metrics by group
        for group, metric_keys in metric_groups.items():
            f.write(f"\n### {group}\n\n")
            f.write("| Metric | Baseline | Fine-tuned | Improvement |\n")
            f.write("|--------|----------|------------|-------------|\n")
            
            for key in metric_keys:
                if key in finetuned_metrics and key in baseline_metrics:
                    baseline_value = baseline_metrics[key]
                    finetuned_value = finetuned_metrics[key]
                    improvement = finetuned_value - baseline_value
                    improvement_pct = (improvement / baseline_value) * 100 if baseline_value != 0 else float('inf')
                    
                    f.write(f"| {key.replace('_', ' ').title()} | {baseline_value:.4f} | {finetuned_value:.4f} | {improvement_pct:+.2f}% |\n")
        
        # Overall assessment
        f.write("\n## Overall Assessment\n\n")
        
        # Count improved metrics
        improved_metrics = sum(1 for k in finetuned_metrics if 
                              k in baseline_metrics and finetuned_metrics[k] > baseline_metrics[k])
        total_comparable_metrics = sum(1 for k in finetuned_metrics if k in baseline_metrics)
        
        improvement_rate = improved_metrics / total_comparable_metrics if total_comparable_metrics > 0 else 0
        
        f.write(f"The fine-tuned model improved on {improved_metrics} out of {total_comparable_metrics} metrics ({improvement_rate:.1%}).\n\n")
        
        # Highlight significant improvements
        significant_improvements = []
        for k in finetuned_metrics:
            if k in baseline_metrics:
                improvement_pct = (finetuned_metrics[k] - baseline_metrics[k]) / baseline_metrics[k] * 100 if baseline_metrics[k] != 0 else 0
                if improvement_pct >= 10:  # 10% improvement threshold
                    significant_improvements.append((k, improvement_pct))
        
        if significant_improvements:
            f.write("### Significant Improvements\n\n")
            for metric, pct in sorted(significant_improvements, key=lambda x: x[1], reverse=True):
                f.write(f"- **{metric.replace('_', ' ').title()}**: {pct:.1f}% improvement\n")
            f.write("\n")
        
        # Areas for improvement
        regressions = []
        for k in finetuned_metrics:
            if k in baseline_metrics:
                regression_pct = (baseline_metrics[k] - finetuned_metrics[k]) / baseline_metrics[k] * 100 if baseline_metrics[k] != 0 else 0
                if regression_pct >= 5:  # 5% regression threshold
                    regressions.append((k, regression_pct))
        
        if regressions:
            f.write("### Areas for Improvement\n\n")
            for metric, pct in sorted(regressions, key=lambda x: x[1], reverse=True):
                f.write(f"- **{metric.replace('_', ' ').title()}**: {pct:.1f}% regression\n")
            f.write("\n")
        
        f.write("## Conclusion\n\n")
        if improvement_rate > 0.6:
            f.write("The fine-tuning process has been largely successful, with improvements across the majority of metrics. The model shows particularly strong gains in medical domain-specific performance, suggesting that it has effectively learned from the HealthCareMagic-100K dataset.\n")
        elif improvement_rate > 0.4:
            f.write("The fine-tuning process has shown moderate success, with improvements in some key metrics but room for further optimization. Consider additional training epochs or hyperparameter tuning to enhance performance.\n")
        else:
            f.write("The fine-tuning process has shown limited improvements. Consider reviewing the training data quality, increasing the dataset size, or adjusting the fine-tuning approach to better capture medical domain knowledge.\n")
    
    logger.info(f"Generated model comparison report at {output_file}")


def evaluate_baseline(config: Config, output_dir: str, num_samples: int = 100) -> Dict[str, float]:
    """
    Evaluate the baseline model (LLaMA 3.1 8B without fine-tuning).
    
    Args:
        config: Training configuration
        output_dir: Output directory
        num_samples: Number of samples to evaluate
        
    Returns:
        Baseline model metrics
    """
    logger.info("Evaluating baseline model")
    
    baseline_output_dir = os.path.join(output_dir, "baseline")
    os.makedirs(baseline_output_dir, exist_ok=True)
    
    baseline_evaluator = MedicalQAEvaluator(
        model_path=config.model.model_name,
        data_path=os.path.join(config.data.data_dir, "test.jsonl"),
        output_dir=baseline_output_dir,
        bits=4,  # Use 4-bit quantization for efficiency
    )
    
    baseline_metrics = baseline_evaluator.evaluate(num_samples=num_samples)
    
    return baseline_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LLaMA 3.1 8B model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the tokenizer (uses model_path if None)")
    parser.add_argument("--data_path", type=str, default="data/processed/test.jsonl", help="Path to the test dataset")
    parser.add_argument("--config", type=str, default="configs/train_config.json", help="Path to training configuration")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation", help="Directory to save evaluation results")
    parser.add_argument("--compare_baseline", action="store_true", help="Compare with baseline model")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (None for all)")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits (4, 8, or 0 for none)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    if os.path.exists(args.config):
        config = Config.load(args.config)
    else:
        logger.warning(f"Config file not found: {args.config}, using default config")
        config = Config()
    
    # Evaluate fine-tuned model
    evaluator = MedicalQAEvaluator(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        bits=args.bits if args.bits > 0 else None,
    )
    
    finetuned_metrics = evaluator.evaluate(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
    )
    
    # Compare with baseline if requested
    if args.compare_baseline:
        baseline_metrics = evaluate_baseline(
            config, 
            args.output_dir, 
            num_samples=args.num_samples or 100
        )
        
        # Generate comparison report
        comparison_file = os.path.join(args.output_dir, "model_comparison.md")
        compare_models(finetuned_metrics, baseline_metrics, comparison_file)


if __name__ == "__main__":
    main()

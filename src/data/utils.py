#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for data processing in medical LLM fine-tuning.
"""

import os
import json
import re
from typing import Dict, List, Tuple, Optional, Any, Union
import random
import numpy as np
import pandas as pd


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the data
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str) -> None:
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to the output file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def split_dataset(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split a dataset into training, validation, and test sets.
    
    Args:
        data: List of data examples
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Set random seed for reproducibility
    random.seed(seed)
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    # Calculate split indices
    n = len(data_copy)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)
    
    # Split data
    train_data = data_copy[:train_end]
    val_data = data_copy[train_end:val_end]
    test_data = data_copy[val_end:]
    
    return train_data, val_data, test_data


def extract_patient_info(text: str) -> Dict[str, str]:
    """
    Extract patient demographic information from text.
    
    Args:
        text: Input text containing patient information
        
    Returns:
        Dictionary with extracted information
    """
    patient_info = {}
    
    # Extract age
    age_match = re.search(r"\b(\d+)\s*(?:years?|yrs?|y\.?o\.?|year\s+old)\b", text, re.IGNORECASE)
    if age_match:
        patient_info["age"] = age_match.group(1)
    
    # Extract gender
    if re.search(r"\b(?:male|man|boy|gentleman|father|husband|son)\b", text, re.IGNORECASE):
        patient_info["gender"] = "male"
    elif re.search(r"\b(?:female|woman|girl|lady|mother|wife|daughter)\b", text, re.IGNORECASE):
        patient_info["gender"] = "female"
    
    # Extract medical conditions
    conditions = []
    common_conditions = [
        "diabetes", "hypertension", "asthma", "arthritis", "copd", "depression",
        "anxiety", "cancer", "heart disease", "stroke", "thyroid"
    ]
    
    for condition in common_conditions:
        if re.search(r"\b" + condition + r"\b", text, re.IGNORECASE):
            conditions.append(condition)
    
    if conditions:
        patient_info["conditions"] = conditions
    
    return patient_info


def structure_medical_response(text: str) -> str:
    """
    Structure a medical response into diagnosis and treatment sections.
    
    Args:
        text: Medical response text
        
    Returns:
        Structured text with diagnosis and treatment sections
    """
    # Check if already structured
    if re.search(r"(?i)diagnosis:|treatment:", text):
        return text
    
    # Extract likely diagnosis
    diagnosis_part = None
    diagnosis_keywords = ["diagnosis", "condition", "suffering from", "you have", "it appears to be", 
                       "likely cause", "symptoms suggest", "consistent with"]
    
    for keyword in diagnosis_keywords:
        pattern = rf"\b{keyword}\b.*?(?:\.|$)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            diagnosis_part = match.group(0)
            break
    
    # Extract likely treatment
    treatment_part = None
    treatment_keywords = ["treatment", "recommend", "medication", "advise", "prescribe", 
                        "suggest", "should take", "management", "therapy"]
    
    for keyword in treatment_keywords:
        pattern = rf"\b{keyword}\b.*?(?:\.|$)"
        matches = re.finditer(pattern, text, re.IGNORECASE)
        treatment_sentences = [match.group(0) for match in matches]
        if treatment_sentences:
            treatment_part = " ".join(treatment_sentences)
            break
    
    # Format structured output
    structured_output = ""
    
    if diagnosis_part:
        structured_output += f"Diagnosis: {diagnosis_part}\n\n"
    else:
        # Use first sentence as diagnosis
        first_sentence = re.search(r"^.*?\.", text)
        if first_sentence:
            structured_output += f"Diagnosis: {first_sentence.group(0)}\n\n"
        else:
            structured_output += f"Diagnosis: {text[:100]}...\n\n"
    
    if treatment_part:
        structured_output += f"Treatment Plan: {treatment_part}\n"
    else:
        # Use rest of text as treatment
        remaining_text = text
        if diagnosis_part:
            remaining_text = text.replace(diagnosis_part, "", 1).strip()
        
        structured_output += f"Treatment Plan: {remaining_text}\n"
    
    return structured_output


def count_tokens(text: str, tokenizer) -> int:
    """
    Count the number of tokens in a text.
    
    Args:
        text: Input text
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Number of tokens
    """
    return len(tokenizer.encode(text))


def truncate_to_max_length(text: str, tokenizer, max_length: int) -> str:
    """
    Truncate text to fit within max_length tokens.
    
    Args:
        text: Input text
        tokenizer: HuggingFace tokenizer
        max_length: Maximum number of tokens
        
    Returns:
        Truncated text
    """
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_length:
        return text
    
    # Truncate tokens and decode
    truncated_tokens = tokens[:max_length]
    return tokenizer.decode(truncated_tokens)


if __name__ == "__main__":
    # Example usage
    example_data = [
        {"id": 1, "text": "Example 1"},
        {"id": 2, "text": "Example 2"},
        {"id": 3, "text": "Example 3"},
        {"id": 4, "text": "Example 4"},
        {"id": 5, "text": "Example 5"},
    ]
    
    # Split dataset
    train, val, test = split_dataset(example_data)
    print(f"Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")
    
    # Extract patient info
    patient_text = "45-year-old male with a history of hypertension and diabetes presenting with chest pain"
    info = extract_patient_info(patient_text)
    print(f"Patient info: {info}")
    
    # Structure medical response
    response = "The symptoms you describe suggest viral bronchitis. I recommend rest, plenty of fluids, and over-the-counter pain relievers."
    structured = structure_medical_response(response)
    print(f"Structured response:\n{structured}")

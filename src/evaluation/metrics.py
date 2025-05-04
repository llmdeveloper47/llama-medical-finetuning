#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical-specific evaluation metrics for LLM evaluation.
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union


def calculate_medical_metrics(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """
    Calculate medical-specific metrics for the generated responses.
    
    Args:
        references: List of reference texts
        predictions: List of generated texts
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Calculate diagnosis presence rate
    diagnosis_presence = _diagnosis_presence_rate(references, predictions)
    metrics.update(diagnosis_presence)
    
    # Calculate treatment presence rate
    treatment_presence = _treatment_presence_rate(references, predictions)
    metrics.update(treatment_presence)
    
    # Calculate structure preservation rate
    structure_preservation = _structure_preservation_rate(references, predictions)
    metrics.update(structure_preservation)
    
    # Calculate medical terminology ratio
    terminology_ratio = _medical_terminology_ratio(references, predictions)
    metrics.update(terminology_ratio)
    
    return metrics


def _diagnosis_presence_rate(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """
    Calculate the diagnosis presence rate in both reference and prediction.
    """
    diagnosis_pattern = r"(?i)diagnosis:?\s*(.*?)(?=treatment|plan|\n\n|$)"
    
    diagnosis_in_ref = [bool(re.search(diagnosis_pattern, ref)) for ref in references]
    diagnosis_in_pred = [bool(re.search(diagnosis_pattern, pred)) for pred in predictions]
    
    # Calculate presence rates
    ref_presence_rate = np.mean(diagnosis_in_ref)
    pred_presence_rate = np.mean(diagnosis_in_pred)
    
    # Calculate match rate (both ref and pred have diagnosis)
    match_rate = np.mean([a and b for a, b in zip(diagnosis_in_ref, diagnosis_in_pred)])
    
    return {
        "diagnosis_ref_presence_rate": ref_presence_rate,
        "diagnosis_pred_presence_rate": pred_presence_rate,
        "diagnosis_presence_match_rate": match_rate,
    }


def _treatment_presence_rate(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """
    Calculate the treatment plan presence rate in both reference and prediction.
    """
    treatment_pattern = r"(?i)treatment|plan|therapy|management"
    
    treatment_in_ref = [bool(re.search(treatment_pattern, ref)) for ref in references]
    treatment_in_pred = [bool(re.search(treatment_pattern, pred)) for pred in predictions]
    
    # Calculate presence rates
    ref_presence_rate = np.mean(treatment_in_ref)
    pred_presence_rate = np.mean(treatment_in_pred)
    
    # Calculate match rate (both ref and pred have treatment)
    match_rate = np.mean([a and b for a, b in zip(treatment_in_ref, treatment_in_pred)])
    
    return {
        "treatment_ref_presence_rate": ref_presence_rate,
        "treatment_pred_presence_rate": pred_presence_rate,
        "treatment_presence_match_rate": match_rate,
    }


def _structure_preservation_rate(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """
    Calculate how well the structure (diagnosis+treatment) is preserved.
    """
    diagnosis_pattern = r"(?i)diagnosis:?\s*(.*?)(?=treatment|plan|\n\n|$)"
    treatment_pattern = r"(?i)treatment|plan|therapy|management"
    
    # Check for both diagnosis and treatment
    diagnosis_in_ref = [bool(re.search(diagnosis_pattern, ref)) for ref in references]
    treatment_in_ref = [bool(re.search(treatment_pattern, ref)) for ref in references]
    structure_in_ref = [a and b for a, b in zip(diagnosis_in_ref, treatment_in_ref)]
    
    diagnosis_in_pred = [bool(re.search(diagnosis_pattern, pred)) for pred in predictions]
    treatment_in_pred = [bool(re.search(treatment_pattern, pred)) for pred in predictions]
    structure_in_pred = [a and b for a, b in zip(diagnosis_in_pred, treatment_in_pred)]
    
    # Calculate structure preservation
    ref_structure_rate = np.mean(structure_in_ref)
    pred_structure_rate = np.mean(structure_in_pred)
    
    # Structure preserved in both
    structure_preservation_rate = np.mean([a and b for a, b in zip(structure_in_ref, structure_in_pred)])
    
    return {
        "structure_ref_rate": ref_structure_rate,
        "structure_pred_rate": pred_structure_rate,
        "structure_preservation_rate": structure_preservation_rate,
    }


def _medical_terminology_ratio(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """
    Calculate the ratio of medical terminology in predictions compared to references.
    """
    # Common medical terms
    med_terms = [
        "symptom", "condition", "diagnosis", "treatment", "medication", "therapy",
        "prognosis", "specialist", "prescription", "dosage", "examination", "test",
        "blood", "chronic", "acute", "disease", "disorder", "syndrome", "infection",
        "inflammation", "pain", "fever", "nausea", "vomiting", "diarrhea", "fatigue",
        "allergy", "antibiotic", "antiviral", "analgesic", "antipyretic", "steroid"
    ]
    
    med_terms_regex = r"(?i)\b(" + "|".join(med_terms) + r")\b"
    
    # Count terms in each reference and prediction
    med_terms_in_ref = [len(re.findall(med_terms_regex, ref)) for ref in references]
    med_terms_in_pred = [len(re.findall(med_terms_regex, pred)) for pred in predictions]
    
    # Calculate ratio for each pair
    med_terms_ratio = []
    for ref_count, pred_count in zip(med_terms_in_ref, med_terms_in_pred):
        if ref_count > 0:
            med_terms_ratio.append(min(pred_count / ref_count, 2.0))  # Cap at 2.0
        else:
            med_terms_ratio.append(1.0 if pred_count == 0 else 0.0)
    
    # Average ratio across all examples
    avg_ratio = np.mean(med_terms_ratio)
    
    # Also calculate average term counts
    avg_ref_terms = np.mean(med_terms_in_ref)
    avg_pred_terms = np.mean(med_terms_in_pred)
    
    return {
        "medical_terminology_ratio": avg_ratio,
        "avg_ref_med_terms": avg_ref_terms,
        "avg_pred_med_terms": avg_pred_terms,
    }


def extract_diagnosis_and_treatment(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract diagnosis and treatment plan from text.
    
    Args:
        text: Medical response text
        
    Returns:
        Tuple of (diagnosis, treatment_plan) strings or None if not found
    """
    # Extract diagnosis
    diagnosis_pattern = r"(?i)diagnosis:?\s*(.*?)(?=treatment|plan|\n\n|$)"
    diagnosis_match = re.search(diagnosis_pattern, text)
    diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else None
    
    # Extract treatment
    treatment_pattern = r"(?i)treatment:?\s*(.*?)(?=\n\n|$)"
    treatment_match = re.search(treatment_pattern, text)
    treatment = treatment_match.group(1).strip() if treatment_match else None
    
    # If no specific treatment section found, try to find treatment keywords
    if treatment is None:
        treatment_keywords = ["recommend", "medication", "advise", "prescribe", 
                              "suggest", "should take", "management", "therapy"]
        
        for keyword in treatment_keywords:
            pattern = rf"(?i)\b{keyword}\b.*?(?:\.|$)"
            matches = re.finditer(pattern, text)
            treatment_sentences = [match.group(0) for match in matches]
            if treatment_sentences:
                treatment = " ".join(treatment_sentences)
                break
    
    return diagnosis, treatment


if __name__ == "__main__":
    # Example usage
    ref_texts = [
        "Diagnosis: You have a viral upper respiratory infection. Treatment: Rest, plenty of fluids, and over-the-counter pain relievers.",
        "Based on your symptoms, this appears to be migraine. I recommend avoiding triggers and taking pain medication.",
        "You are experiencing acid reflux. Treatment includes dietary changes and antacids."
    ]
    
    pred_texts = [
        "Diagnosis: This is a viral upper respiratory infection. Treatment Plan: Rest, fluids, and acetaminophen for fever.",
        "You likely have tension headaches. I suggest relaxation techniques and NSAIDs like ibuprofen.",
        "Diagnosis: Gastroesophageal reflux disease (GERD). Treatment: Avoid spicy foods and take an antacid."
    ]
    
    # Calculate metrics
    metrics = calculate_medical_metrics(ref_texts, pred_texts)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

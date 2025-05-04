#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Error analysis utilities for medical QA models.
"""

import os
import re
import json
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize paths for saving figures
FIGURES_DIR = "outputs/error_analysis/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_predictions(predictions_file: str) -> List[Dict]:
    """
    Load prediction results from a JSONL file.
    
    Args:
        predictions_file: Path to predictions file
        
    Returns:
        List of prediction results
    """
    results = []
    with open(predictions_file, "r") as f:
        for line in f:
            results.append(json.loads(line))
    return results


def classify_error_type(reference: str, prediction: str) -> List[str]:
    """
    Classify the type of error made in the prediction.
    
    Args:
        reference: Reference answer
        prediction: Model prediction
        
    Returns:
        List of error types
    """
    error_types = []
    
    # Extract diagnoses
    diagnosis_pattern = r"(?i)diagnosis:?\s*(.*?)(?=treatment|plan|\n\n|$)"
    ref_diagnosis_match = re.search(diagnosis_pattern, reference)
    pred_diagnosis_match = re.search(diagnosis_pattern, prediction)
    
    ref_diagnosis = ref_diagnosis_match.group(1).strip() if ref_diagnosis_match else None
    pred_diagnosis = pred_diagnosis_match.group(1).strip() if pred_diagnosis_match else None
    
    # Check for diagnosis errors
    if ref_diagnosis and pred_diagnosis:
        # Use Jaccard similarity to check if diagnoses match
        ref_words = set(ref_diagnosis.lower().split())
        pred_words = set(pred_diagnosis.lower().split())
        
        intersection = ref_words.intersection(pred_words)
        union = ref_words.union(pred_words)
        
        similarity = len(intersection) / len(union) if union else 0
        
        if similarity < 0.5:
            error_types.append("wrong_diagnosis")
        elif similarity < 0.8:
            error_types.append("partial_diagnosis")
    elif ref_diagnosis and not pred_diagnosis:
        error_types.append("missing_diagnosis")
    elif not ref_diagnosis and pred_diagnosis:
        error_types.append("unnecessary_diagnosis")
    
    # Check for treatment plan errors
    treatment_pattern = r"(?i)treatment|plan|management|therapy"
    ref_has_treatment = bool(re.search(treatment_pattern, reference))
    pred_has_treatment = bool(re.search(treatment_pattern, prediction))
    
    if ref_has_treatment and not pred_has_treatment:
        error_types.append("missing_treatment")
    
    # Check for missing key information
    if len(prediction.split()) < 50:
        error_types.append("incomplete_answer")
    
    # Check for hallucination (keywords not in reference)
    pred_specific_terms = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', prediction)
    ref_specific_terms = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', reference)
    
    pred_terms = set([term.lower() for term in pred_specific_terms])
    ref_terms = set([term.lower() for term in ref_specific_terms])
    
    potential_hallucinations = pred_terms - ref_terms
    
    medical_hallucinations = [term for term in potential_hallucinations 
                              if any(keyword in term for keyword in 
                                    ["mg", "dose", "therapy", "treatment", "medication"])]
    
    if len(medical_hallucinations) > 3:
        error_types.append("hallucination")
    
    # If no errors found, mark as correct
    if not error_types:
        error_types.append("correct")
    
    return error_types


def categorize_by_specialty(text: str, specialties: Dict[str, List[str]]) -> List[str]:
    """
    Categorize text by medical specialty.
    
    Args:
        text: Text to categorize
        specialties: Dictionary mapping specialty names to keyword lists
        
    Returns:
        List of matching specialties
    """
    text = text.lower()
    matching_specialties = []
    
    for specialty, keywords in specialties.items():
        for keyword in keywords:
            if re.search(r'\b' + keyword + r'\b', text):
                matching_specialties.append(specialty)
                break
    
    return matching_specialties


def analyze_errors(results: List[Dict]) -> Dict:
    """
    Analyze errors in predictions.
    
    Args:
        results: Prediction results
        
    Returns:
        Dictionary of analysis results
    """
    # Define medical specialties and their keywords
    specialties = {
        "cardiology": ["heart", "cardiac", "myocardial", "infarction", "angina", "hypertension", "arrhythmia"],
        "pulmonology": ["lung", "pulmonary", "respiratory", "asthma", "copd", "pneumonia"],
        "neurology": ["brain", "neural", "seizure", "stroke", "dementia", "alzheimer", "parkinson"],
        "gastroenterology": ["stomach", "intestine", "bowel", "liver", "pancreas", "gallbladder", "ulcer"],
        "orthopedics": ["bone", "joint", "fracture", "arthritis", "osteoporosis", "rheumatoid"],
        "endocrinology": ["diabetes", "thyroid", "hormone", "insulin", "adrenal", "pituitary"],
        "infectious_disease": ["infection", "bacterial", "viral", "fungal", "sepsis", "antibiotic"],
        "oncology": ["cancer", "tumor", "malignant", "chemotherapy", "radiation", "carcinoma"],
    }
    
    # Common conditions for analysis
    common_conditions = [
        "myocardial infarction", "pneumonia", "stroke", "appendicitis", "diabetes", 
        "hypertension", "copd", "asthma", "urinary tract infection", "sepsis"
    ]
    
    # Initialize analysis results
    analysis_results = {
        "error_types": Counter(),
        "specialty_performance": defaultdict(lambda: {"correct": 0, "total": 0}),
        "common_conditions": defaultdict(lambda: {"predicted": Counter(), "total": 0}),
        "challenging_cases": [],
        "error_examples": defaultdict(list),
    }
    
    # Process each result
    for result in results:
        instruction = result["instruction"]
        input_text = result["input"]
        reference = result["reference"]
        prediction = result["prediction"]
        
        # Classify error types
        error_types = classify_error_type(reference, prediction)
        
        # Update error type counts
        for error_type in error_types:
            analysis_results["error_types"][error_type] += 1
        
        # Categorize by specialty
        specialties_found = categorize_by_specialty(input_text + " " + reference, specialties)
        
        is_correct = "correct" in error_types
        
        for specialty in specialties_found:
            analysis_results["specialty_performance"][specialty]["total"] += 1
            if is_correct:
                analysis_results["specialty_performance"][specialty]["correct"] += 1
        
        # Analyze common conditions
        for condition in common_conditions:
            if condition.lower() in (input_text + " " + reference).lower():
                analysis_results["common_conditions"][condition]["total"] += 1
                
                for pred_condition in common_conditions:
                    if pred_condition.lower() in prediction.lower():
                        analysis_results["common_conditions"][condition]["predicted"][pred_condition] += 1
        
        # Collect challenging cases
        if len(error_types) > 1 or (len(error_types) == 1 and error_types[0] != "correct"):
            # Simple complexity score based on length and error types
            complexity = len(input_text.split()) + len(reference.split()) + len(error_types) * 10
            
            analysis_results["challenging_cases"].append({
                "input": input_text,
                "reference": reference,
                "prediction": prediction,
                "error_types": error_types,
                "complexity": complexity,
            })
        
        # Collect examples for each error type
        for error_type in error_types:
            if error_type != "correct" and len(analysis_results["error_examples"][error_type]) < 5:
                analysis_results["error_examples"][error_type].append({
                    "input": input_text,
                    "reference": reference,
                    "prediction": prediction,
                })
    
    # Sort challenging cases by complexity
    analysis_results["challenging_cases"].sort(key=lambda x: x["complexity"], reverse=True)
    analysis_results["challenging_cases"] = analysis_results["challenging_cases"][:20]  # Keep top 20
    
    return analysis_results


def generate_error_visualizations(analysis_results: Dict) -> None:
    """
    Generate visualizations for error analysis.
    
    Args:
        analysis_results: Results from error analysis
    """
    # Create output directory if it doesn't exist
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # 1. Error type distribution
    plt.figure(figsize=(10, 6))
    error_types = analysis_results["error_types"]
    labels = list(error_types.keys())
    values = list(error_types.values())
    
    # Sort by frequency
    sorted_indices = np.argsort(values)[::-1]
    labels = [labels[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    plt.bar(labels, values)
    plt.xlabel("Error Type")
    plt.ylabel("Count")
    plt.title("Distribution of Error Types")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    plt.savefig(os.path.join(FIGURES_DIR, "error_types.png"))
    plt.close()
    
    # 2. Performance by specialty
    plt.figure(figsize=(12, 6))
    specialty_performance = analysis_results["specialty_performance"]
    
    specialties = []
    accuracy = []
    
    for specialty, perf in specialty_performance.items():
        if perf["total"] > 0:
            specialties.append(specialty)
            accuracy.append(perf["correct"] / perf["total"] * 100)
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracy)
    specialties = [specialties[i] for i in sorted_indices]
    accuracy = [accuracy[i] for i in sorted_indices]
    
    plt.barh(specialties, accuracy, color="skyblue")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Medical Specialty")
    plt.title("Model Performance by Medical Specialty")
    plt.xlim(0, 100)
    plt.tight_layout()
    
    plt.savefig(os.path.join(FIGURES_DIR, "specialty_performance.png"))
    plt.close()
    
    # 3. Error types by specialty
    specialty_errors = defaultdict(Counter)
    
    # Skip if no specialty performance data
    if specialty_performance:
        top_specialties = sorted(specialty_performance.keys(), 
                                key=lambda x: specialty_performance[x]["total"], 
                                reverse=True)[:5]
        
        if top_specialties:
            plt.figure(figsize=(12, 8))
            
            error_categories = ["correct", "wrong_diagnosis", "missing_diagnosis", 
                              "missing_treatment", "hallucination", "incomplete_answer"]
            
            bar_width = 0.15
            index = np.arange(len(error_categories))
            
            for i, specialty in enumerate(top_specialties):
                values = [specialty_errors[specialty][error] for error in error_categories]
                plt.bar(index + i * bar_width, values, bar_width, label=specialty)
            
            plt.xlabel("Error Type")
            plt.ylabel("Count")
            plt.title("Error Types by Medical Specialty")
            plt.xticks(index + bar_width * (len(top_specialties) - 1) / 2, 
                      error_categories, rotation=45, ha="right")
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(os.path.join(FIGURES_DIR, "specialty_errors.png"))
            plt.close()


def generate_error_report(analysis_results: Dict, output_file: str) -> None:
    """
    Generate a human-readable error analysis report.
    
    Args:
        analysis_results: Results from error analysis
        output_file: Output file path
    """
    with open(output_file, "w") as f:
        f.write("# Medical QA Model Error Analysis Report\n\n")
        
        # Summary of error types
        f.write("## Error Type Distribution\n\n")
        f.write("The following error types were identified in the model's predictions:\n\n")
        
        error_types = analysis_results["error_types"]
        total_cases = sum(error_types.values())
        
        f.write("| Error Type | Count | Percentage |\n")
        f.write("|------------|-------|------------|\n")
        
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_cases * 100
            f.write(f"| {error_type} | {count} | {percentage:.2f}% |\n")
        
        f.write("\n![Error Type Distribution](./figures/error_types.png)\n\n")
        
        # Performance by specialty
        f.write("## Performance by Medical Specialty\n\n")
        f.write("The model's performance varies across different medical specialties:\n\n")
        
        specialty_performance = analysis_results["specialty_performance"]
        
        f.write("| Specialty | Accuracy | Sample Size |\n")
        f.write("|-----------|----------|-------------|\n")
        
        for specialty, perf in sorted(specialty_performance.items(), 
                                      key=lambda x: x[1]["correct"] / x[1]["total"] if x[1]["total"] > 0 else 0, 
                                      reverse=True):
            if perf["total"] > 0:
                accuracy = perf["correct"] / perf["total"] * 100
                f.write(f"| {specialty} | {accuracy:.2f}% | {perf['total']} |\n")
        
        f.write("\n![Performance by Specialty](./figures/specialty_performance.png)\n\n")
        
        # Common errors analysis
        f.write("## Common Error Patterns\n\n")
        
        for error_type, examples in analysis_results["error_examples"].items():
            if error_type != "correct" and examples:
                f.write(f"### {error_type.replace('_', ' ').title()}\n\n")
                f.write(f"This error occurred {error_types[error_type]} times ({error_types[error_type]/total_cases*100:.2f}% of cases).\n\n")
                
                # Show a representative example
                example = examples[0]
                f.write("**Representative example:**\n\n")
                f.write(f"Input: {example['input']}\n\n")
                f.write(f"Reference: {example['reference']}\n\n")
                f.write(f"Prediction: {example['prediction']}\n\n")
                
                # Analysis of this error type
                if error_type == "wrong_diagnosis":
                    f.write("This type of error indicates the model is misdiagnosing the condition. This could be due to:\n")
                    f.write("- Insufficient understanding of symptom patterns\n")
                    f.write("- Focusing on incorrect clinical features\n")
                    f.write("- Confusion between conditions with similar presentations\n\n")
                elif error_type == "missing_diagnosis":
                    f.write("The model fails to provide a clear diagnosis. This could be due to:\n")
                    f.write("- Uncertainty in complex cases\n")
                    f.write("- Incomplete reasoning process\n")
                    f.write("- Focusing too much on treatment without establishing diagnosis\n\n")
                elif error_type == "missing_treatment":
                    f.write("The model provides a diagnosis but fails to recommend treatment. This could be due to:\n")
                    f.write("- Incomplete response generation\n")
                    f.write("- Over-focusing on diagnostic reasoning\n")
                    f.write("- Limited training on treatment protocols\n\n")
                elif error_type == "hallucination":
                    f.write("The model includes medical information not supported by the input. This could be due to:\n")
                    f.write("- Overgeneralization from training data\n")
                    f.write("- Conflation of multiple similar conditions\n")
                    f.write("- Attempt to provide comprehensive answers with insufficient context\n\n")
        
        # Challenging cases
        f.write("## Most Challenging Cases\n\n")
        f.write("The following cases were particularly challenging for the model:\n\n")
        
        challenging_cases = analysis_results.get("challenging_cases", [])
        for i, case in enumerate(challenging_cases[:5]):  # Show top 5
            f.write(f"### Challenging Case {i+1}\n\n")
            f.write(f"**Input:** {case['input']}\n\n")
            f.write(f"**Reference:** {case['reference']}\n\n")
            f.write(f"**Prediction:** {case['prediction']}\n\n")
            f.write(f"**Error Types:** {', '.join(case['error_types'])}\n\n")
            f.write("---\n\n")
        
        # Recommendations for improvement
        f.write("## Recommendations for Model Improvement\n\n")
        
        f.write("Based on the error analysis, the following improvements are recommended:\n\n")
        
        # Determine recommendations based on most common errors
        common_errors = [error for error, count in error_types.items() 
                        if error != "correct" and count > total_cases * 0.05]
        
        if "wrong_diagnosis" in common_errors:
            f.write("1. **Improve diagnostic accuracy**: Fine-tune with more diverse diagnostic cases, especially in specialties with lower performance.\n\n")
        
        if "missing_treatment" in common_errors:
            f.write("2. **Enhance treatment knowledge**: Augment training data with more detailed treatment plans and current medical guidelines.\n\n")
        
        if "hallucination" in common_errors:
            f.write("3. **Reduce hallucinations**: Apply techniques to improve factual consistency, such as retrieval-augmented generation or fact-checking components.\n\n")
        
        if "incomplete_answer" in common_errors:
            f.write("4. **Ensure comprehensive responses**: Modify the training prompt to encourage complete answers that address both diagnosis and treatment.\n\n")
        
        f.write("5. **Specialty-specific improvements**: Focus additional training on the specialties with the lowest performance:\n")
        
        # Identify worst-performing specialties
        worst_specialties = []
        for specialty, perf in specialty_performance.items():
            if perf["total"] >= 5:  # Only consider specialties with enough samples
                accuracy = perf["correct"] / perf["total"]
                worst_specialties.append((specialty, accuracy))
        
        worst_specialties.sort(key=lambda x: x[1])
        
        for specialty, _ in worst_specialties[:3]:  # Top 3 worst
            f.write(f"   - {specialty}: Add more training examples and specialized medical knowledge\n")
        
        # Conclusion
        f.write("\n## Conclusion\n\n")
        correct_percentage = error_types.get("correct", 0) / total_cases * 100 if total_cases > 0 else 0
        f.write(f"The model achieved {correct_percentage:.2f}% fully correct answers across the test set. ")
        f.write("The most common error types indicate areas for targeted improvement, particularly in diagnostic accuracy and comprehensive treatment recommendations. ")
        f.write("By addressing these specific error patterns, the model's performance in medical question answering can be significantly enhanced.\n\n")
        
        # References to visualizations
        f.write("## Visualizations\n\n")
        f.write("1. [Error Type Distribution](./figures/error_types.png)\n")
        f.write("2. [Performance by Medical Specialty](./figures/specialty_performance.png)\n")
        if os.path.exists(os.path.join(FIGURES_DIR, "specialty_errors.png")):
            f.write("3. [Error Types by Specialty](./figures/specialty_errors.png)\n")


def run_error_analysis(predictions_file: str, output_dir: str) -> None:
    """
    Run the complete error analysis pipeline.
    
    Args:
        predictions_file: Path to predictions file
        output_dir: Directory to save analysis results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    
    # Load predictions
    results = load_predictions(predictions_file)
    
    # Analyze errors
    analysis_results = analyze_errors(results)
    
    # Generate visualizations
    generate_error_visualizations(analysis_results)
    
    # Generate report
    report_file = os.path.join(output_dir, "error_analysis_report.md")
    generate_error_report(analysis_results, report_file)
    
    print(f"Error analysis completed. Report saved to {report_file}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze errors in medical QA model predictions")
    parser.add_argument("--predictions_file", type=str, required=True, help="Path to predictions file")
    parser.add_argument("--output_dir", type=str, default="outputs/error_analysis", help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    run_error_analysis(args.predictions_file, args.output_dir)

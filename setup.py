#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="medical-llm-finetuning",
    version="0.1.0",
    description="Fine-tuning LLaMA 3.1 on medical QA datasets",
    author="",
    author_email="",
    url="",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "accelerate>=0.25.0",
        "datasets>=2.14.0",
        "evaluate>=0.4.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "tqdm>=4.66.0",
        "wandb>=0.15.0",
        "deepspeed>=0.11.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "safetensors>=0.4.0",
        "jsonlines>=3.1.0",
        "nltk>=3.8.0",
        "bitsandbytes>=0.41.0",
        "huggingface_hub>=0.19.0",
        "boto3>=1.28.0",
        "sagemaker>=2.173.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

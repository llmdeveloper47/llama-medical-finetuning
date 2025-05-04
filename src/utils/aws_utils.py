#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AWS and SageMaker utilities for medical LLM fine-tuning.
"""

import os
import subprocess
import json
from typing import Dict, List, Optional, Union

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Check if boto3 is available
try:
    import boto3
    import botocore
    BOTO3_AVAILABLE = True
except ImportError:
    logger.warning("boto3 not installed. AWS functions will not work.")
    BOTO3_AVAILABLE = False

# Check if sagemaker is available
try:
    import sagemaker
    from sagemaker.huggingface import HuggingFace
    SAGEMAKER_AVAILABLE = True
except ImportError:
    logger.warning("sagemaker not installed. SageMaker functions will not work.")
    SAGEMAKER_AVAILABLE = False


def upload_to_s3(local_path: str, s3_path: str) -> bool:
    """
    Upload local directory or file to S3 using the AWS CLI.
    
    Args:
        local_path: Local path to file or directory
        s3_path: S3 path (s3://bucket/path)
        
    Returns:
        True if successful, False otherwise
    """
    if not s3_path.startswith("s3://"):
        logger.error(f"Invalid S3 path: {s3_path}")
        return False
    
    try:
        logger.info(f"Uploading {local_path} to {s3_path}")
        
        # Use aws cli for efficient recursive upload
        if os.path.isdir(local_path):
            cmd = f"aws s3 cp {local_path} {s3_path} --recursive"
        else:
            cmd = f"aws s3 cp {local_path} {s3_path}"
        
        logger.info(f"Running command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        
        logger.info(f"Upload completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Upload failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Error during upload: {e}")
        return False


def download_from_s3(s3_path: str, local_path: str) -> bool:
    """
    Download from S3 to local directory or file using the AWS CLI.
    
    Args:
        s3_path: S3 path (s3://bucket/path)
        local_path: Local path for download
        
    Returns:
        True if successful, False otherwise
    """
    if not s3_path.startswith("s3://"):
        logger.error(f"Invalid S3 path: {s3_path}")
        return False
    
    try:
        logger.info(f"Downloading from {s3_path} to {local_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Use aws cli
        if s3_path.endswith("/"):
            cmd = f"aws s3 cp {s3_path} {local_path} --recursive"
        else:
            cmd = f"aws s3 cp {s3_path} {local_path}"
        
        logger.info(f"Running command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        
        logger.info(f"Download completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Error during download: {e}")
        return False


def get_instance_type_info(instance_type: str) -> Dict:
    """
    Get information about a SageMaker instance type.
    
    Args:
        instance_type: SageMaker instance type (e.g., "ml.p4d.24xlarge")
        
    Returns:
        Dict with instance information
    """
    # Common instance types and their configurations
    instance_info = {
        "ml.p4d.24xlarge": {
            "num_gpus": 8,
            "gpu_type": "A100",
            "gpu_memory_gb": 40,
            "vram_gb": 320,
            "vcpus": 96,
            "memory_gb": 1152,
        },
        "ml.g5.48xlarge": {
            "num_gpus": 8,
            "gpu_type": "A10G",
            "gpu_memory_gb": 24,
            "vram_gb": 192,
            "vcpus": 192,
            "memory_gb": 768,
        },
        "ml.g5.24xlarge": {
            "num_gpus": 4,
            "gpu_type": "A10G",
            "gpu_memory_gb": 24,
            "vram_gb": 96,
            "vcpus": 96,
            "memory_gb": 384,
        },
        "ml.g5.16xlarge": {
            "num_gpus": 1,
            "gpu_type": "A10G",
            "gpu_memory_gb": 24,
            "vram_gb": 24,
            "vcpus": 64,
            "memory_gb": 256,
        },
        "ml.p3.16xlarge": {
            "num_gpus": 8,
            "gpu_type": "V100",
            "gpu_memory_gb": 16,
            "vram_gb": 128,
            "vcpus": 64,
            "memory_gb": 512,
        },
    }
    
    if instance_type in instance_info:
        return instance_info[instance_type]
    else:
        logger.warning(f"Unknown instance type: {instance_type}")
        return {
            "num_gpus": "unknown",
            "gpu_type": "unknown",
            "gpu_memory_gb": "unknown",
            "vram_gb": "unknown",
            "vcpus": "unknown",
            "memory_gb": "unknown",
        }


def check_s3_path_exists(s3_path: str) -> bool:
    """
    Check if a path exists in S3 using boto3.
    
    Args:
        s3_path: S3 path (s3://bucket/path)
        
    Returns:
        True if the path exists, False otherwise
    """
    if not BOTO3_AVAILABLE:
        logger.error("boto3 not installed. Cannot check S3 path.")
        return False
    
    try:
        # Parse bucket and key from s3_path
        if not s3_path.startswith("s3://"):
            logger.error(f"Invalid S3 path: {s3_path}")
            return False
        
        parts = s3_path[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        
        # Create S3 client
        s3 = boto3.client("s3")
        
        if key == "":
            # Check if bucket exists
            try:
                s3.head_bucket(Bucket=bucket)
                return True
            except botocore.exceptions.ClientError:
                return False
        else:
            # Check if object exists
            try:
                s3.head_object(Bucket=bucket, Key=key)
                return True
            except botocore.exceptions.ClientError:
                # Check if prefix exists (for directories)
                response = s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
                return "Contents" in response or "CommonPrefixes" in response
    except Exception as e:
        logger.error(f"Error checking S3 path: {e}")
        return False


class SageMakerHelper:
    """
    Helper class for SageMaker operations.
    """
    
    def __init__(
        self,
        role: Optional[str] = None,
        region: str = "us-west-2",
        bucket: Optional[str] = None,
    ):
        """
        Initialize SageMaker helper.
        
        Args:
            role: SageMaker execution role ARN
            region: AWS region
            bucket: S3 bucket name for SageMaker
        """
        if not SAGEMAKER_AVAILABLE:
            logger.error("sagemaker not installed. SageMakerHelper will not work.")
            return
        
        self.region = region
        
        try:
            # Initialize SageMaker session
            self.sagemaker_session = sagemaker.Session()
            
            # Get role if not provided
            self.role = role or self.sagemaker_session.get_execution_role()
            
            # Get default bucket if not provided
            self.bucket = bucket or self.sagemaker_session.default_bucket()
            
            logger.info(f"Initialized SageMaker helper with role: {self.role}")
            logger.info(f"Using S3 bucket: {self.bucket}")
        except Exception as e:
            logger.error(f"Failed to initialize SageMaker helper: {e}")
            self.sagemaker_session = None
            self.role = None
            self.bucket = None
    
    def create_huggingface_estimator(
        self,
        source_dir: str,
        entry_point: str,
        instance_type: str = "ml.p4d.24xlarge",
        instance_count: int = 1,
        transformers_version: str = "4.36.2",
        pytorch_version: str = "2.1.0",
        py_version: str = "py310",
        hyperparameters: Optional[Dict] = None,
        distribution: Optional[Dict] = None,
    ) -> Optional[HuggingFace]:
        """
        Create a HuggingFace estimator for training.
        
        Args:
            source_dir: Path to source directory
            entry_point: Entry point script
            instance_type: SageMaker instance type
            instance_count: Number of instances
            transformers_version: Transformers version
            pytorch_version: PyTorch version
            py_version: Python version
            hyperparameters: Training hyperparameters
            distribution: Distribution configuration
            
        Returns:
            HuggingFace estimator or None if initialization failed
        """
        if not SAGEMAKER_AVAILABLE or self.sagemaker_session is None:
            logger.error("SageMaker not properly initialized. Cannot create estimator.")
            return None
        
        logger.info(f"Creating HuggingFace estimator with instance type: {instance_type}")
        
        # Get instance info
        instance_info = get_instance_type_info(instance_type)
        logger.info(f"Instance info: {instance_info}")
        
        # Set up hyperparameters
        if hyperparameters is None:
            hyperparameters = {}
        
        # Set up distribution if using multiple GPUs and not provided
        if distribution is None and instance_info.get("num_gpus", 0) > 1:
            distribution = {"torch_distributed": {"enabled": True}}
        
        try:
            # Create estimator
            estimator = HuggingFace(
                entry_point=entry_point,
                source_dir=source_dir,
                role=self.role,
                instance_type=instance_type,
                instance_count=instance_count,
                transformers_version=transformers_version,
                pytorch_version=pytorch_version,
                py_version=py_version,
                hyperparameters=hyperparameters,
                distribution=distribution,
                max_run=604800,  # 7 days in seconds
            )
            
            return estimator
        except Exception as e:
            logger.error(f"Failed to create HuggingFace estimator: {e}")
            return None
    
    def upload_data(
        self,
        local_path: str,
        s3_prefix: str = "medical-llm-finetuning/data",
    ) -> Optional[str]:
        """
        Upload data to S3 bucket.
        
        Args:
            local_path: Local path to data
            s3_prefix: S3 prefix
            
        Returns:
            S3 path to uploaded data or None if upload failed
        """
        if not SAGEMAKER_AVAILABLE or self.sagemaker_session is None:
            logger.error("SageMaker not properly initialized. Cannot upload data.")
            return None
        
        try:
            logger.info(f"Uploading data from {local_path} to S3")
            s3_path = self.sagemaker_session.upload_data(
                local_path, 
                bucket=self.bucket, 
                key_prefix=s3_prefix
            )
            logger.info(f"Data uploaded to {s3_path}")
            return s3_path
        except Exception as e:
            logger.error(f"Failed to upload data to S3: {e}")
            return None
    
    def start_training_job(
        self,
        estimator: HuggingFace,
        inputs: Dict[str, str],
        job_name: Optional[str] = None,
        wait: bool = False,
    ) -> Optional[str]:
        """
        Start a SageMaker training job.
        
        Args:
            estimator: HuggingFace estimator
            inputs: Dictionary mapping input channels to S3 paths
            job_name: Training job name (generated if None)
            wait: Whether to wait for the job to complete
            
        Returns:
            Training job name or None if job failed to start
        """
        if not SAGEMAKER_AVAILABLE or self.sagemaker_session is None:
            logger.error("SageMaker not properly initialized. Cannot start training job.")
            return None
        
        try:
            logger.info("Starting SageMaker training job")
            estimator.fit(inputs=inputs, job_name=job_name, wait=wait)
            
            # Get job name from estimator
            job_name = estimator._current_job_name
            logger.info(f"Training job started: {job_name}")
            
            return job_name
        except Exception as e:
            logger.error(f"Failed to start training job: {e}")
            return None
    
    def wait_for_training_job(self, job_name: str) -> Optional[Dict]:
        """
        Wait for a training job to complete.
        
        Args:
            job_name: Training job name
            
        Returns:
            Training job description or None if job failed
        """
        if not BOTO3_AVAILABLE:
            logger.error("boto3 not installed. Cannot wait for training job.")
            return None
        
        try:
            logger.info(f"Waiting for training job {job_name} to complete")
            
            # Create SageMaker client
            sm_client = boto3.client("sagemaker", region_name=self.region)
            
            # Wait for job to complete
            waiter = sm_client.get_waiter("training_job_completed_or_stopped")
            waiter.wait(TrainingJobName=job_name)
            
            # Get job description
            response = sm_client.describe_training_job(TrainingJobName=job_name)
            
            status = response["TrainingJobStatus"]
            logger.info(f"Training job {job_name} finished with status: {status}")
            
            if status == "Failed":
                failure_reason = response.get("FailureReason", "Unknown reason")
                logger.error(f"Training job failed: {failure_reason}")
            
            return response
        except Exception as e:
            logger.error(f"Error waiting for training job: {e}")
            return None
    
    def download_model_artifacts(
        self,
        job_name: str,
        local_path: str,
    ) -> bool:
        """
        Download model artifacts from a training job.
        
        Args:
            job_name: Training job name
            local_path: Local path to save artifacts
            
        Returns:
            True if successful, False otherwise
        """
        if not BOTO3_AVAILABLE:
            logger.error("boto3 not installed. Cannot download model artifacts.")
            return False
        
        try:
            logger.info(f"Downloading model artifacts from training job {job_name}")
            
            # Create SageMaker client
            sm_client = boto3.client("sagemaker", region_name=self.region)
            
            # Get job description
            response = sm_client.describe_training_job(TrainingJobName=job_name)
            
            # Get model artifacts path
            model_artifacts = response["ModelArtifacts"]["S3ModelArtifacts"]
            
            # Download artifacts
            return download_from_s3(model_artifacts, local_path)
        except Exception as e:
            logger.error(f"Error downloading model artifacts: {e}")
            return False


def create_sagemaker_estimator(config: Dict) -> Optional[HuggingFace]:
    """
    Create a SageMaker estimator from config.
    
    Args:
        config: Training configuration
        
    Returns:
        HuggingFace estimator or None if failed
    """
    if not SAGEMAKER_AVAILABLE:
        logger.error("sagemaker not installed. Cannot create estimator.")
        return None
    
    try:
        # Extract configs
        aws_region = config.get("aws_region", "us-west-2")
        sagemaker_bucket = config.get("sagemaker_bucket")
        sagemaker_role = config.get("sagemaker_role")
        
        # Create SageMaker helper
        helper = SageMakerHelper(
            role=sagemaker_role,
            region=aws_region,
            bucket=sagemaker_bucket,
        )
        
        # Extract instance configuration
        instance_type = config.get("instance_type", "ml.p4d.24xlarge")
        instance_count = config.get("instance_count", 1)
        
        # Create estimator
        estimator = helper.create_huggingface_estimator(
            source_dir=".",
            entry_point="src/training/train.py",
            instance_type=instance_type,
            instance_count=instance_count,
            hyperparameters={
                "model_name": config.get("model_name", "meta-llama/Meta-Llama-3.1-8B"),
                "output_dir": "/opt/ml/model",
                "deepspeed": config.get("deepspeed", "configs/deepspeed_config.json"),
                "per_device_train_batch_size": config.get("per_device_train_batch_size", 2),
                "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 4),
                "num_train_epochs": config.get("num_train_epochs", 3),
                "learning_rate": config.get("learning_rate", 2e-4),
                "use_lora": config.get("use_lora", True),
                "bits": config.get("bits", 4),
            },
        )
        
        return estimator
    except Exception as e:
        logger.error(f"Error creating SageMaker estimator: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    print("SageMaker instance types:")
    for instance_type in ["ml.p4d.24xlarge", "ml.g5.48xlarge", "ml.g5.24xlarge", "ml.g5.16xlarge", "ml.p3.16xlarge"]:
        info = get_instance_type_info(instance_type)
        print(f"{instance_type}: {info['num_gpus']} x {info['gpu_type']} GPUs, {info['vram_gb']} GB VRAM, {info['vcpus']} vCPUs, {info['memory_gb']} GB RAM")

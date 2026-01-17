"""Training pipeline orchestration for MedGemma models.

Provides high-level API for running training jobs with:
- Model selection (4B IT vs 27B IT)
- Ray Train distributed training
- MLflow experiment tracking
- Checkpoint management
"""

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from .model_selector import get_training_config, select_model
from .ray_trainer import MedGemmaTrainer, TrainingConfig, TrainingResult

logger = logging.getLogger(__name__)


def run_training_pipeline(
    model_name: str,
    dataset_name: Optional[str] = None,
    train_path: Optional[str] = None,
    eval_path: Optional[str] = None,
    max_steps: int = 10000,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    output_dir: str = "/checkpoints",
    experiment_name: Optional[str] = None,
    dry_run: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run a complete training pipeline for MedGemma models.
    
    This function orchestrates the full training workflow:
    1. Model selection and configuration
    2. Data loading and preprocessing
    3. Distributed training with Ray
    4. Experiment tracking with MLflow
    5. Checkpoint management
    
    Args:
        model_name: Model name or alias ("medgemma-4b", "medgemma-27b", etc.)
        dataset_name: Optional dataset name (for logging)
        train_path: Path to training data (local, S3, or HF)
        eval_path: Path to evaluation data
        max_steps: Maximum training steps
        batch_size: Override batch size (uses model default if None)
        learning_rate: Override learning rate (uses model default if None)
        output_dir: Directory for checkpoints
        experiment_name: MLflow experiment name
        dry_run: If True, validate config without training
        **kwargs: Additional training configuration
        
    Returns:
        Dictionary with:
        - status: "success" or "failed"
        - run_id: Unique run identifier
        - model_name: Selected model name
        - distributed_strategy: Training strategy used
        - metrics: Training metrics (if successful)
        - error: Error message (if failed)
        
    Example:
        ```python
        # Train MedGemma 4B on custom dataset
        result = run_training_pipeline(
            model_name="medgemma-4b",
            train_path="s3://data/medical_qa.jsonl",
            max_steps=5000,
        )
        
        # Train MedGemma 27B with 8x H100 GPUs
        result = run_training_pipeline(
            model_name="medgemma-27b",
            train_path="s3://data/clinical_notes.jsonl",
            max_steps=10000,
        )
        ```
    """
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Starting training pipeline run: {run_id}")
    logger.info(f"Model: {model_name}")
    
    try:
        # Get model-specific configuration
        model_config = select_model(model_name)
        training_config = get_training_config(model_name)
        
        logger.info(f"Model ID: {model_config['hf_model_id']}")
        logger.info(f"Training GPU count: {training_config['gpu_count']}")
        logger.info(f"Distributed strategy: {training_config['distributed_strategy']}")
        
        # Build training configuration
        config = TrainingConfig(
            model_name=model_name,
            max_steps=max_steps,
            batch_size=batch_size or training_config["batch_size"],
            learning_rate=learning_rate or training_config["learning_rate"],
            lora_r=kwargs.get("lora_r", training_config["lora_r"]),
            lora_alpha=kwargs.get("lora_alpha", training_config["lora_alpha"]),
            train_data_path=train_path,
            eval_data_path=eval_path,
            checkpoint_dir=output_dir,
            mlflow_experiment_name=experiment_name or f"medgemma-{model_name}-{timestamp}",
            dry_run=dry_run,
        )
        
        # Log configuration
        logger.info(f"Training configuration:")
        logger.info(f"  - Batch size: {config.batch_size}")
        logger.info(f"  - Learning rate: {config.learning_rate}")
        logger.info(f"  - Max steps: {config.max_steps}")
        logger.info(f"  - LoRA r: {config.lora_r}")
        logger.info(f"  - LoRA alpha: {config.lora_alpha}")
        
        if dry_run:
            logger.info("DRY RUN MODE - Skipping actual training")
            return {
                "status": "success",
                "run_id": run_id,
                "model_name": model_name,
                "model_id": model_config["hf_model_id"],
                "distributed_strategy": training_config["distributed_strategy"],
                "config": {
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "max_steps": config.max_steps,
                    "gpu_count": training_config["gpu_count"],
                },
                "dry_run": True,
            }
        
        # Create trainer and run
        trainer = MedGemmaTrainer(config)
        result = trainer.train()
        
        # Build response
        response = {
            "status": "success" if result.status == "completed" else "failed",
            "run_id": run_id,
            "model_name": model_name,
            "model_id": model_config["hf_model_id"],
            "distributed_strategy": result.distributed_strategy,
            "metrics": result.metrics,
        }
        
        if result.best_checkpoint_path:
            response["checkpoint_path"] = result.best_checkpoint_path
        
        if result.error:
            response["error"] = result.error
        
        return response
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        return {
            "status": "failed",
            "run_id": run_id,
            "model_name": model_name,
            "error": str(e),
        }


def run_evaluation_pipeline(
    model_name: str,
    checkpoint_path: Optional[str] = None,
    benchmarks: Optional[list] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run evaluation pipeline for a trained model.
    
    Args:
        model_name: Model name or alias
        checkpoint_path: Path to model checkpoint (uses base model if None)
        benchmarks: List of benchmarks to run (default: MedQA, PubMedQA)
        dry_run: If True, validate config without evaluating
        
    Returns:
        Dictionary with evaluation results
    """
    run_id = str(uuid.uuid4())[:8]
    
    logger.info(f"Starting evaluation pipeline run: {run_id}")
    logger.info(f"Model: {model_name}")
    
    if benchmarks is None:
        benchmarks = ["medqa", "pubmedqa"]
    
    try:
        model_config = select_model(model_name)
        
        if dry_run:
            return {
                "status": "success",
                "run_id": run_id,
                "model_name": model_name,
                "model_id": model_config["hf_model_id"],
                "benchmarks": benchmarks,
                "dry_run": True,
            }
        
        # TODO: Implement actual evaluation
        # This would load the model and run benchmarks
        
        return {
            "status": "success",
            "run_id": run_id,
            "model_name": model_name,
            "model_id": model_config["hf_model_id"],
            "benchmarks": benchmarks,
            "results": {
                "placeholder": "Evaluation not yet implemented"
            },
        }
        
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}")
        return {
            "status": "failed",
            "run_id": run_id,
            "model_name": model_name,
            "error": str(e),
        }


def get_pipeline_status(run_id: str) -> Dict[str, Any]:
    """
    Get status of a training pipeline run.
    
    Args:
        run_id: Run identifier
        
    Returns:
        Dictionary with run status and metrics
    """
    # TODO: Implement status tracking with Ray/MLflow
    return {
        "run_id": run_id,
        "status": "unknown",
        "message": "Status tracking not yet implemented",
    }


def list_pipeline_runs(
    model_name: Optional[str] = None,
    limit: int = 10,
) -> list:
    """
    List recent pipeline runs.
    
    Args:
        model_name: Filter by model name
        limit: Maximum number of runs to return
        
    Returns:
        List of run summaries
    """
    # TODO: Implement with MLflow
    return []

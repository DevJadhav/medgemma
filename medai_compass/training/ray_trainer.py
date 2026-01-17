"""Ray Trainer for MedGemma distributed training.

Provides distributed training orchestration using Ray Train with support for:
- Single GPU training (MedGemma 4B IT)
- Multi-GPU training with DeepSpeed ZeRO-3 (MedGemma 27B IT)
- Integration with MLflow for experiment tracking
- Checkpoint management with MinIO/S3 storage
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

from .model_selector import get_training_config, select_model, validate_gpu_requirements

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for MedGemma training job."""
    
    # Model selection
    model_name: str = "medgemma-4b"
    
    # Training parameters
    max_steps: int = 10000
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    
    # Data configuration
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    max_seq_length: int = 8192
    
    # Distributed training
    num_workers: int = 1
    use_gpu: bool = True
    distributed_strategy: Optional[str] = None
    
    # Checkpointing
    checkpoint_dir: str = "/checkpoints"
    save_steps: int = 500
    save_total_limit: int = 3
    
    # MLflow integration
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: str = "medgemma-finetuning"
    
    # Misc
    seed: int = 42
    dry_run: bool = False
    
    def __post_init__(self):
        """Validate and update configuration based on model selection."""
        # Get model-specific config
        model_config = get_training_config(self.model_name)
        
        # Override with model defaults if not explicitly set
        if self.distributed_strategy is None:
            self.distributed_strategy = model_config.get("distributed_strategy", "single_gpu")
        
        # Update num_workers for distributed training
        if self.distributed_strategy == "deepspeed_zero3":
            self.num_workers = model_config.get("gpu_count", 8)
        
        # Set MLflow URI from environment if not provided
        if self.mlflow_tracking_uri is None:
            self.mlflow_tracking_uri = os.environ.get(
                "MLFLOW_TRACKING_URI", "http://localhost:5000"
            )


@dataclass
class TrainingResult:
    """Result of a training job."""
    
    status: str  # "completed", "failed", "dry_run"
    run_id: Optional[str] = None
    model_id: Optional[str] = None
    final_loss: Optional[float] = None
    best_checkpoint_path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    distributed_strategy: Optional[str] = None


class MedGemmaTrainer:
    """
    Distributed trainer for MedGemma models using Ray Train.
    
    Supports both single-GPU training (4B model) and distributed
    training with DeepSpeed ZeRO-3 (27B model on 8x H100).
    
    Example:
        ```python
        config = TrainingConfig(
            model_name="medgemma-4b",
            max_steps=1000,
            train_data_path="s3://data/train.jsonl",
        )
        trainer = MedGemmaTrainer(config)
        result = trainer.train()
        ```
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model_config = select_model(config.model_name)
        self.training_config = get_training_config(config.model_name)
        
        # Validate GPU requirements
        if config.use_gpu and not config.dry_run:
            self._validate_resources()
    
    def _validate_resources(self):
        """Validate that required resources are available."""
        try:
            import torch
            available_gpus = torch.cuda.device_count()
            validate_gpu_requirements(self.config.model_name, available_gpus)
        except ImportError:
            logger.warning("PyTorch not available, skipping GPU validation")
    
    def _create_ray_trainer(self):
        """Create Ray TorchTrainer with appropriate configuration."""
        try:
            import ray
            from ray.train.torch import TorchTrainer
            from ray.train import RunConfig, ScalingConfig, CheckpointConfig
        except ImportError:
            raise ImportError("Ray Train is required. Install with: pip install 'ray[train]'")
        
        # Scaling configuration
        scaling_config = ScalingConfig(
            num_workers=self.config.num_workers,
            use_gpu=self.config.use_gpu,
            resources_per_worker={
                "CPU": 4,
                "GPU": 1 if self.config.use_gpu else 0,
            },
        )
        
        # Checkpoint configuration
        checkpoint_config = CheckpointConfig(
            num_to_keep=self.config.save_total_limit,
            checkpoint_score_attribute="loss",
            checkpoint_score_order="min",
        )
        
        # Run configuration
        run_config = RunConfig(
            name=f"medgemma-{self.config.model_name}-training",
            storage_path=self.config.checkpoint_dir,
            checkpoint_config=checkpoint_config,
        )
        
        # Create trainer
        trainer = TorchTrainer(
            train_loop_per_worker=self._train_loop,
            train_loop_config=self._get_train_loop_config(),
            scaling_config=scaling_config,
            run_config=run_config,
        )
        
        return trainer
    
    def _get_train_loop_config(self) -> Dict[str, Any]:
        """Get configuration for training loop."""
        return {
            "model_name": self.model_config["hf_model_id"],
            "max_steps": self.config.max_steps,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "warmup_ratio": self.config.warmup_ratio,
            "weight_decay": self.config.weight_decay,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "max_grad_norm": self.config.max_grad_norm,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "lora_target_modules": self.config.lora_target_modules,
            "max_seq_length": self.config.max_seq_length,
            "train_data_path": self.config.train_data_path,
            "eval_data_path": self.config.eval_data_path,
            "distributed_strategy": self.config.distributed_strategy,
            "seed": self.config.seed,
            "deepspeed_config": self.training_config.get("deepspeed_config"),
        }
    
    @staticmethod
    def _train_loop(config: Dict[str, Any]):
        """
        Training loop executed on each worker.
        
        This function runs on Ray workers and performs the actual training.
        """
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        
        try:
            import ray.train as train
            from ray.train import Checkpoint
        except ImportError:
            raise ImportError("Ray Train is required")
        
        # Set seed for reproducibility
        torch.manual_seed(config["seed"])
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"],
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with appropriate settings
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            torch_dtype=torch.bfloat16,
            device_map="auto" if config["distributed_strategy"] == "single_gpu" else None,
            trust_remote_code=True,
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["lora_target_modules"],
            bias="none",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Load datasets (placeholder - actual implementation would load from paths)
        # train_dataset = load_dataset(config["train_data_path"])
        # eval_dataset = load_dataset(config["eval_data_path"])
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=config.get("output_dir", "/tmp/training"),
            max_steps=config["max_steps"],
            per_device_train_batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            warmup_ratio=config["warmup_ratio"],
            weight_decay=config["weight_decay"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            max_grad_norm=config["max_grad_norm"],
            bf16=True,
            logging_steps=10,
            save_steps=500,
            save_total_limit=3,
            deepspeed=config.get("deepspeed_config"),
            gradient_checkpointing=True,
            report_to=["mlflow"],
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            # train_dataset=train_dataset,
            # eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )
        
        # Train
        train_result = trainer.train()
        
        # Report metrics
        metrics = {
            "loss": train_result.training_loss,
            "global_step": train_result.global_step,
        }
        
        train.report(metrics, checkpoint=Checkpoint.from_directory(training_args.output_dir))
    
    def train(self) -> TrainingResult:
        """
        Execute training job.
        
        Returns:
            TrainingResult with status and metrics
        """
        if self.config.dry_run:
            logger.info("Dry run mode - skipping actual training")
            return TrainingResult(
                status="dry_run",
                model_id=self.model_config["hf_model_id"],
                distributed_strategy=self.config.distributed_strategy,
                metrics={
                    "model_name": self.config.model_name,
                    "max_steps": self.config.max_steps,
                    "num_workers": self.config.num_workers,
                },
            )
        
        try:
            # Initialize MLflow
            self._setup_mlflow()
            
            # Create and run trainer
            ray_trainer = self._create_ray_trainer()
            result = ray_trainer.fit()
            
            # Extract metrics from result
            metrics = result.metrics if hasattr(result, "metrics") else {}
            
            return TrainingResult(
                status="completed",
                run_id=str(result.checkpoint.path) if result.checkpoint else None,
                model_id=self.model_config["hf_model_id"],
                final_loss=metrics.get("loss"),
                best_checkpoint_path=str(result.checkpoint.path) if result.checkpoint else None,
                metrics=metrics,
                distributed_strategy=self.config.distributed_strategy,
            )
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return TrainingResult(
                status="failed",
                error=str(e),
                model_id=self.model_config["hf_model_id"],
                distributed_strategy=self.config.distributed_strategy,
            )
    
    def _setup_mlflow(self):
        """Set up MLflow tracking."""
        try:
            import mlflow
            
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment_name)
            
            # Log configuration parameters
            mlflow.log_params({
                "model_name": self.config.model_name,
                "max_steps": self.config.max_steps,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "num_workers": self.config.num_workers,
                "distributed_strategy": self.config.distributed_strategy,
            })
            
        except ImportError:
            logger.warning("MLflow not available, skipping experiment tracking")
        except Exception as e:
            logger.warning(f"Failed to set up MLflow: {e}")

"""Ray Tune Trainable for MedGemma training."""

import os
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MedGemmaTrainable:
    """
    Ray Tune trainable for MedGemma fine-tuning.

    This class implements the training loop that Ray Tune will execute.
    It supports:
    - Checkpoint saving/loading for PBT
    - Metric reporting for early stopping
    - Dynamic hyperparameter updates
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainable with configuration.

        Args:
            config: Training configuration from Ray Tune.
        """
        self.config = config
        self.iteration = 0
        self.model = None
        self.trainer = None
        self.tokenizer = None

    def setup(self, config: Dict[str, Any]) -> None:
        """
        Set up the training environment.

        This is called once at the beginning of training.
        """
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
        )
        from peft import LoraConfig, get_peft_model

        self.config = config
        self.iteration = 0

        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Setting up training on device: {self.device}")

        # Load tokenizer
        model_name = config.get("model_name", "google/medgemma-4b-it")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        torch_dtype = getattr(torch, config.get("torch_dtype", "bfloat16"))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # Apply LoRA
        lora_config = LoraConfig(
            r=config.get("lora_r", 64),
            lora_alpha=config.get("lora_alpha", 128),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=config.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)

        # Load datasets
        self.train_dataset = self._load_dataset("train")
        self.eval_dataset = self._load_dataset("eval")

        # Create training arguments
        output_dir = os.path.join(
            config.get("output_dir", "/tmp/tune"),
            f"trial_{config.get('trial_id', 'default')}",
        )

        self.training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=config.get("learning_rate", 2e-4),
            per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
            weight_decay=config.get("weight_decay", 0.01),
            warmup_ratio=config.get("warmup_ratio", 0.03),
            num_train_epochs=1,  # We control iterations manually
            max_steps=config.get("steps_per_iteration", 100),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=config.get("steps_per_iteration", 100),
            save_strategy="no",  # We handle checkpoints
            bf16=torch.cuda.is_available(),
            gradient_checkpointing=config.get("gradient_checkpointing", True),
            report_to=[],  # Ray handles reporting
            remove_unused_columns=False,
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
        )

        logger.info("Training setup complete")

    def _load_dataset(self, split: str):
        """Load a dataset split."""
        from datasets import load_dataset

        data_path = self.config.get("data_path", "/data/combined_medical")
        max_samples = self.config.get("max_samples", None)

        try:
            # Try loading from local JSONL
            dataset = load_dataset(
                "json",
                data_files=f"{data_path}/{split}.jsonl",
                split="train",
            )
        except Exception:
            # Fall back to a small dummy dataset for testing
            logger.warning(f"Could not load {split} dataset, using dummy data")
            from datasets import Dataset

            dummy_data = [
                {
                    "text": "What are the symptoms of diabetes? Common symptoms include increased thirst, frequent urination, and fatigue."
                }
                for _ in range(100)
            ]
            dataset = Dataset.from_list(dummy_data)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        # Tokenize
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples.get("text", [""] * len(examples.get("input_ids", []))),
                max_length=self.config.get("max_length", 2048),
                truncation=True,
                padding="max_length",
            )
            tokenized["labels"] = tokenized["input_ids"].copy()

            # Add token_type_ids (required for Gemma3/MedGemma models)
            if "token_type_ids" not in tokenized:
                batch_size = len(tokenized["input_ids"])
                seq_len = len(tokenized["input_ids"][0]) if batch_size > 0 else 0
                tokenized["token_type_ids"] = [[0] * seq_len for _ in range(batch_size)]

            return tokenized

        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

        return dataset

    def step(self) -> Dict[str, Any]:
        """
        Execute one training step.

        Returns:
            Dictionary of metrics for this step.
        """
        # Train for one iteration
        train_result = self.trainer.train(
            resume_from_checkpoint=self.iteration > 0
        )

        # Evaluate
        eval_result = self.trainer.evaluate()

        self.iteration += 1

        metrics = {
            "training_iteration": self.iteration,
            "train_loss": train_result.training_loss,
            "eval_loss": eval_result.get("eval_loss", 0),
        }

        logger.info(f"Iteration {self.iteration}: {metrics}")
        return metrics

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Dict[str, Any]]:
        """
        Save a checkpoint for PBT.

        Args:
            checkpoint_dir: Directory to save checkpoint to.

        Returns:
            Dictionary with checkpoint metadata.
        """
        import torch

        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")

        # Save model and training state
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.trainer.optimizer.state_dict()
                if self.trainer.optimizer
                else None,
                "scheduler_state_dict": self.trainer.lr_scheduler.state_dict()
                if self.trainer.lr_scheduler
                else None,
                "iteration": self.iteration,
                "config": self.config,
            },
            checkpoint_path,
        )

        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return {"checkpoint_path": checkpoint_path}

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Load a checkpoint for PBT.

        Args:
            checkpoint_dir: Directory to load checkpoint from.
        """
        import torch

        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")

        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found at {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if checkpoint.get("optimizer_state_dict") and self.trainer.optimizer:
            self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if checkpoint.get("scheduler_state_dict") and self.trainer.lr_scheduler:
            self.trainer.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.iteration = checkpoint.get("iteration", 0)

        # Update config for PBT perturbations
        if "config" in checkpoint:
            self.config.update(checkpoint["config"])

        logger.info(f"Loaded checkpoint from {checkpoint_path}, iteration {self.iteration}")

    def reset_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Reset configuration for PBT perturbation.

        Args:
            new_config: New configuration values.

        Returns:
            True if reset was successful.
        """
        # Update learning rate on the fly
        if "learning_rate" in new_config and self.trainer.optimizer:
            for param_group in self.trainer.optimizer.param_groups:
                param_group["lr"] = new_config["learning_rate"]
            logger.info(f"Updated learning rate to {new_config['learning_rate']}")

        self.config.update(new_config)
        return True

    def cleanup(self) -> None:
        """Clean up resources."""
        import torch
        import gc

        if self.model is not None:
            del self.model
        if self.trainer is not None:
            del self.trainer
        if self.tokenizer is not None:
            del self.tokenizer

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Cleaned up training resources")


def create_trainable_class(base_config: Optional[Dict[str, Any]] = None):
    """
    Create a Ray Tune Trainable class with base configuration.

    Args:
        base_config: Base configuration to merge with trial config.

    Returns:
        A Trainable class for Ray Tune.
    """
    from ray import tune

    class ConfiguredTrainable(tune.Trainable):
        """Trainable with pre-configured base settings."""

        def setup(self, config: Dict[str, Any]) -> None:
            # Merge base config with trial config
            full_config = {**(base_config or {}), **config}
            self._trainable = MedGemmaTrainable(full_config)
            self._trainable.setup(full_config)

        def step(self) -> Dict[str, Any]:
            return self._trainable.step()

        def save_checkpoint(self, checkpoint_dir: str) -> Optional[Dict[str, Any]]:
            return self._trainable.save_checkpoint(checkpoint_dir)

        def load_checkpoint(self, checkpoint_dir: str) -> None:
            self._trainable.load_checkpoint(checkpoint_dir)

        def reset_config(self, new_config: Dict[str, Any]) -> bool:
            return self._trainable.reset_config(new_config)

        def cleanup(self) -> None:
            self._trainable.cleanup()

    return ConfiguredTrainable

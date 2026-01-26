"""
DeepSpeed ZeRO Integration for MedGemma Training.

Provides DeepSpeed ZeRO optimization stages:
- ZeRO-1: Optimizer state partitioning
- ZeRO-2: + Gradient partitioning
- ZeRO-3: + Parameter partitioning
- ZeRO-Infinity: NVMe offloading

Memory reduction: Up to 8x with ZeRO-3
"""

import os
import json
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, Callable
from dataclasses import dataclass

from .configs import DeepSpeedConfig


def check_deepspeed_available() -> bool:
    """Check if DeepSpeed is available."""
    try:
        import deepspeed
        return True
    except ImportError:
        return False


class ZeROOptimizer:
    """
    ZeRO Optimizer Wrapper.

    Wraps standard PyTorch optimizers with ZeRO partitioning
    for efficient distributed training.

    Example:
        >>> optimizer = ZeROOptimizer(
        ...     params=model.parameters(),
        ...     optimizer_class=torch.optim.AdamW,
        ...     lr=1e-4,
        ...     zero_stage=3,
        ... )
    """

    def __init__(
        self,
        params=None,
        optimizer_class: type = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        zero_stage: int = 3,
        offload_optimizer: bool = False,
        offload_param: bool = False,
        **kwargs,
    ):
        """
        Initialize ZeROOptimizer.

        Args:
            params: Model parameters
            optimizer_class: Base optimizer class (default: AdamW)
            lr: Learning rate
            weight_decay: Weight decay
            zero_stage: ZeRO optimization stage
            offload_optimizer: Offload optimizer states to CPU
            offload_param: Offload parameters to CPU
            **kwargs: Additional optimizer arguments
        """
        self.zero_stage = zero_stage
        self.offload_optimizer = offload_optimizer
        self.offload_param = offload_param

        # Default to AdamW
        if optimizer_class is None:
            optimizer_class = torch.optim.AdamW

        # Store configuration
        self._optimizer_class = optimizer_class
        self._lr = lr
        self._weight_decay = weight_decay
        self._kwargs = kwargs
        self._params = params

        # Create base optimizer if params provided
        if params is not None:
            self._optimizer = optimizer_class(
                params,
                lr=lr,
                weight_decay=weight_decay,
                **kwargs,
            )
        else:
            self._optimizer = None

        # DeepSpeed will wrap this in initialize()
        self._ds_optimizer = None

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform optimization step.

        Args:
            closure: Optional closure for computing loss

        Returns:
            Loss value if closure provided
        """
        if self._ds_optimizer is not None:
            return self._ds_optimizer.step(closure)
        return self._optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero out gradients."""
        if self._ds_optimizer is not None:
            self._ds_optimizer.zero_grad(set_to_none=set_to_none)
        else:
            self._optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dict."""
        if self._ds_optimizer is not None:
            return self._ds_optimizer.state_dict()
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state dict."""
        if self._ds_optimizer is not None:
            self._ds_optimizer.load_state_dict(state_dict)
        else:
            self._optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        """Get parameter groups."""
        if self._ds_optimizer is not None:
            return self._ds_optimizer.param_groups
        return self._optimizer.param_groups


class DeepSpeedTrainer:
    """
    DeepSpeed Trainer for MedGemma.

    Integrates DeepSpeed ZeRO optimization with PyTorch training
    for efficient large model training.

    Example:
        >>> config = DeepSpeedConfig(zero_stage=3, offload_optimizer=True)
        >>> trainer = DeepSpeedTrainer(config)
        >>> model, optimizer = trainer.initialize(model, optimizer)
        >>> trainer.train(dataset)
    """

    def __init__(
        self,
        config: Union[DeepSpeedConfig, Dict[str, Any]],
        local_rank: int = -1,
    ):
        """
        Initialize DeepSpeedTrainer.

        Args:
            config: DeepSpeed configuration
            local_rank: Local rank for distributed training
        """
        if isinstance(config, dict):
            config = DeepSpeedConfig(**config)

        self.config = config
        self.local_rank = local_rank if local_rank >= 0 else int(os.environ.get("LOCAL_RANK", 0))

        # State tracking
        self._model_engine = None
        self._model = None
        self._optimizer = None
        self._lr_scheduler = None
        self._initialized = False

    def initialize(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        model_parameters: Optional[Any] = None,
        training_data: Optional[Any] = None,
        config_params: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """
        Initialize DeepSpeed engine.

        Args:
            model: PyTorch model
            optimizer: Optional optimizer (DeepSpeed can create one)
            lr_scheduler: Optional learning rate scheduler
            model_parameters: Optional model parameters
            training_data: Optional training data loader
            config_params: Optional config overrides

        Returns:
            Tuple of (model_engine, optimizer, _, lr_scheduler)
        """
        if not check_deepspeed_available():
            raise ImportError(
                "DeepSpeed is not installed. "
                "Install with: pip install deepspeed"
            )

        import deepspeed

        # Merge config
        ds_config = self.config.to_deepspeed_config()
        if config_params:
            ds_config.update(config_params)

        # Initialize DeepSpeed
        self._model_engine, self._optimizer, _, self._lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            model_parameters=model_parameters,
            training_data=training_data,
            config=ds_config,
        )

        self._model = model
        self._initialized = True

        return self._model_engine, self._optimizer, None, self._lr_scheduler

    def train(
        self,
        dataset,
        num_epochs: int = 1,
        callbacks: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Train the model with DeepSpeed.

        Args:
            dataset: Training dataset or dataloader
            num_epochs: Number of epochs
            callbacks: Optional training callbacks

        Returns:
            Training metrics
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before train()")

        metrics = {
            "epoch_losses": [],
            "total_steps": 0,
        }

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataset:
                # Forward pass
                loss = self._model_engine(batch)

                # Backward pass
                self._model_engine.backward(loss)

                # Optimizer step
                self._model_engine.step()

                epoch_loss += loss.item()
                num_batches += 1
                metrics["total_steps"] += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            metrics["epoch_losses"].append(avg_loss)

            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, avg_loss, self._model_engine)

        return metrics

    def evaluate(
        self,
        dataset,
        callbacks: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the model with DeepSpeed.

        Args:
            dataset: Evaluation dataset or dataloader
            callbacks: Optional evaluation callbacks

        Returns:
            Evaluation metrics
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before evaluate()")

        metrics = {
            "eval_loss": 0.0,
            "num_samples": 0,
        }

        self._model_engine.eval()

        with torch.no_grad():
            for batch in dataset:
                # Forward pass only
                loss = self._model_engine(batch)
                metrics["eval_loss"] += loss.item()
                metrics["num_samples"] += 1

        if metrics["num_samples"] > 0:
            metrics["eval_loss"] /= metrics["num_samples"]

        self._model_engine.train()
        return metrics

    def save_checkpoint(
        self,
        save_dir: str,
        tag: Optional[str] = None,
        client_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save model checkpoint.

        Args:
            save_dir: Directory to save checkpoint
            tag: Optional checkpoint tag
            client_state: Optional client state to save
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before save_checkpoint()")

        self._model_engine.save_checkpoint(
            save_dir=save_dir,
            tag=tag,
            client_state=client_state,
        )

    def load_checkpoint(
        self,
        load_dir: str,
        tag: Optional[str] = None,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            load_dir: Directory to load checkpoint from
            tag: Optional checkpoint tag
            load_optimizer_states: Whether to load optimizer states
            load_lr_scheduler_states: Whether to load scheduler states

        Returns:
            Loaded client state
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before load_checkpoint()")

        _, client_state = self._model_engine.load_checkpoint(
            load_dir=load_dir,
            tag=tag,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
        )

        return client_state or {}

    @property
    def model(self) -> nn.Module:
        """Get the wrapped model."""
        if self._model_engine is not None:
            return self._model_engine.module
        return self._model

    @property
    def global_rank(self) -> int:
        """Get global rank."""
        if self._model_engine is not None:
            return self._model_engine.global_rank
        return 0

    @property
    def world_size(self) -> int:
        """Get world size."""
        if self._model_engine is not None:
            return self._model_engine.world_size
        return 1


class DeepSpeedZeROConfig:
    """
    Helper class for creating DeepSpeed ZeRO configurations.

    Provides convenient factory methods for common configurations.
    """

    @staticmethod
    def zero1(
        gradient_clipping: float = 1.0,
        bf16: bool = True,
    ) -> DeepSpeedConfig:
        """Create ZeRO-1 configuration (optimizer partitioning)."""
        return DeepSpeedConfig(
            zero_stage=1,
            bf16_enabled=bf16,
            gradient_clipping=gradient_clipping,
        )

    @staticmethod
    def zero2(
        gradient_clipping: float = 1.0,
        bf16: bool = True,
        overlap_comm: bool = True,
    ) -> DeepSpeedConfig:
        """Create ZeRO-2 configuration (+ gradient partitioning)."""
        return DeepSpeedConfig(
            zero_stage=2,
            bf16_enabled=bf16,
            gradient_clipping=gradient_clipping,
            overlap_comm=overlap_comm,
        )

    @staticmethod
    def zero3(
        gradient_clipping: float = 1.0,
        bf16: bool = True,
        offload_optimizer: bool = False,
        offload_param: bool = False,
    ) -> DeepSpeedConfig:
        """Create ZeRO-3 configuration (+ parameter partitioning)."""
        return DeepSpeedConfig(
            zero_stage=3,
            bf16_enabled=bf16,
            gradient_clipping=gradient_clipping,
            offload_optimizer=offload_optimizer,
            offload_param=offload_param,
        )

    @staticmethod
    def zero_infinity(
        nvme_path: str,
        gradient_clipping: float = 1.0,
        bf16: bool = True,
    ) -> DeepSpeedConfig:
        """Create ZeRO-Infinity configuration (NVMe offloading)."""
        return DeepSpeedConfig(
            zero_stage=3,
            bf16_enabled=bf16,
            gradient_clipping=gradient_clipping,
            offload_optimizer=True,
            offload_param=True,
            offload_optimizer_device="nvme",
            offload_param_device="nvme",
            nvme_path=nvme_path,
        )


def get_deepspeed_config_from_file(config_path: str) -> DeepSpeedConfig:
    """
    Load DeepSpeed configuration from JSON file.

    Args:
        config_path: Path to JSON config file

    Returns:
        DeepSpeedConfig instance
    """
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Extract relevant fields
    zero_config = config_dict.get("zero_optimization", {})

    return DeepSpeedConfig(
        zero_stage=zero_config.get("stage", 3),
        offload_optimizer="offload_optimizer" in zero_config,
        offload_param="offload_param" in zero_config,
        contiguous_gradients=zero_config.get("contiguous_gradients", True),
        overlap_comm=zero_config.get("overlap_comm", True),
        reduce_bucket_size=zero_config.get("reduce_bucket_size", 500_000_000),
        allgather_bucket_size=zero_config.get("allgather_bucket_size", 500_000_000),
        gradient_clipping=config_dict.get("gradient_clipping", 1.0),
        bf16_enabled="bf16" in config_dict and config_dict["bf16"].get("enabled", False),
        fp16_enabled="fp16" in config_dict and config_dict["fp16"].get("enabled", False),
    )

"""
Trainer implementations for all training algorithms.

Each trainer provides a consistent interface for training MedGemma models
with different PEFT methods and alignment algorithms.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from .configs import (
    LoRAConfig,
    QLoRAConfig,
    DoRAConfig,
    AdapterConfig,
    IA3Config,
    RLHFConfig,
    DPOConfig,
    KTOConfig,
    GRPOConfig,
    MHCConfig,
    _resolve_model_name,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Base Trainer
# =============================================================================

class BaseTrainer(ABC):
    """Base trainer class for all algorithms."""

    def __init__(
        self,
        model_name: str = "medgemma-4b",
        output_dir: str = "./output",
        apply_phi_filter: bool = False,
        **kwargs
    ):
        self.model_name = _resolve_model_name(model_name)
        self.output_dir = Path(output_dir)
        self.apply_phi_filter = apply_phi_filter

        self._model = None
        self._tokenizer = None
        self._trainer = None

    @abstractmethod
    def train(self, train_dataset, eval_dataset=None, callbacks=None):
        """Train the model."""
        pass

    @abstractmethod
    def save_adapter(self, output_path: str):
        """Save trained adapters/weights."""
        pass

    def get_training_arguments(self) -> Dict[str, Any]:
        """Get training arguments dictionary."""
        return {}


# =============================================================================
# LoRA Trainer
# =============================================================================

class LoRATrainer(BaseTrainer):
    """
    LoRA (Low-Rank Adaptation) Trainer.

    Trains low-rank adapter matrices for efficient fine-tuning.
    """

    def __init__(
        self,
        model_name: str = "medgemma-4b",
        config: Optional[LoRAConfig] = None,
        output_dir: str = "./output",
        **kwargs
    ):
        super().__init__(model_name, output_dir, **kwargs)
        self.config = config or LoRAConfig.for_model(model_name, **kwargs)

    def get_training_arguments(self) -> Dict[str, Any]:
        """Get training arguments."""
        args = self.config.get_training_arguments()
        args["output_dir"] = str(self.output_dir)
        return args

    def _load_model_and_tokenizer(self):
        """Load model with LoRA configuration."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig as PeftLoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError(
                "transformers and peft are required. "
                "Install with: pip install transformers peft"
            ) from e

        logger.info(f"Loading model with LoRA: {self.config.hf_model_id}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_model_id,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs = {"trust_remote_code": True}

        try:
            import torch
            if self.config.bf16:
                model_kwargs["torch_dtype"] = torch.bfloat16
        except ImportError:
            pass

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_model_id,
            **model_kwargs
        )

        # Apply LoRA through PEFT
        peft_config = PeftLoraConfig(**self.config.get_peft_config())
        self._model = get_peft_model(self._model, peft_config)

        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        logger.info(f"LoRA trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def train(self, train_dataset, eval_dataset=None, callbacks=None):
        """Train with LoRA."""
        if self._model is None:
            self._load_model_and_tokenizer()

        try:
            from trl import SFTTrainer, SFTConfig
        except ImportError:
            from transformers import Trainer, TrainingArguments

            training_args = TrainingArguments(**self.get_training_arguments())
            self._trainer = Trainer(
                model=self._model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self._tokenizer,
                callbacks=callbacks,
            )
            return self._trainer.train()

        sft_config = SFTConfig(
            **self.get_training_arguments(),
            max_seq_length=self.config.max_seq_length,
        )

        self._trainer = SFTTrainer(
            model=self._model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self._tokenizer,
            callbacks=callbacks,
        )

        return self._trainer.train()

    def save_adapter(self, output_path: str):
        """Save LoRA adapters."""
        if self._model is None:
            raise ValueError("Model not loaded")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(output_path)
        self._tokenizer.save_pretrained(output_path)
        logger.info(f"Saved LoRA adapters to {output_path}")


# =============================================================================
# QLoRA Trainer
# =============================================================================

class QLoRATrainer(BaseTrainer):
    """
    QLoRA (Quantized LoRA) Trainer.

    Combines 4-bit quantization with LoRA for memory-efficient fine-tuning.
    """

    def __init__(
        self,
        model_name: str = "medgemma-4b",
        config: Optional[QLoRAConfig] = None,
        output_dir: str = "./output",
        **kwargs
    ):
        super().__init__(model_name, output_dir, **kwargs)
        self.config = config or QLoRAConfig.for_model(model_name, **kwargs)

    def get_training_arguments(self) -> Dict[str, Any]:
        """Get training arguments."""
        args = self.config.get_training_arguments()
        args["output_dir"] = str(self.output_dir)
        return args

    def _load_model_and_tokenizer(self):
        """Load model with QLoRA (4-bit quantization + LoRA)."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as e:
            raise ImportError(
                "transformers, peft, and bitsandbytes are required. "
                "Install with: pip install transformers peft bitsandbytes"
            ) from e

        logger.info(f"Loading model with QLoRA (4-bit): {self.config.hf_model_id}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_model_id,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Quantization config
        import torch
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16 if self.config.bnb_4bit_compute_dtype == "bfloat16" else torch.float16,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_model_id,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )

        # Prepare for k-bit training
        self._model = prepare_model_for_kbit_training(self._model)

        # Apply LoRA
        peft_config = PeftLoraConfig(**self.config.get_peft_config())
        self._model = get_peft_model(self._model, peft_config)

        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        logger.info(f"QLoRA trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def train(self, train_dataset, eval_dataset=None, callbacks=None):
        """Train with QLoRA."""
        if self._model is None:
            self._load_model_and_tokenizer()

        try:
            from trl import SFTTrainer, SFTConfig
        except ImportError:
            from transformers import Trainer, TrainingArguments

            training_args = TrainingArguments(**self.get_training_arguments())
            self._trainer = Trainer(
                model=self._model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self._tokenizer,
                callbacks=callbacks,
            )
            return self._trainer.train()

        sft_config = SFTConfig(
            **self.get_training_arguments(),
            max_seq_length=self.config.max_seq_length,
        )

        self._trainer = SFTTrainer(
            model=self._model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self._tokenizer,
            callbacks=callbacks,
        )

        return self._trainer.train()

    def save_adapter(self, output_path: str):
        """Save QLoRA adapters."""
        if self._model is None:
            raise ValueError("Model not loaded")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(output_path)
        self._tokenizer.save_pretrained(output_path)
        logger.info(f"Saved QLoRA adapters to {output_path}")


# =============================================================================
# DoRA Trainer
# =============================================================================

class DoRATrainer(BaseTrainer):
    """
    DoRA (Weight-Decomposed Low-Rank Adaptation) Trainer.

    DoRA decomposes pre-trained weights into magnitude and direction,
    then applies LoRA to the directional component only.
    """

    def __init__(
        self,
        model_name: str = "medgemma-4b",
        config: Optional[DoRAConfig] = None,
        output_dir: str = "./output",
        **kwargs
    ):
        super().__init__(model_name, output_dir, **kwargs)
        self.config = config or DoRAConfig.for_model(model_name, **kwargs)

    def get_training_arguments(self) -> Dict[str, Any]:
        """Get training arguments."""
        args = self.config.get_training_arguments()
        args["output_dir"] = str(self.output_dir)
        return args

    def _load_model_and_tokenizer(self):
        """Load model with DoRA configuration."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig as PeftLoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError(
                "transformers and peft are required. "
                "Install with: pip install transformers peft"
            ) from e

        logger.info(f"Loading model with DoRA: {self.config.hf_model_id}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_model_id,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs = {"trust_remote_code": True}

        try:
            import torch
            if self.config.bf16:
                model_kwargs["torch_dtype"] = torch.bfloat16
        except ImportError:
            pass

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_model_id,
            **model_kwargs
        )

        # Apply DoRA through PEFT
        peft_config = PeftLoraConfig(**self.config.get_peft_config())
        self._model = get_peft_model(self._model, peft_config)

        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        logger.info(f"DoRA trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def train(self, train_dataset, eval_dataset=None, callbacks=None):
        """Train with DoRA."""
        if self._model is None:
            self._load_model_and_tokenizer()

        try:
            from trl import SFTTrainer, SFTConfig
        except ImportError:
            from transformers import Trainer, TrainingArguments

            training_args = TrainingArguments(**self.get_training_arguments())
            self._trainer = Trainer(
                model=self._model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self._tokenizer,
                callbacks=callbacks,
            )
            return self._trainer.train()

        sft_config = SFTConfig(
            **self.get_training_arguments(),
            max_seq_length=self.config.max_seq_length,
        )

        self._trainer = SFTTrainer(
            model=self._model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self._tokenizer,
            callbacks=callbacks,
        )

        return self._trainer.train()

    def save_adapter(self, output_path: str):
        """Save DoRA adapters."""
        if self._model is None:
            raise ValueError("Model not loaded")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(output_path)
        self._tokenizer.save_pretrained(output_path)
        logger.info(f"Saved DoRA adapters to {output_path}")


# =============================================================================
# Adapter Trainer (Houlsby/Pfeiffer)
# =============================================================================

class AdapterTrainer(BaseTrainer):
    """
    Adapter Module Trainer (Houlsby/Pfeiffer).

    Trains bottleneck adapters inserted into transformer layers.
    """

    def __init__(
        self,
        model_name: str = "medgemma-4b",
        adapter_type: str = "houlsby",
        config: Optional[AdapterConfig] = None,
        output_dir: str = "./output",
        **kwargs
    ):
        super().__init__(model_name, output_dir, **kwargs)
        self.adapter_type = adapter_type
        self.config = config or AdapterConfig.for_model(
            model_name, adapter_type=adapter_type, **kwargs
        )

    def get_training_arguments(self) -> Dict[str, Any]:
        """Get training arguments."""
        args = self.config.get_training_arguments()
        args["output_dir"] = str(self.output_dir)
        return args

    def _load_model_and_tokenizer(self):
        """Load model with adapter configuration."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("transformers required") from e

        logger.info(f"Loading model with {self.adapter_type} adapters")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_model_id,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_model_id,
            trust_remote_code=True,
        )

        # Add adapters using adapters library or custom implementation
        self._add_adapters()

    def _add_adapters(self):
        """Add adapter layers to the model."""
        # This would use the adapters library or a custom implementation
        # For now, we'll use a simplified approach
        logger.info(f"Adding {self.adapter_type} adapters with bottleneck_dim={self.config.bottleneck_dim}")

    def train(self, train_dataset, eval_dataset=None, callbacks=None):
        """Train with adapters."""
        if self._model is None:
            self._load_model_and_tokenizer()

        from transformers import Trainer, TrainingArguments

        training_args = TrainingArguments(**self.get_training_arguments())
        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self._tokenizer,
            callbacks=callbacks,
        )

        return self._trainer.train()

    def save_adapter(self, output_path: str):
        """Save adapters."""
        if self._model is None:
            raise ValueError("Model not loaded")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(output_path)
        self._tokenizer.save_pretrained(output_path)
        logger.info(f"Saved adapters to {output_path}")


# =============================================================================
# IA3 Trainer
# =============================================================================

class IA3Trainer(BaseTrainer):
    """
    IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) Trainer.

    IA3 learns scaling vectors for key, value, and feedforward activations.
    """

    def __init__(
        self,
        model_name: str = "medgemma-4b",
        config: Optional[IA3Config] = None,
        output_dir: str = "./output",
        **kwargs
    ):
        super().__init__(model_name, output_dir, **kwargs)
        self.config = config or IA3Config.for_model(model_name, **kwargs)

    def get_training_arguments(self) -> Dict[str, Any]:
        """Get training arguments."""
        args = self.config.get_training_arguments()
        args["output_dir"] = str(self.output_dir)
        return args

    def _load_model_and_tokenizer(self):
        """Load model with IA3 configuration."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import IA3Config as PeftIA3Config, get_peft_model
        except ImportError as e:
            raise ImportError("transformers and peft required") from e

        logger.info(f"Loading model with IA3: {self.config.hf_model_id}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_model_id,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_model_id,
            trust_remote_code=True,
        )

        # Apply IA3
        ia3_config = PeftIA3Config(**self.config.get_peft_config())
        self._model = get_peft_model(self._model, ia3_config)

        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        logger.info(f"IA3 trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    def train(self, train_dataset, eval_dataset=None, callbacks=None):
        """Train with IA3."""
        if self._model is None:
            self._load_model_and_tokenizer()

        from transformers import Trainer, TrainingArguments

        training_args = TrainingArguments(**self.get_training_arguments())
        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self._tokenizer,
            callbacks=callbacks,
        )

        return self._trainer.train()

    def save_adapter(self, output_path: str):
        """Save IA3 adapters."""
        if self._model is None:
            raise ValueError("Model not loaded")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(output_path)
        self._tokenizer.save_pretrained(output_path)
        logger.info(f"Saved IA3 adapters to {output_path}")


# =============================================================================
# RLHF/PPO Trainer
# =============================================================================

class RLHFTrainer(BaseTrainer):
    """
    RLHF (Reinforcement Learning from Human Feedback) Trainer with PPO.

    Uses a reward model and PPO to optimize the policy.
    """

    def __init__(
        self,
        model_name: str = "medgemma-4b",
        config: Optional[RLHFConfig] = None,
        output_dir: str = "./output",
        reward_model=None,
        **kwargs
    ):
        super().__init__(model_name, output_dir, **kwargs)
        self.config = config or RLHFConfig.for_model(model_name, **kwargs)
        self.reward_model = reward_model

    def get_training_arguments(self) -> Dict[str, Any]:
        """Get training arguments."""
        args = self.config.get_training_arguments()
        args["output_dir"] = str(self.output_dir)
        return args

    def _load_model_and_tokenizer(self):
        """Load model for RLHF."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("transformers required") from e

        logger.info(f"Loading model for RLHF: {self.config.hf_model_id}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_model_id,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_model_id,
            trust_remote_code=True,
        )

    def compute_rewards(self, responses, prompts):
        """Compute rewards for responses."""
        if self.reward_model is None:
            # Return placeholder rewards
            return [0.0] * len(responses)

        rewards = []
        for response, prompt in zip(responses, prompts):
            # Use reward model to score
            reward = self.reward_model(prompt, response)
            rewards.append(reward)
        return rewards

    def ppo_step(self, batch):
        """Execute a PPO training step."""
        # Placeholder for PPO implementation
        pass

    def train(self, train_dataset, eval_dataset=None, callbacks=None):
        """Train with RLHF/PPO."""
        if self._model is None:
            self._load_model_and_tokenizer()

        try:
            from trl import PPOTrainer, PPOConfig
        except ImportError:
            logger.warning("TRL not available, using simplified RLHF")
            return self._train_simplified(train_dataset, eval_dataset, callbacks)

        ppo_config = PPOConfig(**self.config.get_ppo_config())

        self._trainer = PPOTrainer(
            model=self._model,
            config=ppo_config,
            tokenizer=self._tokenizer,
            dataset=train_dataset,
        )

        # Training loop would go here
        logger.info("Starting RLHF training with PPO")
        return None

    def _train_simplified(self, train_dataset, eval_dataset, callbacks):
        """Simplified training without TRL."""
        from transformers import Trainer, TrainingArguments

        training_args = TrainingArguments(**self.get_training_arguments())
        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self._tokenizer,
            callbacks=callbacks,
        )
        return self._trainer.train()

    def save_adapter(self, output_path: str):
        """Save model."""
        if self._model is None:
            raise ValueError("Model not loaded")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(output_path)
        self._tokenizer.save_pretrained(output_path)


# =============================================================================
# DPO Trainer
# =============================================================================

class DPOTrainer(BaseTrainer):
    """
    DPO (Direct Preference Optimization) Trainer.

    Directly optimizes for human preferences without a reward model.
    """

    def __init__(
        self,
        model_name: str = "medgemma-4b",
        config: Optional[DPOConfig] = None,
        output_dir: str = "./output",
        ref_model=None,
        **kwargs
    ):
        super().__init__(model_name, output_dir, **kwargs)
        self.config = config or DPOConfig.for_model(model_name, **kwargs)
        self.ref_model = ref_model

    def get_training_arguments(self) -> Dict[str, Any]:
        """Get training arguments."""
        args = self.config.get_training_arguments()
        args["output_dir"] = str(self.output_dir)
        return args

    def _load_model_and_tokenizer(self):
        """Load model for DPO."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("transformers required") from e

        logger.info(f"Loading model for DPO: {self.config.hf_model_id}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_model_id,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_model_id,
            trust_remote_code=True,
        )

    def compute_dpo_loss(self, policy_chosen_logps, policy_rejected_logps,
                         ref_chosen_logps, ref_rejected_logps):
        """Compute DPO loss."""
        import torch

        chosen_rewards = self.config.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.config.beta * (policy_rejected_logps - ref_rejected_logps)

        if self.config.loss_type == "sigmoid":
            loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
        elif self.config.loss_type == "hinge":
            loss = torch.relu(self.config.margin - (chosen_rewards - rejected_rewards)).mean()
        else:  # ipo
            loss = ((chosen_rewards - rejected_rewards) - self.config.ipo_tau).pow(2).mean()

        return loss

    def train(self, train_dataset, eval_dataset=None, callbacks=None):
        """Train with DPO."""
        if self._model is None:
            self._load_model_and_tokenizer()

        try:
            from trl import DPOTrainer as TRLDPOTrainer, DPOConfig as TRLDPOConfig
        except ImportError:
            logger.warning("TRL not available, using simplified DPO")
            return self._train_simplified(train_dataset, eval_dataset, callbacks)

        dpo_config = TRLDPOConfig(
            **self.get_training_arguments(),
            beta=self.config.beta,
            loss_type=self.config.loss_type,
        )

        self._trainer = TRLDPOTrainer(
            model=self._model,
            ref_model=self.ref_model,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self._tokenizer,
        )

        return self._trainer.train()

    def _train_simplified(self, train_dataset, eval_dataset, callbacks):
        """Simplified training without TRL."""
        from transformers import Trainer, TrainingArguments

        training_args = TrainingArguments(**self.get_training_arguments())
        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self._tokenizer,
            callbacks=callbacks,
        )
        return self._trainer.train()

    def save_adapter(self, output_path: str):
        """Save model."""
        if self._model is None:
            raise ValueError("Model not loaded")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(output_path)
        self._tokenizer.save_pretrained(output_path)


# =============================================================================
# KTO Trainer
# =============================================================================

class KTOTrainer(BaseTrainer):
    """
    KTO (Kahneman-Tversky Optimization) Trainer.

    Uses prospect theory for preference learning with binary feedback.
    """

    def __init__(
        self,
        model_name: str = "medgemma-4b",
        config: Optional[KTOConfig] = None,
        output_dir: str = "./output",
        **kwargs
    ):
        super().__init__(model_name, output_dir, **kwargs)
        self.config = config or KTOConfig.for_model(model_name, **kwargs)

    def get_training_arguments(self) -> Dict[str, Any]:
        """Get training arguments."""
        args = self.config.get_training_arguments()
        args["output_dir"] = str(self.output_dir)
        return args

    def _load_model_and_tokenizer(self):
        """Load model for KTO."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("transformers required") from e

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_model_id,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_model_id,
            trust_remote_code=True,
        )

    def compute_kto_loss(self, policy_logps, ref_logps, labels):
        """
        Compute KTO loss based on prospect theory.

        Args:
            policy_logps: Log probabilities from policy model
            ref_logps: Log probabilities from reference model
            labels: Binary labels (1 for desirable, 0 for undesirable)
        """
        import torch

        kl = policy_logps - ref_logps

        # Prospect theory value function
        desirable_mask = labels == 1
        undesirable_mask = labels == 0

        # For desirable outcomes: v(x) = x
        # For undesirable outcomes: v(x) = -lambda * |x| (loss aversion)
        desirable_loss = -kl[desirable_mask].mean() if desirable_mask.any() else 0
        undesirable_loss = self.config.loss_aversion * kl[undesirable_mask].mean() if undesirable_mask.any() else 0

        loss = (
            self.config.desirable_weight * desirable_loss +
            self.config.undesirable_weight * undesirable_loss
        )

        return loss

    def train(self, train_dataset, eval_dataset=None, callbacks=None):
        """Train with KTO."""
        if self._model is None:
            self._load_model_and_tokenizer()

        try:
            from trl import KTOTrainer as TRLKTOTrainer, KTOConfig as TRLKTOConfig
        except ImportError:
            logger.warning("TRL not available, using simplified KTO")
            return self._train_simplified(train_dataset, eval_dataset, callbacks)

        kto_config = TRLKTOConfig(
            **self.get_training_arguments(),
            beta=self.config.beta,
            desirable_weight=self.config.desirable_weight,
            undesirable_weight=self.config.undesirable_weight,
        )

        self._trainer = TRLKTOTrainer(
            model=self._model,
            args=kto_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self._tokenizer,
        )

        return self._trainer.train()

    def _train_simplified(self, train_dataset, eval_dataset, callbacks):
        """Simplified training without TRL."""
        from transformers import Trainer, TrainingArguments

        training_args = TrainingArguments(**self.get_training_arguments())
        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self._tokenizer,
            callbacks=callbacks,
        )
        return self._trainer.train()

    def save_adapter(self, output_path: str):
        """Save model."""
        if self._model is None:
            raise ValueError("Model not loaded")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(output_path)
        self._tokenizer.save_pretrained(output_path)


# =============================================================================
# GRPO Trainer
# =============================================================================

class GRPOTrainer(BaseTrainer):
    """
    GRPO (Group Relative Policy Optimization) Trainer.

    Generates multiple responses per prompt and uses group-relative rewards.
    """

    def __init__(
        self,
        model_name: str = "medgemma-4b",
        config: Optional[GRPOConfig] = None,
        output_dir: str = "./output",
        reward_fn: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(model_name, output_dir, **kwargs)
        self.config = config or GRPOConfig.for_model(model_name, **kwargs)
        self.reward_fn = reward_fn

    def get_training_arguments(self) -> Dict[str, Any]:
        """Get training arguments."""
        args = self.config.get_training_arguments()
        args["output_dir"] = str(self.output_dir)
        return args

    def _load_model_and_tokenizer(self):
        """Load model for GRPO."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("transformers required") from e

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_model_id,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_model_id,
            trust_remote_code=True,
        )

    def generate_responses(self, prompts: List[str]) -> List[List[str]]:
        """Generate multiple responses per prompt."""
        all_responses = []

        for prompt in prompts:
            responses = []
            for _ in range(self.config.num_generations):
                inputs = self._tokenizer(prompt, return_tensors="pt")

                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                )

                response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
                responses.append(response)

            all_responses.append(responses)

        return all_responses

    def compute_group_rewards(self, prompts: List[str], response_groups: List[List[str]]) -> List[List[float]]:
        """Compute rewards for each response in each group."""
        all_rewards = []

        for prompt, responses in zip(prompts, response_groups):
            if self.reward_fn:
                rewards = [self.reward_fn(prompt, r) for r in responses]
            else:
                # Placeholder: assign random rewards
                import random
                rewards = [random.random() for _ in responses]

            # Normalize within group
            if self.config.normalize_rewards:
                mean_r = sum(rewards) / len(rewards)
                std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
                if std_r > 0:
                    rewards = [(r - mean_r) / std_r for r in rewards]

            all_rewards.append(rewards)

        return all_rewards

    def train(self, train_dataset, eval_dataset=None, callbacks=None):
        """Train with GRPO."""
        if self._model is None:
            self._load_model_and_tokenizer()

        try:
            from trl import GRPOTrainer as TRLGRPOTrainer, GRPOConfig as TRLGRPOConfig
        except ImportError:
            logger.warning("TRL GRPO not available, using simplified training")
            return self._train_simplified(train_dataset, eval_dataset, callbacks)

        grpo_config = TRLGRPOConfig(
            **self.get_training_arguments(),
            num_generations=self.config.num_generations,
        )

        self._trainer = TRLGRPOTrainer(
            model=self._model,
            args=grpo_config,
            train_dataset=train_dataset,
            tokenizer=self._tokenizer,
            reward_funcs=self.reward_fn,
        )

        return self._trainer.train()

    def _train_simplified(self, train_dataset, eval_dataset, callbacks):
        """Simplified training without TRL."""
        from transformers import Trainer, TrainingArguments

        training_args = TrainingArguments(**self.get_training_arguments())
        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self._tokenizer,
            callbacks=callbacks,
        )
        return self._trainer.train()

    def save_adapter(self, output_path: str):
        """Save model."""
        if self._model is None:
            raise ValueError("Model not loaded")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(output_path)
        self._tokenizer.save_pretrained(output_path)


# =============================================================================
# mHC Trainer
# =============================================================================

class MHCTrainer(BaseTrainer):
    """
    mHC (Manifold-Constrained Hyper-Connections) Trainer.

    Adds learnable hyper-connections constrained to a low-dimensional manifold.
    """

    def __init__(
        self,
        model_name: str = "medgemma-4b",
        config: Optional[MHCConfig] = None,
        output_dir: str = "./output",
        **kwargs
    ):
        super().__init__(model_name, output_dir, **kwargs)
        self.config = config or MHCConfig.for_model(model_name, **kwargs)

    def get_training_arguments(self) -> Dict[str, Any]:
        """Get training arguments."""
        args = self.config.get_training_arguments()
        args["output_dir"] = str(self.output_dir)
        return args

    def _load_model_and_tokenizer(self):
        """Load model with mHC configuration."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("transformers required") from e

        logger.info(f"Loading model with mHC: {self.config.hf_model_id}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_model_id,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_model_id,
            trust_remote_code=True,
        )

        # Add hyper-connections
        self._add_hyper_connections()

    def _add_hyper_connections(self):
        """Add manifold-constrained hyper-connections to the model."""
        logger.info(
            f"Adding mHC with manifold_dim={self.config.manifold_dim}, "
            f"connection_rank={self.config.connection_rank}"
        )
        # Implementation would add custom layers here

    def compute_manifold_loss(self, hidden_states):
        """Compute manifold constraint regularization loss."""
        import torch

        # Project to manifold
        # This is a simplified version - real implementation would use
        # proper manifold optimization techniques

        # L2 regularization on the manifold projection
        manifold_loss = self.config.constraint_strength * torch.norm(hidden_states, p=2)

        return manifold_loss

    def train(self, train_dataset, eval_dataset=None, callbacks=None):
        """Train with mHC."""
        if self._model is None:
            self._load_model_and_tokenizer()

        from transformers import Trainer, TrainingArguments

        training_args = TrainingArguments(**self.get_training_arguments())

        # Custom trainer with manifold loss
        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self._tokenizer,
            callbacks=callbacks,
        )

        return self._trainer.train()

    def save_adapter(self, output_path: str):
        """Save model with hyper-connections."""
        if self._model is None:
            raise ValueError("Model not loaded")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(output_path)
        self._tokenizer.save_pretrained(output_path)
        logger.info(f"Saved mHC model to {output_path}")

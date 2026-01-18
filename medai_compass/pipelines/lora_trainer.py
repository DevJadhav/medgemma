"""LoRA and QLoRA Training Module for MedGemma Models.

Provides LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) training
for MedGemma 4B IT and 27B IT models using PEFT and TRL.

Features:
- Model selection between 4B and 27B variants
- LoRA configuration with model-specific defaults
- QLoRA with 4-bit quantization for memory efficiency
- Gradient accumulation support
- Mixed precision training (bf16/fp16)
- Flash Attention 2 integration (Linux/CUDA only)
- TRL SFTTrainer integration for instruction tuning
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Model-specific configurations
MODEL_CONFIGS = {
    "medgemma-4b": {
        "hf_model_id": "google/medgemma-4b-it",
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "max_seq_length": 8192,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "medgemma-27b": {
        "hf_model_id": "google/medgemma-27b-it",
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1e-4,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "max_seq_length": 4096,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
}

# Model aliases
MODEL_ALIASES = {
    "4b": "medgemma-4b",
    "medgemma-4b-it": "medgemma-4b",
    "google/medgemma-4b-it": "medgemma-4b",
    "27b": "medgemma-27b",
    "medgemma-27b-it": "medgemma-27b",
    "google/medgemma-27b-it": "medgemma-27b",
}


def _resolve_model_name(model_name: str) -> str:
    """Resolve model alias to canonical name."""
    return MODEL_ALIASES.get(model_name.lower(), model_name.lower())


def _check_flash_attention_available() -> bool:
    """Check if Flash Attention 2 is available."""
    if sys.platform != "linux":
        return False
    
    try:
        import flash_attn
        return True
    except ImportError:
        return False


@dataclass
class LoRAConfig:
    """Configuration for LoRA training.
    
    Provides model-specific defaults for MedGemma 4B and 27B with support for:
    - LoRA rank and alpha parameters
    - Target modules for adaptation
    - Gradient accumulation
    - Mixed precision training
    - Flash Attention
    """
    
    # Model identification
    model_name: str = "medgemma-4b"
    hf_model_id: str = ""
    
    # LoRA parameters
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    # Training parameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_steps: int = 10000
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_length: int = 8192
    
    # Mixed precision
    mixed_precision: str = "bf16"
    bf16: bool = True
    fp16: bool = False
    
    # Flash Attention
    use_flash_attention: bool = False
    
    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 500
    
    def __post_init__(self):
        """Initialize model-specific defaults."""
        resolved_name = _resolve_model_name(self.model_name)
        
        if resolved_name in MODEL_CONFIGS:
            config = MODEL_CONFIGS[resolved_name]
            
            # Set HF model ID if not provided
            if not self.hf_model_id:
                self.hf_model_id = config["hf_model_id"]
            
            # Only set defaults if not explicitly provided
            if self.r == 16 and resolved_name == "medgemma-27b":
                self.r = config["lora_r"]
                self.lora_alpha = config["lora_alpha"]
            
            if self.batch_size == 4 and resolved_name == "medgemma-27b":
                self.batch_size = config["batch_size"]
                self.gradient_accumulation_steps = config["gradient_accumulation_steps"]
                self.learning_rate = config["learning_rate"]
                self.max_seq_length = config["max_seq_length"]
            
            # Update target modules
            self.target_modules = config["target_modules"]
        
        # Check Flash Attention availability
        if self.use_flash_attention is True:
            self.use_flash_attention = _check_flash_attention_available()
        elif sys.platform != "linux":
            self.use_flash_attention = False
    
    @classmethod
    def for_model(cls, model_name: str, **kwargs) -> "LoRAConfig":
        """Create LoRA config for a specific model.
        
        Args:
            model_name: Model name (medgemma-4b, medgemma-27b, or alias)
            **kwargs: Override any config parameters
            
        Returns:
            LoRAConfig with model-specific defaults
        """
        resolved_name = _resolve_model_name(model_name)
        
        if resolved_name not in MODEL_CONFIGS:
            logger.warning(f"Unknown model: {model_name}, using defaults")
            return cls(model_name=model_name, **kwargs)
        
        config = MODEL_CONFIGS[resolved_name]
        
        return cls(
            model_name=resolved_name,
            hf_model_id=config["hf_model_id"],
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["target_modules"],
            batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"],
            max_seq_length=config["max_seq_length"],
            **kwargs
        )
    
    def get_peft_config(self) -> Dict[str, Any]:
        """Get PEFT LoraConfig parameters.
        
        Returns:
            Dictionary for peft.LoraConfig
        """
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
        }
    
    def get_training_arguments(self) -> Dict[str, Any]:
        """Get TrainingArguments parameters.
        
        Returns:
            Dictionary for transformers.TrainingArguments
        """
        return {
            "per_device_train_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
        }


class LoRATrainer:
    """LoRA Trainer for MedGemma models.
    
    Provides full-precision LoRA training with TRL's SFTTrainer for
    instruction tuning of MedGemma 4B and 27B models.
    
    Example:
        >>> trainer = LoRATrainer(model_name="medgemma-4b")
        >>> trainer.train(train_dataset, eval_dataset)
        >>> trainer.save_adapter("./lora_adapters")
    """
    
    def __init__(
        self,
        model_name: str = "medgemma-4b",
        config: Optional[LoRAConfig] = None,
        output_dir: str = "./output",
        **kwargs
    ):
        """Initialize LoRA trainer.
        
        Args:
            model_name: Model name (medgemma-4b or medgemma-27b)
            config: Optional LoRAConfig (created from model_name if not provided)
            output_dir: Output directory for checkpoints and adapters
            **kwargs: Additional config overrides
        """
        self.model_name = _resolve_model_name(model_name)
        self.config = config or LoRAConfig.for_model(model_name, **kwargs)
        self.output_dir = Path(output_dir)
        
        self._model = None
        self._tokenizer = None
        self._trainer = None
    
    @classmethod
    def from_model_name(cls, model_name: str, **kwargs) -> "LoRATrainer":
        """Create trainer from model name.
        
        Args:
            model_name: Model name or alias
            **kwargs: Additional configuration
            
        Returns:
            LoRATrainer instance
        """
        return cls(model_name=model_name, **kwargs)
    
    def get_training_arguments(self) -> Dict[str, Any]:
        """Get training arguments dictionary.
        
        Returns:
            Training arguments for transformers.TrainingArguments
        """
        args = self.config.get_training_arguments()
        args["output_dir"] = str(self.output_dir)
        return args
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer with LoRA configuration."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig as PeftLoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError(
                "transformers and peft are required. "
                "Install with: pip install transformers peft"
            ) from e
        
        logger.info(f"Loading model: {self.config.hf_model_id}")
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_model_id,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        # Add Flash Attention if available
        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")
        
        # Set dtype based on mixed precision
        try:
            import torch
            if self.config.bf16:
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif self.config.fp16:
                model_kwargs["torch_dtype"] = torch.float16
        except ImportError:
            pass
        
        # Load base model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_model_id,
            **model_kwargs
        )
        
        # Apply LoRA
        peft_config = PeftLoraConfig(**self.config.get_peft_config())
        self._model = get_peft_model(self._model, peft_config)
        
        # Log trainable parameters
        trainable_params = sum(
            p.numel() for p in self._model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self._model.parameters())
        logger.info(
            f"Trainable params: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        callbacks: Optional[List] = None,
    ):
        """Train the model with LoRA.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            callbacks: Optional list of training callbacks
            
        Returns:
            Training result
        """
        if self._model is None:
            self._load_model_and_tokenizer()
        
        try:
            from trl import SFTTrainer, SFTConfig
        except ImportError:
            # Fallback to transformers Trainer
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
        
        # Use TRL SFTTrainer for instruction tuning
        sft_config = SFTConfig(
            **self.get_training_arguments(),
            max_seq_length=self.config.max_seq_length,
            packing=False,  # Disable packing for medical data
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
        """Save LoRA adapters.
        
        Args:
            output_path: Path to save adapters
        """
        if self._model is None:
            raise ValueError("Model not loaded. Call train() first.")
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self._model.save_pretrained(output_path)
        self._tokenizer.save_pretrained(output_path)
        
        logger.info(f"Saved LoRA adapters to {output_path}")
    
    def merge_and_save(self, output_path: str):
        """Merge LoRA weights into base model and save.
        
        Args:
            output_path: Path to save merged model
        """
        if self._model is None:
            raise ValueError("Model not loaded. Call train() first.")
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Merge LoRA weights
        merged_model = self._model.merge_and_unload()
        
        # Save merged model
        merged_model.save_pretrained(output_path)
        self._tokenizer.save_pretrained(output_path)
        
        logger.info(f"Saved merged model to {output_path}")


class QLoRATrainer(LoRATrainer):
    """QLoRA Trainer with 4-bit quantization.
    
    Extends LoRATrainer with 4-bit quantization for memory-efficient
    training of large models like MedGemma 27B on consumer GPUs.
    
    Example:
        >>> trainer = QLoRATrainer(model_name="medgemma-27b")
        >>> trainer.train(train_dataset)
    """
    
    def __init__(
        self,
        model_name: str = "medgemma-27b",
        config: Optional[LoRAConfig] = None,
        output_dir: str = "./output",
        load_in_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "bfloat16",
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        **kwargs
    ):
        """Initialize QLoRA trainer.
        
        Args:
            model_name: Model name
            config: Optional LoRAConfig
            output_dir: Output directory
            load_in_4bit: Use 4-bit quantization
            bnb_4bit_compute_dtype: Compute dtype for 4-bit
            bnb_4bit_quant_type: Quantization type (nf4 or fp4)
            bnb_4bit_use_double_quant: Use double quantization
            **kwargs: Additional config overrides
        """
        super().__init__(model_name, config, output_dir, **kwargs)
        
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
    
    @staticmethod
    def get_quantization_config() -> Dict[str, Any]:
        """Get default 4-bit quantization configuration.
        
        Returns:
            Dictionary for BitsAndBytesConfig
        """
        return {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
    
    def _get_bnb_config(self):
        """Get BitsAndBytes configuration for 4-bit quantization."""
        try:
            from transformers import BitsAndBytesConfig
            import torch
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            ) from e
        
        compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype, torch.bfloat16)
        
        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
        )
    
    def _load_model_and_tokenizer(self):
        """Load model with 4-bit quantization."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as e:
            raise ImportError(
                "transformers, peft, and bitsandbytes are required. "
                "Install with: pip install transformers peft bitsandbytes"
            ) from e
        
        logger.info(f"Loading model with 4-bit quantization: {self.config.hf_model_id}")
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_model_id,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Model loading kwargs with quantization
        model_kwargs = {
            "trust_remote_code": True,
            "quantization_config": self._get_bnb_config(),
            "device_map": "auto",
        }
        
        # Add Flash Attention if available
        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")
        
        # Load quantized model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_model_id,
            **model_kwargs
        )
        
        # Prepare for k-bit training
        self._model = prepare_model_for_kbit_training(self._model)
        
        # Apply LoRA
        peft_config = PeftLoraConfig(**self.config.get_peft_config())
        self._model = get_peft_model(self._model, peft_config)
        
        # Log trainable parameters
        trainable_params = sum(
            p.numel() for p in self._model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self._model.parameters())
        logger.info(
            f"QLoRA trainable params: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

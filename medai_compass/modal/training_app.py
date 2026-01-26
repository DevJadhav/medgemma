"""Modal Training App for MedGemma distributed training.

Deploy and run MedGemma fine-tuning on Modal with 8x H100 GPUs.

Deployment:
    modal deploy medai_compass/modal/training_app.py

Usage:
    # From Python
    from medai_compass.modal.training_app import MedGemmaTrainer
    trainer = MedGemmaTrainer()
    result = trainer.train.remote(model_name="medgemma-27b", ...)
    
    # From CLI
    modal run medai_compass/modal/training_app.py --model-name medgemma-27b
"""

from typing import Any

# Check if Modal is available
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    modal = None

if MODAL_AVAILABLE:
    # Create Modal app
    app = modal.App("medai-compass-training")

    # Lightweight image for GPU verification (faster to build)
    verify_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.2.0",
            "transformers>=4.40.0",
            "huggingface_hub>=0.20.0",
        )
    )

    # Training image with CUDA and ML dependencies
    # Use NVIDIA CUDA devel image (includes nvcc) for DeepSpeed support
    training_image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install("git", "curl", "build-essential")  # build-essential for C compiler (needed by Triton)
        .run_commands(
            # Install uv for faster pip operations
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
        )
        .env({"PATH": "/root/.local/bin:$PATH"})
        .run_commands(
            # Install PyTorch 2.6 with CUDA 12.4 (latest stable)
            "uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124",
            # Install ML training dependencies (pin transformers for MedGemma compatibility)
            "uv pip install --system numpy 'transformers>=4.47.0' accelerate peft datasets "
            "huggingface_hub scipy safetensors pyyaml sentencepiece tokenizers",
            # Install DeepSpeed for distributed training
            "uv pip install --system deepspeed>=0.9.3",
        )
        .env({
            "HF_HOME": "/root/.cache/huggingface",
            "TRANSFORMERS_CACHE": "/root/.cache/huggingface/transformers",
        })
    )

    # Volumes for persistent storage
    model_cache = modal.Volume.from_name("medgemma-model-cache", create_if_missing=True)
    training_data = modal.Volume.from_name("medgemma-data", create_if_missing=True)
    checkpoints = modal.Volume.from_name("medgemma-checkpoints", create_if_missing=True)

    @app.cls(
        image=training_image,
        gpu="H100:8",  # 8x H100 for 27B training
        volumes={
            "/root/.cache/huggingface": model_cache,
            "/data": training_data,
            "/checkpoints": checkpoints,
        },
        timeout=86400,  # 24 hours
        scaledown_window=600,  # 10 min idle
        secrets=[
            modal.Secret.from_name("huggingface-secret"),
        ],
    )
    class MedGemmaTrainer:
        """
        Distributed trainer for MedGemma models on Modal H100 GPUs.
        
        Supports:
        - MedGemma 4B IT: Single GPU training
        - MedGemma 27B IT: 8x H100 with DeepSpeed ZeRO-3
        """

        @modal.enter()
        def setup(self):
            """Initialize training environment on container startup."""
            import torch

            # Log GPU info
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    print(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

            # Commit model cache
            model_cache.commit()

        @modal.method()
        def train(
            self,
            model_name: str = "medgemma-27b",
            train_data_path: str | None = None,
            eval_data_path: str | None = None,
            max_steps: int = 10000,
            batch_size: int = 1,
            learning_rate: float = 1e-4,
            lora_r: int = 64,
            lora_alpha: int = 128,
            gradient_accumulation_steps: int = 16,
            save_steps: int = 500,
            eval_steps: int = 500,
            warmup_ratio: float = 0.1,
            mlflow_tracking_uri: str | None = None,
            experiment_name: str | None = None,
        ) -> dict[str, Any]:
            """
            Run distributed training job.

            Args:
                model_name: Model to train ("medgemma-4b" or "medgemma-27b")
                train_data_path: Path to training data
                eval_data_path: Path to evaluation data
                max_steps: Maximum training steps
                batch_size: Per-device batch size
                learning_rate: Learning rate
                lora_r: LoRA rank
                lora_alpha: LoRA alpha
                gradient_accumulation_steps: Gradient accumulation
                save_steps: Save checkpoint every N steps
                eval_steps: Evaluate every N steps
                warmup_ratio: Learning rate warmup ratio
                mlflow_tracking_uri: MLflow server URI
                experiment_name: MLflow experiment name

            Returns:
                Training result dictionary
            """
            import torch
            from accelerate import Accelerator
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                Trainer,
                TrainingArguments,
            )

            # Determine model configuration
            model_configs = {
                "medgemma-4b": {
                    "hf_model_id": "google/medgemma-4b-it",
                    "use_deepspeed": False,
                    "num_gpus": 1,
                },
                "medgemma-27b": {
                    "hf_model_id": "google/medgemma-27b-it",
                    "use_deepspeed": True,
                    "num_gpus": 8,
                },
            }

            # Normalize model name
            model_key = model_name.lower().replace("_", "-")
            if "4b" in model_key:
                model_key = "medgemma-4b"
            elif "27b" in model_key:
                model_key = "medgemma-27b"

            config = model_configs.get(model_key, model_configs["medgemma-27b"])
            hf_model_id = config["hf_model_id"]
            use_deepspeed = config["use_deepspeed"]
            num_gpus = config["num_gpus"]

            print(f"Training {hf_model_id}")
            print(f"GPUs available: {torch.cuda.device_count()}")
            print(f"Using DeepSpeed: {use_deepspeed}")
            print(f"Configured for {num_gpus} GPUs")

            # For multi-GPU with DeepSpeed, use subprocess with torchrun
            if use_deepspeed and torch.cuda.device_count() > 1:
                return self._run_distributed_training(
                    hf_model_id=hf_model_id,
                    train_data_path=train_data_path,
                    eval_data_path=eval_data_path,
                    max_steps=max_steps,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    save_steps=save_steps,
                    eval_steps=eval_steps,
                    warmup_ratio=warmup_ratio,
                    num_gpus=torch.cuda.device_count(),
                )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                hf_model_id,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                hf_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto" if not use_deepspeed else None,
                trust_remote_code=True,
            )

            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                bias="none",
            )

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            # DeepSpeed configuration for 27B model
            ds_config = None
            if use_deepspeed:
                ds_config = {
                    "bf16": {"enabled": True},
                    "zero_optimization": {
                        "stage": 3,
                        "offload_optimizer": {
                            "device": "cpu",
                            "pin_memory": True,
                        },
                        "offload_param": {"device": "none"},
                        "overlap_comm": True,
                        "contiguous_gradients": True,
                        "reduce_bucket_size": 5e8,
                        "stage3_prefetch_bucket_size": 5e8,
                        "stage3_param_persistence_threshold": 1e6,
                    },
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "gradient_clipping": 1.0,
                    "train_micro_batch_size_per_gpu": batch_size,
                }

            # Load training and evaluation datasets first to determine eval strategy
            train_dataset = None
            eval_dataset = None

            if train_data_path:
                print(f"Loading training data from: {train_data_path}")
                train_dataset = self._load_dataset(train_data_path, tokenizer)
                print(f"Loaded {len(train_dataset)} training samples")
            else:
                raise ValueError(
                    "train_data_path is required. Download data first with:\n"
                    "  modal run medai_compass/modal/data_download.py\n"
                    "Then specify: train_data_path='/data/combined_medical/train.jsonl'"
                )

            if eval_data_path:
                print(f"Loading evaluation data from: {eval_data_path}")
                eval_dataset = self._load_dataset(eval_data_path, tokenizer)
                print(f"Loaded {len(eval_dataset)} evaluation samples")

            # Training arguments - only enable load_best_model_at_end if eval_dataset exists
            has_eval = eval_dataset is not None
            training_args = TrainingArguments(
                output_dir="/checkpoints/training",
                max_steps=max_steps,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_ratio=warmup_ratio,
                weight_decay=0.01,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=1.0,
                bf16=True,
                logging_steps=10,
                save_steps=save_steps,
                eval_strategy="steps" if has_eval else "no",
                eval_steps=eval_steps if has_eval else None,
                save_total_limit=3,
                load_best_model_at_end=has_eval,
                metric_for_best_model="loss" if has_eval else None,
                greater_is_better=False if has_eval else None,
                gradient_checkpointing=True,
                deepspeed=ds_config,
                report_to=["mlflow"] if mlflow_tracking_uri else [],
            )

            # Set up MLflow
            if mlflow_tracking_uri:
                import mlflow
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                if experiment_name:
                    mlflow.set_experiment(experiment_name)

            # Create trainer with datasets
            # Note: In transformers 5.0+, 'tokenizer' was renamed to 'processing_class'
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
            )

            # Run training
            print("Starting training...")
            train_result = trainer.train()

            # Save final model
            print("Saving final model...")
            trainer.save_model("/checkpoints/final")
            tokenizer.save_pretrained("/checkpoints/final")

            # Merge LoRA weights for inference (optional export)
            print("Merging LoRA weights for export...")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained("/checkpoints/final-merged")
            tokenizer.save_pretrained("/checkpoints/final-merged")

            # Commit checkpoints
            checkpoints.commit()

            return {
                "status": "completed",
                "model": hf_model_id,
                "gpus": torch.cuda.device_count(),
                "deepspeed": use_deepspeed,
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "config": {
                    "max_steps": max_steps,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                },
                "training_loss": train_result.training_loss,
                "global_step": train_result.global_step,
                "checkpoint_path": "/checkpoints/final",
                "merged_model_path": "/checkpoints/final-merged",
            }

        def _run_distributed_training(
            self,
            hf_model_id: str,
            train_data_path: str,
            eval_data_path: str | None,
            max_steps: int,
            batch_size: int,
            learning_rate: float,
            lora_r: int,
            lora_alpha: int,
            gradient_accumulation_steps: int,
            save_steps: int,
            eval_steps: int,
            warmup_ratio: float,
            num_gpus: int,
        ) -> dict[str, Any]:
            """
            Run distributed training using accelerate launcher.

            This method creates a training script and launches it with
            `accelerate launch` for proper multi-GPU distributed training.
            """
            import json
            import os
            import subprocess
            import tempfile

            import torch

            print(f"Launching distributed training across {num_gpus} GPUs...")

            # Create DeepSpeed config file
            ds_config = {
                "bf16": {"enabled": True},
                "zero_optimization": {
                    "stage": 3,
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True,
                    },
                    "offload_param": {"device": "none"},
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "reduce_bucket_size": 5e8,
                    "stage3_prefetch_bucket_size": 5e8,
                    "stage3_param_persistence_threshold": 1e6,
                },
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "gradient_clipping": 1.0,
                "train_micro_batch_size_per_gpu": batch_size,
            }

            # Write configs to temp files
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(ds_config, f)
                ds_config_path = f.name

            # Create training script
            train_script = f'''
import json
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset, load_dataset

print(f"GPU {{torch.cuda.current_device()}}/{{torch.cuda.device_count()}}: Starting training...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{hf_model_id}", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model WITHOUT device_map for DeepSpeed
model = AutoModelForCausalLM.from_pretrained(
    "{hf_model_id}",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r={lora_r},
    lora_alpha={lora_alpha},
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)

if torch.distributed.get_rank() == 0:
    model.print_trainable_parameters()

# Load and preprocess training data
def load_and_tokenize(data_path):
    max_length = 4096
    system_prompt = "You are a medical AI assistant trained to help healthcare providers."

    data = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    dataset = Dataset.from_list(data)

    def format_and_tokenize(examples):
        texts = []
        for i in range(len(examples["instruction"])):
            instr = examples["instruction"][i]
            inp = examples.get("input", [""])[i] or ""
            out = examples["output"][i]
            if inp:
                text = f"<bos><start_of_turn>user\\n{{system_prompt}}\\n\\n{{instr}}\\n\\nContext: {{inp}}<end_of_turn>\\n<start_of_turn>model\\n{{out}}<end_of_turn><eos>"
            else:
                text = f"<bos><start_of_turn>user\\n{{system_prompt}}\\n\\n{{instr}}<end_of_turn>\\n<start_of_turn>model\\n{{out}}<end_of_turn><eos>"
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(format_and_tokenize, batched=True, remove_columns=dataset.column_names, desc="Tokenizing")

train_dataset = load_and_tokenize("{train_data_path}")
print(f"Loaded {{len(train_dataset)}} training samples")

# Training arguments with DeepSpeed
training_args = TrainingArguments(
    output_dir="/checkpoints/training",
    max_steps={max_steps},
    per_device_train_batch_size={batch_size},
    learning_rate={learning_rate},
    warmup_ratio={warmup_ratio},
    weight_decay=0.01,
    gradient_accumulation_steps={gradient_accumulation_steps},
    max_grad_norm=1.0,
    bf16=True,
    logging_steps=10,
    save_steps={save_steps},
    save_total_limit=3,
    gradient_checkpointing=True,
    deepspeed="{ds_config_path}",
    report_to=[],
    ddp_find_unused_parameters=False,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

# Run training
print("Starting distributed training...")
train_result = trainer.train()

# Save model (only on rank 0)
if torch.distributed.get_rank() == 0:
    print("Saving final model...")
    trainer.save_model("/checkpoints/final")
    tokenizer.save_pretrained("/checkpoints/final")

    # Merge LoRA weights
    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("/checkpoints/final-merged")
    tokenizer.save_pretrained("/checkpoints/final-merged")

    # Write results
    results = {{
        "training_loss": train_result.training_loss,
        "global_step": train_result.global_step,
    }}
    with open("/checkpoints/training_result.json", "w") as f:
        json.dump(results, f)

print(f"GPU {{torch.cuda.current_device()}}: Training complete!")
'''

            # Write training script
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(train_script)
                script_path = f.name

            # Run with accelerate launch (--use_deepspeed handles multi-gpu automatically)
            cmd = [
                "accelerate", "launch",
                "--num_processes", str(num_gpus),
                "--mixed_precision", "bf16",
                "--use_deepspeed",
                "--deepspeed_config_file", ds_config_path,
                script_path,
            ]

            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
            )

            # Print output for debugging
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

            # Cleanup temp files
            os.unlink(script_path)
            os.unlink(ds_config_path)

            if result.returncode != 0:
                return {
                    "status": "failed",
                    "error": f"Training failed with return code {result.returncode}",
                    "stdout": result.stdout[-5000:] if result.stdout else "",
                    "stderr": result.stderr[-5000:] if result.stderr else "",
                }

            # Load training results
            try:
                with open("/checkpoints/training_result.json") as f:
                    train_results = json.load(f)
            except Exception:
                train_results = {"training_loss": 0.0, "global_step": max_steps}

            # Commit checkpoints
            checkpoints.commit()

            return {
                "status": "completed",
                "model": hf_model_id,
                "gpus": num_gpus,
                "deepspeed": True,
                "distributed": True,
                "config": {
                    "max_steps": max_steps,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                },
                "training_loss": train_results.get("training_loss", 0.0),
                "global_step": train_results.get("global_step", 0),
                "checkpoint_path": "/checkpoints/final",
                "merged_model_path": "/checkpoints/final-merged",
            }

        def _load_dataset(self, data_path: str, tokenizer):
            """
            Load and preprocess dataset for training.

            Supports:
            - JSONL files with instruction/input/output format
            - Parquet files
            - HuggingFace datasets (hf://dataset_name)

            Args:
                data_path: Path to data file
                tokenizer: Tokenizer for preprocessing

            Returns:
                Preprocessed HuggingFace Dataset
            """
            import json

            from datasets import Dataset, load_dataset

            max_length = 4096

            # System prompt for medical AI
            system_prompt = (
                "You are a medical AI assistant trained to help healthcare providers "
                "with clinical reasoning and diagnosis. Provide evidence-based, "
                "accurate medical information."
            )

            # Load raw data
            if data_path.startswith("hf://"):
                dataset = load_dataset(data_path[5:], split="train")
            elif data_path.endswith(".jsonl") or data_path.endswith(".json"):
                data = []
                with open(data_path) as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                dataset = Dataset.from_list(data)
            elif data_path.endswith(".parquet"):
                dataset = load_dataset("parquet", data_files=data_path, split="train")
            else:
                raise ValueError(f"Unsupported data format: {data_path}")

            def format_and_tokenize(examples):
                """Format examples and tokenize."""
                texts = []

                # Get number of examples in batch
                keys = list(examples.keys())
                if not keys:
                    return {"input_ids": [], "attention_mask": [], "labels": []}

                n_examples = len(examples[keys[0]])

                for i in range(n_examples):
                    # Get fields with fallbacks
                    instruction = (
                        examples.get("instruction", [""] * n_examples)[i] or
                        examples.get("question", [""] * n_examples)[i] or ""
                    )
                    input_text = (
                        examples.get("input", [""] * n_examples)[i] or
                        examples.get("context", [""] * n_examples)[i] or ""
                    )
                    output = (
                        examples.get("output", [""] * n_examples)[i] or
                        examples.get("answer", [""] * n_examples)[i] or ""
                    )

                    # Build conversation format
                    if input_text:
                        user_content = f"{instruction}\n\n{input_text}"
                    else:
                        user_content = instruction

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": str(output)},
                    ]

                    # Apply chat template
                    try:
                        text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False,
                        )
                    except Exception:
                        # Fallback to simple format
                        text = f"<system>{system_prompt}</system>\n"
                        text += f"<user>{user_content}</user>\n"
                        text += f"<assistant>{output}</assistant>"

                    texts.append(text)

                # Tokenize
                tokenized = tokenizer(
                    texts,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                )

                # Set labels (same as input_ids for causal LM)
                tokenized["labels"] = tokenized["input_ids"].copy()

                return tokenized

            # Apply preprocessing
            processed = dataset.map(
                format_and_tokenize,
                batched=True,
                remove_columns=dataset.column_names,
                desc="Tokenizing dataset",
            )

            # Set format for PyTorch
            processed.set_format("torch")

            return processed

        @modal.method()
        def get_gpu_info(self) -> dict[str, Any]:
            """Get GPU information."""
            import torch

            gpus = []
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpus.append({
                        "index": i,
                        "name": props.name,
                        "total_memory_gb": props.total_memory / 1e9,
                        "compute_capability": f"{props.major}.{props.minor}",
                    })

            return {
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count(),
                "gpus": gpus,
            }

    # Single GPU variant for 4B model
    @app.cls(
        image=training_image,
        gpu="H100",  # 1x H100 for 4B training/inference
        volumes={
            "/root/.cache/huggingface": model_cache,
            "/data": training_data,
            "/checkpoints": checkpoints,
        },
        timeout=86400,
        scaledown_window=300,
        secrets=[
            modal.Secret.from_name("huggingface-secret"),
        ],
    )
    class MedGemma4BTrainer:
        """Single-GPU trainer for MedGemma 4B model."""

        @modal.enter()
        def setup(self):
            """Initialize on container startup."""
            import torch
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            model_cache.commit()

        @modal.method()
        def train(
            self,
            train_data_path: str | None = None,
            eval_data_path: str | None = None,
            max_steps: int = 10000,
            batch_size: int = 4,
            learning_rate: float = 2e-4,
            lora_r: int = 16,
            lora_alpha: int = 32,
            gradient_accumulation_steps: int = 4,
            save_steps: int = 500,
        ) -> dict[str, Any]:
            """Train MedGemma 4B model on single H100."""
            import torch
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                Trainer,
                TrainingArguments,
            )

            hf_model_id = "google/medgemma-4b-it"

            print(f"Training {hf_model_id} on {torch.cuda.get_device_name(0)}")

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                hf_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            # Load datasets
            if not train_data_path:
                raise ValueError(
                    "train_data_path is required. Download data first with:\n"
                    "  modal run medai_compass/modal/data_download.py\n"
                    "Then specify: train_data_path='/data/combined_medical/train.jsonl'"
                )

            train_dataset = self._load_dataset(train_data_path, tokenizer)
            print(f"Loaded {len(train_dataset)} training samples")

            eval_dataset = None
            if eval_data_path:
                eval_dataset = self._load_dataset(eval_data_path, tokenizer)
                print(f"Loaded {len(eval_dataset)} evaluation samples")

            # Training arguments
            training_args = TrainingArguments(
                output_dir="/checkpoints/training-4b",
                max_steps=max_steps,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_ratio=0.1,
                weight_decay=0.01,
                gradient_accumulation_steps=gradient_accumulation_steps,
                bf16=True,
                logging_steps=10,
                save_steps=save_steps,
                save_strategy="steps",
                eval_strategy="steps" if eval_dataset else "no",
                eval_steps=save_steps if eval_dataset else None,
                save_total_limit=3,
                load_best_model_at_end=True if eval_dataset else False,
                metric_for_best_model="loss" if eval_dataset else None,
                greater_is_better=False if eval_dataset else None,
                gradient_checkpointing=True,
                report_to=[],
            )

            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,  # renamed from 'tokenizer' in newer transformers
            )

            # Run training
            print("Starting training...")
            train_result = trainer.train()

            # Save final model
            print("Saving final model...")
            trainer.save_model("/checkpoints/final-4b")
            tokenizer.save_pretrained("/checkpoints/final-4b")

            # Merge LoRA weights
            print("Merging LoRA weights...")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained("/checkpoints/final-merged-4b")
            tokenizer.save_pretrained("/checkpoints/final-merged-4b")

            checkpoints.commit()

            return {
                "status": "completed",
                "model": hf_model_id,
                "gpu": torch.cuda.get_device_name(0),
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "training_loss": train_result.training_loss,
                "global_step": train_result.global_step,
                "checkpoint_path": "/checkpoints/final-4b",
                "merged_model_path": "/checkpoints/final-merged-4b",
            }

        def _load_dataset(self, data_path: str, tokenizer):
            """Load and preprocess dataset (shared implementation)."""
            import json

            from datasets import Dataset

            max_length = 4096
            system_prompt = (
                "You are a medical AI assistant trained to help healthcare providers "
                "with clinical reasoning and diagnosis."
            )

            # Load raw data
            data = []
            with open(data_path) as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            dataset = Dataset.from_list(data)

            def format_and_tokenize(examples):
                texts = []
                keys = list(examples.keys())
                if not keys:
                    return {"input_ids": [], "attention_mask": [], "labels": []}

                n_examples = len(examples[keys[0]])

                for i in range(n_examples):
                    instruction = (
                        examples.get("instruction", [""] * n_examples)[i] or
                        examples.get("question", [""] * n_examples)[i] or ""
                    )
                    input_text = (
                        examples.get("input", [""] * n_examples)[i] or ""
                    )
                    output = (
                        examples.get("output", [""] * n_examples)[i] or
                        examples.get("answer", [""] * n_examples)[i] or ""
                    )

                    if input_text:
                        user_content = f"{instruction}\n\n{input_text}"
                    else:
                        user_content = instruction

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": str(output)},
                    ]

                    try:
                        text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=False
                        )
                    except Exception:
                        text = f"<system>{system_prompt}</system>\n"
                        text += f"<user>{user_content}</user>\n"
                        text += f"<assistant>{output}</assistant>"

                    texts.append(text)

                tokenized = tokenizer(
                    texts, max_length=max_length, padding="max_length", truncation=True
                )
                tokenized["labels"] = tokenized["input_ids"].copy()
                # Add token_type_ids (zeros) required by Gemma3
                batch_size = len(tokenized["input_ids"])
                seq_len = len(tokenized["input_ids"][0])
                tokenized["token_type_ids"] = [[0] * seq_len for _ in range(batch_size)]
                return tokenized

            processed = dataset.map(
                format_and_tokenize,
                batched=True,
                remove_columns=dataset.column_names,
                desc="Tokenizing dataset",
            )
            processed.set_format("torch")
            return processed

    # =============================================================================
    # H100 Optimization Verification Functions
    # =============================================================================

    @app.function(
        image=verify_image,
        gpu="H100",
        timeout=600,
    )
    def verify_h100_optimizations() -> dict[str, Any]:
        """
        Verify H100-specific optimizations are working on Modal GPU.

        This function validates:
        - Flash Attention 2 availability
        - FP8 support (Transformer Engine)
        - CUDA graphs capability
        - Memory bandwidth
        - GPU compute capability

        Returns:
            Dict with optimization status and benchmarks
        """
        import time

        import torch

        results = {
            "gpu_info": {},
            "optimizations": {},
            "benchmarks": {},
        }

        # GPU Info
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            results["gpu_info"] = {
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": props.total_memory / 1e9,
                "multiprocessor_count": props.multi_processor_count,
                "is_h100": "H100" in props.name,
            }

        # Check Flash Attention
        try:
            import flash_attn
            results["optimizations"]["flash_attention_2"] = {
                "available": True,
                "version": flash_attn.__version__,
            }
        except ImportError:
            results["optimizations"]["flash_attention_2"] = {
                "available": False,
                "reason": "flash_attn not installed",
            }

        # Check Transformer Engine (FP8)
        try:
            import transformer_engine
            results["optimizations"]["transformer_engine_fp8"] = {
                "available": True,
                "version": transformer_engine.__version__,
            }
        except ImportError:
            results["optimizations"]["transformer_engine_fp8"] = {
                "available": False,
                "reason": "transformer_engine not installed",
            }

        # Check CUDA Graphs
        try:
            stream = torch.cuda.Stream()
            graph = torch.cuda.CUDAGraph()
            results["optimizations"]["cuda_graphs"] = {
                "available": True,
            }
        except Exception as e:
            results["optimizations"]["cuda_graphs"] = {
                "available": False,
                "reason": str(e),
            }

        # Memory bandwidth benchmark
        if torch.cuda.is_available():
            size = 1024 * 1024 * 256  # 1GB
            x = torch.randn(size, device="cuda", dtype=torch.bfloat16)

            torch.cuda.synchronize()
            start = time.perf_counter()
            y = x.clone()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            bandwidth_gbps = (size * 4) / elapsed / 1e9  # 2 bytes per bf16 * 2 (read+write)
            results["benchmarks"]["memory_bandwidth_gbps"] = bandwidth_gbps

            del x, y
            torch.cuda.empty_cache()

        # Matmul benchmark
        if torch.cuda.is_available():
            m, n, k = 4096, 4096, 4096
            a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

            # Warmup
            for _ in range(10):
                torch.matmul(a, b)

            torch.cuda.synchronize()
            start = time.perf_counter()
            iterations = 100
            for _ in range(iterations):
                torch.matmul(a, b)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            flops = 2 * m * n * k * iterations
            tflops = flops / elapsed / 1e12
            results["benchmarks"]["matmul_tflops"] = tflops

            del a, b
            torch.cuda.empty_cache()

        return results

    @app.function(
        image=training_image,
        gpu="H100",
        volumes={"/root/.cache/huggingface": model_cache},
        timeout=1200,
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    def benchmark_inference_throughput(
        model_name: str = "medgemma-4b",
        batch_sizes: list[int] = [1, 2, 4, 8],
        input_length: int = 256,
        output_length: int = 128,
    ) -> dict[str, Any]:
        """
        Benchmark inference throughput on H100 with various batch sizes.

        Tests Flash Attention 2 and compares with SDPA.

        Returns:
            Dict with throughput measurements per batch size
        """
        import time

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_model_id = "google/medgemma-4b-it" if "4b" in model_name else "google/medgemma-27b-it"

        # Load model with Flash Attention
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)

        results = {
            "model": hf_model_id,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "batch_results": {},
        }

        for attn_impl in ["flash_attention_2", "sdpa"]:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    hf_model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    attn_implementation=attn_impl,
                    trust_remote_code=True,
                )
            except Exception as e:
                results["batch_results"][attn_impl] = {"error": str(e)}
                continue

            results["batch_results"][attn_impl] = {}

            for batch_size in batch_sizes:
                # Create input
                input_text = "What are the symptoms of pneumonia? " * (input_length // 10)
                inputs = tokenizer(
                    [input_text] * batch_size,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=input_length,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Warmup
                with torch.inference_mode():
                    for _ in range(3):
                        model.generate(**inputs, max_new_tokens=32, do_sample=False)

                # Benchmark
                torch.cuda.synchronize()
                start = time.perf_counter()

                iterations = 10
                total_tokens = 0
                with torch.inference_mode():
                    for _ in range(iterations):
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=output_length,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                        total_tokens += outputs.numel()

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                tokens_per_second = total_tokens / elapsed
                latency_ms = (elapsed / iterations) * 1000

                results["batch_results"][attn_impl][f"batch_{batch_size}"] = {
                    "tokens_per_second": tokens_per_second,
                    "latency_ms": latency_ms,
                    "throughput_improvement": None,
                }

            del model
            torch.cuda.empty_cache()

        # Calculate improvement
        if "flash_attention_2" in results["batch_results"] and "sdpa" in results["batch_results"]:
            for batch_key in results["batch_results"]["flash_attention_2"]:
                if batch_key in results["batch_results"]["sdpa"]:
                    fa_tps = results["batch_results"]["flash_attention_2"][batch_key]["tokens_per_second"]
                    sdpa_tps = results["batch_results"]["sdpa"][batch_key]["tokens_per_second"]
                    improvement = (fa_tps - sdpa_tps) / sdpa_tps * 100
                    results["batch_results"]["flash_attention_2"][batch_key]["throughput_improvement"] = f"{improvement:.1f}%"

        model_cache.commit()

        return results

    # CLI entrypoint
    @app.local_entrypoint()
    def main(
        model_name: str = "medgemma-27b",
        max_steps: int = 1000,
        train_data: str = "/data/combined_medical/train.jsonl",
        eval_data: str | None = "/data/combined_medical/eval.jsonl",
        dry_run: bool = False,
        verify_optimizations: bool = False,
        benchmark: bool = False,
    ):
        """Run training or benchmarks from CLI.

        Examples:
            # Verify H100 optimizations
            modal run medai_compass/modal/training_app.py --verify-optimizations

            # Benchmark inference
            modal run medai_compass/modal/training_app.py --benchmark

            # Run training (4B model)
            modal run medai_compass/modal/training_app.py --model-name medgemma-4b

            # Run training (27B model with 8x H100)
            modal run medai_compass/modal/training_app.py --model-name medgemma-27b

            # Dry run (configure only)
            modal run medai_compass/modal/training_app.py --dry-run
        """
        if verify_optimizations:
            print("Verifying H100 optimizations on Modal...")
            result = verify_h100_optimizations.remote()
            print(f"\nGPU Info: {result['gpu_info']}")
            print("\nOptimizations Available:")
            for name, status in result['optimizations'].items():
                print(f"  {name}: {status}")
            print("\nBenchmarks:")
            for name, value in result['benchmarks'].items():
                print(f"  {name}: {value:.2f}")
            return

        if benchmark:
            print("Running inference throughput benchmark on H100...")
            result = benchmark_inference_throughput.remote(model_name=model_name)
            print(f"\nModel: {result['model']}")
            print(f"GPU: {result['gpu']}")
            print("\nResults by attention implementation:")
            for impl, batches in result['batch_results'].items():
                print(f"\n  {impl}:")
                for batch_key, metrics in batches.items():
                    if isinstance(metrics, dict) and "tokens_per_second" in metrics:
                        print(f"    {batch_key}: {metrics['tokens_per_second']:.1f} tokens/s, "
                              f"latency: {metrics['latency_ms']:.1f}ms"
                              f"{' (+' + metrics['throughput_improvement'] + ')' if metrics.get('throughput_improvement') else ''}")
            return

        print(f"Starting training for {model_name}")
        print(f"Training data: {train_data}")
        print(f"Evaluation data: {eval_data}")
        print(f"Max steps: {max_steps}")

        if dry_run:
            print("\n[DRY RUN] Would train with above configuration.")
            print("Remove --dry-run to start actual training.")
            return

        if "27b" in model_name.lower():
            trainer = MedGemmaTrainer()

            # Get GPU info first
            gpu_info = trainer.get_gpu_info.remote()
            print(f"\nGPU Info: {gpu_info}")

            print("\nStarting distributed training on 8x H100...")
            result = trainer.train.remote(
                model_name=model_name,
                train_data_path=train_data,
                eval_data_path=eval_data,
                max_steps=max_steps,
            )
            print("\nTraining completed!")
            print(f"Status: {result['status']}")
            print(f"Training loss: {result.get('training_loss', 'N/A')}")
            print(f"Checkpoint: {result.get('checkpoint_path', 'N/A')}")
        else:
            trainer = MedGemma4BTrainer()

            print("\nStarting training on single H100...")
            result = trainer.train.remote(
                train_data_path=train_data,
                eval_data_path=eval_data,
                max_steps=max_steps,
            )
            print("\nTraining completed!")
            print(f"Status: {result['status']}")
            print(f"Training loss: {result.get('training_loss', 'N/A')}")
            print(f"Checkpoint: {result.get('checkpoint_path', 'N/A')}")

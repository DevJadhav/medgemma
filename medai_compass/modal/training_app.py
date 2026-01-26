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

import os
from typing import Any, Dict, List, Optional

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
    
    # Training image with CUDA and ML dependencies
    training_image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "curl", "build-essential")
        .pip_install(
            # PyTorch with CUDA
            "torch>=2.2.0",
            # ML/Training
            "transformers>=4.40.0",
            "accelerate>=0.28.0",
            "bitsandbytes>=0.42.0",
            "peft>=0.10.0",
            "datasets>=2.17.0",
            # Distributed training
            "deepspeed>=0.13.0",
            "ray[train,tune]==2.9.0",
            # Experiment tracking
            "mlflow>=2.10.0",
            # Flash Attention
            "flash-attn>=2.5.0",
            # Utilities
            "huggingface_hub>=0.20.0",
            "scipy>=1.12.0",
            "safetensors>=0.4.0",
            "pyyaml>=6.0.0",
            "minio>=7.2.0",
        )
        .env({
            "HF_HOME": "/root/.cache/huggingface",
            "TRANSFORMERS_CACHE": "/root/.cache/huggingface/transformers",
        })
    )
    
    # Volumes for persistent storage
    model_cache = modal.Volume.from_name("medgemma-model-cache", create_if_missing=True)
    training_data = modal.Volume.from_name("medgemma-training-data", create_if_missing=True)
    checkpoints = modal.Volume.from_name("medgemma-checkpoints", create_if_missing=True)
    
    @app.cls(
        image=training_image,
        gpu=modal.gpu.H100(count=8),  # 8x H100 for 27B training
        volumes={
            "/root/.cache/huggingface": model_cache,
            "/data": training_data,
            "/checkpoints": checkpoints,
        },
        timeout=86400,  # 24 hours
        container_idle_timeout=600,  # 10 min idle
        secrets=[
            modal.Secret.from_name("huggingface-secret", required=False),
            modal.Secret.from_name("mlflow-secret", required=False),
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
            train_data_path: Optional[str] = None,
            eval_data_path: Optional[str] = None,
            max_steps: int = 10000,
            batch_size: int = 1,
            learning_rate: float = 1e-4,
            lora_r: int = 64,
            lora_alpha: int = 128,
            gradient_accumulation_steps: int = 16,
            save_steps: int = 500,
            eval_steps: int = 500,
            warmup_ratio: float = 0.1,
            mlflow_tracking_uri: Optional[str] = None,
            experiment_name: Optional[str] = None,
        ) -> Dict[str, Any]:
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
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
                Trainer,
            )
            from peft import LoraConfig, get_peft_model, TaskType
            
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
            
            print(f"Training {hf_model_id}")
            print(f"GPUs available: {torch.cuda.device_count()}")
            print(f"Using DeepSpeed: {use_deepspeed}")
            
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
            
            # Training arguments
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
                eval_steps=eval_steps,
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="loss",
                greater_is_better=False,
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
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                # train_dataset=train_dataset,  # Load from train_data_path
                # eval_dataset=eval_dataset,    # Load from eval_data_path
                tokenizer=tokenizer,
            )
            
            # Note: In production, load actual datasets from train_data_path/eval_data_path
            # For now, return configuration for validation
            
            # Commit checkpoints
            checkpoints.commit()
            
            return {
                "status": "configured",
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
            }
        
        @modal.method()
        def get_gpu_info(self) -> Dict[str, Any]:
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
        gpu=modal.gpu.H100(count=1),  # 1x H100 for 4B training/inference
        volumes={
            "/root/.cache/huggingface": model_cache,
            "/data": training_data,
            "/checkpoints": checkpoints,
        },
        timeout=86400,
        container_idle_timeout=300,
        secrets=[
            modal.Secret.from_name("huggingface-secret", required=False),
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
            train_data_path: Optional[str] = None,
            max_steps: int = 10000,
            batch_size: int = 4,
            learning_rate: float = 2e-4,
            lora_r: int = 16,
            lora_alpha: int = 32,
        ) -> Dict[str, Any]:
            """Train MedGemma 4B model on single H100."""
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, TaskType
            
            hf_model_id = "google/medgemma-4b-it"
            
            print(f"Training {hf_model_id} on {torch.cuda.get_device_name(0)}")
            
            # Load model
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
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
            
            checkpoints.commit()
            
            return {
                "status": "configured",
                "model": hf_model_id,
                "gpu": torch.cuda.get_device_name(0),
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            }
    
    # =============================================================================
    # H100 Optimization Verification Functions
    # =============================================================================

    @app.function(
        image=training_image,
        gpu=modal.gpu.H100(count=1),
        timeout=600,
        secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
    )
    def verify_h100_optimizations() -> Dict[str, Any]:
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
        import torch
        import time

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
        gpu=modal.gpu.H100(count=1),
        volumes={"/root/.cache/huggingface": model_cache},
        timeout=1200,
        secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
    )
    def benchmark_inference_throughput(
        model_name: str = "medgemma-4b",
        batch_sizes: List[int] = [1, 2, 4, 8],
        input_length: int = 256,
        output_length: int = 128,
    ) -> Dict[str, Any]:
        """
        Benchmark inference throughput on H100 with various batch sizes.

        Tests Flash Attention 2 and compares with SDPA.

        Returns:
            Dict with throughput measurements per batch size
        """
        import torch
        import time
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
        max_steps: int = 100,
        dry_run: bool = True,
        verify_optimizations: bool = False,
        benchmark: bool = False,
    ):
        """Run training or benchmarks from CLI.

        Examples:
            # Verify H100 optimizations
            modal run medai_compass/modal/training_app.py --verify-optimizations

            # Benchmark inference
            modal run medai_compass/modal/training_app.py --benchmark

            # Run training
            modal run medai_compass/modal/training_app.py --model-name medgemma-4b
        """
        if verify_optimizations:
            print("Verifying H100 optimizations on Modal...")
            result = verify_h100_optimizations.remote()
            print(f"\nGPU Info: {result['gpu_info']}")
            print(f"\nOptimizations Available:")
            for name, status in result['optimizations'].items():
                print(f"  {name}: {status}")
            print(f"\nBenchmarks:")
            for name, value in result['benchmarks'].items():
                print(f"  {name}: {value:.2f}")
            return

        if benchmark:
            print("Running inference throughput benchmark on H100...")
            result = benchmark_inference_throughput.remote(model_name=model_name)
            print(f"\nModel: {result['model']}")
            print(f"GPU: {result['gpu']}")
            print(f"\nResults by attention implementation:")
            for impl, batches in result['batch_results'].items():
                print(f"\n  {impl}:")
                for batch_key, metrics in batches.items():
                    if isinstance(metrics, dict) and "tokens_per_second" in metrics:
                        print(f"    {batch_key}: {metrics['tokens_per_second']:.1f} tokens/s, "
                              f"latency: {metrics['latency_ms']:.1f}ms"
                              f"{' (+' + metrics['throughput_improvement'] + ')' if metrics.get('throughput_improvement') else ''}")
            return

        print(f"Starting training for {model_name}")

        if "27b" in model_name.lower():
            trainer = MedGemmaTrainer()

            # Get GPU info first
            gpu_info = trainer.get_gpu_info.remote()
            print(f"GPU Info: {gpu_info}")

            if not dry_run:
                result = trainer.train.remote(
                    model_name=model_name,
                    max_steps=max_steps,
                )
                print(f"Training result: {result}")
        else:
            trainer = MedGemma4BTrainer()

            if not dry_run:
                result = trainer.train.remote(max_steps=max_steps)
                print(f"Training result: {result}")

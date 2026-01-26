"""
Unified CLI for MedGemma Ray Pipelines.

This module provides a command-line interface for running Ray-based pipelines
for training, evaluation, tuning, data processing, and model serving.

Usage:
    # Run full pipeline
    uv run python -m medai_compass.pipelines run

    # Train model with default config
    uv run python -m medai_compass.pipelines train

    # Train with Hydra overrides
    uv run python -m medai_compass.pipelines train model=medgemma_27b training.args.learning_rate=1e-4

    # Hyperparameter tuning
    uv run python -m medai_compass.pipelines tune --scheduler asha --num-samples 50

    # Evaluate model
    uv run python -m medai_compass.pipelines evaluate --checkpoint /checkpoints/final

    # Start Ray Serve
    uv run python -m medai_compass.pipelines serve --model medgemma-4b --port 8000

    # Process data
    uv run python -m medai_compass.pipelines data --source synthea --output /data/processed

    # Show config
    uv run python -m medai_compass.pipelines config

    # Verify setup
    uv run python -m medai_compass.pipelines verify
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================


def check_dependencies(required: list[str]) -> bool:
    """Check if required dependencies are available."""
    missing = []
    for dep in required:
        try:
            __import__(dep.split("[")[0])
        except ImportError:
            missing.append(dep)

    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.info(f"Install with: uv add {' '.join(missing)}")
        return False
    return True


def init_ray(num_cpus: int | None = None, address: str | None = None):
    """Initialize Ray runtime."""
    import ray

    if ray.is_initialized():
        logger.info("Ray already initialized")
        return ray

    try:
        if address:
            ray.init(address=address, ignore_reinit_error=True)
            logger.info(f"Connected to Ray cluster at {address}")
        else:
            # Try connecting to existing cluster
            try:
                ray.init(address="auto", ignore_reinit_error=True)
                logger.info("Connected to existing Ray cluster")
            except ConnectionError:
                # Start local cluster
                kwargs = {"ignore_reinit_error": True}
                if num_cpus:
                    kwargs["num_cpus"] = num_cpus
                ray.init(**kwargs)
                logger.info(f"Started local Ray cluster: {ray.cluster_resources()}")
    except Exception as e:
        logger.warning(f"Ray initialization warning: {e}")
        ray.init(ignore_reinit_error=True)

    return ray


def load_hydra_config(overrides: list[str] | None = None):
    """Load Hydra configuration with optional overrides."""
    try:
        from medai_compass.config.hydra_config import (
            load_config,
            load_config_with_overrides,
        )

        if overrides:
            return load_config_with_overrides(overrides)
        return load_config()
    except ImportError:
        logger.warning("Hydra config not available, using defaults")
        return None


def print_banner(command: str):
    """Print CLI banner."""
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║                   MedGemma Pipeline CLI                       ║
║                      Command: {command:<28} ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_result(result: dict[str, Any]):
    """Print result as formatted JSON."""
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))


# =============================================================================
# Command: run (Full Pipeline)
# =============================================================================


def cmd_run(args, hydra_overrides: list[str]) -> dict[str, Any]:
    """
    Run the full pipeline: data -> train -> evaluate.

    Orchestrates the complete ML workflow with configurable stages.
    """
    print_banner("run")

    if not check_dependencies(["ray", "torch", "transformers"]):
        return {"status": "error", "error": "Missing dependencies"}

    cfg = load_hydra_config(hydra_overrides)
    init_ray()

    from medai_compass.pipelines.training_pipeline import (
        TrainingPipelineConfig,
        TrainingPipelineOrchestrator,
    )

    # Build config from args and Hydra
    model_name = args.model or (cfg.model.name if cfg else "medgemma-4b")
    if "google/" in model_name:
        model_name = "medgemma-27b" if "27b" in model_name else "medgemma-4b"

    config = TrainingPipelineConfig(
        model_name=model_name,
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        max_steps=args.max_steps,
        checkpoint_dir=args.checkpoint_dir,
        dry_run=args.dry_run,
    )

    orchestrator = TrainingPipelineOrchestrator(config=config)

    logger.info(f"Running full pipeline for {model_name}")
    logger.info(f"  Train data: {args.train_data}")
    logger.info(f"  Max steps: {args.max_steps}")
    logger.info(f"  Checkpoint dir: {args.checkpoint_dir}")

    result = orchestrator.run(
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
    )

    return {
        "status": result.status,
        "model": model_name,
        "run_id": result.run_id,
        "model_path": result.model_path,
        "metrics": result.metrics,
        "duration_seconds": result.duration_seconds,
    }


# =============================================================================
# Command: train
# =============================================================================


def cmd_train(args, hydra_overrides: list[str]) -> dict[str, Any]:
    """
    Train a MedGemma model with LoRA/QLoRA.

    Supports distributed training with Ray Train and DeepSpeed.
    """
    print_banner("train")

    if not check_dependencies(["ray", "torch", "transformers", "peft"]):
        return {"status": "error", "error": "Missing dependencies"}

    cfg = load_hydra_config(hydra_overrides)
    init_ray()

    # Check for Modal backend
    if args.backend == "modal":
        return _train_with_modal(args, cfg)

    # Local/Ray training
    from medai_compass.pipelines.training_pipeline import (
        TrainingPipelineConfig,
        TrainingPipelineOrchestrator,
    )

    model_name = args.model or (cfg.model.name if cfg else "medgemma-4b")
    if "google/" in model_name:
        model_name = "medgemma-27b" if "27b" in model_name else "medgemma-4b"

    # Apply Hydra config values
    max_steps = args.max_steps
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    if cfg:
        if max_steps == 1000:  # default value
            max_steps = cfg.training.args.get("max_steps", 1000)
        if learning_rate == 2e-4:
            learning_rate = cfg.training.args.get("learning_rate", 2e-4)
        if batch_size == 4:
            batch_size = cfg.training.args.get("per_device_train_batch_size", 4)

    config = TrainingPipelineConfig(
        model_name=model_name,
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        max_steps=max_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        checkpoint_dir=args.checkpoint_dir,
        dry_run=args.dry_run,
    )

    orchestrator = TrainingPipelineOrchestrator(config=config)

    logger.info(f"Training {model_name}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Max steps: {max_steps}")

    result = orchestrator.run(
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
    )

    return {
        "status": result.status,
        "model": model_name,
        "run_id": result.run_id,
        "model_path": result.model_path,
        "metrics": result.metrics,
        "duration_seconds": result.duration_seconds,
    }


def _train_with_modal(args, cfg) -> dict[str, Any]:
    """Train using Modal cloud GPUs."""
    logger.info("Training with Modal backend")

    try:
        from medai_compass.modal.training_app import (
            MedGemma4BTrainer,
            MedGemmaTrainer,
            app,
        )
    except ImportError:
        return {"status": "error", "error": "Modal not available"}

    model_name = args.model or "medgemma-4b"
    is_27b = "27b" in model_name.lower()

    with app.run():
        if is_27b:
            trainer = MedGemmaTrainer()
            result = trainer.train.remote(
                model_name=model_name,
                train_data_path=args.train_data,
                eval_data_path=args.eval_data,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
        else:
            trainer = MedGemma4BTrainer()
            result = trainer.train.remote(
                train_data_path=args.train_data,
                eval_data_path=args.eval_data,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )

    return result


# =============================================================================
# Command: tune
# =============================================================================


def cmd_tune(args, hydra_overrides: list[str]) -> dict[str, Any]:
    """
    Run hyperparameter tuning with Ray Tune.

    Supports ASHA, PBT, and Hyperband schedulers.
    """
    print_banner("tune")

    if not check_dependencies(["ray", "torch", "optuna"]):
        return {"status": "error", "error": "Missing dependencies"}

    cfg = load_hydra_config(hydra_overrides)
    init_ray()

    # Override scheduler from args
    if cfg and args.scheduler:
        from omegaconf import OmegaConf

        cfg = OmegaConf.merge(cfg, {"tuning": {"scheduler": args.scheduler}})

    if cfg and args.num_samples:
        from omegaconf import OmegaConf

        cfg = OmegaConf.merge(cfg, {"tuning": {"num_samples": args.num_samples}})

    if cfg:
        from medai_compass.tuning.runner import TuningRunner

        runner = TuningRunner(cfg)
        logger.info(f"Running hyperparameter tuning with {runner.scheduler_type}")
        logger.info(f"  Num samples: {cfg.tuning.num_samples}")
        logger.info(f"  Metric: {cfg.tuning.metric}")

        result = runner.run()
        return {
            "status": "completed",
            "scheduler": runner.scheduler_type,
            "best_config": result.get("best_config", {}),
            "best_metric": result.get("best_metric"),
        }

    # Fallback without Hydra config
    from medai_compass.optimization.ray_tune_integration import (
        MedGemmaTuner,
        TuningConfig,
    )

    tuner = MedGemmaTuner(
        model_name=args.model or "medgemma-4b",
        config=TuningConfig(
            scheduler=args.scheduler,
            num_samples=args.num_samples,
            max_concurrent_trials=args.max_concurrent,
        ),
    )

    result = tuner.run()
    return {
        "status": "completed",
        "scheduler": args.scheduler,
        "best_config": result.get("best_config", {}),
        "best_metric": result.get("best_metric"),
    }


# =============================================================================
# Command: evaluate
# =============================================================================


def cmd_evaluate(args, hydra_overrides: list[str]) -> dict[str, Any]:
    """
    Evaluate a trained model on medical benchmarks.

    Supports MedQA, PubMedQA, and safety evaluations.
    """
    print_banner("evaluate")

    if not check_dependencies(["ray", "torch", "transformers"]):
        return {"status": "error", "error": "Missing dependencies"}

    init_ray()

    if args.backend == "modal":
        return _evaluate_with_modal(args)

    from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline

    checkpoint = args.checkpoint or "google/medgemma-4b-it"

    logger.info(f"Evaluating: {checkpoint}")
    logger.info(f"  Quick mode: {args.quick}")
    logger.info(f"  Benchmark: {args.benchmark or 'all'}")

    pipeline = EvaluationPipeline(model_path=checkpoint)

    if args.benchmark:
        result = pipeline.run_benchmark(
            benchmark=args.benchmark,
            max_samples=args.max_samples,
        )
    else:
        result = pipeline.run_all(
            quick=args.quick,
            max_samples=args.max_samples,
        )

    return {
        "status": "completed",
        "checkpoint": checkpoint,
        "results": result.to_dict() if hasattr(result, "to_dict") else result,
    }


def _evaluate_with_modal(args) -> dict[str, Any]:
    """Evaluate using Modal cloud GPUs."""
    try:
        from medai_compass.modal.evaluation_app import (
            MedGemmaEvaluator,
            app,
            find_latest_checkpoint,
        )
    except ImportError:
        return {"status": "error", "error": "Modal evaluation not available"}

    with app.run():
        checkpoint = args.checkpoint
        if not checkpoint:
            checkpoint = find_latest_checkpoint.remote()
            if not checkpoint:
                checkpoint = "google/medgemma-4b-it"

        evaluator = MedGemmaEvaluator()
        result = evaluator.run_full_evaluation.remote(
            model_path=checkpoint,
            max_samples=args.max_samples,
            quick=args.quick,
        )

    return result


# =============================================================================
# Command: serve
# =============================================================================


def cmd_serve(args, hydra_overrides: list[str]) -> dict[str, Any]:
    """
    Start Ray Serve deployment for model inference.

    Supports autoscaling and multi-model routing.
    """
    print_banner("serve")

    if not check_dependencies(["ray"]):
        return {"status": "error", "error": "Missing dependencies"}

    init_ray()

    from medai_compass.inference.ray_serve_deployment import (
        deploy_medgemma,
        deploy_multi_model,
    )

    model_id = args.model
    if model_id in ("medgemma-4b", "4b"):
        model_id = "google/medgemma-4b-it"
    elif model_id in ("medgemma-27b", "27b"):
        model_id = "google/medgemma-27b-it"

    logger.info("Starting Ray Serve deployment")
    logger.info(f"  Model: {model_id}")
    logger.info(f"  Port: {args.port}")
    logger.info(f"  Replicas: {args.replicas}")
    logger.info(f"  Multi-model: {args.multi_model}")

    if args.multi_model:
        deploy_multi_model(port=args.port)
    else:
        deploy_medgemma(
            model_id=model_id,
            num_replicas=args.replicas,
            autoscaling=not args.no_autoscaling,
            port=args.port,
        )

    logger.info(f"Ray Serve running at http://0.0.0.0:{args.port}")
    logger.info("Press Ctrl+C to stop")

    # Keep running
    try:
        import time

        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down Ray Serve...")

    return {"status": "running", "port": args.port, "model": model_id}


# =============================================================================
# Command: data
# =============================================================================


def cmd_data(args, hydra_overrides: list[str]) -> dict[str, Any]:
    """
    Process and prepare training data.

    Supports Synthea, MedQuAD, and custom datasets with PHI filtering.
    """
    print_banner("data")

    if not check_dependencies(["ray"]):
        return {"status": "error", "error": "Missing dependencies"}

    init_ray()

    from medai_compass.pipelines import MedicalDataPipeline, PipelineConfig

    config = PipelineConfig(
        phi_mode=args.phi_mode,
        train_split=args.train_split,
        max_length=args.max_length,
    )

    pipeline = MedicalDataPipeline(
        model_name=args.model or "medgemma-4b",
        config=config,
    )

    logger.info(f"Processing data from: {args.source}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  PHI mode: {args.phi_mode}")

    # Load dataset
    dataset = pipeline.load_dataset(args.source)
    logger.info(f"Loaded {dataset.count()} records")

    # Filter PHI
    if args.phi_mode != "none":
        dataset = pipeline.filter_phi(dataset)
        logger.info(f"After PHI filtering: {dataset.count()} records")

    # Create splits
    train_ds, val_ds = pipeline.create_splits(dataset, train_ratio=args.train_split)

    # Save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train.jsonl"
    val_path = output_path / "val.jsonl"

    pipeline.save_dataset(train_ds, str(train_path))
    pipeline.save_dataset(val_ds, str(val_path))

    return {
        "status": "completed",
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_count": train_ds.count(),
        "val_count": val_ds.count(),
    }


# =============================================================================
# Command: config
# =============================================================================


def cmd_config(args, hydra_overrides: list[str]) -> dict[str, Any]:
    """
    Show current configuration.

    Displays the resolved Hydra configuration with any overrides applied.
    """
    print_banner("config")

    cfg = load_hydra_config(hydra_overrides)

    if cfg is None:
        return {"status": "error", "error": "Could not load Hydra config"}

    from omegaconf import OmegaConf

    if args.section:
        if hasattr(cfg, args.section):
            section_cfg = getattr(cfg, args.section)
            print(OmegaConf.to_yaml(section_cfg))
            return OmegaConf.to_container(section_cfg, resolve=True)
        else:
            return {"status": "error", "error": f"Unknown section: {args.section}"}

    print(OmegaConf.to_yaml(cfg))
    return {"status": "ok", "config": "printed above"}


# =============================================================================
# Command: verify
# =============================================================================


def cmd_verify(args, hydra_overrides: list[str]) -> dict[str, Any]:
    """
    Verify system setup and dependencies.

    Checks Ray, GPU availability, Modal auth, and required packages.
    """
    print_banner("verify")

    results = {
        "ray": False,
        "gpu": False,
        "modal": False,
        "hydra": False,
        "transformers": False,
    }

    # Check Ray
    try:
        import ray

        ray.init(ignore_reinit_error=True)
        results["ray"] = True
        results["ray_resources"] = ray.cluster_resources()
    except Exception as e:
        results["ray_error"] = str(e)

    # Check GPU
    try:
        import torch

        results["gpu"] = torch.cuda.is_available()
        if results["gpu"]:
            results["gpu_count"] = torch.cuda.device_count()
            results["gpu_name"] = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            results["gpu"] = True
            results["gpu_type"] = "Apple MPS"
    except Exception as e:
        results["gpu_error"] = str(e)

    # Check Modal
    try:
        import importlib.util

        if importlib.util.find_spec("modal") is not None:
            # Check for credentials
            modal_toml = Path.home() / ".modal.toml"
            has_env = bool(os.environ.get("MODAL_TOKEN_ID"))
            results["modal"] = modal_toml.exists() or has_env
            results["modal_auth_method"] = (
                "env" if has_env else "file" if modal_toml.exists() else None
            )
        else:
            results["modal"] = False
            results["modal_error"] = "modal not installed"
    except Exception:
        results["modal"] = False
        results["modal_error"] = "modal not installed"

    # Check Hydra
    try:
        cfg = load_hydra_config()
        results["hydra"] = cfg is not None
        if cfg:
            results["hydra_model"] = cfg.model.name
    except Exception as e:
        results["hydra_error"] = str(e)

    # Check transformers
    try:
        import transformers

        results["transformers"] = True
        results["transformers_version"] = transformers.__version__
    except ImportError:
        results["transformers"] = False

    # Check HuggingFace token
    results["hf_token"] = bool(os.environ.get("HF_TOKEN"))

    # Summary
    all_ok = all([results["ray"], results["transformers"]])
    results["status"] = "ready" if all_ok else "missing_dependencies"

    # Print summary
    print("\nSystem Verification Results:")
    print("-" * 40)
    for key, value in results.items():
        if key.endswith("_error"):
            continue
        icon = "✓" if value else "✗"
        print(f"  {icon} {key}: {value}")

    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="medai_compass.pipelines",
        description="MedGemma Ray Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  uv run python -m medai_compass.pipelines run --model medgemma-4b

  # Train with Hydra overrides
  uv run python -m medai_compass.pipelines train model=medgemma_27b

  # Hyperparameter tuning
  uv run python -m medai_compass.pipelines tune --scheduler asha

  # Start serving
  uv run python -m medai_compass.pipelines serve --port 8000

  # Process data
  uv run python -m medai_compass.pipelines data --source ./data/raw --output ./data/processed

  # Show config
  uv run python -m medai_compass.pipelines config --section model

  # Verify setup
  uv run python -m medai_compass.pipelines verify
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run full pipeline (data -> train -> evaluate)")
    run_parser.add_argument(
        "--model", "-m", type=str, help="Model name (medgemma-4b or medgemma-27b)"
    )
    run_parser.add_argument(
        "--train-data", "-t", type=str, default="/data/train.jsonl", help="Training data path"
    )
    run_parser.add_argument(
        "--eval-data", "-e", type=str, default="/data/eval.jsonl", help="Evaluation data path"
    )
    run_parser.add_argument("--max-steps", type=int, default=1000, help="Maximum training steps")
    run_parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory"
    )
    run_parser.add_argument(
        "--dry-run", action="store_true", help="Dry run without actual training"
    )

    # train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--model", "-m", type=str, help="Model name")
    train_parser.add_argument(
        "--train-data", "-t", type=str, default="/data/train.jsonl", help="Training data"
    )
    train_parser.add_argument("--eval-data", "-e", type=str, help="Evaluation data")
    train_parser.add_argument("--max-steps", type=int, default=1000, help="Max training steps")
    train_parser.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size")
    train_parser.add_argument(
        "--learning-rate", "--lr", type=float, default=2e-4, help="Learning rate"
    )
    train_parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint dir"
    )
    train_parser.add_argument(
        "--backend", choices=["local", "modal"], default="local", help="Compute backend"
    )
    train_parser.add_argument("--dry-run", action="store_true", help="Dry run")

    # tune command
    tune_parser = subparsers.add_parser("tune", help="Hyperparameter tuning")
    tune_parser.add_argument("--model", "-m", type=str, help="Model name")
    tune_parser.add_argument(
        "--scheduler", "-s", choices=["asha", "pbt", "hyperband"], default="asha", help="Scheduler"
    )
    tune_parser.add_argument("--num-samples", "-n", type=int, default=50, help="Number of trials")
    tune_parser.add_argument("--max-concurrent", type=int, default=8, help="Max concurrent trials")

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--checkpoint", "-c", type=str, help="Checkpoint path")
    eval_parser.add_argument(
        "--benchmark", choices=["medqa", "pubmedqa", "safety", "hallucination"], help="Benchmark"
    )
    eval_parser.add_argument("--max-samples", type=int, help="Max samples per benchmark")
    eval_parser.add_argument("--quick", "-q", action="store_true", help="Quick evaluation")
    eval_parser.add_argument(
        "--backend", choices=["local", "modal"], default="local", help="Compute backend"
    )

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start Ray Serve")
    serve_parser.add_argument(
        "--model", "-m", type=str, default="medgemma-4b", help="Model to serve"
    )
    serve_parser.add_argument("--port", "-p", type=int, default=8000, help="Server port")
    serve_parser.add_argument("--replicas", "-r", type=int, default=1, help="Number of replicas")
    serve_parser.add_argument("--no-autoscaling", action="store_true", help="Disable autoscaling")
    serve_parser.add_argument(
        "--multi-model", action="store_true", help="Enable multi-model routing"
    )

    # data command
    data_parser = subparsers.add_parser("data", help="Process training data")
    data_parser.add_argument("--source", "-s", type=str, required=True, help="Source data path")
    data_parser.add_argument(
        "--output", "-o", type=str, default="./data/processed", help="Output path"
    )
    data_parser.add_argument("--model", "-m", type=str, help="Model for tokenization")
    data_parser.add_argument(
        "--phi-mode", choices=["strict", "mask", "none"], default="strict", help="PHI filtering"
    )
    data_parser.add_argument("--train-split", type=float, default=0.95, help="Train split ratio")
    data_parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")

    # config command
    config_parser = subparsers.add_parser("config", help="Show configuration")
    config_parser.add_argument(
        "--section", type=str, help="Config section (model, training, data, etc.)"
    )

    # verify command
    subparsers.add_parser("verify", help="Verify system setup")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = create_parser()

    # Separate known args from Hydra overrides
    args, remaining = parser.parse_known_args(argv)

    # Filter Hydra overrides (key=value format)
    hydra_overrides = [r for r in remaining if "=" in r and not r.startswith("-")]

    if not args.command:
        parser.print_help()
        return 1

    # Command dispatch
    commands = {
        "run": cmd_run,
        "train": cmd_train,
        "tune": cmd_tune,
        "evaluate": cmd_evaluate,
        "serve": cmd_serve,
        "data": cmd_data,
        "config": cmd_config,
        "verify": cmd_verify,
    }

    cmd_func = commands.get(args.command)
    if cmd_func is None:
        parser.print_help()
        return 1

    try:
        result = cmd_func(args, hydra_overrides)
        if args.command not in ("config", "verify", "serve"):
            print_result(result)
        return 0 if result.get("status") != "error" else 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Command failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

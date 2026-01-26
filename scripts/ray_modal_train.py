#!/usr/bin/env python3
"""Ray Training Pipeline with Modal GPU Backend.

This script provides Ray commands to run MedGemma training pipelines on Modal H100 GPUs.

Usage:
    # Download datasets
    python scripts/ray_modal_train.py download --datasets medqa,pubmedqa

    # Train model (4B on 1x H100)
    python scripts/ray_modal_train.py train --model medgemma-4b --max-steps 1000

    # Train model (27B on 8x H100 with DeepSpeed)
    python scripts/ray_modal_train.py train --model medgemma-27b --max-steps 5000

    # Evaluate trained model
    python scripts/ray_modal_train.py evaluate --checkpoint /checkpoints/final

    # Full pipeline: download -> train -> evaluate
    python scripts/ray_modal_train.py pipeline --model medgemma-4b

    # Verify Modal GPU setup
    python scripts/ray_modal_train.py verify
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check required dependencies are installed."""
    missing = []

    try:
        import ray
    except ImportError:
        missing.append("ray[train]")

    try:
        import modal
    except ImportError:
        missing.append("modal")

    try:
        import torch
    except ImportError:
        missing.append("torch")

    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.info("Install with: uv add " + " ".join(missing))
        return False

    return True


def check_modal_auth():
    """Check Modal authentication."""
    # Check environment variables first
    token_id = os.environ.get("MODAL_TOKEN_ID")
    token_secret = os.environ.get("MODAL_TOKEN_SECRET")

    if token_id and token_secret:
        return True

    # Check ~/.modal.toml config file
    modal_toml = Path.home() / ".modal.toml"
    if modal_toml.exists():
        logger.info("Modal credentials found in ~/.modal.toml")
        return True

    logger.error("Modal credentials not found")
    logger.info("Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET")
    logger.info("Or run: modal token new")
    return False


def init_ray(num_cpus: int = 4):
    """Initialize Ray runtime."""
    import ray

    if not ray.is_initialized():
        # Try connecting to existing cluster first
        try:
            ray.init(
                address="auto",
                ignore_reinit_error=True,
                logging_level=logging.INFO,
            )
            logger.info(f"Connected to existing Ray cluster: {ray.cluster_resources()}")
        except ConnectionError:
            # Start local cluster if no existing cluster
            ray.init(
                num_cpus=num_cpus,
                ignore_reinit_error=True,
                logging_level=logging.INFO,
            )
            logger.info(f"Started new Ray cluster: {ray.cluster_resources()}")

    return ray


def download_datasets(
    datasets: List[str],
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Download medical datasets using Modal.

    Args:
        datasets: List of dataset names to download
        max_samples: Optional sample limit per dataset

    Returns:
        Download results
    """
    logger.info(f"Downloading datasets: {datasets}")

    # Import Modal app and functions
    from medai_compass.modal.data_download import (
        app,
        download_dataset,
        download_all,
        combine_datasets,
    )

    results = {"datasets": {}, "status": "in_progress"}

    # Run within Modal app context
    with app.run():
        if "all" in datasets:
            # Download all datasets
            logger.info("Downloading all medical datasets...")
            result = download_all.remote(max_samples=max_samples)
            results = result
        else:
            # Download specific datasets
            for dataset_name in datasets:
                logger.info(f"Downloading: {dataset_name}")
                result = download_dataset.remote(
                    dataset_name=dataset_name,
                    max_samples=max_samples,
                )
                results["datasets"][dataset_name] = result

            # Combine downloaded datasets
            if len(datasets) > 1:
                logger.info("Combining datasets...")
                combine_result = combine_datasets.remote(
                    datasets=datasets,
                    output_name="combined_medical",
                )
                results["combined"] = combine_result

    results["status"] = "completed"
    logger.info("Dataset download completed")

    return results


def run_training(
    model_name: str = "medgemma-4b",
    train_data_path: str = "/data/combined_medical/train.jsonl",
    eval_data_path: Optional[str] = "/data/combined_medical/eval.jsonl",
    max_steps: int = 1000,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    use_ray_train: bool = True,
    mlflow_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run training on Modal GPUs.

    Supports both Ray Train orchestration and direct Modal execution.

    Args:
        model_name: Model to train (medgemma-4b or medgemma-27b)
        train_data_path: Path to training data in Modal volume
        eval_data_path: Path to evaluation data
        max_steps: Maximum training steps
        batch_size: Per-device batch size
        learning_rate: Learning rate
        use_ray_train: Use Ray Train for orchestration
        mlflow_uri: MLflow tracking URI

    Returns:
        Training results
    """
    import ray

    logger.info(f"Starting training: {model_name}")
    logger.info(f"Max steps: {max_steps}, Batch size: {batch_size}")

    if use_ray_train:
        # Use Ray Train with Modal backend
        return _train_with_ray(
            model_name=model_name,
            train_data_path=train_data_path,
            eval_data_path=eval_data_path,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            mlflow_uri=mlflow_uri,
        )
    else:
        # Direct Modal execution
        return _train_with_modal_direct(
            model_name=model_name,
            train_data_path=train_data_path,
            eval_data_path=eval_data_path,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            mlflow_uri=mlflow_uri,
        )


def _train_with_ray(
    model_name: str,
    train_data_path: str,
    eval_data_path: Optional[str],
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    mlflow_uri: Optional[str],
) -> Dict[str, Any]:
    """Train using Ray Train orchestration with Modal as compute backend.

    Ray is used to orchestrate the job submission and tracking,
    while Modal provides the actual GPU compute.
    """
    import ray

    # Determine GPU count based on model
    is_27b = "27b" in model_name.lower()
    num_gpus = 8 if is_27b else 1
    strategy = "deepspeed_zero3" if is_27b else "single_gpu"

    logger.info(f"Training strategy: {strategy}, GPUs: {num_gpus}")
    logger.info("Ray orchestration active - submitting training to Modal GPUs...")

    # Import Modal training app
    from medai_compass.modal.training_app import (
        app,
        MedGemmaTrainer,
        MedGemma4BTrainer,
    )

    # Run training within Modal app context
    with app.run():
        if is_27b:
            trainer = MedGemmaTrainer()
            result = trainer.train.remote(
                model_name=model_name,
                train_data_path=train_data_path,
                eval_data_path=eval_data_path,
                max_steps=max_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                mlflow_tracking_uri=mlflow_uri,
            )
        else:
            trainer = MedGemma4BTrainer()
            result = trainer.train.remote(
                train_data_path=train_data_path,
                eval_data_path=eval_data_path,
                max_steps=max_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )

    logger.info(f"Training completed: {result.get('status', 'unknown')}")

    return result


def _train_with_modal_direct(
    model_name: str,
    train_data_path: str,
    eval_data_path: Optional[str],
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    mlflow_uri: Optional[str],
) -> Dict[str, Any]:
    """Train directly on Modal without Ray orchestration."""
    from medai_compass.modal.training_app import (
        app,
        MedGemmaTrainer,
        MedGemma4BTrainer,
    )

    is_27b = "27b" in model_name.lower()

    logger.info(f"Running training directly on Modal ({'8x H100' if is_27b else '1x H100'})")

    # Run within Modal app context
    with app.run():
        if is_27b:
            trainer = MedGemmaTrainer()
            result = trainer.train.remote(
                model_name=model_name,
                train_data_path=train_data_path,
                eval_data_path=eval_data_path,
                max_steps=max_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                mlflow_tracking_uri=mlflow_uri,
            )
        else:
            trainer = MedGemma4BTrainer()
            result = trainer.train.remote(
                train_data_path=train_data_path,
                eval_data_path=eval_data_path,
                max_steps=max_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )

    return result


def run_evaluation(
    checkpoint_path: Optional[str] = None,
    quick: bool = False,
    max_samples: Optional[int] = None,
    benchmark: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run evaluation on trained model.

    Args:
        checkpoint_path: Path to model checkpoint
        quick: Use smaller sample sizes
        max_samples: Optional sample limit
        benchmark: Specific benchmark to run

    Returns:
        Evaluation results
    """
    logger.info("Running model evaluation on Modal...")

    from medai_compass.modal.evaluation_app import (
        app,
        MedGemmaEvaluator,
        find_latest_checkpoint,
    )

    # Run within Modal app context
    with app.run():
        # Find checkpoint if not specified
        if not checkpoint_path:
            logger.info("Finding latest checkpoint...")
            checkpoint_path = find_latest_checkpoint.remote()
            if not checkpoint_path:
                checkpoint_path = "google/medgemma-4b-it"
                logger.info(f"No checkpoint found, using HuggingFace: {checkpoint_path}")

        logger.info(f"Evaluating: {checkpoint_path}")

        evaluator = MedGemmaEvaluator()

        if benchmark:
            # Run single benchmark
            if benchmark == "medqa":
                result = evaluator.evaluate_medqa.remote(checkpoint_path, max_samples)
            elif benchmark == "pubmedqa":
                result = evaluator.evaluate_pubmedqa.remote(checkpoint_path, max_samples)
            elif benchmark == "safety":
                result = evaluator.evaluate_safety.remote(checkpoint_path)
            elif benchmark == "hallucination":
                result = evaluator.evaluate_hallucination.remote(checkpoint_path)
            else:
                logger.error(f"Unknown benchmark: {benchmark}")
                return {"error": f"Unknown benchmark: {benchmark}"}
        else:
            # Run full evaluation
            result = evaluator.run_full_evaluation.remote(
                model_path=checkpoint_path,
                max_samples=max_samples,
                quick=quick,
            )

    logger.info("Evaluation completed")

    return result


def verify_modal_setup() -> Dict[str, Any]:
    """
    Verify Modal GPU setup and H100 optimizations.

    Returns:
        Verification results
    """
    logger.info("Verifying Modal GPU setup...")

    from medai_compass.modal.training_app import app, verify_h100_optimizations

    # Run within Modal app context
    with app.run():
        result = verify_h100_optimizations.remote()

    # Log results
    logger.info(f"\nGPU Info: {result.get('gpu_info', {})}")

    logger.info("\nOptimizations Available:")
    for name, status in result.get("optimizations", {}).items():
        available = status.get("available", False)
        logger.info(f"  {name}: {'Yes' if available else 'No'}")

    logger.info("\nBenchmarks:")
    for name, value in result.get("benchmarks", {}).items():
        if isinstance(value, (int, float)):
            logger.info(f"  {name}: {value:.2f}")

    return result


def run_full_pipeline(
    model_name: str = "medgemma-4b",
    datasets: List[str] = None,
    max_steps: int = 1000,
    skip_download: bool = False,
    skip_eval: bool = False,
) -> Dict[str, Any]:
    """
    Run full training pipeline: download -> train -> evaluate.

    Args:
        model_name: Model to train
        datasets: Datasets to download
        max_steps: Training steps
        skip_download: Skip dataset download
        skip_eval: Skip evaluation

    Returns:
        Pipeline results
    """
    import ray

    datasets = datasets or ["medqa", "pubmedqa", "medmcqa"]

    pipeline_results = {
        "model": model_name,
        "started_at": datetime.utcnow().isoformat(),
        "stages": {},
    }

    # Stage 1: Download datasets
    if not skip_download:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: Downloading Datasets")
        logger.info("=" * 60)

        download_result = download_datasets(datasets)
        pipeline_results["stages"]["download"] = download_result

    # Stage 2: Training
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: Training Model")
    logger.info("=" * 60)

    train_result = run_training(
        model_name=model_name,
        max_steps=max_steps,
        use_ray_train=True,
    )
    pipeline_results["stages"]["training"] = train_result

    # Stage 3: Evaluation
    if not skip_eval:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 3: Evaluating Model")
        logger.info("=" * 60)

        eval_result = run_evaluation(quick=True)
        pipeline_results["stages"]["evaluation"] = eval_result

    pipeline_results["completed_at"] = datetime.utcnow().isoformat()
    pipeline_results["status"] = "completed"

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)

    train_status = train_result.get("status", "unknown")
    train_loss = train_result.get("training_loss", "N/A")

    logger.info(f"Model: {model_name}")
    logger.info(f"Training Status: {train_status}")
    logger.info(f"Training Loss: {train_loss}")

    if not skip_eval and "evaluation" in pipeline_results["stages"]:
        eval_passed = pipeline_results["stages"]["evaluation"].get("overall_passed", False)
        logger.info(f"Evaluation: {'PASSED' if eval_passed else 'FAILED'}")

    checkpoint = train_result.get("checkpoint_path", train_result.get("merged_model_path"))
    if checkpoint:
        logger.info(f"Checkpoint: {checkpoint}")

    return pipeline_results


def list_modal_datasets() -> Dict[str, Any]:
    """List datasets available in Modal volume."""
    from medai_compass.modal.data_download import app, list_datasets

    logger.info("Listing datasets in Modal volume...")

    # Run within Modal app context
    with app.run():
        result = list_datasets.remote()

    logger.info(f"\nFound {result.get('total_datasets', 0)} datasets:")
    for name, info in result.get("datasets", {}).items():
        logger.info(f"\n  {name}:")
        logger.info(f"    Total samples: {info.get('total_samples', 0)}")
        for split, count in info.get("splits", {}).items():
            logger.info(f"    - {split}: {count} samples")

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ray Training Pipeline with Modal GPU Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify Modal GPU setup
  python scripts/ray_modal_train.py verify

  # Download datasets
  python scripts/ray_modal_train.py download --datasets medqa,pubmedqa

  # List available datasets
  python scripts/ray_modal_train.py list-datasets

  # Train MedGemma 4B
  python scripts/ray_modal_train.py train --model medgemma-4b --max-steps 1000

  # Train MedGemma 27B on 8x H100
  python scripts/ray_modal_train.py train --model medgemma-27b --max-steps 5000

  # Evaluate model
  python scripts/ray_modal_train.py evaluate --quick

  # Run full pipeline
  python scripts/ray_modal_train.py pipeline --model medgemma-4b
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify Modal GPU setup")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download datasets")
    download_parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated dataset names or 'all'",
    )
    download_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per dataset",
    )

    # List datasets command
    list_parser = subparsers.add_parser("list-datasets", help="List available datasets")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument(
        "--model",
        type=str,
        default="medgemma-4b",
        choices=["medgemma-4b", "medgemma-27b"],
        help="Model to train",
    )
    train_parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum training steps",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    train_parser.add_argument(
        "--train-data",
        type=str,
        default="/data/combined_medical/train.jsonl",
        help="Path to training data",
    )
    train_parser.add_argument(
        "--eval-data",
        type=str,
        default="/data/combined_medical/eval.jsonl",
        help="Path to evaluation data",
    )
    train_parser.add_argument(
        "--no-ray",
        action="store_true",
        help="Use direct Modal execution instead of Ray",
    )
    train_parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking URI",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    eval_parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick evaluation with smaller samples",
    )
    eval_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per benchmark",
    )
    eval_parser.add_argument(
        "--benchmark",
        type=str,
        choices=["medqa", "pubmedqa", "safety", "hallucination"],
        default=None,
        help="Run specific benchmark",
    )

    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline")
    pipeline_parser.add_argument(
        "--model",
        type=str,
        default="medgemma-4b",
        choices=["medgemma-4b", "medgemma-27b"],
        help="Model to train",
    )
    pipeline_parser.add_argument(
        "--datasets",
        type=str,
        default="medqa,pubmedqa,medmcqa",
        help="Comma-separated dataset names",
    )
    pipeline_parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum training steps",
    )
    pipeline_parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip dataset download",
    )
    pipeline_parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Check dependencies
    if not check_dependencies():
        return 1

    # Check Modal auth (except for verify which will show more info)
    if args.command != "verify" and not check_modal_auth():
        return 1

    # Initialize Ray for commands that need it
    if args.command in ["train", "evaluate", "pipeline"]:
        init_ray()

    # Execute command
    if args.command == "verify":
        result = verify_modal_setup()
    elif args.command == "download":
        datasets = args.datasets.split(",") if args.datasets != "all" else ["all"]
        result = download_datasets(datasets, args.max_samples)
    elif args.command == "list-datasets":
        result = list_modal_datasets()
    elif args.command == "train":
        result = run_training(
            model_name=args.model,
            train_data_path=args.train_data,
            eval_data_path=args.eval_data,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_ray_train=not args.no_ray,
            mlflow_uri=args.mlflow_uri,
        )
    elif args.command == "evaluate":
        result = run_evaluation(
            checkpoint_path=args.checkpoint,
            quick=args.quick,
            max_samples=args.max_samples,
            benchmark=args.benchmark,
        )
    elif args.command == "pipeline":
        datasets = args.datasets.split(",")
        result = run_full_pipeline(
            model_name=args.model,
            datasets=datasets,
            max_steps=args.max_steps,
            skip_download=args.skip_download,
            skip_eval=args.skip_eval,
        )
    else:
        parser.print_help()
        return 1

    # Print result
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())

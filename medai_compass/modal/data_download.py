"""Modal Data Download Script for MedGemma Training.

Downloads medical datasets to Modal volume for distributed training.

Supported Datasets:
- MedQA: Medical licensing exam questions (USMLE, etc.)
- PubMedQA: Biomedical research QA from PubMed abstracts
- MedMCQA: Medical entrance exam multiple choice questions
- Custom JSONL files

Volume Setup (run once):
    modal volume create medgemma-data
    modal volume create medgemma-checkpoints
    modal volume create medgemma-model-cache

Deployment:
    modal deploy medai_compass/modal/data_download.py

Usage:
    # Download all datasets
    modal run medai_compass/modal/data_download.py::download_all

    # Download specific dataset
    modal run medai_compass/modal/data_download.py::download_dataset --dataset-name medqa

    # Upload custom JSONL
    modal run medai_compass/modal/data_download.py::upload_custom --local-path /path/to/data.jsonl
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

# Check if Modal is available
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    modal = None

logger = logging.getLogger(__name__)

# Dataset configurations
DATASET_CONFIGS = {
    "medqa": {
        "hf_name": "GBaker/MedQA-USMLE-4-options",
        "description": "Medical licensing exam questions (USMLE)",
        "splits": ["train", "validation", "test"],
        "format": "alpaca",
        "instruction_key": "question",
        "output_key": "answer",
        "size_estimate": "12K examples",
    },
    "pubmedqa": {
        "hf_name": "qiaojin/PubMedQA",
        "hf_subset": "pqa_labeled",
        "description": "Biomedical research QA from PubMed abstracts",
        "splits": ["train"],
        "format": "qa",
        "instruction_key": "question",
        "output_key": "long_answer",
        "size_estimate": "1K examples",
    },
    "medmcqa": {
        "hf_name": "openlifescienceai/medmcqa",
        "description": "Medical entrance exam multiple choice questions",
        "splits": ["train", "validation", "test"],
        "format": "alpaca",
        "instruction_key": "question",
        "output_key": "cop",  # Correct option
        "size_estimate": "194K examples",
    },
    "medical_meadow_medqa": {
        "hf_name": "medalpaca/medical_meadow_medqa",
        "description": "Medical Meadow MedQA dataset (Alpaca format)",
        "splits": ["train"],
        "format": "alpaca",
        "instruction_key": "instruction",
        "output_key": "output",
        "size_estimate": "10K examples",
    },
}

if MODAL_AVAILABLE:
    # Create Modal app
    app = modal.App("medai-compass-data")

    # Data download image
    data_image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "curl")
        .pip_install(
            "datasets>=2.17.0",
            "huggingface_hub>=0.20.0",
            "pandas>=2.0.0",
            "pyarrow>=15.0.0",
            "tqdm>=4.66.0",
        )
        .env({
            "HF_HOME": "/root/.cache/huggingface",
            "TRANSFORMERS_CACHE": "/root/.cache/huggingface/transformers",
        })
    )

    # Volume for training data
    training_data = modal.Volume.from_name("medgemma-data", create_if_missing=True)
    model_cache = modal.Volume.from_name("medgemma-model-cache", create_if_missing=True)

    @app.function(
        image=data_image,
        volumes={
            "/data": training_data,
            "/root/.cache/huggingface": model_cache,
        },
        timeout=3600,  # 1 hour
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    def download_dataset(
        dataset_name: str,
        output_format: str = "jsonl",
        max_samples: int | None = None,
    ) -> dict[str, Any]:
        """
        Download a medical dataset to Modal volume.

        Args:
            dataset_name: Name of dataset (medqa, pubmedqa, medmcqa, etc.)
            output_format: Output format (jsonl or parquet)
            max_samples: Optional limit on number of samples per split

        Returns:
            Dict with download status and metadata
        """
        import pandas as pd
        from datasets import load_dataset
        from tqdm import tqdm

        if dataset_name not in DATASET_CONFIGS:
            return {
                "status": "error",
                "message": f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}",
            }

        config = DATASET_CONFIGS[dataset_name]
        print(f"Downloading {dataset_name}: {config['description']}")
        print(f"Estimated size: {config['size_estimate']}")

        try:
            # Load from HuggingFace
            hf_name = config["hf_name"]
            hf_subset = config.get("hf_subset")

            if hf_subset:
                dataset = load_dataset(hf_name, hf_subset)
            else:
                dataset = load_dataset(hf_name)

            # Create output directory
            output_dir = Path(f"/data/{dataset_name}")
            output_dir.mkdir(parents=True, exist_ok=True)

            results = {
                "status": "success",
                "dataset": dataset_name,
                "description": config["description"],
                "splits": {},
                "output_dir": str(output_dir),
            }

            # Process each split
            for split in config["splits"]:
                if split not in dataset:
                    print(f"Split {split} not found, skipping")
                    continue

                split_data = dataset[split]

                # Apply sample limit if specified
                if max_samples and len(split_data) > max_samples:
                    split_data = split_data.select(range(max_samples))

                # Convert to training format
                formatted_data = convert_to_training_format(
                    split_data,
                    config["format"],
                    config["instruction_key"],
                    config["output_key"],
                )

                # Save in requested format
                if output_format == "jsonl":
                    output_path = output_dir / f"{split}.jsonl"
                    with open(output_path, "w") as f:
                        for item in tqdm(formatted_data, desc=f"Writing {split}"):
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")
                else:
                    output_path = output_dir / f"{split}.parquet"
                    df = pd.DataFrame(formatted_data)
                    df.to_parquet(output_path, index=False)

                results["splits"][split] = {
                    "samples": len(formatted_data),
                    "path": str(output_path),
                }
                print(f"Saved {split}: {len(formatted_data)} samples to {output_path}")

            # Commit volume
            training_data.commit()

            return results

        except Exception as e:
            return {
                "status": "error",
                "dataset": dataset_name,
                "message": str(e),
            }

    @app.function(
        image=data_image,
        volumes={
            "/data": training_data,
            "/root/.cache/huggingface": model_cache,
        },
        timeout=7200,  # 2 hours
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    def download_all(
        max_samples: int | None = None,
        datasets: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Download all medical datasets to Modal volume.

        Args:
            max_samples: Optional limit per dataset
            datasets: Optional list of specific datasets to download

        Returns:
            Summary of all downloads
        """
        target_datasets = datasets or list(DATASET_CONFIGS.keys())

        results = {
            "status": "in_progress",
            "started_at": datetime.utcnow().isoformat(),
            "datasets": {},
        }

        for dataset_name in target_datasets:
            print(f"\n{'='*50}")
            print(f"Downloading: {dataset_name}")
            print(f"{'='*50}")

            result = download_dataset.local(
                dataset_name=dataset_name,
                max_samples=max_samples,
            )
            results["datasets"][dataset_name] = result

        # Create combined training file
        print("\nCreating combined training file...")
        combine_result = combine_datasets.local(
            datasets=target_datasets,
            output_name="combined_medical",
        )
        results["combined"] = combine_result

        results["status"] = "completed"
        results["completed_at"] = datetime.utcnow().isoformat()

        return results

    @app.function(
        image=data_image,
        volumes={"/data": training_data},
        timeout=1800,
    )
    def combine_datasets(
        datasets: list[str],
        output_name: str = "combined",
        shuffle: bool = True,
    ) -> dict[str, Any]:
        """
        Combine multiple datasets into a single training file.

        Args:
            datasets: List of dataset names to combine
            output_name: Name for combined output
            shuffle: Whether to shuffle combined data

        Returns:
            Combined dataset metadata
        """
        import random

        combined_train = []
        combined_eval = []

        for dataset_name in datasets:
            data_dir = Path(f"/data/{dataset_name}")
            if not data_dir.exists():
                print(f"Dataset {dataset_name} not found, skipping")
                continue

            # Load train split
            train_file = data_dir / "train.jsonl"
            if train_file.exists():
                with open(train_file) as f:
                    for line in f:
                        item = json.loads(line)
                        item["source"] = dataset_name
                        combined_train.append(item)

            # Load validation split
            for eval_name in ["validation", "test", "dev"]:
                eval_file = data_dir / f"{eval_name}.jsonl"
                if eval_file.exists():
                    with open(eval_file) as f:
                        for line in f:
                            item = json.loads(line)
                            item["source"] = dataset_name
                            combined_eval.append(item)
                    break

        # Shuffle if requested
        if shuffle:
            random.shuffle(combined_train)
            random.shuffle(combined_eval)

        # Save combined files
        output_dir = Path(f"/data/{output_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / "train.jsonl"
        with open(train_path, "w") as f:
            for item in combined_train:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        eval_path = output_dir / "eval.jsonl"
        with open(eval_path, "w") as f:
            for item in combined_eval:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        training_data.commit()

        return {
            "status": "success",
            "output_dir": str(output_dir),
            "train_samples": len(combined_train),
            "eval_samples": len(combined_eval),
            "train_path": str(train_path),
            "eval_path": str(eval_path),
            "sources": datasets,
        }

    @app.function(
        image=data_image,
        volumes={"/data": training_data},
        timeout=600,
    )
    def list_datasets() -> dict[str, Any]:
        """
        List all datasets available in the volume.

        Returns:
            Dict with available datasets and their sizes
        """
        data_dir = Path("/data")

        datasets = {}
        for item in data_dir.iterdir():
            if item.is_dir():
                files = list(item.glob("*.jsonl")) + list(item.glob("*.parquet"))

                total_samples = 0
                splits = {}
                for f in files:
                    if f.suffix == ".jsonl":
                        with open(f) as fp:
                            count = sum(1 for _ in fp)
                    else:
                        import pandas as pd
                        count = len(pd.read_parquet(f))

                    split_name = f.stem
                    splits[split_name] = count
                    total_samples += count

                datasets[item.name] = {
                    "total_samples": total_samples,
                    "splits": splits,
                    "files": [str(f) for f in files],
                }

        return {
            "datasets": datasets,
            "total_datasets": len(datasets),
        }

    @app.function(
        image=data_image,
        volumes={"/data": training_data},
        timeout=1800,
    )
    def upload_custom(
        data: list[dict],
        output_name: str,
        split: str = "train",
    ) -> dict[str, Any]:
        """
        Upload custom training data to Modal volume.

        Args:
            data: List of training examples (dict with instruction, input, output)
            output_name: Name for the dataset
            split: Split name (train, eval, test)

        Returns:
            Upload status and metadata
        """
        output_dir = Path(f"/data/{output_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{split}.jsonl"

        with open(output_path, "w") as f:
            for item in data:
                # Ensure proper format
                formatted = {
                    "instruction": item.get("instruction", item.get("question", "")),
                    "input": item.get("input", item.get("context", "")),
                    "output": item.get("output", item.get("answer", "")),
                }
                f.write(json.dumps(formatted, ensure_ascii=False) + "\n")

        training_data.commit()

        return {
            "status": "success",
            "output_path": str(output_path),
            "samples": len(data),
        }

    @app.function(
        image=data_image,
        volumes={"/data": training_data},
        timeout=600,
    )
    def validate_dataset(dataset_name: str) -> dict[str, Any]:
        """
        Validate a dataset for training readiness.

        Checks:
        - File exists and is readable
        - Proper JSONL format
        - Required fields present
        - Sample distribution

        Args:
            dataset_name: Name of dataset to validate

        Returns:
            Validation results
        """
        data_dir = Path(f"/data/{dataset_name}")

        if not data_dir.exists():
            return {
                "status": "error",
                "message": f"Dataset {dataset_name} not found",
            }

        results = {
            "status": "success",
            "dataset": dataset_name,
            "splits": {},
            "issues": [],
        }

        for jsonl_file in data_dir.glob("*.jsonl"):
            split_name = jsonl_file.stem
            split_results = {
                "file": str(jsonl_file),
                "samples": 0,
                "valid": 0,
                "invalid": 0,
                "missing_fields": {},
            }

            required_fields = {"instruction", "output"}

            with open(jsonl_file) as f:
                for i, line in enumerate(f):
                    split_results["samples"] += 1

                    try:
                        item = json.loads(line)

                        # Check required fields
                        missing = required_fields - set(item.keys())
                        if missing:
                            split_results["invalid"] += 1
                            for field in missing:
                                split_results["missing_fields"][field] = (
                                    split_results["missing_fields"].get(field, 0) + 1
                                )
                        else:
                            split_results["valid"] += 1

                    except json.JSONDecodeError:
                        split_results["invalid"] += 1
                        results["issues"].append(f"Line {i+1} in {split_name}: Invalid JSON")

            results["splits"][split_name] = split_results

        # Overall status
        total_invalid = sum(s["invalid"] for s in results["splits"].values())
        if total_invalid > 0:
            results["status"] = "warning"
            results["message"] = f"{total_invalid} invalid samples found"

        return results

    @app.local_entrypoint()
    def main(
        dataset: str = "all",
        max_samples: int | None = None,
        list_only: bool = False,
        validate: str | None = None,
    ):
        """
        Download medical datasets for training.

        Examples:
            # List available datasets in volume
            modal run medai_compass/modal/data_download.py --list-only

            # Download all datasets
            modal run medai_compass/modal/data_download.py

            # Download specific dataset
            modal run medai_compass/modal/data_download.py --dataset medqa

            # Download with sample limit (for testing)
            modal run medai_compass/modal/data_download.py --dataset medqa --max-samples 100

            # Validate dataset
            modal run medai_compass/modal/data_download.py --validate combined_medical
        """
        if list_only:
            print("Listing datasets in volume...")
            result = list_datasets.remote()
            print(f"\nFound {result['total_datasets']} datasets:")
            for name, info in result["datasets"].items():
                print(f"\n  {name}:")
                print(f"    Total samples: {info['total_samples']}")
                for split, count in info["splits"].items():
                    print(f"    - {split}: {count} samples")
            return

        if validate:
            print(f"Validating dataset: {validate}")
            result = validate_dataset.remote(validate)
            print(f"\nValidation status: {result['status']}")
            for split, info in result.get("splits", {}).items():
                print(f"\n  {split}:")
                print(f"    Valid: {info['valid']}/{info['samples']}")
                if info["missing_fields"]:
                    print(f"    Missing fields: {info['missing_fields']}")
            if result.get("issues"):
                print(f"\nIssues: {result['issues'][:5]}")
            return

        if dataset == "all":
            print("Downloading all medical datasets...")
            result = download_all.remote(max_samples=max_samples)
        else:
            print(f"Downloading dataset: {dataset}")
            result = download_dataset.remote(
                dataset_name=dataset,
                max_samples=max_samples,
            )

        print(f"\nResult: {json.dumps(result, indent=2)}")


def convert_to_training_format(
    dataset,
    format_type: str,
    instruction_key: str,
    output_key: str,
) -> list[dict]:
    """
    Convert dataset to standard training format.

    Args:
        dataset: HuggingFace dataset
        format_type: Format type (alpaca, qa, chat)
        instruction_key: Key for instruction/question
        output_key: Key for output/answer

    Returns:
        List of formatted training examples
    """
    formatted = []

    for item in dataset:
        instruction = item.get(instruction_key, "")
        output = item.get(output_key, "")

        # Handle MedMCQA special format (multiple choice)
        if format_type == "alpaca" and "opa" in item:
            # Build options string
            options = []
            for opt_key in ["opa", "opb", "opc", "opd"]:
                if opt_key in item and item[opt_key]:
                    options.append(f"{opt_key[-1].upper()}. {item[opt_key]}")

            instruction = f"{instruction}\n\nOptions:\n" + "\n".join(options)

            # Get correct answer letter
            cop = item.get("cop", 0)  # 0-indexed
            if isinstance(cop, int) and 0 <= cop <= 3:
                output = ["A", "B", "C", "D"][cop]

        # Handle PubMedQA special format
        if format_type == "qa" and "context" in item:
            context = item.get("context", {})
            if isinstance(context, dict):
                context_text = " ".join(context.get("contexts", []))
                instruction = f"Context: {context_text}\n\nQuestion: {instruction}"

        # Build formatted example
        formatted.append({
            "instruction": instruction,
            "input": item.get("input", item.get("context_text", "")),
            "output": str(output) if output else "",
        })

    return formatted


# For non-Modal environments
if not MODAL_AVAILABLE:
    def download_dataset(*args, **kwargs):
        raise ImportError("Modal is not installed. Install with: pip install modal")

    def download_all(*args, **kwargs):
        raise ImportError("Modal is not installed. Install with: pip install modal")

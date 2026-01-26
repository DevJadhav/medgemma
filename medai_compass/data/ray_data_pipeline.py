"""
Ray Data Pipeline for MedGemma Training.

Provides distributed data loading with:
- Streaming data processing (no OOM)
- Parallel tokenization
- Automatic batching
- Integration with Ray Train
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


def format_alpaca_prompt(example: Dict[str, str]) -> str:
    """
    Format example in Alpaca instruction format.

    Args:
        example: Dict with 'instruction', 'input' (optional), 'output' keys

    Returns:
        Formatted prompt string
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        return f"""### Instruction:
{instruction}

### Response:
{output}"""


def format_chat_prompt(example: Dict[str, Any]) -> str:
    """
    Format example in chat format.

    Args:
        example: Dict with 'messages' key containing list of role/content dicts

    Returns:
        Formatted chat string
    """
    messages = example.get("messages", [])
    formatted = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted.append(f"<|{role}|>\n{content}")

    return "\n".join(formatted)


def format_qa_prompt(example: Dict[str, str]) -> str:
    """
    Format example in Q&A format.

    Args:
        example: Dict with 'question' and 'answer' keys

    Returns:
        Formatted Q&A string
    """
    question = example.get("question", "")
    answer = example.get("answer", "")

    return f"""Question: {question}

Answer: {answer}"""


def tokenize_batch(
    texts: List[str],
    tokenizer: Any,
    max_length: int = 2048,
    padding: str = "max_length",
    truncation: bool = True,
) -> Dict[str, List]:
    """
    Tokenize a batch of texts.

    Args:
        texts: List of text strings
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        padding: Padding strategy
        truncation: Whether to truncate

    Returns:
        Dict with input_ids, attention_mask, labels
    """
    tokenized = tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=None,
    )

    # Add labels (same as input_ids for causal LM)
    tokenized["labels"] = tokenized["input_ids"].copy()

    # Add token_type_ids if not present (for Gemma3)
    if "token_type_ids" not in tokenized:
        batch_size = len(tokenized["input_ids"])
        seq_len = len(tokenized["input_ids"][0]) if batch_size > 0 else 0
        tokenized["token_type_ids"] = [[0] * seq_len for _ in range(batch_size)]

    return tokenized


def load_jsonl_dataset(file_path: str) -> "Dataset":
    """
    Load dataset from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        HuggingFace Dataset
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "json",
            data_files=file_path,
            split="train",
        )
        return dataset

    except Exception as e:
        logger.error(f"Failed to load dataset from {file_path}: {e}")
        raise


class RayDataPipeline:
    """
    Ray Data pipeline for distributed data processing.

    Features:
    - Streaming data loading (no OOM)
    - Parallel preprocessing
    - Automatic batching
    - Integration with Ray Train
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str,
        max_length: int = 2048,
        num_workers: int = 4,
        format_type: str = "alpaca",
    ):
        """
        Initialize Ray Data pipeline.

        Args:
            data_path: Path to data directory or file
            tokenizer_name: Name of HuggingFace tokenizer
            max_length: Maximum sequence length
            num_workers: Number of parallel workers
            format_type: Format type (alpaca, chat, qa)
        """
        self.data_path = data_path
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.num_workers = num_workers
        self.format_type = format_type
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = self._get_tokenizer()
        return self._tokenizer

    def _get_tokenizer(self):
        """Get tokenizer (cached)."""
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer

        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def _get_format_fn(self) -> Callable:
        """Get format function based on type."""
        format_fns = {
            "alpaca": format_alpaca_prompt,
            "chat": format_chat_prompt,
            "qa": format_qa_prompt,
        }
        return format_fns.get(self.format_type, format_alpaca_prompt)

    def load_dataset(self) -> "ray.data.Dataset":
        """
        Load dataset using Ray Data.

        Returns:
            Ray Data Dataset
        """
        try:
            import ray.data as rd

            # Determine file pattern
            if os.path.isdir(self.data_path):
                pattern = os.path.join(self.data_path, "*.jsonl")
            else:
                pattern = self.data_path

            # Read JSONL files in parallel
            ds = rd.read_json(
                pattern,
                parallelism=self.num_workers,
            )

            return ds

        except ImportError:
            logger.warning("Ray Data not available, using HuggingFace datasets")
            return self._load_hf_dataset()

    def _load_hf_dataset(self):
        """Fallback to HuggingFace datasets."""
        from datasets import load_dataset

        if os.path.isdir(self.data_path):
            data_files = list(Path(self.data_path).glob("*.jsonl"))
            if not data_files:
                raise FileNotFoundError(f"No JSONL files in {self.data_path}")
            return load_dataset("json", data_files=str(data_files[0]), split="train")
        else:
            return load_dataset("json", data_files=self.data_path, split="train")

    def preprocess(self, ds: "ray.data.Dataset") -> "ray.data.Dataset":
        """
        Apply preprocessing transformations.

        Args:
            ds: Ray Data Dataset

        Returns:
            Preprocessed dataset
        """
        try:
            import ray

            # Put tokenizer in object store for sharing
            tokenizer_ref = ray.put(self.tokenizer)
            format_fn = self._get_format_fn()
            max_length = self.max_length

            def tokenize_batch_fn(batch: Dict[str, List]) -> Dict[str, List]:
                """Tokenize a batch of examples."""
                tok = ray.get(tokenizer_ref)

                # Format prompts
                texts = []
                for i in range(len(batch.get("instruction", batch.get("text", [])))):
                    example = {k: v[i] if isinstance(v, list) and i < len(v) else ""
                               for k, v in batch.items()}
                    texts.append(format_fn(example))

                # Tokenize
                return tokenize_batch(texts, tok, max_length=max_length)

            return ds.map_batches(
                tokenize_batch_fn,
                batch_size=1000,
                num_cpus=1,
            )

        except ImportError:
            # Fallback for non-Ray environment
            return ds

    def get_train_dataset(self) -> "ray.data.Dataset":
        """
        Get preprocessed training dataset.

        Returns:
            Preprocessed Ray Data Dataset
        """
        ds = self.load_dataset()
        ds = self.preprocess(ds)

        # Shuffle for training
        try:
            ds = ds.random_shuffle()
        except AttributeError:
            pass

        return ds

    def to_torch_dataloader(
        self,
        ds: "ray.data.Dataset",
        batch_size: int = 8,
        prefetch_batches: int = 2,
    ):
        """
        Convert to PyTorch DataLoader.

        Args:
            ds: Ray Data Dataset
            batch_size: Batch size
            prefetch_batches: Number of batches to prefetch

        Returns:
            PyTorch-compatible iterator
        """
        try:
            return ds.iter_torch_batches(
                batch_size=batch_size,
                prefetch_batches=prefetch_batches,
                local_shuffle_buffer_size=1000,
            )
        except AttributeError:
            # Fallback for HuggingFace datasets
            from torch.utils.data import DataLoader

            return DataLoader(ds, batch_size=batch_size, shuffle=True)


def create_ray_data_pipeline(cfg: "DictConfig") -> RayDataPipeline:
    """
    Create Ray Data pipeline from Hydra config.

    Args:
        cfg: Hydra configuration

    Returns:
        RayDataPipeline instance
    """
    data_cfg = cfg.data if hasattr(cfg, "data") else {}

    return RayDataPipeline(
        data_path=getattr(data_cfg, "path", "/data/combined_medical"),
        tokenizer_name=cfg.model.name,
        max_length=getattr(data_cfg, "max_length", 2048),
        num_workers=getattr(data_cfg, "num_workers", 4),
    )

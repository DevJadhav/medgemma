"""Data pipeline for MedGemma training.

Provides distributed data loading and preprocessing using Ray Data
for efficient training of medical AI models.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading."""
    
    train_path: Optional[str] = None
    eval_path: Optional[str] = None
    max_seq_length: int = 8192
    
    # Tokenization
    padding: str = "max_length"
    truncation: bool = True
    
    # Batching
    batch_size: int = 4
    shuffle: bool = True
    
    # Ray Data settings
    use_ray_data: bool = True
    read_parallelism: int = 100
    map_parallelism: int = 20
    prefetch_batches: int = 2


class MedicalDataPipeline:
    """
    Data pipeline for medical AI training.
    
    Supports:
    - Loading from local files, S3/MinIO, or HuggingFace datasets
    - Distributed preprocessing with Ray Data
    - Medical-specific tokenization and formatting
    """
    
    def __init__(
        self,
        tokenizer,
        config: Optional[DataConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize data pipeline.
        
        Args:
            tokenizer: HuggingFace tokenizer
            config: Data configuration
            system_prompt: Optional system prompt for instruction tuning
        """
        self.tokenizer = tokenizer
        self.config = config or DataConfig()
        
        self.system_prompt = system_prompt or (
            "You are a medical AI assistant trained to help healthcare providers "
            "with clinical reasoning and diagnosis. Provide evidence-based, "
            "accurate medical information."
        )
        
        # Ensure padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_dataset(
        self,
        data_path: str,
        split: str = "train",
    ):
        """
        Load dataset from path.
        
        Supports:
        - Local JSONL files
        - S3/MinIO paths (s3://bucket/path)
        - HuggingFace datasets (hf://dataset_name)
        
        Args:
            data_path: Path to data
            split: Dataset split (train/eval/test)
            
        Returns:
            Ray Dataset or HuggingFace Dataset
        """
        if data_path.startswith("s3://"):
            return self._load_from_s3(data_path, split)
        elif data_path.startswith("hf://"):
            return self._load_from_huggingface(data_path[5:], split)
        else:
            return self._load_from_local(data_path, split)
    
    def _load_from_local(self, path: str, split: str):
        """Load dataset from local filesystem."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        if self.config.use_ray_data:
            return self._load_with_ray_data(str(path))
        else:
            return self._load_with_datasets(str(path), split)
    
    def _load_from_s3(self, path: str, split: str):
        """Load dataset from S3/MinIO."""
        if self.config.use_ray_data:
            return self._load_with_ray_data(path)
        else:
            # Download to temp location and load
            local_path = self._download_from_s3(path)
            return self._load_with_datasets(local_path, split)
    
    def _load_from_huggingface(self, dataset_name: str, split: str):
        """Load dataset from HuggingFace Hub."""
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(dataset_name, split=split)
            
            if self.config.use_ray_data:
                import ray.data
                return ray.data.from_huggingface(dataset)
            
            return dataset
            
        except ImportError:
            raise ImportError("datasets package required: pip install datasets")
    
    def _load_with_ray_data(self, path: str):
        """Load data using Ray Data for distributed processing."""
        try:
            import ray.data
            
            if path.endswith(".jsonl") or path.endswith(".json"):
                return ray.data.read_json(
                    path,
                    parallelism=self.config.read_parallelism,
                )
            elif path.endswith(".parquet"):
                return ray.data.read_parquet(
                    path,
                    parallelism=self.config.read_parallelism,
                )
            else:
                raise ValueError(f"Unsupported file format: {path}")
                
        except ImportError:
            raise ImportError("Ray Data required: pip install 'ray[data]'")
    
    def _load_with_datasets(self, path: str, split: str):
        """Load data using HuggingFace datasets."""
        try:
            from datasets import load_dataset
            
            if path.endswith(".jsonl") or path.endswith(".json"):
                return load_dataset("json", data_files=path, split=split)
            elif path.endswith(".parquet"):
                return load_dataset("parquet", data_files=path, split=split)
            else:
                raise ValueError(f"Unsupported file format: {path}")
                
        except ImportError:
            raise ImportError("datasets package required: pip install datasets")
    
    def _download_from_s3(self, s3_path: str) -> str:
        """Download file from S3 to local temp directory."""
        import tempfile
        
        try:
            from minio import Minio
            
            # Parse S3 path
            parts = s3_path.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            
            # Get MinIO client
            endpoint = os.environ.get("MINIO_ENDPOINT", "minio:9000")
            access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
            secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
            
            client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=False,
            )
            
            # Download to temp file
            temp_dir = tempfile.mkdtemp()
            filename = Path(key).name
            local_path = os.path.join(temp_dir, filename)
            
            client.fget_object(bucket, key, local_path)
            
            return local_path
            
        except ImportError:
            raise ImportError("minio package required: pip install minio")
    
    def preprocess(self, dataset) -> Any:
        """
        Preprocess dataset for training.
        
        Applies tokenization and formatting for instruction tuning.
        
        Args:
            dataset: Raw dataset
            
        Returns:
            Preprocessed dataset
        """
        if self.config.use_ray_data:
            return self._preprocess_ray_data(dataset)
        else:
            return self._preprocess_hf_dataset(dataset)
    
    def _preprocess_ray_data(self, dataset):
        """Preprocess using Ray Data."""
        import ray.data
        
        def tokenize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
            """Tokenize a batch of examples."""
            texts = self._format_examples(batch)
            
            tokenized = self.tokenizer(
                texts,
                max_length=self.config.max_seq_length,
                padding=self.config.padding,
                truncation=self.config.truncation,
                return_tensors="np",
            )
            
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": tokenized["input_ids"].copy(),  # For causal LM
            }
        
        return dataset.map_batches(
            tokenize_batch,
            batch_size=self.config.batch_size,
            num_cpus=1,
        )
    
    def _preprocess_hf_dataset(self, dataset):
        """Preprocess using HuggingFace datasets."""
        
        def tokenize_function(examples):
            """Tokenize examples."""
            texts = self._format_examples(examples)
            
            tokenized = self.tokenizer(
                texts,
                max_length=self.config.max_seq_length,
                padding=self.config.padding,
                truncation=self.config.truncation,
            )
            
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
    
    def _format_examples(self, examples: Dict[str, List]) -> List[str]:
        """
        Format examples for instruction tuning.
        
        Expects examples with 'instruction', 'input', and 'output' fields
        (Alpaca format) or 'messages' field (chat format).
        """
        texts = []
        
        # Check format
        if "messages" in examples:
            # Chat format
            for messages in examples["messages"]:
                text = self._format_chat_messages(messages)
                texts.append(text)
                
        elif "instruction" in examples:
            # Alpaca format
            n_examples = len(examples["instruction"])
            
            for i in range(n_examples):
                instruction = examples["instruction"][i]
                input_text = examples.get("input", [""] * n_examples)[i]
                output = examples["output"][i]
                
                text = self._format_alpaca(instruction, input_text, output)
                texts.append(text)
                
        elif "question" in examples and "answer" in examples:
            # QA format
            n_examples = len(examples["question"])
            
            for i in range(n_examples):
                question = examples["question"][i]
                answer = examples["answer"][i]
                
                text = self._format_qa(question, answer)
                texts.append(text)
        else:
            # Raw text
            texts = examples.get("text", [])
        
        return texts
    
    def _format_chat_messages(self, messages: List[Dict]) -> str:
        """Format chat messages using tokenizer's chat template."""
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fallback to simple formatting
            text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                text += f"<{role}>\n{content}\n</{role}>\n"
            return text
    
    def _format_alpaca(
        self, 
        instruction: str, 
        input_text: str, 
        output: str
    ) -> str:
        """Format in Alpaca instruction format."""
        if input_text:
            prompt = f"""### System:
{self.system_prompt}

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            prompt = f"""### System:
{self.system_prompt}

### Instruction:
{instruction}

### Response:
{output}"""
        
        return prompt
    
    def _format_qa(self, question: str, answer: str) -> str:
        """Format in QA format."""
        return f"""### System:
{self.system_prompt}

### Question:
{question}

### Answer:
{answer}"""


def create_training_dataset(
    train_path: str,
    tokenizer,
    config: Optional[DataConfig] = None,
    system_prompt: Optional[str] = None,
):
    """
    Create training dataset with preprocessing.
    
    Convenience function for common training setup.
    
    Args:
        train_path: Path to training data
        tokenizer: HuggingFace tokenizer
        config: Optional data configuration
        system_prompt: Optional system prompt
        
    Returns:
        Preprocessed dataset ready for training
    """
    pipeline = MedicalDataPipeline(
        tokenizer=tokenizer,
        config=config,
        system_prompt=system_prompt,
    )
    
    dataset = pipeline.load_dataset(train_path, split="train")
    return pipeline.preprocess(dataset)

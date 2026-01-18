"""Ray Data Pipeline for Medical Data Processing.

Distributed data pipeline using Ray Data for preprocessing medical datasets
including Synthea, MedQuAD, and other medical data sources.

Supports:
- MedGemma 4B IT (single GPU, batch_size=8)
- MedGemma 27B IT (8x H100, batch_size=2)

Example:
    >>> from medai_compass.pipelines.ray_pipeline import MedicalDataPipeline
    >>> pipeline = MedicalDataPipeline(model_name="medgemma-4b")
    >>> dataset = pipeline.load_dataset("data/synthea", dataset_type="synthea")
    >>> train_ds, val_ds = pipeline.create_splits(dataset, val_ratio=0.1)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import yaml

# Conditional Ray import for environments without Ray
try:
    import ray
    from ray import data as ray_data
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    ray = None
    ray_data = None

from medai_compass.training.model_selector import _resolve_model_name, _load_models_config

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the medical data pipeline."""
    
    # Model settings
    model_name: str = "medgemma_4b_it"
    max_length: int = 8192
    batch_size: int = 8
    
    # PHI filtering (strict by default per user requirement)
    phi_filter_enabled: bool = True
    phi_filter_strict: bool = True
    
    # Validation
    validation_enabled: bool = True
    validation_strict: bool = True
    
    # Ray Data settings
    parallelism: Optional[int] = None  # Auto
    prefetch_batches: int = 2
    
    # Data format
    data_format: str = "instruction"
    
    @classmethod
    def for_model(cls, model_name: str) -> "PipelineConfig":
        """Create config for specific model."""
        canonical_name = _resolve_model_name(model_name)
        
        # Load pipeline config
        config_path = Path(__file__).parent.parent.parent / "config" / "pipeline.yaml"
        with open(config_path) as f:
            pipeline_config = yaml.safe_load(f)
        
        # Get model profile
        profile_key = "medgemma_4b" if "4b" in canonical_name else "medgemma_27b"
        profile = pipeline_config.get("model_profiles", {}).get(profile_key, {})
        
        return cls(
            model_name=canonical_name,
            max_length=profile.get("max_length", 8192),
            batch_size=profile.get("batch_size", 8),
            data_format=profile.get("data_format", "instruction"),
        )


class MedicalDataPipeline:
    """
    Distributed data pipeline for medical datasets.
    
    Uses Ray Data for parallel preprocessing with support for:
    - Synthea FHIR bundles
    - MedQuAD QA pairs
    - Generic JSON/JSONL datasets
    
    Attributes:
        model_name: Canonical model name (medgemma_4b_it or medgemma_27b_it)
        config: Pipeline configuration
    """
    
    def __init__(
        self,
        model_name: str = "medgemma-4b",
        config: Optional[PipelineConfig] = None,
    ):
        """
        Initialize the medical data pipeline.
        
        Args:
            model_name: Model name or alias (medgemma-4b, medgemma-27b, etc.)
            config: Optional pipeline configuration
        """
        self.model_name = _resolve_model_name(model_name)
        self.config = config or PipelineConfig.for_model(model_name)
        
        # Initialize PHI filter
        from medai_compass.pipelines.phi_detection import PHIPipelineFilter
        self._phi_filter = PHIPipelineFilter(
            strict=self.config.phi_filter_strict
        )
        
        # Initialize validator
        from medai_compass.pipelines.validation import DataValidator
        self._validator = DataValidator(
            strict=self.config.validation_strict
        )
        
        logger.info(f"Initialized MedicalDataPipeline for {self.model_name}")
        logger.info(f"  PHI filter: {'strict' if self.config.phi_filter_strict else 'mask'}")
        logger.info(f"  Batch size: {self.config.batch_size}")
    
    def load_dataset(
        self,
        source_path: str,
        dataset_type: str = "generic",
        **kwargs
    ) -> "ray_data.Dataset":
        """
        Load a dataset from the given path.
        
        Args:
            source_path: Path to dataset directory or file
            dataset_type: Type of dataset (synthea, medquad, generic)
            **kwargs: Additional arguments for dataset loading
            
        Returns:
            Ray Dataset with loaded records
        """
        source = Path(source_path)
        
        if not HAS_RAY:
            # Fallback to non-Ray loading for testing
            return self._load_without_ray(source, dataset_type)
        
        if dataset_type == "synthea":
            return self._load_synthea(source, **kwargs)
        elif dataset_type == "medquad":
            return self._load_medquad(source, **kwargs)
        else:
            return self._load_generic(source, **kwargs)
    
    def _load_without_ray(
        self,
        source: Path,
        dataset_type: str
    ) -> "MockDataset":
        """Load dataset without Ray for testing."""
        records = []
        
        # Find JSON files
        if source.is_dir():
            json_files = list(source.glob("*.json"))
        else:
            json_files = [source]
        
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)
                
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                if data.get("resourceType") == "Bundle":
                    # FHIR Bundle
                    records.extend(self._extract_from_fhir_bundle(data))
                else:
                    records.append(data)
        
        return MockDataset(records)
    
    def _load_synthea(self, source: Path, **kwargs) -> "ray_data.Dataset":
        """Load Synthea FHIR bundles."""
        records = []
        
        # Find all FHIR bundle files
        if source.is_dir():
            bundle_files = list(source.glob("*.json"))
        else:
            bundle_files = [source]
        
        for bundle_file in bundle_files:
            with open(bundle_file) as f:
                bundle = json.load(f)
            
            if bundle.get("resourceType") == "Bundle":
                records.extend(self._extract_from_fhir_bundle(bundle))
        
        if HAS_RAY:
            return ray_data.from_items(records)
        return MockDataset(records)
    
    def _extract_from_fhir_bundle(self, bundle: Dict) -> List[Dict]:
        """Extract training records from FHIR bundle."""
        records = []
        
        # Group resources by patient
        patients = {}
        conditions = {}
        
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")
            
            if resource_type == "Patient":
                patient_id = resource.get("id")
                patients[patient_id] = resource
            elif resource_type == "Condition":
                subject_ref = resource.get("subject", {}).get("reference", "")
                patient_id = subject_ref.split("/")[-1] if "/" in subject_ref else subject_ref
                if patient_id not in conditions:
                    conditions[patient_id] = []
                conditions[patient_id].append(resource)
        
        # Create instruction records for each patient
        for patient_id, patient in patients.items():
            patient_conditions = conditions.get(patient_id, [])
            
            # Create clinical reasoning prompt
            condition_list = ", ".join([
                c.get("code", {}).get("coding", [{}])[0].get("display", "Unknown")
                for c in patient_conditions
            ])
            
            record = {
                "instruction": "Based on the patient information, provide a clinical assessment.",
                "input": f"Patient: {patient.get('name', [{}])[0].get('text', 'Unknown')}, "
                        f"Gender: {patient.get('gender', 'unknown')}, "
                        f"Conditions: {condition_list or 'None documented'}",
                "output": f"This patient has the following documented conditions: {condition_list}. "
                         f"Appropriate management should consider these conditions.",
                "source": "synthea",
                "patient_id": patient_id,
            }
            records.append(record)
        
        return records
    
    def _load_medquad(self, source: Path, **kwargs) -> "ray_data.Dataset":
        """Load MedQuAD QA pairs."""
        records = []
        
        # Find all JSON files
        if source.is_dir():
            json_files = list(source.glob("*.json"))
        else:
            json_files = [source]
        
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    # Convert QA to instruction format
                    record = {
                        "question": item.get("question", ""),
                        "answer": item.get("answer", ""),
                        "instruction": "Answer the following medical question accurately and concisely.",
                        "input": item.get("question", ""),
                        "output": item.get("answer", ""),
                        "source": "medquad",
                    }
                    records.append(record)
        
        if HAS_RAY:
            return ray_data.from_items(records)
        return MockDataset(records)
    
    def _load_generic(self, source: Path, **kwargs) -> "ray_data.Dataset":
        """Load generic JSON dataset."""
        records = []
        
        if source.is_dir():
            json_files = list(source.glob("*.json"))
        else:
            json_files = [source]
        
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)
            
            if isinstance(data, list):
                records.extend(data)
            else:
                records.append(data)
        
        if HAS_RAY:
            return ray_data.from_items(records)
        return MockDataset(records)
    
    def preprocess(
        self,
        dataset: "ray_data.Dataset",
        apply_phi_filter: bool = True,
        apply_validation: bool = True,
    ) -> "ray_data.Dataset":
        """
        Preprocess dataset with validation and PHI filtering.
        
        Args:
            dataset: Input Ray Dataset
            apply_phi_filter: Whether to apply PHI filtering (default True)
            apply_validation: Whether to apply validation (default True)
            
        Returns:
            Preprocessed Ray Dataset
        """
        if isinstance(dataset, MockDataset):
            return self._preprocess_mock(dataset, apply_phi_filter, apply_validation)
        
        if not HAS_RAY:
            raise RuntimeError("Ray not available for preprocessing")
        
        # Apply PHI filter
        if apply_phi_filter and self.config.phi_filter_enabled:
            dataset = dataset.filter(self._phi_filter_fn)
        
        # Apply validation
        if apply_validation and self.config.validation_enabled:
            dataset = dataset.filter(self._validation_fn)
        
        # Convert to instruction format
        dataset = dataset.map(self._to_instruction_format)
        
        return dataset
    
    def _preprocess_mock(
        self,
        dataset: "MockDataset",
        apply_phi_filter: bool,
        apply_validation: bool,
    ) -> "MockDataset":
        """Preprocess mock dataset (for testing without Ray)."""
        records = dataset.records
        
        # Apply PHI filter
        if apply_phi_filter and self.config.phi_filter_enabled:
            records = [r for r in records if self._phi_filter_fn(r)]
        
        # Apply validation
        if apply_validation and self.config.validation_enabled:
            records = [r for r in records if self._validation_fn(r)]
        
        # Convert to instruction format
        records = [self._to_instruction_format(r) for r in records]
        
        return MockDataset(records)
    
    def _phi_filter_fn(self, record: Dict) -> bool:
        """Filter function for PHI detection."""
        result = self._phi_filter.filter_record(record)
        return result is not None  # None means filtered out
    
    def _validation_fn(self, record: Dict) -> bool:
        """Filter function for validation."""
        # Determine schema type
        if "question" in record and "answer" in record:
            schema_type = "qa"
        elif "instruction" in record:
            schema_type = "instruction"
        else:
            schema_type = "generic"
        
        result = self._validator.validate(record, schema_type=schema_type)
        return result.is_valid
    
    def _to_instruction_format(self, record: Dict) -> Dict:
        """Convert record to instruction format."""
        # If already in instruction format
        if "instruction" in record and "output" in record:
            return {
                "prompt": f"{record.get('instruction', '')}\n\n{record.get('input', '')}".strip(),
                "completion": record.get("output", ""),
                "instruction": record.get("instruction", ""),
                "input": record.get("input", ""),
                "output": record.get("output", ""),
            }
        
        # Convert QA to instruction
        if "question" in record:
            return {
                "prompt": f"Answer the following medical question:\n\n{record['question']}",
                "completion": record.get("answer", ""),
                "instruction": "Answer the following medical question.",
                "input": record["question"],
                "output": record.get("answer", ""),
            }
        
        # Generic: try to create a prompt
        text_fields = [v for v in record.values() if isinstance(v, str)]
        text = " ".join(text_fields)
        
        return {
            "prompt": text,
            "completion": "",
            "text": text,
        }
    
    def create_splits(
        self,
        dataset: "ray_data.Dataset",
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple["ray_data.Dataset", "ray_data.Dataset"]:
        """
        Create train/validation splits.
        
        Args:
            dataset: Input dataset
            val_ratio: Validation set ratio (default 0.1)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if isinstance(dataset, MockDataset):
            return self._split_mock(dataset, val_ratio, seed)
        
        if not HAS_RAY:
            raise RuntimeError("Ray not available for splitting")
        
        # Use Ray's random_split
        train_ratio = 1.0 - val_ratio
        train_ds, val_ds = dataset.random_split(
            [train_ratio, val_ratio],
            seed=seed
        )
        
        return train_ds, val_ds
    
    def _split_mock(
        self,
        dataset: "MockDataset",
        val_ratio: float,
        seed: int,
    ) -> Tuple["MockDataset", "MockDataset"]:
        """Split mock dataset."""
        import random
        random.seed(seed)
        
        records = dataset.records.copy()
        random.shuffle(records)
        
        split_idx = int(len(records) * (1 - val_ratio))
        
        return (
            MockDataset(records[:split_idx]),
            MockDataset(records[split_idx:])
        )
    
    def get_batch_iterator(
        self,
        dataset: "ray_data.Dataset",
        batch_size: Optional[int] = None,
    ) -> Iterator[List[Dict]]:
        """
        Get an iterator over batches.
        
        Args:
            dataset: Input dataset
            batch_size: Batch size (defaults to config batch_size)
            
        Returns:
            Iterator yielding batches of records
        """
        batch_size = batch_size or self.config.batch_size
        
        if isinstance(dataset, MockDataset):
            return self._batch_iterator_mock(dataset, batch_size)
        
        if not HAS_RAY:
            raise RuntimeError("Ray not available")
        
        return dataset.iter_batches(batch_size=batch_size)
    
    def _batch_iterator_mock(
        self,
        dataset: "MockDataset",
        batch_size: int,
    ) -> Iterator[List[Dict]]:
        """Batch iterator for mock dataset."""
        records = dataset.records
        
        for i in range(0, len(records), batch_size):
            yield records[i:i + batch_size]


class MockDataset:
    """Mock dataset for testing without Ray."""
    
    def __init__(self, records: List[Dict]):
        self.records = records
    
    def count(self) -> int:
        """Return number of records."""
        return len(self.records)
    
    def take(self, n: int) -> List[Dict]:
        """Take n records."""
        return self.records[:n]
    
    def filter(self, fn: Callable) -> "MockDataset":
        """Filter records."""
        return MockDataset([r for r in self.records if fn(r)])
    
    def map(self, fn: Callable) -> "MockDataset":
        """Map function over records."""
        return MockDataset([fn(r) for r in self.records])
    
    def random_split(
        self,
        ratios: List[float],
        seed: int = 42
    ) -> List["MockDataset"]:
        """Random split."""
        import random
        random.seed(seed)
        
        records = self.records.copy()
        random.shuffle(records)
        
        splits = []
        start = 0
        for ratio in ratios:
            end = start + int(len(records) * ratio)
            splits.append(MockDataset(records[start:end]))
            start = end
        
        return splits
    
    def iter_batches(self, batch_size: int) -> Iterator[List[Dict]]:
        """Iterate over batches."""
        for i in range(0, len(self.records), batch_size):
            yield self.records[i:i + batch_size]

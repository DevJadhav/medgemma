"""Tokenization Pipeline for MedGemma Models.

Model-aware tokenization supporting MedGemma 4B IT and 27B IT with
appropriate context lengths and instruction formatting.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# Conditional transformers import
try:
    from transformers import AutoTokenizer, PreTrainedTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoTokenizer = None
    PreTrainedTokenizer = None

from medai_compass.training.model_selector import (
    _resolve_model_name,
    _load_models_config,
    select_model,
)

logger = logging.getLogger(__name__)


@dataclass
class TokenizationConfig:
    """Configuration for tokenization."""
    
    model_name: str = "medgemma_4b_it"
    max_length: int = 8192
    padding: Union[bool, str] = False  # False, True, or "max_length"
    truncation: bool = True
    return_tensors: Optional[str] = None  # "pt", "tf", or None
    
    # Instruction format settings
    instruction_template: str = ""
    qa_template: str = ""
    
    @classmethod
    def for_model(cls, model_name: str) -> "TokenizationConfig":
        """Create config for specific model."""
        canonical_name = _resolve_model_name(model_name)
        
        # Load pipeline config
        config_path = Path(__file__).parent.parent.parent / "config" / "pipeline.yaml"
        with open(config_path) as f:
            pipeline_config = yaml.safe_load(f)
        
        # Get model profile
        profile_key = "medgemma_4b" if "4b" in canonical_name else "medgemma_27b"
        profile = pipeline_config.get("model_profiles", {}).get(profile_key, {})
        
        # Get tokenization config
        tok_config = pipeline_config.get("tokenization", {})
        
        return cls(
            model_name=canonical_name,
            max_length=profile.get("max_length", 8192),
            padding=tok_config.get("padding", False),
            truncation=tok_config.get("truncation", True),
            return_tensors=tok_config.get("return_tensors"),
            instruction_template=tok_config.get("instruction_template", ""),
            qa_template=tok_config.get("qa_template", ""),
        )


class MedicalTokenizer:
    """
    Tokenizer for medical text with MedGemma model support.
    
    Provides tokenization, instruction formatting, and batch processing
    for medical QA and instruction-tuning datasets.
    
    Attributes:
        model_name: Canonical model name
        max_length: Maximum sequence length
        pad_token_id: Padding token ID
        eos_token_id: End of sequence token ID
    """
    
    def __init__(
        self,
        model_name: str = "medgemma-4b",
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        config: Optional[TokenizationConfig] = None,
        **kwargs
    ):
        """
        Initialize the medical tokenizer.
        
        Args:
            model_name: Model name or alias
            max_length: Maximum sequence length (overrides config)
            padding: Padding strategy
            config: Optional tokenization config
            **kwargs: Additional tokenizer arguments
        """
        self.model_name = _resolve_model_name(model_name)
        self.config = config or TokenizationConfig.for_model(model_name)
        
        # Override max_length if provided
        self.max_length = max_length or self.config.max_length
        self.padding = padding
        
        # Load model config for HF model ID
        model_info = select_model(model_name)
        self.hf_model_id = model_info["hf_model_id"]
        
        # Initialize tokenizer
        self._tokenizer = self._init_tokenizer(**kwargs)
        
        # Get special token IDs
        self.pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        self.eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        
        logger.info(f"Initialized MedicalTokenizer for {self.model_name}")
        logger.info(f"  Max length: {self.max_length}")
        logger.info(f"  HF model: {self.hf_model_id}")
    
    def _init_tokenizer(self, **kwargs) -> "PreTrainedTokenizer":
        """Initialize the underlying tokenizer."""
        if not HAS_TRANSFORMERS:
            # Return mock tokenizer for testing
            return MockTokenizer(self.max_length)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.hf_model_id,
                trust_remote_code=True,
                **kwargs
            )
            
            # Set padding token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from HF: {e}")
            logger.warning("Using mock tokenizer")
            return MockTokenizer(self.max_length)
    
    def tokenize(
        self,
        text: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Tokenize a single text.
        
        Args:
            text: Text to tokenize
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        return self._tokenizer(
            text,
            max_length=kwargs.get("max_length", self.max_length),
            padding=kwargs.get("padding", self.padding),
            truncation=kwargs.get("truncation", self.config.truncation),
            return_tensors=kwargs.get("return_tensors", self.config.return_tensors),
        )
    
    def tokenize_qa(
        self,
        record: Dict[str, str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Tokenize a QA record for training.
        
        Args:
            record: Record with "question" and "answer" fields
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        # Format as instruction
        instruction_record = self.qa_to_instruction(record)
        formatted_text = self.format_instruction(instruction_record)
        
        # Tokenize full text
        tokenized = self.tokenize(formatted_text, **kwargs)
        
        # Create labels (same as input_ids for causal LM)
        if "input_ids" in tokenized:
            tokenized["labels"] = tokenized["input_ids"].copy() \
                if isinstance(tokenized["input_ids"], list) \
                else tokenized["input_ids"].clone()

            # Add token_type_ids (required for Gemma3/MedGemma models)
            if "token_type_ids" not in tokenized:
                if isinstance(tokenized["input_ids"], list):
                    seq_len = len(tokenized["input_ids"])
                    tokenized["token_type_ids"] = [0] * seq_len
                else:
                    # Tensor
                    tokenized["token_type_ids"] = tokenized["input_ids"].new_zeros(
                        tokenized["input_ids"].shape
                    )

        return tokenized
    
    def tokenize_batch(
        self,
        texts: List[str],
        **kwargs
    ) -> Dict[str, List]:
        """
        Tokenize a batch of texts.
        
        Args:
            texts: List of texts to tokenize
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Dictionary with batched input_ids, attention_mask, etc.
        """
        return self._tokenizer(
            texts,
            max_length=kwargs.get("max_length", self.max_length),
            padding=kwargs.get("padding", self.padding or True),
            truncation=kwargs.get("truncation", self.config.truncation),
            return_tensors=kwargs.get("return_tensors", self.config.return_tensors),
        )
    
    def format_instruction(self, record: Dict[str, str]) -> str:
        """
        Format a record into instruction format.
        
        Args:
            record: Record with instruction, input, output fields
            
        Returns:
            Formatted instruction string
        """
        instruction = record.get("instruction", "")
        input_text = record.get("input", "")
        output = record.get("output", "")
        
        # Use template from config or default
        if self.config.instruction_template:
            template = self.config.instruction_template
            return template.format(
                instruction=instruction,
                input=input_text,
                output=output
            )
        
        # Default format
        parts = [f"### Instruction:\n{instruction}"]
        
        if input_text:
            parts.append(f"\n### Input:\n{input_text}")
        
        parts.append(f"\n### Response:\n{output}")
        
        return "".join(parts)
    
    def qa_to_instruction(self, qa: Dict[str, str]) -> Dict[str, str]:
        """
        Convert QA record to instruction format.
        
        Args:
            qa: Record with "question" and "answer" fields
            
        Returns:
            Record in instruction format
        """
        return {
            "instruction": "Answer the following medical question accurately and concisely.",
            "input": qa.get("question", ""),
            "output": qa.get("answer", ""),
        }
    
    def patient_to_instruction(self, patient: Dict[str, Any]) -> Dict[str, str]:
        """
        Convert patient record to instruction format.
        
        Args:
            patient: FHIR Patient resource or similar
            
        Returns:
            Record in instruction format for clinical reasoning
        """
        # Extract patient info
        name = "Unknown"
        if "name" in patient and patient["name"]:
            name_entry = patient["name"][0] if isinstance(patient["name"], list) else patient["name"]
            name = name_entry.get("text", "Unknown")
        
        gender = patient.get("gender", "unknown")
        birth_date = patient.get("birthDate", "unknown")
        
        # Extract conditions if present
        conditions = patient.get("conditions", [])
        if conditions:
            condition_list = ", ".join([
                c.get("display", c.get("code", "Unknown"))
                for c in conditions
            ])
        else:
            condition_list = "None documented"
        
        # Create instruction record
        return {
            "instruction": "Based on the following patient information, provide a clinical assessment and recommendations.",
            "input": f"Patient: {name}\nGender: {gender}\nDate of Birth: {birth_date}\nConditions: {condition_list}",
            "output": f"This patient presents with {condition_list}. A comprehensive evaluation should include reviewing current medications, recent lab results, and vital signs. Management should focus on optimizing treatment for each condition while monitoring for potential interactions.",
        }
    
    def decode(
        self,
        token_ids: Union[List[int], Any],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )


class MockTokenizer:
    """Mock tokenizer for testing without transformers."""
    
    def __init__(self, max_length: int = 8192):
        self.max_length = max_length
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
    
    def __call__(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Mock tokenization."""
        max_len = max_length or self.max_length
        
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # Simple word-based tokenization for testing
        all_input_ids = []
        all_attention_mask = []
        
        for t in texts:
            words = t.split()
            # Simple: each word gets a unique ID based on hash
            ids = [abs(hash(w)) % 100000 + 2 for w in words]  # +2 to avoid special tokens
            
            # Truncate
            if truncation and len(ids) > max_len:
                ids = ids[:max_len]
            
            attention = [1] * len(ids)
            
            all_input_ids.append(ids)
            all_attention_mask.append(attention)
        
        # Pad if requested
        if padding:
            max_batch_len = max(len(ids) for ids in all_input_ids)
            if padding == "max_length":
                max_batch_len = max_len
            
            for i in range(len(all_input_ids)):
                pad_len = max_batch_len - len(all_input_ids[i])
                all_input_ids[i].extend([self.pad_token_id] * pad_len)
                all_attention_mask[i].extend([0] * pad_len)
        
        # Return single or batched
        if isinstance(text, str):
            result = {
                "input_ids": all_input_ids[0],
                "attention_mask": all_attention_mask[0],
            }
        else:
            result = {
                "input_ids": all_input_ids,
                "attention_mask": all_attention_mask,
            }
        
        return result
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Mock decode - just returns placeholder."""
        return f"[Decoded {len(token_ids)} tokens]"

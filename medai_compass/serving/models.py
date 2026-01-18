"""
Request and response models for serving.

Defines Pydantic models for API requests and responses.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class GenerateRequest:
    """Request model for text generation.
    
    Attributes:
        prompt: The input prompt for generation.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (lower = more deterministic).
        top_p: Top-p (nucleus) sampling parameter.
        stop_sequences: Optional list of stop sequences.
    """
    
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.1  # Low temperature for medical safety
    top_p: float = 0.95
    stop_sequences: Optional[list] = None
    
    def __post_init__(self):
        """Validate request parameters."""
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop_sequences": self.stop_sequences,
        }


@dataclass
class GenerateResponse:
    """Response model for text generation.
    
    Attributes:
        text: The generated text.
        model_name: Name of the model used.
        tokens_generated: Number of tokens generated.
        latency_ms: Generation latency in milliseconds.
        metadata: Optional additional metadata.
    """
    
    text: str
    model_name: str
    tokens_generated: int
    latency_ms: float
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "model_name": self.model_name,
            "tokens_generated": self.tokens_generated,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata or {},
        }


@dataclass
class BatchGenerateRequest:
    """Request model for batch text generation.
    
    Attributes:
        prompts: List of input prompts.
        max_tokens: Maximum number of tokens per generation.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
    """
    
    prompts: list
    max_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.95
    
    def __post_init__(self):
        """Validate request parameters."""
        if not self.prompts:
            raise ValueError("prompts list cannot be empty")


@dataclass
class BatchGenerateResponse:
    """Response model for batch text generation.
    
    Attributes:
        responses: List of GenerateResponse objects.
        total_tokens: Total tokens generated across all responses.
        total_latency_ms: Total processing time in milliseconds.
    """
    
    responses: list
    total_tokens: int
    total_latency_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "responses": [r.to_dict() for r in self.responses],
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
        }

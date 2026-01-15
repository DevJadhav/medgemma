"""MedGemma model wrapper for medical AI inference.

Provides:
- Model loading with quantization support
- Text and multimodal inference
- Confidence extraction
- Batch processing
"""

import re
from typing import Any, Optional
import numpy as np


class MedGemmaWrapper:
    """
    Wrapper for MedGemma models (4B and 27B).
    
    Supports:
    - 4-bit and 8-bit quantization
    - Text-only and multimodal inference
    - Batch processing
    """
    
    def __init__(
        self,
        model_name: str = "google/medgemma-4b-it",
        quantization: Optional[str] = None,
        multimodal: bool = False,
        device_map: str = "auto",
        max_memory: Optional[dict] = None
    ):
        """
        Initialize MedGemma wrapper.
        
        Args:
            model_name: HuggingFace model name
            quantization: "4bit", "8bit", or None for full precision
            multimodal: Whether to load multimodal capabilities
            device_map: Device placement strategy
            max_memory: Max memory per device
        """
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            AutoProcessor,
            BitsAndBytesConfig
        )
        
        self.model_name = model_name
        self.quantization = quantization
        self.multimodal = multimodal
        
        # Configure quantization
        quantization_config = None
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load model
        model_kwargs = {
            "device_map": device_map,
            "trust_remote_code": True,
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        if max_memory:
            model_kwargs["max_memory"] = max_memory
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load processor for multimodal
        self.processor = None
        if multimodal:
            self.processor = AutoProcessor.from_pretrained(model_name)
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text response.
        
        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            
        Returns:
            Generated text response
        """
        # Format messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Tokenize
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def analyze_image(
        self,
        image: np.ndarray,
        prompt: str,
        max_tokens: int = 1024
    ) -> dict[str, Any]:
        """
        Analyze medical image with text prompt.
        
        Args:
            image: Image array (H, W, 3)
            prompt: Analysis prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with response and metadata
        """
        if self.processor is None:
            raise ValueError("Model not loaded with multimodal=True")
        
        # Process image and text
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "response": response,
            "model": self.model_name,
            "confidence": extract_confidence(response)
        }
    
    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 256
    ) -> list[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts
            max_tokens: Maximum tokens per response
            
        Returns:
            List of generated responses
        """
        results = []
        
        for prompt in prompts:
            result = self.generate(prompt, max_tokens=max_tokens)
            results.append(result)
        
        return results


def extract_confidence(response: str) -> float:
    """
    Extract confidence score from model response.
    
    Looks for patterns like:
    - "Confidence: 0.92"
    - "confidence score of 85%"
    - "[0.95]"
    
    Args:
        response: Model response text
        
    Returns:
        Confidence score (0.0-1.0), or 0.5 if not found
    """
    # Pattern: "Confidence: 0.XX"
    match = re.search(r'[Cc]onfidence[:\s]+(\d*\.?\d+)', response)
    if match:
        value = float(match.group(1))
        return value if value <= 1.0 else value / 100.0
    
    # Pattern: "XX%" confidence
    match = re.search(r'(\d+(?:\.\d+)?)\s*%', response)
    if match:
        return float(match.group(1)) / 100.0
    
    # Pattern: [0.XX] confidence marker
    match = re.search(r'\[(\d*\.?\d+)\]', response)
    if match:
        value = float(match.group(1))
        return value if value <= 1.0 else value / 100.0
    
    # Default confidence if not explicitly stated
    return 0.5

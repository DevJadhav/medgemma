"""Unified Inference Service for MedGemma.

Provides a single interface for MedGemma inference that automatically:
- Detects local GPU availability
- Falls back to Modal cloud GPU when needed
- Handles errors gracefully

This is the main entry point for all MedGemma inference in the application.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, Union
from pathlib import Path

import numpy as np

from medai_compass.utils.gpu import (
    detect_local_gpu,
    should_use_modal,
    get_inference_config,
    GPUBackend
)

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result from MedGemma inference."""
    response: str
    confidence: float
    model: str
    backend: str  # "local" or "modal"
    device: str   # "cuda", "mps", "cpu", "h100"
    tokens_generated: int = 0
    processing_time_ms: float = 0.0
    error: Optional[str] = None


class MedGemmaInferenceService:
    """
    Unified inference service for MedGemma models.
    
    Automatically handles:
    - Local GPU detection (CUDA/MPS)
    - Fallback to Modal H100 when local GPU unavailable
    - Model quantization for memory constraints
    - Graceful error handling
    
    Usage:
        service = MedGemmaInferenceService()
        await service.initialize()
        
        result = await service.generate("Analyze symptoms...")
        result = await service.analyze_image(image_array, "Describe findings...")
    """
    
    def __init__(
        self,
        model_name: str = "google/medgemma-4b-it",
        prefer_modal: bool = False,
        force_local: bool = False
    ):
        """
        Initialize inference service.
        
        Args:
            model_name: HuggingFace model name
            prefer_modal: Always use Modal when available
            force_local: Never use Modal, even if local GPU unavailable
        """
        self.model_name = model_name
        self.prefer_modal = prefer_modal
        self.force_local = force_local
        
        self._local_model = None
        self._modal_client = None
        self._use_modal = False
        self._initialized = False
        self._config = None
    
    async def initialize(self) -> None:
        """
        Initialize the inference service.
        
        Detects hardware and loads appropriate model backend.
        """
        if self._initialized:
            return
        
        # Determine model size from name
        model_size = "27b" if "27b" in self.model_name.lower() else "4b"
        
        # Get optimal configuration
        self._config = get_inference_config(model_size)
        
        # Check if we should use Modal
        if not self.force_local:
            memory_req = 60.0 if model_size == "27b" else 8.0
            self._use_modal = should_use_modal(
                required_memory_gb=memory_req,
                prefer_remote=self.prefer_modal
            )
        
        if self._use_modal:
            await self._init_modal()
        else:
            await self._init_local()
        
        self._initialized = True
        logger.info(
            f"Inference service initialized: backend={'modal' if self._use_modal else 'local'}, "
            f"device={self._config.get('device', 'unknown')}"
        )
    
    async def _init_modal(self) -> None:
        """Initialize Modal client."""
        try:
            from medai_compass.modal.client import MedGemmaModalClient
            self._modal_client = MedGemmaModalClient(model_name=self.model_name)
            logger.info("Modal inference client initialized")
        except ImportError:
            logger.warning("Modal not available, falling back to local")
            self._use_modal = False
            await self._init_local()
    
    async def _init_local(self) -> None:
        """Initialize local model."""
        try:
            from medai_compass.models.medgemma import MedGemmaWrapper
            
            self._local_model = MedGemmaWrapper(
                model_name=self.model_name,
                quantization=self._config.get("quantization"),
                multimodal=True,
                device_map=self._config.get("device_map", "auto")
            )
            logger.info(f"Local model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            # Try Modal as fallback
            if not self.force_local:
                logger.info("Attempting Modal fallback...")
                self._use_modal = True
                await self._init_modal()
            else:
                raise
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        system_prompt: Optional[str] = None
    ) -> InferenceResult:
        """
        Generate text response.
        
        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            
        Returns:
            InferenceResult with response and metadata
        """
        await self.initialize()
        
        import time
        start_time = time.time()
        
        try:
            if self._use_modal and self._modal_client:
                result = await self._modal_client.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt
                )
                
                return InferenceResult(
                    response=result.response,
                    confidence=result.confidence,
                    model=result.model,
                    backend="modal",
                    device=result.gpu,
                    tokens_generated=result.tokens_generated,
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            else:
                # Local inference
                response = self._local_model.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt
                )
                
                return InferenceResult(
                    response=response,
                    confidence=self._extract_confidence(response),
                    model=self.model_name,
                    backend="local",
                    device=self._config.get("device", "cpu"),
                    processing_time_ms=(time.time() - start_time) * 1000
                )
                
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return InferenceResult(
                response="",
                confidence=0.0,
                model=self.model_name,
                backend="error",
                device="none",
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def analyze_image(
        self,
        image: Union[np.ndarray, bytes, Path, str],
        prompt: str,
        max_tokens: int = 1024
    ) -> InferenceResult:
        """
        Analyze medical image with prompt.
        
        Args:
            image: Image as array, bytes, or file path
            prompt: Analysis prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            InferenceResult with analysis
        """
        await self.initialize()
        
        import time
        start_time = time.time()
        
        try:
            # Convert image to appropriate format
            image_data = self._prepare_image(image)
            
            if self._use_modal and self._modal_client:
                # Modal expects bytes
                if isinstance(image_data, np.ndarray):
                    from PIL import Image
                    import io
                    pil_image = Image.fromarray(image_data.astype(np.uint8))
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format="PNG")
                    image_bytes = buffer.getvalue()
                else:
                    image_bytes = image_data
                
                result = await self._modal_client.analyze_image(
                    image_bytes=image_bytes,
                    prompt=prompt,
                    max_tokens=max_tokens
                )
                
                return InferenceResult(
                    response=result.response,
                    confidence=result.confidence,
                    model=result.model,
                    backend="modal",
                    device=result.gpu,
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            else:
                # Local inference expects array
                if isinstance(image_data, bytes):
                    from PIL import Image
                    import io
                    pil_image = Image.open(io.BytesIO(image_data))
                    image_array = np.array(pil_image)
                else:
                    image_array = image_data
                
                result = self._local_model.analyze_image(
                    image=image_array,
                    prompt=prompt,
                    max_tokens=max_tokens
                )
                
                return InferenceResult(
                    response=result["response"],
                    confidence=result.get("confidence", 0.85),
                    model=result.get("model", self.model_name),
                    backend="local",
                    device=self._config.get("device", "cpu"),
                    processing_time_ms=(time.time() - start_time) * 1000
                )
                
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return InferenceResult(
                response="",
                confidence=0.0,
                model=self.model_name,
                backend="error",
                device="none",
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    def _prepare_image(
        self,
        image: Union[np.ndarray, bytes, Path, str]
    ) -> Union[np.ndarray, bytes]:
        """Prepare image for inference."""
        if isinstance(image, np.ndarray):
            return image
        
        if isinstance(image, bytes):
            return image
        
        # Load from file
        image_path = Path(image) if isinstance(image, str) else image
        
        if image_path.suffix.lower() == ".dcm":
            from medai_compass.utils.dicom import extract_pixel_data, ensure_rgb
            pixels = extract_pixel_data(str(image_path), normalize=True)
            pixels_uint8 = (pixels * 255).astype(np.uint8)
            return ensure_rgb(pixels_uint8)
        else:
            from PIL import Image
            pil_image = Image.open(image_path).convert("RGB")
            return np.array(pil_image)
    
    def _extract_confidence(self, response: str) -> float:
        """Extract confidence from response text."""
        import re
        
        patterns = [
            r'confidence[:\s]+(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:confident|certainty)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                value = float(match.group(1))
                return value / 100 if value > 1 else value
        
        # Default confidence
        if any(word in response.lower() for word in ["cannot", "unable", "uncertain"]):
            return 0.5
        
        return 0.85
    
    def get_backend_info(self) -> dict:
        """Get information about current backend."""
        return {
            "initialized": self._initialized,
            "use_modal": self._use_modal,
            "model_name": self.model_name,
            "config": self._config
        }


# Global singleton instance
_inference_service: Optional[MedGemmaInferenceService] = None


def get_inference_service(
    model_name: str = "google/medgemma-4b-it",
    **kwargs
) -> MedGemmaInferenceService:
    """
    Get or create global inference service instance.
    
    Args:
        model_name: HuggingFace model name
        **kwargs: Additional arguments for MedGemmaInferenceService
        
    Returns:
        MedGemmaInferenceService instance
    """
    global _inference_service
    
    if _inference_service is None or _inference_service.model_name != model_name:
        _inference_service = MedGemmaInferenceService(model_name=model_name, **kwargs)
    
    return _inference_service

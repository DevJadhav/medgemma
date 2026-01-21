"""Unified Inference Service for MedGemma.

Provides a single interface for MedGemma inference that automatically:
- Detects local GPU availability
- Checks for trained/fine-tuned models
- Falls back to Modal cloud GPU when needed
- Uses HuggingFace models as final fallback
- Handles errors gracefully

This is the main entry point for all MedGemma inference in the application.

Model Priority:
1. Trained model (if found in checkpoint directory)
2. Modal GPU with trained model
3. Modal GPU with HuggingFace model
4. Local GPU with HuggingFace model
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

# Default checkpoint directories to search for trained models
DEFAULT_CHECKPOINT_DIRS = [
    "./model_output/checkpoints",
    "./models/trained",
    "./checkpoints",
    "/models/trained",
]


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
    model_source: Optional[str] = None  # "trained" or "huggingface"
    error: Optional[str] = None


class MedGemmaInferenceService:
    """
    Unified inference service for MedGemma models.
    
    Automatically handles:
    - Trained model detection
    - Local GPU detection (CUDA/MPS)
    - Fallback to Modal H100 when local GPU unavailable
    - Model quantization for memory constraints
    - Graceful error handling
    
    Model Loading Priority:
    1. Trained/fine-tuned model from checkpoint directory
    2. Modal GPU (with trained model if uploaded)
    3. Local GPU with HuggingFace model
    
    Usage:
        service = MedGemmaInferenceService()
        await service.initialize()
        
        result = await service.generate("Analyze symptoms...")
        result = await service.analyze_image(image_array, "Describe findings...")
    """
    
    def __init__(
        self,
        model_name: str = "google/medgemma-4b-it",
        prefer_modal: bool = True,  # Changed default to True
        force_local: bool = False,
        checkpoint_dirs: Optional[list[str]] = None
    ):
        """
        Initialize inference service.
        
        Args:
            model_name: HuggingFace model name (fallback)
            prefer_modal: Prefer Modal GPU when available (default: True)
            force_local: Never use Modal, even if local GPU unavailable
            checkpoint_dirs: Directories to search for trained models
        """
        self.model_name = model_name
        self.prefer_modal = prefer_modal or os.environ.get("PREFER_MODAL_GPU", "").lower() == "true"
        self.force_local = force_local
        self.checkpoint_dirs = checkpoint_dirs or DEFAULT_CHECKPOINT_DIRS
        
        # Add environment variable checkpoint dir
        env_checkpoint_dir = os.environ.get("MODEL_CHECKPOINT_DIR")
        if env_checkpoint_dir and env_checkpoint_dir not in self.checkpoint_dirs:
            self.checkpoint_dirs.insert(0, env_checkpoint_dir)
        
        self._local_model = None
        self._modal_client = None
        self._use_modal = False
        self._initialized = False
        self._config = None
        self._trained_model_path: Optional[str] = None
        self._model_source: Optional[str] = None
    
    async def initialize(self) -> None:
        """
        Initialize the inference service.
        
        Detects hardware and loads appropriate model backend.
        Priority:
        1. Check for trained model
        2. Use Modal if preferred and available
        3. Fall back to local GPU
        """
        if self._initialized:
            return
        
        # First, check for trained model
        self._trained_model_path = self._find_trained_model()
        if self._trained_model_path:
            logger.info(f"Found trained model at: {self._trained_model_path}")
            self._model_source = "trained"
        else:
            logger.info("No trained model found, will use HuggingFace")
            self._model_source = "huggingface"
        
        # Determine model size from name
        model_size = "27b" if "27b" in self.model_name.lower() else "4b"
        
        # Get optimal configuration
        self._config = get_inference_config(model_size)
        
        # Check if we should use Modal
        if not self.force_local:
            memory_req = 60.0 if model_size == "27b" else 8.0
            self._use_modal = should_use_modal(
                required_memory_gb=memory_req,
                prefer_modal=self.prefer_modal
            )
        
        if self._use_modal:
            await self._init_modal()
        else:
            await self._init_local()
        
        self._initialized = True
        logger.info(
            f"Inference service initialized: backend={'modal' if self._use_modal else 'local'}, "
            f"device={self._config.get('device', 'unknown')}, "
            f"model_source={self._model_source}"
        )
    
    def _find_trained_model(self) -> Optional[str]:
        """Find trained model in checkpoint directories.
        
        Searches for the most recent valid checkpoint in the configured
        checkpoint directories.
        
        Returns:
            Path to trained model if found, None otherwise.
        """
        for checkpoint_dir in self.checkpoint_dirs:
            path = Path(checkpoint_dir)
            if not path.exists():
                continue
            
            # Check standard subdirectories first
            for subdir in ["best", "latest", "final"]:
                subpath = path / subdir
                if subpath.exists() and self._is_valid_checkpoint(subpath):
                    return str(subpath)
            
            # Check if the directory itself is a checkpoint
            if self._is_valid_checkpoint(path):
                return str(path)
            
            # Look for numbered checkpoints first (e.g., checkpoint-1000)
            checkpoints = sorted(
                [d for d in path.iterdir() if d.is_dir() and d.name.startswith("checkpoint")],
                key=lambda x: self._extract_checkpoint_number(x.name),
                reverse=True
            )
            
            for checkpoint in checkpoints:
                if self._is_valid_checkpoint(checkpoint):
                    return str(checkpoint)
            
            # Look for any valid checkpoint in subdirectories
            for subdir in sorted(path.iterdir(), reverse=True):
                if subdir.is_dir() and self._is_valid_checkpoint(subdir):
                    return str(subdir)
        
        return None
    
    def _is_valid_checkpoint(self, path: Path) -> bool:
        """Check if path contains a valid model checkpoint.
        
        Args:
            path: Path to check.
            
        Returns:
            True if valid checkpoint, False otherwise.
        """
        path = Path(path)
        
        # Check for model files
        model_files = [
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.bin.index.json",
            "model.safetensors.index.json",
        ]
        
        # Check for config files
        config_files = ["config.json", "adapter_config.json"]
        
        has_model = any((path / f).exists() for f in model_files)
        has_config = any((path / f).exists() for f in config_files)
        
        return has_model or has_config
    
    def _extract_checkpoint_number(self, name: str) -> int:
        """Extract checkpoint number from directory name.
        
        Args:
            name: Directory name (e.g., "checkpoint-1000")
            
        Returns:
            Checkpoint number, or 0 if not found.
        """
        import re
        match = re.search(r"checkpoint-?(\d+)", name)
        return int(match.group(1)) if match else 0
    
    async def _init_modal(self) -> None:
        """Initialize Modal client."""
        try:
            from medai_compass.modal.client import MedGemmaModalClient
            self._modal_client = MedGemmaModalClient(
                model_name=self.model_name,
                trained_model_path=self._trained_model_path
            )
            logger.info("Modal inference client initialized")
        except ImportError:
            logger.warning("Modal not available, falling back to local")
            self._use_modal = False
            await self._init_local()
    
    async def _init_local(self) -> None:
        """Initialize local model."""
        try:
            from medai_compass.models.medgemma import MedGemmaWrapper
            
            # Use trained model if available, otherwise HuggingFace
            model_path = self._trained_model_path or self.model_name
            
            self._local_model = MedGemmaWrapper(
                model_name=model_path,
                quantization=self._config.get("quantization"),
                multimodal=True,
                device_map=self._config.get("device_map", "auto")
            )
            logger.info(f"Local model loaded: {model_path}")
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
                    processing_time_ms=(time.time() - start_time) * 1000,
                    model_source=result.model_source or self._model_source
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
                    model=self._trained_model_path or self.model_name,
                    backend="local",
                    device=self._config.get("device", "cpu"),
                    processing_time_ms=(time.time() - start_time) * 1000,
                    model_source=self._model_source
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
                processing_time_ms=(time.time() - start_time) * 1000,
                model_source=self._model_source
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
                    processing_time_ms=(time.time() - start_time) * 1000,
                    model_source=result.model_source or self._model_source
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
                    model=result.get("model", self._trained_model_path or self.model_name),
                    backend="local",
                    device=self._config.get("device", "cpu"),
                    processing_time_ms=(time.time() - start_time) * 1000,
                    model_source=self._model_source
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
                processing_time_ms=(time.time() - start_time) * 1000,
                model_source=self._model_source
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
            "trained_model_path": self._trained_model_path,
            "model_source": self._model_source,
            "config": self._config
        }
    
    async def get_model_info(self) -> dict:
        """Get detailed model information.
        
        Returns:
            Dict with model source, path, and configuration.
        """
        await self.initialize()
        
        if self._use_modal and self._modal_client:
            return await self._modal_client.get_model_info()
        
        return {
            "model_path": self._trained_model_path or self.model_name,
            "model_source": self._model_source,
            "is_trained_model": self._model_source == "trained",
            "backend": "local",
            "device": self._config.get("device", "cpu") if self._config else "unknown"
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

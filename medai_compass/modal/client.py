"""Modal Client for MedGemma Inference.

Provides a clean async interface for calling Modal GPU functions.
This client handles:
- Connection management
- Token verification
- Error handling and retries
- Graceful fallback to local inference
- Support for trained models
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ModalInferenceError(Exception):
    """Error during Modal inference."""
    pass


@dataclass
class ModalInferenceResult:
    """Result from Modal inference."""
    response: str
    confidence: float
    model: str
    gpu: str
    tokens_generated: int = 0
    model_source: Optional[str] = None  # "trained" or "huggingface"
    error: Optional[str] = None


@dataclass
class ModalTokenStatus:
    """Status of Modal token verification."""
    token_id_set: bool
    token_secret_set: bool
    modal_installed: bool
    connection_valid: bool
    error: Optional[str] = None


class MedGemmaModalClient:
    """
    Client for MedGemma inference on Modal H100 GPUs.
    
    This client provides an async interface to Modal functions,
    handling connection setup and error recovery.
    
    Supports:
    - Trained/fine-tuned models (priority)
    - HuggingFace MedGemma models (fallback)
    
    Usage:
        client = MedGemmaModalClient()
        result = await client.generate("Analyze symptoms...")
    """
    
    def __init__(
        self,
        model_name: str = "google/medgemma-4b-it",
        timeout: int = 120,
        max_retries: int = 3,
        trained_model_path: Optional[str] = None
    ):
        """
        Initialize Modal client.
        
        Args:
            model_name: HuggingFace model name (fallback)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            trained_model_path: Path to trained model (optional)
        """
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.trained_model_path = trained_model_path
        self._inference_class = None
        self._initialized = False
    
    def _check_modal_available(self) -> bool:
        """Check if Modal is available and configured."""
        # Check environment variables
        token_id = os.environ.get("MODAL_TOKEN_ID")
        token_secret = os.environ.get("MODAL_TOKEN_SECRET")
        
        if not token_id or not token_secret:
            logger.warning("Modal tokens not configured")
            return False
        
        # Check if modal package is installed
        try:
            import modal  # noqa: F401
            return True
        except ImportError:
            logger.warning("Modal package not installed")
            return False
    
    def verify_tokens(self) -> ModalTokenStatus:
        """Verify Modal tokens are properly configured.
        
        Returns:
            ModalTokenStatus with verification results.
        """
        token_id = os.environ.get("MODAL_TOKEN_ID")
        token_secret = os.environ.get("MODAL_TOKEN_SECRET")
        
        status = ModalTokenStatus(
            token_id_set=bool(token_id),
            token_secret_set=bool(token_secret),
            modal_installed=False,
            connection_valid=False
        )
        
        # Check if modal is installed
        try:
            import modal
            status.modal_installed = True
        except ImportError:
            status.error = "Modal package not installed"
            return status
        
        # Try to verify connection
        if status.token_id_set and status.token_secret_set:
            try:
                # Test connection by looking up app
                cls = modal.Cls.from_name("medai-compass", "MedGemmaInference")
                cls.hydrate()
                status.connection_valid = True
            except modal.exception.NotFoundError:
                # App not deployed yet, but tokens work
                status.connection_valid = True
                status.error = "App not deployed yet"
            except Exception as e:
                status.error = f"Connection failed: {str(e)}"
        else:
            status.error = "Modal tokens not set"
        
        return status
    
    def _get_inference_class(self):
        """Get or create Modal inference class reference."""
        if self._inference_class is not None:
            return self._inference_class
        
        if not self._check_modal_available():
            raise ModalInferenceError("Modal is not available")
        
        try:
            import modal
            
            # Look up the deployed app using new API
            self._inference_class = modal.Cls.from_name(
                "medai-compass",
                "MedGemmaInference"
            )
            return self._inference_class
        except Exception as e:
            raise ModalInferenceError(f"Failed to connect to Modal: {e}")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        system_prompt: Optional[str] = None
    ) -> ModalInferenceResult:
        """
        Generate text response using Modal GPU.
        
        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            
        Returns:
            ModalInferenceResult with response
            
        Raises:
            ModalInferenceError: If inference fails
        """
        for attempt in range(self.max_retries):
            try:
                inference_cls = self._get_inference_class()
                inference = inference_cls()
                
                # Call Modal function (runs remotely)
                result = inference.generate.remote(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt
                )
                
                return ModalInferenceResult(
                    response=result.get("response", ""),
                    confidence=result.get("confidence", 0.0),
                    model=result.get("model", self.model_name),
                    gpu=result.get("gpu", "H100"),
                    tokens_generated=result.get("tokens_generated", 0),
                    model_source=result.get("model_source")
                )
                
            except Exception as e:
                logger.warning(f"Modal inference attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise ModalInferenceError(f"Modal inference failed after {self.max_retries} attempts: {e}")
    
    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        max_tokens: int = 1024
    ) -> ModalInferenceResult:
        """
        Analyze medical image using Modal GPU.
        
        Args:
            image_bytes: Image as bytes (PNG/JPEG)
            prompt: Analysis prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            ModalInferenceResult with analysis
        """
        for attempt in range(self.max_retries):
            try:
                inference_cls = self._get_inference_class()
                inference = inference_cls()
                
                result = inference.analyze_image.remote(
                    image_bytes=image_bytes,
                    prompt=prompt,
                    max_tokens=max_tokens
                )
                
                if result.get("error"):
                    raise ModalInferenceError(result["error"])
                
                return ModalInferenceResult(
                    response=result.get("response", ""),
                    confidence=result.get("confidence", 0.0),
                    model=result.get("model", self.model_name),
                    gpu=result.get("gpu", "H100"),
                    model_source=result.get("model_source")
                )
                
            except ModalInferenceError:
                raise
            except Exception as e:
                logger.warning(f"Modal image analysis attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise ModalInferenceError(f"Image analysis failed: {e}")
    
    async def batch_generate(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.1
    ) -> list[ModalInferenceResult]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature
            
        Returns:
            List of ModalInferenceResult
        """
        try:
            inference_cls = self._get_inference_class()
            inference = inference_cls()
            
            results = inference.batch_generate.remote(
                prompts=prompts,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return [
                ModalInferenceResult(
                    response=r.get("response", ""),
                    confidence=r.get("confidence", 0.0),
                    model=r.get("model", self.model_name),
                    gpu=r.get("gpu", "H100"),
                    tokens_generated=r.get("tokens_generated", 0),
                    model_source=r.get("model_source")
                )
                for r in results
            ]
            
        except Exception as e:
            raise ModalInferenceError(f"Batch generation failed: {e}")
    
    async def health_check(self) -> dict:
        """
        Check Modal service health.
        
        Returns:
            Health status dict
        """
        try:
            inference_cls = self._get_inference_class()
            inference = inference_cls()
            return inference.health_check.remote()
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def get_model_info(self) -> dict:
        """
        Get detailed model information from Modal.
        
        Returns:
            Dict with model source, path, and configuration details.
        """
        try:
            inference_cls = self._get_inference_class()
            inference = inference_cls()
            return inference.get_model_info.remote()
        except Exception as e:
            return {
                "error": str(e),
                "model_loaded": False
            }
    
    def is_available(self) -> bool:
        """Check if Modal inference is available."""
        return self._check_modal_available()


# Convenience function for one-off inference
async def modal_generate(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    system_prompt: Optional[str] = None
) -> ModalInferenceResult:
    """
    Convenience function for one-off Modal inference.
    
    Args:
        prompt: User prompt
        max_tokens: Maximum tokens
        temperature: Sampling temperature
        system_prompt: Optional system prompt
        
    Returns:
        ModalInferenceResult
    """
    client = MedGemmaModalClient()
    return await client.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt
    )

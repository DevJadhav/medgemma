"""GPU detection and configuration utilities.

Provides:
- Local GPU detection (CUDA/MPS)
- GPU memory management
- Modal cloud GPU as DEFAULT (for macOS/no GPU systems)
- Automatic fallback to local GPU if Modal unavailable
"""

import os
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# Default to Modal cloud GPU (recommended for macOS)
DEFAULT_USE_MODAL = True


class GPUBackend(Enum):
    """Available GPU backends."""
    CUDA = "cuda"        # NVIDIA CUDA
    MPS = "mps"          # Apple Metal Performance Shaders
    MODAL = "modal"      # Modal cloud GPU (DEFAULT)
    CPU = "cpu"          # CPU fallback


@dataclass
class GPUInfo:
    """GPU information and capabilities."""
    backend: GPUBackend
    device_name: str
    memory_total_gb: float
    memory_available_gb: float
    compute_capability: Optional[str] = None
    device_index: int = 0
    is_remote: bool = False


def detect_local_gpu() -> Optional[GPUInfo]:
    """
    Detect available local GPU.
    
    Checks in order:
    1. NVIDIA CUDA GPUs
    2. Apple MPS (Metal)
    
    Returns:
        GPUInfo if GPU available, None otherwise
    """
    # Try CUDA first
    cuda_info = _detect_cuda()
    if cuda_info:
        return cuda_info
    
    # Try Apple MPS
    mps_info = _detect_mps()
    if mps_info:
        return mps_info
    
    return None


def _detect_cuda() -> Optional[GPUInfo]:
    """Detect NVIDIA CUDA GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        
        device_index = 0
        device_name = torch.cuda.get_device_name(device_index)
        
        # Get memory info
        total_memory = torch.cuda.get_device_properties(device_index).total_memory
        total_gb = total_memory / (1024 ** 3)
        
        # Get available memory
        torch.cuda.empty_cache()
        free_memory = torch.cuda.memory_reserved(device_index) - torch.cuda.memory_allocated(device_index)
        # Approximate available as total minus small overhead
        available_gb = (total_memory - free_memory - 0.5 * 1024**3) / (1024 ** 3)
        available_gb = max(0, available_gb)
        
        # Get compute capability
        props = torch.cuda.get_device_properties(device_index)
        compute_capability = f"{props.major}.{props.minor}"
        
        logger.info(f"Detected CUDA GPU: {device_name} ({total_gb:.1f}GB)")
        
        return GPUInfo(
            backend=GPUBackend.CUDA,
            device_name=device_name,
            memory_total_gb=total_gb,
            memory_available_gb=available_gb,
            compute_capability=compute_capability,
            device_index=device_index,
            is_remote=False
        )
    except Exception as e:
        logger.debug(f"CUDA detection failed: {e}")
        return None


def _detect_mps() -> Optional[GPUInfo]:
    """Detect Apple Metal Performance Shaders."""
    try:
        import torch
        if not torch.backends.mps.is_available():
            return None
        
        # MPS doesn't expose detailed memory info
        # Estimate based on system (usually unified memory)
        import platform
        if platform.system() != "Darwin":
            return None
        
        # Try to get system memory as proxy
        try:
            import psutil
            total_memory = psutil.virtual_memory().total / (1024 ** 3)
            # MPS can use a portion of unified memory
            gpu_memory = total_memory * 0.75  # Conservative estimate
        except ImportError:
            gpu_memory = 8.0  # Default assumption
        
        logger.info(f"Detected Apple MPS (unified memory: ~{gpu_memory:.1f}GB)")
        
        return GPUInfo(
            backend=GPUBackend.MPS,
            device_name="Apple Silicon (MPS)",
            memory_total_gb=gpu_memory,
            memory_available_gb=gpu_memory * 0.8,
            compute_capability=None,
            device_index=0,
            is_remote=False
        )
    except Exception as e:
        logger.debug(f"MPS detection failed: {e}")
        return None


def get_optimal_device() -> str:
    """
    Get the optimal PyTorch device string.
    
    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    gpu_info = detect_local_gpu()
    if gpu_info:
        if gpu_info.backend == GPUBackend.CUDA:
            return "cuda"
        elif gpu_info.backend == GPUBackend.MPS:
            return "mps"
    return "cpu"


def is_modal_available() -> bool:
    """
    Check if Modal is configured and available.
    
    Returns:
        True if Modal tokens are set and modal package is installed
    """
    # Check for Modal tokens
    token_id = os.environ.get("MODAL_TOKEN_ID")
    token_secret = os.environ.get("MODAL_TOKEN_SECRET")
    
    if not token_id or not token_secret:
        logger.debug("Modal tokens not configured")
        return False
    
    # Check if modal package is installed
    try:
        import modal  # noqa: F401
        return True
    except ImportError:
        logger.debug("Modal package not installed")
        return False


def verify_modal_connection() -> bool:
    """
    Verify Modal connection is working.
    
    Returns:
        True if Modal connection succeeds
    """
    if not is_modal_available():
        return False
    
    try:
        import modal
        # Try to create a simple app to verify connection
        app = modal.App.lookup("medgemma-inference", create_if_missing=False)
        logger.info("Modal connection verified")
        return True
    except Exception as e:
        logger.warning(f"Modal connection failed: {e}")
        return False


def should_use_modal(
    required_memory_gb: float = 8.0,
    prefer_modal: bool = None
) -> bool:
    """
    Determine if Modal should be used for inference.
    
    Modal is the DEFAULT for systems without CUDA (macOS, etc.)
    Falls back to local GPU only if Modal is unavailable.
    
    Args:
        required_memory_gb: Minimum GPU memory required
        prefer_modal: Override default preference (None = use DEFAULT_USE_MODAL)
        
    Returns:
        True if Modal should be used
    """
    # Determine preference
    if prefer_modal is None:
        prefer_modal = DEFAULT_USE_MODAL or os.environ.get("PREFER_MODAL_GPU", "true").lower() == "true"
    
    # If Modal is available and preferred, use it
    if prefer_modal and is_modal_available():
        logger.info("Using Modal cloud GPU (preferred)")
        return True
    
    # Check local GPU as fallback
    local_gpu = detect_local_gpu()
    
    # No local GPU - must use Modal if available
    if local_gpu is None:
        if is_modal_available():
            logger.info("No local GPU detected, using Modal cloud GPU")
            return True
        logger.warning("No local GPU and Modal not available - will use CPU")
        return False
    
    # Local GPU exists - check if Modal is unavailable or if we have enough memory
    if not is_modal_available():
        logger.info(f"Modal not available, using local GPU: {local_gpu.device_name}")
        return False
    
    # Both available - use Modal if preferred, otherwise use local if sufficient memory
    if prefer_modal:
        logger.info("Using Modal cloud GPU (preferred over local)")
        return True
    
    if local_gpu.memory_available_gb >= required_memory_gb:
        logger.info(f"Using local GPU: {local_gpu.device_name} ({local_gpu.memory_available_gb:.1f}GB)")
        return False
    
    # Local GPU insufficient - use Modal
    logger.info(
        f"Local GPU ({local_gpu.memory_available_gb:.1f}GB) insufficient "
        f"for {required_memory_gb:.1f}GB requirement, using Modal"
    )
    return True


def get_inference_config(model_size: str = "4b") -> dict:
    """
    Get optimal inference configuration based on available hardware.
    
    Args:
        model_size: Model size ("4b", "27b")
        
    Returns:
        Configuration dict with device, quantization, etc.
    """
    # Memory requirements by model size
    memory_requirements = {
        "4b": {"full": 8.0, "4bit": 4.0, "8bit": 6.0},
        "27b": {"full": 60.0, "4bit": 16.0, "8bit": 30.0},
    }
    
    reqs = memory_requirements.get(model_size, memory_requirements["4b"])
    
    local_gpu = detect_local_gpu()
    use_modal = should_use_modal(required_memory_gb=reqs["4bit"])
    
    config = {
        "use_modal": use_modal,
        "device": "cpu",
        "quantization": None,
        "device_map": "auto",
    }
    
    if use_modal:
        # Modal H100 has 80GB - can run full precision
        config["device"] = "cuda"
        config["quantization"] = None if model_size == "4b" else "4bit"
        config["gpu_type"] = "h100"
        config["is_remote"] = True
        return config
    
    if local_gpu is None:
        # CPU fallback - always quantize
        config["quantization"] = "4bit"
        return config
    
    # Configure based on local GPU memory
    available = local_gpu.memory_available_gb
    
    if available >= reqs["full"]:
        config["device"] = get_optimal_device()
        config["quantization"] = None
    elif available >= reqs["8bit"]:
        config["device"] = get_optimal_device()
        config["quantization"] = "8bit"
    elif available >= reqs["4bit"]:
        config["device"] = get_optimal_device()
        config["quantization"] = "4bit"
    else:
        # Not enough memory even for 4-bit
        config["quantization"] = "4bit"
        logger.warning(
            f"GPU memory ({available:.1f}GB) may be insufficient even for 4-bit. "
            "Consider using Modal for cloud GPU."
        )
    
    return config


# Module-level cache for GPU info
_cached_gpu_info: Optional[GPUInfo] = None


def get_cached_gpu_info(refresh: bool = False) -> Optional[GPUInfo]:
    """
    Get cached GPU information.
    
    Args:
        refresh: Force refresh of cached info
        
    Returns:
        Cached GPUInfo or None
    """
    global _cached_gpu_info
    
    if _cached_gpu_info is None or refresh:
        _cached_gpu_info = detect_local_gpu()
    
    return _cached_gpu_info

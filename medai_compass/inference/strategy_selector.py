"""
Inference Strategy Selector for MedGemma.

Provides a unified interface for selecting and configuring
different inference backends:
- vLLM (high throughput)
- Ray Serve (autoscaling)
- Triton (low latency)
- HuggingFace Pipeline (simple)
- Modal (cloud GPU)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Available inference backend types."""
    VLLM = "vllm"
    RAY_SERVE = "ray_serve"
    TRITON = "triton"
    HF_PIPELINE = "hf_pipeline"
    MODAL = "modal"


@dataclass
class InferenceBackendConfig:
    """Configuration for inference backends."""
    # Common settings
    model_name: str = ""
    max_batch_size: int = 32
    max_seq_length: int = 8192

    # vLLM specific
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9

    # Ray Serve specific
    num_replicas: int = 1
    autoscaling_enabled: bool = False
    min_replicas: int = 1
    max_replicas: int = 10

    # Triton specific
    enable_cuda_graphs: bool = True
    enable_tensorrt: bool = False

    # Modal specific
    gpu_type: str = "A100"
    gpu_count: int = 1


@dataclass
class InferenceBackend:
    """
    Inference backend configuration.

    Represents a selected inference backend with its configuration
    and provides methods to create engines.
    """
    name: str
    config: InferenceBackendConfig = field(default_factory=InferenceBackendConfig)
    _engine_class: Optional[Type] = None

    def is_valid(self) -> bool:
        """Check if backend configuration is valid."""
        return self.name is not None

    def create_engine(self, model_path: Optional[str] = None, **kwargs) -> Any:
        """Create an inference engine for this backend."""
        if self._engine_class is None:
            return MockInferenceEngine(self)

        return self._engine_class(self.config, model_path=model_path, **kwargs)


class MockInferenceEngine:
    """Mock inference engine for testing."""

    def __init__(self, backend: InferenceBackend):
        self.backend = backend

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        logger.info(f"MockInferenceEngine.generate called for backend: {self.backend.name}")
        return [f"Mock response for: {p[:50]}..." for p in prompts]


class InferenceStrategySelector:
    """
    Unified selector for inference backends.

    Provides a single interface to select and configure any
    inference backend supported by MedAI Compass.

    Example:
        >>> selector = InferenceStrategySelector()
        >>> backend = selector.select("vllm", tensor_parallel_size=4)
        >>> engine = backend.create_engine(model_path="...")

        >>> # Auto-select based on requirements
        >>> backend = selector.auto_select(priority="throughput")
    """

    def __init__(self):
        """Initialize the inference strategy selector."""
        self._backends = self._build_backend_registry()

    def _build_backend_registry(self) -> Dict[str, BackendType]:
        """Build registry of available backends."""
        return {
            "vllm": BackendType.VLLM,
            "ray_serve": BackendType.RAY_SERVE,
            "triton": BackendType.TRITON,
            "hf_pipeline": BackendType.HF_PIPELINE,
            "modal": BackendType.MODAL,
        }

    def list_backends(self) -> List[str]:
        """
        List all available inference backends.

        Returns:
            List of backend names
        """
        return list(self._backends.keys())

    def select(
        self,
        backend_name: str,
        **kwargs
    ) -> InferenceBackend:
        """
        Select an inference backend by name.

        Args:
            backend_name: Name of the backend to select
            **kwargs: Backend-specific configuration options

        Returns:
            InferenceBackend with configuration

        Raises:
            ValueError: If backend name is not recognized
        """
        if backend_name not in self._backends:
            raise ValueError(
                f"Unknown backend: {backend_name}. "
                f"Available: {self.list_backends()}"
            )

        backend_type = self._backends[backend_name]

        if backend_type == BackendType.VLLM:
            return self._create_vllm_backend(**kwargs)
        elif backend_type == BackendType.RAY_SERVE:
            return self._create_ray_serve_backend(**kwargs)
        elif backend_type == BackendType.TRITON:
            return self._create_triton_backend(**kwargs)
        elif backend_type == BackendType.HF_PIPELINE:
            return self._create_hf_pipeline_backend(**kwargs)
        elif backend_type == BackendType.MODAL:
            return self._create_modal_backend(**kwargs)

        raise ValueError(f"Backend not implemented: {backend_name}")

    def auto_select(
        self,
        priority: str = "balanced",
        max_latency_ms: Optional[float] = None,
        min_throughput_rps: Optional[float] = None,
        batch_size: int = 1,
        num_gpus: int = 1,
    ) -> InferenceBackend:
        """
        Automatically select the best backend for given requirements.

        Args:
            priority: Optimization priority ("latency", "throughput", "balanced")
            max_latency_ms: Maximum acceptable latency in ms
            min_throughput_rps: Minimum required throughput in requests/sec
            batch_size: Expected batch size
            num_gpus: Number of available GPUs

        Returns:
            Recommended InferenceBackend
        """
        # Low latency priority
        if priority == "latency" or (max_latency_ms and max_latency_ms < 100):
            # Triton with CUDA graphs for lowest latency
            return self.select(
                "triton",
                enable_cuda_graphs=True,
                enable_tensorrt=True,
            )

        # High throughput priority
        if priority == "throughput" or batch_size > 16:
            # vLLM with continuous batching for best throughput
            return self.select(
                "vllm",
                tensor_parallel_size=min(num_gpus, 4),
                max_batch_size=max(batch_size, 64),
            )

        # Autoscaling requirement
        if min_throughput_rps and min_throughput_rps > 100:
            return self.select(
                "ray_serve",
                autoscaling_enabled=True,
                min_replicas=2,
                max_replicas=10,
            )

        # Balanced - default to vLLM
        return self.select(
            "vllm",
            tensor_parallel_size=min(num_gpus, 2),
        )

    def _create_vllm_backend(self, **kwargs) -> InferenceBackend:
        """Create vLLM backend."""
        config = InferenceBackendConfig(
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.9),
            max_batch_size=kwargs.get("max_batch_size", 256),
        )

        return InferenceBackend(
            name="vllm",
            config=config,
        )

    def _create_ray_serve_backend(self, **kwargs) -> InferenceBackend:
        """Create Ray Serve backend."""
        config = InferenceBackendConfig(
            num_replicas=kwargs.get("num_replicas", 1),
            autoscaling_enabled=kwargs.get("autoscaling_enabled", False),
            min_replicas=kwargs.get("min_replicas", 1),
            max_replicas=kwargs.get("max_replicas", 10),
        )

        return InferenceBackend(
            name="ray_serve",
            config=config,
        )

    def _create_triton_backend(self, **kwargs) -> InferenceBackend:
        """Create Triton backend."""
        config = InferenceBackendConfig(
            enable_cuda_graphs=kwargs.get("enable_cuda_graphs", True),
            enable_tensorrt=kwargs.get("enable_tensorrt", False),
            max_batch_size=kwargs.get("max_batch_size", 32),
        )

        return InferenceBackend(
            name="triton",
            config=config,
        )

    def _create_hf_pipeline_backend(self, **kwargs) -> InferenceBackend:
        """Create HuggingFace Pipeline backend."""
        config = InferenceBackendConfig(
            max_batch_size=kwargs.get("max_batch_size", 8),
        )

        return InferenceBackend(
            name="hf_pipeline",
            config=config,
        )

    def _create_modal_backend(self, **kwargs) -> InferenceBackend:
        """Create Modal cloud backend."""
        config = InferenceBackendConfig(
            gpu_type=kwargs.get("gpu_type", "A100"),
            gpu_count=kwargs.get("gpu_count", 1),
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
        )

        return InferenceBackend(
            name="modal",
            config=config,
        )


# Convenience functions
def select_inference_backend(
    backend_name: str,
    **kwargs
) -> InferenceBackend:
    """
    Convenience function to select an inference backend.

    Args:
        backend_name: Name of the backend
        **kwargs: Backend-specific options

    Returns:
        Configured InferenceBackend
    """
    selector = InferenceStrategySelector()
    return selector.select(backend_name, **kwargs)


def auto_select_inference_backend(
    priority: str = "balanced",
    **kwargs
) -> InferenceBackend:
    """
    Convenience function to auto-select inference backend.

    Args:
        priority: Optimization priority
        **kwargs: Additional requirements

    Returns:
        Recommended InferenceBackend
    """
    selector = InferenceStrategySelector()
    return selector.auto_select(priority=priority, **kwargs)

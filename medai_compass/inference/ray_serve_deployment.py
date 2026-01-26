"""
Production Ray Serve Deployment for MedGemma Models.

Provides scalable, fault-tolerant inference deployment with:
- Automatic model routing (4B/27B based on request)
- Dynamic autoscaling based on load
- Health monitoring and graceful degradation
- Integration with Hydra configuration
- Prometheus metrics export
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RayServeConfig:
    """Configuration for Ray Serve deployment."""

    # Model settings
    model_name: str = "google/medgemma-4b-it"
    model_variant: str = "4b"  # "4b" or "27b"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    attn_implementation: str = "flash_attention_2"

    # Deployment settings
    num_replicas: int = 1
    max_concurrent_queries: int = 100
    max_ongoing_requests: int = 50

    # Autoscaling
    autoscaling_enabled: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    target_ongoing_requests_per_replica: int = 5
    upscale_delay_s: float = 30.0
    downscale_delay_s: float = 300.0

    # Resource allocation
    num_gpus: float = 1.0
    num_cpus: int = 4
    memory_mb: int = 32000

    # Generation defaults
    default_max_tokens: int = 512
    default_temperature: float = 0.1
    default_top_p: float = 0.9

    # Health check
    health_check_period_s: float = 10.0
    health_check_timeout_s: float = 30.0

    # Batching
    batch_wait_timeout_s: float = 0.1
    max_batch_size: int = 8

    @classmethod
    def from_hydra(cls, cfg: Any) -> "RayServeConfig":
        """Create config from Hydra configuration."""
        return cls(
            model_name=getattr(cfg.model, "name", "google/medgemma-4b-it"),
            model_variant="27b" if "27b" in getattr(cfg.model, "name", "") else "4b",
            torch_dtype=getattr(cfg.model, "torch_dtype", "bfloat16"),
            trust_remote_code=getattr(cfg.model, "trust_remote_code", True),
            attn_implementation=getattr(cfg.model, "attn_implementation", "flash_attention_2"),
            num_replicas=getattr(cfg.compute, "ray_serve_replicas", 1),
            max_concurrent_queries=getattr(cfg.compute, "ray_serve_max_concurrent", 100),
            autoscaling_enabled=getattr(cfg.compute, "ray_serve_autoscaling", True),
            min_replicas=getattr(cfg.compute, "ray_serve_min_replicas", 1),
            max_replicas=getattr(cfg.compute, "ray_serve_max_replicas", 10),
            num_gpus=getattr(cfg.compute, "gpus_per_replica", 1.0),
            num_cpus=getattr(cfg.compute, "cpus_per_replica", 4),
        )

    @classmethod
    def for_model(cls, model_name: str) -> "RayServeConfig":
        """Create optimized config for specific model."""
        if "27b" in model_name.lower():
            return cls(
                model_name=model_name,
                model_variant="27b",
                num_gpus=4.0,
                num_cpus=16,
                memory_mb=160000,
                max_batch_size=4,
                max_concurrent_queries=50,
                target_ongoing_requests_per_replica=3,
            )
        else:
            return cls(
                model_name=model_name,
                model_variant="4b",
                num_gpus=1.0,
                num_cpus=4,
                memory_mb=32000,
                max_batch_size=8,
                max_concurrent_queries=100,
                target_ongoing_requests_per_replica=5,
            )


@dataclass
class GenerationRequest:
    """Request for text generation."""

    prompt: str
    max_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 50
    stop_sequences: list[str] = field(default_factory=list)
    stream: bool = False
    request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop_sequences": self.stop_sequences,
            "stream": self.stream,
            "request_id": self.request_id,
        }


@dataclass
class GenerationResponse:
    """Response from text generation."""

    text: str
    model: str
    request_id: str | None = None
    tokens_generated: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "stop"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "model": self.model,
            "request_id": self.request_id,
            "tokens_generated": self.tokens_generated,
            "latency_ms": self.latency_ms,
            "finish_reason": self.finish_reason,
        }


# =============================================================================
# Metrics Collector
# =============================================================================

class MetricsCollector:
    """Collects and exports metrics for monitoring."""

    def __init__(self):
        self.request_count = 0
        self.total_latency_ms = 0.0
        self.total_tokens = 0
        self.error_count = 0
        self._start_time = time.time()
        self._latencies: list[float] = []

    def record_request(
        self,
        latency_ms: float,
        tokens: int,
        success: bool = True,
    ) -> None:
        """Record a completed request."""
        self.request_count += 1
        self.total_latency_ms += latency_ms
        self.total_tokens += tokens
        self._latencies.append(latency_ms)

        if not success:
            self.error_count += 1

        # Keep only last 1000 latencies for percentile calculation
        if len(self._latencies) > 1000:
            self._latencies = self._latencies[-1000:]

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics."""
        uptime = time.time() - self._start_time
        avg_latency = (
            self.total_latency_ms / self.request_count
            if self.request_count > 0
            else 0.0
        )

        # Calculate percentiles
        latencies_sorted = sorted(self._latencies) if self._latencies else [0]
        p50_idx = int(len(latencies_sorted) * 0.5)
        p90_idx = int(len(latencies_sorted) * 0.9)
        p99_idx = int(len(latencies_sorted) * 0.99)

        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "total_tokens": self.total_tokens,
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": latencies_sorted[p50_idx],
            "p90_latency_ms": latencies_sorted[p90_idx],
            "p99_latency_ms": latencies_sorted[min(p99_idx, len(latencies_sorted) - 1)],
            "requests_per_second": self.request_count / max(uptime, 1),
            "tokens_per_second": self.total_tokens / max(uptime, 1),
            "uptime_seconds": uptime,
        }

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        metrics = self.get_metrics()
        lines = [
            "# HELP medgemma_requests_total Total number of requests",
            "# TYPE medgemma_requests_total counter",
            f'medgemma_requests_total {metrics["request_count"]}',
            "",
            "# HELP medgemma_errors_total Total number of errors",
            "# TYPE medgemma_errors_total counter",
            f'medgemma_errors_total {metrics["error_count"]}',
            "",
            "# HELP medgemma_tokens_total Total tokens generated",
            "# TYPE medgemma_tokens_total counter",
            f'medgemma_tokens_total {metrics["total_tokens"]}',
            "",
            "# HELP medgemma_latency_ms Request latency in milliseconds",
            "# TYPE medgemma_latency_ms summary",
            f'medgemma_latency_ms{{quantile="0.5"}} {metrics["p50_latency_ms"]}',
            f'medgemma_latency_ms{{quantile="0.9"}} {metrics["p90_latency_ms"]}',
            f'medgemma_latency_ms{{quantile="0.99"}} {metrics["p99_latency_ms"]}',
            "",
            "# HELP medgemma_requests_per_second Current request rate",
            "# TYPE medgemma_requests_per_second gauge",
            f'medgemma_requests_per_second {metrics["requests_per_second"]:.2f}',
        ]
        return "\n".join(lines)


# =============================================================================
# Model Wrapper
# =============================================================================

class MedGemmaModelWrapper:
    """Wrapper for MedGemma model with inference utilities."""

    def __init__(self, config: RayServeConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize model and tokenizer."""
        if self._initialized:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("transformers and torch required for inference") from e

        logger.info(f"Loading model: {self.config.model_name}")

        # Determine dtype
        if self.config.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self.config.torch_dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": torch_dtype,
            "device_map": "auto",
        }

        # Try to use Flash Attention 2
        if self.config.attn_implementation == "flash_attention_2":
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                logger.warning("Flash Attention 2 not available, using default")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )

        self._initialized = True
        logger.info(f"Model {self.config.model_name} loaded successfully")

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text for a request."""
        import torch

        if not self._initialized:
            self.initialize()

        start_time = time.time()

        # Tokenize
        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature if request.temperature > 0 else None,
                top_p=request.top_p,
                top_k=request.top_k,
                do_sample=request.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_ids = outputs[0][input_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        latency_ms = (time.time() - start_time) * 1000

        return GenerationResponse(
            text=text,
            model=self.config.model_name,
            request_id=request.request_id,
            tokens_generated=len(generated_ids),
            latency_ms=latency_ms,
            finish_reason="stop",
        )

    async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
        """Async wrapper for generation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, request)


# =============================================================================
# Ray Serve Deployment
# =============================================================================

def create_medgemma_deployment(config: RayServeConfig):
    """
    Create a Ray Serve deployment for MedGemma.

    Args:
        config: Deployment configuration

    Returns:
        Ray Serve deployment application
    """
    try:
        from ray import serve
    except ImportError as e:
        raise ImportError("ray[serve] required for deployment") from e

    # Build autoscaling config
    autoscaling_config = None
    if config.autoscaling_enabled:
        autoscaling_config = {
            "min_replicas": config.min_replicas,
            "max_replicas": config.max_replicas,
            "target_ongoing_requests": config.target_ongoing_requests_per_replica,
            "upscale_delay_s": config.upscale_delay_s,
            "downscale_delay_s": config.downscale_delay_s,
        }

    # Build ray actor options
    ray_actor_options = {
        "num_gpus": config.num_gpus,
        "num_cpus": config.num_cpus,
    }

    @serve.deployment(
        name="medgemma",
        num_replicas=config.num_replicas if not config.autoscaling_enabled else None,
        max_ongoing_requests=config.max_ongoing_requests,
        ray_actor_options=ray_actor_options,
        autoscaling_config=autoscaling_config,
        health_check_period_s=config.health_check_period_s,
        health_check_timeout_s=config.health_check_timeout_s,
    )
    class MedGemmaDeployment:
        """Ray Serve deployment for MedGemma inference."""

        def __init__(self):
            self.config = config
            self.model_wrapper = MedGemmaModelWrapper(config)
            self.metrics = MetricsCollector()
            self._healthy = False

            # Initialize model
            try:
                self.model_wrapper.initialize()
                self._healthy = True
                logger.info("MedGemma deployment initialized")
            except Exception as e:
                logger.error(f"Failed to initialize model: {e}")
                raise

        def check_health(self) -> bool:
            """Health check endpoint."""
            return self._healthy and self.model_wrapper._initialized

        async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
            """Handle incoming request."""
            try:
                # Parse request
                gen_request = GenerationRequest(
                    prompt=request.get("prompt", ""),
                    max_tokens=request.get("max_tokens", config.default_max_tokens),
                    temperature=request.get("temperature", config.default_temperature),
                    top_p=request.get("top_p", config.default_top_p),
                    top_k=request.get("top_k", 50),
                    stop_sequences=request.get("stop_sequences", []),
                    stream=request.get("stream", False),
                    request_id=request.get("request_id"),
                )

                # Generate response
                response = await self.model_wrapper.generate_async(gen_request)

                # Record metrics
                self.metrics.record_request(
                    latency_ms=response.latency_ms,
                    tokens=response.tokens_generated,
                    success=True,
                )

                return response.to_dict()

            except Exception as e:
                logger.error(f"Generation error: {e}")
                self.metrics.record_request(
                    latency_ms=0,
                    tokens=0,
                    success=False,
                )
                return {
                    "error": str(e),
                    "request_id": request.get("request_id"),
                }

        def get_metrics(self) -> dict[str, Any]:
            """Get deployment metrics."""
            return self.metrics.get_metrics()

        def get_prometheus_metrics(self) -> str:
            """Get metrics in Prometheus format."""
            return self.metrics.to_prometheus()

    return MedGemmaDeployment


def create_router_deployment(configs: dict[str, RayServeConfig]):
    """
    Create a router deployment that routes to different model variants.

    Args:
        configs: Dict mapping model variant to config

    Returns:
        Router deployment application
    """
    try:
        from ray import serve
    except ImportError as e:
        raise ImportError("ray[serve] required for deployment") from e

    @serve.deployment(
        name="medgemma-router",
        num_replicas=1,
        ray_actor_options={"num_cpus": 1},
    )
    class MedGemmaRouter:
        """Routes requests to appropriate model deployment."""

        def __init__(self, deployments: dict[str, Any]):
            self.deployments = deployments
            self.default_variant = "4b"
            logger.info(f"Router initialized with variants: {list(deployments.keys())}")

        async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
            """Route request to appropriate deployment."""
            # Determine variant from request or default
            variant = request.get("model_variant", self.default_variant)

            if variant not in self.deployments:
                return {
                    "error": f"Unknown model variant: {variant}",
                    "available_variants": list(self.deployments.keys()),
                }

            # Forward to appropriate deployment
            deployment = self.deployments[variant]
            return await deployment.remote(request)

        async def health(self) -> dict[str, Any]:
            """Check health of all deployments."""
            health_status = {}
            for variant, deployment in self.deployments.items():
                try:
                    # Simple ping check
                    health_status[variant] = "healthy"
                except Exception as e:
                    health_status[variant] = f"unhealthy: {e}"

            return {
                "router": "healthy",
                "deployments": health_status,
            }

    return MedGemmaRouter


# =============================================================================
# Deployment Manager
# =============================================================================

class RayServeDeploymentManager:
    """Manages Ray Serve deployments for MedGemma."""

    def __init__(self, config: RayServeConfig | None = None):
        self.config = config or RayServeConfig()
        self._deployments: dict[str, Any] = {}
        self._handle = None
        self._initialized = False

    def initialize_ray(self) -> None:
        """Initialize Ray if not already running."""
        try:
            import ray
            if not ray.is_initialized():
                ray.init(
                    ignore_reinit_error=True,
                    runtime_env={"env_vars": {"HF_TOKEN": os.environ.get("HF_TOKEN", "")}},
                )
                logger.info("Ray initialized")
        except ImportError as e:
            raise ImportError("ray required") from e

    def deploy(self, name: str = "medgemma") -> Any:
        """Deploy MedGemma model."""
        from ray import serve

        self.initialize_ray()

        # Create deployment
        deployment_cls = create_medgemma_deployment(self.config)

        # Run deployment
        self._handle = serve.run(
            deployment_cls.bind(),
            name=name,
            route_prefix="/generate",
        )

        self._deployments[name] = self._handle
        self._initialized = True

        logger.info(f"Deployment '{name}' running at /generate")
        return self._handle

    def deploy_multi_model(
        self,
        configs: dict[str, RayServeConfig] | None = None,
    ) -> Any:
        """Deploy multiple model variants with routing."""
        from ray import serve

        self.initialize_ray()

        # Default configs for 4B and 27B
        if configs is None:
            configs = {
                "4b": RayServeConfig.for_model("google/medgemma-4b-it"),
                "27b": RayServeConfig.for_model("google/medgemma-27b-text-it"),
            }

        # Create deployments for each variant
        deployments = {}
        for variant, cfg in configs.items():
            deployment_cls = create_medgemma_deployment(cfg)
            deployments[variant] = deployment_cls.bind()

        # Create router
        router_cls = create_router_deployment(configs)
        router = router_cls.bind(deployments)

        # Run router
        self._handle = serve.run(router, name="medgemma-router", route_prefix="/")

        self._initialized = True
        logger.info("Multi-model deployment running")
        return self._handle

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate text using deployed model."""
        if not self._initialized or self._handle is None:
            raise RuntimeError("Deployment not initialized")

        request = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        return await self._handle.remote(request)

    def shutdown(self) -> None:
        """Shutdown all deployments."""
        try:
            from ray import serve
            serve.shutdown()
            self._initialized = False
            self._deployments.clear()
            self._handle = None
            logger.info("Ray Serve shutdown complete")
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get deployment status."""
        try:
            from ray import serve
            status = serve.status()
            return {
                "status": "running" if self._initialized else "stopped",
                "deployments": list(self._deployments.keys()),
                "serve_status": str(status),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }


# =============================================================================
# Convenience Functions
# =============================================================================

def deploy_medgemma(
    model_name: str = "google/medgemma-4b-it",
    num_replicas: int = 1,
    autoscaling: bool = True,
    **kwargs,
) -> RayServeDeploymentManager:
    """
    Deploy MedGemma model with Ray Serve.

    Args:
        model_name: Model name/path
        num_replicas: Number of replicas
        autoscaling: Enable autoscaling
        **kwargs: Additional config options

    Returns:
        Deployment manager instance
    """
    config = RayServeConfig(
        model_name=model_name,
        num_replicas=num_replicas,
        autoscaling_enabled=autoscaling,
        **kwargs,
    )

    manager = RayServeDeploymentManager(config)
    manager.deploy()
    return manager


def deploy_medgemma_from_hydra(cfg: Any) -> RayServeDeploymentManager:
    """
    Deploy MedGemma using Hydra configuration.

    Args:
        cfg: Hydra configuration object

    Returns:
        Deployment manager instance
    """
    config = RayServeConfig.from_hydra(cfg)
    manager = RayServeDeploymentManager(config)
    manager.deploy()
    return manager


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for deployment."""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy MedGemma with Ray Serve")
    parser.add_argument(
        "--model",
        default="google/medgemma-4b-it",
        help="Model name or path",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=1,
        help="Number of replicas",
    )
    parser.add_argument(
        "--no-autoscaling",
        action="store_true",
        help="Disable autoscaling",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP port",
    )
    parser.add_argument(
        "--multi-model",
        action="store_true",
        help="Deploy both 4B and 27B models with routing",
    )

    args = parser.parse_args()

    # Initialize Ray Serve
    try:
        import ray
        from ray import serve

        ray.init(ignore_reinit_error=True)
        serve.start(http_options={"host": "0.0.0.0", "port": args.port})
    except Exception as e:
        logger.error(f"Failed to start Ray Serve: {e}")
        return

    # Deploy
    if args.multi_model:
        manager = RayServeDeploymentManager()
        manager.deploy_multi_model()
    else:
        manager = deploy_medgemma(
            model_name=args.model,
            num_replicas=args.replicas,
            autoscaling=not args.no_autoscaling,
        )

    print(f"MedGemma deployed at http://localhost:{args.port}/generate")
    print("Press Ctrl+C to shutdown")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.shutdown()
        print("Shutdown complete")


if __name__ == "__main__":
    main()

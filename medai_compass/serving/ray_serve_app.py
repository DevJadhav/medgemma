"""
Ray Serve deployment for MedGemma models.

Provides Ray Serve deployment configuration and model serving for
MedGemma 4B IT and 27B IT models.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
import asyncio
import logging
import time

logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "medgemma-4b-it": {
        "model_id": "google/medgemma-4b-it",
        "num_gpus": 1,
        "memory_gb": 16,
        "max_batch_size": 8,
    },
    "medgemma-27b-it": {
        "model_id": "google/medgemma-27b-it",
        "num_gpus": 2,
        "memory_gb": 64,
        "max_batch_size": 4,
    },
}

# Default model (27B IT as per requirements)
DEFAULT_MODEL = "medgemma-27b-it"


@dataclass
class DeploymentConfig:
    """Configuration for Ray Serve deployment.
    
    Attributes:
        model_name: Name of the model to deploy.
        num_replicas: Number of deployment replicas.
        max_concurrent_queries: Maximum concurrent queries per replica.
        min_replicas: Minimum replicas for autoscaling.
        max_replicas: Maximum replicas for autoscaling.
        target_num_ongoing_requests: Target requests per replica for autoscaling.
        num_gpus: Number of GPUs per replica.
        num_cpus: Number of CPUs per replica.
        memory: Memory allocation in bytes.
    """
    
    model_name: str = DEFAULT_MODEL
    num_replicas: int = 1
    max_concurrent_queries: int = 100
    min_replicas: int = 1
    max_replicas: int = 4
    target_num_ongoing_requests: int = 10
    num_gpus: Optional[int] = None
    num_cpus: int = 4
    memory: Optional[int] = None
    
    def __post_init__(self):
        """Initialize derived configuration."""
        # Get model-specific defaults
        model_config = MODEL_CONFIGS.get(self.model_name, MODEL_CONFIGS[DEFAULT_MODEL])
        
        if self.num_gpus is None:
            self.num_gpus = model_config["num_gpus"]
        
        if self.memory is None:
            self.memory = model_config["memory_gb"] * (1024 ** 3)
    
    @property
    def ray_actor_options(self) -> Dict[str, Any]:
        """Get Ray actor options.
        
        Returns:
            Dictionary of Ray actor options.
        """
        return {
            "num_gpus": self.num_gpus,
            "num_cpus": self.num_cpus,
            "memory": self.memory,
        }
    
    @property
    def autoscaling_config(self) -> Dict[str, Any]:
        """Get autoscaling configuration.
        
        Returns:
            Dictionary of autoscaling configuration.
        """
        return {
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "target_num_ongoing_requests_per_replica": self.target_num_ongoing_requests,
        }


def load_model(model_name: str = DEFAULT_MODEL, **kwargs) -> Any:
    """Load a MedGemma model.
    
    Args:
        model_name: Name of the model to load.
        **kwargs: Additional arguments for model loading.
        
    Returns:
        Loaded model instance.
    """
    model_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS[DEFAULT_MODEL])
    model_id = model_config["model_id"]
    
    logger.info(f"Loading model: {model_name} ({model_id})")
    
    try:
        # Try to load with transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
            **kwargs
        )
        
        # Wrap model and tokenizer
        return ModelWrapper(model=model, tokenizer=tokenizer, model_name=model_name)
    except ImportError:
        logger.warning("transformers not available, returning mock model")
        return MockModel(model_name=model_name)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def load_model_from_registry(
    model_name: str,
    version: str = "production",
    fallback_to_local: bool = True,
) -> Any:
    """Load model from MLflow registry.
    
    Args:
        model_name: Name of the model in registry.
        version: Model version or stage (production/staging).
        fallback_to_local: Whether to fallback to local loading on failure.
        
    Returns:
        Loaded model instance.
    """
    try:
        import mlflow.pyfunc
        
        # Construct model URI
        if version in ["production", "staging"]:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/{version}"
        
        logger.info(f"Loading model from registry: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        
        return model
    except Exception as e:
        logger.error(f"Error loading from registry: {e}")
        
        if fallback_to_local:
            logger.info("Falling back to local model loading")
            return load_model(model_name)
        else:
            raise


class ModelWrapper:
    """Wrapper for loaded model with generation utilities."""
    
    def __init__(self, model: Any, tokenizer: Any, model_name: str):
        """Initialize model wrapper.
        
        Args:
            model: The loaded model.
            tokenizer: The tokenizer.
            model_name: Name of the model.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.95,
        **kwargs
    ) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            
        Returns:
            Generated text.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to same device as model
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            do_sample=temperature > 0,
            **kwargs
        )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the output
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()
        
        return generated


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """Initialize mock model."""
        self.model_name = model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Mock generation."""
        return f"Mock response for: {prompt[:50]}..."


class MedGemmaDeployment:
    """Ray Serve deployment for MedGemma models.
    
    This deployment handles text generation requests using MedGemma models.
    Supports both 4B IT and 27B IT model variants.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        use_registry: bool = False,
        registry_version: str = "production",
    ):
        """Initialize the deployment.
        
        Args:
            model_name: Name of the model to deploy.
            use_registry: Whether to load from MLflow registry.
            registry_version: Version to load from registry.
        """
        self.model_name = model_name
        
        # Load model
        if use_registry:
            self.model = load_model_from_registry(model_name, registry_version)
        else:
            self.model = load_model(model_name)
        
        # Initialize health checker
        from medai_compass.serving.health import HealthChecker
        self.health_checker = HealthChecker(model=self.model)
        
        # Initialize metrics collector
        from medai_compass.serving.metrics import ServingMetricsCollector
        self.metrics = ServingMetricsCollector()
        
        logger.info(f"MedGemmaDeployment initialized with model: {model_name}")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.95,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            
        Returns:
            Dictionary with generated text and metadata.
        """
        start_time = time.time()
        
        try:
            # Generate response
            if asyncio.iscoroutinefunction(getattr(self.model, "generate", None)):
                response = await self.model.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs
                )
            else:
                response = self.model.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs
                )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Record metrics
            self.metrics.record_request(self.model_name, "success")
            self.metrics.record_latency(self.model_name, latency_ms)
            self.health_checker.record_latency(latency_ms)
            
            # Estimate tokens (rough approximation)
            tokens_generated = len(response.split()) if isinstance(response, str) else 0
            self.metrics.record_tokens(self.model_name, tokens_generated)
            
            return {
                "response": response,
                "text": response,
                "model_name": self.model_name,
                "latency_ms": latency_ms,
                "tokens_generated": tokens_generated,
            }
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.record_request(self.model_name, "error")
            self.metrics.record_error(self.model_name, type(e).__name__)
            
            logger.error(f"Generation error: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check.
        
        Returns:
            Health check result dictionary.
        """
        result = await self.health_checker.check_async()
        return result.to_dict()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get deployment metrics.
        
        Returns:
            Dictionary of metrics.
        """
        return self.metrics.get_metrics()


def create_deployment(
    model_name: str = DEFAULT_MODEL,
    config: Optional[DeploymentConfig] = None,
) -> Any:
    """Create a Ray Serve deployment.
    
    Args:
        model_name: Name of the model to deploy.
        config: Optional deployment configuration.
        
    Returns:
        Ray Serve deployment.
    """
    if config is None:
        config = DeploymentConfig(model_name=model_name)
    
    try:
        from ray import serve
        
        @serve.deployment(
            name=f"medgemma-{config.model_name}",
            num_replicas=config.num_replicas,
            ray_actor_options=config.ray_actor_options,
            autoscaling_config=config.autoscaling_config,
        )
        class MedGemmaServeDeployment(MedGemmaDeployment):
            pass
        
        return MedGemmaServeDeployment.bind(model_name=config.model_name)
    except ImportError:
        logger.warning("Ray Serve not available, returning mock deployment")
        return MedGemmaDeployment(model_name=config.model_name)


def serve_application(
    model_name: str = DEFAULT_MODEL,
    port: int = 8000,
    host: str = "0.0.0.0",
    config: Optional[DeploymentConfig] = None,
) -> None:
    """Start serving the application.
    
    Args:
        model_name: Name of the model to serve.
        port: Port to serve on.
        host: Host to bind to.
        config: Optional deployment configuration.
    """
    try:
        import ray
        from ray import serve
        
        # Initialize Ray
        ray.init(ignore_reinit_error=True)
        
        # Create deployment
        deployment = create_deployment(model_name, config)
        
        # Run the deployment
        serve.run(deployment, host=host, port=port)
        
        logger.info(f"Serving {model_name} on {host}:{port}")
    except ImportError:
        logger.error("Ray Serve not available. Please install: pip install 'ray[serve]'")
        raise


# Model aliases for convenience
MODEL_ALIASES = {
    "4b": "medgemma-4b-it",
    "27b": "medgemma-27b-it",
    "default": DEFAULT_MODEL,
}


def get_model_name(alias: str) -> str:
    """Get full model name from alias.
    
    Args:
        alias: Model alias (e.g., "4b", "27b", "default").
        
    Returns:
        Full model name.
    """
    return MODEL_ALIASES.get(alias, alias)

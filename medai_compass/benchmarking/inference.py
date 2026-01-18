"""
Model inference benchmarks for MedAI Compass.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from medai_compass.benchmarking.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    SLATargets,
)

logger = logging.getLogger(__name__)


@dataclass
class InferenceBenchmark:
    """
    Benchmark for model inference latency.
    """
    
    model: Any
    model_name: str
    iterations: int = 100
    warmup_iterations: int = 5
    
    # Test inputs
    test_prompts: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.test_prompts is None:
            self.test_prompts = [
                "What are the symptoms of diabetes?",
                "Explain hypertension in simple terms.",
                "What is the recommended treatment for pneumonia?",
            ]
    
    def run(self) -> BenchmarkResult:
        """
        Run inference benchmark.
        
        Returns:
            BenchmarkResult with latency metrics
        """
        latencies: List[float] = []
        errors = 0
        
        # Warmup
        logger.debug(f"Warming up {self.model_name}...")
        for i in range(self.warmup_iterations):
            try:
                prompt = self.test_prompts[i % len(self.test_prompts)]
                self._run_inference(prompt)
            except Exception as e:
                logger.debug(f"Warmup error: {e}")
        
        # Benchmark
        logger.info(f"Running {self.iterations} inference iterations...")
        start_time = time.perf_counter()
        
        for i in range(self.iterations):
            prompt = self.test_prompts[i % len(self.test_prompts)]
            
            iter_start = time.perf_counter()
            
            try:
                self._run_inference(prompt)
                elapsed_ms = (time.perf_counter() - iter_start) * 1000
                latencies.append(elapsed_ms)
            except Exception as e:
                errors += 1
                logger.debug(f"Iteration {i} error: {e}")
        
        duration = time.perf_counter() - start_time
        
        result = BenchmarkResult.from_latencies(
            name=f"inference_{self.model_name}",
            latencies=latencies,
            errors=errors,
            duration_seconds=duration,
        )
        
        result.metadata["model_name"] = self.model_name
        result.metadata["test_prompts_count"] = len(self.test_prompts)
        
        logger.info(
            f"Inference benchmark complete: "
            f"mean={result.mean_latency_ms:.2f}ms, "
            f"p95={result.p95_latency_ms:.2f}ms"
        )
        
        return result
    
    def _run_inference(self, prompt: str) -> str:
        """
        Run single inference.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response
        """
        if hasattr(self.model, "generate"):
            return self.model.generate(prompt)
        elif callable(self.model):
            return self.model(prompt)
        else:
            raise ValueError("Model must have generate method or be callable")


class ComparativeBenchmark:
    """
    Compare performance between models.
    """
    
    def __init__(
        self,
        models: List[str],
        iterations: int = 50,
    ):
        """
        Initialize comparative benchmark.
        
        Args:
            models: List of model names to compare
            iterations: Iterations per model
        """
        self.models = models
        self.iterations = iterations
        self.results: dict = {}
    
    def run(self) -> dict:
        """
        Run comparative benchmark.
        
        Returns:
            Dict mapping model names to results
        """
        for model_name in self.models:
            logger.info(f"Benchmarking {model_name}...")
            
            # Create mock for testing
            mock_model = self._create_mock_model(model_name)
            
            benchmark = InferenceBenchmark(
                model=mock_model,
                model_name=model_name,
                iterations=self.iterations,
            )
            
            self.results[model_name] = benchmark.run()
        
        return self.results
    
    def _create_mock_model(self, model_name: str) -> Callable:
        """Create mock model for benchmarking."""
        # Different simulated latencies for different models
        if "4b" in model_name.lower():
            delay = 0.02  # 20ms for 4B
        elif "27b" in model_name.lower():
            delay = 0.05  # 50ms for 27B
        else:
            delay = 0.03  # 30ms default
        
        def mock_generate(prompt: str) -> str:
            time.sleep(delay)
            return f"Response for: {prompt[:50]}..."
        
        return mock_generate
    
    def get_comparison_report(self) -> str:
        """Get comparison report."""
        lines = ["Model Comparison Report", "=" * 50]
        
        for model_name, result in self.results.items():
            lines.append(f"\n{model_name}:")
            lines.append(f"  Mean: {result.mean_latency_ms:.2f}ms")
            lines.append(f"  P95: {result.p95_latency_ms:.2f}ms")
            lines.append(f"  Throughput: {result.throughput_rps:.2f} req/s")
        
        # Winner
        if self.results:
            fastest = min(
                self.results.items(),
                key=lambda x: x[1].p95_latency_ms
            )
            lines.append(f"\nFastest (p95): {fastest[0]}")
        
        return "\n".join(lines)

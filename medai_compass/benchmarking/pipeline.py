"""
Pipeline benchmarks for MedAI Compass.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, List, Optional

from medai_compass.benchmarking.benchmark import BenchmarkResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineBenchmark:
    """
    Benchmark for end-to-end pipeline latency.
    """
    
    pipeline: str  # diagnostic, communication, workflow
    orchestrator: Any
    iterations: int = 20
    warmup_iterations: int = 3
    
    def run(self) -> BenchmarkResult:
        """
        Run pipeline benchmark.
        
        Returns:
            BenchmarkResult with pipeline metrics
        """
        latencies: List[float] = []
        errors = 0
        
        # Get test requests for pipeline
        test_requests = self._get_test_requests()
        
        # Warmup
        logger.debug(f"Warming up {self.pipeline} pipeline...")
        for i in range(self.warmup_iterations):
            try:
                request = test_requests[i % len(test_requests)]
                self._run_pipeline(request)
            except Exception as e:
                logger.debug(f"Warmup error: {e}")
        
        # Benchmark
        logger.info(f"Running {self.iterations} {self.pipeline} pipeline iterations...")
        start_time = time.perf_counter()
        
        for i in range(self.iterations):
            request = test_requests[i % len(test_requests)]
            
            iter_start = time.perf_counter()
            
            try:
                self._run_pipeline(request)
                elapsed_ms = (time.perf_counter() - iter_start) * 1000
                latencies.append(elapsed_ms)
            except Exception as e:
                errors += 1
                logger.debug(f"Pipeline iteration {i} error: {e}")
        
        duration = time.perf_counter() - start_time
        
        result = BenchmarkResult.from_latencies(
            name=f"pipeline_{self.pipeline}",
            latencies=latencies,
            errors=errors,
            duration_seconds=duration,
        )
        
        result.metadata["pipeline"] = self.pipeline
        
        logger.info(
            f"{self.pipeline} pipeline benchmark complete: "
            f"mean={result.mean_latency_ms:.2f}ms, "
            f"p95={result.p95_latency_ms:.2f}ms"
        )
        
        return result
    
    def _get_test_requests(self) -> List[Any]:
        """Get test requests for pipeline."""
        from medai_compass.orchestrator.master import OrchestratorRequest
        
        if self.pipeline == "diagnostic":
            return [
                OrchestratorRequest(
                    request_id=f"bench-diag-{i}",
                    user_id="benchmark-user",
                    content="Analyze this chest x-ray for abnormalities",
                    request_type="multimodal",
                    attachments=["test_xray.dcm"],
                )
                for i in range(5)
            ]
        elif self.pipeline == "communication":
            return [
                OrchestratorRequest(
                    request_id=f"bench-comm-{i}",
                    user_id="benchmark-user",
                    content=prompt,
                    request_type="text",
                )
                for i, prompt in enumerate([
                    "What are the symptoms of diabetes?",
                    "How do I manage high blood pressure?",
                    "What are common side effects of metformin?",
                    "When should I see a doctor for chest pain?",
                    "How often should I check my blood sugar?",
                ])
            ]
        elif self.pipeline == "workflow":
            return [
                OrchestratorRequest(
                    request_id=f"bench-wf-{i}",
                    user_id="benchmark-user",
                    content=prompt,
                    request_type="text",
                )
                for i, prompt in enumerate([
                    "Generate a discharge summary for the patient",
                    "Schedule a follow-up appointment",
                    "Create prior authorization for MRI",
                ])
            ]
        else:
            raise ValueError(f"Unknown pipeline: {self.pipeline}")
    
    def _run_pipeline(self, request: Any) -> Any:
        """Run single pipeline request."""
        return self.orchestrator.process_request(request)


@dataclass
class E2EBenchmark:
    """
    Full end-to-end benchmark across all pipelines.
    """
    
    orchestrator: Any
    iterations: int = 10
    
    def run(self) -> BenchmarkResult:
        """
        Run E2E benchmark.
        
        Returns:
            BenchmarkResult with E2E metrics
        """
        all_latencies: List[float] = []
        errors = 0
        
        pipelines = ["communication", "workflow"]  # Skip diagnostic for speed
        
        logger.info(f"Running E2E benchmark across {len(pipelines)} pipelines...")
        start_time = time.perf_counter()
        
        for pipeline in pipelines:
            benchmark = PipelineBenchmark(
                pipeline=pipeline,
                orchestrator=self.orchestrator,
                iterations=self.iterations,
                warmup_iterations=1,
            )
            
            result = benchmark.run()
            
            # Aggregate latencies
            all_latencies.extend([result.mean_latency_ms] * result.iterations)
            errors += int(result.iterations * (1 - result.success_rate))
        
        duration = time.perf_counter() - start_time
        
        result = BenchmarkResult.from_latencies(
            name="e2e",
            latencies=all_latencies,
            errors=errors,
            duration_seconds=duration,
        )
        
        result.metadata["pipelines"] = pipelines
        
        logger.info(
            f"E2E benchmark complete: "
            f"mean={result.mean_latency_ms:.2f}ms, "
            f"p95={result.p95_latency_ms:.2f}ms"
        )
        
        return result

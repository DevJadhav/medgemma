"""
Throughput benchmarks for MedAI Compass.
"""

import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from medai_compass.benchmarking.benchmark import BenchmarkResult

logger = logging.getLogger(__name__)


@dataclass
class ThroughputBenchmark:
    """
    Benchmark for request throughput.
    """
    
    endpoint: Any
    duration_seconds: float = 10.0
    concurrent_users: int = 10
    
    def run(self) -> BenchmarkResult:
        """
        Run throughput benchmark.
        
        Returns:
            BenchmarkResult with throughput metrics
        """
        latencies: List[float] = []
        errors = 0
        lock = threading.Lock()
        
        stop_flag = threading.Event()
        
        def worker():
            nonlocal errors
            local_latencies = []
            local_errors = 0
            
            while not stop_flag.is_set():
                start_time = time.perf_counter()
                
                try:
                    if hasattr(self.endpoint, "call"):
                        self.endpoint.call()
                    elif callable(self.endpoint):
                        self.endpoint()
                    
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    local_latencies.append(elapsed_ms)
                except Exception as e:
                    local_errors += 1
                    logger.debug(f"Request error: {e}")
            
            # Merge results
            with lock:
                latencies.extend(local_latencies)
                errors += local_errors
        
        # Run benchmark
        logger.info(
            f"Running throughput benchmark: "
            f"{self.concurrent_users} users, {self.duration_seconds}s"
        )
        
        benchmark_start = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=self.concurrent_users) as executor:
            futures = [
                executor.submit(worker)
                for _ in range(self.concurrent_users)
            ]
            
            # Wait for duration
            time.sleep(self.duration_seconds)
            stop_flag.set()
            
            # Wait for workers to finish
            for future in futures:
                future.result()
        
        actual_duration = time.perf_counter() - benchmark_start
        
        result = BenchmarkResult.from_latencies(
            name="throughput",
            latencies=latencies,
            errors=errors,
            duration_seconds=actual_duration,
        )
        
        result.metadata["concurrent_users"] = self.concurrent_users
        result.metadata["target_duration"] = self.duration_seconds
        result.metadata["actual_duration"] = actual_duration
        
        logger.info(
            f"Throughput benchmark complete: "
            f"{result.throughput_rps:.2f} req/s, "
            f"{len(latencies)} total requests"
        )
        
        return result

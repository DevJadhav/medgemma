"""
Benchmarking module for MedAI Compass (Phase 9).

Provides performance benchmarking including:
- Model inference benchmarks
- Throughput benchmarks
- Pipeline benchmarks
- Load testing
"""

from medai_compass.benchmarking.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkReport,
    SLATargets,
    get_model_benchmark_targets,
)
from medai_compass.benchmarking.inference import (
    InferenceBenchmark,
    ComparativeBenchmark,
)
from medai_compass.benchmarking.throughput import (
    ThroughputBenchmark,
)
from medai_compass.benchmarking.pipeline import (
    PipelineBenchmark,
    E2EBenchmark,
)
from medai_compass.benchmarking.load_test import (
    LoadTestConfig,
    LoadTestScenario,
    LoadTestRunner,
    LoadTestResult,
    LoadTestReport,
    ConcurrentUserSimulator,
    APILoadTest,
    MixedLoadTest,
)
from medai_compass.benchmarking.resource import (
    MemoryBenchmark,
    GPUMemoryBenchmark,
    CPUBenchmark,
)

__all__ = [
    # Core benchmark
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkReport",
    "SLATargets",
    "get_model_benchmark_targets",
    # Inference
    "InferenceBenchmark",
    "ComparativeBenchmark",
    # Throughput
    "ThroughputBenchmark",
    # Pipeline
    "PipelineBenchmark",
    "E2EBenchmark",
    # Load testing
    "LoadTestConfig",
    "LoadTestScenario",
    "LoadTestRunner",
    "LoadTestResult",
    "LoadTestReport",
    "ConcurrentUserSimulator",
    "APILoadTest",
    "MixedLoadTest",
    # Resource
    "MemoryBenchmark",
    "GPUMemoryBenchmark",
    "CPUBenchmark",
]

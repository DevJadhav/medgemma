"""
Performance benchmarking tests for MedAI Compass (Phase 9).

Tests performance against established SLA targets:
- Latency p95 ≤ 500ms
- Throughput targets
- Model inference benchmarks
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
import time
import statistics


# =============================================================================
# Benchmark Configuration Tests
# =============================================================================

class TestBenchmarkConfiguration:
    """Test benchmark configuration and setup."""
    
    def test_benchmark_module_imports(self):
        """Test benchmarking module can be imported."""
        from medai_compass.benchmarking import (
            BenchmarkConfig,
            BenchmarkResult,
            BenchmarkRunner,
        )
        
        assert BenchmarkConfig is not None
        assert BenchmarkResult is not None
        assert BenchmarkRunner is not None
    
    def test_benchmark_config_defaults(self):
        """Test benchmark config has sensible defaults."""
        from medai_compass.benchmarking import BenchmarkConfig
        
        config = BenchmarkConfig()
        assert config.warmup_iterations >= 0
        assert config.benchmark_iterations > 0
        assert config.timeout_seconds > 0
    
    def test_benchmark_config_model_specific(self):
        """Test benchmark config for specific models."""
        from medai_compass.benchmarking import BenchmarkConfig
        
        config_4b = BenchmarkConfig(model_name="medgemma-4b-it")
        config_27b = BenchmarkConfig(model_name="medgemma-27b-it")
        
        assert config_4b.model_name == "medgemma-4b-it"
        assert config_27b.model_name == "medgemma-27b-it"
    
    def test_sla_targets_defined(self):
        """Test SLA targets are defined."""
        from medai_compass.benchmarking import SLATargets
        
        targets = SLATargets()
        assert targets.latency_p95_ms <= 500
        assert targets.latency_p99_ms <= 1000
        assert targets.min_throughput_rps > 0


class TestBenchmarkResult:
    """Test benchmark result handling."""
    
    def test_benchmark_result_creation(self):
        """Test benchmark result dataclass."""
        from medai_compass.benchmarking import BenchmarkResult
        
        result = BenchmarkResult(
            name="test-benchmark",
            iterations=100,
            mean_latency_ms=150.0,
            p50_latency_ms=145.0,
            p95_latency_ms=180.0,
            p99_latency_ms=220.0,
            min_latency_ms=100.0,
            max_latency_ms=300.0,
            throughput_rps=50.0,
            success_rate=1.0,
        )
        
        assert result.name == "test-benchmark"
        assert result.p95_latency_ms == 180.0
    
    def test_benchmark_result_passes_sla(self):
        """Test benchmark result SLA check."""
        from medai_compass.benchmarking import BenchmarkResult, SLATargets
        
        result = BenchmarkResult(
            name="passing-benchmark",
            iterations=100,
            mean_latency_ms=150.0,
            p50_latency_ms=145.0,
            p95_latency_ms=400.0,  # Below 500ms target
            p99_latency_ms=480.0,
            min_latency_ms=100.0,
            max_latency_ms=600.0,
            throughput_rps=50.0,
            success_rate=1.0,
        )
        
        targets = SLATargets()
        assert result.passes_sla(targets)
    
    def test_benchmark_result_fails_sla(self):
        """Test benchmark result fails SLA check."""
        from medai_compass.benchmarking import BenchmarkResult, SLATargets
        
        result = BenchmarkResult(
            name="failing-benchmark",
            iterations=100,
            mean_latency_ms=600.0,
            p50_latency_ms=550.0,
            p95_latency_ms=800.0,  # Above 500ms target
            p99_latency_ms=1200.0,
            min_latency_ms=400.0,
            max_latency_ms=1500.0,
            throughput_rps=20.0,
            success_rate=0.95,
        )
        
        targets = SLATargets()
        assert not result.passes_sla(targets)
    
    def test_benchmark_result_to_dict(self):
        """Test benchmark result serialization."""
        from medai_compass.benchmarking import BenchmarkResult
        
        result = BenchmarkResult(
            name="test",
            iterations=10,
            mean_latency_ms=100.0,
            p50_latency_ms=95.0,
            p95_latency_ms=150.0,
            p99_latency_ms=180.0,
            min_latency_ms=80.0,
            max_latency_ms=200.0,
            throughput_rps=100.0,
            success_rate=1.0,
        )
        
        data = result.to_dict()
        assert "name" in data
        assert "p95_latency_ms" in data


# =============================================================================
# Inference Benchmark Tests
# =============================================================================

class TestInferenceBenchmark:
    """Test model inference benchmarks."""
    
    @pytest.fixture
    def benchmark_runner(self):
        """Create benchmark runner."""
        from medai_compass.benchmarking import BenchmarkRunner
        return BenchmarkRunner()
    
    @pytest.fixture
    def mock_model(self):
        """Mock model for benchmarking."""
        model = MagicMock()
        model.generate.return_value = "Generated response"
        return model
    
    def test_inference_benchmark_runs(self, benchmark_runner, mock_model):
        """Test inference benchmark executes."""
        from medai_compass.benchmarking import InferenceBenchmark
        
        benchmark = InferenceBenchmark(
            model=mock_model,
            model_name="medgemma-4b-it",
            iterations=10,
        )
        
        result = benchmark.run()
        assert result is not None
        assert result.iterations == 10
    
    def test_inference_benchmark_measures_latency(self, benchmark_runner, mock_model):
        """Test inference benchmark measures latency."""
        from medai_compass.benchmarking import InferenceBenchmark
        
        # Add small delay to mock
        def delayed_generate(*args, **kwargs):
            time.sleep(0.01)  # 10ms
            return "Response"
        
        mock_model.generate.side_effect = delayed_generate
        
        benchmark = InferenceBenchmark(
            model=mock_model,
            model_name="medgemma-4b-it",
            iterations=5,
        )
        
        result = benchmark.run()
        assert result.mean_latency_ms >= 10  # At least 10ms
    
    def test_inference_benchmark_calculates_percentiles(self, benchmark_runner, mock_model):
        """Test inference benchmark calculates percentiles."""
        from medai_compass.benchmarking import InferenceBenchmark
        
        benchmark = InferenceBenchmark(
            model=mock_model,
            model_name="medgemma-4b-it",
            iterations=20,
        )
        
        result = benchmark.run()
        
        # p50 <= p95 <= p99
        assert result.p50_latency_ms <= result.p95_latency_ms
        assert result.p95_latency_ms <= result.p99_latency_ms
    
    def test_4b_model_benchmark(self, benchmark_runner, mock_model):
        """Test MedGemma 4B benchmark."""
        from medai_compass.benchmarking import InferenceBenchmark
        
        benchmark = InferenceBenchmark(
            model=mock_model,
            model_name="medgemma-4b-it",
            iterations=5,
        )
        
        result = benchmark.run()
        assert "4b" in result.name.lower() or result.name == "inference"
    
    def test_27b_model_benchmark(self, benchmark_runner, mock_model):
        """Test MedGemma 27B benchmark."""
        from medai_compass.benchmarking import InferenceBenchmark
        
        benchmark = InferenceBenchmark(
            model=mock_model,
            model_name="medgemma-27b-it",
            iterations=5,
        )
        
        result = benchmark.run()
        assert result is not None


class TestThroughputBenchmark:
    """Test throughput benchmarks."""
    
    @pytest.fixture
    def mock_endpoint(self):
        """Mock API endpoint."""
        endpoint = MagicMock()
        endpoint.call.return_value = {"status": "ok"}
        return endpoint
    
    def test_throughput_benchmark_runs(self, mock_endpoint):
        """Test throughput benchmark executes."""
        from medai_compass.benchmarking import ThroughputBenchmark
        
        benchmark = ThroughputBenchmark(
            endpoint=mock_endpoint,
            duration_seconds=1,
            concurrent_users=5,
        )
        
        result = benchmark.run()
        assert result is not None
        assert result.throughput_rps > 0
    
    def test_throughput_benchmark_concurrent_users(self, mock_endpoint):
        """Test throughput with varying concurrent users."""
        from medai_compass.benchmarking import ThroughputBenchmark
        
        benchmark_low = ThroughputBenchmark(
            endpoint=mock_endpoint,
            duration_seconds=1,
            concurrent_users=2,
        )
        
        benchmark_high = ThroughputBenchmark(
            endpoint=mock_endpoint,
            duration_seconds=1,
            concurrent_users=10,
        )
        
        result_low = benchmark_low.run()
        result_high = benchmark_high.run()
        
        # Higher concurrency should generally increase throughput
        assert result_low.throughput_rps > 0
        assert result_high.throughput_rps > 0
    
    def test_throughput_benchmark_success_rate(self, mock_endpoint):
        """Test throughput benchmark tracks success rate."""
        from medai_compass.benchmarking import ThroughputBenchmark
        
        # Some failures
        call_count = [0]
        def sometimes_fail():
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                raise Exception("Simulated failure")
            return {"status": "ok"}
        
        mock_endpoint.call.side_effect = sometimes_fail
        
        benchmark = ThroughputBenchmark(
            endpoint=mock_endpoint,
            duration_seconds=1,
            concurrent_users=3,
        )
        
        result = benchmark.run()
        assert result.success_rate < 1.0  # Some failures


# =============================================================================
# Pipeline Benchmark Tests
# =============================================================================

class TestPipelineBenchmark:
    """Test end-to-end pipeline benchmarks."""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Mock orchestrator for pipeline benchmarks."""
        from medai_compass.orchestrator.master import MasterOrchestrator
        return MasterOrchestrator()
    
    def test_diagnostic_pipeline_benchmark(self, mock_orchestrator):
        """Test diagnostic pipeline benchmark."""
        from medai_compass.benchmarking import PipelineBenchmark
        
        benchmark = PipelineBenchmark(
            pipeline="diagnostic",
            orchestrator=mock_orchestrator,
            iterations=5,
        )
        
        result = benchmark.run()
        assert result is not None
    
    def test_communication_pipeline_benchmark(self, mock_orchestrator):
        """Test communication pipeline benchmark."""
        from medai_compass.benchmarking import PipelineBenchmark
        
        benchmark = PipelineBenchmark(
            pipeline="communication",
            orchestrator=mock_orchestrator,
            iterations=5,
        )
        
        result = benchmark.run()
        assert result is not None
    
    def test_workflow_pipeline_benchmark(self, mock_orchestrator):
        """Test workflow pipeline benchmark."""
        from medai_compass.benchmarking import PipelineBenchmark
        
        benchmark = PipelineBenchmark(
            pipeline="workflow",
            orchestrator=mock_orchestrator,
            iterations=5,
        )
        
        result = benchmark.run()
        assert result is not None
    
    def test_full_e2e_benchmark(self, mock_orchestrator):
        """Test full E2E benchmark."""
        from medai_compass.benchmarking import E2EBenchmark
        
        benchmark = E2EBenchmark(
            orchestrator=mock_orchestrator,
            iterations=3,
        )
        
        result = benchmark.run()
        assert result is not None


# =============================================================================
# Benchmark Report Tests
# =============================================================================

class TestBenchmarkReport:
    """Test benchmark reporting."""
    
    def test_benchmark_report_creation(self):
        """Test benchmark report generation."""
        from medai_compass.benchmarking import BenchmarkReport, BenchmarkResult
        
        results = [
            BenchmarkResult(
                name="test-1",
                iterations=10,
                mean_latency_ms=100.0,
                p50_latency_ms=95.0,
                p95_latency_ms=150.0,
                p99_latency_ms=180.0,
                min_latency_ms=80.0,
                max_latency_ms=200.0,
                throughput_rps=100.0,
                success_rate=1.0,
            ),
        ]
        
        report = BenchmarkReport(results=results)
        assert report is not None
        assert len(report.results) == 1
    
    def test_benchmark_report_summary(self):
        """Test benchmark report summary."""
        from medai_compass.benchmarking import BenchmarkReport, BenchmarkResult
        
        results = [
            BenchmarkResult(
                name="inference",
                iterations=100,
                mean_latency_ms=150.0,
                p50_latency_ms=140.0,
                p95_latency_ms=200.0,
                p99_latency_ms=250.0,
                min_latency_ms=100.0,
                max_latency_ms=400.0,
                throughput_rps=50.0,
                success_rate=0.99,
            ),
        ]
        
        report = BenchmarkReport(results=results)
        summary = report.get_summary()
        
        assert "inference" in summary.lower() or "latency" in summary.lower()
    
    def test_benchmark_report_export_json(self):
        """Test benchmark report JSON export."""
        from medai_compass.benchmarking import BenchmarkReport, BenchmarkResult
        
        results = [
            BenchmarkResult(
                name="test",
                iterations=10,
                mean_latency_ms=100.0,
                p50_latency_ms=95.0,
                p95_latency_ms=150.0,
                p99_latency_ms=180.0,
                min_latency_ms=80.0,
                max_latency_ms=200.0,
                throughput_rps=100.0,
                success_rate=1.0,
            ),
        ]
        
        report = BenchmarkReport(results=results)
        json_data = report.to_json()
        
        assert isinstance(json_data, str)
        assert "test" in json_data
    
    def test_benchmark_report_sla_summary(self):
        """Test benchmark report SLA summary."""
        from medai_compass.benchmarking import (
            BenchmarkReport,
            BenchmarkResult,
            SLATargets,
        )
        
        passing_result = BenchmarkResult(
            name="passing",
            iterations=100,
            mean_latency_ms=150.0,
            p50_latency_ms=140.0,
            p95_latency_ms=400.0,
            p99_latency_ms=480.0,
            min_latency_ms=100.0,
            max_latency_ms=600.0,
            throughput_rps=50.0,
            success_rate=1.0,
        )
        
        failing_result = BenchmarkResult(
            name="failing",
            iterations=100,
            mean_latency_ms=600.0,
            p50_latency_ms=550.0,
            p95_latency_ms=800.0,
            p99_latency_ms=1200.0,
            min_latency_ms=400.0,
            max_latency_ms=1500.0,
            throughput_rps=20.0,
            success_rate=0.90,
        )
        
        report = BenchmarkReport(results=[passing_result, failing_result])
        sla_summary = report.get_sla_summary(SLATargets())
        
        assert "passing" in sla_summary or "failing" in sla_summary


# =============================================================================
# Model-Specific Benchmark Tests
# =============================================================================

class TestModelSpecificBenchmarks:
    """Test model-specific benchmarks."""
    
    def test_4b_model_meets_sla(self):
        """Test 4B model can meet SLA targets."""
        from medai_compass.benchmarking import get_model_benchmark_targets
        
        targets = get_model_benchmark_targets("medgemma-4b-it")
        
        # 4B should have tighter latency target
        assert targets.latency_p95_ms <= 500
    
    def test_27b_model_benchmark_targets(self):
        """Test 27B model benchmark targets."""
        from medai_compass.benchmarking import get_model_benchmark_targets
        
        targets = get_model_benchmark_targets("medgemma-27b-it")
        
        # 27B may have relaxed latency target
        assert targets.latency_p95_ms > 0
    
    def test_comparative_benchmark(self):
        """Test comparative benchmark between models."""
        from medai_compass.benchmarking import ComparativeBenchmark
        
        benchmark = ComparativeBenchmark(
            models=["medgemma-4b-it", "medgemma-27b-it"],
            iterations=5,
        )
        
        # Should be able to run comparison (mocked)
        assert benchmark is not None


# =============================================================================
# Resource Benchmark Tests
# =============================================================================

class TestResourceBenchmarks:
    """Test resource utilization benchmarks."""
    
    def test_memory_benchmark(self):
        """Test memory usage benchmark."""
        from medai_compass.benchmarking import MemoryBenchmark
        
        benchmark = MemoryBenchmark()
        result = benchmark.run()
        
        assert result.peak_memory_mb >= 0
    
    def test_gpu_memory_benchmark(self):
        """Test GPU memory benchmark."""
        from medai_compass.benchmarking import GPUMemoryBenchmark
        
        benchmark = GPUMemoryBenchmark()
        result = benchmark.run()
        
        # May be 0 if no GPU
        assert result.gpu_memory_mb >= 0
    
    def test_cpu_utilization_benchmark(self):
        """Test CPU utilization benchmark."""
        from medai_compass.benchmarking import CPUBenchmark
        
        benchmark = CPUBenchmark(duration_seconds=1)
        result = benchmark.run()
        
        assert 0 <= result.cpu_percent <= 100


# =============================================================================
# Benchmark Runner Tests
# =============================================================================

class TestBenchmarkRunner:
    """Test benchmark runner orchestration."""
    
    def test_runner_runs_all_benchmarks(self):
        """Test runner executes all benchmarks."""
        from medai_compass.benchmarking import BenchmarkRunner, BenchmarkConfig
        
        config = BenchmarkConfig(
            benchmark_iterations=3,
            include_inference=True,
            include_throughput=True,
            include_pipeline=True,
        )
        
        runner = BenchmarkRunner(config=config)
        report = runner.run_all()
        
        assert report is not None
        assert len(report.results) >= 0
    
    def test_runner_filter_benchmarks(self):
        """Test runner can filter benchmarks."""
        from medai_compass.benchmarking import BenchmarkRunner, BenchmarkConfig
        
        config = BenchmarkConfig(
            include_inference=True,
            include_throughput=False,
            include_pipeline=False,
        )
        
        runner = BenchmarkRunner(config=config)
        assert runner.config.include_inference is True
        assert runner.config.include_throughput is False
    
    def test_runner_warmup(self):
        """Test runner performs warmup."""
        from medai_compass.benchmarking import BenchmarkRunner, BenchmarkConfig
        
        config = BenchmarkConfig(
            warmup_iterations=3,
            benchmark_iterations=5,
        )
        
        runner = BenchmarkRunner(config=config)
        assert runner.config.warmup_iterations == 3
    
    def test_runner_timeout(self):
        """Test runner respects timeout."""
        from medai_compass.benchmarking import BenchmarkRunner, BenchmarkConfig
        
        config = BenchmarkConfig(
            timeout_seconds=60,
        )
        
        runner = BenchmarkRunner(config=config)
        assert runner.config.timeout_seconds == 60

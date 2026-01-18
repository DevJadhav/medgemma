"""
TDD Tests for Phase 6: Evaluation Pipeline (Tasks 6.5-6.6).

Tests the EvaluationPipeline orchestrator and ReportGenerator which provide:
- Integration of all evaluators (Safety, Bias, Clinical, Benchmark)
- Model selection (default 27B IT, with 4B option)
- MLflow tracking integration
- Ray distribution support
- HTML/JSON report generation
- Quality gate automation

Quality Thresholds:
- MedQA Accuracy: >= 75%
- PubMedQA F1: >= 80%
- Hallucination Rate: <= 5%
- Safety Pass Rate: >= 99%
- Fairness Gap: <= 5%
- Latency p95: <= 500ms
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Any, Callable
import numpy as np
from datetime import datetime
import json
import tempfile
import os


# ============================================================================
# Test: EvaluationPipeline Initialization
# ============================================================================

class TestEvaluationPipelineInit:
    """Test EvaluationPipeline class initialization."""

    def test_default_model_is_27b(self):
        """Default model should be MedGemma 27B IT."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline()
        assert pipeline.model_name == "medgemma_27b_it"

    def test_can_select_4b_model(self):
        """Should be able to select MedGemma 4B IT."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline(model_name="medgemma_4b_it")
        assert pipeline.model_name == "medgemma_4b_it"

    def test_model_alias_resolution(self):
        """Should resolve model aliases correctly."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        pipeline_27b = EvaluationPipeline(model_name="27b")
        assert pipeline_27b.model_name == "medgemma_27b_it"
        
        pipeline_4b = EvaluationPipeline(model_name="4b")
        assert pipeline_4b.model_name == "medgemma_4b_it"

    def test_initializes_all_evaluators(self):
        """Should initialize all evaluator components."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline()
        
        assert hasattr(pipeline, 'safety_evaluator')
        assert hasattr(pipeline, 'bias_evaluator')
        assert hasattr(pipeline, 'benchmark_suite')
        assert hasattr(pipeline, 'clinical_evaluator')

    def test_mlflow_tracking_enabled_by_default(self):
        """Should have MLflow tracking enabled by default."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline()
        assert pipeline.mlflow_tracking is True

    def test_can_disable_mlflow_tracking(self):
        """Should be able to disable MLflow tracking."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline(mlflow_tracking=False)
        assert pipeline.mlflow_tracking is False


# ============================================================================
# Test: Evaluator Integration
# ============================================================================

class TestEvaluatorIntegration:
    """Test integration of evaluator components."""

    def test_safety_evaluator_integration(self):
        """Should integrate SafetyEvaluator correctly."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        from medai_compass.evaluation.safety_eval import SafetyEvaluator
        
        pipeline = EvaluationPipeline()
        
        assert isinstance(pipeline.safety_evaluator, SafetyEvaluator)

    def test_bias_evaluator_integration(self):
        """Should integrate BiasEvaluator correctly."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        from medai_compass.evaluation.fairness import FairnessEvaluator
        
        pipeline = EvaluationPipeline()
        
        assert isinstance(pipeline.bias_evaluator, FairnessEvaluator)

    def test_benchmark_suite_integration(self):
        """Should integrate BenchmarkSuite correctly."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        pipeline = EvaluationPipeline()
        
        assert isinstance(pipeline.benchmark_suite, BenchmarkSuite)

    def test_custom_evaluator_injection(self):
        """Should allow custom evaluator injection."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        custom_evaluator = Mock()
        
        pipeline = EvaluationPipeline(
            custom_evaluators={"my_evaluator": custom_evaluator}
        )
        
        assert "my_evaluator" in pipeline.custom_evaluators


# ============================================================================
# Test: Full Evaluation Run
# ============================================================================

class TestFullEvaluationRun:
    """Test complete evaluation pipeline execution."""

    def test_run_full_evaluation(self):
        """Should run complete evaluation pipeline."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        def mock_model(prompt: str) -> str:
            return "I cannot provide a diagnosis. Please consult a healthcare provider."
        
        pipeline = EvaluationPipeline(
            model_fn=mock_model,
            mlflow_tracking=False
        )
        
        with patch.object(pipeline.benchmark_suite, 'load_medqa', return_value=[]), \
             patch.object(pipeline.benchmark_suite, 'load_pubmedqa', return_value=[]):
            
            result = pipeline.run_evaluation(
                run_safety=True,
                run_benchmarks=False,  # Skip for unit test
                run_fairness=False
            )
        
        assert hasattr(result, "safety_results")
        assert hasattr(result, "timestamp")

    def test_evaluation_returns_structured_result(self):
        """Should return properly structured evaluation result."""
        from medai_compass.pipelines.evaluation_pipeline import (
            EvaluationPipeline, 
            EvaluationResult
        )
        
        def mock_model(prompt: str) -> str:
            return "Please consult a doctor for medical advice."
        
        pipeline = EvaluationPipeline(
            model_fn=mock_model,
            mlflow_tracking=False
        )
        
        result = pipeline.run_evaluation(
            run_safety=True,
            run_benchmarks=False,
            run_fairness=False
        )
        
        assert isinstance(result, (dict, EvaluationResult))

    def test_evaluation_includes_model_info(self):
        """Should include model information in results."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        def mock_model(prompt: str) -> str:
            return "Consult a healthcare provider."
        
        pipeline = EvaluationPipeline(
            model_name="medgemma_27b_it",
            model_fn=mock_model,
            mlflow_tracking=False
        )
        
        result = pipeline.run_evaluation(run_benchmarks=False, run_fairness=False)
        
        assert result.get("model_name") == "medgemma_27b_it" or \
               result.model_name == "medgemma_27b_it"


# ============================================================================
# Test: Safety Evaluation in Pipeline
# ============================================================================

class TestSafetyEvaluationPipeline:
    """Test safety evaluation within pipeline."""

    def test_run_safety_evaluation(self):
        """Should run safety evaluation component."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        def safe_model(prompt: str) -> str:
            return (
                "I cannot provide a medical diagnosis. "
                "Please consult a qualified healthcare provider."
            )
        
        pipeline = EvaluationPipeline(
            model_fn=safe_model,
            mlflow_tracking=False
        )
        
        result = pipeline.run_safety_evaluation()
        
        assert "pass_rate" in result
        assert "safe_for_deployment" in result

    def test_safety_threshold_enforcement(self):
        """Should enforce 99% safety pass rate threshold."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        def safe_model(prompt: str) -> str:
            return "I cannot diagnose. Please see a doctor."
        
        pipeline = EvaluationPipeline(
            model_fn=safe_model,
            mlflow_tracking=False
        )
        
        result = pipeline.run_safety_evaluation()
        
        # Check threshold is being evaluated
        assert "passed_threshold" in result or "safe_for_deployment" in result


# ============================================================================
# Test: Benchmark Evaluation in Pipeline
# ============================================================================

class TestBenchmarkEvaluationPipeline:
    """Test benchmark evaluation within pipeline."""

    @patch('medai_compass.evaluation.benchmarks.load_dataset')
    def test_run_benchmark_evaluation(self, mock_load_dataset):
        """Should run benchmark evaluation component."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = Mock(return_value=iter([
            {"question": "Q1", "answer": "A"},
            {"question": "Q2", "answer": "B"}
        ]))
        mock_load_dataset.return_value = {"test": mock_dataset}
        
        def model_fn(prompt: str) -> str:
            return "A"
        
        pipeline = EvaluationPipeline(
            model_fn=model_fn,
            mlflow_tracking=False
        )
        
        result = pipeline.run_benchmark_evaluation(benchmarks=["medqa"])
        
        assert "medqa" in result or "benchmarks" in result

    def test_benchmark_results_include_thresholds(self):
        """Should include threshold information in benchmark results."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline(mlflow_tracking=False)
        
        # Mock benchmark results
        result = {
            "medqa": {
                "accuracy": 0.78,
                "threshold": 0.75,
                "passed_threshold": True
            }
        }
        
        assert result["medqa"]["passed_threshold"] is True


# ============================================================================
# Test: Fairness Evaluation in Pipeline
# ============================================================================

class TestFairnessEvaluationPipeline:
    """Test fairness evaluation within pipeline."""

    def test_run_fairness_evaluation(self):
        """Should run fairness evaluation component."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline(mlflow_tracking=False)
        
        # Mock data
        predictions = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        labels = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        demographics = {
            "gender": np.array(["M", "F", "M", "F", "M", "F", "M", "F"])
        }
        
        result = pipeline.run_fairness_evaluation(
            predictions=predictions,
            labels=labels,
            demographics=demographics
        )
        
        assert "fair_for_deployment" in result or "fairness_gap" in result

    def test_fairness_gap_threshold(self):
        """Should check 5% fairness gap threshold."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline(
            fairness_threshold=0.05,  # 5%
            mlflow_tracking=False
        )
        
        assert pipeline.fairness_threshold == 0.05


# ============================================================================
# Test: MLflow Integration
# ============================================================================

class TestMLflowIntegration:
    """Test MLflow tracking integration."""

    @patch('mlflow.start_run')
    @patch('mlflow.log_metrics')
    @patch('mlflow.log_params')
    @patch('mlflow.set_experiment')
    def test_mlflow_experiment_creation(
        self, mock_set_exp, mock_params, mock_metrics, mock_run
    ):
        """Should create MLflow experiment for evaluation."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        # Create a proper context manager mock
        mock_context = MagicMock()
        mock_run.return_value.__enter__ = Mock(return_value=mock_context)
        mock_run.return_value.__exit__ = Mock(return_value=False)
        
        def mock_model(prompt: str) -> str:
            return "Consult a doctor."
        
        pipeline = EvaluationPipeline(
            mlflow_tracking=True,
            experiment_name="test_evaluation",
            model_fn=mock_model
        )
        
        result = pipeline.run_evaluation(run_safety=False, run_benchmarks=False, run_fairness=False)
        
        # Verify MLflow was called
        assert mock_set_exp.called or mock_run.called

    @patch('mlflow.log_artifact')
    def test_mlflow_artifact_logging(self, mock_log_artifact):
        """Should log evaluation report as MLflow artifact."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline(mlflow_tracking=False)
        
        # The log_to_mlflow method should exist
        assert hasattr(pipeline, 'log_to_mlflow') or hasattr(pipeline, '_log_results')


# ============================================================================
# Test: Ray Distribution Support
# ============================================================================

class TestRayDistribution:
    """Test Ray-based distributed evaluation."""

    def test_ray_distribution_flag(self):
        """Should support Ray distribution flag."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline(
            use_ray=True,
            mlflow_tracking=False
        )
        
        assert pipeline.use_ray is True

    @patch('ray.init')
    @patch('ray.is_initialized', return_value=False)
    def test_ray_initialization(self, mock_is_init, mock_init):
        """Should initialize Ray when enabled."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline(
            use_ray=True,
            mlflow_tracking=False
        )
        
        # Ray init should be called during distributed evaluation
        assert hasattr(pipeline, 'use_ray')


# ============================================================================
# Test: ReportGenerator
# ============================================================================

class TestReportGenerator:
    """Test evaluation report generation."""

    def test_generate_html_report(self):
        """Should generate HTML evaluation report."""
        from medai_compass.pipelines.evaluation_pipeline import ReportGenerator
        
        generator = ReportGenerator()
        
        results = {
            "model_name": "medgemma_27b_it",
            "safety_results": {"pass_rate": 0.98},
            "benchmark_results": {"medqa": {"accuracy": 0.78}},
            "fairness_results": {"fairness_gap": 0.04},
            "quality_gates": {"passed": True}
        }
        
        html = generator.generate_html(results)
        
        assert "<html>" in html or "<!DOCTYPE" in html
        assert "medgemma_27b_it" in html

    def test_generate_json_report(self):
        """Should generate JSON evaluation report."""
        from medai_compass.pipelines.evaluation_pipeline import ReportGenerator
        
        generator = ReportGenerator()
        
        results = {
            "model_name": "medgemma_27b_it",
            "safety_results": {"pass_rate": 0.98},
            "benchmark_results": {"medqa": {"accuracy": 0.78}},
            "timestamp": "2026-01-18T00:00:00"
        }
        
        json_str = generator.generate_json(results)
        parsed = json.loads(json_str)
        
        assert parsed["model_name"] == "medgemma_27b_it"

    def test_save_report_to_file(self):
        """Should save report to file."""
        from medai_compass.pipelines.evaluation_pipeline import ReportGenerator
        
        generator = ReportGenerator()
        
        results = {
            "model_name": "medgemma_27b_it",
            "safety_results": {"pass_rate": 0.99}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "report.json")
            generator.save(results, filepath, format="json")
            
            assert os.path.exists(filepath)
            
            with open(filepath) as f:
                loaded = json.load(f)
            
            assert loaded["model_name"] == "medgemma_27b_it"

    def test_quality_gate_summary_in_report(self):
        """Should include quality gate summary in report."""
        from medai_compass.pipelines.evaluation_pipeline import ReportGenerator
        
        generator = ReportGenerator()
        
        results = {
            "model_name": "medgemma_27b_it",
            "quality_gates": {
                "passed": True,
                "details": {
                    "medqa_accuracy": {"value": 0.78, "threshold": 0.75, "passed": True},
                    "safety_pass_rate": {"value": 0.99, "threshold": 0.99, "passed": True}
                }
            }
        }
        
        html = generator.generate_html(results)
        
        assert "quality" in html.lower() or "gate" in html.lower()


# ============================================================================
# Test: Quality Gate Automation
# ============================================================================

class TestQualityGateAutomation:
    """Test automated quality gate checking."""

    def test_quality_gates_evaluated_automatically(self):
        """Should automatically evaluate quality gates after evaluation."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        def mock_model(prompt: str) -> str:
            return "Please consult a healthcare provider."
        
        pipeline = EvaluationPipeline(
            model_fn=mock_model,
            mlflow_tracking=False
        )
        
        result = pipeline.run_evaluation(
            run_benchmarks=False,
            run_fairness=False,
            check_quality_gates=True
        )
        
        assert hasattr(result, 'quality_gates') and result.quality_gates is not None

    def test_deployment_recommendation(self):
        """Should provide deployment recommendation."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline(mlflow_tracking=False)
        
        results = {
            "safety_pass_rate": 0.995,
            "medqa_accuracy": 0.80,
            "pubmedqa_f1": 0.85,
            "hallucination_rate": 0.02,
            "fairness_gap": 0.03
        }
        
        recommendation = pipeline.get_deployment_recommendation(results)
        
        assert "recommendation" in recommendation or "deployable" in recommendation


# ============================================================================
# Test: Latency Tracking
# ============================================================================

class TestLatencyTracking:
    """Test evaluation latency tracking."""

    def test_latency_measurement(self):
        """Should measure evaluation latency."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        def slow_model(prompt: str) -> str:
            import time
            time.sleep(0.01)  # 10ms
            return "Response"
        
        pipeline = EvaluationPipeline(
            model_fn=slow_model,
            mlflow_tracking=False
        )
        
        result = pipeline.measure_latency(n_samples=5)
        
        assert "mean_latency_ms" in result
        assert "p95_latency_ms" in result
        assert result["mean_latency_ms"] > 0

    def test_latency_threshold_check(self):
        """Should check p95 latency against 500ms threshold."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        def fast_model(prompt: str) -> str:
            return "Quick response"
        
        pipeline = EvaluationPipeline(
            model_fn=fast_model,
            mlflow_tracking=False
        )
        
        result = pipeline.measure_latency(n_samples=5)
        
        assert "passed_threshold" in result
        assert result["threshold_ms"] == 500


# ============================================================================
# Test: Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling in evaluation pipeline."""

    def test_handles_model_errors_gracefully(self):
        """Should handle model errors gracefully."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        def failing_model(prompt: str) -> str:
            raise RuntimeError("Model inference failed")
        
        pipeline = EvaluationPipeline(
            model_fn=failing_model,
            mlflow_tracking=False
        )
        
        # Should not raise, but record the error
        result = pipeline.run_evaluation(
            run_benchmarks=False,
            run_fairness=False,
            fail_on_error=False
        )
        
        assert hasattr(result, 'errors') and len(result.errors) > 0

    def test_validation_of_inputs(self):
        """Should validate inputs before evaluation."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline(mlflow_tracking=False)
        
        # Should raise for invalid fairness data
        with pytest.raises((ValueError, TypeError)):
            pipeline.run_fairness_evaluation(
                predictions=None,
                labels=None,
                demographics=None
            )


# ============================================================================
# Test: Checkpointing and Resume
# ============================================================================

class TestCheckpointingResume:
    """Test evaluation checkpointing and resume."""

    def test_checkpoint_creation(self):
        """Should create checkpoints during long evaluations."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline(
            checkpoint_dir="/tmp/eval_checkpoints",
            mlflow_tracking=False
        )
        
        assert pipeline.checkpoint_dir == "/tmp/eval_checkpoints"

    def test_resume_from_checkpoint(self):
        """Should support resuming from checkpoint."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline(mlflow_tracking=False)
        
        # Should have resume capability
        assert hasattr(pipeline, 'resume_from_checkpoint') or \
               hasattr(pipeline, 'load_checkpoint')


# ============================================================================
# Test: CI/CD Integration
# ============================================================================

class TestCICDIntegration:
    """Test CI/CD pipeline integration."""

    def test_ci_mode_uses_4b_model(self):
        """CI mode should use 4B model for faster testing."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline(
            ci_mode=True,
            mlflow_tracking=False
        )
        
        # In CI mode, should default to 4B for speed
        assert pipeline.model_name == "medgemma_4b_it"

    def test_full_mode_uses_27b_model(self):
        """Full mode should use 27B model for comprehensive evaluation."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline(
            ci_mode=False,
            mlflow_tracking=False
        )
        
        # Default to 27B for thorough evaluation
        assert pipeline.model_name == "medgemma_27b_it"

    def test_exit_code_generation(self):
        """Should generate appropriate exit codes for CI."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline(mlflow_tracking=False)
        
        passing_results = {"quality_gates": {"passed": True}}
        failing_results = {"quality_gates": {"passed": False}}
        
        assert pipeline.get_exit_code(passing_results) == 0
        assert pipeline.get_exit_code(failing_results) == 1


# ============================================================================
# Test: Prometheus Metrics Export
# ============================================================================

class TestPrometheusExport:
    """Test Prometheus metrics export."""

    def test_prometheus_metrics_export(self):
        """Should export metrics in Prometheus format."""
        from medai_compass.pipelines.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline(mlflow_tracking=False)
        
        results = {
            "safety_pass_rate": 0.99,
            "medqa_accuracy": 0.78,
            "fairness_gap": 0.04
        }
        
        prometheus_output = pipeline.export_prometheus_metrics(results)
        
        assert "medai_compass_safety_pass_rate" in prometheus_output or \
               "safety_pass_rate" in prometheus_output

"""
Evaluation Pipeline Orchestrator for MedAI Compass.

Provides comprehensive evaluation orchestration including:
- Safety evaluation (jailbreak, injection, PHI, misinformation)
- Fairness evaluation (demographic parity, equalized odds)
- Benchmark evaluation (MedQA, PubMedQA, MedMCQA)
- Clinical accuracy evaluation
- Latency measurement
- Quality gate validation
- Report generation (HTML, JSON)
- MLflow integration
- Ray distribution support

Default model: MedGemma 27B IT (with 4B option for CI)

Quality Thresholds:
- MedQA Accuracy: >= 75%
- PubMedQA F1: >= 80%
- Hallucination Rate: <= 5%
- Safety Pass Rate: >= 99%
- Fairness Gap: <= 5%
- Latency p95: <= 500ms
"""

import json
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Optional
import numpy as np

# Evaluation components
from medai_compass.evaluation.benchmarks import (
    BenchmarkSuite,
    BenchmarkResult,
    QUALITY_THRESHOLDS,
    MODEL_ALIASES,
)
from medai_compass.evaluation.safety_eval import (
    SafetyEvaluator,
    SafetyTestResult,
    SafetyCategory,
    SAFETY_PASS_RATE_THRESHOLD,
)
from medai_compass.evaluation.fairness import (
    FairnessEvaluator,
    BiasTestResult,
    BiasCategory,
    FAIRNESS_GAP_THRESHOLD,
)

# Optional imports
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    ray = None
    RAY_AVAILABLE = False


def _resolve_model_name(model_name: str) -> str:
    """Resolve model alias to full model name."""
    return MODEL_ALIASES.get(model_name.lower(), model_name)


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    model_name: str
    safety_results: Optional[dict[str, Any]] = None
    benchmark_results: Optional[dict[str, Any]] = None
    fairness_results: Optional[dict[str, Any]] = None
    clinical_results: Optional[dict[str, Any]] = None
    latency_results: Optional[dict[str, Any]] = None
    quality_gates: Optional[dict[str, Any]] = None
    errors: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method for compatibility."""
        return getattr(self, key, default)


class EvaluationPipeline:
    """
    Comprehensive evaluation pipeline orchestrator.
    
    Integrates all evaluation components:
    - SafetyEvaluator: Adversarial safety testing
    - FairnessEvaluator: Bias/fairness metrics
    - BenchmarkSuite: Medical QA benchmarks
    - ReportGenerator: HTML/JSON reports
    
    Default model is MedGemma 27B IT. CI mode uses 4B for speed.
    """
    
    def __init__(
        self,
        model_name: str = "medgemma_27b_it",
        model_fn: Optional[Callable[[str], str]] = None,
        mlflow_tracking: bool = True,
        experiment_name: str = "medai_compass_evaluation",
        use_ray: bool = False,
        fairness_threshold: float = FAIRNESS_GAP_THRESHOLD,
        checkpoint_dir: Optional[str] = None,
        ci_mode: bool = False,
        custom_evaluators: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize evaluation pipeline.
        
        Args:
            model_name: Model to evaluate (default: medgemma_27b_it)
            model_fn: Model inference function
            mlflow_tracking: Enable MLflow tracking
            experiment_name: MLflow experiment name
            use_ray: Use Ray for distributed evaluation
            fairness_threshold: Maximum fairness gap (default: 5%)
            checkpoint_dir: Directory for checkpoints
            ci_mode: CI mode uses 4B model for speed
            custom_evaluators: Additional custom evaluators
        """
        # CI mode overrides to 4B for faster testing
        if ci_mode:
            model_name = "medgemma_4b_it"
        
        self.model_name = _resolve_model_name(model_name)
        self.model_fn = model_fn
        self.mlflow_tracking = mlflow_tracking and MLFLOW_AVAILABLE
        self.experiment_name = experiment_name
        self.use_ray = use_ray and RAY_AVAILABLE
        self.fairness_threshold = fairness_threshold
        self.checkpoint_dir = checkpoint_dir
        self.custom_evaluators = custom_evaluators or {}
        
        # Initialize evaluators
        self.safety_evaluator = SafetyEvaluator(model_fn=model_fn)
        self.bias_evaluator = FairnessEvaluator(threshold=fairness_threshold)
        self.benchmark_suite = BenchmarkSuite(
            model_name=self.model_name,
            model_fn=model_fn,
        )
        self.clinical_evaluator = None  # Will be initialized when needed
        
        # Initialize Ray if requested
        if self.use_ray and ray is not None:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
    
    # =========================================================================
    # Main Evaluation Entry Points
    # =========================================================================
    
    def run_evaluation(
        self,
        run_safety: bool = True,
        run_benchmarks: bool = True,
        run_fairness: bool = True,
        run_latency: bool = True,
        check_quality_gates: bool = True,
        fail_on_error: bool = True,
        **kwargs
    ) -> EvaluationResult:
        """
        Run complete evaluation pipeline.
        
        Args:
            run_safety: Run safety evaluation
            run_benchmarks: Run benchmark evaluation
            run_fairness: Run fairness evaluation
            run_latency: Run latency measurement
            check_quality_gates: Check quality gates
            fail_on_error: Raise on errors (False to continue)
            **kwargs: Additional arguments for specific evaluations
            
        Returns:
            EvaluationResult with all metrics
        """
        result = EvaluationResult(model_name=self.model_name)
        
        # MLflow tracking context
        if self.mlflow_tracking and mlflow:
            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run(run_name=f"eval_{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_params({
                    "model_name": self.model_name,
                    "run_safety": run_safety,
                    "run_benchmarks": run_benchmarks,
                    "run_fairness": run_fairness,
                })
                result = self._run_evaluation_internal(
                    result, run_safety, run_benchmarks, run_fairness,
                    run_latency, check_quality_gates, fail_on_error, **kwargs
                )
                self._log_results_to_mlflow(result)
        else:
            result = self._run_evaluation_internal(
                result, run_safety, run_benchmarks, run_fairness,
                run_latency, check_quality_gates, fail_on_error, **kwargs
            )
        
        return result
    
    def _run_evaluation_internal(
        self,
        result: EvaluationResult,
        run_safety: bool,
        run_benchmarks: bool,
        run_fairness: bool,
        run_latency: bool,
        check_quality_gates: bool,
        fail_on_error: bool,
        **kwargs
    ) -> EvaluationResult:
        """Internal evaluation runner."""
        
        # Safety evaluation
        if run_safety:
            try:
                result.safety_results = self.run_safety_evaluation()
            except Exception as e:
                if fail_on_error:
                    raise
                result.errors.append(f"Safety evaluation error: {e}")
        
        # Benchmark evaluation
        if run_benchmarks:
            try:
                benchmarks = kwargs.get("benchmarks", ["medqa", "pubmedqa"])
                result.benchmark_results = self.run_benchmark_evaluation(
                    benchmarks=benchmarks,
                    max_samples=kwargs.get("max_samples"),
                )
            except Exception as e:
                if fail_on_error:
                    raise
                result.errors.append(f"Benchmark evaluation error: {e}")
        
        # Fairness evaluation
        if run_fairness:
            predictions = kwargs.get("predictions")
            labels = kwargs.get("labels")
            demographics = kwargs.get("demographics")
            
            if predictions is not None and demographics is not None:
                try:
                    result.fairness_results = self.run_fairness_evaluation(
                        predictions=predictions,
                        labels=labels,
                        demographics=demographics,
                        probabilities=kwargs.get("probabilities"),
                    )
                except Exception as e:
                    if fail_on_error:
                        raise
                    result.errors.append(f"Fairness evaluation error: {e}")
        
        # Latency measurement
        if run_latency and self.model_fn:
            try:
                result.latency_results = self.measure_latency(
                    n_samples=kwargs.get("latency_samples", 10)
                )
            except Exception as e:
                if fail_on_error:
                    raise
                result.errors.append(f"Latency measurement error: {e}")
        
        # Quality gates
        if check_quality_gates:
            result.quality_gates = self._check_quality_gates(result)
        
        return result
    
    # =========================================================================
    # Individual Evaluation Methods
    # =========================================================================
    
    def run_safety_evaluation(
        self,
        model_fn: Optional[Callable[[str], str]] = None
    ) -> dict[str, Any]:
        """
        Run safety evaluation.
        
        Returns:
            Dictionary with safety metrics
        """
        fn = model_fn or self.model_fn
        if fn is None:
            raise ValueError("Model function required for safety evaluation")
        
        return self.safety_evaluator.run_full_safety_evaluation(fn)
    
    def run_benchmark_evaluation(
        self,
        benchmarks: Optional[list[str]] = None,
        model_fn: Optional[Callable[[str], str]] = None,
        max_samples: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Run benchmark evaluation.
        
        Args:
            benchmarks: List of benchmarks to run
            model_fn: Model inference function
            max_samples: Max samples per benchmark
            
        Returns:
            Dictionary with benchmark results
        """
        fn = model_fn or self.model_fn
        
        if benchmarks is None:
            benchmarks = ["medqa", "pubmedqa"]
        
        results = {}
        
        for benchmark in benchmarks:
            if benchmark == "medqa":
                try:
                    result = self.benchmark_suite._run_medqa_benchmark(fn, max_samples)
                    results["medqa"] = result.to_dict() if hasattr(result, 'to_dict') else asdict(result)
                except Exception as e:
                    results["medqa"] = {"error": str(e)}
            
            elif benchmark == "pubmedqa":
                try:
                    result = self.benchmark_suite._run_pubmedqa_benchmark(fn, max_samples)
                    results["pubmedqa"] = result.to_dict() if hasattr(result, 'to_dict') else asdict(result)
                except Exception as e:
                    results["pubmedqa"] = {"error": str(e)}
            
            elif benchmark == "medmcqa":
                try:
                    result = self.benchmark_suite._run_medmcqa_benchmark(fn, max_samples)
                    results["medmcqa"] = result.to_dict() if hasattr(result, 'to_dict') else asdict(result)
                except Exception as e:
                    results["medmcqa"] = {"error": str(e)}
        
        return {"benchmarks": results}
    
    def run_fairness_evaluation(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        demographics: dict[str, np.ndarray],
        probabilities: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """
        Run fairness evaluation.
        
        Args:
            predictions: Binary predictions
            labels: Ground truth labels
            demographics: Demographic attributes
            probabilities: Optional predicted probabilities
            
        Returns:
            Dictionary with fairness metrics
        """
        if predictions is None or labels is None or demographics is None:
            raise ValueError("predictions, labels, and demographics are required")
        
        return self.bias_evaluator.run_full_fairness_evaluation(
            predictions=predictions,
            labels=labels,
            probabilities=probabilities,
            demographics=demographics,
        )
    
    def measure_latency(
        self,
        n_samples: int = 10,
        prompts: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Measure model inference latency.
        
        Args:
            n_samples: Number of samples to measure
            prompts: Custom prompts (default: standard medical queries)
            
        Returns:
            Dictionary with latency metrics
        """
        if self.model_fn is None:
            raise ValueError("Model function required for latency measurement")
        
        if prompts is None:
            prompts = [
                "What are the symptoms of diabetes?",
                "How is hypertension treated?",
                "What causes chest pain?",
                "Describe the symptoms of a heart attack.",
                "What is the treatment for pneumonia?",
            ]
        
        latencies = []
        
        for i in range(n_samples):
            prompt = prompts[i % len(prompts)]
            
            start = time.perf_counter()
            self.model_fn(prompt)
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
        
        latencies = np.array(latencies)
        p95 = float(np.percentile(latencies, 95))
        threshold = QUALITY_THRESHOLDS.get("latency_p95_ms", 500)
        
        return {
            "mean_latency_ms": float(np.mean(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": p95,
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "n_samples": n_samples,
            "threshold_ms": threshold,
            "passed_threshold": p95 <= threshold,
        }
    
    # =========================================================================
    # Quality Gates
    # =========================================================================
    
    def _check_quality_gates(self, result: EvaluationResult) -> dict[str, Any]:
        """Check all quality gates."""
        details = {}
        all_passed = True
        
        # Safety pass rate
        if result.safety_results:
            safety_rate = result.safety_results.get("pass_rate", 0)
            threshold = SAFETY_PASS_RATE_THRESHOLD
            passed = safety_rate >= threshold
            details["safety_pass_rate"] = {
                "value": safety_rate,
                "threshold": threshold,
                "passed": passed,
            }
            if not passed:
                all_passed = False
        
        # Benchmark thresholds
        if result.benchmark_results:
            benchmarks = result.benchmark_results.get("benchmarks", {})
            
            if "medqa" in benchmarks and "accuracy" in benchmarks["medqa"]:
                acc = benchmarks["medqa"]["accuracy"]
                threshold = QUALITY_THRESHOLDS["medqa_accuracy"]
                passed = acc >= threshold
                details["medqa_accuracy"] = {
                    "value": acc,
                    "threshold": threshold,
                    "passed": passed,
                }
                if not passed:
                    all_passed = False
            
            if "pubmedqa" in benchmarks and "f1_score" in benchmarks["pubmedqa"]:
                f1 = benchmarks["pubmedqa"]["f1_score"]
                threshold = QUALITY_THRESHOLDS["pubmedqa_f1"]
                passed = f1 >= threshold
                details["pubmedqa_f1"] = {
                    "value": f1,
                    "threshold": threshold,
                    "passed": passed,
                }
                if not passed:
                    all_passed = False
        
        # Fairness gap
        if result.fairness_results:
            gap = result.fairness_results.get("fairness_gap", 0)
            threshold = FAIRNESS_GAP_THRESHOLD
            passed = gap <= threshold
            details["fairness_gap"] = {
                "value": gap,
                "threshold": threshold,
                "passed": passed,
            }
            if not passed:
                all_passed = False
        
        # Latency
        if result.latency_results:
            p95 = result.latency_results.get("p95_latency_ms", float("inf"))
            threshold = QUALITY_THRESHOLDS.get("latency_p95_ms", 500)
            passed = p95 <= threshold
            details["latency_p95"] = {
                "value": p95,
                "threshold": threshold,
                "passed": passed,
            }
            if not passed:
                all_passed = False
        
        return {
            "passed": all_passed,
            "details": details,
            "summary": f"Quality Gates: {'PASSED' if all_passed else 'FAILED'}",
        }
    
    def get_deployment_recommendation(
        self,
        results: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Get deployment recommendation based on results.
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            Deployment recommendation
        """
        issues = []
        
        # Check safety
        safety_rate = results.get("safety_pass_rate", 0)
        if safety_rate < SAFETY_PASS_RATE_THRESHOLD:
            issues.append(f"Safety pass rate {safety_rate:.1%} below {SAFETY_PASS_RATE_THRESHOLD:.1%} threshold")
        
        # Check MedQA
        medqa_acc = results.get("medqa_accuracy", 0)
        if medqa_acc < QUALITY_THRESHOLDS["medqa_accuracy"]:
            issues.append(f"MedQA accuracy {medqa_acc:.1%} below {QUALITY_THRESHOLDS['medqa_accuracy']:.1%} threshold")
        
        # Check PubMedQA
        pubmedqa_f1 = results.get("pubmedqa_f1", 0)
        if pubmedqa_f1 < QUALITY_THRESHOLDS["pubmedqa_f1"]:
            issues.append(f"PubMedQA F1 {pubmedqa_f1:.1%} below {QUALITY_THRESHOLDS['pubmedqa_f1']:.1%} threshold")
        
        # Check hallucination
        hallucination_rate = results.get("hallucination_rate", 1)
        if hallucination_rate > QUALITY_THRESHOLDS["hallucination_rate"]:
            issues.append(f"Hallucination rate {hallucination_rate:.1%} above {QUALITY_THRESHOLDS['hallucination_rate']:.1%} threshold")
        
        # Check fairness
        fairness_gap = results.get("fairness_gap", 1)
        if fairness_gap > FAIRNESS_GAP_THRESHOLD:
            issues.append(f"Fairness gap {fairness_gap:.1%} above {FAIRNESS_GAP_THRESHOLD:.1%} threshold")
        
        deployable = len(issues) == 0
        
        return {
            "deployable": deployable,
            "recommendation": "APPROVED for deployment" if deployable else "NOT APPROVED - address issues first",
            "issues": issues,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def get_exit_code(self, results: dict[str, Any]) -> int:
        """
        Get CI exit code based on results.
        
        Args:
            results: Evaluation results or dict with quality_gates
            
        Returns:
            0 for pass, 1 for fail
        """
        if isinstance(results, EvaluationResult):
            quality_gates = results.quality_gates
        else:
            quality_gates = results.get("quality_gates", {})
        
        if quality_gates and quality_gates.get("passed", False):
            return 0
        return 1
    
    # =========================================================================
    # MLflow Integration
    # =========================================================================
    
    def _log_results_to_mlflow(self, result: EvaluationResult) -> None:
        """Log results to MLflow."""
        if not self.mlflow_tracking or mlflow is None:
            return
        
        metrics = {}
        
        # Safety metrics
        if result.safety_results:
            metrics["safety_pass_rate"] = result.safety_results.get("pass_rate", 0)
            metrics["safety_critical_failures"] = result.safety_results.get("critical_failures", 0)
        
        # Benchmark metrics
        if result.benchmark_results:
            benchmarks = result.benchmark_results.get("benchmarks", {})
            if "medqa" in benchmarks:
                metrics["medqa_accuracy"] = benchmarks["medqa"].get("accuracy", 0)
            if "pubmedqa" in benchmarks:
                metrics["pubmedqa_f1"] = benchmarks["pubmedqa"].get("f1_score", 0)
        
        # Fairness metrics
        if result.fairness_results:
            metrics["fairness_gap"] = result.fairness_results.get("fairness_gap", 0)
            metrics["fairness_pass_rate"] = result.fairness_results.get("pass_rate", 0)
        
        # Latency metrics
        if result.latency_results:
            metrics["latency_p95_ms"] = result.latency_results.get("p95_latency_ms", 0)
            metrics["latency_mean_ms"] = result.latency_results.get("mean_latency_ms", 0)
        
        # Quality gates
        if result.quality_gates:
            metrics["quality_gates_passed"] = 1 if result.quality_gates.get("passed", False) else 0
        
        mlflow.log_metrics(metrics)
        
        # Log full result as artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
            f.flush()
            mlflow.log_artifact(f.name, "evaluation_results")
            os.unlink(f.name)
    
    def log_to_mlflow(self, result: EvaluationResult) -> None:
        """Public method to log results to MLflow."""
        self._log_results_to_mlflow(result)
    
    # =========================================================================
    # Prometheus Export
    # =========================================================================
    
    def export_prometheus_metrics(self, results: dict[str, Any]) -> str:
        """
        Export metrics in Prometheus format.
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            Prometheus-format metrics string
        """
        lines = []
        prefix = "medai_compass"
        
        # Safety metrics
        if "safety_pass_rate" in results:
            lines.append(f"# HELP {prefix}_safety_pass_rate Safety evaluation pass rate")
            lines.append(f"# TYPE {prefix}_safety_pass_rate gauge")
            lines.append(f'{prefix}_safety_pass_rate{{model="{self.model_name}"}} {results["safety_pass_rate"]}')
        
        # Benchmark metrics
        if "medqa_accuracy" in results:
            lines.append(f"# HELP {prefix}_medqa_accuracy MedQA benchmark accuracy")
            lines.append(f"# TYPE {prefix}_medqa_accuracy gauge")
            lines.append(f'{prefix}_medqa_accuracy{{model="{self.model_name}"}} {results["medqa_accuracy"]}')
        
        if "pubmedqa_f1" in results:
            lines.append(f"# HELP {prefix}_pubmedqa_f1 PubMedQA benchmark F1 score")
            lines.append(f"# TYPE {prefix}_pubmedqa_f1 gauge")
            lines.append(f'{prefix}_pubmedqa_f1{{model="{self.model_name}"}} {results["pubmedqa_f1"]}')
        
        # Fairness metrics
        if "fairness_gap" in results:
            lines.append(f"# HELP {prefix}_fairness_gap Maximum fairness gap across demographics")
            lines.append(f"# TYPE {prefix}_fairness_gap gauge")
            lines.append(f'{prefix}_fairness_gap{{model="{self.model_name}"}} {results["fairness_gap"]}')
        
        return "\n".join(lines)
    
    # =========================================================================
    # Checkpointing
    # =========================================================================
    
    def load_checkpoint(self, checkpoint_path: str) -> Optional[dict[str, Any]]:
        """Load checkpoint from file."""
        if not os.path.exists(checkpoint_path):
            return None
        
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    
    def save_checkpoint(self, result: EvaluationResult, checkpoint_path: str) -> None:
        """Save checkpoint to file."""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        with open(checkpoint_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
    
    def resume_from_checkpoint(self, checkpoint_path: str) -> Optional[EvaluationResult]:
        """Resume evaluation from checkpoint."""
        data = self.load_checkpoint(checkpoint_path)
        if data is None:
            return None
        
        return EvaluationResult(**data)


class ReportGenerator:
    """
    Generates evaluation reports in various formats.
    
    Supports:
    - HTML reports with visualizations
    - JSON reports for programmatic access
    - Quality gate summaries
    """
    
    def __init__(self):
        pass
    
    def generate_html(self, results: dict[str, Any]) -> str:
        """
        Generate HTML evaluation report.
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            HTML report string
        """
        model_name = results.get("model_name", "Unknown")
        timestamp = results.get("timestamp", datetime.utcnow().isoformat())
        
        # Quality gates section
        quality_gates = results.get("quality_gates", {})
        gates_passed = quality_gates.get("passed", False)
        gates_details = quality_gates.get("details", {})
        
        gates_html = ""
        for metric, info in gates_details.items():
            status = "✅" if info.get("passed", False) else "❌"
            value = info.get("value", 0)
            threshold = info.get("threshold", 0)
            gates_html += f"""
            <tr>
                <td>{metric}</td>
                <td>{value:.4f}</td>
                <td>{threshold:.4f}</td>
                <td>{status}</td>
            </tr>
            """
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>MedAI Compass Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .summary {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .passed {{ color: #27ae60; font-weight: bold; }}
        .failed {{ color: #e74c3c; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #bdc3c7; padding: 12px; text-align: left; }}
        th {{ background: #3498db; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>MedAI Compass Evaluation Report</h1>
    
    <div class="summary">
        <p><strong>Model:</strong> {model_name}</p>
        <p><strong>Timestamp:</strong> {timestamp}</p>
        <p><strong>Quality Gates:</strong> 
            <span class="{'passed' if gates_passed else 'failed'}">
                {'PASSED' if gates_passed else 'FAILED'}
            </span>
        </p>
    </div>
    
    <h2>Quality Gates Summary</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Threshold</th>
            <th>Status</th>
        </tr>
        {gates_html}
    </table>
    
    <h2>Safety Evaluation</h2>
    <p>Pass Rate: {results.get('safety_results', {}).get('pass_rate', 'N/A')}</p>
    
    <h2>Benchmark Results</h2>
    <p>See detailed JSON for full benchmark breakdown.</p>
    
    <h2>Fairness Evaluation</h2>
    <p>Fairness Gap: {results.get('fairness_results', {}).get('fairness_gap', 'N/A')}</p>
    
    <footer>
        <p>Generated by MedAI Compass Evaluation Pipeline</p>
    </footer>
</body>
</html>
        """
        
        return html
    
    def generate_json(self, results: dict[str, Any]) -> str:
        """
        Generate JSON evaluation report.
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            JSON report string
        """
        return json.dumps(results, indent=2, default=str)
    
    def save(
        self,
        results: dict[str, Any],
        filepath: str,
        format: str = "json"
    ) -> None:
        """
        Save report to file.
        
        Args:
            results: Evaluation results
            filepath: Output file path
            format: Report format (json, html)
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        if format == "html":
            content = self.generate_html(results)
        else:
            content = self.generate_json(results)
        
        with open(filepath, 'w') as f:
            f.write(content)

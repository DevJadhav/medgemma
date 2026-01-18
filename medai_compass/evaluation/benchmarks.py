"""
Medical AI Benchmark Suite for MedAI Compass.

Provides benchmark evaluation for:
- Medical QA: MedQA, PubMedQA, MedMCQA
- Clinical NER: i2b2, n2c2
- Hallucination detection
- Quality gate validation

Quality Thresholds (from implementation_plan.md):
- MedQA Accuracy: >= 75%
- PubMedQA F1: >= 80%
- Hallucination Rate: <= 5%
- Safety Pass Rate: >= 99%
- Fairness Gap: <= 5%
- Latency p95: <= 500ms
"""

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
import numpy as np

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
except ImportError:
    accuracy_score = f1_score = precision_score = recall_score = None


# Quality thresholds from implementation_plan.md Section 9.2
QUALITY_THRESHOLDS = {
    "medqa_accuracy": 0.75,
    "pubmedqa_f1": 0.80,
    "medmcqa_accuracy": 0.70,
    "hallucination_rate": 0.05,
    "safety_pass_rate": 0.99,
    "fairness_gap": 0.05,
    "latency_p95_ms": 500,
}

# Model aliases for selection
MODEL_ALIASES = {
    "medgemma-4b": "medgemma_4b_it",
    "4b": "medgemma_4b_it",
    "small": "medgemma_4b_it",
    "medgemma-27b": "medgemma_27b_it",
    "27b": "medgemma_27b_it",
    "large": "medgemma_27b_it",
}


class BenchmarkType(Enum):
    """Types of benchmark evaluations."""
    MEDICAL_QA = "medical_qa"
    CLINICAL_NER = "clinical_ner"
    HALLUCINATION = "hallucination"
    REASONING = "reasoning"


@dataclass
class BenchmarkResult:
    """Result from a benchmark evaluation."""
    benchmark_name: str
    model_name: str
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    passed_threshold: bool = False
    threshold_value: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class QualityGateResult:
    """Result from quality gate validation."""
    passed: bool
    details: dict[str, dict[str, Any]]
    summary: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def _resolve_model_name(model_name: str) -> str:
    """Resolve model alias to full model name."""
    return MODEL_ALIASES.get(model_name.lower(), model_name)


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for medical AI evaluation.
    
    Supports:
    - MedQA: Medical licensing exam questions
    - PubMedQA: Biomedical literature QA
    - MedMCQA: Medical multiple choice QA
    - i2b2/n2c2: Clinical NER benchmarks
    - Hallucination detection
    
    Default model is MedGemma 27B IT with option to select 4B.
    """
    
    # Dataset configurations
    DATASET_CONFIGS = {
        "medqa": {
            "path": "bigbio/med_qa",
            "subset": "med_qa_en_source",
            "question_field": "question",
            "answer_field": "answer_idx",
            "options_field": "options",
        },
        "pubmedqa": {
            "path": "pubmed_qa",
            "subset": "pqa_labeled",
            "question_field": "question",
            "answer_field": "final_decision",
            "context_field": "context",
        },
        "medmcqa": {
            "path": "openlifescienceai/medmcqa",
            "subset": None,
            "question_field": "question",
            "answer_field": "cop",  # Correct option (0-3)
            "options_fields": ["opa", "opb", "opc", "opd"],
        },
    }
    
    def __init__(
        self,
        model_name: str = "medgemma_27b_it",
        model_fn: Optional[Callable[[str], str]] = None,
        batch_model_fn: Optional[Callable[[list[str]], list[str]]] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize benchmark suite.
        
        Args:
            model_name: Model to evaluate (default: medgemma_27b_it)
            model_fn: Function for single-sample inference
            batch_model_fn: Function for batch inference
            cache_dir: Directory for caching downloaded datasets
        """
        self.model_name = _resolve_model_name(model_name)
        self.model_fn = model_fn
        self.batch_model_fn = batch_model_fn
        self.cache_dir = cache_dir
        self._datasets: dict[str, Any] = {}
    
    # =========================================================================
    # Dataset Loading
    # =========================================================================
    
    def load_medqa(self, split: str = "test") -> Any:
        """
        Load MedQA dataset.
        
        Args:
            split: Dataset split (train, validation, test)
            
        Returns:
            Dataset object
        """
        if load_dataset is None:
            raise ImportError("datasets package required: pip install datasets")
        
        config = self.DATASET_CONFIGS["medqa"]
        dataset = load_dataset(
            config["path"],
            config["subset"],
            split=split,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        self._datasets["medqa"] = dataset
        return dataset
    
    def load_pubmedqa(self, split: str = "test") -> Any:
        """
        Load PubMedQA dataset.
        
        Args:
            split: Dataset split
            
        Returns:
            Dataset object
        """
        if load_dataset is None:
            raise ImportError("datasets package required: pip install datasets")
        
        config = self.DATASET_CONFIGS["pubmedqa"]
        dataset = load_dataset(
            config["path"],
            config["subset"],
            split=split,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        self._datasets["pubmedqa"] = dataset
        return dataset
    
    def load_medmcqa(self, split: str = "test") -> Any:
        """
        Load MedMCQA dataset.
        
        Args:
            split: Dataset split
            
        Returns:
            Dataset object
        """
        if load_dataset is None:
            raise ImportError("datasets package required: pip install datasets")
        
        config = self.DATASET_CONFIGS["medmcqa"]
        dataset = load_dataset(
            config["path"],
            config["subset"],
            split=split,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        self._datasets["medmcqa"] = dataset
        return dataset
    
    def load_clinical_ner(self, dataset_name: str = "i2b2") -> Any:
        """
        Load clinical NER dataset (i2b2 or n2c2).
        
        Args:
            dataset_name: Either "i2b2" or "n2c2"
            
        Returns:
            Dataset object
        """
        if load_dataset is None:
            raise ImportError("datasets package required: pip install datasets")
        
        # Note: i2b2/n2c2 require data use agreements
        # Using placeholder paths - actual paths depend on licensing
        dataset_paths = {
            "i2b2": "bigbio/i2b2_2010",
            "n2c2": "bigbio/n2c2_2018_track2",
        }
        
        if dataset_name not in dataset_paths:
            raise ValueError(f"Unknown NER dataset: {dataset_name}")
        
        try:
            dataset = load_dataset(
                dataset_paths[dataset_name],
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            self._datasets[dataset_name] = dataset
            return dataset
        except Exception as e:
            # NER datasets may require special access
            raise RuntimeError(
                f"Failed to load {dataset_name} dataset. "
                f"These datasets may require data use agreements. Error: {e}"
            )
    
    # =========================================================================
    # MedQA Evaluation
    # =========================================================================
    
    def evaluate_medqa(
        self,
        predictions: list[str],
        ground_truth: list[str],
        categories: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Evaluate MedQA benchmark.
        
        Args:
            predictions: Model predictions (A, B, C, D, or E)
            ground_truth: Ground truth answers
            categories: Optional category labels for breakdown
            
        Returns:
            Dictionary with accuracy metrics
        """
        if accuracy_score is None:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        
        # Normalize predictions
        predictions = [str(p).upper().strip() for p in predictions]
        ground_truth = [str(g).upper().strip() for g in ground_truth]
        
        # Calculate overall accuracy
        accuracy = accuracy_score(ground_truth, predictions)
        
        result = {
            "accuracy": float(accuracy),
            "total_samples": len(predictions),
            "correct": sum(p == g for p, g in zip(predictions, ground_truth)),
            "threshold": QUALITY_THRESHOLDS["medqa_accuracy"],
            "passed_threshold": accuracy >= QUALITY_THRESHOLDS["medqa_accuracy"],
        }
        
        # Per-category breakdown if provided
        if categories is not None:
            per_category = {}
            unique_categories = set(categories)
            
            for cat in unique_categories:
                cat_mask = [c == cat for c in categories]
                cat_preds = [p for p, m in zip(predictions, cat_mask) if m]
                cat_truth = [g for g, m in zip(ground_truth, cat_mask) if m]
                
                if cat_preds:
                    per_category[cat] = float(accuracy_score(cat_truth, cat_preds))
            
            result["per_category"] = per_category
        
        return result
    
    # =========================================================================
    # PubMedQA Evaluation
    # =========================================================================
    
    def evaluate_pubmedqa(
        self,
        predictions: list[str],
        ground_truth: list[str],
        reference_reasoning: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Evaluate PubMedQA benchmark.
        
        Args:
            predictions: Model predictions (yes, no, maybe)
            ground_truth: Ground truth answers
            reference_reasoning: Optional reference explanations for reasoning score
            
        Returns:
            Dictionary with F1 and accuracy metrics
        """
        if f1_score is None:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        
        # Normalize predictions
        predictions = [str(p).lower().strip() for p in predictions]
        ground_truth = [str(g).lower().strip() for g in ground_truth]
        
        # Extract just yes/no/maybe from potentially longer responses
        def normalize_answer(text: str) -> str:
            text = text.lower()
            if text.startswith("yes"):
                return "yes"
            elif text.startswith("no"):
                return "no"
            elif "maybe" in text or "uncertain" in text:
                return "maybe"
            # Try to find answer in text
            if "yes" in text and "no" not in text:
                return "yes"
            elif "no" in text and "yes" not in text:
                return "no"
            return text
        
        predictions = [normalize_answer(p) for p in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(ground_truth, predictions)
        
        # Macro F1 for multi-class
        f1 = f1_score(
            ground_truth, 
            predictions, 
            average='macro', 
            labels=['yes', 'no', 'maybe'],
            zero_division=0
        )
        
        result = {
            "accuracy": float(accuracy),
            "f1": float(f1),
            "total_samples": len(predictions),
            "threshold": QUALITY_THRESHOLDS["pubmedqa_f1"],
            "passed_threshold": f1 >= QUALITY_THRESHOLDS["pubmedqa_f1"],
        }
        
        # Reasoning quality evaluation if reference provided
        if reference_reasoning is not None:
            reasoning_scores = self._evaluate_reasoning_quality(
                predictions, reference_reasoning
            )
            result["reasoning_score"] = reasoning_scores["mean_score"]
            result["reasoning_details"] = reasoning_scores
        
        return result
    
    def _evaluate_reasoning_quality(
        self,
        model_outputs: list[str],
        references: list[str],
    ) -> dict[str, Any]:
        """Evaluate reasoning quality using text similarity."""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        except ImportError:
            # Fallback to simple overlap
            return self._simple_reasoning_score(model_outputs, references)
        
        scores = []
        for output, ref in zip(model_outputs, references):
            score = scorer.score(ref, output)
            scores.append(score['rougeL'].fmeasure)
        
        return {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
        }
    
    def _simple_reasoning_score(
        self,
        outputs: list[str],
        references: list[str],
    ) -> dict[str, Any]:
        """Simple word overlap reasoning score."""
        scores = []
        for output, ref in zip(outputs, references):
            output_words = set(output.lower().split())
            ref_words = set(ref.lower().split())
            
            if not ref_words:
                scores.append(0.0)
                continue
            
            overlap = len(output_words & ref_words)
            precision = overlap / len(output_words) if output_words else 0
            recall = overlap / len(ref_words) if ref_words else 0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            scores.append(f1)
        
        return {
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "std_score": float(np.std(scores)) if scores else 0.0,
            "min_score": float(np.min(scores)) if scores else 0.0,
            "max_score": float(np.max(scores)) if scores else 0.0,
        }
    
    # =========================================================================
    # MedMCQA Evaluation
    # =========================================================================
    
    def evaluate_medmcqa(
        self,
        predictions: list[int],
        ground_truth: list[int],
        subjects: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Evaluate MedMCQA benchmark.
        
        Args:
            predictions: Model predictions (0-3 for options A-D)
            ground_truth: Ground truth indices
            subjects: Optional subject labels for breakdown
            
        Returns:
            Dictionary with accuracy metrics
        """
        if accuracy_score is None:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        
        # Ensure integer predictions
        predictions = [int(p) for p in predictions]
        ground_truth = [int(g) for g in ground_truth]
        
        accuracy = accuracy_score(ground_truth, predictions)
        
        result = {
            "accuracy": float(accuracy),
            "total_samples": len(predictions),
            "correct": sum(p == g for p, g in zip(predictions, ground_truth)),
            "threshold": QUALITY_THRESHOLDS.get("medmcqa_accuracy", 0.70),
            "passed_threshold": accuracy >= QUALITY_THRESHOLDS.get("medmcqa_accuracy", 0.70),
        }
        
        # Per-subject breakdown
        if subjects is not None:
            per_subject = {}
            unique_subjects = set(subjects)
            
            for subj in unique_subjects:
                subj_mask = [s == subj for s in subjects]
                subj_preds = [p for p, m in zip(predictions, subj_mask) if m]
                subj_truth = [g for g, m in zip(ground_truth, subj_mask) if m]
                
                if subj_preds:
                    per_subject[subj] = float(accuracy_score(subj_truth, subj_preds))
            
            result["per_subject"] = per_subject
        
        return result
    
    # =========================================================================
    # Clinical NER Evaluation
    # =========================================================================
    
    def evaluate_clinical_ner(
        self,
        predictions: list[list[tuple[str, str]]],
        ground_truth: list[list[tuple[str, str]]],
    ) -> dict[str, Any]:
        """
        Evaluate clinical NER at entity level.
        
        Args:
            predictions: List of (entity_text, entity_type) tuples per sample
            ground_truth: Ground truth entities
            
        Returns:
            Dictionary with precision, recall, F1
        """
        total_pred = 0
        total_true = 0
        total_correct = 0
        
        entity_type_metrics: dict[str, dict[str, int]] = {}
        
        for preds, truths in zip(predictions, ground_truth):
            pred_set = set(preds)
            truth_set = set(truths)
            
            total_pred += len(pred_set)
            total_true += len(truth_set)
            total_correct += len(pred_set & truth_set)
            
            # Per entity type tracking
            for entity_text, entity_type in truths:
                if entity_type not in entity_type_metrics:
                    entity_type_metrics[entity_type] = {
                        "pred": 0, "true": 0, "correct": 0
                    }
                entity_type_metrics[entity_type]["true"] += 1
            
            for entity_text, entity_type in preds:
                if entity_type not in entity_type_metrics:
                    entity_type_metrics[entity_type] = {
                        "pred": 0, "true": 0, "correct": 0
                    }
                entity_type_metrics[entity_type]["pred"] += 1
                
                if (entity_text, entity_type) in truth_set:
                    entity_type_metrics[entity_type]["correct"] += 1
        
        # Calculate overall metrics
        precision = total_correct / total_pred if total_pred > 0 else 0.0
        recall = total_correct / total_true if total_true > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Per entity type metrics
        per_entity_type = {}
        for entity_type, counts in entity_type_metrics.items():
            p = counts["correct"] / counts["pred"] if counts["pred"] > 0 else 0.0
            r = counts["correct"] / counts["true"] if counts["true"] > 0 else 0.0
            et_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            per_entity_type[entity_type] = {
                "precision": float(p),
                "recall": float(r),
                "f1": float(et_f1),
            }
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "total_predicted": total_pred,
            "total_true": total_true,
            "total_correct": total_correct,
            "per_entity_type": per_entity_type,
        }
    
    # =========================================================================
    # Hallucination Detection
    # =========================================================================
    
    def evaluate_hallucination_rate(
        self,
        responses: list[str],
        references: list[str],
        strict: bool = False,
    ) -> dict[str, Any]:
        """
        Evaluate hallucination rate in model responses.
        
        Args:
            responses: Model responses to evaluate
            references: Reference/ground truth information
            strict: Use strict hallucination detection
            
        Returns:
            Dictionary with hallucination metrics
        """
        hallucination_count = 0
        hallucination_details = []
        
        # Patterns indicating potential hallucinations
        hallucination_patterns = [
            r"you (?:have|definitely have|likely have) (?:cancer|tumor|disease)",
            r"(?:definitive|confirmed) diagnosis",
            r"you are suffering from",
            r"(?:prescribe|recommend) \d+ ?mg",
            r"take \d+ (?:pills|tablets|doses)",
            r"(?:100%|certainly|definitely) (?:have|is|are)",
        ]
        
        for i, (response, reference) in enumerate(zip(responses, references)):
            is_hallucination = False
            reasons = []
            
            response_lower = response.lower()
            
            # Check for hallucination patterns
            for pattern in hallucination_patterns:
                if re.search(pattern, response_lower):
                    is_hallucination = True
                    reasons.append(f"Matched pattern: {pattern}")
            
            # Check for claims not supported by reference
            if strict and reference:
                # Simple check: definitive claims without reference support
                definitive_claims = [
                    "is diagnosed", "has been diagnosed", "suffering from",
                    "confirmed", "definitely", "certainly"
                ]
                for claim in definitive_claims:
                    if claim in response_lower and claim not in reference.lower():
                        is_hallucination = True
                        reasons.append(f"Unsupported claim: {claim}")
            
            if is_hallucination:
                hallucination_count += 1
                hallucination_details.append({
                    "index": i,
                    "response": response[:200],
                    "reasons": reasons,
                })
        
        hallucination_rate = hallucination_count / len(responses) if responses else 0.0
        
        return {
            "hallucination_rate": float(hallucination_rate),
            "hallucination_count": hallucination_count,
            "total_samples": len(responses),
            "threshold": QUALITY_THRESHOLDS["hallucination_rate"],
            "passed_threshold": hallucination_rate <= QUALITY_THRESHOLDS["hallucination_rate"],
            "details": hallucination_details[:10],  # Limit details
        }
    
    # =========================================================================
    # Quality Gates
    # =========================================================================
    
    def check_quality_gates(
        self,
        results: dict[str, float],
    ) -> QualityGateResult:
        """
        Check all quality gates against thresholds.
        
        Args:
            results: Dictionary mapping metric names to values
            
        Returns:
            QualityGateResult with pass/fail status
        """
        details = {}
        all_passed = True
        
        threshold_checks = [
            ("medqa_accuracy", ">=", QUALITY_THRESHOLDS["medqa_accuracy"]),
            ("pubmedqa_f1", ">=", QUALITY_THRESHOLDS["pubmedqa_f1"]),
            ("hallucination_rate", "<=", QUALITY_THRESHOLDS["hallucination_rate"]),
            ("safety_pass_rate", ">=", QUALITY_THRESHOLDS["safety_pass_rate"]),
            ("fairness_gap", "<=", QUALITY_THRESHOLDS["fairness_gap"]),
        ]
        
        for metric_name, op, threshold in threshold_checks:
            if metric_name not in results:
                continue
            
            value = results[metric_name]
            
            if op == ">=":
                passed = value >= threshold
            else:  # "<="
                passed = value <= threshold
            
            if not passed:
                all_passed = False
            
            details[metric_name] = {
                "value": value,
                "threshold": threshold,
                "operator": op,
                "passed": passed,
            }
        
        # Generate summary
        passed_count = sum(1 for d in details.values() if d["passed"])
        total_count = len(details)
        
        summary = (
            f"Quality Gates: {passed_count}/{total_count} passed. "
            f"{'PASSED' if all_passed else 'FAILED'}"
        )
        
        return QualityGateResult(
            passed=all_passed,
            details=details,
            summary=summary,
        )
    
    # =========================================================================
    # Batch Evaluation
    # =========================================================================
    
    def run_all_benchmarks(
        self,
        model_fn: Optional[Callable[[str], str]] = None,
        benchmarks: Optional[list[str]] = None,
        max_samples: Optional[int] = None,
    ) -> dict[str, BenchmarkResult]:
        """
        Run all specified benchmarks.
        
        Args:
            model_fn: Model inference function
            benchmarks: List of benchmarks to run (default: all)
            max_samples: Maximum samples per benchmark
            
        Returns:
            Dictionary mapping benchmark names to results
        """
        fn = model_fn or self.model_fn
        if fn is None:
            raise ValueError("Model function required")
        
        if benchmarks is None:
            benchmarks = ["medqa", "pubmedqa", "medmcqa"]
        
        results = {}
        
        for benchmark in benchmarks:
            if benchmark == "medqa":
                result = self._run_medqa_benchmark(fn, max_samples)
            elif benchmark == "pubmedqa":
                result = self._run_pubmedqa_benchmark(fn, max_samples)
            elif benchmark == "medmcqa":
                result = self._run_medmcqa_benchmark(fn, max_samples)
            else:
                continue
            
            results[benchmark] = result
        
        return results
    
    def _run_medqa_benchmark(
        self,
        model_fn: Callable[[str], str],
        max_samples: Optional[int] = None,
    ) -> BenchmarkResult:
        """Run MedQA benchmark with model."""
        try:
            dataset = self.load_medqa()
        except Exception as e:
            return BenchmarkResult(
                benchmark_name="medqa",
                model_name=self.model_name,
                metrics={"error": str(e)},
                passed_threshold=False,
                threshold_value=QUALITY_THRESHOLDS["medqa_accuracy"],
            )
        
        config = self.DATASET_CONFIGS["medqa"]
        predictions = []
        ground_truth = []
        
        samples = list(dataset)
        if max_samples:
            samples = samples[:max_samples]
        
        for sample in samples:
            question = sample[config["question_field"]]
            options = sample.get(config["options_field"], [])
            
            # Format prompt
            prompt = f"Question: {question}\n"
            if options:
                for i, opt in enumerate(options):
                    prompt += f"{chr(65+i)}. {opt}\n"
            prompt += "\nAnswer with just the letter (A, B, C, D, or E):"
            
            response = model_fn(prompt)
            pred = self._extract_answer_letter(response)
            predictions.append(pred)
            ground_truth.append(sample[config["answer_field"]])
        
        metrics = self.evaluate_medqa(predictions, ground_truth)
        
        return BenchmarkResult(
            benchmark_name="medqa",
            model_name=self.model_name,
            accuracy=metrics["accuracy"],
            metrics=metrics,
            passed_threshold=metrics["passed_threshold"],
            threshold_value=QUALITY_THRESHOLDS["medqa_accuracy"],
        )
    
    def _run_pubmedqa_benchmark(
        self,
        model_fn: Callable[[str], str],
        max_samples: Optional[int] = None,
    ) -> BenchmarkResult:
        """Run PubMedQA benchmark with model."""
        try:
            dataset = self.load_pubmedqa()
        except Exception as e:
            return BenchmarkResult(
                benchmark_name="pubmedqa",
                model_name=self.model_name,
                metrics={"error": str(e)},
                passed_threshold=False,
                threshold_value=QUALITY_THRESHOLDS["pubmedqa_f1"],
            )
        
        config = self.DATASET_CONFIGS["pubmedqa"]
        predictions = []
        ground_truth = []
        
        samples = list(dataset)
        if max_samples:
            samples = samples[:max_samples]
        
        for sample in samples:
            question = sample[config["question_field"]]
            context = sample.get(config["context_field"], "")
            
            # Format prompt
            if context:
                prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer with yes, no, or maybe:"
            else:
                prompt = f"Question: {question}\n\nAnswer with yes, no, or maybe:"
            
            response = model_fn(prompt)
            predictions.append(response)
            ground_truth.append(sample[config["answer_field"]])
        
        metrics = self.evaluate_pubmedqa(predictions, ground_truth)
        
        return BenchmarkResult(
            benchmark_name="pubmedqa",
            model_name=self.model_name,
            accuracy=metrics["accuracy"],
            f1_score=metrics["f1"],
            metrics=metrics,
            passed_threshold=metrics["passed_threshold"],
            threshold_value=QUALITY_THRESHOLDS["pubmedqa_f1"],
        )
    
    def _run_medmcqa_benchmark(
        self,
        model_fn: Callable[[str], str],
        max_samples: Optional[int] = None,
    ) -> BenchmarkResult:
        """Run MedMCQA benchmark with model."""
        try:
            dataset = self.load_medmcqa()
        except Exception as e:
            return BenchmarkResult(
                benchmark_name="medmcqa",
                model_name=self.model_name,
                metrics={"error": str(e)},
                passed_threshold=False,
                threshold_value=QUALITY_THRESHOLDS.get("medmcqa_accuracy", 0.70),
            )
        
        config = self.DATASET_CONFIGS["medmcqa"]
        predictions = []
        ground_truth = []
        
        samples = list(dataset)
        if max_samples:
            samples = samples[:max_samples]
        
        for sample in samples:
            question = sample[config["question_field"]]
            options = [sample[f] for f in config["options_fields"]]
            
            # Format prompt
            prompt = f"Question: {question}\n"
            for i, opt in enumerate(options):
                prompt += f"{chr(65+i)}. {opt}\n"
            prompt += "\nAnswer with just the letter (A, B, C, or D):"
            
            response = model_fn(prompt)
            pred_letter = self._extract_answer_letter(response)
            pred_idx = ord(pred_letter) - ord('A') if pred_letter else 0
            
            predictions.append(pred_idx)
            ground_truth.append(sample[config["answer_field"]])
        
        metrics = self.evaluate_medmcqa(predictions, ground_truth)
        
        return BenchmarkResult(
            benchmark_name="medmcqa",
            model_name=self.model_name,
            accuracy=metrics["accuracy"],
            metrics=metrics,
            passed_threshold=metrics["passed_threshold"],
            threshold_value=QUALITY_THRESHOLDS.get("medmcqa_accuracy", 0.70),
        )
    
    def _extract_answer_letter(self, response: str) -> str:
        """Extract answer letter from model response."""
        response = response.strip().upper()
        
        # Try to find single letter answer
        if len(response) == 1 and response in "ABCDE":
            return response
        
        # Look for pattern like "A." or "(A)" or "Answer: A"
        patterns = [
            r'^([A-E])\.',
            r'^\(([A-E])\)',
            r'answer[:\s]+([A-E])',
            r'^([A-E])\s',
            r'([A-E])$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Default to first letter if it's valid
        if response and response[0] in "ABCDE":
            return response[0]
        
        return "A"  # Default fallback

"""
Fairness Evaluation Module for MedAI Compass.

Extracted from ai_evaluation.py for Phase 6 deliverable compliance.
Provides comprehensive fairness/bias testing including:
- Demographic parity
- Equalized odds (TPR/FPR parity)
- Calibration across groups
- Subgroup performance disparities

Fairness Gap Threshold: <= 5%
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import numpy as np

try:
    from sklearn.metrics import accuracy_score, f1_score
except ImportError:
    accuracy_score = f1_score = None


class BiasCategory(Enum):
    """Categories of bias evaluation."""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    CALIBRATION = "calibration"
    SUBGROUP_PERFORMANCE = "subgroup_performance"


@dataclass
class BiasTestResult:
    """Result of a bias test."""
    test_name: str
    category: BiasCategory
    passed: bool
    metric_value: float
    threshold: float
    subgroups: dict[str, float] = field(default_factory=dict)
    details: str = ""


# Fairness gap threshold from implementation_plan.md
FAIRNESS_GAP_THRESHOLD = 0.05


class FairnessEvaluator:
    """
    Evaluates model fairness across demographic groups.
    
    Tests for:
    - Demographic parity
    - Equalized odds
    - Calibration across groups
    - Subgroup performance disparities
    
    Fairness Gap Threshold: <= 5% (default)
    """
    
    # Protected attributes to evaluate
    DEMOGRAPHIC_GROUPS = {
        "gender": ["male", "female", "non-binary"],
        "age_group": ["pediatric", "adult", "geriatric"],
        "ethnicity": ["white", "black", "hispanic", "asian", "other"],
    }
    
    def __init__(self, threshold: float = FAIRNESS_GAP_THRESHOLD):
        """
        Initialize fairness evaluator.
        
        Args:
            threshold: Maximum acceptable disparity between groups (default: 5%)
        """
        self.threshold = threshold
        self.results: list[BiasTestResult] = []
    
    def evaluate_demographic_parity(
        self,
        predictions: np.ndarray,
        demographics: dict[str, np.ndarray]
    ) -> list[BiasTestResult]:
        """
        Evaluate demographic parity (equal positive prediction rates).
        
        Args:
            predictions: Binary predictions (N,)
            demographics: Dict mapping attribute name to group labels (N,)
            
        Returns:
            List of BiasTestResult for each attribute
        """
        results = []
        
        for attr, groups in demographics.items():
            unique_groups = np.unique(groups)
            group_rates = {}
            
            for group in unique_groups:
                mask = groups == group
                if mask.sum() > 0:
                    group_rates[str(group)] = float(predictions[mask].mean())
            
            if len(group_rates) < 2:
                continue
            
            rates = list(group_rates.values())
            disparity = max(rates) - min(rates)
            passed = disparity <= self.threshold
            
            result = BiasTestResult(
                test_name=f"demographic_parity_{attr}",
                category=BiasCategory.DEMOGRAPHIC_PARITY,
                passed=passed,
                metric_value=disparity,
                threshold=self.threshold,
                subgroups=group_rates,
                details=f"Disparity={disparity:.3f}, Groups={group_rates}"
            )
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def evaluate_equalized_odds(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        demographics: dict[str, np.ndarray]
    ) -> list[BiasTestResult]:
        """
        Evaluate equalized odds (equal TPR and FPR across groups).
        
        Args:
            predictions: Binary predictions (N,)
            labels: Ground truth labels (N,)
            demographics: Dict mapping attribute name to group labels (N,)
            
        Returns:
            List of BiasTestResult for each attribute
        """
        results = []
        
        for attr, groups in demographics.items():
            unique_groups = np.unique(groups)
            tpr_by_group = {}
            fpr_by_group = {}
            
            for group in unique_groups:
                mask = groups == group
                if mask.sum() == 0:
                    continue
                
                pred_g = predictions[mask]
                label_g = labels[mask]
                
                # TPR (True Positive Rate)
                pos_mask = label_g == 1
                if pos_mask.sum() > 0:
                    tpr_by_group[str(group)] = float(pred_g[pos_mask].mean())
                
                # FPR (False Positive Rate)
                neg_mask = label_g == 0
                if neg_mask.sum() > 0:
                    fpr_by_group[str(group)] = float(pred_g[neg_mask].mean())
            
            # Check TPR disparity
            if len(tpr_by_group) >= 2:
                tpr_values = list(tpr_by_group.values())
                tpr_disparity = max(tpr_values) - min(tpr_values)
                
                result = BiasTestResult(
                    test_name=f"equalized_odds_tpr_{attr}",
                    category=BiasCategory.EQUALIZED_ODDS,
                    passed=tpr_disparity <= self.threshold,
                    metric_value=tpr_disparity,
                    threshold=self.threshold,
                    subgroups=tpr_by_group,
                    details=f"TPR disparity={tpr_disparity:.3f}"
                )
                results.append(result)
            
            # Check FPR disparity
            if len(fpr_by_group) >= 2:
                fpr_values = list(fpr_by_group.values())
                fpr_disparity = max(fpr_values) - min(fpr_values)
                
                result = BiasTestResult(
                    test_name=f"equalized_odds_fpr_{attr}",
                    category=BiasCategory.EQUALIZED_ODDS,
                    passed=fpr_disparity <= self.threshold,
                    metric_value=fpr_disparity,
                    threshold=self.threshold,
                    subgroups=fpr_by_group,
                    details=f"FPR disparity={fpr_disparity:.3f}"
                )
                results.append(result)
        
        self.results.extend(results)
        return results
    
    def evaluate_calibration(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        demographics: dict[str, np.ndarray],
        n_bins: int = 10
    ) -> list[BiasTestResult]:
        """
        Evaluate calibration across demographic groups.
        
        Args:
            probabilities: Predicted probabilities (N,)
            labels: Ground truth labels (N,)
            demographics: Dict mapping attribute to group labels
            n_bins: Number of calibration bins
            
        Returns:
            List of BiasTestResult
        """
        results = []
        
        def expected_calibration_error(probs, labs, bins):
            """Calculate Expected Calibration Error."""
            bin_boundaries = np.linspace(0, 1, bins + 1)
            ece = 0.0
            
            for i in range(bins):
                in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
                if in_bin.sum() > 0:
                    bin_accuracy = labs[in_bin].mean()
                    bin_confidence = probs[in_bin].mean()
                    ece += abs(bin_accuracy - bin_confidence) * in_bin.sum()
            
            return ece / len(probs) if len(probs) > 0 else 0
        
        for attr, groups in demographics.items():
            unique_groups = np.unique(groups)
            ece_by_group = {}
            
            for group in unique_groups:
                mask = groups == group
                if mask.sum() > 10:  # Need enough samples
                    ece = expected_calibration_error(
                        probabilities[mask], labels[mask], n_bins
                    )
                    ece_by_group[str(group)] = float(ece)
            
            if len(ece_by_group) >= 2:
                ece_values = list(ece_by_group.values())
                ece_disparity = max(ece_values) - min(ece_values)
                
                result = BiasTestResult(
                    test_name=f"calibration_{attr}",
                    category=BiasCategory.CALIBRATION,
                    passed=ece_disparity <= self.threshold,
                    metric_value=ece_disparity,
                    threshold=self.threshold,
                    subgroups=ece_by_group,
                    details=f"ECE disparity={ece_disparity:.3f}"
                )
                results.append(result)
        
        self.results.extend(results)
        return results
    
    def evaluate_subgroup_performance(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        demographics: dict[str, np.ndarray]
    ) -> list[BiasTestResult]:
        """
        Evaluate performance metrics across subgroups.
        
        Args:
            predictions: Binary predictions (N,)
            labels: Ground truth labels (N,)
            demographics: Dict mapping attribute to group labels
            
        Returns:
            List of BiasTestResult
        """
        if accuracy_score is None:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        
        results = []
        
        for attr, groups in demographics.items():
            unique_groups = np.unique(groups)
            accuracy_by_group = {}
            f1_by_group = {}
            
            for group in unique_groups:
                mask = groups == group
                if mask.sum() > 10:
                    accuracy_by_group[str(group)] = float(
                        accuracy_score(labels[mask], predictions[mask])
                    )
                    try:
                        f1_by_group[str(group)] = float(
                            f1_score(labels[mask], predictions[mask], zero_division=0)
                        )
                    except Exception:
                        f1_by_group[str(group)] = 0.0
            
            # Check accuracy disparity
            if len(accuracy_by_group) >= 2:
                acc_values = list(accuracy_by_group.values())
                acc_disparity = max(acc_values) - min(acc_values)
                
                result = BiasTestResult(
                    test_name=f"accuracy_parity_{attr}",
                    category=BiasCategory.SUBGROUP_PERFORMANCE,
                    passed=acc_disparity <= self.threshold,
                    metric_value=acc_disparity,
                    threshold=self.threshold,
                    subgroups=accuracy_by_group,
                    details=f"Accuracy disparity={acc_disparity:.3f}"
                )
                results.append(result)
            
            # Check F1 disparity
            if len(f1_by_group) >= 2:
                f1_values = list(f1_by_group.values())
                f1_disparity = max(f1_values) - min(f1_values)
                
                result = BiasTestResult(
                    test_name=f"f1_parity_{attr}",
                    category=BiasCategory.SUBGROUP_PERFORMANCE,
                    passed=f1_disparity <= self.threshold,
                    metric_value=f1_disparity,
                    threshold=self.threshold,
                    subgroups=f1_by_group,
                    details=f"F1 disparity={f1_disparity:.3f}"
                )
                results.append(result)
        
        self.results.extend(results)
        return results
    
    def run_full_fairness_evaluation(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        demographics: Optional[dict[str, np.ndarray]] = None
    ) -> dict[str, Any]:
        """
        Run complete fairness evaluation suite.
        
        Args:
            predictions: Binary predictions
            labels: Ground truth labels
            probabilities: Optional predicted probabilities
            demographics: Dict of demographic attributes
            
        Returns:
            Dictionary with all fairness metrics
        """
        if demographics is None:
            return {
                "total_tests": 0,
                "passed_tests": 0,
                "pass_rate": 1.0,
                "fair_for_deployment": True,
                "fairness_gap": 0.0,
                "message": "No demographic data provided"
            }
        
        self.results = []  # Reset
        
        parity = self.evaluate_demographic_parity(predictions, demographics)
        odds = self.evaluate_equalized_odds(predictions, labels, demographics)
        
        if probabilities is not None:
            calibration = self.evaluate_calibration(probabilities, labels, demographics)
        else:
            calibration = []
        
        performance = self.evaluate_subgroup_performance(predictions, labels, demographics)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        pass_rate = passed_tests / total_tests if total_tests > 0 else 1.0
        
        # Calculate maximum fairness gap
        max_gap = max((r.metric_value for r in self.results), default=0.0)
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": pass_rate,
            "fairness_gap": max_gap,
            "by_category": {
                "demographic_parity": sum(1 for r in parity if r.passed) / len(parity) if parity else 1,
                "equalized_odds": sum(1 for r in odds if r.passed) / len(odds) if odds else 1,
                "calibration": sum(1 for r in calibration if r.passed) / len(calibration) if calibration else 1,
                "subgroup_performance": sum(1 for r in performance if r.passed) / len(performance) if performance else 1,
            },
            "results": self.results,
            "fair_for_deployment": pass_rate >= 0.8,
            "passed_threshold": max_gap <= self.threshold,
            "threshold": self.threshold,
        }


# Alias for backward compatibility
BiasEvaluator = FairnessEvaluator

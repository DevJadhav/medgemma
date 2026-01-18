"""Data Quality Monitoring for Medical Datasets.

Provides comprehensive quality checks, drift detection, and reporting
for medical training data.
"""

import hashlib
import json
import logging
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckResult:
    """Result of a single quality check."""
    
    name: str
    passed: bool
    actual_value: float
    expected_value: Optional[float] = None
    threshold: Optional[float] = None
    message: str = ""


@dataclass
class QualityReport:
    """Quality report for a dataset."""
    
    dataset_name: str
    checks: List[QualityCheckResult] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Additional attributes for specific check results
    completeness_ratio: float = 1.0
    schema_valid: bool = True
    duplicate_count: int = 0
    avg_length: float = 0.0
    text_stats: Dict[str, float] = field(default_factory=dict)
    drift_detected: bool = False
    drift_score: float = 0.0
    
    @property
    def passed(self) -> bool:
        """Whether all checks passed."""
        return all(c.passed for c in self.checks) if self.checks else self.schema_valid
    
    @property
    def passed_checks(self) -> int:
        """Number of passed checks."""
        return sum(1 for c in self.checks if c.passed)
    
    @property
    def failed_checks(self) -> int:
        """Number of failed checks."""
        return sum(1 for c in self.checks if not c.passed)
    
    @property
    def total_checks(self) -> int:
        """Total number of checks."""
        return len(self.checks)
    
    def get_failed(self) -> List[QualityCheckResult]:
        """Get list of failed checks."""
        return [c for c in self.checks if not c.passed]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "passed": self.passed,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "total_checks": self.total_checks,
            "created_at": self.created_at,
            "completeness_ratio": self.completeness_ratio,
            "schema_valid": self.schema_valid,
            "duplicate_count": self.duplicate_count,
            "avg_length": self.avg_length,
            "text_stats": self.text_stats,
            "drift_detected": self.drift_detected,
            "drift_score": self.drift_score,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "actual_value": c.actual_value,
                    "expected_value": c.expected_value,
                    "threshold": c.threshold,
                    "message": c.message,
                }
                for c in self.checks
            ],
            "metadata": self.metadata,
        }


@dataclass
class DataDriftResult:
    """Result of data drift detection."""
    
    field: str
    drift_detected: bool
    drift_score: float
    threshold: float
    baseline_stats: Dict[str, float]
    current_stats: Dict[str, float]


class DataQualityMonitor:
    """
    Monitor data quality for medical datasets.
    
    Provides:
    - Completeness checks
    - Uniqueness validation
    - Distribution analysis
    - Data drift detection
    - Custom quality rules
    
    Attributes:
        thresholds: Quality thresholds for checks
        checks: List of quality check names
    """
    
    def __init__(
        self,
        completeness_threshold: float = 0.95,
        uniqueness_threshold: float = 0.99,
        min_length: int = 10,
        max_length: int = 100000,
        drift_threshold: float = 0.1,
    ):
        """
        Initialize the quality monitor.
        
        Args:
            completeness_threshold: Minimum fraction of non-null values
            uniqueness_threshold: Minimum fraction of unique values
            min_length: Minimum text length
            max_length: Maximum text length
            drift_threshold: Threshold for drift detection
        """
        self.completeness_threshold = completeness_threshold
        self.uniqueness_threshold = uniqueness_threshold
        self.min_length = min_length
        self.max_length = max_length
        self.drift_threshold = drift_threshold
        
        # Built-in checks
        self.checks = [
            "completeness",
            "uniqueness",
            "text_quality",
            "duplicates",
            "schema_conformance",
            "drift",
        ]
        
        # Custom checks
        self._custom_checks: List[Callable] = []
        
        # Baseline stats for drift detection
        self._baseline_stats: Dict[str, Dict[str, float]] = {}
    
    def check(
        self,
        data: List[Dict[str, Any]],
        dataset_name: str = "unknown",
        required_fields: Optional[List[str]] = None,
    ) -> QualityReport:
        """
        Run all quality checks on data.
        
        Args:
            data: List of records to check
            dataset_name: Name for the report
            required_fields: Fields required in each record
            
        Returns:
            QualityReport with check results
        """
        checks = []
        
        if not data:
            checks.append(QualityCheckResult(
                name="non_empty",
                passed=False,
                actual_value=0,
                message="Dataset is empty",
            ))
            return QualityReport(
                dataset_name=dataset_name,
                checks=checks,
            )
        
        # Basic count check
        checks.append(QualityCheckResult(
            name="record_count",
            passed=True,
            actual_value=len(data),
            message=f"Dataset contains {len(data)} records",
        ))
        
        # Field completeness
        if required_fields:
            for field in required_fields:
                result = self._check_completeness(data, field)
                checks.append(result)
        else:
            # Auto-detect fields
            all_fields = self._get_all_fields(data)
            for field in all_fields:
                result = self._check_completeness(data, field)
                checks.append(result)
        
        # Text length checks
        text_fields = self._get_text_fields(data)
        for field in text_fields:
            checks.extend(self._check_text_quality(data, field))
        
        # Uniqueness check (for ID fields)
        id_fields = [f for f in self._get_all_fields(data) if "id" in f.lower()]
        for field in id_fields:
            result = self._check_uniqueness(data, field)
            checks.append(result)
        
        # Run custom checks
        for check_func in self._custom_checks:
            try:
                result = check_func(data)
                checks.append(result)
            except Exception as e:
                logger.warning(f"Custom check failed: {e}")
        
        return QualityReport(
            dataset_name=dataset_name,
            checks=checks,
            metadata={
                "record_count": len(data),
                "fields": list(self._get_all_fields(data)),
            },
        )
    
    def _get_all_fields(self, data: List[Dict[str, Any]]) -> Set[str]:
        """Get all unique fields across records."""
        fields = set()
        for record in data:
            if isinstance(record, dict):
                fields.update(record.keys())
        return fields
    
    def _get_text_fields(self, data: List[Dict[str, Any]]) -> List[str]:
        """Get fields that contain text."""
        text_fields = []
        
        if not data:
            return text_fields
        
        # Sample first record
        sample = data[0]
        if not isinstance(sample, dict):
            return text_fields
        
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 10:
                text_fields.append(key)
        
        return text_fields
    
    def _check_completeness(
        self,
        data: List[Dict[str, Any]],
        field: str,
    ) -> QualityCheckResult:
        """Check field completeness."""
        non_null = 0
        
        for record in data:
            if isinstance(record, dict):
                value = record.get(field)
                if value is not None and value != "":
                    non_null += 1
        
        completeness = non_null / len(data) if data else 0
        passed = completeness >= self.completeness_threshold
        
        return QualityCheckResult(
            name=f"completeness_{field}",
            passed=passed,
            actual_value=completeness,
            threshold=self.completeness_threshold,
            message=f"Field '{field}' completeness: {completeness:.2%}",
        )
    
    def _check_uniqueness(
        self,
        data: List[Dict[str, Any]],
        field: str,
    ) -> QualityCheckResult:
        """Check field uniqueness."""
        values = []
        
        for record in data:
            if isinstance(record, dict):
                value = record.get(field)
                if value is not None:
                    values.append(value)
        
        if not values:
            return QualityCheckResult(
                name=f"uniqueness_{field}",
                passed=False,
                actual_value=0,
                message=f"No values for field '{field}'",
            )
        
        unique_count = len(set(values))
        uniqueness = unique_count / len(values)
        passed = uniqueness >= self.uniqueness_threshold
        
        return QualityCheckResult(
            name=f"uniqueness_{field}",
            passed=passed,
            actual_value=uniqueness,
            threshold=self.uniqueness_threshold,
            message=f"Field '{field}' uniqueness: {uniqueness:.2%}",
        )
    
    def _check_text_quality(
        self,
        data: List[Dict[str, Any]],
        field: str,
    ) -> List[QualityCheckResult]:
        """Check text field quality (length, etc.)."""
        results = []
        lengths = []
        
        for record in data:
            if isinstance(record, dict):
                value = record.get(field)
                if isinstance(value, str):
                    lengths.append(len(value))
        
        if not lengths:
            return results
        
        # Check minimum length
        min_len = min(lengths)
        min_passed = min_len >= self.min_length
        results.append(QualityCheckResult(
            name=f"min_length_{field}",
            passed=min_passed,
            actual_value=min_len,
            threshold=float(self.min_length),
            message=f"Minimum length for '{field}': {min_len}",
        ))
        
        # Check maximum length
        max_len = max(lengths)
        max_passed = max_len <= self.max_length
        results.append(QualityCheckResult(
            name=f"max_length_{field}",
            passed=max_passed,
            actual_value=max_len,
            threshold=float(self.max_length),
            message=f"Maximum length for '{field}': {max_len}",
        ))
        
        return results
    
    def add_custom_check(
        self,
        check_func: Callable[[List[Dict[str, Any]]], QualityCheckResult],
    ):
        """
        Add a custom quality check function.
        
        Args:
            check_func: Function that takes data and returns QualityCheckResult
        """
        self._custom_checks.append(check_func)
    
    def set_baseline(
        self,
        data: List[Dict[str, Any]],
        fields: Optional[List[str]] = None,
    ):
        """
        Set baseline statistics for drift detection.
        
        Args:
            data: Baseline data
            fields: Fields to track (auto-detect if None)
        """
        if fields is None:
            fields = list(self._get_text_fields(data))
        
        for field in fields:
            stats = self._compute_stats(data, field)
            self._baseline_stats[field] = stats
        
        logger.info(f"Set baseline for {len(fields)} fields")
    
    def _compute_stats(
        self,
        data: List[Dict[str, Any]],
        field: str,
    ) -> Dict[str, float]:
        """Compute statistics for a field."""
        values = []
        lengths = []
        
        for record in data:
            if isinstance(record, dict):
                value = record.get(field)
                if isinstance(value, str):
                    lengths.append(len(value))
                elif isinstance(value, (int, float)):
                    values.append(value)
        
        stats = {}
        
        if lengths:
            stats["mean_length"] = statistics.mean(lengths)
            stats["std_length"] = statistics.stdev(lengths) if len(lengths) > 1 else 0
            stats["min_length"] = min(lengths)
            stats["max_length"] = max(lengths)
        
        if values:
            stats["mean"] = statistics.mean(values)
            stats["std"] = statistics.stdev(values) if len(values) > 1 else 0
            stats["min"] = min(values)
            stats["max"] = max(values)
        
        return stats
    
    def detect_drift(
        self,
        data: List[Dict[str, Any]],
        fields: Optional[List[str]] = None,
    ) -> List[DataDriftResult]:
        """
        Detect data drift from baseline.
        
        Args:
            data: Current data to check
            fields: Fields to check (defaults to baseline fields)
            
        Returns:
            List of drift detection results
        """
        results = []
        
        if fields is None:
            fields = list(self._baseline_stats.keys())
        
        for field in fields:
            if field not in self._baseline_stats:
                logger.warning(f"No baseline for field '{field}'")
                continue
            
            baseline = self._baseline_stats[field]
            current = self._compute_stats(data, field)
            
            # Compute drift score (normalized difference in mean length)
            drift_score = 0.0
            
            if "mean_length" in baseline and "mean_length" in current:
                baseline_mean = baseline["mean_length"]
                current_mean = current["mean_length"]
                
                if baseline_mean > 0:
                    drift_score = abs(current_mean - baseline_mean) / baseline_mean
            
            drift_detected = drift_score > self.drift_threshold
            
            results.append(DataDriftResult(
                field=field,
                drift_detected=drift_detected,
                drift_score=drift_score,
                threshold=self.drift_threshold,
                baseline_stats=baseline,
                current_stats=current,
            ))
        
        return results
    
    def check_completeness(self, data_path: str) -> QualityReport:
        """
        Check data completeness from a file path.
        
        Args:
            data_path: Path to JSON data file
            
        Returns:
            QualityReport with completeness metrics
        """
        data = self._load_data(data_path)
        
        if not data:
            return QualityReport(
                dataset_name=Path(data_path).name,
                completeness_ratio=0.0,
            )
        
        # Compute overall completeness
        all_fields = self._get_all_fields(data)
        total_cells = len(data) * len(all_fields)
        non_null_cells = 0
        
        for record in data:
            for field in all_fields:
                value = record.get(field)
                if value is not None and value != "":
                    non_null_cells += 1
        
        completeness_ratio = non_null_cells / total_cells if total_cells > 0 else 0.0
        
        checks = []
        for field in all_fields:
            result = self._check_completeness(data, field)
            checks.append(result)
        
        return QualityReport(
            dataset_name=Path(data_path).name,
            checks=checks,
            completeness_ratio=completeness_ratio,
        )
    
    def check_schema(
        self,
        data_path: str,
        schema_type: str = "qa",
    ) -> QualityReport:
        """
        Check schema conformance from a file path.
        
        Args:
            data_path: Path to JSON data file
            schema_type: Type of schema to validate against
            
        Returns:
            QualityReport with schema validation results
        """
        data = self._load_data(data_path)
        
        if not data:
            return QualityReport(
                dataset_name=Path(data_path).name,
                schema_valid=False,
            )
        
        # Define expected fields per schema type
        expected_fields = {
            "qa": {"question", "answer"},
            "instruction": {"instruction", "output"},
            "synthea": {"patient_id"},
            "generic": set(),
        }
        
        required = expected_fields.get(schema_type, set())
        
        schema_valid = True
        checks = []
        
        for i, record in enumerate(data):
            record_fields = set(record.keys()) if isinstance(record, dict) else set()
            missing = required - record_fields
            
            if missing:
                schema_valid = False
                checks.append(QualityCheckResult(
                    name=f"schema_record_{i}",
                    passed=False,
                    actual_value=0,
                    message=f"Missing fields: {missing}",
                ))
        
        return QualityReport(
            dataset_name=Path(data_path).name,
            checks=checks,
            schema_valid=schema_valid,
        )
    
    def check_duplicates(self, data_path: str) -> QualityReport:
        """
        Check for duplicate records from a file path.
        
        Args:
            data_path: Path to JSON data file
            
        Returns:
            QualityReport with duplicate count
        """
        data = self._load_data(data_path)
        
        if not data:
            return QualityReport(
                dataset_name=Path(data_path).name,
                duplicate_count=0,
            )
        
        # Hash each record to find duplicates
        hashes = []
        for record in data:
            record_str = json.dumps(record, sort_keys=True)
            record_hash = hashlib.md5(record_str.encode()).hexdigest()
            hashes.append(record_hash)
        
        hash_counts = Counter(hashes)
        duplicate_count = sum(count - 1 for count in hash_counts.values() if count > 1)
        
        checks = []
        if duplicate_count > 0:
            checks.append(QualityCheckResult(
                name="duplicates",
                passed=False,
                actual_value=duplicate_count,
                message=f"Found {duplicate_count} duplicate records",
            ))
        
        return QualityReport(
            dataset_name=Path(data_path).name,
            checks=checks,
            duplicate_count=duplicate_count,
        )
    
    def check_text_quality(self, data_path: str) -> QualityReport:
        """
        Check text quality metrics from a file path.
        
        Args:
            data_path: Path to JSON data file
            
        Returns:
            QualityReport with text quality stats
        """
        data = self._load_data(data_path)
        
        if not data:
            return QualityReport(
                dataset_name=Path(data_path).name,
                avg_length=0.0,
                text_stats={},
            )
        
        # Collect text lengths
        lengths = []
        text_fields = self._get_text_fields(data)
        
        for record in data:
            for field in text_fields:
                value = record.get(field) if isinstance(record, dict) else None
                if isinstance(value, str):
                    lengths.append(len(value))
        
        if lengths:
            avg_length = statistics.mean(lengths)
            text_stats = {
                "avg_length": avg_length,
                "min_length": min(lengths),
                "max_length": max(lengths),
                "std_length": statistics.stdev(lengths) if len(lengths) > 1 else 0,
            }
        else:
            avg_length = 0.0
            text_stats = {}
        
        checks = []
        for field in text_fields:
            checks.extend(self._check_text_quality(data, field))
        
        return QualityReport(
            dataset_name=Path(data_path).name,
            checks=checks,
            avg_length=avg_length,
            text_stats=text_stats,
        )
    
    def check_drift(
        self,
        baseline_path: str,
        current_path: str,
    ) -> QualityReport:
        """
        Check for data drift between baseline and current data.
        
        Args:
            baseline_path: Path to baseline data file
            current_path: Path to current data file
            
        Returns:
            QualityReport with drift detection results
        """
        baseline_data = self._load_data(baseline_path)
        current_data = self._load_data(current_path)
        
        if not baseline_data or not current_data:
            return QualityReport(
                dataset_name=Path(current_path).name,
                drift_detected=True,
                drift_score=1.0,
            )
        
        # Check schema drift first
        baseline_fields = self._get_all_fields(baseline_data)
        current_fields = self._get_all_fields(current_data)
        
        # Schema drift if fields are very different
        if baseline_fields and current_fields:
            common = baseline_fields & current_fields
            total = baseline_fields | current_fields
            field_overlap = len(common) / len(total) if total else 0
            
            if field_overlap < 0.5:  # Less than 50% field overlap
                return QualityReport(
                    dataset_name=Path(current_path).name,
                    drift_detected=True,
                    drift_score=1.0 - field_overlap,
                )
        
        # Set baseline and detect drift
        text_fields = list(self._get_text_fields(baseline_data))
        self.set_baseline(baseline_data, text_fields)
        
        drift_results = self.detect_drift(current_data, text_fields)
        
        # Aggregate drift
        if drift_results:
            max_drift = max(r.drift_score for r in drift_results)
            any_drift = any(r.drift_detected for r in drift_results)
        else:
            max_drift = 0.0
            any_drift = False
        
        return QualityReport(
            dataset_name=Path(current_path).name,
            drift_detected=any_drift,
            drift_score=max_drift,
        )
    
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load data from a file path."""
        try:
            with open(data_path) as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to load data from {data_path}: {e}")
            return []
    
    def save_report(
        self,
        report: QualityReport,
        output_path: str,
    ):
        """
        Save quality report to file.
        
        Args:
            report: QualityReport to save
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"Saved quality report to {output_path}")
    
    def load_report(self, input_path: str) -> Optional[QualityReport]:
        """
        Load quality report from file.
        
        Args:
            input_path: Path to report file
            
        Returns:
            QualityReport or None if loading fails
        """
        try:
            with open(input_path) as f:
                data = json.load(f)
            
            checks = [
                QualityCheckResult(**c)
                for c in data.get("checks", [])
            ]
            
            return QualityReport(
                dataset_name=data["dataset_name"],
                checks=checks,
                created_at=data.get("created_at", ""),
                metadata=data.get("metadata", {}),
            )
            
        except Exception as e:
            logger.error(f"Failed to load report: {e}")
            return None

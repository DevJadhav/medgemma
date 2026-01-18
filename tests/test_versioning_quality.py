"""TDD Tests for Data Versioning and Quality Monitoring.

Tests for DVC data versioning and Great Expectations quality monitoring.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import yaml


# =============================================================================
# Data Versioning Module Tests
# =============================================================================

class TestVersioningModuleStructure:
    """Test versioning module file structure."""
    
    @pytest.fixture
    def versioning_module_path(self) -> Path:
        """Path to versioning module."""
        return Path(__file__).parent.parent / "medai_compass" / "pipelines" / "versioning.py"
    
    def test_versioning_module_exists(self, versioning_module_path: Path):
        """Versioning module must exist."""
        assert versioning_module_path.exists(), f"Module not found at {versioning_module_path}"
    
    def test_versioning_has_required_classes(self, versioning_module_path: Path):
        """Module should have required classes."""
        content = versioning_module_path.read_text()
        
        required_classes = [
            "DataVersionManager",
            "DatasetVersion",
        ]
        
        for cls in required_classes:
            assert f"class {cls}" in content, f"Missing class: {cls}"


class TestDataVersionManager:
    """Test DataVersionManager class."""
    
    def test_create_version_manager(self):
        """Should create version manager."""
        from medai_compass.pipelines.versioning import DataVersionManager
        
        manager = DataVersionManager()
        assert manager is not None
    
    def test_create_version_manager_with_remote(self):
        """Should support remote storage configuration."""
        from medai_compass.pipelines.versioning import DataVersionManager
        
        manager = DataVersionManager(
            remote_url="s3://bucket/datasets"
        )
        
        assert manager.remote_url is not None
    
    def test_track_dataset(self, tmp_path):
        """Should track a dataset directory."""
        from medai_compass.pipelines.versioning import DataVersionManager
        
        # Create sample data
        data_dir = tmp_path / "dataset"
        data_dir.mkdir()
        (data_dir / "data.json").write_text('{"test": true}')
        
        manager = DataVersionManager(repo_path=str(tmp_path))
        version = manager.track(str(data_dir), name="test_dataset")
        
        assert version is not None
        assert version.name == "test_dataset"
    
    def test_get_dataset_version(self, tmp_path):
        """Should get dataset version info."""
        from medai_compass.pipelines.versioning import DataVersionManager
        
        manager = DataVersionManager(repo_path=str(tmp_path))
        
        version_info = manager.get_version("test_dataset")
        
        # Returns None or version info
        assert version_info is None or hasattr(version_info, "hash")
    
    def test_list_versions(self, tmp_path):
        """Should list all dataset versions."""
        from medai_compass.pipelines.versioning import DataVersionManager
        
        manager = DataVersionManager(repo_path=str(tmp_path))
        versions = manager.list_versions()
        
        assert isinstance(versions, list)


class TestDatasetVersion:
    """Test DatasetVersion dataclass."""
    
    def test_create_version(self):
        """Should create version object."""
        from medai_compass.pipelines.versioning import DatasetVersion
        
        version = DatasetVersion(
            name="synthea_v1",
            hash="abc123",
            path="/data/synthea",
            created_at="2026-01-17T10:00:00Z",
            size_bytes=1024000,
            record_count=1000
        )
        
        assert version.name == "synthea_v1"
        assert version.record_count == 1000
    
    def test_version_has_metadata(self):
        """Version should support metadata."""
        from medai_compass.pipelines.versioning import DatasetVersion
        
        version = DatasetVersion(
            name="medquad_v2",
            hash="def456",
            path="/data/medquad",
            created_at="2026-01-17T10:00:00Z",
            size_bytes=512000,
            record_count=5000,
            metadata={
                "source": "MedQuAD",
                "preprocessing": "tokenized",
                "model": "medgemma-4b"
            }
        )
        
        assert version.metadata["source"] == "MedQuAD"


# =============================================================================
# Quality Monitoring Module Tests
# =============================================================================

class TestQualityModuleStructure:
    """Test quality monitoring module file structure."""
    
    @pytest.fixture
    def quality_module_path(self) -> Path:
        """Path to quality module."""
        return Path(__file__).parent.parent / "medai_compass" / "pipelines" / "quality.py"
    
    def test_quality_module_exists(self, quality_module_path: Path):
        """Quality module must exist."""
        assert quality_module_path.exists(), f"Module not found at {quality_module_path}"
    
    def test_quality_has_required_classes(self, quality_module_path: Path):
        """Module should have required classes."""
        content = quality_module_path.read_text()
        
        required_classes = [
            "DataQualityMonitor",
            "QualityReport",
        ]
        
        for cls in required_classes:
            assert f"class {cls}" in content, f"Missing class: {cls}"


class TestDataQualityMonitor:
    """Test DataQualityMonitor class."""
    
    def test_create_monitor(self):
        """Should create quality monitor."""
        from medai_compass.pipelines.quality import DataQualityMonitor
        
        monitor = DataQualityMonitor()
        assert monitor is not None
    
    def test_monitor_has_expectations(self):
        """Monitor should have data expectations."""
        from medai_compass.pipelines.quality import DataQualityMonitor
        
        monitor = DataQualityMonitor()
        
        # Should have some default expectations
        assert hasattr(monitor, "expectations") or hasattr(monitor, "checks")
    
    def test_run_quality_checks(self, tmp_path):
        """Should run quality checks on dataset."""
        from medai_compass.pipelines.quality import DataQualityMonitor
        
        # Create sample data
        records = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]
        data_path = tmp_path / "quality_test.json"
        with open(data_path, "w") as f:
            json.dump(records, f)
        
        monitor = DataQualityMonitor()
        report = monitor.check(str(data_path))
        
        assert report is not None
        assert hasattr(report, "passed") or hasattr(report, "success")


class TestQualityReport:
    """Test QualityReport dataclass."""
    
    def test_create_report(self):
        """Should create quality report."""
        from medai_compass.pipelines.quality import QualityReport, QualityCheckResult
        
        # Create checks that will result in all passing
        checks = [
            QualityCheckResult(name="check1", passed=True, actual_value=1.0),
            QualityCheckResult(name="check2", passed=True, actual_value=1.0),
        ]
        
        report = QualityReport(
            dataset_name="synthea_v1",
            checks=checks,
        )
        
        assert report.passed is True
        assert report.failed_checks == 0
        assert report.total_checks == 2
    
    def test_report_with_failures(self):
        """Should handle report with failures."""
        from medai_compass.pipelines.quality import QualityReport, QualityCheckResult
        
        # Create checks with some failures
        checks = [
            QualityCheckResult(name="check1", passed=True, actual_value=1.0),
            QualityCheckResult(name="check2", passed=True, actual_value=1.0),
            QualityCheckResult(name="check3", passed=True, actual_value=1.0),
            QualityCheckResult(name="check4", passed=True, actual_value=1.0),
            QualityCheckResult(name="check5", passed=True, actual_value=1.0),
            QualityCheckResult(name="check6", passed=True, actual_value=1.0),
            QualityCheckResult(name="check7", passed=True, actual_value=1.0),
            QualityCheckResult(name="schema_valid", passed=False, actual_value=0.0, message="Missing field"),
            QualityCheckResult(name="completeness", passed=False, actual_value=0.5),
            QualityCheckResult(name="duplicates", passed=False, actual_value=5.0),
        ]
        
        report = QualityReport(
            dataset_name="bad_data",
            checks=checks,
        )
        
        assert report.passed is False
        assert report.failed_checks == 3
        assert report.passed_checks == 7


# =============================================================================
# Quality Check Tests
# =============================================================================

class TestQualityChecks:
    """Test individual quality checks."""
    
    @pytest.fixture
    def sample_records(self) -> List[Dict]:
        """Sample records for quality testing."""
        return [
            {"question": "What is hypertension?", "answer": "High blood pressure."},
            {"question": "What causes diabetes?", "answer": "Insulin resistance."},
            {"question": "Treatment for pneumonia?", "answer": "Antibiotics."},
        ]
    
    def test_check_completeness(self, sample_records, tmp_path):
        """Should check data completeness."""
        from medai_compass.pipelines.quality import DataQualityMonitor
        
        data_path = tmp_path / "complete.json"
        with open(data_path, "w") as f:
            json.dump(sample_records, f)
        
        monitor = DataQualityMonitor()
        report = monitor.check_completeness(str(data_path))
        
        # All records are complete
        assert report.completeness_ratio >= 0.9
    
    def test_check_schema_conformance(self, sample_records, tmp_path):
        """Should check schema conformance."""
        from medai_compass.pipelines.quality import DataQualityMonitor
        
        data_path = tmp_path / "schema_test.json"
        with open(data_path, "w") as f:
            json.dump(sample_records, f)
        
        monitor = DataQualityMonitor()
        report = monitor.check_schema(str(data_path), schema_type="qa")
        
        assert report.schema_valid is True
    
    def test_check_duplicates(self, tmp_path):
        """Should detect duplicate records."""
        from medai_compass.pipelines.quality import DataQualityMonitor
        
        records = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q1", "answer": "A1"},  # Duplicate
            {"question": "Q2", "answer": "A2"},
        ]
        
        data_path = tmp_path / "duplicates.json"
        with open(data_path, "w") as f:
            json.dump(records, f)
        
        monitor = DataQualityMonitor()
        report = monitor.check_duplicates(str(data_path))
        
        assert report.duplicate_count == 1
    
    def test_check_text_quality(self, sample_records, tmp_path):
        """Should check text quality metrics."""
        from medai_compass.pipelines.quality import DataQualityMonitor
        
        data_path = tmp_path / "text_quality.json"
        with open(data_path, "w") as f:
            json.dump(sample_records, f)
        
        monitor = DataQualityMonitor()
        report = monitor.check_text_quality(str(data_path))
        
        assert hasattr(report, "avg_length") or hasattr(report, "text_stats")


# =============================================================================
# Data Drift Detection Tests
# =============================================================================

class TestDataDriftDetection:
    """Test data drift detection."""
    
    @pytest.fixture
    def baseline_data(self, tmp_path) -> Path:
        """Create baseline dataset."""
        records = [
            {"question": f"Medical question {i}?", "answer": f"Answer {i}."}
            for i in range(100)
        ]
        
        path = tmp_path / "baseline.json"
        with open(path, "w") as f:
            json.dump(records, f)
        
        return path
    
    @pytest.fixture
    def current_data_similar(self, tmp_path) -> Path:
        """Create current dataset similar to baseline."""
        records = [
            {"question": f"Medical query {i}?", "answer": f"Response {i}."}
            for i in range(100)
        ]
        
        path = tmp_path / "current_similar.json"
        with open(path, "w") as f:
            json.dump(records, f)
        
        return path
    
    @pytest.fixture
    def current_data_drifted(self, tmp_path) -> Path:
        """Create current dataset with significant drift."""
        # Very different structure
        records = [
            {"text": f"Completely different format {i}"}
            for i in range(100)
        ]
        
        path = tmp_path / "current_drifted.json"
        with open(path, "w") as f:
            json.dump(records, f)
        
        return path
    
    def test_detect_no_drift(self, baseline_data, current_data_similar):
        """Should not detect drift for similar data."""
        from medai_compass.pipelines.quality import DataQualityMonitor
        
        monitor = DataQualityMonitor()
        drift_report = monitor.check_drift(
            baseline_path=str(baseline_data),
            current_path=str(current_data_similar)
        )
        
        assert drift_report.drift_detected is False or drift_report.drift_score < 0.3
    
    def test_detect_schema_drift(self, baseline_data, current_data_drifted):
        """Should detect schema drift."""
        from medai_compass.pipelines.quality import DataQualityMonitor
        
        monitor = DataQualityMonitor()
        drift_report = monitor.check_drift(
            baseline_path=str(baseline_data),
            current_path=str(current_data_drifted)
        )
        
        assert drift_report.drift_detected is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestVersioningQualityIntegration:
    """Test integration between versioning and quality."""
    
    def test_version_includes_quality_report(self, tmp_path):
        """Dataset version should include quality report."""
        from medai_compass.pipelines.versioning import DataVersionManager
        from medai_compass.pipelines.quality import DataQualityMonitor
        
        # Create sample data
        data_dir = tmp_path / "versioned_data"
        data_dir.mkdir()
        records = [{"question": "Q", "answer": "A"}]
        with open(data_dir / "data.json", "w") as f:
            json.dump(records, f)
        
        manager = DataVersionManager(repo_path=str(tmp_path))
        monitor = DataQualityMonitor()
        
        # Run quality check
        report = monitor.check(str(data_dir / "data.json"))
        
        # Track with quality metadata
        version = manager.track(
            str(data_dir),
            name="test_with_quality",
            quality_report=report
        )
        
        assert version.metadata.get("quality_passed") is not None


# =============================================================================
# Pipeline Configuration Tests
# =============================================================================

class TestPipelineConfiguration:
    """Test pipeline configuration file."""
    
    def test_pipeline_config_exists(self):
        """Pipeline config file should exist."""
        config_path = Path(__file__).parent.parent / "config" / "pipeline.yaml"
        assert config_path.exists(), f"Config not found at {config_path}"
    
    def test_config_has_quality_section(self):
        """Config should have quality monitoring section."""
        config_path = Path(__file__).parent.parent / "config" / "pipeline.yaml"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert "quality" in config or "monitoring" in config
    
    def test_config_has_versioning_section(self):
        """Config should have versioning section."""
        config_path = Path(__file__).parent.parent / "config" / "pipeline.yaml"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert "versioning" in config or "dvc" in config

"""
Tests for auto-retraining triggers module (Phase 8: Monitoring & Observability).

TDD tests for retraining triggers based on drift, performance, and scheduling.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio


# ============================================================================
# Test: Retraining Configuration
# ============================================================================

class TestRetrainingConfig:
    """Test retraining trigger configuration."""
    
    def test_retraining_config_defaults(self):
        """Test default retraining configuration."""
        from medai_compass.monitoring.retraining_trigger import RetrainingConfig
        
        config = RetrainingConfig()
        
        assert config.model_name == "medgemma-27b-it"
        assert config.min_samples_for_retrain == 1000
        assert config.drift_threshold == 0.2
        assert config.accuracy_threshold == 0.75
        assert config.max_retrains_per_day == 2
    
    def test_retraining_config_4b_model(self):
        """Test configuration for 4B model."""
        from medai_compass.monitoring.retraining_trigger import RetrainingConfig
        
        config = RetrainingConfig(model_name="medgemma-4b-it")
        
        assert config.model_name == "medgemma-4b-it"


# ============================================================================
# Test: Trigger Types
# ============================================================================

class TestTriggerTypes:
    """Test trigger type definitions."""
    
    def test_trigger_type_enum(self):
        """Test TriggerType enum values."""
        from medai_compass.monitoring.retraining_trigger import TriggerType
        
        assert TriggerType.DRIFT.value == "drift"
        assert TriggerType.ACCURACY.value == "accuracy"
        assert TriggerType.SCHEDULED.value == "scheduled"
        assert TriggerType.MANUAL.value == "manual"
    
    def test_trigger_reason_dataclass(self):
        """Test TriggerReason dataclass."""
        from medai_compass.monitoring.retraining_trigger import TriggerReason, TriggerType
        
        reason = TriggerReason(
            trigger_type=TriggerType.DRIFT,
            description="Input drift detected",
            metric_name="drift_score",
            metric_value=0.25,
            threshold=0.2
        )
        
        assert reason.trigger_type == TriggerType.DRIFT
        assert reason.metric_value > reason.threshold


# ============================================================================
# Test: Drift-Based Trigger
# ============================================================================

class TestDriftBasedTrigger:
    """Test drift-based retraining triggers."""
    
    def test_drift_trigger_initialization(self):
        """Test drift trigger initialization."""
        from medai_compass.monitoring.retraining_trigger import (
            DriftBasedTrigger, RetrainingConfig
        )
        
        config = RetrainingConfig(drift_threshold=0.2)
        trigger = DriftBasedTrigger(config)
        
        assert trigger.threshold == 0.2
    
    def test_drift_trigger_not_triggered(self):
        """Test drift trigger when drift is below threshold."""
        from medai_compass.monitoring.retraining_trigger import (
            DriftBasedTrigger, RetrainingConfig
        )
        
        config = RetrainingConfig(drift_threshold=0.2)
        trigger = DriftBasedTrigger(config)
        
        result = trigger.evaluate(drift_score=0.15)
        
        assert result.should_retrain is False
    
    def test_drift_trigger_activated(self):
        """Test drift trigger when drift exceeds threshold."""
        from medai_compass.monitoring.retraining_trigger import (
            DriftBasedTrigger, RetrainingConfig
        )
        
        config = RetrainingConfig(drift_threshold=0.2)
        trigger = DriftBasedTrigger(config)
        
        result = trigger.evaluate(drift_score=0.25)
        
        assert result.should_retrain is True
        assert result.reason.trigger_type.value == "drift"
    
    def test_drift_trigger_multiple_metrics(self):
        """Test drift trigger with multiple drift metrics."""
        from medai_compass.monitoring.retraining_trigger import (
            DriftBasedTrigger, RetrainingConfig
        )
        
        config = RetrainingConfig(drift_threshold=0.2)
        trigger = DriftBasedTrigger(config)
        
        drift_metrics = {
            "input_drift": 0.15,
            "output_drift": 0.25,  # Exceeds threshold
            "concept_drift": 0.10
        }
        
        result = trigger.evaluate_multiple(drift_metrics)
        
        assert result.should_retrain is True


# ============================================================================
# Test: Accuracy-Based Trigger
# ============================================================================

class TestAccuracyBasedTrigger:
    """Test accuracy-based retraining triggers."""
    
    def test_accuracy_trigger_initialization(self):
        """Test accuracy trigger initialization."""
        from medai_compass.monitoring.retraining_trigger import (
            AccuracyBasedTrigger, RetrainingConfig
        )
        
        config = RetrainingConfig(accuracy_threshold=0.75)
        trigger = AccuracyBasedTrigger(config)
        
        assert trigger.threshold == 0.75
    
    def test_accuracy_trigger_not_triggered(self):
        """Test accuracy trigger when accuracy is above threshold."""
        from medai_compass.monitoring.retraining_trigger import (
            AccuracyBasedTrigger, RetrainingConfig
        )
        
        config = RetrainingConfig(accuracy_threshold=0.75)
        trigger = AccuracyBasedTrigger(config)
        
        result = trigger.evaluate(accuracy=0.82)
        
        assert result.should_retrain is False
    
    def test_accuracy_trigger_activated(self):
        """Test accuracy trigger when accuracy drops below threshold."""
        from medai_compass.monitoring.retraining_trigger import (
            AccuracyBasedTrigger, RetrainingConfig
        )
        
        config = RetrainingConfig(accuracy_threshold=0.75)
        trigger = AccuracyBasedTrigger(config)
        
        result = trigger.evaluate(accuracy=0.70)
        
        assert result.should_retrain is True
        assert result.reason.trigger_type.value == "accuracy"
    
    def test_accuracy_trigger_with_trend(self):
        """Test accuracy trigger considering trend."""
        from medai_compass.monitoring.retraining_trigger import (
            AccuracyBasedTrigger, RetrainingConfig
        )
        
        config = RetrainingConfig(accuracy_threshold=0.75)
        trigger = AccuracyBasedTrigger(config, consider_trend=True)
        
        # Declining accuracy trend
        accuracy_history = [0.85, 0.82, 0.79, 0.76, 0.74]
        
        result = trigger.evaluate_with_history(accuracy_history)
        
        assert result.should_retrain is True


# ============================================================================
# Test: Scheduled Trigger
# ============================================================================

class TestScheduledTrigger:
    """Test scheduled retraining triggers."""
    
    def test_scheduled_trigger_initialization(self):
        """Test scheduled trigger initialization."""
        from medai_compass.monitoring.retraining_trigger import (
            ScheduledTrigger, RetrainingConfig
        )
        
        config = RetrainingConfig()
        trigger = ScheduledTrigger(config, schedule_days=7)
        
        assert trigger.schedule_days == 7
    
    def test_scheduled_trigger_not_due(self):
        """Test scheduled trigger when not due."""
        from medai_compass.monitoring.retraining_trigger import (
            ScheduledTrigger, RetrainingConfig
        )
        
        config = RetrainingConfig()
        trigger = ScheduledTrigger(config, schedule_days=7)
        
        # Last retrain was 3 days ago
        trigger.last_retrain = datetime.now() - timedelta(days=3)
        
        result = trigger.evaluate()
        
        assert result.should_retrain is False
    
    def test_scheduled_trigger_due(self):
        """Test scheduled trigger when due."""
        from medai_compass.monitoring.retraining_trigger import (
            ScheduledTrigger, RetrainingConfig
        )
        
        config = RetrainingConfig()
        trigger = ScheduledTrigger(config, schedule_days=7)
        
        # Last retrain was 8 days ago
        trigger.last_retrain = datetime.now() - timedelta(days=8)
        
        result = trigger.evaluate()
        
        assert result.should_retrain is True
        assert result.reason.trigger_type.value == "scheduled"


# ============================================================================
# Test: Data Volume Trigger
# ============================================================================

class TestDataVolumeTrigger:
    """Test data volume based triggers."""
    
    def test_data_volume_trigger_initialization(self):
        """Test data volume trigger initialization."""
        from medai_compass.monitoring.retraining_trigger import (
            DataVolumeTrigger, RetrainingConfig
        )
        
        config = RetrainingConfig(min_samples_for_retrain=1000)
        trigger = DataVolumeTrigger(config)
        
        assert trigger.min_samples == 1000
    
    def test_data_volume_trigger_not_enough_data(self):
        """Test data volume trigger when not enough data."""
        from medai_compass.monitoring.retraining_trigger import (
            DataVolumeTrigger, RetrainingConfig
        )
        
        config = RetrainingConfig(min_samples_for_retrain=1000)
        trigger = DataVolumeTrigger(config)
        
        result = trigger.evaluate(new_samples=500)
        
        assert result.should_retrain is False
    
    def test_data_volume_trigger_sufficient_data(self):
        """Test data volume trigger when sufficient data."""
        from medai_compass.monitoring.retraining_trigger import (
            DataVolumeTrigger, RetrainingConfig
        )
        
        config = RetrainingConfig(min_samples_for_retrain=1000)
        trigger = DataVolumeTrigger(config)
        
        result = trigger.evaluate(new_samples=1500)
        
        assert result.should_retrain is True


# ============================================================================
# Test: Composite Trigger
# ============================================================================

class TestCompositeTrigger:
    """Test composite triggers combining multiple conditions."""
    
    def test_composite_trigger_any(self):
        """Test composite trigger with ANY logic."""
        from medai_compass.monitoring.retraining_trigger import (
            CompositeTrigger, DriftBasedTrigger, AccuracyBasedTrigger,
            RetrainingConfig
        )
        
        config = RetrainingConfig()
        drift_trigger = DriftBasedTrigger(config)
        accuracy_trigger = AccuracyBasedTrigger(config)
        
        composite = CompositeTrigger(
            triggers=[drift_trigger, accuracy_trigger],
            mode="any"
        )
        
        # Only drift exceeds threshold
        metrics = {
            "drift_score": 0.25,  # Exceeds 0.2 threshold
            "accuracy": 0.80     # Above 0.75 threshold
        }
        
        result = composite.evaluate(metrics)
        
        assert result.should_retrain is True
    
    def test_composite_trigger_all(self):
        """Test composite trigger with ALL logic."""
        from medai_compass.monitoring.retraining_trigger import (
            CompositeTrigger, DriftBasedTrigger, AccuracyBasedTrigger,
            RetrainingConfig
        )
        
        config = RetrainingConfig()
        drift_trigger = DriftBasedTrigger(config)
        accuracy_trigger = AccuracyBasedTrigger(config)
        
        composite = CompositeTrigger(
            triggers=[drift_trigger, accuracy_trigger],
            mode="all"
        )
        
        # Only drift exceeds threshold
        metrics = {
            "drift_score": 0.25,
            "accuracy": 0.80
        }
        
        result = composite.evaluate(metrics)
        
        # Should NOT trigger because accuracy is still good
        assert result.should_retrain is False


# ============================================================================
# Test: Retraining Manager
# ============================================================================

class TestRetrainingManager:
    """Test unified retraining management."""
    
    def test_retraining_manager_initialization(self):
        """Test retraining manager initialization."""
        from medai_compass.monitoring.retraining_trigger import (
            RetrainingManager, RetrainingConfig
        )
        
        config = RetrainingConfig(model_name="medgemma-27b-it")
        manager = RetrainingManager(config)
        
        assert manager.drift_trigger is not None
        assert manager.accuracy_trigger is not None
    
    def test_add_custom_trigger(self):
        """Test adding custom triggers."""
        from medai_compass.monitoring.retraining_trigger import (
            RetrainingManager, RetrainingConfig, ScheduledTrigger
        )
        
        config = RetrainingConfig()
        manager = RetrainingManager(config)
        
        scheduled = ScheduledTrigger(config, schedule_days=14)
        manager.add_trigger(scheduled)
        
        assert len(manager.triggers) >= 3
    
    def test_evaluate_all_triggers(self):
        """Test evaluating all triggers."""
        from medai_compass.monitoring.retraining_trigger import (
            RetrainingManager, RetrainingConfig
        )
        
        config = RetrainingConfig()
        manager = RetrainingManager(config)
        
        metrics = {
            "drift_score": 0.15,
            "accuracy": 0.80
        }
        
        result = manager.evaluate(metrics)
        
        assert "should_retrain" in result
        assert "reasons" in result
    
    def test_rate_limiting(self):
        """Test rate limiting of retraining triggers."""
        from medai_compass.monitoring.retraining_trigger import (
            RetrainingManager, RetrainingConfig
        )
        
        config = RetrainingConfig(max_retrains_per_day=2)
        manager = RetrainingManager(config)
        
        # Simulate multiple triggers in one day
        metrics = {"drift_score": 0.30, "accuracy": 0.60}
        
        # First two should work
        result1 = manager.evaluate(metrics)
        manager.record_retrain()
        
        result2 = manager.evaluate(metrics)
        manager.record_retrain()
        
        # Third should be rate limited
        result3 = manager.evaluate(metrics)
        
        assert result3["rate_limited"] is True


# ============================================================================
# Test: Retraining Actions
# ============================================================================

class TestRetrainingActions:
    """Test retraining action execution."""
    
    @pytest.mark.asyncio
    async def test_trigger_retraining_pipeline(self):
        """Test triggering the retraining pipeline."""
        from medai_compass.monitoring.retraining_trigger import (
            RetrainingManager, RetrainingConfig
        )
        
        config = RetrainingConfig()
        manager = RetrainingManager(config)
        
        with patch.object(manager, '_start_retraining_job') as mock_job:
            mock_job.return_value = {"job_id": "retrain-001", "status": "started"}
            
            result = await manager.trigger_retrain()
            
            assert result["status"] == "started"
    
    def test_generate_retraining_config(self):
        """Test generating retraining configuration."""
        from medai_compass.monitoring.retraining_trigger import (
            RetrainingManager, RetrainingConfig, TriggerReason, TriggerType
        )
        
        config = RetrainingConfig(model_name="medgemma-27b-it")
        manager = RetrainingManager(config)
        
        reason = TriggerReason(
            trigger_type=TriggerType.DRIFT,
            description="Input drift detected",
            metric_name="drift_score",
            metric_value=0.25,
            threshold=0.2
        )
        
        retrain_config = manager.generate_retrain_config(reason)
        
        assert retrain_config["model_name"] == "medgemma-27b-it"
        assert "trigger_reason" in retrain_config


# ============================================================================
# Test: Retraining History
# ============================================================================

class TestRetrainingHistory:
    """Test retraining history tracking."""
    
    def test_record_retraining_event(self):
        """Test recording retraining events."""
        from medai_compass.monitoring.retraining_trigger import RetrainingHistory
        
        history = RetrainingHistory()
        
        history.record(
            model_name="medgemma-27b-it",
            trigger_type="drift",
            success=True
        )
        
        assert len(history.events) == 1
    
    def test_get_retraining_stats(self):
        """Test getting retraining statistics."""
        from medai_compass.monitoring.retraining_trigger import RetrainingHistory
        
        history = RetrainingHistory()
        
        for i in range(5):
            history.record(
                model_name="medgemma-27b-it",
                trigger_type="drift" if i % 2 == 0 else "accuracy",
                success=i < 4  # 4 successes, 1 failure
            )
        
        stats = history.get_stats()
        
        assert stats["total_retrains"] == 5
        assert stats["success_rate"] == 0.8
    
    def test_get_retrains_in_period(self):
        """Test getting retrains in time period."""
        from medai_compass.monitoring.retraining_trigger import RetrainingHistory
        
        history = RetrainingHistory()
        
        for i in range(3):
            history.record(
                model_name="medgemma-27b-it",
                trigger_type="drift",
                success=True
            )
        
        today_count = history.get_count_today()
        
        assert today_count == 3


# ============================================================================
# Test: Model-Specific Retraining Thresholds
# ============================================================================

class TestModelSpecificRetrainingThresholds:
    """Test model-specific retraining thresholds."""
    
    def test_27b_model_thresholds(self):
        """Test retraining thresholds for 27B model."""
        from medai_compass.monitoring.retraining_trigger import get_model_retrain_config
        
        config = get_model_retrain_config("medgemma-27b-it")
        
        assert config.drift_threshold == 0.2
        assert config.accuracy_threshold == 0.75  # MedQA threshold
    
    def test_4b_model_thresholds(self):
        """Test retraining thresholds for 4B model."""
        from medai_compass.monitoring.retraining_trigger import get_model_retrain_config
        
        config = get_model_retrain_config("medgemma-4b-it")
        
        # Both models should have same quality thresholds
        assert config.accuracy_threshold == 0.75


# ============================================================================
# Test: Integration with Deployment Pipeline
# ============================================================================

class TestDeploymentPipelineIntegration:
    """Test integration with deployment pipeline."""
    
    @pytest.mark.asyncio
    async def test_retrain_triggers_deployment(self):
        """Test that successful retrain triggers deployment."""
        from medai_compass.monitoring.retraining_trigger import (
            RetrainingManager, RetrainingConfig
        )
        
        config = RetrainingConfig()
        manager = RetrainingManager(config)
        
        with patch.object(manager, '_deploy_new_model') as mock_deploy:
            mock_deploy.return_value = {"status": "deployed", "version": "1.0.1"}
            
            # Simulate successful retrain
            result = await manager.retrain_and_deploy()
            
            mock_deploy.assert_called_once()
    
    def test_retrain_rollback_on_failure(self):
        """Test rollback configuration on retrain failure."""
        from medai_compass.monitoring.retraining_trigger import (
            RetrainingManager, RetrainingConfig
        )
        
        config = RetrainingConfig()
        manager = RetrainingManager(config)
        
        rollback_config = manager.get_rollback_config()
        
        assert "previous_version" in rollback_config
        assert rollback_config["auto_rollback"] is True

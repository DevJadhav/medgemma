"""
Tests for alerting module (Phase 8: Monitoring & Observability).

TDD tests for alert rules, notification channels, and alert management.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio


# ============================================================================
# Test: Alert Severity
# ============================================================================

class TestAlertSeverity:
    """Test alert severity levels."""
    
    def test_alert_severity_enum(self):
        """Test AlertSeverity enum values."""
        from medai_compass.monitoring.alerting import AlertSeverity
        
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"
        assert AlertSeverity.EMERGENCY.value == "emergency"
    
    def test_severity_ordering(self):
        """Test severity ordering for comparison."""
        from medai_compass.monitoring.alerting import AlertSeverity
        
        # Higher severity should have higher priority value
        assert AlertSeverity.EMERGENCY.priority > AlertSeverity.CRITICAL.priority
        assert AlertSeverity.CRITICAL.priority > AlertSeverity.WARNING.priority
        assert AlertSeverity.WARNING.priority > AlertSeverity.INFO.priority


# ============================================================================
# Test: Alert Configuration
# ============================================================================

class TestAlertConfig:
    """Test alert configuration."""
    
    def test_alert_config_defaults(self):
        """Test default alert configuration."""
        from medai_compass.monitoring.alerting import AlertConfig
        
        config = AlertConfig()
        
        assert config.model_name == "medgemma-27b-it"
        assert config.cooldown_seconds == 300
        assert config.max_alerts_per_hour == 10
    
    def test_alert_config_thresholds(self):
        """Test alert thresholds configuration."""
        from medai_compass.monitoring.alerting import AlertConfig
        
        config = AlertConfig(
            latency_warning_ms=400,
            latency_critical_ms=800,
            drift_warning_score=0.1,
            drift_critical_score=0.2
        )
        
        assert config.latency_warning_ms == 400
        assert config.latency_critical_ms == 800
        assert config.drift_warning_score == 0.1
        assert config.drift_critical_score == 0.2


# ============================================================================
# Test: Alert Definition
# ============================================================================

class TestAlertDefinition:
    """Test alert definition dataclass."""
    
    def test_alert_creation(self):
        """Test creating an alert."""
        from medai_compass.monitoring.alerting import Alert, AlertSeverity
        
        alert = Alert(
            id="alert-001",
            name="High Latency",
            severity=AlertSeverity.WARNING,
            message="P95 latency exceeded threshold",
            metric_name="latency_p95_ms",
            metric_value=550.0,
            threshold=500.0
        )
        
        assert alert.id == "alert-001"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.metric_value > alert.threshold
    
    def test_alert_to_dict(self):
        """Test alert serialization."""
        from medai_compass.monitoring.alerting import Alert, AlertSeverity
        
        alert = Alert(
            id="alert-001",
            name="High Latency",
            severity=AlertSeverity.WARNING,
            message="P95 latency exceeded",
            metric_name="latency_p95_ms",
            metric_value=550.0,
            threshold=500.0
        )
        
        alert_dict = alert.to_dict()
        
        assert alert_dict["id"] == "alert-001"
        assert alert_dict["severity"] == "warning"
        assert "timestamp" in alert_dict


# ============================================================================
# Test: Alert Rules
# ============================================================================

class TestAlertRules:
    """Test alert rule definitions and evaluation."""
    
    def test_latency_alert_rule(self):
        """Test latency-based alert rule."""
        from medai_compass.monitoring.alerting import LatencyAlertRule, AlertSeverity
        
        rule = LatencyAlertRule(
            name="p95_latency",
            warning_threshold=400,
            critical_threshold=800
        )
        
        # Below warning
        result = rule.evaluate(350)
        assert result is None
        
        # Warning level
        result = rule.evaluate(500)
        assert result.severity == AlertSeverity.WARNING
        
        # Critical level
        result = rule.evaluate(900)
        assert result.severity == AlertSeverity.CRITICAL
    
    def test_drift_alert_rule(self):
        """Test drift-based alert rule."""
        from medai_compass.monitoring.alerting import DriftAlertRule, AlertSeverity
        
        rule = DriftAlertRule(
            name="input_drift",
            warning_threshold=0.1,
            critical_threshold=0.2
        )
        
        # No drift
        result = rule.evaluate(0.05)
        assert result is None
        
        # Warning drift
        result = rule.evaluate(0.15)
        assert result.severity == AlertSeverity.WARNING
        
        # Critical drift
        result = rule.evaluate(0.25)
        assert result.severity == AlertSeverity.CRITICAL
    
    def test_accuracy_alert_rule(self):
        """Test accuracy-based alert rule."""
        from medai_compass.monitoring.alerting import AccuracyAlertRule, AlertSeverity
        
        rule = AccuracyAlertRule(
            name="model_accuracy",
            warning_threshold=0.75,  # MedQA threshold
            critical_threshold=0.70
        )
        
        # Good accuracy
        result = rule.evaluate(0.85)
        assert result is None
        
        # Warning level
        result = rule.evaluate(0.73)
        assert result.severity == AlertSeverity.WARNING
        
        # Critical level
        result = rule.evaluate(0.68)
        assert result.severity == AlertSeverity.CRITICAL
    
    def test_safety_alert_rule(self):
        """Test safety-based alert rule (always critical)."""
        from medai_compass.monitoring.alerting import SafetyAlertRule, AlertSeverity
        
        rule = SafetyAlertRule(
            name="safety_score",
            threshold=0.99  # 99% safety threshold
        )
        
        # Safe
        result = rule.evaluate(0.995)
        assert result is None
        
        # Safety violation - always emergency
        result = rule.evaluate(0.98)
        assert result.severity == AlertSeverity.EMERGENCY


# ============================================================================
# Test: Notification Channels (Base Interface)
# ============================================================================

class TestNotificationChannels:
    """Test notification channel interface and implementations."""
    
    def test_notification_channel_interface(self):
        """Test base NotificationChannel interface."""
        from medai_compass.monitoring.alerting import NotificationChannel
        
        class TestChannel(NotificationChannel):
            async def send(self, alert):
                return True
        
        channel = TestChannel()
        assert hasattr(channel, 'send')
    
    def test_mock_notification_channel(self):
        """Test mock notification channel for testing."""
        from medai_compass.monitoring.alerting import MockNotificationChannel, Alert, AlertSeverity
        
        channel = MockNotificationChannel()
        
        alert = Alert(
            id="test-001",
            name="Test Alert",
            severity=AlertSeverity.WARNING,
            message="Test message",
            metric_name="test",
            metric_value=1.0,
            threshold=0.5
        )
        
        # Should always succeed
        result = asyncio.run(channel.send(alert))
        assert result is True
        assert len(channel.sent_alerts) == 1
    
    def test_log_notification_channel(self):
        """Test logging notification channel."""
        from medai_compass.monitoring.alerting import LogNotificationChannel, Alert, AlertSeverity
        
        channel = LogNotificationChannel()
        
        alert = Alert(
            id="test-001",
            name="Test Alert",
            severity=AlertSeverity.CRITICAL,
            message="Critical issue",
            metric_name="latency",
            metric_value=1000.0,
            threshold=500.0
        )
        
        with patch('logging.Logger.critical') as mock_log:
            result = asyncio.run(channel.send(alert))
            assert result is True


# ============================================================================
# Test: Optional Notification Channels (Env Var Configured)
# ============================================================================

class TestOptionalNotificationChannels:
    """Test optional notification channels configured via env vars."""
    
    @patch.dict('os.environ', {'SLACK_WEBHOOK_URL': 'https://hooks.slack.com/test'})
    def test_slack_channel_configured(self):
        """Test Slack channel when configured."""
        from medai_compass.monitoring.alerting import SlackNotificationChannel
        
        channel = SlackNotificationChannel()
        
        assert channel.is_configured() is True
        assert channel.webhook_url == 'https://hooks.slack.com/test'
    
    def test_slack_channel_not_configured(self):
        """Test Slack channel when not configured."""
        from medai_compass.monitoring.alerting import SlackNotificationChannel
        
        with patch.dict('os.environ', {}, clear=True):
            channel = SlackNotificationChannel()
            
            assert channel.is_configured() is False
    
    @patch.dict('os.environ', {'PAGERDUTY_ROUTING_KEY': 'test-key'})
    def test_pagerduty_channel_configured(self):
        """Test PagerDuty channel when configured."""
        from medai_compass.monitoring.alerting import PagerDutyNotificationChannel
        
        channel = PagerDutyNotificationChannel()
        
        assert channel.is_configured() is True
    
    def test_pagerduty_channel_not_configured(self):
        """Test PagerDuty channel when not configured."""
        from medai_compass.monitoring.alerting import PagerDutyNotificationChannel
        
        with patch.dict('os.environ', {}, clear=True):
            channel = PagerDutyNotificationChannel()
            
            assert channel.is_configured() is False


# ============================================================================
# Test: Alert Manager
# ============================================================================

class TestAlertManager:
    """Test alert manager functionality."""
    
    def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        from medai_compass.monitoring.alerting import AlertManager, AlertConfig
        
        config = AlertConfig()
        manager = AlertManager(config)
        
        assert manager.rules is not None
        assert manager.channels is not None
    
    def test_add_rule(self):
        """Test adding alert rules."""
        from medai_compass.monitoring.alerting import (
            AlertManager, AlertConfig, LatencyAlertRule
        )
        
        config = AlertConfig()
        manager = AlertManager(config)
        
        rule = LatencyAlertRule(
            name="p95_latency",
            warning_threshold=400,
            critical_threshold=800
        )
        
        manager.add_rule(rule)
        
        assert len(manager.rules) == 1
    
    def test_add_channel(self):
        """Test adding notification channels."""
        from medai_compass.monitoring.alerting import (
            AlertManager, AlertConfig, MockNotificationChannel
        )
        
        config = AlertConfig()
        manager = AlertManager(config)
        
        channel = MockNotificationChannel()
        manager.add_channel(channel)
        
        assert len(manager.channels) == 1
    
    def test_evaluate_metrics(self):
        """Test evaluating metrics against rules."""
        from medai_compass.monitoring.alerting import (
            AlertManager, AlertConfig, LatencyAlertRule
        )
        
        config = AlertConfig()
        manager = AlertManager(config)
        
        manager.add_rule(LatencyAlertRule(
            name="p95_latency",
            warning_threshold=400,
            critical_threshold=800
        ))
        
        metrics = {"p95_latency": 550}
        alerts = manager.evaluate(metrics)
        
        assert len(alerts) == 1
    
    @pytest.mark.asyncio
    async def test_send_alerts(self):
        """Test sending alerts through channels."""
        from medai_compass.monitoring.alerting import (
            AlertManager, AlertConfig, LatencyAlertRule,
            MockNotificationChannel
        )
        
        config = AlertConfig()
        manager = AlertManager(config)
        
        manager.add_rule(LatencyAlertRule(
            name="p95_latency",
            warning_threshold=400,
            critical_threshold=800
        ))
        
        channel = MockNotificationChannel()
        manager.add_channel(channel)
        
        metrics = {"p95_latency": 550}
        await manager.check_and_alert(metrics)
        
        assert len(channel.sent_alerts) == 1


# ============================================================================
# Test: Alert Cooldown
# ============================================================================

class TestAlertCooldown:
    """Test alert cooldown functionality."""
    
    def test_cooldown_prevents_duplicates(self):
        """Test cooldown prevents duplicate alerts."""
        from medai_compass.monitoring.alerting import (
            AlertManager, AlertConfig, LatencyAlertRule
        )
        
        config = AlertConfig(cooldown_seconds=300)
        manager = AlertManager(config)
        
        manager.add_rule(LatencyAlertRule(
            name="p95_latency",
            warning_threshold=400,
            critical_threshold=800
        ))
        
        metrics = {"p95_latency": 550}
        
        # First evaluation should trigger
        alerts1 = manager.evaluate(metrics)
        assert len(alerts1) == 1
        
        # Second evaluation should be suppressed
        alerts2 = manager.evaluate(metrics)
        assert len(alerts2) == 0
    
    def test_cooldown_expires(self):
        """Test cooldown expiration allows new alerts."""
        from medai_compass.monitoring.alerting import (
            AlertManager, AlertConfig, LatencyAlertRule
        )
        
        config = AlertConfig(cooldown_seconds=1)
        manager = AlertManager(config)
        
        manager.add_rule(LatencyAlertRule(
            name="p95_latency",
            warning_threshold=400,
            critical_threshold=800
        ))
        
        metrics = {"p95_latency": 550}
        
        # First evaluation
        alerts1 = manager.evaluate(metrics)
        assert len(alerts1) == 1
        
        # Wait for cooldown
        import time
        time.sleep(1.5)
        
        # Should trigger again
        alerts2 = manager.evaluate(metrics)
        assert len(alerts2) == 1


# ============================================================================
# Test: Alert History
# ============================================================================

class TestAlertHistory:
    """Test alert history tracking."""
    
    def test_record_alert(self):
        """Test recording alerts in history."""
        from medai_compass.monitoring.alerting import AlertHistory, Alert, AlertSeverity
        
        history = AlertHistory()
        
        alert = Alert(
            id="test-001",
            name="Test Alert",
            severity=AlertSeverity.WARNING,
            message="Test",
            metric_name="test",
            metric_value=1.0,
            threshold=0.5
        )
        
        history.record(alert)
        
        assert len(history.alerts) == 1
    
    def test_get_alerts_by_severity(self):
        """Test filtering alerts by severity."""
        from medai_compass.monitoring.alerting import AlertHistory, Alert, AlertSeverity
        
        history = AlertHistory()
        
        for i, severity in enumerate([AlertSeverity.INFO, AlertSeverity.WARNING, 
                                       AlertSeverity.CRITICAL, AlertSeverity.WARNING]):
            alert = Alert(
                id=f"test-{i}",
                name="Test",
                severity=severity,
                message="Test",
                metric_name="test",
                metric_value=1.0,
                threshold=0.5
            )
            history.record(alert)
        
        warnings = history.get_by_severity(AlertSeverity.WARNING)
        
        assert len(warnings) == 2
    
    def test_get_alert_count_by_time(self):
        """Test getting alert count by time window."""
        from medai_compass.monitoring.alerting import AlertHistory, Alert, AlertSeverity
        
        history = AlertHistory()
        
        for i in range(5):
            alert = Alert(
                id=f"test-{i}",
                name="Test",
                severity=AlertSeverity.WARNING,
                message="Test",
                metric_name="test",
                metric_value=1.0,
                threshold=0.5
            )
            history.record(alert)
        
        count = history.get_count_in_window(hours=1)
        
        assert count == 5


# ============================================================================
# Test: Model-Specific Alerts
# ============================================================================

class TestModelSpecificAlerts:
    """Test model-specific alert configurations."""
    
    def test_27b_model_thresholds(self):
        """Test alert thresholds for 27B model."""
        from medai_compass.monitoring.alerting import get_model_alert_config
        
        config = get_model_alert_config("medgemma-27b-it")
        
        assert config.latency_warning_ms == 400
        assert config.latency_critical_ms == 800
    
    def test_4b_model_thresholds(self):
        """Test alert thresholds for 4B model (tighter)."""
        from medai_compass.monitoring.alerting import get_model_alert_config
        
        config_4b = get_model_alert_config("medgemma-4b-it")
        config_27b = get_model_alert_config("medgemma-27b-it")
        
        # 4B should have stricter latency thresholds
        assert config_4b.latency_warning_ms < config_27b.latency_warning_ms


# ============================================================================
# Test: Prometheus Alert Integration
# ============================================================================

class TestPrometheusAlertIntegration:
    """Test integration with existing Prometheus alerts."""
    
    def test_export_prometheus_rules(self):
        """Test exporting alert rules to Prometheus format."""
        from medai_compass.monitoring.alerting import (
            AlertManager, AlertConfig, LatencyAlertRule,
            export_prometheus_rules
        )
        
        config = AlertConfig()
        manager = AlertManager(config)
        
        manager.add_rule(LatencyAlertRule(
            name="p95_latency",
            warning_threshold=400,
            critical_threshold=800
        ))
        
        prometheus_rules = export_prometheus_rules(manager)
        
        assert "groups" in prometheus_rules
        assert len(prometheus_rules["groups"]) > 0
    
    def test_sync_with_existing_rules(self):
        """Test syncing with existing Prometheus alert rules."""
        from medai_compass.monitoring.alerting import sync_prometheus_rules
        
        existing_rules_path = "docker/prometheus/alert.rules.yml"
        
        # Should not raise
        sync_result = sync_prometheus_rules(existing_rules_path)
        
        assert "synced" in sync_result or "error" in sync_result

"""
Alerting Module (Phase 8: Monitoring & Observability).

Provides alerting capabilities including alert rules, notification channels,
and alert management with cooldown and rate limiting.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


# ============================================================================
# Alert Severity
# ============================================================================

class AlertSeverity(Enum):
    """Alert severity levels with priority ordering."""
    
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    
    @property
    def priority(self) -> int:
        """Get priority value for comparison."""
        priorities = {
            "info": 1,
            "warning": 2,
            "critical": 3,
            "emergency": 4,
        }
        return priorities[self.value]


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AlertConfig:
    """Configuration for alerting system.
    
    Attributes:
        model_name: Name of the model being monitored
        cooldown_seconds: Minimum time between duplicate alerts
        max_alerts_per_hour: Rate limit for alerts
        latency_warning_ms: Latency threshold for warning
        latency_critical_ms: Latency threshold for critical
        drift_warning_score: Drift score threshold for warning
        drift_critical_score: Drift score threshold for critical
    """
    
    model_name: str = "medgemma-27b-it"
    cooldown_seconds: int = 300
    max_alerts_per_hour: int = 10
    latency_warning_ms: float = 400
    latency_critical_ms: float = 800
    drift_warning_score: float = 0.1
    drift_critical_score: float = 0.2
    accuracy_warning: float = 0.75
    accuracy_critical: float = 0.70
    safety_threshold: float = 0.99


def get_model_alert_config(model_name: str) -> AlertConfig:
    """Get alert configuration for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        AlertConfig for the model
    """
    configs = {
        "medgemma-27b-it": AlertConfig(
            model_name="medgemma-27b-it",
            latency_warning_ms=400,
            latency_critical_ms=800,
        ),
        "medgemma-4b-it": AlertConfig(
            model_name="medgemma-4b-it",
            latency_warning_ms=200,  # Tighter thresholds for faster model
            latency_critical_ms=400,
        ),
    }
    
    return configs.get(model_name, AlertConfig(model_name=model_name))


# ============================================================================
# Alert Definition
# ============================================================================

@dataclass
class Alert:
    """Represents an alert.
    
    Attributes:
        id: Unique alert identifier
        name: Alert name
        severity: Alert severity level
        message: Alert message
        metric_name: Name of the metric that triggered the alert
        metric_value: Current metric value
        threshold: Threshold that was exceeded
        timestamp: When the alert was created
        details: Additional alert details
    """
    
    id: str
    name: str
    severity: AlertSeverity
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary.
        
        Returns:
            Alert as dictionary
        """
        return {
            "id": self.id,
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


# ============================================================================
# Alert Rules
# ============================================================================

class AlertRule(ABC):
    """Base class for alert rules."""
    
    @abstractmethod
    def evaluate(self, value: float) -> Optional[Alert]:
        """Evaluate the rule against a value.
        
        Args:
            value: Metric value to evaluate
            
        Returns:
            Alert if triggered, None otherwise
        """
        pass


class LatencyAlertRule(AlertRule):
    """Alert rule for latency metrics."""
    
    def __init__(
        self,
        name: str,
        warning_threshold: float,
        critical_threshold: float,
    ):
        """Initialize latency alert rule.
        
        Args:
            name: Rule name
            warning_threshold: Threshold for warning alerts
            critical_threshold: Threshold for critical alerts
        """
        self.name = name
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
    
    def evaluate(self, value: float) -> Optional[Alert]:
        """Evaluate latency against thresholds.
        
        Args:
            value: Latency in milliseconds
            
        Returns:
            Alert if triggered
        """
        if value >= self.critical_threshold:
            return Alert(
                id=str(uuid.uuid4()),
                name=f"High {self.name}",
                severity=AlertSeverity.CRITICAL,
                message=f"{self.name} of {value:.1f}ms exceeds critical threshold",
                metric_name=self.name,
                metric_value=value,
                threshold=self.critical_threshold,
            )
        elif value >= self.warning_threshold:
            return Alert(
                id=str(uuid.uuid4()),
                name=f"Elevated {self.name}",
                severity=AlertSeverity.WARNING,
                message=f"{self.name} of {value:.1f}ms exceeds warning threshold",
                metric_name=self.name,
                metric_value=value,
                threshold=self.warning_threshold,
            )
        
        return None


class DriftAlertRule(AlertRule):
    """Alert rule for drift detection."""
    
    def __init__(
        self,
        name: str,
        warning_threshold: float,
        critical_threshold: float,
    ):
        """Initialize drift alert rule.
        
        Args:
            name: Rule name
            warning_threshold: Threshold for warning alerts
            critical_threshold: Threshold for critical alerts
        """
        self.name = name
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
    
    def evaluate(self, value: float) -> Optional[Alert]:
        """Evaluate drift score against thresholds.
        
        Args:
            value: Drift score
            
        Returns:
            Alert if triggered
        """
        if value >= self.critical_threshold:
            return Alert(
                id=str(uuid.uuid4()),
                name=f"Critical {self.name}",
                severity=AlertSeverity.CRITICAL,
                message=f"{self.name} score of {value:.3f} exceeds critical threshold",
                metric_name=self.name,
                metric_value=value,
                threshold=self.critical_threshold,
            )
        elif value >= self.warning_threshold:
            return Alert(
                id=str(uuid.uuid4()),
                name=f"{self.name} Warning",
                severity=AlertSeverity.WARNING,
                message=f"{self.name} score of {value:.3f} exceeds warning threshold",
                metric_name=self.name,
                metric_value=value,
                threshold=self.warning_threshold,
            )
        
        return None


class AccuracyAlertRule(AlertRule):
    """Alert rule for accuracy metrics (inverted - alerts when BELOW threshold)."""
    
    def __init__(
        self,
        name: str,
        warning_threshold: float,
        critical_threshold: float,
    ):
        """Initialize accuracy alert rule.
        
        Args:
            name: Rule name
            warning_threshold: Threshold for warning (alerts below this)
            critical_threshold: Threshold for critical (alerts below this)
        """
        self.name = name
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
    
    def evaluate(self, value: float) -> Optional[Alert]:
        """Evaluate accuracy against thresholds.
        
        Args:
            value: Accuracy score
            
        Returns:
            Alert if triggered
        """
        if value < self.critical_threshold:
            return Alert(
                id=str(uuid.uuid4()),
                name=f"Critical {self.name} Drop",
                severity=AlertSeverity.CRITICAL,
                message=f"{self.name} of {value:.3f} below critical threshold",
                metric_name=self.name,
                metric_value=value,
                threshold=self.critical_threshold,
            )
        elif value < self.warning_threshold:
            return Alert(
                id=str(uuid.uuid4()),
                name=f"Low {self.name}",
                severity=AlertSeverity.WARNING,
                message=f"{self.name} of {value:.3f} below warning threshold",
                metric_name=self.name,
                metric_value=value,
                threshold=self.warning_threshold,
            )
        
        return None


class SafetyAlertRule(AlertRule):
    """Alert rule for safety metrics (always emergency severity)."""
    
    def __init__(self, name: str, threshold: float):
        """Initialize safety alert rule.
        
        Args:
            name: Rule name
            threshold: Safety threshold (alerts below this)
        """
        self.name = name
        self.threshold = threshold
    
    def evaluate(self, value: float) -> Optional[Alert]:
        """Evaluate safety score against threshold.
        
        Args:
            value: Safety score
            
        Returns:
            Alert if triggered (always emergency)
        """
        if value < self.threshold:
            return Alert(
                id=str(uuid.uuid4()),
                name=f"Safety Violation - {self.name}",
                severity=AlertSeverity.EMERGENCY,
                message=f"Safety score {value:.4f} below threshold {self.threshold}",
                metric_name=self.name,
                metric_value=value,
                threshold=self.threshold,
            )
        
        return None


# ============================================================================
# Notification Channels
# ============================================================================

class NotificationChannel(ABC):
    """Base class for notification channels."""
    
    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Send an alert through this channel.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if successful
        """
        pass


class MockNotificationChannel(NotificationChannel):
    """Mock notification channel for testing."""
    
    def __init__(self):
        """Initialize mock channel."""
        self.sent_alerts: list[Alert] = []
    
    async def send(self, alert: Alert) -> bool:
        """Record alert in sent list.
        
        Args:
            alert: Alert to send
            
        Returns:
            Always True
        """
        self.sent_alerts.append(alert)
        return True


class LogNotificationChannel(NotificationChannel):
    """Notification channel that logs alerts."""
    
    def __init__(self):
        """Initialize log channel."""
        self.logger = logging.getLogger("alerts")
    
    async def send(self, alert: Alert) -> bool:
        """Log the alert.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if successful
        """
        log_method = getattr(self.logger, alert.severity.value, self.logger.warning)
        log_method(
            f"[{alert.severity.value.upper()}] {alert.name}: {alert.message}"
        )
        return True


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel (configured via env var)."""
    
    def __init__(self):
        """Initialize Slack channel."""
        self.webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    
    def is_configured(self) -> bool:
        """Check if Slack is configured.
        
        Returns:
            True if configured
        """
        return bool(self.webhook_url)
    
    async def send(self, alert: Alert) -> bool:
        """Send alert to Slack.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if successful
        """
        if not self.is_configured():
            logger.warning("Slack not configured, skipping notification")
            return False
        
        try:
            from slack_sdk.webhook.async_client import AsyncWebhookClient
            
            client = AsyncWebhookClient(self.webhook_url)
            
            # Format message
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ffcc00",
                AlertSeverity.CRITICAL: "#ff6600",
                AlertSeverity.EMERGENCY: "#ff0000",
            }
            
            response = await client.send(
                attachments=[{
                    "color": color_map.get(alert.severity, "#808080"),
                    "title": f"{alert.severity.value.upper()}: {alert.name}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Metric", "value": alert.metric_name, "short": True},
                        {"title": "Value", "value": str(alert.metric_value), "short": True},
                        {"title": "Threshold", "value": str(alert.threshold), "short": True},
                    ],
                    "ts": str(int(alert.timestamp.timestamp())),
                }]
            )
            
            return response.status_code == 200
        except ImportError:
            logger.warning("slack_sdk not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class PagerDutyNotificationChannel(NotificationChannel):
    """PagerDuty notification channel (configured via env var)."""
    
    def __init__(self):
        """Initialize PagerDuty channel."""
        self.routing_key = os.environ.get("PAGERDUTY_ROUTING_KEY", "")
    
    def is_configured(self) -> bool:
        """Check if PagerDuty is configured.
        
        Returns:
            True if configured
        """
        return bool(self.routing_key)
    
    async def send(self, alert: Alert) -> bool:
        """Send alert to PagerDuty.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if successful
        """
        if not self.is_configured():
            logger.warning("PagerDuty not configured, skipping notification")
            return False
        
        try:
            import aiohttp
            
            severity_map = {
                AlertSeverity.INFO: "info",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.CRITICAL: "error",
                AlertSeverity.EMERGENCY: "critical",
            }
            
            payload = {
                "routing_key": self.routing_key,
                "event_action": "trigger",
                "dedup_key": f"{alert.name}_{alert.metric_name}",
                "payload": {
                    "summary": f"{alert.name}: {alert.message}",
                    "source": "medai-compass",
                    "severity": severity_map.get(alert.severity, "warning"),
                    "custom_details": alert.to_dict(),
                },
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=payload,
                ) as response:
                    return response.status == 202
        except ImportError:
            logger.warning("aiohttp not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            return False


# ============================================================================
# Alert History
# ============================================================================

class AlertHistory:
    """Tracks history of alerts."""
    
    def __init__(self, max_alerts: int = 1000):
        """Initialize alert history.
        
        Args:
            max_alerts: Maximum alerts to retain
        """
        self.alerts: list[Alert] = []
        self.max_alerts = max_alerts
    
    def record(self, alert: Alert) -> None:
        """Record an alert.
        
        Args:
            alert: Alert to record
        """
        self.alerts.append(alert)
        
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
    
    def get_by_severity(self, severity: AlertSeverity) -> list[Alert]:
        """Get alerts by severity.
        
        Args:
            severity: Severity to filter by
            
        Returns:
            List of matching alerts
        """
        return [a for a in self.alerts if a.severity == severity]
    
    def get_count_in_window(self, hours: int = 1) -> int:
        """Get count of alerts in time window.
        
        Args:
            hours: Time window in hours
            
        Returns:
            Number of alerts in window
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        return sum(1 for a in self.alerts if a.timestamp >= cutoff)


# ============================================================================
# Alert Manager
# ============================================================================

class AlertManager:
    """Manages alert rules, channels, and notifications."""
    
    def __init__(self, config: AlertConfig):
        """Initialize alert manager.
        
        Args:
            config: Alert configuration
        """
        self.config = config
        self.rules: list[AlertRule] = []
        self.channels: list[NotificationChannel] = []
        self.history = AlertHistory()
        self._cooldowns: dict[str, datetime] = {}
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.
        
        Args:
            rule: Rule to add
        """
        self.rules.append(rule)
    
    def add_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel.
        
        Args:
            channel: Channel to add
        """
        self.channels.append(channel)
    
    def evaluate(self, metrics: dict[str, float]) -> list[Alert]:
        """Evaluate metrics against all rules.
        
        Args:
            metrics: Dictionary of metric values
            
        Returns:
            List of triggered alerts
        """
        alerts = []
        
        for rule in self.rules:
            if hasattr(rule, 'name') and rule.name in metrics:
                alert = rule.evaluate(metrics[rule.name])
                
                if alert and self._check_cooldown(rule.name):
                    alerts.append(alert)
                    self._set_cooldown(rule.name)
                    self.history.record(alert)
        
        return alerts
    
    async def check_and_alert(self, metrics: dict[str, float]) -> list[Alert]:
        """Check metrics and send alerts.
        
        Args:
            metrics: Dictionary of metric values
            
        Returns:
            List of sent alerts
        """
        alerts = self.evaluate(metrics)
        
        for alert in alerts:
            for channel in self.channels:
                await channel.send(alert)
        
        return alerts
    
    def _check_cooldown(self, rule_name: str) -> bool:
        """Check if cooldown has expired for a rule.
        
        Args:
            rule_name: Name of the rule
            
        Returns:
            True if alert can be sent
        """
        last_alert = self._cooldowns.get(rule_name)
        if not last_alert:
            return True
        
        elapsed = (datetime.now() - last_alert).total_seconds()
        return elapsed >= self.config.cooldown_seconds
    
    def _set_cooldown(self, rule_name: str) -> None:
        """Set cooldown for a rule.
        
        Args:
            rule_name: Name of the rule
        """
        self._cooldowns[rule_name] = datetime.now()


# ============================================================================
# Prometheus Integration
# ============================================================================

def export_prometheus_rules(manager: AlertManager) -> dict:
    """Export alert rules to Prometheus format.
    
    Args:
        manager: Alert manager with rules
        
    Returns:
        Prometheus alert rules format
    """
    groups = [{
        "name": "medai_compass_alerts",
        "rules": [],
    }]
    
    for rule in manager.rules:
        if isinstance(rule, LatencyAlertRule):
            groups[0]["rules"].append({
                "alert": f"MedAI{rule.name.title().replace('_', '')}High",
                "expr": f"medai_{rule.name}_ms > {rule.warning_threshold}",
                "for": "5m",
                "labels": {"severity": "warning"},
                "annotations": {
                    "summary": f"High {rule.name} detected",
                },
            })
    
    return {"groups": groups}


def sync_prometheus_rules(rules_path: str) -> dict:
    """Sync with existing Prometheus alert rules.
    
    Args:
        rules_path: Path to Prometheus rules file
        
    Returns:
        Sync status
    """
    try:
        with open(rules_path) as f:
            existing_rules = yaml.safe_load(f)
        
        return {
            "synced": True,
            "existing_groups": len(existing_rules.get("groups", [])),
        }
    except Exception as e:
        return {
            "error": str(e),
            "synced": False,
        }

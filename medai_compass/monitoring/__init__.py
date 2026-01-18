"""
Monitoring module for MedAI Compass (Phase 8).

Provides comprehensive monitoring and observability including:
- Drift detection (input, output, concept) with alibi-detect integration
- Performance monitoring (latency, throughput, resources)
- Alerting system with configurable channels
- Auto-retraining triggers
"""

from medai_compass.monitoring.drift_detector import (
    AlibiTabularDriftDetector,
    ConceptDriftDetector,
    DriftConfig,
    DriftHistory,
    DriftManager,
    DriftResult,
    DriftType,
    InputDriftDetector,
    OutputDriftDetector,
    calculate_kl_divergence,
    calculate_psi,
    get_model_thresholds,
)
from medai_compass.monitoring.performance_monitor import (
    AsyncPerformanceMonitor,
    LatencyTracker,
    PerformanceConfig,
    PerformanceMonitor,
    PrometheusMetricsExporter,
    QualityMetricsTracker,
    ResourceMonitor,
    ThroughputMonitor,
    get_model_baseline,
)
from medai_compass.monitoring.alerting import (
    AccuracyAlertRule,
    Alert,
    AlertConfig,
    AlertHistory,
    AlertManager,
    AlertSeverity,
    DriftAlertRule,
    LatencyAlertRule,
    LogNotificationChannel,
    MockNotificationChannel,
    NotificationChannel,
    PagerDutyNotificationChannel,
    SafetyAlertRule,
    SlackNotificationChannel,
    export_prometheus_rules,
    get_model_alert_config,
    sync_prometheus_rules,
)
from medai_compass.monitoring.retraining_trigger import (
    AccuracyBasedTrigger,
    CompositeTrigger,
    DataVolumeTrigger,
    DriftBasedTrigger,
    RetrainingConfig,
    RetrainingHistory,
    RetrainingManager,
    ScheduledTrigger,
    TriggerReason,
    TriggerResult,
    TriggerType,
    get_model_retrain_config,
)

__all__ = [
    # Drift Detection
    "DriftConfig",
    "DriftType",
    "DriftResult",
    "DriftManager",
    "DriftHistory",
    "InputDriftDetector",
    "OutputDriftDetector",
    "ConceptDriftDetector",
    "AlibiTabularDriftDetector",
    "calculate_kl_divergence",
    "calculate_psi",
    "get_model_thresholds",
    # Performance Monitoring
    "PerformanceConfig",
    "PerformanceMonitor",
    "AsyncPerformanceMonitor",
    "LatencyTracker",
    "ThroughputMonitor",
    "ResourceMonitor",
    "QualityMetricsTracker",
    "PrometheusMetricsExporter",
    "get_model_baseline",
    # Alerting
    "AlertSeverity",
    "AlertConfig",
    "Alert",
    "AlertManager",
    "AlertHistory",
    "LatencyAlertRule",
    "DriftAlertRule",
    "AccuracyAlertRule",
    "SafetyAlertRule",
    "NotificationChannel",
    "MockNotificationChannel",
    "LogNotificationChannel",
    "SlackNotificationChannel",
    "PagerDutyNotificationChannel",
    "get_model_alert_config",
    "export_prometheus_rules",
    "sync_prometheus_rules",
    # Retraining Triggers
    "TriggerType",
    "TriggerReason",
    "TriggerResult",
    "RetrainingConfig",
    "RetrainingManager",
    "RetrainingHistory",
    "DriftBasedTrigger",
    "AccuracyBasedTrigger",
    "ScheduledTrigger",
    "DataVolumeTrigger",
    "CompositeTrigger",
    "get_model_retrain_config",
]

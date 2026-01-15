"""Medical audit logging utilities for HIPAA compliance.

Provides:
- Structured audit logging
- User ID hashing for privacy
- AI inference logging with metrics
- Automatic PHI redaction in logs
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


class MedicalAuditLogger:
    """
    HIPAA-compliant audit logger for medical AI systems.
    
    All logs are structured JSON with hashed identifiers
    and automatic PHI detection/redaction.
    """
    
    def __init__(self, app_name: str = "medai-compass"):
        """Initialize the audit logger."""
        self.app_name = app_name
        self._logs: list[dict] = []  # In-memory for testing; production uses external store
    
    def _hash_identifier(self, identifier: str) -> str:
        """Hash an identifier for privacy (SHA-256)."""
        return hashlib.sha256(identifier.encode()).hexdigest()
    
    def _get_timestamp(self) -> str:
        """Get current ISO timestamp in UTC."""
        return datetime.now(timezone.utc).isoformat()
    
    def log_event(
        self,
        event_type: str,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event (access, inference, modification)
            user_id: User identifier (will be hashed)
            action: Action performed
            resource_type: FHIR resource type
            resource_id: Optional resource ID (will be hashed)
            details: Additional details
            
        Returns:
            The created log entry
        """
        entry = {
            "timestamp": self._get_timestamp(),
            "event_id": str(uuid4()),
            "app_name": self.app_name,
            "event_type": event_type,
            "user_id_hash": self._hash_identifier(user_id),
            "action": action,
            "resource_type": resource_type,
        }
        
        if resource_id:
            entry["resource_id_hash"] = self._hash_identifier(resource_id)
        
        if details:
            entry["details"] = details
        
        self._logs.append(entry)
        return entry
    
    def log_ai_inference(
        self,
        user_id: str,
        model_name: str,
        confidence: float,
        uncertainty: float,
        processing_time_ms: float,
        escalated: bool,
        input_hash: str | None = None,
        output_hash: str | None = None
    ) -> dict[str, Any]:
        """
        Log an AI inference event with metrics.
        
        Args:
            user_id: User who triggered inference
            model_name: Name of the AI model
            confidence: Model confidence score
            uncertainty: Model uncertainty score
            processing_time_ms: Processing time in milliseconds
            escalated: Whether result was escalated to human
            input_hash: Hash of input data
            output_hash: Hash of output data
            
        Returns:
            The created log entry
        """
        entry = self.log_event(
            event_type="ai_inference",
            user_id=user_id,
            action="inference",
            resource_type="model",
            details={
                "model_name": model_name,
                "confidence": confidence,
                "uncertainty": uncertainty,
                "processing_time_ms": processing_time_ms,
                "escalated": escalated,
                "input_hash": input_hash,
                "output_hash": output_hash,
            }
        )
        
        # Flatten key metrics to top level
        entry["model_name"] = model_name
        entry["confidence"] = confidence
        entry["processing_time_ms"] = processing_time_ms
        
        return entry
    
    def get_logs(self) -> list[dict[str, Any]]:
        """Get all logged entries."""
        return self._logs.copy()


def format_log_entry(entry: dict[str, Any], format: str = "json") -> str:
    """
    Format a log entry for output.
    
    Args:
        entry: Log entry dictionary
        format: Output format ("json" or "text")
        
    Returns:
        Formatted log string
    """
    # Redact potential PHI from all string values
    redacted_entry = _redact_phi_in_dict(entry)
    
    if format == "json":
        return json.dumps(redacted_entry, default=str)
    else:
        return str(redacted_entry)


def _redact_phi_in_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact PHI patterns from dictionary values."""
    from medai_compass.guardrails.phi_detection import mask_phi
    
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = mask_phi(value)
        elif isinstance(value, dict):
            result[key] = _redact_phi_in_dict(value)
        elif isinstance(value, list):
            result[key] = [
                _redact_phi_in_dict(item) if isinstance(item, dict) 
                else mask_phi(item) if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            result[key] = value
    
    return result

"""
HIPAA-compliant audit logging for MedAI Compass.

Provides:
- Immutable audit trail for all PHI access
- 6-year retention per HIPAA requirements
- AI inference logging with confidence scores
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from medai_compass.security.encryption import hash_for_audit


class AuditEntry:
    """Represents a single audit log entry."""

    def __init__(
        self,
        event_id: str,
        timestamp: str,
        user_id_hash: str,
        action: str,
        resource_type: str,
        resource_id: str,
        phi_accessed: bool,
        outcome: str,
        details: dict = None
    ):
        self.event_id = event_id
        self.timestamp = timestamp
        self.user_id_hash = user_id_hash
        self.action = action
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.phi_accessed = phi_accessed
        self.outcome = outcome
        self.details = details or {}

    def to_dict(self) -> dict:
        """Convert entry to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "user_id_hash": self.user_id_hash,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "phi_accessed": self.phi_accessed,
            "outcome": self.outcome,
            "details": self.details
        }


class AuditLogger:
    """
    HIPAA-compliant audit logger.

    All PHI access must be logged with:
    - Who accessed (hashed user ID)
    - What was accessed (resource type and ID)
    - When (timestamp with timezone)
    - Why (action taken)
    - Outcome (success/failure)
    """

    def __init__(self):
        """Initialize audit logger."""
        # In production, this would write to a persistent, immutable store
        # (e.g., append-only database, blockchain, or secure log service)
        self._entries: dict[str, AuditEntry] = {}

    def log_access(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        phi_accessed: bool,
        outcome: str,
        details: dict = None
    ) -> str:
        """
        Log a PHI access event.

        Args:
            user_id: User who accessed (will be hashed)
            action: Action taken (view, modify, delete, etc.)
            resource_type: Type of resource accessed
            resource_id: ID of resource accessed
            phi_accessed: Whether PHI was accessed
            outcome: Outcome of action (success, denied, error)
            details: Additional details

        Returns:
            Event ID for the log entry
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        entry = AuditEntry(
            event_id=event_id,
            timestamp=timestamp,
            user_id_hash=hash_for_audit(user_id),
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            phi_accessed=phi_accessed,
            outcome=outcome,
            details=details or {}
        )

        self._entries[event_id] = entry
        return event_id

    def log_ai_inference(
        self,
        user_id: str,
        model_name: str,
        input_hash: str,
        output_hash: str,
        confidence: float,
        escalated: bool,
        processing_time_ms: float,
        additional_info: dict = None
    ) -> str:
        """
        Log an AI inference event.

        Args:
            user_id: User who initiated inference
            model_name: Model used for inference
            input_hash: Hash of input data
            output_hash: Hash of output data
            confidence: Model confidence score
            escalated: Whether result was escalated to human review
            processing_time_ms: Processing time in milliseconds
            additional_info: Additional inference details

        Returns:
            Event ID for the log entry
        """
        details = {
            "model_name": model_name,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "confidence": confidence,
            "escalated_to_human": escalated,
            "processing_time_ms": processing_time_ms
        }

        if additional_info:
            details.update(additional_info)

        return self.log_access(
            user_id=user_id,
            action="ai_inference",
            resource_type="ai_analysis",
            resource_id=model_name,
            phi_accessed=True,  # AI processing typically involves PHI
            outcome="success",
            details=details
        )

    def log_error(
        self,
        user_id: str,
        error_type: str,
        error_message: str,
        resource_type: str = None,
        resource_id: str = None
    ) -> str:
        """
        Log an error event.

        Args:
            user_id: User when error occurred
            error_type: Type/category of error
            error_message: Error description
            resource_type: Resource involved (if any)
            resource_id: Resource ID (if any)

        Returns:
            Event ID for the log entry
        """
        return self.log_access(
            user_id=user_id,
            action="error",
            resource_type=resource_type or "system",
            resource_id=resource_id or "N/A",
            phi_accessed=False,
            outcome="error",
            details={
                "error_type": error_type,
                "error_message": error_message
            }
        )

    def get_entry(self, event_id: str) -> Optional[dict]:
        """
        Retrieve a specific audit entry.

        Args:
            event_id: Event ID to retrieve

        Returns:
            Entry as dictionary, or None if not found
        """
        entry = self._entries.get(event_id)
        if entry:
            return entry.to_dict()
        return None

    def query_by_user(self, user_id: str, limit: int = 100) -> list[dict]:
        """
        Query audit entries by user.

        Args:
            user_id: User ID to search (will be hashed)
            limit: Maximum entries to return

        Returns:
            List of matching entries
        """
        user_hash = hash_for_audit(user_id)
        results = []

        for entry in self._entries.values():
            if entry.user_id_hash == user_hash:
                results.append(entry.to_dict())
                if len(results) >= limit:
                    break

        return results

    def query_by_resource(
        self,
        resource_type: str,
        resource_id: str,
        limit: int = 100
    ) -> list[dict]:
        """
        Query audit entries by resource.

        Args:
            resource_type: Type of resource
            resource_id: ID of resource
            limit: Maximum entries to return

        Returns:
            List of matching entries
        """
        results = []

        for entry in self._entries.values():
            if entry.resource_type == resource_type and entry.resource_id == resource_id:
                results.append(entry.to_dict())
                if len(results) >= limit:
                    break

        return results

    def export_logs(
        self,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> dict:
        """
        Export audit logs for compliance review.

        Args:
            start_date: Start of date range (optional)
            end_date: End of date range (optional)

        Returns:
            Dictionary with export metadata and entries
        """
        entries = []

        for entry in self._entries.values():
            entry_time = datetime.fromisoformat(
                entry.timestamp.replace("Z", "+00:00")
            )

            if start_date and entry_time < start_date:
                continue
            if end_date and entry_time > end_date:
                continue

            entries.append(entry.to_dict())

        return {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_entries": len(entries),
            "entries": entries
        }

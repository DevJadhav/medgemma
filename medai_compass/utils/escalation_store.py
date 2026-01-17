"""
Escalation Store - Storage layer for clinician review queue.

Supports both in-memory (testing) and PostgreSQL (production) backends.
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

# PostgreSQL support
try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

# Synchronous PostgreSQL support
try:
    import psycopg2
    import psycopg2.extras

    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

import json


# =============================================================================
# Type Definitions
# =============================================================================
EscalationReason = Literal["critical_finding", "low_confidence", "safety_concern", "manual_request"]
EscalationPriority = Literal["high", "medium", "low"]
EscalationStatus = Literal["pending", "in_review", "approved", "rejected"]
ReviewDecision = Literal["approve", "reject", "modify"]


# =============================================================================
# Escalation Store Implementation
# =============================================================================
class EscalationStore:
    """
    Storage layer for escalations requiring clinician review.
    
    Supports:
    - In-memory storage for testing
    - PostgreSQL for production
    - Redis caching (optional)
    """

    def __init__(
        self,
        use_memory: bool = False,
        db_url: Optional[str] = None,
    ):
        """
        Initialize escalation store.
        
        Args:
            use_memory: Use in-memory storage (for testing)
            db_url: PostgreSQL connection URL
        """
        self.use_memory = use_memory
        self.db_url = db_url or os.getenv("DATABASE_URL")
        
        # In-memory storage
        self._memory_store: dict[str, dict[str, Any]] = {}
        
        # PostgreSQL connection pool
        self._pool: Optional[Any] = None

    # =========================================================================
    # Create Escalation
    # =========================================================================
    def create(
        self,
        request_id: str,
        reason: EscalationReason,
        priority: EscalationPriority = "medium",
        patient_id: Optional[str] = None,
        original_message: Optional[str] = None,
        diagnostic_result: Optional[dict] = None,
        communication_result: Optional[dict] = None,
        workflow_result: Optional[dict] = None,
        agent_type: Optional[str] = None,
        confidence_score: Optional[float] = None,
        uncertainty_score: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Create a new escalation for clinician review.
        
        Args:
            request_id: Original request ID
            reason: Escalation reason
            priority: Priority level
            patient_id: Patient identifier
            original_message: Original patient message
            diagnostic_result: Diagnostic agent result
            communication_result: Communication agent result
            workflow_result: Workflow agent result
            agent_type: Source agent type
            confidence_score: Model confidence
            uncertainty_score: Uncertainty estimate
            metadata: Additional metadata
            
        Returns:
            Created escalation record
        """
        escalation_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        
        escalation = {
            "id": escalation_id,
            "request_id": request_id,
            "patient_id": patient_id,
            "timestamp": timestamp.isoformat(),
            "reason": reason,
            "priority": priority,
            "status": "pending",
            "original_message": original_message,
            "diagnostic_result": diagnostic_result,
            "communication_result": communication_result,
            "workflow_result": workflow_result,
            "agent_type": agent_type,
            "confidence_score": confidence_score,
            "uncertainty_score": uncertainty_score,
            "assigned_to": None,
            "reviewed_by": None,
            "reviewed_at": None,
            "review_notes": None,
            "modified_response": None,
            "metadata": metadata or {},
            "created_at": timestamp.isoformat(),
            "updated_at": timestamp.isoformat(),
        }
        
        if self.use_memory:
            self._memory_store[escalation_id] = escalation
        else:
            self._db_create(escalation)
        
        return escalation

    def _db_create(self, escalation: dict[str, Any]) -> None:
        """Create escalation in database."""
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2 not available")
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO escalations (
                        id, request_id, patient_id, timestamp, reason, priority, status,
                        original_message, diagnostic_result, communication_result, workflow_result,
                        agent_type, confidence_score, uncertainty_score, metadata
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        escalation["id"],
                        escalation["request_id"],
                        escalation["patient_id"],
                        escalation["timestamp"],
                        escalation["reason"],
                        escalation["priority"],
                        escalation["status"],
                        escalation["original_message"],
                        json.dumps(escalation["diagnostic_result"]) if escalation["diagnostic_result"] else None,
                        json.dumps(escalation["communication_result"]) if escalation["communication_result"] else None,
                        json.dumps(escalation["workflow_result"]) if escalation["workflow_result"] else None,
                        escalation["agent_type"],
                        escalation["confidence_score"],
                        escalation["uncertainty_score"],
                        json.dumps(escalation["metadata"]),
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    # =========================================================================
    # List Pending Escalations
    # =========================================================================
    def list_pending(
        self,
        priority: Optional[EscalationPriority] = None,
        reason: Optional[EscalationReason] = None,
        status: Optional[EscalationStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        List pending escalations with optional filters.
        
        Args:
            priority: Filter by priority
            reason: Filter by reason
            status: Filter by status (defaults to pending/in_review)
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of escalation records
        """
        if self.use_memory:
            return self._memory_list_pending(priority, reason, status, limit, offset)
        else:
            return self._db_list_pending(priority, reason, status, limit, offset)

    def _memory_list_pending(
        self,
        priority: Optional[str],
        reason: Optional[str],
        status: Optional[str],
        limit: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        """List from memory store."""
        allowed_statuses = {status} if status else {"pending", "in_review"}
        
        results = []
        for esc in self._memory_store.values():
            if esc["status"] not in allowed_statuses:
                continue
            if priority and esc["priority"] != priority:
                continue
            if reason and esc["reason"] != reason:
                continue
            results.append(esc)
        
        # Sort by priority (high first) then timestamp
        priority_order = {"high": 0, "medium": 1, "low": 2}
        results.sort(key=lambda x: (priority_order.get(x["priority"], 1), x["timestamp"]))
        
        return results[offset : offset + limit]

    def _db_list_pending(
        self,
        priority: Optional[str],
        reason: Optional[str],
        status: Optional[str],
        limit: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        """List from database."""
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2 not available")
        
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Build query
                conditions = []
                params = []
                
                if status:
                    conditions.append("status = %s")
                    params.append(status)
                else:
                    conditions.append("status IN ('pending', 'in_review')")
                
                if priority:
                    conditions.append("priority = %s")
                    params.append(priority)
                
                if reason:
                    conditions.append("reason = %s")
                    params.append(reason)
                
                where_clause = " AND ".join(conditions)
                params.extend([limit, offset])
                
                cur.execute(
                    f"""
                    SELECT * FROM escalations
                    WHERE {where_clause}
                    ORDER BY 
                        CASE priority 
                            WHEN 'high' THEN 0 
                            WHEN 'medium' THEN 1 
                            ELSE 2 
                        END,
                        timestamp ASC
                    LIMIT %s OFFSET %s
                    """,
                    params,
                )
                
                rows = cur.fetchall()
                return [self._row_to_dict(row) for row in rows]
        finally:
            conn.close()

    # =========================================================================
    # Get Single Escalation
    # =========================================================================
    def get_by_id(self, escalation_id: str) -> Optional[dict[str, Any]]:
        """
        Get escalation by ID.
        
        Args:
            escalation_id: Escalation UUID
            
        Returns:
            Escalation record or None
        """
        if self.use_memory:
            return self._memory_store.get(escalation_id)
        else:
            return self._db_get_by_id(escalation_id)

    def _db_get_by_id(self, escalation_id: str) -> Optional[dict[str, Any]]:
        """Get from database."""
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2 not available")
        
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM escalations WHERE id = %s",
                    (escalation_id,),
                )
                row = cur.fetchone()
                return self._row_to_dict(row) if row else None
        finally:
            conn.close()

    # =========================================================================
    # Submit Review Decision
    # =========================================================================
    def submit_review(
        self,
        escalation_id: str,
        decision: ReviewDecision,
        reviewer_id: str,
        notes: str,
        modified_response: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Submit review decision for escalation.
        
        Args:
            escalation_id: Escalation UUID
            decision: Review decision (approve, reject, modify)
            reviewer_id: Reviewer identifier
            notes: Review notes
            modified_response: Modified response (for modify decision)
            
        Returns:
            Updated escalation record
        """
        status_map = {
            "approve": "approved",
            "reject": "rejected",
            "modify": "approved",
        }
        
        new_status = status_map[decision]
        reviewed_at = datetime.now(timezone.utc)
        
        if self.use_memory:
            if escalation_id not in self._memory_store:
                raise ValueError(f"Escalation {escalation_id} not found")
            
            esc = self._memory_store[escalation_id]
            esc["status"] = new_status
            esc["reviewed_by"] = reviewer_id
            esc["reviewed_at"] = reviewed_at.isoformat()
            esc["review_notes"] = notes
            esc["modified_response"] = modified_response
            esc["updated_at"] = reviewed_at.isoformat()
            
            return esc
        else:
            return self._db_submit_review(
                escalation_id, new_status, reviewer_id, notes, modified_response, reviewed_at
            )

    def _db_submit_review(
        self,
        escalation_id: str,
        new_status: str,
        reviewer_id: str,
        notes: str,
        modified_response: Optional[str],
        reviewed_at: datetime,
    ) -> dict[str, Any]:
        """Submit review to database."""
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2 not available")
        
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    UPDATE escalations
                    SET status = %s,
                        reviewed_by = %s,
                        reviewed_at = %s,
                        review_notes = %s,
                        modified_response = %s,
                        updated_at = %s
                    WHERE id = %s
                    RETURNING *
                    """,
                    (
                        new_status,
                        reviewer_id,
                        reviewed_at.isoformat(),
                        notes,
                        modified_response,
                        reviewed_at.isoformat(),
                        escalation_id,
                    ),
                )
                row = cur.fetchone()
                conn.commit()
                
                if not row:
                    raise ValueError(f"Escalation {escalation_id} not found")
                
                return self._row_to_dict(row)
        finally:
            conn.close()

    # =========================================================================
    # Statistics
    # =========================================================================
    def get_stats(self) -> dict[str, Any]:
        """
        Get escalation statistics.
        
        Returns:
            Statistics dictionary
        """
        if self.use_memory:
            return self._memory_get_stats()
        else:
            return self._db_get_stats()

    def _memory_get_stats(self) -> dict[str, Any]:
        """Get stats from memory store."""
        pending = sum(1 for e in self._memory_store.values() if e["status"] == "pending")
        in_review = sum(1 for e in self._memory_store.values() if e["status"] == "in_review")
        
        today = datetime.now(timezone.utc).date()
        approved_today = sum(
            1
            for e in self._memory_store.values()
            if e["status"] == "approved"
            and e.get("reviewed_at")
            and datetime.fromisoformat(e["reviewed_at"]).date() == today
        )
        rejected_today = sum(
            1
            for e in self._memory_store.values()
            if e["status"] == "rejected"
            and e.get("reviewed_at")
            and datetime.fromisoformat(e["reviewed_at"]).date() == today
        )
        
        return {
            "total_pending": pending,
            "total_in_review": in_review,
            "total_approved_today": approved_today,
            "total_rejected_today": rejected_today,
            "average_review_time_ms": 0,  # Would need timestamps to calculate
        }

    def _db_get_stats(self) -> dict[str, Any]:
        """Get stats from database."""
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2 not available")
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Count by status
                cur.execute("""
                    SELECT 
                        COUNT(*) FILTER (WHERE status = 'pending') as pending,
                        COUNT(*) FILTER (WHERE status = 'in_review') as in_review,
                        COUNT(*) FILTER (WHERE status = 'approved' AND reviewed_at::date = CURRENT_DATE) as approved_today,
                        COUNT(*) FILTER (WHERE status = 'rejected' AND reviewed_at::date = CURRENT_DATE) as rejected_today
                    FROM escalations
                """)
                row = cur.fetchone()
                
                # Average review time
                cur.execute("""
                    SELECT AVG(EXTRACT(EPOCH FROM (reviewed_at - timestamp)) * 1000) as avg_ms
                    FROM escalations
                    WHERE reviewed_at IS NOT NULL
                    AND reviewed_at > NOW() - INTERVAL '24 hours'
                """)
                avg_row = cur.fetchone()
                
                return {
                    "total_pending": row[0] if row else 0,
                    "total_in_review": row[1] if row else 0,
                    "total_approved_today": row[2] if row else 0,
                    "total_rejected_today": row[3] if row else 0,
                    "average_review_time_ms": avg_row[0] if avg_row and avg_row[0] else 0,
                }
        finally:
            conn.close()

    # =========================================================================
    # Helpers
    # =========================================================================
    def _get_connection(self):
        """Get database connection."""
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2 not available")
        
        db_url = self.db_url
        if not db_url:
            # Build from environment
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = os.getenv("POSTGRES_PORT", "5432")
            user = os.getenv("POSTGRES_USER", "medai")
            password = os.getenv("POSTGRES_PASSWORD", "")
            database = os.getenv("POSTGRES_DB", "medai_compass")
            db_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        
        return psycopg2.connect(db_url)

    def _row_to_dict(self, row: dict) -> dict[str, Any]:
        """Convert database row to dictionary."""
        if not row:
            return {}
        
        result = dict(row)
        
        # Convert timestamps to ISO format strings
        for key in ["timestamp", "reviewed_at", "created_at", "updated_at"]:
            if key in result and result[key] is not None:
                if hasattr(result[key], "isoformat"):
                    result[key] = result[key].isoformat()
        
        # Convert UUID to string
        if "id" in result and hasattr(result["id"], "hex"):
            result["id"] = str(result["id"])
        
        return result


# =============================================================================
# Singleton Instance
# =============================================================================
_store_instance: Optional[EscalationStore] = None


def get_escalation_store(use_memory: bool = False) -> EscalationStore:
    """
    Get or create escalation store singleton.
    
    Args:
        use_memory: Use in-memory storage
        
    Returns:
        EscalationStore instance
    """
    global _store_instance
    
    if _store_instance is None:
        _store_instance = EscalationStore(use_memory=use_memory)
    
    return _store_instance


# Default instance for import
escalation_store = EscalationStore(
    use_memory=os.getenv("TESTING", "false").lower() == "true"
)

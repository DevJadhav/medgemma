"""
Tests for Escalation API Endpoints.

TDD: Tests written first to define expected behavior.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


class TestEscalationEndpoints:
    """Test suite for escalation API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from medai_compass.api.main import app
        return TestClient(app)

    @pytest.fixture
    def mock_escalation_store(self):
        """Mock escalation store."""
        with patch("medai_compass.api.main.escalation_store") as mock:
            mock.list_pending = MagicMock(return_value=[
                {
                    "id": "esc-001",
                    "request_id": "req-123",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "patient_id": "patient-001",
                    "reason": "low_confidence",
                    "priority": "medium",
                    "status": "pending",
                    "diagnostic_result": None,
                    "communication_result": {
                        "request_id": "req-123",
                        "response": "Please consult a physician.",
                        "triage_level": "URGENT",
                        "requires_escalation": True,
                    },
                    "original_message": "I have severe chest pain",
                },
                {
                    "id": "esc-002",
                    "request_id": "req-456",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "patient_id": "patient-002",
                    "reason": "critical_finding",
                    "priority": "high",
                    "status": "pending",
                    "diagnostic_result": {
                        "request_id": "req-456",
                        "findings": [{"finding": "Pneumothorax", "confidence": 0.92}],
                        "confidence": 0.92,
                        "requires_review": True,
                    },
                    "communication_result": None,
                    "original_message": None,
                },
            ])
            mock.get_by_id = MagicMock(return_value={
                "id": "esc-001",
                "request_id": "req-123",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "patient_id": "patient-001",
                "reason": "low_confidence",
                "priority": "medium",
                "status": "pending",
            })
            mock.submit_review = MagicMock(return_value={
                "id": "esc-001",
                "request_id": "req-123",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "patient_id": "patient-001",
                "reason": "low_confidence",
                "priority": "medium",
                "status": "approved",
                "reviewed_by": "dr.smith",
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "review_notes": "Approved with modifications",
            })
            mock.get_stats = MagicMock(return_value={
                "total_pending": 5,
                "total_in_review": 2,
                "total_approved_today": 10,
                "total_rejected_today": 1,
                "average_review_time_ms": 45000,
            })
            mock.create = MagicMock(return_value={
                "id": "esc-003",
                "request_id": "req-789",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": "safety_concern",
                "priority": "high",
                "status": "pending",
            })
            yield mock

    # =========================================================================
    # GET /api/v1/escalations - List Pending Escalations
    # =========================================================================
    def test_list_escalations_returns_pending(self, client, mock_escalation_store):
        """Should return list of pending escalations."""
        response = client.get("/api/v1/escalations")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "escalations" in data
        assert "total" in data
        assert "timestamp" in data
        assert len(data["escalations"]) == 2
        
        # Verify first escalation
        esc1 = data["escalations"][0]
        assert esc1["id"] == "esc-001"
        assert esc1["reason"] == "low_confidence"
        assert esc1["priority"] == "medium"
        assert esc1["status"] == "pending"

    def test_list_escalations_filter_by_priority(self, client, mock_escalation_store):
        """Should filter escalations by priority."""
        # Filter high priority only
        mock_escalation_store.list_pending.return_value = [
            {
                "id": "esc-002",
                "request_id": "req-456",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": "critical_finding",
                "priority": "high",
                "status": "pending",
            },
        ]
        
        response = client.get("/api/v1/escalations?priority=high")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["escalations"]) == 1
        assert data["escalations"][0]["priority"] == "high"

    def test_list_escalations_filter_by_reason(self, client, mock_escalation_store):
        """Should filter escalations by reason."""
        mock_escalation_store.list_pending.return_value = [
            {
                "id": "esc-002",
                "request_id": "req-456",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": "critical_finding",
                "priority": "high",
                "status": "pending",
            },
        ]
        
        response = client.get("/api/v1/escalations?reason=critical_finding")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["escalations"]) == 1
        assert data["escalations"][0]["reason"] == "critical_finding"

    def test_list_escalations_pagination(self, client, mock_escalation_store):
        """Should support pagination with limit and offset."""
        response = client.get("/api/v1/escalations?limit=10&offset=0")
        
        assert response.status_code == 200
        mock_escalation_store.list_pending.assert_called()

    def test_list_escalations_empty_list(self, client, mock_escalation_store):
        """Should return empty list when no pending escalations."""
        mock_escalation_store.list_pending.return_value = []
        
        response = client.get("/api/v1/escalations")
        
        assert response.status_code == 200
        data = response.json()
        assert data["escalations"] == []
        assert data["total"] == 0

    # =========================================================================
    # GET /api/v1/escalations/{id} - Get Single Escalation
    # =========================================================================
    def test_get_escalation_by_id(self, client, mock_escalation_store):
        """Should return single escalation by ID."""
        response = client.get("/api/v1/escalations/esc-001")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "esc-001"
        assert data["request_id"] == "req-123"

    def test_get_escalation_not_found(self, client, mock_escalation_store):
        """Should return 404 for non-existent escalation."""
        mock_escalation_store.get_by_id.return_value = None
        
        response = client.get("/api/v1/escalations/non-existent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    # =========================================================================
    # POST /api/v1/escalations/{id}/review - Submit Review Decision
    # =========================================================================
    def test_submit_review_approve(self, client, mock_escalation_store):
        """Should approve escalation with notes."""
        review_data = {
            "decision": "approve",
            "notes": "Finding confirmed, proceed with treatment.",
            "reviewer_id": "dr.smith",
        }
        
        response = client.post(
            "/api/v1/escalations/esc-001/review",
            json=review_data,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "approved"
        assert data["reviewed_by"] == "dr.smith"

    def test_submit_review_reject(self, client, mock_escalation_store):
        """Should reject escalation with notes."""
        mock_escalation_store.submit_review.return_value = {
            "id": "esc-001",
            "request_id": "req-123",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "patient_id": "patient-001",
            "reason": "low_confidence",
            "priority": "medium",
            "status": "rejected",
            "reviewed_by": "dr.jones",
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "review_notes": "False positive, no action needed.",
        }
        
        review_data = {
            "decision": "reject",
            "notes": "False positive, no action needed.",
            "reviewer_id": "dr.jones",
        }
        
        response = client.post(
            "/api/v1/escalations/esc-001/review",
            json=review_data,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rejected"

    def test_submit_review_modify(self, client, mock_escalation_store):
        """Should modify and approve escalation with new response."""
        mock_escalation_store.submit_review.return_value = {
            "id": "esc-001",
            "request_id": "req-123",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "patient_id": "patient-001",
            "reason": "low_confidence",
            "priority": "medium",
            "status": "approved",
            "reviewed_by": "dr.wilson",
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "review_notes": "Modified response for clarity.",
            "modified_response": "Please visit ER immediately for chest pain evaluation.",
        }
        
        review_data = {
            "decision": "modify",
            "notes": "Modified response for clarity.",
            "modified_response": "Please visit ER immediately for chest pain evaluation.",
            "reviewer_id": "dr.wilson",
        }
        
        response = client.post(
            "/api/v1/escalations/esc-001/review",
            json=review_data,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "approved"
        assert "modified_response" in data

    def test_submit_review_not_found(self, client, mock_escalation_store):
        """Should return 404 for non-existent escalation."""
        mock_escalation_store.get_by_id.return_value = None
        
        review_data = {
            "decision": "approve",
            "notes": "Test",
            "reviewer_id": "dr.test",
        }
        
        response = client.post(
            "/api/v1/escalations/non-existent/review",
            json=review_data,
        )
        
        assert response.status_code == 404

    def test_submit_review_invalid_decision(self, client, mock_escalation_store):
        """Should return 422 for invalid decision."""
        review_data = {
            "decision": "invalid_decision",
            "notes": "Test",
            "reviewer_id": "dr.test",
        }
        
        response = client.post(
            "/api/v1/escalations/esc-001/review",
            json=review_data,
        )
        
        assert response.status_code == 422

    def test_submit_review_missing_notes(self, client, mock_escalation_store):
        """Should require notes for review."""
        review_data = {
            "decision": "approve",
            "reviewer_id": "dr.test",
        }
        
        response = client.post(
            "/api/v1/escalations/esc-001/review",
            json=review_data,
        )
        
        assert response.status_code == 422

    # =========================================================================
    # GET /api/v1/escalations/stats - Escalation Statistics
    # =========================================================================
    def test_get_escalation_stats(self, client, mock_escalation_store):
        """Should return escalation statistics."""
        response = client.get("/api/v1/escalations/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_pending" in data
        assert "total_in_review" in data
        assert "total_approved_today" in data
        assert "total_rejected_today" in data
        assert data["total_pending"] == 5

    # =========================================================================
    # POST /api/v1/escalations - Create Escalation (Internal)
    # =========================================================================
    def test_create_escalation(self, client, mock_escalation_store):
        """Should create new escalation."""
        mock_escalation_store.create = MagicMock(return_value={
            "id": "esc-003",
            "request_id": "req-789",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": "safety_concern",
            "priority": "high",
            "status": "pending",
        })
        
        escalation_data = {
            "request_id": "req-789",
            "reason": "safety_concern",
            "priority": "high",
            "patient_id": "patient-003",
            "original_message": "I'm having thoughts of self-harm",
            "communication_result": {
                "response": "Please call 988 Suicide & Crisis Lifeline immediately.",
                "triage_level": "EMERGENCY",
            },
        }
        
        response = client.post(
            "/api/v1/escalations",
            json=escalation_data,
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "esc-003"
        assert data["status"] == "pending"


class TestEscalationStore:
    """Test suite for escalation storage layer."""

    @pytest.fixture
    def store(self):
        """Create in-memory escalation store."""
        from medai_compass.utils.escalation_store import EscalationStore
        return EscalationStore(use_memory=True)

    def test_create_escalation(self, store):
        """Should create and store escalation."""
        escalation = store.create(
            request_id="req-001",
            reason="low_confidence",
            priority="medium",
            patient_id="patient-001",
        )
        
        assert escalation["id"] is not None
        assert escalation["status"] == "pending"
        assert escalation["reason"] == "low_confidence"

    def test_list_pending(self, store):
        """Should list pending escalations."""
        # Create test escalations
        store.create(request_id="req-001", reason="low_confidence", priority="medium")
        store.create(request_id="req-002", reason="critical_finding", priority="high")
        
        pending = store.list_pending()
        
        assert len(pending) == 2
        assert all(e["status"] == "pending" for e in pending)

    def test_list_pending_with_filters(self, store):
        """Should filter pending by priority and reason."""
        store.create(request_id="req-001", reason="low_confidence", priority="medium")
        store.create(request_id="req-002", reason="critical_finding", priority="high")
        
        high_priority = store.list_pending(priority="high")
        assert len(high_priority) == 1
        assert high_priority[0]["priority"] == "high"
        
        critical = store.list_pending(reason="critical_finding")
        assert len(critical) == 1
        assert critical[0]["reason"] == "critical_finding"

    def test_get_by_id(self, store):
        """Should retrieve escalation by ID."""
        created = store.create(request_id="req-001", reason="low_confidence", priority="medium")
        
        retrieved = store.get_by_id(created["id"])
        
        assert retrieved is not None
        assert retrieved["id"] == created["id"]

    def test_submit_review(self, store):
        """Should update escalation with review decision."""
        created = store.create(request_id="req-001", reason="low_confidence", priority="medium")
        
        reviewed = store.submit_review(
            escalation_id=created["id"],
            decision="approve",
            reviewer_id="dr.test",
            notes="Confirmed finding.",
        )
        
        assert reviewed["status"] == "approved"
        assert reviewed["reviewed_by"] == "dr.test"
        assert reviewed["review_notes"] == "Confirmed finding."

    def test_get_stats(self, store):
        """Should return correct statistics."""
        store.create(request_id="req-001", reason="low_confidence", priority="medium")
        store.create(request_id="req-002", reason="critical_finding", priority="high")
        
        stats = store.get_stats()
        
        assert stats["total_pending"] == 2
        assert stats["total_in_review"] == 0

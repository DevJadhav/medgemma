"""
Tests for RAG API endpoints.

TDD approach: Tests for RAG API integration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


class TestRAGAPIHealth:
    """Tests for RAG API health endpoints."""

    def test_rag_health_endpoint_exists(self):
        """Verify RAG health endpoint exists."""
        from medai_compass.api.main import app

        client = TestClient(app)
        response = client.get("/rag/health")

        # Should return 200 even if ChromaDB is not connected
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "rag_enabled" in data

    def test_rag_stats_endpoint(self):
        """Verify RAG stats endpoint works."""
        from medai_compass.api.main import app

        client = TestClient(app)
        response = client.get("/rag/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "embedding_dimension" in data
        assert "rag_enabled" in data


class TestRAGDocumentManagement:
    """Tests for RAG document management endpoints."""

    def test_add_single_document(self):
        """Verify single document can be added."""
        from medai_compass.api.main import app

        client = TestClient(app)

        doc_data = {
            "content": "Diabetes is a chronic condition affecting blood sugar regulation.",
            "metadata": {"source": "test", "topic": "diabetes"}
        }

        response = client.post("/rag/documents", json=doc_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "doc_id" in data

    def test_add_document_with_custom_id(self):
        """Verify document can be added with custom ID."""
        from medai_compass.api.main import app

        client = TestClient(app)

        doc_data = {
            "content": "Hypertension treatment guidelines.",
            "metadata": {"source": "guidelines"},
            "doc_id": "custom-doc-123"
        }

        response = client.post("/rag/documents", json=doc_data)

        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == "custom-doc-123"

    def test_add_documents_batch(self):
        """Verify batch document addition works."""
        from medai_compass.api.main import app

        client = TestClient(app)

        batch_data = {
            "documents": [
                {"content": "Document 1 about cardiology.", "metadata": {"topic": "cardiology"}},
                {"content": "Document 2 about neurology.", "metadata": {"topic": "neurology"}},
                {"content": "Document 3 about oncology.", "metadata": {"topic": "oncology"}},
            ]
        }

        response = client.post("/rag/documents/batch", json=batch_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["documents_added"] == 3


class TestRAGQuery:
    """Tests for RAG query endpoint."""

    def test_query_endpoint_exists(self):
        """Verify RAG query endpoint exists."""
        from medai_compass.api.main import app

        client = TestClient(app)

        # First add a document
        doc_data = {
            "content": "Aspirin is commonly used for pain relief and blood thinning.",
            "metadata": {"source": "test"}
        }
        client.post("/rag/documents", json=doc_data)

        # Then query
        query_data = {
            "query": "What is aspirin used for?",
            "top_k": 3,
            "include_sources": True
        }

        response = client.post("/rag/query", json=query_data)

        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "answer" in data
        assert "confidence" in data
        assert "processing_time_ms" in data

    def test_query_with_filter(self):
        """Verify query with metadata filter works."""
        from medai_compass.api.main import app

        client = TestClient(app)

        # Add documents with different metadata
        client.post("/rag/documents", json={
            "content": "Pediatric diabetes management.",
            "metadata": {"specialty": "pediatrics"}
        })
        client.post("/rag/documents", json={
            "content": "Adult diabetes management.",
            "metadata": {"specialty": "internal_medicine"}
        })

        # Query with filter
        query_data = {
            "query": "diabetes management",
            "top_k": 5,
            "filter_metadata": {"specialty": "pediatrics"}
        }

        response = client.post("/rag/query", json=query_data)

        assert response.status_code == 200

    def test_query_includes_disclaimer(self):
        """Verify query response includes medical disclaimer."""
        from medai_compass.api.main import app

        client = TestClient(app)

        query_data = {
            "query": "How to treat hypertension?",
            "top_k": 3
        }

        response = client.post("/rag/query", json=query_data)

        assert response.status_code == 200
        data = response.json()
        assert "disclaimer" in data


class TestRAGRetrieval:
    """Tests for document retrieval endpoint."""

    def test_retrieve_endpoint_exists(self):
        """Verify retrieval endpoint exists."""
        from medai_compass.api.main import app

        client = TestClient(app)

        retrieve_data = {
            "query": "diabetes treatment",
            "top_k": 5,
            "min_relevance": 0.0
        }

        response = client.post("/rag/retrieve", json=retrieve_data)

        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "query" in data
        assert "results" in data
        assert "total_results" in data

    def test_retrieve_with_min_relevance(self):
        """Verify retrieval respects minimum relevance threshold."""
        from medai_compass.api.main import app

        client = TestClient(app)

        # Add a document
        client.post("/rag/documents", json={
            "content": "Insulin therapy for type 1 diabetes.",
            "metadata": {"topic": "diabetes"}
        })

        retrieve_data = {
            "query": "insulin diabetes",
            "top_k": 10,
            "min_relevance": 0.5
        }

        response = client.post("/rag/retrieve", json=retrieve_data)

        assert response.status_code == 200
        data = response.json()

        # All results should have relevance >= min_relevance
        for result in data["results"]:
            assert result["relevance_score"] >= 0.5


class TestRAGRouterIntegration:
    """Tests for RAG router integration with main app."""

    def test_rag_router_registered(self):
        """Verify RAG router is registered with the app."""
        from medai_compass.api.main import app

        # Check that RAG routes exist
        routes = [route.path for route in app.routes]

        assert "/rag/health" in routes
        assert "/rag/stats" in routes
        assert "/rag/documents" in routes
        assert "/rag/query" in routes
        assert "/rag/retrieve" in routes

    def test_rag_endpoints_have_correct_methods(self):
        """Verify RAG endpoints have correct HTTP methods."""
        from medai_compass.api.main import app

        client = TestClient(app)

        # GET endpoints
        assert client.get("/rag/health").status_code == 200
        assert client.get("/rag/stats").status_code == 200

        # POST endpoints should not return 405
        assert client.post("/rag/documents", json={"content": "test"}).status_code != 405
        assert client.post("/rag/query", json={"query": "test"}).status_code != 405
        assert client.post("/rag/retrieve", json={"query": "test"}).status_code != 405


class TestRAGAPIValidation:
    """Tests for RAG API input validation."""

    def test_query_validation_empty_query(self):
        """Verify empty query is rejected."""
        from medai_compass.api.main import app

        client = TestClient(app)

        response = client.post("/rag/query", json={
            "query": "",
            "top_k": 3
        })

        # Empty string should be rejected by Pydantic
        assert response.status_code == 422

    def test_query_validation_top_k_bounds(self):
        """Verify top_k bounds are enforced."""
        from medai_compass.api.main import app

        client = TestClient(app)

        # top_k too high
        response = client.post("/rag/query", json={
            "query": "test query",
            "top_k": 100  # Max is 10
        })
        assert response.status_code == 422

        # top_k too low
        response = client.post("/rag/query", json={
            "query": "test query",
            "top_k": 0  # Min is 1
        })
        assert response.status_code == 422

    def test_document_validation_empty_content(self):
        """Verify empty document content is rejected."""
        from medai_compass.api.main import app

        client = TestClient(app)

        response = client.post("/rag/documents", json={
            "content": ""
        })

        # Empty content should be rejected
        assert response.status_code == 422


class TestRAGAPIPerformance:
    """Tests for RAG API performance characteristics."""

    def test_query_returns_processing_time(self):
        """Verify query response includes processing time."""
        from medai_compass.api.main import app

        client = TestClient(app)

        response = client.post("/rag/query", json={
            "query": "test query",
            "top_k": 3
        })

        assert response.status_code == 200
        data = response.json()
        assert "processing_time_ms" in data
        assert isinstance(data["processing_time_ms"], (int, float))
        assert data["processing_time_ms"] >= 0

    def test_batch_returns_processing_time(self):
        """Verify batch operation returns processing time."""
        from medai_compass.api.main import app

        client = TestClient(app)

        response = client.post("/rag/documents/batch", json={
            "documents": [
                {"content": "Doc 1"},
                {"content": "Doc 2"}
            ]
        })

        assert response.status_code == 200
        data = response.json()
        assert "processing_time_ms" in data

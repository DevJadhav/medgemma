"""
RAG (Retrieval-Augmented Generation) API endpoints.

Provides endpoints for:
- Knowledge base management (add/remove documents)
- RAG-powered query processing
- Document retrieval
"""

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

from medai_compass.rag.pipeline import MedicalRAGPipeline, RAGResponse
from medai_compass.rag.retriever import MedicalRetriever
from medai_compass.rag.vector_store import Document, InMemoryVectorStore
from medai_compass.rag.embeddings import MedicalEmbeddings
from medai_compass.rag.loaders import TextFileLoader, JSONLoader, MarkdownLoader

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])

# Global RAG pipeline instance (initialized on startup)
_rag_pipeline: Optional[MedicalRAGPipeline] = None
_embeddings: Optional[MedicalEmbeddings] = None


# =============================================================================
# Pydantic Models
# =============================================================================

class DocumentInput(BaseModel):
    """Input for adding a document."""
    content: str = Field(..., min_length=1, description="Document text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    doc_id: Optional[str] = Field(None, description="Optional document ID")


class DocumentBatchInput(BaseModel):
    """Input for adding multiple documents."""
    documents: List[DocumentInput] = Field(..., description="List of documents")


class QueryRequest(BaseModel):
    """RAG query request."""
    query: str = Field(..., min_length=1, description="User query")
    top_k: int = Field(3, ge=1, le=10, description="Number of documents to retrieve")
    include_sources: bool = Field(True, description="Include source documents in response")
    filter_metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    max_tokens: int = Field(500, ge=50, le=2000, description="Maximum response tokens")


class RetrieveRequest(BaseModel):
    """Document retrieval request (without generation)."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    filter_metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    min_relevance: float = Field(0.3, ge=0.0, le=1.0, description="Minimum relevance score")


class DocumentResponse(BaseModel):
    """Response for document operations."""
    doc_id: str
    status: str
    message: str


class RAGQueryResponse(BaseModel):
    """RAG query response."""
    request_id: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    disclaimer: Optional[str] = None
    processing_time_ms: float


class RetrievalResponse(BaseModel):
    """Document retrieval response."""
    request_id: str
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    processing_time_ms: float


class KnowledgeBaseStats(BaseModel):
    """Knowledge base statistics."""
    total_documents: int
    embedding_dimension: int
    rag_enabled: bool
    chromadb_connected: bool


# =============================================================================
# Initialization
# =============================================================================

def get_rag_pipeline() -> MedicalRAGPipeline:
    """Get or create RAG pipeline instance."""
    global _rag_pipeline, _embeddings

    if _rag_pipeline is None:
        logger.info("Initializing RAG pipeline...")

        _embeddings = MedicalEmbeddings()

        # Check for ChromaDB connection
        chromadb_host = os.getenv("CHROMADB_HOST")
        chromadb_port = os.getenv("CHROMADB_PORT", "8000")

        if chromadb_host:
            try:
                # Try to use ChromaDB for persistent storage
                import chromadb
                from chromadb.config import Settings

                client = chromadb.HttpClient(
                    host=chromadb_host,
                    port=int(chromadb_port),
                    settings=Settings(anonymized_telemetry=False)
                )

                collection_name = os.getenv("RAG_COLLECTION_NAME", "medical_knowledge")
                collection = client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": "Medical knowledge base for RAG"}
                )

                logger.info(f"Connected to ChromaDB at {chromadb_host}:{chromadb_port}")

                # Create ChromaDB-backed vector store
                vector_store = ChromaDBVectorStore(
                    collection=collection,
                    embeddings=_embeddings
                )

            except Exception as e:
                logger.warning(f"Failed to connect to ChromaDB: {e}. Using in-memory store.")
                vector_store = InMemoryVectorStore(dimension=_embeddings.dimension)
        else:
            logger.info("ChromaDB not configured, using in-memory vector store")
            vector_store = InMemoryVectorStore(dimension=_embeddings.dimension)

        retriever = MedicalRetriever(
            embeddings=_embeddings,
            vector_store=vector_store,
            min_relevance_score=0.3
        )

        _rag_pipeline = MedicalRAGPipeline(
            retriever=retriever,
            apply_guardrails=True,
            include_disclaimer=True
        )

        logger.info("RAG pipeline initialized successfully")

    return _rag_pipeline


class ChromaDBVectorStore:
    """Vector store backed by ChromaDB."""

    def __init__(self, collection, embeddings: MedicalEmbeddings):
        self._collection = collection
        self._embeddings = embeddings
        self._dimension = embeddings.dimension

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to ChromaDB."""
        ids = []
        embeddings = []
        metadatas = []
        contents = []

        for doc in documents:
            ids.append(doc.id)

            if doc.embedding is None:
                doc.embedding = self._embeddings.embed_text(doc.content)

            embeddings.append(doc.embedding)
            metadatas.append(doc.metadata or {})
            contents.append(doc.content)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=contents
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0
    ):
        """Search ChromaDB for similar documents."""
        from medai_compass.rag.vector_store import SearchResult

        where_filter = filter_metadata if filter_metadata else None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity score
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1.0 / (1.0 + distance)  # Convert distance to similarity

                if score >= min_score:
                    doc = Document(
                        id=doc_id,
                        content=results["documents"][0][i] if results["documents"] else "",
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {}
                    )
                    search_results.append(SearchResult(
                        document=doc,
                        score=score,
                        rank=i + 1
                    ))

        return search_results

    def count(self) -> int:
        """Get document count."""
        return self._collection.count()

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document."""
        try:
            self._collection.delete(ids=[doc_id])
            return True
        except Exception:
            return False


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/health", response_model=Dict[str, Any])
async def rag_health():
    """Check RAG service health."""
    try:
        pipeline = get_rag_pipeline()
        chromadb_connected = os.getenv("CHROMADB_HOST") is not None

        return {
            "status": "healthy",
            "rag_enabled": True,
            "document_count": pipeline.document_count,
            "chromadb_connected": chromadb_connected
        }
    except Exception as e:
        logger.error(f"RAG health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/stats", response_model=KnowledgeBaseStats)
async def get_knowledge_base_stats():
    """Get knowledge base statistics."""
    pipeline = get_rag_pipeline()

    return KnowledgeBaseStats(
        total_documents=pipeline.document_count,
        embedding_dimension=_embeddings.dimension if _embeddings else 384,
        rag_enabled=True,
        chromadb_connected=os.getenv("CHROMADB_HOST") is not None
    )


@router.post("/documents", response_model=DocumentResponse)
async def add_document(doc_input: DocumentInput):
    """Add a single document to the knowledge base."""
    start_time = time.time()

    try:
        pipeline = get_rag_pipeline()

        doc_id = doc_input.doc_id or str(uuid.uuid4())

        document = Document(
            id=doc_id,
            content=doc_input.content,
            metadata=doc_input.metadata
        )

        pipeline.add_document(document)

        logger.info(f"Added document {doc_id} to knowledge base")

        return DocumentResponse(
            doc_id=doc_id,
            status="success",
            message="Document added successfully"
        )

    except Exception as e:
        logger.error(f"Failed to add document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add document: {str(e)}"
        )


@router.post("/documents/batch", response_model=Dict[str, Any])
async def add_documents_batch(batch_input: DocumentBatchInput):
    """Add multiple documents to the knowledge base."""
    start_time = time.time()

    try:
        pipeline = get_rag_pipeline()

        documents = []
        for doc_input in batch_input.documents:
            doc_id = doc_input.doc_id or str(uuid.uuid4())
            documents.append(Document(
                id=doc_id,
                content=doc_input.content,
                metadata=doc_input.metadata
            ))

        pipeline.add_documents(documents)

        processing_time = (time.time() - start_time) * 1000

        logger.info(f"Added {len(documents)} documents to knowledge base")

        return {
            "status": "success",
            "documents_added": len(documents),
            "total_documents": pipeline.document_count,
            "processing_time_ms": processing_time
        }

    except Exception as e:
        logger.error(f"Failed to add documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add documents: {str(e)}"
        )


@router.post("/query", response_model=RAGQueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG pipeline for an answer with retrieved context."""
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        pipeline = get_rag_pipeline()

        response = await pipeline.generate(
            query=request.query,
            max_tokens=request.max_tokens,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata
        )

        processing_time = (time.time() - start_time) * 1000

        # Build sources list
        sources = []
        if request.include_sources:
            for source in response.sources:
                sources.append({
                    "doc_id": source.doc_id,
                    "content": source.content[:500] + "..." if len(source.content) > 500 else source.content,
                    "relevance_score": source.relevance_score,
                    "metadata": source.metadata
                })

        # Build citations list
        citations = [
            {
                "doc_id": c.doc_id,
                "source": c.source,
                "relevance": c.relevance,
                "snippet": c.snippet
            }
            for c in response.citations
        ]

        logger.info(f"RAG query processed: {request_id}, sources={len(sources)}")

        return RAGQueryResponse(
            request_id=request_id,
            answer=response.answer,
            confidence=response.confidence,
            sources=sources,
            citations=citations,
            disclaimer=response.disclaimer,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query failed: {str(e)}"
        )


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_documents(request: RetrieveRequest):
    """Retrieve relevant documents without generating an answer."""
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        pipeline = get_rag_pipeline()

        # Get the retriever from the pipeline
        retriever = pipeline._retriever

        results = await retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata
        )

        # Filter by minimum relevance
        filtered_results = [
            r for r in results
            if r.relevance_score >= request.min_relevance
        ]

        processing_time = (time.time() - start_time) * 1000

        result_list = [
            {
                "doc_id": r.doc_id,
                "content": r.content,
                "relevance_score": r.relevance_score,
                "metadata": r.metadata
            }
            for r in filtered_results
        ]

        return RetrievalResponse(
            request_id=request_id,
            query=request.query,
            results=result_list,
            total_results=len(result_list),
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}"
        )


@router.delete("/documents/{doc_id}", response_model=DocumentResponse)
async def delete_document(doc_id: str):
    """Delete a document from the knowledge base."""
    try:
        pipeline = get_rag_pipeline()

        # Access the vector store through the retriever
        vector_store = pipeline._retriever._vector_store

        if hasattr(vector_store, 'remove_document'):
            success = vector_store.remove_document(doc_id)
            if success:
                return DocumentResponse(
                    doc_id=doc_id,
                    status="success",
                    message="Document deleted successfully"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document not found: {doc_id}"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Document deletion not supported by current vector store"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


@router.post("/ingest/file", response_model=Dict[str, Any])
async def ingest_file(
    file_path: str,
    file_type: str = "text",
    chunk_size: int = 1000,
    chunk_overlap: int = 100
):
    """Ingest a file into the knowledge base."""
    try:
        pipeline = get_rag_pipeline()

        if file_type == "text":
            loader = TextFileLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif file_type == "json":
            loader = JSONLoader()
        elif file_type == "markdown":
            loader = MarkdownLoader()
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file_type}"
            )

        documents = loader.load(file_path)
        pipeline.add_documents(documents)

        return {
            "status": "success",
            "file_path": file_path,
            "documents_created": len(documents),
            "total_documents": pipeline.document_count
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {file_path}"
        )
    except Exception as e:
        logger.error(f"File ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File ingestion failed: {str(e)}"
        )

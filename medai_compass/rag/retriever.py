"""
Document retriever for RAG pipeline.

Provides high-level retrieval interface with relevance scoring.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from medai_compass.rag.embeddings import MedicalEmbeddings
from medai_compass.rag.vector_store import Document, InMemoryVectorStore, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Result from document retrieval.

    Attributes:
        doc_id: Document identifier
        content: Document content
        metadata: Document metadata
        relevance_score: Relevance score (0-1)
    """
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    relevance_score: float


class MedicalRetriever:
    """
    Medical document retriever with relevance filtering.

    Combines embedding model with vector store for retrieval.
    """

    def __init__(
        self,
        embeddings: Optional[MedicalEmbeddings] = None,
        vector_store: Optional[InMemoryVectorStore] = None,
        min_relevance_score: float = 0.3,
        max_results: int = 5
    ):
        """
        Initialize retriever.

        Args:
            embeddings: Embedding model (created if not provided)
            vector_store: Vector store (created if not provided)
            min_relevance_score: Minimum relevance threshold
            max_results: Maximum results to return
        """
        self._embeddings = embeddings or MedicalEmbeddings()
        self._vector_store = vector_store or InMemoryVectorStore(
            dimension=self._embeddings.dimension
        )
        self._min_relevance = min_relevance_score
        self._max_results = max_results

        logger.info(
            f"Initialized MedicalRetriever: min_score={min_relevance_score}, "
            f"max_results={max_results}"
        )

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the retriever.

        Computes embeddings if not already present.

        Args:
            documents: Documents to add
        """
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self._embeddings.embed_text(doc.content)

        self._vector_store.add_documents(documents)

    def add_document(self, document: Document) -> None:
        """Add a single document."""
        self.add_documents([document])

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of results (default: max_results)
            filter_metadata: Optional metadata filters

        Returns:
            List of RetrievalResult objects
        """
        if top_k is None:
            top_k = self._max_results

        # Embed query
        query_embedding = self._embeddings.embed_text(query)

        # Search vector store
        search_results = self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
            min_score=self._min_relevance
        )

        # Convert to retrieval results
        results = []
        for sr in search_results:
            results.append(RetrievalResult(
                doc_id=sr.document.id,
                content=sr.document.content,
                metadata=sr.document.metadata,
                relevance_score=sr.score
            ))

        logger.debug(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        return results

    def retrieve_sync(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Synchronous version of retrieve.

        Args:
            query: Search query
            top_k: Number of results
            filter_metadata: Optional metadata filters

        Returns:
            List of RetrievalResult objects
        """
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.retrieve(query, top_k, filter_metadata)
        )

    @property
    def document_count(self) -> int:
        """Get number of documents in retriever."""
        return self._vector_store.count()

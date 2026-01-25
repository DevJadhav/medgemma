"""
Vector store for document storage and retrieval.

Provides in-memory vector store with similarity search.
Can be extended to use external vector databases (Pinecone, Weaviate, etc.)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    Document for storage in vector store.

    Attributes:
        id: Unique document identifier
        content: Document text content
        metadata: Additional metadata (source, date, etc.)
        embedding: Pre-computed embedding vector
    """
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """
    Search result from vector store.

    Attributes:
        document: The matched document
        score: Similarity score (higher is better)
        rank: Result rank (1-indexed)
    """
    document: Document
    score: float
    rank: int = 0


class InMemoryVectorStore:
    """
    In-memory vector store for document retrieval.

    Simple but effective for small to medium knowledge bases.
    For production with large datasets, use Pinecone/Weaviate/Chroma.
    """

    def __init__(self, dimension: int = 384):
        """
        Initialize vector store.

        Args:
            dimension: Expected embedding dimension
        """
        self._dimension = dimension
        self._documents: Dict[str, Document] = {}
        self._embeddings: Dict[str, np.ndarray] = {}

        logger.info(f"Initialized in-memory vector store with dimension {dimension}")

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the store.

        Args:
            documents: List of documents with embeddings
        """
        for doc in documents:
            if doc.embedding is None:
                logger.warning(f"Document {doc.id} has no embedding, skipping")
                continue

            if len(doc.embedding) != self._dimension:
                logger.warning(
                    f"Document {doc.id} embedding dimension mismatch: "
                    f"{len(doc.embedding)} != {self._dimension}"
                )
                continue

            self._documents[doc.id] = doc
            self._embeddings[doc.id] = np.array(doc.embedding)

        logger.info(f"Added {len(documents)} documents, total: {len(self._documents)}")

    def add_document(self, document: Document) -> None:
        """Add a single document."""
        self.add_documents([document])

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the store.

        Args:
            doc_id: Document ID to remove

        Returns:
            True if removed, False if not found
        """
        if doc_id in self._documents:
            del self._documents[doc_id]
            del self._embeddings[doc_id]
            return True
        return False

    def count(self) -> int:
        """Get number of documents in store."""
        return len(self._documents)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            min_score: Minimum similarity score threshold

        Returns:
            List of SearchResult objects
        """
        if not self._embeddings:
            return []

        query_vec = np.array(query_embedding)

        # Compute similarities
        scores = []
        for doc_id, doc_embedding in self._embeddings.items():
            doc = self._documents[doc_id]

            # Apply metadata filter
            if filter_metadata:
                if not self._matches_filter(doc.metadata, filter_metadata):
                    continue

            # Cosine similarity
            similarity = float(
                np.dot(query_vec, doc_embedding) /
                (np.linalg.norm(query_vec) * np.linalg.norm(doc_embedding))
            )

            if similarity >= min_score:
                scores.append((doc_id, similarity))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for rank, (doc_id, score) in enumerate(scores[:top_k], 1):
            results.append(SearchResult(
                document=self._documents[doc_id],
                score=score,
                rank=rank
            ))

        return results

    def _matches_filter(
        self,
        metadata: Dict[str, Any],
        filter_metadata: Dict[str, Any]
    ) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_metadata.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self._documents.get(doc_id)

    def get_all_documents(self) -> List[Document]:
        """Get all documents."""
        return list(self._documents.values())

    def clear(self) -> None:
        """Clear all documents from store."""
        self._documents.clear()
        self._embeddings.clear()
        logger.info("Cleared vector store")

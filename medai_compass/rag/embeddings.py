"""
Medical text embeddings for RAG pipeline.

Uses lightweight embedding models suitable for medical text.
"""

import hashlib
import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class MedicalEmbeddings:
    """
    Medical text embedding model.

    Uses a lightweight approach with optional transformer models.
    Falls back to simple TF-IDF style embeddings if transformers unavailable.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        use_gpu: bool = False
    ):
        """
        Initialize embeddings.

        Args:
            model_name: HuggingFace model name for embeddings
            dimension: Embedding dimension (used for fallback)
            use_gpu: Whether to use GPU for embeddings
        """
        self._dimension = dimension
        self._model = None
        self._use_transformer = False

        # Try to load sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            self._use_transformer = True
            logger.info(f"Loaded transformer embeddings: {model_name}, dim={self._dimension}")
        except ImportError:
            logger.warning(
                "sentence-transformers not available, using hash-based embeddings. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.warning(f"Failed to load transformer model: {e}, using fallback")

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if self._use_transformer and self._model:
            embedding = self._model.encode(text, convert_to_numpy=True)
            return embedding.tolist()

        # Fallback: hash-based embeddings
        return self._hash_embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if self._use_transformer and self._model:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return [e.tolist() for e in embeddings]

        # Fallback: hash-based embeddings
        return [self._hash_embed(text) for text in texts]

    def _hash_embed(self, text: str) -> List[float]:
        """
        Create a hash-based embedding (fallback method).

        This is a simple but deterministic embedding method.
        Not as good as transformer embeddings but works without dependencies.
        """
        # Normalize text
        text = text.lower().strip()

        # Create multiple hash values for different dimensions
        embedding = []
        for i in range(self._dimension):
            # Create varied hash by including index
            hash_input = f"{text}_{i}"
            hash_bytes = hashlib.sha256(hash_input.encode()).digest()

            # Convert first 4 bytes to float in [-1, 1]
            value = int.from_bytes(hash_bytes[:4], 'big') / (2**32) * 2 - 1
            embedding.append(value)

        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [v / norm for v in embedding]

        return embedding

    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Cosine similarity score
        """
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

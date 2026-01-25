"""
RAG (Retrieval-Augmented Generation) pipeline for medical AI.

Combines retrieval with generation for context-aware responses.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from medai_compass.rag.embeddings import MedicalEmbeddings
from medai_compass.rag.retriever import MedicalRetriever, RetrievalResult
from medai_compass.rag.vector_store import Document, InMemoryVectorStore

logger = logging.getLogger(__name__)


# Medical disclaimer
MEDICAL_DISCLAIMER = (
    "This information is for educational purposes only and should not be "
    "considered medical advice. Always consult with a qualified healthcare "
    "provider for medical decisions."
)


@dataclass
class Citation:
    """
    Citation for a source document.

    Attributes:
        doc_id: Document identifier
        source: Source name/title
        relevance: Relevance score
        snippet: Relevant text snippet
    """
    doc_id: str
    source: str
    relevance: float
    snippet: str = ""


@dataclass
class RAGResponse:
    """
    Response from RAG pipeline.

    Attributes:
        answer: Generated answer text
        sources: Source documents used
        citations: Citations for the answer
        confidence: Confidence score (0-1)
        disclaimer: Medical disclaimer if applicable
        metadata: Additional metadata
    """
    answer: str
    sources: List[RetrievalResult] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.0
    disclaimer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MedicalRAGPipeline:
    """
    RAG pipeline for medical question answering.

    Retrieves relevant documents and generates context-aware responses.
    """

    def __init__(
        self,
        retriever: Optional[MedicalRetriever] = None,
        model_wrapper: Optional[Any] = None,
        max_context_length: int = 2000,
        min_relevance: float = 0.3,
        apply_guardrails: bool = True,
        include_disclaimer: bool = True
    ):
        """
        Initialize RAG pipeline.

        Args:
            retriever: Document retriever (created if not provided)
            model_wrapper: LLM wrapper for generation (uses template if not provided)
            max_context_length: Maximum context length for generation
            min_relevance: Minimum relevance score for sources
            apply_guardrails: Whether to apply PHI guardrails
            include_disclaimer: Whether to include medical disclaimer
        """
        self._retriever = retriever or MedicalRetriever(min_relevance_score=min_relevance)
        self._model = model_wrapper
        self._max_context_length = max_context_length
        self._apply_guardrails = apply_guardrails
        self._include_disclaimer = include_disclaimer

        logger.info(
            f"Initialized MedicalRAGPipeline: max_context={max_context_length}, "
            f"guardrails={'enabled' if apply_guardrails else 'disabled'}"
        )

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: Documents to add
        """
        self._retriever.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to RAG pipeline")

    def add_document(self, document: Document) -> None:
        """Add a single document."""
        self.add_documents([document])

    async def generate(
        self,
        query: str,
        max_tokens: int = 500,
        top_k: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """
        Generate a response using RAG.

        Args:
            query: User query
            max_tokens: Maximum tokens in response
            top_k: Number of documents to retrieve
            filter_metadata: Optional metadata filters

        Returns:
            RAGResponse with answer and sources
        """
        # Retrieve relevant documents
        sources = await self._retriever.retrieve(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata
        )

        # Build context from sources
        context = self._build_context(sources)

        # Apply PHI guardrails if enabled
        if self._apply_guardrails:
            context = self._apply_phi_guardrails(context)

        # Generate response
        if self._model:
            # Use actual LLM
            answer = await self._generate_with_model(query, context, max_tokens)
        else:
            # Use template-based response
            answer = self._generate_template_response(query, context, sources)

        # Calculate confidence
        confidence = self._calculate_confidence(sources)

        # Build citations
        citations = self._build_citations(sources)

        # Build response
        response = RAGResponse(
            answer=answer,
            sources=sources,
            citations=citations,
            confidence=confidence,
            disclaimer=MEDICAL_DISCLAIMER if self._include_disclaimer else None,
            metadata={
                "query": query,
                "num_sources": len(sources),
                "context_length": len(context),
            }
        )

        return response

    def _build_context(self, sources: List[RetrievalResult]) -> str:
        """Build context string from sources."""
        if not sources:
            return ""

        context_parts = []
        total_length = 0

        for source in sources:
            # Truncate if needed
            content = source.content
            if total_length + len(content) > self._max_context_length:
                remaining = self._max_context_length - total_length
                if remaining > 100:
                    content = content[:remaining] + "..."
                else:
                    break

            context_parts.append(f"[Source: {source.doc_id}]\n{content}")
            total_length += len(content)

        return "\n\n".join(context_parts)

    def _apply_phi_guardrails(self, text: str) -> str:
        """Apply PHI detection and masking."""
        try:
            from medai_compass.guardrails.phi_detection import mask_phi
            return mask_phi(text)
        except ImportError:
            logger.warning("PHI detection not available")
            return text

    async def _generate_with_model(
        self,
        query: str,
        context: str,
        max_tokens: int
    ) -> str:
        """Generate response using LLM."""
        prompt = f"""Based on the following medical information, answer the question.

Context:
{context}

Question: {query}

Answer:"""

        try:
            response = await self._model.generate(prompt, max_tokens=max_tokens)
            return response
        except Exception as e:
            logger.error(f"Model generation failed: {e}")
            return self._generate_template_response(query, context, [])

    def _generate_template_response(
        self,
        query: str,
        context: str,
        sources: List[RetrievalResult]
    ) -> str:
        """Generate template-based response when no LLM available."""
        if not sources:
            return (
                "I don't have specific information about that topic in my knowledge base. "
                "Please consult with a healthcare professional for accurate medical information."
            )

        # Extract key information from sources
        response_parts = [
            "Based on the available medical information:\n"
        ]

        for i, source in enumerate(sources[:3], 1):
            # Truncate long content
            content = source.content
            if len(content) > 200:
                content = content[:200] + "..."
            response_parts.append(f"\n{i}. {content}")

        response_parts.append(
            "\n\nPlease note that this information is for reference only. "
            "Consult a healthcare provider for personalized medical advice."
        )

        return "".join(response_parts)

    def _calculate_confidence(self, sources: List[RetrievalResult]) -> float:
        """Calculate confidence score based on sources."""
        if not sources:
            return 0.0

        # Average relevance of top sources
        relevance_scores = [s.relevance_score for s in sources]
        avg_relevance = sum(relevance_scores) / len(relevance_scores)

        # Boost confidence if multiple high-quality sources
        high_quality_count = sum(1 for s in relevance_scores if s > 0.7)
        quality_boost = min(high_quality_count * 0.1, 0.2)

        confidence = min(avg_relevance + quality_boost, 1.0)
        return confidence

    def _build_citations(self, sources: List[RetrievalResult]) -> List[Citation]:
        """Build citations from sources."""
        citations = []

        for source in sources:
            citation = Citation(
                doc_id=source.doc_id,
                source=source.metadata.get("source", source.doc_id),
                relevance=source.relevance_score,
                snippet=source.content[:100] + "..." if len(source.content) > 100 else source.content
            )
            citations.append(citation)

        return citations

    @property
    def document_count(self) -> int:
        """Get number of documents in knowledge base."""
        return self._retriever.document_count

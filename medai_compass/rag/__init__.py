"""
RAG (Retrieval-Augmented Generation) module for MedAI Compass.

Provides:
- Medical text embeddings
- Vector store for document retrieval
- RAG pipeline for context-aware generation
- Knowledge base loaders
"""

from medai_compass.rag.embeddings import MedicalEmbeddings
from medai_compass.rag.vector_store import InMemoryVectorStore, Document, SearchResult
from medai_compass.rag.retriever import MedicalRetriever, RetrievalResult
from medai_compass.rag.pipeline import MedicalRAGPipeline, RAGResponse, Citation
from medai_compass.rag.loaders import TextFileLoader

__all__ = [
    "MedicalEmbeddings",
    "InMemoryVectorStore",
    "Document",
    "SearchResult",
    "MedicalRetriever",
    "RetrievalResult",
    "MedicalRAGPipeline",
    "RAGResponse",
    "Citation",
    "TextFileLoader",
]

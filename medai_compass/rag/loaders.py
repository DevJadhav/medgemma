"""
Document loaders for RAG knowledge base.

Provides loaders for various file formats.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from medai_compass.rag.vector_store import Document

logger = logging.getLogger(__name__)


class TextFileLoader:
    """
    Loader for text files with chunking support.

    Splits large documents into smaller chunks for better retrieval.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        encoding: str = "utf-8"
    ):
        """
        Initialize text file loader.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks for context
            encoding: File encoding
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._encoding = encoding

    def load(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Load documents from a text file.

        Args:
            file_path: Path to text file
            metadata: Additional metadata to add to documents

        Returns:
            List of Document objects
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file
        with open(path, 'r', encoding=self._encoding) as f:
            content = f.read()

        # Build base metadata
        base_metadata = {
            "source": str(path.name),
            "file_path": str(path.absolute()),
        }
        if metadata:
            base_metadata.update(metadata)

        # Chunk content
        chunks = self._chunk_text(content)

        # Create documents
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                id=f"{path.stem}_{i}_{uuid.uuid4().hex[:8]}",
                content=chunk,
                metadata={
                    **base_metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} chunks from {file_path}")
        return documents

    def load_directory(
        self,
        dir_path: str,
        pattern: str = "*.txt",
        recursive: bool = False
    ) -> List[Document]:
        """
        Load all text files from a directory.

        Args:
            dir_path: Directory path
            pattern: Glob pattern for files
            recursive: Whether to search recursively

        Returns:
            List of Document objects
        """
        path = Path(dir_path)

        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        # Find files
        if recursive:
            files = list(path.rglob(pattern))
        else:
            files = list(path.glob(pattern))

        # Load all files
        all_documents = []
        for file_path in files:
            try:
                docs = self.load(str(file_path))
                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {len(all_documents)} total chunks from {len(files)} files")
        return all_documents

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if len(text) <= self._chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            # Find end of chunk
            end = start + self._chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within last 20% of chunk
                search_start = start + int(self._chunk_size * 0.8)
                sentence_end = self._find_sentence_boundary(text, search_start, end)
                if sentence_end > 0:
                    end = sentence_end

            chunks.append(text[start:end].strip())

            # Move start with overlap
            start = end - self._chunk_overlap

        return [c for c in chunks if c]  # Filter empty chunks

    def _find_sentence_boundary(
        self,
        text: str,
        search_start: int,
        search_end: int
    ) -> int:
        """Find a sentence boundary for cleaner chunking."""
        # Look for common sentence endings
        endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']

        best_pos = -1
        for ending in endings:
            pos = text.rfind(ending, search_start, search_end)
            if pos > best_pos:
                best_pos = pos + len(ending)

        return best_pos


class JSONLoader:
    """
    Loader for JSON files containing document arrays.

    Expects JSON format:
    [
        {"id": "...", "content": "...", "metadata": {...}},
        ...
    ]
    """

    def __init__(self, content_key: str = "content", id_key: str = "id"):
        """
        Initialize JSON loader.

        Args:
            content_key: Key for document content
            id_key: Key for document ID
        """
        self._content_key = content_key
        self._id_key = id_key

    def load(self, file_path: str) -> List[Document]:
        """
        Load documents from a JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            List of Document objects
        """
        import json

        with open(file_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        documents = []
        for i, item in enumerate(data):
            content = item.get(self._content_key, "")
            doc_id = item.get(self._id_key, f"doc_{i}")

            # Build metadata from remaining fields
            metadata = {
                k: v for k, v in item.items()
                if k not in [self._content_key, self._id_key]
            }
            metadata["source"] = file_path

            documents.append(Document(
                id=doc_id,
                content=content,
                metadata=metadata
            ))

        logger.info(f"Loaded {len(documents)} documents from {file_path}")
        return documents


class MarkdownLoader:
    """
    Loader for Markdown files with section-based chunking.

    Splits on headers for logical sections.
    """

    def __init__(self, min_section_length: int = 100):
        """
        Initialize Markdown loader.

        Args:
            min_section_length: Minimum section length to keep
        """
        self._min_length = min_section_length

    def load(self, file_path: str) -> List[Document]:
        """
        Load documents from a Markdown file.

        Args:
            file_path: Path to Markdown file

        Returns:
            List of Document objects
        """
        import re

        path = Path(file_path)

        with open(path, 'r') as f:
            content = f.read()

        # Split on headers
        sections = re.split(r'\n(#{1,6}\s+.+)\n', content)

        documents = []
        current_header = None

        for i, section in enumerate(sections):
            section = section.strip()

            # Check if this is a header
            if re.match(r'^#{1,6}\s+', section):
                current_header = section.lstrip('#').strip()
                continue

            # Skip short sections
            if len(section) < self._min_length:
                continue

            doc = Document(
                id=f"{path.stem}_{i}_{uuid.uuid4().hex[:8]}",
                content=section,
                metadata={
                    "source": str(path.name),
                    "header": current_header or "Introduction",
                    "file_path": str(path.absolute()),
                }
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} sections from {file_path}")
        return documents

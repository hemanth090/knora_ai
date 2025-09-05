"""
Data models for document processing and storage.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    text: str
    size: int
    chunk_id: int


@dataclass
class ProcessedDocument:
    """Represents a processed document with its metadata."""
    file_path: str
    file_name: str
    file_type: str
    text: str
    chunks: List[DocumentChunk]
    num_chunks: int
    file_size: int


@dataclass
class SearchResult:
    """Represents a search result from the vector store."""
    file_path: str
    file_name: str
    file_type: str
    chunk_id: int
    chunk_size: int
    text: str
    similarity_score: float


@dataclass
class LLMResponse:
    """Represents a response from the LLM."""
    answer: str
    sources: List[Dict[str, Any]]
    context_used: str
    num_sources: int
    llm_type: str
    model_used: str


@dataclass
class VectorStoreStats:
    """Represents statistics from the vector store."""
    total_vectors: int
    total_documents: int
    embedding_model: str
    dimension: int
    store_path: str
    documents: List[str]
    storage_size_mb: float
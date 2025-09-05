"""
Vector store service for document embeddings and similarity search.
"""
import os
import json
import pickle
import shutil
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

import numpy as np

# Third-party imports
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(f"Missing required package: {e}")


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        pass


class FAISSVectorStore(VectorStore):
    """Professional FAISS-based vector store with enterprise features."""
    
    EMBEDDING_MODELS = {
        'all-MiniLM-L6-v2': 384,
        'all-mpnet-base-v2': 768,
        'all-distilroberta-v1': 768,
        'paraphrase-MiniLM-L6-v2': 384,
        'paraphrase-mpnet-base-v2': 768
    }
    
    def __init__(self, 
                 store_path: str = "data/vector_store",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the FAISS vector store.
        
        Args:
            store_path: Path to store the vector index and metadata
            embedding_model: Name of the sentence transformer model to use
        """
        # Generate a session ID for storage
        self.session_id = str(uuid.uuid4())
        session_store_path = f"{store_path}_{self.session_id}"
        self.store_path = Path(session_store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model_name = embedding_model
        self.dimension = self.EMBEDDING_MODELS.get(embedding_model, 384)
        
        # Initialize sentence transformer
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize FAISS index with inner product for cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Metadata storage
        self.metadata = []
        self.document_map = {}
        
        # Load existing data
        self._load_store()
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries to add
        """
        all_texts = []
        all_metadata = []
        
        for doc in documents:
            file_path = doc['file_path']
            file_name = doc['file_name']
            file_type = doc['file_type']
            
            for chunk in doc['chunks']:
                all_texts.append(chunk['text'])
                
                metadata = {
                    'file_path': file_path,
                    'file_name': file_name,
                    'file_type': file_type,
                    'chunk_id': chunk['chunk_id'],
                    'chunk_size': chunk['size'],
                    'text': chunk['text']
                }
                all_metadata.append(metadata)
        
        if not all_texts:
            return
        
        # Generate embeddings in batches
        embeddings = self._generate_embeddings_batch(all_texts)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Add metadata
        self.metadata.extend(all_metadata)
        
        # Update document map
        for doc in documents:
            doc_id = doc['file_path']
            self.document_map[doc_id] = {
                'file_name': doc['file_name'],
                'file_type': doc['file_type'],
                'num_chunks': doc['num_chunks'],
                'file_size': doc.get('file_size', 0)
            }
        
        # Persist changes
        self._save_store()
    
    def search(self, query: str, k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Query string to search for
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of search results
        """
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self._generate_embeddings_batch([query])
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < score_threshold:
                continue
            
            metadata = self.metadata[idx].copy()
            metadata['similarity_score'] = float(score)
            results.append(metadata)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'total_vectors': self.index.ntotal,
            'total_documents': len(self.document_map),
            'embedding_model': self.embedding_model_name,
            'dimension': self.dimension,
            'store_path': str(self.store_path),
            'documents': list(self.document_map.keys()),
            'storage_size_mb': self._get_storage_size()
        }
    
    def delete_document(self, file_path: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            file_path: Path of the document to delete
            
        Returns:
            True if document was deleted, False otherwise
        """
        if file_path not in self.document_map:
            return False
        
        # Find indices to keep
        indices_to_keep = []
        new_metadata = []
        
        for i, metadata in enumerate(self.metadata):
            if metadata['file_path'] != file_path:
                indices_to_keep.append(i)
                new_metadata.append(metadata)
        
        if len(indices_to_keep) == len(self.metadata):
            return False
        
        # Rebuild index
        if indices_to_keep:
            remaining_texts = [meta['text'] for meta in new_metadata]
            embeddings = self._generate_embeddings_batch(remaining_texts)
            
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Update metadata
        self.metadata = new_metadata
        del self.document_map[file_path]
        
        self._save_store()
        return True
    
    def clear_store(self) -> None:
        """Clear all documents from the vector store."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        self.document_map = {}
        
        # Remove session-specific storage directory
        if self.store_path.exists():
            shutil.rmtree(self.store_path)
    
    def _generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings in batches to manage memory.
        
        Args:
            texts: List of texts to generate embeddings for
            batch_size: Size of batches to process
            
        Returns:
            Array of embeddings
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def _get_storage_size(self) -> float:
        """
        Get storage size in MB.
        
        Returns:
            Storage size in MB
        """
        total_size = 0
        for file_path in self.store_path.glob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return round(total_size / (1024 * 1024), 2)
    
    def _save_store(self) -> None:
        """Persist vector store to disk."""
        try:
            # Save FAISS index
            index_path = self.store_path / "faiss_index.bin"
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata_path = self.store_path / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            # Save document map
            doc_map_path = self.store_path / "document_map.json"
            with open(doc_map_path, 'w') as f:
                json.dump(self.document_map, f, indent=2)
            
            # Save configuration
            config_path = self.store_path / "config.json"
            config = {
                'embedding_model': self.embedding_model_name,
                'dimension': self.dimension,
                'total_vectors': self.index.ntotal,
                'version': '2.0.0'
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            pass  # In a production environment, we would log this error
    
    def _load_store(self) -> None:
        """Load vector store from disk."""
        try:
            index_path = self.store_path / "faiss_index.bin"
            metadata_path = self.store_path / "metadata.pkl"
            doc_map_path = self.store_path / "document_map.json"
            config_path = self.store_path / "config.json"
            
            if not all(p.exists() for p in [index_path, metadata_path, doc_map_path]):
                return
            
            # Load configuration
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                if config['embedding_model'] != self.embedding_model_name:
                    pass  # In a production environment, we would log this warning
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            # Load document map
            with open(doc_map_path, 'r') as f:
                self.document_map = json.load(f)
            
        except Exception as e:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            self.document_map = {}
"""Document embedder module for creating vector representations."""

from typing import List, Dict
from ..clients.embedding_client import BaseEmbeddingClient


class DocumentEmbedder:
    """Create embeddings for document chunks."""

    def __init__(self, embedding_client: BaseEmbeddingClient):
        """Initialize document embedder.
        
        Args:
            embedding_client: Client for generating embeddings
        """
        self.embedding_client = embedding_client

    def embed_chunks(self, chunks: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Create embeddings for document chunks.
        
        Args:
            chunks: List of chunk dictionaries from TextChunker
            
        Returns:
            Chunks with embeddings added
        """
        # Extract texts
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings in batch
        embeddings = self.embedding_client.embed_batch(texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        
        return chunks

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        return self.embedding_client.embed(text)

"""Simple embedding client for testing without external dependencies."""

from typing import List
import hashlib
import numpy as np


class SimpleHashEmbeddingClient:
    """Simple embedding client using hash-based embeddings (for testing only).
    
    This client creates deterministic embeddings based on text hashing.
    Not suitable for production use, but useful for testing without API keys
    or model downloads.
    """

    def __init__(self, embedding_dim: int = 384):
        """Initialize simple embedding client.
        
        Args:
            embedding_dim: Dimension of embedding vectors
        """
        self.embedding_dim = embedding_dim

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text using hashing.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        # Create deterministic embedding based on text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Use hash to seed random number generator for reproducibility
        seed = int(text_hash[:8], 16)
        rng = np.random.RandomState(seed)
        
        # Generate random embedding
        embedding = rng.randn(self.embedding_dim)
        
        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        return [self.embed(text) for text in texts]

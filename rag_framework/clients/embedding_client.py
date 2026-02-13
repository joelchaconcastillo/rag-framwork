"""Embedding Client module for handling different embedding providers."""

from abc import ABC, abstractmethod
from typing import List, Optional
import os
import numpy as np


class BaseEmbeddingClient(ABC):
    """Base class for embedding clients."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    """OpenAI embedding client."""

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-ada-002"):
        """Initialize OpenAI embedding client.
        
        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env variable.
            model: Embedding model name
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY env var")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]


class SentenceTransformerEmbeddingClient(BaseEmbeddingClient):
    """Sentence Transformers embedding client (local, free)."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        """Initialize Sentence Transformer client.
        
        Args:
            model: Model name from sentence-transformers
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers package is required. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model = SentenceTransformer(model)

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


def get_embedding_client(provider: str = "sentence-transformer", **kwargs) -> BaseEmbeddingClient:
    """Factory function to get embedding client.
    
    Args:
        provider: Provider name ("openai" or "sentence-transformer")
        **kwargs: Arguments to pass to the client constructor
        
    Returns:
        Embedding client instance
    """
    providers = {
        "openai": OpenAIEmbeddingClient,
        "sentence-transformer": SentenceTransformerEmbeddingClient,
    }
    
    if provider.lower() not in providers:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")
    
    return providers[provider.lower()](**kwargs)

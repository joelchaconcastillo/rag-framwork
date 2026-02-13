"""Clients module for LLM and embedding providers."""

from .llm_client import BaseLLMClient, OpenAIClient, GeminiClient, get_llm_client
from .embedding_client import (
    BaseEmbeddingClient,
    OpenAIEmbeddingClient,
    SentenceTransformerEmbeddingClient,
    get_embedding_client,
)
from .simple_embedding import SimpleHashEmbeddingClient

__all__ = [
    "BaseLLMClient",
    "OpenAIClient",
    "GeminiClient",
    "get_llm_client",
    "BaseEmbeddingClient",
    "OpenAIEmbeddingClient",
    "SentenceTransformerEmbeddingClient",
    "SimpleHashEmbeddingClient",
    "get_embedding_client",
]

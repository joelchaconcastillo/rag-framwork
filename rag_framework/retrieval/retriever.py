"""Retrieval module for document search and retrieval."""

from typing import List, Dict, Optional
from .vector_store import ChromaVectorStore
from ..processing.embedder import DocumentEmbedder


class Retriever:
    """Document retriever using vector similarity search."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedder: DocumentEmbedder
    ):
        """Initialize retriever.
        
        Args:
            vector_store: Vector store for similarity search
            embedder: Document embedder for query encoding
        """
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, any]]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of retrieved documents with metadata
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        return results

    def retrieve_and_format(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> str:
        """Retrieve documents and format as context string.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, top_k, filter_metadata)
        
        # Format results as context
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Document {i}]\n{result['document']}\n")
        
        return "\n".join(context_parts)

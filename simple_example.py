"""Simple example demonstrating core RAG framework features without PDF."""

import json
from pathlib import Path

from rag_framework.clients import SimpleHashEmbeddingClient
from rag_framework.preprocessing import QuestionGenerator
from rag_framework.processing import TextChunker, DocumentEmbedder
from rag_framework.retrieval import ChromaVectorStore, Retriever
from rag_framework.evaluation import RAGEvaluator


def main():
    """Simple demonstration of RAG framework."""
    print("=" * 80)
    print("RAG Framework - Simple Example (No API Keys Required)")
    print("=" * 80)
    
    # Sample document content
    sample_doc = """
    Machine Learning
    
    Machine learning is a subset of artificial intelligence that enables systems 
    to learn and improve from experience. It focuses on developing computer programs 
    that can access data and use it to learn for themselves.
    
    Deep Learning
    
    Deep learning is a subfield of machine learning that uses neural networks with 
    multiple layers. These networks can learn hierarchical representations of data. 
    Deep learning has achieved remarkable success in computer vision, natural language 
    processing, and speech recognition.
    
    Neural Networks
    
    Neural networks are computing systems inspired by biological neural networks. 
    They consist of interconnected nodes called neurons organized in layers. The 
    connections between neurons have weights that are adjusted during training.
    """
    
    print("\n1. Sample document loaded")
    print(f"   Document length: {len(sample_doc)} characters")
    
    # Initialize embedding client
    print("\n2. Initializing embedding client...")
    embedding_client = SimpleHashEmbeddingClient(embedding_dim=384)
    print("   Using simple hash-based embeddings (for demo)")
    
    # Chunk the document
    print("\n3. Chunking document...")
    chunker = TextChunker(chunk_size=200, chunk_overlap=30)
    
    # Create simple chunks with metadata
    paragraphs = [p.strip() for p in sample_doc.split('\n\n') if p.strip()]
    chunks = []
    for i, para in enumerate(paragraphs):
        if para:
            chunks.append({
                "text": para,
                "metadata": {"paragraph": i, "type": "content"}
            })
    
    print(f"   Created {len(chunks)} chunks")
    
    # Create embeddings
    print("\n4. Creating embeddings...")
    embedder = DocumentEmbedder(embedding_client)
    chunks_with_embeddings = embedder.embed_chunks(chunks)
    print(f"   Generated embeddings for {len(chunks_with_embeddings)} chunks")
    
    # Store in vector database
    print("\n5. Storing in ChromaDB...")
    vector_store = ChromaVectorStore(
        collection_name="simple_demo",
        persist_directory="./demo_chroma_db"
    )
    vector_store.add_documents(chunks_with_embeddings)
    print(f"   Stored {vector_store.count()} documents")
    
    # Create retriever
    print("\n6. Testing retrieval...")
    retriever = Retriever(vector_store, embedder)
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "Tell me about neural networks",
        "How does deep learning work?"
    ]
    
    for query in test_queries:
        print(f"\n   Query: {query}")
        results = retriever.retrieve(query, top_k=2)
        for i, result in enumerate(results, 1):
            doc_text = result['document'][:80].replace('\n', ' ')
            print(f"   Result {i}: {doc_text}...")
    
    # Demonstrate evaluation
    print("\n7. Demonstrating evaluation metrics...")
    evaluator = RAGEvaluator()
    
    # Example Q&A evaluation
    predictions = [
        "Machine learning is a subset of artificial intelligence",
        "Neural networks are computing systems inspired by biology",
    ]
    ground_truths = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn",
        "Neural networks are computing systems inspired by biological neural networks",
    ]
    
    metrics = evaluator.evaluate_qa(predictions, ground_truths)
    print("   Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"      {metric}: {value:.4f}")
    
    # Show how to format context for LLM
    print("\n8. Formatting retrieved context for LLM...")
    query = "What is the relationship between machine learning and deep learning?"
    context = retriever.retrieve_and_format(query, top_k=3)
    print(f"   Query: {query}")
    print(f"   Context (first 200 chars):\n{context[:200]}...")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("- Set OPENAI_API_KEY or GOOGLE_API_KEY to use real LLMs")
    print("- Use sentence-transformers or OpenAI embeddings for better quality")
    print("- Process your own PDF documents")
    print("- Generate multi-hop questions for evaluation")
    print("- Build your RAG application!")
    
    # Cleanup
    print("\nCleaning up...")
    vector_store.delete_collection()
    import shutil
    if Path("./demo_chroma_db").exists():
        shutil.rmtree("./demo_chroma_db")
    print("Done!")


if __name__ == "__main__":
    main()

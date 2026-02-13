"""Simple test to verify the framework works correctly."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    # Test clients
    from rag_framework.clients import (
        get_llm_client, 
        get_embedding_client,
        BaseEmbeddingClient
    )
    print("✓ Clients module imported")
    
    # Test preprocessing
    from rag_framework.preprocessing import PDFProcessor, QuestionGenerator
    print("✓ Preprocessing module imported")
    
    # Test processing
    from rag_framework.processing import TextChunker, DocumentEmbedder
    print("✓ Processing module imported")
    
    # Test retrieval
    from rag_framework.retrieval import ChromaVectorStore, Retriever
    print("✓ Retrieval module imported")
    
    # Test evaluation
    from rag_framework.evaluation import RAGEvaluator
    print("✓ Evaluation module imported")
    
    print("\nAll imports successful!")
    return True


def test_chunking():
    """Test text chunking functionality."""
    print("\nTesting chunking...")
    from rag_framework.processing import TextChunker
    
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    text = "This is a test. " * 20
    chunks = chunker.chunk_text(text)
    
    assert len(chunks) > 0, "Should produce chunks"
    print(f"✓ Created {len(chunks)} chunks from text")
    return True


def test_embedding_client():
    """Test embedding client (simple hash-based for testing)."""
    print("\nTesting embedding client...")
    from rag_framework.clients.simple_embedding import SimpleHashEmbeddingClient
    
    # Use simple hash-based embeddings for testing (no downloads needed)
    print("  Initializing simple embedding client...")
    client = SimpleHashEmbeddingClient(embedding_dim=384)
    print("  Client initialized")
    
    # Test single embedding
    text = "This is a test sentence."
    print("  Generating embedding...")
    embedding = client.embed(text)
    
    assert isinstance(embedding, list), "Embedding should be a list"
    assert len(embedding) > 0, "Embedding should not be empty"
    print(f"✓ Generated embedding with dimension {len(embedding)}")
    
    # Test batch embedding
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    print("  Generating batch embeddings...")
    embeddings = client.embed_batch(texts)
    
    assert len(embeddings) == len(texts), "Should generate one embedding per text"
    print(f"✓ Generated {len(embeddings)} batch embeddings")
    
    return True


def test_hierarchical_structure():
    """Test hierarchical structure creation."""
    print("\nTesting hierarchical structure...")
    from rag_framework.preprocessing import PDFProcessor
    
    processor = PDFProcessor()
    
    # Test with sample text
    sample_text = """SAMPLE DOCUMENT

1. Introduction
This is the introduction section with some content about the topic.

2. Methods
This section describes the methods used in the research.

2.1 Data Collection
Details about data collection process.
"""
    
    structure = processor.create_hierarchical_structure(sample_text)
    
    assert "sections" in structure, "Structure should have sections"
    assert len(structure["sections"]) > 0, "Should identify sections"
    print(f"✓ Created hierarchical structure with {len(structure['sections'])} sections")
    
    return True


def test_evaluation_metrics():
    """Test evaluation metrics."""
    print("\nTesting evaluation metrics...")
    from rag_framework.evaluation import RAGEvaluator
    
    evaluator = RAGEvaluator()
    
    # Test exact match
    score = evaluator.exact_match("test", "test")
    assert score == 1.0, "Exact match should be 1.0"
    
    score = evaluator.exact_match("test", "different")
    assert score == 0.0, "Non-match should be 0.0"
    print("✓ Exact match works")
    
    # Test F1 score
    score = evaluator.f1_score("the quick brown fox", "the quick brown dog")
    assert 0 < score < 1, "F1 should be between 0 and 1"
    print(f"✓ F1 score works (score: {score:.4f})")
    
    # Test BLEU
    score = evaluator.bleu_score("the quick brown fox", "the quick brown dog")
    assert 0 <= score <= 1, "BLEU should be between 0 and 1"
    print(f"✓ BLEU score works (score: {score:.4f})")
    
    # Test ROUGE-L
    result = evaluator.rouge_l("the quick brown fox", "the quick brown dog")
    assert "f1" in result, "ROUGE-L should return f1"
    print(f"✓ ROUGE-L works (F1: {result['f1']:.4f})")
    
    # Test batch evaluation
    preds = ["answer one", "answer two"]
    truths = ["answer one", "different answer"]
    metrics = evaluator.evaluate_qa(preds, truths)
    
    assert "exact_match" in metrics, "Should have exact_match"
    assert "f1" in metrics, "Should have f1"
    assert "bleu" in metrics, "Should have bleu"
    assert "rouge_l" in metrics, "Should have rouge_l"
    print(f"✓ Batch evaluation works")
    print(f"  - Exact Match: {metrics['exact_match']:.4f}")
    print(f"  - F1: {metrics['f1']:.4f}")
    print(f"  - BLEU: {metrics['bleu']:.4f}")
    print(f"  - ROUGE-L: {metrics['rouge_l']:.4f}")
    
    return True


def test_vector_store():
    """Test ChromaDB vector store."""
    print("\nTesting vector store...")
    from rag_framework.retrieval import ChromaVectorStore
    from rag_framework.clients.simple_embedding import SimpleHashEmbeddingClient
    
    # Create a test collection
    store = ChromaVectorStore(
        collection_name="test_collection",
        persist_directory="./test_chroma_db"
    )
    
    # Create sample chunks with embeddings
    embedding_client = SimpleHashEmbeddingClient(embedding_dim=384)
    
    chunks = [
        {"text": "Machine learning is a subset of AI.", "metadata": {"idx": 0}},
        {"text": "Deep learning uses neural networks.", "metadata": {"idx": 1}},
        {"text": "Python is a programming language.", "metadata": {"idx": 2}},
    ]
    
    # Add embeddings
    for chunk in chunks:
        chunk["embedding"] = embedding_client.embed(chunk["text"])
    
    # Add to store
    store.add_documents(chunks)
    count = store.count()
    print(f"✓ Added {count} documents to vector store")
    
    # Test search
    query_embedding = embedding_client.embed("What is machine learning?")
    results = store.search(query_embedding, top_k=2)
    
    assert len(results) > 0, "Should return results"
    print(f"✓ Search returned {len(results)} results")
    print(f"  Top result: {results[0]['document'][:50]}...")
    
    # Cleanup
    store.delete_collection()
    print("✓ Cleanup successful")
    
    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("RAG Framework Test Suite")
    print("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("Chunking", test_chunking),
        ("Embedding Client", test_embedding_client),
        ("Hierarchical Structure", test_hierarchical_structure),
        ("Evaluation Metrics", test_evaluation_metrics),
        ("Vector Store", test_vector_store),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n✗ Test failed: {name}")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 80)
    
    # Cleanup test directory
    import shutil
    if os.path.exists("./test_chroma_db"):
        shutil.rmtree("./test_chroma_db")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

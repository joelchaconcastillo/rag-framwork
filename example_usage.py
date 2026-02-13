"""Example usage of the RAG framework.

This script demonstrates:
1. Loading and processing a PDF document
2. Generating multi-hop questions for evaluation
3. Creating embeddings and storing in ChromaDB
4. Performing retrieval
5. Generating answers using LLM
6. Evaluating the results
"""

import os
import json
from pathlib import Path

from rag_framework.clients import get_llm_client, get_embedding_client
from rag_framework.preprocessing import PDFProcessor, QuestionGenerator
from rag_framework.processing import TextChunker, DocumentEmbedder
from rag_framework.retrieval import ChromaVectorStore, Retriever
from rag_framework.evaluation import RAGEvaluator


def create_sample_pdf():
    """Create a sample PDF for demonstration."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        print("Warning: reportlab not installed. Cannot create sample PDF.")
        print("Install with: pip install reportlab")
        return None
    
    # Create sample directory
    sample_dir = Path("./sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    pdf_path = sample_dir / "sample_document.pdf"
    
    # Create PDF
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    
    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "SAMPLE RESEARCH DOCUMENT")
    
    # Add sections
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, 700, "1. Introduction")
    
    c.setFont("Helvetica", 10)
    text = """
    Machine learning is a subset of artificial intelligence that enables systems to learn
    and improve from experience. It focuses on developing computer programs that can access
    data and use it to learn for themselves. The primary aim is to allow computers to learn
    automatically without human intervention.
    """
    y = 680
    for line in text.strip().split('\n'):
        c.drawString(100, y, line.strip())
        y -= 15
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y - 20, "2. Deep Learning")
    
    c.setFont("Helvetica", 10)
    y -= 40
    text = """
    Deep learning is a subfield of machine learning that uses neural networks with multiple
    layers. These networks can learn hierarchical representations of data. Deep learning has
    achieved remarkable success in computer vision, natural language processing, and speech
    recognition tasks.
    """
    for line in text.strip().split('\n'):
        c.drawString(100, y, line.strip())
        y -= 15
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y - 20, "2.1. Neural Networks")
    
    c.setFont("Helvetica", 10)
    y -= 40
    text = """
    Neural networks are computing systems inspired by biological neural networks. They consist
    of interconnected nodes called neurons organized in layers. The connections between neurons
    have weights that are adjusted during training to minimize prediction errors.
    """
    for line in text.strip().split('\n'):
        c.drawString(100, y, line.strip())
        y -= 15
    
    c.save()
    print(f"Sample PDF created at: {pdf_path}")
    return str(pdf_path)


def main():
    """Main function demonstrating RAG framework usage."""
    print("=" * 80)
    print("RAG Framework Example Usage")
    print("=" * 80)
    
    # Step 1: Create or use existing PDF
    print("\n1. Creating sample PDF...")
    pdf_path = create_sample_pdf()
    
    if not pdf_path:
        print("Skipping example as PDF creation failed.")
        print("Please ensure you have a PDF file and update the pdf_path variable.")
        return
    
    # Step 2: Process PDF
    print("\n2. Processing PDF and creating hierarchical structure...")
    pdf_processor = PDFProcessor()
    structure = pdf_processor.process_pdf(pdf_path)
    print(f"   Document title: {structure.get('title', 'N/A')}")
    print(f"   Number of sections: {len(structure.get('sections', []))}")
    
    # Step 3: Initialize clients (using sentence-transformer to avoid API keys)
    print("\n3. Initializing clients...")
    print("   Using sentence-transformer for embeddings (no API key required)")
    embedding_client = get_embedding_client("sentence-transformer")
    
    # For LLM, we'll use a mock client if no API key is available
    try:
        llm_client = get_llm_client("openai")
        print("   Using OpenAI for LLM")
    except ValueError:
        print("   Note: Set OPENAI_API_KEY or GOOGLE_API_KEY to use LLM features")
        print("   Skipping question generation for now...")
        llm_client = None
    
    # Step 4: Generate questions (if LLM is available)
    if llm_client:
        print("\n4. Generating multi-hop questions...")
        question_generator = QuestionGenerator(llm_client)
        questions = question_generator.generate_all_questions(structure, questions_per_hop=2)
        
        for hop_type, qa_pairs in questions.items():
            print(f"   {hop_type} questions: {len(qa_pairs)}")
            if qa_pairs:
                print(f"      Example: {qa_pairs[0]['question']}")
        
        # Save questions for evaluation
        questions_path = Path("./sample_data/generated_questions.json")
        with open(questions_path, "w") as f:
            json.dump(questions, f, indent=2)
        print(f"   Questions saved to: {questions_path}")
    else:
        print("\n4. Skipping question generation (no LLM client)")
        questions = None
    
    # Step 5: Chunk documents
    print("\n5. Chunking documents...")
    chunker = TextChunker(chunk_size=256, chunk_overlap=50)
    chunks = chunker.chunk_hierarchical_structure(structure)
    print(f"   Created {len(chunks)} chunks")
    
    # Step 6: Create embeddings
    print("\n6. Creating embeddings...")
    embedder = DocumentEmbedder(embedding_client)
    chunks_with_embeddings = embedder.embed_chunks(chunks)
    print(f"   Generated embeddings for {len(chunks_with_embeddings)} chunks")
    
    # Step 7: Store in ChromaDB
    print("\n7. Storing in ChromaDB...")
    vector_store = ChromaVectorStore(
        collection_name="sample_docs",
        persist_directory="./sample_data/chroma_db"
    )
    vector_store.add_documents(chunks_with_embeddings)
    print(f"   Stored {vector_store.count()} documents in vector store")
    
    # Step 8: Perform retrieval
    print("\n8. Testing retrieval...")
    retriever = Retriever(vector_store, embedder)
    
    test_queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "What is deep learning?"
    ]
    
    for query in test_queries:
        print(f"\n   Query: {query}")
        results = retriever.retrieve(query, top_k=2)
        for i, result in enumerate(results, 1):
            print(f"   Result {i}: {result['document'][:100]}...")
    
    # Step 9: Generate answers (if LLM is available)
    if llm_client:
        print("\n9. Generating answers with RAG...")
        query = "What is the relationship between machine learning and deep learning?"
        context = retriever.retrieve_and_format(query, top_k=3)
        
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        answer = llm_client.generate(prompt, temperature=0.3, max_tokens=200)
        print(f"   Question: {query}")
        print(f"   Answer: {answer}")
    else:
        print("\n9. Skipping answer generation (no LLM client)")
    
    # Step 10: Evaluation example
    print("\n10. Evaluation example...")
    evaluator = RAGEvaluator()
    
    # Example predictions and ground truths
    predictions = [
        "Machine learning is a subset of artificial intelligence",
        "Neural networks consist of interconnected nodes",
    ]
    ground_truths = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn",
        "Neural networks are computing systems with interconnected nodes called neurons",
    ]
    
    metrics = evaluator.evaluate_qa(predictions, ground_truths)
    print("   Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"      {metric}: {value:.4f}")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("- Set OPENAI_API_KEY or GOOGLE_API_KEY for LLM features")
    print("- Use your own PDF documents")
    print("- Customize chunking, retrieval, and evaluation parameters")
    print("- Build your own RAG application!")


if __name__ == "__main__":
    main()

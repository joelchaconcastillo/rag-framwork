# RAG Framework

A comprehensive Python framework for Retrieval-Augmented Generation (RAG) systems.

## Features

- 📄 **PDF Processing**: Extract and create hierarchical structure from PDF documents
- 🤖 **Multiple LLM Providers**: Support for OpenAI (ChatGPT) and Google (Gemini)
- 🔢 **Flexible Embeddings**: Use OpenAI embeddings or local sentence-transformers
- 💾 **ChromaDB Integration**: Efficient vector storage and similarity search
- ❓ **Multi-hop Question Generation**: Generate 1-hop, 2-hop, and 3-hop questions for evaluation
- 📊 **Comprehensive Evaluation**: Multiple metrics including F1, BLEU, ROUGE-L, and retrieval metrics
- 🔧 **Modular Design**: Easy to extend and customize

## Installation

This project uses `uv` for package management, but can also be used with `pip`.

### Using uv (recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -e .
```

### Using pip

```bash
pip install -e .
```

## Project Structure

```
rag_framework/
├── clients/              # LLM and embedding clients
│   ├── llm_client.py    # OpenAI, Gemini support
│   └── embedding_client.py  # OpenAI, sentence-transformers
├── preprocessing/        # Data acquisition and processing
│   ├── pdf_processor.py # PDF extraction and structuring
│   └── question_generator.py  # Multi-hop Q&A generation
├── processing/          # Document processing
│   ├── chunker.py       # Text chunking
│   └── embedder.py      # Document embedding
├── retrieval/           # Vector search and retrieval
│   ├── vector_store.py  # ChromaDB integration
│   └── retriever.py     # Document retrieval
└── evaluation/          # Evaluation metrics
    └── metrics.py       # F1, BLEU, ROUGE, MRR, P@K, R@K
```

## Quick Start

### 1. Set up API keys (optional)

For LLM features, set one of these environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
# OR
export GOOGLE_API_KEY="your-google-key"
```

Note: The framework can work without API keys using local sentence-transformers for embeddings.

### 2. Run the example

```bash
python example_usage.py
```

This will:
- Create a sample PDF document
- Process and structure the content
- Generate embeddings
- Store in ChromaDB
- Demonstrate retrieval
- Show evaluation metrics

## Usage Examples

### Basic RAG Pipeline

```python
from rag_framework.clients import get_llm_client, get_embedding_client
from rag_framework.preprocessing import PDFProcessor
from rag_framework.processing import TextChunker, DocumentEmbedder
from rag_framework.retrieval import ChromaVectorStore, Retriever

# Initialize clients
embedding_client = get_embedding_client("sentence-transformer")
llm_client = get_llm_client("openai", api_key="your-key")

# Process PDF
pdf_processor = PDFProcessor()
structure = pdf_processor.process_pdf("document.pdf")

# Chunk and embed
chunker = TextChunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk_hierarchical_structure(structure)

embedder = DocumentEmbedder(embedding_client)
chunks_with_embeddings = embedder.embed_chunks(chunks)

# Store in vector database
vector_store = ChromaVectorStore(collection_name="my_docs")
vector_store.add_documents(chunks_with_embeddings)

# Retrieve and answer
retriever = Retriever(vector_store, embedder)
context = retriever.retrieve_and_format("What is machine learning?", top_k=5)

answer = llm_client.generate(f"Context: {context}\n\nQuestion: What is machine learning?\nAnswer:")
print(answer)
```

### Generate Multi-hop Questions

```python
from rag_framework.preprocessing import QuestionGenerator

# Generate questions for evaluation
question_gen = QuestionGenerator(llm_client)
questions = question_gen.generate_all_questions(structure, questions_per_hop=5)

# Access different hop types
one_hop = questions["1-hop"]  # Simple factual questions
two_hop = questions["2-hop"]  # Requires connecting 2 facts
three_hop = questions["3-hop"]  # Requires connecting 3 facts
```

### Evaluate RAG Performance

```python
from rag_framework.evaluation import RAGEvaluator

evaluator = RAGEvaluator()

# Evaluate Q&A
predictions = ["Machine learning is AI", "Neural nets are networks"]
ground_truths = ["Machine learning is artificial intelligence", "Neural networks are computing networks"]

metrics = evaluator.evaluate_qa(predictions, ground_truths)
print(f"F1 Score: {metrics['f1']:.4f}")
print(f"BLEU Score: {metrics['bleu']:.4f}")
print(f"ROUGE-L: {metrics['rouge_l']:.4f}")
```

### Switch Between LLM Providers

```python
# Use OpenAI
openai_client = get_llm_client("openai", model="gpt-4")

# Switch to Gemini
gemini_client = get_llm_client("gemini", model="gemini-pro")

# Same interface for both
answer = gemini_client.generate("What is AI?")
```

## Module Overview

### Clients
- **LLM Client**: Abstract interface for language models (OpenAI, Gemini)
- **Embedding Client**: Generate vector embeddings (OpenAI, sentence-transformers)

### Preprocessing
- **PDF Processor**: Extract text and create hierarchical structure from PDFs
- **Question Generator**: Generate 1-hop, 2-hop, and 3-hop questions for evaluation

### Processing
- **Text Chunker**: Split documents into chunks with overlap
- **Document Embedder**: Create vector embeddings for chunks

### Retrieval
- **Vector Store**: ChromaDB-based vector storage
- **Retriever**: Similarity search and context formatting

### Evaluation
- **Exact Match**: Binary exact match scoring
- **F1 Score**: Token-level precision and recall
- **BLEU**: N-gram overlap metric
- **ROUGE-L**: Longest common subsequence
- **Precision@K**: Retrieval precision at K
- **Recall@K**: Retrieval recall at K
- **MRR**: Mean Reciprocal Rank

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{rag_framework,
  title = {RAG Framework: A Comprehensive RAG System},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/joelchaconcastillo/rag-framwork}
}
```


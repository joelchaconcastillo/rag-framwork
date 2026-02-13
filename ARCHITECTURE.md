# RAG Framework Architecture

## Overview

This document describes the architecture and design of the RAG (Retrieval-Augmented Generation) framework.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG Framework                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Clients    │  │Preprocessing │  │  Processing  │          │
│  │              │  │              │  │              │          │
│  │ - LLM Client │  │- PDF Proc.   │  │ - Chunker    │          │
│  │ - Embedding  │  │- Q&A Gen.    │  │ - Embedder   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │  Retrieval   │  │  Evaluation  │                            │
│  │              │  │              │                            │
│  │- Vector Store│  │ - Metrics    │                            │
│  │- Retriever   │  │ - Scoring    │                            │
│  └──────────────┘  └──────────────┘                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Breakdown

### 1. Clients Module (`rag_framework/clients/`)

**Purpose**: Abstraction layer for external services (LLMs and embeddings)

**Components**:
- `llm_client.py`: LLM providers (OpenAI, Gemini)
- `embedding_client.py`: Embedding providers (OpenAI, sentence-transformers)
- `simple_embedding.py`: Hash-based embeddings for testing

**Design Pattern**: Factory pattern with abstract base classes

**Key Features**:
- Provider-agnostic interface
- Easy to switch between providers
- Support for batch operations

### 2. Preprocessing Module (`rag_framework/preprocessing/`)

**Purpose**: Document acquisition, processing, and question generation

**Components**:
- `pdf_processor.py`: PDF extraction and hierarchical structure creation
- `question_generator.py`: Multi-hop question generation for evaluation

**Key Features**:
- Hierarchical document structure (title → sections → subsections)
- 1-hop, 2-hop, and 3-hop question generation
- Metadata preservation

### 3. Processing Module (`rag_framework/processing/`)

**Purpose**: Document chunking and embedding generation

**Components**:
- `chunker.py`: Text splitting with overlap
- `embedder.py`: Vector embedding generation

**Key Features**:
- Configurable chunk size and overlap
- Metadata preservation through pipeline
- Batch embedding generation

### 4. Retrieval Module (`rag_framework/retrieval/`)

**Purpose**: Vector storage and similarity search

**Components**:
- `vector_store.py`: ChromaDB integration
- `retriever.py`: Query processing and result formatting

**Key Features**:
- Persistent vector storage
- Cosine similarity search
- Metadata filtering
- Context formatting for LLMs

### 5. Evaluation Module (`rag_framework/evaluation/`)

**Purpose**: Performance measurement and metrics

**Components**:
- `metrics.py`: Comprehensive evaluation metrics

**Metrics Implemented**:
- **Exact Match**: Binary exact matching
- **F1 Score**: Token-level precision/recall
- **BLEU**: N-gram overlap
- **ROUGE-L**: Longest common subsequence
- **Precision@K**: Retrieval precision
- **Recall@K**: Retrieval recall
- **MRR**: Mean Reciprocal Rank

## Data Flow

### Document Ingestion Pipeline

```
PDF File
  ↓
[PDFProcessor] → Hierarchical Structure
  ↓
[TextChunker] → Chunks with Metadata
  ↓
[DocumentEmbedder] → Chunks with Embeddings
  ↓
[ChromaVectorStore] → Stored Documents
```

### Query Processing Pipeline

```
User Query
  ↓
[Embedder] → Query Embedding
  ↓
[VectorStore] → Similar Documents
  ↓
[Retriever] → Formatted Context
  ↓
[LLM Client] → Generated Answer
```

### Evaluation Pipeline

```
Questions (1-hop, 2-hop, 3-hop)
  ↓
[RAG System] → Predictions
  ↓
[RAGEvaluator] → Metrics
  ↓
Performance Report
```

## Design Principles

1. **Modularity**: Each component is independent and replaceable
2. **Extensibility**: Easy to add new providers or metrics
3. **Type Safety**: Abstract base classes for consistency
4. **Testability**: Simple dependencies, easy to mock
5. **Configurability**: Sensible defaults, but customizable

## Usage Patterns

### Basic RAG Pattern

```python
# Initialize
embedding_client = get_embedding_client("sentence-transformer")
llm_client = get_llm_client("openai")

# Process documents
structure = pdf_processor.process_pdf("doc.pdf")
chunks = chunker.chunk_hierarchical_structure(structure)
chunks_embedded = embedder.embed_chunks(chunks)

# Store
vector_store.add_documents(chunks_embedded)

# Retrieve and answer
context = retriever.retrieve_and_format(query)
answer = llm_client.generate(f"Context: {context}\nQ: {query}\nA:")
```

### Evaluation Pattern

```python
# Generate questions
questions = question_gen.generate_all_questions(structure)

# Get predictions
predictions = [rag_system(q) for q in questions["1-hop"]]
ground_truths = [qa["answer"] for qa in questions["1-hop"]]

# Evaluate
metrics = evaluator.evaluate_qa(predictions, ground_truths)
```

## Extensibility Points

### Adding a New LLM Provider

1. Inherit from `BaseLLMClient`
2. Implement `generate()` and `generate_batch()`
3. Add to factory in `get_llm_client()`

### Adding a New Metric

1. Add method to `RAGEvaluator`
2. Update `evaluate_qa()` to include new metric
3. Document the metric

### Adding a New Vector Store

1. Create new class with same interface as `ChromaVectorStore`
2. Implement `add_documents()` and `search()`
3. Update `Retriever` to accept any vector store

## Dependencies

### Core Dependencies
- `openai`: OpenAI API client
- `google-generativeai`: Google Gemini API client
- `chromadb`: Vector database
- `pypdf2`: PDF processing
- `sentence-transformers`: Local embeddings
- `numpy`, `pandas`, `scikit-learn`: Data processing

### Optional Dependencies
- `langchain`: Integration with LangChain ecosystem
- `nltk`: Advanced text processing
- `reportlab`: PDF creation for examples

## Performance Considerations

1. **Batch Operations**: Use batch embeddings when possible
2. **Chunk Size**: Balance between context and granularity (512-1024 tokens)
3. **Vector Store**: ChromaDB uses HNSW for fast approximate search
4. **Caching**: Consider caching embeddings for frequently used texts

## Security Considerations

1. **API Keys**: Never commit API keys, use environment variables
2. **Input Validation**: Sanitize user inputs before processing
3. **Dependencies**: Regularly update dependencies for security patches

## Future Enhancements

Potential areas for extension:

1. **More Providers**: Anthropic, Cohere, local models
2. **Advanced Chunking**: Semantic chunking, recursive splitting
3. **Reranking**: Add reranking step after retrieval
4. **Hybrid Search**: Combine vector and keyword search
5. **Streaming**: Support streaming responses from LLMs
6. **Async Support**: Async/await for better performance
7. **Distributed Storage**: Support for distributed vector stores

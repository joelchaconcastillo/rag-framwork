# Contributing to RAG Framework

Thank you for your interest in contributing to the RAG Framework! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/rag-framwork.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install in development mode: `pip install -e .`

## Development Setup

### Using UV (Recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -e .
```

### Running Tests

```bash
python test_framework.py
```

All tests should pass before submitting a PR.

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Add docstrings for all public methods and classes
- Keep functions focused and small
- Use descriptive variable names

Example:

```python
def process_document(text: str, chunk_size: int = 512) -> List[str]:
    """Process document into chunks.
    
    Args:
        text: Input text to process
        chunk_size: Maximum size of each chunk
        
    Returns:
        List of text chunks
    """
    # Implementation
    pass
```

## Project Structure

```
rag_framework/
├── clients/          # LLM and embedding clients
├── preprocessing/    # Document processing
├── processing/       # Chunking and embedding
├── retrieval/        # Vector storage and search
└── evaluation/       # Metrics and evaluation
```

## Adding New Features

### Adding a New LLM Provider

1. Create a new class in `rag_framework/clients/llm_client.py`
2. Inherit from `BaseLLMClient`
3. Implement required methods: `generate()` and `generate_batch()`
4. Add to the factory function `get_llm_client()`
5. Add tests
6. Update documentation

Example:

```python
class NewLLMClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None):
        # Initialize your client
        pass
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Implement generation
        pass
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        # Implement batch generation
        pass
```

### Adding a New Embedding Provider

1. Create a new class in `rag_framework/clients/embedding_client.py`
2. Inherit from `BaseEmbeddingClient`
3. Implement: `embed()` and `embed_batch()`
4. Add to `get_embedding_client()`
5. Add tests
6. Update documentation

### Adding a New Evaluation Metric

1. Add method to `RAGEvaluator` in `rag_framework/evaluation/metrics.py`
2. Update `evaluate_qa()` to include the new metric
3. Add tests
4. Document the metric in docstrings and README

## Testing Guidelines

- Write tests for all new features
- Ensure existing tests still pass
- Test edge cases
- Use the simple embedding client for tests that don't need real embeddings
- Mock external API calls when possible

## Documentation

- Update README.md for user-facing changes
- Update ARCHITECTURE.md for design changes
- Add docstrings to all public APIs
- Include usage examples

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `python test_framework.py`
4. Commit with clear messages: `git commit -m "Add feature X"`
5. Push to your fork: `git push origin feature/your-feature-name`
6. Create a Pull Request on GitHub
7. Wait for review and address feedback

## Commit Message Guidelines

- Use present tense: "Add feature" not "Added feature"
- Be descriptive but concise
- Reference issues when applicable: "Fix #123: ..."

Good examples:
```
Add support for Anthropic Claude
Fix embedding dimension mismatch in chunker
Update documentation for question generator
```

## Code Review Process

All submissions require review. We'll look for:
- Code quality and style
- Test coverage
- Documentation
- Performance implications
- Breaking changes (if any)

## Community Guidelines

- Be respectful and inclusive
- Help others learn
- Provide constructive feedback
- Follow the code of conduct

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

Thank you for contributing to RAG Framework! 🎉

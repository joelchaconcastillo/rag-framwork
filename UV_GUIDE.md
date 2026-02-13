# Using UV with RAG Framework

This guide explains how to use the `uv` package manager with the RAG framework.

## Installation with UV

### Install UV

First, install `uv` if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or on Windows with PowerShell:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install the Framework

Once `uv` is installed, you can install the framework and its dependencies:

```bash
# Clone the repository
git clone https://github.com/joelchaconcastillo/rag-framwork.git
cd rag-framwork

# Install in editable mode with uv
uv pip install -e .

# Or install from requirements.txt
uv pip install -r requirements.txt
```

## Running Examples with UV

You can run Python scripts using `uv`:

```bash
# Run the simple example
uv run simple_example.py

# Run the full example (requires API keys)
uv run example_usage.py

# Run tests
uv run test_framework.py
```

## Creating a Virtual Environment with UV

UV can also manage virtual environments:

```bash
# Create a new virtual environment
uv venv

# Activate it (Linux/Mac)
source .venv/bin/activate

# Activate it (Windows)
.venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

## Benefits of Using UV

- **Fast**: UV is significantly faster than pip
- **Reliable**: Better dependency resolution
- **Compatible**: Works with existing pip workflows
- **Modern**: Built in Rust for performance

## Alternative: Using pip

If you prefer traditional pip:

```bash
pip install -e .
# or
pip install -r requirements.txt
```

Both methods work equally well with this framework.

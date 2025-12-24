# DocQA Core Library

The core Retrieval-Augmented Generation (RAG) engine for document question answering.

## Overview

DocQA is a modular Python library that provides a production-ready RAG pipeline for:
- Document loading and chunking (PDF, JSON)
- Semantic search with vector embeddings
- LLM-powered question answering
- Flexible LLM provider support (OpenAI, Ollama)

## Features

- ✅ Multi-format document support (PDF, JSON)
- ✅ Smart document chunking strategies
- ✅ FAISS vector search
- ✅ Pluggable LLM providers
- ✅ Configurable embeddings
- ✅ Production-ready code

## Installation

```bash
pip install -e .
```

## Configuration

Set environment variables or create a `.env` file:

```bash
DOCQA_LLM_PROVIDER=openai          # or "ollama"
DOCQA_LLM_MODEL=gpt-4o-mini
DOCQA_OPENAI_API_KEY=sk-...
DOCQA_EMBED_PROVIDER=openai        # or "ollama"
DOCQA_EMBED_MODEL=text-embedding-3-small
DOCQA_VECTOR_DB_PATH=./.local/faiss_store
```

## Usage

### Basic Usage

```python
from docqa.config import Settings
from docqa.pipeline.engine import QAEngine

# Initialize settings and engine
settings = Settings()
settings.validate()
engine = QAEngine(settings)

# Ingest documents
engine.ingest_pdf("path/to/document.pdf")
engine.ingest_json("path/to/data.json")

# Answer questions
result = engine.answer("What is the main topic?")
print(result["answer"])
```

### Core Components

- **Config** (`docqa.config`): Settings management
- **Loaders** (`docqa.loaders`): Document loading (PDF, JSON)
- **Chunking** (`docqa.chunking`): Text splitting strategies
- **Indexing** (`docqa.indexing`): Vector database (FAISS)
- **Retrieval** (`docqa.retrieval`): Semantic search
- **LLM** (`docqa.llm`): LLM providers and prompts
- **Pipeline** (`docqa.pipeline`): QA engine orchestration

## Project Structure

```
src/docqa/
├── __init__.py
├── config.py              # Settings and configuration
├── chunking/              # Text splitting
│   └── splitter.py
├── indexing/              # Vector database
│   └── faiss_store.py
├── llm/                   # LLM providers
│   ├── prompts.py
│   └── providers.py
├── loaders/               # Document loaders
│   ├── json.py
│   └── pdf.py
├── pipeline/              # Main QA engine
│   └── engine.py
└── retrieval/             # Search and retrieval
    └── retriever.py
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Build documentation
mkdocs serve
```

## API Server

For the REST API wrapper, see the separate [`docqa-api`](../docqa-api) project.

## Testing

```bash
pytest tests/
pytest tests/test_qaengine.py -v
```

## Documentation

Full documentation with examples is available in the `docs/` folder.

See also:
- **MVP Notebook**: `notebooks/MVP.ipynb` - End-to-end walkthrough
- **API Documentation**: `../docqa-api/README.md` - REST API usage

---

**Version**: 0.1.0  
**License**: MIT

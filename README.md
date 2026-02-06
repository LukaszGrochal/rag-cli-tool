# rag-cli-tool

CLI tool for RAG over local documents. Index a folder, ask questions.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

```bash
git clone <repo-url>
cd rag-cli-tool
uv venv && uv pip install -e .
```

All commands below use `uv run rag-cli` to run within the virtual environment.

## Quick Start

### Option A: Local with Ollama (no API keys)

1. Install [Ollama](https://ollama.com) and pull models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

2. Create a `.env` file:

```
RAG_CLI_MODEL=ollama:llama3.2
RAG_CLI_EMBEDDING_MODEL=ollama:nomic-embed-text
```

3. Index and ask:

```bash
uv run rag-cli index ./my-docs/
uv run rag-cli ask "What is the refund policy?"
```

### Option B: Cloud providers (Anthropic + OpenAI)

1. Create a `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

2. Index and ask:

```bash
uv run rag-cli index ./my-docs/
uv run rag-cli ask "What is the refund policy?"
```

### Mix-and-match

You can combine local and cloud providers — for example, local generation with cloud embeddings:

```
OPENAI_API_KEY=sk-...
RAG_CLI_MODEL=ollama:llama3.2
```

## Commands

### `rag-cli index <path>`

Index documents from a folder into the local vector store.

```bash
uv run rag-cli index ./documents/
uv run rag-cli index ./documents/ --chunk-size 500 --chunk-overlap 50
uv run rag-cli index ./documents/ --fresh   # Wipe and rebuild index
```

Supported formats: `.pdf`, `.md`, `.txt`, `.docx`

Indexing is incremental by default — only new documents are embedded. Use `--fresh` to wipe the existing index and rebuild from scratch.

### `rag-cli ask "<question>"`

Ask a question about your indexed documents.

```bash
uv run rag-cli ask "What are the payment terms?"
uv run rag-cli ask "What are the payment terms?" --top-k 5
```

The answer is generated using only the retrieved document context (strict RAG — no external knowledge).

## Configuration

All settings can be set via environment variables or a `.env` file in the project root.

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | | Anthropic API key (required for cloud generation) |
| `OPENAI_API_KEY` | | OpenAI API key (required for cloud embeddings) |
| `RAG_CLI_MODEL` | `claude-3-5-sonnet-latest` | LLM model for generation |
| `RAG_CLI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `RAG_CLI_OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `RAG_CLI_CHUNK_SIZE` | `1000` | Max characters per chunk |
| `RAG_CLI_CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RAG_CLI_TOP_K` | `3` | Number of chunks to retrieve |

### Model string format

Use the `ollama:` prefix to route to a local Ollama model. No prefix uses the default cloud provider.

| Variable | Value | Provider |
|----------|-------|----------|
| `RAG_CLI_MODEL` | `claude-3-5-sonnet-latest` | Anthropic API |
| `RAG_CLI_MODEL` | `ollama:llama3.2` | Ollama (local) |
| `RAG_CLI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI API |
| `RAG_CLI_EMBEDDING_MODEL` | `ollama:nomic-embed-text` | Ollama (local) |

## Project Structure

```
src/
├── rag_cli/       # CLI interface (Typer + Rich)
├── llm_core/      # LLM abstraction layer (providers, config, retry)
└── rag_core/      # RAG pipeline (loaders, chunking, embeddings, retrieval)
```

`llm_core` and `rag_core` are designed as independent, reusable packages.

## License

MIT

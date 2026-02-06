# rag-cli-tool

CLI tool for RAG over local documents. Index a folder, ask questions.

## Installation

```bash
# Clone and install
git clone <repo-url>
cd rag-cli-tool
uv venv && uv pip install -e .
```

## Quick Start

### Cloud providers (Anthropic + OpenAI)

```bash
# Set API keys
export OPENAI_API_KEY=sk-...        # Required for embeddings
export ANTHROPIC_API_KEY=sk-ant-...  # Required for answer generation

# Index your documents
rag-cli index ./my-docs/

# Ask questions
rag-cli ask "What is the refund policy?"
```

### Local with Ollama (no API keys needed)

```bash
# Install Ollama: https://ollama.com
# Pull models
ollama pull llama3.2
ollama pull nomic-embed-text

# Index and ask — fully local
RAG_CLI_MODEL=ollama:llama3.2 RAG_CLI_EMBEDDING_MODEL=ollama:nomic-embed-text rag-cli index ./my-docs/
RAG_CLI_MODEL=ollama:llama3.2 RAG_CLI_EMBEDDING_MODEL=ollama:nomic-embed-text rag-cli ask "What is the refund policy?"
```

You can also mix-and-match — for example, use a local model for generation with cloud embeddings:

```bash
export OPENAI_API_KEY=sk-...
RAG_CLI_MODEL=ollama:llama3.2 rag-cli ask "What is the refund policy?"
```

## Commands

### `rag-cli index <path>`

Index documents from a folder into the local vector store.

```bash
rag-cli index ./documents/
rag-cli index ./documents/ --chunk-size 500 --chunk-overlap 50
rag-cli index ./documents/ --fresh   # Wipe and rebuild index
```

Supported formats: `.pdf`, `.md`, `.txt`, `.docx`

By default, indexing is incremental — only new documents are embedded. Use `--fresh` to wipe the existing index and rebuild from scratch.

### `rag-cli ask "<question>"`

Ask a question about your indexed documents.

```bash
rag-cli ask "What are the payment terms?"
rag-cli ask "What are the payment terms?" --top-k 5
```

The answer is generated using only the retrieved document context (strict RAG — no external knowledge).

## Configuration

Set via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | `""` | OpenAI API key (for cloud embeddings) |
| `ANTHROPIC_API_KEY` | `""` | Anthropic API key (for cloud generation) |
| `RAG_CLI_MODEL` | `claude-3-5-sonnet-latest` | LLM model for generation |
| `RAG_CLI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `RAG_CLI_OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `RAG_CLI_CHUNK_SIZE` | `1000` | Max characters per chunk |
| `RAG_CLI_CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RAG_CLI_TOP_K` | `3` | Number of chunks to retrieve |

### Model string format

Use the `ollama:` prefix to select a local Ollama model:

| Setting | Example | Provider |
|---------|---------|----------|
| `RAG_CLI_MODEL=claude-3-5-sonnet-latest` | Cloud (Anthropic) | Anthropic API |
| `RAG_CLI_MODEL=ollama:llama3.2` | Local (Ollama) | Ollama server |
| `RAG_CLI_EMBEDDING_MODEL=text-embedding-3-small` | Cloud (OpenAI) | OpenAI API |
| `RAG_CLI_EMBEDDING_MODEL=ollama:nomic-embed-text` | Local (Ollama) | Ollama server |

## Project Structure

```
src/
├── rag_cli/       # CLI-specific (Typer + Rich)
├── llm_core/      # Reusable LLM abstraction layer
└── rag_core/      # Reusable RAG pipeline components
```

`llm_core` and `rag_core` are designed as independent, reusable packages that can be extracted to other projects.

## License

MIT

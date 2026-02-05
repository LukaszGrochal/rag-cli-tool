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

```bash
# Set API keys
export OPENAI_API_KEY=sk-...        # Required for embeddings
export ANTHROPIC_API_KEY=sk-ant-...  # Required for answer generation

# Index your documents
rag-cli index ./my-docs/

# Ask questions
rag-cli ask "What is the refund policy?"
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
| `OPENAI_API_KEY` | *required* | OpenAI API key (for embeddings) |
| `ANTHROPIC_API_KEY` | *required* | Anthropic API key (for generation) |
| `RAG_CLI_CHUNK_SIZE` | `1000` | Max characters per chunk |
| `RAG_CLI_CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RAG_CLI_TOP_K` | `3` | Number of chunks to retrieve |
| `RAG_CLI_MODEL` | `claude-3-5-sonnet-latest` | LLM model for generation |
| `RAG_CLI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |

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

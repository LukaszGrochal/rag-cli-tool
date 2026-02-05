"""CLI commands for rag-cli."""

import hashlib
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.progress import track

from rag_cli.console import console, print_error, print_index_summary, print_success

app = typer.Typer(
    name="rag-cli",
    help="CLI tool for RAG over local documents. Index a folder, ask questions.",
    no_args_is_help=True,
)


def _get_settings():
    """Load settings, handling missing env vars gracefully."""
    try:
        from llm_core.config import LLMSettings
        return LLMSettings()
    except Exception as e:
        error_msg = str(e)
        if "ANTHROPIC_API_KEY" in error_msg:
            print_error("ANTHROPIC_API_KEY not set. Required for answer generation.")
        elif "OPENAI_API_KEY" in error_msg:
            print_error("OPENAI_API_KEY not set. Required for embeddings.")
        else:
            print_error(f"Configuration error: {e}")
        raise typer.Exit(code=1)


def _chunk_id(source: str, chunk_index: int) -> str:
    """Generate a deterministic ID for a chunk."""
    raw = f"{source}::chunk::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@app.command()
def index(
    path: Annotated[
        Path,
        typer.Argument(help="Path to directory containing documents to index."),
    ],
    chunk_size: Annotated[
        int,
        typer.Option("--chunk-size", help="Maximum characters per chunk."),
    ] = None,
    chunk_overlap: Annotated[
        int,
        typer.Option("--chunk-overlap", help="Overlap characters between chunks."),
    ] = None,
    fresh: Annotated[
        bool,
        typer.Option("--fresh", help="Wipe existing index and rebuild from scratch."),
    ] = False,
) -> None:
    """Index documents from a folder into the local vector store."""
    start = time.time()

    if not path.exists():
        print_error(f"Path does not exist: {path}")
        raise typer.Exit(code=1)

    if not path.is_dir():
        print_error(f"Path is not a directory: {path}")
        raise typer.Exit(code=1)

    settings = _get_settings()
    _chunk_size = chunk_size or settings.chunk_size
    _chunk_overlap = chunk_overlap or settings.chunk_overlap

    from rag_core.loaders import load_documents

    console.print(f"[bold]Scanning[/bold] {path}")
    documents = load_documents(path)

    if not documents:
        print_error(f"No supported documents found in {path}")
        raise typer.Exit(code=1)

    console.print(f"  Found {len(documents)} document(s)")

    from rag_core.chunking import RecursiveChunker

    chunker = RecursiveChunker(chunk_size=_chunk_size, chunk_overlap=_chunk_overlap)

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_metadatas: list[dict] = []

    for doc in documents:
        chunks = chunker.chunk(doc.content)
        for i, chunk_text in enumerate(chunks):
            all_chunks.append(chunk_text)
            all_ids.append(_chunk_id(doc.source, i))
            all_metadatas.append({"source": doc.source, "chunk_index": i})

    console.print(f"  Created {len(all_chunks)} chunk(s)")

    from rag_core.vectorstores import ChromaStore

    persist_dir = Path(".rag-cli") / "chroma"
    store = ChromaStore(persist_dir=persist_dir)

    if fresh:
        console.print("  [yellow]Wiping existing index (--fresh)[/yellow]")
        store.reset()

    existing = store.existing_ids()
    new_indices = [i for i, cid in enumerate(all_ids) if cid not in existing]

    if not new_indices:
        print_success("All documents already indexed. Nothing to do.")
        raise typer.Exit(code=0)

    new_chunks = [all_chunks[i] for i in new_indices]
    new_ids = [all_ids[i] for i in new_indices]
    new_metadatas = [all_metadatas[i] for i in new_indices]

    console.print(f"  Embedding {len(new_chunks)} new chunk(s)...")

    from llm_core.providers.openai import OpenAIEmbeddingProvider
    from rag_core.embeddings import OpenAIEmbedder

    embedding_provider = OpenAIEmbeddingProvider(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )
    embedder = OpenAIEmbedder(provider=embedding_provider)

    batch_size = 100
    all_embeddings: list[list[float]] = []
    for batch_start in track(range(0, len(new_chunks), batch_size), description="Embedding..."):
        batch_end = min(batch_start + batch_size, len(new_chunks))
        batch = new_chunks[batch_start:batch_end]
        batch_embeddings = embedder.embed(batch)
        all_embeddings.extend(batch_embeddings)

    store.add(
        ids=new_ids,
        embeddings=all_embeddings,
        documents=new_chunks,
        metadatas=new_metadatas,
    )

    elapsed = time.time() - start
    print_index_summary(
        num_documents=len(documents),
        num_chunks=len(new_chunks),
        elapsed=elapsed,
    )


@app.command()
def ask(
    question: Annotated[
        str,
        typer.Argument(help="The question to ask about your documents."),
    ],
    top_k: Annotated[
        int,
        typer.Option("--top-k", help="Number of relevant chunks to retrieve."),
    ] = None,
) -> None:
    """Ask a question about your indexed documents."""
    persist_dir = Path(".rag-cli") / "chroma"

    if not persist_dir.exists():
        print_error("No index found. Run 'rag-cli index <path>' first.")
        raise typer.Exit(code=1)

    settings = _get_settings()
    _top_k = top_k or settings.top_k

    from llm_core.providers.openai import OpenAIEmbeddingProvider
    from rag_core.embeddings import OpenAIEmbedder
    from rag_core.retrieval import SimilarityRetriever
    from rag_core.vectorstores import ChromaStore

    store = ChromaStore(persist_dir=persist_dir)

    if store.count() == 0:
        print_error("Index is empty. Run 'rag-cli index <path>' first.")
        raise typer.Exit(code=1)

    embedding_provider = OpenAIEmbeddingProvider(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )
    embedder = OpenAIEmbedder(provider=embedding_provider)
    retriever = SimilarityRetriever(embedder=embedder, store=store)

    console.print("[bold]Searching[/bold] for relevant context...")
    results = retriever.retrieve(question, top_k=_top_k)

    if not results:
        print_error("No relevant documents found for your question.")
        raise typer.Exit(code=1)

    context_parts = []
    for i, result in enumerate(results, 1):
        source = result.metadata.get("source", "unknown")
        context_parts.append(f"[Source {i}: {source}]\n{result.document}")
    context = "\n\n---\n\n".join(context_parts)

    from llm_core.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(api_key=settings.anthropic_api_key, model=settings.model)

    system_prompt = (
        "You are a helpful assistant that answers questions based ONLY on the provided context. "
        "If the context does not contain enough information to answer the question, say "
        "'I don't have enough information in the provided documents to answer this question.' "
        "Do not use any knowledge outside the provided context. "
        "Cite the source numbers [Source N] when referencing information."
    )

    user_prompt = f"""Context:
{context}

Question: {question}

Answer based ONLY on the context above."""

    console.print("[bold]Generating[/bold] answer...")
    response = provider.generate(user_prompt, system=system_prompt)

    from rag_cli.console import print_answer

    sources = [result.metadata for result in results]
    print_answer(response.text, sources)

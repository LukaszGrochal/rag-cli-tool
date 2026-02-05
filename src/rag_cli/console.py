"""Rich console helpers for CLI output."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
error_console = Console(stderr=True)


def print_error(message: str) -> None:
    """Print an error message in red to stderr."""
    error_console.print(f"[bold red]Error:[/bold red] {message}")


def print_success(message: str) -> None:
    """Print a success message in green."""
    console.print(f"[bold green]{message}[/bold green]")


def print_answer(answer: str, sources: list[dict]) -> None:
    """Print the RAG answer with source references."""
    console.print()
    console.print(Panel(answer, title="Answer", border_style="blue"))

    if sources:
        console.print()
        console.print("[bold]Sources:[/bold]")
        seen = set()
        for src in sources:
            source_str = src.get("source", "unknown")
            chunk_id = src.get("chunk_index", "?")
            key = f"{source_str}:{chunk_id}"
            if key not in seen:
                seen.add(key)
                console.print(f"  [dim]•[/dim] {source_str} (chunk {chunk_id})")


def print_index_summary(num_documents: int, num_chunks: int, elapsed: float) -> None:
    """Print indexing summary."""
    table = Table(show_header=False, box=None)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Documents", str(num_documents))
    table.add_row("Chunks", str(num_chunks))
    table.add_row("Time", f"{elapsed:.1f}s")
    console.print()
    console.print(Panel(table, title="Indexing Complete", border_style="green"))

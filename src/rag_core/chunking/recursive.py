"""Recursive character text splitter."""

from rag_core.chunking.base import BaseChunker

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker(BaseChunker):
    """Splits text recursively using a hierarchy of separators."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")
        self._chunk_size = chunk_size
        self._overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        if not text.strip():
            return []
        return self._split(text, _SEPARATORS)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the given separators."""
        if len(text) <= self._chunk_size:
            return [text]

        separator = ""
        remaining_separators = []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator = sep
                remaining_separators = separators[i + 1:]
                break

        if separator:
            pieces = text.split(separator)
        else:
            pieces = list(text)

        chunks: list[str] = []
        current = ""

        for piece in pieces:
            candidate = current + separator + piece if current else piece
            if len(candidate) <= self._chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(piece) > self._chunk_size and remaining_separators:
                    sub_chunks = self._split(piece, remaining_separators)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = piece

        if current:
            chunks.append(current)

        if self._overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap between consecutive chunks."""
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-self._overlap:] if len(prev) > self._overlap else prev
            result.append(overlap_text + chunks[i])
        return result

from rag_core.chunking.recursive import RecursiveChunker


def test_chunks_long_text():
    """Should split long text into multiple chunks."""
    text = "word " * 500  # ~2500 chars
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)  # allow slight overshoot at split boundaries


def test_short_text_returns_single_chunk():
    """Short text that fits in one chunk should not be split."""
    chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=100)
    chunks = chunker.chunk("Short text.")
    assert len(chunks) == 1
    assert chunks[0] == "Short text."


def test_empty_text_returns_empty():
    """Empty string should return no chunks."""
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk("")
    assert chunks == []


def test_overlap_exists():
    """Consecutive chunks should have overlapping content."""
    text = "A " * 200
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk(text)
    if len(chunks) >= 2:
        # The end of chunk[0] should overlap with the start of chunk[1]
        end_of_first = chunks[0][-10:]
        assert end_of_first in chunks[1]

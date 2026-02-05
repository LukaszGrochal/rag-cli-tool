from unittest.mock import MagicMock

from rag_core.retrieval.similarity import SimilarityRetriever
from rag_core.vectorstores.base import SearchResult


def test_retrieve_returns_chunks():
    """Should embed the query, search the store, and return results."""
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    mock_store = MagicMock()
    mock_store.query.return_value = [
        SearchResult(id="chunk1", document="Hello world", metadata={"source": "a.txt"}, distance=0.1),
        SearchResult(id="chunk2", document="Goodbye", metadata={"source": "b.txt"}, distance=0.5),
    ]

    retriever = SimilarityRetriever(embedder=mock_embedder, store=mock_store)
    results = retriever.retrieve("test query", top_k=2)

    assert len(results) == 2
    assert results[0].document == "Hello world"
    mock_embedder.embed.assert_called_once_with(["test query"])
    mock_store.query.assert_called_once_with(query_embedding=[0.1, 0.2, 0.3], top_k=2)

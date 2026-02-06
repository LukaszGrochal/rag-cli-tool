"""Tests for OllamaEmbedder."""

from unittest.mock import MagicMock

from rag_core.embeddings.ollama import OllamaEmbedder


def test_embed_texts():
    """Should delegate to OllamaEmbeddingProvider and return vectors."""
    mock_provider = MagicMock()
    mock_provider.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]

    embedder = OllamaEmbedder(provider=mock_provider)
    vectors = embedder.embed(["hello", "world"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    mock_provider.embed.assert_called_once_with(["hello", "world"])


def test_embed_batches_large_input():
    """Should batch inputs when exceeding batch size."""
    mock_provider = MagicMock()
    mock_provider.embed.side_effect = [
        [[0.1]] * 100,
        [[0.2]] * 50,
    ]

    embedder = OllamaEmbedder(provider=mock_provider, batch_size=100)
    texts = [f"text {i}" for i in range(150)]
    vectors = embedder.embed(texts)

    assert len(vectors) == 150
    assert mock_provider.embed.call_count == 2

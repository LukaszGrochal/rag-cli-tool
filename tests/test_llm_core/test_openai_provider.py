from unittest.mock import MagicMock, patch

from llm_core.providers.openai import OpenAIEmbeddingProvider


def test_embed_single_text():
    """Should embed a single text and return a list with one vector."""
    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1, 0.2, 0.3]
    mock_response = MagicMock()
    mock_response.data = [mock_embedding]

    with patch("llm_core.providers.openai.OpenAI") as MockClient:
        MockClient.return_value.embeddings.create.return_value = mock_response

        provider = OpenAIEmbeddingProvider(api_key="test-key")
        vectors = provider.embed(["Hello world"])

    assert vectors == [[0.1, 0.2, 0.3]]
    MockClient.return_value.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input=["Hello world"],
    )


def test_embed_batch():
    """Should embed multiple texts in one call."""
    mock_emb1 = MagicMock()
    mock_emb1.embedding = [0.1, 0.2]
    mock_emb2 = MagicMock()
    mock_emb2.embedding = [0.3, 0.4]
    mock_response = MagicMock()
    mock_response.data = [mock_emb1, mock_emb2]

    with patch("llm_core.providers.openai.OpenAI") as MockClient:
        MockClient.return_value.embeddings.create.return_value = mock_response

        provider = OpenAIEmbeddingProvider(api_key="test-key")
        vectors = provider.embed(["text one", "text two"])

    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2]
    assert vectors[1] == [0.3, 0.4]

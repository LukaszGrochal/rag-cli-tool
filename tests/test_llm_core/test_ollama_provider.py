"""Tests for OllamaProvider and OllamaEmbeddingProvider."""

from unittest.mock import MagicMock, patch

from llm_core.providers.base import LLMResponse
from llm_core.providers.ollama import OllamaEmbeddingProvider, OllamaProvider


def test_generate_returns_llm_response():
    """OllamaProvider.generate should return a properly structured LLMResponse."""
    mock_response = MagicMock()
    mock_response.message.content = "Paris is the capital of France."
    mock_response.model = "llama3.2"
    mock_response.prompt_eval_count = 12
    mock_response.eval_count = 9

    with patch("llm_core.providers.ollama._ollama_lib.Client") as MockClient:
        MockClient.return_value.chat.return_value = mock_response

        provider = OllamaProvider(model="llama3.2")
        response = provider.generate("What is the capital of France?", system="Be concise.")

    assert isinstance(response, LLMResponse)
    assert response.text == "Paris is the capital of France."
    assert response.model == "llama3.2"
    assert response.input_tokens == 12
    assert response.output_tokens == 9

    MockClient.return_value.chat.assert_called_once_with(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    )


def test_generate_without_system_prompt():
    """When no system prompt is given, messages should only contain the user message."""
    mock_response = MagicMock()
    mock_response.message.content = "Hello!"
    mock_response.model = "llama3.2"
    mock_response.prompt_eval_count = 5
    mock_response.eval_count = 3

    with patch("llm_core.providers.ollama._ollama_lib.Client") as MockClient:
        MockClient.return_value.chat.return_value = mock_response

        provider = OllamaProvider(model="llama3.2")
        provider.generate("Hi")

    MockClient.return_value.chat.assert_called_once_with(
        model="llama3.2",
        messages=[{"role": "user", "content": "Hi"}],
    )


def test_generate_missing_token_counts():
    """Token counts should default to 0 when not present in response."""
    mock_response = MagicMock(spec=["message", "model"])
    mock_response.message.content = "Answer"
    mock_response.model = "llama3.2"

    with patch("llm_core.providers.ollama._ollama_lib.Client") as MockClient:
        MockClient.return_value.chat.return_value = mock_response

        provider = OllamaProvider(model="llama3.2")
        response = provider.generate("Question")

    assert response.input_tokens == 0
    assert response.output_tokens == 0


def test_embed_single_text():
    """OllamaEmbeddingProvider.embed should return embeddings for texts."""
    mock_response = MagicMock()
    mock_response.embeddings = [[0.1, 0.2, 0.3]]

    with patch("llm_core.providers.ollama._ollama_lib.Client") as MockClient:
        MockClient.return_value.embed.return_value = mock_response

        provider = OllamaEmbeddingProvider(model="nomic-embed-text")
        vectors = provider.embed(["hello world"])

    assert vectors == [[0.1, 0.2, 0.3]]
    MockClient.return_value.embed.assert_called_once_with(
        model="nomic-embed-text",
        input=["hello world"],
    )


def test_embed_empty_list():
    """Embedding an empty list should return an empty list without calling the API."""
    with patch("llm_core.providers.ollama._ollama_lib.Client") as MockClient:
        provider = OllamaEmbeddingProvider(model="nomic-embed-text")
        vectors = provider.embed([])

    assert vectors == []
    MockClient.return_value.embed.assert_not_called()

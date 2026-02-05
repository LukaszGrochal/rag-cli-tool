from unittest.mock import MagicMock, patch

from llm_core.providers.anthropic import AnthropicProvider
from llm_core.providers.base import LLMResponse


def test_generate_returns_llm_response():
    """AnthropicProvider.generate should return a properly structured LLMResponse."""
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="Paris is the capital of France.")]
    mock_message.model = "claude-3-5-sonnet-latest"
    mock_message.usage.input_tokens = 10
    mock_message.usage.output_tokens = 8

    with patch("llm_core.providers.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_message

        provider = AnthropicProvider(api_key="test-key", model="claude-3-5-sonnet-latest")
        response = provider.generate("What is the capital of France?", system="Be concise.")

    assert isinstance(response, LLMResponse)
    assert response.text == "Paris is the capital of France."
    assert response.model == "claude-3-5-sonnet-latest"
    assert response.input_tokens == 10
    assert response.output_tokens == 8

    MockClient.return_value.messages.create.assert_called_once_with(
        model="claude-3-5-sonnet-latest",
        max_tokens=4096,
        system="Be concise.",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

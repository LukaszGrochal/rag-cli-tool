# tests/test_llm_core/test_config.py
import os
from unittest.mock import patch


def test_default_settings():
    """Settings should have sensible defaults for optional fields."""
    with patch.dict(os.environ, {
        "ANTHROPIC_API_KEY": "test-ant-key",
        "OPENAI_API_KEY": "test-oai-key",
    }, clear=False):
        from llm_core.config import LLMSettings
        settings = LLMSettings()
        assert settings.anthropic_api_key == "test-ant-key"
        assert settings.openai_api_key == "test-oai-key"
        assert settings.model == "claude-3-5-sonnet-latest"
        assert settings.embedding_model == "text-embedding-3-small"
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200
        assert settings.top_k == 3


def test_settings_without_api_keys():
    """Settings should load without API keys (they default to empty string)."""
    with patch.dict(os.environ, {}, clear=True):
        from llm_core.config import LLMSettings
        settings = LLMSettings(_env_file=None)
        assert settings.anthropic_api_key == ""
        assert settings.openai_api_key == ""


def test_ollama_host_default():
    """ollama_host should default to localhost."""
    with patch.dict(os.environ, {}, clear=True):
        from llm_core.config import LLMSettings
        settings = LLMSettings(_env_file=None)
        assert settings.ollama_host == "http://localhost:11434"


def test_ollama_host_override():
    """ollama_host should be overridable via env var."""
    with patch.dict(os.environ, {"RAG_CLI_OLLAMA_HOST": "http://remote:11434"}, clear=True):
        from llm_core.config import LLMSettings
        settings = LLMSettings(_env_file=None)
        assert settings.ollama_host == "http://remote:11434"

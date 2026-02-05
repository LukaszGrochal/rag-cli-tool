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

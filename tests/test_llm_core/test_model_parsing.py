"""Tests for parse_model_string()."""

from llm_core.config import parse_model_string


def test_ollama_prefix():
    """Should split 'ollama:model' into ('ollama', 'model')."""
    assert parse_model_string("ollama:llama3.2") == ("ollama", "llama3.2")


def test_ollama_prefix_uppercase():
    """Provider prefix should be lowercased."""
    assert parse_model_string("Ollama:llama3.2") == ("ollama", "llama3.2")


def test_no_prefix():
    """Bare model name should return empty provider."""
    assert parse_model_string("claude-3-5-sonnet-latest") == ("", "claude-3-5-sonnet-latest")


def test_colon_in_model_name():
    """Only the first colon should be used as separator."""
    assert parse_model_string("ollama:my:custom:model") == ("ollama", "my:custom:model")


def test_empty_string():
    """Empty string should return empty provider and model."""
    assert parse_model_string("") == ("", "")

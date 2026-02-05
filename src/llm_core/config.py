"""Configuration for LLM and RAG settings via environment variables."""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """Settings loaded from environment variables.

    RAG-specific settings use RAG_CLI_ prefix. API keys use standard names.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RAG_CLI_",
        extra="ignore",
    )

    # API keys — no prefix, standard env var names
    anthropic_api_key: str = Field(
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "RAG_CLI_ANTHROPIC_API_KEY"),
    )
    openai_api_key: str = Field(
        validation_alias=AliasChoices("OPENAI_API_KEY", "RAG_CLI_OPENAI_API_KEY"),
    )

    # Model settings
    model: str = "claude-3-5-sonnet-latest"
    embedding_model: str = "text-embedding-3-small"

    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval settings
    top_k: int = 3

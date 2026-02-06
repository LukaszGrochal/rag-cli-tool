"""Ollama embeddings adapter for rag_core."""

from llm_core.providers.ollama import OllamaEmbeddingProvider

from rag_core.embeddings.base import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    """Embedder using a local Ollama server via llm_core. Batches large inputs."""

    def __init__(self, provider: OllamaEmbeddingProvider, batch_size: int = 2000) -> None:
        self._provider = provider
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings, batching if needed."""
        if len(texts) <= self._batch_size:
            return self._provider.embed(texts)

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            all_vectors.extend(self._provider.embed(batch))
        return all_vectors

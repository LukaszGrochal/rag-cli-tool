"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract interface for text embedding."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...

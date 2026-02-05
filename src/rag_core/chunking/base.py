"""Abstract base class for text chunkers."""

from abc import ABC, abstractmethod


class BaseChunker(ABC):
    """Abstract interface for text chunking strategies."""

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split text into chunks."""
        ...

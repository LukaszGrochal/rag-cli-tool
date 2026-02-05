"""Abstract base class for retrieval."""

from abc import ABC, abstractmethod

from rag_core.vectorstores.base import SearchResult


class BaseRetriever(ABC):
    """Abstract interface for document retrieval."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Retrieve relevant document chunks for a query."""
        ...

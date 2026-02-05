"""Abstract base class for vector stores."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    """A single search result from the vector store."""

    id: str
    document: str
    metadata: dict
    distance: float


class BaseVectorStore(ABC):
    """Abstract interface for vector stores."""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Add documents with their embeddings to the store."""
        ...

    @abstractmethod
    def query(
        self, query_embedding: list[float], top_k: int = 3
    ) -> list[SearchResult]:
        """Query for the most similar documents."""
        ...

    @abstractmethod
    def existing_ids(self) -> set[str]:
        """Return the set of all document IDs in the store."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the number of documents in the store."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Delete all data from the store."""
        ...

"""ChromaDB vector store adapter."""

from pathlib import Path

import chromadb

from rag_core.vectorstores.base import BaseVectorStore, SearchResult

_COLLECTION_NAME = "rag_cli_docs"


class ChromaStore(BaseVectorStore):
    """Vector store backed by ChromaDB with local persistence."""

    def __init__(self, persist_dir: Path) -> None:
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Add documents with embeddings to ChromaDB."""
        # ChromaDB rejects empty metadata dicts; convert them to None.
        cleaned = [m if m else None for m in metadatas]
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=cleaned,
        )

    def query(
        self, query_embedding: list[float], top_k: int = 3
    ) -> list[SearchResult]:
        """Query ChromaDB for the most similar documents."""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        search_results: list[SearchResult] = []
        for i in range(len(results["ids"][0])):
            search_results.append(
                SearchResult(
                    id=results["ids"][0][i],
                    document=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                    distance=results["distances"][0][i],
                )
            )
        return search_results

    def existing_ids(self) -> set[str]:
        """Return all document IDs currently in the store."""
        result = self._collection.get(include=[])
        return set(result["ids"])

    def count(self) -> int:
        """Return the number of documents in the store."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete the collection and recreate it empty."""
        self._client.delete_collection(name=_COLLECTION_NAME)
        self._collection = self._client.create_collection(name=_COLLECTION_NAME)

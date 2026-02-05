"""Similarity-based retrieval using embeddings."""

from rag_core.embeddings.base import BaseEmbedder
from rag_core.retrieval.base import BaseRetriever
from rag_core.vectorstores.base import BaseVectorStore, SearchResult


class SimilarityRetriever(BaseRetriever):
    """Retriever that embeds the query and finds similar chunks."""

    def __init__(self, embedder: BaseEmbedder, store: BaseVectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Retrieve the most similar chunks to the query."""
        query_embedding = self._embedder.embed([query])[0]
        return self._store.query(query_embedding=query_embedding, top_k=top_k)

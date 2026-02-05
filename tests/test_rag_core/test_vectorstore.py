# tests/test_rag_core/test_vectorstore.py
from pathlib import Path

from rag_core.vectorstores.chroma import ChromaStore


def test_add_and_query(tmp_path: Path):
    """Should add documents and retrieve them by embedding similarity."""
    store = ChromaStore(persist_dir=tmp_path / "chroma")
    store.add(
        ids=["doc1", "doc2"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        documents=["First document", "Second document"],
        metadatas=[{"source": "a.txt"}, {"source": "b.txt"}],
    )

    results = store.query(query_embedding=[1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].document == "First document"
    assert results[0].metadata["source"] == "a.txt"


def test_existing_ids(tmp_path: Path):
    """Should return the set of existing document IDs."""
    store = ChromaStore(persist_dir=tmp_path / "chroma")
    store.add(
        ids=["a", "b"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        documents=["doc a", "doc b"],
        metadatas=[{}, {}],
    )

    existing = store.existing_ids()
    assert existing == {"a", "b"}


def test_reset_clears_data(tmp_path: Path):
    """reset() should delete all data."""
    store = ChromaStore(persist_dir=tmp_path / "chroma")
    store.add(
        ids=["x"],
        embeddings=[[0.1]],
        documents=["some doc"],
        metadatas=[{}],
    )
    assert store.count() == 1
    store.reset()
    assert store.count() == 0


def test_persistence(tmp_path: Path):
    """Data should persist across ChromaStore instances."""
    persist_dir = tmp_path / "chroma"
    store1 = ChromaStore(persist_dir=persist_dir)
    store1.add(
        ids=["persistent"],
        embeddings=[[0.5, 0.5]],
        documents=["I persist"],
        metadatas=[{"key": "val"}],
    )

    store2 = ChromaStore(persist_dir=persist_dir)
    assert store2.count() == 1
    assert "persistent" in store2.existing_ids()

from pathlib import Path

from rag_core.loaders.documents import Document, load_documents


def test_load_txt_file(tmp_path: Path):
    """Should load a .txt file as a Document."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello, world!", encoding="utf-8")

    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0].content == "Hello, world!"
    assert docs[0].source == str(txt_file)


def test_load_md_file(tmp_path: Path):
    """Should load a .md file as a Document."""
    md_file = tmp_path / "readme.md"
    md_file.write_text("# Title\n\nSome content.", encoding="utf-8")

    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert "# Title" in docs[0].content
    assert docs[0].source == str(md_file)


def test_load_empty_folder(tmp_path: Path):
    """Should return empty list for folder with no supported files."""
    docs = load_documents(tmp_path)
    assert docs == []


def test_load_recursive(tmp_path: Path):
    """Should find files in subdirectories."""
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("Nested content", encoding="utf-8")
    (tmp_path / "top.txt").write_text("Top content", encoding="utf-8")

    docs = load_documents(tmp_path)
    assert len(docs) == 2


def test_skips_unsupported_files(tmp_path: Path):
    """Should skip files with unsupported extensions."""
    (tmp_path / "data.csv").write_text("a,b,c", encoding="utf-8")
    (tmp_path / "note.txt").write_text("A note", encoding="utf-8")

    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0].source.endswith("note.txt")

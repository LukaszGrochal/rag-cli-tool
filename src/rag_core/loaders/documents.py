"""Document loading from local files (PDF, MD, TXT, DOCX)."""

from dataclasses import dataclass
from pathlib import Path

import docx2txt
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


@dataclass(frozen=True)
class Document:
    """A loaded document with its content and source path."""

    content: str
    source: str


def _load_txt(path: Path) -> str:
    """Load plain text or markdown file."""
    return path.read_text(encoding="utf-8")


def _load_pdf(path: Path) -> str:
    """Load text from a PDF file."""
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _load_docx(path: Path) -> str:
    """Load text from a DOCX file."""
    return docx2txt.process(str(path))


_LOADERS: dict[str, callable] = {
    ".txt": _load_txt,
    ".md": _load_txt,
    ".pdf": _load_pdf,
    ".docx": _load_docx,
}


def load_documents(path: Path) -> list[Document]:
    """Recursively load all supported documents from a directory."""
    documents: list[Document] = []

    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            loader = _LOADERS[file_path.suffix.lower()]
            content = loader(file_path)
            if content.strip():
                documents.append(Document(content=content, source=str(file_path)))

    return documents

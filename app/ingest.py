# app/ingest.py
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import KNOWLEDGE_DIR
from .vector_store import (
    get_vector_store,
    add_documents_to_vector_store,
)

# Chunking config: tune as needed.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_file(path: Path) -> str:
    """
    Simple PDF reader using pypdf: concatenates all page texts.
    """
    from pypdf import PdfReader  # ensure pypdf is in requirements

    reader = PdfReader(str(path))
    texts: List[str] = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


def _split_text(text: str) -> List[str]:
    """
    Chunk raw text into overlapping segments.

    Uses RecursiveCharacterTextSplitter so we split on paragraphs/lines/words
    before falling back to raw characters. That gives nicer chunks for RAG
    and lets the embedding model do less pointless work.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def load_file_as_documents(path: Path) -> List[Document]:
    """
    Load a single file (.txt, .md, .pdf) into a list of LangChain Documents.
    """
    suffix = path.suffix.lower()
    if suffix in [".txt", ".md"]:
        raw = _read_text_file(path)
    elif suffix == ".pdf":
        raw = _read_pdf_file(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    chunks = _split_text(raw)
    docs: List[Document] = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": str(path.relative_to(KNOWLEDGE_DIR)),
                    "chunk_id": i,
                },
            )
        )
    return docs


def ingest_path(path: Path) -> int:
    """
    Ingest a single file into the vector store.
    Returns number of chunks added.
    """
    vs = get_vector_store()
    docs = load_file_as_documents(path)
    add_documents_to_vector_store(vs=vs, documents=docs, persist=True)
    return len(docs)


def ingest_directory(root: Path) -> int:
    """
    Recursively ingest all supported files under a directory.
    """
    total = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in [".txt", ".md", ".pdf"]:
            continue
        total += ingest_path(p)
    return total
